import os
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from beamz.const import µm
from beamz.design.core import Design
from beamz.devices.monitors.monitors import Monitor
from beamz.simulation.boundaries import PML, Boundary
from beamz.simulation.compiled import (
    EngineState,
    MonitorState,
    compile_simulation,
    monitor_frequency_size,
    monitor_state_size,
)
from beamz.simulation.fields import Fields
from beamz.simulation.ops import advance_e_field, advance_h_field


class Simulation:
    """FDTD simulation class supporting both 2D and 3D electromagnetic simulations."""

    def __init__(
        self,
        design: Design = None,
        devices: list = None,
        boundaries: list[Boundary] = None,
        thermal=None,
        resolution: float = 0.02 * µm,
        time: np.ndarray = None,
        plane_2d: str = "xy",
    ):
        self.design = design
        devices = devices or []
        boundaries = boundaries or []
        self.resolution = resolution
        self.is_3d = design.is_3d and design.depth > 0
        self.plane_2d = plane_2d.lower()
        if self.plane_2d not in ["xy", "yz", "xz"]:
            self.plane_2d = "xy"

        # Get material grids from design (design owns the material grids, we reference them)
        permittivity, conductivity, permeability = design.get_material_grids(resolution)

        # Initialize time stepping first
        if time is None or len(time) < 2:
            raise ValueError("FDTD requires a time array with at least two entries")
        self.time, self.dt, self.num_steps = time, float(time[1] - time[0]), len(time)
        self.t, self.current_step = float(time[0]), 0

        # Check for PML boundaries before creating fields (to avoid double material init)
        pml_boundaries = [b for b in boundaries if isinstance(b, PML)]

        # Create field storage (fields owns the E/H field arrays, references material grids)
        self.fields = Fields(
            permittivity,
            conductivity,
            permeability,
            resolution,
            plane_2d=self.plane_2d,
            _init_materials=not pml_boundaries,
        )

        # Initialize PML regions if present
        if pml_boundaries:
            # Create PML regions (do this once, not every timestep)
            pml_data = {}
            for pml in pml_boundaries:
                pml_data.update(
                    pml.create_pml_regions(
                        self.fields, design, resolution, self.dt, plane_2d=self.plane_2d
                    )
                )
            self.pml_data = pml_data

            # Set effective conductivity for PML
            self.fields.set_pml_conductivity(pml_data)
        else:
            self.pml_data = None

        # Store device references (no duplication)
        self.devices = devices

        # Store boundary references (no duplication)
        self.boundaries = boundaries

        # Optional thermal coupling
        self.thermal = thermal
        if self.thermal is not None and getattr(self.thermal, "enabled", True):
            self.thermal.initialize(self)

        # Compiled program cache for v0.3 packed-source/monitor execution.
        self._compiled_program = None
        self._compiled_program_signature = None
        self._compiled_program_cache = {}

    def step(self):
        """Perform one FDTD time step with correct Huygens source timing.

        Order: H-update → M-injection → E-update → J-injection → legacy sources
        """
        if self.current_step >= self.num_steps:
            return False

        # Legacy devices (only have inject(), no inject_h/inject_e): inject before update
        self._inject_legacy_sources()

        # Collect source terms from legacy devices (if any)
        source_j, source_m = self._collect_source_terms()

        # 1. H update
        self.fields.update_h(self.dt, source_m=source_m)

        # 2. M injection (modifies H after update)
        self._inject_h_sources()

        # 3. E update (uses modified H)
        self.fields.update_e(self.dt, source_j=source_j)

        # 4. J injection (modifies E after update)
        self._inject_e_sources()

        # Record monitor data (if monitors are in devices)
        self._record_monitors()

        # Update coupled physics (thermal)
        if self.thermal is not None and getattr(self.thermal, "enabled", True):
            self.thermal.step(self)

        # Update time and step counter
        self.t += self.dt
        self.current_step += 1
        return True

    def _record_monitors(self):
        """Record data from Monitor devices during simulation."""
        for device in self.devices:
            if isinstance(device, Monitor) and device.should_record(self.current_step):
                if not self.is_3d:
                    device.record_fields(
                        self.fields.Ez,
                        self.fields.Hx,
                        self.fields.Hy,
                        self.t,
                        self.resolution,
                        self.resolution,
                        self.current_step,
                    )
                else:
                    device.record_fields(
                        self.fields.Ex,
                        self.fields.Ey,
                        self.fields.Ez,
                        self.fields.Hx,
                        self.fields.Hy,
                        self.fields.Hz,
                        self.t,
                        self.resolution,
                        self.resolution,
                        self.resolution,
                        self.current_step,
                    )

    def _inject_h_sources(self):
        """Inject magnetic currents (M) into H-fields after H update."""
        for device in self.devices:
            if hasattr(device, "inject_h"):
                device.inject_h(
                    self.fields,
                    self.t,
                    self.dt,
                    self.current_step,
                    self.resolution,
                    self.design,
                )

    def _inject_e_sources(self):
        """Inject electric currents (J) into E-fields after E update."""
        for device in self.devices:
            if hasattr(device, "inject_e"):
                device.inject_e(
                    self.fields,
                    self.t,
                    self.dt,
                    self.current_step,
                    self.resolution,
                    self.design,
                )

    def _inject_legacy_sources(self):
        """Inject from devices that only have inject() (no inject_h/inject_e)."""
        for device in self.devices:
            if hasattr(device, "inject") and not hasattr(device, "inject_h"):
                device.inject(
                    self.fields,
                    self.t,
                    self.dt,
                    self.current_step,
                    self.resolution,
                    self.design,
                )

    def _collect_source_terms(self):
        """Collect electric and magnetic current sources from all devices."""
        source_j = {}  # Electric currents for E-field update
        source_m = {}  # Magnetic currents for H-field update

        for device in self.devices:
            if hasattr(device, "get_source_terms"):
                j, m = device.get_source_terms(
                    self.fields,
                    self.t,
                    self.dt,
                    self.current_step,
                    self.resolution,
                    self.design,
                )
                for key, val in j.items():
                    source_j.setdefault(key, []).append(val)
                for key, val in m.items():
                    source_m.setdefault(key, []).append(val)

        return source_j, source_m

    def _create_jit_step(self):
        """Create a JIT-compiled FDTD step function for maximum performance.

        Returns a pure function that takes field arrays and returns updated field arrays.
        """
        resolution = self.resolution
        dt = self.dt
        plane_2d = self.plane_2d

        # Material parameters (static for the simulation)
        eps_x, sig_x, region_x = (
            self.fields.eps_x,
            self.fields.sig_x,
            self.fields.region_x,
        )
        eps_y, sig_y, region_y = (
            self.fields.eps_y,
            self.fields.sig_y,
            self.fields.region_y,
        )
        eps_z, sig_z, region_z = (
            self.fields.eps_z,
            self.fields.sig_z,
            self.fields.region_z,
        )
        sigma_m_hx = self.fields.sigma_m_hx
        sigma_m_hy = self.fields.sigma_m_hy
        sigma_m_hz = self.fields.sigma_m_hz

        from beamz.simulation.ops import (
            curl_e_to_h_2d,
            curl_e_to_h_3d,
            curl_h_to_e_2d,
            curl_h_to_e_3d,
        )

        if self.is_3d:

            @jax.jit
            def step(Ex, Ey, Ez, Hx, Hy, Hz):
                curlE_x, curlE_y, curlE_z = curl_e_to_h_3d(Ex, Ey, Ez, resolution)
                Hx_new = advance_h_field(Hx, curlE_x, sigma_m_hx, dt)
                Hy_new = advance_h_field(Hy, curlE_y, sigma_m_hy, dt)
                Hz_new = advance_h_field(Hz, curlE_z, sigma_m_hz, dt)
                curlH_x, curlH_y, curlH_z = curl_h_to_e_3d(
                    Hx_new,
                    Hy_new,
                    Hz_new,
                    resolution,
                    ex_shape=Ex.shape,
                    ey_shape=Ey.shape,
                    ez_shape=Ez.shape,
                )
                Ex_new = advance_e_field(Ex, curlH_x, sig_x, eps_x, dt, region_x)
                Ey_new = advance_e_field(Ey, curlH_y, sig_y, eps_y, dt, region_y)
                Ez_new = advance_e_field(Ez, curlH_z, sig_z, eps_z, dt, region_z)
                return Ex_new, Ey_new, Ez_new, Hx_new, Hy_new, Hz_new

        else:

            @jax.jit
            def step(Ex, Ey, Ez, Hx, Hy, Hz):
                curlE_x, curlE_y, curlE_z = curl_e_to_h_2d(
                    (Ex, Ey, Ez), resolution, plane=plane_2d
                )
                Hx_new = advance_h_field(Hx, curlE_x, sigma_m_hx, dt)
                Hy_new = advance_h_field(Hy, curlE_y, sigma_m_hy, dt)
                Hz_new = advance_h_field(Hz, curlE_z, sigma_m_hz, dt)
                curlH_x, curlH_y, curlH_z = curl_h_to_e_2d(
                    (Hx_new, Hy_new, Hz_new),
                    resolution,
                    (Ex.shape, Ey.shape, Ez.shape),
                    plane=plane_2d,
                )
                Ex_new = advance_e_field(Ex, curlH_x, sig_x, eps_x, dt, region_x)
                Ey_new = advance_e_field(Ey, curlH_y, sig_y, eps_y, dt, region_y)
                Ez_new = advance_e_field(Ez, curlH_z, sig_z, eps_z, dt, region_z)
                return Ex_new, Ey_new, Ez_new, Hx_new, Hy_new, Hz_new

        return step

    def _create_jit_step_h(self):
        """Create a JIT-compiled H-update function."""
        resolution = self.resolution
        dt = self.dt
        plane_2d = self.plane_2d
        sigma_m_hx = self.fields.sigma_m_hx
        sigma_m_hy = self.fields.sigma_m_hy
        sigma_m_hz = self.fields.sigma_m_hz

        from beamz.simulation.ops import curl_e_to_h_2d, curl_e_to_h_3d

        if self.is_3d:

            @jax.jit
            def step_h(Ex, Ey, Ez, Hx, Hy, Hz):
                curlE_x, curlE_y, curlE_z = curl_e_to_h_3d(Ex, Ey, Ez, resolution)
                Hx_new = advance_h_field(Hx, curlE_x, sigma_m_hx, dt)
                Hy_new = advance_h_field(Hy, curlE_y, sigma_m_hy, dt)
                Hz_new = advance_h_field(Hz, curlE_z, sigma_m_hz, dt)
                return Hx_new, Hy_new, Hz_new

        else:

            @jax.jit
            def step_h(Ex, Ey, Ez, Hx, Hy, Hz):
                curlE_x, curlE_y, curlE_z = curl_e_to_h_2d(
                    (Ex, Ey, Ez), resolution, plane=plane_2d
                )
                Hx_new = advance_h_field(Hx, curlE_x, sigma_m_hx, dt)
                Hy_new = advance_h_field(Hy, curlE_y, sigma_m_hy, dt)
                Hz_new = advance_h_field(Hz, curlE_z, sigma_m_hz, dt)
                return Hx_new, Hy_new, Hz_new

        return step_h

    def _create_jit_step_e(self):
        """Create a JIT-compiled E-update function."""
        resolution = self.resolution
        dt = self.dt
        plane_2d = self.plane_2d
        eps_x, sig_x, region_x = (
            self.fields.eps_x,
            self.fields.sig_x,
            self.fields.region_x,
        )
        eps_y, sig_y, region_y = (
            self.fields.eps_y,
            self.fields.sig_y,
            self.fields.region_y,
        )
        eps_z, sig_z, region_z = (
            self.fields.eps_z,
            self.fields.sig_z,
            self.fields.region_z,
        )

        from beamz.simulation.ops import curl_h_to_e_2d, curl_h_to_e_3d

        if self.is_3d:

            @jax.jit
            def step_e(Ex, Ey, Ez, Hx, Hy, Hz):
                curlH_x, curlH_y, curlH_z = curl_h_to_e_3d(
                    Hx,
                    Hy,
                    Hz,
                    resolution,
                    ex_shape=Ex.shape,
                    ey_shape=Ey.shape,
                    ez_shape=Ez.shape,
                )
                Ex_new = advance_e_field(Ex, curlH_x, sig_x, eps_x, dt, region_x)
                Ey_new = advance_e_field(Ey, curlH_y, sig_y, eps_y, dt, region_y)
                Ez_new = advance_e_field(Ez, curlH_z, sig_z, eps_z, dt, region_z)
                return Ex_new, Ey_new, Ez_new

        else:

            @jax.jit
            def step_e(Ex, Ey, Ez, Hx, Hy, Hz):
                curlH_x, curlH_y, curlH_z = curl_h_to_e_2d(
                    (Hx, Hy, Hz),
                    resolution,
                    (Ex.shape, Ey.shape, Ez.shape),
                    plane=plane_2d,
                )
                Ex_new = advance_e_field(Ex, curlH_x, sig_x, eps_x, dt, region_x)
                Ey_new = advance_e_field(Ey, curlH_y, sig_y, eps_y, dt, region_y)
                Ez_new = advance_e_field(Ez, curlH_z, sig_z, eps_z, dt, region_z)
                return Ex_new, Ey_new, Ez_new

        return step_e

    def compile(self, num_steps=None):
        """Compile the v0.3 packed-data simulation program."""
        if num_steps is None:
            num_steps = self.num_steps - self.current_step
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")

        loop_kind_env = os.getenv("BEAMZ_COMPILED_LOOP_KIND", "scan").strip().lower()
        if loop_kind_env in {"fori", "fori_loop", "fori-loop"}:
            loop_kind = "fori_loop"
        elif loop_kind_env == "scan":
            loop_kind = "scan"
        else:
            raise ValueError("Invalid BEAMZ_COMPILED_LOOP_KIND (use: scan, fori_loop).")
        e_shell_split = os.getenv("BEAMZ_ENABLE_E_SHELL_SPLIT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        h_shell_split = os.getenv("BEAMZ_ENABLE_H_SHELL_SPLIT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        source_single_slab_dense = os.getenv(
            "BEAMZ_SOURCE_SINGLE_SLAB_DENSE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

        signature = (
            num_steps,
            self.fields.permittivity.shape,
            self.is_3d,
            self.plane_2d,
            loop_kind,
            e_shell_split,
            h_shell_split,
            source_single_slab_dense,
        )
        cached = self._compiled_program_cache.get(signature)
        if cached is not None:
            self._compiled_program = cached
            self._compiled_program_signature = signature
            return cached

        run_cfg = SimpleNamespace(
            fields=self.fields,
            resolution=self.resolution,
            dt=self.dt,
            num_steps=num_steps,
            plane_2d=self.plane_2d,
            is_3d=self.is_3d,
            total_steps=self.num_steps,
            t0=float(self.time[0]),
            precision="float32",
            loop_kind=loop_kind,
            source_single_slab_dense=source_single_slab_dense,
        )
        program = compile_simulation(
            design=self.design,
            devices=self.devices,
            boundaries=self.boundaries,
            run_cfg=run_cfg,
        )
        self._compiled_program_cache[signature] = program
        self._compiled_program = program
        self._compiled_program_signature = signature
        return program

    def run_compiled(
        self, num_steps=None, record_interval=None, record_fields=None, progress=True
    ):
        """Run simulation using the v0.3 single-program compiled scan engine.

        Notes:
        - Source/monitor callbacks are compiled as packed specs.
        - Monitor results are accumulated in-loop and written back to Monitor objects.
        - Field history recording is optional and chunked via repeated compiled runs.
        """
        if self.thermal is not None and getattr(self.thermal, "enabled", True):
            raise NotImplementedError(
                "run_compiled currently does not support thermal coupling."
            )

        if num_steps is None:
            num_steps = self.num_steps - self.current_step
        num_steps = int(num_steps)
        if num_steps <= 0:
            return None

        if record_fields is None:
            record_fields = ["Ez"]

        record_every = int(record_interval) if record_interval else None
        if record_every is not None and record_every <= 0:
            raise ValueError("record_interval must be a positive integer")

        field_history = {name: [] for name in record_fields} if record_every else None

        # Run in one chunk for max TCUPS by default. For field snapshots, run in equal chunks.
        chunk_size = record_every if record_every else num_steps
        steps_remaining = num_steps
        steps_done = 0
        monitor_state: MonitorState | None = None

        while steps_remaining > 0:
            this_chunk = min(chunk_size, steps_remaining)
            program = self.compile(num_steps=this_chunk)

            if progress and steps_done == 0 and program.compile_count == 0:
                print(
                    "● JIT compiling v0.3 packed FDTD program...", end=" ", flush=True
                )

            engine_state = EngineState(
                ex=self.fields.Ex,
                ey=self.fields.Ey,
                ez=self.fields.Ez,
                hx=self.fields.Hx,
                hy=self.fields.Hy,
                hz=self.fields.Hz,
                t=jnp.asarray(self.t, dtype=jnp.float32),
                current_step=jnp.asarray(self.current_step, dtype=jnp.int32),
            )

            if monitor_state is None:
                if program.monitor_specs:
                    max_records = max(
                        1, monitor_state_size(program.monitor_specs, num_steps)
                    )
                    max_freq = monitor_frequency_size(program.monitor_specs)
                    monitor_state = MonitorState(
                        powers=jnp.zeros(
                            (len(program.monitor_specs), max_records), dtype=jnp.float32
                        ),
                        timestamps=jnp.zeros(
                            (len(program.monitor_specs), max_records), dtype=jnp.float32
                        ),
                        counts=jnp.zeros(
                            (len(program.monitor_specs),), dtype=jnp.int32
                        ),
                        freq_flux_re=jnp.zeros(
                            (len(program.monitor_specs), max_freq), dtype=jnp.float32
                        ),
                        freq_flux_im=jnp.zeros(
                            (len(program.monitor_specs), max_freq), dtype=jnp.float32
                        ),
                        freq_phase_re=jnp.ones(
                            (len(program.monitor_specs), max_freq), dtype=jnp.float32
                        ),
                        freq_phase_im=jnp.zeros(
                            (len(program.monitor_specs), max_freq), dtype=jnp.float32
                        ),
                    )
                else:
                    monitor_state = MonitorState(
                        powers=jnp.zeros((0, 0), dtype=jnp.float32),
                        timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
                        counts=jnp.zeros((0,), dtype=jnp.int32),
                        freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
                        freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
                        freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
                        freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
                    )

            engine_state, monitor_state, _ = program.run(
                engine_state=engine_state,
                monitor_state=monitor_state,
            )
            engine_state.ez.block_until_ready()

            if progress and steps_done == 0:
                print("done!")

            self.fields.Ex = engine_state.ex
            self.fields.Ey = engine_state.ey
            self.fields.Ez = engine_state.ez
            self.fields.Hx = engine_state.hx
            self.fields.Hy = engine_state.hy
            self.fields.Hz = engine_state.hz
            self.t = float(np.asarray(engine_state.t))
            self.current_step = int(np.asarray(engine_state.current_step))

            if field_history is not None and (self.current_step % record_every == 0):
                for name in record_fields:
                    if hasattr(self.fields, name):
                        field_history[name].append(np.array(getattr(self.fields, name)))

            steps_done += this_chunk
            steps_remaining -= this_chunk

            if progress and num_steps > 0:
                pct = 100.0 * steps_done / num_steps
                print(
                    f"\r● Progress: {pct:.0f}% ({steps_done}/{num_steps} steps)",
                    end="",
                    flush=True,
                )

        if progress:
            print()

        if monitor_state is not None:
            program.apply_monitor_state(monitor_state)

        result = {}
        if field_history is not None:
            result["fields"] = {
                k: np.stack(v) if len(v) > 0 else np.zeros((0,))
                for k, v in field_history.items()
            }
        monitors = [device for device in self.devices if isinstance(device, Monitor)]
        if monitors:
            result["monitors"] = monitors
        return result if result else None

    def run_fast(
        self, num_steps=None, record_interval=None, record_fields=None, progress=True
    ):
        """Backward-compatible alias to `run_compiled` in v0.3."""
        return self.run_compiled(
            num_steps=num_steps,
            record_interval=record_interval,
            record_fields=record_fields,
            progress=progress,
        )

    def run_jit_scan(self, num_steps=None, progress=True):
        """Backward-compatible alias to `run_compiled` in v0.3."""
        return self.run_compiled(
            num_steps=num_steps,
            record_interval=None,
            record_fields=None,
            progress=progress,
        )

    def run(self, **kwargs):
        """Run complete FDTD simulation with optional live field visualization.

        Accepts all visualization parameters (animate_live, cmap, save_video, etc.).
        See beamz.visual.runner.VizConfig for the full list of options.

        Returns:
            dict with keys:
                - 'fields': dict of field histories if save_fields was provided
                - 'monitors': list of Monitor objects with recorded data
                - 'animation': JupyterAnimator object if running in Jupyter with animate_live
        """
        # Default non-visual path uses the compiled engine in v0.3.
        wants_live_viz = any(
            kwargs.get(k) is not None
            for k in ("animate_live", "save_video", "jupyter_live")
        )
        if not wants_live_viz:
            save_fields = kwargs.get("save_fields")
            field_subsample = int(kwargs.get("field_subsample", 1))
            progress = bool(kwargs.get("progress", False))
            record_interval = field_subsample if save_fields else None
            return self.run_compiled(
                num_steps=None,
                record_interval=record_interval,
                record_fields=save_fields,
                progress=progress,
            )

        from beamz.visual.runner import run_with_visualization

        return run_with_visualization(self, **kwargs)
