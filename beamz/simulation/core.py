import os
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from beamz.const import µm
from beamz.design.core import Design
from beamz.devices.monitors.monitors import Monitor
from beamz.devices.sources.solve import solve_modes
from beamz.simulation.boundaries import PML, Boundary
from beamz.simulation.compiled import (
    EngineState,
    MonitorState,
    compile_simulation,
    monitor_dft_point_size,
    monitor_frequency_size,
    monitor_state_size,
)
from beamz.simulation.fields import Fields
from beamz.simulation.ops import advance_e_field, advance_h_field


@dataclass(frozen=True)
class PortSpec:
    name: str
    monitor_name: str
    direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"]
    polarization: Literal["tm", "te"]
    mode_index: int = 0
    reference_monitor: str | None = None


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
                new_data = pml.create_pml_regions(
                    self.fields, design, resolution, self.dt, plane_2d=self.plane_2d
                )
                if not pml_data:
                    pml_data = dict(new_data)
                    continue

                if "mask" in new_data and "mask" in pml_data:
                    pml_data["mask"] = pml_data["mask"] | new_data["mask"]
                elif "mask" in new_data:
                    pml_data["mask"] = new_data["mask"]

                for key, value in new_data.items():
                    if key == "mask":
                        continue
                    if key in pml_data:
                        pml_data[key] = pml_data[key] + value
                    else:
                        pml_data[key] = value
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
        self._compiled_monitor_state = None

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
            if not isinstance(device, Monitor):
                continue
            should_record = device.should_record(self.current_step)
            dft_every_step = bool(
                getattr(device, "dft_enabled", False)
                and getattr(device, "dft_record_every_step", True)
            )
            if should_record or dft_every_step:
                if not self.is_3d:
                    device.record_fields_2d(
                        self.fields.Ez,
                        self.fields.Hx,
                        self.fields.Hy,
                        self.t,
                        self.resolution,
                        self.resolution,
                        self.current_step,
                        Ex=self.fields.Ex,
                        Ey=self.fields.Ey,
                        Hz=self.fields.Hz,
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
        if self.current_step == 0:
            self._compiled_monitor_state = None

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
                if (
                    self._compiled_monitor_state is not None
                    and program.monitor_specs
                    and int(np.asarray(self._compiled_monitor_state.counts.shape[0]))
                    == len(program.monitor_specs)
                ):
                    monitor_state = self._compiled_monitor_state
                elif program.monitor_specs:
                    records_horizon = max(1, int(self.num_steps - self.current_step))
                    max_records = max(
                        1, monitor_state_size(program.monitor_specs, records_horizon)
                    )
                    max_freq = monitor_frequency_size(program.monitor_specs)
                    max_points = monitor_dft_point_size(program.monitor_specs)
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
                        dft_vec_re=jnp.zeros(
                            (len(program.monitor_specs), 6, max_freq, max_points),
                            dtype=jnp.float32,
                        ),
                        dft_vec_im=jnp.zeros(
                            (len(program.monitor_specs), 6, max_freq, max_points),
                            dtype=jnp.float32,
                        ),
                        dft_weight_sum=jnp.zeros(
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
                        dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
                        dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
                        dft_weight_sum=jnp.zeros((0, 0), dtype=jnp.float32),
                    )
            self._compiled_monitor_state = monitor_state

            engine_state, monitor_state, _ = program.run(
                engine_state=engine_state,
                monitor_state=monitor_state,
            )
            engine_state.ez.block_until_ready()
            self._compiled_monitor_state = monitor_state

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

    def _get_monitor_trace(self, monitor, field_component="Ez", reduction="mean"):
        """Reduce monitor field snapshots to a 1D time trace."""
        if field_component not in monitor.fields:
            raise ValueError(
                f"Monitor '{monitor.name}' has no field '{field_component}'. "
                f"Available: {sorted(monitor.fields.keys())}"
            )

        raw = monitor.fields[field_component]
        if raw is None or len(raw) == 0:
            raise ValueError(
                f"Monitor '{monitor.name}' has no recorded '{field_component}' data."
            )

        values = np.asarray(raw)
        if values.ndim == 1:
            trace = values
        else:
            flattened = values.reshape(values.shape[0], -1)
            reduction_key = str(reduction).lower()
            if reduction_key == "mean":
                trace = np.mean(flattened, axis=1)
            elif reduction_key == "sum":
                trace = np.sum(flattened, axis=1)
            elif reduction_key == "max_abs":
                trace = np.max(np.abs(flattened), axis=1)
            else:
                raise ValueError(
                    f"Unsupported reduction '{reduction}'. "
                    "Use one of {'mean', 'sum', 'max_abs'}."
                )

        time_values = np.asarray(monitor.fields.get("t", []), dtype=float)
        if time_values.size < trace.shape[0]:
            if hasattr(self, "time") and len(self.time) >= trace.shape[0]:
                time_values = np.asarray(self.time[: trace.shape[0]], dtype=float)
            else:
                time_values = np.arange(trace.shape[0], dtype=float) * float(self.dt)

        return np.asarray(trace), np.asarray(time_values)

    @staticmethod
    def _safe_ratio(num, den, eps=1e-18):
        out = np.zeros_like(num, dtype=np.complex128)
        valid = np.abs(den) > eps
        out[valid] = num[valid] / den[valid]
        return out

    @staticmethod
    def _normalize_portspecs(ports):
        if isinstance(ports, dict):
            values = list(ports.values())
        else:
            values = list(ports)
        if not values:
            raise ValueError("ports must contain at least one PortSpec.")

        normalized = {}
        for item in values:
            if isinstance(item, PortSpec):
                spec = item
            else:
                spec = PortSpec(
                    name=item["name"],
                    monitor_name=item["monitor_name"],
                    direction=item["direction"],
                    polarization=item["polarization"],
                    mode_index=int(item.get("mode_index", 0)),
                    reference_monitor=item.get("reference_monitor"),
                )
            if spec.direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
                raise ValueError(f"Unsupported port direction '{spec.direction}'.")
            pol = str(spec.polarization).lower()
            if pol not in {"tm", "te"}:
                raise ValueError(f"Unsupported polarization '{spec.polarization}'.")
            normalized[spec.name] = PortSpec(
                name=spec.name,
                monitor_name=spec.monitor_name,
                direction=spec.direction,
                polarization=pol,
                mode_index=int(spec.mode_index),
                reference_monitor=spec.reference_monitor,
            )
        return normalized

    def _named_monitors(self):
        return {
            device.name: device
            for device in self.devices
            if isinstance(device, Monitor) and getattr(device, "name", None)
        }

    def _sample_monitor_component_spectrum(
        self,
        monitor,
        component,
        frequencies=None,
        window="hann",
    ):
        if component not in monitor.fields:
            raise ValueError(
                f"Monitor '{monitor.name}' has no field '{component}'. "
                f"Available: {sorted(monitor.fields.keys())}"
            )
        raw = monitor.fields[component]
        if raw is None or len(raw) == 0:
            raise ValueError(
                f"Monitor '{monitor.name}' has no recorded '{component}' data."
            )
        values = np.asarray(raw)
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)

        t = np.asarray(monitor.fields.get("t", []), dtype=float)
        n = min(values.shape[0], t.size)
        if n < 2:
            raise ValueError(
                f"Monitor '{monitor.name}' has insufficient samples for FFT extraction."
            )
        values = values[:n]
        t = t[:n]
        values = values - np.mean(values, axis=0, keepdims=True)

        win_key = str(window).lower() if window is not None else "none"
        if win_key in {"hann", "hanning"}:
            w = np.hanning(n)
        elif win_key in {"none", "rect", "rectangular"}:
            w = np.ones(n, dtype=float)
        else:
            raise ValueError(f"Unsupported window '{window}'.")
        values = values * w[:, None]

        dt = float(np.mean(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"Invalid dt inferred from monitor '{monitor.name}'.")
        if np.iscomplexobj(values):
            freq_bins = np.fft.fftfreq(n, d=dt)
            spec_bins = np.fft.fft(values, axis=0)
            keep = freq_bins >= 0
            freq_bins = freq_bins[keep]
            spec_bins = spec_bins[keep]
        else:
            freq_bins = np.fft.rfftfreq(n, d=dt)
            spec_bins = np.fft.rfft(values, axis=0)

        if frequencies is None:
            return freq_bins, spec_bins

        requested = np.atleast_1d(np.asarray(frequencies, dtype=float))
        sampled = np.empty((len(requested), spec_bins.shape[1]), dtype=np.complex128)
        for col in range(spec_bins.shape[1]):
            real_part = np.interp(
                requested, freq_bins, np.real(spec_bins[:, col]), left=0.0, right=0.0
            )
            imag_part = np.interp(
                requested, freq_bins, np.imag(spec_bins[:, col]), left=0.0, right=0.0
            )
            sampled[:, col] = real_part + 1j * imag_part
        return requested, sampled

    @staticmethod
    def _resample_complex_matrix(freq_src, values_src, freq_dst):
        src = np.asarray(values_src, dtype=np.complex128)
        if src.ndim == 1:
            src = src[:, None]
        if np.allclose(freq_src, freq_dst, rtol=1e-9, atol=0.0) and src.shape[0] == len(freq_dst):
            return src
        out = np.empty((len(freq_dst), src.shape[1]), dtype=np.complex128)
        for col in range(src.shape[1]):
            re = np.interp(freq_dst, freq_src, np.real(src[:, col]), left=0.0, right=0.0)
            im = np.interp(freq_dst, freq_src, np.imag(src[:, col]), left=0.0, right=0.0)
            out[:, col] = re + 1j * im
        return out

    def _sample_monitor_component_dft(self, monitor, component, frequencies):
        if not hasattr(monitor, "get_dft_component"):
            raise ValueError(
                f"Monitor '{monitor.name}' does not support DFT accumulation."
            )
        freq_src = np.asarray(monitor.get_dft_frequencies(), dtype=float)
        if freq_src.size == 0:
            raise ValueError(
                f"Monitor '{monitor.name}' has no configured DFT frequencies."
            )
        values_src = np.asarray(monitor.get_dft_component(component), dtype=np.complex128)
        freq_dst = np.atleast_1d(np.asarray(frequencies, dtype=float))
        return freq_dst, self._resample_complex_matrix(freq_src, values_src, freq_dst)

    def _demodulate_monitor_component(
        self,
        monitor,
        component,
        frequency,
        t_start=None,
        avg_cycles=12,
        window="hann",
    ):
        """Demodulate one monitor component at a single CW frequency.

        Returns the complex amplitude vector over monitor samples.
        """
        if component not in monitor.fields:
            raise ValueError(
                f"Monitor '{monitor.name}' has no field '{component}'. "
                f"Available: {sorted(monitor.fields.keys())}"
            )
        raw = monitor.fields[component]
        if raw is None or len(raw) == 0:
            raise ValueError(
                f"Monitor '{monitor.name}' has no recorded '{component}' data."
            )
        values = np.asarray(raw)
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)

        t = np.asarray(monitor.fields.get("t", []), dtype=float)
        n = min(values.shape[0], t.size)
        if n < 2:
            raise ValueError(
                f"Monitor '{monitor.name}' has insufficient samples for demodulation."
            )
        values = values[:n]
        t = t[:n]
        f0 = float(frequency)
        if not np.isfinite(f0) or f0 <= 0:
            raise ValueError(f"frequency must be positive, got {frequency!r}")

        if t_start is None:
            mask = np.ones(n, dtype=bool)
        else:
            mask = t >= float(t_start)
        if np.count_nonzero(mask) < 2:
            raise ValueError(
                f"Monitor '{monitor.name}' has insufficient post-transient samples."
            )
        t_sel = t[mask]
        v_sel = values[mask]

        if avg_cycles is not None:
            cycles = float(avg_cycles)
            if cycles > 0:
                span = cycles / f0
                t_end = t_sel[0] + span
                keep = t_sel <= t_end
                if np.count_nonzero(keep) >= 2:
                    t_sel = t_sel[keep]
                    v_sel = v_sel[keep]

        n_sel = t_sel.size
        if n_sel < 2:
            raise ValueError(
                f"Monitor '{monitor.name}' has insufficient samples in demod window."
            )
        win_key = str(window).lower() if window is not None else "none"
        if win_key in {"hann", "hanning"}:
            w = np.hanning(n_sel)
        elif win_key in {"none", "rect", "rectangular"}:
            w = np.ones(n_sel, dtype=float)
        else:
            raise ValueError(f"Unsupported window '{window}'.")

        carrier = np.exp(-1j * 2.0 * np.pi * f0 * t_sel)[:, None]
        denom = max(float(np.sum(w)), 1e-18)
        demod = (2.0 / denom) * np.sum((w[:, None] * v_sel) * carrier, axis=0)
        return np.asarray(demod, dtype=np.complex128)

    @staticmethod
    def _mode_components_for_port(spec):
        axis = spec.direction[1]
        tm_map = {
            "x": ("Ez", "Hy", 2, 1, -1.0),
            "y": ("Ez", "Hx", 2, 0, 1.0),
            "z": ("Ey", "Hx", 1, 0, -1.0),
        }
        te_map = {
            "x": ("Ey", "Hz", 1, 2, 1.0),
            "y": ("Ex", "Hz", 0, 2, -1.0),
            "z": ("Ex", "Hy", 0, 1, 1.0),
        }
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported port axis '{axis}'.")
        e_comp, h_comp, e_idx, h_idx, sign = (
            tm_map[axis] if spec.polarization == "tm" else te_map[axis]
        )
        return {
            "axis": axis,
            "e_component": e_comp,
            "h_component": h_comp,
            "e_mode_index": e_idx,
            "h_mode_index": h_idx,
            "signed_flux_sign": sign,
        }

    def _monitor_profile_slice(self, monitor, axis, pad_cells):
        perm = np.asarray(self.fields.permittivity)
        if perm.ndim == 3:
            z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
                self.resolution,
                self.resolution,
                self.resolution,
                perm.shape,
            )

            def _clamp(idx, limit):
                if isinstance(idx, slice):
                    start = 0 if idx.start is None else int(idx.start)
                    stop = limit if idx.stop is None else int(idx.stop)
                    start = max(0, min(start, max(limit - 1, 0)))
                    stop = max(start + 1, min(stop, limit))
                    return slice(start, stop)
                ii = int(idx)
                return max(0, min(ii, limit - 1))

            z_idx = _clamp(z_idx, perm.shape[0])
            y_idx = _clamp(y_idx, perm.shape[1])
            x_idx = _clamp(x_idx, perm.shape[2])
            eps_slice = np.asarray(perm[z_idx, y_idx, x_idx], dtype=np.complex128)
            if eps_slice.ndim != 2:
                eps_slice = np.atleast_2d(eps_slice)
            npts = int(eps_slice.size)
            local_idx = np.arange(npts, dtype=int)
            d_area = float(self.resolution) * float(self.resolution)
            return eps_slice, local_idx, d_area
        if perm.ndim != 2:
            raise NotImplementedError("Modal extraction supports 2D or 3D only.")
        points = monitor.get_grid_points_2d(self.resolution, self.resolution)
        if not points:
            raise ValueError(f"Monitor '{monitor.name}' contains no sample points.")
        p = np.asarray(points, dtype=float)
        if axis == "x":
            x_idx = int(np.clip(round(float(np.mean(p[:, 0]))), 0, perm.shape[1] - 1))
            eps_profile_full = perm[:, x_idx]
            sample_idx = np.asarray(
                [int(np.clip(pi[1], 0, perm.shape[0] - 1)) for pi in points], dtype=int
            )
        else:
            y_idx = int(np.clip(round(float(np.mean(p[:, 1]))), 0, perm.shape[0] - 1))
            eps_profile_full = perm[y_idx, :]
            sample_idx = np.asarray(
                [int(np.clip(pi[0], 0, perm.shape[1] - 1)) for pi in points], dtype=int
            )
        lo = max(0, int(np.min(sample_idx)) - int(pad_cells))
        hi = min(len(eps_profile_full), int(np.max(sample_idx)) + int(pad_cells) + 1)
        local_idx = np.clip(sample_idx - lo, 0, max(hi - lo - 1, 0))
        if len(points) > 1:
            step_idx = np.diff(np.asarray(points, dtype=float), axis=0)
            dl = float(np.mean(np.linalg.norm(step_idx, axis=1))) * float(self.resolution)
        else:
            dl = float(self.resolution)
        dl = max(dl, float(self.resolution) * 1e-9)
        return np.asarray(eps_profile_full[lo:hi], dtype=np.complex128), local_idx, dl

    def _build_port_projection(self, spec, monitor, frequency, cache, mode_pad_cells=6):
        key = (spec.name, monitor.name, float(frequency))
        cached = cache.get(key)
        if cached is not None:
            return cached

        parts = self._mode_components_for_port(spec)
        eps_profile, local_idx, dl = self._monitor_profile_slice(
            monitor, parts["axis"], mode_pad_cells
        )
        omega = 2.0 * np.pi * float(frequency)
        _, e_fields, h_fields, _ = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=float(self.resolution),
            m=spec.mode_index + 1,
            direction=spec.direction,
            filter_pol=spec.polarization,
            return_fields=True,
        )

        mode = int(spec.mode_index)
        e_fwd_full = np.asarray(
            np.squeeze(e_fields[mode][parts["e_mode_index"]]), dtype=np.complex128
        )
        h_fwd_full = np.asarray(
            np.squeeze(h_fields[mode][parts["h_mode_index"]]), dtype=np.complex128
        )
        if self.is_3d:
            if e_fwd_full.ndim == 1:
                e_fwd_full = e_fwd_full[:, None]
            if h_fwd_full.ndim == 1:
                h_fwd_full = h_fwd_full[:, None]
            e_flat = e_fwd_full.reshape(-1)
            h_flat = h_fwd_full.reshape(-1)
            n_target = min(e_flat.size, h_flat.size)
            try:
                n_monitor = int(
                    np.asarray(
                        monitor.get_dft_component(parts["e_component"]),
                        dtype=np.complex128,
                    ).shape[1]
                )
                n_target = min(n_target, n_monitor)
            except Exception:
                pass
            idx = np.asarray(local_idx, dtype=int)
            if idx.size > n_target:
                idx = idx[:n_target]
            e_fwd = e_flat[idx]
            h_fwd = h_flat[idx]
        else:
            if e_fwd_full.ndim > 1:
                e_fwd_full = e_fwd_full[:, 0]
            if h_fwd_full.ndim > 1:
                h_fwd_full = h_fwd_full[:, 0]
            e_fwd = e_fwd_full[local_idx]
            h_fwd = h_fwd_full[local_idx]

        if h_fwd.size:
            i_max = int(np.argmax(np.abs(h_fwd)))
            phase = np.angle(h_fwd[i_max])
            phase_rot = np.exp(-1j * phase)
            e_fwd = e_fwd * phase_rot
            h_fwd = h_fwd * phase_rot

        pm = 0.5 * np.real(
            np.sum(parts["signed_flux_sign"] * e_fwd * np.conjugate(h_fwd)) * dl
        )
        norm = np.sqrt(max(abs(pm), 1e-30))
        e_fwd = e_fwd / norm
        h_fwd = h_fwd / norm
        e_bwd = e_fwd.copy()
        h_bwd = -h_fwd.copy()

        mode_matrix = np.column_stack(
            [
                np.concatenate([e_fwd, h_fwd]),
                np.concatenate([e_bwd, h_bwd]),
            ]
        )
        projection = {
            "e_component": parts["e_component"],
            "h_component": parts["h_component"],
            "mode_matrix": mode_matrix,
            "condition_number": float(np.linalg.cond(mode_matrix)),
            "pinv": np.linalg.pinv(mode_matrix),
        }
        cache[key] = projection
        return projection

    def extract_port_waves(
        self,
        ports,
        frequencies,
        mode_strategy="per_frequency",
        window="hann",
        return_power=True,
    ):
        """Broadband modal extraction using FFT bins.

        Fast and convenient for sweeps, but less robust than CW demodulation
        for strict passivity/loss assessment.
        """
        if (not self.is_3d) and self.plane_2d != "xy":
            raise NotImplementedError(
                "extract_port_waves currently supports 2D simulations in the xy plane."
            )

        port_map = self._normalize_portspecs(ports)
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if freqs.size == 0:
            raise ValueError("frequencies must contain at least one value.")
        if np.any(freqs <= 0):
            raise ValueError("frequencies must be strictly positive.")

        strategy = str(mode_strategy).lower()
        if strategy not in {"per_frequency", "single", "single_frequency", "center"}:
            raise ValueError(
                f"Unsupported mode_strategy '{mode_strategy}'. "
                "Use 'per_frequency' or 'single'."
            )
        single_freq = float(np.median(freqs))

        monitor_by_name = self._named_monitors()
        for spec in port_map.values():
            if spec.monitor_name not in monitor_by_name:
                raise ValueError(f"Missing monitor '{spec.monitor_name}' for port '{spec.name}'.")
            if spec.reference_monitor and spec.reference_monitor not in monitor_by_name:
                raise ValueError(
                    f"Missing reference monitor '{spec.reference_monitor}' for port '{spec.name}'."
                )

        spectrum_cache = {}
        projection_cache = {}
        waves = {}
        for spec in port_map.values():
            main_monitor = monitor_by_name[spec.monitor_name]
            parts = self._mode_components_for_port(spec)
            for comp in (parts["e_component"], parts["h_component"]):
                key = (main_monitor.name, comp)
                if key not in spectrum_cache:
                    _, spectrum_cache[key] = self._sample_monitor_component_spectrum(
                        main_monitor, comp, frequencies=freqs, window=window
                    )

            a_plus = np.zeros(freqs.size, dtype=np.complex128)
            a_minus = np.zeros(freqs.size, dtype=np.complex128)
            for idx, f in enumerate(freqs):
                f_mode = float(f if strategy == "per_frequency" else single_freq)
                proj = self._build_port_projection(
                    spec, main_monitor, f_mode, projection_cache
                )
                field_vec = np.concatenate(
                    [
                        spectrum_cache[(main_monitor.name, proj["e_component"])][idx],
                        spectrum_cache[(main_monitor.name, proj["h_component"])][idx],
                    ]
                )
                coeff = proj["pinv"] @ field_vec
                a_plus[idx], a_minus[idx] = coeff[0], coeff[1]

            port_waves = {"a_plus": a_plus, "a_minus": a_minus}
            if return_power:
                port_waves["P_plus"] = np.abs(a_plus) ** 2
                port_waves["P_minus"] = np.abs(a_minus) ** 2

            if spec.reference_monitor:
                ref_monitor = monitor_by_name[spec.reference_monitor]
                for comp in (parts["e_component"], parts["h_component"]):
                    key = (ref_monitor.name, comp)
                    if key not in spectrum_cache:
                        _, spectrum_cache[key] = self._sample_monitor_component_spectrum(
                            ref_monitor, comp, frequencies=freqs, window=window
                        )

                a_incident = np.zeros(freqs.size, dtype=np.complex128)
                for idx, f in enumerate(freqs):
                    f_mode = float(f if strategy == "per_frequency" else single_freq)
                    proj = self._build_port_projection(
                        spec, ref_monitor, f_mode, projection_cache
                    )
                    field_vec = np.concatenate(
                        [
                            spectrum_cache[(ref_monitor.name, proj["e_component"])][idx],
                            spectrum_cache[(ref_monitor.name, proj["h_component"])][idx],
                        ]
                    )
                    coeff = proj["pinv"] @ field_vec
                    a_incident[idx] = coeff[0]
                port_waves["a_incident"] = a_incident
                if return_power:
                    port_waves["P_incident"] = np.abs(a_incident) ** 2

            waves[spec.name] = port_waves
        return waves

    def extract_port_waves_dft(
        self,
        ports,
        frequencies,
        min_incident_db=-40.0,
        return_power=True,
    ):
        """Extract modal port waves from in-simulation DFT monitor accumulators."""
        del min_incident_db  # Used in get_S_matrix_modal_dft validity masking.
        if (not self.is_3d) and self.plane_2d != "xy":
            raise NotImplementedError(
                "extract_port_waves_dft currently supports 2D simulations in the xy plane."
            )

        port_map = self._normalize_portspecs(ports)
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if freqs.size == 0:
            raise ValueError("frequencies must contain at least one value.")
        if np.any(freqs <= 0):
            raise ValueError("frequencies must be strictly positive.")

        monitor_by_name = self._named_monitors()
        for spec in port_map.values():
            main = monitor_by_name.get(spec.monitor_name)
            if main is None:
                raise ValueError(f"Missing monitor '{spec.monitor_name}' for port '{spec.name}'.")
            if not getattr(main, "dft_enabled", False):
                raise ValueError(
                    f"Monitor '{spec.monitor_name}' must be created with dft_enabled=True."
                )
            if spec.reference_monitor:
                ref = monitor_by_name.get(spec.reference_monitor)
                if ref is None:
                    raise ValueError(
                        f"Missing reference monitor '{spec.reference_monitor}' for port '{spec.name}'."
                    )
                if not getattr(ref, "dft_enabled", False):
                    raise ValueError(
                        f"Reference monitor '{spec.reference_monitor}' must have dft_enabled=True."
                    )

        dft_cache = {}
        projection_cache = {}
        waves = {}
        for spec in port_map.values():
            parts = self._mode_components_for_port(spec)
            main_monitor = monitor_by_name[spec.monitor_name]
            for comp in (parts["e_component"], parts["h_component"]):
                key = (main_monitor.name, comp)
                if key not in dft_cache:
                    _, dft_cache[key] = self._sample_monitor_component_dft(
                        main_monitor, comp, frequencies=freqs
                    )

            a_plus = np.zeros(freqs.size, dtype=np.complex128)
            a_minus = np.zeros(freqs.size, dtype=np.complex128)
            cond_main = np.zeros(freqs.size, dtype=float)
            for idx, f in enumerate(freqs):
                proj = self._build_port_projection(
                    spec, main_monitor, float(f), projection_cache
                )
                field_vec = np.concatenate(
                    [
                        dft_cache[(main_monitor.name, proj["e_component"])][idx],
                        dft_cache[(main_monitor.name, proj["h_component"])][idx],
                    ]
                )
                coeff = proj["pinv"] @ field_vec
                a_plus[idx], a_minus[idx] = coeff[0], coeff[1]
                cond_main[idx] = float(proj.get("condition_number", np.nan))

            port_waves = {
                "a_plus": a_plus,
                "a_minus": a_minus,
                "condition_number": cond_main,
            }
            if return_power:
                port_waves["P_plus"] = np.abs(a_plus) ** 2
                port_waves["P_minus"] = np.abs(a_minus) ** 2

            if spec.reference_monitor:
                ref_monitor = monitor_by_name[spec.reference_monitor]
                for comp in (parts["e_component"], parts["h_component"]):
                    key = (ref_monitor.name, comp)
                    if key not in dft_cache:
                        _, dft_cache[key] = self._sample_monitor_component_dft(
                            ref_monitor, comp, frequencies=freqs
                        )
                a_incident = np.zeros(freqs.size, dtype=np.complex128)
                cond_ref = np.zeros(freqs.size, dtype=float)
                for idx, f in enumerate(freqs):
                    proj = self._build_port_projection(
                        spec, ref_monitor, float(f), projection_cache
                    )
                    field_vec = np.concatenate(
                        [
                            dft_cache[(ref_monitor.name, proj["e_component"])][idx],
                            dft_cache[(ref_monitor.name, proj["h_component"])][idx],
                        ]
                    )
                    coeff = proj["pinv"] @ field_vec
                    a_incident[idx] = coeff[0]
                    cond_ref[idx] = float(proj.get("condition_number", np.nan))
                port_waves["a_incident"] = a_incident
                port_waves["reference_condition_number"] = cond_ref
                if return_power:
                    port_waves["P_incident"] = np.abs(a_incident) ** 2

            waves[spec.name] = port_waves
        return waves

    def get_S_matrix_modal_dft(
        self,
        source_port,
        ports,
        output_ports=None,
        frequencies=None,
        as_sax=True,
        return_diagnostics=True,
        min_incident_db=-40.0,
    ):
        """Broadband modal S extraction from in-simulation DFT monitor accumulators."""
        port_map = self._normalize_portspecs(ports)
        if source_port not in port_map:
            raise ValueError(f"source_port '{source_port}' not found in ports.")

        monitor_by_name = self._named_monitors()
        if frequencies is None:
            src_spec = port_map[source_port]
            ref_name = src_spec.reference_monitor or src_spec.monitor_name
            src_monitor = monitor_by_name.get(ref_name)
            if src_monitor is None:
                raise ValueError(f"Missing source/reference monitor '{ref_name}'.")
            frequencies = src_monitor.get_dft_frequencies()
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

        waves = self.extract_port_waves_dft(
            ports=port_map.values(),
            frequencies=frequencies,
            min_incident_db=min_incident_db,
            return_power=True,
        )

        if output_ports is None:
            output_ports = list(port_map.keys())
        else:
            output_ports = list(output_ports)
        missing = [name for name in output_ports if name not in port_map]
        if missing:
            raise ValueError(f"output_ports contains unknown ports: {missing}")

        a_incident = np.asarray(
            waves[source_port].get("a_incident", waves[source_port]["a_plus"]),
            dtype=np.complex128,
        )
        max_incident = float(np.max(np.abs(a_incident))) if a_incident.size else 0.0
        rel_floor = max_incident * (10.0 ** (float(min_incident_db) / 20.0))
        abs_floor = max(1e-18, rel_floor)
        valid_mask = np.abs(a_incident) >= abs_floor

        s_matrix = {}
        for out_port in output_ports:
            b_out = np.asarray(waves[out_port]["a_minus"], dtype=np.complex128)
            ratio = self._safe_ratio(b_out, a_incident)
            ratio = np.where(valid_mask, ratio, 0.0 + 0.0j)
            s_matrix[(out_port, source_port)] = ratio

        self.s_matrix_frequencies = np.asarray(frequencies, dtype=float)
        if as_sax:
            try:
                import sax
            except ImportError as exc:
                raise ImportError(
                    "sax is required for as_sax=True. Install it with `pip install sax`."
                ) from exc
            s_output = sax.sdict(s_matrix)
        else:
            s_output = s_matrix

        if not return_diagnostics:
            return s_output

        p_in = np.abs(a_incident) ** 2
        p_guided_out = np.zeros_like(p_in, dtype=float)
        for out_port in output_ports:
            p_guided_out += np.abs(waves[out_port]["a_minus"]) ** 2
        power_sum = p_guided_out / np.maximum(p_in, 1e-18)
        loss_est = 1.0 - power_sum
        power_sum = np.where(valid_mask, power_sum, np.nan)
        loss_est = np.where(valid_mask, loss_est, np.nan)

        diagnostics = {
            "frequencies": np.asarray(frequencies, dtype=float),
            "source_port": source_port,
            "output_ports": output_ports,
            "waves": waves,
            "P_in": p_in,
            "P_guided_out": p_guided_out,
            "power_sum": power_sum,
            "loss_est": loss_est,
            "valid_mask": valid_mask,
            "condition_numbers": {
                name: {
                    "monitor": np.asarray(data.get("condition_number", []), dtype=float),
                    "reference": np.asarray(
                        data.get("reference_condition_number", []), dtype=float
                    ),
                }
                for name, data in waves.items()
            },
        }
        return {"s_matrix": s_output, "diagnostics": diagnostics}

    def extract_port_waves_cw(
        self,
        ports,
        frequency,
        steady_start_time=None,
        avg_cycles=12,
        window="hann",
        mode_strategy="per_frequency",
        return_power=True,
    ):
        """CW modal extraction at one frequency using complex demodulation."""
        if self.is_3d or self.plane_2d != "xy":
            raise NotImplementedError(
                "extract_port_waves_cw currently supports 2D simulations in the xy plane."
            )

        port_map = self._normalize_portspecs(ports)
        f = float(frequency)
        if not np.isfinite(f) or f <= 0:
            raise ValueError(f"frequency must be positive, got {frequency!r}")

        strategy = str(mode_strategy).lower()
        if strategy not in {"per_frequency", "single", "single_frequency", "center"}:
            raise ValueError(
                f"Unsupported mode_strategy '{mode_strategy}'. "
                "Use 'per_frequency' or 'single'."
            )
        f_mode = f

        monitor_by_name = self._named_monitors()
        for spec in port_map.values():
            if spec.monitor_name not in monitor_by_name:
                raise ValueError(f"Missing monitor '{spec.monitor_name}' for port '{spec.name}'.")
            if spec.reference_monitor and spec.reference_monitor not in monitor_by_name:
                raise ValueError(
                    f"Missing reference monitor '{spec.reference_monitor}' for port '{spec.name}'."
                )

        projection_cache = {}
        waves = {}
        for spec in port_map.values():
            parts = self._mode_components_for_port(spec)
            main_monitor = monitor_by_name[spec.monitor_name]
            proj = self._build_port_projection(
                spec,
                main_monitor,
                f_mode if strategy == "per_frequency" else f,
                projection_cache,
            )
            e_main = self._demodulate_monitor_component(
                main_monitor,
                parts["e_component"],
                frequency=f,
                t_start=steady_start_time,
                avg_cycles=avg_cycles,
                window=window,
            )
            h_main = self._demodulate_monitor_component(
                main_monitor,
                parts["h_component"],
                frequency=f,
                t_start=steady_start_time,
                avg_cycles=avg_cycles,
                window=window,
            )
            coeff = proj["pinv"] @ np.concatenate([e_main, h_main])
            a_plus = np.complex128(coeff[0])
            a_minus = np.complex128(coeff[1])
            port_waves = {"a_plus": a_plus, "a_minus": a_minus}
            if return_power:
                port_waves["P_plus"] = float(np.abs(a_plus) ** 2)
                port_waves["P_minus"] = float(np.abs(a_minus) ** 2)

            if spec.reference_monitor:
                ref_monitor = monitor_by_name[spec.reference_monitor]
                ref_proj = self._build_port_projection(
                    spec,
                    ref_monitor,
                    f_mode if strategy == "per_frequency" else f,
                    projection_cache,
                )
                e_ref = self._demodulate_monitor_component(
                    ref_monitor,
                    parts["e_component"],
                    frequency=f,
                    t_start=steady_start_time,
                    avg_cycles=avg_cycles,
                    window=window,
                )
                h_ref = self._demodulate_monitor_component(
                    ref_monitor,
                    parts["h_component"],
                    frequency=f,
                    t_start=steady_start_time,
                    avg_cycles=avg_cycles,
                    window=window,
                )
                ref_coeff = ref_proj["pinv"] @ np.concatenate([e_ref, h_ref])
                a_incident = np.complex128(ref_coeff[0])
                port_waves["a_incident"] = a_incident
                if return_power:
                    port_waves["P_incident"] = float(np.abs(a_incident) ** 2)

            waves[spec.name] = port_waves
        return waves

    def get_S_matrix_modal(
        self,
        source_port,
        ports,
        output_ports=None,
        frequencies=None,
        mode_strategy="per_frequency",
        as_sax=True,
        return_diagnostics=True,
    ):
        """Broadband modal S-matrix extraction from FFT-sampled monitor spectra.

        This method is fast and useful for exploratory sweeps. For strict
        passivity/loss checks, prefer get_S_matrix_modal_cw(...).
        """
        port_map = self._normalize_portspecs(ports)
        if source_port not in port_map:
            raise ValueError(f"source_port '{source_port}' not found in ports.")

        monitor_by_name = self._named_monitors()
        if frequencies is None:
            src_spec = port_map[source_port]
            ref_name = src_spec.reference_monitor or src_spec.monitor_name
            src_monitor = monitor_by_name.get(ref_name)
            if src_monitor is None:
                raise ValueError(f"Missing source/reference monitor '{ref_name}'.")
            src_parts = self._mode_components_for_port(src_spec)
            frequencies, _ = self._sample_monitor_component_spectrum(
                src_monitor, src_parts["e_component"], frequencies=None, window="hann"
            )
        else:
            frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

        waves = self.extract_port_waves(
            ports=port_map.values(),
            frequencies=frequencies,
            mode_strategy=mode_strategy,
            window="hann",
            return_power=True,
        )

        if output_ports is None:
            output_ports = list(port_map.keys())
        else:
            output_ports = list(output_ports)
        missing = [name for name in output_ports if name not in port_map]
        if missing:
            raise ValueError(f"output_ports contains unknown ports: {missing}")

        a_incident = waves[source_port].get("a_incident", waves[source_port]["a_plus"])
        s_matrix = {}
        for out_port in output_ports:
            b_out = waves[out_port]["a_minus"]
            s_matrix[(out_port, source_port)] = self._safe_ratio(b_out, a_incident)

        self.s_matrix_frequencies = np.asarray(frequencies, dtype=float)
        if as_sax:
            try:
                import sax
            except ImportError as exc:
                raise ImportError(
                    "sax is required for as_sax=True. Install it with `pip install sax`."
                ) from exc
            s_output = sax.sdict(s_matrix)
        else:
            s_output = s_matrix

        if not return_diagnostics:
            return s_output

        p_in = np.abs(a_incident) ** 2
        p_guided_out = np.zeros_like(p_in, dtype=float)
        for out_port in output_ports:
            p_guided_out += np.abs(waves[out_port]["a_minus"]) ** 2
        power_sum = p_guided_out / np.maximum(p_in, 1e-18)
        diagnostics = {
            "frequencies": np.asarray(frequencies, dtype=float),
            "source_port": source_port,
            "output_ports": output_ports,
            "waves": waves,
            "P_in": p_in,
            "P_guided_out": p_guided_out,
            "power_sum": power_sum,
            "loss_est": 1.0 - power_sum,
        }
        return {"s_matrix": s_output, "diagnostics": diagnostics}

    def get_S_matrix_modal_cw(
        self,
        source_port,
        ports,
        output_ports=None,
        frequency=None,
        steady_start_time=None,
        avg_cycles=12,
        window="hann",
        mode_strategy="per_frequency",
        as_sax=True,
        return_diagnostics=True,
    ):
        """CW modal S extraction for one source/one frequency.

        Recommended when physically reliable passivity/loss diagnostics matter.
        """
        if frequency is None:
            raise ValueError("frequency is required for get_S_matrix_modal_cw.")

        port_map = self._normalize_portspecs(ports)
        if source_port not in port_map:
            raise ValueError(f"source_port '{source_port}' not found in ports.")

        waves = self.extract_port_waves_cw(
            ports=port_map.values(),
            frequency=frequency,
            steady_start_time=steady_start_time,
            avg_cycles=avg_cycles,
            window=window,
            mode_strategy=mode_strategy,
            return_power=True,
        )

        if output_ports is None:
            output_ports = list(port_map.keys())
        else:
            output_ports = list(output_ports)
        missing = [name for name in output_ports if name not in port_map]
        if missing:
            raise ValueError(f"output_ports contains unknown ports: {missing}")

        a_incident = waves[source_port].get("a_incident", waves[source_port]["a_plus"])
        s_matrix = {}
        for out_port in output_ports:
            b_out = waves[out_port]["a_minus"]
            ratio = self._safe_ratio(np.asarray([b_out]), np.asarray([a_incident]))[0]
            s_matrix[(out_port, source_port)] = np.complex128(ratio)

        self.s_matrix_frequencies = np.asarray([float(frequency)], dtype=float)
        if as_sax:
            try:
                import sax
            except ImportError as exc:
                raise ImportError(
                    "sax is required for as_sax=True. Install it with `pip install sax`."
                ) from exc
            s_output = sax.sdict(s_matrix)
        else:
            s_output = s_matrix

        if not return_diagnostics:
            return s_output

        p_in = float(np.abs(a_incident) ** 2)
        p_guided_out = float(
            np.sum([np.abs(waves[out]["a_minus"]) ** 2 for out in output_ports])
        )
        power_sum = p_guided_out / max(p_in, 1e-18)
        diagnostics = {
            "frequency": float(frequency),
            "source_port": source_port,
            "output_ports": output_ports,
            "waves": waves,
            "P_in": p_in,
            "P_guided_out": p_guided_out,
            "power_sum": power_sum,
            "loss_est": 1.0 - power_sum,
        }
        return {"s_matrix": s_output, "diagnostics": diagnostics}

    def get_s_matrix_modal(self, *args, **kwargs):
        return self.get_S_matrix_modal(*args, **kwargs)

    def get_s_matrix_modal_dft(self, *args, **kwargs):
        return self.get_S_matrix_modal_dft(*args, **kwargs)

    def get_s_matrix_modal_cw(self, *args, **kwargs):
        return self.get_S_matrix_modal_cw(*args, **kwargs)

    def get_S_matrix(self, *args, **kwargs):
        raise RuntimeError(
            "Simulation.get_S_matrix(...) is deprecated and removed. "
            "Use Simulation.get_S_matrix_modal(...)."
        )

    def get_s_matrix(self, *args, **kwargs):
        raise RuntimeError(
            "Simulation.get_s_matrix(...) is deprecated and removed. "
            "Use Simulation.get_s_matrix_modal(...)."
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
