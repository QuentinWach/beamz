import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import µm
from beamz.design.core import Design
from beamz.devices.monitors.monitors import Monitor
from beamz.devices.sources.mode import (
    _make_3d_mode_basis_profiles,
    _modal_overlap_3d_profiles,
)
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
from beamz.simulation.modal import (
    _build_port_projection as _build_port_projection_impl,
    modal_power_3d as _modal_power_3d_impl,
    monitor_profile_slice as _monitor_profile_slice_impl,
    _project_modal_coefficients_3d as _project_modal_coefficients_3d_impl,
    remap_3d_solver_components as _remap_3d_solver_components_impl,
)
from beamz.simulation.ports import (
    format_s_matrix_output as _format_s_matrix_output_impl,
    mode_components_for_port as _mode_components_for_port_impl,
    named_monitors as _named_monitors_impl,
    normalize_portspecs as _normalize_portspecs_impl,
    safe_ratio as _safe_ratio_impl,
    select_wave_component as _select_wave_component_impl,
)
from beamz.simulation.spectral import (
    demodulate_monitor_component as _demodulate_monitor_component_impl,
    get_monitor_trace as _get_monitor_trace_impl,
    monitor_projection_phase as _monitor_projection_phase_impl,
    resample_complex_matrix as _resample_complex_matrix_impl,
    sample_monitor_component_dft as _sample_monitor_component_dft_impl,
    sample_monitor_component_spectrum as _sample_monitor_component_spectrum_impl,
)
from beamz.simulation.sparams import (
    extract_port_waves as _extract_port_waves_impl,
    extract_port_waves_cw as _extract_port_waves_cw_impl,
    extract_port_waves_dft as _extract_port_waves_dft_impl,
    get_s_matrix_modal as _get_s_matrix_modal_impl,
    get_s_matrix_modal_cw as _get_s_matrix_modal_cw_impl,
    get_s_matrix_modal_dft as _get_s_matrix_modal_dft_impl,
)
from beamz.simulation.step import (
    collect_source_terms as _collect_source_terms_impl,
    inject_e_sources as _inject_e_sources_impl,
    inject_h_sources as _inject_h_sources_impl,
    inject_legacy_sources as _inject_legacy_sources_impl,
    record_monitors as _record_monitors_impl,
    run_step as _run_step_impl,
)
from beamz.simulation.view import (
    run as _run_impl,
    show as _show_impl,
    to_scene as _to_scene_impl,
)


@dataclass(frozen=True)
class PortSpec:
    name: str
    monitor_name: str
    direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"]
    polarization: Literal["tm", "te"]
    mode_index: int = 0
    reference_monitor: str | None = None
    incident_wave: Literal["plus", "minus", "auto"] = "plus"
    scattered_wave: Literal["plus", "minus", "auto"] = "minus"


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
        return _run_step_impl(self)

    def _record_monitors(self):
        """Record data from Monitor devices during simulation."""
        _record_monitors_impl(self)

    def _inject_h_sources(self):
        """Inject magnetic currents (M) into H-fields after H update."""
        _inject_h_sources_impl(self)

    def _inject_e_sources(self):
        """Inject electric currents (J) into E-fields after E update."""
        _inject_e_sources_impl(self)

    def _inject_legacy_sources(self):
        """Inject from devices that only have inject() (no inject_h/inject_e)."""
        _inject_legacy_sources_impl(self)

    def _collect_source_terms(self):
        """Collect electric and magnetic current sources from all devices."""
        return _collect_source_terms_impl(self)

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

    def run_compiled_until_decay(
        self,
        monitors,
        *,
        min_time_s=0.0,
        chunk_steps=None,
        lookback_records=12,
        decay_ratio=1e-3,
        progress=True,
    ):
        """Run compiled chunks until monitor power decays after a minimum time."""
        total_steps = int(self.num_steps - self.current_step)
        if total_steps <= 0:
            return 0
        dt = float(self.dt)
        chunk_steps = (
            max(64, min(512, int(np.ceil(total_steps / 24.0))))
            if chunk_steps is None
            else max(1, int(chunk_steps))
        )
        lookback_records = max(2, int(lookback_records))
        min_steps = int(np.ceil(max(0.0, float(min_time_s)) / max(dt, 1e-30)))
        steps_done = 0
        peak = 0.0

        while steps_done < total_steps:
            this_chunk = min(chunk_steps, total_steps - steps_done)
            self.run_compiled(num_steps=this_chunk, progress=False)
            steps_done += this_chunk

            histories = [
                np.abs(np.asarray(mon.power_history, dtype=np.float64))
                for mon in monitors
                if len(mon.power_history)
            ]
            tail = np.inf
            if histories:
                peak = max(peak, max(float(np.max(hist)) for hist in histories))
                tail = max(
                    float(np.max(hist[-lookback_records:])) for hist in histories
                )

            if progress:
                pct = 100.0 * steps_done / max(total_steps, 1)
                print(
                    f"\r● Progress: {pct:.0f}% ({steps_done}/{total_steps} steps)",
                    end="",
                    flush=True,
                )

            if (
                steps_done >= min_steps
                and peak > 0.0
                and np.isfinite(tail)
                and tail <= float(decay_ratio) * peak
            ):
                break

        if progress:
            print()
        return steps_done

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
        return _get_monitor_trace_impl(
            self,
            monitor,
            field_component=field_component,
            reduction=reduction,
        )

    @staticmethod
    def _safe_ratio(num, den, eps=1e-18):
        return _safe_ratio_impl(num, den, eps=eps)

    @staticmethod
    def _select_wave_component(
        wave_data,
        selector="minus",
        *,
        use_reference=False,
    ):
        return _select_wave_component_impl(
            wave_data,
            selector=selector,
            use_reference=use_reference,
        )

    @staticmethod
    def _format_s_matrix_output(s_matrix, as_sax):
        return _format_s_matrix_output_impl(s_matrix, as_sax)

    @staticmethod
    def _normalize_portspecs(ports):
        return _normalize_portspecs_impl(ports, PortSpec)

    def _named_monitors(self):
        return _named_monitors_impl(self.devices)

    def _sample_monitor_component_spectrum(
        self,
        monitor,
        component,
        frequencies=None,
        window="hann",
    ):
        return _sample_monitor_component_spectrum_impl(
            self,
            monitor,
            component,
            frequencies=frequencies,
            window=window,
        )

    @staticmethod
    def _resample_complex_matrix(freq_src, values_src, freq_dst):
        return _resample_complex_matrix_impl(freq_src, values_src, freq_dst)

    @staticmethod
    def _monitor_projection_phase(component, frequencies, dt):
        return _monitor_projection_phase_impl(component, frequencies, dt)

    def _sample_monitor_component_dft(self, monitor, component, frequencies):
        return _sample_monitor_component_dft_impl(
            self, monitor, component, frequencies
        )

    def _demodulate_monitor_component(
        self,
        monitor,
        component,
        frequency,
        t_start=None,
        avg_cycles=12,
        window="hann",
    ):
        return _demodulate_monitor_component_impl(
            self,
            monitor,
            component,
            frequency,
            t_start=t_start,
            avg_cycles=avg_cycles,
            window=window,
        )

    @staticmethod
    def _mode_components_for_port(spec):
        return _mode_components_for_port_impl(spec)

    @staticmethod
    def _remap_3d_solver_components(ex, ey, ez, hx, hy, hz, axis):
        return _remap_3d_solver_components_impl(ex, ey, ez, hx, hy, hz, axis)

    def _monitor_profile_slice(self, monitor, axis, pad_cells):
        return _monitor_profile_slice_impl(self, monitor, axis, pad_cells)

    def _build_port_projection(self, spec, monitor, frequency, cache, mode_pad_cells=6):
        return _build_port_projection_impl(
            self,
            spec,
            monitor,
            frequency,
            cache,
            mode_pad_cells=mode_pad_cells,
            solve_modes_fn=solve_modes,
            mode_basis_builder=_make_3d_mode_basis_profiles,
            modal_overlap_fn=_modal_overlap_3d_profiles,
        )

    @staticmethod
    def _modal_power_3d(mode_components, axis, d_area):
        return _modal_power_3d_impl(mode_components, axis, d_area)

    @staticmethod
    def _project_modal_coefficients_3d(
        field_components, projection, apply_calibration=True
    ):
        return _project_modal_coefficients_3d_impl(
            field_components,
            projection,
            apply_calibration=apply_calibration,
            modal_overlap_fn=_modal_overlap_3d_profiles,
        )

    def extract_port_waves(
        self,
        ports,
        frequencies,
        mode_strategy="per_frequency",
        window="hann",
        return_power=True,
    ):
        return _extract_port_waves_impl(
            self,
            ports,
            frequencies,
            mode_strategy=mode_strategy,
            window=window,
            return_power=return_power,
        )

    def extract_port_waves_dft(
        self,
        ports,
        frequencies,
        min_incident_db=-40.0,
        return_power=True,
    ):
        return _extract_port_waves_dft_impl(
            self,
            ports,
            frequencies,
            min_incident_db=min_incident_db,
            return_power=return_power,
        )

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
        return _get_s_matrix_modal_dft_impl(
            self,
            source_port,
            ports,
            output_ports=output_ports,
            frequencies=frequencies,
            as_sax=as_sax,
            return_diagnostics=return_diagnostics,
            min_incident_db=min_incident_db,
        )

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
        return _extract_port_waves_cw_impl(
            self,
            ports,
            frequency,
            steady_start_time=steady_start_time,
            avg_cycles=avg_cycles,
            window=window,
            mode_strategy=mode_strategy,
            return_power=return_power,
        )

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
        return _get_s_matrix_modal_impl(
            self,
            source_port,
            ports,
            output_ports=output_ports,
            frequencies=frequencies,
            mode_strategy=mode_strategy,
            as_sax=as_sax,
            return_diagnostics=return_diagnostics,
        )

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
        return _get_s_matrix_modal_cw_impl(
            self,
            source_port,
            ports,
            output_ports=output_ports,
            frequency=frequency,
            steady_start_time=steady_start_time,
            avg_cycles=avg_cycles,
            window=window,
            mode_strategy=mode_strategy,
            as_sax=as_sax,
            return_diagnostics=return_diagnostics,
        )

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
        return _run_impl(self, **kwargs)

    def to_scene(self):
        """Build a 3D scene representation of the simulation setup."""
        return _to_scene_impl(self)

    def show(self, *, mode="auto", open_browser=True, **kwargs):
        """Display the simulation setup in the interactive 3D scene viewer."""
        return _show_impl(
            self, mode=mode, open_browser=open_browser, **kwargs
        )
