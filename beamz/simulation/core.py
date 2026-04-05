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
from beamz.simulation import jit, runtime
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
        return jit.create_step(self)

    def _create_jit_step_h(self):
        """Create a JIT-compiled H-update function."""
        return jit.create_step_h(self)

    def _create_jit_step_e(self):
        """Create a JIT-compiled E-update function."""
        return jit.create_step_e(self)

    def compile(self, num_steps=None):
        """Compile the v0.3 packed-data simulation program."""
        return runtime.compile_program(self, num_steps=num_steps)

    def run_compiled(
        self, num_steps=None, record_interval=None, record_fields=None, progress=True
    ):
        return runtime.run_compiled(
            self,
            num_steps=num_steps,
            record_interval=record_interval,
            record_fields=record_fields,
            progress=progress,
        )

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
        return runtime.run_compiled_until_decay(
            self,
            monitors,
            min_time_s=min_time_s,
            chunk_steps=chunk_steps,
            lookback_records=lookback_records,
            decay_ratio=decay_ratio,
            progress=progress,
        )

    def run_fast(
        self, num_steps=None, record_interval=None, record_fields=None, progress=True
    ):
        """Backward-compatible alias to `run_compiled` in v0.3."""
        return runtime.run_fast(
            self,
            num_steps=num_steps,
            record_interval=record_interval,
            record_fields=record_fields,
            progress=progress,
        )

    def run_jit_scan(self, num_steps=None, progress=True):
        """Backward-compatible alias to `run_compiled` in v0.3."""
        return runtime.run_jit_scan(self, num_steps=num_steps, progress=progress)

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
