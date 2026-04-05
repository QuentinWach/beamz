from dataclasses import dataclass
from typing import Literal

import numpy as np

from beamz.const import µm
from beamz.design.core import Design
from beamz.devices.sources.mode import (
    _make_3d_mode_basis_profiles,
    _modal_overlap_3d_profiles,
)
from beamz.devices.sources.solve import solve_modes
from beamz.simulation.boundaries import PML, Boundary
from beamz.simulation.fields import Fields
from beamz.simulation.spec import SimulationSpec, build_simulation_spec
from beamz.simulation.state import SimulationRuntime
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

    _SPEC_FIELDS = frozenset(SimulationSpec.__dataclass_fields__.keys())
    _RUNTIME_MAP = {
        "fields": "fields",
        "dt": "dt",
        "num_steps": "num_steps",
        "t": "t",
        "current_step": "current_step",
        "pml_data": "pml_data",
        "_compiled_program": "compiled_program",
        "_compiled_program_signature": "compiled_program_signature",
        "_compiled_program_cache": "compiled_program_cache",
        "_compiled_monitor_state": "compiled_monitor_state",
    }

    def __init__(
        self,
        design: Design = None,
        devices: list = None,
        boundaries: list[Boundary] = None,
        resolution: float = 0.02 * µm,
        time: np.ndarray = None,
        plane_2d: str = "xy",
    ):
        object.__setattr__(
            self,
            "spec",
            build_simulation_spec(
                design=design,
                devices=devices,
                boundaries=boundaries,
                resolution=resolution,
                time=time,
                plane_2d=plane_2d,
            ),
        )
        object.__setattr__(self, "runtime", SimulationRuntime())

        # Get material grids from design (design owns the material grids, we reference them)
        permittivity, conductivity, permeability = self.design.get_material_grids(
            self.resolution
        )

        runtime = self.runtime
        runtime.dt = float(self.time[1] - self.time[0])
        runtime.num_steps = len(self.time)
        runtime.t = float(self.time[0])
        runtime.current_step = 0

        # Check for PML boundaries before creating fields (to avoid double material init)
        pml_boundaries = [b for b in self.boundaries if isinstance(b, PML)]

        # Create field storage (fields owns the E/H field arrays, references material grids)
        runtime.fields = Fields(
            permittivity,
            conductivity,
            permeability,
            self.resolution,
            plane_2d=self.plane_2d,
            _init_materials=not pml_boundaries,
        )

        # Initialize PML regions if present
        if pml_boundaries:
            # Create PML regions (do this once, not every timestep)
            pml_data = {}
            for pml in pml_boundaries:
                new_data = pml.create_pml_regions(
                    runtime.fields,
                    self.design,
                    self.resolution,
                    self.dt,
                    plane_2d=self.plane_2d,
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
            runtime.pml_data = pml_data

            # Set effective conductivity for PML
            runtime.fields.set_pml_conductivity(pml_data)
        else:
            runtime.pml_data = None

    def __getattr__(self, name):
        spec = self.__dict__.get("spec")
        if spec is not None and hasattr(spec, name):
            return getattr(spec, name)
        runtime = self.__dict__.get("runtime")
        if runtime is not None and name in self._RUNTIME_MAP:
            return getattr(runtime, self._RUNTIME_MAP[name])
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in {"spec", "runtime"}:
            object.__setattr__(self, name, value)
            return
        if name in self._SPEC_FIELDS and "spec" in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__!s}.{name} is derived from immutable SimulationSpec"
            )
        if name in self._RUNTIME_MAP and "runtime" in self.__dict__:
            setattr(self.runtime, self._RUNTIME_MAP[name], value)
            return
        object.__setattr__(self, name, value)


def _run_fast(sim, num_steps=None, record_interval=None, record_fields=None, progress=True):
    """Backward-compatible alias to `run_compiled` in v0.3."""
    return runtime.run_compiled(
        sim,
        num_steps=num_steps,
        record_interval=record_interval,
        record_fields=record_fields,
        progress=progress,
    )


def _run_jit_scan(sim, num_steps=None, progress=True):
    """Backward-compatible alias to `run_compiled` in v0.3."""
    return runtime.run_compiled(
        sim,
        num_steps=num_steps,
        record_interval=None,
        record_fields=None,
        progress=progress,
    )


def _normalize_portspecs(ports):
    return _normalize_portspecs_impl(ports, PortSpec)


def _named_monitors(sim):
    return _named_monitors_impl(sim.devices)


def _build_port_projection(sim, spec, monitor, frequency, cache, mode_pad_cells=6):
    return _build_port_projection_impl(
        sim,
        spec,
        monitor,
        frequency,
        cache,
        mode_pad_cells=mode_pad_cells,
        solve_modes_fn=solve_modes,
        mode_basis_builder=_make_3d_mode_basis_profiles,
        modal_overlap_fn=_modal_overlap_3d_profiles,
    )


def _project_modal_coefficients_3d(
    field_components, projection, apply_calibration=True
):
    return _project_modal_coefficients_3d_impl(
        field_components,
        projection,
        apply_calibration=apply_calibration,
        modal_overlap_fn=_modal_overlap_3d_profiles,
    )


def _deprecated_get_s_matrix(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "Simulation.get_S_matrix(...) is deprecated and removed. "
        "Use Simulation.get_S_matrix_modal(...)."
    )


def _deprecated_get_s_matrix_lower(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "Simulation.get_s_matrix(...) is deprecated and removed. "
        "Use Simulation.get_s_matrix_modal(...)."
    )


def _monitor_trace(sim, monitor, field_component="Ez", reduction="mean"):
    from beamz.visual.data import Trace1D

    values, coords = _get_monitor_trace_impl(
        sim,
        monitor,
        field_component=field_component,
        reduction=reduction,
    )
    return Trace1D(
        values=values,
        coords=coords,
        coord_label="time",
        value_label=f"{field_component} ({reduction})",
        title=f"{getattr(monitor, 'name', 'monitor')}: {field_component}",
    )


Simulation.step = _run_step_impl
Simulation._record_monitors = _record_monitors_impl
Simulation._inject_h_sources = _inject_h_sources_impl
Simulation._inject_e_sources = _inject_e_sources_impl
Simulation._inject_legacy_sources = _inject_legacy_sources_impl
Simulation._collect_source_terms = _collect_source_terms_impl
Simulation._create_jit_step = jit.create_step
Simulation._create_jit_step_h = jit.create_step_h
Simulation._create_jit_step_e = jit.create_step_e
Simulation.compile = runtime.compile_program
Simulation.run_compiled = runtime.run_compiled
Simulation.run_compiled_until_decay = runtime.run_compiled_until_decay
Simulation.run_fast = _run_fast
Simulation.run_jit_scan = _run_jit_scan
Simulation._get_monitor_trace = _get_monitor_trace_impl
Simulation.monitor_trace = _monitor_trace
Simulation._safe_ratio = staticmethod(_safe_ratio_impl)
Simulation._select_wave_component = staticmethod(_select_wave_component_impl)
Simulation._format_s_matrix_output = staticmethod(_format_s_matrix_output_impl)
Simulation._normalize_portspecs = staticmethod(_normalize_portspecs)
Simulation._named_monitors = _named_monitors
Simulation._sample_monitor_component_spectrum = _sample_monitor_component_spectrum_impl
Simulation._resample_complex_matrix = staticmethod(_resample_complex_matrix_impl)
Simulation._monitor_projection_phase = staticmethod(_monitor_projection_phase_impl)
Simulation._sample_monitor_component_dft = _sample_monitor_component_dft_impl
Simulation._demodulate_monitor_component = _demodulate_monitor_component_impl
Simulation._mode_components_for_port = staticmethod(_mode_components_for_port_impl)
Simulation._remap_3d_solver_components = staticmethod(_remap_3d_solver_components_impl)
Simulation._monitor_profile_slice = _monitor_profile_slice_impl
Simulation._build_port_projection = _build_port_projection
Simulation._modal_power_3d = staticmethod(_modal_power_3d_impl)
Simulation._project_modal_coefficients_3d = staticmethod(_project_modal_coefficients_3d)
Simulation.extract_port_waves = _extract_port_waves_impl
Simulation.extract_port_waves_dft = _extract_port_waves_dft_impl
Simulation.get_S_matrix_modal_dft = _get_s_matrix_modal_dft_impl
Simulation.extract_port_waves_cw = _extract_port_waves_cw_impl
Simulation.get_S_matrix_modal = _get_s_matrix_modal_impl
Simulation.get_S_matrix_modal_cw = _get_s_matrix_modal_cw_impl
Simulation.get_s_matrix_modal = _get_s_matrix_modal_impl
Simulation.get_s_matrix_modal_dft = _get_s_matrix_modal_dft_impl
Simulation.get_s_matrix_modal_cw = _get_s_matrix_modal_cw_impl
Simulation.get_S_matrix = _deprecated_get_s_matrix
Simulation.get_s_matrix = _deprecated_get_s_matrix_lower
Simulation.run = _run_impl
Simulation.to_scene = _to_scene_impl
Simulation.show = _show_impl
