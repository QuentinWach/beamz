"""Static planning for configuration-specialized compiled simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from beamz.devices.compiler import validate_compilable_devices
from beamz.devices.monitors.compiler import (
    CompiledMonitorSpec,
    compile_monitor_specs,
)
from beamz.devices.monitors.monitors import Monitor
from beamz.devices.sources.compiler import (
    CompiledSourceSpec,
    compile_source_specs,
)
from beamz.simulation.boundaries import PML
from beamz.simulation.material_models import CompiledMaterialSpec

CompiledKernelFamily = Literal[
    "engine_only",
    "engine_plus_sources",
    "engine_plus_monitors",
    "engine_plus_sources_and_monitors",
    "engine_plus_material",
]


@dataclass(frozen=True)
class BoundaryPlan:
    family: str
    geometry: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class MaterialPlan:
    family: str
    thermal_family: str
    spec: CompiledMaterialSpec


@dataclass(frozen=True)
class SourcePlan:
    family: str
    timing_family: str
    spec_count: int
    slab_count: int


@dataclass(frozen=True)
class MonitorPlan:
    family: str
    geometry_family: str
    spec_count: int
    power_count: int
    frequency_count: int
    dft_count: int


@dataclass(frozen=True)
class OutputPlan:
    family: str


@dataclass(frozen=True)
class CompilationPlanKey:
    backend_platform: str
    dimension_family: str
    field_shape: tuple[int, ...]
    field_dtype: str
    num_steps: int
    loop_kind: str
    source_single_slab_dense: bool
    temporal_block_steps: int
    boundary_family: str
    boundary_geometry: tuple[tuple[str, int], ...]
    material_family: str
    thermal_family: str
    coefficient_layout_family: str
    source_family: str
    source_timing_family: str
    monitor_family: str
    monitor_geometry_family: str
    output_family: str
    kernel_family: CompiledKernelFamily


@dataclass(frozen=True)
class CompiledPlan:
    key: CompilationPlanKey
    boundary: BoundaryPlan
    material: MaterialPlan
    sources: SourcePlan
    monitors: MonitorPlan
    output: OutputPlan
    source_specs: tuple[CompiledSourceSpec, ...]
    monitor_specs: tuple[CompiledMonitorSpec, ...]
    monitor_devices: tuple[Monitor, ...]

    @property
    def kernel_family(self) -> CompiledKernelFamily:
        return self.key.kernel_family


def build_compilation_plan(
    simulation,
    *,
    backend_platform: str,
    num_steps: int,
    loop_kind: str,
    source_single_slab_dense: bool,
    temporal_block_steps: int,
) -> CompiledPlan:
    validate_compilable_devices(simulation.devices)
    fields = simulation.fields
    source_specs = compile_source_specs(
        devices=simulation.devices,
        fields=fields,
        dt=float(simulation.dt),
        resolution=float(simulation.resolution),
        num_steps=int(num_steps),
        t0=float(simulation.time[0]),
        total_steps=int(simulation.num_steps),
    )
    monitor_specs, _ = compile_monitor_specs(
        devices=simulation.devices,
        fields=fields,
        resolution=float(simulation.resolution),
        num_steps=int(num_steps),
        dt=float(simulation.dt),
    )
    monitor_devices = tuple(
        device for device in simulation.devices if isinstance(device, Monitor)
    )

    boundary_plan = _build_boundary_plan(simulation)
    material_plan = _build_material_plan(simulation)
    source_plan = _build_source_plan(source_specs)
    monitor_plan = _build_monitor_plan(monitor_specs)
    output_plan = OutputPlan(
        family="monitor_state_only" if monitor_specs else "none",
    )

    key = CompilationPlanKey(
        backend_platform=str(backend_platform),
        dimension_family=_dimension_family(simulation.is_3d, simulation.plane_2d),
        field_shape=tuple(int(v) for v in fields.permittivity.shape),
        field_dtype=str(np.asarray(fields.Ex).dtype),
        num_steps=int(num_steps),
        loop_kind=str(loop_kind),
        source_single_slab_dense=bool(source_single_slab_dense),
        temporal_block_steps=int(temporal_block_steps),
        boundary_family=boundary_plan.family,
        boundary_geometry=boundary_plan.geometry,
        material_family=material_plan.family,
        thermal_family=material_plan.thermal_family,
        coefficient_layout_family=_coefficient_layout_family(
            simulation, boundary_plan.family
        ),
        source_family=source_plan.family,
        source_timing_family=source_plan.timing_family,
        monitor_family=monitor_plan.family,
        monitor_geometry_family=monitor_plan.geometry_family,
        output_family=output_plan.family,
        kernel_family=_kernel_family(material_plan, source_plan, monitor_plan),
    )
    return CompiledPlan(
        key=key,
        boundary=boundary_plan,
        material=material_plan,
        sources=source_plan,
        monitors=monitor_plan,
        output=output_plan,
        source_specs=source_specs,
        monitor_specs=monitor_specs,
        monitor_devices=monitor_devices,
    )


def _dimension_family(is_3d: bool, plane_2d: str) -> str:
    if bool(is_3d):
        return "3d"
    return {
        "xy": "2d_xy",
        "xz": "2d_xz",
        "yz": "2d_yz",
    }.get(str(plane_2d).lower(), "2d_xy")


def _build_boundary_plan(simulation) -> BoundaryPlan:
    pml_boundaries = [b for b in simulation.boundaries if isinstance(b, PML)]
    if not pml_boundaries:
        return BoundaryPlan(family="none", geometry=tuple())

    resolution = float(simulation.resolution)
    geometry: list[tuple[str, int]] = []
    for idx, boundary in enumerate(pml_boundaries):
        thickness = int(round(float(getattr(boundary, "thickness", 0.0)) / resolution))
        edges = tuple(str(getattr(boundary, "edges", "all")).replace(" ", "").split(","))
        geometry.append((f"{idx}:{'|'.join(edges)}", max(thickness, 0)))
    return BoundaryPlan(family="pml", geometry=tuple(sorted(geometry)))


def _build_material_plan(simulation) -> MaterialPlan:
    thermal = simulation.thermal
    thermal_family = (
        "compiled_unsupported"
        if thermal is not None and getattr(thermal, "enabled", True)
        else "none"
    )
    return MaterialPlan(
        family="linear_nondispersive",
        thermal_family=thermal_family,
        spec=CompiledMaterialSpec(model_kind="linear"),
    )


def _build_source_plan(source_specs: tuple[CompiledSourceSpec, ...]) -> SourcePlan:
    if not source_specs:
        return SourcePlan(family="none", timing_family="none", spec_count=0, slab_count=0)

    slab_count = sum(1 for spec in source_specs if bool(spec.is_slab))
    if len(source_specs) == 1 and slab_count == 1:
        family = "single_slab"
    elif slab_count == len(source_specs):
        family = "batched_slabs"
    else:
        family = "mixed"

    timings = {str(spec.timing) for spec in source_specs}
    if timings == {"pre_e"}:
        timing_family = "pre_update_only"
    elif timings <= {"h", "e"}:
        timing_family = "split_h_e"
    else:
        timing_family = "mixed"

    return SourcePlan(
        family=family,
        timing_family=timing_family,
        spec_count=len(source_specs),
        slab_count=slab_count,
    )


def _build_monitor_plan(
    monitor_specs: tuple[CompiledMonitorSpec, ...],
) -> MonitorPlan:
    if not monitor_specs:
        return MonitorPlan(
            family="none",
            geometry_family="none",
            spec_count=0,
            power_count=0,
            frequency_count=0,
            dft_count=0,
        )

    power_count = sum(1 for spec in monitor_specs if bool(spec.accumulate_power))
    dft_count = sum(1 for spec in monitor_specs if bool(spec.dft_enabled))
    frequency_count = sum(
        1 for spec in monitor_specs if bool(spec.accumulate_frequency or spec.dft_enabled)
    )

    if dft_count and dft_count == len(monitor_specs):
        family = "dft_only"
    elif frequency_count == 0:
        family = "power_only"
    else:
        family = "mixed"

    if len(monitor_specs) == 1:
        geometry_family = "single_region"
    elif all(bool(spec.is_3d) and not bool(spec.dft_enabled) for spec in monitor_specs):
        geometry_family = "batched_regions"
    else:
        geometry_family = "mixed"

    return MonitorPlan(
        family=family,
        geometry_family=geometry_family,
        spec_count=len(monitor_specs),
        power_count=power_count,
        frequency_count=frequency_count,
        dft_count=dft_count,
    )


def _coefficient_layout_family(simulation, boundary_family: str) -> str:
    if boundary_family != "none":
        return "heterogeneous_array"

    fields = simulation.fields
    arrays = (
        np.asarray(fields.permittivity),
        np.asarray(fields.conductivity),
        np.asarray(fields.permeability),
    )
    if all(_is_uniform_array(arr) for arr in arrays):
        return "uniform_scalar"
    return "heterogeneous_array"


def _is_uniform_array(arr: np.ndarray) -> bool:
    if arr.size == 0:
        return True
    return bool(np.allclose(arr, arr.reshape(-1)[0], rtol=0.0, atol=0.0))


def _kernel_family(
    material_plan: MaterialPlan,
    source_plan: SourcePlan,
    monitor_plan: MonitorPlan,
) -> CompiledKernelFamily:
    if material_plan.family != "linear_nondispersive":
        return "engine_plus_material"
    if source_plan.family == "none" and monitor_plan.family == "none":
        return "engine_only"
    if source_plan.family != "none" and monitor_plan.family == "none":
        return "engine_plus_sources"
    if source_plan.family == "none" and monitor_plan.family != "none":
        return "engine_plus_monitors"
    return "engine_plus_sources_and_monitors"
