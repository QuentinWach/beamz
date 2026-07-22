"""The immutable input contract shared by BeamZ analysis operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from beamz.devices._immutable import immutable_snapshot, readonly_array
from beamz.devices.monitors.monitors import FluxMonitor, ModeMonitor, _Monitor
from beamz.simulation.results import (
    MaterialRegion,
    MonitorResults,
    SimulationMetadata,
    SimulationResults,
)


def static_fields(value):
    """Return field metadata without consulting live runtime state."""
    fields = getattr(value, "fields", None)
    if fields is not None:
        return fields
    compile_simulation = getattr(value, "compile", None)
    if callable(compile_simulation):
        return cast(Any, compile_simulation()).grid
    raise TypeError(f"{type(value).__name__} does not provide field metadata.")


@dataclass(frozen=True, slots=True)
class AnalysisData:
    """Coordinates, fields, materials, frequencies, and monitor geometry."""

    coordinates: SimulationMetadata
    fields: Mapping[str, np.ndarray]
    materials: MaterialRegion | None
    frequencies: np.ndarray
    monitor_geometry: _Monitor | None

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, SimulationMetadata):
            raise TypeError("AnalysisData.coordinates must be SimulationMetadata.")
        if self.materials is not None and not isinstance(
            self.materials, MaterialRegion
        ):
            raise TypeError("AnalysisData.materials must be a MaterialRegion or None.")
        if self.monitor_geometry is not None and not isinstance(
            self.monitor_geometry, _Monitor
        ):
            raise TypeError(
                "AnalysisData.monitor_geometry must be a canonical monitor."
            )
        object.__setattr__(self, "fields", immutable_snapshot(dict(self.fields)))
        object.__setattr__(self, "materials", immutable_snapshot(self.materials))
        object.__setattr__(
            self, "frequencies", readonly_array(self.frequencies, dtype=float)
        )
        object.__setattr__(
            self, "monitor_geometry", immutable_snapshot(self.monitor_geometry)
        )

    name = property(
        lambda self: (
            None if self.monitor_geometry is None else self.monitor_geometry.name
        )
    )
    dt = property(lambda self: self.coordinates.dt)
    resolution = property(lambda self: self.coordinates.resolution)
    is_3d = property(lambda self: self.coordinates.is_3d)
    plane_2d = property(lambda self: self.coordinates.plane_2d)

    def field(self, component: str) -> np.ndarray:
        try:
            return np.asarray(self.fields[component])
        except KeyError as exc:
            raise ValueError(f"No analysis field recorded for {component!r}.") from exc


def _lower_monitor(
    metadata: SimulationMetadata, result: MonitorResults
) -> AnalysisData:
    sample_times = (
        result.field_times if result.field_times.size else result.power_timestamps
    )
    coordinates = (
        replace(metadata, time=sample_times) if sample_times.size else metadata
    )
    fields = dict(result.fields)
    fields.update(
        {
            component: result.get_dft_component(component)
            for component in result.dft_fields
        }
    )
    frequencies = result.get_dft_frequencies()
    has_dft_flux_fields = bool(result.dft_fields) and isinstance(
        result.monitor, (FluxMonitor, ModeMonitor)
    )
    if frequencies.size and (result.power_spectrum.size or has_dft_flux_fields):
        fields["flux"] = result.get_dft_flux()
    if result.power_history.size:
        fields["power"] = result.power_history
    if result.field_steps.size:
        fields["step"] = result.field_steps
    materials = result.material_region or metadata.fields.materials
    return AnalysisData(coordinates, fields, materials, frequencies, result.monitor)


def analysis_inputs(value) -> Mapping[str, AnalysisData]:
    """Return canonical named analysis inputs without live-object adaptation."""
    if isinstance(value, AnalysisData):
        return MappingProxyType({value.name or "analysis": value})
    if isinstance(value, SimulationResults):
        lowered = {
            name: _lower_monitor(value.metadata, result)
            for name, result in value.monitors.items()
        }
        if not lowered:
            lowered["analysis"] = AnalysisData(
                value.metadata,
                {},
                value.metadata.fields.materials,
                np.empty(0, dtype=float),
                None,
            )
        return MappingProxyType(lowered)
    if isinstance(value, Mapping) and all(
        isinstance(item, AnalysisData) for item in value.values()
    ):
        return MappingProxyType(dict(value))
    raise TypeError(
        "Analysis requires AnalysisData, SimulationResults, or a named "
        "mapping of AnalysisData."
    )


def analysis_data(value, name: str | None = None) -> AnalysisData:
    """Return one named contract, or the sole available contract."""
    inputs = analysis_inputs(value)
    if name is not None:
        return inputs[str(name)]
    if len(inputs) != 1:
        raise ValueError("name is required when multiple monitor results are present.")
    return next(iter(inputs.values()))


__all__ = ["AnalysisData", "analysis_data", "analysis_inputs"]
