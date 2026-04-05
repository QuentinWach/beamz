from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from beamz.simulation.boundary_specs import (
    boundary_spec_from_dict,
    boundary_spec_to_dict,
    boundary_to_spec,
)


def _freeze_time_array(time) -> np.ndarray:
    arr = np.asarray(time, dtype=float).copy()
    if arr.ndim != 1:
        raise ValueError("time must be a 1D array")
    if arr.size < 2:
        raise ValueError("FDTD requires a time array with at least two entries")
    if not np.all(np.isfinite(arr)):
        raise ValueError("time must contain only finite values")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("time must be strictly increasing")
    arr.setflags(write=False)
    return arr


def _as_spec(value):
    spec = getattr(value, "spec", None)
    return spec if spec is not None else value


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    design: object
    devices: tuple[object, ...]
    boundaries: tuple[object, ...]
    resolution: float
    time: np.ndarray
    plane_2d: str
    is_3d: bool

    def __post_init__(self):
        if self.design is None:
            raise ValueError("design is required")
        object.__setattr__(self, "design", _as_spec(self.design))
        object.__setattr__(self, "devices", tuple(_as_spec(device) for device in self.devices))
        object.__setattr__(
            self,
            "boundaries",
            tuple(boundary_to_spec(boundary) for boundary in self.boundaries),
        )
        object.__setattr__(self, "resolution", float(self.resolution))
        if not isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("resolution must be a finite positive value")
        object.__setattr__(self, "time", _freeze_time_array(self.time))
        plane = str(self.plane_2d).lower()
        if plane not in {"xy", "yz", "xz"}:
            plane = "xy"
        object.__setattr__(self, "plane_2d", plane)
        object.__setattr__(self, "is_3d", bool(self.is_3d))
        if any(device is None for device in self.devices):
            raise ValueError("devices may not contain None")
        if any(boundary is None for boundary in self.boundaries):
            raise ValueError("boundaries may not contain None")

    def to_dict(self):
        from beamz.devices.monitors.spec import MonitorSpec
        from beamz.devices.sources.spec import source_spec_to_dict

        devices = []
        for device in self.devices:
            if isinstance(device, MonitorSpec):
                devices.append(device.to_dict())
            else:
                devices.append(source_spec_to_dict(device))
        return {
            "type": "SimulationSpec",
            "design": self.design.to_dict(),
            "devices": devices,
            "boundaries": [boundary_spec_to_dict(boundary) for boundary in self.boundaries],
            "resolution": float(self.resolution),
            "time": np.asarray(self.time).tolist(),
            "plane_2d": self.plane_2d,
            "is_3d": bool(self.is_3d),
        }

    @classmethod
    def from_dict(cls, data):
        from beamz.design.spec import DesignSpec
        from beamz.devices.monitors.spec import MonitorSpec
        from beamz.devices.sources.spec import source_spec_from_dict

        devices = []
        for item in data.get("devices", ()):
            if item.get("type") == "MonitorSpec":
                devices.append(MonitorSpec.from_dict(item))
            else:
                devices.append(source_spec_from_dict(item))
        return cls(
            design=DesignSpec.from_dict(data["design"]),
            devices=tuple(devices),
            boundaries=tuple(boundary_spec_from_dict(item) for item in data.get("boundaries", ())),
            resolution=data["resolution"],
            time=data["time"],
            plane_2d=data.get("plane_2d", "xy"),
            is_3d=data.get("is_3d", False),
        )


def build_simulation_spec(
    *,
    design,
    devices=None,
    boundaries=None,
    resolution,
    time,
    plane_2d="xy",
) -> SimulationSpec:
    devices = () if devices is None else tuple(devices)
    boundaries = () if boundaries is None else tuple(boundaries)
    is_3d = bool(design.is_3d and design.depth > 0)
    return SimulationSpec(
        design=design,
        devices=devices,
        boundaries=boundaries,
        resolution=resolution,
        time=time,
        plane_2d=plane_2d,
        is_3d=is_3d,
    )
