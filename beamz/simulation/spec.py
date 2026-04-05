from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _freeze_time_array(time) -> np.ndarray:
    arr = np.asarray(time, dtype=float).copy()
    if arr.ndim != 1:
        raise ValueError("time must be a 1D array")
    if arr.size < 2:
        raise ValueError("FDTD requires a time array with at least two entries")
    arr.setflags(write=False)
    return arr


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
        object.__setattr__(self, "devices", tuple(self.devices))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "resolution", float(self.resolution))
        object.__setattr__(self, "time", _freeze_time_array(self.time))
        plane = str(self.plane_2d).lower()
        if plane not in {"xy", "yz", "xz"}:
            plane = "xy"
        object.__setattr__(self, "plane_2d", plane)
        object.__setattr__(self, "is_3d", bool(self.is_3d))


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
