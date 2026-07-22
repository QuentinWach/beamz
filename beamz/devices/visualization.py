from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class DeviceVisualSpec:
    kind: str
    center: tuple[float, ...]
    size: tuple[float, ...]
    direction: str | None = None
    label: str | None = None
    style: Mapping[str, Any] | None = None

    def __post_init__(self):
        if self.style is not None:
            object.__setattr__(self, "style", MappingProxyType(dict(self.style)))


def visual_spec_from_device(device: object) -> DeviceVisualSpec | None:
    """Return the built-in plot description for a canonical device spec."""
    from beamz.devices.monitors.monitors import _Monitor
    from beamz.devices.sources.specs import (
        GaussianBeamSource,
        GaussianSource,
        ModeSource,
    )

    if isinstance(device, GaussianSource):
        return _gaussian_source_visual(device)
    if isinstance(device, ModeSource):
        return _mode_source_visual(device)
    if isinstance(device, GaussianBeamSource):
        return _gaussian_beam_source_visual(device)
    if isinstance(device, _Monitor):
        return _monitor_visual(device)
    return None


def _center_tuple(value) -> tuple[float, ...]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError("visual center must have at least two coordinates")
    return tuple(map(float, arr))


def _size_tuple(value, fallback: float = 0.25e-6) -> tuple[float, ...]:
    if value is None:
        return (float(fallback), float(fallback))
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return (float(fallback), float(fallback))
    return tuple(float(abs(v)) for v in arr)


def _gaussian_source_visual(source: Any) -> DeviceVisualSpec:
    width = float(source.width)
    return DeviceVisualSpec(
        kind="source",
        center=_center_tuple(source.position),
        size=(width, width),
        style={"shape": "circle"},
    )


def _mode_source_visual(source: Any) -> DeviceVisualSpec:
    return DeviceVisualSpec(
        kind="source",
        center=_center_tuple(source.center),
        size=_size_tuple(source.size),
        direction=str(getattr(source, "signed_direction", "")).lower(),
    )


def _gaussian_beam_source_visual(source: Any) -> DeviceVisualSpec:
    return DeviceVisualSpec(
        kind="source",
        center=_center_tuple(source.center),
        size=_size_tuple(getattr(source, "size", None)),
        direction=str(getattr(source, "direction", "")).lower(),
    )


def _monitor_visual(monitor: object) -> DeviceVisualSpec:
    start = getattr(monitor, "start", None)
    end = getattr(monitor, "end", None)
    if start is not None and end is not None:
        start = _center_tuple(start)
        end = _center_tuple(end)
        return DeviceVisualSpec(
            kind="monitor-line",
            center=tuple(0.5 * (a + b) for a, b in zip(start, end, strict=True)),
            size=tuple(abs(b - a) for a, b in zip(start, end, strict=True)),
            label=getattr(monitor, "name", None),
            style={"start": start[:2], "end": end[:2]},
        )
    center = getattr(monitor, "position", None)
    size = getattr(monitor, "size", None)
    return DeviceVisualSpec(
        kind="monitor-plane",
        center=_center_tuple(center),
        size=_size_tuple(size),
        direction=str(getattr(monitor, "plane_normal", "")).lower() or None,
        label=getattr(monitor, "name", None),
    )
