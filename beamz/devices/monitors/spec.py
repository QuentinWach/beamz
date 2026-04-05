from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_optional_float_array(values) -> np.ndarray:
    arr = np.asarray([] if values is None else values, dtype=float).reshape(-1).copy()
    arr.setflags(write=False)
    return arr


def _as_optional_str_tuple(values) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(str(v) for v in values)


def _normalize_start(start) -> tuple[float, ...]:
    if start is None:
        raise ValueError("start is required")
    vals = tuple(float(v) for v in start)
    if len(vals) not in {2, 3}:
        raise ValueError("start must have length 2 or 3")
    return vals


def _normalize_end(end, *, start_z: float | None = None) -> tuple[float, ...] | None:
    if end is None:
        return None
    vals = tuple(float(v) for v in end)
    if len(vals) == 2 and start_z is not None:
        return (vals[0], vals[1], start_z)
    if len(vals) not in {2, 3}:
        raise ValueError("end must have length 2 or 3")
    return vals


def _normalize_size(size) -> tuple[float, float] | None:
    if size is None:
        return None
    vals = tuple(float(v) for v in size)
    if len(vals) != 2:
        raise ValueError("size must have length 2")
    return vals


def _determine_3d_mode(start, end, design) -> bool:
    if end is not None and len(end) == 3:
        return True
    if len(start) == 3:
        return True
    if end is not None and len(start) == 2 and len(end) == 2:
        return False
    if design is not None and getattr(design, "is_3d", False):
        return True
    return False


def _default_plane_size(design, plane_normal: str) -> tuple[float, float]:
    if design is None:
        return (1e-6, 1e-6)
    if plane_normal == "z":
        return (float(design.width), float(design.height))
    if plane_normal == "y":
        return (float(design.width), float(getattr(design, "depth", 0.0) or design.width))
    return (
        float(design.height),
        float(getattr(design, "depth", 0.0) or design.height),
    )


def _geometry_from_inputs(design, start, end, plane_normal, plane_position, size):
    start = _normalize_start(start)
    start_z = start[2] if len(start) == 3 else None
    end = _normalize_end(end, start_z=start_z)
    is_3d = _determine_3d_mode(start, end, design)

    if not is_3d:
        if end is None:
            end = start
        if len(end) != 2:
            end = (end[0], end[1])
        if len(start) != 2:
            start = (start[0], start[1])
        position = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        return {
            "is_3d": False,
            "start": start,
            "end": end,
            "plane_normal": None,
            "plane_position": 0.0,
            "size": None,
            "monitor_type": "line",
            "position": position,
            "vertices": (),
        }

    if len(start) == 2:
        start = (start[0], start[1], 0.0)

    if end is not None:
        if len(end) == 2:
            end = (end[0], end[1], start[2])
        inferred = plane_normal
        if inferred is None:
            dx = abs(end[0] - start[0])
            dy = abs(end[1] - start[1])
            dz = abs(end[2] - start[2])
            inferred = ("x", "y", "z")[int(np.argmin([dx, dy, dz]))]
        inferred = str(inferred).lower()
        if inferred == "x":
            size = (abs(end[1] - start[1]), abs(end[2] - start[2]))
            plane_position = start[0]
        elif inferred == "y":
            size = (abs(end[0] - start[0]), abs(end[2] - start[2]))
            plane_position = start[1]
        else:
            inferred = "z"
            size = (abs(end[0] - start[0]), abs(end[1] - start[1]))
            plane_position = start[2]
        start = (
            min(start[0], end[0]),
            min(start[1], end[1]),
            min(start[2], end[2]),
        )
        plane_normal = inferred
    else:
        plane_normal = str(plane_normal or "z").lower()
        if plane_position == 0 and start is not None:
            axis_idx = {"x": 0, "y": 1, "z": 2}.get(plane_normal, 2)
            plane_position = start[axis_idx]
        if size is None:
            size = _default_plane_size(design, plane_normal)

    size = _normalize_size(size)
    plane_position = float(plane_position)
    if plane_normal == "z":
        vertices = (
            (start[0], start[1], plane_position),
            (start[0] + size[0], start[1], plane_position),
            (start[0] + size[0], start[1] + size[1], plane_position),
            (start[0], start[1] + size[1], plane_position),
        )
        position = (
            start[0] + size[0] / 2.0,
            start[1] + size[1] / 2.0,
            plane_position,
        )
    elif plane_normal == "y":
        vertices = (
            (start[0], plane_position, start[2]),
            (start[0] + size[0], plane_position, start[2]),
            (start[0] + size[0], plane_position, start[2] + size[1]),
            (start[0], plane_position, start[2] + size[1]),
        )
        position = (
            start[0] + size[0] / 2.0,
            plane_position,
            start[2] + size[1] / 2.0,
        )
    else:
        plane_normal = "x"
        vertices = (
            (plane_position, start[1], start[2]),
            (plane_position, start[1] + size[0], start[2]),
            (plane_position, start[1] + size[0], start[2] + size[1]),
            (plane_position, start[1], start[2] + size[1]),
        )
        position = (
            plane_position,
            start[1] + size[0] / 2.0,
            start[2] + size[1] / 2.0,
        )

    return {
        "is_3d": True,
        "start": start,
        "end": end,
        "plane_normal": plane_normal,
        "plane_position": plane_position,
        "size": size,
        "monitor_type": "plane",
        "position": position,
        "vertices": vertices,
    }


@dataclass(frozen=True, slots=True)
class MonitorSpec:
    start: tuple[float, ...]
    end: tuple[float, ...] | None
    is_3d: bool
    monitor_type: str
    position: tuple[float, ...]
    vertices: tuple[tuple[float, float, float], ...]
    plane_normal: str | None
    plane_position: float
    size: tuple[float, float] | None
    should_record_fields: bool
    accumulate_power: bool
    live_update: bool
    record_interval: int
    max_history_steps: int | None
    dft_frequencies: np.ndarray
    dft_t_start: float
    dft_t_end: float | None
    dft_enabled: bool
    dft_components: tuple[str, ...] | None
    dft_record_every_step: bool
    dft_record_interval: int
    dft_window: str
    name: str | None
    frequency_points: np.ndarray
    frequency_record_interval: int

    def __post_init__(self):
        object.__setattr__(self, "dft_frequencies", _as_optional_float_array(self.dft_frequencies))
        object.__setattr__(self, "frequency_points", _as_optional_float_array(self.frequency_points))
        object.__setattr__(self, "dft_components", _as_optional_str_tuple(self.dft_components))
        object.__setattr__(self, "record_interval", max(1, int(self.record_interval)))
        object.__setattr__(self, "dft_record_interval", max(1, int(self.dft_record_interval)))
        object.__setattr__(self, "frequency_record_interval", max(1, int(self.frequency_record_interval)))
        object.__setattr__(self, "should_record_fields", bool(self.should_record_fields))
        object.__setattr__(self, "accumulate_power", bool(self.accumulate_power))
        object.__setattr__(self, "live_update", bool(self.live_update))
        object.__setattr__(self, "dft_enabled", bool(self.dft_enabled))
        object.__setattr__(self, "dft_record_every_step", bool(self.dft_record_every_step))
        window = str(self.dft_window).lower()
        if window in {"none", "rectangular"}:
            window = "rect"
        if window not in {"rect", "hann"}:
            raise ValueError(f"dft_window must be one of ['rect', 'hann'], got {self.dft_window!r}")
        object.__setattr__(self, "dft_window", window)
        if self.max_history_steps is not None and int(self.max_history_steps) <= 0:
            raise ValueError("max_history_steps must be positive when provided")
        if np.any(self.frequency_points < 0.0):
            raise ValueError("frequency_points must be non-negative frequencies in Hz")
        if self.plane_normal is not None and self.plane_normal not in {"x", "y", "z"}:
            raise ValueError("plane_normal must be one of 'x', 'y', or 'z'")

    @property
    def accumulate_frequency(self) -> bool:
        return bool(self.frequency_points.size > 0)

    def to_dict(self):
        return {
            "type": "MonitorSpec",
            "start": list(self.start),
            "end": None if self.end is None else list(self.end),
            "is_3d": bool(self.is_3d),
            "monitor_type": self.monitor_type,
            "position": list(self.position),
            "vertices": [list(vertex) for vertex in self.vertices],
            "plane_normal": self.plane_normal,
            "plane_position": float(self.plane_position),
            "size": None if self.size is None else list(self.size),
            "should_record_fields": bool(self.should_record_fields),
            "accumulate_power": bool(self.accumulate_power),
            "live_update": bool(self.live_update),
            "record_interval": int(self.record_interval),
            "max_history_steps": self.max_history_steps,
            "dft_frequencies": np.asarray(self.dft_frequencies).tolist(),
            "dft_t_start": float(self.dft_t_start),
            "dft_t_end": self.dft_t_end,
            "dft_enabled": bool(self.dft_enabled),
            "dft_components": None if self.dft_components is None else list(self.dft_components),
            "dft_record_every_step": bool(self.dft_record_every_step),
            "dft_record_interval": int(self.dft_record_interval),
            "dft_window": self.dft_window,
            "name": self.name,
            "frequency_points": np.asarray(self.frequency_points).tolist(),
            "frequency_record_interval": int(self.frequency_record_interval),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            start=tuple(data["start"]),
            end=None if data.get("end") is None else tuple(data["end"]),
            is_3d=data["is_3d"],
            monitor_type=data["monitor_type"],
            position=tuple(data["position"]),
            vertices=tuple(tuple(vertex) for vertex in data.get("vertices", ())),
            plane_normal=data.get("plane_normal"),
            plane_position=data.get("plane_position", 0.0),
            size=None if data.get("size") is None else tuple(data["size"]),
            should_record_fields=data.get("should_record_fields", True),
            accumulate_power=data.get("accumulate_power", True),
            live_update=data.get("live_update", False),
            record_interval=data.get("record_interval", 1),
            max_history_steps=data.get("max_history_steps"),
            dft_frequencies=data.get("dft_frequencies"),
            dft_t_start=data.get("dft_t_start", 0.0),
            dft_t_end=data.get("dft_t_end"),
            dft_enabled=data.get("dft_enabled", False),
            dft_components=data.get("dft_components"),
            dft_record_every_step=data.get("dft_record_every_step", True),
            dft_record_interval=data.get("dft_record_interval", 1),
            dft_window=data.get("dft_window", "rect"),
            name=data.get("name"),
            frequency_points=data.get("frequency_points"),
            frequency_record_interval=data.get("frequency_record_interval", 1),
        )


def build_monitor_spec(
    *,
    design=None,
    start=(0, 0),
    end=None,
    plane_normal=None,
    plane_position=0,
    size=None,
    record_fields=True,
    accumulate_power=True,
    live_update=False,
    record_interval=1,
    max_history_steps=None,
    dft_frequencies=None,
    dft_t_start=0.0,
    dft_t_end=None,
    dft_enabled=False,
    dft_components=None,
    dft_record_every_step=True,
    dft_record_interval=None,
    dft_window="rect",
    name=None,
    frequency_points=None,
    frequency_record_interval=1,
) -> MonitorSpec:
    geometry = _geometry_from_inputs(
        design,
        start,
        end,
        plane_normal,
        plane_position,
        size,
    )
    return MonitorSpec(
        **geometry,
        should_record_fields=record_fields,
        accumulate_power=accumulate_power,
        live_update=live_update,
        record_interval=record_interval,
        max_history_steps=max_history_steps,
        dft_frequencies=_as_optional_float_array(dft_frequencies),
        dft_t_start=float(dft_t_start) if dft_t_start is not None else 0.0,
        dft_t_end=None if dft_t_end is None else float(dft_t_end),
        dft_enabled=dft_enabled,
        dft_components=dft_components,
        dft_record_every_step=dft_record_every_step,
        dft_record_interval=(
            1 if dft_record_interval is None and dft_record_every_step else (
                max(1, int(record_interval)) if dft_record_interval is None else dft_record_interval
            )
        ),
        dft_window=dft_window,
        name=name,
        frequency_points=_as_optional_float_array(frequency_points),
        frequency_record_interval=frequency_record_interval,
    )


def monitor_to_spec(monitor):
    if isinstance(monitor, MonitorSpec):
        return monitor
    spec = getattr(monitor, "spec", None)
    if isinstance(spec, MonitorSpec):
        return spec
    raise TypeError("monitor must be a MonitorSpec or spec-backed monitor facade")
