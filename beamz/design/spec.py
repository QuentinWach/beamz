from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DesignSpec:
    width: float
    height: float
    depth: float
    structures: tuple[object, ...]
    is_3d: bool
    time: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "depth", float(self.depth))
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "structures", tuple(self.structures))
        object.__setattr__(self, "is_3d", bool(self.is_3d))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Design width and height must be positive")


def build_design_spec(*, width, height, depth=0.0, structures=(), time=0.0) -> DesignSpec:
    depth = 0.0 if depth is None else float(depth)
    return DesignSpec(
        width=width,
        height=height,
        depth=depth,
        structures=tuple(structures),
        is_3d=bool(depth > 0),
        time=time,
    )
