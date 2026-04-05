from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from beamz.design.structures import StructureSpec


def _validate_structure(structure):
    if structure is None:
        raise ValueError("structures may not contain None")
    if isinstance(structure, StructureSpec):
        return structure
    spec = getattr(structure, "spec", None)
    if isinstance(spec, StructureSpec):
        return spec
    raise TypeError("structures must be StructureSpec values or spec-backed structure facades")


def _structure_is_3d(structure):
    if bool(getattr(structure, "is_3d", False)) or getattr(structure, "depth", 0) != 0:
        return True
    position = getattr(structure, "position", None)
    if position is not None and len(position) > 2 and position[2] != 0:
        return True
    return any(
        len(vertex) > 2 and vertex[2] != 0 for vertex in getattr(structure, "vertices", ())
    )


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
        object.__setattr__(self, "structures", tuple(_validate_structure(s) for s in self.structures))
        object.__setattr__(self, "is_3d", bool(self.is_3d))
        if (not isfinite(self.width)) or self.width <= 0 or (not isfinite(self.height)) or self.height <= 0:
            raise ValueError("Design width and height must be positive")
        if (not isfinite(self.depth)) or self.depth < 0:
            raise ValueError("Design depth must be finite and non-negative")
        if not isfinite(self.time):
            raise ValueError("Design time must be finite")
        has_3d_content = bool(self.depth > 0) or any(
            _structure_is_3d(structure) for structure in self.structures
        )
        if self.is_3d and not has_3d_content:
            raise ValueError("Design is_3d requires positive depth or at least one 3D structure")


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
