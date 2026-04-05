from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamz.const import µm

_VALID_EDGES = frozenset({"left", "right", "top", "bottom", "front", "back"})


def _normalize_edges(edges):
    if edges == "all":
        return "all"
    values = edges if isinstance(edges, (tuple, list)) else (edges,)
    normalized = tuple(str(edge).lower() for edge in values)
    invalid = sorted(set(normalized) - _VALID_EDGES)
    if invalid:
        raise ValueError(f"invalid boundary edges: {invalid}")
    return normalized


@dataclass(frozen=True, slots=True)
class BoundarySpec:
    edges: str | tuple[str, ...] = "all"
    thickness: float = 1 * µm

    def __post_init__(self):
        object.__setattr__(self, "edges", _normalize_edges(self.edges))
        object.__setattr__(self, "thickness", float(self.thickness))
        if not np.isfinite(self.thickness) or self.thickness <= 0:
            raise ValueError("thickness must be positive")

    def to_dict(self):
        return {
            "type": type(self).__name__,
            "edges": self.edges if self.edges == "all" else list(self.edges),
            "thickness": float(self.thickness),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            edges=data.get("edges", "all"),
            thickness=data.get("thickness", 1 * µm),
        )


@dataclass(frozen=True, slots=True)
class PMLSpec(BoundarySpec):
    sigma_max: float | None = None
    m: int = 3

    def __post_init__(self):
        BoundarySpec.__post_init__(self)
        if self.sigma_max is not None:
            object.__setattr__(self, "sigma_max", float(self.sigma_max))
            if not np.isfinite(self.sigma_max) or self.sigma_max <= 0:
                raise ValueError("sigma_max must be a finite positive value when provided")
        object.__setattr__(self, "m", int(self.m))
        if self.m <= 0:
            raise ValueError("m must be positive")

    def to_dict(self):
        data = BoundarySpec.to_dict(self)
        data.update(
            sigma_max=self.sigma_max,
            m=int(self.m),
        )
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            edges=data.get("edges", "all"),
            thickness=data.get("thickness", 1 * µm),
            sigma_max=data.get("sigma_max"),
            m=data.get("m", 3),
        )


def boundary_to_spec(boundary):
    if isinstance(boundary, (BoundarySpec, PMLSpec)):
        return boundary
    spec = getattr(boundary, "spec", None)
    if isinstance(spec, (BoundarySpec, PMLSpec)):
        return spec
    raise TypeError("boundary must be a boundary spec or spec-backed boundary facade")


def boundary_spec_to_dict(boundary):
    return boundary_to_spec(boundary).to_dict()


def boundary_spec_from_dict(data):
    kind = data.get("type")
    if kind == "BoundarySpec":
        return BoundarySpec.from_dict(data)
    if kind == "PMLSpec":
        return PMLSpec.from_dict(data)
    if kind == "Boundary":
        return BoundarySpec.from_dict(data)
    if kind == "PML":
        return PMLSpec.from_dict(data)
    raise ValueError(f"unknown boundary spec type: {kind!r}")
