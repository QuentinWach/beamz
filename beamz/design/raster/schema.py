"""Small, serializable scene model shared by every raster importer."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from beamz.design.materials import Material

from . import _native  # type: ignore[attr-defined]

_CACHE_SCHEMA_VERSION = 4
_ENGINE_VERSION = str(_native.ENGINE_VERSION)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _points(values: Iterable[Iterable[float]], dimensions: int) -> list[list[float]]:
    result = [[float(value) for value in point] for point in values]
    if any(len(point) != dimensions for point in result):
        raise ValueError(f"Every point must contain {dimensions} coordinates.")
    if not np.isfinite(result).all():
        raise ValueError("Coordinates must be finite.")
    return result


@dataclass(frozen=True, slots=True)
class Polygon:
    exterior: tuple[tuple[float, float], ...]
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "exterior": _points(self.exterior, 2),
            "holes": [_points(hole, 2) for hole in self.holes],
        }


@dataclass(frozen=True, slots=True)
class Box:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "box",
            "bounds": {
                "min": _points((self.minimum,), 3)[0],
                "max": _points((self.maximum,), 3)[0],
            },
        }


@dataclass(frozen=True, slots=True)
class Sphere:
    center: tuple[float, float, float]
    radius: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sphere",
            "center": _points((self.center,), 3)[0],
            "radius": float(self.radius),
        }


@dataclass(frozen=True, slots=True)
class Cylinder:
    center: tuple[float, float]
    radius: float
    z_min: float
    z_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "cylinder",
            "center": _points((self.center,), 2)[0],
            "radius": float(self.radius),
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
        }


@dataclass(frozen=True, slots=True)
class ExtrudedPolygon:
    polygon: Polygon
    z_min: float
    z_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "extruded_polygon",
            "polygon": self.polygon.to_dict(),
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
        }


@dataclass(frozen=True, slots=True)
class TaperedExtrudedPolygon:
    polygon: Polygon
    z_min: float
    z_max: float
    sidewall_angle_degrees: float
    width_to_z: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tapered_extruded_polygon",
            "polygon": self.polygon.to_dict(),
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
            "sidewall_angle_degrees": float(self.sidewall_angle_degrees),
            "width_to_z": float(self.width_to_z),
        }


def _mesh_arrays(vertices: Any, triangles: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ValueError("Mesh vertices must have shape (N, 3) with N > 0.")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,):
        raise ValueError("Mesh triangles must have shape (M, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite.")
    if np.issubdtype(triangles.dtype, np.floating):
        if not np.isfinite(triangles).all() or np.any(triangles != np.floor(triangles)):
            raise ValueError("Mesh triangle indices must be finite integers.")
    elif not np.issubdtype(triangles.dtype, np.integer):
        raise TypeError("Mesh triangle indices must use an integer dtype.")
    if np.any(triangles < 0) or np.any(triangles > np.iinfo(np.uint32).max):
        raise ValueError("Mesh triangle indices must fit in uint32.")
    return vertices, triangles.astype(np.uint32, copy=False)


@dataclass(frozen=True, slots=True)
class MeshInspection:
    vertices: int
    triangles: int
    connected_components: int
    boundary_edges: int
    nonmanifold_edges: int
    inconsistent_edges: int
    degenerate_triangles: int
    self_intersections: int
    signed_volume: float
    valid_for_rasterization: bool

    @property
    def topology_is_valid(self) -> bool:
        return not any(
            (
                self.boundary_edges,
                self.nonmanifold_edges,
                self.inconsistent_edges,
                self.degenerate_triangles,
                self.self_intersections,
            )
        )


def inspect_mesh(vertices: Any, triangles: Any) -> MeshInspection:
    """Run the exact native checks used when a scene is compiled."""

    vertex_array, triangle_array = _mesh_arrays(vertices, triangles)
    report = _native.inspect_mesh(vertex_array.tolist(), triangle_array.tolist())
    return MeshInspection(**json.loads(report))


@dataclass(frozen=True, slots=True, eq=False)
class Mesh:
    vertices: np.ndarray
    triangles: np.ndarray

    def validated_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return _mesh_arrays(self.vertices, self.triangles)

    def to_dict(self) -> dict[str, Any]:
        vertices, triangles = self.validated_arrays()
        return {
            "type": "triangle_mesh",
            "vertices": vertices.tolist(),
            "triangles": triangles.tolist(),
            "bounds": {
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            },
        }


Geometry = Box | Sphere | Cylinder | ExtrudedPolygon | TaperedExtrudedPolygon | Mesh


@dataclass(frozen=True, slots=True)
class Object:
    geometry: Geometry
    material_id: int
    priority: int = 0
    id: int = 0


@dataclass(frozen=True, slots=True)
class Scene:
    materials: tuple[Material, ...]
    objects: tuple[Object, ...] = ()
    background_material: int = 0

    def to_dict(self) -> dict[str, Any]:
        used_ids = {obj.id for obj in self.objects if obj.id}
        next_id = 1
        objects = []
        for obj in self.objects:
            object_id = _integer(obj.id, "Object ID")
            if object_id == 0:
                while next_id in used_ids:
                    next_id += 1
                object_id = next_id
                used_ids.add(object_id)
                next_id += 1
            objects.append(
                {
                    "id": object_id,
                    "material_id": _integer(obj.material_id, "Material ID"),
                    "priority": int(obj.priority),
                    "geometry": obj.geometry.to_dict(),
                }
            )
        return {
            "materials": [material.to_dict() for material in self.materials],
            "objects": objects,
            "background_material": _integer(
                self.background_material, "Background material ID"
            ),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def rasterize(
        self,
        grid: Grid,
        *,
        options: Any = None,
        cache_directory: Any = None,
    ):
        """Rasterize this reusable scene on an explicit Cartesian grid.

        Returns
        -------
        RasterResult
            Immutable cell and Yee-support material tensors.
        """
        from .engine import rasterize

        return rasterize(
            self,
            grid,
            options=options,
            cache_directory=cache_directory,
        )


@dataclass(frozen=True, slots=True)
class Grid:
    """Explicit Cartesian edges; spacing may vary independently on every axis."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray

    def __post_init__(self) -> None:
        for name in ("x_edges", "y_edges", "z_edges"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if values.ndim != 1 or len(values) < 2:
                raise ValueError(f"{name} must be a one-dimensional edge array.")
            if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
                raise ValueError(f"{name} must be finite and strictly increasing.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @classmethod
    def uniform(
        cls,
        minimum: tuple[float, float, float],
        maximum: tuple[float, float, float],
        shape: tuple[int, int, int],
    ) -> Grid:
        if len(minimum) != 3 or len(maximum) != 3 or len(shape) != 3:
            raise ValueError("Uniform-grid bounds and shape must have length 3.")
        counts = tuple(_integer(value, "Grid shape", minimum=1) for value in shape)
        return cls(
            *(
                np.linspace(
                    float(minimum[axis]), float(maximum[axis]), counts[axis] + 1
                )
                for axis in range(3)
            )
        )

    @property
    def edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.x_edges, self.y_edges, self.z_edges

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(len(edges) - 1 for edges in self.edges)  # type: ignore[return-value]

    @property
    def is_uniform(self) -> bool:
        return all(
            np.allclose(np.diff(edges), np.diff(edges)[0], rtol=1e-12, atol=0.0)
            for edges in self.edges
        )
