from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from beamz.design.materials import MaterialProtocol

DEFAULT_COLOR = "#6699cc"


def _position(value, z=None):
    values = tuple(value)
    if len(values) == 2:
        values = (*values, 0.0)
    if len(values) != 3:
        raise ValueError("Position must be (x,y) or (x,y,z)")
    if z is not None:
        values = (values[0], values[1], z)
    result = tuple(float(v) for v in values)
    if not all(np.isfinite(result)):
        raise ValueError(f"Position must be finite, got {value!r}")
    return result


def _vertices(values, z=0.0, *, ccw=True):
    result = []
    for vertex in values or ():
        if len(vertex) == 2:
            result.append((float(vertex[0]), float(vertex[1]), float(z)))
        elif len(vertex) == 3:
            result.append(tuple(float(v) for v in vertex))
        else:
            raise ValueError(f"Vertex must have 2 or 3 coordinates, got {len(vertex)}")
    if ccw and len(result) >= 3:
        area = sum(
            result[i][0] * result[(i + 1) % len(result)][1]
            - result[(i + 1) % len(result)][0] * result[i][1]
            for i in range(len(result))
        )
        if area < 0:
            result.reverse()
    return tuple(result)


def _rotate(path, angle, axis, center):
    cx, cy, cz = center
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    if axis == "z":
        return tuple(
            (
                cx + (x - cx) * cos_a - (y - cy) * sin_a,
                cy + (x - cx) * sin_a + (y - cy) * cos_a,
                z,
            )
            for x, y, z in path
        )
    if axis == "x":
        return tuple(
            (
                x,
                cy + (y - cy) * cos_a - (z - cz) * sin_a,
                cz + (y - cy) * sin_a + (z - cz) * cos_a,
            )
            for x, y, z in path
        )
    if axis == "y":
        return tuple(
            (
                cx + (x - cx) * cos_a + (z - cz) * sin_a,
                y,
                cz - (x - cx) * sin_a + (z - cz) * cos_a,
            )
            for x, y, z in path
        )
    raise ValueError(f"Invalid rotation axis {axis!r}; expected 'x', 'y', or 'z'.")


def _path_center(path):
    return tuple(sum(vertex[i] for vertex in path) / len(path) for i in range(3))


def _positive(name, value, *, allow_zero=False):
    value = float(value)
    valid = np.isfinite(value) and (value >= 0.0 if allow_zero else value > 0.0)
    if not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}, got {value}")
    return value


def _normalize_common(structure):
    """Normalize fields shared by public immutable structure specs."""
    field_names = structure.__dataclass_fields__
    if (
        "material" in field_names
        and structure.material is not None
        and not isinstance(structure.material, MaterialProtocol)
    ):
        raise TypeError(
            f"{type(structure).__name__}.material must satisfy MaterialProtocol."
        )
    if "color" in field_names:
        object.__setattr__(
            structure,
            "color",
            DEFAULT_COLOR if structure.color is None else str(structure.color),
        )
    if "facecolor" in field_names and structure.facecolor is not None:
        object.__setattr__(structure, "facecolor", str(structure.facecolor))
    for name in ("optimize", "sidewall_angle", "width_to_z"):
        if name in field_names:
            value = (
                bool(getattr(structure, name))
                if name == "optimize"
                else float(getattr(structure, name) or 0.0)
            )
            object.__setattr__(structure, name, value)


class Structure:
    """Shared functional operations for immutable geometry specifications."""

    __slots__ = ()

    def updated_copy(self, **changes):
        try:
            return replace(self, **changes)  # type: ignore[arg-type]
        except TypeError as exc:
            fields = getattr(self, "__dataclass_fields__", {})
            unknown = sorted(set(changes) - fields.keys())
            if unknown:
                raise TypeError(
                    f"Unexpected update fields: {', '.join(unknown)}"
                ) from None
            raise exc

    def with_material(self, material):
        return self.updated_copy(material=material)

    def copy(self):
        return self

    def _mesh_feature_sizes(self):
        """Return optional axis-specific physical sizes for automatic meshing."""
        return None


class PlanarStructure(Structure):
    """Geometry operations shared by polygon-backed structures."""

    __slots__ = ()
    vertices: tuple[tuple[float, float, float], ...]
    interiors: tuple[tuple[tuple[float, float, float], ...], ...]
    depth: float
    z: float
    sidewall_angle: float
    width_to_z: float
    material: Any
    color: str
    optimize: bool
    position: tuple[float, float, float]

    def _as_polygon(self, *, vertices=None, interiors=None, z=None, depth=None):
        return Polygon(
            vertices=self.vertices if vertices is None else vertices,
            interiors=self.interiors if interiors is None else interiors,
            material=self.material,
            color=self.color,
            optimize=self.optimize,
            depth=self.depth if depth is None else depth,
            z=self.z if z is None else z,
            sidewall_angle=self.sidewall_angle,
            width_to_z=self.width_to_z,
        )

    def shift(self, x, y, z=0):
        dx, dy, dz = float(x), float(y), float(z)
        if hasattr(self, "position") and not isinstance(self, Polygon):
            px, py, pz = self.position
            return self.updated_copy(position=(px + dx, py + dy, pz + dz), z=pz + dz)
        return self.updated_copy(
            vertices=tuple((vx + dx, vy + dy, vz + dz) for vx, vy, vz in self.vertices),
            interiors=tuple(
                tuple((vx + dx, vy + dy, vz + dz) for vx, vy, vz in path)
                for path in self.interiors
            ),
            z=self.z + dz,
        )

    def scale(self, s_x, s_y=None, s_z=None):
        s_x = float(s_x)
        s_y = s_x if s_y is None else float(s_y)
        s_z = (1.0 if s_y != s_x else s_x) if s_z is None else float(s_z)
        if not self.vertices:
            return self
        center = _path_center(self.vertices)

        def scaled(path):
            return tuple(
                (
                    center[0] + (x - center[0]) * s_x,
                    center[1] + (y - center[1]) * s_y,
                    center[2] + (z - center[2]) * s_z,
                )
                for x, y, z in path
            )

        return self._as_polygon(
            vertices=scaled(self.vertices),
            interiors=tuple(scaled(path) for path in self.interiors),
            depth=self.depth * s_z,
        )

    def rotate(self, angle, axis="z", point=None):
        if not self.vertices:
            return self
        center = _path_center(self.vertices) if point is None else _position(point)
        angle_rad = np.radians(float(angle))
        return self._as_polygon(
            vertices=_rotate(self.vertices, angle_rad, axis, center),
            interiors=tuple(
                _rotate(path, angle_rad, axis, center) for path in self.interiors
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class Polygon(PlanarStructure):
    vertices: Any = ()
    material: Any = None
    color: str = field(default=DEFAULT_COLOR, metadata={"beamz_cache": False})
    optimize: bool = False
    interiors: Any = ()
    depth: float = 0.0
    z: float = 0.0
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0

    def __post_init__(self):
        z = float(0.0 if self.z is None else self.z)
        object.__setattr__(self, "z", z)
        object.__setattr__(
            self,
            "depth",
            _positive(
                "depth", 0.0 if self.depth is None else self.depth, allow_zero=True
            ),
        )
        object.__setattr__(self, "vertices", _vertices(self.vertices, z))
        object.__setattr__(
            self,
            "interiors",
            tuple(_vertices(path, z, ccw=False) for path in (self.interiors or ())),
        )
        _normalize_common(self)


@dataclass(frozen=True, eq=False, slots=True)
class Rectangle(PlanarStructure):
    position: Any = (0.0, 0.0, 0.0)
    width: float = 1.0
    height: float = 1.0
    depth: float = 1.0
    material: Any = None
    color: str = field(default=DEFAULT_COLOR, metadata={"beamz_cache": False})
    is_pml: bool = False
    optimize: bool = False
    z: float | None = None
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0
    vertices: tuple = field(init=False)
    interiors: tuple = field(init=False, default=())

    def __post_init__(self):
        position = _position(self.position, self.z)
        width = _positive("width", self.width)
        height = _positive("height", self.height)
        depth = _positive("depth", self.depth, allow_zero=True)
        x, y, z = position
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "is_pml", bool(self.is_pml))
        object.__setattr__(
            self,
            "vertices",
            (
                (x, y, z),
                (x + width, y, z),
                (x + width, y + height, z),
                (x, y + height, z),
            ),
        )
        _normalize_common(self)

    def scale(self, s_x, s_y=None, s_z=None):
        s_y = s_x if s_y is None else s_y
        s_z = (1.0 if s_y != s_x else s_x) if s_z is None else s_z
        return self.updated_copy(
            width=self.width * float(s_x),
            height=self.height * float(s_y),
            depth=self.depth * float(s_z),
        )

    def _mesh_feature_sizes(self):
        return self.width, self.height, self.depth or None


def _circle_vertices(position, radius, points):
    if int(points) < 3:
        raise ValueError(f"points must be at least 3, got {points}")
    theta = np.linspace(0.0, 2.0 * np.pi, int(points), endpoint=False)
    return tuple(
        (
            position[0] + radius * np.cos(t),
            position[1] + radius * np.sin(t),
            position[2],
        )
        for t in theta
    )


@dataclass(frozen=True, eq=False, slots=True)
class Circle(PlanarStructure):
    position: Any = (0.0, 0.0)
    radius: float = 1.0
    points: int = 32
    material: Any = None
    color: str = field(default=DEFAULT_COLOR, metadata={"beamz_cache": False})
    optimize: bool = False
    depth: float = 0.0
    z: float = 0.0
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0
    vertices: tuple = field(init=False)
    interiors: tuple = field(init=False, default=())

    def __post_init__(self):
        position = _position(self.position, self.z)
        radius = _positive("radius", self.radius)
        depth = _positive("depth", self.depth, allow_zero=True)
        points = int(self.points)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "z", position[2])
        object.__setattr__(self, "vertices", _circle_vertices(position, radius, points))
        _normalize_common(self)

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is not None and not np.isclose(float(s_x), float(s_y)):
            return PlanarStructure.scale(self, s_x, s_y, s_z)
        return self.updated_copy(radius=self.radius * float(s_x))

    def _mesh_feature_sizes(self):
        diameter = 2.0 * self.radius
        return diameter, diameter, self.depth or None


@dataclass(frozen=True, eq=False, slots=True)
class Ring(PlanarStructure):
    position: Any = (0.0, 0.0)
    inner_radius: float = 1.0
    outer_radius: float = 2.0
    material: Any = None
    color: str = field(default=DEFAULT_COLOR, metadata={"beamz_cache": False})
    optimize: bool = False
    points: int = 256
    depth: float = 0.0
    z: float | None = None
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0
    vertices: tuple = field(init=False)
    interiors: tuple = field(init=False)

    def __post_init__(self):
        position = _position(self.position, self.z)
        inner = _positive("inner_radius", self.inner_radius)
        outer = _positive("outer_radius", self.outer_radius)
        if outer <= inner:
            raise ValueError(
                f"outer_radius ({outer}) must be greater than inner_radius ({inner})"
            )
        depth = _positive("depth", self.depth, allow_zero=True)
        points = int(self.points)
        outer_vertices = _circle_vertices(position, outer, points)
        inner_vertices = tuple(reversed(_circle_vertices(position, inner, points)))
        for name, value in (
            ("position", position),
            ("inner_radius", inner),
            ("outer_radius", outer),
            ("points", points),
            ("depth", depth),
            ("z", position[2]),
            ("vertices", outer_vertices),
            ("interiors", (inner_vertices,)),
        ):
            object.__setattr__(self, name, value)
        _normalize_common(self)

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is not None and not np.isclose(float(s_x), float(s_y)):
            return PlanarStructure.scale(self, s_x, s_y, s_z)
        return self.updated_copy(
            inner_radius=self.inner_radius * float(s_x),
            outer_radius=self.outer_radius * float(s_x),
        )

    def _mesh_feature_sizes(self):
        wall = self.outer_radius - self.inner_radius
        return wall, wall, self.depth or None


@dataclass(frozen=True, eq=False, slots=True)
class CircularBend(PlanarStructure):
    position: Any = (0.0, 0.0)
    inner_radius: float = 1.0
    outer_radius: float = 2.0
    angle: float = 90.0
    rotation: float = 0.0
    material: Any = None
    facecolor: str | None = None
    optimize: bool = False
    points: int = 64
    depth: float = 0.0
    z: float = 0.0
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0
    vertices: tuple = field(init=False)
    interiors: tuple = field(init=False, default=())

    def __post_init__(self):
        position = _position(self.position, self.z)
        inner = _positive("inner_radius", self.inner_radius)
        outer = _positive("outer_radius", self.outer_radius)
        if outer <= inner:
            raise ValueError(
                f"outer_radius ({outer}) must be greater than inner_radius ({inner})"
            )
        depth = _positive("depth", self.depth, allow_zero=True)
        points = int(self.points)
        if points < 2:
            raise ValueError(f"points must be at least 2, got {points}")
        theta = np.linspace(0.0, np.radians(float(self.angle)), points) + np.radians(
            float(self.rotation)
        )
        outer_vertices = tuple(
            (
                position[0] + outer * np.cos(t),
                position[1] + outer * np.sin(t),
                position[2],
            )
            for t in theta
        )
        inner_vertices = tuple(
            (
                position[0] + inner * np.cos(t),
                position[1] + inner * np.sin(t),
                position[2],
            )
            for t in reversed(theta)
        )
        for name, value in (
            ("position", position),
            ("inner_radius", inner),
            ("outer_radius", outer),
            ("points", points),
            ("depth", depth),
            ("z", position[2]),
            ("vertices", outer_vertices + inner_vertices),
        ):
            object.__setattr__(self, name, value)
        _normalize_common(self)

    @property
    def color(self):
        return DEFAULT_COLOR if self.facecolor is None else str(self.facecolor)

    def updated_copy(self, **changes):
        if "color" in changes:
            changes["facecolor"] = changes.pop("color")
        return Structure.updated_copy(self, **changes)

    def rotate(self, angle, axis="z", point=None):
        if axis == "z" and (
            point is None or tuple(_position(point)) == tuple(self.position)
        ):
            return self.updated_copy(rotation=(self.rotation + float(angle)) % 360.0)
        return PlanarStructure.rotate(self, angle, axis, point)

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is not None and not np.isclose(float(s_x), float(s_y)):
            return PlanarStructure.scale(self, s_x, s_y, s_z)
        return self.updated_copy(
            inner_radius=self.inner_radius * float(s_x),
            outer_radius=self.outer_radius * float(s_x),
        )

    def _mesh_feature_sizes(self):
        wall = self.outer_radius - self.inner_radius
        return wall, wall, self.depth or None


@dataclass(frozen=True, eq=False, slots=True)
class Taper(PlanarStructure):
    position: Any = (0.0, 0.0)
    input_width: float = 1.0
    output_width: float = 0.5
    length: float = 1.0
    material: Any = None
    color: str = field(default=DEFAULT_COLOR, metadata={"beamz_cache": False})
    optimize: bool = False
    depth: float = 0.0
    z: float = 0.0
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0
    vertices: tuple = field(init=False)
    interiors: tuple = field(init=False, default=())

    def __post_init__(self):
        position = _position(self.position, self.z)
        input_width = _positive("input_width", self.input_width)
        output_width = _positive("output_width", self.output_width)
        length = _positive("length", self.length)
        depth = _positive("depth", self.depth, allow_zero=True)
        x, y, z = position
        vertices = (
            (x, y - input_width / 2, z),
            (x + length, y - output_width / 2, z),
            (x + length, y + output_width / 2, z),
            (x, y + input_width / 2, z),
        )
        for name, value in (
            ("position", position),
            ("input_width", input_width),
            ("output_width", output_width),
            ("length", length),
            ("depth", depth),
            ("z", z),
            ("vertices", vertices),
        ):
            object.__setattr__(self, name, value)
        _normalize_common(self)

    def _mesh_feature_sizes(self):
        return (
            self.length,
            min(self.input_width, self.output_width),
            self.depth or None,
        )


@dataclass(frozen=True, eq=False, slots=True)
class Box(Structure):
    center: Any = (0.0, 0.0, 0.0)
    size: Any = (1.0, 1.0, 1.0)
    material: Any = None

    def __post_init__(self):
        center = _position(self.center)
        size = tuple(self.size)
        if len(size) == 2:
            size = (*size, 0.0)
        if len(size) != 3:
            raise ValueError("Box center and size must be 2D or 3D coordinate tuples.")
        size = tuple(float(value) for value in size)
        if any(np.isnan(value) or value < 0.0 for value in size):
            raise ValueError(f"Box sizes must be non-negative, got {size!r}.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        _normalize_common(self)

    @property
    def lower(self):
        return tuple(c - 0.5 * s for c, s in zip(self.center, self.size, strict=True))

    @property
    def upper(self):
        return tuple(c + 0.5 * s for c, s in zip(self.center, self.size, strict=True))

    position = property(lambda self: self.lower)
    width = property(lambda self: self.size[0])
    height = property(lambda self: self.size[1])
    depth = property(lambda self: self.size[2])
    is_3d = property(lambda self: bool(self.size[2] != 0.0))

    def shifted(self, offset):
        return self.updated_copy(
            center=tuple(c + d for c, d in zip(self.center, offset, strict=True))
        )

    def shift(self, x, y, z=0):
        return self.shifted((float(x), float(y), float(z)))

    def to_rectangle(self, offset=(0.0, 0.0, 0.0), material=None):
        shifted = self.shifted(offset)
        if not all(np.isfinite(value) for value in (*shifted.lower, *shifted.size)):
            raise ValueError(
                "Infinite Box sizes must be clipped by Simulation(domain=...) before rasterization."
            )
        return Rectangle(
            position=shifted.lower,
            width=max(shifted.width, np.finfo(float).eps),
            height=max(shifted.height, np.finfo(float).eps),
            depth=max(shifted.depth, 0.0),
            material=self.material if material is None else material,
        )

    def _mesh_feature_sizes(self):
        return tuple(value if value > 0.0 else None for value in self.size)


@dataclass(frozen=True, eq=False, slots=True)
class Sphere(Structure):
    position: Any = (0.0, 0.0, 0.0)
    radius: float = 1.0
    material: Any = None
    color: str = field(default=DEFAULT_COLOR, metadata={"beamz_cache": False})
    optimize: bool = False
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "position", _position(self.position))
        object.__setattr__(self, "radius", _positive("radius", self.radius))
        _normalize_common(self)

    center = property(lambda self: self.position)
    vertices = property(lambda self: ())
    interiors = property(lambda self: ())
    depth = property(lambda self: 2.0 * self.radius)
    z = property(lambda self: self.position[2] - self.radius)

    def shift(self, x, y, z=0):
        px, py, pz = self.position
        return self.updated_copy(position=(px + float(x), py + float(y), pz + float(z)))

    def scale(self, s_x, s_y=None, s_z=None):
        if any(
            value is not None and not np.isclose(float(s_x), float(value))
            for value in (s_y, s_z)
        ):
            raise ValueError("Sphere scaling must be uniform.")
        return self.updated_copy(radius=self.radius * float(s_x))

    def _mesh_feature_sizes(self):
        diameter = 2.0 * self.radius
        return diameter, diameter, diameter
