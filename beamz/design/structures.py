from __future__ import annotations

import colorsys
import copy
import random
from dataclasses import replace

import numpy as np

from beamz.design.geometry_ops import (
    bend_vertices as _bend_vertices,
    circle_vertices as _circle_vertices,
    freeze_interiors as _freeze_interiors,
    freeze_vertices as _freeze_vertices,
    normalize_position as _normalize_position,
    require_nonnegative as _require_nonnegative,
    require_positive as _require_positive,
    ring_vertices as _ring_vertices,
    rotate_vertices as _rotate_vertices,
    transform_geometry as _transform_geometry,
    vertices_bbox as _vertices_bbox,
    vertices_center as _vertices_center,
)
from beamz.design.materials import material_from_spec, material_to_spec
from beamz.design.structure_specs import StructureSpec, structure_kind as _structure_kind


def _replace_from_bbox(structure, *, include_length=False):
    min_x, min_y, min_z, max_x, max_y, max_z = _vertices_bbox(structure.vertices)
    changes = {"position": (min_x, min_y, min_z)}
    if include_length:
        changes["length"] = max_x - min_x
    else:
        changes.update(width=max_x - min_x, height=max_y - min_y, depth=max_z - min_z)
    return structure._replace_spec(**changes)


def structure_from_spec(spec):
    material = material_from_spec(spec.material)
    kind = _structure_kind(spec)
    if kind == "Rectangle":
        return Rectangle(
            position=spec.position or (0.0, 0.0, spec.z),
            width=spec.width,
            height=spec.height,
            depth=spec.depth,
            material=material,
            color=spec.color,
            is_pml=spec.is_pml,
            optimize=spec.optimize,
        )
    if kind == "Circle":
        return Circle(
            position=spec.position or (0.0, 0.0, spec.z),
            radius=spec.radius,
            points=spec.points or 32,
            material=material,
            color=spec.color,
            optimize=spec.optimize,
            depth=spec.depth,
            z=spec.z,
        )
    if kind == "Ring":
        return Ring(
            position=spec.position or (0.0, 0.0, spec.z),
            inner_radius=spec.inner_radius,
            outer_radius=spec.outer_radius,
            material=material,
            color=spec.color,
            optimize=spec.optimize,
            points=spec.points or 256,
            depth=spec.depth,
            z=spec.z,
        )
    if kind == "CircularBend":
        return CircularBend(
            position=spec.position or (0.0, 0.0, spec.z),
            inner_radius=spec.inner_radius,
            outer_radius=spec.outer_radius,
            angle=spec.angle,
            rotation=spec.rotation or 0.0,
            material=material,
            facecolor=spec.color,
            optimize=spec.optimize,
            points=spec.points or 64,
            depth=spec.depth,
            z=spec.z,
        )
    if kind == "Taper":
        return Taper(
            position=spec.position or (0.0, 0.0, spec.z),
            input_width=spec.input_width,
            output_width=spec.output_width,
            length=spec.length,
            material=material,
            color=spec.color,
            optimize=spec.optimize,
            depth=spec.depth,
            z=spec.z,
        )
    if kind == "Sphere":
        return Sphere(
            position=spec.position or (0.0, 0.0, 0.0),
            radius=spec.radius,
            material=material,
            color=spec.color,
            optimize=spec.optimize,
        )
    poly = Polygon(
        vertices=spec.vertices,
        material=material,
        color=spec.color,
        optimize=spec.optimize,
        interiors=spec.interiors,
        depth=spec.depth,
        z=spec.z,
    )
    poly._replace_spec(position=spec.position)
    return poly


class Polygon:
    _SPEC_FIELDS = frozenset(StructureSpec.__dataclass_fields__.keys())

    def __init__(
        self,
        vertices=None,
        material=None,
        color=None,
        optimize=False,
        interiors=None,
        depth=0,
        z=0,
    ):
        processed_vertices = self._process_vertices(vertices if vertices is not None else [], z)
        processed_interiors = [
            self._process_vertices(interior, z, ensure_ccw=False)
            for interior in (interiors if interiors is not None else [])
        ]
        object.__setattr__(
            self,
            "spec",
            StructureSpec(
                vertices=processed_vertices,
                interiors=processed_interiors,
                material=material,
                color=color if color is not None else self.get_random_color_consistent(),
                optimize=optimize,
                depth=depth if depth is not None else 0,
                z=z if z is not None else 0,
            ),
        )
        object.__setattr__(self, "_material", material if material is not None else material_from_spec(self.spec.material))

    def __getattr__(self, name):
        if name == "material":
            return self.__dict__.get("_material")
        spec = self.__dict__.get("spec")
        if spec is not None and hasattr(spec, name):
            return getattr(spec, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name == "spec":
            object.__setattr__(self, name, value)
            return
        if name == "_material":
            object.__setattr__(self, name, value)
            return
        if name == "material":
            object.__setattr__(self, "_material", value)
            object.__setattr__(self, "spec", replace(self.spec, material=material_to_spec(value)))
            return
        if name in self._SPEC_FIELDS and "spec" in self.__dict__:
            if name == "vertices":
                value = _freeze_vertices(value)
            elif name == "interiors":
                value = _freeze_interiors(value)
            elif name == "position" and value is not None:
                value = tuple(float(v) for v in value)
            elif name == "material":
                object.__setattr__(self, "_material", value)
                value = material_to_spec(value)
            object.__setattr__(self, "spec", replace(self.spec, **{name: value}))
            return
        object.__setattr__(self, name, value)

    def _replace_spec(self, **changes):
        if "vertices" in changes:
            changes["vertices"] = _freeze_vertices(changes["vertices"])
        if "interiors" in changes:
            changes["interiors"] = _freeze_interiors(changes["interiors"])
        if "position" in changes and changes["position"] is not None:
            changes["position"] = tuple(float(v) for v in changes["position"])
        if "material" in changes:
            object.__setattr__(self, "_material", changes["material"])
            changes["material"] = material_to_spec(changes["material"])
        object.__setattr__(self, "spec", replace(self.spec, **changes))
        return self

    def with_spec(self, spec=None, /, **changes):
        base_spec = self.spec if spec is None else spec
        if not isinstance(base_spec, StructureSpec):
            raise TypeError("with_spec expects a StructureSpec or spec field updates")
        if changes:
            base_spec = replace(base_spec, **changes)
        new = copy.copy(self)
        object.__setattr__(new, "spec", base_spec)
        object.__setattr__(new, "_material", material_from_spec(base_spec.material))
        return new

    def _process_vertices(self, vertices, z=0, ensure_ccw=True):
        if not vertices:
            return []
        vertices_3d = self._ensure_3d_vertices(vertices)
        if ensure_ccw and len(vertices_3d) >= 3:
            vertices_2d = [(v[0], v[1]) for v in vertices_3d]
            vertices_2d = self._ensure_ccw_vertices(vertices_2d)
            vertices_3d = [
                (x, y, vertices_3d[i][2] if len(vertices_3d[i]) > 2 else z)
                for i, (x, y) in enumerate(vertices_2d)
            ]
        return vertices_3d

    def _ensure_ccw_vertices(self, vertices_2d):
        if len(vertices_2d) < 3:
            return vertices_2d
        area = 0
        for i in range(len(vertices_2d)):
            j = (i + 1) % len(vertices_2d)
            area += vertices_2d[i][0] * vertices_2d[j][1]
            area -= vertices_2d[j][0] * vertices_2d[i][1]
        if area < 0:
            return vertices_2d[::-1]
        return vertices_2d

    def _ensure_3d_vertices(self, vertices):
        if not vertices:
            return []
        result = []
        for v in vertices:
            if len(v) == 2:
                result.append((v[0], v[1], 0.0))
            elif len(v) == 3:
                result.append(tuple(v))
            else:
                raise ValueError(f"Vertex must have 2 or 3 coordinates, got {len(v)}")
        return result

    def _vertices_2d(self, vertices=None):
        if vertices is None:
            vertices = self.vertices
        return [(v[0], v[1]) for v in vertices]

    def get_random_color_consistent(self, saturation=0.6, value=0.7):
        hue = random.random()
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    def shift(self, x, y, z=0):
        position = self.position
        vertices, interiors = _transform_geometry(
            self.vertices,
            self.interiors,
            lambda v: (v[0] + x, v[1] + y, v[2] + z),
        )
        changes = {"vertices": vertices, "interiors": interiors}
        if position is not None:
            changes["position"] = (position[0] + x, position[1] + y, position[2] + z)
        if self.z is not None:
            changes["z"] = self.z + z
        return self._replace_spec(**changes)

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is None:
            s_y = s_x
        if s_z is None:
            s_z = 1.0 if s_y != s_x else s_x
        if not self.vertices:
            return self
        x_center, y_center, z_center = _vertices_center(self.vertices)
        vertices, interiors = _transform_geometry(
            self.vertices,
            self.interiors,
            lambda v: (
                x_center + (v[0] - x_center) * s_x,
                y_center + (v[1] - y_center) * s_y,
                z_center + (v[2] - z_center) * s_z,
            ),
        )
        return self._replace_spec(vertices=vertices, interiors=interiors)

    def rotate(self, angle, axis="z", point=None):
        if not self.vertices:
            return self
        angle_rad = np.radians(angle)
        center = (
            _vertices_center(self.vertices)
            if point is None
            else (point[0], point[1], point[2] if len(point) > 2 else 0)
        )
        return self._replace_spec(
            vertices=_rotate_vertices(self.vertices, angle_rad, axis, center),
            interiors=[
                _rotate_vertices(path, angle_rad, axis, center)
                for path in self.interiors
                if path
            ],
        )

    def add_to_plot(
        self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None
    ):
        from beamz.visual.design_viz import draw_polygon
        return draw_polygon(
            ax,
            self,
            facecolor=self.color if facecolor is None else facecolor,
            edgecolor=edgecolor,
            alpha=1 if alpha is None else alpha,
            linestyle="-" if linestyle is None else linestyle,
        )

    def copy(self):
        return copy.copy(self)

    def get_bounding_box(self):
        if not self.vertices:
            return (0, 0, 0, 0, 0, 0)
        min_x, min_y, min_z, max_x, max_y, max_z = _vertices_bbox(self.vertices)
        max_z = max(max_z, min_z + getattr(self, "depth", 0))
        return (min_x, min_y, min_z, max_x, max_y, max_z)

    def _point_in_polygon_single_path(self, x, y, path_vertices):
        if not path_vertices:
            return False
        path_2d = self._vertices_2d(path_vertices)
        n = len(path_2d)
        inside = False
        p1x, p1y = path_2d[0]
        for i in range(n + 1):
            p2x, p2y = path_2d[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def point_in_polygon(self, x, y, z=None):
        if z is not None and hasattr(self, "depth") and self.depth > 0:
            if not (self.z <= z <= self.z + self.depth):
                return False
        if not self.vertices:
            return False
        if not self._point_in_polygon_single_path(x, y, self.vertices):
            return False
        for interior in self.interiors:
            if interior and self._point_in_polygon_single_path(x, y, interior):
                return False
        return True


class Rectangle(Polygon):
    def __init__(
        self,
        position=(0, 0, 0),
        width=1,
        height=1,
        depth=1,
        material=None,
        color=None,
        is_pml=False,
        optimize=False,
        z=None,
    ):
        _require_positive("width", width)
        _require_positive("height", height)
        _require_nonnegative("depth", depth)
        position = _normalize_position(position, z)
        x, y, z_pos = position
        vertices = [
            (x, y, z_pos),
            (x + width, y, z_pos),
            (x + width, y + height, z_pos),
            (x, y + height, z_pos),
        ]
        super().__init__(
            vertices=vertices,
            material=material,
            color=color,
            optimize=optimize,
            depth=depth,
            z=z_pos,
        )
        self._replace_spec(
            position=position,
            width=width,
            height=height,
            depth=depth,
            is_pml=is_pml,
        )

    def rotate(self, angle, axis="z", point=None):
        super().rotate(angle, axis, point)
        return _replace_from_bbox(self)

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is None:
            s_y = s_x
        if s_z is None:
            s_z = 1.0 if s_y != s_x else s_x
        super().scale(s_x, s_y, s_z)
        return self._replace_spec(
            width=self.width * s_x,
            height=self.height * s_y,
            depth=self.depth * s_z,
            position=self.position,
        )


class Circle(Polygon):
    def __init__(
        self,
        position=(0, 0),
        radius=1,
        points=32,
        material=None,
        color=None,
        optimize=False,
        depth=0,
        z=0,
    ):
        _require_positive("radius", radius)
        _require_nonnegative("depth", depth)
        position = _normalize_position(position, z)
        vertices = _circle_vertices(position, radius, points)
        super().__init__(
            vertices=vertices,
            material=material,
            color=color,
            optimize=optimize,
            depth=depth,
            z=position[2],
        )
        self._replace_spec(position=position, radius=radius, points=points)

    def scale(self, s_x, s_y=None, s_z=None):
        radius = self.radius * s_x
        return self._replace_spec(
            radius=radius,
            vertices=_circle_vertices(self.position, radius, self.points),
        )


class Ring(Polygon):
    def __init__(
        self,
        position=(0, 0),
        inner_radius=1,
        outer_radius=2,
        material=None,
        color=None,
        optimize=False,
        points=256,
        depth=0,
        z=None,
    ):
        _require_positive("inner_radius", inner_radius)
        if outer_radius <= inner_radius:
            raise ValueError(
                f"outer_radius ({outer_radius}) must be greater than inner_radius ({inner_radius})"
            )
        _require_nonnegative("depth", depth)
        position = _normalize_position(position, z)
        outer_vertices, inner_vertices = _ring_vertices(position, inner_radius, outer_radius, points)
        super().__init__(
            vertices=outer_vertices,
            interiors=[inner_vertices] if inner_vertices else [],
            material=material,
            color=color,
            optimize=optimize,
            depth=depth,
            z=position[2],
        )
        self._replace_spec(
            position=position,
            points=points,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
        )

    def scale(self, s_x, s_y=None, s_z=None):
        inner_radius = self.inner_radius * s_x
        outer_radius = self.outer_radius * s_x
        outer_vertices, inner_vertices = _ring_vertices(
            self.position, inner_radius, outer_radius, self.points
        )
        return self._replace_spec(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            vertices=outer_vertices,
            interiors=[inner_vertices],
        )


class CircularBend(Polygon):
    def __init__(
        self,
        position=(0, 0),
        inner_radius=1,
        outer_radius=2,
        angle=90,
        rotation=0,
        material=None,
        facecolor=None,
        optimize=False,
        points=64,
        depth=0,
        z=0,
    ):
        _require_positive("inner_radius", inner_radius)
        if outer_radius <= inner_radius:
            raise ValueError(
                f"outer_radius ({outer_radius}) must be greater than inner_radius ({inner_radius})"
            )
        _require_positive("angle", angle)
        _require_nonnegative("depth", depth)
        position = _normalize_position(position, z)
        vertices = _bend_vertices(position, inner_radius, outer_radius, angle, rotation, points)
        super().__init__(
            vertices=vertices,
            material=material,
            color=facecolor,
            optimize=optimize,
            depth=depth,
            z=z,
        )
        self._replace_spec(
            position=position,
            points=points,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            angle=angle,
            rotation=rotation,
        )

    def rotate(self, angle, axis="z", point=None):
        if axis == "z":
            self._replace_spec(rotation=(self.rotation + angle) % 360)
        return super().rotate(angle, axis, point or self.position)

    def scale(self, s_x, s_y=None, s_z=None):
        inner_radius = self.inner_radius * s_x
        outer_radius = self.outer_radius * s_x
        return self._replace_spec(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            vertices=_bend_vertices(
                self.position,
                inner_radius,
                outer_radius,
                self.angle,
                self.rotation,
                self.points,
            ),
        )


class Taper(Polygon):
    def __init__(
        self,
        position=(0, 0),
        input_width=1,
        output_width=0.5,
        length=1,
        material=None,
        color=None,
        optimize=False,
        depth=0,
        z=0,
    ):
        _require_positive("input_width", input_width)
        _require_positive("output_width", output_width)
        _require_positive("length", length)
        _require_nonnegative("depth", depth)
        position = _normalize_position(position, z)
        x, y, z_pos = position
        vertices = [
            (x, y - input_width / 2, z_pos),
            (x + length, y - output_width / 2, z_pos),
            (x + length, y + output_width / 2, z_pos),
            (x, y + input_width / 2, z_pos),
        ]
        super().__init__(
            vertices=vertices,
            material=material,
            color=color,
            optimize=optimize,
            depth=depth,
            z=z_pos,
        )
        self._replace_spec(
            position=position,
            input_width=input_width,
            output_width=output_width,
            length=length,
            optimize=optimize,
        )

    def rotate(self, angle, axis="z", point=None):
        super().rotate(angle, axis, point)
        return _replace_from_bbox(self, include_length=True)


class Sphere(Polygon):
    def __init__(
        self, position=(0, 0, 0), radius=1, material=None, color=None, optimize=False
    ):
        _require_positive("radius", radius)
        position = _normalize_position(position)
        super().__init__(
            vertices=[],
            material=material,
            color=color,
            optimize=optimize,
            depth=2 * radius,
            z=position[2] - radius,
        )
        self._replace_spec(position=position, radius=radius)

    def get_bounding_box(self):
        x, y, z = self.position
        r = self.radius
        return (x - r, y - r, z - r, x + r, y + r, z + r)

    def point_in_polygon(self, x, y, z=0):
        cx, cy, cz = self.position
        return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= self.radius**2
