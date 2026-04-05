import colorsys
import random

import numpy as np


def _rotate_vertices(vertices, angle_rad, axis, center):
    """Rotate a list of 3D vertices around center by angle_rad on the given axis."""
    cx, cy, cz = center
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "z":
        return [
            (
                cx + (v[0] - cx) * cos_a - (v[1] - cy) * sin_a,
                cy + (v[0] - cx) * sin_a + (v[1] - cy) * cos_a,
                v[2],
            )
            for v in vertices
        ]
    elif axis == "x":
        return [
            (
                v[0],
                cy + (v[1] - cy) * cos_a - (v[2] - cz) * sin_a,
                cz + (v[1] - cy) * sin_a + (v[2] - cz) * cos_a,
            )
            for v in vertices
        ]
    elif axis == "y":
        return [
            (
                cx + (v[0] - cx) * cos_a + (v[2] - cz) * sin_a,
                v[1],
                cz - (v[0] - cx) * sin_a + (v[2] - cz) * cos_a,
            )
            for v in vertices
        ]
    else:
        raise ValueError(f"Invalid rotation axis '{axis}'. Must be 'x', 'y', or 'z'.")


def _normalize_position(position, z=None):
    """Ensure position is a 3-tuple, optionally overriding z."""
    if len(position) == 2:
        position = (position[0], position[1], 0.0)
    elif len(position) != 3:
        raise ValueError("Position must be (x,y) or (x,y,z)")
    if z is not None:
        position = (position[0], position[1], z)
    return position


def _require_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_nonnegative(name, value):
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _vertices_center(vertices):
    return tuple(sum(v[i] for v in vertices) / len(vertices) for i in range(3))


def _vertices_bbox(vertices):
    x_vals = [v[0] for v in vertices]
    y_vals = [v[1] for v in vertices]
    z_vals = [v[2] for v in vertices]
    return (
        min(x_vals),
        min(y_vals),
        min(z_vals),
        max(x_vals),
        max(y_vals),
        max(z_vals),
    )


def _circle_vertices(position, radius, points, *, theta=None, reverse=False):
    theta = (
        np.linspace(0, 2 * np.pi, points, endpoint=False)
        if theta is None
        else np.asarray(theta, dtype=float)
    )
    angles = reversed(theta) if reverse else theta
    return [
        (
            position[0] + radius * np.cos(t),
            position[1] + radius * np.sin(t),
            position[2],
        )
        for t in angles
    ]


def _ring_vertices(position, inner_radius, outer_radius, points):
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    outer = _circle_vertices(position, outer_radius, points, theta=theta)
    inner = _circle_vertices(position, inner_radius, points, theta=theta, reverse=True)
    return outer, inner


def _bend_vertices(position, inner_radius, outer_radius, angle, rotation, points):
    theta = np.linspace(0, np.radians(angle), points)
    rotation_rad = np.radians(rotation)
    angles = theta + rotation_rad
    outer = _circle_vertices(position, outer_radius, points, theta=angles)
    inner = _circle_vertices(position, inner_radius, points, theta=angles, reverse=True)
    return outer + inner


def _update_box_metrics(shape):
    min_x, min_y, min_z, max_x, max_y, max_z = _vertices_bbox(shape.vertices)
    shape.position = (min_x, min_y, min_z)
    shape.width = max_x - min_x
    shape.height = max_y - min_y
    shape.depth = max_z - min_z


def _default_plot_style(shape, facecolor=None, alpha=None, linestyle=None):
    return (
        shape.color if facecolor is None else facecolor,
        1 if alpha is None else alpha,
        "-" if linestyle is None else linestyle,
    )


class Polygon:
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
        self.vertices = self._process_vertices(
            vertices if vertices is not None else [], z
        )
        self.interiors = [
            self._process_vertices(interior, z, ensure_ccw=False)
            for interior in (interiors if interiors is not None else [])
        ]
        self.material = material
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color_consistent()
        self.depth = depth if depth is not None else 0
        self.z = z if z is not None else 0

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
                result.append(v)
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
        if hasattr(self, "position") and self.position is not None:
            self.position = (
                self.position[0] + x,
                self.position[1] + y,
                self.position[2] + z,
            )
        if self.vertices:
            self.vertices = [(v[0] + x, v[1] + y, v[2] + z) for v in self.vertices]
        self.interiors = [
            [(v[0] + x, v[1] + y, v[2] + z) for v in path]
            for path in self.interiors
            if path
        ]
        return self

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is None:
            s_y = s_x
        if s_z is None:
            s_z = 1.0 if s_y != s_x else s_x
        if self.vertices:
            x_center, y_center, z_center = _vertices_center(self.vertices)
            self.vertices = [
                (
                    x_center + (v[0] - x_center) * s_x,
                    y_center + (v[1] - y_center) * s_y,
                    z_center + (v[2] - z_center) * s_z,
                )
                for v in self.vertices
            ]
            new_interiors_paths = []
            for interior_path in self.interiors:
                if interior_path:
                    new_interiors_paths.append(
                        [
                            (
                                x_center + (v[0] - x_center) * s_x,
                                y_center + (v[1] - y_center) * s_y,
                                z_center + (v[2] - z_center) * s_z,
                            )
                            for v in interior_path
                        ]
                    )
            self.interiors = new_interiors_paths
        return self

    def rotate(self, angle, axis="z", point=None):
        if self.vertices:
            angle_rad = np.radians(angle)
            center = (
                _vertices_center(self.vertices)
                if point is None
                else (point[0], point[1], point[2] if len(point) > 2 else 0)
            )

            self.vertices = _rotate_vertices(self.vertices, angle_rad, axis, center)
            self.interiors = [
                _rotate_vertices(path, angle_rad, axis, center)
                for path in self.interiors
                if path
            ]
        return self

    def add_to_plot(
        self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None
    ):
        from beamz.visual.design_viz import draw_polygon

        return draw_polygon(
            ax,
            self,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linestyle=linestyle,
        )

    def copy(self):
        copied_interiors = (
            [list(path) for path in self.interiors if path] if self.interiors else []
        )
        return Polygon(
            vertices=list(self.vertices) if self.vertices else [],
            interiors=copied_interiors,
            material=self.material,
            color=self.color,
            optimize=self.optimize,
            depth=self.depth,
            z=self.z,
        )

    def get_bounding_box(self):
        if not self.vertices or len(self.vertices) == 0:
            return (0, 0, 0, 0, 0, 0)
        min_x, min_y, min_z, max_x, max_y, max_z = _vertices_bbox(self.vertices)

        # Expand Z-range by depth if present
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
        # 3D containment check if z is provided
        if z is not None and hasattr(self, "depth") and self.depth > 0:
            if not (self.z <= z <= self.z + self.depth):
                return False

        exterior_path = self.vertices
        interior_paths = self.interiors
        if not exterior_path:
            return False
        if not self._point_in_polygon_single_path(x, y, exterior_path):
            return False
        for interior_path_pts in interior_paths:
            if interior_path_pts and self._point_in_polygon_single_path(
                x, y, interior_path_pts
            ):
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
        self.position = position
        self.width = width
        self.height = height
        self.depth = depth
        self.is_pml = is_pml

    def get_bounding_box(self):
        if not hasattr(self, "vertices") or len(self.vertices) == 0:
            x, y, z = self.position
            return (x, y, z, x + self.width, y + self.height, z + self.depth)
        return super().get_bounding_box()

    def rotate(self, angle, axis="z", point=None):
        super().rotate(angle, axis, point)
        _update_box_metrics(self)
        return self

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is None:
            s_y = s_x
        if s_z is None:
            s_z = 1.0 if s_y != s_x else s_x
        super().scale(s_x, s_y, s_z)
        self.width *= s_x
        self.height *= s_y
        self.depth *= s_z
        return self

    def copy(self):
        new_rect = Rectangle(
            self.position,
            self.width,
            self.height,
            self.depth,
            self.material,
            self.color,
            self.is_pml,
            self.optimize,
        )
        new_rect.vertices = [(x, y, z) for x, y, z in self.vertices]
        return new_rect


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
        self.position = position
        self.radius = radius
        self.points = points

    def scale(self, s_x, s_y=None, s_z=None):
        self.radius *= s_x
        self.vertices = _circle_vertices(self.position, self.radius, self.points)
        return self

    def copy(self):
        return Circle(
            position=self.position,
            radius=self.radius,
            points=self.points,
            material=self.material,
            color=self.color,
            optimize=self.optimize,
            depth=self.depth,
            z=self.z,
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
        outer_ext_vertices, inner_int_vertices_cw = _ring_vertices(
            position, inner_radius, outer_radius, points
        )
        super().__init__(
            vertices=outer_ext_vertices,
            interiors=[inner_int_vertices_cw] if inner_int_vertices_cw else [],
            material=material,
            color=color,
            optimize=optimize,
            depth=depth,
            z=position[2],
        )
        self.points = points
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def scale(self, s_x, s_y=None, s_z=None):
        self.inner_radius *= s_x
        self.outer_radius *= s_x
        outer_vertices, inner_vertices = _ring_vertices(
            self.position, self.inner_radius, self.outer_radius, self.points
        )
        self.vertices = outer_vertices
        self.interiors = [inner_vertices]
        return self

    def add_to_plot(
        self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None
    ):
        facecolor, alpha, linestyle = _default_plot_style(
            self, facecolor, alpha, linestyle
        )
        return super().add_to_plot(
            ax,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linestyle=linestyle,
        )

    def copy(self):
        return Ring(
            position=self.position,
            inner_radius=self.inner_radius,
            outer_radius=self.outer_radius,
            material=self.material,
            color=self.color,
            optimize=self.optimize,
            points=self.points,
            depth=self.depth,
            z=self.z,
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
        self.points = points
        vertices = _bend_vertices(position, inner_radius, outer_radius, angle, rotation, points)
        super().__init__(
            vertices=vertices,
            material=material,
            color=facecolor,
            optimize=optimize,
            depth=depth,
            z=z,
        )
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.angle = angle
        self.rotation = rotation

    def rotate(self, angle, axis="z", point=None):
        if axis == "z":
            self.rotation = (self.rotation + angle) % 360
        super().rotate(angle, axis, point or self.position)
        return self

    def scale(self, s_x, s_y=None, s_z=None):
        self.inner_radius *= s_x
        self.outer_radius *= s_x
        self.vertices = _bend_vertices(
            self.position,
            self.inner_radius,
            self.outer_radius,
            self.angle,
            self.rotation,
            self.points,
        )
        return self

    def add_to_plot(
        self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None
    ):
        facecolor, alpha, linestyle = _default_plot_style(
            self, facecolor, alpha, linestyle
        )
        return super().add_to_plot(
            ax,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linestyle=linestyle,
        )

    def copy(self):
        return CircularBend(
            self.position,
            self.inner_radius,
            self.outer_radius,
            self.angle,
            self.rotation,
            self.material,
            self.color,
            self.optimize,
            self.points,
            self.depth,
            self.z,
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
        x, y, z = position
        vertices = [
            (x, y - input_width / 2, z),
            (x + length, y - output_width / 2, z),
            (x + length, y + output_width / 2, z),
            (x, y + input_width / 2, z),
        ]
        super().__init__(
            vertices=vertices,
            material=material,
            color=color,
            optimize=optimize,
            depth=depth,
            z=z,
        )
        self.position = position
        self.input_width = input_width
        self.output_width = output_width
        self.length = length
        self.optimize = optimize

    def rotate(self, angle, axis="z", point=None):
        super().rotate(angle, axis, point)
        min_x, min_y, min_z, max_x, _, _ = _vertices_bbox(self.vertices)
        self.position = (min_x, min_y, min_z)
        self.length = max_x - min_x
        return self

    def copy(self):
        new_taper = Taper(
            self.position,
            self.input_width,
            self.output_width,
            self.length,
            self.material,
            self.color,
            self.optimize,
            self.depth,
            self.z,
        )
        new_taper.vertices = [(x, y, z) for x, y, z in self.vertices]
        return new_taper


class Sphere(Polygon):
    def __init__(
        self, position=(0, 0, 0), radius=1, material=None, color=None, optimize=False
    ):
        """Create a 3D sphere at position (x,y,z) with specified radius."""
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
        self.position = position
        self.radius = radius

    def get_bounding_box(self):
        x, y, z = self.position
        r = self.radius
        return (x - r, y - r, z - r, x + r, y + r, z + r)

    def point_in_polygon(self, x, y, z=0):
        cx, cy, cz = self.position
        return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= self.radius**2

    def copy(self):
        return Sphere(
            self.position, self.radius, self.material, self.color, self.optimize
        )
