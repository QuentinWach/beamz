import colorsys
import random

import numpy as np


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
            self._process_vertices_preserve_orientation(interior, z)
            for interior in (interiors if interiors is not None else [])
        ]
        self.material = material
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color_consistent()
        self.depth = depth if depth is not None else 0
        self.z = z if z is not None else 0

    def _process_vertices(self, vertices, z=0):
        if not vertices:
            return []
        vertices_3d = self._ensure_3d_vertices(vertices)
        vertices_2d = [(v[0], v[1]) for v in vertices_3d]
        if len(vertices_2d) >= 3:
            vertices_2d = self._ensure_ccw_vertices(vertices_2d)
            vertices_3d = [
                (x, y, vertices_3d[i][2] if len(vertices_3d[i]) > 2 else z)
                for i, (x, y) in enumerate(vertices_2d)
            ]
        return vertices_3d

    def _process_vertices_preserve_orientation(self, vertices, z=0):
        if not vertices:
            return []
        vertices_3d = self._ensure_3d_vertices(vertices)
        return [(v[0], v[1], v[2] if len(v) > 2 else z) for v in vertices_3d]

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
        if self.vertices:
            self.vertices = [(v[0] + x, v[1] + y, v[2] + z) for v in self.vertices]
        new_interiors_paths = []
        for interior_path in self.interiors:
            if interior_path:
                new_interiors_paths.append(
                    [(v[0] + x, v[1] + y, v[2] + z) for v in interior_path]
                )
        self.interiors = new_interiors_paths
        return self

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is None:
            s_y = s_x
        if s_z is None:
            s_z = 1.0 if s_y != s_x else s_x
        if self.vertices:
            x_center = sum(v[0] for v in self.vertices) / len(self.vertices)
            y_center = sum(v[1] for v in self.vertices) / len(self.vertices)
            z_center = sum(v[2] for v in self.vertices) / len(self.vertices)
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
            if point is None:
                x_center = sum(v[0] for v in self.vertices) / len(self.vertices)
                y_center = sum(v[1] for v in self.vertices) / len(self.vertices)
                z_center = sum(v[2] for v in self.vertices) / len(self.vertices)
            else:
                x_center, y_center, z_center = (
                    point[0],
                    point[1],
                    point[2] if len(point) > 2 else 0,
                )

            if axis == "z":
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                self.vertices = [
                    (
                        x_center
                        + (v[0] - x_center) * cos_a
                        - (v[1] - y_center) * sin_a,
                        y_center
                        + (v[0] - x_center) * sin_a
                        + (v[1] - y_center) * cos_a,
                        v[2],
                    )
                    for v in self.vertices
                ]
            elif axis == "x":
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                self.vertices = [
                    (
                        v[0],
                        y_center
                        + (v[1] - y_center) * cos_a
                        - (v[2] - z_center) * sin_a,
                        z_center
                        + (v[1] - y_center) * sin_a
                        + (v[2] - z_center) * cos_a,
                    )
                    for v in self.vertices
                ]
            elif axis == "y":
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                self.vertices = [
                    (
                        x_center
                        + (v[0] - x_center) * cos_a
                        + (v[2] - z_center) * sin_a,
                        v[1],
                        z_center
                        - (v[0] - x_center) * sin_a
                        + (v[2] - z_center) * cos_a,
                    )
                    for v in self.vertices
                ]
            else:
                raise ValueError(
                    f"Invalid rotation axis '{axis}'. Must be 'x', 'y', or 'z'."
                )
            new_interiors_paths = []
            for interior_path in self.interiors:
                if interior_path:
                    if axis == "z":
                        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                        new_interiors_paths.append(
                            [
                                (
                                    x_center
                                    + (v[0] - x_center) * cos_a
                                    - (v[1] - y_center) * sin_a,
                                    y_center
                                    + (v[0] - x_center) * sin_a
                                    + (v[1] - y_center) * cos_a,
                                    v[2],
                                )
                                for v in interior_path
                            ]
                        )
                    elif axis == "x":
                        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                        new_interiors_paths.append(
                            [
                                (
                                    v[0],
                                    y_center
                                    + (v[1] - y_center) * cos_a
                                    - (v[2] - z_center) * sin_a,
                                    z_center
                                    + (v[1] - y_center) * sin_a
                                    + (v[2] - z_center) * cos_a,
                                )
                                for v in interior_path
                            ]
                        )
                    elif axis == "y":
                        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                        new_interiors_paths.append(
                            [
                                (
                                    x_center
                                    + (v[0] - x_center) * cos_a
                                    + (v[2] - z_center) * sin_a,
                                    v[1],
                                    z_center
                                    - (v[0] - x_center) * sin_a
                                    + (v[2] - z_center) * cos_a,
                                )
                                for v in interior_path
                            ]
                        )
            self.interiors = new_interiors_paths
        return self

    def add_to_plot(
        self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None
    ):
        from beamz.visual.viz import draw_polygon

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
        x_coords = [v[0] for v in self.vertices]
        y_coords = [v[1] for v in self.vertices]
        z_coords = [v[2] for v in self.vertices]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)

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
        if z is not None:
            if len(position) == 2:
                position = (position[0], position[1], z)
            elif len(position) == 3:
                position = (position[0], position[1], z)
        if len(position) == 2:
            position = (position[0], position[1], 0.0)
        elif len(position) == 3:
            position = position
        else:
            raise ValueError("Position must be (x,y) or (x,y,z)")
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

    def shift(self, x, y, z=0):
        self.position = (
            self.position[0] + x,
            self.position[1] + y,
            self.position[2] + z,
        )
        super().shift(x, y, z)
        return self

    def rotate(self, angle, axis="z", point=None):
        super().rotate(angle, axis, point)
        min_x = min(v[0] for v in self.vertices)
        min_y = min(v[1] for v in self.vertices)
        min_z = min(v[2] for v in self.vertices)
        max_x = max(v[0] for v in self.vertices)
        max_y = max(v[1] for v in self.vertices)
        max_z = max(v[2] for v in self.vertices)
        self.position = (min_x, min_y, min_z)
        self.width = max_x - min_x
        self.height = max_y - min_y
        self.depth = max_z - min_z
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
        if len(position) == 2:
            position = (position[0], position[1], 0.0)
        elif len(position) == 3:
            position = position
        else:
            raise ValueError("Position must be (x,y) or (x,y,z)")
        theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
        vertices = [
            (
                position[0] + radius * np.cos(t),
                position[1] + radius * np.sin(t),
                position[2],
            )
            for t in theta
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
        self.radius = radius
        self.points = points

    def shift(self, x, y, z=0):
        self.position = (
            self.position[0] + x,
            self.position[1] + y,
            self.position[2] + z,
        )
        super().shift(x, y, z)
        return self

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is None:
            s_y = s_x
        if s_z is None:
            s_z = 1.0
        self.radius *= s_x
        theta = np.linspace(0, 2 * np.pi, self.points, endpoint=False)
        self.vertices = [
            (
                self.position[0] + self.radius * np.cos(t),
                self.position[1] + self.radius * np.sin(t),
                self.position[2],
            )
            for t in theta
        ]
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
        if z is not None:
            if len(position) == 2:
                position = (position[0], position[1], z)
            elif len(position) == 3:
                position = (position[0], position[1], z)
        if len(position) == 2:
            position = (position[0], position[1], 0.0)
        elif len(position) == 3:
            position = position
        else:
            raise ValueError("Position must be (x,y) or (x,y,z)")
        theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
        outer_ext_vertices = [
            (
                position[0] + outer_radius * np.cos(t),
                position[1] + outer_radius * np.sin(t),
                position[2],
            )
            for t in theta
        ]
        inner_int_vertices_cw = [
            (
                position[0] + inner_radius * np.cos(t),
                position[1] + inner_radius * np.sin(t),
                position[2],
            )
            for t in reversed(theta)
        ]
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

    def shift(self, x, y, z=0):
        self.position = (
            self.position[0] + x,
            self.position[1] + y,
            self.position[2] + z,
        )
        super().shift(x, y, z)
        return self

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is None:
            s_y = s_x
        if s_z is None:
            s_z = 1.0
        self.inner_radius *= s_x
        self.outer_radius *= s_x
        theta = np.linspace(0, 2 * np.pi, self.points, endpoint=False)
        outer_vertices = [
            (
                self.position[0] + self.outer_radius * np.cos(t),
                self.position[1] + self.outer_radius * np.sin(t),
                self.position[2],
            )
            for t in theta
        ]
        inner_vertices = [
            (
                self.position[0] + self.inner_radius * np.cos(t),
                self.position[1] + self.inner_radius * np.sin(t),
                self.position[2],
            )
            for t in reversed(theta)
        ]
        self.vertices = outer_vertices
        self.interiors = [inner_vertices]
        return self

    def add_to_plot(
        self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None
    ):
        if facecolor is None:
            facecolor = self.color
        if alpha is None:
            alpha = 1
        if linestyle is None:
            linestyle = "-"
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
        if len(position) == 2:
            position = (position[0], position[1], 0.0)
        elif len(position) == 3:
            position = position
        else:
            raise ValueError("Position must be (x,y) or (x,y,z)")
        self.points = points
        theta = np.linspace(0, np.radians(angle), points)
        rotation_rad = np.radians(rotation)
        outer_vertices = [
            (
                position[0] + outer_radius * np.cos(t + rotation_rad),
                position[1] + outer_radius * np.sin(t + rotation_rad),
                position[2],
            )
            for t in theta
        ]
        inner_vertices = [
            (
                position[0] + inner_radius * np.cos(t + rotation_rad),
                position[1] + inner_radius * np.sin(t + rotation_rad),
                position[2],
            )
            for t in reversed(theta)
        ]
        vertices = outer_vertices + inner_vertices
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

    def shift(self, x, y, z=0):
        self.position = (
            self.position[0] + x,
            self.position[1] + y,
            self.position[2] + z,
        )
        super().shift(x, y, z)
        return self

    def rotate(self, angle, axis="z", point=None):
        if axis == "z":
            self.rotation = (self.rotation + angle) % 360
        super().rotate(angle, axis, point or self.position)
        return self

    def scale(self, s_x, s_y=None, s_z=None):
        if s_y is None:
            s_y = s_x
        if s_z is None:
            s_z = 1.0
        self.inner_radius *= s_x
        self.outer_radius *= s_x
        theta = np.linspace(0, np.radians(self.angle), self.points)
        rotation_rad = np.radians(self.rotation)
        outer_vertices = [
            (
                self.position[0] + self.outer_radius * np.cos(t + rotation_rad),
                self.position[1] + self.outer_radius * np.sin(t + rotation_rad),
                self.position[2],
            )
            for t in theta
        ]
        inner_vertices = [
            (
                self.position[0] + self.inner_radius * np.cos(t + rotation_rad),
                self.position[1] + self.inner_radius * np.sin(t + rotation_rad),
                self.position[2],
            )
            for t in reversed(theta)
        ]
        self.vertices = outer_vertices + inner_vertices
        return self

    def add_to_plot(
        self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None
    ):
        if facecolor is None:
            facecolor = self.color
        if alpha is None:
            alpha = 1
        if linestyle is None:
            linestyle = "-"
        # Use parent polygon drawing
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
        if len(position) == 2:
            position = (position[0], position[1], 0.0)
        elif len(position) == 3:
            position = position
        else:
            raise ValueError("Position must be (x,y) or (x,y,z)")
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
        min_x = min(v[0] for v in self.vertices)
        min_y = min(v[1] for v in self.vertices)
        min_z = min(v[2] for v in self.vertices)
        max_x = max(v[0] for v in self.vertices)
        max_y = max(v[1] for v in self.vertices)
        max_z = max(v[2] for v in self.vertices)
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
        if len(position) == 2:
            position = (position[0], position[1], 0.0)
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
