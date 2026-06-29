import colorsys
import random
import warnings

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


class Polygon:
    """Planar polygon geometry with optional holes and extrusion metadata.

    Args:
        vertices: Exterior polygon vertices as 2D or 3D coordinate tuples.
        material: Material assigned to the polygon during meshing.
        color: Display color used by plotting helpers.
        optimize: Whether this polygon participates in optimization workflows.
        interiors: Optional interior paths that define holes.
        depth: Extrusion depth along z.
        z: Lower z coordinate for the extruded polygon.
        sidewall_angle: Sidewall taper angle in degrees.
        width_to_z: Fractional depth used as the sidewall width reference.
    """

    def __init__(
        self,
        vertices=None,
        material=None,
        color=None,
        optimize=False,
        interiors=None,
        depth=0,
        z=0,
        sidewall_angle=0.0,
        width_to_z=0.0,
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
        self.sidewall_angle = float(sidewall_angle or 0.0)
        self.width_to_z = float(width_to_z or 0.0)

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
                center = (
                    sum(v[0] for v in self.vertices) / len(self.vertices),
                    sum(v[1] for v in self.vertices) / len(self.vertices),
                    sum(v[2] for v in self.vertices) / len(self.vertices),
                )
            else:
                center = (point[0], point[1], point[2] if len(point) > 2 else 0)

            self.vertices = _rotate_vertices(self.vertices, angle_rad, axis, center)
            self.interiors = [
                _rotate_vertices(path, angle_rad, axis, center)
                for path in self.interiors
                if path
            ]
        return self

    def to_plot_data(
        self, *, facecolor=None, edgecolor="black", alpha=None, linestyle=None
    ):
        from beamz.visual.data import structure_plot_data

        return structure_plot_data(
            self,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=1.0 if alpha is None else alpha,
            linestyle="-" if linestyle is None else linestyle,
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
            sidewall_angle=self.sidewall_angle,
            width_to_z=self.width_to_z,
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

        expand_xy = self._max_taper_expansion()
        min_x -= expand_xy
        max_x += expand_xy
        min_y -= expand_xy
        max_y += expand_xy

        return (min_x, min_y, min_z, max_x, max_y, max_z)

    def has_tapered_sidewalls(self):
        return (
            abs(float(getattr(self, "sidewall_angle", 0.0))) > 1e-12
            and float(getattr(self, "depth", 0.0)) > 0.0
        )

    def _sidewall_reference_z(self):
        return float(self.z) + float(self.width_to_z) * float(self.depth)

    def _taper_buffer_at_z(self, z):
        if not self.has_tapered_sidewalls():
            return 0.0
        return -(float(z) - self._sidewall_reference_z()) * np.tan(
            np.radians(float(self.sidewall_angle))
        )

    def _max_taper_expansion(self):
        if not self.has_tapered_sidewalls():
            return 0.0
        z0 = float(self.z)
        z1 = z0 + float(self.depth)
        return max(0.0, self._taper_buffer_at_z(z0), self._taper_buffer_at_z(z1))

    def _base_shapely_polygon(self):
        from shapely.geometry import Polygon as ShapelyPolygon

        if not self.vertices:
            return None
        shell = [(float(v[0]), float(v[1])) for v in self.vertices]
        holes = []
        for hole in self.interiors:
            if hole:
                holes.append([(float(v[0]), float(v[1])) for v in hole])
        poly = ShapelyPolygon(shell=shell, holes=holes if holes else None)
        if poly.is_empty:
            return None
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not poly.is_valid:
            return None
        return poly

    def shapely_polygon_at_z(self, z):
        if z is None:
            z = float(self.z)
        if float(self.depth) > 0.0 and not (
            float(self.z) <= float(z) <= float(self.z) + float(self.depth)
        ):
            return None

        poly = self._base_shapely_polygon()
        if poly is None or not self.has_tapered_sidewalls():
            return poly

        offset = self._taper_buffer_at_z(z)
        if abs(offset) <= 1e-18:
            return poly
        buffered = poly.buffer(offset, join_style=2)
        if buffered.is_empty:
            return None
        if not buffered.is_valid:
            buffered = buffered.buffer(0)
        if buffered.is_empty or not buffered.is_valid:
            return None
        return buffered

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

        if z is not None and self.has_tapered_sidewalls():
            from shapely import contains_xy

            poly = self.shapely_polygon_at_z(z)
            if poly is None:
                return False
            return bool(contains_xy(poly.buffer(1e-15), [float(x)], [float(y)])[0])

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
    """Axis-aligned rectangular geometry.

    Args:
        position: Lower-left coordinate as ``(x, y)`` or ``(x, y, z)``.
        width: Rectangle width along x.
        height: Rectangle height along y.
        depth: Extrusion depth along z.
        material: Material assigned to the rectangle during meshing.
        color: Display color used by plotting helpers.
        is_pml: Whether the rectangle represents a PML helper region.
        optimize: Whether this rectangle participates in optimization workflows.
        z: Optional z coordinate overriding ``position``.
        sidewall_angle: Sidewall taper angle in degrees.
        width_to_z: Fractional depth used as the sidewall width reference.
    """

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
        sidewall_angle=0.0,
        width_to_z=0.0,
    ):
        # Validate dimensions
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")

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
            sidewall_angle=sidewall_angle,
            width_to_z=width_to_z,
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
            position=self.position,
            width=self.width,
            height=self.height,
            depth=self.depth,
            material=self.material,
            color=self.color,
            is_pml=self.is_pml,
            optimize=self.optimize,
            sidewall_angle=self.sidewall_angle,
            width_to_z=self.width_to_z,
        )
        new_rect.vertices = [(x, y, z) for x, y, z in self.vertices]
        return new_rect


class Box:
    """Axis-aligned box specified by center and size.

    This is a centered-geometry companion to :class:`Rectangle`. It is intended
    for simulation construction APIs where coordinates are expressed relative to
    the simulation center.

    Args:
        center: Box center as a 2D or 3D coordinate tuple.
        size: Full box size along each axis.
        material: Material assigned when the box is converted to a rectangle.
    """

    def __init__(self, center=(0, 0, 0), size=(1, 1, 1), material=None):
        if len(center) == 2:
            center = (center[0], center[1], 0.0)
        if len(size) == 2:
            size = (size[0], size[1], 0.0)
        if len(center) != 3 or len(size) != 3:
            raise ValueError("Box center and size must be 2D or 3D coordinate tuples.")
        self.center = tuple(float(v) for v in center)
        self.size = tuple(float(v) for v in size)
        finite_sizes = [abs(v) for v in self.size if np.isfinite(v)]
        if any(v < 0 for v in finite_sizes):
            raise ValueError(f"Box sizes must be non-negative, got {size!r}.")
        self.material = material
        self.position = self.lower
        self.width = self.size[0]
        self.height = self.size[1]
        self.depth = self.size[2]
        self.is_3d = bool(self.size[2] != 0)

    @property
    def lower(self):
        return tuple(c - 0.5 * s for c, s in zip(self.center, self.size, strict=True))

    @property
    def upper(self):
        return tuple(c + 0.5 * s for c, s in zip(self.center, self.size, strict=True))

    def shifted(self, offset):
        return Box(
            center=tuple(c + d for c, d in zip(self.center, offset, strict=True)),
            size=self.size,
            material=self.material,
        )

    def to_rectangle(self, offset=(0.0, 0.0, 0.0), material=None):
        shifted = self.shifted(offset)
        lower = shifted.lower
        size = shifted.size
        if not all(np.isfinite(v) for v in (*lower, *size)):
            raise ValueError(
                "Infinite Box sizes must be clipped by the Simulation(domain=...) "
                "constructor before rasterization."
            )
        return Rectangle(
            position=lower,
            width=max(float(size[0]), np.finfo(float).eps),
            height=max(float(size[1]), np.finfo(float).eps),
            depth=max(float(size[2]), 0.0),
            material=material if material is not None else self.material,
        )

    def get_bounding_box(self):
        return (*self.lower, *self.upper)

    def point_in_polygon(self, x, y, z=None):
        lower = self.lower
        upper = self.upper
        inside_xy = lower[0] <= x <= upper[0] and lower[1] <= y <= upper[1]
        if z is None or self.size[2] == 0:
            return inside_xy
        return inside_xy and lower[2] <= z <= upper[2]

    def copy(self):
        return Box(center=self.center, size=self.size, material=self.material)


class Structure:
    """Deprecated compatibility wrapper pairing geometry with a material."""

    def __init__(self, geometry, medium=None, material=None, _warn=True):
        if _warn:
            warnings.warn(
                "Structure is deprecated; attach material=... to the geometry and add "
                "it to a Design instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if medium is not None:
                warnings.warn(
                    "Structure(..., medium=...) is deprecated; use material=... instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        self.geometry = geometry
        self.material = material if material is not None else medium
        if self.material is not None and hasattr(self.geometry, "material"):
            self.geometry.material = self.material
        self.medium = self.material

    def __getattr__(self, name):
        return getattr(self.geometry, name)

    def get_bounding_box(self):
        return self.geometry.get_bounding_box()

    def point_in_polygon(self, x, y, z=None):
        return self.geometry.point_in_polygon(x, y, z)

    def to_beamz_structure(self, offset=(0.0, 0.0, 0.0), domain_size=None):
        geometry = self.geometry
        if isinstance(geometry, Box):
            if domain_size is not None and any(
                not np.isfinite(v) for v in geometry.size
            ):
                clipped_size = tuple(
                    float(domain)
                    if not np.isfinite(size)
                    else min(float(size), float(domain))
                    for size, domain in zip(geometry.size, domain_size, strict=True)
                )
                geometry = Box(center=geometry.center, size=clipped_size)
            return geometry.to_rectangle(offset=offset, material=self.material)

        copied = geometry.copy() if hasattr(geometry, "copy") else geometry
        if hasattr(copied, "shift"):
            copied = copied.shift(*offset)
        if self.material is not None:
            copied.material = self.material
        return copied

    def copy(self):
        geometry = (
            self.geometry.copy() if hasattr(self.geometry, "copy") else self.geometry
        )
        return Structure(geometry=geometry, material=self.material, _warn=False)


class Circle(Polygon):
    """Circular planar geometry approximated by polygon vertices.

    Args:
        position: Circle center as ``(x, y)`` or ``(x, y, z)``.
        radius: Circle radius.
        points: Number of polygon vertices used for the approximation.
        material: Material assigned to the circle during meshing.
        color: Display color used by plotting helpers.
        optimize: Whether this circle participates in optimization workflows.
        depth: Extrusion depth along z.
        z: Lower z coordinate for the extruded circle.
    """

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
        # Validate dimensions
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")

        position = _normalize_position(position)
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
    """Annular planar geometry with inner and outer radii.

    Args:
        position: Ring center as ``(x, y)`` or ``(x, y, z)``.
        inner_radius: Inner radius of the annulus.
        outer_radius: Outer radius of the annulus.
        material: Material assigned to the ring during meshing.
        color: Display color used by plotting helpers.
        optimize: Whether this ring participates in optimization workflows.
        points: Number of polygon vertices used per circular boundary.
        depth: Extrusion depth along z.
        z: Optional z coordinate overriding ``position``.
    """

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
        # Validate dimensions
        if inner_radius <= 0:
            raise ValueError(f"inner_radius must be positive, got {inner_radius}")
        if outer_radius <= inner_radius:
            raise ValueError(
                f"outer_radius ({outer_radius}) must be greater than inner_radius ({inner_radius})"
            )
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")

        position = _normalize_position(position, z)
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

    def to_plot_data(
        self, *, facecolor=None, edgecolor="black", alpha=None, linestyle=None
    ):
        if facecolor is None:
            facecolor = self.color
        if alpha is None:
            alpha = 1
        if linestyle is None:
            linestyle = "-"
        return super().to_plot_data(
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
    """Curved annular waveguide bend geometry.

    Args:
        position: Bend center as ``(x, y)`` or ``(x, y, z)``.
        inner_radius: Inner radius of the bend.
        outer_radius: Outer radius of the bend.
        angle: Bend sweep angle in degrees.
        rotation: Rotation angle in degrees.
        material: Material assigned to the bend during meshing.
        facecolor: Display fill color used by plotting helpers.
        optimize: Whether this bend participates in optimization workflows.
        points: Number of polygon vertices used along each arc.
        depth: Extrusion depth along z.
        z: Lower z coordinate for the extruded bend.
    """

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
        position = _normalize_position(position)
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

    def to_plot_data(
        self, *, facecolor=None, edgecolor="black", alpha=None, linestyle=None
    ):
        if facecolor is None:
            facecolor = self.color
        if alpha is None:
            alpha = 1
        if linestyle is None:
            linestyle = "-"
        return super().to_plot_data(
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
    """Linear taper between an input and output width.

    Args:
        position: Input-center coordinate as ``(x, y)`` or ``(x, y, z)``.
        input_width: Width at the taper input.
        output_width: Width at the taper output.
        length: Taper length along x before rotation.
        material: Material assigned to the taper during meshing.
        color: Display color used by plotting helpers.
        optimize: Whether this taper participates in optimization workflows.
        depth: Extrusion depth along z.
        z: Lower z coordinate for the extruded taper.
    """

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
        position = _normalize_position(position)
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
        # Validate dimensions
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")

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
