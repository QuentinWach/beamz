import random

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from beamz.const import µm
from beamz.design.materials import Material
from beamz.design.structures import (
    Circle,
    CircularBend,
    Polygon,
    Rectangle,
    Ring,
    Sphere,
    Taper,
)


class Design:
    def __init__(
        self,
        width: float = 4 * µm,
        height: float = 4 * µm,
        depth: float = 0,
        material: Material = None,
    ):
        """Create a design domain with specified dimensions and background material."""
        if material is None:
            material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        background = Rectangle(
            position=(0, 0, 0),
            width=width,
            height=height,
            depth=depth,
            material=material,
        )
        self.structures, self.sources, self.monitors = [background], [], []
        self.width, self.height, self.depth, self.time = width, height, depth, 0
        self.is_3d = depth is not None and depth > 0
        self.layers: dict[int, list[Polygon]] = {}

    def __str__(self):
        return f"Design with {len(self.structures)} structures ({'3D' if self.is_3d else '2D'})"

    def __iadd__(self, structure):
        """Implement += operator for adding structures."""
        self.add(structure)
        return self

    def unify_polygons(self):
        """Merge overlapping polygons with the same material properties into unified shapes."""
        material_groups, non_polygon_structures, structures_to_remove = {}, [], []

        for structure in self.structures:
            material = getattr(structure, "material", None)
            if not material:
                non_polygon_structures.append(structure)
                continue

            material_key = (
                getattr(material, "permittivity", None),
                getattr(material, "permeability", None),
                getattr(material, "conductivity", None),
            )

            if material_key not in material_groups:
                material_groups[material_key] = []

            if hasattr(structure, "interiors") and structure.interiors:
                valid_interiors = [
                    list(i_path) for i_path in structure.interiors if i_path
                ]
                if structure.vertices and valid_interiors:
                    shapely_polygon = ShapelyPolygon(
                        shell=structure.vertices, holes=valid_interiors
                    )
                elif structure.vertices:
                    shapely_polygon = ShapelyPolygon(shell=structure.vertices)
                else:
                    non_polygon_structures.append(structure)
                    continue
            elif hasattr(structure, "vertices") and structure.vertices:
                shapely_polygon = ShapelyPolygon(shell=structure.vertices)
            else:
                non_polygon_structures.append(structure)
                continue

            if shapely_polygon.is_valid:
                material_groups[material_key].append((structure, shapely_polygon))
                structures_to_remove.append(structure)
            else:
                non_polygon_structures.append(structure)

        rings_to_preserve = []
        for material_key, structure_group in material_groups.items():
            if len(structure_group) <= 1:
                continue
            rings_in_group = [
                (idx, s)
                for idx, s in enumerate(structure_group)
                if isinstance(s[0], Ring)
            ]
            if not rings_in_group:
                continue
            for ring_idx, (ring, ring_shapely) in rings_in_group:
                rings_to_preserve.append(ring)
                if ring in structures_to_remove:
                    structures_to_remove.remove(ring)

        new_structures = []
        for material_key, structure_group in material_groups.items():
            filtered_group = [
                s for s in structure_group if s[0] not in rings_to_preserve
            ]
            if len(filtered_group) <= 1:
                new_structures.extend([s[0] for s in filtered_group])
                for s in filtered_group:
                    if s[0] in structures_to_remove:
                        structures_to_remove.remove(s[0])
                continue

            shapely_polygons = [p[1] for p in filtered_group]
            material = filtered_group[0][0].material
            merged = unary_union(shapely_polygons)

            if merged.geom_type == "Polygon":
                exterior_coords = list(merged.exterior.coords[:-1])
                interior_coords_lists = [
                    list(interior.coords[:-1]) for interior in merged.interiors
                ]
                if exterior_coords and len(exterior_coords) >= 3:
                    first_structure = filtered_group[0][0]
                    new_poly = Polygon(
                        vertices=exterior_coords,
                        interiors=interior_coords_lists,
                        material=material,
                        depth=getattr(first_structure, "depth", 0),
                        z=getattr(first_structure, "z", 0),
                    )
                    new_structures.append(new_poly)
                else:
                    new_structures.extend([s[0] for s in structure_group])
                    for s_tuple in structure_group:
                        if s_tuple[0] in structures_to_remove:
                            structures_to_remove.remove(s_tuple[0])
            elif merged.geom_type == "MultiPolygon":
                all_valid = True
                temp_new_polys = []
                for geom in merged.geoms:
                    exterior_coords = list(geom.exterior.coords[:-1])
                    interior_coords_lists = [
                        list(interior.coords[:-1]) for interior in geom.interiors
                    ]
                    if exterior_coords and len(exterior_coords) >= 3:
                        first_structure = filtered_group[0][0]
                        new_poly = Polygon(
                            vertices=exterior_coords,
                            interiors=interior_coords_lists,
                            material=material,
                            depth=getattr(first_structure, "depth", 0),
                            z=getattr(first_structure, "z", 0),
                        )
                        temp_new_polys.append(new_poly)
                    else:
                        all_valid = False
                        break

                if all_valid:
                    new_structures.extend(temp_new_polys)
                else:
                    new_structures.extend([s[0] for s in structure_group])
                    for s_tuple in structure_group:
                        if s_tuple[0] in structures_to_remove:
                            structures_to_remove.remove(s_tuple[0])
            else:
                new_structures.extend([s[0] for s in structure_group])
                for s in structure_group:
                    if s[0] in structures_to_remove:
                        structures_to_remove.remove(s[0])

        material_replacements = {}
        for new_struct in new_structures:
            if hasattr(new_struct, "material") and new_struct.material:
                new_material_key = (
                    getattr(new_struct.material, "permittivity", None),
                    getattr(new_struct.material, "permeability", None),
                    getattr(new_struct.material, "conductivity", None),
                )
                for material_key, structure_group in material_groups.items():
                    if len(structure_group) > 1 and material_key == new_material_key:
                        if material_key not in material_replacements:
                            material_replacements[material_key] = []
                        material_replacements[material_key].append(new_struct)
                        break

        rebuilt_structures, material_groups_used = [], set()
        for structure in self.structures:
            if structure in structures_to_remove:
                structure_material_key = None
                if hasattr(structure, "material") and structure.material:
                    structure_material_key = (
                        getattr(structure.material, "permittivity", None),
                        getattr(structure.material, "permeability", None),
                        getattr(structure.material, "conductivity", None),
                    )
                if (
                    structure_material_key
                    and structure_material_key not in material_groups_used
                ):
                    if structure_material_key in material_replacements:
                        rebuilt_structures.extend(
                            material_replacements[structure_material_key]
                        )
                        material_groups_used.add(structure_material_key)
            else:
                rebuilt_structures.append(structure)

        self.structures = rebuilt_structures
        return True

    def add(self, structure: type[Polygon]):
        """Add structure to the design and update 3D flag if needed."""
        from beamz.devices.monitors import Monitor
        from beamz.devices.sources import GaussianSource, ModeSource

        # Set back-reference to design if the structure supports it
        if hasattr(structure, "design"):
            structure.design = self

        if isinstance(structure, Monitor):
            self.monitors.append(structure)
        elif isinstance(structure, (ModeSource, GaussianSource)):
            self.sources.append(structure)
        else:
            self.structures.append(structure)

        if hasattr(structure, "is_3d") and structure.is_3d:
            self.is_3d = True
        if hasattr(structure, "depth") and structure.depth != 0:
            self.is_3d = True
        if (
            hasattr(structure, "position")
            and len(structure.position) > 2
            and structure.position[2] != 0
        ):
            self.is_3d = True
        if hasattr(structure, "vertices") and structure.vertices:
            for vertex in structure.vertices:
                if len(vertex) > 2 and vertex[2] != 0:
                    self.is_3d = True
                    break

    def scatter(
        self,
        structure: type[Polygon],
        n: int = 1000,
        xyrange: tuple[float, float] = (-5 * µm, 5 * µm),
        scale_range: tuple[float, float] = (0.05, 1),
    ):
        """Randomly distribute n copies of the structure across the design domain."""
        for _ in range(n):
            new_structure = structure.copy()
            new_structure.shift(
                random.uniform(xyrange[0], xyrange[1]),
                random.uniform(xyrange[0], xyrange[1]),
            )
            new_structure.rotate(random.uniform(0, 360))
            new_structure.scale(random.uniform(scale_range[0], scale_range[1]))
            self.add(new_structure)

    def get_material_value(self, x: float, y: float, z: float = 0.0):
        """Return material properties at coordinate (x,y,z) prioritizing topmost structure."""
        epsilon, mu, sigma_base = 1.0, 1.0, 0.0

        # TODO: Check if we can reduce this by simply using the Polygon
        for structure in reversed(self.structures):
            if isinstance(structure, Polygon):
                if structure.point_in_polygon(x, y, z):
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break

            elif isinstance(structure, Rectangle):
                if hasattr(structure, "is_pml") and structure.is_pml:
                    continue
                if structure.point_in_polygon(x, y, z):
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, Circle):
                if (
                    np.hypot(x - structure.position[0], y - structure.position[1])
                    <= structure.radius
                ):
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, Ring):
                distance = np.hypot(
                    x - structure.position[0], y - structure.position[1]
                )
                if structure.inner_radius <= distance <= structure.outer_radius:
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, CircularBend):
                distance = np.hypot(
                    x - structure.position[0], y - structure.position[1]
                )
                if structure.inner_radius <= distance <= structure.outer_radius:
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break

        return [epsilon, mu, sigma_base]

    def rasterize(
        self,
        resolution: float,
        grid_type: str = "auto",
        force_recompute: bool = False,
        **kwargs,
    ):
        """Rasterize the design into a mesh grid at specified resolution and cache it internally."""
        from beamz.design.meshing import RegularGrid, RegularGrid3D, create_mesh

        # Return cached grid if resolution matches and no force recompute
        if (
            not force_recompute
            and hasattr(self, "_grid")
            and hasattr(self, "_grid_resolution")
        ):
            if self._grid_resolution == resolution:
                return self._grid

        if isinstance(grid_type, str):
            gt = grid_type.lower()
            if gt in {"regular", "regulargrid", "2d"}:
                grid_cls = RegularGrid
            elif gt in {"regular3d", "3d"}:
                grid_cls = RegularGrid3D
            elif gt in {"auto", "auto-select", "autoselect"}:
                self._grid = create_mesh(self, resolution, **kwargs)
                self._grid_resolution = resolution
                return self._grid
            else:
                return None
        elif isinstance(grid_type, type):
            grid_cls = grid_type

        # If we got here with grid_cls, use it
        if grid_cls is RegularGrid3D:
            resolution_xy, resolution_z = resolution, kwargs.pop("resolution_z", None)
            self._grid = grid_cls(
                self, resolution_xy=resolution_xy, resolution_z=resolution_z
            )
            self._grid_resolution = resolution
        else:
            self._grid = grid_cls(self, resolution, **kwargs)
            self._grid_resolution = resolution

        return self._grid

    def get_material_grids(self, resolution):
        """Get cached rasterized material property arrays at specified resolution as references."""
        if (
            not hasattr(self, "_grid")
            or not hasattr(self, "_grid_resolution")
            or self._grid_resolution != resolution
        ):
            self.rasterize(resolution, grid_type="auto")
        return (
            self._grid.permittivity,
            self._grid.conductivity,
            self._grid.permeability,
        )

    def copy(self):
        """Create a deep copy of the design with all structures and properties."""
        background_material = (
            self.structures[0].material
            if self.structures and hasattr(self.structures[0], "material")
            else None
        )
        new_design = Design(
            width=self.width,
            height=self.height,
            depth=self.depth,
            material=background_material,
        )
        new_design.structures, new_design.sources, new_design.monitors = [], [], []

        # Copy structures
        for structure in self.structures:
            if hasattr(structure, "copy"):
                copied_structure = structure.copy()
                if (
                    hasattr(copied_structure, "material")
                    and copied_structure.material
                    and hasattr(copied_structure.material, "copy")
                ):
                    copied_structure.material = copied_structure.material.copy()
                if hasattr(copied_structure, "design"):
                    copied_structure.design = new_design
                new_design.structures.append(copied_structure)
            else:
                new_design.structures.append(structure)

        # Copy sources
        for source in self.sources:
            if hasattr(source, "copy"):
                copied_source = source.copy()
                if hasattr(copied_source, "design"):
                    copied_source.design = new_design
                new_design.sources.append(copied_source)
            else:
                new_design.sources.append(source)

        # Copy monitors
        for monitor in self.monitors:
            if hasattr(monitor, "copy"):
                copied_monitor = monitor.copy()
                if hasattr(copied_monitor, "design"):
                    copied_monitor.design = new_design
                new_design.monitors.append(copied_monitor)
            else:
                new_design.monitors.append(monitor)

        new_design.is_3d, new_design.depth, new_design.time = (
            self.is_3d,
            self.depth,
            self.time,
        )
        new_design.layers = self.layers.copy() if hasattr(self, "layers") else {}

        return new_design

    def show(self, **kwargs):
        """Display the design using the visualization module."""
        from beamz.visual.viz import show_design

        show_design(self, **kwargs)
