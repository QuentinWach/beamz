import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from beamz.const import µm
from beamz.design.materials import Material
from beamz.design.structures import (
    Polygon,
    Rectangle,
    Ring,
)


def _material_key(material):
    """Return a hashable key for a material based on its physical properties."""
    return (
        getattr(material, "permittivity", None),
        getattr(material, "permeability", None),
        getattr(material, "conductivity", None),
    )


def _to_shapely(structure):
    """Convert a beamz structure to a Shapely polygon, or None if not possible."""
    if hasattr(structure, "interiors") and structure.interiors:
        valid_interiors = [list(i_path) for i_path in structure.interiors if i_path]
        if structure.vertices and valid_interiors:
            poly = ShapelyPolygon(shell=structure.vertices, holes=valid_interiors)
        elif structure.vertices:
            poly = ShapelyPolygon(shell=structure.vertices)
        else:
            return None
    elif hasattr(structure, "vertices") and structure.vertices:
        poly = ShapelyPolygon(shell=structure.vertices)
    else:
        return None
    return poly if poly.is_valid else None


def _group_by_material(structures):
    """Group structures by material key and convert to Shapely polygons.

    Returns (material_groups, structures_to_remove) where material_groups maps
    material_key -> list of (structure, shapely_polygon) tuples.
    """
    material_groups = {}
    structures_to_remove = []
    for structure in structures:
        material = getattr(structure, "material", None)
        if not material:
            continue
        key = _material_key(material)
        shapely_poly = _to_shapely(structure)
        if shapely_poly is None:
            continue
        material_groups.setdefault(key, []).append((structure, shapely_poly))
        structures_to_remove.append(structure)
    return material_groups, structures_to_remove


def _find_rings_to_preserve(material_groups, structures_to_remove):
    """Identify Ring structures that should not be merged (they have interiors)."""
    rings_to_preserve = []
    for structure_group in material_groups.values():
        if len(structure_group) <= 1:
            continue
        for _idx, (struct, _shapely) in enumerate(structure_group):
            if isinstance(struct, Ring):
                rings_to_preserve.append(struct)
                if struct in structures_to_remove:
                    structures_to_remove.remove(struct)
    return rings_to_preserve


def _shapely_to_polygons(merged, material, first_structure):
    """Convert a merged Shapely geometry back to beamz Polygon(s).

    Returns a list of Polygon objects, or None if conversion fails.
    """
    depth = getattr(first_structure, "depth", 0)
    z = getattr(first_structure, "z", 0)

    def _geom_to_polygon(geom):
        exterior_coords = list(geom.exterior.coords[:-1])
        if not exterior_coords or len(exterior_coords) < 3:
            return None
        interior_coords_lists = [
            list(interior.coords[:-1]) for interior in geom.interiors
        ]
        return Polygon(
            vertices=exterior_coords,
            interiors=interior_coords_lists,
            material=material,
            depth=depth,
            z=z,
        )

    if merged.geom_type == "Polygon":
        poly = _geom_to_polygon(merged)
        return [poly] if poly else None
    elif merged.geom_type == "MultiPolygon":
        polys = []
        for geom in merged.geoms:
            poly = _geom_to_polygon(geom)
            if poly is None:
                return None
            polys.append(poly)
        return polys
    return None


def _merge_groups(material_groups, rings_to_preserve, structures_to_remove):
    """Merge each material group using Shapely union.

    Returns (new_structures, updated structures_to_remove).
    """
    new_structures = []
    for material_key, structure_group in material_groups.items():
        filtered_group = [s for s in structure_group if s[0] not in rings_to_preserve]
        if len(filtered_group) <= 1:
            new_structures.extend([s[0] for s in filtered_group])
            for s in filtered_group:
                if s[0] in structures_to_remove:
                    structures_to_remove.remove(s[0])
            continue

        shapely_polygons = [p[1] for p in filtered_group]
        material = filtered_group[0][0].material
        merged = unary_union(shapely_polygons)
        result = _shapely_to_polygons(merged, material, filtered_group[0][0])

        if result is not None:
            new_structures.extend(result)
        else:
            # Fallback: keep originals
            new_structures.extend([s[0] for s in structure_group])
            for s_tuple in structure_group:
                if s_tuple[0] in structures_to_remove:
                    structures_to_remove.remove(s_tuple[0])

    return new_structures, structures_to_remove


def _rebuild_structure_list(
    original, structures_to_remove, new_structures, material_groups
):
    """Rebuild the structure list, replacing merged groups at their original position."""
    material_replacements = {}
    for new_struct in new_structures:
        if not (hasattr(new_struct, "material") and new_struct.material):
            continue
        key = _material_key(new_struct.material)
        for mat_key, group in material_groups.items():
            if len(group) > 1 and mat_key == key:
                material_replacements.setdefault(mat_key, []).append(new_struct)
                break

    rebuilt, used = [], set()
    for structure in original:
        if structure in structures_to_remove:
            if not (hasattr(structure, "material") and structure.material):
                continue
            key = _material_key(structure.material)
            if key not in used and key in material_replacements:
                rebuilt.extend(material_replacements[key])
                used.add(key)
        else:
            rebuilt.append(structure)
    return rebuilt


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
        material_groups, structures_to_remove = _group_by_material(self.structures)
        rings_to_preserve = _find_rings_to_preserve(
            material_groups, structures_to_remove
        )
        new_structures, structures_to_remove = _merge_groups(
            material_groups, rings_to_preserve, structures_to_remove
        )
        self.structures = _rebuild_structure_list(
            self.structures, structures_to_remove, new_structures, material_groups
        )
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

    def get_material_value(self, x: float, y: float, z: float = 0.0):
        """Return material properties at coordinate (x,y,z) prioritizing topmost structure."""
        epsilon, mu, sigma_base = 1.0, 1.0, 0.0

        for structure in reversed(self.structures):
            if getattr(structure, "is_pml", False):
                continue
            if structure.point_in_polygon(x, y, z):
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

    def get_thermal_grids(self, resolution):
        """Get cached rasterized thermal property arrays at specified resolution as references."""
        if (
            not hasattr(self, "_grid")
            or not hasattr(self, "_grid_resolution")
            or self._grid_resolution != resolution
        ):
            self.rasterize(resolution, grid_type="auto")
        if hasattr(self._grid, "get_thermal_grids"):
            return self._grid.get_thermal_grids()
        return None

    def solve_thermal(
        self,
        resolution,
        scenario,
        config=None,
    ):
        """Solve steady-state thermal profile for this design and return thermo-optic result."""
        from beamz.simulation.thermal import solve_static_thermal

        return solve_static_thermal(
            design=self,
            resolution=resolution,
            scenario=scenario,
            config=config,
        )

    def solve_static_thermal(
        self,
        resolution,
        scenario,
        config=None,
    ):
        """Alias for solve_thermal to preserve method surface."""
        return self.solve_thermal(
            resolution=resolution,
            scenario=scenario,
            config=config,
        )

    def sweep_mzi_heater(
        self,
        resolution,
        powers_w,
        heater,
        optical_region,
        arm_length_m,
        wavelength_m,
        group_index,
        scenario_base,
        mode_weight=None,
        config=None,
    ):
        """Run a static thermal heater sweep and return MZI tuning metrics."""
        from beamz.simulation.thermal import sweep_mzi_heater

        return sweep_mzi_heater(
            design=self,
            resolution=resolution,
            powers_w=powers_w,
            heater=heater,
            optical_region=optical_region,
            arm_length_m=arm_length_m,
            wavelength_m=wavelength_m,
            group_index=group_index,
            scenario_base=scenario_base,
            mode_weight=mode_weight,
            config=config,
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
        from beamz.visual.design_viz import show_design

        show_design(self, **kwargs)
