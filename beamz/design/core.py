import hashlib
import json
import os
import time
from pathlib import Path

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

_DEFAULT_DOMAIN_SIZE = object()


def _material_key(material):
    """Return a hashable key for a material based on its physical properties."""
    return (
        getattr(material, "permittivity", None),
        getattr(material, "permeability", None),
        getattr(material, "conductivity", None),
    )


def _merge_group_key(structure):
    """Return a hashable key for planar-union compatibility.

    Structures can only be merged safely when both their material properties and
    their z/depth placement match; the shapely union ignores z entirely.
    """
    return (
        _material_key(getattr(structure, "material", None)),
        getattr(structure, "z", None),
        getattr(structure, "depth", None),
        getattr(structure, "sidewall_angle", None),
        getattr(structure, "width_to_z", None),
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
        key = _merge_group_key(structure)
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
    sidewall_angle = getattr(first_structure, "sidewall_angle", 0.0)
    width_to_z = getattr(first_structure, "width_to_z", 0.0)

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
            sidewall_angle=sidewall_angle,
            width_to_z=width_to_z,
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
        key = _merge_group_key(new_struct)
        for mat_key, group in material_groups.items():
            if len(group) > 1 and mat_key == key:
                material_replacements.setdefault(mat_key, []).append(new_struct)
                break

    rebuilt, used = [], set()
    for structure in original:
        if structure in structures_to_remove:
            if not (hasattr(structure, "material") and structure.material):
                continue
            key = _merge_group_key(structure)
            if key not in used and key in material_replacements:
                rebuilt.extend(material_replacements[key])
                used.add(key)
        else:
            rebuilt.append(structure)
    return rebuilt


RASTER_CACHE_VERSION = "v6"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_jsonable(value):
    """Convert arbitrary values into deterministic JSON-safe primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(f"{value:.16g}")
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size <= 1024:
            return [_to_jsonable(v) for v in arr.reshape(-1).tolist()]
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        return {
            "__ndarray__": {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": digest,
            }
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items())}
    return repr(value)


def _material_signature(material):
    if material is None:
        return None
    keys = (
        "permittivity",
        "permeability",
        "conductivity",
    )
    return {k: _to_jsonable(getattr(material, k, None)) for k in keys}


def _structure_signature(structure):
    sig = {
        "type": type(structure).__name__,
        "bbox": (
            _to_jsonable(structure.get_bounding_box())
            if hasattr(structure, "get_bounding_box")
            else None
        ),
        "material": _material_signature(getattr(structure, "material", None)),
    }
    keys = (
        "position",
        "width",
        "height",
        "depth",
        "z",
        "sidewall_angle",
        "width_to_z",
        "radius",
        "inner_radius",
        "outer_radius",
        "is_pml",
        "optimize",
    )
    for key in keys:
        if hasattr(structure, key):
            sig[key] = _to_jsonable(getattr(structure, key))
    if hasattr(structure, "vertices"):
        sig["vertices"] = _to_jsonable(getattr(structure, "vertices"))
    if hasattr(structure, "interiors"):
        sig["interiors"] = _to_jsonable(getattr(structure, "interiors"))
    return sig


def _boundary_signature(boundary):
    if boundary is None:
        return None
    raw = {}
    for key, val in getattr(boundary, "__dict__", {}).items():
        if key.startswith("_"):
            continue
        raw[key] = _to_jsonable(val)
    return {"type": type(boundary).__name__, "attrs": raw}


def _grid_kind_for_type(grid_type):
    if not isinstance(grid_type, type):
        return None
    for cls in getattr(grid_type, "__mro__", ()):
        name = getattr(cls, "__name__", "").lower()
        if name == "regulargrid3d":
            return "3d"
        if name == "regulargrid":
            return "2d"
    return None


def _grid_kind_for_request(design_obj, grid_type, kwargs):
    explicit_kind = _grid_kind_for_type(grid_type)
    if explicit_kind is not None:
        return explicit_kind
    if isinstance(grid_type, str):
        gt = grid_type.lower()
        if gt in {"regular3d", "3d"}:
            return "3d"
        if gt in {"regular", "regulargrid", "2d"}:
            return "2d"
        if gt in {"auto", "auto-select", "autoselect"}:
            force_3d = bool(kwargs.get("force_3d", False))
            auto_select = bool(kwargs.get("auto_select", True))
            if force_3d or (auto_select and design_obj.is_3d and design_obj.depth > 0):
                return "3d"
            return "2d"
    return "3d" if (design_obj.is_3d and design_obj.depth > 0) else "2d"


def _normalize_aa_config(kwargs):
    mode = str(kwargs.get("aa_mode", "legacy_grid") or "legacy_grid").strip().lower()
    samples = kwargs.get("aa_samples", 64)
    seed = kwargs.get("aa_seed", 0)
    return {
        "mode": mode,
        "samples": int(64 if samples is None else samples),
        "seed": int(0 if seed is None else seed),
        "scramble": "cp_cell_v1" if mode == "stratified_jitter" else "none",
    }


def _design_cache_key(design_obj, resolution, grid_kind, resolution_z, aa_config):
    raster_env = {
        "fast_3d": os.getenv("BEAMZ_RASTER_FAST_3D"),
        "fast_min_voxels": os.getenv("BEAMZ_RASTER_FAST_MIN_VOXELS", "1000000"),
    }
    payload = {
        "version": RASTER_CACHE_VERSION,
        "grid_kind": grid_kind,
        "resolution_xy": _to_jsonable(float(resolution)),
        "resolution_z": _to_jsonable(float(resolution_z)),
        "antialiasing": _to_jsonable(aa_config),
        "raster_env": raster_env,
        "domain": {
            "width": _to_jsonable(float(design_obj.width)),
            "height": _to_jsonable(float(design_obj.height)),
            "depth": _to_jsonable(float(design_obj.depth)),
            "is_3d": bool(design_obj.is_3d),
        },
        "structures": [_structure_signature(s) for s in design_obj.structures],
        "boundaries": [
            _boundary_signature(b) for b in getattr(design_obj, "boundaries", [])
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _raster_cache_dir():
    default_dir = Path(".beamz_cache") / "raster"
    return Path(os.getenv("BEAMZ_RASTER_CACHE_DIR", str(default_dir)))


def _raster_cache_path(cache_key):
    return _raster_cache_dir() / f"{cache_key}.npz"


def _save_grid_to_cache(grid, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        permittivity=np.asarray(grid.permittivity),
        permeability=np.asarray(grid.permeability),
        conductivity=np.asarray(grid.conductivity),
    )


def _build_grid_from_cached_arrays(
    *,
    design_obj,
    resolution,
    resolution_z,
    grid_kind,
    arrays,
):
    from beamz.design.meshing import RegularGrid, RegularGrid3D

    if grid_kind == "3d":
        grid = object.__new__(RegularGrid3D)
        grid.design = design_obj
        grid.resolution = resolution
        grid.resolution_xy = resolution
        grid.resolution_z = resolution_z
        grid.is_3d = True
        grid.dx = resolution
        grid.dy = resolution
        grid.dz = resolution_z
        grid.width = design_obj.width
        grid.height = design_obj.height
        grid.depth = design_obj.depth
    else:
        grid = object.__new__(RegularGrid)
        grid.design = design_obj
        grid.resolution = resolution
        grid.is_3d = False
        grid.dx = resolution
        grid.dy = resolution
        grid.width = design_obj.width
        grid.height = design_obj.height

    for name in (
        "permittivity",
        "permeability",
        "conductivity",
    ):
        setattr(grid, name, np.asarray(arrays[name]))
    grid.shape = grid.permittivity.shape
    return grid


class Design:
    def __init__(
        self,
        width: float | object = _DEFAULT_DOMAIN_SIZE,
        height: float | object = _DEFAULT_DOMAIN_SIZE,
        depth: float | object = _DEFAULT_DOMAIN_SIZE,
        material: Material = None,
        background: Material = None,
    ):
        """Create a design domain with specified dimensions and background material."""
        width_is_default = width is _DEFAULT_DOMAIN_SIZE
        height_is_default = height is _DEFAULT_DOMAIN_SIZE
        depth_is_default = depth is _DEFAULT_DOMAIN_SIZE
        background_arg = background
        if width_is_default:
            width = 4 * µm
        if height_is_default:
            height = 4 * µm
        if depth_is_default:
            depth = 0
        explicit_size = not (
            width_is_default and height_is_default and depth_is_default
        )
        if (
            background_arg is not None
            and material is not None
            and background_arg is not material
        ):
            raise ValueError("Pass only one of background=... or material=....")
        if background_arg is not None:
            material = background_arg
        if material is None:
            material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        background = Rectangle(
            position=(0, 0, 0),
            width=width,
            height=height,
            depth=depth,
            material=material,
        )
        self.structures = [background]
        self.background = material
        self.width, self.height, self.depth, self.time = width, height, depth, 0
        self.is_3d = depth is not None and depth > 0
        self._explicit_size = explicit_size
        self._centered_coordinates = bool(
            background_arg is not None and not explicit_size
        )
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
        """Add a geometric structure to the design and update 3D state."""
        from beamz.devices.monitors import Monitor
        from beamz.devices.sources import GaussianSource, ModeSource

        if isinstance(structure, Monitor):
            raise TypeError(
                "Design only accepts geometry. Pass monitors to "
                "Simulation(monitors=[...]) instead."
            )
        if isinstance(structure, (ModeSource, GaussianSource)):
            raise TypeError(
                "Design only accepts geometry. Pass sources to "
                "Simulation(sources=[...]) instead."
            )

        # Set back-reference to design if the structure supports it
        if hasattr(structure, "design"):
            structure.design = self

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
        """Rasterize design into a mesh grid with in-memory and optional disk cache."""
        from beamz.design.meshing import RegularGrid, RegularGrid3D, create_mesh
        from beamz.visual.helpers import display_status

        timing_enabled = _env_bool("BEAMZ_RASTER_TIMING", True)
        disk_cache_enabled = _env_bool("BEAMZ_RASTER_CACHE", True)
        t_total_start = time.perf_counter()
        grid_kind = _grid_kind_for_request(self, grid_type, kwargs)
        requested_resolution_z_raw = kwargs.get("resolution_z", resolution)
        if requested_resolution_z_raw is None:
            requested_resolution_z_raw = resolution
        requested_resolution_z = float(requested_resolution_z_raw)
        aa_config = _normalize_aa_config(kwargs)
        request_signature = {
            "resolution_xy": float(resolution),
            "resolution_z": requested_resolution_z,
            "grid_kind": grid_kind,
            "aa_config": aa_config,
        }

        # Return cached grid if request signature matches and no force recompute
        if not force_recompute and hasattr(self, "_grid"):
            cached_sig = getattr(self, "_grid_request_signature", None)
            if cached_sig is not None and cached_sig == request_signature:
                return self._grid

        cache_path = None
        if disk_cache_enabled and not force_recompute:
            cache_key = _design_cache_key(
                self,
                resolution=float(resolution),
                grid_kind=grid_kind,
                resolution_z=requested_resolution_z,
                aa_config=aa_config,
            )
            cache_path = _raster_cache_path(cache_key)
            if cache_path.exists():
                t_load = time.perf_counter()
                arrays = np.load(cache_path)
                try:
                    self._grid = _build_grid_from_cached_arrays(
                        design_obj=self,
                        resolution=float(resolution),
                        resolution_z=requested_resolution_z,
                        grid_kind=grid_kind,
                        arrays=arrays,
                    )
                    self._grid_resolution = resolution
                    self._grid_request_signature = request_signature
                finally:
                    arrays.close()
                if timing_enabled:
                    display_status(
                        (
                            f"Raster cache hit ({grid_kind}): {cache_path.name} | "
                            f"load={time.perf_counter() - t_load:.2f}s"
                        ),
                        "success",
                    )
                return self._grid

        t_raster_start = time.perf_counter()
        if isinstance(grid_type, str):
            gt = grid_type.lower()
            if gt in {"regular", "regulargrid", "2d"}:
                grid_cls = RegularGrid
            elif gt in {"regular3d", "3d"}:
                grid_cls = RegularGrid3D
            elif gt in {"auto", "auto-select", "autoselect"}:
                self._grid = create_mesh(self, resolution, **kwargs)
                self._grid_resolution = resolution
                self._grid_request_signature = request_signature
                t_raster_end = time.perf_counter()
                if disk_cache_enabled:
                    if cache_path is None:
                        cache_key = _design_cache_key(
                            self,
                            resolution=float(resolution),
                            grid_kind=(
                                "3d"
                                if (hasattr(self._grid, "is_3d") and self._grid.is_3d)
                                else "2d"
                            ),
                            resolution_z=float(
                                getattr(self._grid, "resolution_z", resolution)
                            ),
                            aa_config=aa_config,
                        )
                        cache_path = _raster_cache_path(cache_key)
                    t_save = time.perf_counter()
                    _save_grid_to_cache(self._grid, cache_path)
                    if timing_enabled:
                        display_status(
                            (
                                f"Raster cache saved: {cache_path.name} | "
                                f"save={time.perf_counter() - t_save:.2f}s"
                            ),
                            "info",
                        )
                if timing_enabled:
                    display_status(
                        (
                            f"Rasterize wall-time: {t_raster_end - t_raster_start:.2f}s | "
                            f"total={time.perf_counter() - t_total_start:.2f}s"
                        ),
                        "info",
                    )
                return self._grid
            else:
                return None
        elif isinstance(grid_type, type):
            grid_cls = grid_type

        # If we got here with grid_cls, use it
        if grid_cls is RegularGrid3D:
            resolution_xy, resolution_z = resolution, kwargs.pop("resolution_z", None)
            self._grid = grid_cls(
                self,
                resolution_xy=resolution_xy,
                resolution_z=resolution_z,
                **kwargs,
            )
            self._grid_resolution = resolution
        else:
            self._grid = grid_cls(self, resolution, **kwargs)
            self._grid_resolution = resolution
        self._grid_request_signature = request_signature

        t_raster_end = time.perf_counter()

        if disk_cache_enabled:
            if cache_path is None:
                cache_key = _design_cache_key(
                    self,
                    resolution=float(resolution),
                    grid_kind=(
                        "3d"
                        if (hasattr(self._grid, "is_3d") and self._grid.is_3d)
                        else "2d"
                    ),
                    resolution_z=float(getattr(self._grid, "resolution_z", resolution)),
                    aa_config=aa_config,
                )
                cache_path = _raster_cache_path(cache_key)
            t_save = time.perf_counter()
            _save_grid_to_cache(self._grid, cache_path)
            if timing_enabled:
                display_status(
                    (
                        f"Raster cache saved: {cache_path.name} | "
                        f"save={time.perf_counter() - t_save:.2f}s"
                    ),
                    "info",
                )

        if timing_enabled:
            display_status(
                (
                    f"Rasterize wall-time: {t_raster_end - t_raster_start:.2f}s | "
                    f"total={time.perf_counter() - t_total_start:.2f}s"
                ),
                "info",
            )

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
            background=background_material,
        )
        new_design.structures = []

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

        new_design.is_3d, new_design.depth, new_design.time = (
            self.is_3d,
            self.depth,
            self.time,
        )
        new_design._explicit_size = self._explicit_size
        new_design._centered_coordinates = self._centered_coordinates
        new_design.background = (
            new_design.structures[0].material if new_design.structures else None
        )
        new_design.layers = self.layers.copy() if hasattr(self, "layers") else {}

        return new_design

    def to_plot_data(self, **kwargs):
        """Return renderer-agnostic design data for manual plotting."""
        from beamz.visual.data import design_plot_data

        return design_plot_data(self, **kwargs)

    def plot(self, **kwargs):
        """Plot the design layout using the matplotlib backend."""
        from beamz.visual.mpl import plot_design

        kwargs.setdefault("show", False)
        return plot_design(self, **kwargs)

    def show(self, **kwargs):
        """Display the design using the matplotlib backend."""
        kwargs.setdefault("show", True)
        return self.plot(**kwargs)
