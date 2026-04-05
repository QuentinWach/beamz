import os
import time
from dataclasses import replace

import numpy as np

from beamz.const import µm
from beamz.design.cache import (
    _cache_path_for_grid,
    _env_bool,
    _grid_kind_for_request,
    _normalize_aa_config,
    _raster_request_signature,
    _save_grid_to_cache,
    _try_load_cached_grid,
)
from beamz.design.materials import Material
from beamz.design.spec import DesignSpec, build_design_spec
from beamz.design.state import DesignState
from beamz.design.merge import (
    _find_rings_to_preserve,
    _group_by_material,
    _material_key,
    _merge_groups,
    _rebuild_structure_list,
    _shapely_to_polygons,
    _to_shapely,
)
from beamz.design.structures import (
    Polygon,
    Rectangle,
    Ring,
)


class Design:
    _SPEC_FIELDS = frozenset(DesignSpec.__dataclass_fields__.keys())
    _STATE_MAP = {
        "sources": "sources",
        "monitors": "monitors",
        "boundaries": "boundaries",
        "layers": "layers",
        "_grid": "grid",
        "_grid_resolution": "grid_resolution",
        "_grid_request_signature": "grid_request_signature",
    }

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
        object.__setattr__(
            self,
            "spec",
            build_design_spec(
                width=width,
                height=height,
                depth=depth if depth is not None else 0.0,
                structures=(background,),
                time=0.0,
            ),
        )
        object.__setattr__(self, "state", DesignState())

    def __str__(self):
        return f"Design with {len(self.structures)} structures ({'3D' if self.is_3d else '2D'})"

    def __getattr__(self, name):
        spec = self.__dict__.get("spec")
        if spec is not None and hasattr(spec, name):
            return getattr(spec, name)
        state = self.__dict__.get("state")
        if state is not None and name in self._STATE_MAP:
            return getattr(state, self._STATE_MAP[name])
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in {"spec", "state"}:
            object.__setattr__(self, name, value)
            return
        if name in self._SPEC_FIELDS and "spec" in self.__dict__:
            new_spec = replace(self.spec, **{name: value})
            object.__setattr__(self, "spec", new_spec)
            if name in {"width", "height", "depth", "structures", "is_3d", "time"}:
                self.state.grid = None
                self.state.grid_resolution = None
                self.state.grid_request_signature = None
            return
        if name in self._STATE_MAP and "state" in self.__dict__:
            setattr(self.state, self._STATE_MAP[name], value)
            return
        object.__setattr__(self, name, value)

    def __iadd__(self, structure):
        """Implement += operator for adding structures."""
        self.add(structure)
        return self

    def _set_grid_state(self, grid, *, resolution, request_signature):
        self.state.grid = grid
        self.state.grid_resolution = resolution
        self.state.grid_request_signature = request_signature

    def _instantiate_grid(self, grid_type, resolution, request_signature, **kwargs):
        from beamz.design.meshing import RegularGrid, RegularGrid3D, create_mesh

        grid_cls = None
        if isinstance(grid_type, str):
            gt = grid_type.lower()
            if gt in {"regular", "regulargrid", "2d"}:
                grid_cls = RegularGrid
            elif gt in {"regular3d", "3d"}:
                grid_cls = RegularGrid3D
            elif gt in {"auto", "auto-select", "autoselect"}:
                grid = create_mesh(self, resolution, **kwargs)
                self._set_grid_state(
                    grid, resolution=resolution, request_signature=request_signature
                )
                return grid
            else:
                return None
        elif isinstance(grid_type, type):
            grid_cls = grid_type

        if grid_cls is RegularGrid3D:
            grid = grid_cls(
                self,
                resolution_xy=resolution,
                resolution_z=kwargs.pop("resolution_z", None),
                **kwargs,
            )
        else:
            grid = grid_cls(self, resolution, **kwargs)
        self._set_grid_state(grid, resolution=resolution, request_signature=request_signature)
        return grid

    @staticmethod
    def _is_monitor(obj):
        module = type(obj).__module__
        return module.startswith("beamz.devices.monitors") or hasattr(obj, "power_history")

    @staticmethod
    def _is_source(obj):
        module = type(obj).__module__
        return module.startswith("beamz.devices.sources") or (
            hasattr(obj, "inject") and hasattr(obj, "signal")
        )

    def unify_polygons(self):
        """Merge overlapping polygons with the same material properties into unified shapes."""
        material_groups, structures_to_remove = _group_by_material(self.structures)
        rings_to_preserve = _find_rings_to_preserve(
            material_groups, structures_to_remove
        )
        new_structures, structures_to_remove = _merge_groups(
            material_groups, rings_to_preserve, structures_to_remove
        )
        self.structures = tuple(
            _rebuild_structure_list(
                list(self.structures), structures_to_remove, new_structures, material_groups
            )
        )
        return True

    def add(self, structure: type[Polygon]):
        """Add structure to the design and update 3D flag if needed."""
        # Set back-reference to design if the structure supports it
        if hasattr(structure, "design"):
            structure.design = self

        if self._is_monitor(structure):
            self.monitors.append(structure)
        elif self._is_source(structure):
            self.sources.append(structure)
        else:
            self.structures = tuple(list(self.structures) + [structure])

        structure_is_3d = bool(getattr(structure, "is_3d", False))
        if hasattr(structure, "depth") and structure.depth != 0:
            structure_is_3d = True
        position = getattr(structure, "position", None)
        if position is not None and len(position) > 2 and position[2] != 0:
            structure_is_3d = True
        if hasattr(structure, "vertices") and structure.vertices:
            for vertex in structure.vertices:
                if len(vertex) > 2 and vertex[2] != 0:
                    structure_is_3d = True
                    break
        if structure_is_3d and not self.is_3d:
            self.is_3d = True

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

    def slice2d(
        self,
        *,
        field="permittivity",
        resolution=None,
        z_index=None,
        z_position=None,
        title=None,
        **raster_kwargs,
    ):
        """Return a rasterized 2D material slice prepared for plotting."""
        from beamz.visual.data import Slice2D

        if resolution is None:
            resolution = getattr(self.state, "grid_resolution", None)
        if resolution is None:
            raise ValueError(
                "resolution is required for Design.slice2d(...) unless the design "
                "has already been rasterized."
            )

        grid = self.rasterize(resolution, **raster_kwargs)
        if not hasattr(grid, field):
            raise ValueError(f"Unknown grid field: {field!r}")

        if getattr(grid, "is_3d", False):
            if z_index is None and z_position is None:
                z_index = grid.shape[0] // 2
            elif z_position is not None:
                z_index = int(float(z_position) / float(grid.resolution_z))
                z_index = max(0, min(grid.shape[0] - 1, z_index))
            values = np.asarray(getattr(grid, field)[z_index, :, :])
            position = float(z_index) * float(grid.resolution_z)
        else:
            values = np.asarray(getattr(grid, field))
            position = None

        plane = "xy"
        extent = (0.0, float(self.width), 0.0, float(self.height))
        default_title = title or f"{field} slice" + (
            f" ({plane}, z={position:.3e} m)" if position is not None else f" ({plane})"
        )
        return Slice2D(
            values=values,
            extent=extent,
            value_label=field,
            plane=plane,
            position=position,
            title=default_title,
            style={"cmap": "viridis", "origin": "lower", "aspect": "equal"},
        )

    def rasterize(
        self,
        resolution: float,
        grid_type: str = "auto",
        force_recompute: bool = False,
        **kwargs,
    ):
        """Rasterize design into a mesh grid with in-memory and optional disk cache."""
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
        request_signature = _raster_request_signature(
            resolution_xy=resolution,
            resolution_z=requested_resolution_z,
            grid_kind=grid_kind,
            aa_config=aa_config,
        )

        # Return cached grid if request signature matches and no force recompute
        if not force_recompute and self.state.grid is not None:
            cached_sig = getattr(self.state, "grid_request_signature", None)
            if cached_sig is not None and cached_sig == request_signature:
                return self.state.grid

        cache_path = None
        if disk_cache_enabled and not force_recompute:
            t_load = time.perf_counter()
            cached_grid, cache_path, _ = _try_load_cached_grid(
                design_obj=self,
                resolution=float(resolution),
                resolution_z=requested_resolution_z,
                grid_kind=grid_kind,
                aa_config=aa_config,
            )
            if cached_grid is not None:
                self._set_grid_state(
                    cached_grid, resolution=resolution, request_signature=request_signature
                )
                if timing_enabled:
                    display_status(
                        (
                            f"Raster cache hit ({grid_kind}): {cache_path.name} | "
                            f"load={time.perf_counter() - t_load:.2f}s"
                        ),
                        "success",
                    )
                return self.state.grid

        t_raster_start = time.perf_counter()
        grid = self._instantiate_grid(
            grid_type, resolution, request_signature=request_signature, **kwargs
        )
        if grid is None:
            return None

        t_raster_end = time.perf_counter()

        if disk_cache_enabled:
            if cache_path is None:
                cache_path, _ = _cache_path_for_grid(
                    design_obj=self,
                    grid=self.state.grid,
                    resolution=float(resolution),
                    aa_config=aa_config,
                )
            t_save = time.perf_counter()
            _save_grid_to_cache(self.state.grid, cache_path)
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

        return self.state.grid

    def get_material_grids(self, resolution):
        """Get cached rasterized material property arrays at specified resolution as references."""
        if (
            self.state.grid is None
            or self.state.grid_resolution != resolution
        ):
            self.rasterize(resolution, grid_type="auto")
        return (
            self.state.grid.permittivity,
            self.state.grid.conductivity,
            self.state.grid.permeability,
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
        new_design.structures = ()
        new_design.sources = []
        new_design.monitors = []
        new_design.boundaries = list(self.boundaries)

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
                new_design.structures = tuple(list(new_design.structures) + [copied_structure])
            else:
                new_design.structures = tuple(list(new_design.structures) + [structure])

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
