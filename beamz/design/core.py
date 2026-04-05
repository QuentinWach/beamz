import os
import time

import numpy as np

from beamz.const import µm
from beamz.design.cache import (
    _boundary_signature,
    _build_grid_from_cached_arrays,
    _design_cache_key,
    _env_bool,
    _grid_kind_for_request,
    _grid_kind_for_type,
    _material_signature,
    _normalize_aa_config,
    _raster_cache_dir,
    _raster_cache_path,
    _save_grid_to_cache,
    _structure_signature,
    _to_jsonable,
)
from beamz.design.materials import Material
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
        scenario=None,
        config=None,
        **kwargs,
    ):
        """Compatibility wrapper with migration hint for the scenario-based static API."""
        legacy_keys = {
            "heater_mask",
            "heater_power",
            "fixed_temp_mask",
            "fixed_temp_value",
        }
        if legacy_keys.intersection(kwargs):
            raise ValueError(
                "Static thermal API changed. Replace legacy kwargs with "
                "Design.solve_thermal(resolution=..., scenario=ThermalScenario(...), "
                "config=StaticThermalConfig(...))."
            )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        if scenario is None:
            raise ValueError(
                "solve_static_thermal now requires scenario=ThermalScenario(...)."
            )
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
