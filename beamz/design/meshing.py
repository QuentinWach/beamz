import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from beamz.design.cache import _env_bool
from beamz.design.grids import MaterialGrids
from beamz.design import raster2d, raster3d
from beamz.visual.helpers import (
    display_status,
)


class BaseMeshGrid:
    """Base class for mesh grids with common functionality."""

    _SUPPORTED_AA_MODES = ("legacy_grid", "stratified_jitter")
    _DEFAULT_AA_MODE = "legacy_grid"
    _DEFAULT_AA_SAMPLES = 64
    _DEFAULT_AA_SEED = 0

    def __init__(self, design, resolution):
        self.design = design
        self.resolution = resolution
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters."""
        if self.resolution <= 0:
            raise ValueError("Resolution must be positive")
        if self.design is None:
            raise ValueError("Design cannot be None")

    def _configure_antialiasing(self, aa_mode=None, aa_samples=None, aa_seed=None):
        """Configure anti-aliasing supersampling behavior."""
        mode = str(aa_mode or self._DEFAULT_AA_MODE).strip().lower()
        if mode not in self._SUPPORTED_AA_MODES:
            raise ValueError(
                f"Unsupported aa_mode={aa_mode!r}. "
                f"Expected one of: {self._SUPPORTED_AA_MODES}"
            )
        samples = self._DEFAULT_AA_SAMPLES if aa_samples is None else int(aa_samples)
        if samples <= 0:
            raise ValueError(f"aa_samples must be positive, got {samples}")
        seed = self._DEFAULT_AA_SEED if aa_seed is None else int(aa_seed)

        self.aa_mode = mode
        self.aa_samples = samples
        self.aa_seed = seed

    @staticmethod
    def _sample_grid_shape(sample_count):
        """Choose an exact grid factorization close to square for N samples."""
        n = max(1, int(sample_count))
        nx = int(np.floor(np.sqrt(float(n))))
        while nx > 1 and (n % nx) != 0:
            nx -= 1
        ny = max(1, n // max(1, nx))
        return max(1, nx), max(1, ny)

    @staticmethod
    def _splitmix64(value):
        """Deterministic 64-bit mixer for stable per-cell scrambling."""
        mask = 0xFFFFFFFFFFFFFFFF
        z = int(value) & mask
        z = (z + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z = z ^ (z >> 31)
        return z & mask

    def _cell_scramble_params_xy(self, cell_i, cell_j, cell_k=0):
        """Return deterministic (shift_x, shift_y, rot90) for a cell."""
        mask = 0xFFFFFFFFFFFFFFFF
        key = int(self.aa_seed) & mask
        key ^= (int(cell_i) * 0x9E3779B185EBCA87) & mask
        key ^= (int(cell_j) * 0xC2B2AE3D27D4EB4F) & mask
        key ^= (int(cell_k) * 0x165667B19E3779F9) & mask

        h1 = self._splitmix64(key ^ 0xA0761D6478BD642F)
        h2 = self._splitmix64(key ^ 0xE7037ED1A0B428DB)

        mantissa_mask = (1 << 53) - 1
        shift_x = ((h1 >> 11) & mantissa_mask) / float(1 << 53)
        shift_y = ((h2 >> 11) & mantissa_mask) / float(1 << 53)
        rot90 = int((h1 ^ h2) & 0x3)
        return shift_x, shift_y, rot90

    def _scramble_offsets_xy_for_cell(
        self, sample_dx, sample_dy, cell_size, cell_i, cell_j, cell_k=0
    ):
        """Apply per-cell CP shift + 90° rotation to stratified jitter samples."""
        if self.aa_mode != "stratified_jitter":
            return sample_dx, sample_dy

        cell_size = float(cell_size)
        u = np.asarray(sample_dx, dtype=float) / cell_size + 0.5
        v = np.asarray(sample_dy, dtype=float) / cell_size + 0.5

        shift_x, shift_y, rot90 = self._cell_scramble_params_xy(cell_i, cell_j, cell_k)
        if rot90 == 1:
            u, v = v, 1.0 - u
        elif rot90 == 2:
            u, v = 1.0 - u, 1.0 - v
        elif rot90 == 3:
            u, v = 1.0 - v, u

        u = np.mod(u + shift_x, 1.0)
        v = np.mod(v + shift_y, 1.0)
        return (u - 0.5) * cell_size, (v - 0.5) * cell_size

    def _build_supersample_offsets_xy(self, cell_size):
        """Build XY supersample offsets for the configured AA mode."""
        cell_size = float(cell_size)
        if self.aa_mode == "legacy_grid":
            nx, ny = self._sample_grid_shape(self.aa_samples)
            ox = ((np.arange(nx, dtype=float) + 0.5) / float(nx) - 0.5) * cell_size
            oy = ((np.arange(ny, dtype=float) + 0.5) / float(ny) - 0.5) * cell_size
            sample_dx, sample_dy = np.meshgrid(ox, oy)
            sample_dx = sample_dx.ravel()
            sample_dy = sample_dy.ravel()
            return sample_dx, sample_dy

        nx, ny = self._sample_grid_shape(self.aa_samples)
        total = nx * ny
        rng = np.random.default_rng(self.aa_seed)
        strata_x = np.tile(np.arange(nx, dtype=float), ny)
        strata_y = np.repeat(np.arange(ny, dtype=float), nx)
        jitter_x = rng.random(total)
        jitter_y = rng.random(total)
        sample_dx = ((strata_x + jitter_x) / float(nx) - 0.5) * cell_size
        sample_dy = ((strata_y + jitter_y) / float(ny) - 0.5) * cell_size
        return sample_dx, sample_dy

    @staticmethod
    def _structure_polygon_2d(structure):
        """Convert a 2D polygonal structure to a valid Shapely polygon."""
        vertices = getattr(structure, "vertices", None) or []
        if len(vertices) < 3:
            return None

        shell = [(float(x), float(y)) for x, y, *_ in vertices]
        holes = []
        for hole in getattr(structure, "interiors", None) or []:
            if len(hole) >= 3:
                holes.append([(float(x), float(y)) for x, y, *_ in hole])

        poly = ShapelyPolygon(shell=shell, holes=holes)
        if poly.is_empty:
            return None
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not poly.is_valid or poly.geom_type != "Polygon":
            return None
        return poly

    def _build_supersample_offsets_z(self, cell_size_z, depth_samples):
        """Build Z supersample offsets for 3D fallback paths."""
        if depth_samples <= 1:
            return np.array([0.0], dtype=float)

        cell_size_z = float(cell_size_z)
        if self.aa_mode == "legacy_grid":
            return np.array([-0.25, 0.0, 0.25], dtype=float) * cell_size_z

        n = int(depth_samples)
        rng = np.random.default_rng(self.aa_seed + 7919)
        bins = np.arange(n, dtype=float)
        jitter = rng.random(n)
        return ((bins + jitter) / float(n) - 0.5) * cell_size_z

    def _get_material_properties_safe(self, material, x=0, y=0, z=0):
        """Safely get material properties from either Material or CustomMaterial objects."""
        if material is None:
            return 1.0, 1.0, 0.0

        # Check if this is a CustomMaterial (has getter methods)
        if hasattr(material, "get_permittivity"):
            try:
                permittivity = material.get_permittivity(x, y, z)
                permeability = material.get_permeability(x, y, z)
                conductivity = material.get_conductivity(x, y, z)

                # Handle numpy arrays vs scalars
                if hasattr(permittivity, "item"):
                    permittivity = permittivity.item()
                if hasattr(permeability, "item"):
                    permeability = permeability.item()
                if hasattr(conductivity, "item"):
                    conductivity = conductivity.item()

                return permittivity, permeability, conductivity
            except Exception as e:
                print(f"Warning: CustomMaterial evaluation failed: {e}, using defaults")
                return (
                    getattr(material, "default_permittivity", 1.0),
                    getattr(material, "default_permeability", 1.0),
                    getattr(material, "default_conductivity", 0.0),
                )

        # Traditional Material object (direct attributes)
        elif hasattr(material, "permittivity"):
            return (
                getattr(material, "permittivity", 1.0),
                getattr(material, "permeability", 1.0),
                getattr(material, "conductivity", 0.0),
            )

        # Fallback for unknown material types
        else:
            print(
                f"Warning: Unknown material type {type(material)}, using vacuum properties"
            )
            return 1.0, 1.0, 0.0

    def _get_thermal_properties_safe(self, material, x=0, y=0, z=0):
        """Safely get thermal properties from material objects."""
        if material is None:
            return 0.0, 0.0, 0.0, 0.0, 300.0

        # CustomMaterial or Material: thermal params are constants
        k = getattr(material, "k", 0.0)
        rho = getattr(material, "rho", 0.0)
        cp = getattr(material, "cp", 0.0)
        dn_dT = getattr(material, "dn_dT", 0.0)
        T0 = getattr(material, "T0", 300.0)
        return k, rho, cp, dn_dT, T0

    def _get_all_material_props(self, material, x=0, y=0, z=0):
        """Get all 8 material properties as a single tuple matching MaterialGrids.NAMES order."""
        perm, permb, cond = self._get_material_properties_safe(material, x, y, z)
        k, rho, cp, dn_dT, T0 = self._get_thermal_properties_safe(material, x, y, z)
        return (perm, permb, cond, k, rho, cp, dn_dT, T0)

    def get_thermal_grids(self):
        """Get thermal property grids."""
        return self.k, self.rho, self.cp, self.dn_dT, self.T0

    def get_material_grids(self, resolution=None):
        """Get the material property grids."""
        return self.permittivity, self.conductivity, self.permeability

    def rasterize(self, resolution=None):
        """Return self if resolution matches, otherwise raise."""
        if resolution is None or resolution == self.resolution:
            return self
        raise ValueError(
            "Cannot re-rasterize with different resolution. Use Design.rasterize()"
        )


class RegularGrid(BaseMeshGrid):
    """2D Regular grid meshing for 2D designs (backwards compatible)."""

    def __init__(
        self,
        design,
        resolution,
        aa_mode="legacy_grid",
        aa_samples=64,
        aa_seed=0,
    ):
        super().__init__(design, resolution)
        self._configure_antialiasing(
            aa_mode=aa_mode,
            aa_samples=aa_samples,
            aa_seed=aa_seed,
        )

        # Check if this is actually a 2D design
        if design.is_3d and design.depth > 0:
            display_status(
                "Warning: Using 2D RegularGrid for a 3D design. Use RegularGrid3D for proper 3D meshing.",
                "warning",
            )

        # Determine is_3d property for compatibility with Simulation class
        self.is_3d = design.is_3d and design.depth > 0

        # Rasterize the design (assigns all 8 material grids via MaterialGrids)
        self.__rasterize__()

        # Set grid properties
        self.shape = self.permittivity.shape
        self.dx = self.resolution
        self.dy = self.resolution
        self.width = self.design.width
        self.height = self.design.height

    def __rasterize__(self):
        """Rasterize the design into a 2D grid."""
        raster2d.rasterize(self)

    @staticmethod
    def _is_axis_aligned(structure):
        """Check if a Rectangle is axis-aligned (not rotated)."""
        return raster2d.is_axis_aligned(structure)

    def _get_bbox_indices(self, structure, grid_height, grid_width, cell_size):
        """Get bounding box grid indices for a structure."""
        return raster2d.get_bbox_indices(structure, grid_height, grid_width, cell_size)

    def _supersample_cell(
        self,
        cx,
        cy,
        sample_dx,
        sample_dy,
        num_samples,
        contains_fn,
        *,
        cell_i=None,
        cell_j=None,
        cell_k=0,
        cell_size=None,
    ):
        """Count how many configured sample points are inside the shape."""
        return raster2d.supersample_cell(
            self,
            cx,
            cy,
            sample_dx,
            sample_dy,
            num_samples,
            contains_fn,
            cell_i=cell_i,
            cell_j=cell_j,
            cell_k=cell_k,
            cell_size=cell_size,
        )

    def _rasterize_rectangle(
        self,
        structure,
        grids,
        props,
        is_custom_material,
        grid_height,
        grid_width,
        cell_size,
        x_centers,
        y_centers,
        sample_dx,
        sample_dy,
        num_samples,
    ):
        """Exact area coverage for axis-aligned rectangles."""
        raster2d.rasterize_rectangle(
            self,
            structure,
            grids,
            props,
            is_custom_material,
            grid_height,
            grid_width,
            cell_size,
            x_centers,
            y_centers,
            sample_dx,
            sample_dy,
            num_samples,
        )

    def _rasterize_circle(
        self,
        structure,
        grids,
        props,
        min_i,
        min_j,
        max_i,
        max_j,
        cell_size,
        x_centers,
        y_centers,
        sample_dx,
        sample_dy,
        num_samples,
    ):
        """Fast path for circles using distance-based classification."""
        raster2d.rasterize_circle(
            self,
            structure,
            grids,
            props,
            min_i,
            min_j,
            max_i,
            max_j,
            cell_size,
            x_centers,
            y_centers,
            sample_dx,
            sample_dy,
            num_samples,
        )

    def _rasterize_ring(
        self,
        structure,
        grids,
        props,
        min_i,
        min_j,
        max_i,
        max_j,
        cell_size,
        x_centers,
        y_centers,
        sample_dx,
        sample_dy,
        num_samples,
    ):
        """Fast path for rings using distance-based classification."""
        raster2d.rasterize_ring(
            self,
            structure,
            grids,
            props,
            min_i,
            min_j,
            max_i,
            max_j,
            cell_size,
            x_centers,
            y_centers,
            sample_dx,
            sample_dy,
            num_samples,
        )

    def _rasterize_polygon(
        self,
        structure,
        grids,
        props,
        is_custom_material,
        min_i,
        min_j,
        max_i,
        max_j,
        cell_size,
        x_centers,
        y_centers,
        sample_dx,
        sample_dy,
        num_samples,
    ):
        """General path for polygons and complex shapes."""
        raster2d.rasterize_polygon(
            self,
            structure,
            grids,
            props,
            is_custom_material,
            min_i,
            min_j,
            max_i,
            max_j,
            cell_size,
            x_centers,
            y_centers,
            sample_dx,
            sample_dy,
            num_samples,
        )

    def show(self, field: str = "permittivity"):
        """Display the rasterized grid with properly scaled SI units."""
        from beamz.visual.overlays import show_mesh_grid

        grid = getattr(self, field, None)
        if grid is not None:
            show_mesh_grid(grid, self.design, field)
        else:
            print("Grid not rasterized yet.")


class RegularGrid3D(BaseMeshGrid):
    """3D Regular grid meshing for 3D designs."""

    def __init__(
        self,
        design,
        resolution_xy=None,
        resolution_z=None,
        aa_mode="legacy_grid",
        aa_samples=64,
        aa_seed=0,
    ):
        # Handle different resolution input formats
        if isinstance(design, (int, float)) and resolution_xy is None:
            # Legacy format: RegularGrid3D(resolution) - set uniform resolution
            resolution = design
            design = resolution_xy  # Second argument is actually design
            resolution_xy = resolution
            resolution_z = resolution
        elif resolution_xy is None:
            # Default to design.resolution if available, otherwise same as xy
            resolution_xy = getattr(design, "resolution", resolution_xy)
            resolution_z = resolution_xy
        elif resolution_z is None:
            # Only xy resolution provided, use same for z
            resolution_z = resolution_xy

        super().__init__(design, resolution_xy)
        self._configure_antialiasing(
            aa_mode=aa_mode,
            aa_samples=aa_samples,
            aa_seed=aa_seed,
        )

        # Store separate resolutions for xy and z
        self.resolution_xy = resolution_xy
        self.resolution_z = resolution_z

        # Rasterize the design (assigns all 8 material grids via MaterialGrids)
        self.__rasterize_3d__()

        # Calculate grid dimensions for status message
        width, height, depth = self.design.width, self.design.height, self.design.depth
        grid_width = int(width / self.resolution_xy)
        grid_height = int(height / self.resolution_xy)
        grid_depth = int(depth / self.resolution_z) if depth > 0 else 1

        # Set grid properties
        self.shape = self.permittivity.shape
        self.dx = self.resolution_xy
        self.dy = self.resolution_xy
        self.dz = self.resolution_z
        self.width = self.design.width
        self.height = self.design.height
        self.depth = self.design.depth
        display_status(
            f"Created 3D mesh: {grid_width} × {grid_height} × {grid_depth} cells",
            "success",
        )

    def __rasterize_3d__(self):
        """Rasterize the design into a 3D grid."""
        raster3d.rasterize(self)

    def _process_3d_pml(
        self,
        permittivity,
        permeability,
        conductivity,
        x_centers,
        y_centers,
        z_centers,
        dt_estimate,
    ):
        """Process 3D PML boundaries and add conductivity to the grid."""
        raster3d.process_pml(
            self,
            permittivity,
            permeability,
            conductivity,
            x_centers,
            y_centers,
            z_centers,
            dt_estimate,
        )

    @staticmethod
    def _is_axis_aligned(structure):
        """Check if a rectangle is axis-aligned (not rotated)."""
        return raster3d.is_axis_aligned(structure)

    def _get_bbox_indices_3d(
        self,
        structure,
        *,
        grid_height,
        grid_width,
        grid_depth,
        cell_size_xy,
        cell_size_z,
        margin_cells=1,
    ):
        """Get clipped 3D bbox index range for a structure."""
        return raster3d.get_bbox_indices(
            structure,
            grid_height=grid_height,
            grid_width=grid_width,
            grid_depth=grid_depth,
            cell_size_xy=cell_size_xy,
            cell_size_z=cell_size_z,
            margin_cells=margin_cells,
        )

    def _rasterize_rectangle_3d_fast(
        self,
        *,
        structure,
        grids,
        props,
        grid_height,
        grid_width,
        grid_depth,
        cell_size_xy,
        cell_size_z,
    ):
        """Fast fill for axis-aligned rectangular prisms."""
        raster3d.rasterize_rectangle(
            structure=structure,
            grids=grids,
            props=props,
            grid_height=grid_height,
            grid_width=grid_width,
            grid_depth=grid_depth,
            cell_size_xy=cell_size_xy,
            cell_size_z=cell_size_z,
        )

    def _rasterize_polygon_3d_fast(
        self,
        *,
        structure,
        grids,
        props,
        min_i,
        min_j,
        min_k,
        max_i,
        max_j,
        max_k,
        x_centers,
        y_centers,
        cell_size_xy,
        cell_size_z,
    ):
        """Vectorized anti-aliased fill for extruded polygons.

        Uses configurable supersampling in XY and exact voxel overlap in Z so imported
        taper polygons get the same subpixel smoothing behavior as fallback paths.
        """
        return raster3d.rasterize_polygon(
            self,
            structure=structure,
            grids=grids,
            props=props,
            min_i=min_i,
            min_j=min_j,
            min_k=min_k,
            max_i=max_i,
            max_j=max_j,
            max_k=max_k,
            x_centers=x_centers,
            y_centers=y_centers,
            cell_size_xy=cell_size_xy,
            cell_size_z=cell_size_z,
        )

    def _rasterize_structure_3d_fallback(
        self,
        *,
        structure,
        grids,
        props,
        min_i,
        min_j,
        min_k,
        max_i,
        max_j,
        max_k,
        cell_size_xy,
        cell_size_z,
        x_centers,
        y_centers,
        z_centers,
    ):
        """Fallback supersampling path for non-rectilinear 3D structures."""
        raster3d.rasterize_fallback(
            self,
            structure=structure,
            grids=grids,
            props=props,
            min_i=min_i,
            min_j=min_j,
            min_k=min_k,
            max_i=max_i,
            max_j=max_j,
            max_k=max_k,
            cell_size_xy=cell_size_xy,
            cell_size_z=cell_size_z,
            x_centers=x_centers,
            y_centers=y_centers,
            z_centers=z_centers,
        )

    def get_2d_slice(self, z_index=None, z_position=None):
        """Extract a 2D slice from the 3D grid.

        Args:
            z_index: Index of the z-layer to extract
            z_position: Physical z-position to extract (will find nearest layer)

        Returns:
            dict with 'permittivity', 'permeability', 'conductivity' 2D arrays
        """
        if z_index is None and z_position is None:
            z_index = self.shape[0] // 2  # Middle layer
        elif z_position is not None:
            z_index = int(z_position / self.resolution_z)
            z_index = max(0, min(self.shape[0] - 1, z_index))

        return {
            "permittivity": self.permittivity[z_index, :, :],
            "permeability": self.permeability[z_index, :, :],
            "conductivity": self.conductivity[z_index, :, :],
        }

    def show_3d(self, field="permittivity", slice_spacing=1, alpha=0.3):
        """Display 3D visualization of the mesh."""
        from beamz.visual.overlays import show_mesh_3d

        grid = getattr(self, field, None)
        if grid is None:
            raise ValueError(f"Unknown field: {field}")
        show_mesh_3d(grid, self.design, field, slice_spacing, alpha)

    def show(self, field="permittivity", z_index=None, z_position=None):
        """Display a 2D slice of the 3D mesh (backwards compatible interface)."""
        from beamz.visual.overlays import show_mesh_slice

        slice_data = self.get_2d_slice(z_index, z_position)
        grid = slice_data[field]

        if grid is not None:
            z_idx = z_index if z_index is not None else self.shape[0] // 2
            show_mesh_slice(grid, self.design, field, z_idx, self.resolution_z)
        else:
            print("Grid not rasterized yet.")


# Convenience functions for automatic mesh selection
def create_mesh(design, resolution, auto_select=True, force_3d=False, **kwargs):
    """Create a mesh automatically selecting 2D or 3D based on design properties.

    Args:
        design: Design object to mesh
        resolution: Mesh resolution (or xy resolution for 3D)
        auto_select: If True, automatically choose between 2D and 3D meshing
        force_3d: If True, force 3D meshing even for 2D designs
        **kwargs: Extra mesh kwargs forwarded to RegularGrid/RegularGrid3D

    Returns:
        RegularGrid or RegularGrid3D instance
    """
    resolution_z = kwargs.pop("resolution_z", None)
    if force_3d or (auto_select and design.is_3d and design.depth > 0):
        display_status("Auto-selecting 3D meshing for 3D design", "info")
        return RegularGrid3D(
            design,
            resolution_xy=resolution,
            resolution_z=resolution_z,
            **kwargs,
        )
    else:
        if auto_select and design.is_3d:
            display_status(
                "Auto-selecting 2D meshing for effectively 2D design (depth=0)", "info"
            )
        return RegularGrid(design, resolution, **kwargs)
