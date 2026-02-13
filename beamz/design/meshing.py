import numpy as np

from beamz.design.structures import Rectangle
from beamz.visual.helpers import (
    create_rich_progress,
    display_status,
)


class MaterialGrids:
    """Bundles the 8 material property arrays with bulk operations."""
    NAMES = ('permittivity', 'permeability', 'conductivity', 'k', 'rho', 'cp', 'dn_dT', 'T0')
    DEFAULTS = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0)

    def __init__(self, shape):
        for name, default in zip(self.NAMES, self.DEFAULTS):
            setattr(self, name, np.full(shape, default))

    def fill_all(self, props):
        """Fill all grids with material property tuple."""
        for name, val in zip(self.NAMES, props):
            getattr(self, name).fill(val)

    def set_at(self, idx, props):
        """Set all properties at index (i,j) or (k,i,j)."""
        for name, val in zip(self.NAMES, props):
            getattr(self, name)[idx] = val

    def blend_at(self, idx, props, factor):
        """Blend properties at index with given factor."""
        for name, val in zip(self.NAMES, props):
            arr = getattr(self, name)
            arr[idx] = arr[idx] * (1 - factor) + val * factor

    def set_region(self, slices, props):
        """Set all properties for a slice/index-array region."""
        for name, val in zip(self.NAMES, props):
            getattr(self, name)[slices] = val

    def assign_to(self, target):
        """Copy all grids as attributes onto target object."""
        for name in self.NAMES:
            setattr(target, name, getattr(self, name))


class BaseMeshGrid:
    """Base class for mesh grids with common functionality."""

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

    def __init__(self, design, resolution):
        super().__init__(design, resolution)

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
        """Painters algorithm: rasterize design into a grid using super-sampling.

        Iterates through structures in order (background first, foreground last).
        Uses bounding-box clipping and fast paths for axis-aligned rectangles,
        circles, and rings. Boundary cells get 3x3 super-sampling for anti-aliasing.
        """
        width, height = self.design.width, self.design.height
        grid_width, grid_height = int(width / self.resolution), int(
            height / self.resolution
        )
        cell_size = self.resolution

        # Create grid of cell centers
        x_centers = np.linspace(0.5 * cell_size, width - 0.5 * cell_size, grid_width)
        y_centers = np.linspace(0.5 * cell_size, height - 0.5 * cell_size, grid_height)

        # Precompute offsets for all 9 sample points
        offsets = np.array([-0.25, 0, 0.25]) * cell_size
        sample_dx, sample_dy = np.meshgrid(offsets, offsets)
        sample_dx = sample_dx.flatten()
        sample_dy = sample_dy.flatten()
        num_samples = len(sample_dx)

        grids = MaterialGrids((grid_height, grid_width))

        # Fill background (first structure)
        if len(self.design.structures) > 0:
            background = self.design.structures[0]
            if hasattr(background, "material") and background.material is not None:
                grids.fill_all(self._get_all_material_props(background.material))

        # Process remaining structures
        with create_rich_progress() as progress:
            task = progress.add_task(
                "Rasterizing structures...", total=len(self.design.structures)
            )
            progress.update(task, advance=1)  # Skip background

            for idx in range(1, len(self.design.structures)):
                structure = self.design.structures[idx]

                if hasattr(structure, "is_pml") and structure.is_pml:
                    progress.update(task, advance=1)
                    continue
                if not hasattr(structure, "material") or structure.material is None:
                    progress.update(task, advance=1)
                    continue

                is_custom_material = hasattr(structure.material, "get_permittivity")
                props = None if is_custom_material else self._get_all_material_props(structure.material)

                try:
                    bbox_indices = self._get_bbox_indices(
                        structure, grid_height, grid_width, cell_size
                    )
                    if bbox_indices is None:
                        progress.update(task, advance=1)
                        continue
                    min_i, min_j, max_i, max_j = bbox_indices

                    # Dispatch to shape-specific fast path
                    if isinstance(structure, Rectangle) and self._is_axis_aligned(structure):
                        self._rasterize_rectangle(
                            structure, grids, props, is_custom_material,
                            grid_height, grid_width, cell_size,
                            x_centers, y_centers, sample_dx, sample_dy, num_samples,
                        )
                    elif hasattr(structure, "radius") and not hasattr(structure, "inner_radius"):
                        self._rasterize_circle(
                            structure, grids, props,
                            min_i, min_j, max_i, max_j, cell_size,
                            x_centers, y_centers, sample_dx, sample_dy, num_samples,
                        )
                    elif hasattr(structure, "inner_radius") and hasattr(structure, "outer_radius"):
                        self._rasterize_ring(
                            structure, grids, props,
                            min_i, min_j, max_i, max_j, cell_size,
                            x_centers, y_centers, sample_dx, sample_dy, num_samples,
                        )
                    else:
                        self._rasterize_polygon(
                            structure, grids, props, is_custom_material,
                            min_i, min_j, max_i, max_j, cell_size,
                            x_centers, y_centers, sample_dx, sample_dy, num_samples,
                        )

                except (AttributeError, TypeError) as e:
                    print(
                        f"Warning: Structure {type(structure)} doesn't have proper bounding box: {e}"
                    )

                progress.update(task, advance=1)

        grids.assign_to(self)

    @staticmethod
    def _is_axis_aligned(structure):
        """Check if a Rectangle is axis-aligned (not rotated)."""
        return (
            structure.vertices[0][0] == structure.position[0]
            and structure.vertices[0][1] == structure.position[1]
        )

    def _get_bbox_indices(self, structure, grid_height, grid_width, cell_size):
        """Get bounding box grid indices for a structure. Returns None if outside grid."""
        bbox = structure.get_bounding_box()
        if bbox is None:
            return None

        if len(bbox) == 6:
            min_x, min_y, _, max_x, max_y, _ = bbox
        elif len(bbox) == 4:
            min_x, min_y, max_x, max_y = bbox
        else:
            raise ValueError(f"Invalid bounding box format: {bbox}")

        min_i = max(0, int(min_y / cell_size) - 1)
        min_j = max(0, int(min_x / cell_size) - 1)
        max_i = min(grid_height, int(np.ceil(max_y / cell_size)) + 1)
        max_j = min(grid_width, int(np.ceil(max_x / cell_size)) + 1)

        if min_i >= grid_height or min_j >= grid_width or max_i <= 0 or max_j <= 0:
            return None
        return min_i, min_j, max_i, max_j

    def _supersample_cell(self, cx, cy, sample_dx, sample_dy, num_samples, contains_fn):
        """Count how many of the 3x3 sample points are inside the shape."""
        count = 0
        for k in range(num_samples):
            if contains_fn(cx + sample_dx[k], cy + sample_dy[k]):
                count += 1
        return count

    def _rasterize_rectangle(self, structure, grids, props, is_custom_material,
                             grid_height, grid_width, cell_size,
                             x_centers, y_centers, sample_dx, sample_dy, num_samples):
        """Fast path for axis-aligned rectangles."""
        rect_min_j = max(0, int(structure.position[0] / cell_size))
        rect_min_i = max(0, int(structure.position[1] / cell_size))
        rect_max_j = min(grid_width, int(np.ceil((structure.position[0] + structure.width) / cell_size)))
        rect_max_i = min(grid_height, int(np.ceil((structure.position[1] + structure.height) / cell_size)))

        # Interior cells (fully covered)
        inner_min_j = max(0, int((structure.position[0] + 0.25 * cell_size) / cell_size))
        inner_min_i = max(0, int((structure.position[1] + 0.25 * cell_size) / cell_size))
        inner_max_j = min(grid_width, int(np.floor((structure.position[0] + structure.width - 0.25 * cell_size) / cell_size)))
        inner_max_i = min(grid_height, int(np.floor((structure.position[1] + structure.height - 0.25 * cell_size) / cell_size)))

        if inner_max_i > inner_min_i and inner_max_j > inner_min_j:
            if is_custom_material:
                for i in range(inner_min_i, inner_max_i):
                    for j in range(inner_min_j, inner_max_j):
                        p = self._get_all_material_props(structure.material, x_centers[j], y_centers[i])
                        grids.set_at((i, j), p)
            else:
                s = np.s_[inner_min_i:inner_max_i, inner_min_j:inner_max_j]
                grids.set_region(s, props)

        # Boundary cells (need super-sampling)
        boundary_mask = np.zeros((rect_max_i - rect_min_i, rect_max_j - rect_min_j), dtype=bool)
        if rect_min_i < inner_min_i:
            boundary_mask[: inner_min_i - rect_min_i, :] = True
        if inner_max_i < rect_max_i:
            boundary_mask[inner_max_i - rect_min_i :, :] = True
        if rect_min_j < inner_min_j:
            boundary_mask[:, : inner_min_j - rect_min_j] = True
        if inner_max_j < rect_max_j:
            boundary_mask[:, inner_max_j - rect_min_j :] = True

        sx, sy = structure.position[0], structure.position[1]
        sw, sh = structure.width, structure.height

        boundary_indices = np.where(boundary_mask)
        for idx in range(len(boundary_indices[0])):
            i = boundary_indices[0][idx] + rect_min_i
            j = boundary_indices[1][idx] + rect_min_j
            cx, cy = x_centers[j], y_centers[i]

            samples_inside = self._supersample_cell(
                cx, cy, sample_dx, sample_dy, num_samples,
                lambda x, y: sx <= x < sx + sw and sy <= y < sy + sh,
            )
            if samples_inside > 0:
                blend_factor = samples_inside / num_samples
                if is_custom_material:
                    props = self._get_all_material_props(structure.material, cx, cy)
                grids.blend_at((i, j), props, blend_factor)

    def _rasterize_circle(self, structure, grids, props,
                          min_i, min_j, max_i, max_j, cell_size,
                          x_centers, y_centers, sample_dx, sample_dy, num_samples):
        """Fast path for circles using distance-based classification."""
        center_x, center_y = structure.position[0], structure.position[1]
        radius = structure.radius

        j_indices = np.arange(min_j, max_j)
        i_indices = np.arange(min_i, max_i)
        X, Y = np.meshgrid(x_centers[j_indices], y_centers[i_indices])
        distances = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        diag = 0.3536 * cell_size  # sqrt(2)/4
        fully_inside = distances + diag <= radius
        boundary = (distances - diag <= radius) & ~fully_inside

        # Bulk set fully inside cells
        local_i, local_j = np.where(fully_inside)
        if len(local_i) > 0:
            grids.set_region((local_i + min_i, local_j + min_j), props)

        # Super-sample boundary cells
        boundary_i, boundary_j = np.where(boundary)
        for idx in range(len(boundary_i)):
            i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
            cx, cy = x_centers[j], y_centers[i]
            samples_inside = self._supersample_cell(
                cx, cy, sample_dx, sample_dy, num_samples,
                lambda x, y: np.hypot(x - center_x, y - center_y) <= radius,
            )
            if samples_inside > 0:
                grids.blend_at((i, j), props, samples_inside / num_samples)

    def _rasterize_ring(self, structure, grids, props,
                        min_i, min_j, max_i, max_j, cell_size,
                        x_centers, y_centers, sample_dx, sample_dy, num_samples):
        """Fast path for rings using distance-based classification."""
        center_x, center_y = structure.position[0], structure.position[1]
        inner_radius = structure.inner_radius
        outer_radius = structure.outer_radius

        j_indices = np.arange(min_j, max_j)
        i_indices = np.arange(min_i, max_i)
        X, Y = np.meshgrid(x_centers[j_indices], y_centers[i_indices])
        distances = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        diag = 0.3536 * cell_size
        fully_inside = (distances - diag >= inner_radius) & (distances + diag <= outer_radius)
        inner_boundary = (distances - diag <= inner_radius) & (distances + diag >= inner_radius)
        outer_boundary = (distances - diag <= outer_radius) & (distances + diag >= outer_radius)
        boundary = inner_boundary | outer_boundary

        # Bulk set fully inside cells
        local_i, local_j = np.where(fully_inside)
        if len(local_i) > 0:
            grids.set_region((local_i + min_i, local_j + min_j), props)

        # Super-sample boundary cells
        boundary_i, boundary_j = np.where(boundary)
        for idx in range(len(boundary_i)):
            i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
            cx, cy = x_centers[j], y_centers[i]
            samples_inside = self._supersample_cell(
                cx, cy, sample_dx, sample_dy, num_samples,
                lambda x, y: inner_radius <= np.hypot(x - center_x, y - center_y) <= outer_radius,
            )
            if samples_inside > 0:
                grids.blend_at((i, j), props, samples_inside / num_samples)

    def _rasterize_polygon(self, structure, grids, props, is_custom_material,
                           min_i, min_j, max_i, max_j, cell_size,
                           x_centers, y_centers, sample_dx, sample_dy, num_samples):
        """General path for polygons and complex shapes."""
        if hasattr(structure, "point_in_polygon"):
            contains_func = lambda x, y: structure.point_in_polygon(x, y)
        else:
            contains_func = lambda x, y: any(
                val != def_val
                for val, def_val in zip(
                    self.design.get_material_value(x, y, z=0), [1.0, 1.0, 0.0]
                )
            )

        if hasattr(structure, "vertices") and len(getattr(structure, "vertices", [])) > 0:
            # Classify cells as fully-inside, boundary, or remaining
            inside_mask = np.zeros((max_i - min_i, max_j - min_j), dtype=bool)
            boundary_mask = np.zeros((max_i - min_i, max_j - min_j), dtype=bool)
            sample_points = [(0, 0), (-0.4, -0.4), (-0.4, 0.4), (0.4, -0.4), (0.4, 0.4)]

            for i_rel in range(max_i - min_i):
                for j_rel in range(max_j - min_j):
                    cx = x_centers[j_rel + min_j]
                    cy = y_centers[i_rel + min_i]
                    points_inside = 0
                    center_inside = False
                    if contains_func(cx, cy):
                        center_inside = True
                        points_inside += 1
                    for dx_pt, dy_pt in sample_points[1:]:
                        if contains_func(cx + dx_pt * cell_size, cy + dy_pt * cell_size):
                            points_inside += 1
                    if center_inside and points_inside == len(sample_points):
                        inside_mask[i_rel, j_rel] = True
                    elif points_inside > 0:
                        boundary_mask[i_rel, j_rel] = True

            # Set fully inside cells
            inside_i, inside_j = np.where(inside_mask)
            for idx in range(len(inside_i)):
                i, j = inside_i[idx] + min_i, inside_j[idx] + min_j
                grids.set_at((i, j), props)

            # Blend boundary + remaining cells via super-sampling
            for mask in (boundary_mask, ~inside_mask & ~boundary_mask):
                bi, bj = np.where(mask)
                for idx in range(len(bi)):
                    i, j = bi[idx] + min_i, bj[idx] + min_j
                    cx, cy = x_centers[j], y_centers[i]
                    samples_inside = self._supersample_cell(
                        cx, cy, sample_dx, sample_dy, num_samples, contains_func,
                    )
                    if samples_inside > 0:
                        grids.blend_at((i, j), props, samples_inside / num_samples)
        else:
            # Direct super-sampling for all cells in bounding box
            for i in range(min_i, max_i):
                for j in range(min_j, max_j):
                    cx, cy = x_centers[j], y_centers[i]
                    samples_inside = self._supersample_cell(
                        cx, cy, sample_dx, sample_dy, num_samples, contains_func,
                    )
                    if samples_inside > 0:
                        grids.blend_at((i, j), props, samples_inside / num_samples)

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

    def __init__(self, design, resolution_xy=None, resolution_z=None):
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
        """3D rasterization using layered 2D approach with z-layer processing."""
        width, height, depth = self.design.width, self.design.height, self.design.depth
        grid_width = int(width / self.resolution_xy)
        grid_height = int(height / self.resolution_xy)
        grid_depth = int(depth / self.resolution_z) if depth > 0 else 1

        cell_size_xy = self.resolution_xy
        cell_size_z = self.resolution_z

        # Create grid of cell centers
        x_centers = np.linspace(0.5 * cell_size_xy, width - 0.5 * cell_size_xy, grid_width)
        y_centers = np.linspace(0.5 * cell_size_xy, height - 0.5 * cell_size_xy, grid_height)
        z_centers = (
            np.linspace(0.5 * cell_size_z, depth - 0.5 * cell_size_z, grid_depth)
            if depth > 0 else [0]
        )

        # Precompute offsets for 3D super-sampling (3x3x3 = 27 samples)
        offsets_xy = np.array([-0.25, 0, 0.25]) * cell_size_xy
        offsets_z = np.array([-0.25, 0, 0.25]) * cell_size_z if depth > 0 else [0]
        sdx, sdy, sdz = np.meshgrid(offsets_xy, offsets_xy, offsets_z)
        sdx, sdy, sdz = sdx.flatten(), sdy.flatten(), sdz.flatten()
        num_samples = len(sdx)

        # Estimate dt for PML calculations
        c = 3e8
        dt_estimate = 0.5 * self.resolution / (c * np.sqrt(2))

        grids = MaterialGrids((grid_depth, grid_height, grid_width))

        # Fill background
        if len(self.design.structures) > 0:
            background = self.design.structures[0]
            if hasattr(background, "material") and background.material is not None:
                grids.fill_all(self._get_all_material_props(background.material))

        # Process structures layer by layer
        with create_rich_progress() as progress:
            task = progress.add_task(
                "Rasterizing 3D structures...", total=len(self.design.structures)
            )
            progress.update(task, advance=1)

            for idx in range(1, len(self.design.structures)):
                structure = self.design.structures[idx]

                if hasattr(structure, "is_pml") and structure.is_pml:
                    progress.update(task, advance=1)
                    continue
                if not hasattr(structure, "material") or structure.material is None:
                    progress.update(task, advance=1)
                    continue

                props = self._get_all_material_props(structure.material)

                try:
                    bbox = structure.get_bounding_box()
                    if bbox is None:
                        raise AttributeError("Bounding box is None")

                    if len(bbox) == 6:
                        min_x, min_y, min_z, max_x, max_y, max_z = bbox
                    else:
                        min_x, min_y, max_x, max_y = bbox
                        min_z, max_z = 0, 0

                    min_i = max(0, int(min_y / cell_size_xy) - 1)
                    min_j = max(0, int(min_x / cell_size_xy) - 1)
                    min_k = max(0, int(min_z / cell_size_z) - 1) if depth > 0 else 0
                    max_i = min(grid_height, int(np.ceil(max_y / cell_size_xy)) + 1)
                    max_j = min(grid_width, int(np.ceil(max_x / cell_size_xy)) + 1)
                    max_k = (
                        min(grid_depth, int(np.ceil(max_z / cell_size_z)) + 1)
                        if depth > 0 else 1
                    )

                    if (min_i >= grid_height or min_j >= grid_width or min_k >= grid_depth
                            or max_i <= 0 or max_j <= 0 or max_k <= 0):
                        progress.update(task, advance=1)
                        continue

                    # Build 3D containment function
                    if hasattr(structure, "point_in_polygon"):
                        contains_fn = lambda x, y, z: structure.point_in_polygon(x, y, z)
                    else:
                        contains_fn = lambda x, y, z: any(
                            val != def_val
                            for val, def_val in zip(
                                self.design.get_material_value(x, y, z), [1.0, 1.0, 0.0]
                            )
                        )

                    for k in range(min_k, max_k):
                        z_center = z_centers[k]
                        for i in range(min_i, max_i):
                            for j in range(min_j, max_j):
                                x_center = x_centers[j]
                                y_center = y_centers[i]

                                samples_inside = 0
                                for si in range(num_samples):
                                    if contains_fn(
                                        x_center + sdx[si],
                                        y_center + sdy[si],
                                        z_center + sdz[si],
                                    ):
                                        samples_inside += 1

                                if samples_inside > 0:
                                    grids.blend_at(
                                        (k, i, j), props, samples_inside / num_samples
                                    )

                except (AttributeError, TypeError) as e:
                    display_status(
                        f"Warning: Structure {type(structure)} processing failed: {e}",
                        "warning",
                    )

                progress.update(task, advance=1)

        # Process 3D PML boundaries
        self._process_3d_pml(
            grids.permittivity, grids.permeability, grids.conductivity,
            x_centers, y_centers, z_centers, dt_estimate,
        )

        grids.assign_to(self)

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
        if not hasattr(self.design, "boundaries") or not self.design.boundaries:
            return

        with create_rich_progress() as progress:
            task = progress.add_task(
                "Processing 3D PML boundaries...", total=len(self.design.boundaries)
            )

            for boundary in self.design.boundaries:
                # Add PML conductivity to the global 3D conductivity grid for all 6 faces
                for k, z in enumerate(z_centers):
                    for i, y in enumerate(y_centers):
                        for j, x in enumerate(x_centers):
                            # Calculate PML conductivity at this point (x,y,z)
                            pml_conductivity = boundary.get_conductivity(
                                x,
                                y,
                                z,
                                dx=self.resolution_xy,
                                dt=dt_estimate,
                                eps_avg=permittivity[k, i, j],
                                width=self.design.width,
                                height=self.design.height,
                                depth=self.design.depth,
                            )
                            if pml_conductivity > 0:
                                conductivity[k, i, j] += pml_conductivity

                progress.update(task, advance=1)

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
def create_mesh(design, resolution, auto_select=True, force_3d=False):
    """Create a mesh automatically selecting 2D or 3D based on design properties.

    Args:
        design: Design object to mesh
        resolution: Mesh resolution (or xy resolution for 3D)
        auto_select: If True, automatically choose between 2D and 3D meshing
        force_3d: If True, force 3D meshing even for 2D designs

    Returns:
        RegularGrid or RegularGrid3D instance
    """
    if force_3d or (auto_select and design.is_3d and design.depth > 0):
        display_status("Auto-selecting 3D meshing for 3D design", "info")
        return RegularGrid3D(design, resolution)
    else:
        if auto_select and design.is_3d:
            display_status(
                "Auto-selecting 2D meshing for effectively 2D design (depth=0)", "info"
            )
        return RegularGrid(design, resolution)


