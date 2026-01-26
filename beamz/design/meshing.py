import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from beamz.design.structures import Rectangle
from beamz.visual.helpers import (
    create_rich_progress,
    display_status,
    get_si_scale_and_label,
)


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
        self.is_3d = False
        if design.is_3d and design.depth > 0:
            self.is_3d = True

        # Calculate 2D grid dimensions
        width, height = self.design.width, self.design.height
        grid_width = int(width / self.resolution)
        grid_height = int(height / self.resolution)

        # Initialize 2D material grids
        self.permittivity = np.zeros((grid_height, grid_width))
        self.permeability = np.zeros((grid_height, grid_width))
        self.conductivity = np.zeros((grid_height, grid_width))

        # Rasterize the design
        self.__rasterize__()

        # Set grid properties
        self.shape = self.permittivity.shape
        self.dx = self.resolution
        self.dy = self.resolution
        self.width = self.design.width
        self.height = self.design.height

    def rasterize(self, resolution=None):
        """Mock rasterize method to return self if resolution matches."""
        if resolution is None or resolution == self.resolution:
            return self
        else:
            raise ValueError(
                "RegularGrid cannot re-rasterize itself with different resolution. Use Design.rasterize()"
            )

    def get_material_grids(self, resolution=None):
        """Get the material property grids."""
        return self.permittivity, self.conductivity, self.permeability

    def __rasterize__(self):
        """Painters algorithm to rasterize the design into a grid using super-sampling
        by utilizing the ordered nature of the structures and their bounding boxes.
        We iterate through the sorted list of objects:
        1. First, draw the background layer without any anti-aliasing or boundary box consideration.
        2. Then take the boundary box of the next object and create a mask for the material arrays.
        3. Then use super-sampling over that boundary box to draw this object.
        4. Do this until all objects are drawn.

        TODO:
            + Refactor into more readable code with distinct repeatable functions.
            + Write detailed documentation (see Quentin's personal notes for details).
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
        dx, dy = np.meshgrid(offsets, offsets)
        dx = dx.flatten()
        dy = dy.flatten()
        num_samples = len(dx)

        # Estimate dt for PML calculations
        c = 3e8  # Speed of light
        dt_estimate = 0.5 * self.resolution / (c * np.sqrt(2))

        # Initialize material grids with vacuum (air) properties
        permittivity = np.ones((grid_height, grid_width))
        permeability = np.ones((grid_height, grid_width))
        conductivity = np.zeros((grid_height, grid_width))

        # Start with the background (first structure)
        if len(self.design.structures) > 0:
            background = self.design.structures[0]
            if hasattr(background, "material") and background.material is not None:
                # Get background material properties safely
                bg_perm, bg_permb, bg_cond = self._get_material_properties_safe(
                    background.material
                )
                # Fast fill for background
                permittivity.fill(bg_perm)
                permeability.fill(bg_permb)
                conductivity.fill(bg_cond)

        # Process remaining structures in reverse order (foreground objects last)
        # Note: we process in ORIGINAL order, not reversed, because we want background first
        with create_rich_progress() as progress:
            task = progress.add_task(
                "Rasterizing structures...", total=len(self.design.structures)
            )
            progress.update(task, advance=1)  # Skip the background we already processed
            for idx in range(1, len(self.design.structures)):
                structure = self.design.structures[idx]
                # Skip PML visualization structures or structures without material (sources, monitors, etc.)
                if hasattr(structure, "is_pml") and structure.is_pml:
                    progress.update(task, advance=1)
                    continue
                if not hasattr(structure, "material") or structure.material is None:
                    progress.update(task, advance=1)
                    continue
                # Check if this is a CustomMaterial that needs spatial evaluation
                is_custom_material = hasattr(structure.material, "get_permittivity")
                if is_custom_material:
                    # For CustomMaterial, we'll evaluate at each spatial location during rasterization
                    mat_perm, mat_permb, mat_cond = None, None, None
                else:
                    # Cache material properties for performance (traditional Material objects)
                    mat_perm, mat_permb, mat_cond = self._get_material_properties_safe(
                        structure.material
                    )
                try:
                    # Get bounding box of the structure
                    bbox = structure.get_bounding_box()
                    if bbox is None:
                        raise AttributeError("Bounding box is None")

                    # Handle both 2D and 3D bounding boxes
                    if (
                        len(bbox) == 6
                    ):  # 3D bounding box: (min_x, min_y, min_z, max_x, max_y, max_z)
                        min_x, min_y, min_z, max_x, max_y, max_z = bbox
                    elif (
                        len(bbox) == 4
                    ):  # 2D bounding box: (min_x, min_y, max_x, max_y)
                        min_x, min_y, max_x, max_y = bbox
                    else:
                        raise ValueError(f"Invalid bounding box format: {bbox}")

                    # Convert to grid indices
                    min_i = max(0, int(min_y / cell_size) - 1)
                    min_j = max(0, int(min_x / cell_size) - 1)
                    max_i = min(grid_height, int(np.ceil(max_y / cell_size)) + 1)
                    max_j = min(grid_width, int(np.ceil(max_x / cell_size)) + 1)
                    # Skip if bounding box is outside grid
                    if (
                        min_i >= grid_height
                        or min_j >= grid_width
                        or max_i <= 0
                        or max_j <= 0
                    ):
                        progress.update(task, advance=1)
                        continue
                    # Fast paths for different structure types
                    if isinstance(structure, Rectangle) and all(
                        v == 0
                        for v in [
                            structure.vertices[0][0] - structure.position[0],
                            structure.vertices[0][1] - structure.position[1],
                        ]
                    ):
                        # FAST PATH: Axis-aligned rectangle
                        # Define rectangle bounds for grid indices
                        rect_min_j = max(0, int(structure.position[0] / cell_size))
                        rect_min_i = max(0, int(structure.position[1] / cell_size))
                        rect_max_j = min(
                            grid_width,
                            int(
                                np.ceil(
                                    (structure.position[0] + structure.width)
                                    / cell_size
                                )
                            ),
                        )
                        rect_max_i = min(
                            grid_height,
                            int(
                                np.ceil(
                                    (structure.position[1] + structure.height)
                                    / cell_size
                                )
                            ),
                        )
                        # Identify interior and boundary cells
                        inner_min_j = max(
                            0,
                            int((structure.position[0] + 0.25 * cell_size) / cell_size),
                        )
                        inner_min_i = max(
                            0,
                            int((structure.position[1] + 0.25 * cell_size) / cell_size),
                        )
                        inner_max_j = min(
                            grid_width,
                            int(
                                np.floor(
                                    (
                                        structure.position[0]
                                        + structure.width
                                        - 0.25 * cell_size
                                    )
                                    / cell_size
                                )
                            ),
                        )
                        inner_max_i = min(
                            grid_height,
                            int(
                                np.floor(
                                    (
                                        structure.position[1]
                                        + structure.height
                                        - 0.25 * cell_size
                                    )
                                    / cell_size
                                )
                            ),
                        )
                        # Fast fill interior cells (fully covered, no need for sampling)
                        if inner_max_i > inner_min_i and inner_max_j > inner_min_j:
                            if is_custom_material:
                                # Evaluate CustomMaterial at each interior point
                                for i in range(inner_min_i, inner_max_i):
                                    for j in range(inner_min_j, inner_max_j):
                                        x, y = x_centers[j], y_centers[i]
                                        perm, permb, cond = (
                                            self._get_material_properties_safe(
                                                structure.material, x, y
                                            )
                                        )
                                        permittivity[i, j] = perm
                                        permeability[i, j] = permb
                                        conductivity[i, j] = cond
                            else:
                                permittivity[
                                    inner_min_i:inner_max_i, inner_min_j:inner_max_j
                                ] = mat_perm
                                permeability[
                                    inner_min_i:inner_max_i, inner_min_j:inner_max_j
                                ] = mat_permb
                                conductivity[
                                    inner_min_i:inner_max_i, inner_min_j:inner_max_j
                                ] = mat_cond
                        # Calculate boundary region cells (those that need super-sampling)
                        # This is more efficient than checking each cell individually
                        boundary_mask = np.zeros(
                            (rect_max_i - rect_min_i, rect_max_j - rect_min_j),
                            dtype=bool,
                        )
                        # Top and bottom boundaries
                        if rect_min_i < inner_min_i:
                            boundary_mask[: inner_min_i - rect_min_i, :] = True
                        if inner_max_i < rect_max_i:
                            boundary_mask[inner_max_i - rect_min_i :, :] = True
                        # Left and right boundaries
                        if rect_min_j < inner_min_j:
                            boundary_mask[:, : inner_min_j - rect_min_j] = True
                        if inner_max_j < rect_max_j:
                            boundary_mask[:, inner_max_j - rect_min_j :] = True
                        # Process boundary cells with super-sampling
                        boundary_indices = np.where(boundary_mask)
                        for idx in range(len(boundary_indices[0])):
                            i_rel, j_rel = (
                                boundary_indices[0][idx],
                                boundary_indices[1][idx],
                            )
                            i, j = i_rel + rect_min_i, j_rel + rect_min_j
                            # Cell center
                            center_x = x_centers[j]
                            center_y = y_centers[i]
                            # Count samples inside rectangle
                            samples_inside = 0
                            for k in range(num_samples):
                                x_sample = center_x + dx[k]
                                y_sample = center_y + dy[k]
                                if (
                                    structure.position[0]
                                    <= x_sample
                                    < structure.position[0] + structure.width
                                    and structure.position[1]
                                    <= y_sample
                                    < structure.position[1] + structure.height
                                ):
                                    samples_inside += 1
                            if samples_inside > 0:
                                # Calculate blend factor
                                blend_factor = samples_inside / num_samples
                                # Update material properties
                                if is_custom_material:
                                    # Evaluate CustomMaterial at cell center for boundary blending
                                    x, y = x_centers[j], y_centers[i]
                                    mat_perm, mat_permb, mat_cond = (
                                        self._get_material_properties_safe(
                                            structure.material, x, y
                                        )
                                    )
                                permittivity[i, j] = (
                                    permittivity[i, j] * (1 - blend_factor)
                                    + mat_perm * blend_factor
                                )
                                permeability[i, j] = (
                                    permeability[i, j] * (1 - blend_factor)
                                    + mat_permb * blend_factor
                                )
                                conductivity[i, j] = (
                                    conductivity[i, j] * (1 - blend_factor)
                                    + mat_cond * blend_factor
                                )

                    elif hasattr(structure, "radius"):  # Circle
                        # FAST PATH: Circle
                        # Get circle parameters
                        if len(structure.position) == 3:
                            center_x, center_y, _ = structure.position  # 3D position
                        else:
                            center_x, center_y = structure.position  # 2D position
                        radius = structure.radius
                        # Create local coordinate arrays for the bounding box region
                        j_indices = np.arange(min_j, max_j)
                        i_indices = np.arange(min_i, max_i)
                        local_x = x_centers[j_indices]
                        local_y = y_centers[i_indices]
                        # Create a grid of coordinates
                        X, Y = np.meshgrid(local_x, local_y)
                        # Calculate distances from center
                        distances = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
                        # Find cells fully inside circle (all sample points inside)
                        fully_inside = (
                            distances + 0.3536 * cell_size <= radius
                        )  # sqrt(2)/4 ≈ 0.3536 for diagonal
                        # Find cells potentially on the boundary (need super-sampling)
                        boundary = (
                            distances - 0.3536 * cell_size <= radius
                        ) & ~fully_inside
                        # Fast update for fully inside cells
                        local_i, local_j = np.where(fully_inside)
                        global_i, global_j = local_i + min_i, local_j + min_j
                        if len(global_i) > 0:
                            permittivity[global_i, global_j] = mat_perm
                            permeability[global_i, global_j] = mat_permb
                            conductivity[global_i, global_j] = mat_cond
                        # Super-sample for boundary cells
                        boundary_i, boundary_j = np.where(boundary)
                        for idx in range(len(boundary_i)):
                            i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
                            # Cell center
                            center_x_cell = x_centers[j]
                            center_y_cell = y_centers[i]
                            # Count samples inside circle
                            samples_inside = 0
                            for k in range(num_samples):
                                x_sample = center_x_cell + dx[k]
                                y_sample = center_y_cell + dy[k]
                                if (
                                    np.hypot(x_sample - center_x, y_sample - center_y)
                                    <= radius
                                ):
                                    samples_inside += 1
                            if samples_inside > 0:
                                # Calculate blend factor
                                blend_factor = samples_inside / num_samples
                                # Update material properties
                                permittivity[i, j] = (
                                    permittivity[i, j] * (1 - blend_factor)
                                    + mat_perm * blend_factor
                                )
                                permeability[i, j] = (
                                    permeability[i, j] * (1 - blend_factor)
                                    + mat_permb * blend_factor
                                )
                                conductivity[i, j] = (
                                    conductivity[i, j] * (1 - blend_factor)
                                    + mat_cond * blend_factor
                                )

                    elif hasattr(structure, "inner_radius") and hasattr(
                        structure, "outer_radius"
                    ):  # Ring
                        # FAST PATH: Ring
                        # Get ring parameters
                        if len(structure.position) == 3:
                            center_x, center_y, _ = structure.position  # 3D position
                        else:
                            center_x, center_y = structure.position  # 2D position
                        inner_radius = structure.inner_radius
                        outer_radius = structure.outer_radius
                        # Create local coordinate arrays for the bounding box region
                        j_indices = np.arange(min_j, max_j)
                        i_indices = np.arange(min_i, max_i)
                        local_x = x_centers[j_indices]
                        local_y = y_centers[i_indices]
                        # Create a grid of coordinates
                        X, Y = np.meshgrid(local_x, local_y)
                        # Calculate distances from center
                        distances = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
                        # Find cells fully inside ring (all sample points inside)
                        fully_inside = (
                            distances - 0.3536 * cell_size >= inner_radius
                        ) & (distances + 0.3536 * cell_size <= outer_radius)
                        # Find cells potentially on the boundary (need super-sampling)
                        inner_boundary = (
                            distances - 0.3536 * cell_size <= inner_radius
                        ) & (distances + 0.3536 * cell_size >= inner_radius)
                        outer_boundary = (
                            distances - 0.3536 * cell_size <= outer_radius
                        ) & (distances + 0.3536 * cell_size >= outer_radius)
                        boundary = inner_boundary | outer_boundary
                        # Fast update for fully inside cells
                        local_i, local_j = np.where(fully_inside)
                        global_i, global_j = local_i + min_i, local_j + min_j
                        if len(global_i) > 0:
                            permittivity[global_i, global_j] = mat_perm
                            permeability[global_i, global_j] = mat_permb
                            conductivity[global_i, global_j] = mat_cond
                        # Super-sample for boundary cells
                        boundary_i, boundary_j = np.where(boundary)
                        for idx in range(len(boundary_i)):
                            i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
                            # Cell center
                            center_x_cell = x_centers[j]
                            center_y_cell = y_centers[i]
                            # Count samples inside ring
                            samples_inside = 0
                            for k in range(num_samples):
                                x_sample = center_x_cell + dx[k]
                                y_sample = center_y_cell + dy[k]
                                distance = np.hypot(
                                    x_sample - center_x, y_sample - center_y
                                )
                                if inner_radius <= distance <= outer_radius:
                                    samples_inside += 1
                            if samples_inside > 0:
                                # Calculate blend factor
                                blend_factor = samples_inside / num_samples
                                # Update material properties
                                permittivity[i, j] = (
                                    permittivity[i, j] * (1 - blend_factor)
                                    + mat_perm * blend_factor
                                )
                                permeability[i, j] = (
                                    permeability[i, j] * (1 - blend_factor)
                                    + mat_permb * blend_factor
                                )
                                conductivity[i, j] = (
                                    conductivity[i, j] * (1 - blend_factor)
                                    + mat_cond * blend_factor
                                )
                    else:
                        # GENERAL PATH: For polygons and complex shapes
                        # Select the appropriate containment function
                        if hasattr(structure, "point_in_polygon"):
                            contains_func = lambda x, y: structure.point_in_polygon(
                                x, y
                            )
                        else:
                            # Fallback method using material values
                            contains_func = lambda x, y: any(
                                val != def_val
                                for val, def_val in zip(
                                    self.design.get_material_value(x, y, z=0),
                                    [1.0, 1.0, 0.0],
                                )
                            )
                        # First, try to identify fully inside cells if possible to minimize super-sampling
                        if (
                            hasattr(structure, "vertices")
                            and len(getattr(structure, "vertices", [])) > 0
                        ):
                            # Sample a center grid of points in each cell to detect likely inside areas
                            # This is a heuristic to identify cells likely fully inside
                            inside_mask = np.zeros(
                                (max_i - min_i, max_j - min_j), dtype=bool
                            )
                            boundary_mask = np.zeros(
                                (max_i - min_i, max_j - min_j), dtype=bool
                            )
                            # Sample 5 points per cell (center and corners) to identify inside/boundary cells
                            sample_points = [
                                (0, 0),
                                (-0.4, -0.4),
                                (-0.4, 0.4),
                                (0.4, -0.4),
                                (0.4, 0.4),
                            ]
                            for i_rel in range(max_i - min_i):
                                for j_rel in range(max_j - min_j):
                                    i, j = i_rel + min_i, j_rel + min_j
                                    # Get cell center
                                    center_x = x_centers[j]
                                    center_y = y_centers[i]
                                    # Track points inside/outside
                                    points_inside = 0
                                    center_inside = False
                                    # Check center point first
                                    if contains_func(center_x, center_y):
                                        center_inside = True
                                        points_inside += 1
                                    # Check corner points
                                    for dx_pt, dy_pt in sample_points[1:]:
                                        x_pt = center_x + dx_pt * cell_size
                                        y_pt = center_y + dy_pt * cell_size
                                        if contains_func(x_pt, y_pt):
                                            points_inside += 1
                                    # If center is inside and all sample points are inside
                                    if center_inside and points_inside == len(
                                        sample_points
                                    ):
                                        inside_mask[i_rel, j_rel] = True
                                    # If some points are inside and some are outside
                                    elif points_inside > 0:
                                        boundary_mask[i_rel, j_rel] = True

                            # Fast update for fully inside cells
                            inside_i, inside_j = np.where(inside_mask)
                            for idx in range(len(inside_i)):
                                i, j = inside_i[idx] + min_i, inside_j[idx] + min_j
                                permittivity[i, j] = mat_perm
                                permeability[i, j] = mat_permb
                                conductivity[i, j] = mat_cond

                            # Super-sample for boundary cells
                            boundary_i, boundary_j = np.where(boundary_mask)
                            for idx in range(len(boundary_i)):
                                i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
                                # Cell center
                                center_x = x_centers[j]
                                center_y = y_centers[i]
                                # Count samples inside shape
                                samples_inside = 0
                                for k in range(num_samples):
                                    x_sample = center_x + dx[k]
                                    y_sample = center_y + dy[k]
                                    if contains_func(x_sample, y_sample):
                                        samples_inside += 1
                                if samples_inside > 0:
                                    # Calculate blend factor
                                    blend_factor = samples_inside / num_samples
                                    # Update material properties
                                    permittivity[i, j] = (
                                        permittivity[i, j] * (1 - blend_factor)
                                        + mat_perm * blend_factor
                                    )
                                    permeability[i, j] = (
                                        permeability[i, j] * (1 - blend_factor)
                                        + mat_permb * blend_factor
                                    )
                                    conductivity[i, j] = (
                                        conductivity[i, j] * (1 - blend_factor)
                                        + mat_cond * blend_factor
                                    )

                            # Check remaining cells not marked as inside or boundary
                            remaining_i, remaining_j = np.where(
                                ~inside_mask & ~boundary_mask
                            )
                            for idx in range(len(remaining_i)):
                                i, j = (
                                    remaining_i[idx] + min_i,
                                    remaining_j[idx] + min_j,
                                )
                                # Cell center
                                center_x = x_centers[j]
                                center_y = y_centers[i]
                                # Super-sample
                                samples_inside = 0
                                for k in range(num_samples):
                                    x_sample = center_x + dx[k]
                                    y_sample = center_y + dy[k]
                                    if contains_func(x_sample, y_sample):
                                        samples_inside += 1
                                if samples_inside > 0:
                                    # Calculate blend factor
                                    blend_factor = samples_inside / num_samples
                                    # Update material properties
                                    permittivity[i, j] = (
                                        permittivity[i, j] * (1 - blend_factor)
                                        + mat_perm * blend_factor
                                    )
                                    permeability[i, j] = (
                                        permeability[i, j] * (1 - blend_factor)
                                        + mat_permb * blend_factor
                                    )
                                    conductivity[i, j] = (
                                        conductivity[i, j] * (1 - blend_factor)
                                        + mat_cond * blend_factor
                                    )
                        else:
                            # Direct super-sampling for all cells in bounding box
                            for i in range(min_i, max_i):
                                for j in range(min_j, max_j):
                                    # Cell center
                                    center_x = x_centers[j]
                                    center_y = y_centers[i]
                                    # Super-sample
                                    samples_inside = 0
                                    for k in range(num_samples):
                                        x_sample = center_x + dx[k]
                                        y_sample = center_y + dy[k]
                                        if contains_func(x_sample, y_sample):
                                            samples_inside += 1

                                    if samples_inside > 0:
                                        # Calculate blend factor
                                        blend_factor = samples_inside / num_samples
                                        # Update material properties
                                        permittivity[i, j] = (
                                            permittivity[i, j] * (1 - blend_factor)
                                            + mat_perm * blend_factor
                                        )
                                        permeability[i, j] = (
                                            permeability[i, j] * (1 - blend_factor)
                                            + mat_permb * blend_factor
                                        )
                                        conductivity[i, j] = (
                                            conductivity[i, j] * (1 - blend_factor)
                                            + mat_cond * blend_factor
                                        )

                except (AttributeError, TypeError) as e:
                    print(
                        f"Warning: Structure {type(structure)} doesn't have proper bounding box: {e}"
                    )

                progress.update(task, advance=1)

        # Assign final arrays to class instance
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity

    def show(self, field: str = "permittivity"):
        """Display the rasterized grid with properly scaled SI units."""
        if field == "permittivity":
            grid = self.permittivity
        elif field == "permeability":
            grid = self.permeability
        elif field == "conductivity":
            grid = self.conductivity
        if grid is not None:
            # Determine appropriate SI unit and scale
            max_dim = max(self.design.width, self.design.height)
            if max_dim >= 1e-3:
                scale, unit = 1e3, "mm"
            elif max_dim >= 1e-6:
                scale, unit = 1e6, "µm"
            elif max_dim >= 1e-9:
                scale, unit = 1e9, "nm"
            else:
                scale, unit = 1, "m"
            # Calculate figure size based on grid dimensions
            grid_height, grid_width = grid.shape
            aspect_ratio = grid_width / grid_height
            base_size = 2.5  # Base size for the smaller dimension
            if aspect_ratio > 1:
                figsize = (base_size * aspect_ratio, base_size)
            else:
                figsize = (base_size, base_size / aspect_ratio)
            # Make the actual figure
            plt.figure(figsize=figsize)
            plt.imshow(
                grid,
                origin="lower",
                cmap="Grays",
                extent=(0, self.design.width, 0, self.design.height),
            )
            plt.colorbar(label=field)
            plt.title("Rasterized Design Grid")
            plt.xlabel(f"X ({unit})")
            plt.ylabel(f"Y ({unit})")
            # Update tick labels with scaled values
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
            plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
            plt.tight_layout()
            plt.show()
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

        # Calculate 3D grid dimensions
        width, height, depth = self.design.width, self.design.height, self.design.depth
        grid_width = int(width / self.resolution_xy)
        grid_height = int(height / self.resolution_xy)
        grid_depth = int(depth / self.resolution_z) if depth > 0 else 1

        # Initialize 3D material grids
        self.permittivity = np.zeros((grid_depth, grid_height, grid_width))
        self.permeability = np.zeros((grid_depth, grid_height, grid_width))
        self.conductivity = np.zeros((grid_depth, grid_height, grid_width))

        # Rasterize the design
        self.__rasterize_3d__()

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

        # Create grid of cell centers for xy plane
        x_centers = np.linspace(
            0.5 * cell_size_xy, width - 0.5 * cell_size_xy, grid_width
        )
        y_centers = np.linspace(
            0.5 * cell_size_xy, height - 0.5 * cell_size_xy, grid_height
        )
        z_centers = (
            np.linspace(0.5 * cell_size_z, depth - 0.5 * cell_size_z, grid_depth)
            if depth > 0
            else [0]
        )

        # Precompute offsets for 3D super-sampling (3x3x3 = 27 samples)
        offsets_xy = np.array([-0.25, 0, 0.25]) * cell_size_xy
        offsets_z = np.array([-0.25, 0, 0.25]) * cell_size_z if depth > 0 else [0]
        dx, dy, dz = np.meshgrid(offsets_xy, offsets_xy, offsets_z)
        dx, dy, dz = dx.flatten(), dy.flatten(), dz.flatten()
        num_samples = len(dx)

        # Estimate dt for PML calculations
        c = 3e8  # Speed of light
        dt_estimate = 0.5 * self.resolution / (c * np.sqrt(2))

        # Initialize material grids with vacuum properties
        permittivity = np.ones((grid_depth, grid_height, grid_width))
        permeability = np.ones((grid_depth, grid_height, grid_width))
        conductivity = np.zeros((grid_depth, grid_height, grid_width))

        # Start with the background (first structure)
        if len(self.design.structures) > 0:
            background = self.design.structures[0]
            if hasattr(background, "material") and background.material is not None:
                permittivity.fill(background.material.permittivity)
                permeability.fill(background.material.permeability)
                conductivity.fill(background.material.conductivity)

        # Process structures layer by layer
        with create_rich_progress() as progress:
            task = progress.add_task(
                "Rasterizing 3D structures...", total=len(self.design.structures)
            )
            progress.update(task, advance=1)  # Skip background

            for idx in range(1, len(self.design.structures)):
                structure = self.design.structures[idx]

                # Skip PML visualization structures
                if hasattr(structure, "is_pml") and structure.is_pml:
                    progress.update(task, advance=1)
                    continue
                # Skip structures without material
                if not hasattr(structure, "material") or structure.material is None:
                    progress.update(task, advance=1)
                    continue

                # Cache material properties
                mat_perm, mat_permb, mat_cond = self._get_material_properties_safe(
                    structure.material
                )

                try:
                    # Get 3D bounding box
                    bbox = structure.get_bounding_box()
                    if bbox is None:
                        raise AttributeError("Bounding box is None")

                    if len(bbox) == 6:  # 3D bounding box
                        min_x, min_y, min_z, max_x, max_y, max_z = bbox
                    else:  # 2D bounding box - extend to 3D
                        min_x, min_y, max_x, max_y = bbox
                        min_z, max_z = 0, 0

                    # Convert to grid indices
                    min_i = max(0, int(min_y / cell_size_xy) - 1)
                    min_j = max(0, int(min_x / cell_size_xy) - 1)
                    min_k = max(0, int(min_z / cell_size_z) - 1) if depth > 0 else 0
                    max_i = min(grid_height, int(np.ceil(max_y / cell_size_xy)) + 1)
                    max_j = min(grid_width, int(np.ceil(max_x / cell_size_xy)) + 1)
                    max_k = (
                        min(grid_depth, int(np.ceil(max_z / cell_size_z)) + 1)
                        if depth > 0
                        else 1
                    )

                    # Skip if bounding box is outside grid
                    if (
                        min_i >= grid_height
                        or min_j >= grid_width
                        or min_k >= grid_depth
                        or max_i <= 0
                        or max_j <= 0
                        or max_k <= 0
                    ):
                        progress.update(task, advance=1)
                        continue

                    # Process each z-layer
                    for k in range(min_k, max_k):
                        z_center = z_centers[k]

                        # Process xy grid for this z-layer
                        for i in range(min_i, max_i):
                            for j in range(min_j, max_j):
                                x_center = x_centers[j]
                                y_center = y_centers[i]

                                # Super-sample this cell
                                samples_inside = 0
                                for sample_idx in range(num_samples):
                                    x_sample = x_center + dx[sample_idx]
                                    y_sample = y_center + dy[sample_idx]
                                    z_sample = z_center + dz[sample_idx]

                                    # Check if sample point is inside structure
                                    if hasattr(structure, "point_in_polygon"):
                                        # Use 3D-aware point-in-polygon
                                        if structure.point_in_polygon(
                                            x_sample, y_sample, z_sample
                                        ):
                                            samples_inside += 1
                                    else:
                                        # Fallback: use design's material value method
                                        material_vals = self.design.get_material_value(
                                            x_sample, y_sample, z_sample
                                        )
                                        # Check if material is different from background
                                        if any(
                                            val != def_val
                                            for val, def_val in zip(
                                                material_vals, [1.0, 1.0, 0.0]
                                            )
                                        ):
                                            samples_inside += 1

                                if samples_inside > 0:
                                    # Calculate blend factor
                                    blend_factor = samples_inside / num_samples

                                    # Update material properties with blending
                                    permittivity[k, i, j] = (
                                        permittivity[k, i, j] * (1 - blend_factor)
                                        + mat_perm * blend_factor
                                    )
                                    permeability[k, i, j] = (
                                        permeability[k, i, j] * (1 - blend_factor)
                                        + mat_permb * blend_factor
                                    )
                                    conductivity[k, i, j] = (
                                        conductivity[k, i, j] * (1 - blend_factor)
                                        + mat_cond * blend_factor
                                    )

                except (AttributeError, TypeError) as e:
                    display_status(
                        f"Warning: Structure {type(structure)} processing failed: {e}",
                        "warning",
                    )

                progress.update(task, advance=1)

        # Process 3D PML boundaries
        self._process_3d_pml(
            permittivity,
            permeability,
            conductivity,
            x_centers,
            y_centers,
            z_centers,
            dt_estimate,
        )

        # Assign final arrays
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity

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
        if field == "permittivity":
            grid = self.permittivity
        elif field == "permeability":
            grid = self.permeability
        elif field == "conductivity":
            grid = self.conductivity
        else:
            raise ValueError(f"Unknown field: {field}")

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Create coordinate grids
        nz, ny, nx = grid.shape
        x = np.linspace(0, self.design.width, nx)
        y = np.linspace(0, self.design.height, ny)
        z = np.linspace(0, self.design.depth, nz)

        # Show slices with different spacing
        for k in range(0, nz, slice_spacing):
            X, Y = np.meshgrid(x, y)
            Z = np.full_like(X, z[k])
            colors = grid[k, :, :]

            # Plot surface
            surf = ax.plot_surface(
                X,
                Y,
                Z,
                facecolors=plt.cm.viridis(colors),
                alpha=alpha,
                linewidth=0,
                antialiased=True,
            )

        # Set labels and title
        max_dim = max(self.design.width, self.design.height, self.design.depth)
        if max_dim >= 1e-3:
            scale, unit = 1e3, "mm"
        elif max_dim >= 1e-6:
            scale, unit = 1e6, "µm"
        elif max_dim >= 1e-9:
            scale, unit = 1e9, "nm"
        else:
            scale, unit = 1, "m"

        ax.set_xlabel(f"X ({unit})")
        ax.set_ylabel(f"Y ({unit})")
        ax.set_zlabel(f"Z ({unit})")
        ax.set_title(f"3D {field.capitalize()} Distribution")

        # Scale the axes
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
        ax.zaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")

        plt.tight_layout()
        plt.show()

    def show(self, field="permittivity", z_index=None, z_position=None):
        """Display a 2D slice of the 3D mesh (backwards compatible interface)."""
        slice_data = self.get_2d_slice(z_index, z_position)
        grid = slice_data[field]

        if grid is not None:
            # Determine appropriate SI unit and scale
            max_dim = max(self.design.width, self.design.height)
            if max_dim >= 1e-3:
                scale, unit = 1e3, "mm"
            elif max_dim >= 1e-6:
                scale, unit = 1e6, "µm"
            elif max_dim >= 1e-9:
                scale, unit = 1e9, "nm"
            else:
                scale, unit = 1, "m"

            # Calculate figure size based on grid dimensions
            grid_height, grid_width = grid.shape
            aspect_ratio = grid_width / grid_height
            base_size = 2.5
            if aspect_ratio > 1:
                figsize = (base_size * aspect_ratio, base_size)
            else:
                figsize = (base_size, base_size / aspect_ratio)

            # Create the plot
            plt.figure(figsize=figsize)
            plt.imshow(
                grid,
                origin="lower",
                cmap="Grays",
                extent=(0, self.design.width, 0, self.design.height),
            )
            plt.colorbar(label=field)

            # Determine z-layer info
            z_idx = z_index if z_index is not None else self.shape[0] // 2
            z_pos = z_idx * self.resolution_z
            plt.title(f"3D {field.capitalize()} at z = {z_pos*scale:.2f} {unit}")

            plt.xlabel(f"X ({unit})")
            plt.ylabel(f"Y ({unit})")

            # Update tick labels with scaled values
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
            plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")

            plt.tight_layout()
            plt.show()
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


def mesh_2d(design, resolution):
    """Create a 2D mesh (backwards compatible function)."""
    return RegularGrid(design, resolution)


def mesh_3d(design, resolution_xy, resolution_z=None):
    """Create a 3D mesh with optional separate z-resolution."""
    return RegularGrid3D(design, resolution_xy, resolution_z)


# Export classes and functions
__all__ = [
    "BaseMeshGrid",
    "RegularGrid",
    "RegularGrid3D",
    "create_mesh",
    "mesh_2d",
    "mesh_3d",
]
