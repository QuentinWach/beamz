import os
import time

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box
from shapely.prepared import prep

try:
    from matplotlib.path import Path as MplPath
except Exception:  # pragma: no cover - matplotlib is expected but keep fallback safe
    MplPath = None

from beamz.design.structures import Rectangle
from beamz.visual.helpers import (
    create_rich_progress,
    display_status,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class MaterialGrids:
    """Bundles the 8 material property arrays with bulk operations."""

    NAMES = (
        "permittivity",
        "permeability",
        "conductivity",
        "k",
        "rho",
        "cp",
        "dn_dT",
        "T0",
    )
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
        """Painters algorithm: rasterize design into a grid using super-sampling.

        Iterates through structures in order (background first, foreground last).
        Uses bounding-box clipping and fast paths for axis-aligned rectangles,
        circles, and rings. Boundary cells get configurable super-sampling
        for anti-aliasing.
        """
        width, height = self.design.width, self.design.height
        grid_width, grid_height = int(width / self.resolution), int(
            height / self.resolution
        )
        cell_size = self.resolution

        # Create grid of cell centers
        x_centers = np.linspace(0.5 * cell_size, width - 0.5 * cell_size, grid_width)
        y_centers = np.linspace(0.5 * cell_size, height - 0.5 * cell_size, grid_height)

        # Precompute subpixel offsets based on AA mode
        sample_dx, sample_dy = self._build_supersample_offsets_xy(cell_size)
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
                props = (
                    None
                    if is_custom_material
                    else self._get_all_material_props(structure.material)
                )

                try:
                    bbox_indices = self._get_bbox_indices(
                        structure, grid_height, grid_width, cell_size
                    )
                    if bbox_indices is None:
                        progress.update(task, advance=1)
                        continue
                    min_i, min_j, max_i, max_j = bbox_indices

                    # Dispatch to shape-specific fast path
                    if isinstance(structure, Rectangle) and self._is_axis_aligned(
                        structure
                    ):
                        self._rasterize_rectangle(
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
                    elif hasattr(structure, "radius") and not hasattr(
                        structure, "inner_radius"
                    ):
                        self._rasterize_circle(
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
                    elif hasattr(structure, "inner_radius") and hasattr(
                        structure, "outer_radius"
                    ):
                        self._rasterize_ring(
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
                    else:
                        self._rasterize_polygon(
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
        local_dx, local_dy = sample_dx, sample_dy
        if cell_i is not None and cell_j is not None:
            local_dx, local_dy = self._scramble_offsets_xy_for_cell(
                sample_dx=sample_dx,
                sample_dy=sample_dy,
                cell_size=(self.resolution if cell_size is None else cell_size),
                cell_i=cell_i,
                cell_j=cell_j,
                cell_k=cell_k,
            )

        count = 0
        for k in range(num_samples):
            if contains_fn(cx + local_dx[k], cy + local_dy[k]):
                count += 1
        return count

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
        rect_min_j = max(0, int(structure.position[0] / cell_size))
        rect_min_i = max(0, int(structure.position[1] / cell_size))
        rect_max_j = min(
            grid_width,
            int(np.ceil((structure.position[0] + structure.width) / cell_size)),
        )
        rect_max_i = min(
            grid_height,
            int(np.ceil((structure.position[1] + structure.height) / cell_size)),
        )

        sx, sy = structure.position[0], structure.position[1]
        sw, sh = structure.width, structure.height
        j_idx = np.arange(rect_min_j, rect_max_j, dtype=float)
        i_idx = np.arange(rect_min_i, rect_max_i, dtype=float)
        cell_x0 = j_idx * cell_size
        cell_x1 = cell_x0 + cell_size
        cell_y0 = i_idx * cell_size
        cell_y1 = cell_y0 + cell_size

        overlap_x = np.clip(
            np.minimum(cell_x1, sx + sw) - np.maximum(cell_x0, sx), 0.0, cell_size
        )
        overlap_y = np.clip(
            np.minimum(cell_y1, sy + sh) - np.maximum(cell_y0, sy), 0.0, cell_size
        )
        coverage = np.outer(overlap_y, overlap_x) / float(cell_size * cell_size)

        local_i, local_j = np.where(coverage >= 1.0 - 1e-15)
        for idx in range(len(local_i)):
            i = local_i[idx] + rect_min_i
            j = local_j[idx] + rect_min_j
            rect_props = props
            if is_custom_material:
                rect_props = self._get_all_material_props(
                    structure.material, x_centers[j], y_centers[i]
                )
            grids.set_at((i, j), rect_props)

        boundary_i, boundary_j = np.where((coverage > 0.0) & (coverage < 1.0 - 1e-15))
        for idx in range(len(boundary_i)):
            i = boundary_i[idx] + rect_min_i
            j = boundary_j[idx] + rect_min_j
            rect_props = props
            if is_custom_material:
                rect_props = self._get_all_material_props(
                    structure.material, x_centers[j], y_centers[i]
                )
            grids.blend_at(
                (i, j), rect_props, float(coverage[boundary_i[idx], boundary_j[idx]])
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
                cx,
                cy,
                sample_dx,
                sample_dy,
                num_samples,
                lambda x, y: np.hypot(x - center_x, y - center_y) <= radius,
                cell_i=i,
                cell_j=j,
                cell_size=cell_size,
            )
            if samples_inside > 0:
                grids.blend_at((i, j), props, samples_inside / num_samples)

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
        center_x, center_y = structure.position[0], structure.position[1]
        inner_radius = structure.inner_radius
        outer_radius = structure.outer_radius

        j_indices = np.arange(min_j, max_j)
        i_indices = np.arange(min_i, max_i)
        X, Y = np.meshgrid(x_centers[j_indices], y_centers[i_indices])
        distances = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        diag = 0.3536 * cell_size
        fully_inside = (distances - diag >= inner_radius) & (
            distances + diag <= outer_radius
        )
        inner_boundary = (distances - diag <= inner_radius) & (
            distances + diag >= inner_radius
        )
        outer_boundary = (distances - diag <= outer_radius) & (
            distances + diag >= outer_radius
        )
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
                cx,
                cy,
                sample_dx,
                sample_dy,
                num_samples,
                lambda x, y: inner_radius
                <= np.hypot(x - center_x, y - center_y)
                <= outer_radius,
                cell_i=i,
                cell_j=j,
                cell_size=cell_size,
            )
            if samples_inside > 0:
                grids.blend_at((i, j), props, samples_inside / num_samples)

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
        polygon = self._structure_polygon_2d(structure)
        if polygon is not None:
            prepared_polygon = prep(polygon)
            cell_area = float(cell_size * cell_size)
            for i in range(min_i, max_i):
                cell_y0 = i * cell_size
                cell_y1 = cell_y0 + cell_size
                cy = y_centers[i]
                for j in range(min_j, max_j):
                    cell_x0 = j * cell_size
                    cell_x1 = cell_x0 + cell_size
                    cx = x_centers[j]
                    cell = shapely_box(cell_x0, cell_y0, cell_x1, cell_y1)
                    if prepared_polygon.contains(cell):
                        cell_props = props
                        if is_custom_material:
                            cell_props = self._get_all_material_props(
                                structure.material, cx, cy
                            )
                        grids.set_at((i, j), cell_props)
                        continue
                    if not prepared_polygon.intersects(cell):
                        continue

                    blend_factor = polygon.intersection(cell).area / cell_area
                    if blend_factor <= 0.0:
                        continue
                    cell_props = props
                    if is_custom_material:
                        cell_props = self._get_all_material_props(
                            structure.material, cx, cy
                        )
                    grids.blend_at((i, j), cell_props, float(blend_factor))
            return

        if hasattr(structure, "point_in_polygon"):
            contains_func = lambda x, y: structure.point_in_polygon(x, y)
        else:
            contains_func = lambda x, y: any(
                val != def_val
                for val, def_val in zip(
                    self.design.get_material_value(x, y, z=0), [1.0, 1.0, 0.0]
                )
            )

        if (
            hasattr(structure, "vertices")
            and len(getattr(structure, "vertices", [])) > 0
        ):
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
                        if contains_func(
                            cx + dx_pt * cell_size, cy + dy_pt * cell_size
                        ):
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
                        cx,
                        cy,
                        sample_dx,
                        sample_dy,
                        num_samples,
                        contains_func,
                        cell_i=i,
                        cell_j=j,
                        cell_size=cell_size,
                    )
                    if samples_inside > 0:
                        grids.blend_at((i, j), props, samples_inside / num_samples)
        else:
            # Direct super-sampling for all cells in bounding box
            for i in range(min_i, max_i):
                for j in range(min_j, max_j):
                    cx, cy = x_centers[j], y_centers[i]
                    samples_inside = self._supersample_cell(
                        cx,
                        cy,
                        sample_dx,
                        sample_dy,
                        num_samples,
                        contains_func,
                        cell_i=i,
                        cell_j=j,
                        cell_size=cell_size,
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
        """3D rasterization using vectorized shape fills and bounded fallbacks."""
        width, height, depth = self.design.width, self.design.height, self.design.depth
        grid_width = int(width / self.resolution_xy)
        grid_height = int(height / self.resolution_xy)
        grid_depth = int(depth / self.resolution_z) if depth > 0 else 1

        t_total_start = time.perf_counter()
        cell_size_xy = self.resolution_xy
        cell_size_z = self.resolution_z

        # Create grid of cell centers
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

        # Estimate dt for PML calculations
        c = 3e8
        dt_estimate = 0.5 * self.resolution / (c * np.sqrt(2))

        grids = MaterialGrids((grid_depth, grid_height, grid_width))
        t_setup_end = time.perf_counter()

        # Fill background
        if len(self.design.structures) > 0:
            background = self.design.structures[0]
            if hasattr(background, "material") and background.material is not None:
                grids.fill_all(self._get_all_material_props(background.material))

        timing_enabled = _env_bool("BEAMZ_RASTER_TIMING", True)
        voxel_count = int(grid_width) * int(grid_height) * int(grid_depth)
        fast_min_voxels = int(
            float(os.getenv("BEAMZ_RASTER_FAST_MIN_VOXELS", 1_000_000))
        )
        fast_env = os.getenv("BEAMZ_RASTER_FAST_3D")
        if fast_env is None:
            prefer_fast = voxel_count >= fast_min_voxels
        else:
            prefer_fast = _env_bool("BEAMZ_RASTER_FAST_3D", True)
        fast_rect_count = 0
        fast_poly_count = 0
        fallback_count = 0

        t_struct_start = time.perf_counter()
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
                    bbox = self._get_bbox_indices_3d(
                        structure,
                        grid_height=grid_height,
                        grid_width=grid_width,
                        grid_depth=grid_depth,
                        cell_size_xy=cell_size_xy,
                        cell_size_z=cell_size_z,
                        margin_cells=1,
                    )
                    if bbox is None:
                        progress.update(task, advance=1)
                        continue
                    min_i, min_j, min_k, max_i, max_j, max_k = bbox

                    fast_done = False
                    if (
                        prefer_fast
                        and isinstance(structure, Rectangle)
                        and self._is_axis_aligned(structure)
                    ):
                        self._rasterize_rectangle_3d_fast(
                            structure=structure,
                            grids=grids,
                            props=props,
                            grid_height=grid_height,
                            grid_width=grid_width,
                            grid_depth=grid_depth,
                            cell_size_xy=cell_size_xy,
                            cell_size_z=cell_size_z,
                        )
                        fast_rect_count += 1
                        fast_done = True

                    if not fast_done and prefer_fast:
                        poly_done = self._rasterize_polygon_3d_fast(
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
                        if poly_done:
                            fast_poly_count += 1
                            fast_done = True

                    if not fast_done:
                        self._rasterize_structure_3d_fallback(
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
                        fallback_count += 1

                except (AttributeError, TypeError) as e:
                    display_status(
                        f"Warning: Structure {type(structure)} processing failed: {e}",
                        "warning",
                    )

                progress.update(task, advance=1)
        t_struct_end = time.perf_counter()

        # Process 3D PML boundaries
        t_pml_start = time.perf_counter()
        self._process_3d_pml(
            grids.permittivity,
            grids.permeability,
            grids.conductivity,
            x_centers,
            y_centers,
            z_centers,
            dt_estimate,
        )
        t_pml_end = time.perf_counter()

        grids.assign_to(self)
        t_total_end = time.perf_counter()

        if timing_enabled:
            display_status(
                (
                    "3D raster timing: "
                    f"setup={t_setup_end - t_total_start:.2f}s, "
                    f"structures={t_struct_end - t_struct_start:.2f}s, "
                    f"pml={t_pml_end - t_pml_start:.2f}s, "
                    f"total={t_total_end - t_total_start:.2f}s"
                ),
                "info",
            )
            display_status(
                (
                    "3D raster kernels: "
                    f"fast_enabled={prefer_fast}, "
                    f"fast_rect={fast_rect_count}, "
                    f"fast_poly={fast_poly_count}, "
                    f"fallback={fallback_count}"
                ),
                "info",
            )

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

    @staticmethod
    def _is_axis_aligned(structure):
        """Check if a rectangle is axis-aligned (not rotated)."""
        return (
            hasattr(structure, "vertices")
            and structure.vertices
            and structure.vertices[0][0] == structure.position[0]
            and structure.vertices[0][1] == structure.position[1]
        )

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
        bbox = structure.get_bounding_box()
        if bbox is None:
            return None
        if len(bbox) == 6:
            min_x, min_y, min_z, max_x, max_y, max_z = bbox
        elif len(bbox) == 4:
            min_x, min_y, max_x, max_y = bbox
            min_z = getattr(structure, "z", 0.0)
            max_z = min_z + float(getattr(structure, "depth", 0.0))
            if max_z <= min_z:
                max_z = min_z + cell_size_z
        else:
            return None

        margin = int(max(0, margin_cells))
        min_i = max(0, int(min_y / cell_size_xy) - margin)
        min_j = max(0, int(min_x / cell_size_xy) - margin)
        min_k = max(0, int(min_z / cell_size_z) - margin) if grid_depth > 1 else 0
        max_i = min(grid_height, int(np.ceil(max_y / cell_size_xy)) + margin)
        max_j = min(grid_width, int(np.ceil(max_x / cell_size_xy)) + margin)
        max_k = (
            min(grid_depth, int(np.ceil(max_z / cell_size_z)) + margin)
            if grid_depth > 1
            else 1
        )

        if min_i >= max_i or min_j >= max_j or min_k >= max_k:
            return None
        return min_i, min_j, min_k, max_i, max_j, max_k

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
        x0, y0, z0 = structure.position
        x1 = x0 + float(structure.width)
        y1 = y0 + float(structure.height)
        z1 = z0 + float(getattr(structure, "depth", 0.0))
        if z1 <= z0:
            z1 = z0 + cell_size_z

        i0 = max(0, int(np.floor(y0 / cell_size_xy)))
        j0 = max(0, int(np.floor(x0 / cell_size_xy)))
        k0 = max(0, int(np.floor(z0 / cell_size_z))) if grid_depth > 1 else 0
        i1 = min(grid_height, int(np.ceil(y1 / cell_size_xy)))
        j1 = min(grid_width, int(np.ceil(x1 / cell_size_xy)))
        k1 = min(grid_depth, int(np.ceil(z1 / cell_size_z))) if grid_depth > 1 else 1
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            return

        # Compute exact axis-aligned overlap fractions per voxel to preserve
        # anti-aliased boundaries without expensive per-sample point checks.
        x_edges0 = np.arange(j0, j1, dtype=float) * cell_size_xy
        x_edges1 = x_edges0 + cell_size_xy
        y_edges0 = np.arange(i0, i1, dtype=float) * cell_size_xy
        y_edges1 = y_edges0 + cell_size_xy
        z_edges0 = np.arange(k0, k1, dtype=float) * cell_size_z
        z_edges1 = z_edges0 + cell_size_z

        fx = (
            np.clip(
                np.minimum(x_edges1, x1) - np.maximum(x_edges0, x0), 0.0, cell_size_xy
            )
            / cell_size_xy
        )
        fy = (
            np.clip(
                np.minimum(y_edges1, y1) - np.maximum(y_edges0, y0), 0.0, cell_size_xy
            )
            / cell_size_xy
        )
        fz = (
            np.clip(
                np.minimum(z_edges1, z1) - np.maximum(z_edges0, z0), 0.0, cell_size_z
            )
            / cell_size_z
        )

        frac = fz[:, None, None] * fy[None, :, None] * fx[None, None, :]
        if not np.any(frac > 0.0):
            return

        full_mask = frac >= (1.0 - 1e-12)
        blend_mask = (frac > 0.0) & ~full_mask

        if np.any(full_mask):
            for name, val in zip(MaterialGrids.NAMES, props):
                arr = getattr(grids, name)[k0:k1, i0:i1, j0:j1]
                arr[full_mask] = val

        if np.any(blend_mask):
            for name, val in zip(MaterialGrids.NAMES, props):
                arr = getattr(grids, name)[k0:k1, i0:i1, j0:j1]
                f = frac[blend_mask]
                arr[blend_mask] = arr[blend_mask] * (1.0 - f) + val * f

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
        if MplPath is None:
            return False
        if not hasattr(structure, "vertices") or not structure.vertices:
            return False
        if hasattr(structure, "radius"):
            # Circle/ring currently uses fallback path.
            return False
        if float(getattr(structure, "depth", 0.0)) <= 0.0:
            return False
        if min_i >= max_i or min_j >= max_j or min_k >= max_k:
            return False

        verts3 = np.asarray(structure.vertices, dtype=float)
        if verts3.ndim != 2 or verts3.shape[0] < 3:
            return False
        if verts3.shape[1] >= 3 and np.ptp(verts3[:, 2]) > 1e-12:
            # Non-planar polygons should use full fallback.
            return False

        verts = np.asarray([(v[0], v[1]) for v in structure.vertices], dtype=float)
        if verts.ndim != 2 or verts.shape[0] < 3:
            return False

        z0 = float(getattr(structure, "z", np.min(verts3[:, 2])))
        z1 = z0 + float(getattr(structure, "depth", 0.0))
        if z1 <= z0:
            z1 = float(np.max(verts3[:, 2]))
        if z1 <= z0:
            z1 = z0 + float(cell_size_z)

        x_local = x_centers[min_j:max_j]
        y_local = y_centers[min_i:max_i]
        xx, yy = np.meshgrid(x_local, y_local)
        outer_path = MplPath(verts)
        interiors = getattr(structure, "interiors", None) or []

        hole_paths = []
        for hole in interiors:
            iv3 = np.asarray(hole, dtype=float)
            if iv3.ndim == 2 and iv3.shape[1] >= 3 and np.ptp(iv3[:, 2]) > 1e-12:
                return False
            iv = np.asarray([(v[0], v[1]) for v in hole], dtype=float)
            if iv.ndim == 2 and iv.shape[0] >= 3:
                hole_paths.append(MplPath(iv))

        sample_dx, sample_dy = self._build_supersample_offsets_xy(cell_size_xy)
        n_samples_xy = float(sample_dx.size)
        inside_count = np.zeros(xx.shape, dtype=float)

        shift_x_map = None
        shift_y_map = None
        rot_map = None
        if self.aa_mode == "stratified_jitter":
            shift_x_map = np.empty(xx.shape, dtype=float)
            shift_y_map = np.empty(xx.shape, dtype=float)
            rot_map = np.empty(xx.shape, dtype=np.uint8)
            for i_rel in range(xx.shape[0]):
                cell_i = min_i + i_rel
                for j_rel in range(xx.shape[1]):
                    cell_j = min_j + j_rel
                    sx, sy, rot90 = self._cell_scramble_params_xy(cell_i, cell_j, 0)
                    shift_x_map[i_rel, j_rel] = sx
                    shift_y_map[i_rel, j_rel] = sy
                    rot_map[i_rel, j_rel] = rot90

        for sidx in range(sample_dx.size):
            if self.aa_mode == "stratified_jitter":
                u0 = sample_dx[sidx] / float(cell_size_xy) + 0.5
                v0 = sample_dy[sidx] / float(cell_size_xy) + 0.5
                u_rot = np.where(
                    rot_map == 0,
                    u0,
                    np.where(
                        rot_map == 1, v0, np.where(rot_map == 2, 1.0 - u0, 1.0 - v0)
                    ),
                )
                v_rot = np.where(
                    rot_map == 0,
                    v0,
                    np.where(
                        rot_map == 1, 1.0 - u0, np.where(rot_map == 2, 1.0 - v0, u0)
                    ),
                )
                cell_dx = (np.mod(u_rot + shift_x_map, 1.0) - 0.5) * float(cell_size_xy)
                cell_dy = (np.mod(v_rot + shift_y_map, 1.0) - 0.5) * float(cell_size_xy)
                points = np.column_stack(
                    ((xx + cell_dx).ravel(), (yy + cell_dy).ravel())
                )
            else:
                points = np.column_stack(
                    ((xx + sample_dx[sidx]).ravel(), (yy + sample_dy[sidx]).ravel())
                )
            inside = outer_path.contains_points(points, radius=1e-15).reshape(xx.shape)
            for hole_path in hole_paths:
                inside &= ~hole_path.contains_points(points, radius=1e-15).reshape(
                    xx.shape
                )
            inside_count += inside.astype(float)
        frac_xy = inside_count / n_samples_xy
        if not np.any(frac_xy > 0.0):
            return True

        z_edges0 = np.arange(min_k, max_k, dtype=float) * float(cell_size_z)
        z_edges1 = z_edges0 + float(cell_size_z)
        frac_z = np.clip(
            np.minimum(z_edges1, z1) - np.maximum(z_edges0, z0),
            0.0,
            float(cell_size_z),
        ) / float(cell_size_z)
        frac = frac_z[:, None, None] * frac_xy[None, :, :]
        if not np.any(frac > 0.0):
            return True

        full_mask = frac >= (1.0 - 1e-12)
        blend_mask = (frac > 0.0) & ~full_mask

        if np.any(full_mask):
            for name, val in zip(MaterialGrids.NAMES, props):
                arr = getattr(grids, name)[min_k:max_k, min_i:max_i, min_j:max_j]
                arr[full_mask] = val

        if np.any(blend_mask):
            for name, val in zip(MaterialGrids.NAMES, props):
                arr = getattr(grids, name)[min_k:max_k, min_i:max_i, min_j:max_j]
                f = frac[blend_mask]
                arr[blend_mask] = arr[blend_mask] * (1.0 - f) + val * f

        return True

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
        if min_i >= max_i or min_j >= max_j or min_k >= max_k:
            return

        sample_dx, sample_dy = self._build_supersample_offsets_xy(cell_size_xy)
        offsets_z = self._build_supersample_offsets_z(
            cell_size_z=cell_size_z,
            depth_samples=(3 if len(z_centers) > 1 else 1),
        )
        num_samples = sample_dx.size * offsets_z.size

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
                y_center = y_centers[i]
                for j in range(min_j, max_j):
                    x_center = x_centers[j]
                    cell_dx, cell_dy = self._scramble_offsets_xy_for_cell(
                        sample_dx=sample_dx,
                        sample_dy=sample_dy,
                        cell_size=cell_size_xy,
                        cell_i=i,
                        cell_j=j,
                        cell_k=k,
                    )
                    inside = 0
                    for z_off in offsets_z:
                        for sidx in range(cell_dx.size):
                            if contains_fn(
                                x_center + cell_dx[sidx],
                                y_center + cell_dy[sidx],
                                z_center + z_off,
                            ):
                                inside += 1
                    if inside > 0:
                        grids.blend_at((k, i, j), props, inside / float(num_samples))

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
