import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


class MaterialGrids:
    """Bundles the core EM material property arrays with bulk operations."""

    NAMES = ("permittivity", "permeability", "conductivity")
    DEFAULTS = (1.0, 1.0, 0.0)

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
    """Base class for mesh grids with shared AA and material helpers."""

    _SUPPORTED_AA_MODES = ("legacy_grid", "stratified_jitter")
    _DEFAULT_AA_MODE = "legacy_grid"
    _DEFAULT_AA_SAMPLES = 64
    _DEFAULT_AA_SEED = 0

    def __init__(self, design, resolution):
        self.design = design
        self.resolution = resolution
        self._validate_inputs()

    def _validate_inputs(self):
        if self.resolution <= 0:
            raise ValueError("Resolution must be positive")
        if self.design is None:
            raise ValueError("Design cannot be None")

    def _configure_antialiasing(self, aa_mode=None, aa_samples=None, aa_seed=None):
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
        """Apply per-cell CP shift and 90-degree rotation to AA samples."""
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
            return sample_dx.ravel(), sample_dy.ravel()

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
        """Convert a polygonal structure to a valid Shapely polygon."""
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
        """Safely get material properties from Material or CustomMaterial-like objects."""
        if material is None:
            return 1.0, 1.0, 0.0

        if hasattr(material, "get_permittivity"):
            try:
                permittivity = material.get_permittivity(x, y, z)
                permeability = material.get_permeability(x, y, z)
                conductivity = material.get_conductivity(x, y, z)
                if hasattr(permittivity, "item"):
                    permittivity = permittivity.item()
                if hasattr(permeability, "item"):
                    permeability = permeability.item()
                if hasattr(conductivity, "item"):
                    conductivity = conductivity.item()
                return permittivity, permeability, conductivity
            except Exception as exc:
                print(f"Warning: CustomMaterial evaluation failed: {exc}, using defaults")
                return (
                    getattr(material, "default_permittivity", 1.0),
                    getattr(material, "default_permeability", 1.0),
                    getattr(material, "default_conductivity", 0.0),
                )

        if hasattr(material, "permittivity"):
            return (
                getattr(material, "permittivity", 1.0),
                getattr(material, "permeability", 1.0),
                getattr(material, "conductivity", 0.0),
            )

        print(f"Warning: Unknown material type {type(material)}, using vacuum properties")
        return 1.0, 1.0, 0.0

    def _get_all_material_props(self, material, x=0, y=0, z=0):
        """Get material properties as a tuple matching MaterialGrids.NAMES."""
        return self._get_material_properties_safe(material, x, y, z)

    def get_material_grids(self, resolution=None):
        """Return the rasterized material grids."""
        return self.permittivity, self.conductivity, self.permeability

    def rasterize(self, resolution=None):
        """Return self if resolution matches, otherwise raise."""
        if resolution is None or resolution == self.resolution:
            return self
        raise ValueError(
            "Cannot re-rasterize with different resolution. Use Design.rasterize()"
        )
