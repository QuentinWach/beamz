from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from shapely import area as shapely_area
from shapely import box as vectorized_box
from shapely import intersection
from shapely.geometry import Polygon as ShapelyPolygon

from beamz._helpers import (
    create_plain_progress,
    display_status,
    env_bool,
)
from beamz.const import LIGHT_SPEED
from beamz.design.materials import CustomMaterial, MaterialProtocol
from beamz.design.structures import (
    Circle,
    PlanarStructure,
    Rectangle,
    Ring,
)


@dataclass(frozen=True, slots=True)
class RasterShape:
    """Normalized geometry contract consumed by both rasterizers."""

    source: object
    kind: str
    bounds: tuple[float, float, float, float, float, float]
    material: object
    polygon: object | None

    @classmethod
    def from_structure(cls, structure):
        bounds = tuple(float(value) for value in structure.get_bounding_box())
        if len(bounds) != 6:
            raise ValueError(f"Invalid structure bounds: {bounds!r}")
        return cls(
            source=structure,
            kind=structure.raster_kind,
            bounds=bounds,
            material=structure.material,
            polygon=BaseMeshGrid._structure_polygon_2d(structure),
        )


@dataclass(frozen=True, slots=True)
class RasterContext:
    """Dimension-aware coordinates used by the shared rasterization loop."""

    dimensions: int
    cell_sizes: tuple[float, ...]
    grid_shape: tuple[int, ...]
    storage_shape: tuple[int, ...]
    centers: tuple[np.ndarray, ...]

    def point(self, storage_index):
        physical_index = tuple(reversed(storage_index))
        values = tuple(
            self.centers[axis][physical_index[axis]] for axis in range(self.dimensions)
        )
        return (*values, None) if self.dimensions == 2 else values


def _cell_count(length: float, step: float) -> int:
    """Convert an extent to cells without losing mathematically integral grids.

    SI geometry commonly arrives through sums of decimal nanometre values. A
    quotient that is mathematically 126 can consequently be represented just
    below 126, where a bare ``int`` would silently drop an entire boundary cell.
    Preserve the historical floor behavior for genuinely non-integral extents.
    """

    scaled = float(length) / float(step)
    nearest = int(round(scaled))
    if np.isclose(scaled, nearest, rtol=1e-12, atol=1e-12):
        return nearest
    return int(scaled)


@dataclass(frozen=True)
class GridSpec:
    """Configure automatic or explicit FDTD spatial discretization.

    Parameters
    ----------
    min_steps_per_wvl : float, default=10.0
        Minimum cells per wavelength inside the highest-index material.
    wavelength : float, optional
        Vacuum wavelength in metres used to derive automatic resolution.
    resolution : float, optional
        Explicit uniform cell size in metres. When present, it takes precedence
        over ``wavelength`` and ``min_steps_per_wvl``.
    courant : float, default=0.99
        Fraction of the dimensional Courant stability limit used for time steps.

    Notes
    -----
    Use :meth:`auto` for wavelength-driven grids and :meth:`uniform` for an
    explicit cell size.

    Examples
    --------
    >>> grid = GridSpec.auto(wavelength=1.55e-6, min_steps_per_wvl=12)
    >>> grid = GridSpec.uniform(20e-9, courant=0.9)
    """

    min_steps_per_wvl: float = 10.0
    wavelength: float | None = None
    resolution: float | None = None
    courant: float = 0.99

    @classmethod
    def auto(
        cls,
        *,
        min_steps_per_wvl: float = 10.0,
        wavelength: float | None = None,
        courant: float = 0.99,
    ) -> GridSpec:
        """Create a wavelength-driven automatic grid specification.

        Parameters
        ----------
        min_steps_per_wvl : float, default=10
            Minimum cells per local material wavelength.
        wavelength : float, optional
            Free-space design wavelength in metres.
        courant : float, default=0.99
            Fraction of the dimensional Courant stability limit.

        Returns
        -------
        GridSpec
            Immutable wavelength-driven grid policy.

        Examples
        --------
        >>> grid = GridSpec.auto(wavelength=1.55e-6, min_steps_per_wvl=12)
        """
        return cls(float(min_steps_per_wvl), wavelength, None, float(courant))

    @classmethod
    def uniform(cls, resolution: float, *, courant: float = 0.99) -> GridSpec:
        """Create a grid specification with an explicit uniform cell size.

        Parameters
        ----------
        resolution : float
            Cell spacing in metres.
        courant : float, default=0.99
            Fraction of the dimensional Courant stability limit.

        Returns
        -------
        GridSpec
            Immutable uniform-grid policy.

        Examples
        --------
        >>> grid = GridSpec.uniform(20e-9)
        """
        return cls(resolution=float(resolution), courant=float(courant))

    def resolve_resolution(self, *, max_index: float = 1.0) -> float:
        """Return the explicit or wavelength-derived cell size in metres.

        Parameters
        ----------
        max_index : float, default=1
            Largest refractive index represented by the grid.

        Returns
        -------
        float
            Uniform cell size in metres.

        Raises
        ------
        ValueError
            If neither an explicit resolution nor an automatic wavelength is set.

        Notes
        -----
        Automatic resolution is the vacuum wavelength divided by the product of
        ``max_index`` and ``min_steps_per_wvl``.
        """
        if self.resolution is not None:
            return float(self.resolution)
        if self.wavelength is None:
            raise ValueError(
                "GridSpec.auto requires wavelength when resolution is absent."
            )
        return float(self.wavelength) / (
            max(float(max_index), 1.0) * float(self.min_steps_per_wvl)
        )

    def resolve_time_step(self, resolution: float, *, dims: int) -> float:
        """Return a Courant-limited time step in seconds.

        Parameters
        ----------
        resolution : float
            Uniform cell spacing in metres.
        dims : int
            Number of active spatial dimensions.

        Returns
        -------
        float
            Courant-limited timestep in seconds.

        Examples
        --------
        >>> dt = GridSpec.uniform(20e-9).resolve_time_step(20e-9, dims=3)
        """
        return (
            float(self.courant)
            * float(resolution)
            / (LIGHT_SPEED * np.sqrt(float(max(1, int(dims)))))
        )


class MaterialGrids:
    """Bundles electromagnetic material grids with compact default-valued channels."""

    NAMES = (
        "permittivity",
        "permeability",
        "conductivity",
    )
    DEFAULTS = (1.0, 1.0, 0.0)
    DENSE_NAMES = frozenset(("permittivity",))
    DTYPE = np.float32

    def __init__(self, shape):
        self.shape = tuple(int(v) for v in shape)
        self._values = {}
        for name, default in zip(self.NAMES, self.DEFAULTS, strict=True):
            self._values[name] = (
                np.full(self.shape, default, dtype=self.DTYPE)
                if name in self.DENSE_NAMES
                else self.DTYPE(default)
            )

    def __getattr__(self, name):
        if name in self.NAMES:
            return self._values[name]
        raise AttributeError(name)

    @staticmethod
    def _same_scalar(left, right):
        left_arr = np.asarray(left)
        right_arr = np.asarray(right)
        if left_arr.shape != () or right_arr.shape != ():
            return False
        return bool(np.isclose(float(left_arr), float(right_arr), rtol=0.0, atol=0.0))

    def _materialize(self, name):
        value = self._values[name]
        if np.asarray(value).shape == ():
            value = np.full(self.shape, float(value), dtype=self.DTYPE)
            self._values[name] = value
        return value

    def fill_all(self, props):
        """Fill all grids with material property tuple."""
        for name, val in zip(self.NAMES, props, strict=True):
            if name in self.DENSE_NAMES:
                self._materialize(name).fill(val)
            else:
                self._values[name] = self.DTYPE(val)

    def blend_at(self, idx, props, factor):
        """Blend properties at index with given factor."""
        for name, val in zip(self.NAMES, props, strict=True):
            self.blend_channel_at(name, idx, val, factor)

    def blend_channel_at(self, name, idx, val, factor):
        current = self._values[name]
        if self._same_scalar(current, val):
            return
        arr = self._materialize(name)
        arr[idx] = arr[idx] * (1 - factor) + val * factor

    def set_region(self, slices, props):
        """Set all properties for a slice/index-array region."""
        for name, val in zip(self.NAMES, props, strict=True):
            self.set_channel_region(name, slices, val)

    def set_channel_region(self, name, slices, val):
        current = self._values[name]
        if self._same_scalar(current, val):
            return
        self._materialize(name)[slices] = val

    def set_masked_region(self, name, region, mask, val):
        if not np.any(mask):
            return
        current = self._values[name]
        if self._same_scalar(current, val):
            return
        view = self._materialize(name)[region]
        view[mask] = val

    def blend_masked_region(self, name, region, mask, val, factors):
        if not np.any(mask):
            return
        current = self._values[name]
        if self._same_scalar(current, val):
            return
        view = self._materialize(name)[region]
        f = np.asarray(factors)
        view[mask] = view[mask] * (1.0 - f) + val * f

    def assign_to(self, target):
        """Copy all grids as attributes onto target object."""
        for name in self.NAMES:
            setattr(target, name, self._values[name])


class BaseMeshGrid:
    """Construction-time mesh builder that freezes before public use."""

    _SUPPORTED_AA_MODES = ("legacy_grid", "stratified_jitter")
    _DEFAULT_AA_MODE = "legacy_grid"
    _DEFAULT_AA_SAMPLES = 64
    _DEFAULT_AA_SEED = 0

    shape: tuple[int, ...]
    permittivity: np.ndarray
    permeability: np.ndarray
    conductivity: np.ndarray

    def __init__(self, design, resolution, *, progress: bool = False):
        object.__setattr__(self, "_frozen", False)
        self.design = design
        self.resolution = resolution
        self.progress = bool(progress)
        self._validate_inputs()

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("Rasterized material grids are immutable.")
        object.__setattr__(self, name, value)

    def freeze(self):
        """Make rasterized arrays and metadata immutable, then return self."""
        for name in MaterialGrids.NAMES:
            value = getattr(self, name, None)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        object.__setattr__(self, "_frozen", True)
        return self

    def updated_copy(self, **materials):
        """Return a grid with functionally replaced material channels."""
        unknown = set(materials) - set(MaterialGrids.NAMES)
        if unknown:
            raise TypeError(f"Unsupported material grid fields: {sorted(unknown)!r}")
        copied = object.__new__(type(self))
        for name, value in vars(self).items():
            object.__setattr__(copied, name, value)
        object.__setattr__(copied, "_frozen", False)
        for name, value in materials.items():
            array = np.array(value, copy=True)
            if array.shape != self.shape:
                raise ValueError(
                    f"{name} shape {array.shape} does not match grid shape {self.shape}"
                )
            setattr(copied, name, array)
        return copied.freeze()

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

    def _raster_context(self, *cell_sizes):
        dimensions = len(cell_sizes)
        extents = (float(self.design.width), float(self.design.height))
        if dimensions == 3:
            extents += (float(self.design.depth),)
        counts = tuple(
            _cell_count(length, step) if length > 0.0 else 1
            for length, step in zip(extents, cell_sizes, strict=True)
        )
        centers = tuple(
            np.linspace(0.5 * step, length - 0.5 * step, count)
            if length > 0.0
            else np.asarray([0.0])
            for length, step, count in zip(extents, cell_sizes, counts, strict=True)
        )
        return RasterContext(
            dimensions,
            tuple(map(float, cell_sizes)),
            counts,
            tuple(counts[axis] for axis in reversed(range(dimensions))),
            centers,
        )

    def _rasterize_design(self, context):
        """Run shared normalization, clipping, sampling, and reporting."""
        started = time.perf_counter()
        grids = MaterialGrids(context.storage_shape)
        background_z = 0.0 if context.dimensions == 3 else None
        grids.fill_all(self._get_material_props(self.design.background, z=background_z))
        setup_finished = time.perf_counter()
        counts = {}

        with create_plain_progress(enabled=self.progress) as progress:
            task = progress.add_task(
                "Rasterizing structures...", total=len(self.design.structures)
            )
            for structure in self.design.structures:
                try:
                    shape = self._raster_shape(structure)
                    if shape is None:
                        continue
                    bbox = self._bbox_indices(shape, context)
                    if bbox is None:
                        continue
                    custom = isinstance(shape.material, CustomMaterial)
                    props = (
                        None
                        if custom
                        else self._get_material_props(shape.material, z=background_z)
                    )
                    sampler = (
                        (
                            lambda index, material=shape.material: (
                                self._get_material_props(
                                    material, *context.point(index)
                                )
                            )
                        )
                        if custom
                        else None
                    )
                    kernel = self._paint_shape(
                        shape, grids, bbox, context, props, sampler
                    )
                    counts[kernel] = counts.get(kernel, 0) + 1
                except (AttributeError, TypeError) as exc:
                    raise TypeError(
                        f"Failed to rasterize {type(structure).__name__}."
                    ) from exc
                finally:
                    progress.update(task, advance=1)

        structures_finished = time.perf_counter()
        grids.assign_to(self)
        if env_bool("BEAMZ_RASTER_TIMING", context.dimensions == 3):
            finished = time.perf_counter()
            kernels = ", ".join(f"{name}={count}" for name, count in counts.items())
            display_status(
                f"{context.dimensions}D raster timing: "
                f"setup={setup_finished - started:.2f}s, "
                f"structures={structures_finished - setup_finished:.2f}s, "
                f"total={finished - started:.2f}s; kernels: {kernels or 'none'}",
                "info",
            )

    def _paint_shape(self, shape, grids, bbox, context, props, sampler):
        raise NotImplementedError

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
        if not isinstance(structure, PlanarStructure):
            return None
        vertices = structure.vertices
        if len(vertices) < 3:
            return None

        shell = [(float(x), float(y)) for x, y, *_ in vertices]
        holes = []
        for hole in structure.interiors:
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

    @staticmethod
    def _bbox_indices(shape, context, margin=1):
        """Clip bounds and return starts/stops in kernel index order."""
        bounds = list(shape.bounds)
        if context.dimensions == 3:
            axis = 2
            upper = axis + 3
            if bounds[upper] <= bounds[axis]:
                bounds[upper] = bounds[axis] + context.cell_sizes[axis]
        lower = bounds[:3][: context.dimensions]
        upper = bounds[3:][: context.dimensions]
        margin = max(0, int(margin))
        starts = tuple(
            max(0, int(np.floor(value / step)) - margin)
            for value, step in zip(lower, context.cell_sizes, strict=True)
        )
        stops = tuple(
            min(count, int(np.ceil(value / step)) + margin)
            for value, step, count in zip(
                upper, context.cell_sizes, context.grid_shape, strict=True
            )
        )
        if any(start >= stop for start, stop in zip(starts, stops, strict=True)):
            return None
        order = (1, 0) if context.dimensions == 2 else (1, 0, 2)
        return tuple(starts[axis] for axis in order) + tuple(
            stops[axis] for axis in order
        )

    @staticmethod
    def _axis_coverage(lower, upper, start, stop, cell_size):
        cell_lower = np.arange(start, stop, dtype=float) * cell_size
        return (
            np.clip(
                np.minimum(cell_lower + cell_size, upper)
                - np.maximum(cell_lower, lower),
                0.0,
                cell_size,
            )
            / cell_size
        )

    @classmethod
    def _product_coverage(cls, axes):
        """Return separable cell coverage for axes in array storage order."""
        coverage = 1.0
        dims = len(axes)
        for axis, (lower, upper, start, stop, cell_size) in enumerate(axes):
            values = cls._axis_coverage(lower, upper, start, stop, cell_size)
            shape = [1] * dims
            shape[axis] = values.size
            coverage = coverage * values.reshape(shape)
        return np.asarray(coverage)

    @staticmethod
    def _raster_shape(structure):
        """Return the normalized shape, or ``None`` for non-material geometry."""
        if isinstance(structure, Rectangle) and structure.is_pml:
            return None
        shape = RasterShape.from_structure(structure)
        return shape if shape.material is not None else None

    @staticmethod
    def _apply_coverage(grids, region, props, coverage, sampler=None):
        coverage = np.asarray(coverage, dtype=float)
        if sampler is not None:
            origin = tuple(int(part.start or 0) for part in region)
            for local in zip(*np.where(coverage > 0.0), strict=True):
                index = tuple(
                    value + offset for value, offset in zip(local, origin, strict=True)
                )
                grids.blend_at(index, sampler(index), float(coverage[local]))
            return
        full = coverage >= 1.0 - 1e-12
        partial = (coverage > 0.0) & ~full
        for name, value in zip(MaterialGrids.NAMES, props, strict=True):
            grids.set_masked_region(name, region, full, value)
            grids.blend_masked_region(name, region, partial, value, coverage[partial])

    def _paint_axis_aligned(self, shape, grids, context, props, sampler):
        """Paint a 2D rectangle or 3D prism with exact cell coverage."""
        bbox = self._bbox_indices(shape, context, margin=0)
        if bbox is None:
            return
        dimensions = context.dimensions
        starts, stops = bbox[:dimensions], bbox[dimensions:]
        order = (1, 0) if dimensions == 2 else (1, 0, 2)
        physical_starts = tuple(starts[order.index(axis)] for axis in range(dimensions))
        physical_stops = tuple(stops[order.index(axis)] for axis in range(dimensions))
        lower = list(shape.bounds[:3][:dimensions])
        upper = list(shape.bounds[3:][:dimensions])
        if dimensions == 3 and upper[2] <= lower[2]:
            upper[2] = lower[2] + context.cell_sizes[2]
        storage_axes = tuple(reversed(range(dimensions)))
        storage_starts = tuple(physical_starts[axis] for axis in storage_axes)
        storage_stops = tuple(physical_stops[axis] for axis in storage_axes)
        region = tuple(
            slice(start, stop)
            for start, stop in zip(storage_starts, storage_stops, strict=True)
        )
        aligned = all(
            np.isclose(
                lower[axis],
                physical_starts[axis] * context.cell_sizes[axis],
                rtol=0.0,
                atol=1e-12,
            )
            and np.isclose(
                upper[axis],
                physical_stops[axis] * context.cell_sizes[axis],
                rtol=0.0,
                atol=1e-12,
            )
            for axis in range(dimensions)
        )
        if aligned and sampler is None:
            grids.set_region(region, props)
            return
        coverage = self._product_coverage(
            tuple(
                (
                    lower[axis],
                    upper[axis],
                    physical_starts[axis],
                    physical_stops[axis],
                    context.cell_sizes[axis],
                )
                for axis in storage_axes
            )
        )
        self._apply_coverage(grids, region, props, coverage, sampler)

    @staticmethod
    def _polygon_coverage(polygon, *, min_i, min_j, max_i, max_j, cell_size):
        """Compute exact vectorized XY cell coverage for a polygon."""
        x0 = np.arange(min_j, max_j, dtype=float) * cell_size
        y0 = np.arange(min_i, max_i, dtype=float) * cell_size
        xx, yy = np.meshgrid(x0, y0)
        cells = vectorized_box(xx, yy, xx + cell_size, yy + cell_size)
        return np.asarray(shapely_area(intersection(cells, polygon)), dtype=float) / (
            cell_size * cell_size
        )

    def _paint_polygon_exact(self, shape, grids, bbox, context, props, sampler):
        """Paint a planar polygon, optionally extruded through Z."""
        polygon = shape.polygon
        if polygon is None:
            return False
        min_i, min_j, *rest = bbox
        split = context.dimensions
        max_i, max_j = bbox[split : split + 2]
        coverage_xy = self._polygon_coverage(
            polygon,
            min_i=min_i,
            min_j=min_j,
            max_i=max_i,
            max_j=max_j,
            cell_size=context.cell_sizes[0],
        )
        if context.dimensions == 2:
            region = (slice(min_i, max_i), slice(min_j, max_j))
            self._apply_coverage(grids, region, props, coverage_xy, sampler)
            return True

        structure = shape.source
        if not isinstance(structure, PlanarStructure) or isinstance(
            structure, (Circle, Ring)
        ):
            return False
        vertices = np.asarray(structure.vertices, dtype=float)
        if (
            shape.bounds[5] <= shape.bounds[2]
            or vertices.ndim != 2
            or vertices.shape[0] < 3
            or (vertices.shape[1] >= 3 and np.ptp(vertices[:, 2]) > 1e-12)
            or structure.has_tapered_sidewalls()
        ):
            return False
        min_k, max_k = rest[0], bbox[-1]
        coverage_z = self._axis_coverage(
            shape.bounds[2],
            shape.bounds[5],
            min_k,
            max_k,
            context.cell_sizes[2],
        )
        region = (
            slice(min_k, max_k),
            slice(min_i, max_i),
            slice(min_j, max_j),
        )
        self._apply_coverage(
            grids, region, props, coverage_z[:, None, None] * coverage_xy, sampler
        )
        return True

    @staticmethod
    def _is_axis_aligned(shape):
        return isinstance(shape.source, Rectangle)

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

    @staticmethod
    def _get_material_props(material, x=0, y=0, z=None):
        """Return scalar material channels or propagate invalid material behavior."""
        if material is None:
            return MaterialGrids.DEFAULTS
        if not isinstance(material, MaterialProtocol):
            raise TypeError(f"Unsupported material type: {type(material).__name__}.")
        values = material.get_sample(x, y, z)
        return tuple(np.asarray(value).item() for value in values)

    def rasterize(self, resolution=None):
        """Return self if resolution matches, otherwise raise."""
        if resolution is None or resolution == self.resolution:
            return self
        raise ValueError(
            "Cannot re-rasterize with different resolution. Use Design.rasterize()"
        )

    def plot(self, **kwargs):
        """Plot a rasterized grid field using the matplotlib backend."""
        from beamz.analysis.plotting import plot_grid

        kwargs.setdefault("show", False)
        return plot_grid(self, **kwargs)

    def show(self, **kwargs):
        """Display a rasterized grid field using the matplotlib backend."""
        kwargs.setdefault("show", True)
        return self.plot(**kwargs)


class RegularGrid(BaseMeshGrid):
    """2D Regular grid meshing for 2D designs (backwards compatible)."""

    def __init__(
        self,
        design,
        resolution,
        aa_mode="legacy_grid",
        aa_samples=64,
        aa_seed=0,
        progress=False,
    ):
        super().__init__(design, resolution, progress=progress)
        self._configure_antialiasing(
            aa_mode=aa_mode,
            aa_samples=aa_samples,
            aa_seed=aa_seed,
        )

        # Check if this is actually a 2D design
        if design.is_3d and design.depth > 0:
            warnings.warn(
                "Using 2D RegularGrid for a 3D design; use RegularGrid3D for proper 3D meshing.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Determine is_3d property for compatibility with Simulation class
        self.is_3d = design.is_3d and design.depth > 0

        context = self._raster_context(self.resolution, self.resolution)
        self._sample_dx, self._sample_dy = self._build_supersample_offsets_xy(
            self.resolution
        )
        self._rasterize_design(context)

        # Set grid properties
        self.shape = self.permittivity.shape
        self.dx = self.resolution
        self.dy = self.resolution
        self.width = self.design.width
        self.height = self.design.height
        self.depth = 0.0
        self.freeze()

    def _paint_shape(self, shape, grids, bbox, context, props, sampler):
        if shape.kind == "rectangle" and self._is_axis_aligned(shape):
            self._paint_axis_aligned(shape, grids, context, props, sampler)
            return "rectangle"
        if shape.kind in {"circle", "ring"}:
            self._rasterize_radial(shape.source, grids, props, sampler, bbox, context)
            return "radial"
        if self._paint_polygon_exact(shape, grids, bbox, context, props, sampler):
            return "polygon"
        self._rasterize_polygon_fallback(
            shape.source, grids, props, sampler, bbox, context
        )
        return "polygon"

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

        return sum(
            bool(contains_fn(cx + dx, cy + dy))
            for dx, dy in zip(
                local_dx[:num_samples], local_dy[:num_samples], strict=True
            )
        )

    def _rasterize_radial(
        self,
        structure,
        grids,
        props,
        sampler,
        bbox,
        context,
    ):
        """Paint a circle or annulus with one radial coverage kernel."""
        min_i, min_j, max_i, max_j = bbox
        cell_size = context.cell_sizes[0]
        x_centers, y_centers = context.centers
        sample_dx, sample_dy = self._sample_dx, self._sample_dy
        num_samples = len(sample_dx)
        center_x, center_y = structure.position[:2]
        inner = float(structure.inner_radius) if isinstance(structure, Ring) else 0.0
        outer = float(
            structure.outer_radius if isinstance(structure, Ring) else structure.radius
        )
        x, y = np.meshgrid(
            x_centers[min_j:max_j],
            y_centers[min_i:max_i],
        )
        distance = np.hypot(x - center_x, y - center_y)
        half_diagonal = 0.3536 * cell_size
        full = (distance - half_diagonal >= inner) & (distance + half_diagonal <= outer)
        candidates = (
            (distance + half_diagonal >= inner)
            & (distance - half_diagonal <= outer)
            & ~full
        )

        def contains(px, py):
            radius = np.hypot(px - center_x, py - center_y)
            return inner <= radius <= outer

        coverage = full.astype(float)
        for local_i, local_j in zip(*np.where(candidates), strict=True):
            i, j = local_i + min_i, local_j + min_j
            count = self._supersample_cell(
                x_centers[j],
                y_centers[i],
                sample_dx,
                sample_dy,
                num_samples,
                contains,
                cell_i=i,
                cell_j=j,
                cell_size=cell_size,
            )
            coverage[local_i, local_j] = count / num_samples
        self._apply_coverage(
            grids,
            (slice(min_i, max_i), slice(min_j, max_j)),
            props,
            coverage,
            sampler,
        )

    def _rasterize_polygon_fallback(
        self,
        structure,
        grids,
        props,
        sampler,
        bbox,
        context,
    ):
        """Supersample geometry that cannot provide an exact polygon."""
        min_i, min_j, max_i, max_j = bbox
        cell_size = context.cell_sizes[0]
        x_centers, y_centers = context.centers
        sample_dx, sample_dy = self._sample_dx, self._sample_dy
        num_samples = len(sample_dx)
        contains = structure.point_in_polygon

        coverage = np.zeros((max_i - min_i, max_j - min_j), dtype=float)
        for local_i in range(coverage.shape[0]):
            for local_j in range(coverage.shape[1]):
                i, j = local_i + min_i, local_j + min_j
                coverage[local_i, local_j] = (
                    self._supersample_cell(
                        x_centers[j],
                        y_centers[i],
                        sample_dx,
                        sample_dy,
                        num_samples,
                        contains,
                        cell_i=i,
                        cell_j=j,
                        cell_size=cell_size,
                    )
                    / num_samples
                )
        self._apply_coverage(
            grids,
            (slice(min_i, max_i), slice(min_j, max_j)),
            props,
            coverage,
            sampler,
        )


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
        progress=False,
    ):
        if resolution_xy is None:
            raise ValueError("RegularGrid3D requires an xy resolution.")
        if resolution_z is None:
            # Only xy resolution provided, use same for z
            resolution_z = resolution_xy

        if resolution_xy is None or resolution_z is None:
            raise ValueError("RegularGrid3D requires xy and z resolutions.")
        resolution_xy = float(resolution_xy)
        resolution_z = float(resolution_z)

        super().__init__(design, resolution_xy, progress=progress)
        self._configure_antialiasing(
            aa_mode=aa_mode,
            aa_samples=aa_samples,
            aa_seed=aa_seed,
        )

        self.resolution_xy = resolution_xy
        self.resolution_z = resolution_z
        context = self._raster_context(
            self.resolution_xy, self.resolution_xy, self.resolution_z
        )
        self._prefer_exact = env_bool("BEAMZ_RASTER_EXACT_3D", True)
        self._rasterize_design(context)

        self.shape = self.permittivity.shape
        self.dx = self.resolution_xy
        self.dy = self.resolution_xy
        self.dz = self.resolution_z
        self.width = self.design.width
        self.height = self.design.height
        self.depth = self.design.depth
        display_status(
            "Created 3D mesh: " + " × ".join(map(str, context.grid_shape)) + " cells",
            "success",
        )
        self.freeze()

    def _paint_shape(self, shape, grids, bbox, context, props, sampler):
        structure = shape.source
        if (
            self._prefer_exact
            and shape.kind == "rectangle"
            and self._is_axis_aligned(shape)
            and (
                not isinstance(structure, PlanarStructure)
                or not structure.has_tapered_sidewalls()
            )
        ):
            self._paint_axis_aligned(shape, grids, context, props, sampler)
            return "rectangle"
        if self._prefer_exact and self._paint_polygon_exact(
            shape, grids, bbox, context, props, sampler
        ):
            return "polygon"
        self._rasterize_structure_3d_fallback(
            structure, grids, props, sampler, bbox, context
        )
        return "fallback"

    def _rasterize_structure_3d_fallback(
        self,
        structure,
        grids,
        props,
        sampler,
        bbox,
        context,
    ):
        """Fallback supersampling path for non-rectilinear 3D structures."""
        min_i, min_j, min_k, max_i, max_j, max_k = bbox
        if min_i >= max_i or min_j >= max_j or min_k >= max_k:
            return

        cell_size_xy, _, cell_size_z = context.cell_sizes
        x_centers, y_centers, z_centers = context.centers
        sample_dx, sample_dy = self._build_supersample_offsets_xy(cell_size_xy)
        offsets_z = self._build_supersample_offsets_z(
            cell_size_z=cell_size_z,
            depth_samples=(3 if len(z_centers) > 1 else 1),
        )
        num_samples = sample_dx.size * offsets_z.size

        contains_fn = structure.point_in_polygon

        coverage = np.zeros((max_k - min_k, max_i - min_i, max_j - min_j), dtype=float)
        for local_k, k in enumerate(range(min_k, max_k)):
            z_center = z_centers[k]
            for local_i, i in enumerate(range(min_i, max_i)):
                y_center = y_centers[i]
                for local_j, j in enumerate(range(min_j, max_j)):
                    x_center = x_centers[j]
                    cell_dx, cell_dy = self._scramble_offsets_xy_for_cell(
                        sample_dx=sample_dx,
                        sample_dy=sample_dy,
                        cell_size=cell_size_xy,
                        cell_i=i,
                        cell_j=j,
                        cell_k=k,
                    )
                    inside = sum(
                        bool(
                            contains_fn(
                                x_center + dx,
                                y_center + dy,
                                z_center + z_off,
                            )
                        )
                        for z_off in offsets_z
                        for dx, dy in zip(cell_dx, cell_dy, strict=True)
                    )
                    coverage[local_k, local_i, local_j] = inside / num_samples
        self._apply_coverage(
            grids,
            (
                slice(min_k, max_k),
                slice(min_i, max_i),
                slice(min_j, max_j),
            ),
            props,
            coverage,
            sampler,
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

        def slice_channel(value):
            arr = np.asarray(value)
            if arr.shape == ():
                return arr
            return arr[z_index, :, :]

        return {
            "permittivity": slice_channel(self.permittivity),
            "permeability": slice_channel(self.permeability),
            "conductivity": slice_channel(self.conductivity),
        }


# Convenience functions for automatic mesh selection
def create_mesh(
    design, resolution, auto_select=True, force_3d=False, progress=False, **kwargs
):
    """Create a mesh automatically selecting 2D or 3D based on design properties.

    Args:
        design: Design object to mesh
        resolution: Mesh resolution (or xy resolution for 3D)
        auto_select: If True, automatically choose between 2D and 3D meshing
        force_3d: If True, force 3D meshing even for 2D designs
        progress: Emit rasterization progress to stdout when True.
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
            progress=progress,
            **kwargs,
        )
    else:
        if auto_select and design.is_3d:
            display_status(
                "Auto-selecting 2D meshing for effectively 2D design (depth=0)", "info"
            )
        return RegularGrid(design, resolution, progress=progress, **kwargs)
