"""Physics- and geometry-aware solver grid-spacing policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.design.grid import RectilinearGrid
from beamz.design.mesher import _GradedMesher
from beamz.design.structures import Polygon

AxisValues = tuple[float, float, float]
OptionalAxisValues = tuple[float | None, float | None, float | None]

_AXIS_NORMAL_THRESHOLD = 0.2
_EQUIVALENT_ENVELOPE_RATIO = 1.05


def _xyz(values, name: str, *, fill: float) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) == 2:
        result = (*result, fill)
    if len(result) != 3:
        raise ValueError(f"{name} must contain two or three values.")
    return result


@dataclass(frozen=True, slots=True)
class MeshOverride:
    """Specify a meshing-only rectangular region with explicit axis spacings.

    ``None`` disables the override along an axis. An enforced override replaces
    material-derived spacing constraints in its region; a normal override can
    only refine them.
    """

    center: tuple[float, ...]
    size: tuple[float, ...]
    dl: float | tuple[float | None, ...]
    enforced: bool = False

    def __post_init__(self) -> None:
        center = _xyz(self.center, "MeshOverride center", fill=0.0)
        size = _xyz(self.size, "MeshOverride size", fill=np.inf)
        if not np.all(np.isfinite(center)):
            raise ValueError("MeshOverride center must be finite.")
        if any(np.isnan(value) or value <= 0.0 for value in size):
            raise ValueError("MeshOverride size must contain positive values.")
        if np.isscalar(self.dl):
            scalar = float(cast(float, self.dl))
            spacing: OptionalAxisValues = (scalar, scalar, scalar)
        else:
            raw = tuple(cast(tuple[float | None, ...], self.dl))
            if len(raw) == 2:
                raw = (*raw, None)
            if len(raw) != 3:
                raise ValueError("MeshOverride dl must contain two or three values.")
            spacing = cast(
                OptionalAxisValues,
                tuple(None if value is None else float(value) for value in raw),
            )
        if all(value is None for value in spacing) or any(
            value is not None and (not np.isfinite(value) or value <= 0.0)
            for value in spacing
        ):
            raise ValueError("MeshOverride dl must contain a positive spacing.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "dl", spacing)
        object.__setattr__(self, "enforced", bool(self.enforced))

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (
            tuple(c - 0.5 * s for c, s in zip(self.center, self.size, strict=True)),
            tuple(c + 0.5 * s for c, s in zip(self.center, self.size, strict=True)),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
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
    max_scale : float, default=1.3
        Hard maximum ratio between neighboring cell widths on a graded axis.
    min_steps_per_sim_size : float, default=10.0
        Minimum number of cells across each active simulation dimension.
    min_feature_cells : float, default=1.0
        Minimum target cells across detected structure thicknesses and gaps.
        Values above one opt into additional geometry-driven refinement beyond
        the material-wavelength target.
    dl_min, dl_max : float, optional
        Global lower and upper bounds for local automatic cell-width targets.
    overrides : tuple[MeshOverride, ...]
        Rectangular regions that refine or replace automatic spacing targets.
    snapping_points : tuple[tuple[float | None, ...], ...]
        User coordinates that must be grid edges along the selected axes.
    max_cells_per_axis : int, optional
        Safety limit for realized cells along any active axis. ``None`` disables it.
    max_total_cells : int, optional
        Safety limit for the active Cartesian cell product. ``None`` disables it.
    """

    min_steps_per_wvl: float = 10.0
    wavelength: float | None = None
    resolution: float | None = None
    courant: float = 0.99
    max_scale: float = 1.3
    min_steps_per_sim_size: float = 10.0
    min_feature_cells: float = 1.0
    dl_min: float | None = None
    dl_max: float | None = None
    overrides: tuple[MeshOverride, ...] = ()
    snapping_points: tuple[tuple[float | None, ...], ...] = ()
    max_cells_per_axis: int | None = 200_000
    max_total_cells: int | None = 20_000_000

    @property
    def is_automatic(self) -> bool:
        """Return whether this policy realizes a geometry-aware grid.

        Returns
        -------
        bool
            ``True`` when no explicit uniform resolution was supplied.
        """
        return self.resolution is None

    def __post_init__(self) -> None:
        positive = {
            "min_steps_per_wvl": self.min_steps_per_wvl,
            "courant": self.courant,
            "max_scale": self.max_scale,
            "min_steps_per_sim_size": self.min_steps_per_sim_size,
            "min_feature_cells": self.min_feature_cells,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"GridSpec {name} must be finite and positive.")
        if float(self.courant) > 1.0:
            raise ValueError("GridSpec courant cannot exceed one.")
        if not 1.0 < float(self.max_scale) < 2.0:
            raise ValueError("GridSpec max_scale must be strictly between 1 and 2.")
        for name in ("wavelength", "resolution", "dl_min", "dl_max"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or float(value) <= 0.0):
                raise ValueError(f"GridSpec {name} must be positive and finite.")
        for name in ("max_cells_per_axis", "max_total_cells"):
            value = getattr(self, name)
            if value is None:
                continue
            integer = int(value)
            if isinstance(value, (bool, np.bool_)) or integer != value or integer <= 0:
                raise ValueError(f"GridSpec {name} must be a positive integer or None.")
            object.__setattr__(self, name, integer)
        if (
            self.dl_min is not None
            and self.dl_max is not None
            and self.dl_min > self.dl_max
        ):
            raise ValueError("GridSpec dl_min cannot exceed dl_max.")
        overrides = tuple(self.overrides)
        if any(not isinstance(value, MeshOverride) for value in overrides):
            raise TypeError("GridSpec overrides must contain MeshOverride values.")
        points = []
        for point in self.snapping_points:
            values = tuple(point)
            if len(values) == 2:
                values = (*values, None)
            if len(values) != 3:
                raise ValueError(
                    "GridSpec snapping points must have two or three values."
                )
            normalized = tuple(
                None if value is None else float(value) for value in values
            )
            if any(
                value is not None and not np.isfinite(value) for value in normalized
            ):
                raise ValueError("GridSpec snapping points must be finite.")
            points.append(normalized)
        object.__setattr__(self, "overrides", overrides)
        object.__setattr__(self, "snapping_points", tuple(points))

    @classmethod
    def auto(
        cls,
        *,
        min_steps_per_wvl: float = 10.0,
        wavelength: float | None = None,
        courant: float = 0.99,
        max_scale: float = 1.3,
        min_steps_per_sim_size: float = 10.0,
        min_feature_cells: float = 1.0,
        dl_min: float | None = None,
        dl_max: float | None = None,
        overrides: tuple[MeshOverride, ...] = (),
        snapping_points: tuple[tuple[float | None, ...], ...] = (),
        max_cells_per_axis: int | None = 200_000,
        max_total_cells: int | None = 20_000_000,
    ) -> GridSpec:
        """Create a wavelength- and geometry-aware nonuniform grid policy.

        Returns
        -------
        GridSpec
            Immutable wavelength-driven grid policy.
        """
        return cls(
            min_steps_per_wvl=float(min_steps_per_wvl),
            wavelength=wavelength,
            courant=float(courant),
            max_scale=float(max_scale),
            min_steps_per_sim_size=float(min_steps_per_sim_size),
            min_feature_cells=float(min_feature_cells),
            dl_min=dl_min,
            dl_max=dl_max,
            overrides=overrides,
            snapping_points=snapping_points,
            max_cells_per_axis=max_cells_per_axis,
            max_total_cells=max_total_cells,
        )

    @classmethod
    def uniform(
        cls,
        resolution: float,
        *,
        courant: float = 0.99,
        max_cells_per_axis: int | None = 200_000,
        max_total_cells: int | None = 20_000_000,
    ) -> GridSpec:
        """Create a grid specification with an explicit uniform cell size.

        Returns
        -------
        GridSpec
            Immutable uniform-grid policy.
        """
        return cls(
            resolution=float(resolution),
            courant=float(courant),
            max_cells_per_axis=max_cells_per_axis,
            max_total_cells=max_total_cells,
        )

    def _target_spacing(self, *, max_index: float = 1.0) -> float:
        """Return the explicit or wavelength-derived target spacing."""
        if self.resolution is not None:
            return float(self.resolution)
        if self.wavelength is None:
            raise ValueError(
                "GridSpec.auto requires wavelength when resolution is absent."
            )
        return float(self.wavelength) / (
            max(float(max_index), 1.0) * float(self.min_steps_per_wvl)
        )

    def realize(self, design: Any) -> RectilinearGrid:
        """Realize this policy for a concrete design domain and material stack.

        Returns
        -------
        RectilinearGrid
            Physical grid edges satisfying this policy for ``design``.
        """
        if self.is_automatic:
            if self.wavelength is None:
                raise ValueError("GridSpec.auto requires wavelength.")
            grid = _realize_graded_grid(design, self)
        else:
            grid = _realize_uniform_grid(design, self._target_spacing(), spec=self)
        dimensions = 3 if float(design.depth) > 0.0 else 2
        return _validate_grid_budget(grid, self, dimensions=dimensions)

    def resolve_time_step(
        self, resolution: float | RectilinearGrid, *, dims: int
    ) -> float:
        """Return a Courant-limited time step in seconds.

        Returns
        -------
        float
            Courant-limited time step in seconds.
        """
        if isinstance(resolution, RectilinearGrid):
            active_axes = ("x", "y", "z") if int(dims) == 3 else ("x", "y")
            return resolution.cfl_time_step(
                self.courant,
                active_axes=active_axes,
            )
        return (
            float(self.courant)
            * float(resolution)
            / (LIGHT_SPEED * np.sqrt(float(max(1, int(dims)))))
        )


@dataclass(frozen=True, slots=True)
class _Region:
    lower: AxisValues
    upper: AxisValues
    spacing: OptionalAxisValues
    enforced: bool = False


@dataclass(frozen=True, slots=True)
class _MeshFeatureRegion:
    """Localized, axis-aware spacing constraint produced by geometry analysis."""

    lower: AxisValues
    upper: AxisValues
    spacing: OptionalAxisValues
    reason: str


@dataclass(frozen=True, slots=True)
class _OpposingFeature:
    """A reliable local distance between opposing polygon boundary faces."""

    lower: tuple[float, float]
    upper: tuple[float, float]
    width: float
    normal: tuple[float, float]
    reason: str


def _validate_grid_budget(
    grid: RectilinearGrid, spec: GridSpec, *, dimensions: int
) -> RectilinearGrid:
    """Reject a realized grid before material and field arrays can exhaust memory."""
    _validate_grid_shape_budget(
        grid.shape,
        spec,
        dimensions=dimensions,
        context="realized grid",
        minimum_spacing=grid.minimum_spacing,
    )
    return grid


def _validate_grid_shape_budget(
    shape: tuple[int, int, int],
    spec: GridSpec,
    *,
    dimensions: int,
    context: str,
    minimum_spacing: float | None = None,
) -> None:
    """Reject predicted cell counts before allocating edge or material arrays."""
    dims = 3 if int(dimensions) == 3 else 2
    active_shape = shape[:dims]
    total_cells = math.prod(active_shape)
    violations = []
    if spec.max_cells_per_axis is not None:
        oversized = [
            f"{'xyz'[axis]}={count:,}"
            for axis, count in enumerate(active_shape)
            if count > spec.max_cells_per_axis
        ]
        if oversized:
            violations.append(
                "axis limit "
                f"{spec.max_cells_per_axis:,} exceeded by {', '.join(oversized)}"
            )
    if spec.max_total_cells is not None and total_cells > spec.max_total_cells:
        violations.append(
            f"total limit {spec.max_total_cells:,} exceeded by {total_cells:,} cells"
        )
    if not violations:
        return
    bytes_per_cell = 128 if dims == 3 else 64
    estimated_gib = total_cells * bytes_per_cell / 1024**3
    spacing_detail = (
        ""
        if minimum_spacing is None
        else f" The smallest requested spacing is {minimum_spacing:.6g}."
    )
    raise ValueError(
        f"Grid budget exceeded before allocation for predicted active shape "
        f"{active_shape} ({context}; estimated setup storage {estimated_gib:.2f} "
        f"GiB): {'; '.join(violations)}.{spacing_detail} Increase the matching "
        "GridSpec budget explicitly, raise dl_min, or relax local refinement."
    )


def _material_index(material: Any) -> float:
    maximum = getattr(material, "max_permittivity", None)
    if maximum is None:
        raise ValueError("Automatic meshing requires finite material permittivity.")
    value = float(np.real(maximum))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Automatic meshing requires positive material permittivity.")
    return float(np.sqrt(max(value, 1.0)))


def _design_extents(design: Any) -> tuple[float, float, float]:
    depth = float(design.depth)
    return float(design.width), float(design.height), depth if depth > 0.0 else 1.0


def _realize_uniform_grid(
    design: Any, resolution: float, *, spec: GridSpec | None = None
) -> RectilinearGrid:
    """Cover a design with the exact isotropic spacing used by scalar rasterization."""
    spacing = float(resolution)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Uniform grid resolution must be positive and finite.")

    def count(extent: float) -> int:
        ratio = float(extent) / spacing
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(ratio))
        return max(1, math.ceil(ratio - tolerance))

    extents = _design_extents(design)
    nx, ny = count(extents[0]), count(extents[1])
    dimensions = 3 if float(design.depth) > 0.0 else 2
    nz = count(extents[2]) if dimensions == 3 else 1
    if spec is not None:
        _validate_grid_shape_budget(
            (nx, ny, nz),
            spec,
            dimensions=dimensions,
            context="uniform spacing preflight",
            minimum_spacing=spacing,
        )
    if float(design.depth) > 0.0:
        return RectilinearGrid.uniform(
            (0.0, 0.0, 0.0),
            (nx * spacing, ny * spacing, nz * spacing),
            (nx, ny, nz),
        )
    return RectilinearGrid.uniform(
        (0.0, 0.0, 0.0),
        (nx * spacing, ny * spacing, 1.0),
        (nx, ny, 1),
    )


def _design_coordinate_offset(design: Any) -> AxisValues:
    """Translate centered public geometry into the positive raster domain."""
    if not bool(getattr(design, "_centered_coordinates", False)):
        return (0.0, 0.0, 0.0)
    extents = _design_extents(design)
    return (
        0.5 * extents[0],
        0.5 * extents[1],
        0.5 * extents[2] if float(design.depth) > 0.0 else 0.0,
    )


def _structure_bounds(
    structure: Any, *, two_dimensional: bool
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if hasattr(structure, "lower") and hasattr(structure, "upper"):
        return cast(AxisValues, tuple(map(float, structure.lower))), cast(
            AxisValues, tuple(map(float, structure.upper))
        )
    if hasattr(structure, "radius") and not getattr(structure, "vertices", ()):
        center = tuple(map(float, structure.center))
        radius = float(structure.radius)
        return cast(AxisValues, tuple(value - radius for value in center)), cast(
            AxisValues, tuple(value + radius for value in center)
        )
    vertices = np.asarray(getattr(structure, "vertices", ()), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3:
        raise TypeError(f"Cannot derive meshing bounds for {type(structure).__name__}.")
    lower = [float(np.min(vertices[:, 0])), float(np.min(vertices[:, 1]))]
    upper = [float(np.max(vertices[:, 0])), float(np.max(vertices[:, 1]))]
    if two_dimensional:
        return cast(AxisValues, (*lower, 0.0)), cast(AxisValues, (*upper, 1.0))
    z = float(getattr(structure, "z", np.min(vertices[:, 2])))
    depth = float(getattr(structure, "depth", 0.0))
    return cast(AxisValues, (*lower, z)), cast(AxisValues, (*upper, z + depth))


def _polygon_opposing_features(structure: Polygon) -> list[_OpposingFeature]:
    """Return reliable local distances between opposing material faces.

    Boundary vertex spacing is deliberately excluded: refining a curved polygon's
    tessellation must not change its physical mesh target.
    """
    from shapely.geometry import LineString
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry.polygon import orient

    shell = [(float(x), float(y)) for x, y, _z in structure.vertices]
    holes = [
        [(float(x), float(y)) for x, y, _z in path] for path in structure.interiors
    ]
    geometry = orient(ShapelyPolygon(shell, holes=holes), sign=1.0)
    if geometry.is_empty or not geometry.is_valid:
        return []
    if not geometry.interiors and geometry.equals(geometry.convex_hull):
        coordinates = np.asarray(geometry.exterior.coords[:-1], dtype=np.float64)
        edges = np.roll(coordinates, -1, axis=0) - coordinates
        edge_lengths = np.linalg.norm(edges, axis=1)
        normals = np.column_stack((-edges[:, 1], edges[:, 0])) / edge_lengths[:, None]
        widths = np.asarray([float(np.ptp(coordinates @ normal)) for normal in normals])
        index = int(np.argmin(widths))
        width = float(widths[index])
        if not np.isfinite(width) or width <= 0.0:
            return []
        bounds = tuple(map(float, geometry.bounds))
        return [
            _OpposingFeature(
                (bounds[0], bounds[1]),
                (bounds[2], bounds[3]),
                width,
                (float(normals[index, 0]), float(normals[index, 1])),
                "minimum convex Feret width",
            )
        ]

    def canonical_ring(ring) -> np.ndarray:
        coordinates = np.asarray(ring.coords[:-1], dtype=np.float64)
        if coordinates.shape[0] < 2:
            return coordinates
        start = min(
            range(coordinates.shape[0]),
            key=lambda index: (
                *coordinates[index],
                *coordinates[(index + 1) % coordinates.shape[0]],
                *coordinates[index - 1],
            ),
        )
        return np.roll(coordinates, -start, axis=0)

    exterior = canonical_ring(geometry.exterior)
    interiors = sorted(
        (canonical_ring(ring) for ring in geometry.interiors),
        key=lambda coordinates: tuple(coordinates.reshape(-1)),
    )
    rings = (exterior, *interiors)
    segments = []
    ring_sizes = [coordinates.shape[0] for coordinates in rings]
    for ring_id, coordinates in enumerate(rings):
        for local_id, start in enumerate(coordinates):
            end = coordinates[(local_id + 1) % coordinates.shape[0]]
            if np.linalg.norm(end - start) <= np.finfo(float).tiny:
                continue
            segments.append((start, end, ring_id, local_id))
    segments.sort(
        key=lambda segment: (
            *segment[0],
            *segment[1],
            segment[2],
            segment[3],
        )
    )
    starts = [segment[0] for segment in segments]
    ends = [segment[1] for segment in segments]
    ring_ids = [segment[2] for segment in segments]
    local_ids = [segment[3] for segment in segments]
    if len(starts) < 2:
        return []

    start_array = np.asarray(starts)
    end_array = np.asarray(ends)
    vectors = end_array - start_array
    lengths = np.linalg.norm(vectors, axis=1)
    tangents = vectors / lengths[:, None]
    # Oriented exterior and interior rings both keep material on their left.
    inward_normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    midpoints = 0.5 * (start_array + end_array)
    ring_ids_array = np.asarray(ring_ids)
    local_ids_array = np.asarray(local_ids)
    tolerance = 128.0 * np.finfo(float).eps * max(1.0, *geometry.bounds)
    geometry_with_tolerance = geometry.buffer(tolerance)
    features = []

    for index in range(len(starts)):
        normal = inward_normals[index]
        offsets = start_array - midpoints[index]
        denominators = normal[0] * vectors[:, 1] - normal[1] * vectors[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            ray_distances = (
                offsets[:, 0] * vectors[:, 1] - offsets[:, 1] * vectors[:, 0]
            ) / denominators
            segment_positions = (
                offsets[:, 0] * normal[1] - offsets[:, 1] * normal[0]
            ) / denominators
        valid = np.arange(len(starts)) != index
        same_ring = ring_ids_array == ring_ids_array[index]
        local_delta = np.abs(local_ids_array - local_ids_array[index])
        ring_size = ring_sizes[ring_ids[index]]
        adjacent = same_ring & ((local_delta <= 1) | (local_delta >= ring_size - 1))
        valid &= ~adjacent
        valid &= np.einsum("ij,j->i", inward_normals, inward_normals[index]) < -0.95
        valid &= np.abs(denominators) > tolerance
        valid &= np.abs(ray_distances) > tolerance
        valid &= segment_positions >= -1e-12
        valid &= segment_positions <= 1.0 + 1e-12
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            continue
        candidates = candidates[
            np.argsort(np.abs(ray_distances[candidates]), kind="stable")
        ]
        first_point = midpoints[index]
        for other in candidates:
            signed_distance = float(ray_distances[other])
            distance = abs(signed_distance)
            second_point = first_point + signed_distance * normal
            connector = LineString((first_point, second_point))
            material_between = geometry_with_tolerance.covers(connector)
            gap_between = float(connector.intersection(geometry).length) <= tolerance
            if not material_between and not gap_between:
                continue
            direction = second_point - first_point
            feature_normal = direction / distance
            padding = 0.5 * distance
            feature_points = np.vstack(
                (start_array[index], end_array[index], first_point, second_point)
            )
            lower = np.min(feature_points, axis=0) - padding
            upper = np.max(feature_points, axis=0) + padding
            features.append(
                _OpposingFeature(
                    (float(lower[0]), float(lower[1])),
                    (float(upper[0]), float(upper[1])),
                    distance,
                    (float(feature_normal[0]), float(feature_normal[1])),
                    (
                        "opposing polygon faces"
                        if material_between
                        else "opposing polygon gap faces"
                    ),
                )
            )
            # The nearest full connector defines the local thickness. Farther
            # crossings are bulk polygon dimensions and should not add constraints.
            break

    return features


def _merge_mesh_feature_regions(
    regions: list[_MeshFeatureRegion],
) -> list[_MeshFeatureRegion]:
    """Canonicalize overlapping local corridors into a 1D spacing envelope."""
    if not regions:
        return []
    global_lower = tuple(
        min(region.lower[axis] for region in regions) for axis in range(3)
    )
    global_upper = tuple(
        max(region.upper[axis] for region in regions) for axis in range(3)
    )
    axis_envelopes: list[list[tuple[float, float, float, str]]] = []
    for axis in range(2):
        selected = [region for region in regions if region.spacing[axis] is not None]
        if not selected:
            axis_envelopes.append([])
            continue

        def active_spacing(
            region: _MeshFeatureRegion, selected_axis: int = axis
        ) -> float:
            value = region.spacing[selected_axis]
            assert value is not None
            return float(value)

        raw_coordinates = sorted(
            {
                value
                for region in selected
                for value in (region.lower[axis], region.upper[axis])
            }
        )
        smallest_spacing = min(active_spacing(region) for region in selected)
        coordinate_clusters: list[list[float]] = []
        for value in raw_coordinates:
            if (
                not coordinate_clusters
                or value - coordinate_clusters[-1][0] >= smallest_spacing
            ):
                coordinate_clusters.append([value])
            else:
                coordinate_clusters[-1].append(value)
        coordinates = [float(np.mean(cluster)) for cluster in coordinate_clusters]
        intervals: list[tuple[float, float, float, str]] = []
        for lower, upper in zip(coordinates[:-1], coordinates[1:], strict=True):
            if upper <= lower:
                continue
            midpoint = 0.5 * (lower + upper)
            covering = [
                region
                for region in selected
                if region.lower[axis] <= midpoint <= region.upper[axis]
            ]
            if not covering:
                continue
            spacing = min(active_spacing(region) for region in covering)
            finest = min(covering, key=active_spacing)
            reason = finest.reason
            if intervals:
                old_lower, old_upper, old_spacing, old_reason = intervals[-1]
                ratio = max(old_spacing, spacing) / min(old_spacing, spacing)
                if old_upper == lower and ratio <= 1.25:
                    intervals[-1] = (
                        old_lower,
                        upper,
                        min(old_spacing, spacing),
                        old_reason if old_spacing <= spacing else reason,
                    )
                    continue
            intervals.append((lower, upper, spacing, reason))
        axis_envelopes.append(intervals)

    x_intervals, y_intervals = axis_envelopes
    if len(x_intervals) == len(y_intervals) and all(
        max(x_item[2], y_item[2]) / min(x_item[2], y_item[2])
        <= _EQUIVALENT_ENVELOPE_RATIO
        and abs(x_item[0] - y_item[0]) <= max(x_item[2], y_item[2])
        and abs(x_item[1] - y_item[1]) <= max(x_item[2], y_item[2])
        for x_item, y_item in zip(x_intervals, y_intervals, strict=True)
    ):
        canonical = [
            (
                0.5 * (x_item[0] + y_item[0]),
                0.5 * (x_item[1] + y_item[1]),
                min(x_item[2], y_item[2]),
                x_item[3] if x_item[2] <= y_item[2] else y_item[3],
            )
            for x_item, y_item in zip(x_intervals, y_intervals, strict=True)
        ]
        axis_envelopes = [canonical, canonical]

    result = []
    for axis, intervals in enumerate(axis_envelopes):
        for lower, upper, spacing, reason in intervals:
            region_lower = list(global_lower)
            region_upper = list(global_upper)
            region_lower[axis], region_upper[axis] = lower, upper
            axis_spacing: list[float | None] = [None, None, None]
            axis_spacing[axis] = spacing
            result.append(
                _MeshFeatureRegion(
                    cast(AxisValues, tuple(region_lower)),
                    cast(AxisValues, tuple(region_upper)),
                    cast(OptionalAxisValues, tuple(axis_spacing)),
                    reason,
                )
            )
    return result


def _polygon_mesh_feature_regions(
    structure: Polygon,
    *,
    material_spacing: float,
    spec: GridSpec,
) -> list[_MeshFeatureRegion]:
    """Convert opposing polygon faces into local, axis-specific constraints."""
    depth = float(structure.depth)
    z_lower = float(structure.z)
    z_upper = z_lower + depth if depth > 0.0 else 1.0
    regions = []
    for feature in _polygon_opposing_features(structure):
        target = _clamp_spacing(feature.width / spec.min_feature_cells, spec)
        if target >= material_spacing * (1.0 - 64.0 * np.finfo(float).eps):
            continue
        direction = np.abs(np.asarray(feature.normal, dtype=np.float64))
        # Ignore nearly tangential projections. The threshold also provides enough
        # angular overlap for coarse curve tessellations to form one stable envelope.
        active = direction >= _AXIS_NORMAL_THRESHOLD
        if not np.any(active):
            active[int(np.argmax(direction))] = True
        for axis in np.flatnonzero(active):
            spacing = cast(
                OptionalAxisValues,
                tuple(target if index == int(axis) else None for index in range(3)),
            )
            regions.append(
                _MeshFeatureRegion(
                    (feature.lower[0], feature.lower[1], z_lower),
                    (feature.upper[0], feature.upper[1], z_upper),
                    spacing,
                    f"{feature.reason} ({feature.width:.6g})",
                )
            )
    return _merge_mesh_feature_regions(regions)


def _feature_sizes(
    structure: Any,
) -> tuple[float | None, float | None, float | None]:
    semantic = structure._mesh_feature_sizes()
    if semantic is not None:
        return cast(
            tuple[float | None, float | None, float | None],
            tuple(
                value
                if value is None
                else (float(value) if np.isfinite(value) and value > 0.0 else None)
                for value in semantic
            ),
        )
    width = getattr(structure, "width", None)
    height = getattr(structure, "height", None)
    depth = getattr(structure, "depth", None)
    return tuple(
        value if value is not None and np.isfinite(value) and value > 0.0 else None
        for value in (width, height, depth)
    )  # type: ignore[return-value]


def _clamp_spacing(value: float, spec: GridSpec) -> float:
    if spec.dl_min is not None:
        value = max(value, float(spec.dl_min))
    if spec.dl_max is not None:
        value = min(value, float(spec.dl_max))
    return value


def _overlap(first: _Region, second: _Region, transverse_axes: tuple[int, ...]) -> bool:
    return all(
        first.lower[axis] < second.upper[axis]
        and second.lower[axis] < first.upper[axis]
        for axis in transverse_axes
    )


def _geometry_regions(
    design: Any, spec: GridSpec, coordinate_offset: AxisValues
) -> list[_Region]:
    extents = _design_extents(design)
    if spec.wavelength is None:
        raise ValueError("GridSpec.auto requires wavelength.")
    wavelength = float(spec.wavelength)
    two_dimensional = float(design.depth) == 0.0
    regions = []
    material_regions = []
    for structure in getattr(design, "structures", ()):
        raw_lower, raw_upper = _structure_bounds(
            structure, two_dimensional=two_dimensional
        )
        lower = cast(
            AxisValues,
            tuple(
                max(0.0, raw_lower[axis] + coordinate_offset[axis]) for axis in range(3)
            ),
        )
        upper = cast(
            AxisValues,
            tuple(
                min(extents[axis], raw_upper[axis] + coordinate_offset[axis])
                for axis in range(3)
            ),
        )
        if any(
            upper[axis] <= lower[axis] for axis in range(2 if two_dimensional else 3)
        ):
            continue
        material_spacing = _clamp_spacing(
            wavelength / (_material_index(structure.material) * spec.min_steps_per_wvl),
            spec,
        )
        features = _feature_sizes(structure)
        spacing = tuple(
            _clamp_spacing(
                min(material_spacing, feature / spec.min_feature_cells)
                if feature is not None
                else material_spacing,
                spec,
            )
            for feature in features
        )
        material_region = _Region(lower, upper, cast(OptionalAxisValues, spacing))
        regions.append(material_region)
        material_regions.append(material_region)
        if isinstance(structure, Polygon) and spec.min_feature_cells > 1.0:
            for feature_region in _polygon_mesh_feature_regions(
                structure,
                material_spacing=material_spacing,
                spec=spec,
            ):
                feature_lower = cast(
                    AxisValues,
                    tuple(
                        max(
                            0.0,
                            feature_region.lower[axis] + coordinate_offset[axis],
                        )
                        for axis in range(3)
                    ),
                )
                feature_upper = cast(
                    AxisValues,
                    tuple(
                        min(
                            extents[axis],
                            feature_region.upper[axis] + coordinate_offset[axis],
                        )
                        for axis in range(3)
                    ),
                )
                snapped_lower = list(feature_lower)
                snapped_upper = list(feature_upper)
                for axis, feature_spacing in enumerate(feature_region.spacing):
                    if feature_spacing is None:
                        continue
                    if abs(snapped_lower[axis] - lower[axis]) < feature_spacing:
                        snapped_lower[axis] = lower[axis]
                    if abs(snapped_upper[axis] - upper[axis]) < feature_spacing:
                        snapped_upper[axis] = upper[axis]
                feature_lower = cast(AxisValues, tuple(snapped_lower))
                feature_upper = cast(AxisValues, tuple(snapped_upper))
                if any(
                    feature_upper[axis] <= feature_lower[axis]
                    for axis in range(2 if two_dimensional else 3)
                ):
                    continue
                regions.append(
                    _Region(
                        feature_lower,
                        feature_upper,
                        feature_region.spacing,
                    )
                )

    active_count = 2 if two_dimensional else 3
    for first_index, first in enumerate(material_regions):
        for second in material_regions[first_index + 1 :]:
            for axis in range(active_count):
                transverse = tuple(
                    index for index in range(active_count) if index != axis
                )
                if not _overlap(first, second, transverse):
                    continue
                if first.upper[axis] <= second.lower[axis]:
                    gap_lower, gap_upper = first.upper[axis], second.lower[axis]
                elif second.upper[axis] <= first.lower[axis]:
                    gap_lower, gap_upper = second.upper[axis], first.lower[axis]
                else:
                    continue
                gap = gap_upper - gap_lower
                gap_spacing = _clamp_spacing(gap / spec.min_feature_cells, spec)
                lower = [0.0, 0.0, 0.0]
                upper = list(extents)
                lower[axis], upper[axis] = gap_lower, gap_upper
                regions.append(
                    _Region(
                        cast(AxisValues, tuple(lower)),
                        cast(AxisValues, tuple(upper)),
                        cast(
                            OptionalAxisValues,
                            tuple(
                                gap_spacing if index == axis else None
                                for index in range(3)
                            ),
                        ),
                    )
                )

    for override in spec.overrides:
        raw_lower, raw_upper = override.bounds
        lower = cast(
            AxisValues,
            tuple(
                max(0.0, raw_lower[axis] + coordinate_offset[axis]) for axis in range(3)
            ),
        )
        upper = cast(
            AxisValues,
            tuple(
                min(extents[axis], raw_upper[axis] + coordinate_offset[axis])
                for axis in range(3)
            ),
        )
        if any(upper[axis] <= lower[axis] for axis in range(active_count)):
            continue
        override_spacing = cast(OptionalAxisValues, override.dl)
        spacing = tuple(
            None if value is None else _clamp_spacing(value, spec)
            for value in override_spacing
        )
        regions.append(
            _Region(lower, upper, cast(OptionalAxisValues, spacing), override.enforced)
        )
    return regions


def _unique_coordinates(values: list[float], extent: float) -> np.ndarray:
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(extent))
    result = []
    for value in sorted(float(np.clip(item, 0.0, extent)) for item in values):
        if not result or value - result[-1] > tolerance:
            result.append(value)
    result[0], result[-1] = 0.0, extent
    return np.asarray(result, dtype=np.float64)


def _axis_constraints(
    axis: int,
    extent: float,
    background_spacing: float,
    regions: list[_Region],
    spec: GridSpec,
    coordinate_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = [0.0, extent]
    effective_regions = []
    for region in regions:
        if region.spacing[axis] is None:
            continue
        coordinates.extend((region.lower[axis], region.upper[axis]))
        effective_regions.append(region)
    for point in spec.snapping_points:
        value = point[axis]
        if value is not None:
            value = float(value) + coordinate_offset
        if value is not None and 0.0 < value < extent:
            coordinates.append(value)
    coords = _unique_coordinates(coordinates, extent)
    limits = np.empty(coords.size - 1, dtype=np.float64)
    for index, midpoint in enumerate(0.5 * (coords[:-1] + coords[1:])):
        limit = background_spacing
        for region in effective_regions:
            if (
                not region.enforced
                and region.lower[axis] <= midpoint <= region.upper[axis]
            ):
                limit = min(limit, float(region.spacing[axis]))
        for region in effective_regions:
            if region.enforced and region.lower[axis] <= midpoint <= region.upper[axis]:
                limit = float(region.spacing[axis])
        limits[index] = _clamp_spacing(limit, spec)
    return coords, limits


def _realize_graded_grid(design: Any, spec: GridSpec) -> RectilinearGrid:
    if spec.wavelength is None:
        raise ValueError("GridSpec.auto requires wavelength.")
    wavelength = float(spec.wavelength)
    extents = _design_extents(design)
    coordinate_offset = _design_coordinate_offset(design)
    regions = _geometry_regions(design, spec, coordinate_offset)
    background_spacing = _clamp_spacing(
        wavelength / (_material_index(design.background) * spec.min_steps_per_wvl),
        spec,
    )
    active_count = 2 if float(design.depth) == 0.0 else 3
    dimensions = active_count
    constraints = []
    predicted_counts = []
    minimum_spacing = np.inf
    for axis in range(active_count):
        domain_limit = extents[axis] / spec.min_steps_per_sim_size
        coords, limits = _axis_constraints(
            axis,
            extents[axis],
            min(background_spacing, domain_limit),
            regions,
            spec,
            coordinate_offset[axis],
        )
        constraints.append((coords, limits))
        minimum_spacing = min(minimum_spacing, float(np.min(limits)))
        predicted_counts.append(
            sum(
                max(1, math.ceil(float(length) / float(limit) - 1e-12))
                for length, limit in zip(np.diff(coords), limits, strict=True)
            )
        )
    if active_count == 2:
        predicted_counts.append(1)
    _validate_grid_shape_budget(
        cast(tuple[int, int, int], tuple(predicted_counts)),
        spec,
        dimensions=dimensions,
        context="graded-spacing lower-bound preflight",
        minimum_spacing=float(minimum_spacing),
    )

    mesher = _GradedMesher(max_scale=spec.max_scale)
    edges = []
    for axis, (coords, limits) in enumerate(constraints):
        axis_edges = mesher.make_axis_edges(
            coords,
            limits,
            max_cells=spec.max_cells_per_axis,
            axis_name="xyz"[axis],
        )
        edges.append(axis_edges)
        predicted_counts[axis] = int(axis_edges.size - 1)
        _validate_grid_shape_budget(
            cast(tuple[int, int, int], tuple(predicted_counts)),
            spec,
            dimensions=dimensions,
            context=f"after realizing {'xyz'[axis]} axis",
            minimum_spacing=float(minimum_spacing),
        )
    if active_count == 2:
        edges.append(np.asarray([0.0, 1.0]))
    return RectilinearGrid(*edges)


__all__ = ["GridSpec", "MeshOverride"]
