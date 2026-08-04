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
        )

    @classmethod
    def uniform(cls, resolution: float, *, courant: float = 0.99) -> GridSpec:
        """Create a grid specification with an explicit uniform cell size.

        Returns
        -------
        GridSpec
            Immutable uniform-grid policy.
        """
        return cls(resolution=float(resolution), courant=float(courant))

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
            return _realize_graded_grid(design, self)
        return _realize_uniform_grid(design, self._target_spacing())

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


def _realize_uniform_grid(design: Any, resolution: float) -> RectilinearGrid:
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
    if float(design.depth) > 0.0:
        nz = count(extents[2])
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


def _polygon_opposing_width(structure: Polygon) -> float | None:
    """Return the narrowest reliable distance between opposing material faces.

    Boundary vertex spacing is deliberately excluded: refining a curved polygon's
    tessellation must not change its physical mesh target.
    """
    from shapely.geometry import LineString
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry.polygon import orient
    from shapely.ops import nearest_points

    shell = [(float(x), float(y)) for x, y, _z in structure.vertices]
    holes = [
        [(float(x), float(y)) for x, y, _z in path] for path in structure.interiors
    ]
    geometry = orient(ShapelyPolygon(shell, holes=holes), sign=1.0)
    if geometry.is_empty or not geometry.is_valid:
        return None
    if not geometry.interiors and geometry.equals(geometry.convex_hull):
        coordinates = np.asarray(geometry.exterior.coords[:-1], dtype=np.float64)
        edges = np.roll(coordinates, -1, axis=0) - coordinates
        edge_lengths = np.linalg.norm(edges, axis=1)
        normals = np.column_stack((-edges[:, 1], edges[:, 0])) / edge_lengths[:, None]
        width = min(float(np.ptp(coordinates @ normal)) for normal in normals)
        return width if np.isfinite(width) and width > 0.0 else None

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    ring_ids: list[int] = []
    local_ids: list[int] = []
    ring_sizes: list[int] = []
    rings = (geometry.exterior, *geometry.interiors)
    for ring_id, ring in enumerate(rings):
        coordinates = np.asarray(ring.coords, dtype=np.float64)
        ring_sizes.append(coordinates.shape[0] - 1)
        for local_id, (start, end) in enumerate(
            zip(coordinates[:-1], coordinates[1:], strict=True)
        ):
            if np.linalg.norm(end - start) <= np.finfo(float).tiny:
                continue
            starts.append(start)
            ends.append(end)
            ring_ids.append(ring_id)
            local_ids.append(local_id)
    if len(starts) < 2:
        return None

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
    best = np.inf

    for index in range(len(starts)):
        delta = midpoints - midpoints[index]
        distances = np.linalg.norm(delta, axis=1)
        valid = np.arange(len(starts)) > index
        same_ring = ring_ids_array == ring_ids_array[index]
        local_delta = np.abs(local_ids_array - local_ids_array[index])
        ring_size = ring_sizes[ring_ids[index]]
        adjacent = same_ring & ((local_delta <= 1) | (local_delta >= ring_size - 1))
        valid &= ~adjacent
        valid &= np.einsum("ij,j->i", inward_normals, inward_normals[index]) < -0.95
        nonzero = distances > tolerance
        valid &= nonzero
        safe_distances = np.where(nonzero, distances, 1.0)
        valid &= (
            np.einsum("ij,j->i", delta, inward_normals[index]) / safe_distances > 0.8
        )
        valid &= np.einsum("ij,ij->i", -delta, inward_normals) / safe_distances > 0.8
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            continue
        candidates = candidates[np.argsort(distances[candidates])[:8]]
        first = LineString((start_array[index], end_array[index]))
        for other in candidates:
            second = LineString((start_array[other], end_array[other]))
            point_a, point_b = nearest_points(first, second)
            distance = float(point_a.distance(point_b))
            if not tolerance < distance < best:
                continue
            connector = LineString((point_a, point_b))
            if geometry.covers(connector):
                best = distance

    return float(best) if np.isfinite(best) and best > 0.0 else None


def _feature_sizes(
    structure: Any, *, detect_polygon_features: bool
) -> tuple[float | None, float | None, float | None]:
    if isinstance(structure, Polygon) and detect_polygon_features:
        clearance = _polygon_opposing_width(structure)
        depth = float(structure.depth)
        return clearance, clearance, depth if depth > 0.0 else None
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
        features = _feature_sizes(
            structure, detect_polygon_features=spec.min_feature_cells > 1.0
        )
        spacing = tuple(
            _clamp_spacing(
                min(material_spacing, feature / spec.min_feature_cells)
                if feature is not None
                else material_spacing,
                spec,
            )
            for feature in features
        )
        regions.append(_Region(lower, upper, cast(OptionalAxisValues, spacing)))

    active_count = 2 if two_dimensional else 3
    for first_index, first in enumerate(tuple(regions)):
        for second in tuple(regions)[first_index + 1 :]:
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
    mesher = _GradedMesher(max_scale=spec.max_scale)
    active_count = 2 if float(design.depth) == 0.0 else 3
    edges = []
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
        edges.append(mesher.make_axis_edges(coords, limits))
    if active_count == 2:
        edges.append(np.asarray([0.0, 1.0]))
    return RectilinearGrid(*edges)


__all__ = ["GridSpec", "MeshOverride"]
