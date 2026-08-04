"""Physics- and geometry-aware solver grid-spacing policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.design.grid import RectilinearGrid
from beamz.design.mesher import GradedMesher


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
            spacing: tuple[float | None, ...] = (float(self.dl),) * 3  # type: ignore[arg-type]
        else:
            raw = tuple(self.dl)
            if len(raw) == 2:
                raw = (*raw, None)
            if len(raw) != 3:
                raise ValueError("MeshOverride dl must contain two or three values.")
            spacing = tuple(None if value is None else float(value) for value in raw)
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
    nonuniform : bool, default=False
        Generate a geometry-aware rectilinear grid instead of a uniform grid.
        Prefer :meth:`graded` when constructing this policy directly.
    max_scale : float, default=1.3
        Hard maximum ratio between neighboring cell widths on a graded axis.
    min_steps_per_sim_size : float, default=10.0
        Minimum number of cells across each active simulation dimension.
    min_feature_cells : float, default=4.0
        Minimum target cells across detected structure thicknesses and gaps.
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
    nonuniform: bool = False
    max_scale: float = 1.3
    min_steps_per_sim_size: float = 10.0
    min_feature_cells: float = 4.0
    dl_min: float | None = None
    dl_max: float | None = None
    overrides: tuple[MeshOverride, ...] = ()
    snapping_points: tuple[tuple[float | None, ...], ...] = ()

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
        if self.nonuniform and not 1.0 < float(self.max_scale) < 2.0:
            raise ValueError(
                "A nonuniform GridSpec max_scale must be strictly between 1 and 2."
            )
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
        nonuniform: bool = False,
        max_scale: float = 1.3,
        min_steps_per_sim_size: float = 10.0,
        min_feature_cells: float = 4.0,
        dl_min: float | None = None,
        dl_max: float | None = None,
        overrides: tuple[MeshOverride, ...] = (),
        snapping_points: tuple[tuple[float | None, ...], ...] = (),
    ) -> GridSpec:
        """Create a wavelength-driven automatic grid specification.

        Returns
        -------
        GridSpec
            Immutable wavelength-driven grid policy.
        """
        return cls(
            min_steps_per_wvl=float(min_steps_per_wvl),
            wavelength=wavelength,
            courant=float(courant),
            nonuniform=bool(nonuniform),
            max_scale=float(max_scale),
            min_steps_per_sim_size=float(min_steps_per_sim_size),
            min_feature_cells=float(min_feature_cells),
            dl_min=dl_min,
            dl_max=dl_max,
            overrides=overrides,
            snapping_points=snapping_points,
        )

    @classmethod
    def graded(
        cls,
        *,
        wavelength: float,
        min_steps_per_wvl: float = 10.0,
        max_scale: float = 1.3,
        min_steps_per_sim_size: float = 10.0,
        min_feature_cells: float = 4.0,
        dl_min: float | None = None,
        dl_max: float | None = None,
        overrides: tuple[MeshOverride, ...] = (),
        snapping_points: tuple[tuple[float | None, ...], ...] = (),
        courant: float = 0.99,
    ) -> GridSpec:
        """Create a geometry-aware, smoothly graded rectilinear grid policy."""
        return cls.auto(
            wavelength=wavelength,
            min_steps_per_wvl=min_steps_per_wvl,
            courant=courant,
            nonuniform=True,
            max_scale=max_scale,
            min_steps_per_sim_size=min_steps_per_sim_size,
            min_feature_cells=min_feature_cells,
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

    def resolve_resolution(self, *, max_index: float = 1.0) -> float:
        """Return the explicit or wavelength-derived cell size in metres.

        Returns
        -------
        float
            Uniform cell size in metres.
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

    def realize(self, design: Any) -> RectilinearGrid:
        """Realize this policy for a concrete design domain and material stack."""
        if self.nonuniform:
            if self.wavelength is None:
                raise ValueError("GridSpec.graded requires wavelength.")
            return _realize_graded_grid(design, self)
        resolution = (
            self.resolve_resolution()
            if self.resolution is not None
            else self.resolve_resolution(max_index=_maximum_design_index(design))
        )
        extents = _design_extents(design)
        shape = tuple(
            1
            if axis == 2 and float(design.depth) == 0.0
            else max(1, int(np.ceil(value / resolution)))
            for axis, value in enumerate(extents)
        )
        return RectilinearGrid.uniform((0.0, 0.0, 0.0), extents, shape)

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
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    spacing: tuple[float | None, float | None, float | None]
    enforced: bool = False


def _material_index(material: Any) -> float:
    maximum = getattr(material, "max_permittivity", None)
    if maximum is None:
        raise ValueError("Automatic meshing requires finite material permittivity.")
    value = float(np.real(maximum))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Automatic meshing requires positive material permittivity.")
    return float(np.sqrt(max(value, 1.0)))


def _maximum_design_index(design: Any) -> float:
    materials = [design.background]
    materials.extend(
        structure.material
        for structure in getattr(design, "structures", ())
        if getattr(structure, "material", None) is not None
    )
    return max(_material_index(material) for material in materials)


def _design_extents(design: Any) -> tuple[float, float, float]:
    depth = float(design.depth)
    return float(design.width), float(design.height), depth if depth > 0.0 else 1.0


def _structure_bounds(
    structure: Any, *, two_dimensional: bool
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if hasattr(structure, "lower") and hasattr(structure, "upper"):
        return tuple(map(float, structure.lower)), tuple(map(float, structure.upper))
    if hasattr(structure, "radius") and not getattr(structure, "vertices", ()):
        center = tuple(map(float, structure.center))
        radius = float(structure.radius)
        return (
            tuple(value - radius for value in center),
            tuple(value + radius for value in center),
        )  # type: ignore[return-value]
    vertices = np.asarray(getattr(structure, "vertices", ()), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3:
        raise TypeError(f"Cannot derive meshing bounds for {type(structure).__name__}.")
    lower = [float(np.min(vertices[:, 0])), float(np.min(vertices[:, 1]))]
    upper = [float(np.max(vertices[:, 0])), float(np.max(vertices[:, 1]))]
    if two_dimensional:
        return (*lower, 0.0), (*upper, 1.0)
    z = float(getattr(structure, "z", np.min(vertices[:, 2])))
    depth = float(getattr(structure, "depth", 0.0))
    return (*lower, z), (*upper, z + depth)


def _feature_sizes(structure: Any) -> tuple[float | None, float | None, float | None]:
    if hasattr(structure, "inner_radius") and hasattr(structure, "outer_radius"):
        wall = float(structure.outer_radius) - float(structure.inner_radius)
        return wall, wall, float(getattr(structure, "depth", 0.0)) or None
    if hasattr(structure, "size"):
        size = tuple(float(value) for value in structure.size)
        if len(size) == 2:
            size = (*size, 0.0)
        return tuple(
            value if np.isfinite(value) and value > 0.0 else None for value in size
        )  # type: ignore[return-value]
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


def _geometry_regions(design: Any, spec: GridSpec) -> list[_Region]:
    extents = _design_extents(design)
    two_dimensional = float(design.depth) == 0.0
    regions = []
    for structure in getattr(design, "structures", ()):
        raw_lower, raw_upper = _structure_bounds(
            structure, two_dimensional=two_dimensional
        )
        lower = tuple(max(0.0, raw_lower[axis]) for axis in range(3))
        upper = tuple(min(extents[axis], raw_upper[axis]) for axis in range(3))
        if any(
            upper[axis] <= lower[axis] for axis in range(2 if two_dimensional else 3)
        ):
            continue
        material_spacing = _clamp_spacing(
            float(spec.wavelength)
            / (_material_index(structure.material) * spec.min_steps_per_wvl),
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
        regions.append(_Region(lower, upper, spacing))

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
                        tuple(lower),
                        tuple(upper),
                        tuple(
                            gap_spacing if index == axis else None for index in range(3)
                        ),
                    )
                )

    for override in spec.overrides:
        raw_lower, raw_upper = override.bounds
        lower = tuple(max(0.0, raw_lower[axis]) for axis in range(3))
        upper = tuple(min(extents[axis], raw_upper[axis]) for axis in range(3))
        if any(upper[axis] <= lower[axis] for axis in range(active_count)):
            continue
        spacing = tuple(
            None if value is None else _clamp_spacing(value, spec)
            for value in override.dl
        )
        regions.append(_Region(lower, upper, spacing, override.enforced))
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
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = [0.0, extent]
    effective_regions = []
    for region in regions:
        if region.spacing[axis] is None:
            continue
        coordinates.extend((region.lower[axis], region.upper[axis]))
        effective_regions.append(region)
    for point in spec.snapping_points:
        if point[axis] is not None and 0.0 < point[axis] < extent:
            coordinates.append(float(point[axis]))
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
    extents = _design_extents(design)
    regions = _geometry_regions(design, spec)
    background_spacing = _clamp_spacing(
        float(spec.wavelength)
        / (_material_index(design.background) * spec.min_steps_per_wvl),
        spec,
    )
    mesher = GradedMesher(max_scale=spec.max_scale)
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
        )
        edges.append(mesher.make_axis_edges(coords, limits))
    if active_count == 2:
        edges.append(np.asarray([0.0, 1.0]))
    return RectilinearGrid(*edges)


__all__ = ["GridSpec", "MeshOverride"]
