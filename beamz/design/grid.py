"""Canonical realized rectilinear grid geometry for rasterization and simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from beamz.const import LIGHT_SPEED

Axis = Literal["x", "y", "z"]

_AXES: tuple[Axis, Axis, Axis] = ("x", "y", "z")


def _axis_name(axis: Axis | int) -> Axis:
    if isinstance(axis, (int, np.integer)):
        if int(axis) not in range(3):
            raise ValueError(f"Grid axis index must be 0, 1, or 2; got {axis!r}.")
        return _AXES[int(axis)]
    normalized = str(axis).lower()
    if normalized not in _AXES:
        raise ValueError(f"Grid axis must be 'x', 'y', or 'z'; got {axis!r}.")
    return normalized  # type: ignore[return-value]


def _positive_counts(shape) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError("Grid shape must contain three positive integer counts.")
    counts = tuple(int(value) for value in shape)
    if any(
        count <= 0 or value != count for value, count in zip(shape, counts, strict=True)
    ):
        raise ValueError("Grid shape must contain three positive integer counts.")
    return counts  # type: ignore[return-value]


def _readonly_edges(values, name: str) -> np.ndarray:
    edges = np.array(values, dtype=np.float64, copy=True)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"{name} must be a one-dimensional edge array.")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    edges.setflags(write=False)
    return edges


def _widths_are_equal(widths: np.ndarray, reference: float, edges: np.ndarray) -> bool:
    # Absolute tolerances classify photonic-scale stretched meshes as uniform. Scale
    # the roundoff floor with the coordinate magnitude instead.
    roundoff = (
        8.0 * np.finfo(edges.dtype).eps * float(np.max(np.abs(edges), initial=0.0))
    )
    return bool(
        np.max(np.abs(widths - reference), initial=0.0)
        <= 1e-12 * abs(reference) + roundoff
    )


@dataclass(frozen=True, slots=True, eq=False)
class RectilinearGrid:
    """Realized Cartesian grid described by physical XYZ cell edges in metres.

    Uniform and nonuniform simulations share this representation. Grid-generation
    policies may decide where the edges belong, but numerical consumers derive all
    coordinates, distances, integration weights, and CFL limits from these arrays.
    """

    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray

    def __post_init__(self) -> None:
        for name in ("x_edges", "y_edges", "z_edges"):
            object.__setattr__(self, name, _readonly_edges(getattr(self, name), name))

    @classmethod
    def uniform(
        cls,
        minimum: tuple[float, float, float],
        maximum: tuple[float, float, float],
        shape: tuple[int, int, int],
    ) -> RectilinearGrid:
        """Realize equal-width cells between explicit lower and upper corners."""
        if len(minimum) != 3 or len(maximum) != 3 or len(shape) != 3:
            raise ValueError("Uniform-grid bounds and shape must have length 3.")
        counts = _positive_counts(shape)
        if any(
            not np.isfinite((lower, upper)).all() or float(upper) <= float(lower)
            for lower, upper in zip(minimum, maximum, strict=True)
        ):
            raise ValueError("Uniform-grid bounds must be finite and increasing.")
        return cls(
            *(
                np.linspace(
                    float(minimum[index]),
                    float(maximum[index]),
                    counts[index] + 1,
                    dtype=np.float64,
                )
                for index in range(3)
            )
        )

    @classmethod
    def from_spacing(
        cls,
        shape: tuple[int, int, int],
        spacing: float | tuple[float, float, float],
        *,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> RectilinearGrid:
        """Realize constant per-axis spacings from a physical lower corner."""
        counts = _positive_counts(shape)
        spacings = (
            (float(spacing),) * 3
            if np.asarray(spacing).ndim == 0
            else tuple(float(value) for value in spacing)
        )
        if len(spacings) != 3 or any(
            not np.isfinite(value) or value <= 0.0 for value in spacings
        ):
            raise ValueError("Grid spacing must contain three positive finite values.")
        lower = tuple(float(value) for value in origin)
        if len(lower) != 3 or not np.all(np.isfinite(lower)):
            raise ValueError("Grid origin must contain three finite coordinates.")
        return cls(
            *(
                lower[index]
                + spacings[index] * np.arange(counts[index] + 1, dtype=np.float64)
                for index in range(3)
            )
        )

    @property
    def edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.x_edges, self.y_edges, self.z_edges

    def axis_edges(self, axis: Axis | int) -> np.ndarray:
        return getattr(self, f"{_axis_name(axis)}_edges")

    def cell_widths(self, axis: Axis | int) -> np.ndarray:
        return np.diff(self.axis_edges(axis))

    def centers(self, axis: Axis | int) -> np.ndarray:
        edges = self.axis_edges(axis)
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def shape(self) -> tuple[int, int, int]:
        """Cell counts in public ``(x, y, z)`` order."""
        return tuple(int(edges.size - 1) for edges in self.edges)  # type: ignore[return-value]

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.shape[::-1]

    @property
    def is_axis_uniform(self) -> bool:
        return all(
            _widths_are_equal(
                self.cell_widths(axis), self.cell_widths(axis)[0], self.axis_edges(axis)
            )
            for axis in _AXES
        )

    @property
    def is_uniform(self) -> bool:
        reference = float(self.cell_widths("x")[0])
        return all(
            _widths_are_equal(self.cell_widths(axis), reference, self.axis_edges(axis))
            for axis in _AXES
        )

    @property
    def metric_kind(
        self,
    ) -> Literal["isotropic_uniform", "axis_uniform", "rectilinear"]:
        return self.metric_kind_for(_AXES)

    def metric_kind_for(
        self, active_axes: tuple[Axis, ...]
    ) -> Literal["isotropic_uniform", "axis_uniform", "rectilinear"]:
        """Classify derivative metrics over the axes active in a solver."""
        axes = tuple(_axis_name(axis) for axis in active_axes)
        if not axes:
            raise ValueError("Metric classification requires at least one active axis.")
        widths = tuple(self.cell_widths(axis) for axis in axes)
        reference = float(widths[0][0])
        if all(
            _widths_are_equal(values, reference, self.axis_edges(axis))
            for axis, values in zip(axes, widths, strict=True)
        ):
            return "isotropic_uniform"
        if all(
            _widths_are_equal(values, float(values[0]), self.axis_edges(axis))
            for axis, values in zip(axes, widths, strict=True)
        ):
            return "axis_uniform"
        return "rectilinear"

    @property
    def uniform_spacing(self) -> float:
        if not self.is_uniform:
            raise ValueError("This operation requires one uniform grid spacing.")
        return float(self.cell_widths("x")[0])

    @property
    def min_spacings(self) -> tuple[float, float, float]:
        return tuple(float(np.min(self.cell_widths(axis))) for axis in _AXES)  # type: ignore[return-value]

    @property
    def minimum_spacing(self) -> float:
        return min(self.min_spacings)

    @property
    def origin(self) -> tuple[float, float, float]:
        return tuple(float(edges[0]) for edges in self.edges)  # type: ignore[return-value]

    @property
    def maximum(self) -> tuple[float, float, float]:
        return tuple(float(edges[-1]) for edges in self.edges)  # type: ignore[return-value]

    @property
    def extent(self) -> tuple[float, float, float]:
        return tuple(float(edges[-1] - edges[0]) for edges in self.edges)  # type: ignore[return-value]

    def translated(self, offset: tuple[float, float, float]) -> RectilinearGrid:
        """Return the same cell widths in a translated coordinate frame."""
        shift = tuple(float(value) for value in offset)
        if len(shift) != 3 or not np.all(np.isfinite(shift)):
            raise ValueError("Grid translation must contain three finite values.")
        return type(self)(
            *(edges + shift[index] for index, edges in enumerate(self.edges))
        )

    def axis_extent(
        self, axis: Axis | int, bounds: tuple[int, int] | None = None
    ) -> float:
        edges = self.axis_edges(axis)
        lower, upper = (0, edges.size - 1) if bounds is None else bounds
        return float(edges[int(upper)] - edges[int(lower)])

    def coord_to_edge_index(
        self,
        axis: Axis | int,
        coordinate: float,
        *,
        snap: Literal["nearest", "lower", "upper"] = "nearest",
    ) -> int:
        edges = self.axis_edges(axis)
        coordinate = float(coordinate)
        if snap == "lower":
            return int(
                np.clip(
                    np.searchsorted(edges, coordinate, side="right") - 1,
                    0,
                    edges.size - 1,
                )
            )
        if snap == "upper":
            return int(
                np.clip(
                    np.searchsorted(edges, coordinate, side="left"), 0, edges.size - 1
                )
            )
        if snap != "nearest":
            raise ValueError(f"Unknown grid snapping rule {snap!r}.")
        upper = int(
            np.clip(np.searchsorted(edges, coordinate, side="left"), 0, edges.size - 1)
        )
        lower = max(0, upper - 1)
        return (
            lower
            if abs(coordinate - edges[lower]) <= abs(edges[upper] - coordinate)
            else upper
        )

    def cell_volume(self) -> np.ndarray:
        dx, dy, dz = (self.cell_widths(axis) for axis in _AXES)
        return dz[:, None, None] * dy[None, :, None] * dx[None, None, :]

    def face_area(self, normal_axis: Axis | int) -> np.ndarray:
        normal = _axis_name(normal_axis)
        widths = {axis: self.cell_widths(axis) for axis in _AXES}
        transverse = tuple(axis for axis in _AXES if axis != normal)
        first, second = transverse
        return widths[first][:, None] * widths[second][None, :]

    def cfl_time_step(
        self,
        courant: float = 1.0,
        *,
        active_axes: tuple[Axis, ...] = _AXES,
    ) -> float:
        axes = tuple(_axis_name(axis) for axis in active_axes)
        if not axes:
            raise ValueError("CFL calculation requires at least one active axis.")
        inverse_metric = sum(
            1.0 / float(np.min(self.cell_widths(axis))) ** 2 for axis in axes
        )
        return float(courant) / (LIGHT_SPEED * float(np.sqrt(inverse_metric)))

    def canonical_spec(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.edges

    def __eq__(self, other) -> bool:
        if not isinstance(other, RectilinearGrid):
            return NotImplemented
        return all(
            np.array_equal(lhs, rhs)
            for lhs, rhs in zip(self.edges, other.edges, strict=True)
        )

    def __hash__(self) -> int:
        return hash(
            tuple(
                (array.shape, array.dtype.str, array.tobytes()) for array in self.edges
            )
        )


# The raster API historically exposed this shorter name. It remains an alias so
# rasterization and simulation cannot drift into competing grid representations.
Grid = RectilinearGrid


__all__ = ["Axis", "Grid", "RectilinearGrid"]
