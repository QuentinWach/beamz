"""Small public data models used by the renamed MicroMode API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index
from typing import Literal, SupportsIndex, cast

import numpy as np

AverageMethod = Literal["arithmetic", "harmonic", "geometric", "min", "max"]
SliceAxis = Literal["x", "y", 0, 1]
BoundaryCondition = Literal["pec", "pmc"]


# These dataclasses are intentionally lightweight: they validate user-facing
# arrays and options before the numerical code receives them.
@dataclass(frozen=True)
class PmlSpec:
    """Perfectly matched layer settings for mode solves."""

    num_cells: tuple[int, int] = (0, 0)
    sigma_max: float = 2.0
    kappa_min: float = 1.0
    kappa_max: float = 3.0
    order: int = 3

    def __post_init__(self) -> None:
        """Normalize constructor inputs and enforce model invariants."""
        # First normalize the two edge counts because they control PML indexing.
        if len(self.num_cells) != 2:
            raise ValueError("num_cells must contain two non-negative integers")
        num_cells = (
            _coerce_integral("num_cells", self.num_cells[0], minimum=0),
            _coerce_integral("num_cells", self.num_cells[1], minimum=0),
        )
        object.__setattr__(self, "num_cells", num_cells)

        # Stretch parameters must be positive scalars; zero would remove the
        # damping profile or create invalid divisions in the PML factors.
        for name in ("sigma_max", "kappa_min", "kappa_max"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

        # The outer PML stretch should never be weaker than the inner stretch.
        if self.kappa_max < self.kappa_min:
            raise ValueError("kappa_max must be greater than or equal to kappa_min")
        object.__setattr__(
            self, "order", _coerce_integral("order", self.order, minimum=1)
        )

    @classmethod
    def from_num_cells(cls, num_cells: tuple[int, int]) -> PmlSpec:
        """Build a PML specification from x/y cell counts."""
        return cls(num_cells=num_cells)

    def as_dict(self) -> dict[str, float | int | tuple[int, int]]:
        """Return a JSON-friendly representation of the specification."""
        return {
            "num_cells": self.num_cells,
            "sigma_max": self.sigma_max,
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
            "order": self.order,
        }

    def profile_dict(self) -> dict[str, float | int]:
        """Return only the scalar stretch-profile settings used by the solver."""
        return {
            "sigma_max": self.sigma_max,
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
            "order": self.order,
        }


@dataclass(frozen=True)
class BoundarySpec:
    """Low-edge electric/magnetic wall settings for the mode plane."""

    low: tuple[BoundaryCondition, BoundaryCondition] = ("pec", "pec")

    def __post_init__(self) -> None:
        """Normalize constructor inputs and enforce model invariants."""
        # Keep only the low-edge symmetry flags; high edges are handled by the
        # open-domain derivative/PML construction.
        if len(self.low) != 2:
            raise ValueError("low must contain two boundary conditions")
        normalized = tuple(str(value).lower() for value in self.low)

        # Reject misspellings early so they do not silently become PEC walls.
        unknown = set(normalized).difference({"pec", "pmc"})
        if unknown:
            raise ValueError("boundary conditions must be 'pec' or 'pmc'")
        object.__setattr__(self, "low", normalized)

    @property
    def dmin_pmc(self) -> tuple[bool, bool]:
        """Return whether each low-edge boundary uses magnetic symmetry."""
        return self.low[0] == "pmc", self.low[1] == "pmc"

    @property
    def dmin_pml(self) -> tuple[bool, bool]:
        """Return whether each low-edge boundary is open to PML stretching."""
        return self.low[0] == "pec", self.low[1] == "pec"

    def as_dict(self) -> dict[str, tuple[str, str]]:
        """Return a JSON-friendly representation of the specification."""
        return {"low": self.low}


@dataclass(frozen=True)
class Grid:
    """Mode-plane grid metadata for rasterized material inputs.

    The two edge arrays describe the two tangential axes of the mode plane in
    microns. Beamz can pass its own rasterized grid directly through this
    object without involving any geometry code in MicroMode.
    """

    x_edges: tuple[float, ...]
    y_edges: tuple[float, ...]
    normal_axis: Literal[0, 1, 2] = 2
    normal_coordinate: float = 0.0

    def __post_init__(self) -> None:
        """Normalize constructor inputs and enforce model invariants."""
        # Store immutable float tuples so downstream validation and metadata
        # serialization see one consistent representation.
        x_edges = tuple(float(value) for value in self.x_edges)
        y_edges = tuple(float(value) for value in self.y_edges)
        object.__setattr__(self, "x_edges", x_edges)
        object.__setattr__(self, "y_edges", y_edges)

        # The normal axis determines how local solver fields map back to global
        # x/y/z components, so it must be one of the Cartesian axes.
        if self.normal_axis not in {0, 1, 2}:
            raise ValueError("normal_axis must be 0, 1, or 2")

        # Edge arrays define cell widths; non-monotonic values would produce
        # negative or zero finite-difference spacings.
        for name, values in {"x_edges": x_edges, "y_edges": y_edges}.items():
            if len(values) < 2:
                raise ValueError(f"{name} must contain at least two values")
            array = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(array)) or np.any(np.diff(array) <= 0):
                raise ValueError(f"{name} must be finite and strictly increasing")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the number of cells along the two mode-plane axes."""
        return len(self.x_edges) - 1, len(self.y_edges) - 1


@dataclass(frozen=True)
class Materials:
    """Rasterized material tensors on a :class:`Grid`.

    ``eps_tensor`` and ``mu_tensor`` use shape ``(3, 3, nx, ny)``. Diagonal
    scalar grids can be created with :meth:`from_diagonal`; full tensor grids
    from Beamz can use :meth:`from_components`.
    """

    grid: Grid
    eps_tensor: np.ndarray
    mu_tensor: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Normalize constructor inputs and enforce model invariants."""
        # Epsilon is required and must already be sampled on the solver cells.
        eps_tensor = np.asarray(self.eps_tensor, dtype=np.complex128)
        if eps_tensor.shape != (3, 3, *self.grid.shape):
            raise ValueError(
                "eps_tensor must have shape (3, 3, nx, ny) matching the grid"
            )

        # Permeability defaults to identity so non-magnetic material grids stay
        # concise at call sites while the solver always receives full tensors.
        if self.mu_tensor is None:
            mu_tensor = np.zeros_like(eps_tensor)
            for axis in range(3):
                mu_tensor[axis, axis, :, :] = 1.0
        else:
            mu_tensor = np.asarray(self.mu_tensor, dtype=np.complex128)
            if mu_tensor.shape != eps_tensor.shape:
                raise ValueError("mu_tensor must have the same shape as eps_tensor")

        # NaNs and infinities poison sparse assembly, so reject them at the data
        # model boundary instead of deep inside ARPACK.
        if not np.all(np.isfinite(eps_tensor)) or not np.all(np.isfinite(mu_tensor)):
            raise ValueError("material tensors must contain finite values")
        object.__setattr__(self, "eps_tensor", eps_tensor)
        object.__setattr__(self, "mu_tensor", mu_tensor)

    @classmethod
    def from_diagonal(
        cls,
        *,
        eps_xx: np.ndarray,
        x_edges: Sequence[float],
        y_edges: Sequence[float],
        eps_yy: np.ndarray | None = None,
        eps_zz: np.ndarray | None = None,
        mu_xx: np.ndarray | None = None,
        mu_yy: np.ndarray | None = None,
        mu_zz: np.ndarray | None = None,
        normal_axis: Literal[0, 1, 2] = 2,
        normal_coordinate: float = 0.0,
    ) -> Materials:
        """Build a material grid from scalar or diagonal tensor components."""
        # Build and validate grid metadata before interpreting any component
        # arrays, because the grid shape drives every component check.
        grid = Grid(
            tuple(float(value) for value in x_edges),
            tuple(float(value) for value in y_edges),
            normal_axis=normal_axis,
            normal_coordinate=normal_coordinate,
        )

        # Missing yy/zz components inherit xx, matching isotropic scalar input.
        eps_diag = _stack_diagonal_components("eps", grid.shape, eps_xx, eps_yy, eps_zz)
        mu_diag = _stack_diagonal_components(
            "mu",
            grid.shape,
            np.ones(grid.shape, dtype=np.complex128) if mu_xx is None else mu_xx,
            np.ones(grid.shape, dtype=np.complex128) if mu_yy is None else mu_yy,
            np.ones(grid.shape, dtype=np.complex128) if mu_zz is None else mu_zz,
        )

        # Store the result in the same full-tensor layout used by tensorial
        # inputs; diagonal detection later chooses the faster solver path.
        return cls(
            grid=grid,
            eps_tensor=_diagonal_to_full_tensor(eps_diag),
            mu_tensor=_diagonal_to_full_tensor(mu_diag),
        )

    @classmethod
    def from_components(
        cls,
        *,
        x_edges: Sequence[float],
        y_edges: Sequence[float],
        eps_xx: np.ndarray,
        eps_yy: np.ndarray | None = None,
        eps_zz: np.ndarray | None = None,
        eps_xy: np.ndarray | None = None,
        eps_xz: np.ndarray | None = None,
        eps_yx: np.ndarray | None = None,
        eps_yz: np.ndarray | None = None,
        eps_zx: np.ndarray | None = None,
        eps_zy: np.ndarray | None = None,
        mu_xx: np.ndarray | None = None,
        mu_yy: np.ndarray | None = None,
        mu_zz: np.ndarray | None = None,
        mu_xy: np.ndarray | None = None,
        mu_xz: np.ndarray | None = None,
        mu_yx: np.ndarray | None = None,
        mu_yz: np.ndarray | None = None,
        mu_zx: np.ndarray | None = None,
        mu_zy: np.ndarray | None = None,
        normal_axis: Literal[0, 1, 2] = 2,
        normal_coordinate: float = 0.0,
    ) -> Materials:
        """Build a material grid from diagonal and off-diagonal tensor components."""
        # Full component construction shares the same grid validation as the
        # diagonal constructor, then overlays optional off-diagonal entries.
        grid = Grid(
            tuple(float(value) for value in x_edges),
            tuple(float(value) for value in y_edges),
            normal_axis=normal_axis,
            normal_coordinate=normal_coordinate,
        )

        # Start with diagonal tensors so scalar and anisotropic grids follow one
        # predictable path before optional coupling components are assigned.
        eps_diag = _stack_diagonal_components("eps", grid.shape, eps_xx, eps_yy, eps_zz)
        eps_tensor = _diagonal_to_full_tensor(eps_diag)
        _assign_tensor_offdiagonal(
            "eps",
            eps_tensor,
            grid.shape,
            xy=eps_xy,
            xz=eps_xz,
            yx=eps_yx,
            yz=eps_yz,
            zx=eps_zx,
            zy=eps_zy,
        )

        # Magnetic coupling components are optional; absent mu entries represent
        # identity permeability on the diagonal and zero off-diagonal coupling.
        mu_diag = _stack_diagonal_components(
            "mu",
            grid.shape,
            np.ones(grid.shape, dtype=np.complex128) if mu_xx is None else mu_xx,
            np.ones(grid.shape, dtype=np.complex128) if mu_yy is None else mu_yy,
            np.ones(grid.shape, dtype=np.complex128) if mu_zz is None else mu_zz,
        )
        mu_tensor = _diagonal_to_full_tensor(mu_diag)
        _assign_tensor_offdiagonal(
            "mu",
            mu_tensor,
            grid.shape,
            xy=mu_xy,
            xz=mu_xz,
            yx=mu_yx,
            yz=mu_yz,
            zx=mu_zx,
            zy=mu_zy,
        )

        # The dataclass constructor performs the final finite-value and shape
        # validation on the assembled full tensors.
        return cls(grid=grid, eps_tensor=eps_tensor, mu_tensor=mu_tensor)

    @classmethod
    def from_slice(
        cls,
        *,
        coord_edges: Sequence[float],
        eps_xx: np.ndarray,
        axis: SliceAxis = "x",
        invariant_width: float = 1.0,
        invariant_coordinate: float = 0.0,
        eps_yy: np.ndarray | None = None,
        eps_zz: np.ndarray | None = None,
        eps_xy: np.ndarray | None = None,
        eps_xz: np.ndarray | None = None,
        eps_yx: np.ndarray | None = None,
        eps_yz: np.ndarray | None = None,
        eps_zx: np.ndarray | None = None,
        eps_zy: np.ndarray | None = None,
        mu_xx: np.ndarray | None = None,
        mu_yy: np.ndarray | None = None,
        mu_zz: np.ndarray | None = None,
        mu_xy: np.ndarray | None = None,
        mu_xz: np.ndarray | None = None,
        mu_yx: np.ndarray | None = None,
        mu_yz: np.ndarray | None = None,
        mu_zx: np.ndarray | None = None,
        mu_zy: np.ndarray | None = None,
        normal_axis: Literal[0, 1, 2] = 2,
        normal_coordinate: float = 0.0,
    ) -> Materials:
        """Build a one-dimensional mode slice.

        ``axis="x"`` makes the supplied material arrays vary along the first
        mode-plane axis and inserts one invariant cell along the second axis.
        ``axis="y"`` does the opposite. The invariant cell has
        ``invariant_width`` in microns so integrations and unnormalized overlap
        values keep the expected physical scale.
        """

        axis_index = _normalize_slice_axis(axis)
        edge_values = tuple(float(value) for value in coord_edges)

        # A slice is still represented as a 2D mode plane, so one axis gets the
        # user coordinates and the other gets a single finite-width cell.
        if len(edge_values) < 2:
            raise ValueError("coord_edges must contain at least two values")
        if invariant_width <= 0.0 or not np.isfinite(invariant_width):
            raise ValueError("invariant_width must be finite and positive")
        invariant_edges = (
            float(invariant_coordinate) - 0.5 * float(invariant_width),
            float(invariant_coordinate) + 0.5 * float(invariant_width),
        )
        x_edges = edge_values if axis_index == 0 else invariant_edges
        y_edges = invariant_edges if axis_index == 0 else edge_values
        cell_count = len(edge_values) - 1

        def expand(label: str, values: np.ndarray | None) -> np.ndarray | None:
            """Expand a one-dimensional component onto the padded mode-plane grid."""
            # Optional components stay absent so from_components can apply its
            # diagonal/permeability defaults in one place.
            if values is None:
                return None
            array = np.asarray(values, dtype=np.complex128)
            if array.shape != (cell_count,):
                raise ValueError(
                    f"{label} must have shape {(cell_count,)} for a one-dimensional slice"
                )

            # Insert the singleton invariant axis on the side implied by the
            # requested varying axis.
            return array[:, None] if axis_index == 0 else array[None, :]

        expanded_eps_xx = expand("eps_xx", eps_xx)
        if expanded_eps_xx is None:
            raise ValueError("eps_xx is required")

        # Delegate to the full component constructor so slice and grid inputs
        # share tensor assembly and validation behavior.
        return cls.from_components(
            x_edges=x_edges,
            y_edges=y_edges,
            eps_xx=expanded_eps_xx,
            eps_yy=expand("eps_yy", eps_yy),
            eps_zz=expand("eps_zz", eps_zz),
            eps_xy=expand("eps_xy", eps_xy),
            eps_xz=expand("eps_xz", eps_xz),
            eps_yx=expand("eps_yx", eps_yx),
            eps_yz=expand("eps_yz", eps_yz),
            eps_zx=expand("eps_zx", eps_zx),
            eps_zy=expand("eps_zy", eps_zy),
            mu_xx=expand("mu_xx", mu_xx),
            mu_yy=expand("mu_yy", mu_yy),
            mu_zz=expand("mu_zz", mu_zz),
            mu_xy=expand("mu_xy", mu_xy),
            mu_xz=expand("mu_xz", mu_xz),
            mu_yx=expand("mu_yx", mu_yx),
            mu_yz=expand("mu_yz", mu_yz),
            mu_zx=expand("mu_zx", mu_zx),
            mu_zy=expand("mu_zy", mu_zy),
            normal_axis=normal_axis,
            normal_coordinate=normal_coordinate,
        )

    @classmethod
    def from_subpixel_diagonal(
        cls,
        *,
        eps_xx: np.ndarray,
        x_edges: Sequence[float],
        y_edges: Sequence[float],
        subpixel_shape: tuple[int, int],
        eps_yy: np.ndarray | None = None,
        eps_zz: np.ndarray | None = None,
        mu_xx: np.ndarray | None = None,
        mu_yy: np.ndarray | None = None,
        mu_zz: np.ndarray | None = None,
        average: AverageMethod = "arithmetic",
        normal_axis: Literal[0, 1, 2] = 2,
        normal_coordinate: float = 0.0,
    ) -> Materials:
        """Build a grid from higher-resolution samples inside each solver cell.

        The sample arrays may either have shape ``(nx * sx, ny * sy)`` or
        ``(nx, ny, sx, sy)``, where ``(sx, sy)`` is ``subpixel_shape``. This is
        useful when BeamZ rasterizes geometry at a finer resolution than the
        mode-solver grid and wants deterministic cell averaging before solving.
        """

        x_edge_values = tuple(float(value) for value in x_edges)
        y_edge_values = tuple(float(value) for value in y_edges)
        shape = (len(x_edge_values) - 1, len(y_edge_values) - 1)

        # Average each supplied high-resolution component independently before
        # reusing the ordinary diagonal tensor constructor.
        averaged = {
            "eps_xx": cls.average_subpixels(
                eps_xx, shape=shape, subpixel_shape=subpixel_shape, method=average
            ),
            "eps_yy": None
            if eps_yy is None
            else cls.average_subpixels(
                eps_yy, shape=shape, subpixel_shape=subpixel_shape, method=average
            ),
            "eps_zz": None
            if eps_zz is None
            else cls.average_subpixels(
                eps_zz, shape=shape, subpixel_shape=subpixel_shape, method=average
            ),
            "mu_xx": None
            if mu_xx is None
            else cls.average_subpixels(
                mu_xx, shape=shape, subpixel_shape=subpixel_shape, method=average
            ),
            "mu_yy": None
            if mu_yy is None
            else cls.average_subpixels(
                mu_yy, shape=shape, subpixel_shape=subpixel_shape, method=average
            ),
            "mu_zz": None
            if mu_zz is None
            else cls.average_subpixels(
                mu_zz, shape=shape, subpixel_shape=subpixel_shape, method=average
            ),
        }
        return cls.from_diagonal(
            x_edges=x_edge_values,
            y_edges=y_edge_values,
            normal_axis=normal_axis,
            normal_coordinate=normal_coordinate,
            **averaged,
        )

    @staticmethod
    def average_subpixels(
        values: np.ndarray,
        *,
        shape: tuple[int, int],
        subpixel_shape: tuple[int, int],
        method: AverageMethod = "arithmetic",
    ) -> np.ndarray:
        """Average high-resolution cell samples down to a solver grid.

        ``values`` may be supplied as ``(nx * sx, ny * sy)`` or
        ``(nx, ny, sx, sy)``. ``harmonic`` averaging is useful for some
        interface-normal material components; ``arithmetic`` is the default
        because it preserves a straightforward fill-fraction interpretation.
        """

        nx, ny = (int(shape[0]), int(shape[1]))
        sx, sy = (int(subpixel_shape[0]), int(subpixel_shape[1]))

        # Validate the target grouping before reshaping so shape errors are
        # reported in terms of solver cells and subpixel samples.
        if nx <= 0 or ny <= 0:
            raise ValueError("shape must contain positive cell counts")
        if sx <= 0 or sy <= 0:
            raise ValueError("subpixel_shape must contain positive sample counts")
        array = np.asarray(values, dtype=np.complex128)

        # Accept either a dense raster or an already grouped raster; both become
        # (nx, ny, sx, sy) before the selected averaging rule is applied.
        if array.shape == (nx * sx, ny * sy):
            grouped = array.reshape(nx, sx, ny, sy).transpose(0, 2, 1, 3)
        elif array.shape == (nx, ny, sx, sy):
            grouped = array
        else:
            raise ValueError(
                f"subpixel values must have shape {(nx * sx, ny * sy)} or {(nx, ny, sx, sy)}"
            )

        # Choose the averaging rule explicitly so interface-sensitive callers can
        # request harmonic/geometric behavior without changing solver code.
        if method == "arithmetic":
            return grouped.mean(axis=(2, 3))
        if method == "harmonic":
            if np.any(np.isclose(grouped, 0.0)):
                raise ValueError("harmonic subpixel averaging requires nonzero samples")
            return 1.0 / np.mean(1.0 / grouped, axis=(2, 3))
        if method == "geometric":
            if np.any(np.real(grouped) <= 0.0) or np.any(
                np.abs(np.imag(grouped)) > 1e-14
            ):
                raise ValueError(
                    "geometric subpixel averaging requires positive real samples"
                )
            return np.exp(np.mean(np.log(grouped.real), axis=(2, 3))).astype(
                np.complex128
            )
        if method == "min":
            return grouped.min(axis=(2, 3))
        if method == "max":
            return grouped.max(axis=(2, 3))
        raise ValueError(
            "method must be one of 'arithmetic', 'harmonic', 'geometric', 'min', or 'max'"
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the number of cells along the two mode-plane axes."""
        return self.grid.shape

    @property
    def is_diagonal(self) -> bool:
        """Return whether epsilon and mu contain only diagonal components."""
        # A boolean mask lets the check stay independent of grid shape and
        # catches off-diagonal entries across every solver cell.
        off_diagonal = np.ones((3, 3), dtype=bool)
        np.fill_diagonal(off_diagonal, False)
        mu_tensor = self._resolved_mu_tensor()
        return bool(
            np.all(np.abs(self.eps_tensor[off_diagonal]) <= 1e-12)
            and np.all(np.abs(mu_tensor[off_diagonal]) <= 1e-12)
        )

    def flat_eps_tensor(self) -> np.ndarray:
        """Return epsilon as flattened per-cell 3x3 tensors."""
        return self.eps_tensor.reshape(3, 3, -1)

    def flat_mu_tensor(self) -> np.ndarray:
        """Return mu as flattened per-cell 3x3 tensors."""
        return self._resolved_mu_tensor().reshape(3, 3, -1)

    def _resolved_mu_tensor(self) -> np.ndarray:
        """Return the initialized permeability tensor."""
        if self.mu_tensor is None:
            raise RuntimeError("mu_tensor was not initialized")
        return self.mu_tensor

    def diagonal_eps(self) -> np.ndarray:
        """Return the three diagonal epsilon components as a stacked array."""
        return np.stack([self.eps_tensor[axis, axis] for axis in range(3)], axis=0)


def _stack_diagonal_components(
    label: str,
    shape: tuple[int, int],
    xx: np.ndarray,
    yy: np.ndarray | None,
    zz: np.ndarray | None,
) -> np.ndarray:
    """Validate and stack x/y/z diagonal tensor components."""
    # xx is the required scalar/diagonal component; yy and zz inherit it unless
    # anisotropic values are provided.
    xx_array = np.asarray(xx, dtype=np.complex128)
    if xx_array.shape != shape:
        raise ValueError(f"{label}_xx must have shape {shape}")
    yy_array = xx_array if yy is None else np.asarray(yy, dtype=np.complex128)
    zz_array = xx_array if zz is None else np.asarray(zz, dtype=np.complex128)

    # All diagonal components must be cell-centered arrays matching the grid.
    if yy_array.shape != shape or zz_array.shape != shape:
        raise ValueError(
            f"{label}_xx, {label}_yy, and {label}_zz must have shape {shape}"
        )
    return np.stack([xx_array, yy_array, zz_array], axis=0)


def _coerce_integral(name: str, value: object, *, minimum: int) -> int:
    """Coerce index-like values while rejecting floats and booleans."""
    # bool implements the integer protocol, but accepting it would make option
    # typos like num_cells=(True, False) look valid.
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must contain integers")
    try:
        integer = index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise ValueError(f"{name} must contain integers") from exc

    # Keep the minimum check here so every PML/order caller reports consistent
    # validation messages.
    if integer < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must contain non-negative integers")
        raise ValueError(f"{name} must be positive")
    return int(integer)


def _normalize_slice_axis(axis: SliceAxis) -> int:
    """Map slice-axis aliases onto a zero-based axis index."""
    if axis in {"x", 0}:
        return 0
    if axis in {"y", 1}:
        return 1
    raise ValueError("axis must be 'x', 'y', 0, or 1")


def _diagonal_to_full_tensor(diagonal: np.ndarray) -> np.ndarray:
    """Expand diagonal components into a full 3x3 tensor grid."""
    # The solver consumes a full tensor layout even for diagonal materials; all
    # off-diagonal entries remain zero.
    tensor = np.zeros((3, 3, *diagonal.shape[1:]), dtype=np.complex128)
    for axis in range(3):
        tensor[axis, axis, :, :] = diagonal[axis]
    return tensor


def _assign_tensor_offdiagonal(
    label: str,
    tensor: np.ndarray,
    shape: tuple[int, int],
    *,
    xy: np.ndarray | None,
    xz: np.ndarray | None,
    yx: np.ndarray | None,
    yz: np.ndarray | None,
    zx: np.ndarray | None,
    zy: np.ndarray | None,
) -> None:
    """Validate and assign optional off-diagonal tensor components."""
    # Iterate through physical tensor suffixes instead of open-coding each
    # assignment, which keeps validation messages and row/column mapping aligned.
    for (row, col), suffix, values in [
        ((0, 1), "xy", xy),
        ((0, 2), "xz", xz),
        ((1, 0), "yx", yx),
        ((1, 2), "yz", yz),
        ((2, 0), "zx", zx),
        ((2, 1), "zy", zy),
    ]:
        if values is None:
            continue
        array = np.asarray(values, dtype=np.complex128)

        # Off-diagonal components are cell-centered just like the diagonal
        # components; no broadcasting is allowed because that can hide mistakes.
        if array.shape != shape:
            raise ValueError(f"{label}_{suffix} must have shape {shape}")
        tensor[row, col, :, :] = array


@dataclass(frozen=True)
class Spec:
    """Mode solver options for grid solves."""

    num_modes: int = 1
    target_neff: float | None = None
    pml: PmlSpec | tuple[int, int] | None = None
    boundary: BoundarySpec | tuple[BoundaryCondition, BoundaryCondition] | None = None
    angle_theta: float = 0.0
    angle_phi: float = 0.0
    bend_radius: float | None = None
    bend_axis: Literal[0, 1] | None = None

    def __post_init__(self) -> None:
        """Normalize constructor inputs and enforce model invariants."""
        # Validate simple scalar controls before expanding nested model objects.
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")
        if self.target_neff is not None and self.target_neff <= 0:
            raise ValueError("target_neff must be positive")

        # Accept compact tuple inputs at the API boundary, but store canonical
        # model objects so solve_modes can unpack a Spec without extra branches.
        pml = self.pml
        pml = PmlSpec() if pml is None else pml
        if not isinstance(pml, PmlSpec):
            pml = PmlSpec.from_num_cells(pml)
        object.__setattr__(self, "pml", pml)
        boundary = BoundarySpec() if self.boundary is None else self.boundary
        if not isinstance(boundary, BoundarySpec):
            boundary = BoundarySpec(low=boundary)
        object.__setattr__(self, "boundary", boundary)

        # Bend transforms need both a radius and an axis because the Jacobian
        # depends on which transverse coordinate measures curvature.
        if self.bend_radius is not None and np.isclose(self.bend_radius, 0.0):
            raise ValueError("bend_radius magnitude must be larger than 0")
        if self.bend_radius is not None and self.bend_axis is None:
            raise ValueError("bend_axis must be set when bend_radius is set")
        if self.bend_axis is not None and self.bend_axis not in {0, 1}:
            raise ValueError("bend_axis must be 0 or 1")

    @property
    def has_angle(self) -> bool:
        """Return whether an angular coordinate transform is requested."""
        return abs(float(self.angle_theta)) > 0.0

    @property
    def has_bend(self) -> bool:
        """Return whether a bend transform is requested."""
        return self.bend_radius is not None

    @property
    def has_transform(self) -> bool:
        """Return whether any coordinate transform is requested."""
        return self.has_angle or self.has_bend
