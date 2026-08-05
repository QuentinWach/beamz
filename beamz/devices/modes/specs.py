"""BeamZ-facing immutable mode selection and result values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from beamz.devices._immutable import readonly_array


@dataclass(frozen=True)
class ModeSpec:
    """Configure mode selection for previews, sources, and monitors.

    Parameters
    ----------
    num_modes : int, default=1
        Number of candidate eigenmodes to solve; clamped to at least one.
    mode_index : int, default=0
        Zero-based selected mode after polarization-aware ordering.
    polarization : {"te", "tm"}, optional
        Preferred polarization used to order candidate modes.
    target_neff : float, optional
        Effective index near which the eigensolver should search.
    num_freqs : int, optional
        Number of frequency samples used for a broadband mode representation.

    Examples
    --------
    >>> spec = ModeSpec(num_modes=4, mode_index=1, polarization="te")

    Notes
    -----
    ``num_modes`` is the number of eigensolver candidates, whereas ``mode_index``
    selects the mode launched or projected after polarization-aware ordering.
    ``num_freqs`` controls broadband mode reconstruction and is independent of the
    frequencies configured on a monitor.
    """

    num_modes: int = 1
    mode_index: int = 0
    polarization: str | None = None
    target_neff: float | None = None
    num_freqs: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_modes", max(1, int(self.num_modes)))
        object.__setattr__(self, "mode_index", max(0, int(self.mode_index)))
        if self.num_freqs is not None:
            object.__setattr__(self, "num_freqs", max(1, int(self.num_freqs)))
        if self.polarization is not None:
            pol = str(self.polarization).lower()
            if pol not in {"te", "tm"}:
                raise ValueError(f"polarization must be 'te' or 'tm', got {pol!r}.")
            object.__setattr__(self, "polarization", pol)


@dataclass(frozen=True)
class ModeData:
    """Store immutable modal fields returned by a mode solve.

    Parameters
    ----------
    frequencies : array-like
        Solved frequencies in hertz, shape ``(num_frequencies,)``.
    neffs : array-like
        Complex effective indices, shape ``(num_frequencies, num_modes)``.
    e_fields : array-like
        Complex electric modal fields with frequency and mode leading axes.
    h_fields : array-like
        Complex magnetic modal fields matching ``e_fields``.
    eps_profiles : array-like
        Relative-permittivity cross sections used by the eigensolver.
    resolution : float
        Cross-section cell size in metres.
    solver_direction : str, optional
        Signed propagation direction used by the eigensolver.
    axis : str, optional
        Plane-normal axis, one of ``"x"``, ``"y"``, or ``"z"``.
    center : tuple, optional
        Mode-plane center in ``(x, y, z)`` order, in metres.
    plane : object, optional
        Original public plane specification.
    crop_slices : tuple of slice, optional
        Slices locating the solved aperture inside the full material plane.
    eps_profile_fulls : array-like, optional
        Full uncropped relative-permittivity planes for later analysis.
    grid_edges : tuple of array-like, optional
        Physical cell-edge coordinates for each transverse field axis, in metres.
        The arrays follow the storage order of the trailing modal-field axes.
    transverse_axes : tuple of str, optional
        Physical axis names corresponding to ``grid_edges``.

    Notes
    -----
    Array inputs are copied and made read-only.
    """

    frequencies: np.ndarray
    neffs: np.ndarray
    e_fields: np.ndarray
    h_fields: np.ndarray
    eps_profiles: np.ndarray
    resolution: float
    solver_direction: str | None = None
    axis: str | None = None
    center: tuple[float, float, float] | None = None
    plane: Any = field(default=None, compare=False, hash=False, repr=False)
    crop_slices: tuple[slice, ...] | None = field(
        default=None, compare=False, hash=False
    )
    eps_profile_fulls: np.ndarray | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    grid_edges: tuple[np.ndarray, ...] | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    transverse_axes: tuple[str, ...] | None = field(
        default=None, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frequencies", readonly_array(self.frequencies, dtype=float)
        )
        object.__setattr__(
            self, "neffs", readonly_array(self.neffs, dtype=np.complex128)
        )
        object.__setattr__(
            self, "e_fields", readonly_array(self.e_fields, dtype=np.complex128)
        )
        object.__setattr__(
            self, "h_fields", readonly_array(self.h_fields, dtype=np.complex128)
        )
        object.__setattr__(self, "eps_profiles", readonly_array(self.eps_profiles))
        if self.eps_profile_fulls is not None:
            object.__setattr__(
                self, "eps_profile_fulls", readonly_array(self.eps_profile_fulls)
            )
        if self.grid_edges is not None:
            edges = tuple(
                readonly_array(values, dtype=float) for values in self.grid_edges
            )
            if not edges:
                raise ValueError("grid_edges must contain at least one edge array.")
            field_shape = self.e_fields.shape[-len(edges) :]
            for index, (values, cell_count) in enumerate(
                zip(edges, field_shape, strict=True)
            ):
                if values.ndim != 1 or values.size != int(cell_count) + 1:
                    raise ValueError(
                        "grid_edges must have one more entry than their modal-field "
                        f"axis; axis {index} has {values.shape} for {cell_count} cells."
                    )
                if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
                    raise ValueError(
                        "grid_edges must be finite and strictly increasing."
                    )
            object.__setattr__(self, "grid_edges", edges)
        if self.transverse_axes is not None:
            axes = tuple(str(axis).lower() for axis in self.transverse_axes)
            if any(axis not in {"x", "y", "z"} for axis in axes):
                raise ValueError("transverse_axes entries must be 'x', 'y', or 'z'.")
            if self.grid_edges is not None and len(axes) != len(self.grid_edges):
                raise ValueError(
                    "transverse_axes must provide one name per grid edge array."
                )
            object.__setattr__(self, "transverse_axes", axes)
        if self.center is not None:
            object.__setattr__(self, "center", tuple(float(v) for v in self.center[:3]))

    def _frequency_index(self, f=None) -> int:
        if self.frequencies.size == 0:
            raise ValueError("ModeData contains no frequencies.")
        if f is None:
            return 0
        return int(
            np.argmin(np.abs(np.asarray(self.frequencies, dtype=float) - float(f)))
        )

    def selected_mode(self, *, f=None, mode_index=0):
        """Return arrays for the nearest frequency and selected mode index."""
        f_idx = self._frequency_index(f)
        m_idx = int(mode_index)
        neff = np.asarray(self.neffs)[f_idx, m_idx]
        e_field = np.asarray(self.e_fields)[f_idx, m_idx]
        h_field = np.asarray(self.h_fields)[f_idx, m_idx]
        eps_full = (
            np.asarray(self.eps_profile_fulls)[f_idx]
            if self.eps_profile_fulls is not None
            else np.asarray(self.eps_profiles)[f_idx]
        )
        return f_idx, m_idx, neff, e_field, h_field, eps_full
