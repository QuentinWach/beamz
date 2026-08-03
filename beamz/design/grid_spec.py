"""Solver grid-spacing policy, independent of geometry rasterization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamz.const import LIGHT_SPEED


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

        Returns
        -------
        GridSpec
            Immutable wavelength-driven grid policy.
        """
        return cls(float(min_steps_per_wvl), wavelength, None, float(courant))

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

    def resolve_time_step(self, resolution: float, *, dims: int) -> float:
        """Return a Courant-limited time step in seconds.

        Returns
        -------
        float
            Courant-limited time step in seconds.
        """
        return (
            float(self.courant)
            * float(resolution)
            / (LIGHT_SPEED * np.sqrt(float(max(1, int(dims)))))
        )
