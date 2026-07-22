from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from beamz._cache_tokens import cache_token
from beamz.devices._immutable import readonly_array


@dataclass(frozen=True, slots=True, eq=False)
class MaterialGrid:
    """Store an immutable, center-sampled material discretization.

    Parameters
    ----------
    permittivity : array-like
        Relative-permittivity samples in ``(y, x)`` or ``(z, y, x)`` order.
    conductivity : array-like
        Electrical-conductivity samples in siemens per metre.
    permeability : array-like
        Relative-permeability samples matching ``permittivity``.
    resolution : float
        Uniform spatial cell size in metres.
    shape : tuple of int
        Material array shape. It must follow array storage order, not public
        coordinate order.

    Notes
    -----
    Input arrays are copied and made read-only during construction.
    """

    permittivity: npt.ArrayLike
    conductivity: npt.ArrayLike
    permeability: npt.ArrayLike
    resolution: float
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "permittivity", readonly_array(self.permittivity))
        object.__setattr__(self, "conductivity", readonly_array(self.conductivity))
        object.__setattr__(self, "permeability", readonly_array(self.permeability))
        object.__setattr__(self, "resolution", float(self.resolution))
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))

    @classmethod
    def from_grid(cls, grid, *, resolution: float) -> MaterialGrid:
        """Snapshot a rasterizer object into an immutable material grid.

        Parameters
        ----------
        grid : object
            Rasterizer result exposing permittivity, conductivity, and
            permeability arrays.
        resolution : float
            Uniform cell spacing in metres.
        """
        permittivity = grid.permittivity
        conductivity = grid.conductivity
        permeability = grid.permeability
        return cls(
            permittivity,
            conductivity,
            permeability,
            float(resolution),
            tuple(int(v) for v in np.asarray(permittivity).shape),
        )

    def field_arrays(self) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Return permittivity, conductivity, and permeability arrays."""
        return self.permittivity, self.conductivity, self.permeability

    def canonical_spec(self):
        """Return values defining material-grid cache identity."""
        return (*self.field_arrays(), self.resolution, self.shape)

    def __eq__(self, other):
        if not isinstance(other, MaterialGrid):
            return NotImplemented
        return cache_token(self.canonical_spec()) == cache_token(other.canonical_spec())

    def __hash__(self):
        return hash(cache_token(self.canonical_spec()))


def build_material_grid(
    design,
    resolution: float,
    *,
    grid_type: str = "auto",
    force_recompute: bool = False,
    progress: bool = False,
    **kwargs,
) -> MaterialGrid:
    """Discretize a design into center-sampled material-property grids.

    Parameters
    ----------
    design : Design
        Immutable geometry and material specification to rasterize.
    resolution : float
        Uniform spatial cell size in metres.
    grid_type : str, default="auto"
        Rasterizer selection forwarded to ``design.rasterize``.
    force_recompute : bool, default=False
        Rebuild the grid even when a matching cached discretization exists.
    progress : bool, default=False
        Emit rasterization progress when the design supports it.
    **kwargs
        Additional rasterizer-specific options.

    Returns
    -------
    MaterialGrid
        Read-only permittivity, conductivity, and permeability arrays.

    """
    grid = design.rasterize(
        resolution,
        grid_type=grid_type,
        force_recompute=force_recompute,
        progress=progress,
        **kwargs,
    )
    return MaterialGrid.from_grid(grid, resolution=resolution)
