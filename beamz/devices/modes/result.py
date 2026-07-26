"""Minimal result returned by the BeamZ-native mode solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import xarray as xr


@dataclass(frozen=True)
class Result:
    """Solved effective indices, component fields, and solver diagnostics."""

    n_complex: xr.DataArray
    field_components: dict[str, xr.DataArray]
    n_group: xr.DataArray | None = None
    dispersion: xr.DataArray | None = None
    solver_info: dict[str, Any] | None = None

    @property
    def n_eff(self) -> xr.DataArray:
        return self.n_complex.real

    @property
    def k_eff(self) -> xr.DataArray:
        return self.n_complex.imag
