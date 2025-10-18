"""Field storage and update logic for FDTD simulations."""

# This module relies on helpers in `beamz.simulation.ops` for curl and Yee-step operations.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.simulation import ops


@dataclass(slots=True)
class GridSpacing:
    dx: float
    dy: float
    dz: Optional[float] = None

    @property
    def is_3d(self) -> bool:
        return self.dz is not None


class Fields:
    """Container for electric and magnetic field arrays and their updates."""

    def __init__(
        self,
        backend,
        epsilon_r,
        sigma,
        grid_shape: Tuple[int, ...],
        spacing: GridSpacing,
        complex_dtype=np.complex128,
    ) -> None:
        self.backend = backend
        self.spacing = spacing
        self.complex_dtype = complex_dtype

        self.epsilon_r = backend.from_numpy(epsilon_r)
        self.sigma = backend.from_numpy(sigma)

        self.is_complex_backend = True
        if spacing.is_3d:
            nz, ny, nx = grid_shape
            self._init_fields_3d(nx, ny, nz)
        else:
            ny, nx = grid_shape
            self._init_fields_2d(ny, nx)

    def _to_numpy(self, array):
        """Return a NumPy view/copy of the backend array."""
        return np.asarray(self.backend.to_numpy(array))

    def _from_numpy(self, array):
        """Wrap a NumPy array back into the backend type."""
        return self.backend.from_numpy(array)

    def _init_fields_2d(self, ny: int, nx: int) -> None:
        try:
            self.Ez = self.backend.zeros((ny, nx), dtype=self.complex_dtype)
        except TypeError:
            self.Ez = self.backend.zeros((ny, nx))
            self.is_complex_backend = False
        self.Hx = self.backend.zeros((ny, nx - 1))
        self.Hy = self.backend.zeros((ny - 1, nx))

    def _init_fields_3d(self, nx: int, ny: int, nz: int) -> None:
        try:
            self.Ex = self.backend.zeros((nz, ny, nx - 1), dtype=self.complex_dtype)
            self.Ey = self.backend.zeros((nz, ny - 1, nx), dtype=self.complex_dtype)
            self.Ez = self.backend.zeros((nz - 1, ny, nx), dtype=self.complex_dtype)
            self.Hx = self.backend.zeros((nz - 1, ny - 1, nx), dtype=self.complex_dtype)
            self.Hy = self.backend.zeros((nz - 1, ny, nx - 1), dtype=self.complex_dtype)
            self.Hz = self.backend.zeros((nz, ny - 1, nx - 1), dtype=self.complex_dtype)
        except TypeError:
            self.Ex = self.backend.zeros((nz, ny, nx - 1))
            self.Ey = self.backend.zeros((nz, ny - 1, nx))
            self.Ez = self.backend.zeros((nz - 1, ny, nx))
            self.Hx = self.backend.zeros((nz - 1, ny - 1, nx))
            self.Hy = self.backend.zeros((nz - 1, ny, nx - 1))
            self.Hz = self.backend.zeros((nz, ny - 1, nx - 1))
            self.is_complex_backend = False

    @property
    def is_3d(self) -> bool:
        return self.spacing.is_3d

    def update(self, dt: float) -> None:
        if self.is_3d:
            self._update_3d(dt)
        else:
            self._update_2d(dt)

    def _update_2d(self, dt: float) -> None:
        Ez = self._to_numpy(self.Ez)
        Hx = self._to_numpy(self.Hx)
        Hy = self._to_numpy(self.Hy)
        sigma = self._to_numpy(self.sigma)
        epsilon_r = self._to_numpy(self.epsilon_r)

        curlE_x, curlE_y = ops.curl_e_to_h_2d(Ez, self.spacing.dx, self.spacing.dy)
        sigma_m_x, sigma_m_y = ops.magnetic_conductivity_terms_2d(sigma)

        Hx = ops.advance_h_field(Hx, curlE_x, sigma_m_x, dt)
        Hy = ops.advance_h_field(Hy, -curlE_y, sigma_m_y, dt)

        curlH = ops.curl_h_to_e_2d(Hx, Hy, self.spacing.dx, self.spacing.dy, Ez.shape)

        Ez_new = ops.advance_e_field_2d(Ez, curlH, sigma, epsilon_r, dt)

        self.Hx = self._from_numpy(Hx)
        self.Hy = self._from_numpy(Hy)
        self.Ez = self._from_numpy(Ez_new)
        self.is_complex_backend = np.iscomplexobj(Ez_new)

    def _update_3d(self, dt: float) -> None:
        dx = self.spacing.dx
        dy = self.spacing.dy
        dz = self.spacing.dz
        if dz is None:
            raise ValueError("3D update requires a dz spacing value")

        Ex = self._to_numpy(self.Ex)
        Ey = self._to_numpy(self.Ey)
        Ez = self._to_numpy(self.Ez)
        Hx = self._to_numpy(self.Hx)
        Hy = self._to_numpy(self.Hy)
        Hz = self._to_numpy(self.Hz)
        sigma = self._to_numpy(self.sigma)
        eps_r = self._to_numpy(self.epsilon_r)

        curlE_x, curlE_y, curlE_z = ops.curl_e_to_h_3d(Ex, Ey, Ez, dx, dy, dz)
        sigma_m_hx, sigma_m_hy, sigma_m_hz = ops.magnetic_conductivity_terms_3d(
            sigma, Hx.shape, Hy.shape, Hz.shape
        )

        Hx = ops.advance_h_field(Hx, curlE_x, sigma_m_hx, dt)
        Hy = ops.advance_h_field(Hy, curlE_y, sigma_m_hy, dt)
        Hz = ops.advance_h_field(Hz, curlE_z, sigma_m_hz, dt)

        curlH_x, curlH_y, curlH_z = ops.curl_h_to_e_3d(Hx, Hy, Hz, dx, dy, dz)
        eps_x, sig_x, region_x = ops.material_slice_for_e(eps_r, sigma, orientation="x")
        eps_y, sig_y, region_y = ops.material_slice_for_e(eps_r, sigma, orientation="y")
        eps_z, sig_z, region_z = ops.material_slice_for_e(eps_r, sigma, orientation="z")

        Ex = ops.advance_e_field(Ex, curlH_x, eps_x, sig_x, region_x, dt)
        Ey = ops.advance_e_field(Ey, curlH_y, eps_y, sig_y, region_y, dt)
        Ez = ops.advance_e_field(Ez, curlH_z, eps_z, sig_z, region_z, dt)

        self.Ex = self._from_numpy(Ex)
        self.Ey = self._from_numpy(Ey)
        self.Ez = self._from_numpy(Ez)
        self.Hx = self._from_numpy(Hx)
        self.Hy = self._from_numpy(Hy)
        self.Hz = self._from_numpy(Hz)
        self.is_complex_backend = np.iscomplexobj(Ex) or np.iscomplexobj(Ey) or np.iscomplexobj(Ez)

    def to_numpy(self, field_name: str):
        field = getattr(self, field_name)
        return self.backend.to_numpy(field)

