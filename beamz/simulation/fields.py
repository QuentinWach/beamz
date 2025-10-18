"""Field storage and update logic for FDTD simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from beamz.const import EPS_0, MU_0


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
        hx_new, hy_new = self.backend.update_h_fields(
            self.Hx,
            self.Hy,
            self.Ez,
            self.sigma,
            self.spacing.dx,
            self.spacing.dy,
            dt,
            MU_0,
            EPS_0,
        )
        if hx_new is not self.Hx:
            self.Hx[...] = hx_new
        if hy_new is not self.Hy:
            self.Hy[...] = hy_new

        ez_new = self.backend.update_e_field(
            self.Ez,
            self.Hx,
            self.Hy,
            self.sigma,
            self.epsilon_r,
            self.spacing.dx,
            self.spacing.dy,
            dt,
            EPS_0,
        )
        if ez_new is not self.Ez:
            self.Ez[...] = ez_new

    def _update_3d(self, dt: float) -> None:
        dx = self.spacing.dx
        dy = self.spacing.dy
        dz = self.spacing.dz
        if dz is None:
            raise ValueError("3D update requires a dz spacing value")

        mu0 = MU_0
        eps0 = EPS_0

        Ex = self.Ex
        Ey = self.Ey
        Ez = self.Ez
        Hx = self.Hx
        Hy = self.Hy
        Hz = self.Hz
        eps_r = self.epsilon_r
        sigma = self.sigma

        sigma_m_hx = (sigma[:-1, :-1, :] * mu0 / eps0) if sigma.ndim == 3 else sigma * 0
        sigma_m_hy = (sigma[:-1, :, :-1] * mu0 / eps0) if sigma.ndim == 3 else sigma * 0
        sigma_m_hz = (sigma[:, :-1, :-1] * mu0 / eps0) if sigma.ndim == 3 else sigma * 0

        dEz_dy = (Ez[:, 1:, :] - Ez[:, :-1, :]) / dy
        dEy_dz = (Ey[1:, :, :] - Ey[:-1, :, :]) / dz
        curlE_x = dEz_dy - dEy_dz

        dEx_dz = (Ex[1:, :, :] - Ex[:-1, :, :]) / dz
        dEz_dx = (Ez[:, :, 1:] - Ez[:, :, :-1]) / dx
        curlE_y = dEx_dz - dEz_dx

        dEy_dx = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dx
        dEx_dy = (Ex[:, 1:, :] - Ex[:, :-1, :]) / dy
        curlE_z = dEy_dx - dEx_dy

        denom_hx = 1.0 + sigma_m_hx * dt / (2.0 * mu0)
        factor_hx = (1.0 - sigma_m_hx * dt / (2.0 * mu0)) / denom_hx
        source_hx = (dt / mu0) / denom_hx

        denom_hy = 1.0 + sigma_m_hy * dt / (2.0 * mu0)
        factor_hy = (1.0 - sigma_m_hy * dt / (2.0 * mu0)) / denom_hy
        source_hy = (dt / mu0) / denom_hy

        denom_hz = 1.0 + sigma_m_hz * dt / (2.0 * mu0)
        factor_hz = (1.0 - sigma_m_hz * dt / (2.0 * mu0)) / denom_hz
        source_hz = (dt / mu0) / denom_hz

        Hx[:] = factor_hx * Hx - source_hx * curlE_x
        Hy[:] = factor_hy * Hy - source_hy * curlE_y
        Hz[:] = factor_hz * Hz - source_hz * curlE_z

        if eps_r.ndim == 3:
            eps_ex = eps_r[1:-1, 1:-1, :-1]
            sig_ex = sigma[1:-1, 1:-1, :-1]
            eps_ey = eps_r[1:-1, :-1, 1:-1]
            sig_ey = sigma[1:-1, :-1, 1:-1]
            eps_ez = eps_r[:-1, 1:-1, 1:-1]
            sig_ez = sigma[:-1, 1:-1, 1:-1]
        else:
            eps_ex = eps_r
            sig_ex = sigma
            eps_ey = eps_r
            sig_ey = sigma
            eps_ez = eps_r
            sig_ez = sigma

        dHz_dy_ex = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy
        dHy_dz_ex = (Hy[1:, :, :] - Hy[:-1, :, :]) / dz
        curlH_x = dHz_dy_ex[1:-1, :, :] - dHy_dz_ex[:, 1:-1, :]

        dHx_dz_ey = (Hx[1:, :, :] - Hx[:-1, :, :]) / dz
        dHz_dx_ey = (Hz[:, :, 1:] - Hz[:, :, :-1]) / dx
        curlH_y = dHx_dz_ey[:, :, 1:-1] - dHz_dx_ey[1:-1, :, :]

        dHy_dx_ez = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dx
        dHx_dy_ez = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy
        curlH_z = dHy_dx_ez[:, 1:-1, :] - dHx_dy_ez[:, :, 1:-1]

        denom_ex = 1.0 + sig_ex * dt / (2.0 * eps0 * eps_ex)
        factor_ex = (1.0 - sig_ex * dt / (2.0 * eps0 * eps_ex)) / denom_ex
        source_ex = (dt / (eps0 * eps_ex)) / denom_ex

        denom_ey = 1.0 + sig_ey * dt / (2.0 * eps0 * eps_ey)
        factor_ey = (1.0 - sig_ey * dt / (2.0 * eps0 * eps_ey)) / denom_ey
        source_ey = (dt / (eps0 * eps_ey)) / denom_ey

        denom_ez = 1.0 + sig_ez * dt / (2.0 * eps0 * eps_ez)
        factor_ez = (1.0 - sig_ez * dt / (2.0 * eps0 * eps_ez)) / denom_ez
        source_ez = (dt / (eps0 * eps_ez)) / denom_ez

        Ex[1:-1, 1:-1, :] = factor_ex * Ex[1:-1, 1:-1, :] + source_ex * curlH_x
        Ey[1:-1, :, 1:-1] = factor_ey * Ey[1:-1, :, 1:-1] + source_ey * curlH_y
        Ez[:, 1:-1, 1:-1] = factor_ez * Ez[:, 1:-1, 1:-1] + source_ez * curlH_z

    def to_numpy(self, field_name: str):
        field = getattr(self, field_name)
        return self.backend.to_numpy(field)


