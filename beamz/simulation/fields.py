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

        curlE_x = (Ez[:, 1:] - Ez[:, :-1]) / self.spacing.dy
        curlE_y = (Ez[1:, :] - Ez[:-1, :]) / self.spacing.dx

        sigma_m_x = sigma[:, :-1] * MU_0 / EPS_0
        sigma_m_y = sigma[:-1, :] * MU_0 / EPS_0

        denom_x = 1.0 + sigma_m_x * dt / (2.0 * MU_0)
        factor_x = (1.0 - sigma_m_x * dt / (2.0 * MU_0)) / denom_x
        source_x = (dt / MU_0) / denom_x
        Hx = factor_x * Hx - source_x * curlE_x

        denom_y = 1.0 + sigma_m_y * dt / (2.0 * MU_0)
        factor_y = (1.0 - sigma_m_y * dt / (2.0 * MU_0)) / denom_y
        source_y = (dt / MU_0) / denom_y
        Hy = factor_y * Hy + source_y * curlE_y

        curlH = np.zeros_like(Ez)
        curlH[1:-1, 1:-1] = (
            (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / self.spacing.dx
            - (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / self.spacing.dy
        )

        sig = sigma[1:-1, 1:-1]
        eps = epsilon_r[1:-1, 1:-1]
        denom = 1.0 + sig * dt / (2.0 * EPS_0 * eps)
        factor = (1.0 - sig * dt / (2.0 * EPS_0 * eps)) / denom
        source = (dt / (EPS_0 * eps)) / denom

        Ez_new = Ez.copy()
        Ez_new[1:-1, 1:-1] = factor * Ez[1:-1, 1:-1] + source * curlH[1:-1, 1:-1]

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

        curlE_x, curlE_y, curlE_z = self._curl_e_to_h_3d(Ex, Ey, Ez, dx, dy, dz)
        sigma_m_hx, sigma_m_hy, sigma_m_hz = self._magnetic_conductivity_terms_3d(sigma)

        Hx = self._advance_h_field(Hx, curlE_x, sigma_m_hx, dt)
        Hy = self._advance_h_field(Hy, curlE_y, sigma_m_hy, dt)
        Hz = self._advance_h_field(Hz, curlE_z, sigma_m_hz, dt)

        curlH_x, curlH_y, curlH_z = self._curl_h_to_e_3d(Hx, Hy, Hz, dx, dy, dz)
        Ex = self._advance_e_field(Ex, curlH_x, eps_r, sigma, dt, orientation="x")
        Ey = self._advance_e_field(Ey, curlH_y, eps_r, sigma, dt, orientation="y")
        Ez = self._advance_e_field(Ez, curlH_z, eps_r, sigma, dt, orientation="z")

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

    # --- 2D helper methods -------------------------------------------------

    def _curl_e_to_h_2d(self, Ez, dx, dy):
        curl_e_x = (Ez[:, 1:] - Ez[:, :-1]) / dy
        curl_e_y = (Ez[1:, :] - Ez[:-1, :]) / dx
        return curl_e_x, curl_e_y

    def _curl_h_to_e_2d(self, Hx, Hy, dx, dy):
        curl = np.zeros((Hy.shape[0] + 1, Hx.shape[1] + 1), dtype=np.result_type(Hx, Hy))
        curl[1:-1, 1:-1] = (
            (Hy[:, 1:] - Hy[:, :-1]) / dx -
            (Hx[1:, :] - Hx[:-1, :]) / dy
        )
        return curl

    def _advance_e_2d(self, Ez, curlH, sigma, epsilon_r, dt):
        Ez_new = Ez.copy()
        interior = (slice(1, -1), slice(1, -1))
        sig = sigma[interior]
        eps = epsilon_r[interior]
        denom = 1.0 + sig * dt / (2.0 * EPS_0 * eps)
        factor = (1.0 - sig * dt / (2.0 * EPS_0 * eps)) / denom
        source = (dt / (EPS_0 * eps)) / denom
        Ez_new[interior] = factor * Ez[interior] + source * curlH[interior]
        return Ez_new

    # --- 3D helper methods -------------------------------------------------

    def _curl_e_to_h_3d(self, Ex, Ey, Ez, dx, dy, dz):
        dEz_dy = (Ez[:, 1:, :] - Ez[:, :-1, :]) / dy
        dEy_dz = (Ey[1:, :, :] - Ey[:-1, :, :]) / dz
        curlE_x = dEz_dy - dEy_dz

        dEx_dz = (Ex[1:, :, :] - Ex[:-1, :, :]) / dz
        dEz_dx = (Ez[:, :, 1:] - Ez[:, :, :-1]) / dx
        curlE_y = dEx_dz - dEz_dx

        dEy_dx = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dx
        dEx_dy = (Ex[:, 1:, :] - Ex[:, :-1, :]) / dy
        curlE_z = dEy_dx - dEx_dy
        return curlE_x, curlE_y, curlE_z

    def _magnetic_conductivity_terms_3d(self, sigma):
        if sigma.ndim == 3:
            sigma_m_hx = sigma[:-1, :-1, :] * MU_0 / EPS_0
            sigma_m_hy = sigma[:-1, :, :-1] * MU_0 / EPS_0
            sigma_m_hz = sigma[:, :-1, :-1] * MU_0 / EPS_0
        else:
            sigma_m_hx = np.zeros_like(self.Hx)
            sigma_m_hy = np.zeros_like(self.Hy)
            sigma_m_hz = np.zeros_like(self.Hz)
        return sigma_m_hx, sigma_m_hy, sigma_m_hz

    def _advance_h_field(self, field, curlE, sigma_m, dt):
        denom = 1.0 + sigma_m * dt / (2.0 * MU_0)
        factor = (1.0 - sigma_m * dt / (2.0 * MU_0)) / denom
        source = (dt / MU_0) / denom
        return factor * field - source * curlE

    def _curl_h_to_e_3d(self, Hx, Hy, Hz, dx, dy, dz):
        dHz_dy = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy
        dHy_dz = (Hy[1:, :, :] - Hy[:-1, :, :]) / dz
        curlH_x = dHz_dy[1:-1, :, :] - dHy_dz[:, 1:-1, :]

        dHx_dz = (Hx[1:, :, :] - Hx[:-1, :, :]) / dz
        dHz_dx = (Hz[:, :, 1:] - Hz[:, :, :-1]) / dx
        curlH_y = dHx_dz[:, :, 1:-1] - dHz_dx[1:-1, :, :]

        dHy_dx = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dx
        dHx_dy = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy
        curlH_z = dHy_dx[:, 1:-1, :] - dHx_dy[:, :, 1:-1]
        return curlH_x, curlH_y, curlH_z

    def _advance_e_field(self, field, curlH, eps_r, sigma, dt, orientation: str):
        updated = field.copy()
        if orientation == "x":
            eps = eps_r[1:-1, 1:-1, :-1]
            sig = sigma[1:-1, 1:-1, :-1]
            region = (slice(1, -1), slice(1, -1), slice(None))
        elif orientation == "y":
            eps = eps_r[1:-1, :-1, 1:-1]
            sig = sigma[1:-1, :-1, 1:-1]
            region = (slice(1, -1), slice(None), slice(1, -1))
        else:
            eps = eps_r[:-1, 1:-1, 1:-1]
            sig = sigma[:-1, 1:-1, 1:-1]
            region = (slice(None), slice(1, -1), slice(1, -1))

        current = field[region]
        denom = 1.0 + sig * dt / (2.0 * EPS_0 * eps)
        factor = (1.0 - sig * dt / (2.0 * EPS_0 * eps)) / denom
        source = (dt / (EPS_0 * eps)) / denom
        updated[region] = factor * current + source * curlH
        return updated

