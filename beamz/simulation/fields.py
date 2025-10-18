"""Field storage and update logic for FDTD simulations."""

from __future__ import annotations
import numpy as np
from beamz.simulation import ops


class Fields:
    """Container for E/H field arrays on staggered Yee grid with FDTD update logic."""

    def __init__(self, epsilon_r, sigma, grid_shape, dx, dy, dz=None):
        """Initialize field arrays on a Yee grid for 2D (Ez, Hx, Hy) or 3D (Ex, Ey, Ez, Hx, Hy, Hz) simulations."""
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.epsilon_r = np.asarray(epsilon_r)
        self.sigma = np.asarray(sigma)

        if dz is not None:
            nz, ny, nx = grid_shape
            self._init_fields_3d(nx, ny, nz)
            self.update = self._update_3d
            self._curl_e_to_h = ops.curl_e_to_h_3d
            self._curl_h_to_e = ops.curl_h_to_e_3d
            self._material_slice = ops.material_slice_for_e_3d
            self.sigma_m_hx, self.sigma_m_hy, self.sigma_m_hz = ops.magnetic_conductivity_terms_3d(
                self.sigma, self.Hx.shape, self.Hy.shape, self.Hz.shape)
            self.eps_x, self.sig_x, self.region_x = self._material_slice(self.epsilon_r, self.sigma, orientation="x")
            self.eps_y, self.sig_y, self.region_y = self._material_slice(self.epsilon_r, self.sigma, orientation="y")
            self.eps_z, self.sig_z, self.region_z = self._material_slice(self.epsilon_r, self.sigma, orientation="z")
        else:
            ny, nx = grid_shape
            self._init_fields_2d(ny, nx)
            self.update = self._update_2d
            self._curl_e_to_h = ops.curl_e_to_h_2d
            self._curl_h_to_e = ops.curl_h_to_e_2d
            self._material_slice = ops.material_slice_for_e_2d
            self.sigma_m_x, self.sigma_m_y = ops.magnetic_conductivity_terms_2d(self.sigma, self.Hx.shape, self.Hy.shape)
            self.eps_region, self.sig_region, self.region = self._material_slice(self.epsilon_r, self.sigma)

    def _init_fields_2d(self, ny, nx):
        """Initialize 2D TM mode field arrays (Ez, Hx, Hy) on staggered Yee grid."""
        self.Ez = np.zeros((ny, nx))
        self.Hx = np.zeros((ny, nx - 1))
        self.Hy = np.zeros((ny - 1, nx))

    def _init_fields_3d(self, nx, ny, nz):
        """Initialize 3D field arrays (Ex, Ey, Ez, Hx, Hy, Hz) with proper Yee grid staggering."""
        self.Ex = np.zeros((nz, ny, nx - 1))
        self.Ey = np.zeros((nz, ny - 1, nx))
        self.Ez = np.zeros((nz - 1, ny, nx))
        self.Hx = np.zeros((nz - 1, ny - 1, nx))
        self.Hy = np.zeros((nz - 1, ny, nx - 1))
        self.Hz = np.zeros((nz, ny - 1, nx - 1))

    def _update_2d(self, dt):
        """Execute one 2D FDTD time step: H from curl(E) via Faraday's law, then E from curl(H) via Ampere's law."""
        curlE_x, curlE_y = self._curl_e_to_h(self.Ez, self.dx, self.dy)
        self.Hx = ops.advance_h_field(self.Hx, curlE_x, self.sigma_m_x, dt)
        self.Hy = ops.advance_h_field(self.Hy, curlE_y, self.sigma_m_y, dt)
        (curlH_z,) = self._curl_h_to_e(self.Hx, self.Hy, self.dx, self.dy, self.Ez.shape)
        self.Ez = ops.advance_e_field(self.Ez, curlH_z, self.sig_region, self.eps_region, dt, self.region)

    def _update_3d(self, dt):
        """Execute one 3D FDTD time step: H from curl(E) via Faraday's law, then E from curl(H) via Ampere's law."""
        curlE_x, curlE_y, curlE_z = self._curl_e_to_h(self.Ex, self.Ey, self.Ez, self.dx, self.dy, self.dz)
        self.Hx = ops.advance_h_field(self.Hx, curlE_x, self.sigma_m_hx, dt)
        self.Hy = ops.advance_h_field(self.Hy, curlE_y, self.sigma_m_hy, dt)
        self.Hz = ops.advance_h_field(self.Hz, curlE_z, self.sigma_m_hz, dt)
        curlH_x, curlH_y, curlH_z = self._curl_h_to_e(self.Hx, self.Hy, self.Hz, self.dx, self.dy, self.dz)
        self.Ex = ops.advance_e_field(self.Ex, curlH_x, self.sig_x, self.eps_x, dt, self.region_x)
        self.Ey = ops.advance_e_field(self.Ey, curlH_y, self.sig_y, self.eps_y, dt, self.region_y)
        self.Ez = ops.advance_e_field(self.Ez, curlH_z, self.sig_z, self.eps_z, dt, self.region_z)