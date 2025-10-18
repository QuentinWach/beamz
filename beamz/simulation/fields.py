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
        else:
            ny, nx = grid_shape
            self._init_fields_2d(ny, nx)

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

    def update(self, dt):
        """Advance all fields by one FDTD time step: update H from curl(E), then E from curl(H)."""
        if self.dz is not None: self._update_3d(dt)
        else: self._update_2d(dt)

    def _update_2d(self, dt):
        """Execute one 2D FDTD time step: H from curl(E) via Faraday's law, then E from curl(H) via Ampere's law."""
        spacings = (self.dx, self.dy, 0.0)
        curlE_x, curlE_y = ops.curl_e_to_h((self.Ez,), spacings)
        sigma_m_x, sigma_m_y = ops.magnetic_conductivity_terms(self.sigma, (self.Hx.shape, self.Hy.shape))
        self.Hx = ops.advance_h_field(self.Hx, curlE_x, sigma_m_x, dt)
        self.Hy = ops.advance_h_field(self.Hy, curlE_y, sigma_m_y, dt)
        (curlH_z,) = ops.curl_h_to_e((self.Hx, self.Hy), spacings, self.Ez.shape)
        _, _, region = ops.material_slice_for_e(self.epsilon_r, self.sigma, orientation=None)
        self.Ez = ops.advance_e_field(self.Ez, curlH_z, self.sigma, self.epsilon_r, dt, region)

    def _update_3d(self, dt):
        """Execute one 3D FDTD time step: H from curl(E) via Faraday's law, then E from curl(H) via Ampere's law."""
        spacings = (self.dx, self.dy, self.dz)
        curlE_x, curlE_y, curlE_z = ops.curl_e_to_h((self.Ex, self.Ey, self.Ez), spacings)
        sigma_m_hx, sigma_m_hy, sigma_m_hz = ops.magnetic_conductivity_terms(
            self.sigma, (self.Hx.shape, self.Hy.shape, self.Hz.shape))
        self.Hx = ops.advance_h_field(self.Hx, curlE_x, sigma_m_hx, dt)
        self.Hy = ops.advance_h_field(self.Hy, curlE_y, sigma_m_hy, dt)
        self.Hz = ops.advance_h_field(self.Hz, curlE_z, sigma_m_hz, dt)
        curlH_x, curlH_y, curlH_z = ops.curl_h_to_e((self.Hx, self.Hy, self.Hz), spacings, self.Ex.shape)
        eps_x, sig_x, region_x = ops.material_slice_for_e(self.epsilon_r, self.sigma, orientation="x")
        eps_y, sig_y, region_y = ops.material_slice_for_e(self.epsilon_r, self.sigma, orientation="y")
        eps_z, sig_z, region_z = ops.material_slice_for_e(self.epsilon_r, self.sigma, orientation="z")
        self.Ex = ops.advance_e_field(self.Ex, curlH_x, sig_x, eps_x, dt, region_x)
        self.Ey = ops.advance_e_field(self.Ey, curlH_y, sig_y, eps_y, dt, region_y)
        self.Ez = ops.advance_e_field(self.Ez, curlH_z, sig_z, eps_z, dt, region_z)