"""Field storage and update logic for FDTD simulations."""

from __future__ import annotations
import numpy as np
from beamz.simulation import ops


class GridSpacing:
    """Spatial resolution of the FDTD grid (dx, dy, dz)."""
    
    def __init__(self, dx, dy, dz=None):
        self.dx = dx
        self.dy = dy
        self.dz = dz
    
    def is_3d(self):
        """Check if this is a 3D grid (has z-spacing)."""
        return self.dz is not None


class Fields:
    """Container for E/H field arrays on staggered Yee grid with FDTD update logic."""

    def __init__(self, backend, epsilon_r, sigma, grid_shape, spacing, complex_dtype=np.complex128):
        """Initialize field arrays on a Yee grid for 2D (Ez, Hx, Hy) or 3D (Ex, Ey, Ez, Hx, Hy, Hz) simulations."""
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

    def _init_fields_2d(self, ny, nx):
        """Initialize 2D TM mode field arrays (Ez, Hx, Hy) on staggered Yee grid."""
        self.Ez = self.backend.zeros((ny, nx), dtype=self.complex_dtype)
        self.Hx = self.backend.zeros((ny, nx - 1))  # Staggered half-cell in x
        self.Hy = self.backend.zeros((ny - 1, nx))  # Staggered half-cell in y

    def _init_fields_3d(self, nx, ny, nz):
        """Initialize 3D field arrays (Ex, Ey, Ez, Hx, Hy, Hz) with proper Yee grid staggering."""
        # E-field components, each staggered in one direction
        self.Ex = self.backend.zeros((nz, ny, nx - 1), dtype=self.complex_dtype)
        self.Ey = self.backend.zeros((nz, ny - 1, nx), dtype=self.complex_dtype)
        self.Ez = self.backend.zeros((nz - 1, ny, nx), dtype=self.complex_dtype)
        # H-field components, each staggered in two directions
        self.Hx = self.backend.zeros((nz - 1, ny - 1, nx), dtype=self.complex_dtype)
        self.Hy = self.backend.zeros((nz - 1, ny, nx - 1), dtype=self.complex_dtype)
        self.Hz = self.backend.zeros((nz, ny - 1, nx - 1), dtype=self.complex_dtype)


    def is_3d(self):
        """Check if simulation is 3D."""
        return self.spacing.is_3d()

    def update(self, dt):
        """Advance all fields by one FDTD time step: update H from curl(E), then E from curl(H)."""
        if self.is_3d: self._update_3d(dt)
        else: self._update_2d(dt)

    def _update_2d(self, dt):
        """Execute one 2D FDTD time step: H from curl(E) via Faraday's law, then E from curl(H) via Ampere's law."""
        # Convert backend arrays to NumPy for computation
        Ez = self._to_numpy(self.Ez)
        Hx = self._to_numpy(self.Hx)
        Hy = self._to_numpy(self.Hy)
        sigma = self._to_numpy(self.sigma)
        epsilon_r = self._to_numpy(self.epsilon_r)

        # Step 1: Update H-fields from curl of E
        spacings = (self.spacing.dx, self.spacing.dy, 0.0)
        curlE_x, curlE_y = ops.curl_e_to_h((Ez,), spacings)
        # Compute magnetic conductivity for PML absorption
        sigma_m_x, sigma_m_y = ops.magnetic_conductivity_terms(sigma, (Hx.shape, Hy.shape))
        # Advance H using implicit time-stepping for stability
        Hx = ops.advance_h_field(Hx, curlE_x, sigma_m_x, dt)
        Hy = ops.advance_h_field(Hy, curlE_y, sigma_m_y, dt)

        # Step 2: Update E-field from curl of H
        (curlH_z,) = ops.curl_h_to_e((Hx, Hy), spacings, Ez.shape)
        _, _, region = ops.material_slice_for_e(epsilon_r, sigma, orientation=None)
        # Advance E with dispersive/conductive material updates
        Ez_new = ops.advance_e_field(Ez, curlH_z, sigma, epsilon_r, dt, region)

        # Store updated fields back to backend
        self.Hx = self._from_numpy(Hx)
        self.Hy = self._from_numpy(Hy)
        self.Ez = self._from_numpy(Ez_new)
        self.is_complex_backend = np.iscomplexobj(Ez_new)

    def _update_3d(self, dt):
        """Execute one 3D FDTD time step: H from curl(E) via Faraday's law, then E from curl(H) via Ampere's law."""
        dx = self.spacing.dx
        dy = self.spacing.dy
        dz = self.spacing.dz
        if dz is None:
            raise ValueError("3D update requires a dz spacing value")

        # Convert all field components to NumPy
        Ex = self._to_numpy(self.Ex)
        Ey = self._to_numpy(self.Ey)
        Ez = self._to_numpy(self.Ez)
        Hx = self._to_numpy(self.Hx)
        Hy = self._to_numpy(self.Hy)
        Hz = self._to_numpy(self.Hz)
        sigma = self._to_numpy(self.sigma)
        eps_r = self._to_numpy(self.epsilon_r)

        # Step 1: Update H-fields from curl of E (Faraday's law: ∂H/∂t = -∇×E/μ₀)
        spacings = (dx, dy, dz)
        curlE_x, curlE_y, curlE_z = ops.curl_e_to_h((Ex, Ey, Ez), spacings)
        # Compute magnetic conductivity terms for each H component (for PML)
        sigma_m_hx, sigma_m_hy, sigma_m_hz = ops.magnetic_conductivity_terms(
            sigma, (Hx.shape, Hy.shape, Hz.shape)
        )
        # Advance each H component with implicit time-stepping
        Hx = ops.advance_h_field(Hx, curlE_x, sigma_m_hx, dt)
        Hy = ops.advance_h_field(Hy, curlE_y, sigma_m_hy, dt)
        Hz = ops.advance_h_field(Hz, curlE_z, sigma_m_hz, dt)

        # Step 2: Update E-fields from curl of H (Ampere's law: ∂E/∂t = ∇×H/(ε₀εᵣ) - σE/ε₀εᵣ)
        curlH_x, curlH_y, curlH_z = ops.curl_h_to_e((Hx, Hy, Hz), spacings, Ex.shape)
        # Extract material parameters at appropriate staggered positions for each E component
        eps_x, sig_x, region_x = ops.material_slice_for_e(eps_r, sigma, orientation="x")
        eps_y, sig_y, region_y = ops.material_slice_for_e(eps_r, sigma, orientation="y")
        eps_z, sig_z, region_z = ops.material_slice_for_e(eps_r, sigma, orientation="z")
        # Advance each E component with conductive and dispersive material response
        Ex = ops.advance_e_field(Ex, curlH_x, sig_x, eps_x, dt, region_x)
        Ey = ops.advance_e_field(Ey, curlH_y, sig_y, eps_y, dt, region_y)
        Ez = ops.advance_e_field(Ez, curlH_z, sig_z, eps_z, dt, region_z)

        # Store all updated fields back to backend
        self.Ex = self._from_numpy(Ex)
        self.Ey = self._from_numpy(Ey)
        self.Ez = self._from_numpy(Ez)
        self.Hx = self._from_numpy(Hx)
        self.Hy = self._from_numpy(Hy)
        self.Hz = self._from_numpy(Hz)
        self.is_complex_backend = np.iscomplexobj(Ex) or np.iscomplexobj(Ey) or np.iscomplexobj(Ez)

    def to_numpy(self, field_name):
        """Convert a field component (e.g., 'Ex', 'Hy', 'Ez') to a NumPy array for analysis or visualization."""
        field = getattr(self, field_name)
        return self.backend.to_numpy(field)

