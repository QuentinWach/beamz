"""Field storage and update logic for FDTD simulations."""

from __future__ import annotations
import numpy as np
from beamz.simulation import ops


class Fields:
    """Container for E/H field arrays on staggered Yee grid with FDTD update logic."""

    def __init__(self, permittivity, conductivity, permeability, resolution, pml_regions=None):
        """Initialize field arrays on a Yee grid for 2D (Ez, Hx, Hy) or 3D (Ex, Ey, Ez, Hx, Hy, Hz) simulations."""
        self.resolution = resolution
        # Store references to material grids owned by Design (no copying)
        self.permittivity = permittivity
        self.conductivity = conductivity
        self.permeability = permeability
        
        # Initialize PML regions if present
        if pml_regions:
            self._init_upml_fields(pml_regions)
        else:
            self.has_pml = False
        
        # Infer dimensionality and shape from material arrays
        is_3d = self.permittivity.ndim == 3
        grid_shape = self.permittivity.shape

        if is_3d:
            nz, ny, nx = grid_shape
            self._init_fields_3d(nx, ny, nz)
            self.update = self._update_3d
            self._curl_e_to_h = ops.curl_e_to_h_3d
            self._curl_h_to_e = ops.curl_h_to_e_3d
            self._material_slice = ops.material_slice_for_e_3d
            self.sigma_m_hx, self.sigma_m_hy, self.sigma_m_hz = ops.magnetic_conductivity_terms_3d(
                self.conductivity, self.permeability, self.Hx.shape, self.Hy.shape, self.Hz.shape)
            self.eps_x, self.sig_x, self.region_x = self._material_slice(self.permittivity, self.conductivity, orientation="x")
            self.eps_y, self.sig_y, self.region_y = self._material_slice(self.permittivity, self.conductivity, orientation="y")
            self.eps_z, self.sig_z, self.region_z = self._material_slice(self.permittivity, self.conductivity, orientation="z")
        else:
            ny, nx = grid_shape
            self._init_fields_2d(ny, nx)
            self.update = self._update_2d
            self._curl_e_to_h = ops.curl_e_to_h_2d
            self._curl_h_to_e = ops.curl_h_to_e_2d
            self._material_slice = ops.material_slice_for_e_2d
            self.sigma_m_x, self.sigma_m_y = ops.magnetic_conductivity_terms_2d(self.conductivity, self.permeability, self.Hx.shape, self.Hy.shape)
            self.eps_region, self.sig_region, self.region = self._material_slice(self.permittivity, self.conductivity)

    def _init_upml_fields(self, pml_regions):
        """Initialize auxiliary fields for split-field UPML."""
        self.has_pml = True
        self.pml_regions = pml_regions
        
        # Initialize split fields now that we have PML data
        if not self.permittivity.ndim == 3:  # 2D
            self._init_split_fields_2d()
        else:  # 3D
            self._init_split_fields_3d()
    
    def _init_split_fields_2d(self):
        """Initialize split-field components for 2D UPML."""
        if self.has_pml:
            # 2D TM mode: Ez splits into Ez_x and Ez_y
            self.Ez_x = np.zeros_like(self.Ez)
            self.Ez_y = np.zeros_like(self.Ez)
            # Hx and Hy don't split in standard UPML for TM mode
    
    def _init_split_fields_3d(self):
        """Initialize split-field components for 3D UPML."""
        if self.has_pml:
            # 3D: all components split
            self.Ex_y = np.zeros_like(self.Ex)
            self.Ex_z = np.zeros_like(self.Ex)
            self.Ey_x = np.zeros_like(self.Ey)
            self.Ey_z = np.zeros_like(self.Ey)
            self.Ez_x = np.zeros_like(self.Ez)
            self.Ez_y = np.zeros_like(self.Ez)
            # Similar for H fields...

    def _init_fields_3d(self, nx, ny, nz):
        """Initialize 3D field arrays (Ex, Ey, Ez, Hx, Hy, Hz) with proper Yee grid staggering."""
        self.Ex = np.zeros((nz, ny, nx - 1))
        self.Ey = np.zeros((nz, ny - 1, nx))
        self.Ez = np.zeros((nz - 1, ny, nx))
        self.Hx = np.zeros((nz - 1, ny - 1, nx))
        self.Hy = np.zeros((nz - 1, ny, nx - 1))
        self.Hz = np.zeros((nz, ny - 1, nx - 1))

    def _init_fields_2d(self, ny, nx):
        """Initialize 2D TM mode field arrays (Ez, Hx, Hy) on staggered Yee grid."""
        self.Ez = np.zeros((ny, nx))
        self.Hx = np.zeros((ny, nx - 1))
        self.Hy = np.zeros((ny - 1, nx))

    def _update_2d(self, dt, source_j=None, source_m=None):
        """Execute one 2D FDTD time step."""
        if self.has_pml:
            # Use UPML split-field updates
            from beamz.simulation.ops import update_e_field_upml_2d
            
            # Update H fields
            curlE_x, curlE_y = self._curl_e_to_h(self.Ez, self.resolution)
            
            # Handle legacy magnetic current M_y if passed
            if source_m and 'Hy' in source_m:
                m_y, indices = source_m['Hy']
                curlE_y_with_source = curlE_y.copy()
                y_slice, x_idx = indices
                curlE_y_with_source[y_slice, x_idx] += m_y
                self.Hy = ops.advance_h_field(self.Hy, curlE_y_with_source, self.sigma_m_y, dt)
            else:
                self.Hy = ops.advance_h_field(self.Hy, curlE_y, self.sigma_m_y, dt)
            
            self.Hx = ops.advance_h_field(self.Hx, curlE_x, self.sigma_m_x, dt)
            
            # Update E field with UPML
            self.Ez = update_e_field_upml_2d(self.Ez, self.Ez_x, self.Ez_y, self.Hx, self.Hy,
                                          self.pml_regions, self.permittivity, 
                                          self.conductivity, self.resolution, dt)
        else:
            # Standard update
            curlE_x, curlE_y = self._curl_e_to_h(self.Ez, self.resolution)
            
            # Handle legacy magnetic current M_y if passed
            if source_m and 'Hy' in source_m:
                m_y, indices = source_m['Hy']
                curlE_y_with_source = curlE_y.copy()
                y_slice, x_idx = indices
                curlE_y_with_source[y_slice, x_idx] += m_y
                self.Hy = ops.advance_h_field(self.Hy, curlE_y_with_source, self.sigma_m_y, dt)
            else:
                self.Hy = ops.advance_h_field(self.Hy, curlE_y, self.sigma_m_y, dt)
            
            self.Hx = ops.advance_h_field(self.Hx, curlE_x, self.sigma_m_x, dt)
            
            # Update E field
            (curlH_z,) = self._curl_h_to_e(self.Hx, self.Hy, self.resolution, self.Ez.shape)
            
            # Handle legacy electric current J_z if passed
            if source_j and 'Ez' in source_j:
                j_z, indices = source_j['Ez']
                curlH_z_with_source = curlH_z.copy()
                y_slice, x_idx = indices
                curlH_z_with_source[y_slice, x_idx] += j_z
            else:
                curlH_z_with_source = curlH_z
            
            self.Ez = ops.advance_e_field(self.Ez, curlH_z_with_source, self.sig_region, self.eps_region, dt, self.region)

    def _update_3d(self, dt):
        """Execute one 3D FDTD time step: H from curl(E) via Faraday's law, then E from curl(H) via Ampere's law."""
        curlE_x, curlE_y, curlE_z = self._curl_e_to_h(self.Ex, self.Ey, self.Ez, self.resolution)
        self.Hx = ops.advance_h_field(self.Hx, curlE_x, self.sigma_m_hx, dt)
        self.Hy = ops.advance_h_field(self.Hy, curlE_y, self.sigma_m_hy, dt)
        self.Hz = ops.advance_h_field(self.Hz, curlE_z, self.sigma_m_hz, dt)
        curlH_x, curlH_y, curlH_z = self._curl_h_to_e(self.Hx, self.Hy, self.Hz, self.resolution)
        self.Ex = ops.advance_e_field(self.Ex, curlH_x, self.sig_x, self.eps_x, dt, self.region_x)
        self.Ey = ops.advance_e_field(self.Ey, curlH_y, self.sig_y, self.eps_y, dt, self.region_y)
        self.Ez = ops.advance_e_field(self.Ez, curlH_z, self.sig_z, self.eps_z, dt, self.region_z)

    def update(self, dt, source_j=None, source_m=None):
        """Execute one FDTD time step with optional source injection."""
        if not self.permittivity.ndim == 3:  # 2D
            self._update_2d(dt, source_j=source_j, source_m=source_m)
        else:  # 3D
            self._update_3d(dt)