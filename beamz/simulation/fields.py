"""Field storage and update logic for FDTD simulations."""

from __future__ import annotations

import jax.numpy as jnp

from beamz.simulation import ops


class Fields:
    """Container for E/H field arrays on staggered Yee grid with FDTD update logic."""

    def __init__(
        self,
        permittivity,
        conductivity,
        permeability,
        resolution,
        pml_regions=None,
        plane_2d="xy",
    ):
        """Initialize field arrays on a Yee grid for 2D (all 6 components) or 3D (Ex, Ey, Ez, Hx, Hy, Hz) simulations."""
        self.resolution = resolution
        self.plane_2d = plane_2d
        # Store references to material grids owned by Design (convert to JAX arrays)
        self.permittivity = jnp.asarray(permittivity)
        self.conductivity = jnp.asarray(conductivity)
        self.permeability = jnp.asarray(permeability)

        # Initialize PML regions if present
        if pml_regions:
            # We don't use split-field anymore, but we might store regions for visualization
            self.has_pml = True
            self.pml_regions = pml_regions
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
            self._init_material_parameters_3d()
        else:
            dim1, dim2 = grid_shape
            self._init_fields_2d(dim1, dim2)
            self.update = self._update_2d
            self._curl_e_to_h = ops.curl_e_to_h_2d
            self._curl_h_to_e = ops.curl_h_to_e_2d
            self._init_material_parameters_2d()

    def set_pml_conductivity(self, pml_data):
        """Set effective conductivity for PML regions (replaces split-field initialization)."""
        self.has_pml = True
        # Convert PML data arrays to JAX
        self.pml_data = {
            k: jnp.asarray(v) if hasattr(v, "__array__") else v
            for k, v in pml_data.items()
        }

        # Recalculate material parameters with PML conductivity added
        if self.permittivity.ndim == 3:  # 3D
            self._init_material_parameters_3d()
        else:  # 2D
            self._init_material_parameters_2d()

    def _init_material_parameters_3d(self):
        """Initialize 3D material parameters including PML conductivity if present."""
        base_sigma = self.conductivity

        if self.has_pml and hasattr(self, "pml_data"):
            sigma_pml = jnp.zeros_like(base_sigma)
            if "sigma_x" in self.pml_data:
                sigma_pml = sigma_pml + self.pml_data["sigma_x"]
            if "sigma_y" in self.pml_data:
                sigma_pml = sigma_pml + self.pml_data["sigma_y"]
            if "sigma_z" in self.pml_data:
                sigma_pml = sigma_pml + self.pml_data["sigma_z"]
            # Use maximum to avoid double-counting if meshing.py already added it
            total_sigma = jnp.maximum(base_sigma, sigma_pml)
        else:
            total_sigma = base_sigma

        self.eps_x, self.sig_x, self.region_x = ops.material_slice_for_e_3d(
            self.permittivity, total_sigma, "x"
        )
        self.eps_y, self.sig_y, self.region_y = ops.material_slice_for_e_3d(
            self.permittivity, total_sigma, "y"
        )
        self.eps_z, self.sig_z, self.region_z = ops.material_slice_for_e_3d(
            self.permittivity, total_sigma, "z"
        )

        self.sigma_m_hx, self.sigma_m_hy, self.sigma_m_hz = (
            ops.magnetic_conductivity_terms_3d(
                total_sigma,
                self.permeability,
                self.Hx.shape,
                self.Hy.shape,
                self.Hz.shape,
            )
        )

    def _init_material_parameters_2d(self):
        """Initialize 2D material parameters including PML conductivity if present."""
        # Base conductivity from design
        base_sigma = self.conductivity

        # Add PML conductivity if present
        if self.has_pml and hasattr(self, "pml_data"):
            # Combine profiles based on plane
            sigma_pml = jnp.zeros_like(base_sigma)
            if self.plane_2d == "xy":
                if "sigma_x" in self.pml_data:
                    sigma_pml = sigma_pml + self.pml_data["sigma_x"]
                if "sigma_y" in self.pml_data:
                    sigma_pml = sigma_pml + self.pml_data["sigma_y"]
            elif self.plane_2d == "yz":
                if "sigma_y" in self.pml_data:
                    sigma_pml = sigma_pml + self.pml_data["sigma_y"]
                if "sigma_z" in self.pml_data:
                    sigma_pml = sigma_pml + self.pml_data["sigma_z"]
            elif self.plane_2d == "xz":
                if "sigma_x" in self.pml_data:
                    sigma_pml = sigma_pml + self.pml_data["sigma_x"]
                if "sigma_z" in self.pml_data:
                    sigma_pml = sigma_pml + self.pml_data["sigma_z"]

            total_sigma = base_sigma + sigma_pml
        else:
            total_sigma = base_sigma

        # Initialize slicing and magnetic terms based on plane
        # Note: We use the same total_sigma for all components as a simplification
        # Ideally, we should stagger sigma for each component, but grid-colocated sigma is a standard approx

        # Setup slices for all 3 E-components
        self.eps_x, self.sig_x, self.region_x = ops.material_slice_for_e_2d_component(
            self.permittivity, total_sigma, "x", self.plane_2d
        )
        self.eps_y, self.sig_y, self.region_y = ops.material_slice_for_e_2d_component(
            self.permittivity, total_sigma, "y", self.plane_2d
        )
        self.eps_z, self.sig_z, self.region_z = ops.material_slice_for_e_2d_component(
            self.permittivity, total_sigma, "z", self.plane_2d
        )

        # Setup magnetic conductivity for H-field updates
        self.sigma_m_hx, self.sigma_m_hy, self.sigma_m_hz = (
            ops.magnetic_conductivity_terms_2d_full(
                total_sigma,
                self.permeability,
                self.Hx.shape,
                self.Hy.shape,
                self.Hz.shape,
                self.plane_2d,
            )
        )

    def _init_upml_fields(self, pml_regions):
        """Initialize auxiliary fields for split-field UPML. (Deprecated/No-op)"""
        pass

    def _init_split_fields_2d(self):
        """Initialize split-field components for 2D UPML. (Deprecated/No-op)"""
        pass

    def _init_split_fields_3d(self):
        """Initialize split-field components for 3D UPML."""
        if self.has_pml:
            # 3D: all components split
            self.Ex_y = jnp.zeros_like(self.Ex)
            self.Ex_z = jnp.zeros_like(self.Ex)
            self.Ey_x = jnp.zeros_like(self.Ey)
            self.Ey_z = jnp.zeros_like(self.Ey)
            self.Ez_x = jnp.zeros_like(self.Ez)
            self.Ez_y = jnp.zeros_like(self.Ez)
            # Similar for H fields...

    def _init_fields_3d(self, nx, ny, nz):
        """Initialize 3D field arrays (Ex, Ey, Ez, Hx, Hy, Hz) with proper Yee grid staggering."""
        self.Ex = jnp.zeros((nz, ny, nx - 1))
        self.Ey = jnp.zeros((nz, ny - 1, nx))
        self.Ez = jnp.zeros((nz - 1, ny, nx))
        self.Hx = jnp.zeros((nz - 1, ny - 1, nx))
        self.Hy = jnp.zeros((nz - 1, ny, nx - 1))
        self.Hz = jnp.zeros((nz, ny - 1, nx - 1))

    def _init_fields_2d(self, dim1, dim2):
        """Initialize 2D field arrays (Ex, Ey, Ez, Hx, Hy, Hz) on staggered Yee grid for the selected plane."""
        # dim1, dim2 correspond to the two active dimensions
        # xy: (y, x), yz: (z, y), xz: (z, x)

        if self.plane_2d == "xy":
            ny, nx = dim1, dim2
            # TM set (Ez, Hx, Hy)
            self.Ez = jnp.zeros((ny, nx))
            self.Hx = jnp.zeros((ny, nx - 1))
            self.Hy = jnp.zeros((ny - 1, nx))
            # TE set (Hz, Ex, Ey)
            self.Hz = jnp.zeros((ny - 1, nx - 1))
            self.Ex = jnp.zeros((ny, nx - 1))
            self.Ey = jnp.zeros((ny - 1, nx))

        elif self.plane_2d == "yz":
            nz, ny = dim1, dim2
            # Invariant in x. We map standard 3D staggering to 2D slice.
            # 3D: Ex(z,y,x-1/2), Ey(z,y-1/2,x), Ez(z-1/2,y,x)
            # 2D yz (x-invariant):
            # Ex is normal to plane -> (nz, ny) [like Ez in xy]
            # Ey is in plane -> (nz, ny-1)
            # Ez is in plane -> (nz-1, ny)

            # TE-like set (Ex, Hy, Hz)
            self.Ex = jnp.zeros((nz, ny))
            self.Hy = jnp.zeros((nz, ny - 1))
            self.Hz = jnp.zeros((nz - 1, ny))

            # TM-like set (Hx, Ey, Ez)
            self.Hx = jnp.zeros((nz - 1, ny - 1))
            self.Ey = jnp.zeros((nz, ny - 1))
            self.Ez = jnp.zeros((nz - 1, ny))

        elif self.plane_2d == "xz":
            nz, nx = dim1, dim2
            # Invariant in y.
            # Ey is normal to plane -> (nz, nx)
            # Ex is in plane -> (nz, nx-1)
            # Ez is in plane -> (nz-1, nx)

            # TE-like set (Ey, Hx, Hz)
            self.Ey = jnp.zeros((nz, nx))
            self.Hx = jnp.zeros((nz, nx - 1))
            self.Hz = jnp.zeros((nz - 1, nx))

            # TM-like set (Hy, Ex, Ez)
            self.Hy = jnp.zeros((nz - 1, nx - 1))
            self.Ex = jnp.zeros((nz, nx - 1))
            self.Ez = jnp.zeros((nz - 1, nx))

    def available_components(self):
        """Return list of available field components."""
        if self.permittivity.ndim == 3:
            return ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        else:
            # All 6 are available in 2D full mode
            return ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

    def _update_2d(self, dt, source_j=None, source_m=None):
        """Execute one 2D FDTD time step (all 6 components) for the selected plane."""
        # 1. Update H fields from E fields
        curlE_x, curlE_y, curlE_z = self._curl_e_to_h(
            (self.Ex, self.Ey, self.Ez), self.resolution, plane=self.plane_2d
        )

        # Inject magnetic sources (if any) - use .at[].add() for JAX functional updates
        if source_m:
            if "Hx" in source_m:
                val, indices = source_m["Hx"]
                curlE_x = curlE_x.at[indices].add(val)
            if "Hy" in source_m:
                val, indices = source_m["Hy"]
                curlE_y = curlE_y.at[indices].add(val)
            if "Hz" in source_m:
                val, indices = source_m["Hz"]
                curlE_z = curlE_z.at[indices].add(val)

        self.Hx = ops.advance_h_field(self.Hx, curlE_x, self.sigma_m_hx, dt)
        self.Hy = ops.advance_h_field(self.Hy, curlE_y, self.sigma_m_hy, dt)
        self.Hz = ops.advance_h_field(self.Hz, curlE_z, self.sigma_m_hz, dt)

        # 2. Update E fields from H fields
        curlH_x, curlH_y, curlH_z = self._curl_h_to_e(
            (self.Hx, self.Hy, self.Hz),
            self.resolution,
            (self.Ex.shape, self.Ey.shape, self.Ez.shape),
            plane=self.plane_2d,
        )

        # Inject electric sources (if any) - use .at[].add() for JAX functional updates
        if source_j:
            if "Ex" in source_j:
                val, indices = source_j["Ex"]
                curlH_x = curlH_x.at[indices].add(val)
            if "Ey" in source_j:
                val, indices = source_j["Ey"]
                curlH_y = curlH_y.at[indices].add(val)
            if "Ez" in source_j:
                val, indices = source_j["Ez"]
                curlH_z = curlH_z.at[indices].add(val)

        self.Ex = ops.advance_e_field(
            self.Ex, curlH_x, self.sig_x, self.eps_x, dt, self.region_x
        )
        self.Ey = ops.advance_e_field(
            self.Ey, curlH_y, self.sig_y, self.eps_y, dt, self.region_y
        )
        self.Ez = ops.advance_e_field(
            self.Ez, curlH_z, self.sig_z, self.eps_z, dt, self.region_z
        )

    def _update_3d(self, dt, source_j=None, source_m=None):
        """Execute one 3D FDTD time step: H from curl(E) via Faraday's law, then E from curl(H) via Ampere's law."""
        # 1. Update H fields from E fields
        curlE_x, curlE_y, curlE_z = self._curl_e_to_h(
            self.Ex, self.Ey, self.Ez, self.resolution
        )

        # Inject magnetic sources (if any) - use .at[].add() for JAX functional updates
        if source_m:
            if "Hx" in source_m:
                val, indices = source_m["Hx"]
                curlE_x = curlE_x.at[indices].add(val)
            if "Hy" in source_m:
                val, indices = source_m["Hy"]
                curlE_y = curlE_y.at[indices].add(val)
            if "Hz" in source_m:
                val, indices = source_m["Hz"]
                curlE_z = curlE_z.at[indices].add(val)

        self.Hx = ops.advance_h_field(self.Hx, curlE_x, self.sigma_m_hx, dt)
        self.Hy = ops.advance_h_field(self.Hy, curlE_y, self.sigma_m_hy, dt)
        self.Hz = ops.advance_h_field(self.Hz, curlE_z, self.sigma_m_hz, dt)

        # 2. Update E fields from H fields
        curlH_x, curlH_y, curlH_z = self._curl_h_to_e(
            self.Hx,
            self.Hy,
            self.Hz,
            self.resolution,
            ex_shape=self.Ex.shape,
            ey_shape=self.Ey.shape,
            ez_shape=self.Ez.shape,
        )

        # Inject electric sources (if any) - use .at[].add() for JAX functional updates
        if source_j:
            if "Ex" in source_j:
                val, indices = source_j["Ex"]
                curlH_x = curlH_x.at[indices].add(val)
            if "Ey" in source_j:
                val, indices = source_j["Ey"]
                curlH_y = curlH_y.at[indices].add(val)
            if "Ez" in source_j:
                val, indices = source_j["Ez"]
                curlH_z = curlH_z.at[indices].add(val)

        self.Ex = ops.advance_e_field(
            self.Ex, curlH_x, self.sig_x, self.eps_x, dt, self.region_x
        )
        self.Ey = ops.advance_e_field(
            self.Ey, curlH_y, self.sig_y, self.eps_y, dt, self.region_y
        )
        self.Ez = ops.advance_e_field(
            self.Ez, curlH_z, self.sig_z, self.eps_z, dt, self.region_z
        )

    def update(self, dt, source_j=None, source_m=None):
        """Execute one FDTD time step with optional source injection."""
        if not self.permittivity.ndim == 3:  # 2D
            self._update_2d(dt, source_j=source_j, source_m=source_m)
        else:  # 3D
            self._update_3d(dt, source_j=source_j, source_m=source_m)
