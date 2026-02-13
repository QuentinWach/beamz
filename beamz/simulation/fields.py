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
        plane_2d="xy",
        _init_materials=True,
    ):
        """Initialize field arrays on a Yee grid for 2D (all 6 components) or 3D (Ex, Ey, Ez, Hx, Hy, Hz) simulations."""
        self.resolution = resolution
        self.plane_2d = plane_2d
        # Store references to material grids owned by Design (convert to JAX arrays)
        self.permittivity = jnp.asarray(permittivity)
        self.conductivity = jnp.asarray(conductivity)
        self.permeability = jnp.asarray(permeability)

        self.has_pml = False

        # Infer dimensionality and shape from material arrays
        is_3d = self.permittivity.ndim == 3
        grid_shape = self.permittivity.shape

        if is_3d:
            nz, ny, nx = grid_shape
            self._init_fields_3d(nx, ny, nz)
        else:
            dim1, dim2 = grid_shape
            self._init_fields_2d(dim1, dim2)

        if _init_materials:
            self._init_material_parameters()

    def set_pml_conductivity(self, pml_data):
        """Set effective conductivity for PML regions."""
        self.has_pml = True
        # Convert PML data arrays to JAX
        self.pml_data = {
            k: jnp.asarray(v) if hasattr(v, "__array__") else v
            for k, v in pml_data.items()
        }
        self._init_material_parameters()

    def _init_material_parameters(self):
        """Initialize material parameters including PML conductivity if present."""
        is_3d = self.permittivity.ndim == 3
        base_sigma = self.conductivity

        if self.has_pml and hasattr(self, "pml_data"):
            sigma_pml = jnp.zeros_like(base_sigma)
            if is_3d:
                pml_keys = ("sigma_x", "sigma_y", "sigma_z")
            else:
                pml_keys = {
                    "xy": ("sigma_x", "sigma_y"),
                    "yz": ("sigma_y", "sigma_z"),
                    "xz": ("sigma_x", "sigma_z"),
                }.get(self.plane_2d, ())
            for key in pml_keys:
                if key in self.pml_data:
                    sigma_pml = sigma_pml + self.pml_data[key]
            total_sigma = jnp.maximum(base_sigma, sigma_pml)
        else:
            total_sigma = base_sigma

        if is_3d:
            for comp in ("x", "y", "z"):
                eps, sig, region = ops.material_slice_for_e_3d(
                    self.permittivity, total_sigma, comp
                )
                setattr(self, f"eps_{comp}", eps)
                setattr(self, f"sig_{comp}", sig)
                setattr(self, f"region_{comp}", region)

            self.sigma_m_hx, self.sigma_m_hy, self.sigma_m_hz = (
                ops.magnetic_conductivity_terms_3d(
                    total_sigma,
                    self.permeability,
                    self.Hx.shape,
                    self.Hy.shape,
                    self.Hz.shape,
                )
            )
        else:
            for comp in ("x", "y", "z"):
                eps, sig, region = ops.material_slice_for_e_2d_component(
                    self.permittivity, total_sigma, comp, self.plane_2d
                )
                setattr(self, f"eps_{comp}", eps)
                setattr(self, f"sig_{comp}", sig)
                setattr(self, f"region_{comp}", region)

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
        return ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

    def update(self, dt, source_j=None, source_m=None):
        """Execute one FDTD time step (2D or 3D) with optional source injection."""
        is_3d = self.permittivity.ndim == 3

        # 1. Compute curl(E) for H update
        if is_3d:
            curlE_x, curlE_y, curlE_z = ops.curl_e_to_h_3d(
                self.Ex, self.Ey, self.Ez, self.resolution
            )
        else:
            curlE_x, curlE_y, curlE_z = ops.curl_e_to_h_2d(
                (self.Ex, self.Ey, self.Ez), self.resolution, plane=self.plane_2d
            )

        # Inject magnetic sources
        if source_m:
            for comp in ("Hx", "Hy", "Hz"):
                if comp in source_m:
                    for val, indices in source_m[comp]:
                        if comp == "Hx":
                            curlE_x = curlE_x.at[indices].add(val)
                        elif comp == "Hy":
                            curlE_y = curlE_y.at[indices].add(val)
                        else:
                            curlE_z = curlE_z.at[indices].add(val)

        self.Hx = ops.advance_h_field(self.Hx, curlE_x, self.sigma_m_hx, dt)
        self.Hy = ops.advance_h_field(self.Hy, curlE_y, self.sigma_m_hy, dt)
        self.Hz = ops.advance_h_field(self.Hz, curlE_z, self.sigma_m_hz, dt)

        # 2. Compute curl(H) for E update
        if is_3d:
            curlH_x, curlH_y, curlH_z = ops.curl_h_to_e_3d(
                self.Hx,
                self.Hy,
                self.Hz,
                self.resolution,
                ex_shape=self.Ex.shape,
                ey_shape=self.Ey.shape,
                ez_shape=self.Ez.shape,
            )
        else:
            curlH_x, curlH_y, curlH_z = ops.curl_h_to_e_2d(
                (self.Hx, self.Hy, self.Hz),
                self.resolution,
                (self.Ex.shape, self.Ey.shape, self.Ez.shape),
                plane=self.plane_2d,
            )

        # Inject electric sources
        if source_j:
            for comp in ("Ex", "Ey", "Ez"):
                if comp in source_j:
                    for val, indices in source_j[comp]:
                        if comp == "Ex":
                            curlH_x = curlH_x.at[indices].add(val)
                        elif comp == "Ey":
                            curlH_y = curlH_y.at[indices].add(val)
                        else:
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

    def update_materials(self, permittivity=None, conductivity=None, permeability=None):
        """Update material grids and recompute Yee parameters."""
        if permittivity is not None:
            self.permittivity = jnp.asarray(permittivity)
        if conductivity is not None:
            self.conductivity = jnp.asarray(conductivity)
        if permeability is not None:
            self.permeability = jnp.asarray(permeability)
        self._init_material_parameters()
