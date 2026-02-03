try:
    import jax.numpy as jnp
except ImportError:  # Fallback for environments without JAX
    import numpy as jnp


# =============================================================================
# LEGACY MATERIALS (CONSTANT)
# =============================================================================
# Medium: Dispersionless medium.
class Material:
    def __init__(
        self,
        permittivity=1.0,
        permeability=1.0,
        conductivity=0.0,
        k=0.0,
        rho=0.0,
        cp=0.0,
        dn_dT=0.0,
        T0=300.0,
    ):
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity
        # Thermal parameters (per-material constants)
        self.k = k
        self.rho = rho
        self.cp = cp
        self.dn_dT = dn_dT
        self.T0 = T0

    def get_sample(self):
        return self.permittivity, self.permeability, self.conductivity


# CustomMaterial: Function-based material for inverse design
class CustomMaterial:
    def __init__(
        self,
        permittivity_func=None,
        permeability_func=None,
        conductivity_func=None,
        permittivity_grid=None,
        permeability_grid=None,
        conductivity_grid=None,
        k=0.0,
        rho=0.0,
        cp=0.0,
        dn_dT=0.0,
        T0=300.0,
        bounds=None,
        interpolation="linear",
    ):
        """
        Custom material with spatially-varying properties for inverse design.

        Args:
            permittivity_func: Function that takes (x, y) or (x, y, z) and returns permittivity
            permeability_func: Function that takes (x, y) or (x, y, z) and returns permeability
            conductivity_func: Function that takes (x, y) or (x, y, z) and returns conductivity
            permittivity_grid: 2D numpy array of permittivity values for grid-based interpolation
            permeability_grid: 2D numpy array of permeability values for grid-based interpolation
            conductivity_grid: 2D numpy array of conductivity values for grid-based interpolation
            bounds: Tuple ((x_min, x_max), (y_min, y_max)) defining the spatial bounds for grid interpolation
            interpolation: 'linear', 'cubic', or 'nearest' for grid interpolation

        Examples:
            # Function-based material
            def perm_func(x, y):
                return 2.0 + 0.5 * np.sin(x) * np.cos(y)
            material = CustomMaterial(permittivity_func=perm_func)

            # Grid-based material for inverse design
            perm_grid = np.ones((50, 50)) * 2.0
            perm_grid[20:30, 20:30] = 4.0  # High index region
            material = CustomMaterial(
                permittivity_grid=perm_grid,
                bounds=((0, 10e-6), (0, 10e-6))  # 10 micron x 10 micron
            )
        """
        import numpy as np

        # Store function-based definitions
        self.permittivity_func = permittivity_func
        self.permeability_func = permeability_func
        self.conductivity_func = conductivity_func

        # Store grid-based definitions
        self.permittivity_grid = permittivity_grid
        self.permeability_grid = permeability_grid
        self.conductivity_grid = conductivity_grid
        # Thermal parameters (per-material constants)
        self.k = k
        self.rho = rho
        self.cp = cp
        self.dn_dT = dn_dT
        self.T0 = T0

        # Validate bounds if grid is provided
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError(f"bounds must be ((x_min, x_max), (y_min, y_max)), got {bounds}")
            if bounds[0][0] >= bounds[0][1]:
                raise ValueError(f"Invalid x bounds: x_min={bounds[0][0]} >= x_max={bounds[0][1]}")
            if bounds[1][0] >= bounds[1][1]:
                raise ValueError(f"Invalid y bounds: y_min={bounds[1][0]} >= y_max={bounds[1][1]}")

        # Spatial bounds for grid interpolation
        self.bounds = bounds
        self.interpolation = interpolation

        # Default values
        self.default_permittivity = 1.0
        self.default_permeability = 1.0
        self.default_conductivity = 0.0

        # Create interpolation functions for grids
        if permittivity_grid is not None and bounds is not None:
            self._create_grid_interpolator("permittivity")
        if permeability_grid is not None and bounds is not None:
            self._create_grid_interpolator("permeability")
        if conductivity_grid is not None and bounds is not None:
            self._create_grid_interpolator("conductivity")

    @property
    def permittivity(self):
        """Return representative permittivity for display purposes."""
        if self.permittivity_grid is not None:
            import numpy as np

            return f"grid({np.min(self.permittivity_grid):.3f}-{np.max(self.permittivity_grid):.3f})"
        elif self.permittivity_func is not None:
            return "function"
        else:
            return self.default_permittivity

    @property
    def permeability(self):
        """Return representative permeability for display purposes."""
        if self.permeability_grid is not None:
            import numpy as np

            return f"grid({np.min(self.permeability_grid):.3f}-{np.max(self.permeability_grid):.3f})"
        elif self.permeability_func is not None:
            return "function"
        else:
            return self.default_permeability

    @property
    def conductivity(self):
        """Return representative conductivity for display purposes."""
        if self.conductivity_grid is not None:
            import numpy as np

            return f"grid({np.min(self.conductivity_grid):.3f}-{np.max(self.conductivity_grid):.3f})"
        elif self.conductivity_func is not None:
            return "function"
        else:
            return self.default_conductivity

    def _create_grid_interpolator(self, property_name):
        """Create scipy interpolator for grid-based material property."""
        try:
            import numpy as np
            from scipy.interpolate import RegularGridInterpolator

            grid = getattr(self, f"{property_name}_grid")
            if grid is None:
                return

            # Create coordinate arrays
            x_coords = np.linspace(self.bounds[0][0], self.bounds[0][1], grid.shape[1])
            y_coords = np.linspace(self.bounds[1][0], self.bounds[1][1], grid.shape[0])

            # Create interpolator
            interpolator = RegularGridInterpolator(
                (y_coords, x_coords),
                grid,
                method=self.interpolation,
                bounds_error=False,
                fill_value=getattr(self, f"default_{property_name}"),
            )

            # Store interpolator
            setattr(self, f"_{property_name}_interpolator", interpolator)

        except ImportError:
            print("Warning: scipy not available, using nearest neighbor interpolation")
            setattr(self, f"_{property_name}_interpolator", None)

    def get_permittivity(self, x, y, z=None):
        """Get permittivity at spatial coordinates (x, y, z)."""
        if self.permittivity_func is not None:
            if z is not None:
                return self.permittivity_func(x, y, z)
            else:
                return self.permittivity_func(x, y)
        elif (
            hasattr(self, "_permittivity_interpolator")
            and self._permittivity_interpolator is not None
        ):
            import numpy as np

            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._permittivity_interpolator(points)
        else:
            return self.default_permittivity

    def get_permeability(self, x, y, z=None):
        """Get permeability at spatial coordinates (x, y, z)."""
        if self.permeability_func is not None:
            if z is not None:
                return self.permeability_func(x, y, z)
            else:
                return self.permeability_func(x, y)
        elif (
            hasattr(self, "_permeability_interpolator")
            and self._permeability_interpolator is not None
        ):
            import numpy as np

            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._permeability_interpolator(points)
        else:
            return self.default_permeability

    def get_conductivity(self, x, y, z=None):
        """Get conductivity at spatial coordinates (x, y, z)."""
        if self.conductivity_func is not None:
            if z is not None:
                return self.conductivity_func(x, y, z)
            else:
                return self.conductivity_func(x, y)
        elif (
            hasattr(self, "_conductivity_interpolator")
            and self._conductivity_interpolator is not None
        ):
            import numpy as np

            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._conductivity_interpolator(points)
        else:
            return self.default_conductivity

    def get_sample(self, x=0, y=0, z=None):
        """Get material properties at spatial coordinates for backward compatibility."""
        return (
            self.get_permittivity(x, y, z),
            self.get_permeability(x, y, z),
            self.get_conductivity(x, y, z),
        )

    def update_grid(self, property_name, new_grid):
        """Update material property grid (for optimization)."""
        if property_name == "permittivity":
            self.permittivity_grid = new_grid
            self._create_grid_interpolator("permittivity")
        elif property_name == "permeability":
            self.permeability_grid = new_grid
            self._create_grid_interpolator("permeability")
        elif property_name == "conductivity":
            self.conductivity_grid = new_grid
            self._create_grid_interpolator("conductivity")
        else:
            raise ValueError(f"Unknown property: {property_name}")

    def copy(self):
        """Create a deep copy of the CustomMaterial."""
        import numpy as np

        # Deep copy grids if they exist
        perm_grid = (
            self.permittivity_grid.copy()
            if self.permittivity_grid is not None
            else None
        )
        permeability_grid = (
            self.permeability_grid.copy()
            if self.permeability_grid is not None
            else None
        )
        cond_grid = (
            self.conductivity_grid.copy()
            if self.conductivity_grid is not None
            else None
        )

        # Create new CustomMaterial with copied data
        return CustomMaterial(
            permittivity_func=self.permittivity_func,  # Functions can be shared
            permeability_func=self.permeability_func,
            conductivity_func=self.conductivity_func,
            permittivity_grid=perm_grid,  # Deep copied grids
            permeability_grid=permeability_grid,
            conductivity_grid=cond_grid,
            k=self.k,
            rho=self.rho,
            cp=self.cp,
            dn_dT=self.dn_dT,
            T0=self.T0,
            bounds=self.bounds,  # Bounds can be shared (tuples are immutable)
            interpolation=self.interpolation,
        )


# =============================================================================
# TEMPERATURE-DEPENDENT MATERIAL MODELS
# =============================================================================


class MaterialModel:
    """Temperature-dependent material model base class."""

    def __init__(self, T_ref=300.0, name=None):
        self.T_ref = float(T_ref)
        self.name = name or self.__class__.__name__

    def epsilon_r(self, T, omega=None):
        raise NotImplementedError

    def conductivity(self, T, omega=None):
        return 0.0

    def permeability(self, T, omega=None):
        return 1.0

    def thermal_k(self, T):
        return 0.0

    def heat_capacity(self, T):
        return 0.0

    def density(self, T):
        return 0.0

    @property
    def permittivity(self):
        value = self.epsilon_r(self.T_ref)
        return value.item() if hasattr(value, "item") else value

    @property
    def conductivity_ref(self):
        value = self.conductivity(self.T_ref)
        return value.item() if hasattr(value, "item") else value

    @property
    def permeability_ref(self):
        value = self.permeability(self.T_ref)
        return value.item() if hasattr(value, "item") else value

    @property
    def k(self):
        value = self.thermal_k(self.T_ref)
        return value.item() if hasattr(value, "item") else value

    @property
    def rho(self):
        value = self.density(self.T_ref)
        return value.item() if hasattr(value, "item") else value

    @property
    def cp(self):
        value = self.heat_capacity(self.T_ref)
        return value.item() if hasattr(value, "item") else value


class LinearThermoOpticMaterial(MaterialModel):
    """Linear temperature-dependent material model with optional linear thermal props."""

    def __init__(
        self,
        n0,
        dn_dT=0.0,
        mu_r=1.0,
        sigma0=0.0,
        dsigma_dT=0.0,
        k0=0.0,
        dk_dT=0.0,
        rho0=0.0,
        drho_dT=0.0,
        cp0=0.0,
        dcp_dT=0.0,
        T_ref=300.0,
        name=None,
    ):
        super().__init__(T_ref=T_ref, name=name)
        self.n0 = float(n0)
        self.dn_dT = float(dn_dT)
        self.mu_r = float(mu_r)
        self.sigma0 = float(sigma0)
        self.dsigma_dT = float(dsigma_dT)
        self.k0 = float(k0)
        self.dk_dT = float(dk_dT)
        self.rho0 = float(rho0)
        self.drho_dT = float(drho_dT)
        self.cp0 = float(cp0)
        self.dcp_dT = float(dcp_dT)

    def epsilon_r(self, T, omega=None):
        T = jnp.asarray(T)
        n = self.n0 + self.dn_dT * (T - self.T_ref)
        return n * n

    def conductivity(self, T, omega=None):
        T = jnp.asarray(T)
        return self.sigma0 + self.dsigma_dT * (T - self.T_ref)

    def permeability(self, T, omega=None):
        return self.mu_r

    def thermal_k(self, T):
        T = jnp.asarray(T)
        return self.k0 + self.dk_dT * (T - self.T_ref)

    def heat_capacity(self, T):
        T = jnp.asarray(T)
        return self.cp0 + self.dcp_dT * (T - self.T_ref)

    def density(self, T):
        T = jnp.asarray(T)
        return self.rho0 + self.drho_dT * (T - self.T_ref)


class ConstantMaterialModel(MaterialModel):
    """Material model with constant permittivity and optional linear conductivity/thermal props."""

    def __init__(
        self,
        permittivity,
        permeability=1.0,
        conductivity=0.0,
        dconductivity_dT=0.0,
        k0=0.0,
        dk_dT=0.0,
        rho0=0.0,
        drho_dT=0.0,
        cp0=0.0,
        dcp_dT=0.0,
        T_ref=300.0,
        name=None,
    ):
        super().__init__(T_ref=T_ref, name=name)
        self._permittivity = permittivity
        self._permeability = permeability
        self._conductivity = conductivity
        self._dconductivity_dT = dconductivity_dT
        self.k0 = k0
        self.dk_dT = dk_dT
        self.rho0 = rho0
        self.drho_dT = drho_dT
        self.cp0 = cp0
        self.dcp_dT = dcp_dT

    def epsilon_r(self, T, omega=None):
        return self._permittivity

    def conductivity(self, T, omega=None):
        T = jnp.asarray(T)
        return self._conductivity + self._dconductivity_dT * (T - self.T_ref)

    def permeability(self, T, omega=None):
        return self._permeability

    def thermal_k(self, T):
        T = jnp.asarray(T)
        return self.k0 + self.dk_dT * (T - self.T_ref)

    def heat_capacity(self, T):
        T = jnp.asarray(T)
        return self.cp0 + self.dcp_dT * (T - self.T_ref)

    def density(self, T):
        T = jnp.asarray(T)
        return self.rho0 + self.drho_dT * (T - self.T_ref)


class ConstantMaterialAdapter(MaterialModel):
    """Adapter for legacy Material to a MaterialModel interface."""

    def __init__(self, material: Material):
        super().__init__(T_ref=getattr(material, "T0", 300.0), name=getattr(material, "name", None))
        self.material = material
        self._permittivity = getattr(material, "permittivity", 1.0)
        self._permeability = getattr(material, "permeability", 1.0)
        self._conductivity = getattr(material, "conductivity", 0.0)
        self._k = getattr(material, "k", 0.0)
        self._rho = getattr(material, "rho", 0.0)
        self._cp = getattr(material, "cp", 0.0)
        self.dn_dT = getattr(material, "dn_dT", 0.0)

    def epsilon_r(self, T, omega=None):
        return self._permittivity

    def conductivity(self, T, omega=None):
        return self._conductivity

    def permeability(self, T, omega=None):
        return self._permeability

    def thermal_k(self, T):
        return self._k

    def heat_capacity(self, T):
        return self._cp

    def density(self, T):
        return self._rho


class CustomMaterialAdapter(MaterialModel):
    """Adapter for CustomMaterial with constant thermal properties."""

    def __init__(self, material: CustomMaterial):
        super().__init__(T_ref=getattr(material, "T0", 300.0), name=getattr(material, "name", None))
        self.material = material
        self._permittivity = getattr(material, "default_permittivity", 1.0)
        self._permeability = getattr(material, "default_permeability", 1.0)
        self._conductivity = getattr(material, "default_conductivity", 0.0)
        self._k = getattr(material, "k", 0.0)
        self._rho = getattr(material, "rho", 0.0)
        self._cp = getattr(material, "cp", 0.0)
        self.dn_dT = getattr(material, "dn_dT", 0.0)

    def epsilon_r(self, T, omega=None):
        return self._permittivity

    def conductivity(self, T, omega=None):
        return self._conductivity

    def permeability(self, T, omega=None):
        return self._permeability

    def thermal_k(self, T):
        return self._k

    def heat_capacity(self, T):
        return self._cp

    def density(self, T):
        return self._rho


def is_spatial_material(material) -> bool:
    return hasattr(material, "get_permittivity")


def as_material_model(material):
    if isinstance(material, MaterialModel):
        return material
    if isinstance(material, Material):
        return ConstantMaterialAdapter(material)
    if isinstance(material, CustomMaterial):
        return CustomMaterialAdapter(material)
    return None


# =============================================================================
# ADVANCED DISPERSIVE MATERIAL MODELS
# =============================================================================
#
# These classes implement frequency-dependent material models for accurate
# electromagnetic simulations across a range of frequencies/wavelengths.
#
# For FDTD simulations, dispersive materials are typically handled using
# auxiliary differential equations (ADE) or recursive convolution methods.
# These classes provide the material parameters needed for such implementations.
#
# =============================================================================

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from beamz.const import LIGHT_SPEED, EPS_0
except ImportError:
    LIGHT_SPEED = 299792458.0
    EPS_0 = 8.85e-12


class SellmeierMaterial:
    """
    Dispersive medium described by the Sellmeier equation.

<<<<<<< HEAD
    The Sellmeier equation models the wavelength-dependent refractive index
    of transparent dielectric materials:

        n²(λ) = 1 + Σᵢ (Bᵢ * λ²) / (λ² - Cᵢ)

    where λ is wavelength, Bᵢ are dimensionless coefficients, and Cᵢ are
    wavelength-squared constants (resonance wavelengths squared).

    This model is accurate for wavelengths far from material resonances
    (i.e., in the transparent region).

    Args:
        B_coeffs: List of B coefficients (dimensionless)
        C_coeffs: List of C coefficients in m² (wavelength² of resonances)
        name: Optional material name

    Example:
        # Fused silica (SiO2) Sellmeier coefficients
        sio2 = SellmeierMaterial(
            B_coeffs=[0.6961663, 0.4079426, 0.8974794],
            C_coeffs=[0.0684043e-12, 0.1162414e-12, 9.896161e-12],
            name="SiO2"
        )
        n = sio2.get_refractive_index(1.55e-6)  # n at 1550nm
    """

    def __init__(self, B_coeffs, C_coeffs, name="SellmeierMaterial"):
        if len(B_coeffs) != len(C_coeffs):
            raise ValueError("B_coeffs and C_coeffs must have same length")

        self.B_coeffs = np.array(B_coeffs)
        self.C_coeffs = np.array(C_coeffs)
        self.name = name
        self.num_poles = len(B_coeffs)

    def get_refractive_index(self, wavelength):
        """
        Calculate refractive index at given wavelength.

        Args:
            wavelength: Wavelength in meters (scalar or array)

        Returns:
            Refractive index n (real part)
        """
        wavelength = np.atleast_1d(wavelength)
        wl_sq = wavelength**2

        n_sq = np.ones_like(wl_sq)
        for B, C in zip(self.B_coeffs, self.C_coeffs):
            n_sq = n_sq + B * wl_sq / (wl_sq - C)

        return np.sqrt(n_sq).squeeze()

    def get_permittivity(self, wavelength):
        """
        Calculate relative permittivity at given wavelength.

        Args:
            wavelength: Wavelength in meters

        Returns:
            Relative permittivity ε_r
        """
        n = self.get_refractive_index(wavelength)
        return n**2

    def get_group_index(self, wavelength, delta=1e-9):
        """
        Calculate group index n_g = n - λ(dn/dλ).

        Args:
            wavelength: Wavelength in meters
            delta: Wavelength step for numerical derivative

        Returns:
            Group index n_g
        """
        n = self.get_refractive_index(wavelength)
        n_plus = self.get_refractive_index(wavelength + delta)
        n_minus = self.get_refractive_index(wavelength - delta)
        dn_dlambda = (n_plus - n_minus) / (2 * delta)
        return n - wavelength * dn_dlambda

    def get_dispersion(self, wavelength, delta=1e-9):
        """
        Calculate group velocity dispersion D = -(λ/c)(d²n/dλ²) in ps/(nm·km).

        Args:
            wavelength: Wavelength in meters
            delta: Wavelength step for numerical derivative

        Returns:
            Dispersion D in ps/(nm·km)
        """
        n_plus = self.get_refractive_index(wavelength + delta)
        n = self.get_refractive_index(wavelength)
        n_minus = self.get_refractive_index(wavelength - delta)
        d2n_dlambda2 = (n_plus - 2*n + n_minus) / delta**2

        # Convert to ps/(nm·km)
        D = -(wavelength / LIGHT_SPEED) * d2n_dlambda2
        D = D * 1e12 * 1e-9 * 1e3  # s/m² -> ps/(nm·km)
        return D

    def to_material(self, wavelength):
        """
        Convert to a simple Material at specified wavelength.

        Args:
            wavelength: Wavelength in meters

        Returns:
            Material instance with permittivity at this wavelength
        """
        eps = self.get_permittivity(wavelength)
        return Material(permittivity=eps, permeability=1.0, conductivity=0.0)

    def __repr__(self):
        return f"SellmeierMaterial(name='{self.name}', poles={self.num_poles})"


class DrudeMaterial:
    """
    Dispersive medium described by the Drude model for free electrons.

    The Drude model describes the optical response of metals and
    heavily-doped semiconductors with free carriers:

        ε(ω) = ε_∞ - ωₚ² / (ω² + iγω)

    where:
        ε_∞ = high-frequency permittivity (background)
        ωₚ = plasma frequency (rad/s)
        γ = collision frequency / damping rate (rad/s)
        ω = angular frequency (rad/s)

    Args:
        plasma_freq: Plasma frequency ωₚ in rad/s
        collision_freq: Collision/damping frequency γ in rad/s
        eps_inf: High-frequency permittivity (default 1.0)
        name: Optional material name

    Example:
        # Gold at optical frequencies
        gold = DrudeMaterial(
            plasma_freq=1.37e16,
            collision_freq=4.05e13,
            name="Gold"
        )
        eps = gold.get_permittivity(wavelength=1.55e-6)
    """

    def __init__(self, plasma_freq, collision_freq, eps_inf=1.0, name="DrudeMaterial"):
        self.plasma_freq = plasma_freq  # ωₚ
        self.collision_freq = collision_freq  # γ
        self.eps_inf = eps_inf  # ε_∞
        self.name = name

        # Derived quantities
        self.plasma_wavelength = 2 * np.pi * LIGHT_SPEED / plasma_freq

    @classmethod
    def from_dc_conductivity(cls, dc_conductivity, eps_inf=1.0,
                             collision_freq=None, name="DrudeMaterial"):
        """
        Create DrudeMaterial from DC conductivity.

        For metals, σ_DC = ε₀ * ωₚ² / γ

        Args:
            dc_conductivity: DC conductivity in S/m
            eps_inf: High-frequency permittivity
            collision_freq: If known, collision frequency in rad/s
            name: Material name
        """
        if collision_freq is None:
            # Typical collision frequency for metals
            collision_freq = 1e14  # rad/s

        # ωₚ² = σ_DC * γ / ε₀
        plasma_freq_sq = dc_conductivity * collision_freq / EPS_0
        plasma_freq = np.sqrt(plasma_freq_sq)

        return cls(plasma_freq, collision_freq, eps_inf, name)

    def get_permittivity(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate complex permittivity at given wavelength/frequency.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Complex relative permittivity ε_r
        """
        if omega is None:
            if wavelength is not None:
                omega = 2 * np.pi * LIGHT_SPEED / wavelength
            elif frequency is not None:
                omega = 2 * np.pi * frequency
            else:
                raise ValueError("Must specify wavelength, frequency, or omega")

        omega = np.atleast_1d(omega)

        # Drude model: ε(ω) = ε_∞ - ωₚ² / (ω² + iγω)
        eps = self.eps_inf - self.plasma_freq**2 / (omega**2 + 1j * self.collision_freq * omega)

        return eps.squeeze()

    def get_refractive_index(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate complex refractive index n + ik.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Complex refractive index
        """
        eps = self.get_permittivity(wavelength, frequency, omega)
        return np.sqrt(eps)

    def get_conductivity(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate effective conductivity σ = ω * ε₀ * ε''

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Effective conductivity in S/m
        """
        if omega is None:
            if wavelength is not None:
                omega = 2 * np.pi * LIGHT_SPEED / wavelength
            elif frequency is not None:
                omega = 2 * np.pi * frequency

        eps = self.get_permittivity(omega=omega)
        return omega * EPS_0 * np.abs(eps.imag)

    def get_skin_depth(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate skin depth δ = λ / (2π * k).

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Skin depth in meters
        """
        if wavelength is None and omega is not None:
            wavelength = 2 * np.pi * LIGHT_SPEED / omega
        elif wavelength is None and frequency is not None:
            wavelength = LIGHT_SPEED / frequency

        n_complex = self.get_refractive_index(wavelength=wavelength)
        k = np.abs(n_complex.imag)
        return wavelength / (2 * np.pi * k)

    def to_material(self, wavelength):
        """
        Convert to a simple Material at specified wavelength.

        The loss is represented as conductivity for FDTD compatibility.

        Args:
            wavelength: Wavelength in meters

        Returns:
            Material instance
        """
        eps = self.get_permittivity(wavelength=wavelength)
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        conductivity = omega * EPS_0 * np.abs(eps.imag)
        return Material(permittivity=eps.real, permeability=1.0, conductivity=conductivity)

    def __repr__(self):
        return f"DrudeMaterial(name='{self.name}', ωₚ={self.plasma_freq:.2e}, γ={self.collision_freq:.2e})"


class LorentzMaterial:
    """
    Dispersive medium described by the Lorentz oscillator model.

    The Lorentz model describes bound electron oscillations and is
    used for dielectrics near absorption resonances:

        ε(ω) = ε_∞ + Σⱼ (Δεⱼ * ω₀ⱼ²) / (ω₀ⱼ² - ω² - iγⱼω)

    where:
        ε_∞ = high-frequency permittivity
        Δεⱼ = oscillator strength (dimensionless)
        ω₀ⱼ = resonance frequency (rad/s)
        γⱼ = damping rate (rad/s)

    Args:
        resonances: List of (delta_eps, omega_0, gamma) tuples or
                   list of dicts with keys 'delta_eps', 'omega_0', 'gamma'
        eps_inf: High-frequency permittivity (default 1.0)
        name: Optional material name

    Example:
        # Single Lorentz oscillator
        mat = LorentzMaterial(
            resonances=[(1.0, 5e15, 1e14)],  # (Δε, ω₀, γ)
            eps_inf=2.0,
            name="Lorentz Example"
        )

        # Multiple oscillators
        mat = LorentzMaterial(
            resonances=[
                {'delta_eps': 1.0, 'omega_0': 5e15, 'gamma': 1e14},
                {'delta_eps': 0.5, 'omega_0': 8e15, 'gamma': 2e14},
            ],
            eps_inf=1.0
        )
    """

    def __init__(self, resonances, eps_inf=1.0, name="LorentzMaterial"):
        self.eps_inf = eps_inf
        self.name = name

        # Parse resonances into uniform format
        self.resonances = []
        for res in resonances:
            if isinstance(res, dict):
                self.resonances.append({
                    'delta_eps': res['delta_eps'],
                    'omega_0': res['omega_0'],
                    'gamma': res['gamma']
                })
            else:
                # Assume tuple (delta_eps, omega_0, gamma)
                self.resonances.append({
                    'delta_eps': res[0],
                    'omega_0': res[1],
                    'gamma': res[2]
                })

        self.num_poles = len(self.resonances)

    @classmethod
    def from_wavelength(cls, resonances_wavelength, eps_inf=1.0, name="LorentzMaterial"):
        """
        Create LorentzMaterial with resonances specified by wavelength.

        Args:
            resonances_wavelength: List of (delta_eps, wavelength, linewidth) tuples
                                  wavelength and linewidth in meters
            eps_inf: High-frequency permittivity
            name: Material name
        """
        resonances = []
        for res in resonances_wavelength:
            delta_eps, wavelength, linewidth = res[0], res[1], res[2]
            omega_0 = 2 * np.pi * LIGHT_SPEED / wavelength
            gamma = 2 * np.pi * LIGHT_SPEED * linewidth / wavelength**2
            resonances.append((delta_eps, omega_0, gamma))

        return cls(resonances, eps_inf, name)

    def get_permittivity(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate complex permittivity at given wavelength/frequency.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Complex relative permittivity ε_r
        """
        if omega is None:
            if wavelength is not None:
                omega = 2 * np.pi * LIGHT_SPEED / wavelength
            elif frequency is not None:
                omega = 2 * np.pi * frequency
            else:
                raise ValueError("Must specify wavelength, frequency, or omega")

        omega = np.atleast_1d(omega)

        eps = np.full_like(omega, self.eps_inf, dtype=complex)

        for res in self.resonances:
            delta_eps = res['delta_eps']
            omega_0 = res['omega_0']
            gamma = res['gamma']

            # Lorentz term: Δε * ω₀² / (ω₀² - ω² - iγω)
            eps = eps + delta_eps * omega_0**2 / (omega_0**2 - omega**2 - 1j * gamma * omega)

        return eps.squeeze()

    def get_refractive_index(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate complex refractive index.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Complex refractive index n + ik
        """
        eps = self.get_permittivity(wavelength, frequency, omega)
        return np.sqrt(eps)

    def get_absorption_coefficient(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate absorption coefficient α = 4πk/λ.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Absorption coefficient in 1/m
        """
        if wavelength is None:
            if omega is not None:
                wavelength = 2 * np.pi * LIGHT_SPEED / omega
            elif frequency is not None:
                wavelength = LIGHT_SPEED / frequency

        n_complex = self.get_refractive_index(wavelength=wavelength)
        k = np.abs(n_complex.imag)
        return 4 * np.pi * k / wavelength

    def to_material(self, wavelength):
        """
        Convert to a simple Material at specified wavelength.

        Args:
            wavelength: Wavelength in meters

        Returns:
            Material instance
        """
        eps = self.get_permittivity(wavelength=wavelength)
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        conductivity = omega * EPS_0 * np.abs(eps.imag)
        return Material(permittivity=eps.real, permeability=1.0, conductivity=conductivity)

    def __repr__(self):
        return f"LorentzMaterial(name='{self.name}', poles={self.num_poles})"


class DebyeMaterial:
    """
    Dispersive medium described by the Debye relaxation model.

    The Debye model describes dielectric relaxation in polar materials
    and is commonly used for biological tissues and water at RF/microwave:

        ε(ω) = ε_∞ + Σⱼ Δεⱼ / (1 + iωτⱼ)

    where:
        ε_∞ = high-frequency (optical) permittivity
        Δεⱼ = relaxation strength (εₛ - ε_∞ for single pole)
        τⱼ = relaxation time constant (s)

    Args:
        relaxations: List of (delta_eps, tau) tuples or
                    list of dicts with keys 'delta_eps', 'tau'
        eps_inf: High-frequency permittivity (default 1.0)
        conductivity: Optional DC conductivity in S/m (for ionic conduction)
        name: Optional material name

    Example:
        # Water at microwave frequencies (single Debye pole)
        water = DebyeMaterial(
            relaxations=[(77.0, 8.27e-12)],  # (Δε, τ)
            eps_inf=4.9,
            name="Water (microwave)"
        )

        # Biological tissue (multiple poles)
        tissue = DebyeMaterial(
            relaxations=[
                {'delta_eps': 50.0, 'tau': 7.96e-12},
                {'delta_eps': 3000.0, 'tau': 1.59e-7},
            ],
            eps_inf=4.0,
            conductivity=0.7
        )
    """

    def __init__(self, relaxations, eps_inf=1.0, conductivity=0.0, name="DebyeMaterial"):
        self.eps_inf = eps_inf
        self.dc_conductivity = conductivity
        self.name = name

        # Parse relaxations into uniform format
        self.relaxations = []
        for rel in relaxations:
            if isinstance(rel, dict):
                self.relaxations.append({
                    'delta_eps': rel['delta_eps'],
                    'tau': rel['tau']
                })
            else:
                # Assume tuple (delta_eps, tau)
                self.relaxations.append({
                    'delta_eps': rel[0],
                    'tau': rel[1]
                })

        self.num_poles = len(self.relaxations)

        # Calculate static permittivity
        self.eps_static = eps_inf + sum(r['delta_eps'] for r in self.relaxations)

    def get_permittivity(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate complex permittivity at given wavelength/frequency.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Complex relative permittivity ε_r
        """
        if omega is None:
            if wavelength is not None:
                omega = 2 * np.pi * LIGHT_SPEED / wavelength
            elif frequency is not None:
                omega = 2 * np.pi * frequency
            else:
                raise ValueError("Must specify wavelength, frequency, or omega")

        omega = np.atleast_1d(omega)

        eps = np.full_like(omega, self.eps_inf, dtype=complex)

        # Add Debye relaxation poles
        for rel in self.relaxations:
            delta_eps = rel['delta_eps']
            tau = rel['tau']

            # Debye term: Δε / (1 + iωτ)
            eps = eps + delta_eps / (1 + 1j * omega * tau)

        # Add ionic conductivity contribution: -iσ/(ωε₀)
        if self.dc_conductivity > 0:
            eps = eps - 1j * self.dc_conductivity / (omega * EPS_0)

        return eps.squeeze()

    def get_refractive_index(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate complex refractive index.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Complex refractive index n + ik
        """
        eps = self.get_permittivity(wavelength, frequency, omega)
        return np.sqrt(eps)

    def get_loss_tangent(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate loss tangent tan(δ) = ε''/ε'.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Loss tangent (dimensionless)
        """
        eps = self.get_permittivity(wavelength, frequency, omega)
        return -eps.imag / eps.real

    def to_material(self, frequency):
        """
        Convert to a simple Material at specified frequency.

        Args:
            frequency: Frequency in Hz

        Returns:
            Material instance
        """
        eps = self.get_permittivity(frequency=frequency)
        omega = 2 * np.pi * frequency
        conductivity = omega * EPS_0 * np.abs(eps.imag)
        return Material(permittivity=eps.real, permeability=1.0, conductivity=conductivity)

    def __repr__(self):
        return f"DebyeMaterial(name='{self.name}', poles={self.num_poles}, ε_s={self.eps_static:.1f})"


class PoleResidueMaterial:
    """
    General dispersive medium described by pole-residue pairs.

    The pole-residue model is a general representation of dispersive
    materials that can represent Drude, Lorentz, Debye, and other
    models in a unified framework:

        ε(ω) = ε_∞ + Σⱼ (cⱼ/(iω - aⱼ) + cⱼ*/(iω - aⱼ*))

    where aⱼ are complex poles and cⱼ are complex residues.
    Poles come in conjugate pairs for physical (real) permittivity.

    This model is particularly useful for:
    - Fitting experimental optical data
    - Converting between different dispersion models
    - Interfacing with ADE-FDTD implementations

    Args:
        poles: List of complex poles aⱼ (rad/s)
        residues: List of complex residues cⱼ
        eps_inf: High-frequency permittivity (default 1.0)
        name: Optional material name

    Note:
        Poles and residues should be specified as conjugate pairs
        or as just one of each pair (the conjugate will be added automatically).

    Example:
        # Equivalent to single Drude pole
        drude_pole = -1e14  # -γ (pure real = Drude)
        drude_residue = 1e32 / (2j)  # ωₚ²/(2i)
        mat = PoleResidueMaterial(
            poles=[drude_pole],
            residues=[drude_residue],
            name="Drude-like"
        )
    """

    def __init__(self, poles, residues, eps_inf=1.0, name="PoleResidueMaterial"):
        if len(poles) != len(residues):
            raise ValueError("poles and residues must have same length")

        self.poles = np.array(poles, dtype=complex)
        self.residues = np.array(residues, dtype=complex)
        self.eps_inf = eps_inf
        self.name = name
        self.num_poles = len(poles)

    @classmethod
    def from_drude(cls, plasma_freq, collision_freq, eps_inf=1.0, name="DrudePR"):
        """
        Create PoleResidueMaterial equivalent to Drude model.

        Args:
            plasma_freq: Plasma frequency ωₚ (rad/s)
            collision_freq: Collision frequency γ (rad/s)
            eps_inf: High-frequency permittivity
            name: Material name
        """
        # Drude: ε = ε_∞ - ωₚ²/(ω² + iγω) = ε_∞ + ωₚ²/(iω(iω + γ))
        # In pole-residue form with pole at a = -γ:
        # ε = ε_∞ + c/(iω - a) + c*/(iω - a*) where a = -γ (real)
        # For Drude: c = ωₚ²/2
        pole = -collision_freq  # Real pole
        residue = plasma_freq**2 / 2

        return cls([pole], [residue], eps_inf, name)

    @classmethod
    def from_lorentz(cls, delta_eps, omega_0, gamma, eps_inf=1.0, name="LorentzPR"):
        """
        Create PoleResidueMaterial equivalent to single Lorentz oscillator.

        Args:
            delta_eps: Oscillator strength
            omega_0: Resonance frequency (rad/s)
            gamma: Damping rate (rad/s)
            eps_inf: High-frequency permittivity
            name: Material name
        """
        # Lorentz: ε = ε_∞ + Δε·ω₀²/(ω₀² - ω² - iγω)
        # Poles are at: iω = -γ/2 ± sqrt((γ/2)² - ω₀²)
        # For underdamped (γ < 2ω₀): poles at -γ/2 ± i·sqrt(ω₀² - (γ/2)²)

        gamma_half = gamma / 2
        omega_d = np.sqrt(omega_0**2 - gamma_half**2) if omega_0 > gamma_half else 0

        if omega_d > 0:
            # Underdamped: complex conjugate poles
            pole = -gamma_half + 1j * omega_d
            residue = delta_eps * omega_0**2 / (2j * omega_d)
        else:
            # Overdamped: two real poles (simplified)
            pole = -gamma_half + np.sqrt(gamma_half**2 - omega_0**2)
            residue = delta_eps * omega_0**2 / 2

        return cls([pole], [residue], eps_inf, name)

    @classmethod
    def from_debye(cls, delta_eps, tau, eps_inf=1.0, name="DebyePR"):
        """
        Create PoleResidueMaterial equivalent to single Debye pole.

        Args:
            delta_eps: Relaxation strength
            tau: Relaxation time (s)
            eps_inf: High-frequency permittivity
            name: Material name
        """
        # Debye: ε = ε_∞ + Δε/(1 + iωτ)
        # Rewrite: Δε/(1 + iωτ) = Δε·(−1/τ)/(iω − (−1/τ))
        # Pole at a = -1/τ (real), residue c = -Δε/τ

        pole = -1 / tau
        residue = -delta_eps / tau

        return cls([pole], [residue], eps_inf, name)

    def get_permittivity(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate complex permittivity at given wavelength/frequency.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Complex relative permittivity ε_r
        """
        if omega is None:
            if wavelength is not None:
                omega = 2 * np.pi * LIGHT_SPEED / wavelength
            elif frequency is not None:
                omega = 2 * np.pi * frequency
            else:
                raise ValueError("Must specify wavelength, frequency, or omega")

        omega = np.atleast_1d(omega)

        eps = np.full_like(omega, self.eps_inf, dtype=complex)

        for pole, residue in zip(self.poles, self.residues):
            # Add pole contribution and its conjugate
            eps = eps + residue / (1j * omega - pole)
            eps = eps + np.conj(residue) / (1j * omega - np.conj(pole))

        return eps.squeeze()

    def get_refractive_index(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate complex refractive index.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Complex refractive index n + ik
        """
        eps = self.get_permittivity(wavelength, frequency, omega)
        return np.sqrt(eps)

    def to_material(self, wavelength):
        """
        Convert to a simple Material at specified wavelength.

        Args:
            wavelength: Wavelength in meters

        Returns:
            Material instance
        """
        eps = self.get_permittivity(wavelength=wavelength)
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        conductivity = omega * EPS_0 * np.abs(eps.imag)
        return Material(permittivity=eps.real, permeability=1.0, conductivity=conductivity)

    def get_poles_residues(self):
        """
        Get all poles and residues including conjugates.

        Returns:
            Tuple of (poles, residues) including conjugate pairs
        """
        all_poles = []
        all_residues = []

        for pole, residue in zip(self.poles, self.residues):
            all_poles.extend([pole, np.conj(pole)])
            all_residues.extend([residue, np.conj(residue)])

        return np.array(all_poles), np.array(all_residues)

    def __repr__(self):
        return f"PoleResidueMaterial(name='{self.name}', poles={self.num_poles})"


class DrudeLorentzMaterial:
    """
    Combined Drude-Lorentz model for metals with interband transitions.

    This model combines free electron (Drude) response with bound
    electron (Lorentz) oscillators, commonly used for noble metals
    where interband transitions are significant:

        ε(ω) = ε_∞ - ωₚ²/(ω² + iγω) + Σⱼ (Δεⱼ·ω₀ⱼ²)/(ω₀ⱼ² - ω² - iγⱼω)

    Args:
        plasma_freq: Plasma frequency ωₚ (rad/s) for Drude term
        drude_gamma: Collision frequency γ (rad/s) for Drude term
        lorentz_poles: List of Lorentz (delta_eps, omega_0, gamma) tuples
        eps_inf: High-frequency permittivity (default 1.0)
        name: Optional material name

    Example:
        # Gold with interband transitions
        gold = DrudeLorentzMaterial(
            plasma_freq=1.37e16,
            drude_gamma=1.0e14,
            lorentz_poles=[
                (0.27, 3.87e15, 7.0e14),   # First interband
                (0.10, 4.70e15, 2.0e15),   # Second interband
            ],
            eps_inf=1.0,
            name="Gold (DL)"
        )
    """

    def __init__(self, plasma_freq, drude_gamma, lorentz_poles=None,
                 eps_inf=1.0, name="DrudeLorentzMaterial"):
        self.plasma_freq = plasma_freq
        self.drude_gamma = drude_gamma
        self.eps_inf = eps_inf
        self.name = name

        # Parse Lorentz poles
        self.lorentz_poles = []
        if lorentz_poles:
            for pole in lorentz_poles:
                if isinstance(pole, dict):
                    self.lorentz_poles.append(pole)
                else:
                    self.lorentz_poles.append({
                        'delta_eps': pole[0],
                        'omega_0': pole[1],
                        'gamma': pole[2]
                    })

        self.num_lorentz_poles = len(self.lorentz_poles)

    def get_permittivity(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate complex permittivity at given wavelength/frequency.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Complex relative permittivity ε_r
        """
        if omega is None:
            if wavelength is not None:
                omega = 2 * np.pi * LIGHT_SPEED / wavelength
            elif frequency is not None:
                omega = 2 * np.pi * frequency
            else:
                raise ValueError("Must specify wavelength, frequency, or omega")

        omega = np.atleast_1d(omega)

        # Start with high-frequency permittivity
        eps = np.full_like(omega, self.eps_inf, dtype=complex)

        # Add Drude term
        eps = eps - self.plasma_freq**2 / (omega**2 + 1j * self.drude_gamma * omega)

        # Add Lorentz terms
        for pole in self.lorentz_poles:
            delta_eps = pole['delta_eps']
            omega_0 = pole['omega_0']
            gamma = pole['gamma']

            eps = eps + delta_eps * omega_0**2 / (omega_0**2 - omega**2 - 1j * gamma * omega)

        return eps.squeeze()

    def get_refractive_index(self, wavelength=None, frequency=None, omega=None):
        """
        Calculate complex refractive index.

        Args:
            wavelength: Wavelength in meters
            frequency: Frequency in Hz
            omega: Angular frequency in rad/s

        Returns:
            Complex refractive index n + ik
        """
        eps = self.get_permittivity(wavelength, frequency, omega)
        return np.sqrt(eps)

    def to_material(self, wavelength):
        """
        Convert to a simple Material at specified wavelength.

        Args:
            wavelength: Wavelength in meters

        Returns:
            Material instance
        """
        eps = self.get_permittivity(wavelength=wavelength)
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        conductivity = omega * EPS_0 * np.abs(eps.imag)
        return Material(permittivity=eps.real, permeability=1.0, conductivity=conductivity)

    def __repr__(self):
        return f"DrudeLorentzMaterial(name='{self.name}', Lorentz poles={self.num_lorentz_poles})"


# =============================================================================
# PREDEFINED DISPERSIVE MATERIALS
# =============================================================================

# Fused Silica (SiO2) Sellmeier model (Malitson 1965)
# C values are resonance wavelengths squared: λ² in m²
SiO2_Sellmeier = SellmeierMaterial(
    B_coeffs=[0.6961663, 0.4079426, 0.8974794],
    C_coeffs=[(0.0684043e-6)**2, (0.1162414e-6)**2, (9.896161e-6)**2],
    name="SiO2 (Sellmeier)"
)

# BK7 Glass Sellmeier model (Schott catalog)
# C values are resonance wavelengths squared: λ² in m²
BK7_Sellmeier = SellmeierMaterial(
    B_coeffs=[1.03961212, 0.231792344, 1.01046945],
    C_coeffs=[(0.0776142e-6)**2, (0.141569e-6)**2, (10.176e-6)**2],
    name="BK7 (Sellmeier)"
)

# Gold Drude model (simple)
Gold_Drude = DrudeMaterial(
    plasma_freq=1.37e16,
    collision_freq=4.05e13,
    name="Gold (Drude)"
)

# Silver Drude model
Silver_Drude = DrudeMaterial(
    plasma_freq=1.37e16,
    collision_freq=2.73e13,
    name="Silver (Drude)"
)

# Aluminum Drude model
Aluminum_Drude = DrudeMaterial(
    plasma_freq=2.24e16,
    collision_freq=1.22e14,
    name="Aluminum (Drude)"
)

# Copper Drude model
Copper_Drude = DrudeMaterial(
    plasma_freq=1.12e16,
    collision_freq=1.38e14,
    name="Copper (Drude)"
)

# Water (microwave Debye model)
Water_Debye = DebyeMaterial(
    relaxations=[(77.0, 8.27e-12)],
    eps_inf=4.9,
    name="Water (Debye)"
)

# Gold with interband transitions (Drude-Lorentz)
Gold_DrudeLorentz = DrudeLorentzMaterial(
    plasma_freq=1.37e16,
    drude_gamma=1.0e14,
    lorentz_poles=[
        (0.27, 3.87e15, 7.0e14),
        (0.10, 4.70e15, 2.0e15),
    ],
    eps_inf=1.0,
    name="Gold (Drude-Lorentz)"
)
=======
# Debye: A dispersive medium described by the Debye model.
>>>>>>> main
