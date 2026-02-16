"""Material models for EM simulation.

This module contains:
- `Material`: dispersionless bulk material used directly by the current FDTD core.
- `CustomMaterial`: spatially varying material for inverse design.
- Dispersive material classes for frequency/wavelength-dependent properties.

Dispersive models are currently conversion helpers. They must be converted to a
simple `Material` at an explicit operating wavelength or frequency before use in
`Design`/meshing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warnings

from beamz.const import EPS_0, LIGHT_SPEED, MU_0


UM = 1e-6


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
                raise ValueError(
                    f"bounds must be ((x_min, x_max), (y_min, y_max)), got {bounds}"
                )
            if bounds[0][0] >= bounds[0][1]:
                raise ValueError(
                    f"Invalid x bounds: x_min={bounds[0][0]} >= x_max={bounds[0][1]}"
                )
            if bounds[1][0] >= bounds[1][1]:
                raise ValueError(
                    f"Invalid y bounds: y_min={bounds[1][0]} >= y_max={bounds[1][1]}"
                )

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
            return f"grid({np.min(self.permittivity_grid):.3f}-{np.max(self.permittivity_grid):.3f})"
        elif self.permittivity_func is not None:
            return "function"
        else:
            return self.default_permittivity

    @property
    def permeability(self):
        """Return representative permeability for display purposes."""
        if self.permeability_grid is not None:
            return f"grid({np.min(self.permeability_grid):.3f}-{np.max(self.permeability_grid):.3f})"
        elif self.permeability_func is not None:
            return "function"
        else:
            return self.default_permeability

    @property
    def conductivity(self):
        """Return representative conductivity for display purposes."""
        if self.conductivity_grid is not None:
            return f"grid({np.min(self.conductivity_grid):.3f}-{np.max(self.conductivity_grid):.3f})"
        elif self.conductivity_func is not None:
            return "function"
        else:
            return self.default_conductivity

    def _create_grid_interpolator(self, property_name):
        """Create scipy interpolator for grid-based material property."""
        try:
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


@dataclass(frozen=True)
class DispersiveMetadata:
    source: str = ""
    notes: str = ""
    valid_range: str = ""


class _DispersiveBase:
    """Base class for dispersive models.

    These models are not consumed natively by the current FDTD time-stepper.
    Convert explicitly using `to_material(...)` at the desired operating point.
    """

    def __init__(
        self,
        name: str,
        metadata: DispersiveMetadata | None = None,
        k: float = 0.0,
        rho: float = 0.0,
        cp: float = 0.0,
        dn_dT: float = 0.0,
        T0: float = 300.0,
    ) -> None:
        warnings.warn(
            "beamz.design.materials dispersive classes are deprecated. "
            "Use beamz.components.medium classes (Sellmeier/Drude/Lorentz/Debye/PoleResidue) "
            "and beamz.material_library.material_library instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.name = name
        self.metadata = metadata or DispersiveMetadata()
        self.k = k
        self.rho = rho
        self.cp = cp
        self.dn_dT = dn_dT
        self.T0 = T0

    @staticmethod
    def _resolve_angular_frequency(
        frequency: float | np.ndarray | None,
        wavelength: float | np.ndarray | None,
    ) -> np.ndarray:
        if (frequency is None) == (wavelength is None):
            raise ValueError(
                "Provide exactly one of `frequency` (Hz) or `wavelength` (m)."
            )
        if wavelength is not None:
            wl = np.asarray(wavelength, dtype=float)
            if np.any(wl <= 0):
                raise ValueError("wavelength must be > 0.")
            return 2.0 * np.pi * LIGHT_SPEED / wl
        freq = np.asarray(frequency, dtype=float)
        if np.any(freq <= 0):
            raise ValueError("frequency must be > 0.")
        return 2.0 * np.pi * freq

    @staticmethod
    def _resolve_frequency(
        frequency: float | np.ndarray | None,
        wavelength: float | np.ndarray | None,
    ) -> np.ndarray:
        if wavelength is not None:
            wl = np.asarray(wavelength, dtype=float)
            return LIGHT_SPEED / wl
        if frequency is None:
            raise ValueError(
                "Provide exactly one of `frequency` (Hz) or `wavelength` (m)."
            )
        return np.asarray(frequency, dtype=float)

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def n_complex(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        eps_r = self.epsilon(frequency=frequency, wavelength=wavelength)
        return np.lib.scimath.sqrt(eps_r)

    def to_material(
        self,
        *,
        frequency: float | None = None,
        wavelength: float | None = None,
        permeability: float = 1.0,
    ) -> Material:
        """Convert model to dispersionless Material at one operating point."""
        eps_r = self.epsilon(frequency=frequency, wavelength=wavelength)
        eps_arr = np.asarray(eps_r)
        if eps_arr.size != 1:
            raise ValueError(
                "to_material requires scalar frequency/wavelength, not arrays."
            )

        eps_scalar = complex(eps_arr.reshape(()))
        f_scalar = float(
            np.asarray(
                self._resolve_frequency(frequency=frequency, wavelength=wavelength)
            ).reshape(())
        )
        # Convert Im(epsilon_r) into equivalent sigma in A/(V*m).
        sigma = abs(2.0 * np.pi * f_scalar * EPS_0 * float(np.imag(eps_scalar)))
        return Material(
            permittivity=float(np.real(eps_scalar)),
            permeability=permeability,
            conductivity=sigma,
            k=self.k,
            rho=self.rho,
            cp=self.cp,
            dn_dT=self.dn_dT,
            T0=self.T0,
        )

    def get_sample(self):
        raise ValueError(
            f"{self.__class__.__name__} is dispersive. Convert with "
            "`.to_material(frequency=... or wavelength=...)` before meshing/simulation."
        )


class SellmeierMaterial(_DispersiveBase):
    """Sellmeier wavelength model.

    n^2(lambda_um) = 1 + sum_i B_i * lambda_um^2 / (lambda_um^2 - C_i)
    where C_i are in um^2.
    """

    def __init__(
        self,
        name: str,
        B: list[float] | tuple[float, ...],
        C: list[float] | tuple[float, ...],
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        if self.B.shape != self.C.shape or self.B.ndim != 1:
            raise ValueError("B and C must be 1D arrays of equal length.")

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        if wavelength is None:
            # use exact conversion if frequency is passed
            omega = self._resolve_angular_frequency(frequency=frequency, wavelength=None)
            wavelength = 2.0 * np.pi * LIGHT_SPEED / omega
        lam_um = np.asarray(wavelength, dtype=float) / UM
        lam2 = lam_um**2
        n2 = np.ones_like(lam_um, dtype=float)
        for Bi, Ci in zip(self.B, self.C):
            n2 = n2 + Bi * lam2 / (lam2 - Ci)
        return n2.astype(complex)

    def group_index(self, wavelength: float) -> float:
        """Return group index n_g at wavelength (m) via finite differences."""
        wl = float(wavelength)
        if wl <= 0:
            raise ValueError("wavelength must be > 0.")
        dw = max(1e-12, wl * 1e-6)
        n_m = np.real(self.n_complex(wavelength=wl - dw)).reshape(())
        n_0 = np.real(self.n_complex(wavelength=wl)).reshape(())
        n_p = np.real(self.n_complex(wavelength=wl + dw)).reshape(())
        dn_dlambda = (n_p - n_m) / (2.0 * dw)
        return float(n_0 - wl * dn_dlambda)

    def dispersion_ps_nm_km(self, wavelength: float) -> float:
        """Return chromatic dispersion D in ps/(nm*km)."""
        wl = float(wavelength)
        if wl <= 0:
            raise ValueError("wavelength must be > 0.")
        dw = max(1e-12, wl * 1e-6)
        n_m = np.real(self.n_complex(wavelength=wl - dw)).reshape(())
        n_0 = np.real(self.n_complex(wavelength=wl)).reshape(())
        n_p = np.real(self.n_complex(wavelength=wl + dw)).reshape(())
        d2n_dlambda2 = (n_p - 2.0 * n_0 + n_m) / (dw**2)
        d_si = -(wl / LIGHT_SPEED) * d2n_dlambda2
        return float(d_si * 1e6)


class DrudeMaterial(_DispersiveBase):
    """Drude free-electron model.

    epsilon = eps_inf - omega_p^2 / (omega^2 + i*gamma*omega)
    """

    def __init__(
        self,
        name: str,
        eps_inf: float,
        plasma_frequency: float,
        damping: float,
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.eps_inf = float(eps_inf)
        self.plasma_frequency = float(plasma_frequency)
        self.damping = float(damping)

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        omega = self._resolve_angular_frequency(frequency=frequency, wavelength=wavelength)
        return self.eps_inf - (self.plasma_frequency**2) / (
            omega**2 + 1j * self.damping * omega
        )

    def skin_depth(
        self,
        *,
        frequency: float | None = None,
        wavelength: float | None = None,
    ) -> float:
        """Approximate skin depth from the converted conductivity."""
        mat = self.to_material(frequency=frequency, wavelength=wavelength)
        sigma = max(mat.conductivity, 1e-30)
        f = float(
            np.asarray(self._resolve_frequency(frequency=frequency, wavelength=wavelength)).reshape(())
        )
        omega = 2.0 * np.pi * f
        return float(np.sqrt(2.0 / (omega * MU_0 * sigma)))


class LorentzMaterial(_DispersiveBase):
    """Lorentz oscillator model.

    epsilon = eps_inf + sum_j (f_j * omega_pj^2)/(omega_0j^2 - omega^2 - i*gamma_j*omega)
    """

    def __init__(
        self,
        name: str,
        eps_inf: float,
        resonances: list[float] | tuple[float, ...],
        strengths: list[float] | tuple[float, ...],
        dampings: list[float] | tuple[float, ...],
        plasma_frequencies: list[float] | tuple[float, ...] | None = None,
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.eps_inf = float(eps_inf)
        self.resonances = np.asarray(resonances, dtype=float)
        self.strengths = np.asarray(strengths, dtype=float)
        self.dampings = np.asarray(dampings, dtype=float)
        if plasma_frequencies is None:
            self.plasma_frequencies = np.ones_like(self.resonances)
        else:
            self.plasma_frequencies = np.asarray(plasma_frequencies, dtype=float)
        n = self.resonances.size
        if any(arr.size != n for arr in [self.strengths, self.dampings, self.plasma_frequencies]):
            raise ValueError("Lorentz arrays must have equal length.")

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        omega = self._resolve_angular_frequency(frequency=frequency, wavelength=wavelength)
        eps = np.full_like(omega, self.eps_inf, dtype=complex)
        for omega0, f, gamma, wp in zip(
            self.resonances, self.strengths, self.dampings, self.plasma_frequencies
        ):
            eps = eps + (f * wp**2) / (omega0**2 - omega**2 - 1j * gamma * omega)
        return eps


class DebyeMaterial(_DispersiveBase):
    """Debye dielectric relaxation model.

    epsilon = eps_inf + sum_k Delta_eps_k/(1 + i*omega*tau_k) + sigma_dc/(i*omega*eps0)
    """

    def __init__(
        self,
        name: str,
        eps_inf: float,
        debye_strengths: list[float] | tuple[float, ...],
        relaxation_times: list[float] | tuple[float, ...],
        sigma_dc: float = 0.0,
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.eps_inf = float(eps_inf)
        self.debye_strengths = np.asarray(debye_strengths, dtype=float)
        self.relaxation_times = np.asarray(relaxation_times, dtype=float)
        self.sigma_dc = float(sigma_dc)
        if self.debye_strengths.shape != self.relaxation_times.shape:
            raise ValueError("debye_strengths and relaxation_times must match in size.")

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        omega = self._resolve_angular_frequency(frequency=frequency, wavelength=wavelength)
        eps = np.full_like(omega, self.eps_inf, dtype=complex)
        for delta_eps, tau in zip(self.debye_strengths, self.relaxation_times):
            eps = eps + delta_eps / (1.0 + 1j * omega * tau)
        if self.sigma_dc != 0.0:
            eps = eps + self.sigma_dc / (1j * omega * EPS_0)
        return eps


class PoleResidueMaterial(_DispersiveBase):
    """General pole-residue dispersive representation.

    epsilon = eps_inf + sum_k residues_k / (i*omega - poles_k)
    """

    def __init__(
        self,
        name: str,
        eps_inf: float,
        poles: list[complex] | tuple[complex, ...],
        residues: list[complex] | tuple[complex, ...],
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.eps_inf = float(eps_inf)
        self.poles = np.asarray(poles, dtype=complex)
        self.residues = np.asarray(residues, dtype=complex)
        if self.poles.shape != self.residues.shape:
            raise ValueError("poles and residues must match in size.")

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        omega = self._resolve_angular_frequency(frequency=frequency, wavelength=wavelength)
        eps = np.full_like(omega, self.eps_inf, dtype=complex)
        for pole, residue in zip(self.poles, self.residues):
            eps = eps + residue / (1j * omega - pole)
        return eps


class DrudeLorentzMaterial(_DispersiveBase):
    """Combined Drude + Lorentz model.

    epsilon = eps_inf - drude_wp^2/(omega^2 + i*drude_gamma*omega)
              + sum_j (f_j * wp_j^2)/(omega_0j^2 - omega^2 - i*gamma_j*omega)
    """

    def __init__(
        self,
        name: str,
        eps_inf: float,
        drude_plasma_frequency: float,
        drude_damping: float,
        lorentz_resonances: list[float] | tuple[float, ...],
        lorentz_strengths: list[float] | tuple[float, ...],
        lorentz_dampings: list[float] | tuple[float, ...],
        lorentz_plasma_frequencies: list[float] | tuple[float, ...] | None = None,
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.eps_inf = float(eps_inf)
        self.drude_plasma_frequency = float(drude_plasma_frequency)
        self.drude_damping = float(drude_damping)

        self.lorentz_resonances = np.asarray(lorentz_resonances, dtype=float)
        self.lorentz_strengths = np.asarray(lorentz_strengths, dtype=float)
        self.lorentz_dampings = np.asarray(lorentz_dampings, dtype=float)
        if lorentz_plasma_frequencies is None:
            self.lorentz_plasma_frequencies = np.ones_like(self.lorentz_resonances)
        else:
            self.lorentz_plasma_frequencies = np.asarray(
                lorentz_plasma_frequencies, dtype=float
            )

        n = self.lorentz_resonances.size
        if any(
            arr.size != n
            for arr in [
                self.lorentz_strengths,
                self.lorentz_dampings,
                self.lorentz_plasma_frequencies,
            ]
        ):
            raise ValueError("Lorentz arrays must have equal length.")

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        omega = self._resolve_angular_frequency(frequency=frequency, wavelength=wavelength)
        eps = np.full_like(omega, self.eps_inf, dtype=complex)
        eps = eps - (self.drude_plasma_frequency**2) / (
            omega**2 + 1j * self.drude_damping * omega
        )
        for omega0, f, gamma, wp in zip(
            self.lorentz_resonances,
            self.lorentz_strengths,
            self.lorentz_dampings,
            self.lorentz_plasma_frequencies,
        ):
            eps = eps + (f * wp**2) / (omega0**2 - omega**2 - 1j * gamma * omega)
        return eps


# Predefined dispersive instances.
SiO2_Sellmeier = SellmeierMaterial(
    name="SiO2_Sellmeier",
    B=[0.6961663, 0.4079426, 0.8974794],
    C=[0.0684043**2, 0.1162414**2, 9.896161**2],
    metadata=DispersiveMetadata(
        source="Malitson, JOSA 55(10), 1965",
        notes="Fused silica Sellmeier coefficients",
        valid_range="0.21-3.71 um",
    ),
)

BK7_Sellmeier = SellmeierMaterial(
    name="BK7_Sellmeier",
    B=[1.03961212, 0.231792344, 1.01046945],
    C=[0.00600069867, 0.0200179144, 103.560653],
    metadata=DispersiveMetadata(
        source="SCHOTT BK7 Sellmeier",
        notes="N-BK7 optical glass",
        valid_range="0.3-2.5 um",
    ),
)

Gold_Drude = DrudeMaterial(
    name="Gold_Drude",
    eps_inf=9.5,
    plasma_frequency=1.37e16,
    damping=1.05e14,
    metadata=DispersiveMetadata(
        source="Rakic et al., Applied Optics 37(22), 1998",
        notes="Simple Drude fit for gold",
        valid_range="near IR / visible (fit-dependent)",
    ),
)

Silver_Drude = DrudeMaterial(
    name="Silver_Drude",
    eps_inf=3.7,
    plasma_frequency=1.38e16,
    damping=2.73e13,
    metadata=DispersiveMetadata(
        source="Rakic et al., Applied Optics 37(22), 1998",
        notes="Simple Drude fit for silver",
        valid_range="near IR / visible (fit-dependent)",
    ),
)

Aluminum_Drude = DrudeMaterial(
    name="Aluminum_Drude",
    eps_inf=1.0,
    plasma_frequency=2.24e16,
    damping=1.22e14,
    metadata=DispersiveMetadata(
        source="Rakic et al., Applied Optics 37(22), 1998",
        notes="Simple Drude fit for aluminum",
        valid_range="near IR / visible (fit-dependent)",
    ),
)

Copper_Drude = DrudeMaterial(
    name="Copper_Drude",
    eps_inf=10.8,
    plasma_frequency=1.39e16,
    damping=1.03e14,
    metadata=DispersiveMetadata(
        source="Rakic et al., Applied Optics 37(22), 1998",
        notes="Simple Drude fit for copper",
        valid_range="near IR / visible (fit-dependent)",
    ),
)

Water_Debye = DebyeMaterial(
    name="Water_Debye",
    eps_inf=4.9,
    debye_strengths=[73.0],
    relaxation_times=[8.27e-12],
    sigma_dc=0.0,
    metadata=DispersiveMetadata(
        source="Single-pole room-temperature water Debye approximation",
        notes="Microwave/low-THz dielectric relaxation",
        valid_range="GHz to low-THz",
    ),
)

Gold_DrudeLorentz = DrudeLorentzMaterial(
    name="Gold_DrudeLorentz",
    eps_inf=1.53,
    drude_plasma_frequency=1.299e16,
    drude_damping=1.108e14,
    lorentz_resonances=[4.08e15, 7.14e15],
    lorentz_strengths=[0.76, 0.024],
    lorentz_dampings=[8.30e14, 3.886e15],
    lorentz_plasma_frequencies=[1.299e16, 1.299e16],
    metadata=DispersiveMetadata(
        source="Drude-Lorentz style fit inspired by literature parameterizations",
        notes="Includes approximate interband transitions",
        valid_range="fit-dependent",
    ),
)


__all__ = [
    "Material",
    "CustomMaterial",
    "DispersiveMetadata",
    "SellmeierMaterial",
    "DrudeMaterial",
    "LorentzMaterial",
    "DebyeMaterial",
    "PoleResidueMaterial",
    "DrudeLorentzMaterial",
    "SiO2_Sellmeier",
    "BK7_Sellmeier",
    "Gold_Drude",
    "Silver_Drude",
    "Aluminum_Drude",
    "Copper_Drude",
    "Water_Debye",
    "Gold_DrudeLorentz",
]
