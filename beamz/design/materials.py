"""Material models used by Beamz design and simulation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0


UM = 1e-6


def _as_float_array(value: float | np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _ensure_positive(name: str, value: np.ndarray) -> np.ndarray:
    if np.any(value <= 0):
        raise ValueError(f"{name} must be > 0.")
    return value


class Material:
    """Dispersionless bulk material used directly by the current solver."""

    def __init__(
        self,
        permittivity: float = 1.0,
        permeability: float = 1.0,
        conductivity: float = 0.0,
        k: float = 0.0,
        rho: float = 0.0,
        cp: float = 0.0,
        dn_dT: float = 0.0,
        T0: float = 300.0,
        name: str | None = None,
        frequency_range: tuple[float, float] | None = None,
    ) -> None:
        self.permittivity = float(permittivity)
        self.permeability = float(permeability)
        self.conductivity = float(conductivity)
        self.k = k
        self.rho = rho
        self.cp = cp
        self.dn_dT = dn_dT
        self.T0 = T0
        self.name = name
        self.frequency_range = frequency_range

    def eps_model(self, frequency: float | np.ndarray) -> np.ndarray:
        f = _ensure_positive("frequency", _as_float_array(frequency))
        return self.permittivity + 1j * self.conductivity / (2.0 * np.pi * f * EPS_0)

    def n_model(self, frequency: float | np.ndarray) -> np.ndarray:
        return np.lib.scimath.sqrt(self.eps_model(frequency))

    def to_material(self, **_: Any) -> Material:
        return self

    def get_sample(self) -> tuple[float, float, float]:
        return self.permittivity, self.permeability, self.conductivity

    def show(
        self,
        *,
        wavelength_range_um: tuple[float, float] = (0.4, 2.0),
        num_points: int = 300,
        title: str | None = None,
    ) -> None:
        """Display a compact material summary and constant spectral response.

        Example
        -------
        >>> Material(permittivity=2.1).show()
        """
        from beamz.visual.material_plots import show_material

        show_material(
            self,
            wavelength_range_um=wavelength_range_um,
            num_points=num_points,
            title=title,
        )


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
        self.permittivity_func = permittivity_func
        self.permeability_func = permeability_func
        self.conductivity_func = conductivity_func

        self.permittivity_grid = permittivity_grid
        self.permeability_grid = permeability_grid
        self.conductivity_grid = conductivity_grid

        self.k = k
        self.rho = rho
        self.cp = cp
        self.dn_dT = dn_dT
        self.T0 = T0

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

        self.bounds = bounds
        self.interpolation = interpolation

        self.default_permittivity = 1.0
        self.default_permeability = 1.0
        self.default_conductivity = 0.0

        if permittivity_grid is not None and bounds is not None:
            self._create_grid_interpolator("permittivity")
        if permeability_grid is not None and bounds is not None:
            self._create_grid_interpolator("permeability")
        if conductivity_grid is not None and bounds is not None:
            self._create_grid_interpolator("conductivity")

    @property
    def permittivity(self):
        if self.permittivity_grid is not None:
            return (
                f"grid({np.min(self.permittivity_grid):.3f}-"
                f"{np.max(self.permittivity_grid):.3f})"
            )
        if self.permittivity_func is not None:
            return "function"
        return self.default_permittivity

    @property
    def permeability(self):
        if self.permeability_grid is not None:
            return (
                f"grid({np.min(self.permeability_grid):.3f}-"
                f"{np.max(self.permeability_grid):.3f})"
            )
        if self.permeability_func is not None:
            return "function"
        return self.default_permeability

    @property
    def conductivity(self):
        if self.conductivity_grid is not None:
            return (
                f"grid({np.min(self.conductivity_grid):.3f}-"
                f"{np.max(self.conductivity_grid):.3f})"
            )
        if self.conductivity_func is not None:
            return "function"
        return self.default_conductivity

    def _create_grid_interpolator(self, property_name):
        try:
            from scipy.interpolate import RegularGridInterpolator

            grid = getattr(self, f"{property_name}_grid")
            if grid is None:
                return

            x_coords = np.linspace(self.bounds[0][0], self.bounds[0][1], grid.shape[1])
            y_coords = np.linspace(self.bounds[1][0], self.bounds[1][1], grid.shape[0])

            interpolator = RegularGridInterpolator(
                (y_coords, x_coords),
                grid,
                method=self.interpolation,
                bounds_error=False,
                fill_value=getattr(self, f"default_{property_name}"),
            )

            setattr(self, f"_{property_name}_interpolator", interpolator)
        except ImportError:
            setattr(self, f"_{property_name}_interpolator", None)

    def get_permittivity(self, x, y, z=None):
        if self.permittivity_func is not None:
            return self.permittivity_func(x, y, z) if z is not None else self.permittivity_func(x, y)
        if (
            hasattr(self, "_permittivity_interpolator")
            and self._permittivity_interpolator is not None
        ):
            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._permittivity_interpolator(points)
        return self.default_permittivity

    def get_permeability(self, x, y, z=None):
        if self.permeability_func is not None:
            return self.permeability_func(x, y, z) if z is not None else self.permeability_func(x, y)
        if (
            hasattr(self, "_permeability_interpolator")
            and self._permeability_interpolator is not None
        ):
            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._permeability_interpolator(points)
        return self.default_permeability

    def get_conductivity(self, x, y, z=None):
        if self.conductivity_func is not None:
            return self.conductivity_func(x, y, z) if z is not None else self.conductivity_func(x, y)
        if (
            hasattr(self, "_conductivity_interpolator")
            and self._conductivity_interpolator is not None
        ):
            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._conductivity_interpolator(points)
        return self.default_conductivity

    def get_sample(self, x=0, y=0, z=None):
        return (
            self.get_permittivity(x, y, z),
            self.get_permeability(x, y, z),
            self.get_conductivity(x, y, z),
        )

    def update_grid(self, property_name, new_grid):
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
        perm_grid = self.permittivity_grid.copy() if self.permittivity_grid is not None else None
        permeability_grid = self.permeability_grid.copy() if self.permeability_grid is not None else None
        cond_grid = self.conductivity_grid.copy() if self.conductivity_grid is not None else None

        return CustomMaterial(
            permittivity_func=self.permittivity_func,
            permeability_func=self.permeability_func,
            conductivity_func=self.conductivity_func,
            permittivity_grid=perm_grid,
            permeability_grid=permeability_grid,
            conductivity_grid=cond_grid,
            k=self.k,
            rho=self.rho,
            cp=self.cp,
            dn_dT=self.dn_dT,
            T0=self.T0,
            bounds=self.bounds,
            interpolation=self.interpolation,
        )

    def show(
        self,
        *,
        bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
        grid_shape: tuple[int, int] = (150, 150),
        title: str | None = None,
    ) -> None:
        """Display spatial material fields from grids/functions when available.

        Example
        -------
        >>> mat = CustomMaterial(permittivity_func=lambda x, y: 2.0 + x, bounds=((-1, 1), (-1, 1)))
        >>> mat.show()
        """
        from beamz.visual.material_plots import show_custom_material

        show_custom_material(
            self,
            bounds=bounds,
            grid_shape=grid_shape,
            title=title,
        )


@dataclass(frozen=True)
class DispersiveMetadata:
    source: str = ""
    notes: str = ""
    valid_range: str = ""


class _DispersiveBase:
    """Base class for dispersive material models."""

    def __init__(
        self,
        name: str | None = None,
        metadata: DispersiveMetadata | None = None,
        k: float = 0.0,
        rho: float = 0.0,
        cp: float = 0.0,
        dn_dT: float = 0.0,
        T0: float = 300.0,
        frequency_range: tuple[float, float] | None = None,
    ) -> None:
        self.name = name
        self.metadata = metadata or DispersiveMetadata()
        self.k = k
        self.rho = rho
        self.cp = cp
        self.dn_dT = dn_dT
        self.T0 = T0
        self.frequency_range = frequency_range

    @staticmethod
    def _resolve_angular_frequency(
        frequency: float | np.ndarray | None,
        wavelength: float | np.ndarray | None,
    ) -> np.ndarray:
        if (frequency is None) == (wavelength is None):
            raise ValueError("Provide exactly one of `frequency` (Hz) or `wavelength` (m).")
        if wavelength is not None:
            wl = _ensure_positive("wavelength", _as_float_array(wavelength))
            return 2.0 * np.pi * LIGHT_SPEED / wl
        freq = _ensure_positive("frequency", _as_float_array(frequency))
        return 2.0 * np.pi * freq

    @staticmethod
    def _resolve_frequency(
        frequency: float | np.ndarray | None,
        wavelength: float | np.ndarray | None,
    ) -> np.ndarray:
        if wavelength is not None:
            wl = _ensure_positive("wavelength", _as_float_array(wavelength))
            return LIGHT_SPEED / wl
        if frequency is None:
            raise ValueError("Provide exactly one of `frequency` (Hz) or `wavelength` (m).")
        return _ensure_positive("frequency", _as_float_array(frequency))

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def eps_model(
        self,
        frequency: float | np.ndarray,
    ) -> np.ndarray:
        return self.epsilon(frequency=frequency)

    def n_complex(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        return np.lib.scimath.sqrt(self.epsilon(frequency=frequency, wavelength=wavelength))

    def n_model(self, frequency: float | np.ndarray) -> np.ndarray:
        return self.n_complex(frequency=frequency)

    def show(
        self,
        *,
        wavelength_range_um: tuple[float, float] = (0.4, 2.0),
        num_points: int = 300,
        title: str | None = None,
    ) -> None:
        """Display dispersive material response over wavelength.

        Example
        -------
        >>> SellmeierMaterial(coeffs=((0.696, 0.068**2),)).show()
        """
        from beamz.visual.material_plots import show_dispersive_material

        show_dispersive_material(
            self,
            wavelength_range_um=wavelength_range_um,
            num_points=num_points,
            title=title,
        )

    def to_material(
        self,
        *,
        frequency: float | None = None,
        wavelength: float | None = None,
        permeability: float = 1.0,
    ) -> Material:
        eps = np.asarray(self.epsilon(frequency=frequency, wavelength=wavelength))
        freq = np.asarray(self._resolve_frequency(frequency=frequency, wavelength=wavelength))
        if eps.size != 1 or freq.size != 1:
            raise ValueError("to_material requires scalar frequency/wavelength.")

        eps_scalar = complex(eps.reshape(()))
        f_scalar = float(freq.reshape(()))
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
            name=self.name,
            frequency_range=self.frequency_range,
        )

    def get_sample(self):
        raise ValueError(
            f"{self.__class__.__name__} is dispersive. Convert with "
            "`.to_material(frequency=... or wavelength=...)` first."
        )


class SellmeierMaterial(_DispersiveBase):
    """Sellmeier model where C coefficients are in micron^2."""

    def __init__(
        self,
        coeffs: tuple[tuple[float, float], ...],
        name: str | None = None,
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.coeffs = tuple((float(b), float(c)) for b, c in coeffs)

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        if wavelength is None:
            omega = self._resolve_angular_frequency(frequency=frequency, wavelength=None)
            wavelength = 2.0 * np.pi * LIGHT_SPEED / omega
        lam_um2 = (np.asarray(wavelength, dtype=float) / UM) ** 2
        n2 = np.ones_like(lam_um2, dtype=float)
        for b, c in self.coeffs:
            n2 = n2 + b * lam_um2 / (lam_um2 - c)
        return n2.astype(complex)


class DrudeMaterial(_DispersiveBase):
    """Drude model with coeffs as (plasma_frequency, damping) in Hz."""

    def __init__(
        self,
        coeffs: tuple[tuple[float, float], ...],
        eps_inf: float = 1.0,
        name: str | None = None,
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.coeffs = tuple((float(fp), float(delta)) for fp, delta in coeffs)
        self.eps_inf = float(eps_inf)

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        f = self._resolve_frequency(frequency=frequency, wavelength=wavelength)
        eps = np.full_like(f, self.eps_inf, dtype=complex)
        for fp, delta in self.coeffs:
            eps = eps - (fp**2) / (f**2 + 1j * f * delta)
        return eps

    def skin_depth(
        self,
        *,
        frequency: float | None = None,
        wavelength: float | None = None,
    ) -> float:
        mat = self.to_material(frequency=frequency, wavelength=wavelength)
        sigma = max(mat.conductivity, 1e-30)
        f = float(np.asarray(self._resolve_frequency(frequency=frequency, wavelength=wavelength)).reshape(()))
        omega = 2.0 * np.pi * f
        return float(np.sqrt(2.0 / (omega * MU_0 * sigma)))


class LorentzMaterial(_DispersiveBase):
    """Lorentz model with coeffs (delta_eps, resonance_hz, damping_hz)."""

    def __init__(
        self,
        coeffs: tuple[tuple[float, float, float], ...],
        eps_inf: float = 1.0,
        name: str | None = None,
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.coeffs = tuple((float(de), float(f0), float(delta)) for de, f0, delta in coeffs)
        self.eps_inf = float(eps_inf)

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        f = self._resolve_frequency(frequency=frequency, wavelength=wavelength)
        eps = np.full_like(f, self.eps_inf, dtype=complex)
        for de, f0, delta in self.coeffs:
            eps = eps + (de * f0**2) / (f0**2 - 2j * f * delta - f**2)
        return eps


class DebyeMaterial(_DispersiveBase):
    """Debye model with coeffs (delta_eps, tau_seconds)."""

    def __init__(
        self,
        coeffs: tuple[tuple[float, float], ...],
        eps_inf: float = 1.0,
        name: str | None = None,
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.coeffs = tuple((float(de), float(tau)) for de, tau in coeffs)
        self.eps_inf = float(eps_inf)

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        f = self._resolve_frequency(frequency=frequency, wavelength=wavelength)
        eps = np.full_like(f, self.eps_inf, dtype=complex)
        for de, tau in self.coeffs:
            eps = eps + de / (1 - 1j * f * tau)
        return eps


class PoleResidueMaterial(_DispersiveBase):
    """Pole-residue model.

    epsilon = eps_inf - sum(c/(j*omega + a) + conj(c)/(j*omega + conj(a)))
    """

    def __init__(
        self,
        eps_inf: float = 1.0,
        poles: tuple[tuple[complex, complex], ...] = (),
        name: str | None = None,
        metadata: DispersiveMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, metadata=metadata, **kwargs)
        self.eps_inf = float(eps_inf)
        self.poles = tuple((complex(a), complex(c)) for a, c in poles)

    def epsilon(
        self,
        *,
        frequency: float | np.ndarray | None = None,
        wavelength: float | np.ndarray | None = None,
    ) -> np.ndarray:
        omega = self._resolve_angular_frequency(frequency=frequency, wavelength=wavelength)
        eps = np.full_like(omega, self.eps_inf, dtype=complex)
        for a, c in self.poles:
            eps = eps - c / (1j * omega + a)
            eps = eps - np.conj(c) / (1j * omega + np.conj(a))
        return eps


@dataclass
class Material2D:
    """Simple 2D surface material container with in-plane components."""

    ss: Material | _DispersiveBase
    tt: Material | _DispersiveBase
    name: str | None = None
    frequency_range: tuple[float, float] | None = None

    def eps_model(self, frequency: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.ss.eps_model(frequency), self.tt.eps_model(frequency)

    def show(
        self,
        *,
        wavelength_range_um: tuple[float, float] = (0.4, 2.0),
        num_points: int = 300,
        title: str | None = None,
    ) -> None:
        """Display 2D material component dispersion (ss/tt).

        Example
        -------
        >>> Material2D(ss=Material(permittivity=2.0), tt=Material(permittivity=2.2)).show()
        """
        from beamz.visual.material_plots import show_material2d

        show_material2d(
            self,
            wavelength_range_um=wavelength_range_um,
            num_points=num_points,
            title=title,
        )


@dataclass
class AnisotropicMaterial:
    """Simple anisotropic material container with xx/yy/zz components."""

    xx: Material | _DispersiveBase
    yy: Material | _DispersiveBase
    zz: Material | _DispersiveBase

    def show(
        self,
        *,
        wavelength_range_um: tuple[float, float] = (0.4, 2.0),
        num_points: int = 300,
        title: str | None = None,
    ) -> None:
        """Display anisotropic material dispersion (xx/yy/zz).

        Example
        -------
        >>> AnisotropicMaterial(xx=Material(2.1), yy=Material(2.2), zz=Material(2.3)).show()
        """
        from beamz.visual.material_plots import show_anisotropic_material

        show_anisotropic_material(
            self,
            wavelength_range_um=wavelength_range_um,
            num_points=num_points,
            title=title,
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
    "Material2D",
    "AnisotropicMaterial",
]
