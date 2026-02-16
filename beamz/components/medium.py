"""Medium models for Beamz."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0


UM = 1e-6


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _ensure_frequency(frequency: float | np.ndarray) -> np.ndarray:
    f = _as_float_array(frequency)
    if np.any(f <= 0):
        raise ValueError("frequency must be > 0")
    return f


@dataclass
class Medium:
    """Dispersionless medium."""

    permittivity: float = 1.0
    permeability: float = 1.0
    conductivity: float = 0.0
    name: str | None = None
    frequency_range: tuple[float, float] | None = None

    def eps_model(self, frequency: float | np.ndarray) -> np.ndarray:
        f = _ensure_frequency(frequency)
        return self.permittivity + 1j * self.conductivity / (2.0 * np.pi * f * EPS_0)

    def n_model(self, frequency: float | np.ndarray) -> np.ndarray:
        return np.lib.scimath.sqrt(self.eps_model(frequency))

    def get_sample(self):
        return self.permittivity, self.permeability, self.conductivity

    def to_material(self):
        # Imported lazily to avoid circular imports.
        from beamz.design.materials import Material

        return Material(
            permittivity=self.permittivity,
            permeability=self.permeability,
            conductivity=self.conductivity,
        )


@dataclass
class PECMedium(Medium):
    """Perfect electric conductor placeholder."""

    name: str | None = "PEC"

    def eps_model(self, frequency: float | np.ndarray) -> np.ndarray:
        f = _ensure_frequency(frequency)
        return np.full_like(f, 1e20 + 0j, dtype=complex)


@dataclass
class PMCMedium(Medium):
    """Perfect magnetic conductor placeholder."""

    name: str | None = "PMC"

    def eps_model(self, frequency: float | np.ndarray) -> np.ndarray:
        f = _ensure_frequency(frequency)
        return np.full_like(f, 1.0 + 0j, dtype=complex)


PEC = PECMedium()
PMC = PMCMedium()


@dataclass
class PoleResidue:
    """Dispersive medium in pole-residue form.

    epsilon(omega) = eps_inf - sum(c/(j*omega+a) + conj(c)/(j*omega+conj(a)))
    """

    eps_inf: float = 1.0
    poles: tuple[tuple[complex, complex], ...] = field(default_factory=tuple)
    frequency_range: tuple[float, float] | None = None
    name: str | None = None

    def __post_init__(self):
        normalized = []
        for a, c in self.poles:
            normalized.append((complex(a), complex(c)))
        self.poles = tuple(normalized)

    def eps_model(self, frequency: float | np.ndarray) -> np.ndarray:
        f = _ensure_frequency(frequency)
        omega = 2.0 * np.pi * f
        eps = np.full_like(omega, self.eps_inf, dtype=complex)
        for a, c in self.poles:
            eps = eps - c / (1j * omega + a)
            eps = eps - np.conj(c) / (1j * omega + np.conj(a))
        return eps

    def n_model(self, frequency: float | np.ndarray) -> np.ndarray:
        return np.lib.scimath.sqrt(self.eps_model(frequency))

    def sigma_model(self, frequency: float | np.ndarray) -> np.ndarray:
        eps = self.eps_model(frequency)
        f = _ensure_frequency(frequency)
        return 2.0 * np.pi * f * EPS_0 * np.imag(eps)

    def to_medium(self, frequency: float) -> Medium:
        eps = complex(self.eps_model(frequency).reshape(()))
        sigma = abs(2.0 * np.pi * float(frequency) * EPS_0 * float(np.imag(eps)))
        return Medium(
            permittivity=float(np.real(eps)),
            conductivity=sigma,
            name=self.name,
            frequency_range=self.frequency_range,
        )


@dataclass
class Medium2D:
    """Simple 2D anisotropic surface medium (ss/tt components)."""

    ss: PoleResidue | Medium
    tt: PoleResidue | Medium
    name: str | None = None
    frequency_range: tuple[float, float] | None = None

    def eps_model(self, frequency: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.ss.eps_model(frequency), self.tt.eps_model(frequency)


@dataclass
class AnisotropicMedium:
    """Simple anisotropic medium container (xx/yy/zz)."""

    xx: PoleResidue | Medium
    yy: PoleResidue | Medium
    zz: PoleResidue | Medium


@dataclass
class Sellmeier:
    """Sellmeier model with Tidy3D-style coeffs[(B_i, C_i_um2)]."""

    coeffs: tuple[tuple[float, float], ...]
    name: str | None = None
    frequency_range: tuple[float, float] | None = None

    def _n_model(self, frequency: float | np.ndarray) -> np.ndarray:
        f = _ensure_frequency(frequency)
        wvl_m = LIGHT_SPEED / f
        wvl_um2 = (wvl_m / UM) ** 2
        n2 = np.ones_like(wvl_um2, dtype=float)
        for b, c in self.coeffs:
            n2 = n2 + b * wvl_um2 / (wvl_um2 - c)
        return np.lib.scimath.sqrt(n2 + 0j)

    def eps_model(self, frequency: float | np.ndarray) -> np.ndarray:
        n = self._n_model(frequency)
        return n**2

    @property
    def pole_residue(self) -> PoleResidue:
        # Lightweight compatibility placeholder.
        f0 = 2e14
        eps0 = float(np.real(self.eps_model(f0).reshape(())))
        return PoleResidue(
            eps_inf=eps0,
            poles=(),
            frequency_range=self.frequency_range,
            name=self.name,
        )


@dataclass
class Drude:
    """Drude model with Tidy3D-style coeffs[(f_i, delta_i)] (Hz)."""

    coeffs: tuple[tuple[float, float], ...]
    eps_inf: float = 1.0
    name: str | None = None
    frequency_range: tuple[float, float] | None = None

    def eps_model(self, frequency: float | np.ndarray) -> np.ndarray:
        f = _ensure_frequency(frequency)
        eps = np.full_like(f, self.eps_inf, dtype=complex)
        for fp, delta in self.coeffs:
            eps = eps - (fp**2) / (f**2 + 1j * f * delta)
        return eps

    def skin_depth(self, frequency: float) -> float:
        eps = complex(self.eps_model(float(frequency)).reshape(()))
        sigma = abs(2.0 * np.pi * frequency * EPS_0 * float(np.imag(eps)))
        sigma = max(sigma, 1e-30)
        omega = 2.0 * np.pi * frequency
        return float(np.sqrt(2.0 / (omega * MU_0 * sigma)))


@dataclass
class Lorentz:
    """Lorentz model with coeffs[(Delta_eps_i, f_i, delta_i)] (Hz)."""

    coeffs: tuple[tuple[float, float, float], ...]
    eps_inf: float = 1.0
    name: str | None = None
    frequency_range: tuple[float, float] | None = None

    def eps_model(self, frequency: float | np.ndarray) -> np.ndarray:
        f = _ensure_frequency(frequency)
        eps = np.full_like(f, self.eps_inf, dtype=complex)
        for de, f0, delta in self.coeffs:
            eps = eps + (de * f0**2) / (f0**2 - 2j * f * delta - f**2)
        return eps


@dataclass
class Debye:
    """Debye model with coeffs[(Delta_eps_i, tau_i)] where tau in seconds."""

    coeffs: tuple[tuple[float, float], ...]
    eps_inf: float = 1.0
    name: str | None = None
    frequency_range: tuple[float, float] | None = None

    def eps_model(self, frequency: float | np.ndarray) -> np.ndarray:
        f = _ensure_frequency(frequency)
        eps = np.full_like(f, self.eps_inf, dtype=complex)
        for de, tau in self.coeffs:
            eps = eps + de / (1 - 1j * f * tau)
        return eps


__all__ = [
    "Medium",
    "Medium2D",
    "AnisotropicMedium",
    "PECMedium",
    "PMCMedium",
    "PEC",
    "PMC",
    "PoleResidue",
    "Sellmeier",
    "Drude",
    "Lorentz",
    "Debye",
]
