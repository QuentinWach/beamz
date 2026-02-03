"""
Material Library for BeamZ electromagnetic simulations.

This module provides a comprehensive library of predefined materials with
accurate optical and electromagnetic properties for use in FDTD simulations.

Materials are organized by category:
- Vacuum & Gases
- Dielectrics & Glasses
- Semiconductors
- Insulators & Oxides
- Metals (with Drude model support)
- Polymers
- Liquids
- Specialty Materials

Usage:
    from beamz.design.library import Silicon, SiO2, Gold

    # Use predefined material
    design.add_structure(Rectangle(..., material=Silicon))

    # Get material at specific wavelength
    si = Silicon(wavelength=1.55e-6)  # Silicon at 1550nm

    # Access Sellmeier model for wavelength-dependent properties
    from beamz.design.library import SiliconSellmeier
    n = SiliconSellmeier.get_refractive_index(wavelength=1.31e-6)

References:
    - RefractiveIndex.INFO database
    - Palik, E.D., "Handbook of Optical Constants of Solids"
    - CRC Handbook of Chemistry and Physics
"""

<<<<<<< HEAD
import numpy as np
from beamz.design.materials import Material
from beamz.const import LIGHT_SPEED


# =============================================================================
# VACUUM & GASES
# =============================================================================

class Vacuum(Material):
    """
    Vacuum / Free space.
    n = 1.0, k = 0
    """
    def __init__(self):
        super().__init__(permittivity=1.0, permeability=1.0, conductivity=0.0)
        self.name = "Vacuum"
        self.refractive_index = 1.0


class Air(Material):
    """
    Air at standard temperature and pressure (STP).
    n ≈ 1.000293 at 589nm
    For most simulations, treated as n = 1.0
    """
    def __init__(self, accurate=False):
        n = 1.000293 if accurate else 1.0
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = "Air"
        self.refractive_index = n


# =============================================================================
# DIELECTRICS & GLASSES
# =============================================================================

class SiO2(Material):
    """
    Silicon Dioxide (Fused Silica / Quartz glass).
    n ≈ 1.444 at 1550nm (telecom wavelength)
    Transparent from ~0.2 to ~3.7 μm

    Args:
        wavelength: Optional wavelength in meters for Sellmeier calculation
    """
    # Sellmeier coefficients for fused silica (Malitson 1965)
    # C values are resonance wavelengths squared: λ² in m²
    SELLMEIER_B = [0.6961663, 0.4079426, 0.8974794]
    SELLMEIER_C = [(0.0684043e-6)**2, (0.1162414e-6)**2, (9.896161e-6)**2]  # in m^2

    def __init__(self, wavelength=1.55e-6):
        n = self._sellmeier(wavelength)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = "SiO2"
        self.refractive_index = n
        self.wavelength = wavelength

    @classmethod
    def _sellmeier(cls, wavelength):
        """Calculate refractive index using Sellmeier equation."""
        wl_sq = wavelength**2
        n_sq = 1.0
        for B, C in zip(cls.SELLMEIER_B, cls.SELLMEIER_C):
            n_sq += B * wl_sq / (wl_sq - C)
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get refractive index at specified wavelength."""
        return cls._sellmeier(wavelength)


# Alias for common naming conventions
FusedSilica = SiO2
Silica = SiO2


class Si3N4(Material):
    """
    Silicon Nitride (LPCVD stoichiometric Si3N4).
    n ≈ 2.0 at 1550nm
    Transparent from ~0.25 to ~8 μm
    Common waveguide core material for photonic integrated circuits.

    Args:
        wavelength: Optional wavelength in meters for Sellmeier calculation
    """
    # Sellmeier coefficients for Si3N4 (Luke et al. 2015)
    # C value is resonance wavelength squared: λ² in m²
    SELLMEIER_B = [3.0249]
    SELLMEIER_C = [(0.1353406e-6)**2]  # already squared, in m^2

    def __init__(self, wavelength=1.55e-6):
        n = self._sellmeier(wavelength)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = "Si3N4"
        self.refractive_index = n
        self.wavelength = wavelength

    @classmethod
    def _sellmeier(cls, wavelength):
        """Calculate refractive index using simplified Sellmeier equation."""
        wl_sq = wavelength**2
        n_sq = 1.0
        for B, C in zip(cls.SELLMEIER_B, cls.SELLMEIER_C):
            n_sq += B * wl_sq / (wl_sq - C)
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get refractive index at specified wavelength."""
        return cls._sellmeier(wavelength)


# Aliases
SiliconNitride = Si3N4
SiN = Si3N4


class BK7(Material):
    """
    Schott BK7 borosilicate crown glass.
    n ≈ 1.5168 at 589nm (d-line)
    Common optical glass for lenses and prisms.

    Args:
        wavelength: Optional wavelength in meters for Sellmeier calculation
    """
    # Sellmeier coefficients for BK7 (Schott catalog)
    # C values are resonance wavelengths squared: λ² in m²
    SELLMEIER_B = [1.03961212, 0.231792344, 1.01046945]
    SELLMEIER_C = [(0.0776142e-6)**2, (0.141569e-6)**2, (10.176e-6)**2]  # in m^2

    def __init__(self, wavelength=589e-9):
        n = self._sellmeier(wavelength)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = "BK7"
        self.refractive_index = n
        self.wavelength = wavelength

    @classmethod
    def _sellmeier(cls, wavelength):
        """Calculate refractive index using Sellmeier equation."""
        wl_sq = wavelength**2
        n_sq = 1.0
        for B, C in zip(cls.SELLMEIER_B, cls.SELLMEIER_C):
            n_sq += B * wl_sq / (wl_sq - C)
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get refractive index at specified wavelength."""
        return cls._sellmeier(wavelength)


class SodaLimeGlass(Material):
    """
    Soda-lime glass (common window glass).
    n ≈ 1.52 at 589nm
    """
    def __init__(self):
        super().__init__(permittivity=1.52**2, permeability=1.0, conductivity=0.0)
        self.name = "Soda-Lime Glass"
        self.refractive_index = 1.52


class Sapphire(Material):
    """
    Sapphire (Al2O3 - Aluminum Oxide, crystalline).
    n ≈ 1.77 at 589nm (ordinary ray)
    Birefringent crystal - this uses ordinary ray value.
    Transparent from ~0.15 to ~5.5 μm

    Args:
        wavelength: Optional wavelength in meters for Sellmeier calculation
        ray: 'ordinary' or 'extraordinary' for birefringent calculation
    """
    # Sellmeier coefficients for sapphire ordinary ray (Malitson 1962)
    # C values are resonance wavelengths squared: λ² in m²
    SELLMEIER_B_O = [1.4313493, 0.65054713, 5.3414021]
    SELLMEIER_C_O = [(0.0726631e-6)**2, (0.1193242e-6)**2, (18.028e-6)**2]  # in m^2

    # Extraordinary ray coefficients
    SELLMEIER_B_E = [1.5039759, 0.55069141, 6.5927379]
    SELLMEIER_C_E = [(0.0740288e-6)**2, (0.1216529e-6)**2, (20.072e-6)**2]

    def __init__(self, wavelength=589e-9, ray='ordinary'):
        n = self._sellmeier(wavelength, ray)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = f"Sapphire ({ray})"
        self.refractive_index = n
        self.wavelength = wavelength
        self.ray = ray

    @classmethod
    def _sellmeier(cls, wavelength, ray='ordinary'):
        """Calculate refractive index using Sellmeier equation."""
        wl_sq = wavelength**2
        if ray == 'ordinary':
            B, C = cls.SELLMEIER_B_O, cls.SELLMEIER_C_O
        else:
            B, C = cls.SELLMEIER_B_E, cls.SELLMEIER_C_E
        n_sq = 1.0
        for b, c in zip(B, C):
            n_sq += b * wl_sq / (wl_sq - c)
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength, ray='ordinary'):
        """Get refractive index at specified wavelength."""
        return cls._sellmeier(wavelength, ray)


# Alias
Al2O3 = Sapphire


class Diamond(Material):
    """
    Diamond (crystalline carbon).
    n ≈ 2.42 at 589nm
    Transparent from ~0.22 μm to far infrared.
    Highest thermal conductivity of any natural material.

    Args:
        wavelength: Optional wavelength in meters
    """
    # Sellmeier coefficients for diamond (Peter 1923)
    # C values are resonance wavelengths squared: λ² in m²
    SELLMEIER_B = [0.3306, 4.3356]
    SELLMEIER_C = [0.0, (0.1060e-6)**2]  # already squared, in m^2

    def __init__(self, wavelength=589e-9):
        n = self._sellmeier(wavelength)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = "Diamond"
        self.refractive_index = n
        self.wavelength = wavelength

    @classmethod
    def _sellmeier(cls, wavelength):
        """Calculate refractive index using Sellmeier equation."""
        wl_sq = wavelength**2
        n_sq = 1.0
        for B, C in zip(cls.SELLMEIER_B, cls.SELLMEIER_C):
            if C == 0:
                n_sq += B
            else:
                n_sq += B * wl_sq / (wl_sq - C)
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get refractive index at specified wavelength."""
        return cls._sellmeier(wavelength)


# =============================================================================
# SEMICONDUCTORS
# =============================================================================

class Silicon(Material):
    """
    Crystalline Silicon.
    n ≈ 3.48 at 1550nm (telecom wavelength)
    Transparent for λ > 1.1 μm (bandgap ~1.12 eV)
    Primary material for silicon photonics.

    Args:
        wavelength: Wavelength in meters (default 1550nm)
    """
    # Sellmeier coefficients for silicon (Salzberg & Villa 1957)
    # Valid for λ > 1.1 μm
    SELLMEIER_A = 11.6858
    SELLMEIER_B = 0.939816
    SELLMEIER_C = (1.1071e-6)**2  # in m^2

    def __init__(self, wavelength=1.55e-6):
        n = self._sellmeier(wavelength)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = "Silicon"
        self.refractive_index = n
        self.wavelength = wavelength

    @classmethod
    def _sellmeier(cls, wavelength):
        """Calculate refractive index using Sellmeier-like equation."""
        wl_sq = wavelength**2
        n_sq = cls.SELLMEIER_A + cls.SELLMEIER_B / (wl_sq / cls.SELLMEIER_C - 1)
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get refractive index at specified wavelength."""
        return cls._sellmeier(wavelength)


# Aliases
Si = Silicon
cSi = Silicon


class Germanium(Material):
    """
    Crystalline Germanium.
    n ≈ 4.0 at 10 μm
    Transparent for λ > 1.8 μm (bandgap ~0.67 eV)
    Common infrared optical material.

    Args:
        wavelength: Wavelength in meters (default 10 μm)
    """
    # Sellmeier coefficients for germanium (Icenogle et al. 1976)
    # Valid for 2.5-14 μm at room temperature
    SELLMEIER_A = 9.28156
    SELLMEIER_B = 6.72880
    SELLMEIER_C = 0.44105
    SELLMEIER_D = (0.6649e-6)**2
    SELLMEIER_E = (62.21e-6)**2

    def __init__(self, wavelength=10e-6):
        n = self._calc_n(wavelength)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = "Germanium"
        self.refractive_index = n
        self.wavelength = wavelength

    @classmethod
    def _calc_n(cls, wavelength):
        """Calculate refractive index."""
        wl_sq = wavelength**2
        n_sq = cls.SELLMEIER_A + cls.SELLMEIER_B * wl_sq / (wl_sq - cls.SELLMEIER_D)
        n_sq += cls.SELLMEIER_C * wl_sq / (wl_sq - cls.SELLMEIER_E)
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get refractive index at specified wavelength."""
        return cls._calc_n(wavelength)


# Alias
Ge = Germanium


class GaAs(Material):
    """
    Gallium Arsenide.
    n ≈ 3.4 at 1550nm
    Direct bandgap ~1.42 eV (870nm)
    Common material for lasers and detectors.

    Args:
        wavelength: Wavelength in meters (default 1550nm)
    """
    # Sellmeier coefficients (Skauli et al. 2003)
    SELLMEIER_A = 5.372514
    SELLMEIER_B = 5.466742
    SELLMEIER_C = (0.4431307e-6)**2
    SELLMEIER_D = 0.02429960
    SELLMEIER_E = (0.8746453e-6)**2

    def __init__(self, wavelength=1.55e-6):
        n = self._calc_n(wavelength)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = "GaAs"
        self.refractive_index = n
        self.wavelength = wavelength

    @classmethod
    def _calc_n(cls, wavelength):
        """Calculate refractive index."""
        wl_sq = wavelength**2
        n_sq = cls.SELLMEIER_A
        n_sq += cls.SELLMEIER_B * wl_sq / (wl_sq - cls.SELLMEIER_C)
        n_sq += cls.SELLMEIER_D * wl_sq / (wl_sq - cls.SELLMEIER_E)
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get refractive index at specified wavelength."""
        return cls._calc_n(wavelength)


class InP(Material):
    """
    Indium Phosphide.
    n ≈ 3.17 at 1550nm
    Direct bandgap ~1.34 eV (925nm)
    Platform for telecom photonics and lasers.

    Args:
        wavelength: Wavelength in meters (default 1550nm)
    """
    # Sellmeier coefficients (Pettit & Turner 1965)
    SELLMEIER_A = 7.255
    SELLMEIER_B = 2.316
    SELLMEIER_C = (0.3922e-6)**2

    def __init__(self, wavelength=1.55e-6):
        n = self._calc_n(wavelength)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = "InP"
        self.refractive_index = n
        self.wavelength = wavelength

    @classmethod
    def _calc_n(cls, wavelength):
        """Calculate refractive index."""
        wl_sq = wavelength**2
        n_sq = cls.SELLMEIER_A + cls.SELLMEIER_B * wl_sq / (wl_sq - cls.SELLMEIER_C)
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get refractive index at specified wavelength."""
        return cls._calc_n(wavelength)


class LiNbO3(Material):
    """
    Lithium Niobate (extraordinary ray).
    n ≈ 2.14 at 1550nm
    Strong electro-optic and nonlinear optical properties.
    Used in modulators and frequency converters.

    Args:
        wavelength: Wavelength in meters (default 1550nm)
        ray: 'ordinary' or 'extraordinary'
    """
    # Sellmeier coefficients (Jundt 1997) - extraordinary ray
    SELLMEIER_A_E = 5.35583
    SELLMEIER_B_E = 0.100473
    SELLMEIER_C_E = (0.20692e-6)**2
    SELLMEIER_D_E = 100.0
    SELLMEIER_E_E = (11.34927e-6)**2

    # Ordinary ray
    SELLMEIER_A_O = 5.756
    SELLMEIER_B_O = 0.0983
    SELLMEIER_C_O = (0.2020e-6)**2
    SELLMEIER_D_O = 189.32
    SELLMEIER_E_O = (12.52e-6)**2

    def __init__(self, wavelength=1.55e-6, ray='extraordinary'):
        n = self._calc_n(wavelength, ray)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = f"LiNbO3 ({ray})"
        self.refractive_index = n
        self.wavelength = wavelength
        self.ray = ray

    @classmethod
    def _calc_n(cls, wavelength, ray='extraordinary'):
        """Calculate refractive index."""
        wl_sq = wavelength**2
        if ray == 'extraordinary':
            n_sq = cls.SELLMEIER_A_E
            n_sq += cls.SELLMEIER_B_E / (wl_sq / cls.SELLMEIER_C_E - 1)
            n_sq -= cls.SELLMEIER_D_E * wl_sq / cls.SELLMEIER_E_E
        else:
            n_sq = cls.SELLMEIER_A_O
            n_sq += cls.SELLMEIER_B_O / (wl_sq / cls.SELLMEIER_C_O - 1)
            n_sq -= cls.SELLMEIER_D_O * wl_sq / cls.SELLMEIER_E_O
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength, ray='extraordinary'):
        """Get refractive index at specified wavelength."""
        return cls._calc_n(wavelength, ray)


# Alias
LithiumNiobate = LiNbO3


# =============================================================================
# METALS (using effective permittivity from Drude model parameters)
# =============================================================================

class Gold(Material):
    """
    Gold (Au).
    Highly conductive metal with plasmonic properties.

    At optical frequencies, gold is dispersive. This class provides
    an effective permittivity at a specific wavelength using the Drude model.

    Drude parameters:
        plasma frequency ωp ≈ 1.37e16 rad/s
        collision frequency γ ≈ 4.05e13 rad/s

    Args:
        wavelength: Wavelength in meters (default 1550nm)
    """
    # Drude model parameters for gold
    PLASMA_FREQ = 1.37e16  # rad/s
    COLLISION_FREQ = 4.05e13  # rad/s
    DC_CONDUCTIVITY = 4.1e7  # S/m

    def __init__(self, wavelength=1.55e-6):
        eps_real, eps_imag = self._drude_permittivity(wavelength)
        # For FDTD, we use the real part and represent loss via conductivity
        # conductivity = omega * eps_0 * eps_imag
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps_0 = 8.85e-12
        conductivity = omega * eps_0 * abs(eps_imag)
        super().__init__(permittivity=eps_real, permeability=1.0, conductivity=conductivity)
        self.name = "Gold"
        self.wavelength = wavelength
        self.eps_complex = complex(eps_real, eps_imag)

    @classmethod
    def _drude_permittivity(cls, wavelength):
        """Calculate complex permittivity using Drude model."""
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps = 1 - cls.PLASMA_FREQ**2 / (omega**2 + 1j * cls.COLLISION_FREQ * omega)
        return eps.real, eps.imag

    @classmethod
    def get_permittivity(cls, wavelength):
        """Get complex permittivity at specified wavelength."""
        eps_real, eps_imag = cls._drude_permittivity(wavelength)
        return complex(eps_real, eps_imag)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get complex refractive index at specified wavelength."""
        eps = cls.get_permittivity(wavelength)
        n_complex = np.sqrt(eps)
        return n_complex


# Alias
Au = Gold


class Silver(Material):
    """
    Silver (Ag).
    Lowest loss metal at optical frequencies.
    Excellent plasmonic properties.

    Drude parameters:
        plasma frequency ωp ≈ 1.37e16 rad/s
        collision frequency γ ≈ 2.73e13 rad/s

    Args:
        wavelength: Wavelength in meters (default 1550nm)
    """
    PLASMA_FREQ = 1.37e16  # rad/s
    COLLISION_FREQ = 2.73e13  # rad/s
    DC_CONDUCTIVITY = 6.3e7  # S/m

    def __init__(self, wavelength=1.55e-6):
        eps_real, eps_imag = self._drude_permittivity(wavelength)
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps_0 = 8.85e-12
        conductivity = omega * eps_0 * abs(eps_imag)
        super().__init__(permittivity=eps_real, permeability=1.0, conductivity=conductivity)
        self.name = "Silver"
        self.wavelength = wavelength
        self.eps_complex = complex(eps_real, eps_imag)

    @classmethod
    def _drude_permittivity(cls, wavelength):
        """Calculate complex permittivity using Drude model."""
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps = 1 - cls.PLASMA_FREQ**2 / (omega**2 + 1j * cls.COLLISION_FREQ * omega)
        return eps.real, eps.imag

    @classmethod
    def get_permittivity(cls, wavelength):
        """Get complex permittivity at specified wavelength."""
        eps_real, eps_imag = cls._drude_permittivity(wavelength)
        return complex(eps_real, eps_imag)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get complex refractive index at specified wavelength."""
        return np.sqrt(cls.get_permittivity(wavelength))


# Alias
Ag = Silver


class Copper(Material):
    """
    Copper (Cu).
    Common conductive metal.

    Drude parameters:
        plasma frequency ωp ≈ 1.12e16 rad/s
        collision frequency γ ≈ 1.38e14 rad/s

    Args:
        wavelength: Wavelength in meters (default 1550nm)
    """
    PLASMA_FREQ = 1.12e16  # rad/s
    COLLISION_FREQ = 1.38e14  # rad/s
    DC_CONDUCTIVITY = 5.96e7  # S/m

    def __init__(self, wavelength=1.55e-6):
        eps_real, eps_imag = self._drude_permittivity(wavelength)
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps_0 = 8.85e-12
        conductivity = omega * eps_0 * abs(eps_imag)
        super().__init__(permittivity=eps_real, permeability=1.0, conductivity=conductivity)
        self.name = "Copper"
        self.wavelength = wavelength
        self.eps_complex = complex(eps_real, eps_imag)

    @classmethod
    def _drude_permittivity(cls, wavelength):
        """Calculate complex permittivity using Drude model."""
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps = 1 - cls.PLASMA_FREQ**2 / (omega**2 + 1j * cls.COLLISION_FREQ * omega)
        return eps.real, eps.imag

    @classmethod
    def get_permittivity(cls, wavelength):
        """Get complex permittivity at specified wavelength."""
        eps_real, eps_imag = cls._drude_permittivity(wavelength)
        return complex(eps_real, eps_imag)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get complex refractive index at specified wavelength."""
        return np.sqrt(cls.get_permittivity(wavelength))


# Alias
Cu = Copper


class Aluminum(Material):
    """
    Aluminum (Al).
    Good reflector across broad spectrum including UV.

    Drude parameters:
        plasma frequency ωp ≈ 2.24e16 rad/s
        collision frequency γ ≈ 1.22e14 rad/s

    Args:
        wavelength: Wavelength in meters (default 1550nm)
    """
    PLASMA_FREQ = 2.24e16  # rad/s
    COLLISION_FREQ = 1.22e14  # rad/s
    DC_CONDUCTIVITY = 3.77e7  # S/m

    def __init__(self, wavelength=1.55e-6):
        eps_real, eps_imag = self._drude_permittivity(wavelength)
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps_0 = 8.85e-12
        conductivity = omega * eps_0 * abs(eps_imag)
        super().__init__(permittivity=eps_real, permeability=1.0, conductivity=conductivity)
        self.name = "Aluminum"
        self.wavelength = wavelength
        self.eps_complex = complex(eps_real, eps_imag)

    @classmethod
    def _drude_permittivity(cls, wavelength):
        """Calculate complex permittivity using Drude model."""
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps = 1 - cls.PLASMA_FREQ**2 / (omega**2 + 1j * cls.COLLISION_FREQ * omega)
        return eps.real, eps.imag

    @classmethod
    def get_permittivity(cls, wavelength):
        """Get complex permittivity at specified wavelength."""
        eps_real, eps_imag = cls._drude_permittivity(wavelength)
        return complex(eps_real, eps_imag)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get complex refractive index at specified wavelength."""
        return np.sqrt(cls.get_permittivity(wavelength))


# Alias
Al = Aluminum


class Chromium(Material):
    """
    Chromium (Cr).
    Adhesion layer material in nanofabrication.

    Args:
        wavelength: Wavelength in meters (default 1550nm)
    """
    PLASMA_FREQ = 1.49e16  # rad/s
    COLLISION_FREQ = 4.6e14  # rad/s
    DC_CONDUCTIVITY = 7.9e6  # S/m

    def __init__(self, wavelength=1.55e-6):
        eps_real, eps_imag = self._drude_permittivity(wavelength)
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps_0 = 8.85e-12
        conductivity = omega * eps_0 * abs(eps_imag)
        super().__init__(permittivity=eps_real, permeability=1.0, conductivity=conductivity)
        self.name = "Chromium"
        self.wavelength = wavelength
        self.eps_complex = complex(eps_real, eps_imag)

    @classmethod
    def _drude_permittivity(cls, wavelength):
        """Calculate complex permittivity using Drude model."""
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps = 1 - cls.PLASMA_FREQ**2 / (omega**2 + 1j * cls.COLLISION_FREQ * omega)
        return eps.real, eps.imag

    @classmethod
    def get_permittivity(cls, wavelength):
        """Get complex permittivity at specified wavelength."""
        eps_real, eps_imag = cls._drude_permittivity(wavelength)
        return complex(eps_real, eps_imag)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get complex refractive index at specified wavelength."""
        return np.sqrt(cls.get_permittivity(wavelength))


# Alias
Cr = Chromium


class Titanium(Material):
    """
    Titanium (Ti).
    Adhesion layer and structural material.

    Args:
        wavelength: Wavelength in meters (default 1550nm)
    """
    PLASMA_FREQ = 1.0e16  # rad/s
    COLLISION_FREQ = 2.0e14  # rad/s
    DC_CONDUCTIVITY = 2.38e6  # S/m

    def __init__(self, wavelength=1.55e-6):
        eps_real, eps_imag = self._drude_permittivity(wavelength)
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps_0 = 8.85e-12
        conductivity = omega * eps_0 * abs(eps_imag)
        super().__init__(permittivity=eps_real, permeability=1.0, conductivity=conductivity)
        self.name = "Titanium"
        self.wavelength = wavelength
        self.eps_complex = complex(eps_real, eps_imag)

    @classmethod
    def _drude_permittivity(cls, wavelength):
        """Calculate complex permittivity using Drude model."""
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        eps = 1 - cls.PLASMA_FREQ**2 / (omega**2 + 1j * cls.COLLISION_FREQ * omega)
        return eps.real, eps.imag

    @classmethod
    def get_permittivity(cls, wavelength):
        """Get complex permittivity at specified wavelength."""
        eps_real, eps_imag = cls._drude_permittivity(wavelength)
        return complex(eps_real, eps_imag)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get complex refractive index at specified wavelength."""
        return np.sqrt(cls.get_permittivity(wavelength))


# Alias
Ti = Titanium


# =============================================================================
# POLYMERS
# =============================================================================

class PMMA(Material):
    """
    Poly(methyl methacrylate) - Acrylic / Plexiglass.
    n ≈ 1.49 at 589nm
    Common resist material for electron-beam lithography.
    """
    def __init__(self):
        super().__init__(permittivity=1.49**2, permeability=1.0, conductivity=0.0)
        self.name = "PMMA"
        self.refractive_index = 1.49


class SU8(Material):
    """
    SU-8 photoresist.
    n ≈ 1.58 at 633nm
    Thick photoresist for high aspect ratio structures.
    """
    def __init__(self):
        super().__init__(permittivity=1.58**2, permeability=1.0, conductivity=0.0)
        self.name = "SU-8"
        self.refractive_index = 1.58


class Polystyrene(Material):
    """
    Polystyrene (PS).
    n ≈ 1.59 at 589nm
    Common polymer for microspheres and thin films.
    """
    def __init__(self):
        super().__init__(permittivity=1.59**2, permeability=1.0, conductivity=0.0)
        self.name = "Polystyrene"
        self.refractive_index = 1.59


class PDMS(Material):
    """
    Polydimethylsiloxane - Silicone rubber.
    n ≈ 1.41 at 589nm
    Flexible, transparent, biocompatible polymer.
    """
    def __init__(self):
        super().__init__(permittivity=1.41**2, permeability=1.0, conductivity=0.0)
        self.name = "PDMS"
        self.refractive_index = 1.41


class HSQ(Material):
    """
    Hydrogen silsesquioxane.
    n ≈ 1.39 at 633nm
    High-resolution negative-tone electron-beam resist.
    """
    def __init__(self):
        super().__init__(permittivity=1.39**2, permeability=1.0, conductivity=0.0)
        self.name = "HSQ"
        self.refractive_index = 1.39


# =============================================================================
# LIQUIDS
# =============================================================================

class Water(Material):
    """
    Water (H2O) at 25°C.
    n ≈ 1.333 at 589nm

    Args:
        wavelength: Wavelength in meters for Sellmeier calculation
    """
    # Sellmeier coefficients for water (Daimon & Masumura 2007)
    # C and E values are resonance wavelengths squared: λ² in m²
    SELLMEIER_A = 0.5684027565
    SELLMEIER_B = 0.1726177391
    SELLMEIER_C = (0.05084151894e-6)**2  # already squared, in m^2
    SELLMEIER_D = 0.02086189578
    SELLMEIER_E = (0.1818018908e-6)**2  # already squared, in m^2

    def __init__(self, wavelength=589e-9):
        n = self._calc_n(wavelength)
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=0.0)
        self.name = "Water"
        self.refractive_index = n
        self.wavelength = wavelength

    @classmethod
    def _calc_n(cls, wavelength):
        """Calculate refractive index."""
        wl_sq = wavelength**2
        n_sq = 1 + cls.SELLMEIER_A * wl_sq / (wl_sq - cls.SELLMEIER_C)
        n_sq += cls.SELLMEIER_B * wl_sq / (wl_sq - cls.SELLMEIER_E)
        n_sq += cls.SELLMEIER_D
        return np.sqrt(n_sq)

    @classmethod
    def get_refractive_index(cls, wavelength):
        """Get refractive index at specified wavelength."""
        return cls._calc_n(wavelength)


# Alias
H2O = Water


class Ethanol(Material):
    """
    Ethanol (C2H5OH) at 25°C.
    n ≈ 1.361 at 589nm
    """
    def __init__(self):
        super().__init__(permittivity=1.361**2, permeability=1.0, conductivity=0.0)
        self.name = "Ethanol"
        self.refractive_index = 1.361


class IPA(Material):
    """
    Isopropyl Alcohol (2-Propanol).
    n ≈ 1.377 at 589nm
    Common cleaning solvent in semiconductor processing.
    """
    def __init__(self):
        super().__init__(permittivity=1.377**2, permeability=1.0, conductivity=0.0)
        self.name = "IPA"
        self.refractive_index = 1.377


class Glycerol(Material):
    """
    Glycerol (Glycerin).
    n ≈ 1.474 at 589nm
    Index-matching fluid for microscopy.
    """
    def __init__(self):
        super().__init__(permittivity=1.474**2, permeability=1.0, conductivity=0.0)
        self.name = "Glycerol"
        self.refractive_index = 1.474


class ImmersionOil(Material):
    """
    Microscope immersion oil.
    n ≈ 1.518 at 589nm (matched to glass coverslip)
    """
    def __init__(self):
        super().__init__(permittivity=1.518**2, permeability=1.0, conductivity=0.0)
        self.name = "Immersion Oil"
        self.refractive_index = 1.518


# =============================================================================
# SPECIALTY & 2D MATERIALS
# =============================================================================

class ITO(Material):
    """
    Indium Tin Oxide (90% In2O3, 10% SnO2).
    n ≈ 1.9 at 550nm (varies with deposition conditions)
    Transparent conductive oxide.

    Args:
        wavelength: Wavelength in meters
        conductivity: DC conductivity in S/m (varies 1e4 to 1e6 S/m)
    """
    def __init__(self, wavelength=550e-9, conductivity=1e5):
        # ITO has complex wavelength-dependent behavior
        # This is a simplified model
        n = 1.9  # Approximate
        super().__init__(permittivity=n**2, permeability=1.0, conductivity=conductivity)
        self.name = "ITO"
        self.refractive_index = n


class TiO2(Material):
    """
    Titanium Dioxide (Rutile).
    n ≈ 2.6 at 589nm (ordinary ray)
    High refractive index dielectric for photonics.
    """
    def __init__(self):
        super().__init__(permittivity=2.6**2, permeability=1.0, conductivity=0.0)
        self.name = "TiO2"
        self.refractive_index = 2.6


class HfO2(Material):
    """
    Hafnium Dioxide.
    n ≈ 2.0 at 633nm
    High-k dielectric material.
    """
    def __init__(self):
        super().__init__(permittivity=2.0**2, permeability=1.0, conductivity=0.0)
        self.name = "HfO2"
        self.refractive_index = 2.0


class Ta2O5(Material):
    """
    Tantalum Pentoxide.
    n ≈ 2.1 at 633nm
    Optical coating material.
    """
    def __init__(self):
        super().__init__(permittivity=2.1**2, permeability=1.0, conductivity=0.0)
        self.name = "Ta2O5"
        self.refractive_index = 2.1


class ZnO(Material):
    """
    Zinc Oxide.
    n ≈ 1.95 at 633nm
    Wide bandgap semiconductor with piezoelectric properties.
    """
    def __init__(self):
        super().__init__(permittivity=1.95**2, permeability=1.0, conductivity=0.0)
        self.name = "ZnO"
        self.refractive_index = 1.95


class AlN(Material):
    """
    Aluminum Nitride.
    n ≈ 2.2 at 633nm
    Piezoelectric material for MEMS and photonics.
    """
    def __init__(self):
        super().__init__(permittivity=2.2**2, permeability=1.0, conductivity=0.0)
        self.name = "AlN"
        self.refractive_index = 2.2


class MgF2(Material):
    """
    Magnesium Fluoride.
    n ≈ 1.38 at 589nm
    Low refractive index material for anti-reflection coatings.
    """
    def __init__(self):
        super().__init__(permittivity=1.38**2, permeability=1.0, conductivity=0.0)
        self.name = "MgF2"
        self.refractive_index = 1.38


class CaF2(Material):
    """
    Calcium Fluoride.
    n ≈ 1.43 at 589nm
    UV and IR transparent optical material.
    """
    def __init__(self):
        super().__init__(permittivity=1.43**2, permeability=1.0, conductivity=0.0)
        self.name = "CaF2"
        self.refractive_index = 1.43


class BaF2(Material):
    """
    Barium Fluoride.
    n ≈ 1.47 at 589nm
    IR transparent to ~15 μm.
    """
    def __init__(self):
        super().__init__(permittivity=1.47**2, permeability=1.0, conductivity=0.0)
        self.name = "BaF2"
        self.refractive_index = 1.47


class ZnSe(Material):
    """
    Zinc Selenide.
    n ≈ 2.4 at 10.6 μm (CO2 laser wavelength)
    Common IR window and lens material.
    """
    def __init__(self):
        super().__init__(permittivity=2.4**2, permeability=1.0, conductivity=0.0)
        self.name = "ZnSe"
        self.refractive_index = 2.4


class ZnS(Material):
    """
    Zinc Sulfide.
    n ≈ 2.36 at 589nm
    IR optical material.
    """
    def __init__(self):
        super().__init__(permittivity=2.36**2, permeability=1.0, conductivity=0.0)
        self.name = "ZnS"
        self.refractive_index = 2.36


# =============================================================================
# PERFECT CONDUCTORS & SPECIAL MATERIALS
# =============================================================================

class PEC(Material):
    """
    Perfect Electric Conductor.
    Idealized material with infinite conductivity.
    Use very high conductivity value for FDTD approximation.
    """
    def __init__(self):
        super().__init__(permittivity=1.0, permeability=1.0, conductivity=1e10)
        self.name = "PEC"


class PMC(Material):
    """
    Perfect Magnetic Conductor.
    Idealized material with infinite magnetic permeability.
    Note: This is an approximation; true PMC doesn't exist in nature.
    """
    def __init__(self):
        super().__init__(permittivity=1.0, permeability=1e6, conductivity=0.0)
        self.name = "PMC"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def list_materials():
    """Return a list of all available materials."""
    materials = [
        # Vacuum & Gases
        "Vacuum", "Air",
        # Dielectrics & Glasses
        "SiO2", "FusedSilica", "Si3N4", "SiliconNitride", "SiN", "BK7",
        "SodaLimeGlass", "Sapphire", "Al2O3", "Diamond",
        # Semiconductors
        "Silicon", "Si", "Germanium", "Ge", "GaAs", "InP", "LiNbO3",
        # Metals
        "Gold", "Au", "Silver", "Ag", "Copper", "Cu", "Aluminum", "Al",
        "Chromium", "Cr", "Titanium", "Ti",
        # Polymers
        "PMMA", "SU8", "Polystyrene", "PDMS", "HSQ",
        # Liquids
        "Water", "H2O", "Ethanol", "IPA", "Glycerol", "ImmersionOil",
        # Specialty
        "ITO", "TiO2", "HfO2", "Ta2O5", "ZnO", "AlN",
        "MgF2", "CaF2", "BaF2", "ZnSe", "ZnS",
        # Special
        "PEC", "PMC"
    ]
    return materials


def get_material(name, **kwargs):
    """
    Get a material instance by name.

    Args:
        name: Material name (e.g., "Silicon", "Gold", "SiO2")
        **kwargs: Additional arguments passed to material constructor
                 (e.g., wavelength=1.55e-6)

    Returns:
        Material instance

    Example:
        si = get_material("Silicon", wavelength=1.31e-6)
        au = get_material("Gold", wavelength=800e-9)
    """
    materials_dict = {
        # Vacuum & Gases
        "vacuum": Vacuum, "air": Air,
        # Dielectrics
        "sio2": SiO2, "fusedsilica": SiO2, "silica": SiO2,
        "si3n4": Si3N4, "siliconnitride": Si3N4, "sin": Si3N4,
        "bk7": BK7, "sodalimeglass": SodaLimeGlass,
        "sapphire": Sapphire, "al2o3": Sapphire,
        "diamond": Diamond,
        # Semiconductors
        "silicon": Silicon, "si": Silicon,
        "germanium": Germanium, "ge": Germanium,
        "gaas": GaAs, "inp": InP,
        "linbo3": LiNbO3, "lithiumniobate": LiNbO3,
        # Metals
        "gold": Gold, "au": Gold,
        "silver": Silver, "ag": Silver,
        "copper": Copper, "cu": Copper,
        "aluminum": Aluminum, "al": Aluminum,
        "chromium": Chromium, "cr": Chromium,
        "titanium": Titanium, "ti": Titanium,
        # Polymers
        "pmma": PMMA, "su8": SU8, "su-8": SU8,
        "polystyrene": Polystyrene, "pdms": PDMS, "hsq": HSQ,
        # Liquids
        "water": Water, "h2o": Water,
        "ethanol": Ethanol, "ipa": IPA,
        "glycerol": Glycerol, "immersionoil": ImmersionOil,
        # Specialty
        "ito": ITO, "tio2": TiO2, "hfo2": HfO2,
        "ta2o5": Ta2O5, "zno": ZnO, "aln": AlN,
        "mgf2": MgF2, "caf2": CaF2, "baf2": BaF2,
        "znse": ZnSe, "zns": ZnS,
        # Special
        "pec": PEC, "pmc": PMC,
    }

    key = name.lower().replace(" ", "").replace("_", "")
    if key not in materials_dict:
        available = list_materials()
        raise ValueError(f"Unknown material: {name}. Available: {available}")

    return materials_dict[key](**kwargs)


def material_info(name):
    """
    Get detailed information about a material.

    Args:
        name: Material name

    Returns:
        dict with material properties and metadata
    """
    mat = get_material(name)
    info = {
        "name": mat.name if hasattr(mat, 'name') else name,
        "permittivity": mat.permittivity,
        "permeability": mat.permeability,
        "conductivity": mat.conductivity,
    }
    if hasattr(mat, 'refractive_index'):
        info["refractive_index"] = mat.refractive_index
    if hasattr(mat, 'wavelength'):
        info["wavelength"] = mat.wavelength
    if hasattr(mat, 'eps_complex'):
        info["complex_permittivity"] = mat.eps_complex
    return info


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Vacuum & Gases
    'Vacuum', 'Air',
    # Dielectrics & Glasses
    'SiO2', 'FusedSilica', 'Silica', 'Si3N4', 'SiliconNitride', 'SiN',
    'BK7', 'SodaLimeGlass', 'Sapphire', 'Al2O3', 'Diamond',
    # Semiconductors
    'Silicon', 'Si', 'cSi', 'Germanium', 'Ge', 'GaAs', 'InP',
    'LiNbO3', 'LithiumNiobate',
    # Metals
    'Gold', 'Au', 'Silver', 'Ag', 'Copper', 'Cu', 'Aluminum', 'Al',
    'Chromium', 'Cr', 'Titanium', 'Ti',
    # Polymers
    'PMMA', 'SU8', 'Polystyrene', 'PDMS', 'HSQ',
    # Liquids
    'Water', 'H2O', 'Ethanol', 'IPA', 'Glycerol', 'ImmersionOil',
    # Specialty
    'ITO', 'TiO2', 'HfO2', 'Ta2O5', 'ZnO', 'AlN',
    'MgF2', 'CaF2', 'BaF2', 'ZnSe', 'ZnS',
    # Special
    'PEC', 'PMC',
    # Functions
    'list_materials', 'get_material', 'material_info',
]
=======
# Copper
>>>>>>> main
