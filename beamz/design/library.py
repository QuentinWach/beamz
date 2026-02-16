"""Predefined material catalog for BEAMZ.

This module provides:
- A static library of dispersionless `Material` presets (55+ entries).
- Alias-aware lookup helpers.
- Metadata access for provenance/reference wavelengths.
- Symbolic `PEC`/`PMC` entries for future boundary-condition workflows.

All optical constants in the static table are referenced to 1.55 um.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED
from beamz.design.materials import Material


REFERENCE_WAVELENGTH_M = 1.55e-6
REFERENCE_FREQUENCY_HZ = LIGHT_SPEED / REFERENCE_WAVELENGTH_M


def _eps_sigma_from_nk(n: float, kappa: float) -> tuple[float, float]:
    eps_complex = complex(n, kappa) ** 2
    eps_r = float(np.real(eps_complex))
    sigma = abs(2.0 * np.pi * REFERENCE_FREQUENCY_HZ * EPS_0 * float(np.imag(eps_complex)))
    return eps_r, sigma


def _entry(
    canonical_name: str,
    *,
    category: str,
    aliases: list[str] | None = None,
    source: str,
    notes: str = "",
    n: float | None = None,
    kappa: float = 0.0,
    permittivity: float | None = None,
    permeability: float = 1.0,
    conductivity: float | None = None,
    thermal: dict[str, float] | None = None,
    symbolic: bool = False,
) -> dict[str, Any]:
    aliases = aliases or []
    thermal = thermal or {}

    if symbolic:
        eps_r = None
        sigma = None
    else:
        if permittivity is None:
            if n is None:
                raise ValueError(f"Material {canonical_name} requires n or permittivity")
            eps_r, sigma_from_nk = _eps_sigma_from_nk(n=n, kappa=kappa)
            sigma = sigma_from_nk if conductivity is None else float(conductivity)
        else:
            eps_r = float(permittivity)
            sigma = 0.0 if conductivity is None else float(conductivity)

    return {
        "canonical_name": canonical_name,
        "aliases": aliases,
        "category": category,
        "source": source,
        "notes": notes,
        "reference_wavelength_m": REFERENCE_WAVELENGTH_M,
        "symbolic": symbolic,
        "n": n,
        "kappa": kappa,
        "permittivity": eps_r,
        "permeability": float(permeability),
        "conductivity": sigma,
        "thermal": {
            "k": float(thermal.get("k", 0.0)),
            "rho": float(thermal.get("rho", 0.0)),
            "cp": float(thermal.get("cp", 0.0)),
            "dn_dT": float(thermal.get("dn_dT", 0.0)),
            "T0": float(thermal.get("T0", 300.0)),
        },
    }


_MATERIALS: dict[str, dict[str, Any]] = {
    # Vacuum and gases
    "Vacuum": _entry(
        "Vacuum",
        category="vacuum_gases",
        aliases=["free_space"],
        source="Physical constants",
        notes="Reference vacuum medium",
        permittivity=1.0,
    ),
    "Air": _entry(
        "Air",
        category="vacuum_gases",
        aliases=["atm", "atmosphere"],
        source="Standard optics approximation",
        notes="Dry air around 1 atm",
        n=1.00027,
    ),
    # Dielectrics and glasses
    "SiO2": _entry(
        "SiO2",
        category="dielectrics_glasses",
        aliases=["silica", "fused_silica"],
        source="Common telecom approximation",
        notes="Fused silica at 1.55 um",
        n=1.444,
        thermal={"k": 1.38, "dn_dT": 1.0e-5},
    ),
    "Si3N4": _entry(
        "Si3N4",
        category="dielectrics_glasses",
        aliases=["sin", "sinx", "silicon_nitride"],
        source="Common photonics approximation",
        notes="Stoichiometric silicon nitride",
        n=2.04,
        thermal={"k": 20.0, "dn_dT": 2.45e-5},
    ),
    "BK7": _entry(
        "BK7",
        category="dielectrics_glasses",
        aliases=["nbk7"],
        source="SCHOTT BK7 refractive index",
        n=1.5007,
    ),
    "Sapphire": _entry(
        "Sapphire",
        category="dielectrics_glasses",
        aliases=["al2o3_crystal"],
        source="Typical ordinary index at 1.55 um",
        n=1.75,
    ),
    "Diamond": _entry(
        "Diamond",
        category="dielectrics_glasses",
        aliases=["c_diamond"],
        source="Typical CVD diamond index",
        n=2.40,
    ),
    "Quartz": _entry(
        "Quartz",
        category="dielectrics_glasses",
        aliases=["alpha_quartz"],
        source="Typical crystalline quartz index",
        n=1.54,
    ),
    "Borosilicate": _entry(
        "Borosilicate",
        category="dielectrics_glasses",
        aliases=["pyrex"],
        source="Typical borosilicate glass data",
        n=1.47,
    ),
    "Alumina": _entry(
        "Alumina",
        category="dielectrics_glasses",
        aliases=["Al2O3"],
        source="Typical alumina ceramic properties",
        n=1.76,
    ),
    "Ta2O5": _entry(
        "Ta2O5",
        category="dielectrics_glasses",
        aliases=["tantalum_pentoxide"],
        source="Thin-film optics approximation",
        n=2.10,
    ),
    "SiON": _entry(
        "SiON",
        category="dielectrics_glasses",
        aliases=["silicon_oxynitride"],
        source="Typical SiON waveguide composition",
        n=1.80,
    ),
    "YAG": _entry(
        "YAG",
        category="dielectrics_glasses",
        aliases=["y3al5o12"],
        source="YAG crystal index approximation",
        n=1.82,
    ),
    "Chalcogenide": _entry(
        "Chalcogenide",
        category="dielectrics_glasses",
        aliases=["chalcogenide_glass", "as2s3"],
        source="As2S3-family index approximation",
        n=2.40,
    ),
    # Semiconductors
    "Silicon": _entry(
        "Silicon",
        category="semiconductors",
        aliases=["si", "csi"],
        source="Telecom silicon index",
        n=3.476,
        thermal={"k": 130.0, "dn_dT": 1.86e-4},
    ),
    "Germanium": _entry(
        "Germanium",
        category="semiconductors",
        aliases=["ge"],
        source="Typical Ge index at 1.55 um",
        n=4.00,
    ),
    "GaAs": _entry(
        "GaAs",
        category="semiconductors",
        aliases=["gallium_arsenide"],
        source="Typical GaAs index at 1.55 um",
        n=3.37,
    ),
    "InP": _entry(
        "InP",
        category="semiconductors",
        aliases=["indium_phosphide"],
        source="Typical InP index at 1.55 um",
        n=3.17,
    ),
    "LiNbO3": _entry(
        "LiNbO3",
        category="semiconductors",
        aliases=["linbo3", "lithium_niobate", "lno"],
        source="Typical extraordinary/ordinary averaged index",
        n=2.20,
        thermal={"dn_dT": 3.0e-5},
    ),
    "GaN": _entry(
        "GaN",
        category="semiconductors",
        aliases=["gallium_nitride"],
        source="Typical GaN index",
        n=2.32,
    ),
    "SiC": _entry(
        "SiC",
        category="semiconductors",
        aliases=["silicon_carbide"],
        source="Typical SiC index",
        n=2.60,
    ),
    "InGaAs": _entry(
        "InGaAs",
        category="semiconductors",
        aliases=["ingaas"],
        source="Typical InGaAs near-IR value",
        n=3.50,
    ),
    "AlGaAs": _entry(
        "AlGaAs",
        category="semiconductors",
        aliases=["algaas"],
        source="Typical AlGaAs near-IR value",
        n=3.20,
    ),
    "CdTe": _entry(
        "CdTe",
        category="semiconductors",
        aliases=["cadmium_telluride"],
        source="Typical CdTe index",
        n=2.70,
    ),
    # Metals
    "Gold": _entry(
        "Gold",
        category="metals",
        aliases=["au"],
        source="Approximate optical constants at 1.55 um",
        notes="See Drude/Drude-Lorentz models for dispersion",
        n=0.55,
        kappa=11.5,
    ),
    "Silver": _entry(
        "Silver",
        category="metals",
        aliases=["ag"],
        source="Approximate optical constants at 1.55 um",
        notes="See Drude model for dispersion",
        n=0.14,
        kappa=11.2,
    ),
    "Copper": _entry(
        "Copper",
        category="metals",
        aliases=["cu"],
        source="Approximate optical constants at 1.55 um",
        notes="See Drude model for dispersion",
        n=0.31,
        kappa=10.9,
    ),
    "Aluminum": _entry(
        "Aluminum",
        category="metals",
        aliases=["al"],
        source="Approximate optical constants at 1.55 um",
        notes="See Drude model for dispersion",
        n=1.44,
        kappa=14.6,
    ),
    "Chromium": _entry(
        "Chromium",
        category="metals",
        aliases=["cr"],
        source="Approximate optical constants at 1.55 um",
        notes="Lossy transition metal",
        n=3.30,
        kappa=4.30,
    ),
    "Titanium": _entry(
        "Titanium",
        category="metals",
        aliases=["ti"],
        source="Approximate optical constants at 1.55 um",
        notes="Transition metal, Drude-like behavior",
        n=3.00,
        kappa=3.50,
    ),
    "Nickel": _entry(
        "Nickel",
        category="metals",
        aliases=["ni"],
        source="Approximate optical constants at 1.55 um",
        n=2.00,
        kappa=4.50,
    ),
    "Platinum": _entry(
        "Platinum",
        category="metals",
        aliases=["pt"],
        source="Approximate optical constants at 1.55 um",
        n=2.30,
        kappa=4.10,
    ),
    "Tungsten": _entry(
        "Tungsten",
        category="metals",
        aliases=["w"],
        source="Approximate optical constants at 1.55 um",
        n=3.70,
        kappa=2.70,
    ),
    "Iron": _entry(
        "Iron",
        category="metals",
        aliases=["fe"],
        source="Approximate optical constants at 1.55 um",
        n=2.90,
        kappa=3.30,
    ),
    # Polymers
    "PMMA": _entry(
        "PMMA",
        category="polymers",
        aliases=["acrylic", "plexiglass"],
        source="Typical PMMA refractive index",
        n=1.49,
    ),
    "SU8": _entry(
        "SU8",
        category="polymers",
        aliases=["su-8", "epoxy_photoresist"],
        source="Typical SU-8 index",
        n=1.57,
    ),
    "Polystyrene": _entry(
        "Polystyrene",
        category="polymers",
        aliases=["ps"],
        source="Typical polystyrene index",
        n=1.59,
    ),
    "PDMS": _entry(
        "PDMS",
        category="polymers",
        aliases=["silicone_elastomer"],
        source="Typical PDMS index",
        n=1.41,
    ),
    "HSQ": _entry(
        "HSQ",
        category="polymers",
        aliases=["hydrogen_silsesquioxane"],
        source="Typical HSQ-like index after cure",
        n=1.40,
    ),
    "Polycarbonate": _entry(
        "Polycarbonate",
        category="polymers",
        aliases=["pc"],
        source="Typical polycarbonate index",
        n=1.58,
    ),
    "COC": _entry(
        "COC",
        category="polymers",
        aliases=["cyclic_olefin_copolymer"],
        source="Typical COC index",
        n=1.53,
    ),
    "ParyleneC": _entry(
        "ParyleneC",
        category="polymers",
        aliases=["parylene_c"],
        source="Typical Parylene-C index",
        n=1.64,
    ),
    "PTFE": _entry(
        "PTFE",
        category="polymers",
        aliases=["teflon"],
        source="Typical PTFE optical approximation",
        n=1.35,
    ),
    "Epoxy": _entry(
        "Epoxy",
        category="polymers",
        aliases=["optical_epoxy"],
        source="Typical optical epoxy index",
        n=1.55,
    ),
    # Liquids
    "Water": _entry(
        "Water",
        category="liquids",
        aliases=["h2o", "deionized_water"],
        source="Typical liquid water index at 1.55 um",
        n=1.318,
    ),
    "Ethanol": _entry(
        "Ethanol",
        category="liquids",
        aliases=["etoh"],
        source="Typical ethanol index",
        n=1.353,
    ),
    "IPA": _entry(
        "IPA",
        category="liquids",
        aliases=["isopropanol", "isopropyl_alcohol"],
        source="Typical IPA index",
        n=1.377,
    ),
    "Glycerol": _entry(
        "Glycerol",
        category="liquids",
        aliases=["glycerin"],
        source="Typical glycerol index",
        n=1.472,
    ),
    "Acetone": _entry(
        "Acetone",
        category="liquids",
        aliases=["propanone"],
        source="Typical acetone index",
        n=1.358,
    ),
    "Methanol": _entry(
        "Methanol",
        category="liquids",
        aliases=["meoh"],
        source="Typical methanol index",
        n=1.329,
    ),
    "Toluene": _entry(
        "Toluene",
        category="liquids",
        aliases=["methylbenzene"],
        source="Typical toluene index",
        n=1.496,
    ),
    # Specialty
    "ITO": _entry(
        "ITO",
        category="specialty",
        aliases=["indium_tin_oxide"],
        source="Approximate ITO optical constants",
        notes="Conductive oxide with process-dependent properties",
        n=1.90,
        kappa=0.02,
        conductivity=2.0e5,
    ),
    "TiO2": _entry(
        "TiO2",
        category="specialty",
        aliases=["tio2", "titania"],
        source="Typical TiO2 high-index dielectric",
        n=2.40,
    ),
    "HfO2": _entry(
        "HfO2",
        category="specialty",
        aliases=["hafnia"],
        source="Typical HfO2 index",
        n=1.95,
    ),
    "ZnO": _entry(
        "ZnO",
        category="specialty",
        aliases=["zno"],
        source="Typical ZnO index",
        n=2.00,
    ),
    "AlN": _entry(
        "AlN",
        category="specialty",
        aliases=["aln", "aluminum_nitride"],
        source="Typical AlN index",
        n=2.10,
    ),
    "MgF2": _entry(
        "MgF2",
        category="specialty",
        aliases=["mgf2", "magnesium_fluoride"],
        source="Typical MgF2 low-index value",
        n=1.38,
    ),
    "CaF2": _entry(
        "CaF2",
        category="specialty",
        aliases=["caf2", "calcium_fluoride"],
        source="Typical CaF2 low-index value",
        n=1.43,
    ),
    "ZnSe": _entry(
        "ZnSe",
        category="specialty",
        aliases=["znse"],
        source="Typical ZnSe value",
        n=2.43,
    ),
    "ZnS": _entry(
        "ZnS",
        category="specialty",
        aliases=["zns"],
        source="Typical ZnS value",
        n=2.20,
    ),
    "SiO": _entry(
        "SiO",
        category="specialty",
        aliases=["silicon_monoxide"],
        source="Typical SiO thin-film approximation",
        n=1.90,
    ),
    "Al2O3": _entry(
        "Al2O3",
        category="specialty",
        aliases=["alumina_film"],
        source="Typical Al2O3 thin-film value",
        n=1.63,
    ),
    "Nb2O5": _entry(
        "Nb2O5",
        category="specialty",
        aliases=["niobium_pentoxide"],
        source="Typical Nb2O5 high-index dielectric",
        n=2.24,
    ),
    # Symbolic placeholders
    "PEC": _entry(
        "PEC",
        category="special",
        aliases=["perfect_electric_conductor"],
        source="Maxwell boundary-condition concept",
        notes="Symbolic placeholder. Use boundary conditions, not bulk fill.",
        symbolic=True,
    ),
    "PMC": _entry(
        "PMC",
        category="special",
        aliases=["perfect_magnetic_conductor"],
        source="Maxwell boundary-condition concept",
        notes="Symbolic placeholder. Use boundary conditions, not bulk fill.",
        symbolic=True,
    ),
}


_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical_name, payload in _MATERIALS.items():
    _ALIAS_TO_CANONICAL[canonical_name.lower()] = canonical_name
    for alias in payload["aliases"]:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical_name


def _resolve_material_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("material name must be a non-empty string")
    key = name.strip().lower()
    canonical = _ALIAS_TO_CANONICAL.get(key)
    if canonical is None:
        raise KeyError(f"Unknown material '{name}'. Use list_materials() to inspect options.")
    return canonical


def list_materials(
    category: str | None = None,
    include_symbolic: bool = True,
) -> list[str]:
    """List available canonical material names.

    Args:
        category: Optional category filter (case-insensitive).
        include_symbolic: Include symbolic entries (`PEC`, `PMC`) when True.
    """
    if category is None:
        names = [
            name
            for name, payload in _MATERIALS.items()
            if include_symbolic or not payload["symbolic"]
        ]
    else:
        cat = category.strip().lower()
        names = [
            name
            for name, payload in _MATERIALS.items()
            if payload["category"].lower() == cat
            and (include_symbolic or not payload["symbolic"])
        ]

    return sorted(names, key=str.lower)


def material_info(name: str) -> dict[str, Any]:
    """Return metadata and physical table values for one material."""
    canonical = _resolve_material_name(name)
    return deepcopy(_MATERIALS[canonical])


def get_material(name: str, allow_symbolic: bool = False):
    """Get a predefined `Material` by name or alias.

    Symbolic entries (`PEC`/`PMC`) do not map to physical bulk materials.
    If `allow_symbolic=True` and a symbolic entry is requested, this function
    returns its metadata dictionary.
    """
    canonical = _resolve_material_name(name)
    payload = _MATERIALS[canonical]

    if payload["symbolic"]:
        if allow_symbolic:
            return material_info(canonical)
        raise ValueError(
            f"{canonical} is symbolic, not a bulk `Material`. "
            f"Use material_info('{canonical}') for metadata."
        )

    return Material(
        permittivity=payload["permittivity"],
        permeability=payload["permeability"],
        conductivity=payload["conductivity"],
        k=payload["thermal"]["k"],
        rho=payload["thermal"]["rho"],
        cp=payload["thermal"]["cp"],
        dn_dT=payload["thermal"]["dn_dT"],
        T0=payload["thermal"]["T0"],
    )


__all__ = [
    "REFERENCE_WAVELENGTH_M",
    "list_materials",
    "get_material",
    "material_info",
]
