"""Starter material presets for common photonics workflows.

This module intentionally provides a compact set of baseline presets.
They are convenient defaults, not a full dispersion-aware material DB.
"""

from __future__ import annotations

from beamz.design.materials import Material

# Baseline dielectric presets (permittivity = n^2).
VACUUM = Material(permittivity=1.0)
AIR = Material(permittivity=1.0006**2)
SIO2 = Material(permittivity=1.444**2)
SIN = Material(permittivity=2.0**2)
SI3N4 = Material(permittivity=2.0**2)

# Simple conductive placeholders for coarse EM setup (non-dispersive).
GOLD = Material(permittivity=1.0, conductivity=4.1e7)
ALUMINUM = Material(permittivity=1.0, conductivity=3.5e7)
COPPER = Material(permittivity=1.0, conductivity=5.96e7)


def by_name(name: str) -> Material:
    """Return a new `Material` instance for a known preset name."""
    key = str(name).strip().lower()
    table = {
        "vacuum": VACUUM,
        "air": AIR,
        "sio2": SIO2,
        "sin": SIN,
        "si3n4": SI3N4,
        "gold": GOLD,
        "aluminum": ALUMINUM,
        "copper": COPPER,
    }
    if key not in table:
        raise ValueError(
            f"Unknown material preset {name!r}. "
            f"Available: {sorted(table.keys())}"
        )

    src = table[key]
    return Material(
        permittivity=src.permittivity,
        permeability=src.permeability,
        conductivity=src.conductivity,
        k=src.k,
        rho=src.rho,
        cp=src.cp,
        dn_dT=src.dn_dT,
        T0=src.T0,
    )


__all__ = [
    "VACUUM",
    "AIR",
    "SIO2",
    "SIN",
    "SI3N4",
    "GOLD",
    "ALUMINUM",
    "COPPER",
    "by_name",
]
