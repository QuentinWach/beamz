"""Deprecated Beamz material-library helpers.

Use `beamz.material_library.material_library`.
This module is kept as a migration shim.
"""

from __future__ import annotations

from copy import deepcopy
import warnings

from beamz.components.medium import Medium, Medium2D, PoleResidue
from beamz.const import LIGHT_SPEED
from beamz.design.materials import Material
from beamz.material_library.material_library import material_library


REFERENCE_WAVELENGTH_M = 1.55e-6


def _warn_deprecated() -> None:
    warnings.warn(
        "beamz.design.library is deprecated. Use `beamz.material_library.material_library` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def _resolve_item(name: str):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("material name must be a non-empty string")

    needle = name.strip().lower()
    for key, item in material_library.items():
        if key.lower() == needle:
            return key, item
        if hasattr(item, "name") and str(item.name).lower() == needle:
            return key, item

    raise KeyError(f"Unknown material '{name}'.")


def _pole_residue_to_material(pr: PoleResidue, wavelength: float = REFERENCE_WAVELENGTH_M) -> Material:
    frequency = LIGHT_SPEED / wavelength
    med = pr.to_medium(frequency)
    return Material(
        permittivity=med.permittivity,
        permeability=med.permeability,
        conductivity=med.conductivity,
    )


def list_materials(category: str | None = None, include_symbolic: bool = True) -> list[str]:
    """Deprecated helper: list material keys from new material_library.

    Category filtering is no longer applicable in the same form and is ignored.
    """
    _warn_deprecated()
    names = []
    for key, item in material_library.items():
        if not include_symbolic and key in {"PEC", "PMC"}:
            continue
        if isinstance(item, type):
            continue
        names.append(key)
    names = sorted(set(names), key=str.lower)
    return names


def material_info(name: str) -> dict:
    """Deprecated helper: return metadata summary from new material_library."""
    _warn_deprecated()
    key, item = _resolve_item(name)

    if isinstance(item, type):
        return {
            "key": key,
            "name": key,
            "kind": "parametric_class",
            "class_name": item.__name__,
        }

    payload = {
        "key": key,
        "name": item.name,
        "default": getattr(item, "default", None),
        "variants": sorted(list(getattr(item, "variants", {}).keys())),
        "symbolic": key in {"PEC", "PMC"},
        "reference_wavelength_m": REFERENCE_WAVELENGTH_M,
    }
    return deepcopy(payload)


def get_material(name: str, allow_symbolic: bool = False):
    """Deprecated helper: return design.Material converted from new default medium."""
    _warn_deprecated()
    key, item = _resolve_item(name)

    if key in {"PEC", "PMC"}:
        if allow_symbolic:
            return material_info(key)
        raise ValueError(
            f"{key} is symbolic. Use material_info('{key}') or the new material_library API."
        )

    if isinstance(item, type):
        if allow_symbolic:
            return material_info(key)
        raise ValueError(
            f"{key} is a parametric class entry. Instantiate via `beamz.Graphene(...)`."
        )

    med = item.medium
    if isinstance(med, PoleResidue):
        return _pole_residue_to_material(med)
    if isinstance(med, Medium):
        return Material(
            permittivity=med.permittivity,
            permeability=med.permeability,
            conductivity=med.conductivity,
        )
    if isinstance(med, Medium2D):
        # Migration fallback: use in-plane `ss` component.
        if isinstance(med.ss, PoleResidue):
            return _pole_residue_to_material(med.ss)
        if isinstance(med.ss, Medium):
            return Material(
                permittivity=med.ss.permittivity,
                permeability=med.ss.permeability,
                conductivity=med.ss.conductivity,
            )

    raise ValueError(f"Unsupported medium conversion path for material '{name}'.")


__all__ = [
    "REFERENCE_WAVELENGTH_M",
    "list_materials",
    "get_material",
    "material_info",
]
