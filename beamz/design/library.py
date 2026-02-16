"""Deprecated Beamz material-library helpers.

Use `beamz.material_library.material_library`.
This module is kept as a migration shim.
"""

from __future__ import annotations

from copy import deepcopy
import warnings

from beamz.design.materials import (
    AnisotropicMaterial,
    Material,
    Material2D,
)
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


def _to_design_material(model) -> Material:
    if isinstance(model, Material):
        return Material(
            permittivity=model.permittivity,
            permeability=model.permeability,
            conductivity=model.conductivity,
            k=model.k,
            rho=model.rho,
            cp=model.cp,
            dn_dT=model.dn_dT,
            T0=model.T0,
            name=model.name,
            frequency_range=model.frequency_range,
        )

    if isinstance(model, Material2D):
        return _to_design_material(model.ss)

    if isinstance(model, AnisotropicMaterial):
        return _to_design_material(model.xx)

    if hasattr(model, "to_material"):
        return model.to_material(wavelength=REFERENCE_WAVELENGTH_M)

    raise ValueError(f"Unsupported medium conversion path for type '{type(model).__name__}'.")


def list_materials(category: str | None = None, include_symbolic: bool = True) -> list[str]:
    """Deprecated helper: list material keys from new material_library.

    Category filtering is no longer applicable in the same form and is ignored.
    """
    _warn_deprecated()
    names = []
    for key in material_library:
        if not include_symbolic and key in {"PEC", "PMC"}:
            continue
        names.append(key)
    return sorted(set(names), key=str.lower)


def material_info(name: str) -> dict:
    """Deprecated helper: return metadata summary from new material_library."""
    _warn_deprecated()
    key, item = _resolve_item(name)

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

    medium_or_callable = item.medium
    if callable(medium_or_callable):
        medium = medium_or_callable("z")
    else:
        medium = medium_or_callable

    return _to_design_material(medium)


__all__ = [
    "REFERENCE_WAVELENGTH_M",
    "list_materials",
    "get_material",
    "material_info",
]
