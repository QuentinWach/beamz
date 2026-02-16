"""Curated material library defined directly in Python.

All registered materials are implemented as functions in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from beamz.design.materials import (
    AnisotropicMaterial,
    DebyeMaterial,
    DrudeMaterial,
    LorentzMaterial,
    Material,
    Material2D,
    PoleResidueMaterial,
    SellmeierMaterial,
)


def _rad_s_to_hz(omega: float) -> float:
    return omega / (2.0 * 3.141592653589793)


@dataclass(frozen=True)
class ReferenceData:
    doi: str | None = None
    journal: str | None = None
    url: str | None = None
    manufacturer: str | None = None
    datasheet_title: str | None = None


@dataclass(frozen=True)
class ExportData:
    source_repo: str
    source_commit: str
    generated_utc: str
    source_module: str
    material_count: int


MaterialModel = (
    Material
    | PoleResidueMaterial
    | SellmeierMaterial
    | DrudeMaterial
    | LorentzMaterial
    | DebyeMaterial
    | Material2D
)


@dataclass
class VariantItem:
    medium: MaterialModel
    reference: list[ReferenceData] | None = None
    data_url: str | None = None
    notes: str | None = None
    frequency_range: tuple[float, float] | None = None


@dataclass
class MaterialItem:
    name: str
    variants: dict[str, VariantItem]
    default: str

    def __post_init__(self):
        if self.default not in self.variants:
            raise ValueError(
                f"Default variant '{self.default}' not found in variants of {self.name}."
            )

    def __getitem__(self, variant_name: str) -> MaterialModel:
        return self.variants[variant_name].medium

    @property
    def medium(self) -> MaterialModel:
        return self.variants[self.default].medium


@dataclass
class VariantItem2D:
    medium: Material2D
    reference: list[ReferenceData] | None = None
    data_url: str | None = None
    notes: str | None = None
    frequency_range: tuple[float, float] | None = None


@dataclass
class MaterialItem2D(MaterialItem):
    variants: dict[str, VariantItem2D]


@dataclass
class VariantItemUniaxial:
    ordinary: Material | PoleResidueMaterial | SellmeierMaterial
    extraordinary: Material | PoleResidueMaterial | SellmeierMaterial
    reference: list[ReferenceData] | None = None
    data_url: str | None = None
    notes: str | None = None
    frequency_range: tuple[float, float] | None = None

    def medium(self, optical_axis: int | str = 2) -> AnisotropicMaterial:
        axes = {0: "x", 1: "y", 2: "z", "x": "x", "y": "y", "z": "z"}
        axis = axes.get(optical_axis)
        if axis is None:
            raise ValueError("optical_axis must be one of 0,1,2,'x','y','z'.")
        xx = self.extraordinary if axis == "x" else self.ordinary
        yy = self.extraordinary if axis == "y" else self.ordinary
        zz = self.extraordinary if axis == "z" else self.ordinary
        return AnisotropicMaterial(xx=xx, yy=yy, zz=zz)


@dataclass
class MaterialItemUniaxial(MaterialItem):
    variants: dict[str, VariantItemUniaxial]

    def medium(self, optical_axis: int | str = 2) -> AnisotropicMaterial:
        return self.variants[self.default].medium(optical_axis)


class MaterialLibrary(dict):
    def __str__(self) -> str:
        lines = ["Material Library Summary:"]
        for key, item in self.items():
            lines.append(
                f"  - Key: {key}, Name: {item.name}, "
                f"Default Variant: {item.default}, # Variants: {len(item.variants)}"
            )
        return "\n".join(lines)


# --- Material model functions ----------------------------------------------------------------


def vacuum() -> Material:
    return Material(name="Vacuum", permittivity=1.0, permeability=1.0, conductivity=0.0)


def air() -> Material:
    return Material(name="Air", permittivity=1.00054, permeability=1.0, conductivity=0.0)


def sio2() -> SellmeierMaterial:
    # Malitson (1965), fused silica, C in um^2
    return SellmeierMaterial(
        name="SiO2_Malitson1965",
        coeffs=(
            (0.6961663, 0.0684043**2),
            (0.4079426, 0.1162414**2),
            (0.8974794, 9.896161**2),
        ),
    )


def si3n4() -> SellmeierMaterial:
    # Luke et al. (2015) PECVD Si3N4 fit; single-term Sellmeier approximation.
    return SellmeierMaterial(
        name="Si3N4_Luke2015",
        coeffs=((2.8939, 0.13967**2),),
    )


def silicon() -> LorentzMaterial:
    # Compact Lorentz fit targeting near-IR silicon index behavior around telecom.
    return LorentzMaterial(
        name="Silicon_LorentzNearIR",
        eps_inf=7.6,
        coeffs=((4.5, 3.5e15, 1.0e14),),
    )


def linbo3_ordinary() -> SellmeierMaterial:
    # Zelmon et al. (1997), congruent LiNbO3 ordinary index.
    return SellmeierMaterial(
        name="LiNbO3_o_Zelmon1997",
        coeffs=(
            (2.6734, 0.01764),
            (1.2290, 0.05914),
            (12.614, 474.60),
        ),
    )


def linbo3_extraordinary() -> SellmeierMaterial:
    # Zelmon et al. (1997), congruent LiNbO3 extraordinary index.
    return SellmeierMaterial(
        name="LiNbO3_e_Zelmon1997",
        coeffs=(
            (2.9804, 0.02047),
            (0.5981, 0.0666),
            (8.9543, 416.08),
        ),
    )


def gold() -> DrudeMaterial:
    # Rakic et al. (1998) simple Drude parameters.
    return DrudeMaterial(
        name="Gold_Rakic1998",
        eps_inf=9.5,
        coeffs=((_rad_s_to_hz(1.37e16), _rad_s_to_hz(1.05e14)),),
    )


def aluminum() -> DrudeMaterial:
    return DrudeMaterial(
        name="Aluminum_Rakic1998",
        eps_inf=1.0,
        coeffs=((_rad_s_to_hz(2.24e16), _rad_s_to_hz(1.22e14)),),
    )


def copper() -> DrudeMaterial:
    return DrudeMaterial(
        name="Copper_Rakic1998",
        eps_inf=10.8,
        coeffs=((_rad_s_to_hz(1.39e16), _rad_s_to_hz(1.03e14)),),
    )


def water() -> DebyeMaterial:
    # Single-pole room-temperature water Debye model (microwave / low-THz).
    return DebyeMaterial(
        name="Water_Debye",
        eps_inf=4.9,
        coeffs=((73.0, 8.27e-12),),
    )


def ito() -> DrudeMaterial:
    # Representative conductive-oxide Drude parameters for NIR behavior.
    return DrudeMaterial(
        name="ITO_Drude",
        eps_inf=3.8,
        coeffs=((_rad_s_to_hz(3.2e15), _rad_s_to_hz(8.0e13)),),
    )


# --- Registry builders -----------------------------------------------------------------------


def _bulk_item(name: str, fn, notes: str) -> MaterialItem:
    return MaterialItem(
        name=name,
        default="Default",
        variants={"Default": VariantItem(medium=fn(), notes=notes)},
    )


def vacuum_item() -> MaterialItem:
    return _bulk_item("Vacuum", vacuum, "Reference vacuum material")


def air_item() -> MaterialItem:
    return _bulk_item("Air", air, "Dry air approximation near 1.55 um")


def sio2_item() -> MaterialItem:
    return _bulk_item("Silicon Dioxide", sio2, "Sellmeier model (Malitson 1965)")


def si3n4_item() -> MaterialItem:
    return _bulk_item("Silicon Nitride", si3n4, "Sellmeier model (Luke 2015)")


def silicon_item() -> MaterialItem:
    return _bulk_item("Silicon", silicon, "Lorentz model (near-IR compact fit)")


def linbo3_item() -> MaterialItemUniaxial:
    return MaterialItemUniaxial(
        name="Lithium Niobate",
        default="Default",
        variants={
            "Default": VariantItemUniaxial(
                ordinary=linbo3_ordinary(),
                extraordinary=linbo3_extraordinary(),
                notes="Uniaxial Sellmeier model (Zelmon 1997)",
            )
        },
    )


def gold_item() -> MaterialItem:
    return _bulk_item("Gold", gold, "Drude model (Rakic 1998)")


def aluminum_item() -> MaterialItem:
    return _bulk_item("Aluminum", aluminum, "Drude model (Rakic 1998)")


def copper_item() -> MaterialItem:
    return _bulk_item("Copper", copper, "Drude model (Rakic 1998)")


def water_item() -> MaterialItem:
    return _bulk_item("Water", water, "Debye model (single-pole)")


def ito_item() -> MaterialItem:
    return _bulk_item("Indium Tin Oxide", ito, "Drude model (representative NIR fit)")


def build_material_library() -> MaterialLibrary:
    return MaterialLibrary(
        {
            "Vacuum": vacuum_item(),
            "Air": air_item(),
            "SiO2": sio2_item(),
            "Si3N4": si3n4_item(),
            "Silicon": silicon_item(),
            "LiNbO3": linbo3_item(),
            "Gold": gold_item(),
            "Aluminum": aluminum_item(),
            "Copper": copper_item(),
            "Water": water_item(),
            "ITO": ito_item(),
        }
    )


def export_matlib_to_file(fname: str | Path = "matlib.json") -> None:
    out = {}
    for key, item in material_library.items():
        if isinstance(item, MaterialItemUniaxial):
            out[key] = {
                "name": item.name,
                "default": item.default,
                "variants": {
                    variant_key: {
                        "ordinary": repr(variant_value.ordinary),
                        "extraordinary": repr(variant_value.extraordinary),
                        "data_url": variant_value.data_url,
                    }
                    for variant_key, variant_value in item.variants.items()
                },
            }
            continue

        out[key] = {
            "name": item.name,
            "default": item.default,
            "variants": {
                variant_key: {"data_url": variant_value.data_url}
                for variant_key, variant_value in item.variants.items()
            },
        }

    Path(fname).write_text(json.dumps(out, indent=2, sort_keys=True))


material_library = build_material_library()
material_library_export = ExportData(
    source_repo="https://github.com/QuentinWach/beamz",
    source_commit="curated-materials-dispersive-v1",
    generated_utc="",
    source_module="beamz.design.library",
    material_count=len(material_library),
)


__all__ = [
    "ReferenceData",
    "ExportData",
    "VariantItem",
    "MaterialItem",
    "VariantItem2D",
    "MaterialItem2D",
    "VariantItemUniaxial",
    "MaterialItemUniaxial",
    "MaterialLibrary",
    "vacuum",
    "air",
    "sio2",
    "si3n4",
    "silicon",
    "linbo3_ordinary",
    "linbo3_extraordinary",
    "gold",
    "aluminum",
    "copper",
    "water",
    "ito",
    "build_material_library",
    "material_library_export",
    "material_library",
    "export_matlib_to_file",
]
