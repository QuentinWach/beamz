"""Curated material library with a compact Material-first API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beamz.design.materials import (
    AnisotropicMaterial,
    DebyeMaterial,
    DrudeMaterial,
    LorentzMaterial,
    Material,
    Material2D,
    PEC,
    PMC,
    PECMaterial,
    PMCMaterial,
    PoleResidueMaterial,
    SellmeierMaterial,
)
from beamz.material_library.material_reference import ReferenceData


MaterialModel = (
    Material
    | PoleResidueMaterial
    | SellmeierMaterial
    | DrudeMaterial
    | LorentzMaterial
    | DebyeMaterial
    | Material2D
    | PECMaterial
    | PMCMaterial
)


@dataclass(frozen=True)
class ExportData:
    source_repo: str
    source_commit: str
    generated_utc: str
    source_module: str
    material_count: int


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
            if hasattr(item, "variants"):
                lines.append(
                    f"  - Key: {key}, Name: {item.name}, "
                    f"Default Variant: {item.default}, # Variants: {len(item.variants)}"
                )
            else:
                lines.append(f"  - Key: {key}, Type: {type(item).__name__}")
        return "\n".join(lines)


def _deserialize_complex(val: Any) -> complex:
    if isinstance(val, dict) and val.get("__complex__") is True:
        return complex(val["real"], val["imag"])
    if isinstance(val, list) and len(val) == 2 and all(
        isinstance(x, (int, float)) for x in val
    ):
        return complex(val[0], val[1])
    return complex(val)


def _build_reference_list(data: list[dict[str, Any]] | None) -> list[ReferenceData] | None:
    if not data:
        return None
    allowed = {"doi", "journal", "url", "manufacturer", "datasheet_title"}
    refs = []
    for entry in data:
        filtered = {k: v for k, v in entry.items() if k in allowed}
        refs.append(ReferenceData(**filtered))
    return refs


def _build_medium(payload: dict[str, Any]) -> MaterialModel:
    model = payload["model"]
    params = payload.get("params", {})

    if model == "Material":
        fr = params.get("frequency_range")
        return Material(
            name=params.get("name"),
            permittivity=float(params.get("permittivity", 1.0)),
            permeability=float(params.get("permeability", 1.0)),
            conductivity=float(params.get("conductivity", 0.0)),
            frequency_range=tuple(fr) if fr is not None else None,
        )

    if model == "PoleResidueMaterial":
        poles = []
        for pair in params.get("poles", []):
            poles.append((_deserialize_complex(pair[0]), _deserialize_complex(pair[1])))
        fr = params.get("frequency_range")
        return PoleResidueMaterial(
            name=params.get("name"),
            eps_inf=float(params.get("eps_inf", 1.0)),
            poles=tuple(poles),
            frequency_range=tuple(fr) if fr is not None else None,
        )

    if model == "SellmeierMaterial":
        coeffs = tuple((float(b), float(c)) for b, c in params.get("coeffs", []))
        fr = params.get("frequency_range")
        return SellmeierMaterial(
            name=params.get("name"),
            coeffs=coeffs,
            frequency_range=tuple(fr) if fr is not None else None,
        )

    if model == "DrudeMaterial":
        coeffs = tuple((float(fp), float(delta)) for fp, delta in params.get("coeffs", []))
        fr = params.get("frequency_range")
        return DrudeMaterial(
            name=params.get("name"),
            coeffs=coeffs,
            eps_inf=float(params.get("eps_inf", 1.0)),
            frequency_range=tuple(fr) if fr is not None else None,
        )

    if model == "LorentzMaterial":
        coeffs = tuple(
            (float(de), float(f0), float(delta))
            for de, f0, delta in params.get("coeffs", [])
        )
        fr = params.get("frequency_range")
        return LorentzMaterial(
            name=params.get("name"),
            coeffs=coeffs,
            eps_inf=float(params.get("eps_inf", 1.0)),
            frequency_range=tuple(fr) if fr is not None else None,
        )

    if model == "DebyeMaterial":
        coeffs = tuple((float(de), float(tau)) for de, tau in params.get("coeffs", []))
        fr = params.get("frequency_range")
        return DebyeMaterial(
            name=params.get("name"),
            coeffs=coeffs,
            eps_inf=float(params.get("eps_inf", 1.0)),
            frequency_range=tuple(fr) if fr is not None else None,
        )

    if model == "Material2D":
        fr = params.get("frequency_range")
        return Material2D(
            name=params.get("name"),
            ss=_build_medium(params["ss"]),
            tt=_build_medium(params["tt"]),
            frequency_range=tuple(fr) if fr is not None else None,
        )

    if model == "PECMaterial":
        return PEC

    if model == "PMCMaterial":
        return PMC

    raise ValueError(f"Unsupported model type '{model}'.")


def _load_data() -> tuple[ExportData, list[dict[str, Any]]]:
    try:
        from beamz.material_library.data._generated import (
            MATERIAL_LIBRARY_EXPORT,
            MATERIAL_LIBRARY_ITEMS,
        )

        export = ExportData(**MATERIAL_LIBRARY_EXPORT)
        items = MATERIAL_LIBRARY_ITEMS
        return export, items
    except Exception:
        data_path = Path(__file__).resolve().parent / "data" / "materials.normalized.json"
        payload = json.loads(data_path.read_text())
        export = ExportData(**payload["export"])
        return export, payload["materials"]


def _build_material_library(items: list[dict[str, Any]]) -> MaterialLibrary:
    out = MaterialLibrary()
    for item in items:
        key = item["key"]
        kind = item["kind"]
        variants_raw = item["variants"]

        if kind == "uniaxial":
            variants: dict[str, VariantItemUniaxial] = {}
            for variant_key, variant_value in variants_raw.items():
                variants[variant_key] = VariantItemUniaxial(
                    ordinary=_build_medium(variant_value["ordinary"]),
                    extraordinary=_build_medium(variant_value["extraordinary"]),
                    reference=_build_reference_list(variant_value.get("reference")),
                    data_url=variant_value.get("data_url"),
                    notes=variant_value.get("notes"),
                    frequency_range=tuple(variant_value["frequency_range"])
                    if variant_value.get("frequency_range")
                    else None,
                )
            out[key] = MaterialItemUniaxial(
                name=item["name"],
                variants=variants,
                default=item["default"],
            )
            continue

        if kind == "2d":
            variants_2d: dict[str, VariantItem2D] = {}
            for variant_key, variant_value in variants_raw.items():
                variants_2d[variant_key] = VariantItem2D(
                    medium=_build_medium(variant_value["medium"]),
                    reference=_build_reference_list(variant_value.get("reference")),
                    data_url=variant_value.get("data_url"),
                    notes=variant_value.get("notes"),
                    frequency_range=tuple(variant_value["frequency_range"])
                    if variant_value.get("frequency_range")
                    else None,
                )
            out[key] = MaterialItem2D(
                name=item["name"],
                variants=variants_2d,
                default=item["default"],
            )
            continue

        variants_bulk: dict[str, VariantItem] = {}
        for variant_key, variant_value in variants_raw.items():
            variants_bulk[variant_key] = VariantItem(
                medium=_build_medium(variant_value["medium"]),
                reference=_build_reference_list(variant_value.get("reference")),
                data_url=variant_value.get("data_url"),
                notes=variant_value.get("notes"),
                frequency_range=tuple(variant_value["frequency_range"])
                if variant_value.get("frequency_range")
                else None,
            )
        out[key] = MaterialItem(
            name=item["name"],
            variants=variants_bulk,
            default=item["default"],
        )

    return out


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


material_library_export, _material_items = _load_data()
material_library = _build_material_library(_material_items)


__all__ = [
    "ExportData",
    "VariantItem",
    "MaterialItem",
    "VariantItem2D",
    "MaterialItem2D",
    "VariantItemUniaxial",
    "MaterialItemUniaxial",
    "MaterialLibrary",
    "material_library_export",
    "material_library",
    "export_matlib_to_file",
]
