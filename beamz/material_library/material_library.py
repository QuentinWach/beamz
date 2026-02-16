"""Material library structures for Beamz."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beamz.components.medium import (
    AnisotropicMedium,
    Medium,
    Medium2D,
    PEC,
    PMC,
    PECMedium,
    PMCMedium,
    PoleResidue,
)
from beamz.material_library.material_reference import ReferenceData
from beamz.material_library.parametric_materials import Graphene


@dataclass(frozen=True)
class ExportData:
    source_repo: str
    source_commit: str
    generated_utc: str
    source_module: str
    material_count: int


@dataclass
class VariantItem:
    medium: PoleResidue | Medium | Medium2D | PECMedium | PMCMedium
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

    def __getitem__(self, variant_name: str):
        return self.variants[variant_name].medium

    @property
    def medium(self):
        return self.variants[self.default].medium


@dataclass
class VariantItem2D:
    medium: Medium2D
    reference: list[ReferenceData] | None = None
    data_url: str | None = None
    notes: str | None = None
    frequency_range: tuple[float, float] | None = None


@dataclass
class MaterialItem2D(MaterialItem):
    variants: dict[str, VariantItem2D]


@dataclass
class VariantItemUniaxial:
    ordinary: PoleResidue
    extraordinary: PoleResidue
    reference: list[ReferenceData] | None = None
    data_url: str | None = None
    notes: str | None = None
    frequency_range: tuple[float, float] | None = None

    def medium(self, optical_axis: int | str = 2) -> AnisotropicMedium:
        axes = {0: "x", 1: "y", 2: "z", "x": "x", "y": "y", "z": "z"}
        axis = axes.get(optical_axis)
        if axis is None:
            raise ValueError("optical_axis must be one of 0,1,2,'x','y','z'.")
        xx = self.extraordinary if axis == "x" else self.ordinary
        yy = self.extraordinary if axis == "y" else self.ordinary
        zz = self.extraordinary if axis == "z" else self.ordinary
        return AnisotropicMedium(xx=xx, yy=yy, zz=zz)


@dataclass
class MaterialItemUniaxial(MaterialItem):
    variants: dict[str, VariantItemUniaxial]

    def medium(self, optical_axis: int | str = 2) -> AnisotropicMedium:
        return self.variants[self.default].medium(optical_axis)


class MaterialLibrary(dict):
    def __str__(self) -> str:
        lines = ["Material Library Summary:"]
        for key, item in self.items():
            if isinstance(item, type):
                lines.append(f"  - Key: {key}")
            elif hasattr(item, "variants"):
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


def _build_pole_residue(params: dict[str, Any]) -> PoleResidue:
    poles = []
    for pair in params.get("poles", []):
        a = _deserialize_complex(pair[0])
        c = _deserialize_complex(pair[1])
        poles.append((a, c))
    fr = params.get("frequency_range")
    fr_t = tuple(fr) if fr is not None else None
    return PoleResidue(
        name=params.get("name"),
        eps_inf=float(params.get("eps_inf", 1.0)),
        poles=tuple(poles),
        frequency_range=fr_t,
    )


def _build_medium(payload: dict[str, Any]):
    model = payload["model"]
    params = payload.get("params", {})
    if model == "PoleResidue":
        return _build_pole_residue(params)
    if model == "Medium":
        fr = params.get("frequency_range")
        return Medium(
            name=params.get("name"),
            permittivity=float(params.get("permittivity", 1.0)),
            permeability=float(params.get("permeability", 1.0)),
            conductivity=float(params.get("conductivity", 0.0)),
            frequency_range=tuple(fr) if fr is not None else None,
        )
    if model == "Medium2D":
        fr = params.get("frequency_range")
        return Medium2D(
            name=params.get("name"),
            ss=_build_medium(params["ss"]),
            tt=_build_medium(params["tt"]),
            frequency_range=tuple(fr) if fr is not None else None,
        )
    if model == "PECMedium":
        return PEC
    if model == "PMCMedium":
        return PMC
    raise ValueError(f"Unsupported model type '{model}'.")


def _load_data() -> tuple[ExportData, list[dict[str, Any]]]:
    try:
        from beamz.material_library.data._generated import MATERIAL_LIBRARY_EXPORT, MATERIAL_LIBRARY_ITEMS

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

        if kind == "parametric_class":
            out[key] = Graphene
            continue

        variants_raw = item["variants"]

        if kind == "uniaxial":
            variants: dict[str, VariantItemUniaxial] = {}
            for vk, vv in variants_raw.items():
                ordinary = _build_medium(vv["ordinary"])
                extraordinary = _build_medium(vv["extraordinary"])
                variants[vk] = VariantItemUniaxial(
                    ordinary=ordinary,
                    extraordinary=extraordinary,
                    reference=_build_reference_list(vv.get("reference")),
                    data_url=vv.get("data_url"),
                    notes=vv.get("notes"),
                    frequency_range=tuple(vv["frequency_range"])
                    if vv.get("frequency_range")
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
            for vk, vv in variants_raw.items():
                medium = _build_medium(vv["medium"])
                variants_2d[vk] = VariantItem2D(
                    medium=medium,
                    reference=_build_reference_list(vv.get("reference")),
                    data_url=vv.get("data_url"),
                    notes=vv.get("notes"),
                    frequency_range=tuple(vv["frequency_range"])
                    if vv.get("frequency_range")
                    else None,
                )
            out[key] = MaterialItem2D(
                name=item["name"],
                variants=variants_2d,
                default=item["default"],
            )
            continue

        variants_bulk: dict[str, VariantItem] = {}
        for vk, vv in variants_raw.items():
            medium = _build_medium(vv["medium"])
            variants_bulk[vk] = VariantItem(
                medium=medium,
                reference=_build_reference_list(vv.get("reference")),
                data_url=vv.get("data_url"),
                notes=vv.get("notes"),
                frequency_range=tuple(vv["frequency_range"])
                if vv.get("frequency_range")
                else None,
            )
        out[key] = MaterialItem(
            name=item["name"],
            variants=variants_bulk,
            default=item["default"],
        )

    # Add symbolic placeholders explicitly to guarantee availability.
    out.setdefault(
        "PEC",
        MaterialItem(
            name="Perfect Electric Conductor",
            variants={"default": VariantItem(medium=PEC)},
            default="default",
        ),
    )
    out.setdefault(
        "PMC",
        MaterialItem(
            name="Perfect Magnetic Conductor",
            variants={"default": VariantItem(medium=PMC)},
            default="default",
        ),
    )

    return out


def export_matlib_to_file(fname: str | Path = "matlib.json") -> None:
    """Export simplified material library to JSON file."""
    out = {}
    for key, item in material_library.items():
        if isinstance(item, type):
            continue
        if isinstance(item, MaterialItemUniaxial):
            out[key] = {
                "name": item.name,
                "default": item.default,
                "variants": {
                    vk: {
                        "ordinary": vv.ordinary,
                        "extraordinary": vv.extraordinary,
                        "data_url": vv.data_url,
                    }
                    for vk, vv in item.variants.items()
                },
            }
            continue
        out[key] = {
            "name": item.name,
            "default": item.default,
            "variants": {vk: {"data_url": vv.data_url} for vk, vv in item.variants.items()},
        }

    Path(fname).write_text(json.dumps(out, default=str, indent=2, sort_keys=True))


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
