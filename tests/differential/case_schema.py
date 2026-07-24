"""Strict loader for solver-neutral differential-validation cases."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

SCHEMA_VERSION = "beamz.differential/v1"
ORACLE_KINDS = ("analytical", "invariant", "solver_consensus", "regression")
ORACLE_PRIORITY = {kind: index for index, kind in enumerate(ORACLE_KINDS)}


@dataclass(frozen=True, slots=True)
class ObservableSpec:
    """One portable scalar result requested from every case adapter."""

    name: str
    unit: str
    tolerance: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObservableSpec:
        required = {"name", "unit", "tolerance"}
        if set(payload) != required:
            raise ValueError(
                f"observable keys must be {sorted(required)}, got {sorted(payload)}"
            )
        values = {key: str(payload[key]).strip() for key in required}
        if not all(values.values()):
            raise ValueError("observable name, unit, and tolerance must be non-empty")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class DifferentialCase:
    """A physical case that can be lowered by independent solver adapters."""

    name: str
    description: str
    dimension: int
    wavelength_m: float
    resolution_m: float
    oracle_kind: str
    oracle_reference: str
    materials: Mapping[str, float]
    geometry: Mapping[str, Any]
    observables: tuple[ObservableSpec, ...]
    adapters: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.description:
            raise ValueError("case name and description must be non-empty")
        if self.dimension not in {2, 3}:
            raise ValueError("case dimension must be 2 or 3")
        if self.wavelength_m <= 0.0 or self.resolution_m <= 0.0:
            raise ValueError("wavelength_m and resolution_m must be positive")
        if self.oracle_kind not in ORACLE_PRIORITY:
            raise ValueError(f"unsupported oracle kind {self.oracle_kind!r}")
        if not self.oracle_reference:
            raise ValueError("oracle reference must be non-empty")
        if not self.observables:
            raise ValueError("case must request at least one observable")
        if len({item.name for item in self.observables}) != len(self.observables):
            raise ValueError("observable names must be unique within a case")
        if not self.adapters or "beamz" not in self.adapters:
            raise ValueError("case adapters must include 'beamz'")
        if len(set(self.adapters)) != len(self.adapters):
            raise ValueError("case adapter names must be unique")
        material_values = {
            str(key): float(value) for key, value in self.materials.items()
        }
        if not material_values or any(
            not math.isfinite(value) or value <= 0.0
            for value in material_values.values()
        ):
            raise ValueError("material refractive indices must be positive and finite")
        geometry = dict(self.geometry)
        try:
            json.dumps(geometry, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("geometry must contain finite JSON values") from error
        object.__setattr__(self, "materials", MappingProxyType(material_values))
        object.__setattr__(self, "geometry", MappingProxyType(geometry))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DifferentialCase:
        expected = {
            "schema_version",
            "name",
            "description",
            "dimension",
            "wavelength_m",
            "resolution_m",
            "oracle",
            "materials",
            "geometry",
            "observables",
            "adapters",
        }
        if set(payload) != expected:
            raise ValueError(
                f"case keys must be {sorted(expected)}, got {sorted(payload)}"
            )
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported differential schema {payload['schema_version']!r}"
            )
        oracle = payload["oracle"]
        if not isinstance(oracle, Mapping) or set(oracle) != {"kind", "reference"}:
            raise ValueError("oracle must contain exactly kind and reference")
        observables = tuple(
            ObservableSpec.from_dict(item) for item in payload["observables"]
        )
        return cls(
            name=str(payload["name"]).strip(),
            description=str(payload["description"]).strip(),
            dimension=int(payload["dimension"]),
            wavelength_m=float(payload["wavelength_m"]),
            resolution_m=float(payload["resolution_m"]),
            oracle_kind=str(oracle["kind"]),
            oracle_reference=str(oracle["reference"]).strip(),
            materials=payload["materials"],
            geometry=payload["geometry"],
            observables=observables,
            adapters=tuple(str(item).strip() for item in payload["adapters"]),
        )


def load_case(path: Path) -> DifferentialCase:
    """Load one UTF-8 JSON case and reject non-object roots."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"differential case {path} must contain a JSON object")
    return DifferentialCase.from_dict(payload)
