"""Reference metadata for material-library entries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceData:
    doi: str | None = None
    journal: str | None = None
    url: str | None = None
    manufacturer: str | None = None
    datasheet_title: str | None = None


__all__ = ["ReferenceData"]
