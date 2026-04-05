from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DesignState:
    sources: list = field(default_factory=list)
    monitors: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)
    layers: dict = field(default_factory=dict)
    grid: object = None
    grid_resolution: float | None = None
    grid_request_signature: dict | None = None
