"""Native finite-difference mode solving for BeamZ devices."""

from __future__ import annotations

# Re-export the small public API from package root so users do not need to know
# the internal module split.
from .result import Result
from .solver import solve_grid
from .specs import ModeData, ModeSpec

# BeamZ users configure modes through ModeSpec/ModeData. ``solve_grid`` and its
# Result are the only supported low-level escape hatch; placement contracts and
# numerical implementation models remain internal.
__all__ = [
    "ModeData",
    "ModeSpec",
    "Result",
    "solve_grid",
]
