"""Native finite-difference mode solving for BeamZ devices."""

from __future__ import annotations

# Re-export the small public API from package root so users do not need to know
# the internal module split.
from .constants import C_0, EPSILON_0
from .discrete import DiscreteMode, ModePlaneSpec, solve_beamz_mode
from .models import BoundarySpec, Grid, Materials, PmlSpec, Spec
from .result import Result, overlap
from .solver import solve_grid, solve_modes, solve_slice
from .specs import ModeData, ModeSpec
from .sweep import Sweep, track_modes_by_overlap

# BeamZ users configure modes through ModeSpec/ModeData. ``solve_grid`` and its
# Result are the only supported low-level escape hatch; placement contracts and
# numerical implementation models remain internal.
__all__ = [
    "ModeData",
    "ModeSpec",
    "Result",
    "solve_grid",
]
