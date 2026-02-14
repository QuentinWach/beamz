"""
Simulation module for BEAMZ - Contains FDTD simulation and field operations.
"""

from beamz.design.meshing import RegularGrid
from beamz.simulation.core import Simulation
from beamz.simulation.thermal import (
    StaticThermalResult,
    StaticThermalSolver,
    ThermalConfig,
    ThermalCoupling,
    solve_static_thermal,
)

__all__ = [
    "RegularGrid",
    "Simulation",
    "ThermalConfig",
    "ThermalCoupling",
    "StaticThermalResult",
    "StaticThermalSolver",
    "solve_static_thermal",
]
