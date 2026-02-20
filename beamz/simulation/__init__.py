"""
Simulation module for BEAMZ - Contains FDTD simulation and field operations.
"""

from beamz.design.meshing import RegularGrid
from beamz.simulation.compiled import (
    CompiledRunConfig,
    CompiledSimulation,
    EngineState,
    MonitorState,
    RunState,
    compile_simulation,
)
from beamz.simulation.core import Simulation
from beamz.simulation.thermal import (
    ConvectionBC,
    MZITuningResult,
    StaticThermalConfig,
    StaticThermalResult,
    StaticThermalSolver,
    ThermalBoundaryProfile,
    ThermalConfig,
    ThermalCoupling,
    ThermalScenario,
    ThermalSink,
    ThermalSource,
    solve_static_thermal,
    solve_thermal,
    sweep_mzi_heater,
)

__all__ = [
    "RegularGrid",
    "Simulation",
    "CompiledRunConfig",
    "CompiledSimulation",
    "EngineState",
    "MonitorState",
    "RunState",
    "compile_simulation",
    "ThermalConfig",
    "ThermalCoupling",
    "StaticThermalConfig",
    "StaticThermalResult",
    "ThermalSource",
    "ThermalSink",
    "ConvectionBC",
    "ThermalBoundaryProfile",
    "ThermalScenario",
    "MZITuningResult",
    "StaticThermalSolver",
    "solve_thermal",
    "solve_static_thermal",
    "sweep_mzi_heater",
]
