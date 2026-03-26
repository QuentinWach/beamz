"""Simulation module for BEAMZ."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "RegularGrid",
    "Simulation",
    "PortSpec",
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

_MODULE_EXPORTS = {
    "RegularGrid": ("beamz.design.meshing", "RegularGrid"),
    "Simulation": ("beamz.simulation.core", "Simulation"),
    "PortSpec": ("beamz.simulation.core", "PortSpec"),
    "CompiledRunConfig": ("beamz.simulation.compiled", "CompiledRunConfig"),
    "CompiledSimulation": ("beamz.simulation.compiled", "CompiledSimulation"),
    "EngineState": ("beamz.simulation.compiled", "EngineState"),
    "MonitorState": ("beamz.simulation.compiled", "MonitorState"),
    "RunState": ("beamz.simulation.compiled", "RunState"),
    "compile_simulation": ("beamz.simulation.compiled", "compile_simulation"),
    "ThermalConfig": ("beamz.simulation.thermal", "ThermalConfig"),
    "ThermalCoupling": ("beamz.simulation.thermal", "ThermalCoupling"),
    "StaticThermalConfig": ("beamz.simulation.thermal", "StaticThermalConfig"),
    "StaticThermalResult": ("beamz.simulation.thermal", "StaticThermalResult"),
    "ThermalSource": ("beamz.simulation.thermal", "ThermalSource"),
    "ThermalSink": ("beamz.simulation.thermal", "ThermalSink"),
    "ConvectionBC": ("beamz.simulation.thermal", "ConvectionBC"),
    "ThermalBoundaryProfile": ("beamz.simulation.thermal", "ThermalBoundaryProfile"),
    "ThermalScenario": ("beamz.simulation.thermal", "ThermalScenario"),
    "MZITuningResult": ("beamz.simulation.thermal", "MZITuningResult"),
    "StaticThermalSolver": ("beamz.simulation.thermal", "StaticThermalSolver"),
    "solve_thermal": ("beamz.simulation.thermal", "solve_thermal"),
    "solve_static_thermal": ("beamz.simulation.thermal", "solve_static_thermal"),
    "sweep_mzi_heater": ("beamz.simulation.thermal", "sweep_mzi_heater"),
}


def __getattr__(name: str):
    if name not in _MODULE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _MODULE_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
