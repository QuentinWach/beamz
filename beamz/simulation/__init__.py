"""Simulation module for BEAMZ."""

from beamz.design.meshing import GridSpec
from beamz.devices.boundaries import PEC, PML, Absorber
from beamz.devices.modes.specs import ModeSpec
from beamz.devices.ports import Port
from beamz.devices.sources.time import GaussianPulse
from beamz.simulation.api import Simulation
from beamz.simulation.model import SimulationState
from beamz.simulation.results import MonitorResults, SimulationResults, SimulationRun

# Keep this facade deliberately small: importing ``beamz.simulation`` should expose
# stable user concepts without pulling private compiler implementation names into API docs.
inf = float("inf")

__all__ = [
    "Simulation",
    "Port",
    "MonitorResults",
    "SimulationResults",
    "SimulationRun",
    "SimulationState",
    "GridSpec",
    "GaussianPulse",
    "ModeSpec",
    "inf",
    "PML",
    "PEC",
    "Absorber",
]
