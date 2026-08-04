"""
BeamZ - A Python package for electromagnetic simulations.
"""

import beamz.design as design
import beamz.optimization as optimization
from beamz._helpers import (
    calc_optimal_fdtd_params,
    create_plain_progress,
    display_status,
    dxdt,
    get_si_scale_and_label,
)
from beamz.analysis import SParameterResult

# Import constants from the const module
from beamz.const import (
    EPS_0,
    LIGHT_SPEED,
    MU_0,
    VAC_PERMEABILITY,
    VAC_PERMITTIVITY,
    nm,
    um,
    µm,
    μm,  # noqa: F811 - public alias using Greek mu codepoint
)

# Import design-related classes and functions
from beamz.design.core import Design
from beamz.design.grid import AxisGridQuality, Grid, GridQualityReport, RectilinearGrid

# Import simulation-related classes and functions
from beamz.design.grid_spec import GridSpec, MeshOverride
from beamz.design.materials import Material
from beamz.design.structures import (
    Box,
    Circle,
    CircularBend,
    Polygon,
    Rectangle,
    Ring,
    Sphere,
    Taper,
)
from beamz.devices.boundaries import PEC, PML, Absorber
from beamz.devices.monitors import FieldMonitor, FieldRecorder, FluxMonitor, ModeMonitor
from beamz.devices.ports import Port
from beamz.devices.sources import (
    CustomSource,
    GaussianBeamSource,
    GaussianPulse,
    GaussianSource,
    ModeData,
    ModeSource,
    ModeSpec,
    SampledSignal,
)
from beamz.devices.sources.time import ramped_cosine
from beamz.simulation.api import Simulation
from beamz.simulation.model import SimulationState
from beamz.simulation.results import MonitorResults, SimulationResults, SimulationRun

inf = float("inf")

# Python normalizes identifiers, so preserve the micro-sign spelling as an explicit
# module key alongside the Greek-mu import alias.
globals()["µm"] = μm

# Define what should be available with "from beamz import *"
__all__ = [
    "LIGHT_SPEED",
    "VAC_PERMITTIVITY",
    "VAC_PERMEABILITY",
    "EPS_0",
    "MU_0",
    "um",
    "nm",
    "µm",  # pyright: ignore[reportUnsupportedDunderAll] -- Unicode public alias
    "μm",
    "Material",
    "design",
    "optimization",
    "Design",
    "Box",
    "Rectangle",
    "Circle",
    "Ring",
    "CircularBend",
    "Polygon",
    "Taper",
    "Sphere",
    "ModeSource",
    "ModeData",
    "GaussianSource",
    "GaussianBeamSource",
    "CustomSource",
    "FieldMonitor",
    "FieldRecorder",
    "FluxMonitor",
    "ModeMonitor",
    "Port",
    "ramped_cosine",
    "Simulation",
    "GridSpec",
    "MeshOverride",
    "Grid",
    "RectilinearGrid",
    "AxisGridQuality",
    "GridQualityReport",
    "GaussianPulse",
    "SampledSignal",
    "ModeSpec",
    "inf",
    "MonitorResults",
    "SimulationResults",
    "SimulationRun",
    "SimulationState",
    "PML",
    "PEC",
    "Absorber",
    "display_status",
    "create_plain_progress",
    "get_si_scale_and_label",
    "calc_optimal_fdtd_params",
    "dxdt",
    "SParameterResult",
]


# Version information
__version__ = "0.4.3"
