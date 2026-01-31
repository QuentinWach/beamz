"""
BeamZ - A Python package for electromagnetic simulations.
"""

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
    μm,
)

# Import design-related classes and functions
from beamz.design.core import Design
from beamz.design.materials import CustomMaterial, Material

# Import simulation-related classes and functions
from beamz.design.meshing import RegularGrid
from beamz.design.structures import (
    Circle,
    CircularBend,
    Polygon,
    Rectangle,
    Ring,
    Sphere,
    Taper,
)
from beamz.devices.monitors import Monitor
from beamz.devices.sources import GaussianSource, ModeSource
from beamz.devices.sources.mode import solve_modes
from beamz.devices.sources.signals import plot_signal, ramped_cosine

# from beamz.optimization.optimizers import Optimizer  # TODO: Re-enable when optimizers module is created
from beamz.optimization.topology import compute_overlap_gradient
from beamz.simulation.boundaries import ABC, PML, Boundary, PeriodicBoundary
from beamz.simulation.core import Simulation

# Import UI helpers
from beamz.visual.helpers import (
    calc_optimal_fdtd_params,
    code_preview,
    create_rich_progress,
    display_header,
    display_optimization_progress,
    display_parameters,
    display_results,
    display_simulation_status,
    display_status,
    display_time_elapsed,
    get_si_scale_and_label,
    tree_view,
)

# Import optimization-related classes
# (Currently empty, to be filled as the module grows)


# Prepare a dictionary of all our exports
_exports = {
    # Constants
    "LIGHT_SPEED": LIGHT_SPEED,
    "VAC_PERMITTIVITY": VAC_PERMITTIVITY,
    "VAC_PERMEABILITY": VAC_PERMEABILITY,
    "EPS_0": EPS_0,
    "MU_0": MU_0,
    "um": um,
    "nm": nm,
    "µm": µm,
    "μm": μm,
    # Materials
    "Material": Material,
    "CustomMaterial": CustomMaterial,
    # Structures
    "Design": Design,
    "Rectangle": Rectangle,
    "Circle": Circle,
    "Ring": Ring,
    "CircularBend": CircularBend,
    "Polygon": Polygon,
    "Taper": Taper,
    "Sphere": Sphere,
    # Sources
    "ModeSource": ModeSource,
    "GaussianSource": GaussianSource,
    # Monitors
    "Monitor": Monitor,
    # Signals
    "ramped_cosine": ramped_cosine,
    "plot_signal": plot_signal,
    # Mode calculations
    "solve_modes": solve_modes,
    # Simulation
    "RegularGrid": RegularGrid,
    "Simulation": Simulation,
    # Boundaries
    "Boundary": Boundary,
    "PML": PML,
    "ABC": ABC,
    "PeriodicBoundary": PeriodicBoundary,
    # Optimization
    # 'Optimizer': Optimizer,  # TODO: Re-enable when optimizers module is created
    "compute_overlap_gradient": compute_overlap_gradient,
    # UI helpers
    "display_header": display_header,
    "display_status": display_status,
    "create_rich_progress": create_rich_progress,
    "display_parameters": display_parameters,
    "display_results": display_results,
    "display_simulation_status": display_simulation_status,
    "display_optimization_progress": display_optimization_progress,
    "display_time_elapsed": display_time_elapsed,
    "tree_view": tree_view,
    "code_preview": code_preview,
    "get_si_scale_and_label": get_si_scale_and_label,
    "calc_optimal_fdtd_params": calc_optimal_fdtd_params,
}

# Update module's dictionary with our exports
globals().update(_exports)

# Define what should be available with "from beamz import *"
__all__ = list(_exports.keys())

# Version information
__version__ = "0.1.20"
