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
from beamz.devices.sources.signals import ramped_cosine
from beamz.visual.source_plots import plot_signal

from beamz.optimization.topology import (
    TopologyManager,
    compute_overlap_gradient,
    create_optimization_mask,
)
from beamz.optimization.autodiff import transform_density
from beamz.simulation.boundaries import PML, Boundary
from beamz.simulation.core import Simulation
from beamz.multiphysics.thermal import (
    StaticThermalSolve,
    ThermalParams,
    ThermoPhysics,
    apply_static_thermal,
)

# Import UI helpers
from beamz.visual.helpers import (
    calc_optimal_fdtd_params,
    create_rich_progress,
    display_status,
    get_si_scale_and_label,
)


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
    # Multiphysics
    "ThermalParams": ThermalParams,
    "ThermoPhysics": ThermoPhysics,
    "StaticThermalSolve": StaticThermalSolve,
    "apply_static_thermal": apply_static_thermal,
    # Boundaries
    "Boundary": Boundary,
    "PML": PML,
    # Optimization
    "TopologyManager": TopologyManager,
    "compute_overlap_gradient": compute_overlap_gradient,
    "create_optimization_mask": create_optimization_mask,
    "transform_density": transform_density,
    # UI helpers
    "display_status": display_status,
    "create_rich_progress": create_rich_progress,
    "get_si_scale_and_label": get_si_scale_and_label,
    "calc_optimal_fdtd_params": calc_optimal_fdtd_params,
}

# Update module's dictionary with our exports
globals().update(_exports)

# Define what should be available with "from beamz import *"
__all__ = list(_exports.keys())

# Version information
__version__ = "0.1.20"
