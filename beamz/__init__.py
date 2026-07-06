"""
BeamZ - A Python package for electromagnetic simulations.
"""

import beamz.design as design

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
from beamz.data import colocate_dataset, field_intensity, poynting_vector

# Import design-related classes and functions
from beamz.design.core import Design
from beamz.design.materials import CustomMaterial, Material, Medium

# Import simulation-related classes and functions
from beamz.design.meshing import RegularGrid
from beamz.design.structures import (
    Box,
    Circle,
    CircularBend,
    Polygon,
    Rectangle,
    Ring,
    Sphere,
    Structure,
    Taper,
)
from beamz.devices.monitors import FieldMonitor, FluxMonitor, ModeMonitor, Monitor
from beamz.devices.ports import Port
from beamz.devices.sources import (
    GaussianBeamSource,
    GaussianSource,
    ModeData,
    ModeSolver,
    ModeSource,
)
from beamz.devices.sources.mode import solve_modes
from beamz.devices.sources.signals import plot_signal, ramped_cosine, signal_plot_data
from beamz.optimization.autodiff import transform_density
from beamz.optimization.topology import (
    TopologyManager,
    compute_overlap_gradient,
    create_optimization_mask,
)
from beamz.simulation.boundaries import PEC, PML, AbsorbingLayer, Boundary
from beamz.simulation.compiled import (
    CompiledRunConfig,
    CompiledSimulation,
    EngineState,
    MonitorState,
    RunState,
    ShardingConfig,
    compile_simulation,
)
from beamz.simulation.core import (
    MonitorResults,
    PortSpec,
    Simulation,
    SimulationResults,
)
from beamz.simulation.specs import BoundarySpec, GaussianPulse, GridSpec, ModeSpec, inf

# Import UI helpers
from beamz.visual.helpers import (
    calc_optimal_fdtd_params,
    create_plain_progress,
    display_status,
    dxdt,
    get_si_scale_and_label,
)
from beamz.visual.mpl import (
    mode_field_component_pairs,
    plot_mode_fields,
    plot_simulation_field,
    plot_simulation_permittivity,
    plot_source_signal,
    plot_source_spectrum,
    plot_tidy3d_cross_sections,
    plot_tidy3d_dft_field,
    plot_tidy3d_field_frame,
    plot_tidy3d_mode_components,
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
    "Medium": Medium,
    "CustomMaterial": CustomMaterial,
    # Structures
    "design": design,
    "Design": Design,
    "Box": Box,
    "Structure": Structure,
    "Rectangle": Rectangle,
    "Circle": Circle,
    "Ring": Ring,
    "CircularBend": CircularBend,
    "Polygon": Polygon,
    "Taper": Taper,
    "Sphere": Sphere,
    # Sources
    "ModeSource": ModeSource,
    "ModeSolver": ModeSolver,
    "ModeData": ModeData,
    "GaussianSource": GaussianSource,
    "GaussianBeamSource": GaussianBeamSource,
    # Monitors
    "Monitor": Monitor,
    "FieldMonitor": FieldMonitor,
    "FluxMonitor": FluxMonitor,
    "ModeMonitor": ModeMonitor,
    "Port": Port,
    # Signals
    "ramped_cosine": ramped_cosine,
    "signal_plot_data": signal_plot_data,
    "plot_signal": plot_signal,
    "plot_source_signal": plot_source_signal,
    "plot_source_spectrum": plot_source_spectrum,
    # Mode calculations
    "solve_modes": solve_modes,
    # Simulation
    "RegularGrid": RegularGrid,
    "Simulation": Simulation,
    "GridSpec": GridSpec,
    "GaussianPulse": GaussianPulse,
    "ModeSpec": ModeSpec,
    "BoundarySpec": BoundarySpec,
    "inf": inf,
    "PortSpec": PortSpec,
    "MonitorResults": MonitorResults,
    "SimulationResults": SimulationResults,
    "colocate_dataset": colocate_dataset,
    "field_intensity": field_intensity,
    "poynting_vector": poynting_vector,
    "plot_simulation_field": plot_simulation_field,
    "plot_simulation_permittivity": plot_simulation_permittivity,
    "mode_field_component_pairs": mode_field_component_pairs,
    "plot_mode_fields": plot_mode_fields,
    "plot_tidy3d_cross_sections": plot_tidy3d_cross_sections,
    "plot_tidy3d_dft_field": plot_tidy3d_dft_field,
    "plot_tidy3d_field_frame": plot_tidy3d_field_frame,
    "plot_tidy3d_mode_components": plot_tidy3d_mode_components,
    "CompiledRunConfig": CompiledRunConfig,
    "CompiledSimulation": CompiledSimulation,
    "EngineState": EngineState,
    "MonitorState": MonitorState,
    "RunState": RunState,
    "ShardingConfig": ShardingConfig,
    "compile_simulation": compile_simulation,
    # Boundaries
    "Boundary": Boundary,
    "AbsorbingLayer": AbsorbingLayer,
    "PML": PML,
    "PEC": PEC,
    # Optimization
    "TopologyManager": TopologyManager,
    "compute_overlap_gradient": compute_overlap_gradient,
    "create_optimization_mask": create_optimization_mask,
    "transform_density": transform_density,
    # UI helpers
    "display_status": display_status,
    "create_plain_progress": create_plain_progress,
    "get_si_scale_and_label": get_si_scale_and_label,
    "calc_optimal_fdtd_params": calc_optimal_fdtd_params,
    "dxdt": dxdt,
}

# Update module's dictionary with our exports
globals().update(_exports)

# Define what should be available with "from beamz import *"
__all__ = list(_exports.keys())

# Version information
__version__ = "0.4.3"
