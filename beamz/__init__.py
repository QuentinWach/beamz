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
<<<<<<< HEAD
from beamz.design.materials import (
    Material, CustomMaterial,
    # Advanced dispersive material models
    SellmeierMaterial, DrudeMaterial, LorentzMaterial,
    DebyeMaterial, PoleResidueMaterial, DrudeLorentzMaterial,
    # Predefined dispersive materials
    SiO2_Sellmeier, BK7_Sellmeier,
    Gold_Drude, Silver_Drude, Aluminum_Drude, Copper_Drude,
    Water_Debye, Gold_DrudeLorentz,
)
from beamz.design.library import (
    # Material library - Vacuum & Gases
    Vacuum, Air,
    # Dielectrics & Glasses
    SiO2, FusedSilica, Silica, Si3N4, SiliconNitride, SiN,
    BK7, SodaLimeGlass, Sapphire, Al2O3, Diamond,
    # Semiconductors
    Silicon, Si, Germanium, Ge, GaAs, InP, LiNbO3, LithiumNiobate,
    # Metals
    Gold, Au, Silver, Ag, Copper, Cu, Aluminum, Al, Chromium, Cr, Titanium, Ti,
    # Polymers
    PMMA, SU8, Polystyrene, PDMS, HSQ,
    # Liquids
    Water, H2O, Ethanol, IPA, Glycerol, ImmersionOil,
    # Specialty materials
    ITO, TiO2, HfO2, Ta2O5, ZnO, AlN, MgF2, CaF2, BaF2, ZnSe, ZnS,
    # Special materials
    PEC, PMC,
    # Utility functions
    list_materials, get_material, material_info,
)
from beamz.design.structures import (
    Rectangle, Circle, Ring, 
    CircularBend, Polygon, Taper, Sphere
)
from beamz.devices.sources import ModeSource, GaussianSource
from beamz.devices.monitors import Monitor
from beamz.devices.sources.signals import ramped_cosine, plot_signal
from beamz.devices.sources.mode import solve_modes
=======
from beamz.design.materials import CustomMaterial, Material
>>>>>>> main

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
from beamz.optimization.topology import (
    TopologyManager,
    compute_overlap_gradient,
    create_optimization_mask,
)
from beamz.optimization.autodiff import transform_density
from beamz.simulation.boundaries import ABC, PML, Boundary, PeriodicBoundary
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
<<<<<<< HEAD
    'LIGHT_SPEED': LIGHT_SPEED,
    'VAC_PERMITTIVITY': VAC_PERMITTIVITY,
    'VAC_PERMEABILITY': VAC_PERMEABILITY,
    'EPS_0': EPS_0,
    'MU_0': MU_0,
    'um': um,
    'nm': nm,
    'µm': µm,
    'μm': μm,
    
    # Materials - Basic
    'Material': Material,
    'CustomMaterial': CustomMaterial,

    # Materials - Advanced dispersive models
    'SellmeierMaterial': SellmeierMaterial,
    'DrudeMaterial': DrudeMaterial,
    'LorentzMaterial': LorentzMaterial,
    'DebyeMaterial': DebyeMaterial,
    'PoleResidueMaterial': PoleResidueMaterial,
    'DrudeLorentzMaterial': DrudeLorentzMaterial,

    # Materials - Predefined dispersive
    'SiO2_Sellmeier': SiO2_Sellmeier,
    'BK7_Sellmeier': BK7_Sellmeier,
    'Gold_Drude': Gold_Drude,
    'Silver_Drude': Silver_Drude,
    'Aluminum_Drude': Aluminum_Drude,
    'Copper_Drude': Copper_Drude,
    'Water_Debye': Water_Debye,
    'Gold_DrudeLorentz': Gold_DrudeLorentz,

    # Materials - Library (Vacuum & Gases)
    'Vacuum': Vacuum,
    'Air': Air,

    # Materials - Dielectrics & Glasses
    'SiO2': SiO2,
    'FusedSilica': FusedSilica,
    'Silica': Silica,
    'Si3N4': Si3N4,
    'SiliconNitride': SiliconNitride,
    'SiN': SiN,
    'BK7': BK7,
    'SodaLimeGlass': SodaLimeGlass,
    'Sapphire': Sapphire,
    'Al2O3': Al2O3,
    'Diamond': Diamond,

    # Materials - Semiconductors
    'Silicon': Silicon,
    'Si': Si,
    'Germanium': Germanium,
    'Ge': Ge,
    'GaAs': GaAs,
    'InP': InP,
    'LiNbO3': LiNbO3,
    'LithiumNiobate': LithiumNiobate,

    # Materials - Metals
    'Gold': Gold,
    'Au': Au,
    'Silver': Silver,
    'Ag': Ag,
    'Copper': Copper,
    'Cu': Cu,
    'Aluminum': Aluminum,
    'Al': Al,
    'Chromium': Chromium,
    'Cr': Cr,
    'Titanium': Titanium,
    'Ti': Ti,

    # Materials - Polymers
    'PMMA': PMMA,
    'SU8': SU8,
    'Polystyrene': Polystyrene,
    'PDMS': PDMS,
    'HSQ': HSQ,

    # Materials - Liquids
    'Water': Water,
    'H2O': H2O,
    'Ethanol': Ethanol,
    'IPA': IPA,
    'Glycerol': Glycerol,
    'ImmersionOil': ImmersionOil,

    # Materials - Specialty
    'ITO': ITO,
    'TiO2': TiO2,
    'HfO2': HfO2,
    'Ta2O5': Ta2O5,
    'ZnO': ZnO,
    'AlN': AlN,
    'MgF2': MgF2,
    'CaF2': CaF2,
    'BaF2': BaF2,
    'ZnSe': ZnSe,
    'ZnS': ZnS,

    # Materials - Special
    'PEC': PEC,
    'PMC': PMC,

    # Materials - Utility functions
    'list_materials': list_materials,
    'get_material': get_material,
    'material_info': material_info,
    
=======
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
>>>>>>> main
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
    "ABC": ABC,
    "PeriodicBoundary": PeriodicBoundary,
    # Optimization
    # 'Optimizer': Optimizer,  # TODO: Re-enable when optimizers module is created
    "TopologyManager": TopologyManager,
    "compute_overlap_gradient": compute_overlap_gradient,
    "create_optimization_mask": create_optimization_mask,
    "transform_density": transform_density,
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
