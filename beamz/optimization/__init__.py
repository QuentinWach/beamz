"""Adjoint-based optimization helpers for BEAMZ."""

from . import adjoint_memmap, topology
from .fabrication import (
    Brush2D,
    GeneratedDesign,
    GeneratorState,
    brush_feasibility_errors,
    circular_brush,
    conditional_generator,
    filtered_reward,
    generator_state,
    is_brush_feasible,
    morphological_opening,
    notched_square_brush,
    straight_through_gradient,
)
from .polygonize import (
    density_to_polygons,
    density_to_shapely_geometry,
    shapely_geometry_to_polygons,
)
from .problems import (
    DifferentiablePortProjector,
    InverseDesignProblem,
    PortSweepResult,
)
from .projections import smoothed_heaviside, subpixel_smoothed_projection
from .topology import TopologySpec, TopologyState
from .trainable import DesignRegion, DifferentiableResult, DifferentiableSimulation

__all__ = [
    "topology",
    "adjoint_memmap",
    "smoothed_heaviside",
    "subpixel_smoothed_projection",
    "density_to_shapely_geometry",
    "shapely_geometry_to_polygons",
    "density_to_polygons",
    "TopologySpec",
    "TopologyState",
    "DesignRegion",
    "DifferentiableResult",
    "DifferentiableSimulation",
    "DifferentiablePortProjector",
    "InverseDesignProblem",
    "PortSweepResult",
    "Brush2D",
    "GeneratedDesign",
    "GeneratorState",
    "circular_brush",
    "notched_square_brush",
    "morphological_opening",
    "brush_feasibility_errors",
    "is_brush_feasible",
    "conditional_generator",
    "filtered_reward",
    "generator_state",
    "straight_through_gradient",
]
