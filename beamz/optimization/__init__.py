"""Adjoint-based optimization helpers for BEAMZ."""

from . import adjoint_memmap, topology
from .polygonize import (
    density_to_polygons,
    density_to_shapely_geometry,
    shapely_geometry_to_polygons,
)
from .projections import smoothed_heaviside, subpixel_smoothed_projection

__all__ = [
    "topology",
    "adjoint_memmap",
    "smoothed_heaviside",
    "subpixel_smoothed_projection",
    "density_to_shapely_geometry",
    "shapely_geometry_to_polygons",
    "density_to_polygons",
]
