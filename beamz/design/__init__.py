"""
Design module for BEAMZ - Contains components for designing photonic structures.
"""

from beamz.design.core import Design
from beamz.design.discretization import MaterialGrid, build_material_grid
from beamz.design.gds import ImportedComponent, export_gds, import_component, import_gds
from beamz.design.grid import Grid, RectilinearGrid
from beamz.design.grid_spec import GridSpec
from beamz.design.materials import Material
from beamz.design.structures import (
    Box,
    Circle,
    CircularBend,
    Polygon,
    Rectangle,
    Ring,
    Taper,
)

__all__ = [
    "Material",
    "Design",
    "MaterialGrid",
    "build_material_grid",
    "Box",
    "Rectangle",
    "Circle",
    "Ring",
    "CircularBend",
    "Polygon",
    "Taper",
    "GridSpec",
    "Grid",
    "RectilinearGrid",
    "ImportedComponent",
    "import_component",
    "import_gds",
    "export_gds",
]
