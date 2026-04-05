"""
Design module for BEAMZ - Contains components for designing photonic structures.
"""

from beamz.design import io
from beamz.design.core import Design
from beamz.design.materials import (
    CustomMaterial,
    CustomMaterialSpec,
    Material,
    MaterialSpec,
)
from beamz.design.meshing import RegularGrid, RegularGrid3D, create_mesh
from beamz.design.spec import DesignSpec
from beamz.design.structure_specs import StructureSpec
from beamz.design.structures import (
    Circle,
    CircularBend,
    Polygon,
    Rectangle,
    Ring,
    Taper,
)

__all__ = [
    "Material",
    "CustomMaterial",
    "MaterialSpec",
    "CustomMaterialSpec",
    "Design",
    "DesignSpec",
    "StructureSpec",
    "Rectangle",
    "Circle",
    "Ring",
    "CircularBend",
    "Polygon",
    "Taper",
    "RegularGrid",
    "RegularGrid3D",
    "create_mesh",
    "io",
]
