"""
Design module for BEAMZ - Contains components for designing photonic structures.
"""

from beamz.design.core import Design
from beamz.design.materials import (
    AnisotropicMaterial,
    CustomMaterial,
    DebyeMaterial,
    DrudeMaterial,
    LorentzMaterial,
    Material,
    Material2D,
    PoleResidueMaterial,
    SellmeierMaterial,
)
from beamz.design.meshing import RegularGrid, RegularGrid3D, create_mesh
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
    "SellmeierMaterial",
    "DrudeMaterial",
    "LorentzMaterial",
    "DebyeMaterial",
    "PoleResidueMaterial",
    "Material2D",
    "AnisotropicMaterial",
    "Design",
    "Rectangle",
    "Circle",
    "Ring",
    "CircularBend",
    "Polygon",
    "Taper",
    "RegularGrid",
    "RegularGrid3D",
    "create_mesh",
]
