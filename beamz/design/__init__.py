"""
Design module for BEAMZ - Contains components for designing photonic structures.
"""

from beamz.design.materials import Material, CustomMaterial
from beamz.design.core import Design
from beamz.design.structures import Rectangle, Circle, Ring, CircularBend, Polygon, Taper
from beamz.design.meshing import RegularGrid, RegularGrid3D, create_mesh

__all__ = ['Material', 'CustomMaterial', 'Design', 'Rectangle', 'Circle', 'Ring', 'CircularBend', 'Polygon', 'Taper',
           'RegularGrid', 'RegularGrid3D', 'create_mesh']
