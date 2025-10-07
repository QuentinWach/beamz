"""
Design module for BEAMZ - Contains components for designing photonic structures.
"""

from beamz.design.materials import Material, CustomMaterial
from beamz.design.core import Design
from beamz.design.structures import (
    Rectangle, Circle, Ring, CircularBend, Polygon, Taper
)
from beamz.design.pml import PML
from beamz.devices.sources import ModeSource, GaussianSource
from beamz.devices.monitors import Monitor
from beamz.devices.signals import ramped_cosine, plot_signal
from beamz.devices.mode import solve_modes

__all__ = [
    'Material', 'CustomMaterial',
    'Design', 'Rectangle', 'Circle', 'Ring',
    'CircularBend', 'Polygon', 'Taper', 'PML',
    'ModeSource', 'GaussianSource',
    'Monitor',
    'ramped_cosine', 'plot_signal'
]
