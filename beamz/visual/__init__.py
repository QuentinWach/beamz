"""
Visual module for BEAMZ - Contains visualization and UI helpers.
"""

from beamz.visual.animation import (
    JupyterAnimator,
    animate_manual_field,
    is_jupyter_environment,
)
from beamz.visual.data import Slice2D, Trace1D
from beamz.visual.design_3d import show_design_3d
from beamz.visual.design_viz import (
    draw_boundary,
    draw_polygon,
    show_design,
    show_design_2d,
)
from beamz.visual.helpers import (
    calc_optimal_fdtd_params,
    check_fdtd_stability,
    create_rich_progress,
    display_status,
    get_si_scale_and_label,
)
from beamz.visual.scene import beamz_to_scene, view3d
from beamz.visual.video import VideoRecorder

__all__ = [
    "draw_polygon",
    "draw_boundary",
    "show_design",
    "show_design_2d",
    "show_design_3d",
    "animate_manual_field",
    "VideoRecorder",
    "JupyterAnimator",
    "is_jupyter_environment",
    "Slice2D",
    "Trace1D",
    "display_status",
    "create_rich_progress",
    "get_si_scale_and_label",
    "check_fdtd_stability",
    "calc_optimal_fdtd_params",
    "beamz_to_scene",
    "view3d",
]
