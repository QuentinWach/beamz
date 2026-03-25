"""
Visual module for BEAMZ - Contains visualization and UI helpers.
"""

from beamz.visual.animation import (
    JupyterAnimator,
    animate_manual_field,
    is_jupyter_environment,
)
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
from beamz.visual.example_plots import plot_simulation_overview, plot_sparameters_db
from beamz.visual.scene import beamz_to_scene, view3d
from beamz.visual.source_plots import plot_signal, show_mode_profile
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
    "display_status",
    "create_rich_progress",
    "get_si_scale_and_label",
    "check_fdtd_stability",
    "calc_optimal_fdtd_params",
    "plot_simulation_overview",
    "plot_sparameters_db",
    "plot_signal",
    "show_mode_profile",
    "beamz_to_scene",
    "view3d",
]
