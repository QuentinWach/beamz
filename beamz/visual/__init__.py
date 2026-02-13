"""
Visual module for BEAMZ - Contains visualization and UI helpers.
"""

from beamz.visual.helpers import (
    calc_optimal_fdtd_params,
    check_fdtd_stability,
    create_rich_progress,
    display_status,
    get_si_scale_and_label,
)
from beamz.visual.viz import (
    JupyterAnimator,
    VideoRecorder,
    animate_manual_field,
    draw_polygon,
    is_jupyter_environment,
    plot_fdtd_field,
    plot_fdtd_power,
    show_design,
    show_design_2d,
    show_design_3d,
)

__all__ = [
    "draw_polygon",
    "show_design",
    "show_design_2d",
    "show_design_3d",
    "plot_fdtd_field",
    "plot_fdtd_power",
    "animate_manual_field",
    "VideoRecorder",
    "JupyterAnimator",
    "is_jupyter_environment",
    "display_status",
    "create_rich_progress",
    "get_si_scale_and_label",
    "check_fdtd_stability",
    "calc_optimal_fdtd_params",
]
