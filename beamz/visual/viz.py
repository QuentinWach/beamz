"""Backward-compatible re-exports from visual submodules."""

from beamz.visual.animation import (
    JupyterAnimator,
    animate_fdtd_live,
    animate_manual_field,
    get_twilight_zero_cmap,
    is_jupyter_environment,
    save_fdtd_animation,
)
from beamz.visual.design_viz import (
    _add_direction_arrow_to_3d_plot,
    _add_gaussian_source_to_3d_plot,
    _add_mode_source_to_3d_plot,
    _add_monitor_to_3d_plot,
    _get_deterministic_color,
    determine_if_3d,
    draw_boundary,
    draw_pml,
    draw_polygon,
    show_design,
    show_design_2d,
    show_design_3d,
    structure_to_3d_mesh,
)
from beamz.visual.fields import close_fdtd_figure, plot_fdtd_field, plot_fdtd_power
from beamz.visual.video import VideoRecorder

__all__ = [
    "draw_polygon",
    "draw_pml",
    "draw_boundary",
    "determine_if_3d",
    "show_design",
    "show_design_2d",
    "show_design_3d",
    "plot_fdtd_field",
    "plot_fdtd_power",
    "close_fdtd_figure",
    "animate_fdtd_live",
    "animate_manual_field",
    "save_fdtd_animation",
    "get_twilight_zero_cmap",
    "is_jupyter_environment",
    "VideoRecorder",
    "JupyterAnimator",
    "structure_to_3d_mesh",
]
