"""
Visual module for BEAMZ - Contains visualization and UI helpers.
"""

from beamz.visual.viz import (
    draw_polygon,
    show_design,
    show_design_2d,
    show_design_3d,
    plot_fdtd_field,
    animate_fdtd_live,
    save_fdtd_animation,
    plot_fdtd_power,
    close_fdtd_figure,
    animate_manual_field,
    VideoRecorder,
    JupyterAnimator,
    is_jupyter_environment
)

from beamz.visual.helpers import (
    display_status,
    display_header,
    display_parameters,
    display_results,
    create_rich_progress,
    get_si_scale_and_label,
    check_fdtd_stability,
    calc_optimal_fdtd_params
)

__all__ = [
    'draw_polygon',
    'show_design',
    'show_design_2d',
    'show_design_3d',
    'plot_fdtd_field',
    'animate_fdtd_live',
    'save_fdtd_animation',
    'plot_fdtd_power',
    'close_fdtd_figure',
    'animate_manual_field',
    'VideoRecorder',
    'JupyterAnimator',
    'is_jupyter_environment',
    'display_status',
    'display_header',
    'display_parameters',
    'display_results',
    'create_rich_progress',
    'get_si_scale_and_label',
    'check_fdtd_stability',
    'calc_optimal_fdtd_params'
]

