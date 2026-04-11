"""Backward-compatible data helpers for monitor visualization."""

from beamz.visual.data import monitor_field_plot_data, monitor_power_plot_data


def plot_monitor_fields(*args, **kwargs):
    raise RuntimeError(
        "plot_monitor_fields() was removed from beamz. "
        "Use monitor_field_plot_data() and render in examples."
    )


def plot_monitor_power(*args, **kwargs):
    raise RuntimeError(
        "plot_monitor_power() was removed from beamz. "
        "Use monitor_power_plot_data() and render in examples."
    )


def animate_monitor_fields(*args, **kwargs):
    raise RuntimeError(
        "animate_monitor_fields() was removed from beamz. "
        "Use monitor_field_plot_data() across recorded frames and animate in examples."
    )


__all__ = [
    "animate_monitor_fields",
    "monitor_field_plot_data",
    "monitor_power_plot_data",
    "plot_monitor_fields",
    "plot_monitor_power",
]
