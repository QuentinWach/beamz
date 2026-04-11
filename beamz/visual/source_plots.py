"""Backward-compatible data helpers for source visualization."""

from beamz.visual.data import mode_profile_data, signal_plot_data


def plot_signal(*args, **kwargs):
    raise RuntimeError(
        "plot_signal() was removed from beamz. "
        "Use signal_plot_data() and render in examples."
    )


def show_mode_profile(*args, **kwargs):
    raise RuntimeError(
        "show_mode_profile() was removed from beamz. "
        "Use mode_profile_data() and render in examples."
    )


__all__ = ["mode_profile_data", "plot_signal", "show_mode_profile", "signal_plot_data"]
