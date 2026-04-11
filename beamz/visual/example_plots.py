"""Legacy entry points removed from core BeamZ."""


def plot_simulation_overview(*args, **kwargs):
    raise RuntimeError(
        "plot_simulation_overview() was removed from beamz. "
        "Render overview plots directly inside examples."
    )


def plot_sparameters_db(*args, **kwargs):
    raise RuntimeError(
        "plot_sparameters_db() was removed from beamz. "
        "Render S-parameter plots directly inside examples."
    )


__all__ = ["plot_simulation_overview", "plot_sparameters_db"]
