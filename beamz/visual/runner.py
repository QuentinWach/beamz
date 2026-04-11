"""Legacy matplotlib runner removed from core BeamZ."""


def run_with_visualization(*args, **kwargs):
    raise RuntimeError(
        "Matplotlib-backed simulation rendering was removed from beamz. "
        "Use beamz.simulation.snapshots.run_with_snapshots() through Simulation.run()."
    )


__all__ = ["run_with_visualization"]
