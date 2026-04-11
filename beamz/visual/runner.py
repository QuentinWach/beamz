"""Legacy matplotlib runner removed from core BeamZ."""


def run_with_visualization(*args, **kwargs):
    raise RuntimeError(
        "Matplotlib-backed simulation rendering was removed from beamz. "
        "Use Simulation.run(...) to collect data, then render it in examples."
    )


__all__ = ["run_with_visualization"]
