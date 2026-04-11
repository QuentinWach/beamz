"""Legacy animation entry points removed from core BeamZ."""


class JupyterAnimator:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "JupyterAnimator was removed from beamz. "
            "Use Simulation.run(snapshot_field=..., snapshot_callback=...) and animate in examples."
        )


def animate_manual_field(*args, **kwargs):
    raise RuntimeError(
        "animate_manual_field() was removed from beamz. "
        "Animate snapshots directly in examples."
    )


def is_jupyter_environment():
    return False


__all__ = ["JupyterAnimator", "animate_manual_field", "is_jupyter_environment"]
