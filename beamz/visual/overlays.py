"""Legacy overlay helpers removed from core BeamZ."""

from beamz.visual.data import _get_deterministic_color


def __getattr__(name):
    raise AttributeError(
        f"beamz.visual.overlays.{name} is not available. "
        "Matplotlib overlay helpers were removed from beamz."
    )


__all__ = ["_get_deterministic_color"]
