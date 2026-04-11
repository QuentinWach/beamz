"""Legacy video recording entry points removed from core BeamZ."""


class VideoRecorder:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "VideoRecorder was removed from beamz. "
            "Use Simulation.run(snapshot_field=...) and encode video in examples."
        )


__all__ = ["VideoRecorder"]
