import numpy as np

from beamz.visual.animation import JupyterAnimator


def test_animator_update_persists_overlay_metadata():
    animator = JupyterAnimator(store_frames=True, live_display=False)
    sources = [object()]
    monitors = [object()]
    boundaries = [object()]
    design = object()

    animator.update(
        np.ones((2, 2)),
        t=0.0,
        step=0,
        num_steps=1,
        design=design,
        boundaries=boundaries,
        sources=sources,
        monitors=monitors,
    )

    assert animator.metadata["design"] is design
    assert animator.metadata["boundaries"] is boundaries
    assert animator.metadata["sources"] is sources
    assert animator.metadata["monitors"] is monitors


def test_animator_replay_uses_stored_overlay_metadata(monkeypatch):
    animator = JupyterAnimator(store_frames=True, live_display=False)
    sources = [object()]
    monitors = [object()]
    boundaries = [object()]
    design = object()

    animator.update(
        np.ones((2, 2)),
        t=0.0,
        step=0,
        num_steps=1,
        design=design,
        boundaries=boundaries,
        sources=sources,
        monitors=monitors,
    )

    seen = {}

    def capture(ax, design_arg, boundaries_arg, sources_arg, monitors_arg):
        seen["design"] = design_arg
        seen["boundaries"] = boundaries_arg
        seen["sources"] = sources_arg
        seen["monitors"] = monitors_arg

    monkeypatch.setattr(animator, "_add_overlays", capture)
    fig, anim = animator._build_replay(fps=30)

    assert fig is not None
    assert anim is not None
    assert seen == {
        "design": design,
        "boundaries": boundaries,
        "sources": sources,
        "monitors": monitors,
    }
