"""Visualization-facing helpers for simulation sessions."""

from __future__ import annotations


def run(sim, **kwargs):
    """Run complete FDTD simulation with optional live field visualization."""
    wants_live_viz = any(
        kwargs.get(key) is not None
        for key in ("animate_live", "save_video", "jupyter_live")
    )
    if not wants_live_viz:
        save_fields = kwargs.get("save_fields")
        field_subsample = int(kwargs.get("field_subsample", 1))
        progress = bool(kwargs.get("progress", False))
        record_interval = field_subsample if save_fields else None
        return sim.run_compiled(
            num_steps=None,
            record_interval=record_interval,
            record_fields=save_fields,
            progress=progress,
        )

    from beamz.visual.runner import run_with_visualization

    return run_with_visualization(sim, **kwargs)


def to_scene(sim):
    """Build a 3D scene representation of the simulation setup."""
    from beamz.visual.scene import simulation_to_scene

    return simulation_to_scene(sim)


def show(sim, *, mode="auto", open_browser=True, **kwargs):
    """Display the simulation setup in the interactive 3D scene viewer."""
    from beamz.visual.scene import view3d

    return view3d(to_scene(sim), mode=mode, open_browser=open_browser, **kwargs)
