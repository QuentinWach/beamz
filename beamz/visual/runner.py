"""Simulation runner with visualization support.

Extracted from simulation/core.py to decouple the FDTD physics engine
from animation, video recording, and Jupyter display concerns.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class VizConfig:
    """Configuration for visualization during simulation run."""

    animate_live: str = None
    animation_interval: int = 10
    axis_scale: tuple = None
    cmap: str = "twilight_zero"
    clean_visualization: bool = False
    wavelength: float = None
    line_color: str = "gray"
    line_opacity: float = 0.5
    interpolation: str = "bicubic"
    save_video: str = None
    video_fps: int = 30
    video_dpi: int = 150
    video_field: str = None
    save_fields: list = None
    field_subsample: int = 1
    jupyter_live: bool = None
    store_animation: bool = True


def run_with_visualization(sim, **kwargs):
    """Run simulation with optional live visualization, video recording, etc.

    This is the implementation behind Simulation.run(). It wraps the physics
    stepping loop with visualization, field saving, and video recording.

    Args:
        sim: Simulation instance.
        **kwargs: Visualization parameters (see VizConfig fields).

    Returns:
        dict with 'fields', 'monitors', and/or 'animation' keys.
    """
    cfg = VizConfig(**kwargs)

    active_monitor, use_jupyter, jupyter_animator, video_recorder, viz_context = (
        _setup_visualization(sim, cfg)
    )

    field_history = {name: [] for name in (cfg.save_fields or [])}

    try:
        while sim.step():
            _store_fields(sim, field_history, cfg)

            if sim.current_step % cfg.animation_interval != 0:
                continue

            if video_recorder:
                _record_video_frame(sim, video_recorder, cfg)

            if cfg.animate_live:
                viz_context = _update_live_display(
                    sim, cfg, active_monitor, use_jupyter, jupyter_animator, viz_context
                )
    finally:
        if video_recorder:
            video_recorder.save()
        if jupyter_animator:
            jupyter_animator.finalize()
        if not use_jupyter and viz_context and viz_context.get("fig"):
            import matplotlib.pyplot as plt

            plt.show(block=False)
            print("Simulation complete. Close the plot window to continue.")

    return _collect_results(sim, field_history, cfg, jupyter_animator)


def _setup_visualization(sim, cfg):
    """Set up all visualization components."""
    active_monitor = None
    if cfg.animate_live and sim.is_3d:
        active_monitor = next((d for d in sim.monitors if d.is_3d), None)
        if not active_monitor:
            cfg.animate_live = None

    if cfg.animate_live:
        available = sim.fields.available_components()
        if cfg.animate_live not in available:
            cfg.animate_live = None

    if cfg.wavelength is None:
        for device in sim.sources:
            if hasattr(device, "wavelength"):
                cfg.wavelength = device.wavelength
                break

    from beamz.visual.animation import JupyterAnimator, is_jupyter_environment

    use_jupyter = (
        cfg.jupyter_live if cfg.jupyter_live is not None else is_jupyter_environment()
    )

    jupyter_animator = None
    if cfg.animate_live and use_jupyter:
        jupyter_animator = JupyterAnimator(
            cmap=cfg.cmap,
            axis_scale=cfg.axis_scale,
            clean_visualization=cfg.clean_visualization,
            wavelength=cfg.wavelength,
            line_color=cfg.line_color,
            line_opacity=cfg.line_opacity,
            interpolation=cfg.interpolation,
            live_display=True,
            store_frames=cfg.store_animation,
        )

    video_recorder = None
    if cfg.save_video:
        from beamz.visual.video import VideoRecorder

        record_field = cfg.video_field or cfg.animate_live or "Ez"
        available = sim.fields.available_components()
        if record_field not in available:
            print(
                f"Warning: Field '{record_field}' not found for video. Available: {available}"
            )
            record_field = available[0] if available else None
        if record_field:
            video_recorder = VideoRecorder(
                filename=cfg.save_video,
                fps=cfg.video_fps,
                dpi=cfg.video_dpi,
                cmap=cfg.cmap,
                axis_scale=cfg.axis_scale,
                clean_visualization=cfg.clean_visualization,
                wavelength=cfg.wavelength,
                line_color=cfg.line_color,
                line_opacity=cfg.line_opacity,
                interpolation=cfg.interpolation,
            )

    return active_monitor, use_jupyter, jupyter_animator, video_recorder, None


def _store_fields(sim, field_history, cfg):
    """Store field snapshots if requested."""
    if not cfg.save_fields or sim.current_step % cfg.field_subsample != 0:
        return
    for field_name in cfg.save_fields:
        if hasattr(sim.fields, field_name):
            field_history[field_name].append(getattr(sim.fields, field_name).copy())


def _record_video_frame(sim, video_recorder, cfg):
    """Record a single video frame."""
    record_field = cfg.video_field or cfg.animate_live or "Ez"
    if not hasattr(sim.fields, record_field):
        return
    field_display = getattr(sim.fields, record_field)
    if "E" in record_field:
        field_display = field_display * 1e-6
    video_recorder.add_frame(
        field_display,
        t=sim.t,
        step=sim.current_step,
        num_steps=sim.num_steps,
        field_name=record_field,
        units="V/µm" if "E" in record_field else "A/m",
        extent=(0, sim.design.width, 0, sim.design.height),
        design=sim.design,
        boundaries=sim.boundaries,
        plane_2d=sim.plane_2d,
    )


def _update_live_display(
    sim, cfg, active_monitor, use_jupyter, jupyter_animator, viz_context
):
    """Update live animation display and return updated viz_context."""
    from beamz.visual.animation import animate_manual_field

    if sim.is_3d and active_monitor:
        if (
            cfg.animate_live in active_monitor.fields
            and active_monitor.fields[cfg.animate_live]
        ):
            field_display = active_monitor.fields[cfg.animate_live][-1]
            extent = (
                active_monitor.start[0],
                active_monitor.start[0] + active_monitor.size[0],
                active_monitor.start[1],
                active_monitor.start[1] + active_monitor.size[1],
            )
        else:
            return viz_context
    else:
        field_display = getattr(sim.fields, cfg.animate_live)
        extent = (0, sim.design.width, 0, sim.design.height)

    if "E" in cfg.animate_live:
        field_display = field_display * 1e-6
    units = "V/µm" if "E" in cfg.animate_live else "A/m"

    if use_jupyter and jupyter_animator:
        jupyter_animator.update(
            field_display,
            t=sim.t,
            step=sim.current_step,
            num_steps=sim.num_steps,
            field_name=cfg.animate_live,
            units=units,
            extent=extent,
            design=sim.design,
            boundaries=sim.boundaries,
            plane_2d=sim.plane_2d,
        )
    else:
        title = f"{cfg.animate_live} at t = {sim.t:.2e} s (step {sim.current_step}/{sim.num_steps})"
        viz_context = animate_manual_field(
            field_display,
            context=viz_context,
            extent=extent,
            title=title,
            units=units,
            design=sim.design,
            boundaries=sim.boundaries,
            pause=0.001,
            axis_scale=cfg.axis_scale,
            cmap=cfg.cmap,
            clean_visualization=cfg.clean_visualization,
            wavelength=cfg.wavelength,
            line_color=cfg.line_color,
            line_opacity=cfg.line_opacity,
            plane_2d=sim.plane_2d,
            interpolation=cfg.interpolation,
        )
    return viz_context


def _collect_results(sim, field_history, cfg, jupyter_animator):
    """Collect and return simulation results."""
    from beamz.simulation.core import SimulationResults

    monitors = [device for device in sim.monitors if hasattr(device, "power_history")]
    fields = field_history if cfg.save_fields else None
    animation = (
        jupyter_animator
        if jupyter_animator is not None and jupyter_animator.frames
        else None
    )
    return SimulationResults.from_run(
        sim,
        fields=fields,
        monitors=monitors,
        animation=animation,
    )
