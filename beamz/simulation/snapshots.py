"""Snapshot helpers for compiled simulation runs."""

from __future__ import annotations

import numpy as np

from beamz.visual.data import snapshot_payload, simulation_plot_data


def run_with_snapshots(sim, **kwargs):
    """Backward-compatible wrapper around Simulation.run()."""
    return sim.run(**kwargs)


def validate_snapshot_field(sim, snapshot_field: str) -> None:
    """Validate that a requested snapshot field exists on the simulation."""
    available = sim.fields.available_components()
    if snapshot_field not in available:
        raise ValueError(
            f"Field '{snapshot_field}' not found for snapshots. Available: {available}"
        )


def _snapshot_units_and_scale(field_name: str) -> tuple[str, float]:
    if "E" in field_name:
        return "V/µm", 1e-6
    return "A/m", 1.0


def collect_compiled_snapshots(sim, *, field_name: str, snapshot_data, layout=None):
    """Build host payloads from buffers emitted by the compiled kernel."""
    frames, steps, times, count = snapshot_data
    count_int = int(np.asarray(count))
    if count_int <= 0:
        return [], layout

    if layout is None:
        layout = simulation_plot_data(sim)

    extent = (0.0, sim.design.width, 0.0, sim.design.height)
    units, scale = _snapshot_units_and_scale(field_name)
    frames_np = np.asarray(frames[:count_int], dtype=np.float32)
    steps_np = np.asarray(steps[:count_int], dtype=np.int32)
    times_np = np.asarray(times[:count_int], dtype=np.float32)

    snapshots = []
    for frame, step, time_value in zip(frames_np, steps_np, times_np, strict=False):
        snapshots.append(
            snapshot_payload(
                field=frame * scale,
                field_name=field_name,
                t=time_value,
                step=step,
                num_steps=sim.num_steps,
                extent=extent,
                units=units,
                plane_2d=sim.plane_2d,
                layout=layout,
            )
        )
    return snapshots, layout
