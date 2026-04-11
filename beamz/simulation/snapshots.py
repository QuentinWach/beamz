"""Snapshot streaming for simulation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from beamz.visual.data import snapshot_payload


@dataclass
class SnapshotConfig:
    """Configuration for streaming field snapshots during a run."""

    snapshot_field: str | None = None
    snapshot_interval: int = 10
    snapshot_callback: Callable[[dict], None] | None = None
    store_snapshots: bool = True
    save_fields: list[str] | None = None
    field_subsample: int = 1
    progress: bool = False


def run_with_snapshots(sim, **kwargs):
    """Run a simulation and optionally stream/store field snapshots."""
    cfg = SnapshotConfig(**kwargs)
    field_history = {name: [] for name in (cfg.save_fields or [])}
    snapshots = []
    layout = None

    available = sim.fields.available_components()
    if cfg.snapshot_field is not None and cfg.snapshot_field not in available:
        raise ValueError(
            f"Field '{cfg.snapshot_field}' not found for snapshots. Available: {available}"
        )

    while sim.step():
        _store_fields(sim, field_history, cfg)

        if cfg.snapshot_field is None:
            continue
        if sim.current_step % cfg.snapshot_interval != 0:
            continue

        snapshot = _build_snapshot(sim, cfg, layout)
        if layout is None:
            layout = snapshot["layout"]
        if cfg.snapshot_callback is not None:
            cfg.snapshot_callback(snapshot)
        if cfg.store_snapshots:
            snapshots.append(snapshot)

    return _collect_results(sim, field_history, cfg, snapshots)


def _store_fields(sim, field_history, cfg):
    if not cfg.save_fields or sim.current_step % cfg.field_subsample != 0:
        return
    for field_name in cfg.save_fields:
        if hasattr(sim.fields, field_name):
            field_history[field_name].append(getattr(sim.fields, field_name).copy())


def _build_snapshot(sim, cfg, layout):
    field_name = cfg.snapshot_field
    extent = (0.0, sim.design.width, 0.0, sim.design.height)
    field = getattr(sim.fields, field_name)
    units = "V/µm" if "E" in field_name else "A/m"
    if "E" in field_name:
        field = field * 1e-6

    return snapshot_payload(
        field=field,
        field_name=field_name,
        t=sim.t,
        step=sim.current_step,
        num_steps=sim.num_steps,
        extent=extent,
        units=units,
        plane_2d=sim.plane_2d,
        simulation=sim,
        layout=layout,
    )


def _collect_results(sim, field_history, cfg, snapshots):
    from beamz.simulation.core import SimulationResults

    monitors = [device for device in sim.monitors if hasattr(device, "power_history")]
    fields = field_history if cfg.save_fields else None
    return SimulationResults.from_run(
        sim,
        fields=fields,
        monitors=monitors,
        snapshots=snapshots,
    )
