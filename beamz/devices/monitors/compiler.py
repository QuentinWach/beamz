"""Compile Monitor objects into static packed monitor specs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from beamz.devices.monitors.monitors import Monitor


@dataclass(frozen=True)
class CompiledMonitorSpec:
    """Packed monitor descriptor consumed by compiled step kernels."""

    name: str
    monitor_index: int
    is_3d: bool
    record_interval: int
    accumulate_power: bool
    power_scale: float

    # 2D fields
    x_ez: jnp.ndarray | None = None
    y_ez: jnp.ndarray | None = None
    valid_ez: jnp.ndarray | None = None
    x_hx: jnp.ndarray | None = None
    y_hx: jnp.ndarray | None = None
    valid_hx: jnp.ndarray | None = None
    x_hy: jnp.ndarray | None = None
    y_hy: jnp.ndarray | None = None
    valid_hy: jnp.ndarray | None = None

    # 3D fields
    ex_idx: tuple[Any, ...] | None = None
    ey_idx: tuple[Any, ...] | None = None
    ez_idx: tuple[Any, ...] | None = None
    hx_idx: tuple[Any, ...] | None = None
    hy_idx: tuple[Any, ...] | None = None
    hz_idx: tuple[Any, ...] | None = None
    min_dim0: int = 0
    min_dim1: int = 0


def _clip_indices(x_idx: np.ndarray, y_idx: np.ndarray, shape: tuple[int, int]):
    h, w = shape
    valid = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
    x = np.clip(x_idx, 0, max(w - 1, 0)).astype(np.int32)
    y = np.clip(y_idx, 0, max(h - 1, 0)).astype(np.int32)
    return x, y, valid.astype(np.float32)


def _clamp_3d_index(idx, limit: int):
    if isinstance(idx, int):
        return int(min(max(0, idx), limit - 1))
    start = idx.start if idx.start is not None else 0
    stop = idx.stop if idx.stop is not None else limit
    start = max(0, min(start, limit - 1))
    stop = max(start, min(stop, limit))
    return slice(start, stop)


def _compile_monitor_3d_indices(monitor: Monitor, resolution: float, shape_3d: dict[str, tuple[int, ...]]):
    idx_map: dict[str, tuple[Any, ...]] = {}
    dim0 = []
    dim1 = []

    for name, shape in shape_3d.items():
        z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
            resolution,
            resolution,
            resolution,
            shape,
        )
        z_idx = _clamp_3d_index(z_idx, shape[0])
        y_idx = _clamp_3d_index(y_idx, shape[1])
        x_idx = _clamp_3d_index(x_idx, shape[2])
        idx = (z_idx, y_idx, x_idx)
        idx_map[name] = idx

        # infer resulting 2D slice shape
        arr = np.zeros(shape, dtype=np.float32)
        sliced = arr[idx]
        if sliced.ndim != 2:
            sliced = np.atleast_2d(sliced)
        dim0.append(sliced.shape[0])
        dim1.append(sliced.shape[1])

    return idx_map, int(min(dim0)), int(min(dim1))


def compile_monitor_specs(
    devices: list,
    fields,
    resolution: float,
    num_steps: int,
    dt: float,
) -> tuple[tuple[CompiledMonitorSpec, ...], int]:
    """Compile monitor devices into packed monitor specs.

    Returns
    -------
    specs:
        Tuple of monitor specs.
    max_records:
        Maximum number of records per monitor row in monitor-state buffers.
    """
    monitors = [d for d in devices if isinstance(d, Monitor)]
    if not monitors:
        return tuple(), 0

    specs: list[CompiledMonitorSpec] = []
    max_records = 0

    for mon_idx, monitor in enumerate(monitors):
        interval = max(1, int(monitor.record_interval))
        records = int(math.ceil(num_steps / interval))
        max_records = max(max_records, records)

        if not monitor.is_3d:
            points = monitor.get_grid_points_2d(resolution, resolution)
            if points:
                x_raw = np.asarray([p[0] for p in points], dtype=np.int32)
                y_raw = np.asarray([p[1] for p in points], dtype=np.int32)
            else:
                x_raw = np.zeros((0,), dtype=np.int32)
                y_raw = np.zeros((0,), dtype=np.int32)

            x_ez, y_ez, v_ez = _clip_indices(x_raw, y_raw, tuple(fields.Ez.shape))
            x_hx, y_hx, v_hx = _clip_indices(x_raw, y_raw, tuple(fields.Hx.shape))
            x_hy, y_hy, v_hy = _clip_indices(x_raw, y_raw, tuple(fields.Hy.shape))

            specs.append(
                CompiledMonitorSpec(
                    name=monitor.name or f"monitor_{mon_idx}",
                    monitor_index=mon_idx,
                    is_3d=False,
                    record_interval=interval,
                    accumulate_power=bool(monitor.accumulate_power),
                    power_scale=float(resolution * resolution),
                    x_ez=jnp.asarray(x_ez),
                    y_ez=jnp.asarray(y_ez),
                    valid_ez=jnp.asarray(v_ez),
                    x_hx=jnp.asarray(x_hx),
                    y_hx=jnp.asarray(y_hx),
                    valid_hx=jnp.asarray(v_hx),
                    x_hy=jnp.asarray(x_hy),
                    y_hy=jnp.asarray(y_hy),
                    valid_hy=jnp.asarray(v_hy),
                )
            )
        else:
            shape_3d = {
                "Ex": tuple(fields.Ex.shape),
                "Ey": tuple(fields.Ey.shape),
                "Ez": tuple(fields.Ez.shape),
                "Hx": tuple(fields.Hx.shape),
                "Hy": tuple(fields.Hy.shape),
                "Hz": tuple(fields.Hz.shape),
            }
            idx_map, min_dim0, min_dim1 = _compile_monitor_3d_indices(
                monitor,
                resolution,
                shape_3d,
            )

            specs.append(
                CompiledMonitorSpec(
                    name=monitor.name or f"monitor_{mon_idx}",
                    monitor_index=mon_idx,
                    is_3d=True,
                    record_interval=interval,
                    accumulate_power=bool(monitor.accumulate_power),
                    power_scale=float(resolution * resolution),
                    ex_idx=idx_map["Ex"],
                    ey_idx=idx_map["Ey"],
                    ez_idx=idx_map["Ez"],
                    hx_idx=idx_map["Hx"],
                    hy_idx=idx_map["Hy"],
                    hz_idx=idx_map["Hz"],
                    min_dim0=min_dim0,
                    min_dim1=min_dim1,
                )
            )

    del dt
    return tuple(specs), max_records
