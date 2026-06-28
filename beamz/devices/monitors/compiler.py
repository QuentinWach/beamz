"""Compile Monitor objects into static packed monitor specs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from beamz.const import µm
from beamz.devices.monitors.monitors import (
    Monitor,
    _line_integral_scale_2d,
    _line_normal_2d,
    _plane_axes_for_normal_3d,
)


@dataclass(frozen=True)
class CompiledMonitorSpec:
    """Packed monitor descriptor consumed by compiled step kernels."""

    name: str
    monitor_index: int
    is_3d: bool
    record_interval: int
    accumulate_power: bool
    power_scale: float
    normal_axis: int = -1
    normal_sign: float = 1.0
    accumulate_frequency: bool = False
    freq_record_interval: int = 1
    freq_count: int = 0
    freq_hz: jnp.ndarray | None = None
    freq_rot_re: jnp.ndarray | None = None
    freq_rot_im: jnp.ndarray | None = None
    dft_enabled: bool = False
    dft_record_interval: int = 1
    dft_t_start: float = 0.0
    dft_t_end: float = np.inf
    dft_window_code: int = 0  # 0=rect, 1=hann
    dft_normalization_code: int = 0  # 0=native, 1=physical
    dft_length_unit: float = 1e-6
    dft_centered_tm_xy_sampling: bool = False
    dft_point_count: int = 0
    dft_component_mask: jnp.ndarray | None = None
    dft_target_x: jnp.ndarray | None = None
    dft_target_y: jnp.ndarray | None = None

    # 2D fields
    x_ex: jnp.ndarray | None = None
    y_ex: jnp.ndarray | None = None
    valid_ex: jnp.ndarray | None = None
    x_ey: jnp.ndarray | None = None
    y_ey: jnp.ndarray | None = None
    valid_ey: jnp.ndarray | None = None
    x_ez: jnp.ndarray | None = None
    y_ez: jnp.ndarray | None = None
    valid_ez: jnp.ndarray | None = None
    x_hx: jnp.ndarray | None = None
    y_hx: jnp.ndarray | None = None
    valid_hx: jnp.ndarray | None = None
    x_hy: jnp.ndarray | None = None
    y_hy: jnp.ndarray | None = None
    valid_hy: jnp.ndarray | None = None
    x_hz: jnp.ndarray | None = None
    y_hz: jnp.ndarray | None = None
    valid_hz: jnp.ndarray | None = None

    # 3D fields
    ex_idx: tuple[Any, ...] | None = None
    ey_idx: tuple[Any, ...] | None = None
    ez_idx: tuple[Any, ...] | None = None
    hx_idx: tuple[Any, ...] | None = None
    hy_idx: tuple[Any, ...] | None = None
    hz_idx: tuple[Any, ...] | None = None
    ex_interp_flat_idx: jnp.ndarray | None = None
    ex_interp_weights: jnp.ndarray | None = None
    ey_interp_flat_idx: jnp.ndarray | None = None
    ey_interp_weights: jnp.ndarray | None = None
    ez_interp_flat_idx: jnp.ndarray | None = None
    ez_interp_weights: jnp.ndarray | None = None
    hx_interp_flat_idx: jnp.ndarray | None = None
    hx_interp_weights: jnp.ndarray | None = None
    hy_interp_flat_idx: jnp.ndarray | None = None
    hy_interp_weights: jnp.ndarray | None = None
    hz_interp_flat_idx: jnp.ndarray | None = None
    hz_interp_weights: jnp.ndarray | None = None
    min_dim0: int = 0
    min_dim1: int = 0


@dataclass(frozen=True)
class BatchedMonitorData:
    """Stacked 3D monitor data for fori_loop-based power computation.

    Pre-computed raveled indices allow uniform gather ops inside a
    fori_loop body, keeping HLO size constant regardless of monitor count.
    """

    n_monitors: int
    monitor_indices: jnp.ndarray  # (n,) int32 — maps batch idx → monitor_state row
    record_intervals: jnp.ndarray  # (n,) int32
    accumulate_flags: jnp.ndarray  # (n,) bool
    power_scales: jnp.ndarray  # (n,) float32
    # Raveled indices per component: (n, max_points) int32
    ex_interp_flat_idx: jnp.ndarray
    ex_interp_weights: jnp.ndarray
    ey_interp_flat_idx: jnp.ndarray
    ey_interp_weights: jnp.ndarray
    ez_interp_flat_idx: jnp.ndarray
    ez_interp_weights: jnp.ndarray
    hx_interp_flat_idx: jnp.ndarray
    hx_interp_weights: jnp.ndarray
    hy_interp_flat_idx: jnp.ndarray
    hy_interp_weights: jnp.ndarray
    hz_interp_flat_idx: jnp.ndarray
    hz_interp_weights: jnp.ndarray
    valid_mask: jnp.ndarray  # (n, max_points) float32
    normal_axes: jnp.ndarray  # (n,) int32; 0=x,1=y,2=z, -1=unknown
    normal_signs: jnp.ndarray  # (n,) float32
    # Frequency-domain in-loop accumulation metadata
    freq_enabled: jnp.ndarray  # (n,) bool
    freq_record_intervals: jnp.ndarray  # (n,) int32
    freq_hz: jnp.ndarray  # (n, max_freq) float32
    freq_rot_re: jnp.ndarray  # (n, max_freq) float32
    freq_rot_im: jnp.ndarray  # (n, max_freq) float32
    freq_mask: jnp.ndarray  # (n, max_freq) float32
    dft_enabled: jnp.ndarray  # (n,) bool
    dft_record_intervals: jnp.ndarray  # (n,) int32
    dft_t_start: jnp.ndarray  # (n,) float32
    dft_t_end: jnp.ndarray  # (n,) float32
    dft_window_code: jnp.ndarray  # (n,) int32
    dft_normalization_code: jnp.ndarray  # (n,) int32
    dft_length_unit: jnp.ndarray  # (n,) float32
    dft_component_mask: jnp.ndarray  # (n, 6) float32


def compile_batched_monitor_data(
    specs: tuple[CompiledMonitorSpec, ...],
    field_shapes: dict[str, tuple[int, ...]],
) -> BatchedMonitorData | None:
    """Compile 3D monitors into batched form for fori_loop update."""
    specs_3d = [s for s in specs if s.is_3d]
    if not specs_3d:
        return None

    n = len(specs_3d)
    components = [
        ("Ex", "ex_idx"),
        ("Ey", "ey_idx"),
        ("Ez", "ez_idx"),
        ("Hx", "hx_idx"),
        ("Hy", "hy_idx"),
        ("Hz", "hz_idx"),
    ]

    all_flat: dict[str, list[np.ndarray]] = {c: [] for c, _ in components}
    all_weights: dict[str, list[np.ndarray]] = {c: [] for c, _ in components}
    all_n_points: list[int] = []

    for spec in specs_3d:
        n_pts = spec.min_dim0 * spec.min_dim1
        all_n_points.append(n_pts)
        for comp, attr in components:
            flat_idx = np.asarray(
                getattr(spec, f"{comp.lower()}_interp_flat_idx"), dtype=np.int32
            ).reshape(n_pts, 8)
            weights = np.asarray(
                getattr(spec, f"{comp.lower()}_interp_weights"), dtype=np.float32
            ).reshape(n_pts, 8)
            all_flat[comp].append(flat_idx)
            all_weights[comp].append(weights)

    max_points = max(all_n_points) if all_n_points else 0
    if max_points == 0:
        return None

    max_freq = max(int(s.freq_count) for s in specs_3d) if specs_3d else 0

    def _pad_stack(flat_list: list[np.ndarray], dtype) -> jnp.ndarray:
        padded = []
        for flat in flat_list:
            p = np.zeros((max_points, 8), dtype=dtype)
            p[: flat.shape[0], : flat.shape[1]] = flat
            padded.append(p)
        return jnp.array(np.stack(padded))

    valid = np.zeros((n, max_points), dtype=np.float32)
    for i, n_pts in enumerate(all_n_points):
        valid[i, :n_pts] = 1.0

    freq_mask = np.zeros((n, max_freq), dtype=np.float32)
    freq_hz = np.zeros((n, max_freq), dtype=np.float32)
    freq_rot_re = np.ones((n, max_freq), dtype=np.float32)
    freq_rot_im = np.zeros((n, max_freq), dtype=np.float32)
    dft_enabled = np.zeros((n,), dtype=bool)
    dft_record_intervals = np.ones((n,), dtype=np.int32)
    dft_t_start = np.zeros((n,), dtype=np.float32)
    dft_t_end = np.full((n,), np.inf, dtype=np.float32)
    dft_window_code = np.zeros((n,), dtype=np.int32)
    dft_normalization_code = np.zeros((n,), dtype=np.int32)
    dft_length_unit = np.ones((n,), dtype=np.float32)
    dft_component_mask = np.zeros((n, 6), dtype=np.float32)
    for i, spec in enumerate(specs_3d):
        if (
            spec.freq_count > 0
            and spec.freq_hz is not None
            and spec.freq_rot_re is not None
            and spec.freq_rot_im is not None
        ):
            cnt = int(spec.freq_count)
            freq_mask[i, :cnt] = 1.0
            freq_hz[i, :cnt] = np.asarray(spec.freq_hz, dtype=np.float32)[:cnt]
            freq_rot_re[i, :cnt] = np.asarray(spec.freq_rot_re, dtype=np.float32)[:cnt]
            freq_rot_im[i, :cnt] = np.asarray(spec.freq_rot_im, dtype=np.float32)[:cnt]
        dft_enabled[i] = bool(spec.dft_enabled and spec.freq_count > 0)
        dft_record_intervals[i] = int(max(1, spec.dft_record_interval))
        dft_t_start[i] = float(spec.dft_t_start)
        dft_t_end[i] = float(spec.dft_t_end)
        dft_window_code[i] = int(spec.dft_window_code)
        dft_normalization_code[i] = int(spec.dft_normalization_code)
        dft_length_unit[i] = float(spec.dft_length_unit)
        if spec.dft_component_mask is not None:
            dft_component_mask[i, :] = np.asarray(
                spec.dft_component_mask, dtype=np.float32
            )[:6]

    return BatchedMonitorData(
        n_monitors=n,
        monitor_indices=jnp.array([s.monitor_index for s in specs_3d], dtype=jnp.int32),
        record_intervals=jnp.array(
            [s.record_interval for s in specs_3d], dtype=jnp.int32
        ),
        accumulate_flags=jnp.array([s.accumulate_power for s in specs_3d]),
        power_scales=jnp.array([s.power_scale for s in specs_3d], dtype=jnp.float32),
        ex_interp_flat_idx=_pad_stack(all_flat["Ex"], np.int32),
        ex_interp_weights=_pad_stack(all_weights["Ex"], np.float32),
        ey_interp_flat_idx=_pad_stack(all_flat["Ey"], np.int32),
        ey_interp_weights=_pad_stack(all_weights["Ey"], np.float32),
        ez_interp_flat_idx=_pad_stack(all_flat["Ez"], np.int32),
        ez_interp_weights=_pad_stack(all_weights["Ez"], np.float32),
        hx_interp_flat_idx=_pad_stack(all_flat["Hx"], np.int32),
        hx_interp_weights=_pad_stack(all_weights["Hx"], np.float32),
        hy_interp_flat_idx=_pad_stack(all_flat["Hy"], np.int32),
        hy_interp_weights=_pad_stack(all_weights["Hy"], np.float32),
        hz_interp_flat_idx=_pad_stack(all_flat["Hz"], np.int32),
        hz_interp_weights=_pad_stack(all_weights["Hz"], np.float32),
        valid_mask=jnp.array(valid),
        normal_axes=jnp.array([int(s.normal_axis) for s in specs_3d], dtype=jnp.int32),
        normal_signs=jnp.array(
            [float(s.normal_sign) for s in specs_3d], dtype=jnp.float32
        ),
        freq_enabled=jnp.array([bool(s.accumulate_frequency) for s in specs_3d]),
        freq_record_intervals=jnp.array(
            [max(1, int(s.freq_record_interval)) for s in specs_3d], dtype=jnp.int32
        ),
        freq_hz=jnp.array(freq_hz, dtype=jnp.float32),
        freq_rot_re=jnp.array(freq_rot_re, dtype=jnp.float32),
        freq_rot_im=jnp.array(freq_rot_im, dtype=jnp.float32),
        freq_mask=jnp.array(freq_mask, dtype=jnp.float32),
        dft_enabled=jnp.array(dft_enabled),
        dft_record_intervals=jnp.array(dft_record_intervals, dtype=jnp.int32),
        dft_t_start=jnp.array(dft_t_start, dtype=jnp.float32),
        dft_t_end=jnp.array(dft_t_end, dtype=jnp.float32),
        dft_window_code=jnp.array(dft_window_code, dtype=jnp.int32),
        dft_normalization_code=jnp.array(dft_normalization_code, dtype=jnp.int32),
        dft_length_unit=jnp.array(dft_length_unit, dtype=jnp.float32),
        dft_component_mask=jnp.array(dft_component_mask, dtype=jnp.float32),
    )


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


def _compile_monitor_3d_indices(
    monitor: Monitor, resolution: float, shape_3d: dict[str, tuple[int, ...]]
):
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


def _linear_interp_plan_1d(
    src_coords: np.ndarray,
    dst_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(src_coords, dtype=np.float64).reshape(-1)
    dst = np.asarray(dst_coords, dtype=np.float64).reshape(-1)
    if src.size == 0:
        raise ValueError("3D monitor interpolation source coordinates cannot be empty.")
    hi = np.searchsorted(src, dst, side="right")
    hi = np.clip(hi, 1, src.size - 1)
    lo = hi - 1
    left_mask = dst <= src[0]
    right_mask = dst >= src[-1]
    lo[left_mask] = 0
    hi[left_mask] = 0
    lo[right_mask] = src.size - 1
    hi[right_mask] = src.size - 1
    denom = src[hi] - src[lo]
    alpha = np.zeros_like(dst, dtype=np.float64)
    interior = (hi != lo) & (np.abs(denom) > 0.0)
    alpha[interior] = (dst[interior] - src[lo[interior]]) / denom[interior]
    w_hi = alpha.astype(np.float32)
    w_lo = (1.0 - alpha).astype(np.float32)
    w_lo[hi == lo] = 1.0
    w_hi[hi == lo] = 0.0
    return lo.astype(np.int32), hi.astype(np.int32), w_lo, w_hi


def _compile_monitor_3d_interpolation(
    monitor: Monitor,
    component: str,
    resolution: float,
    base_shape: tuple[int, int, int],
    field_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    from beamz.simulation.yee import component_coordinates_3d_um

    axis = str(getattr(monitor, "plane_normal", "z")).lower()
    axis0, axis1 = _plane_axes_for_normal_3d(axis)
    target0, target1 = monitor.get_analysis_plane_coords_3d(
        dx=resolution,
        dy=resolution,
        dz=resolution,
        field_shape=base_shape,
    )
    bounds, plane_pos = monitor._analysis_plane_bounds_3d()
    coords_um = component_coordinates_3d_um(
        component,
        tuple(int(v) for v in base_shape),
        float(resolution / µm),
    )
    src0 = np.asarray(coords_um[axis0], dtype=np.float64) * 1e-6
    src1 = np.asarray(coords_um[axis1], dtype=np.float64) * 1e-6
    srcn = np.asarray(coords_um[axis], dtype=np.float64) * 1e-6
    grid0, grid1 = np.meshgrid(
        np.asarray(target0, dtype=np.float64),
        np.asarray(target1, dtype=np.float64),
        indexing="ij",
    )
    flat0 = grid0.reshape(-1)
    flat1 = grid1.reshape(-1)
    flatn = np.full_like(flat0, float(plane_pos), dtype=np.float64)

    lo0, hi0, w0_lo, w0_hi = _linear_interp_plan_1d(src0, flat0)
    lo1, hi1, w1_lo, w1_hi = _linear_interp_plan_1d(src1, flat1)
    lon, hin, wn_lo, wn_hi = _linear_interp_plan_1d(srcn, flatn)

    corner_bits = np.asarray(
        [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ],
        dtype=np.int32,
    )

    idx0 = np.where(corner_bits[:, 0][None, :] == 0, lo0[:, None], hi0[:, None])
    idx1 = np.where(corner_bits[:, 1][None, :] == 0, lo1[:, None], hi1[:, None])
    idxn = np.where(corner_bits[:, 2][None, :] == 0, lon[:, None], hin[:, None])
    w0 = np.where(corner_bits[:, 0][None, :] == 0, w0_lo[:, None], w0_hi[:, None])
    w1 = np.where(corner_bits[:, 1][None, :] == 0, w1_lo[:, None], w1_hi[:, None])
    wn = np.where(corner_bits[:, 2][None, :] == 0, wn_lo[:, None], wn_hi[:, None])
    weights = (w0 * w1 * wn).astype(np.float32)

    index_map = {axis0: idx0, axis1: idx1, axis: idxn}
    z_idx = np.asarray(index_map["z"], dtype=np.int32)
    y_idx = np.asarray(index_map["y"], dtype=np.int32)
    x_idx = np.asarray(index_map["x"], dtype=np.int32)
    flat_idx = np.ravel_multi_index(
        (z_idx, y_idx, x_idx),
        dims=tuple(int(v) for v in field_shape),
    ).astype(np.int32)
    return (
        flat_idx,
        weights,
        int(np.asarray(target0).size),
        int(np.asarray(target1).size),
    )


def sample_compiled_monitor_plane_component_3d(
    field: np.ndarray,
    flat_idx: np.ndarray,
    weights: np.ndarray,
    dim0: int,
    dim1: int,
) -> np.ndarray:
    flat = np.asarray(field, dtype=np.complex128).reshape(-1)
    idx = np.asarray(flat_idx, dtype=np.int32).reshape(-1, 8)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 8)
    sampled = np.sum(flat[idx] * w.astype(np.complex128), axis=1)
    return sampled.reshape(int(dim0), int(dim1))


def _analysis_plane_sample_area(
    coord0: np.ndarray,
    coord1: np.ndarray,
    fallback_step: float,
) -> float:
    def _axis_step(coord):
        arr = np.asarray(coord, dtype=np.float64).reshape(-1)
        if arr.size > 1:
            diffs = np.diff(arr)
            step = float(np.median(np.abs(diffs)))
            if np.isfinite(step) and step > 0.0:
                return step
        return float(fallback_step)

    return float(_axis_step(coord0) * _axis_step(coord1))


def _crop_monitor_3d_interpolation(
    flat_idx: np.ndarray,
    weights: np.ndarray,
    dim0: int,
    dim1: int,
    target_dim0: int,
    target_dim1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a component interpolation plan to the common monitor-plane shape."""
    flat_arr = np.asarray(flat_idx, dtype=np.int32).reshape(int(dim0), int(dim1), 8)
    weight_arr = np.asarray(weights, dtype=np.float32).reshape(int(dim0), int(dim1), 8)
    flat_arr = flat_arr[: int(target_dim0), : int(target_dim1), :]
    weight_arr = weight_arr[: int(target_dim0), : int(target_dim1), :]
    return flat_arr.reshape(-1, 8), weight_arr.reshape(-1, 8)


def _monitor_normal_2d(monitor: Monitor, resolution: float) -> tuple[int, float]:
    snapped = monitor.get_snapped_region(dx=resolution, dy=resolution)
    line_normal = _line_normal_2d(
        getattr(monitor, "start", None),
        getattr(monitor, "end", None),
    )
    if snapped is not None:
        axis = str(snapped.normal_axis).lower()
        sign = (
            float(line_normal[1])
            if line_normal is not None and line_normal[0] == axis
            else 1.0
        )
        return {"x": 0, "y": 1}.get(axis, -1), sign
    if line_normal is None:
        return -1, 1.0
    axis, sign = line_normal
    return {"x": 0, "y": 1}.get(axis, -1), float(sign)


def _compile_monitor_2d_interpolation(
    monitor: Monitor,
    component: str,
    fields,
    resolution: float,
) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    from beamz.simulation.yee import component_coordinates_2d_um

    line_coords = monitor._line_sample_coords_2d(resolution, resolution)
    if line_coords is None:
        return None, None

    plane = getattr(fields, "plane_2d", "xy")
    coords = component_coordinates_2d_um(
        component,
        tuple(int(v) for v in fields.permittivity.shape),
        float(resolution),
        plane,
    )
    if "x" not in coords or "y" not in coords:
        return None, None

    src_x = np.asarray(coords["x"], dtype=np.float64)
    src_y = np.asarray(coords["y"], dtype=np.float64)
    dst_x = np.asarray(line_coords[0], dtype=np.float64).reshape(-1)
    dst_y = np.asarray(line_coords[1], dtype=np.float64).reshape(-1)
    if dst_x.size != dst_y.size:
        return None, None

    x0, x1, wx0, wx1 = _linear_interp_plan_1d(src_x, dst_x)
    y0, y1, wy0, wy1 = _linear_interp_plan_1d(src_y, dst_y)
    bits = np.asarray([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=np.int32)
    x_idx = np.where(bits[:, 1][None, :] == 0, x0[:, None], x1[:, None])
    y_idx = np.where(bits[:, 0][None, :] == 0, y0[:, None], y1[:, None])
    wx = np.where(bits[:, 1][None, :] == 0, wx0[:, None], wx1[:, None])
    wy = np.where(bits[:, 0][None, :] == 0, wy0[:, None], wy1[:, None])
    weights = (wx * wy).astype(np.float32)
    flat_idx = np.ravel_multi_index(
        (y_idx.astype(np.int32), x_idx.astype(np.int32)),
        dims=tuple(int(v) for v in getattr(fields, component).shape),
    ).astype(np.int32)
    return jnp.asarray(flat_idx), jnp.asarray(weights)


def compile_monitor_specs(
    monitors: list,
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
    if not monitors:
        return tuple(), 0

    specs: list[CompiledMonitorSpec] = []
    max_records = 0

    for mon_idx, monitor in enumerate(monitors):
        interval = max(1, int(monitor.record_interval))
        records = int(math.ceil(num_steps / interval))
        max_records = max(max_records, records)

        dft_enabled = bool(getattr(monitor, "dft_enabled", False))
        dft_freqs = np.asarray(
            getattr(monitor, "dft_frequencies", np.zeros((0,))), dtype=np.float64
        ).ravel()
        if dft_freqs.size > 0 and not np.all(np.isfinite(dft_freqs)):
            raise ValueError("Monitor dft_frequencies must be finite values in Hz")
        if dft_freqs.size > 0 and np.any(dft_freqs <= 0.0):
            raise ValueError("Monitor dft_frequencies must be strictly positive")

        flux_freqs = np.asarray(
            getattr(monitor, "power_spectrum_frequencies", np.zeros((0,))),
            dtype=np.float64,
        ).ravel()
        if flux_freqs.size > 0 and not np.all(np.isfinite(flux_freqs)):
            raise ValueError(
                "Monitor power_spectrum_frequencies must be finite values in Hz"
            )
        dft_active = bool(dft_enabled and dft_freqs.size > 0)
        dft_interval = max(
            1,
            int(
                getattr(
                    monitor,
                    "dft_record_interval",
                    (
                        1
                        if bool(getattr(monitor, "dft_record_every_step", True))
                        else interval
                    ),
                )
            ),
        )
        power_spectrum_interval = max(
            1, int(getattr(monitor, "power_spectrum_record_interval", 1))
        )
        if dft_active and flux_freqs.size > 0:
            if dft_freqs.shape != flux_freqs.shape or not np.allclose(
                dft_freqs, flux_freqs, rtol=1e-12, atol=0.0
            ):
                raise ValueError(
                    "A monitor cannot pack DFT component frequencies and "
                    "power_spectrum_frequencies onto different grids. Use matching "
                    "frequencies or separate monitors."
                )
            if dft_interval != power_spectrum_interval:
                raise ValueError(
                    "A monitor cannot use different DFT and power-spectrum record "
                    "intervals. Use matching intervals or separate monitors."
                )
        if dft_active:
            freq_points = dft_freqs
            freq_interval = dft_interval
        else:
            freq_points = flux_freqs
            freq_interval = power_spectrum_interval
        theta = 2.0 * np.pi * freq_points * float(dt) * float(freq_interval)
        freq_rot_re = np.cos(theta).astype(np.float32, copy=False)
        freq_rot_im = np.sin(theta).astype(np.float32, copy=False)
        dft_window = str(getattr(monitor, "dft_window", "rect")).lower()
        if dft_window in {"none", "rectangular"}:
            dft_window = "rect"
        dft_window_code = 1 if dft_window == "hann" else 0
        dft_normalization = str(getattr(monitor, "dft_normalization", "native")).lower()
        dft_normalization_code = 1 if dft_normalization == "physical" else 0
        dft_length_unit = float(getattr(monitor, "dft_length_unit", 1e-6))
        dft_t_end_val = float(
            np.inf
            if getattr(monitor, "dft_t_end", None) is None
            else getattr(monitor, "dft_t_end")
        )
        if dft_window_code == 1 and not np.isfinite(dft_t_end_val):
            dft_window_code = 0
        dft_components = getattr(monitor, "dft_components", None)
        if dft_components is None:
            dft_component_mask = np.ones((6,), dtype=np.float32)
        else:
            wanted = {str(c) for c in dft_components}
            ordered = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            dft_component_mask = np.asarray(
                [1.0 if c in wanted else 0.0 for c in ordered], dtype=np.float32
            )

        if not monitor.is_3d:
            points = monitor.get_grid_points_2d(resolution, resolution)
            line_coords = monitor._line_sample_coords_2d(resolution, resolution)
            if points:
                x_raw = np.asarray([p[0] for p in points], dtype=np.int32)
                y_raw = np.asarray([p[1] for p in points], dtype=np.int32)
            else:
                x_raw = np.zeros((0,), dtype=np.int32)
                y_raw = np.zeros((0,), dtype=np.int32)
            if line_coords is not None:
                dft_target_x = np.asarray(line_coords[0], dtype=np.float32).reshape(-1)
                dft_target_y = np.asarray(line_coords[1], dtype=np.float32).reshape(-1)
            else:
                dft_target_x = np.zeros((0,), dtype=np.float32)
                dft_target_y = np.zeros((0,), dtype=np.float32)
            dft_centered_tm_xy_sampling = bool(
                dft_normalization_code == 1 and dft_target_x.size > 0
            )

            x_ex, y_ex, v_ex = _clip_indices(x_raw, y_raw, tuple(fields.Ex.shape))
            x_ey, y_ey, v_ey = _clip_indices(x_raw, y_raw, tuple(fields.Ey.shape))
            x_ez, y_ez, v_ez = _clip_indices(x_raw, y_raw, tuple(fields.Ez.shape))
            x_hx, y_hx, v_hx = _clip_indices(x_raw, y_raw, tuple(fields.Hx.shape))
            x_hy, y_hy, v_hy = _clip_indices(x_raw, y_raw, tuple(fields.Hy.shape))
            x_hz, y_hz, v_hz = _clip_indices(x_raw, y_raw, tuple(fields.Hz.shape))
            interp_2d = {
                name: _compile_monitor_2d_interpolation(
                    monitor, name, fields, resolution
                )
                for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            }
            normal_axis, normal_sign = _monitor_normal_2d(monitor, resolution)
            power_scale = _line_integral_scale_2d(
                "x" if normal_axis == 0 else "y",
                resolution,
                resolution,
            )

            specs.append(
                CompiledMonitorSpec(
                    name=monitor.name or f"monitor_{mon_idx}",
                    monitor_index=mon_idx,
                    is_3d=False,
                    record_interval=interval,
                    accumulate_power=bool(monitor.accumulate_power),
                    power_scale=float(power_scale),
                    normal_axis=normal_axis,
                    normal_sign=normal_sign,
                    accumulate_frequency=bool(flux_freqs.size > 0),
                    freq_record_interval=freq_interval,
                    freq_count=int(freq_points.size),
                    freq_hz=jnp.asarray(freq_points.astype(np.float32, copy=False)),
                    freq_rot_re=jnp.asarray(freq_rot_re),
                    freq_rot_im=jnp.asarray(freq_rot_im),
                    dft_enabled=bool(dft_enabled and dft_freqs.size > 0),
                    dft_record_interval=freq_interval,
                    dft_t_start=float(getattr(monitor, "dft_t_start", 0.0)),
                    dft_t_end=float(dft_t_end_val),
                    dft_window_code=dft_window_code,
                    dft_normalization_code=dft_normalization_code,
                    dft_length_unit=dft_length_unit,
                    dft_centered_tm_xy_sampling=dft_centered_tm_xy_sampling,
                    dft_point_count=int(
                        dft_target_x.size if dft_target_x.size > 0 else x_ez.size
                    ),
                    dft_component_mask=jnp.asarray(dft_component_mask),
                    dft_target_x=jnp.asarray(dft_target_x),
                    dft_target_y=jnp.asarray(dft_target_y),
                    x_ex=jnp.asarray(x_ex),
                    y_ex=jnp.asarray(y_ex),
                    valid_ex=jnp.asarray(v_ex),
                    x_ey=jnp.asarray(x_ey),
                    y_ey=jnp.asarray(y_ey),
                    valid_ey=jnp.asarray(v_ey),
                    x_ez=jnp.asarray(x_ez),
                    y_ez=jnp.asarray(y_ez),
                    valid_ez=jnp.asarray(v_ez),
                    x_hx=jnp.asarray(x_hx),
                    y_hx=jnp.asarray(y_hx),
                    valid_hx=jnp.asarray(v_hx),
                    x_hy=jnp.asarray(x_hy),
                    y_hy=jnp.asarray(y_hy),
                    valid_hy=jnp.asarray(v_hy),
                    x_hz=jnp.asarray(x_hz),
                    y_hz=jnp.asarray(y_hz),
                    valid_hz=jnp.asarray(v_hz),
                    ex_interp_flat_idx=interp_2d["Ex"][0],
                    ex_interp_weights=interp_2d["Ex"][1],
                    ey_interp_flat_idx=interp_2d["Ey"][0],
                    ey_interp_weights=interp_2d["Ey"][1],
                    ez_interp_flat_idx=interp_2d["Ez"][0],
                    ez_interp_weights=interp_2d["Ez"][1],
                    hx_interp_flat_idx=interp_2d["Hx"][0],
                    hx_interp_weights=interp_2d["Hx"][1],
                    hy_interp_flat_idx=interp_2d["Hy"][0],
                    hy_interp_weights=interp_2d["Hy"][1],
                    hz_interp_flat_idx=interp_2d["Hz"][0],
                    hz_interp_weights=interp_2d["Hz"][1],
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
            logical_shapes_3d = getattr(fields, "_logical_component_shapes", shape_3d)
            base_shape_3d = tuple(
                max(int(shape[axis]) for shape in logical_shapes_3d.values())
                for axis in range(3)
            )
            idx_map, _slice_dim0, _slice_dim1 = _compile_monitor_3d_indices(
                monitor,
                resolution,
                shape_3d,
            )
            target0, target1 = monitor.get_analysis_plane_coords_3d(
                dx=resolution,
                dy=resolution,
                dz=resolution,
                field_shape=base_shape_3d,
            )
            min_dim0 = int(np.asarray(target0).size)
            min_dim1 = int(np.asarray(target1).size)
            interp_map = {}
            interp_dims = {}
            for name, shape in shape_3d.items():
                flat_idx, weights, dim0, dim1 = _compile_monitor_3d_interpolation(
                    monitor,
                    name,
                    resolution,
                    base_shape_3d,
                    shape,
                )
                interp_map[name] = (flat_idx, weights)
                interp_dims[name] = (dim0, dim1)
            for name, (flat_idx, weights) in interp_map.items():
                dim0, dim1 = interp_dims[name]
                flat_idx, weights = _crop_monitor_3d_interpolation(
                    flat_idx,
                    weights,
                    dim0,
                    dim1,
                    min_dim0,
                    min_dim1,
                )
                interp_map[name] = (jnp.asarray(flat_idx), jnp.asarray(weights))
            power_scale = _analysis_plane_sample_area(
                np.asarray(target0, dtype=np.float64),
                np.asarray(target1, dtype=np.float64),
                float(resolution),
            )

            specs.append(
                CompiledMonitorSpec(
                    name=monitor.name or f"monitor_{mon_idx}",
                    monitor_index=mon_idx,
                    is_3d=True,
                    record_interval=interval,
                    accumulate_power=bool(monitor.accumulate_power),
                    power_scale=float(power_scale),
                    normal_axis={"x": 0, "y": 1, "z": 2}.get(
                        str(getattr(monitor, "plane_normal", "z")).lower(), -1
                    ),
                    normal_sign=1.0,
                    accumulate_frequency=bool(flux_freqs.size > 0),
                    freq_record_interval=freq_interval,
                    freq_count=int(freq_points.size),
                    freq_hz=jnp.asarray(freq_points.astype(np.float32, copy=False)),
                    freq_rot_re=jnp.asarray(freq_rot_re),
                    freq_rot_im=jnp.asarray(freq_rot_im),
                    dft_enabled=bool(dft_enabled and dft_freqs.size > 0),
                    dft_record_interval=freq_interval,
                    dft_t_start=float(getattr(monitor, "dft_t_start", 0.0)),
                    dft_t_end=float(dft_t_end_val),
                    dft_window_code=dft_window_code,
                    dft_normalization_code=dft_normalization_code,
                    dft_length_unit=dft_length_unit,
                    dft_point_count=int(min_dim0 * min_dim1),
                    dft_component_mask=jnp.asarray(dft_component_mask),
                    ex_idx=idx_map["Ex"],
                    ey_idx=idx_map["Ey"],
                    ez_idx=idx_map["Ez"],
                    hx_idx=idx_map["Hx"],
                    hy_idx=idx_map["Hy"],
                    hz_idx=idx_map["Hz"],
                    ex_interp_flat_idx=interp_map["Ex"][0],
                    ex_interp_weights=interp_map["Ex"][1],
                    ey_interp_flat_idx=interp_map["Ey"][0],
                    ey_interp_weights=interp_map["Ey"][1],
                    ez_interp_flat_idx=interp_map["Ez"][0],
                    ez_interp_weights=interp_map["Ez"][1],
                    hx_interp_flat_idx=interp_map["Hx"][0],
                    hx_interp_weights=interp_map["Hx"][1],
                    hy_interp_flat_idx=interp_map["Hy"][0],
                    hy_interp_weights=interp_map["Hy"][1],
                    hz_interp_flat_idx=interp_map["Hz"][0],
                    hz_interp_weights=interp_map["Hz"][1],
                    min_dim0=min_dim0,
                    min_dim1=min_dim1,
                )
            )

    return tuple(specs), max_records
