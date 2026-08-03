"""Compile monitor specs into static packed monitor specs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from beamz.devices._placement import (
    line_region_points,
    snap_axis_aligned_line_region_grid,
    snap_plane_region_grid,
)
from beamz.devices.monitors.monitors import (
    FieldRecorder,
    ModeMonitor,
    _line_integral_scale_2d,
    _line_normal_2d,
    _Monitor,
)
from beamz.lattice import (
    common_grid_shape_3d,
    compile_yee_plane_quadrature_3d,
    linear_interpolation_plan,
)


def _empty_array() -> jnp.ndarray:
    return jnp.empty(0, dtype=jnp.float32)


@dataclass(frozen=True, kw_only=True)
class CompiledMonitorSpec:
    """Canonical spatial sampling, reduction, and accumulation plan."""

    name: str
    monitor_index: int
    record_interval: int
    accumulate_power: bool
    power_scale: float
    normal_axis: int = -1
    normal_sign: float = 1.0
    accumulate_frequency: bool = False
    freq_record_interval: int = 1
    freq_count: int = 0
    freq_hz: jnp.ndarray = field(default_factory=_empty_array)
    freq_rot_re: jnp.ndarray = field(default_factory=_empty_array)
    freq_rot_im: jnp.ndarray = field(default_factory=_empty_array)
    dft_enabled: bool = False
    dft_record_interval: int = 1
    dft_t_start: float = 0.0
    dft_t_end: float = np.inf
    dft_window_code: int = 0  # 0=rect, 1=hann
    dft_normalization_code: int = 0  # 0=native, 1=physical
    dft_length_unit: float = 1e-6
    dft_point_count: int = 0
    dft_component_mask: jnp.ndarray = field(default_factory=_empty_array)
    sample_flat_idx: tuple[jnp.ndarray, ...] = ()
    sample_weights: tuple[jnp.ndarray, ...] = ()
    dft_flat_idx: tuple[jnp.ndarray, ...] = ()
    dft_weights: tuple[jnp.ndarray, ...] = ()
    integration_weights: jnp.ndarray = field(default_factory=_empty_array)
    recorder_index: int = -1
    components: tuple[str, ...] = ()
    canonical_components: tuple[str, ...] = ()
    component_signs: tuple[float, ...] = ()
    field_buffer_indices: tuple[int, ...] = ()
    field_shapes: tuple[tuple[int, ...], ...] = ()
    field_interp_flat_idx: tuple[jnp.ndarray, ...] = ()
    field_interp_weights: tuple[jnp.ndarray, ...] = ()


def _clip_indices(x_idx: np.ndarray, y_idx: np.ndarray, shape: tuple[int, int]):
    h, w = shape
    valid = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
    x = np.clip(x_idx, 0, max(w - 1, 0)).astype(np.int32)
    y = np.clip(y_idx, 0, max(h - 1, 0)).astype(np.int32)
    return x, y, valid.astype(np.float32)


def _monitor_normal_2d(
    monitor: _Monitor, resolution: float, snapped=None
) -> tuple[int, float]:
    snapped = snapped or monitor.get_snapped_region(dx=resolution, dy=resolution)
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
    monitor: _Monitor,
    component: str,
    fields,
    resolution: float,
    field_shape=None,
    grid=None,
    region=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    from beamz.lattice import component_coordinates_2d_um

    if grid is not None and region is not None:
        if region.normal_axis == "x":
            interval = region.axis_interval("y")
            dst_y_values = np.asarray(grid.centers("y"))[
                int(interval.start) : int(interval.stop)
            ]
            line_coords = (
                np.full(dst_y_values.shape, float(region.plane_coord)),
                dst_y_values,
            )
        else:
            interval = region.axis_interval("x")
            dst_x_values = np.asarray(grid.centers("x"))[
                int(interval.start) : int(interval.stop)
            ]
            line_coords = (
                dst_x_values,
                np.full(dst_x_values.shape, float(region.plane_coord)),
            )
    else:
        line_coords = monitor._line_sample_coords_2d(
            resolution, resolution, field_shape=field_shape
        )
    if line_coords is None:
        return _empty_array(), _empty_array()

    plane = getattr(fields, "plane_2d", "xy")
    if grid is not None and plane == "xy":
        offsets = {
            "Ez": (0.0, 0.0),
            "Hx": (0.5, 0.0),
            "Hy": (0.0, 0.5),
            "Ex": (0.0, 0.5),
            "Ey": (0.5, 0.0),
            "Hz": (0.5, 0.5),
        }[component]
        coords = {
            "y": grid.y_edges if offsets[0] == 0.0 else grid.centers("y"),
            "x": grid.x_edges if offsets[1] == 0.0 else grid.centers("x"),
        }
    else:
        coords = component_coordinates_2d_um(
            component,
            tuple(int(v) for v in fields.permittivity.shape),
            float(resolution),
            plane,
            getattr(fields, "polarization_2d", "tm"),
        )
    if "x" not in coords or "y" not in coords:
        return _empty_array(), _empty_array()

    src_x = np.asarray(coords["x"], dtype=np.float64)
    src_y = np.asarray(coords["y"], dtype=np.float64)
    dst_x = np.asarray(line_coords[0], dtype=np.float64).reshape(-1)
    dst_y = np.asarray(line_coords[1], dtype=np.float64).reshape(-1)
    if dst_x.size != dst_y.size:
        return _empty_array(), _empty_array()

    x0, x1, wx0, wx1 = linear_interpolation_plan(src_x, dst_x)
    y0, y1, wy0, wy1 = linear_interpolation_plan(src_y, dst_y)
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


def _direct_sampling_plan(x, y, valid, shape):
    """Convert clipped lattice points into the same weighted-gather form as interpolation."""
    flat = np.asarray(np.ravel_multi_index((y, x), dims=shape), dtype=np.int32).reshape(
        -1, 1
    )
    return jnp.asarray(flat), jnp.asarray(valid, dtype=jnp.float32)[:, None]


def _inactive_sampling_plan(point_count: int):
    """Represent an inactive canonical component without a runtime dimension branch."""
    shape = (int(point_count), 1)
    return jnp.zeros(shape, dtype=jnp.int32), jnp.zeros(shape, dtype=jnp.float32)


def compile_monitor_specs(
    monitors: tuple[_Monitor, ...],
    fields,
    resolution: float,
    num_steps: int,
    dt: float,
    plane_2d: str = "xy",
    polarization_2d: str = "tm",
    grid=None,
) -> tuple[tuple[CompiledMonitorSpec, ...], int]:
    """Compile request monitor specs into packed monitor descriptors.

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
    recorder_count = 0
    field_buffer_count = 0
    active_grid = grid if grid is not None else getattr(fields, "geometry", None)
    is_3d_grid = np.asarray(fields.permittivity).ndim == 3
    use_geometry = active_grid is not None and (
        active_grid.metric_kind_for(("x", "y", "z") if is_3d_grid else ("x", "y"))
        != "isotropic_uniform"
        or active_grid.origin != (0.0, 0.0, 0.0)
    )

    for mon_idx, monitor in enumerate(monitors):
        if not isinstance(monitor, _Monitor):
            raise TypeError(f"Unsupported monitor object {type(monitor).__name__!s}.")
        is_3d = np.asarray(fields.permittivity).ndim == 3
        if (
            not is_3d
            and isinstance(monitor, ModeMonitor)
            and monitor.mode_spec.polarization not in {None, polarization_2d}
        ):
            raise ValueError(
                f"ModeMonitor polarization {monitor.mode_spec.polarization!r} does "
                f"not match the Simulation polarization {polarization_2d!r}."
            )
        interval = max(1, int(monitor.interval))
        records = int(math.ceil(num_steps / interval))
        max_records = max(max_records, records)

        if isinstance(monitor, FieldRecorder):
            public_components, canonical_components, signs = [], [], []
            for component in monitor.components:
                canonical, sign = component, 1.0
                if fields.permittivity.ndim != 3:
                    from beamz.lattice import (
                        canonical_component_2d,
                        public_component_2d,
                    )

                    canonical = canonical_component_2d(
                        component, plane_2d, polarization_2d
                    )
                    if canonical is None:
                        raise ValueError(
                            f"FieldRecorder component {component!r} is inactive "
                            f"for the {plane_2d!r} 2D plane."
                        )
                    _, sign = public_component_2d(canonical, plane_2d, polarization_2d)
                public_components.append(component)
                canonical_components.append(canonical)
                signs.append(float(sign))

            shapes: list[tuple[int, ...]] = []
            interp_indices: list[jnp.ndarray] = []
            interp_weights: list[jnp.ndarray] = []
            if monitor.region == "domain":
                logical_shapes = getattr(fields, "_logical_component_shapes", {})
                for canonical in canonical_components:
                    shapes.append(
                        tuple(
                            logical_shapes.get(
                                canonical, getattr(fields, canonical).shape
                            )
                        )
                    )
                    interp_indices.append(_empty_array())
                    interp_weights.append(_empty_array())
            elif fields.permittivity.ndim != 3:
                for canonical in canonical_components:
                    flat_idx, weights = _compile_monitor_2d_interpolation(
                        monitor,
                        canonical,
                        fields,
                        resolution,
                        field_shape=fields.permittivity.shape,
                        grid=active_grid if use_geometry else None,
                        region=(
                            snap_axis_aligned_line_region_grid(
                                monitor.start, monitor.end, active_grid
                            )
                            if use_geometry
                            else None
                        ),
                    )
                    if not flat_idx.size:
                        raise ValueError(
                            "A 2D FieldRecorder slice must be a non-empty "
                            "axis-aligned line."
                        )
                    shapes.append((int(flat_idx.shape[0]),))
                    interp_indices.append(flat_idx)
                    interp_weights.append(weights)
            else:
                field_shapes = {
                    component: tuple(getattr(fields, component).shape)
                    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
                }
                base_shape = common_grid_shape_3d(fields)
                region = (
                    snap_plane_region_grid(
                        center=monitor.center,
                        size=monitor.size,
                        plane_normal=monitor.plane_normal,
                        grid=active_grid,
                    )
                    if use_geometry
                    else monitor.get_snapped_region(
                        dx=resolution,
                        dy=resolution,
                        dz=resolution,
                        field_shape=base_shape,
                    )
                )
                quadrature = compile_yee_plane_quadrature_3d(
                    center=monitor.center,
                    size=monitor.size,
                    normal_axis=monitor.plane_normal,
                    region=region,
                    resolution=resolution,
                    grid_shape=base_shape,
                    component_shapes=field_shapes,
                    grid=active_grid if use_geometry else None,
                )
                output_shape = tuple(
                    int(values.size) for values in quadrature.coordinates
                )
                for canonical in canonical_components:
                    flat_idx, weights = quadrature.plan(canonical)
                    shapes.append(output_shape)
                    interp_indices.append(jnp.asarray(flat_idx))
                    interp_weights.append(jnp.asarray(weights))
            buffer_indices = tuple(
                range(field_buffer_count, field_buffer_count + len(shapes))
            )
            specs.append(
                CompiledMonitorSpec(
                    name=monitor.name or f"monitor_{mon_idx}",
                    monitor_index=mon_idx,
                    record_interval=interval,
                    accumulate_power=False,
                    power_scale=0.0,
                    recorder_index=recorder_count,
                    components=tuple(public_components),
                    canonical_components=tuple(canonical_components),
                    component_signs=tuple(signs),
                    field_buffer_indices=buffer_indices,
                    field_shapes=tuple(shapes),
                    field_interp_flat_idx=tuple(interp_indices),
                    field_interp_weights=tuple(interp_weights),
                )
            )
            recorder_count += 1
            field_buffer_count += len(shapes)
            continue

        dft_freqs = np.asarray(monitor.freqs, dtype=np.float64).ravel()
        if dft_freqs.size > 0 and not np.all(np.isfinite(dft_freqs)):
            raise ValueError("Monitor frequencies must be finite values in Hz")
        if dft_freqs.size > 0 and np.any(dft_freqs <= 0.0):
            raise ValueError("Monitor frequencies must be strictly positive")

        freq_points = dft_freqs
        freq_interval = monitor.interval
        theta = 2.0 * np.pi * freq_points * float(dt) * float(freq_interval)
        freq_rot_re = np.cos(theta).astype(np.float32, copy=False)
        freq_rot_im = np.sin(theta).astype(np.float32, copy=False)
        dft_t_end_val = np.inf
        dft_components = getattr(monitor, "dft_components", None)
        if dft_components is None:
            dft_component_mask = np.ones((6,), dtype=np.float32)
        else:
            wanted = {str(c) for c in dft_components}
            if not is_3d:
                from beamz.lattice import canonical_component_2d

                wanted = {
                    canonical
                    for component in wanted
                    if (
                        canonical := canonical_component_2d(
                            component, plane_2d, polarization_2d
                        )
                    )
                    is not None
                }
            ordered = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            dft_component_mask = np.asarray(
                [1.0 if c in wanted else 0.0 for c in ordered], dtype=np.float32
            )
        common: dict[str, Any] = dict(
            name=monitor.name or f"monitor_{mon_idx}",
            monitor_index=mon_idx,
            record_interval=interval,
            accumulate_power=False,
            accumulate_frequency=False,
            freq_record_interval=freq_interval,
            freq_count=int(freq_points.size),
            freq_hz=jnp.asarray(freq_points, dtype=jnp.float32),
            freq_rot_re=jnp.asarray(freq_rot_re),
            freq_rot_im=jnp.asarray(freq_rot_im),
            dft_enabled=True,
            dft_record_interval=freq_interval,
            dft_t_start=0.0,
            dft_t_end=dft_t_end_val,
            dft_window_code=0,
            dft_normalization_code=0,
            dft_length_unit=1e-6,
            dft_component_mask=jnp.asarray(dft_component_mask),
        )

        if not is_3d:
            region_2d = (
                snap_axis_aligned_line_region_grid(
                    monitor.start, monitor.end, active_grid
                )
                if use_geometry
                else None
            )
            points = (
                line_region_points(region_2d)
                if region_2d is not None
                else monitor.get_grid_points_2d(resolution, resolution)
            )
            if points:
                x_raw = np.asarray([p[0] for p in points], dtype=np.int32)
                y_raw = np.asarray([p[1] for p in points], dtype=np.int32)
            else:
                x_raw = np.zeros((0,), dtype=np.int32)
                y_raw = np.zeros((0,), dtype=np.int32)
            active_components = (
                ("Ez", "Hx", "Hy") if polarization_2d == "tm" else ("Ex", "Ey", "Hz")
            )
            component_plans = {}
            for name in active_components:
                shape = tuple(getattr(fields, name).shape)
                x, y, valid = _clip_indices(x_raw, y_raw, shape)
                flat_idx, weights = _compile_monitor_2d_interpolation(
                    monitor,
                    name,
                    fields,
                    resolution,
                    grid=active_grid if use_geometry else None,
                    region=region_2d,
                )
                component_plans[name] = (
                    (flat_idx, weights)
                    if flat_idx.size
                    else _direct_sampling_plan(x, y, valid, shape)
                )
            point_count = int(next(iter(component_plans.values()))[0].shape[0])
            inactive = _inactive_sampling_plan(point_count)
            sample_plans = tuple(
                component_plans.get(name, inactive)
                for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            )
            dft_plans = sample_plans
            normal_axis, normal_sign = _monitor_normal_2d(
                monitor, resolution, region_2d
            )
            if use_geometry and region_2d is not None:
                tangential = "y" if normal_axis == 0 else "x"
                interval = region_2d.axis_interval(tangential)
                integration_weights = active_grid.cell_widths(tangential)[
                    int(interval.start) : int(interval.stop)
                ]
                power_scale = 1.0
            else:
                integration_weights = np.empty((0,), dtype=np.float32)
                power_scale = _line_integral_scale_2d(
                    "x" if normal_axis == 0 else "y",
                    resolution,
                    resolution,
                )

            specs.append(
                CompiledMonitorSpec(
                    **common,
                    power_scale=float(power_scale),
                    normal_axis=normal_axis,
                    normal_sign=normal_sign,
                    dft_point_count=point_count,
                    sample_flat_idx=tuple(plan[0] for plan in sample_plans),
                    sample_weights=tuple(plan[1] for plan in sample_plans),
                    dft_flat_idx=tuple(plan[0] for plan in dft_plans),
                    dft_weights=tuple(plan[1] for plan in dft_plans),
                    integration_weights=jnp.asarray(
                        integration_weights, dtype=jnp.float32
                    ),
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
            base_shape_3d = common_grid_shape_3d(fields)
            region = (
                snap_plane_region_grid(
                    center=monitor.center,
                    size=monitor.size,
                    plane_normal=monitor.plane_normal,
                    grid=active_grid,
                )
                if use_geometry
                else monitor.get_snapped_region(
                    dx=resolution,
                    dy=resolution,
                    dz=resolution,
                    field_shape=base_shape_3d,
                )
            )
            quadrature = compile_yee_plane_quadrature_3d(
                center=monitor.center,
                size=monitor.size,
                normal_axis=monitor.plane_normal,
                region=region,
                resolution=resolution,
                grid_shape=base_shape_3d,
                component_shapes=shape_3d,
                grid=active_grid if use_geometry else None,
            )
            point_count = quadrature.point_count
            sample_plans = []
            dft_plans = []
            for component_index, name in enumerate(shape_3d):
                needs_dft = bool(dft_component_mask[component_index] > 0.0)
                if needs_dft:
                    flat_idx, weights = quadrature.plan(name)
                    plan = (jnp.asarray(flat_idx), jnp.asarray(weights))
                else:
                    plan = _inactive_sampling_plan(point_count)
                inactive = _inactive_sampling_plan(point_count)
                sample_plans.append(inactive)
                dft_plans.append(plan if needs_dft else inactive)
            specs.append(
                CompiledMonitorSpec(
                    **common,
                    power_scale=quadrature.sample_area,
                    normal_axis=quadrature.normal_axis,
                    normal_sign=1.0,
                    dft_point_count=point_count,
                    sample_flat_idx=tuple(plan[0] for plan in sample_plans),
                    sample_weights=tuple(plan[1] for plan in sample_plans),
                    dft_flat_idx=tuple(plan[0] for plan in dft_plans),
                    dft_weights=tuple(plan[1] for plan in dft_plans),
                    integration_weights=jnp.asarray(
                        quadrature.integration_weights, dtype=jnp.float32
                    ),
                )
            )

    return tuple(specs), max_records
