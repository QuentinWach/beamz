"""Geometry and phase helpers for modal projection."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from beamz.analysis.data import AnalysisData, static_fields
from beamz.const import LIGHT_SPEED, µm
from beamz.devices._placement import snap_plane_region_grid
from beamz.lattice import (
    component_coordinates_3d_um,
    component_coordinates_rectilinear,
    plane_sample_area,
    yee_plane_coordinates_3d,
)


def _grid_shape(sim):
    if isinstance(sim, AnalysisData):
        return sim.coordinates.fields.grid_shape
    fields = static_fields(sim)
    shape = getattr(fields, "grid_shape", None)
    if shape is not None:
        return tuple(int(v) for v in shape)
    return tuple(int(v) for v in np.asarray(fields.permittivity).shape)


def _analysis_grid(sim):
    if isinstance(sim, AnalysisData):
        return sim.coordinates.grid
    try:
        fields = static_fields(sim)
    except TypeError:
        return None
    return getattr(fields, "geometry", None)


def _monitor_projection_phase(component, frequencies, dt):
    """Phase-align raw monitor phasors to the E-field sample time.

    BeamZ's DFT uses the phasor convention

        f(t) = Re{F exp(-i omega t)}
        F ~= 2 sum_t f(t) exp(+i omega t) / sum_t 1

    Monitors are sampled after the E update at timestamp T = t + dt. At
    that instant E is stored at T, while the leapfrog H fields are stored at
    T - dt/2. If a component is actually sampled at T + tau but accumulated
    with exp(+i omega T), the accumulator returns F exp(-i omega tau). To
    recover the common-time modal phasor F, multiply by exp(+i omega tau).
    Therefore E has tau = 0 and H has tau = -dt/2.
    """
    freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
    comp = str(component)
    if comp.startswith("H"):
        return np.exp(-1j * np.pi * freq_arr * float(dt))
    return np.ones_like(freq_arr, dtype=np.complex128)


def _modal_projection_spatial_phase(component, frequencies, plane_delay_s):
    """Phase-align E components from their Yee plane to the H-referenced mode.

    Mode profiles are gauged to the dominant H component, matching the
    ModeSource launch convention. After the temporal Yee correction, E
    samples still need the spatial propagation phase from the E Yee plane
    to that H reference plane.
    """
    freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
    comp = str(component)
    delay = float(plane_delay_s)
    if comp.startswith("E") and delay != 0.0:
        return np.exp(1j * 2.0 * np.pi * freq_arr * delay)
    return np.ones_like(freq_arr, dtype=np.complex128)


def _modal_projection_plane_delay_s(sim, spec, frequency, mode_neff):
    """Return the E-to-H modal-plane delay used by S-parameter projection."""
    if getattr(sim, "is_3d", False):
        # 3D monitors interpolate every recorded component onto the same
        # physical analysis plane. There is no remaining normal-direction
        # Yee half-cell offset to compensate during modal extraction.
        return 0.0
    freq = float(frequency)
    neff = float(np.real(np.asarray(mode_neff)))
    if (not np.isfinite(freq)) or freq <= 0.0:
        return 0.0
    if (not np.isfinite(neff)) or neff <= 0.0:
        return 0.0
    grid = _analysis_grid(sim)
    if grid is not None and grid.metric_kind_for(("x", "y")) != "isotropic_uniform":
        axis = str(getattr(spec, "axis", "x"))
        center = tuple(float(value) for value in getattr(spec, "center", (0.0, 0.0, 0.0)))
        coordinate = center[{"x": 0, "y": 1, "z": 2}[axis]]
        index = int(np.argmin(np.abs(np.asarray(grid.centers(axis)) - coordinate)))
        d_axis = float(grid.cell_widths(axis)[index])
    else:
        d_axis = float(getattr(sim, "resolution", 0.0) or 0.0)
    if (not np.isfinite(d_axis)) or d_axis <= 0.0:
        return 0.0

    direction_sign = +1.0
    delta_s = direction_sign * 0.5 * d_axis
    return float(delta_s * neff / LIGHT_SPEED)


def _apply_modal_projection_spatial_phase(component, values, frequency, projection):
    phase = _modal_projection_spatial_phase(
        component,
        np.asarray([float(frequency)], dtype=float),
        float(projection.get("modal_plane_delay_s", 0.0)),
    )[0]
    return np.asarray(values, dtype=np.complex128) * phase


def _mode_components_for_port(spec):
    axis = spec.axis
    tm_map = {
        "x": ("Ez", "Hy", 2, 1, -1.0),
        "y": ("Ez", "Hx", 2, 0, 1.0),
        "z": ("Ey", "Hx", 1, 0, -1.0),
    }
    te_map = {
        "x": ("Ey", "Hz", 1, 2, 1.0),
        "y": ("Ex", "Hz", 0, 2, -1.0),
        "z": ("Ex", "Hy", 0, 1, 1.0),
    }
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported port axis '{axis}'.")
    e_comp, h_comp, e_idx, h_idx, sign = (
        tm_map[axis] if spec.polarization == "tm" else te_map[axis]
    )
    proj_components_3d = {
        "x": ("Ey", "Ez", "Hy", "Hz"),
        "y": ("Ex", "Ez", "Hx", "Hz"),
        "z": ("Ex", "Ey", "Hx", "Hy"),
    }[axis]
    return {
        "axis": axis,
        "e_component": e_comp,
        "h_component": h_comp,
        "e_mode_index": e_idx,
        "h_mode_index": h_idx,
        "signed_flux_sign": sign,
        "projection_components": (e_comp, h_comp),
        "projection_components_3d": proj_components_3d,
    }


def _plane_axes_for_port_axis(axis: str) -> tuple[str, str]:
    axis = str(axis).lower()
    mapping = {
        "x": ("z", "y"),
        "y": ("z", "x"),
        "z": ("y", "x"),
    }
    try:
        return mapping[axis]
    except KeyError as exc:
        raise ValueError(f"Unsupported port axis {axis!r}.") from exc


def _analysis_plane_sample_area(coord0, coord1, fallback_step: float) -> float:
    return plane_sample_area((coord0, coord1), fallback_step)


def _clamp_monitor_grid_index(idx, limit):
    if isinstance(idx, slice):
        start = 0 if idx.start is None else int(idx.start)
        stop = limit if idx.stop is None else int(idx.stop)
        start = max(0, min(start, max(limit - 1, 0)))
        stop = max(start + 1, min(stop, limit))
        return slice(start, stop)
    return max(0, min(int(idx), limit - 1))


def _component_shape(sim, component):
    if isinstance(sim, AnalysisData):
        return sim.coordinates.fields.component_shapes[component]
    fields = static_fields(sim)
    shapes = getattr(fields, "component_shapes", None)
    if shapes is not None:
        return tuple(int(v) for v in shapes[component])
    return tuple(int(v) for v in np.asarray(getattr(fields, component)).shape)


def _monitor_common_plane_shape_3d(sim, monitor) -> tuple[int, int]:
    dims = []
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        shape = _component_shape(sim, component)
        indices = monitor.get_grid_slice_3d(
            sim.resolution, sim.resolution, sim.resolution, shape
        )
        indices = tuple(
            _clamp_monitor_grid_index(index, limit)
            for index, limit in zip(indices, shape, strict=True)
        )
        sample = np.asarray(np.zeros(shape, dtype=np.float32)[indices])
        dims.append(tuple(int(v) for v in np.atleast_2d(sample).shape[:2]))
    return int(min(dim[0] for dim in dims)), int(min(dim[1] for dim in dims))


def _monitor_analysis_plane_3d(
    sim,
    monitor,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    grid = _analysis_grid(sim)
    if grid is not None and grid.metric_kind != "isotropic_uniform":
        snapped = snap_plane_region_grid(
            center=monitor.center,
            size=monitor.size,
            plane_normal=axis,
            grid=grid,
        )
        return tuple(
            np.asarray(values, dtype=np.float64)
            for values in yee_plane_coordinates_3d(
                monitor.center, monitor.size, axis, snapped, grid=grid
            )
        )
    coordinate_method = getattr(monitor, "get_analysis_plane_coords_3d", None)
    if callable(coordinate_method):
        coord0, coord1 = cast(Any, coordinate_method)(
            dx=sim.resolution,
            dy=sim.resolution,
            dz=sim.resolution,
            field_shape=_grid_shape(sim),
        )
        return np.asarray(coord0, dtype=np.float64), np.asarray(
            coord1, dtype=np.float64
        )

    snapped = monitor.get_snapped_region(
        dx=sim.resolution,
        dy=sim.resolution,
        dz=sim.resolution,
        field_shape=_grid_shape(sim),
    )
    if snapped is None:
        raise ValueError(f"Monitor '{monitor.name}' has no snapped 3D region.")
    axis0, axis1 = _plane_axes_for_port_axis(axis)
    interval0, interval1 = snapped.axis_interval(axis0), snapped.axis_interval(axis1)
    if interval0 is None or interval1 is None:
        raise ValueError(
            f"Monitor '{monitor.name}' is missing tangential intervals for axis '{axis}'."
        )
    coord0 = (
        np.arange(int(interval0.start), int(interval0.stop), dtype=np.float64) + 0.5
    ) * float(sim.resolution)
    coord1 = (
        np.arange(int(interval1.start), int(interval1.stop), dtype=np.float64) + 0.5
    ) * float(sim.resolution)
    common0, common1 = _monitor_common_plane_shape_3d(sim, monitor)
    return coord0[:common0], coord1[:common1]


def _monitor_component_plane_coords_3d(
    sim,
    monitor,
    component: str,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    grid = _analysis_grid(sim)
    if grid is not None and grid.metric_kind != "isotropic_uniform":
        return _monitor_analysis_plane_3d(sim, monitor, axis)
    coordinate_method = getattr(monitor, "get_analysis_plane_coords_3d", None)
    if callable(coordinate_method):
        return _monitor_analysis_plane_3d(sim, monitor, axis)

    field_shape = _component_shape(sim, component)
    indices = monitor.get_grid_slice_3d(
        sim.resolution, sim.resolution, sim.resolution, field_shape
    )
    indices = tuple(
        _clamp_monitor_grid_index(index, limit)
        for index, limit in zip(indices, field_shape, strict=True)
    )
    if grid is None:
        coords = {
            name: values * float(µm)
            for name, values in component_coordinates_3d_um(
                component, _grid_shape(sim), float(sim.resolution / µm)
            ).items()
        }
    else:
        coords = component_coordinates_rectilinear(component, grid)
    axis0, axis1 = _plane_axes_for_port_axis(axis)
    axis_slices: dict[str, Any] = dict(zip(("z", "y", "x"), indices, strict=True))
    common0, common1 = _monitor_common_plane_shape_3d(sim, monitor)
    coord0 = np.asarray(coords[axis0][axis_slices[axis0]], dtype=np.float64)
    coord1 = np.asarray(coords[axis1][axis_slices[axis1]], dtype=np.float64)
    return coord0[:common0], coord1[:common1]
