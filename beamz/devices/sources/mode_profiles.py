"""Shared modal profile and discrete-mode solver helpers."""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import replace

import numpy as np

from beamz.devices._placement import snap_centered_extent
from beamz.devices.modes.discrete import (
    AxisName,
    DiscreteMode,
    ModePlaneSpec,
    solve_beamz_mode,
)
from beamz.lattice import (
    component_shape_3d,
    sample_voxel_grid_at_component_3d,
    sample_voxel_grid_at_e_component_3d_centered,
)

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = {"+x", "-x", "+y", "-y", "+z", "-z"}
_AXIS_POS_3D = {"z": 0, "y": 1, "x": 2}
_TRANSVERSE_AXES_3D: dict[str, tuple[AxisName, AxisName]] = {
    "x": ("z", "y"),
    "y": ("z", "x"),
    "z": ("y", "x"),
}
_MODE_PLANE_APERTURE_PAD_CELLS = 2
_MODE_PLANE_APERTURE_WINDOW_ALPHA = 0.2


def _mode_plane_outer_pad_cells(width, height, resolution) -> int:
    """Return the shared finite-domain padding for 3D modal solves."""
    extent = max(float(width), float(width if height is None else height))
    res = max(float(resolution), 1e-30)
    return int(np.clip(np.ceil(0.5 * extent / res), 8, 48))


def _to_real_profile(profile, imag_ratio_warn=1e-2, eps=1e-30):
    """Project profile to real-valued injection coefficients."""
    arr = np.asarray(profile, dtype=np.complex128)
    re = np.real(arr)
    im = np.imag(arr)
    re_peak = float(np.max(np.abs(re))) if re.size else 0.0
    im_peak = float(np.max(np.abs(im))) if im.size else 0.0
    if re_peak > eps and im_peak / re_peak > imag_ratio_warn:
        logger.debug(
            "Mode profile has non-negligible imaginary content before real projection: "
            "imag/real peak ratio=%.3e",
            im_peak / re_peak,
        )
    return re


def _shift_component_indices_along_axis(indices, axis, shift, field_shape):
    """Shift a component support tuple by integer cells along the propagation axis."""
    if indices is None:
        return None
    axis_pos = _AXIS_POS_3D[axis]
    out = list(indices)
    plane_idx = out[axis_pos]
    if isinstance(plane_idx, slice):
        return None
    plane_new = int(plane_idx) + int(shift)
    if plane_new < 0 or plane_new >= int(field_shape[axis_pos]):
        return None
    out[axis_pos] = plane_new
    return tuple(out)


def _axis_counts_from_grid_shape(grid_shape):
    nz, ny, nx = (int(v) for v in grid_shape)
    return {"z": nz, "y": ny, "x": nx}


def _center_by_axis(center, grid_shape, resolution):
    counts = _axis_counts_from_grid_shape(grid_shape)
    values = tuple(float(v) for v in center)
    return {
        "x": values[0] if len(values) > 0 else 0.5 * counts["x"] * float(resolution),
        "y": values[1] if len(values) > 1 else 0.5 * counts["y"] * float(resolution),
        "z": values[2] if len(values) > 2 else 0.5 * counts["z"] * float(resolution),
    }


def _source_extent_by_axis(axis, width, height):
    height = float(width if height is None else height)
    width = float(width)
    if axis == "x":
        return {"y": width, "z": height}
    if axis == "y":
        return {"x": width, "z": height}
    if axis == "z":
        return {"x": width, "y": height}
    raise ValueError(f"Unsupported mode-source axis {axis!r}")


def _ensure_min_interval(start, stop, limit, min_cells=2):
    start = max(0, min(int(start), int(limit)))
    stop = max(start, min(int(stop), int(limit)))
    need = max(1, int(min_cells))
    while stop - start < need and (start > 0 or stop < int(limit)):
        if start > 0:
            start -= 1
        if stop - start >= need:
            break
        if stop < int(limit):
            stop += 1
    return start, stop


def _local_mode_plane_spec(
    eps_profile,
    *,
    axis,
    grid_shape,
    center,
    width,
    height,
    plane_index,
    offset_index,
    resolution,
    snapped_region=None,
    aperture_pad_cells=2,
    material_origin_zyx=(0, 0, 0),
):
    """Build a compact local mode plane and metadata to shift it globally."""
    axis = str(axis).lower()
    counts = _axis_counts_from_grid_shape(grid_shape)
    centers = _center_by_axis(center, grid_shape, resolution)
    if snapped_region is not None:
        for name in ("x", "y", "z"):
            with suppress(Exception):
                centers[name] = float(snapped_region.axis_coord(name))

    origin = {"z": 0, "y": 0, "x": 0}
    material_origin: dict[str, int] = dict(
        zip(("z", "y", "x"), (int(value) for value in material_origin_zyx), strict=True)
    )
    local_counts = dict(counts)
    crop_slices = []
    pad = max(0, int(aperture_pad_cells))
    extents = _source_extent_by_axis(axis, width, height)

    for transverse_axis in _TRANSVERSE_AXES_3D[axis]:
        interval = None
        if snapped_region is not None:
            try:
                interval = snapped_region.axis_interval(transverse_axis)
            except Exception:
                interval = None
        if interval is None:
            interval = snap_centered_extent(
                centers[transverse_axis],
                extents[transverse_axis],
                float(resolution),
                counts[transverse_axis],
                min_cells=2,
            )
        start = int(interval.start) - pad
        stop = int(interval.stop) + pad
        start, stop = _ensure_min_interval(start, stop, counts[transverse_axis])
        origin[transverse_axis] = int(start)
        local_counts[transverse_axis] = int(stop) - int(start)
        crop_slices.append(
            slice(
                int(start) - material_origin[transverse_axis],
                int(stop) - material_origin[transverse_axis],
            )
        )

    normal_origin = min(int(plane_index), int(offset_index))
    normal_stop = max(int(plane_index), int(offset_index)) + 2
    normal_origin, normal_stop = _ensure_min_interval(
        normal_origin,
        normal_stop,
        counts[axis],
    )
    origin[axis] = int(normal_origin)
    local_counts[axis] = int(normal_stop) - int(normal_origin)
    local_plane_index = int(plane_index) - int(normal_origin)
    local_offset_index = int(offset_index) - int(normal_origin)

    eps_local = np.asarray(eps_profile)[tuple(crop_slices)]
    local_grid_shape = (
        int(local_counts["z"]),
        int(local_counts["y"]),
        int(local_counts["x"]),
    )
    local_center = tuple(
        float(centers[name]) - float(origin[name]) * float(resolution)
        for name in ("x", "y", "z")
    )
    component_shapes = {
        component: component_shape_3d(component, local_grid_shape)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    return {
        "scalar_permittivity": eps_local,
        "grid_shape": local_grid_shape,
        "component_shapes": component_shapes,
        "center": local_center,
        "plane_index": int(local_plane_index),
        "offset_index": int(local_offset_index),
        "origin_zyx": (
            int(origin["z"]),
            int(origin["y"]),
            int(origin["x"]),
        ),
    }


def _solve_mode_plane_3d(
    permittivity,
    permeability,
    *,
    frequency,
    resolution,
    dt,
    axis,
    direction,
    grid_shape,
    center,
    width,
    height,
    plane_index,
    offset_index,
    mode_index,
    polarization,
    target_neff,
    num_modes,
    snapped_region,
    solver_direction=None,
    material_origin_zyx=(0, 0, 0),
    solver=solve_beamz_mode,
) -> DiscreteMode:
    """Solve one finite 3D mode plane and return globally indexed fields."""
    axis_index = _AXIS_POS_3D[axis]
    profile_index = int(plane_index) - int(material_origin_zyx[axis_index])
    eps_profile = np.take(permittivity, profile_index, axis=axis_index)
    local_plane = _local_mode_plane_spec(
        eps_profile,
        axis=axis,
        grid_shape=grid_shape,
        center=center,
        width=width,
        height=height,
        plane_index=plane_index,
        offset_index=offset_index,
        resolution=resolution,
        snapped_region=snapped_region,
        aperture_pad_cells=_mode_plane_outer_pad_cells(width, height, resolution),
        material_origin_zyx=material_origin_zyx,
    )
    target = target_neff
    if target is None:
        target = 0.98 * np.sqrt(
            max(float(np.max(np.real(local_plane["scalar_permittivity"]))), 1e-12)
        )
    sampling_plane = dict(local_plane)
    sampling_plane["origin_zyx"] = tuple(
        int(global_offset) - int(region_offset)
        for global_offset, region_offset in zip(
            local_plane["origin_zyx"], material_origin_zyx, strict=True
        )
    )
    component_permittivity, component_permeability = _local_component_materials(
        permittivity, permeability, sampling_plane
    )
    mode = solver(
        ModePlaneSpec(
            scalar_permittivity=np.asarray(
                local_plane["scalar_permittivity"], dtype=np.complex128
            ),
            frequency=float(frequency),
            resolution=float(resolution),
            dt=None if dt is None else float(dt),
            axis=axis,
            direction=direction,
            solver_direction=solver_direction or direction,
            transverse_axes=_TRANSVERSE_AXES_3D[axis],
            grid_shape=local_plane["grid_shape"],
            component_shapes=local_plane["component_shapes"],
            component_permittivity=component_permittivity,
            component_permeability=component_permeability,
            center=local_plane["center"],
            width=float(width),
            height=float(height),
            plane_index=int(local_plane["plane_index"]),
            offset_index=int(local_plane["offset_index"]),
            mode_index=int(mode_index),
            polarization=polarization,
            target_neff=target,
            num_modes=int(num_modes),
            aperture_pad_cells=_MODE_PLANE_APERTURE_PAD_CELLS,
            aperture_window_alpha=_MODE_PLANE_APERTURE_WINDOW_ALPHA,
        )
    )
    return _shift_discrete_mode_to_global(
        mode,
        origin_zyx=local_plane["origin_zyx"],
        axis=axis,
        resolution=resolution,
    )


def _local_component_materials(permittivity, permeability, local_plane):
    """Sample local voxel materials on the six Yee component lattices."""
    origin = tuple(int(value) for value in local_plane["origin_zyx"])
    grid_shape = tuple(int(value) for value in local_plane["grid_shape"])
    region = tuple(
        slice(start, start + count)
        for start, count in zip(origin, grid_shape, strict=True)
    )

    eps_grid = np.asarray(permittivity)
    mu_grid = np.asarray(permeability)
    if mu_grid.ndim == 0:
        mu_grid = np.full(eps_grid.shape, mu_grid.item(), dtype=mu_grid.dtype)
    else:
        mu_grid = np.broadcast_to(mu_grid, eps_grid.shape)
    eps_local = eps_grid[region]
    mu_local = mu_grid[region]
    shapes = local_plane["component_shapes"]
    component_permittivity = {
        component: np.asarray(
            sample_voxel_grid_at_e_component_3d_centered(
                eps_local,
                component,
                stored_shape=shapes[component],
            )
        )
        for component in ("Ex", "Ey", "Ez")
    }
    component_permeability = {
        component: np.asarray(
            sample_voxel_grid_at_component_3d(
                mu_local,
                component,
                stored_shape=shapes[component],
            )
        )
        for component in ("Hx", "Hy", "Hz")
    }
    return component_permittivity, component_permeability


def _shift_3d_index(index, origin_zyx):
    out = []
    for item, offset in zip(index, origin_zyx, strict=True):
        offset = int(offset)
        if isinstance(item, slice):
            start = None if item.start is None else int(item.start) + offset
            stop = None if item.stop is None else int(item.stop) + offset
            out.append(slice(start, stop, item.step))
        else:
            out.append(int(item) + offset)
    return tuple(out)


def _shift_discrete_mode_to_global(
    discrete_mode: DiscreteMode, *, origin_zyx, axis, resolution
) -> DiscreteMode:
    """Shift a mode result solved on a local crop back to global indices."""
    component_indices = {
        name: _shift_3d_index(index, origin_zyx)
        for name, index in discrete_mode.component_indices.items()
    }
    axis_offset = float(origin_zyx[_AXIS_POS_3D[str(axis).lower()]]) * float(resolution)
    return replace(
        discrete_mode,
        component_indices=component_indices,
        phase_reference_coord=float(discrete_mode.phase_reference_coord) + axis_offset,
        phase_plane_coord=float(discrete_mode.phase_plane_coord) + axis_offset,
    )


def _scale_profiles_for_power(profiles, power):
    """Scale unit-power modal profiles to the requested launched power."""
    power_value = float(power)
    if not np.isfinite(power_value) or power_value < 0.0:
        raise ValueError(
            f"ModeSource power must be a non-negative finite value, got {power!r}."
        )
    if power_value == 1.0:
        return profiles
    scale = float(np.sqrt(power_value))
    for key, value in profiles.items():
        if value is not None:
            profiles[key] = np.asarray(value) * scale
    return profiles


def _scale_pair_for_power(first, second, power):
    power_value = float(power)
    if not np.isfinite(power_value) or power_value < 0.0:
        raise ValueError(
            f"ModeSource power must be a non-negative finite value, got {power!r}."
        )
    if power_value == 1.0:
        return first, second
    scale = float(np.sqrt(power_value))
    return np.asarray(first) * scale, np.asarray(second) * scale
