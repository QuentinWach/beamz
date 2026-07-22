"""Shared modal profile and discrete-mode solver helpers."""

from __future__ import annotations

import logging
from contextlib import suppress
from copy import copy
from dataclasses import is_dataclass, replace
from typing import Any, cast

import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.devices._placement import snap_centered_extent
from beamz.lattice import (
    component_shape_3d,
    sample_voxel_grid_at_component_3d,
    sample_voxel_grid_at_e_component_3d_centered,
    yee_flux,
)

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = {"+x", "-x", "+y", "-y", "+z", "-z"}
_AXIS_POS_3D = {"z": 0, "y": 1, "x": 2}
_TRANSVERSE_AXES_3D = {
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


def _solve_numeric_k_axis(omega, dt, d_axis, neff, eps=1e-30):
    """Solve 1D Yee dispersion for k at a fixed omega (normal incidence)."""
    neff_r = max(float(np.real(neff)), eps)
    d = max(float(d_axis), eps)
    dt_r = max(float(dt), eps)
    omega_r = float(omega)
    k_phys = omega_r * neff_r / LIGHT_SPEED

    S = LIGHT_SPEED * dt_r / (neff_r * d)
    if (not np.isfinite(S)) or eps >= S:
        return k_phys

    rhs = np.sin(0.5 * omega_r * dt_r) / S
    rhs = float(np.clip(rhs, -1.0, 1.0))
    k_num = (2.0 / d) * np.arcsin(rhs)
    if (not np.isfinite(k_num)) or k_num <= eps:
        return k_phys
    return float(k_num)


def _numeric_phase_delay(omega, k_num, delta_s, eps=1e-30):
    """Convert numerical phase advance into a time delay."""
    omega_r = max(abs(float(omega)), eps)
    # Keep the sign: launch direction depends on the signed E/H plane offset.
    return float((float(k_num) * float(delta_s)) / omega_r)


def _axis_index_from_component_indices(indices, axis):
    """Extract scalar axis index from a 3D component index tuple."""
    if indices is None:
        return None
    axis_pos = {"x": 2, "y": 1, "z": 0}[axis]
    val = indices[axis_pos]
    if isinstance(val, slice):
        return None
    return int(val)


def _component_axis_coord(component_name, axis_index, axis, dx, dy, dz):
    """Yee-location coordinate along propagation axis for one component plane index."""
    if axis_index is None:
        return 0.0

    d_axis = {"x": dx, "y": dy, "z": dz}[axis]
    staggered_along_axis = {
        "x": {"Ex", "Hy", "Hz"},
        "y": {"Ey", "Hx", "Hz"},
        "z": {"Ez", "Hx", "Hy"},
    }
    offset = 1.0 if component_name in staggered_along_axis[axis] else 0.5
    return (axis_index + offset) * d_axis


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
):
    """Build a compact local micromode plane and metadata to shift it globally."""
    axis = str(axis).lower()
    counts = _axis_counts_from_grid_shape(grid_shape)
    centers = _center_by_axis(center, grid_shape, resolution)
    if snapped_region is not None:
        for name in ("x", "y", "z"):
            with suppress(Exception):
                centers[name] = float(snapped_region.axis_coord(name))

    origin = {"z": 0, "y": 0, "x": 0}
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
        crop_slices.append(slice(int(start), int(stop)))

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


def _shift_discrete_mode_to_global(discrete_mode, *, origin_zyx, axis, resolution):
    """Shift a micromode result solved on a local crop back to global indices."""
    if discrete_mode is None:
        return None
    component_indices = {
        name: _shift_3d_index(index, origin_zyx)
        for name, index in dict(discrete_mode.component_indices).items()
    }
    axis_offset = float(origin_zyx[_AXIS_POS_3D[str(axis).lower()]]) * float(resolution)
    updates = {
        "component_indices": component_indices,
        "phase_reference_coord": (
            float(discrete_mode.phase_reference_coord) + axis_offset
        ),
        "phase_plane_coord": float(discrete_mode.phase_plane_coord) + axis_offset,
    }
    if is_dataclass(discrete_mode):
        return replace(cast(Any, discrete_mode), **updates)
    shifted = copy(discrete_mode)
    for name, value in updates.items():
        setattr(shifted, name, value)
    return shifted


def _modal_power_3d_from_profiles(profiles, axis, d_area, direction_sign=1.0):
    """Compute phasor power from colocated modal cross-section profiles."""
    normal_axis = {"x": 0, "y": 1, "z": 2}.get(str(axis).lower())
    if normal_axis is None:
        return 0.0
    required = (
        ("Ey", "Ez", "Hy", "Hz"),
        ("Ex", "Ez", "Hx", "Hz"),
        ("Ex", "Ey", "Hx", "Hy"),
    )[normal_axis]
    if any(profiles.get(name) is None for name in required):
        return 0.0

    def plane(value):
        array = np.asarray(value, dtype=np.complex128)
        return array[:, None] if array.ndim == 1 else array

    present = [plane(value) for value in profiles.values() if value is not None]
    shape = tuple(min(array.shape[axis] for array in present) for axis in (0, 1))
    if not all(shape):
        return 0.0
    template = np.zeros(shape, dtype=np.complex128)
    samples = tuple(
        template
        if profiles.get(name) is None
        else plane(profiles[name])[: shape[0], : shape[1]]
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    return float(
        np.asarray(
            yee_flux(
                samples,
                normal_axis,
                normal_sign=float(direction_sign),
                measure=float(d_area),
                phasor=True,
            )
        )
    )


def _normalize_3d_profiles_by_flux(
    profiles,
    axis,
    d_area=1.0,
    direction_sign=1.0,
    eps=1e-18,
):
    """Normalize 3D source profiles so |modal power| equals 1."""
    flux = _modal_power_3d_from_profiles(
        profiles,
        axis=axis,
        d_area=d_area,
        direction_sign=direction_sign,
    )
    if (not np.isfinite(flux)) or abs(flux) <= eps:
        return profiles

    scale = float(np.sqrt(1.0 / max(abs(flux), eps)))
    scale = float(np.clip(scale, 1e-6, 1e6))
    for key, value in profiles.items():
        profiles[key] = np.asarray(value) * scale
    return profiles


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


def _modal_overlap_3d_profiles(
    field_profiles,
    mode_profiles,
    axis,
    d_area,
    direction_sign=1.0,
):
    """Symmetric power overlap between a field sample and a 3D modal basis field."""
    comp_map = {
        "x": ("Ey", "Ez", "Hz", "Hy"),
        "y": ("Ez", "Ex", "Hx", "Hz"),
        "z": ("Ex", "Ey", "Hy", "Hx"),
    }
    try:
        e1, e2, h1, h2 = comp_map[str(axis)]
    except KeyError as exc:
        raise ValueError(f"Unsupported axis {axis!r}") from exc

    n_common = None
    for name in (e1, e2, h1, h2):
        for profiles in (field_profiles, mode_profiles):
            if name not in profiles:
                continue
            arr = np.asarray(profiles[name], dtype=np.complex128).reshape(-1)
            if arr.size <= 0:
                continue
            n_common = arr.size if n_common is None else min(n_common, arr.size)

    n_common = int(max(0, n_common or 0))
    if n_common <= 0:
        return np.complex128(0.0 + 0.0j)

    def _component(profiles, name):
        if name not in profiles:
            return np.zeros((n_common,), dtype=np.complex128)
        arr = np.asarray(profiles[name], dtype=np.complex128).reshape(-1)
        if arr.size >= n_common:
            return arr[:n_common]
        out = np.zeros((n_common,), dtype=np.complex128)
        out[: arr.size] = arr
        return out

    ef1 = _component(field_profiles, e1)
    ef2 = _component(field_profiles, e2)
    hf1 = _component(field_profiles, h1)
    hf2 = _component(field_profiles, h2)
    em1 = _component(mode_profiles, e1)
    em2 = _component(mode_profiles, e2)
    hm1 = _component(mode_profiles, h1)
    hm2 = _component(mode_profiles, h2)

    overlap = (
        0.25
        * np.sum(
            ef1 * np.conjugate(hm1)
            - ef2 * np.conjugate(hm2)
            + np.conjugate(em1) * hf1
            - np.conjugate(em2) * hf2
        )
        * float(d_area)
    )
    return np.complex128(float(direction_sign) * overlap)
