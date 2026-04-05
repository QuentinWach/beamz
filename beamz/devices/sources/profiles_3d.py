"""3D profile construction helpers for mode-source injection."""

import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.devices.sources.profiles_basis import _normalize_3d_profiles_by_flux
from beamz.devices.sources.profiles_common import (
    _impedance_match_3d_tangential_pairs,
    _numeric_impedance_axis,
    _solve_numeric_k_axis,
)
from beamz.devices.sources.windows import (
    _compute_transverse_bounds,
    _crop_and_window_2d,
    _make_tukey_window_2d,
    _stagger_both,
    _stagger_half,
)

_STAGGER_3D = {
    "x": {
        "Ex": None,
        "Ey": ("half", 1),
        "Ez": ("half", 0),
        "Hx": ("both", None),
        "Hy": ("half", 0),
        "Hz": ("half", 1),
    },
    "y": {
        "Ex": ("half", 1),
        "Ey": None,
        "Ez": ("half", 0),
        "Hx": ("half", 0),
        "Hy": ("both", None),
        "Hz": ("half", 1),
    },
    "z": {
        "Ex": ("half", 1),
        "Ey": ("half", 0),
        "Ez": None,
        "Hx": ("half", 0),
        "Hy": ("half", 1),
        "Hz": ("both", None),
    },
}

_AXIS_PROFILE_META = {
    "x": {
        "crop_axes": ("z", "y"),
        "use_jax": True,
        "extra_components": {"_h_component": "Hy", "_e_component": "Ey"},
    },
    "y": {
        "crop_axes": ("z", "x"),
        "use_jax": False,
        "extra_components": {"_h_component": "Hx", "_e_component": "Ex"},
    },
    "z": {
        "crop_axes": ("y", "x"),
        "use_jax": True,
        "extra_components": {"_h_component": "Hx", "_e_component": "Ex"},
    },
}


def _apply_stagger_op(field, op):
    if op is None:
        return field
    kind, axis = op
    if kind == "both":
        return _stagger_both(field)
    if kind == "half":
        return _stagger_half(field, axis=axis)
    raise ValueError(f"Unsupported stagger op {op!r}")


def _build_3d_indices(axis, staggered, bounds, center_idx, offset_idx, grid_shape):
    nz, ny, nx = grid_shape
    limit_slice = lambda field, start, end, dim, limit: slice(
        start, min(end, field.shape[dim], limit)
    )
    if axis == "x":
        z_start, z_end = bounds["z"]
        y_start, y_end = bounds["y"]
        return {
            "Ex": (
                limit_slice(staggered["Ex"], z_start, z_end, 0, nz),
                limit_slice(staggered["Ex"], y_start, y_end, 1, ny),
                offset_idx,
            ),
            "Ey": (
                limit_slice(staggered["Ey"], z_start, z_end, 0, nz),
                limit_slice(staggered["Ey"], y_start, y_end, 1, ny - 1),
                center_idx,
            ),
            "Ez": (
                limit_slice(staggered["Ez"], z_start, z_end, 0, nz - 1),
                limit_slice(staggered["Ez"], y_start, y_end, 1, ny),
                center_idx,
            ),
            "Hx": (
                limit_slice(staggered["Hx"], z_start, z_end, 0, nz - 1),
                limit_slice(staggered["Hx"], y_start, y_end, 1, ny - 1),
                center_idx,
            ),
            "Hy": (
                limit_slice(staggered["Hy"], z_start, z_end, 0, nz - 1),
                limit_slice(staggered["Hy"], y_start, y_end, 1, ny),
                offset_idx,
            ),
            "Hz": (
                limit_slice(staggered["Hz"], z_start, z_end, 0, nz),
                limit_slice(staggered["Hz"], y_start, y_end, 1, ny - 1),
                offset_idx,
            ),
        }

    if axis == "y":
        z_start, z_end = bounds["z"]
        x_start, x_end = bounds["x"]
        return {
            "Ex": (
                limit_slice(staggered["Ex"], z_start, z_end, 0, nz),
                center_idx,
                limit_slice(staggered["Ex"], x_start, x_end, 1, nx - 1),
            ),
            "Ey": (
                limit_slice(staggered["Ey"], z_start, z_end, 0, nz),
                offset_idx,
                limit_slice(staggered["Ey"], x_start, x_end, 1, nx),
            ),
            "Ez": (
                limit_slice(staggered["Ez"], z_start, z_end, 0, nz - 1),
                center_idx,
                limit_slice(staggered["Ez"], x_start, x_end, 1, nx),
            ),
            "Hx": (
                limit_slice(staggered["Hx"], z_start, z_end, 0, nz - 1),
                offset_idx,
                limit_slice(staggered["Hx"], x_start, x_end, 1, nx),
            ),
            "Hy": (
                limit_slice(staggered["Hy"], z_start, z_end, 0, nz - 1),
                center_idx,
                limit_slice(staggered["Hy"], x_start, x_end, 1, nx - 1),
            ),
            "Hz": (
                limit_slice(staggered["Hz"], z_start, z_end, 0, nz),
                offset_idx,
                limit_slice(staggered["Hz"], x_start, x_end, 1, nx - 1),
            ),
        }

    y_start, y_end = bounds["y"]
    x_start, x_end = bounds["x"]
    e_z_idx = int(np.clip(center_idx, 0, nz - 1))
    h_z_idx = int(np.clip(offset_idx, 0, max(nz - 2, 0)))
    ez_z_idx = int(np.clip(center_idx, 0, max(nz - 2, 0)))
    hz_z_idx = int(np.clip(offset_idx, 0, nz - 1))
    return {
        "Ex": (
            e_z_idx,
            limit_slice(staggered["Ex"], y_start, y_end, 0, ny),
            limit_slice(staggered["Ex"], x_start, x_end, 1, nx - 1),
        ),
        "Ey": (
            e_z_idx,
            limit_slice(staggered["Ey"], y_start, y_end, 0, ny - 1),
            limit_slice(staggered["Ey"], x_start, x_end, 1, nx),
        ),
        "Ez": (
            ez_z_idx,
            limit_slice(staggered["Ez"], y_start, y_end, 0, ny),
            limit_slice(staggered["Ez"], x_start, x_end, 1, nx),
        ),
        "Hx": (
            h_z_idx,
            limit_slice(staggered["Hx"], y_start, y_end, 0, ny - 1),
            limit_slice(staggered["Hx"], x_start, x_end, 1, nx),
        ),
        "Hy": (
            h_z_idx,
            limit_slice(staggered["Hy"], y_start, y_end, 0, ny),
            limit_slice(staggered["Hy"], x_start, x_end, 1, nx - 1),
        ),
        "Hz": (
            hz_z_idx,
            limit_slice(staggered["Hz"], y_start, y_end, 0, ny - 1),
            limit_slice(staggered["Hz"], x_start, x_end, 1, nx - 1),
        ),
    }


def _build_3d_profiles(
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
    axis,
    direction,
    center,
    width,
    height,
    center_idx,
    offset_idx,
    grid_shape,
    resolution,
    impedance_neff,
    omega,
    dt,
):
    """Build staggered, windowed, impedance-corrected injection profiles for 3D."""
    dir_sign = 1.0 if direction.startswith("+") else -1.0

    if axis not in _STAGGER_3D:
        raise ValueError(f"Unsupported axis {axis!r} for 3D profile setup")

    staggered = {
        name: _apply_stagger_op(field, _STAGGER_3D[axis][name])
        for name, field in {
            "Ex": Ex,
            "Ey": Ey,
            "Ez": Ez,
            "Hx": Hx,
            "Hy": Hy,
            "Hz": Hz,
        }.items()
    }
    nz, ny, nx = grid_shape
    z_center = center[2] if len(center) > 2 else (nz // 2) * resolution
    if axis == "x":
        bounds = {
            "y": _compute_transverse_bounds(center[1], width, resolution, ny),
            "z": _compute_transverse_bounds(z_center, height, resolution, nz),
        }
    elif axis == "y":
        bounds = {
            "x": _compute_transverse_bounds(center[0], width, resolution, nx),
            "z": _compute_transverse_bounds(z_center, height, resolution, nz),
        }
    elif axis == "z":
        bounds = {
            "x": _compute_transverse_bounds(center[0], width, resolution, nx),
            "y": _compute_transverse_bounds(center[1], height, resolution, ny),
        }
    else:
        raise ValueError(f"Unsupported axis {axis!r} for 3D profile setup")
    indices = _build_3d_indices(
        axis, staggered, bounds, center_idx, offset_idx, grid_shape
    )
    neff_imp_r = max(float(np.real(impedance_neff)), 1e-6)
    if dt is None:
        z_target = np.sqrt(MU_0 / EPS_0) / neff_imp_r
    else:
        k_num_imp = _solve_numeric_k_axis(omega, dt, resolution, neff_imp_r)
        z_target = _numeric_impedance_axis(
            omega, dt, resolution, k_num_imp, neff_imp_r
        )
    staggered["Ex"], staggered["Ey"], staggered["Ez"] = _impedance_match_3d_tangential_pairs(
        axis,
        staggered["Ex"],
        staggered["Ey"],
        staggered["Ez"],
        staggered["Hx"],
        staggered["Hy"],
        staggered["Hz"],
        z_target,
    )

    meta = _AXIS_PROFILE_META[axis]
    primary_axis, secondary_axis = meta["crop_axes"]
    primary_start, primary_end = bounds[primary_axis]
    secondary_start, secondary_end = bounds[secondary_axis]
    profiles = _crop_and_window_all(
        staggered,
        primary_start,
        primary_end,
        secondary_start,
        secondary_end,
        dir_sign,
        use_jax=meta["use_jax"],
        alpha=0.2,
    )
    profiles = _normalize_3d_profiles_by_flux(
        profiles, axis=axis, d_area=float(resolution * resolution)
    )
    extra = {f"_{name}_start": start for name, (start, _) in bounds.items()}
    extra.update({f"_{name}_end": end for name, (_, end) in bounds.items()})
    extra.update(meta["extra_components"])
    return profiles, indices, extra


def _crop_and_window_all(
    staggered,
    z_start,
    z_end,
    t_start,
    t_end,
    dir_sign,
    use_jax,
    alpha=0.3,
):
    """Crop all six staggered profiles and multiply by a 2D Tukey window."""
    ref = next(iter(staggered.values()))
    pz_end = min(z_end, ref.shape[0])
    pt_end = min(t_end, ref.shape[1])
    h_cells = pz_end - z_start
    w_cells = pt_end - t_start

    window = _make_tukey_window_2d(h_cells, w_cells, alpha=alpha, use_jax=use_jax)

    profiles = {}
    for name, field in staggered.items():
        fe = min(z_end, field.shape[0])
        te = min(t_end, field.shape[1])
        profiles[name] = dir_sign * np.real(
            _crop_and_window_2d(field, z_start, fe, t_start, te, window)
        )
    return profiles
