"""Mode-profile math and 3D profile builders for mode sources."""

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.sources.profiles_basis import (
    _backward_3d_mode_from_forward,
    _make_3d_mode_basis_profiles,
    _modal_overlap_3d_profiles,
    _modal_power_2d,
    _modal_power_3d_from_profiles,
    _normalize_2d_pair_by_power,
    _normalize_3d_profiles_by_flux,
    _project_3d_profiles_to_real,
    _to_real_profile,
)
from beamz.devices.sources.windows import (
    _compute_transverse_bounds,
    _crop_and_window_2d,
    _scipy_tukey,
    _make_tukey_window_2d,
    _stagger_both,
    _stagger_half,
)

_VALID_DIRECTIONS = {"+x", "-x", "+y", "-y", "+z", "-z"}


def _impedance_match_e_profile(e_profile, h_profile, z_target, eps=1e-12):
    """Scale E profile so the least-squares modal impedance matches `z_target`."""
    e = np.asarray(e_profile, dtype=np.complex128).reshape(-1)
    h = np.asarray(h_profile, dtype=np.complex128).reshape(-1)
    n = int(min(e.size, h.size))
    if n <= 0:
        return e_profile
    e = e[:n]
    h = h[:n]
    denom = np.sum(h * np.conjugate(h))
    if abs(denom) <= eps:
        return e_profile
    z_est = np.sum(e * np.conjugate(h)) / denom
    z_mag = float(abs(z_est))
    if (not np.isfinite(z_mag)) or z_mag <= eps:
        return e_profile
    scale = float(abs(z_target)) / z_mag
    return np.asarray(e_profile) * scale


def _align_2d_impedance_pair(h_field, e_field, z_target):
    h_profile = np.squeeze(h_field)
    e_profile = np.squeeze(e_field)
    idx_max = np.argmax(np.abs(h_profile))
    phase_ref = np.angle(h_profile.flatten()[idx_max])
    h_profile = h_profile * np.exp(-1j * phase_ref)
    e_profile = e_profile * np.exp(-1j * phase_ref)
    return h_profile, _impedance_match_e_profile(e_profile, h_profile, z_target)


def _stagger_2d_pair(h_field, e_field):
    return 0.5 * (h_field[:-1] + h_field[1:]), 0.5 * (e_field[:-1] + e_field[1:])


def _crop_window_2d_pair(h_profile, e_profile, start: int, end: int):
    stop = min(end, len(h_profile), len(e_profile))
    width = max(0, stop - start)
    window = _scipy_tukey(width, alpha=0.3) if width > 2 else np.ones(max(1, width))
    h_cropped = h_profile[start:stop]
    e_cropped = e_profile[start:stop]
    if len(h_cropped) == len(window):
        h_cropped = h_cropped * window
        e_cropped = e_cropped * window
    return h_cropped, e_cropped


def _finalize_2d_launch_pair(
    h_profile,
    e_profile,
    *,
    sign_h,
    sign_e,
    signed_flux_sign,
    resolution,
):
    h_profile = sign_h * h_profile
    e_profile = sign_e * e_profile
    h_profile, e_profile = _normalize_2d_pair_by_power(
        h_profile,
        e_profile,
        signed_flux_sign=signed_flux_sign,
        dl=resolution,
    )
    h_profile = _to_real_profile(h_profile)
    e_profile = _to_real_profile(e_profile)
    return _normalize_2d_pair_by_power(
        h_profile,
        e_profile,
        signed_flux_sign=signed_flux_sign,
        dl=resolution,
    )


def _solve_numeric_k_axis(omega, dt, d_axis, neff, eps=1e-30):
    """Solve 1D Yee dispersion for k at a fixed omega."""
    neff_r = max(float(np.real(neff)), eps)
    d = max(float(d_axis), eps)
    dt_r = max(float(dt), eps)
    omega_r = float(omega)
    k_phys = omega_r * neff_r / LIGHT_SPEED

    S = LIGHT_SPEED * dt_r / (neff_r * d)
    if (not np.isfinite(S)) or S <= eps:
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
    return float((float(k_num) * float(delta_s)) / omega_r)


def _numeric_impedance_axis(omega, dt, d_axis, k_num, neff, mu_r=1.0, eps=1e-30):
    """Discrete normal-incidence Yee impedance for one propagation axis."""
    neff_r = max(float(np.real(neff)), eps)
    eta0 = np.sqrt(MU_0 / EPS_0)

    denom = np.sin(0.5 * float(k_num) * float(d_axis))
    if abs(denom) <= eps or not np.isfinite(denom):
        return float(eta0 / neff_r)

    numer = (
        float(mu_r)
        * MU_0
        * (float(d_axis) / max(float(dt), eps))
        * np.sin(0.5 * float(omega) * float(dt))
    )
    z_num = abs(numer / denom)
    if (not np.isfinite(z_num)) or z_num <= eps:
        return float(eta0 / neff_r)
    return float(z_num)


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


def _dominant_3d_pair(axis, pol):
    """Dominant launch pair used for 3D delay extraction."""
    pair = {
        ("x", "tm"): ("Ez", "Hy"),
        ("x", "te"): ("Ey", "Hz"),
        ("y", "tm"): ("Ez", "Hx"),
        ("y", "te"): ("Ex", "Hz"),
        ("z", "tm"): ("Ey", "Hx"),
        ("z", "te"): ("Ex", "Hy"),
    }.get((axis, pol))
    if pair is None:
        raise ValueError(f"Unsupported 3D delay pair for axis={axis!r}, pol={pol!r}")
    return pair


def _select_3d_impedance_index(axis, pol, eps_profile_2d, Ex, Ey, Ez, Hx, Hy, Hz):
    """Pick a local modal index for 3D impedance targeting."""
    eps_arr = np.asarray(eps_profile_2d)
    if eps_arr.size == 0:
        return 1.0

    tangential = (
        {"x": (Hy, Hz), "y": (Hx, Hz), "z": (Hx, Hy)}
        if pol == "tm"
        else {"x": (Ey, Ez), "y": (Ex, Ez), "z": (Ex, Ey)}
    )[axis]
    w_sum = 0.0
    ew_sum = 0.0
    for comp in tangential:
        arr = np.asarray(comp)
        if arr.ndim != 2 or arr.size == 0:
            continue
        z = min(eps_arr.shape[0], arr.shape[0])
        t = min(eps_arr.shape[1], arr.shape[1])
        if z <= 0 or t <= 0:
            continue
        w = np.abs(arr[:z, :t]) ** 2
        w_sum += float(np.sum(w))
        ew_sum += float(np.sum(eps_arr[:z, :t] * w))

    if w_sum <= 1e-30:
        return float(np.sqrt(max(float(np.mean(eps_arr)), 1e-30)))
    return float(np.sqrt(max(ew_sum / w_sum, 1e-30)))


def _impedance_match_3d_tangential_pairs(
    axis, Ex_s, Ey_s, Ez_s, Hx_s, Hy_s, Hz_s, Z_target, eps=1e-12
):
    """Match tangential 3D E components to paired H components."""
    e_map = {"Ex": Ex_s, "Ey": Ey_s, "Ez": Ez_s}
    h_map = {"Hx": Hx_s, "Hy": Hy_s, "Hz": Hz_s}
    pair_map = {
        "x": [("Ey", "Hz"), ("Ez", "Hy")],
        "y": [("Ex", "Hz"), ("Ez", "Hx")],
        "z": [("Ex", "Hy"), ("Ey", "Hx")],
    }
    for e_name, h_name in pair_map[axis]:
        e_field = np.asarray(e_map[e_name])
        h_field = np.asarray(h_map[h_name])
        abs_e = np.abs(e_field)
        abs_h = np.abs(h_field)

        h_peak = float(np.max(abs_h))
        if h_peak > eps:
            mask = abs_h > (0.05 * h_peak)
            if np.any(mask):
                local_ratio = abs_e[mask] / (abs_h[mask] + eps)
                ratio = float(np.median(local_ratio))
            else:
                ratio = 0.0
        else:
            ratio = 0.0

        if not np.isfinite(ratio) or ratio <= eps:
            e_norm = float(np.sqrt(np.sum(abs_e**2) + eps))
            h_norm = float(np.sqrt(np.sum(abs_h**2) + eps))
            ratio = e_norm / (h_norm + eps)

        if not np.isfinite(ratio) or ratio <= eps:
            continue
        z_abs = float(abs(Z_target))
        scale = z_abs / ratio
        max_scale = max(20.0, 8.0 * z_abs)
        scale = float(np.clip(scale, 1.0 / max_scale, max_scale))
        e_map[e_name] = e_field * scale
    return e_map["Ex"], e_map["Ey"], e_map["Ez"]


def _parse_direction(direction):
    """Parse and validate direction string into `(direction, axis, sign)`."""
    direction = str(direction).lower()
    if direction not in _VALID_DIRECTIONS:
        valid = ", ".join(sorted(_VALID_DIRECTIONS))
        raise ValueError(f"direction must be one of {{{valid}}}, got {direction!r}")
    axis = direction[1]
    dir_sign = 1.0 if direction.startswith("+") else -1.0
    return direction, axis, dir_sign


def _remap_3d_solver_components(Ex, Ey, Ez, Hx, Hy, Hz, axis):
    """Remap solver x-basis components to the requested propagation axis."""
    if axis == "x":
        return Ex, Ey, Ez, Hx, Hy, Hz
    if axis == "y":
        return -Ey, Ex, Ez, -Hy, Hx, Hz
    if axis == "z":
        return Ey, Ez, Ex, Hy, Hz, Hx
    raise ValueError(f"Unsupported axis {axis!r} for 3D remap")


def _select_3d_phase_ref(axis, pol, Ex, Ey, Ez, Hx, Hy, Hz):
    """Select the 3D phase-reference field using the 2D H-based convention."""
    preferred = {
        ("x", "tm"): Hy,
        ("x", "te"): Hz,
        ("y", "tm"): Hx,
        ("y", "te"): Hz,
        ("z", "tm"): Hx,
        ("z", "te"): Hy,
    }
    tangential_h = {
        "x": (Hy, Hz),
        "y": (Hx, Hz),
        "z": (Hx, Hy),
    }
    key = (axis, pol)
    if key not in preferred:
        raise ValueError(f"Unsupported 3D phase-reference mapping for {key!r}")

    ref_field = preferred[key]
    if float(jnp.max(jnp.abs(ref_field))) >= 1e-9:
        return ref_field

    candidates = tangential_h[axis]
    strengths = [float(jnp.max(jnp.abs(comp))) for comp in candidates]
    return candidates[int(np.argmax(strengths))]


def _select_core_confined_mode_index(eps_profile, e_fields, neff_values):
    """Choose the candidate mode with strongest confinement in the high-index core."""
    if e_fields is None or len(e_fields) <= 1:
        return 0

    eps_arr = np.asarray(eps_profile)
    if eps_arr.size == 0:
        return 0

    eps_flat = np.real(eps_arr).ravel()
    eps_min = float(np.min(eps_flat))
    eps_max = float(np.max(eps_flat))
    if (not np.isfinite(eps_min)) or (not np.isfinite(eps_max)) or (eps_max <= eps_min):
        return 0

    core_mask = eps_flat >= (eps_min + 0.5 * (eps_max - eps_min))
    if not np.any(core_mask):
        return 0

    core_coords = np.argwhere(core_mask.reshape(eps_arr.shape))
    if core_coords.size > 0:
        core_center = np.mean(core_coords, axis=0)
    else:
        core_center = 0.5 * (np.asarray(eps_arr.shape, dtype=float) - 1.0)

    best_idx = 0
    best_core_frac = -np.inf
    best_center_frac = -np.inf
    best_neff = -np.inf

    for idx, field in enumerate(e_fields):
        f = np.asarray(field)
        if f.ndim < 2:
            continue

        e_plane = np.sum(np.abs(f) ** 2, axis=0)
        e_mag = e_plane.ravel()
        n = min(e_mag.size, core_mask.size)
        if n <= 0:
            continue
        e_mag = e_mag[:n]
        mask = core_mask[:n]

        total = float(np.sum(e_mag))
        if (not np.isfinite(total)) or total <= 1e-30:
            continue

        core_frac = float(np.sum(e_mag[mask]) / total)

        e_mag_shape = e_plane.shape
        center_mask = np.ones(e_mag_shape, dtype=bool)
        idx_grids = np.indices(e_mag_shape)
        for axis_i, n_axis in enumerate(e_mag_shape):
            half_span = max(1.0, 0.25 * float(n_axis))
            center_mask &= (
                np.abs(idx_grids[axis_i] - float(core_center[axis_i])) <= half_span
            )
        center_num = float(np.sum(e_plane[center_mask]))
        center_frac = center_num / total
        neff_r = float(np.real(neff_values[idx])) if idx < len(neff_values) else -np.inf

        if (
            (core_frac > best_core_frac + 1e-12)
            or (
                abs(core_frac - best_core_frac) <= 1e-12
                and center_frac > best_center_frac + 1e-12
            )
            or (
                abs(core_frac - best_core_frac) <= 1e-12
                and abs(center_frac - best_center_frac) <= 1e-12
                and neff_r > best_neff
            )
        ):
            best_idx = idx
            best_core_frac = core_frac
            best_center_frac = center_frac
            best_neff = neff_r

    return int(best_idx)


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

