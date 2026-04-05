"""Mode-profile math and 3D profile builders for mode sources."""

import logging

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.sources.windows import (
    _compute_transverse_bounds,
    _crop_and_window_2d,
    _make_tukey_window_2d,
    _stagger_both,
    _stagger_half,
)

logger = logging.getLogger(__name__)

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


def _modal_power_2d(e_profile, h_profile, signed_flux_sign, dl):
    """Return 2D modal power using the same convention as port extraction."""
    e = np.asarray(e_profile, dtype=np.complex128).reshape(-1)
    h = np.asarray(h_profile, dtype=np.complex128).reshape(-1)
    n = int(min(e.size, h.size))
    if n <= 0:
        return 0.0
    e = e[:n]
    h = h[:n]
    p = 0.5 * np.real(np.sum(float(signed_flux_sign) * e * np.conjugate(h)) * float(dl))
    return float(p)


def _normalize_2d_pair_by_power(h_profile, e_profile, signed_flux_sign, dl, eps=1e-30):
    """Normalize a 2D Huygens pair so |modal power| equals 1."""
    h = np.asarray(h_profile)
    e = np.asarray(e_profile)
    p = _modal_power_2d(
        np.asarray(e, dtype=np.complex128),
        np.asarray(h, dtype=np.complex128),
        signed_flux_sign=signed_flux_sign,
        dl=dl,
    )
    if np.isfinite(p) and abs(p) > eps:
        scale = np.sqrt(1.0 / abs(p))
        h = h * scale
        e = e * scale
    return h, e


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


def _weighted_local_index(eps_profile_2d, components, eps=1e-30):
    """Field-energy-weighted refractive index over a 2D cross-section."""
    eps_arr = np.asarray(eps_profile_2d)
    if eps_arr.size == 0:
        return 1.0

    w_sum = 0.0
    ew_sum = 0.0
    for comp in components:
        arr = np.asarray(comp)
        if arr.ndim != 2 or arr.size == 0:
            continue
        z = min(eps_arr.shape[0], arr.shape[0])
        t = min(eps_arr.shape[1], arr.shape[1])
        if z <= 0 or t <= 0:
            continue
        sl_eps = eps_arr[:z, :t]
        sl_arr = arr[:z, :t]
        w = np.abs(sl_arr) ** 2
        w_sum += float(np.sum(w))
        ew_sum += float(np.sum(sl_eps * w))

    if w_sum <= eps:
        return float(np.sqrt(max(float(np.mean(eps_arr)), eps)))
    return float(np.sqrt(max(ew_sum / w_sum, eps)))


def _select_3d_impedance_index(axis, pol, eps_profile_2d, Ex, Ey, Ez, Hx, Hy, Hz):
    """Pick a local modal index for 3D impedance targeting."""
    if pol == "tm":
        tangential = {
            "x": (Hy, Hz),
            "y": (Hx, Hz),
            "z": (Hx, Hy),
        }[axis]
    else:
        tangential = {
            "x": (Ey, Ez),
            "y": (Ex, Ez),
            "z": (Ex, Ey),
        }[axis]
    return _weighted_local_index(eps_profile_2d, tangential)


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


def _compute_3d_bounds(axis, center, width, height, resolution, grid_shape):
    nz, ny, nx = grid_shape
    z_center = center[2] if len(center) > 2 else (nz // 2) * resolution
    bounds = {}
    if axis == "x":
        bounds["y"] = _compute_transverse_bounds(center[1], width, resolution, ny)
        bounds["z"] = _compute_transverse_bounds(z_center, height, resolution, nz)
    elif axis == "y":
        bounds["x"] = _compute_transverse_bounds(center[0], width, resolution, nx)
        bounds["z"] = _compute_transverse_bounds(z_center, height, resolution, nz)
    elif axis == "z":
        bounds["x"] = _compute_transverse_bounds(center[0], width, resolution, nx)
        bounds["y"] = _compute_transverse_bounds(center[1], height, resolution, ny)
    else:
        raise ValueError(f"Unsupported axis {axis!r} for 3D profile setup")
    return bounds


def _slice_limit(field, start, end, dim, grid_limit):
    return slice(start, min(end, field.shape[dim], grid_limit))


def _build_3d_indices(axis, staggered, bounds, center_idx, offset_idx, grid_shape):
    nz, ny, nx = grid_shape
    if axis == "x":
        z_start, z_end = bounds["z"]
        y_start, y_end = bounds["y"]
        return {
            "Ex": (
                _slice_limit(staggered["Ex"], z_start, z_end, 0, nz),
                _slice_limit(staggered["Ex"], y_start, y_end, 1, ny),
                offset_idx,
            ),
            "Ey": (
                _slice_limit(staggered["Ey"], z_start, z_end, 0, nz),
                _slice_limit(staggered["Ey"], y_start, y_end, 1, ny - 1),
                center_idx,
            ),
            "Ez": (
                _slice_limit(staggered["Ez"], z_start, z_end, 0, nz - 1),
                _slice_limit(staggered["Ez"], y_start, y_end, 1, ny),
                center_idx,
            ),
            "Hx": (
                _slice_limit(staggered["Hx"], z_start, z_end, 0, nz - 1),
                _slice_limit(staggered["Hx"], y_start, y_end, 1, ny - 1),
                center_idx,
            ),
            "Hy": (
                _slice_limit(staggered["Hy"], z_start, z_end, 0, nz - 1),
                _slice_limit(staggered["Hy"], y_start, y_end, 1, ny),
                offset_idx,
            ),
            "Hz": (
                _slice_limit(staggered["Hz"], z_start, z_end, 0, nz),
                _slice_limit(staggered["Hz"], y_start, y_end, 1, ny - 1),
                offset_idx,
            ),
        }

    if axis == "y":
        z_start, z_end = bounds["z"]
        x_start, x_end = bounds["x"]
        return {
            "Ex": (
                _slice_limit(staggered["Ex"], z_start, z_end, 0, nz),
                center_idx,
                _slice_limit(staggered["Ex"], x_start, x_end, 1, nx - 1),
            ),
            "Ey": (
                _slice_limit(staggered["Ey"], z_start, z_end, 0, nz),
                offset_idx,
                _slice_limit(staggered["Ey"], x_start, x_end, 1, nx),
            ),
            "Ez": (
                _slice_limit(staggered["Ez"], z_start, z_end, 0, nz - 1),
                center_idx,
                _slice_limit(staggered["Ez"], x_start, x_end, 1, nx),
            ),
            "Hx": (
                _slice_limit(staggered["Hx"], z_start, z_end, 0, nz - 1),
                offset_idx,
                _slice_limit(staggered["Hx"], x_start, x_end, 1, nx),
            ),
            "Hy": (
                _slice_limit(staggered["Hy"], z_start, z_end, 0, nz - 1),
                center_idx,
                _slice_limit(staggered["Hy"], x_start, x_end, 1, nx - 1),
            ),
            "Hz": (
                _slice_limit(staggered["Hz"], z_start, z_end, 0, nz),
                offset_idx,
                _slice_limit(staggered["Hz"], x_start, x_end, 1, nx - 1),
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
            _slice_limit(staggered["Ex"], y_start, y_end, 0, ny),
            _slice_limit(staggered["Ex"], x_start, x_end, 1, nx - 1),
        ),
        "Ey": (
            e_z_idx,
            _slice_limit(staggered["Ey"], y_start, y_end, 0, ny - 1),
            _slice_limit(staggered["Ey"], x_start, x_end, 1, nx),
        ),
        "Ez": (
            ez_z_idx,
            _slice_limit(staggered["Ez"], y_start, y_end, 0, ny),
            _slice_limit(staggered["Ez"], x_start, x_end, 1, nx),
        ),
        "Hx": (
            h_z_idx,
            _slice_limit(staggered["Hx"], y_start, y_end, 0, ny - 1),
            _slice_limit(staggered["Hx"], x_start, x_end, 1, nx),
        ),
        "Hy": (
            h_z_idx,
            _slice_limit(staggered["Hy"], y_start, y_end, 0, ny),
            _slice_limit(staggered["Hy"], x_start, x_end, 1, nx - 1),
        ),
        "Hz": (
            hz_z_idx,
            _slice_limit(staggered["Hz"], y_start, y_end, 0, ny - 1),
            _slice_limit(staggered["Hz"], x_start, x_end, 1, nx - 1),
        ),
    }


def _target_3d_impedance(impedance_neff, omega, dt, d_axis):
    eta0 = np.sqrt(MU_0 / EPS_0)
    neff_imp_r = max(float(np.real(impedance_neff)), 1e-6)
    if dt is None:
        return eta0 / neff_imp_r
    k_num_imp = _solve_numeric_k_axis(omega, dt, d_axis, neff_imp_r)
    return _numeric_impedance_axis(omega, dt, d_axis, k_num_imp, neff_imp_r)


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
    bounds = _compute_3d_bounds(axis, center, width, height, resolution, grid_shape)
    indices = _build_3d_indices(
        axis, staggered, bounds, center_idx, offset_idx, grid_shape
    )
    z_target = _target_3d_impedance(impedance_neff, omega, dt, resolution)
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


def _modal_power_3d_from_profiles(profiles, axis, d_area):
    """Compute 3D modal power from profiles on a cross-section."""
    ex = profiles.get("Ex")
    ey = profiles.get("Ey")
    ez = profiles.get("Ez")
    hx = profiles.get("Hx")
    hy = profiles.get("Hy")
    hz = profiles.get("Hz")
    if any(v is None for v in (ex, ey, ez, hx, hy, hz)):
        return 0.0

    ex = np.asarray(ex, dtype=np.complex128)
    ey = np.asarray(ey, dtype=np.complex128)
    ez = np.asarray(ez, dtype=np.complex128)
    hx = np.asarray(hx, dtype=np.complex128)
    hy = np.asarray(hy, dtype=np.complex128)
    hz = np.asarray(hz, dtype=np.complex128)
    if ex.ndim == 1:
        ex = ex[:, None]
    if ey.ndim == 1:
        ey = ey[:, None]
    if ez.ndim == 1:
        ez = ez[:, None]
    if hx.ndim == 1:
        hx = hx[:, None]
    if hy.ndim == 1:
        hy = hy[:, None]
    if hz.ndim == 1:
        hz = hz[:, None]
    ny = min(
        ex.shape[0], ey.shape[0], ez.shape[0], hx.shape[0], hy.shape[0], hz.shape[0]
    )
    nx = min(
        ex.shape[1], ey.shape[1], ez.shape[1], hx.shape[1], hy.shape[1], hz.shape[1]
    )
    if ny <= 0 or nx <= 0:
        return 0.0

    ex = ex[:ny, :nx]
    ey = ey[:ny, :nx]
    ez = ez[:ny, :nx]
    hx = hx[:ny, :nx]
    hy = hy[:ny, :nx]
    hz = hz[:ny, :nx]

    if axis == "x":
        s_axis = ey * np.conjugate(hz) - ez * np.conjugate(hy)
    elif axis == "y":
        s_axis = ez * np.conjugate(hx) - ex * np.conjugate(hz)
    else:
        s_axis = ex * np.conjugate(hy) - ey * np.conjugate(hx)
    return float(0.5 * np.real(np.sum(s_axis) * float(d_area)))


def _normalize_3d_profiles_by_flux(profiles, axis, d_area=1.0, eps=1e-18):
    """Normalize 3D source profiles so |modal power| equals 1."""
    flux = _modal_power_3d_from_profiles(profiles, axis=axis, d_area=d_area)
    if (not np.isfinite(flux)) or abs(flux) <= eps:
        return profiles

    scale = float(np.sqrt(1.0 / max(abs(flux), eps)))
    scale = float(np.clip(scale, 1e-6, 1e6))
    for key, value in profiles.items():
        profiles[key] = np.asarray(value) * scale
    return profiles


def _backward_3d_mode_from_forward(profiles):
    """Return the backward-going counterpart of a forward 3D modal field set."""
    out = {}
    for key, value in profiles.items():
        arr = np.asarray(value, dtype=np.complex128)
        out[key] = -arr if key.startswith("H") else arr.copy()
    return out


def _make_3d_mode_basis_profiles(profiles, axis, d_area=1.0):
    """Build unit-flux forward/backward 3D basis fields from one solved mode."""
    forward = {
        key: np.asarray(value, dtype=np.complex128) for key, value in profiles.items()
    }
    forward = _normalize_3d_profiles_by_flux(forward, axis=axis, d_area=d_area)
    backward = _backward_3d_mode_from_forward(forward)
    return forward, backward


def _modal_overlap_3d_profiles(field_profiles, mode_profiles, axis, d_area):
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

    arrays = {}
    n_common = None
    for name in (e1, e2, h1, h2):
        f_arr = np.asarray(field_profiles[name], dtype=np.complex128).reshape(-1)
        m_arr = np.asarray(mode_profiles[name], dtype=np.complex128).reshape(-1)
        n_local = int(min(f_arr.size, m_arr.size))
        if n_local <= 0:
            return np.complex128(0.0 + 0.0j)
        n_common = n_local if n_common is None else min(n_common, n_local)
        arrays[name] = (f_arr, m_arr)

    n_common = int(max(0, n_common or 0))
    if n_common <= 0:
        return np.complex128(0.0 + 0.0j)

    ef1 = arrays[e1][0][:n_common]
    ef2 = arrays[e2][0][:n_common]
    hf1 = arrays[h1][0][:n_common]
    hf2 = arrays[h2][0][:n_common]
    em1 = arrays[e1][1][:n_common]
    em2 = arrays[e2][1][:n_common]
    hm1 = arrays[h1][1][:n_common]
    hm2 = arrays[h2][1][:n_common]

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
    return np.complex128(overlap)


def _project_3d_profiles_to_real(profiles):
    """Project 3D source profiles to real-valued runtime injection arrays."""
    out = {}
    for key, value in profiles.items():
        out[key] = _to_real_profile(value)
    return out
