import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.sources.solve import solve_modes

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = {"+x", "-x", "+y", "-y", "+z", "-z"}


@dataclass
class _ModeSourceState:
    """Mutable prepared state for ModeSource."""

    _Ex_profile: np.ndarray | None = None
    _Ey_profile: np.ndarray | None = None
    _Ez_profile: np.ndarray | None = None
    _Hx_profile: np.ndarray | None = None
    _Hy_profile: np.ndarray | None = None
    _Hz_profile: np.ndarray | None = None
    _Ex_indices: tuple | None = None
    _Ey_indices: tuple | None = None
    _Ez_indices: tuple | None = None
    _Hx_indices: tuple | None = None
    _Hy_indices: tuple | None = None
    _Hz_indices: tuple | None = None
    _jz_profile: np.ndarray | None = None
    _my_profile: np.ndarray | None = None
    _mz_profile: np.ndarray | None = None
    _jy_profile: np.ndarray | None = None
    _jx_profile: np.ndarray | None = None
    _ez_indices: tuple | None = None
    _h_indices: tuple | None = None
    _hz_indices: tuple | None = None
    _e_indices: tuple | None = None
    _h_component: str | None = None
    _e_component: str | None = None
    _neff: float | None = None
    _impedance_neff: float | None = None
    _dt_physical: float = 0.0
    _launch_dt: float | None = None
    _initialized: bool = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _jax_tukey_window(M: int, alpha: float = 0.5) -> jnp.ndarray:
    """JAX-compatible Tukey (tapered cosine) window.

    Replaces scipy.signal.windows.tukey for differentiable source initialization.

    Args:
        M: Number of points in the window
        alpha: Shape parameter (0 = rectangular, 1 = Hann)

    Returns:
        The Tukey window as a JAX array
    """
    if M <= 0:
        return jnp.array([])
    if M == 1:
        return jnp.ones(1)

    n = jnp.arange(M)
    width = alpha * (M - 1) / 2.0

    # Avoid division by zero when alpha=0
    width = jnp.maximum(width, 1e-10)

    # Three regions: taper up, flat, taper down
    left_taper = 0.5 * (1 + jnp.cos(jnp.pi * (n / width - 1)))
    right_taper = 0.5 * (1 + jnp.cos(jnp.pi * ((n - (M - 1 - width)) / width)))

    window = jnp.where(
        n < width, left_taper, jnp.where(n > (M - 1) - width, right_taper, 1.0)
    )
    return window


def _scipy_tukey(n, alpha=0.3):
    from scipy.signal.windows import tukey

    return tukey(n, alpha=alpha)


def _crop_and_window_2d(profile, z_s, z_e, t_s, t_e, window):
    """Crop a 2D profile to [z_s:z_e, t_s:t_e] and multiply by window.

    Handles shape mismatches by taking the minimum overlap region.
    """
    cropped = profile[z_s:z_e, t_s:t_e]
    if cropped.size == 0:
        return cropped
    if cropped.shape == window.shape:
        return cropped * window
    z_min = min(cropped.shape[0], window.shape[0])
    t_min = min(cropped.shape[1], window.shape[1])
    return cropped[:z_min, :t_min] * window[:z_min, :t_min]


def _make_tukey_window_2d(height_cells, width_cells, alpha=0.3, use_jax=True):
    """Create a 2D Tukey window via outer product of 1D windows."""
    _make = _jax_tukey_window if use_jax else _scipy_tukey
    _ones = jnp.ones if use_jax else np.ones

    wz = (
        _make(height_cells, alpha=alpha)
        if height_cells > 2
        else _ones(max(1, height_cells))
    )
    wt = (
        _make(width_cells, alpha=alpha)
        if width_cells > 2
        else _ones(max(1, width_cells))
    )

    if use_jax:
        return wz[:, jnp.newaxis] * wt[jnp.newaxis, :]
    return wz[:, np.newaxis] * wt[np.newaxis, :]


def _stagger_half(field, axis):
    """Average adjacent cells along *axis* (0 or 1) for Yee half-grid staggering."""
    if field.shape[axis] <= 1:
        return field
    if axis == 0:
        return 0.5 * (field[:-1, :] + field[1:, :])
    return 0.5 * (field[:, :-1] + field[:, 1:])


def _stagger_both(field):
    """Stagger along both axes (for longitudinal H that sits at half-grid in both)."""
    out = field
    if out.shape[1] > 1:
        out = 0.5 * (out[:, :-1] + out[:, 1:])
    if out.shape[0] > 1:
        out = 0.5 * (out[:-1, :] + out[1:, :])
    return out


def _compute_transverse_bounds(center_val, extent, resolution, grid_max):
    """Return (start_idx, end_idx) for the injection window along a transverse axis."""
    center_idx = int(round(center_val / resolution))
    half_idx = int(round((extent / 2) / resolution))
    return max(0, center_idx - half_idx), min(grid_max, center_idx + half_idx)


def _impedance_match_e_profile(e_profile, h_profile, z_target, eps=1e-12):
    """Scale E profile so the least-squares modal impedance matches ``z_target``."""
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
    """Solve 1D Yee dispersion for k at a fixed omega (normal incidence)."""
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
    # Keep the sign: launch direction depends on the signed E/H plane offset.
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
    """Match tangential 3D E components to paired H components via LS impedance fit."""
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

        # Robust local impedance estimate: use pointwise |E|/|H| over strong-field H support.
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
    return (
        e_map["Ex"],
        e_map["Ey"],
        e_map["Ez"],
    )


def _parse_direction(direction):
    """Parse and validate direction string into (axis, sign)."""
    direction = str(direction).lower()
    if direction not in _VALID_DIRECTIONS:
        valid = ", ".join(sorted(_VALID_DIRECTIONS))
        raise ValueError(f"direction must be one of {{{valid}}}, got {direction!r}")
    axis = direction[1]
    dir_sign = 1.0 if direction.startswith("+") else -1.0
    return direction, axis, dir_sign


def _remap_3d_solver_components(Ex, Ey, Ez, Hx, Hy, Hz, axis):
    """Remap solver x-basis components to requested global propagation axis."""
    if axis == "x":
        return Ex, Ey, Ez, Hx, Hy, Hz
    if axis == "y":
        # Use the same right-handed x'->+y, y'->-x, z'->+z basis as the
        # monitor-side modal extraction. Source-profile parity corrections for
        # y-directed launches are applied after profile construction.
        return -Ey, Ex, Ez, -Hy, Hx, Hz
    if axis == "z":
        # Cyclic remap from x-basis -> z-basis.
        return Ey, Ez, Ex, Hy, Hz, Hx
    raise ValueError(f"Unsupported axis {axis!r} for 3D remap")


def _select_3d_phase_ref(axis, pol, Ex, Ey, Ez, Hx, Hy, Hz):
    """Select 3D phase-reference field using the same H-based convention as 2D."""
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

    # Fallback for near-cutoff/degenerate cases: pick strongest tangential H component.
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

    # Robust core estimate that also works with subpixel-smoothed profiles.
    core_mask = eps_flat >= (eps_min + 0.5 * (eps_max - eps_min))
    if not np.any(core_mask):
        return 0

    # Estimate core center for additional center-confinement scoring.
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

        # Secondary discriminator: prefer modes concentrated near core center.
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
    """Build staggered, windowed, impedance-corrected injection profiles for 3D.

    Returns
    -------
    profiles : dict
        Mapping component name -> numpy real profile array.
    indices : dict
        Mapping component name -> index tuple for field injection.
    """
    nz, ny, nx = grid_shape
    dir_sign = 1.0 if direction.startswith("+") else -1.0
    ETA_0 = np.sqrt(MU_0 / EPS_0)
    neff_imp_r = max(float(np.real(impedance_neff)), 1e-6)
    d_axis = resolution
    if dt is not None:
        k_num_imp = _solve_numeric_k_axis(omega, dt, d_axis, neff_imp_r)
        Z_target = _numeric_impedance_axis(omega, dt, d_axis, k_num_imp, neff_imp_r)
    else:
        Z_target = ETA_0 / neff_imp_r

    if axis == "x":
        return _build_3d_x(
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            dir_sign,
            Z_target,
            center,
            width,
            height,
            center_idx,
            offset_idx,
            nz,
            ny,
            nx,
            resolution,
        )
    if axis == "y":
        return _build_3d_y(
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            dir_sign,
            Z_target,
            center,
            width,
            height,
            center_idx,
            offset_idx,
            nz,
            ny,
            nx,
            resolution,
        )
    if axis == "z":
        return _build_3d_z(
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            dir_sign,
            Z_target,
            center,
            width,
            height,
            center_idx,
            offset_idx,
            nz,
            ny,
            nx,
            resolution,
        )
    raise ValueError(f"Unsupported axis {axis!r} for 3D profile setup")


def _build_3d_x(
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
    dir_sign,
    Z_target,
    center,
    width,
    height,
    center_idx,
    offset_idx,
    nz,
    ny,
    nx,
    resolution,
):
    # --- stagger ---
    Ey_s = _stagger_half(Ey, axis=1)
    Ez_s = _stagger_half(Ez, axis=0)
    Ex_s = Ex
    Hx_s = _stagger_both(Hx)
    Hy_s = _stagger_half(Hy, axis=0)
    Hz_s = _stagger_half(Hz, axis=1)

    # --- transverse bounds ---
    y_start, y_end = _compute_transverse_bounds(center[1], width, resolution, ny)
    z_center = center[2] if len(center) > 2 else (nz // 2) * resolution
    z_start, z_end = _compute_transverse_bounds(z_center, height, resolution, nz)

    # --- indices  (z_slice, y_slice, x_index) ---
    indices = {
        "Ex": (
            slice(z_start, min(z_end, Ex_s.shape[0], nz)),
            slice(y_start, min(y_end, Ex_s.shape[1], ny)),
            offset_idx,
        ),
        "Ey": (
            slice(z_start, min(z_end, Ey_s.shape[0], nz)),
            slice(y_start, min(y_end, Ey_s.shape[1], ny - 1)),
            center_idx,
        ),
        "Ez": (
            slice(z_start, min(z_end, Ez_s.shape[0], nz - 1)),
            slice(y_start, min(y_end, Ez_s.shape[1], ny)),
            center_idx,
        ),
        "Hx": (
            slice(z_start, min(z_end, Hx_s.shape[0], nz - 1)),
            slice(y_start, min(y_end, Hx_s.shape[1], ny - 1)),
            center_idx,
        ),
        "Hy": (
            slice(z_start, min(z_end, Hy_s.shape[0], nz - 1)),
            slice(y_start, min(y_end, Hy_s.shape[1], ny)),
            offset_idx,
        ),
        "Hz": (
            slice(z_start, min(z_end, Hz_s.shape[0], nz)),
            slice(y_start, min(y_end, Hz_s.shape[1], ny - 1)),
            offset_idx,
        ),
    }

    # Match J/M balance using pairwise tangential impedance fits before windowing.
    Ex_s, Ey_s, Ez_s = _impedance_match_3d_tangential_pairs(
        "x",
        Ex_s,
        Ey_s,
        Ez_s,
        Hx_s,
        Hy_s,
        Hz_s,
        Z_target,
    )

    # --- crop & window ---
    staggered = {"Ex": Ex_s, "Ey": Ey_s, "Ez": Ez_s, "Hx": Hx_s, "Hy": Hy_s, "Hz": Hz_s}
    profiles = _crop_and_window_all(
        staggered,
        z_start,
        z_end,
        y_start,
        y_end,
        dir_sign,
        use_jax=True,
        alpha=0.2,
    )
    d_area = float(resolution * resolution)
    profiles = _normalize_3d_profiles_by_flux(profiles, axis="x", d_area=d_area)

    extra = {
        "_y_start": y_start,
        "_y_end": y_end,
        "_z_start": z_start,
        "_z_end": z_end,
        "_h_component": "Hy",
        "_e_component": "Ey",
    }
    return profiles, indices, extra


def _build_3d_y(
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
    dir_sign,
    Z_target,
    center,
    width,
    height,
    center_idx,
    offset_idx,
    nz,
    ny,
    nx,
    resolution,
):
    # --- stagger (y-propagation) ---
    Ex_s = _stagger_half(Ex, axis=1)
    Ey_s = Ey
    Ez_s = _stagger_half(Ez, axis=0)
    Hx_s = _stagger_half(Hx, axis=0)
    Hy_s = _stagger_both(Hy)
    Hz_s = _stagger_half(Hz, axis=1)

    # --- transverse bounds ---
    x_start, x_end = _compute_transverse_bounds(center[0], width, resolution, nx)
    z_center = center[2] if len(center) > 2 else (nz // 2) * resolution
    z_start, z_end = _compute_transverse_bounds(z_center, height, resolution, nz)

    # --- indices  (z_slice, y_index, x_slice) ---
    indices = {
        "Ex": (
            slice(z_start, min(z_end, Ex_s.shape[0], nz)),
            center_idx,
            slice(x_start, min(x_end, Ex_s.shape[1], nx - 1)),
        ),
        "Ey": (
            slice(z_start, min(z_end, Ey_s.shape[0], nz)),
            offset_idx,
            slice(x_start, min(x_end, Ey_s.shape[1], nx)),
        ),
        "Ez": (
            slice(z_start, min(z_end, Ez_s.shape[0], nz - 1)),
            center_idx,
            slice(x_start, min(x_end, Ez_s.shape[1], nx)),
        ),
        "Hx": (
            slice(z_start, min(z_end, Hx_s.shape[0], nz - 1)),
            offset_idx,
            slice(x_start, min(x_end, Hx_s.shape[1], nx)),
        ),
        "Hy": (
            slice(z_start, min(z_end, Hy_s.shape[0], nz - 1)),
            center_idx,
            slice(x_start, min(x_end, Hy_s.shape[1], nx - 1)),
        ),
        "Hz": (
            slice(z_start, min(z_end, Hz_s.shape[0], nz)),
            offset_idx,
            slice(x_start, min(x_end, Hz_s.shape[1], nx - 1)),
        ),
    }

    # Match J/M balance using pairwise tangential impedance fits before windowing.
    Ex_s, Ey_s, Ez_s = _impedance_match_3d_tangential_pairs(
        "y",
        Ex_s,
        Ey_s,
        Ez_s,
        Hx_s,
        Hy_s,
        Hz_s,
        Z_target,
    )

    # --- crop & window ---
    staggered = {"Ex": Ex_s, "Ey": Ey_s, "Ez": Ez_s, "Hx": Hx_s, "Hy": Hy_s, "Hz": Hz_s}
    profiles = _crop_and_window_all(
        staggered,
        z_start,
        z_end,
        x_start,
        x_end,
        dir_sign,
        use_jax=False,
        alpha=0.2,
    )
    d_area = float(resolution * resolution)
    profiles = _normalize_3d_profiles_by_flux(profiles, axis="y", d_area=d_area)
    if dir_sign < 0.0:
        for comp in ("Ex", "Ey", "Ez"):
            if profiles.get(comp) is not None:
                profiles[comp] = -profiles[comp]

    extra = {
        "_x_start": x_start,
        "_x_end": x_end,
        "_z_start": z_start,
        "_z_end": z_end,
        "_h_component": "Hx",
        "_e_component": "Ex",
    }
    return profiles, indices, extra


def _build_3d_z(
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
    dir_sign,
    Z_target,
    center,
    width,
    height,
    center_idx,
    offset_idx,
    nz,
    ny,
    nx,
    resolution,
):
    # --- stagger (z-propagation) ---
    Ex_s = _stagger_half(Ex, axis=1)
    Ey_s = _stagger_half(Ey, axis=0)
    Ez_s = Ez
    Hx_s = _stagger_half(Hx, axis=0)
    Hy_s = _stagger_half(Hy, axis=1)
    Hz_s = _stagger_both(Hz)

    # --- transverse bounds ---
    x_start, x_end = _compute_transverse_bounds(center[0], width, resolution, nx)
    y_start, y_end = _compute_transverse_bounds(center[1], height, resolution, ny)

    e_z_idx = int(np.clip(center_idx, 0, nz - 1))
    h_z_idx = int(np.clip(offset_idx, 0, max(nz - 2, 0)))
    ez_z_idx = int(np.clip(center_idx, 0, max(nz - 2, 0)))
    hz_z_idx = int(np.clip(offset_idx, 0, nz - 1))

    # --- indices  (z_index, y_slice, x_slice) ---
    indices = {
        "Ex": (
            e_z_idx,
            slice(y_start, min(y_end, Ex_s.shape[0], ny)),
            slice(x_start, min(x_end, Ex_s.shape[1], nx - 1)),
        ),
        "Ey": (
            e_z_idx,
            slice(y_start, min(y_end, Ey_s.shape[0], ny - 1)),
            slice(x_start, min(x_end, Ey_s.shape[1], nx)),
        ),
        "Ez": (
            ez_z_idx,
            slice(y_start, min(y_end, Ez_s.shape[0], ny)),
            slice(x_start, min(x_end, Ez_s.shape[1], nx)),
        ),
        "Hx": (
            h_z_idx,
            slice(y_start, min(y_end, Hx_s.shape[0], ny - 1)),
            slice(x_start, min(x_end, Hx_s.shape[1], nx)),
        ),
        "Hy": (
            h_z_idx,
            slice(y_start, min(y_end, Hy_s.shape[0], ny)),
            slice(x_start, min(x_end, Hy_s.shape[1], nx - 1)),
        ),
        "Hz": (
            hz_z_idx,
            slice(y_start, min(y_end, Hz_s.shape[0], ny - 1)),
            slice(x_start, min(x_end, Hz_s.shape[1], nx - 1)),
        ),
    }

    # Match J/M balance using pairwise tangential impedance fits before windowing.
    Ex_s, Ey_s, Ez_s = _impedance_match_3d_tangential_pairs(
        "z",
        Ex_s,
        Ey_s,
        Ez_s,
        Hx_s,
        Hy_s,
        Hz_s,
        Z_target,
    )

    # --- crop & window ---
    staggered = {"Ex": Ex_s, "Ey": Ey_s, "Ez": Ez_s, "Hx": Hx_s, "Hy": Hy_s, "Hz": Hz_s}
    profiles = _crop_and_window_all(
        staggered,
        y_start,
        y_end,
        x_start,
        x_end,
        dir_sign,
        use_jax=True,
        alpha=0.2,
    )
    d_area = float(resolution * resolution)
    profiles = _normalize_3d_profiles_by_flux(profiles, axis="z", d_area=d_area)

    extra = {
        "_x_start": x_start,
        "_x_end": x_end,
        "_y_start": y_start,
        "_y_end": y_end,
        "_h_component": "Hx",
        "_e_component": "Ex",
    }
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


def _runtime_3d_profiles(profiles, axis: str, direction_sign: float):
    """Apply runtime-only gauge corrections without mutating stored profiles."""
    out = dict(profiles)
    if axis != "y":
        return out

    if direction_sign > 0.0:
        if out.get("Ex") is not None:
            out["Ex"] = -out["Ex"]
        if out.get("Hz") is not None:
            out["Hz"] = -out["Hz"]
    else:
        if out.get("Ez") is not None:
            out["Ez"] = -out["Ez"]
        if out.get("Hx") is not None:
            out["Hx"] = -out["Hx"]

    return out


# ---------------------------------------------------------------------------
# Huygens cross-product sign tables
# sign = multiplier in: target += sign * source_profile * sig * dt / (material * dx)
# Only transverse components are injected; longitudinal components are skipped.
# ---------------------------------------------------------------------------
_HUYGENS_SIGNS = {
    "x": {
        "e": [("Ey", "Hz", -1), ("Ez", "Hy", +1)],
        "h": [("Hy", "Ez", -1), ("Hz", "Ey", +1)],
    },
    "y": {
        "e": [("Ex", "Hz", -1), ("Ez", "Hx", +1)],
        "h": [("Hx", "Ez", +1), ("Hz", "Ex", -1)],
    },
    "z": {
        "e": [("Ex", "Hy", -1), ("Ey", "Hx", +1)],
        "h": [("Hx", "Ey", +1), ("Hy", "Ex", -1)],
    },
}


def _get_3d_huygens_terms(axis, pol):
    """Return 3D sign terms with TE gauge parity matched to 2D conventions."""
    e_terms = list(_HUYGENS_SIGNS[axis]["e"])
    h_terms = list(_HUYGENS_SIGNS[axis]["h"])
    return e_terms, h_terms


def _inject_3d_e_fields(fields, profiles, indices, signal_e, dt, resolution, axis, pol):
    """Inject E-field components for 3D Huygens source (J = n x H), after E update."""
    e_terms, _ = _get_3d_huygens_terms(axis, pol)
    for e_comp, h_source, sign in e_terms:
        _inject_e_component(
            fields,
            e_comp,
            profiles,
            indices,
            h_source,
            signal_e,
            dt,
            resolution,
            sign=sign,
        )


def _inject_3d_h_fields(fields, profiles, indices, signal_h, dt, resolution, axis, pol):
    """Inject H-field components for 3D Huygens source (M = -n x E), after H update."""
    _, h_terms = _get_3d_huygens_terms(axis, pol)
    for h_comp, e_source, sign in h_terms:
        _inject_h_component(
            fields,
            h_comp,
            profiles,
            indices,
            e_source,
            signal_h,
            dt,
            resolution,
            sign=sign,
        )


def _inject_3d_fields(
    fields, profiles, indices, signal_e, signal_h, dt, resolution, axis="x", pol="tm"
):
    """Inject all field components into a 3D field object (backward compat wrapper)."""
    _inject_3d_h_fields(fields, profiles, indices, signal_h, dt, resolution, axis, pol)
    _inject_3d_e_fields(fields, profiles, indices, signal_e, dt, resolution, axis, pol)


def _inject_e_component(
    fields, comp, profiles, indices, j_source, sig, dt, res, sign=-1
):
    """Inject one E-field component via J = cross(n, H)."""
    profile = profiles.get(comp)
    idx = indices.get(comp)
    j_term = profiles.get(j_source)
    if profile is None or idx is None or j_term is None:
        return
    target = getattr(fields, comp)[idx]
    j_term = _match_shape(j_term, target.shape)
    if j_term is None:
        logger.debug("Shape mismatch injecting %s, skipping", comp)
        return
    eps = fields.permittivity[idx]
    setattr(
        fields,
        comp,
        getattr(fields, comp)
        .at[idx]
        .add(sign * j_term * sig * dt / (EPS_0 * eps * res)),
    )


def _inject_h_component(
    fields, comp, profiles, indices, m_source, sig, dt, res, sign=-1
):
    """Inject one H-field component via M = -cross(n, E)."""
    profile = profiles.get(comp)
    idx = indices.get(comp)
    m_term = profiles.get(m_source)
    if profile is None or idx is None or m_term is None:
        return
    target = getattr(fields, comp)[idx]
    m_term = _match_shape(m_term, target.shape)
    if m_term is None:
        logger.debug("Shape mismatch injecting %s, skipping", comp)
        return
    mu = getattr(fields, "permeability", None)
    mu_val = mu[idx] if mu is not None else 1.0
    setattr(
        fields,
        comp,
        getattr(fields, comp)
        .at[idx]
        .add(sign * m_term * sig * dt / (MU_0 * mu_val * res)),
    )


def _match_shape(profile, target_shape):
    """Match profile shape to target field shape, trimming or padding as needed."""
    if profile is None:
        return None
    profile = np.squeeze(profile)
    if profile.shape == target_shape:
        return profile
    if profile.ndim == len(target_shape):
        slices = tuple(
            slice(0, min(profile.shape[i], target_shape[i]))
            for i in range(profile.ndim)
        )
        trimmed = profile[slices]
        if trimmed.shape == target_shape:
            return trimmed
        result = np.zeros(target_shape, dtype=profile.dtype)
        result[tuple(slice(0, trimmed.shape[i]) for i in range(trimmed.ndim))] = trimmed
        return result
    return None


# ---------------------------------------------------------------------------
# ModeSource class
# ---------------------------------------------------------------------------


class ModeSource:
    """Huygens mode source on Yee grid supporting ±x/±y in 2D and ±x/±y/±z in 3D.

    In 3D, injects all 6 field components (Ex, Ey, Ez, Hx, Hy, Hz) for accurate
    mode injection, accounting for proper Yee grid staggering.
    """

    _RUNTIME_ATTRS = {
        "_Ex_profile",
        "_Ey_profile",
        "_Ez_profile",
        "_Hx_profile",
        "_Hy_profile",
        "_Hz_profile",
        "_Ex_indices",
        "_Ey_indices",
        "_Ez_indices",
        "_Hx_indices",
        "_Hy_indices",
        "_Hz_indices",
        "_jz_profile",
        "_my_profile",
        "_mz_profile",
        "_jy_profile",
        "_jx_profile",
        "_ez_indices",
        "_h_indices",
        "_hz_indices",
        "_e_indices",
        "_h_component",
        "_e_component",
        "_neff",
        "_impedance_neff",
        "_dt_physical",
        "_launch_dt",
        "_initialized",
        "_resolution",
        "_is_3d",
        "_grid_shape",
        "_axis",
        "_eps_profile_2d",
        "_y_start",
        "_y_end",
        "_x_start",
        "_x_end",
    }

    def __getattr__(self, name):
        if name in self._RUNTIME_ATTRS and "_state" in self.__dict__:
            return getattr(self._state, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in self._RUNTIME_ATTRS and "_state" in self.__dict__:
            setattr(self._state, name, value)
            return
        object.__setattr__(self, name, value)

    def __init__(
        self, grid, center, width, wavelength, pol, signal, direction="+x", height=None
    ):
        self.grid = grid
        self.center = (
            center if isinstance(center, (tuple, list)) else (center, grid.height / 2)
        )
        self.width = width
        self.height = height
        self.wavelength = wavelength
        self.pol = str(pol).lower()
        if self.pol not in {"te", "tm"}:
            raise ValueError(f"pol must be 'te' or 'tm', got {pol!r}")
        self.signal = signal
        self.direction, self._direction_axis, self._direction_sign = _parse_direction(
            direction
        )
        self._state = _ModeSourceState()

    def initialize(self, permittivity, resolution, dt=None):
        """Compute the mode and set up the source currents for all 6 components in 3D."""
        dx = dy = resolution
        is_3d = permittivity.ndim == 3
        self._resolution = resolution
        self._is_3d = is_3d

        if is_3d:
            nz, ny, nx = permittivity.shape
            dz = resolution
            self._grid_shape = (nz, ny, nx)
            if self.height is None:
                self.height = self.width
        else:
            ny, nx = permittivity.shape
            nz = 1
            dz = resolution
            self._grid_shape = (ny, nx)
            self.height = None

        axis = self._direction_axis
        if (not is_3d) and axis == "z":
            raise ValueError(
                "direction '+z'/'-z' requires a 3D permittivity grid; received 2D data"
            )
        self._axis = axis
        self._dt_physical = 0.0
        self._launch_dt = dt

        # 1. Get center index for injection plane
        if axis == "x":
            center_idx = int(np.clip(np.round(self.center[0] / dx - 0.5), 0, nx - 1))
            if self.direction == "+x":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(nx - 2, center_idx + 1)

            if is_3d:
                eps_profile = permittivity[:, :, center_idx]
                self._eps_profile_2d = eps_profile
            else:
                eps_profile = permittivity[:, center_idx]
                self._eps_profile_2d = None

        elif axis == "y":
            center_idx = int(np.clip(np.round(self.center[1] / dy - 0.5), 0, ny - 1))
            if self.direction == "+y":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(ny - 2, center_idx + 1)

            if is_3d:
                eps_profile = permittivity[:, center_idx, :]
                self._eps_profile_2d = eps_profile
            else:
                eps_profile = permittivity[center_idx, :]
                self._eps_profile_2d = None

        else:  # axis == "z" (3D only)
            center_idx = int(np.clip(np.round(self.center[2] / dz - 0.5), 0, nz - 1))
            if self.direction == "+z":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(nz - 2, center_idx + 1)

            eps_profile = permittivity[center_idx, :, :]
            self._eps_profile_2d = eps_profile

        # 2. Solve for mode fields
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        dL = dz if is_3d else (dy if axis == "x" else dx)
        solver_direction = self.direction
        if is_3d and axis == "x":
            # x-directed 3D launches are still anchored to the opposite signed
            # solver branch to match the source-plane staggering conventions.
            solver_direction = ("-" if self.direction.startswith("+") else "+") + axis
        elif is_3d and axis == "y":
            # y-directed 3D launches are solved in a fixed +y basis for a
            # stable gauge; launch direction is handled by source timing.
            solver_direction = "+y"

        eps_profile_arr = np.asarray(eps_profile)
        n_local_max = float(
            np.sqrt(max(float(np.max(np.real(eps_profile_arr))), 1e-12))
        )
        # Bias the solver toward the guided branch in large windows where cladding-like
        # continuum modes can otherwise dominate the sort order.
        target_neff = 0.98 * n_local_max

        mode_candidates = 3
        try:
            neff_val, e_fields, h_fields, _ = solve_modes(
                eps=eps_profile,
                omega=omega,
                dL=dL,
                m=mode_candidates,
                direction=solver_direction,
                filter_pol=self.pol,
                target_neff=target_neff,
                return_fields=True,
            )
        except ValueError:
            neff_val, e_fields, h_fields, _ = solve_modes(
                eps=eps_profile,
                omega=omega,
                dL=dL,
                m=1,
                direction=solver_direction,
                filter_pol=self.pol,
                target_neff=target_neff,
                return_fields=True,
            )

        mode_idx = _select_core_confined_mode_index(eps_profile, e_fields, neff_val)
        self._neff = neff_val[mode_idx]
        E_mode = e_fields[mode_idx]
        H_mode = h_fields[mode_idx]

        # 3. Extract all 6 components and convert to JAX arrays
        Ex_raw = jnp.asarray(jnp.squeeze(E_mode[0]))
        Ey_raw = jnp.asarray(jnp.squeeze(E_mode[1]))
        Ez_raw = jnp.asarray(jnp.squeeze(E_mode[2]))
        Hx_raw = jnp.asarray(jnp.squeeze(H_mode[0]))
        Hy_raw = jnp.asarray(jnp.squeeze(H_mode[1]))
        Hz_raw = jnp.asarray(jnp.squeeze(H_mode[2]))

        if is_3d:
            Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw = (
                _remap_3d_solver_components(
                    Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw, axis
                )
            )

        # 4. Phase-align all components with the same H-based gauge convention as 2D.
        if is_3d:
            ref_field = _select_3d_phase_ref(
                axis, self.pol, Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw
            )
        elif self.pol == "tm":
            ex_max = jnp.max(jnp.abs(Ex_raw))
            ey_max = jnp.max(jnp.abs(Ey_raw))
            ez_max = jnp.max(jnp.abs(Ez_raw))
            ref_field = jnp.where(
                ex_max > ey_max,
                jnp.where(ex_max > ez_max, Ex_raw, Ez_raw),
                jnp.where(ey_max > ez_max, Ey_raw, Ez_raw),
            )
        else:
            ref_field = Ey_raw if axis == "x" else Ex_raw
            ref_field = jnp.where(jnp.max(jnp.abs(ref_field)) < 1e-9, Ez_raw, ref_field)

        idx_max = jnp.argmax(jnp.abs(ref_field))
        phase_ref = jnp.angle(ref_field.flatten()[idx_max])

        Ex_aligned = Ex_raw * jnp.exp(-1j * phase_ref)
        Ey_aligned = Ey_raw * jnp.exp(-1j * phase_ref)
        Ez_aligned = Ez_raw * jnp.exp(-1j * phase_ref)
        Hx_aligned = Hx_raw * jnp.exp(-1j * phase_ref)
        Hy_aligned = Hy_raw * jnp.exp(-1j * phase_ref)
        Hz_aligned = Hz_raw * jnp.exp(-1j * phase_ref)

        # 5. Apply Yee grid staggering and set up indices
        if is_3d:
            self._impedance_neff = _select_3d_impedance_index(
                axis,
                self.pol,
                self._eps_profile_2d,
                Ex_aligned,
                Ey_aligned,
                Ez_aligned,
                Hx_aligned,
                Hy_aligned,
                Hz_aligned,
            )
            self._setup_3d_injection(
                Ex_aligned,
                Ey_aligned,
                Ez_aligned,
                Hx_aligned,
                Hy_aligned,
                Hz_aligned,
                center_idx,
                offset_idx,
                axis,
                nz,
                ny,
                nx,
                resolution,
                omega=omega,
                dt=dt,
            )
        else:
            self._impedance_neff = None
            self._setup_2d_injection(
                E_mode, H_mode, center_idx, offset_idx, axis, ny, nx, resolution
            )

        self._compute_dt_physical(axis, is_3d, dx, dy, dz, dt=dt)
        self._initialized = True

    def _setup_3d_injection(
        self,
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
        center_idx,
        offset_idx,
        axis,
        nz,
        ny,
        nx,
        resolution,
        omega,
        dt,
    ):
        """Set up full 6-component injection for 3D simulations."""
        profiles, indices, extra = _build_3d_profiles(
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            axis=axis,
            direction=self.direction,
            center=self.center,
            width=self.width,
            height=self.height,
            center_idx=center_idx,
            offset_idx=offset_idx,
            grid_shape=(nz, ny, nx),
            resolution=resolution,
            impedance_neff=(
                self._impedance_neff if self._impedance_neff is not None else self._neff
            ),
            omega=omega,
            dt=dt,
        )

        # Store profiles on self
        self._Ex_profile = profiles.get("Ex")
        self._Ey_profile = profiles.get("Ey")
        self._Ez_profile = profiles.get("Ez")
        self._Hx_profile = profiles.get("Hx")
        self._Hy_profile = profiles.get("Hy")
        self._Hz_profile = profiles.get("Hz")

        # Store indices on self
        self._Ex_indices = indices.get("Ex")
        self._Ey_indices = indices.get("Ey")
        self._Ez_indices = indices.get("Ez")
        self._Hx_indices = indices.get("Hx")
        self._Hy_indices = indices.get("Hy")
        self._Hz_indices = indices.get("Hz")

        # Store extra metadata
        for key, val in extra.items():
            setattr(self, key, val)

        # Legacy compatibility
        self._jz_profile = self._Hz_profile
        self._my_profile = self._Ez_profile

    def _setup_2d_injection(
        self, E_mode, H_mode, center_idx, offset_idx, axis, ny, nx, resolution
    ):
        """2D injection setup using explicit global component mapping.

        `solve_modes(..., return_fields=True)` returns fields ordered as:
        E_mode = [Ex, Ey, Ez], H_mode = [Hx, Hy, Hz] in global components.
        We pick the physically matching TE/TM pair for the chosen propagation axis.
        """
        dir_sign = 1.0 if self.direction.startswith("+") else -1.0
        ETA_0 = np.sqrt(MU_0 / EPS_0)
        z_target = ETA_0 / max(np.real(self._neff), 1e-6)

        if axis == "x":
            self._setup_2d_x(
                E_mode,
                H_mode,
                center_idx,
                offset_idx,
                ny,
                nx,
                resolution,
                dir_sign,
                z_target,
            )
        else:
            self._setup_2d_y(
                E_mode,
                H_mode,
                center_idx,
                offset_idx,
                ny,
                nx,
                resolution,
                dir_sign,
                z_target,
            )

    def _setup_2d_x(
        self,
        E_mode,
        H_mode,
        center_idx,
        offset_idx,
        ny,
        nx,
        resolution,
        dir_sign,
        z_target,
    ):
        """2D injection setup for x-propagation."""
        center_y_idx = int(round(self.center[1] / resolution))
        half_width_idx = int(round((self.width / 2) / resolution))
        y_start = max(0, center_y_idx - half_width_idx)
        y_end = min(ny, center_y_idx + half_width_idx)
        y_slice = slice(y_start, y_end)
        self._y_start = y_start
        self._y_end = y_end

        if self.pol == "tm":
            self._ez_indices = (y_slice, center_idx)
            self._h_indices = (y_slice, offset_idx)
            self._h_component = "Hx"

            # +x TM: (Ez, Hy)
            Hy_raw = np.squeeze(H_mode[1])
            Ez_raw = np.squeeze(E_mode[2])

            idx_max = np.argmax(np.abs(Hy_raw))
            phase_ref = np.angle(Hy_raw.flatten()[idx_max])
            Hy_profile = Hy_raw * np.exp(-1j * phase_ref)
            Ez_profile = Ez_raw * np.exp(-1j * phase_ref)
            Ez_profile = _impedance_match_e_profile(Ez_profile, Hy_profile, z_target)

            width_cells = y_end - y_start
            window = self._make_1d_window(width_cells)

            Hy_cropped = Hy_profile[y_start:y_end]
            Ez_cropped = Ez_profile[y_start:y_end]
            if len(Hy_cropped) == len(window):
                Hy_cropped = Hy_cropped * window
                Ez_cropped = Ez_cropped * window

            jz_profile = dir_sign * Hy_cropped
            my_profile = dir_sign * Ez_cropped
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=-1.0, dl=resolution
            )
            jz_profile = _to_real_profile(jz_profile)
            my_profile = _to_real_profile(my_profile)
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=-1.0, dl=resolution
            )

            self._jz_profile = jz_profile
            self._my_profile = my_profile

        else:  # TE
            hz_col = (
                max(0, offset_idx - 1)
                if self.direction == "+x"
                else min(nx - 2, offset_idx)
            )

            self._hz_indices = (slice(y_start, min(y_end, ny - 1)), hz_col)
            self._e_indices = (slice(y_start, min(y_end, ny - 1)), offset_idx)
            self._e_component = "Ey"

            # +x TE: (Ey, Hz)
            Hz_raw = np.squeeze(H_mode[2])
            Ey_raw = np.squeeze(E_mode[1])

            Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
            Ey_staggered = 0.5 * (Ey_raw[:-1] + Ey_raw[1:])

            idx_max = np.argmax(np.abs(Hz_staggered))
            phase_ref = np.angle(Hz_staggered.flatten()[idx_max])
            Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
            Ey_profile = Ey_staggered * np.exp(-1j * phase_ref)
            Ey_profile = _impedance_match_e_profile(Ey_profile, Hz_profile, z_target)

            width_cells = min(y_end, len(Hz_profile)) - y_start
            window = self._make_1d_window(width_cells)

            Hz_cropped = Hz_profile[y_start : min(y_end, len(Hz_profile))]
            Ey_cropped = Ey_profile[y_start : min(y_end, len(Ey_profile))]
            if len(Hz_cropped) == len(window):
                Hz_cropped = Hz_cropped * window
                Ey_cropped = Ey_cropped * window

            # Relative J/M sign controls propagation handedness for TE in x-propagation.
            jy_profile = dir_sign * Hz_cropped
            mz_profile = -dir_sign * Ey_cropped
            jy_profile, mz_profile = _normalize_2d_pair_by_power(
                jy_profile, mz_profile, signed_flux_sign=1.0, dl=resolution
            )
            jy_profile = _to_real_profile(jy_profile)
            mz_profile = _to_real_profile(mz_profile)
            jy_profile, mz_profile = _normalize_2d_pair_by_power(
                jy_profile, mz_profile, signed_flux_sign=1.0, dl=resolution
            )

            self._jy_profile = jy_profile
            self._mz_profile = mz_profile

    def _setup_2d_y(
        self,
        E_mode,
        H_mode,
        center_idx,
        offset_idx,
        ny,
        nx,
        resolution,
        dir_sign,
        z_target,
    ):
        """2D injection setup for y-propagation."""
        center_x_idx = int(round(self.center[0] / resolution))
        half_width_idx = int(round((self.width / 2) / resolution))
        x_start = max(0, center_x_idx - half_width_idx)
        x_end = min(nx, center_x_idx + half_width_idx)
        x_slice = slice(x_start, x_end)
        self._x_start = x_start
        self._x_end = x_end

        if self.pol == "tm":
            self._ez_indices = (center_idx, x_slice)
            self._h_indices = (offset_idx, x_slice)
            self._h_component = "Hy"

            # +y TM uses the rotated x-solver basis: (Ez, Hx) <- (Ez, Hy_xbasis)
            Hx_raw = np.squeeze(H_mode[1])
            Ez_raw = np.squeeze(E_mode[2])

            idx_max = np.argmax(np.abs(Hx_raw))
            phase_ref = np.angle(Hx_raw.flatten()[idx_max])
            Hx_profile = Hx_raw * np.exp(-1j * phase_ref)
            Ez_profile = Ez_raw * np.exp(-1j * phase_ref)
            Ez_profile = _impedance_match_e_profile(Ez_profile, Hx_profile, z_target)

            width_cells = x_end - x_start
            window = self._make_1d_window(width_cells)

            Hx_cropped = Hx_profile[x_start:x_end]
            Ez_cropped = Ez_profile[x_start:x_end]
            if len(Hx_cropped) == len(window):
                Hx_cropped = Hx_cropped * window
                Ez_cropped = Ez_cropped * window

            # Relative J/M sign controls propagation handedness for TM in y-propagation.
            jz_profile = dir_sign * Hx_cropped
            my_profile = -dir_sign * Ez_cropped
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=1.0, dl=resolution
            )
            jz_profile = _to_real_profile(jz_profile)
            my_profile = _to_real_profile(my_profile)
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=1.0, dl=resolution
            )

            self._jz_profile = jz_profile
            self._my_profile = my_profile

        else:  # TE y-prop
            hz_row = (
                max(0, offset_idx - 1)
                if self.direction == "+y"
                else min(ny - 2, offset_idx)
            )

            self._hz_indices = (hz_row, slice(x_start, min(x_end, nx - 1)))
            self._e_indices = (offset_idx, slice(x_start, min(x_end, nx - 1)))
            self._e_component = "Ex"

            # +y TE uses the rotated x-solver basis: (Ex, Hz) <- (Ey_xbasis, Hz_xbasis)
            Hz_raw = np.squeeze(H_mode[2])
            Ex_raw = np.squeeze(E_mode[1])

            Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
            Ex_staggered = 0.5 * (Ex_raw[:-1] + Ex_raw[1:])

            idx_max = np.argmax(np.abs(Hz_staggered))
            phase_ref = np.angle(Hz_staggered.flatten()[idx_max])
            Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
            Ex_profile = Ex_staggered * np.exp(-1j * phase_ref)
            Ex_profile = _impedance_match_e_profile(Ex_profile, Hz_profile, z_target)

            width_cells = min(x_end, len(Hz_profile)) - x_start
            window = self._make_1d_window(width_cells)

            Hz_cropped = Hz_profile[x_start : min(x_end, len(Hz_profile))]
            Ex_cropped = Ex_profile[x_start : min(x_end, len(Ex_profile))]
            if len(Hz_cropped) == len(window):
                Hz_cropped = Hz_cropped * window
                Ex_cropped = Ex_cropped * window

            jx_profile = -dir_sign * Hz_cropped
            mz_profile = -dir_sign * Ex_cropped
            jx_profile, mz_profile = _normalize_2d_pair_by_power(
                jx_profile, mz_profile, signed_flux_sign=-1.0, dl=resolution
            )
            jx_profile = _to_real_profile(jx_profile)
            mz_profile = _to_real_profile(mz_profile)
            jx_profile, mz_profile = _normalize_2d_pair_by_power(
                jx_profile, mz_profile, signed_flux_sign=-1.0, dl=resolution
            )

            self._jx_profile = jx_profile
            self._mz_profile = mz_profile

    @staticmethod
    def _make_1d_window(width_cells, alpha=0.3):
        """Create a 1D Tukey window for smooth edges."""
        if width_cells > 2:
            from scipy.signal.windows import tukey

            return tukey(width_cells, alpha=alpha)
        return np.ones(max(1, width_cells))

    def _compute_dt_physical(self, axis, is_3d, dx, dy, dz=None, dt=None):
        """Compute physical time shift between E and H injection planes."""
        if self._neff is None:
            return
        if dz is None:
            dz = dx

        coord_e = 0.0
        coord_h = 0.0

        if is_3d:
            e_comp, h_comp = _dominant_3d_pair(axis, self.pol)
            e_indices = getattr(self, f"_{e_comp}_indices", None)
            h_indices = getattr(self, f"_{h_comp}_indices", None)
            e_axis_idx = _axis_index_from_component_indices(e_indices, axis)
            h_axis_idx = _axis_index_from_component_indices(h_indices, axis)
            coord_e = _component_axis_coord(e_comp, e_axis_idx, axis, dx, dy, dz)
            coord_h = _component_axis_coord(h_comp, h_axis_idx, axis, dx, dy, dz)
        else:
            if axis == "x":
                if self.pol == "tm":
                    idx_e = self._ez_indices[1] if self._ez_indices else 0
                    idx_h = self._h_indices[1] if self._h_indices else 0
                else:
                    idx_e = self._e_indices[1] if self._e_indices else 0
                    idx_h = self._hz_indices[1] if self._hz_indices else 0
                coord_e = (idx_e + 0.5) * dx
                coord_h = (idx_h + 1.0) * dx
            else:
                if self.pol == "tm":
                    idx_e = self._ez_indices[0] if self._ez_indices else 0
                    idx_h = self._h_indices[0] if self._h_indices else 0
                else:
                    idx_e = self._e_indices[0] if self._e_indices else 0
                    idx_h = self._hz_indices[0] if self._hz_indices else 0
                coord_e = (idx_e + 0.5) * dy
                coord_h = (idx_h + 1.0) * dy

        delta_s = float(coord_e - coord_h)
        if is_3d and dt is not None:
            omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
            d_axis = {"x": dx, "y": dy, "z": dz}[axis]
            k_num = _solve_numeric_k_axis(omega, dt, d_axis, self._neff)
            self._dt_physical = _numeric_phase_delay(omega, k_num, delta_s)
        else:
            self._dt_physical = delta_s * float(np.real(self._neff)) / LIGHT_SPEED

        if self._direction_sign < 0.0:
            self._dt_physical = -self._dt_physical

    def _get_signal_value(self, time, dt):
        """Interpolate signal value at arbitrary time."""
        idx_float = float(time / dt)
        idx_low = int(np.floor(idx_float))
        idx_high = idx_low + 1
        frac = idx_float - idx_low

        if 0 <= idx_low < len(self.signal) - 1:
            return (1.0 - frac) * self.signal[idx_low] + frac * self.signal[idx_high]
        elif idx_low == len(self.signal) - 1:
            return self.signal[idx_low]
        else:
            return 0.0

    def inject_h(self, fields, t, dt, current_step, resolution, design):
        """Inject magnetic current (M) into H-fields after the H update."""
        needs_reinit = (
            (not self._initialized)
            or (self._grid_shape != fields.permittivity.shape)
            or (self._resolution is None)
            or (not np.isclose(self._resolution, resolution))
        )
        if (
            (not needs_reinit)
            and getattr(self, "_is_3d", False)
            and ((self._launch_dt is None) or (not np.isclose(self._launch_dt, dt)))
        ):
            needs_reinit = True
        if needs_reinit:
            self.initialize(fields.permittivity, resolution, dt=dt)

        # M=-n×E is injected on the H update at the standard half-step time.
        signal_value_h = self._get_signal_value(t + 0.5 * dt, dt)

        if self._Ex_profile is not None and self._is_3d:
            self._inject_3d_h(fields, signal_value_h, dt, resolution)
        else:
            self._inject_2d_h(fields, signal_value_h, dt, resolution)

    def inject_e(self, fields, t, dt, current_step, resolution, design):
        """Inject electric current (J) into E-fields after the E update."""
        needs_reinit = (
            (not self._initialized)
            or (self._grid_shape != fields.permittivity.shape)
            or (self._resolution is None)
            or (not np.isclose(self._resolution, resolution))
        )
        if (
            (not needs_reinit)
            and getattr(self, "_is_3d", False)
            and ((self._launch_dt is None) or (not np.isclose(self._launch_dt, dt)))
        ):
            needs_reinit = True
        if needs_reinit:
            self.initialize(fields.permittivity, resolution, dt=dt)

        # J=n×H is evaluated on the E update and needs the physical E/H plane offset.
        # Keep E/H drive samples on the same temporal convention and only apply
        # the physical E/H plane offset correction via _dt_physical.
        signal_time_e = t + 0.5 * dt + self._dt_physical
        signal_value_e = self._get_signal_value(signal_time_e, dt)

        if self._Ex_profile is not None and self._is_3d:
            self._inject_3d_e(fields, signal_value_e, dt, resolution)
        else:
            self._inject_2d_e(fields, signal_value_e, dt, resolution)

    def inject(self, fields, t, dt, current_step, resolution, design):
        """Inject source fields (calls inject_h + inject_e for backward compatibility)."""
        self.inject_h(fields, t, dt, current_step, resolution, design)
        self.inject_e(fields, t, dt, current_step, resolution, design)

    # -- 3D injection (split) ------------------------------------------

    def _get_3d_profiles_and_indices(self):
        profiles = {
            "Ex": self._Ex_profile,
            "Ey": self._Ey_profile,
            "Ez": self._Ez_profile,
            "Hx": self._Hx_profile,
            "Hy": self._Hy_profile,
            "Hz": self._Hz_profile,
        }
        profiles = _runtime_3d_profiles(profiles, self._axis, self._direction_sign)
        indices = {
            "Ex": self._Ex_indices,
            "Ey": self._Ey_indices,
            "Ez": self._Ez_indices,
            "Hx": self._Hx_indices,
            "Hy": self._Hy_indices,
            "Hz": self._Hz_indices,
        }
        return profiles, indices

    def _inject_3d_h(self, fields, signal_h, dt, resolution):
        """Inject H-field components for 3D Huygens source."""
        profiles, indices = self._get_3d_profiles_and_indices()
        _inject_3d_h_fields(
            fields, profiles, indices, signal_h, dt, resolution, self._axis, self.pol
        )

    def _inject_3d_e(self, fields, signal_e, dt, resolution):
        """Inject E-field components for 3D Huygens source."""
        profiles, indices = self._get_3d_profiles_and_indices()
        _inject_3d_e_fields(
            fields, profiles, indices, signal_e, dt, resolution, self._axis, self.pol
        )

    # -- 2D injection (split, with corrected signs) ---------------------

    def _inject_2d_h(self, fields, signal_h, dt, resolution):
        """Inject magnetic current into H-fields for 2D (after H update)."""
        if self.pol == "tm":
            if self._h_indices is not None and self._my_profile is not None:
                mu_val = getattr(fields, "permeability", None)
                mu_at_source = mu_val[self._h_indices] if mu_val is not None else 1.0
                my_term = self._my_profile * signal_h / resolution
                h_injection = -my_term * dt / (MU_0 * mu_at_source)

                if self._h_component == "Hx":
                    fields.Hx = fields.Hx.at[self._h_indices].add(h_injection)
                else:
                    fields.Hy = fields.Hy.at[self._h_indices].add(h_injection)
        else:  # TE
            if self._hz_indices is not None and self._mz_profile is not None:
                mu_val = getattr(fields, "permeability", None)
                mu_at_source = mu_val[self._hz_indices] if mu_val is not None else 1.0
                mz_term = self._mz_profile * signal_h / resolution
                hz_injection = +mz_term * dt / (MU_0 * mu_at_source)
                fields.Hz = fields.Hz.at[self._hz_indices].add(hz_injection)

    def _inject_2d_e(self, fields, signal_e, dt, resolution):
        """Inject electric current into E-fields for 2D (after E update)."""
        if self.pol == "tm":
            if self._ez_indices is not None and self._jz_profile is not None:
                eps_at_source = fields.permittivity[self._ez_indices]
                jz_term = self._jz_profile * signal_e / resolution
                ez_injection = +jz_term * dt / (EPS_0 * eps_at_source)
                fields.Ez = fields.Ez.at[self._ez_indices].add(ez_injection)
        else:  # TE
            if self._e_indices is not None:
                j_profile = (
                    self._jx_profile if self._e_component == "Ex" else self._jy_profile
                )
                if j_profile is not None:
                    eps_at_source = fields.permittivity[self._e_indices]
                    j_term = j_profile * signal_e / resolution
                    e_injection = -j_term * dt / (EPS_0 * eps_at_source)

                    if self._e_component == "Ex":
                        fields.Ex = fields.Ex.at[self._e_indices].add(e_injection)
                    else:
                        fields.Ey = fields.Ey.at[self._e_indices].add(e_injection)

    def mode_profile_data(self, field=None):
        """Return mode-profile arrays and metadata for manual plotting."""
        from beamz.visual.data import mode_profile_data

        return mode_profile_data(self, field=field)

    def to_plot_data(
        self, *, facecolor="none", edgecolor="crimson", alpha=0.8, linestyle="-"
    ):
        """Return a renderer-agnostic source payload."""
        from beamz.visual.data import mode_source_plot_data

        return mode_source_plot_data(
            self,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linestyle=linestyle,
        )
