"""Shared direction, impedance, and mode-selection helpers for mode sources."""

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0

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
