import logging
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from beamz._yee import component_axis_offsets_3d
from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices._placement import snap_centered_extent, snap_mode_source_region
from beamz.devices._runtime import RuntimeStateProxy
from beamz.devices.sources._materials import (
    component_permeability_at,
    component_permittivity_at,
)
from beamz.devices.sources.solve import solve_beamz_mode_plane, solve_modes

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
    _k_num_axis: float | None = None
    _omega_launch: float | None = None
    _phase_ref_coord: float = 0.0
    _phase_plane_coord: float = 0.0
    _discrete_launch_max_shift: int = 2
    _launch_dt: float | None = None
    _snapped_region: object | None = None
    _discrete_mode: object | None = None
    _profiles_are_runtime_oriented: bool = False
    _initialized: bool = False


@dataclass(frozen=True)
class _ModeSource3DResidual:
    """Compact local 3D source residual emitted by ModeSource compilation."""

    component: str
    timing: str
    index: tuple[slice, slice, slice]
    residual: np.ndarray


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _interpolate_time_signal(signal, time, dt):
    """Linearly interpolate a real-valued source signal at an arbitrary time."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size <= 0:
        return 0.0

    idx_float = float(time / dt)
    idx_low = int(np.floor(idx_float))
    idx_high = idx_low + 1
    frac = idx_float - idx_low

    if 0 <= idx_low < arr.size - 1:
        return float((1.0 - frac) * arr[idx_low] + frac * arr[idx_high])
    if idx_low == arr.size - 1:
        return float(arr[idx_low])
    return 0.0


def _analytic_signal_quadrature(signal):
    """Return the Hilbert-transform quadrature of a real source waveform."""
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.zeros_like(arr, dtype=np.float64)

    spectrum = np.fft.fft(arr)
    weights = np.zeros((n,), dtype=np.float64)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[n // 2] = 1.0
        weights[1 : n // 2] = 2.0
    else:
        weights[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * weights)
    return np.asarray(np.imag(analytic), dtype=np.float64)


def _real_phasor_sample(profile, in_phase, quadrature):
    """Evaluate Re(profile * analytic_signal) for real-valued FDTD fields."""
    arr = np.asarray(profile, dtype=np.complex128)
    return np.real(arr) * float(in_phase) - np.imag(arr) * float(quadrature)


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
    interval = snap_centered_extent(center_val, extent, resolution, grid_max)
    return int(interval.start), int(interval.stop)


def _compute_padded_3d_transverse_bounds(
    center_val, extent, resolution, grid_max, pad_cells=2
):
    """Return 3D source bounds with extra cladding cells to reduce modal truncation.

    A tight 3D launch aperture can distort the weaker transverse field pair in
    rectangular guides and launch a stable but non-modal mixture. Pad the
    requested extent symmetrically by a small number of cells and clip to the
    available transverse grid.
    """
    padded_extent = float(extent) + 2.0 * float(max(0, int(pad_cells))) * float(
        resolution
    )
    return _compute_transverse_bounds(center_val, padded_extent, resolution, grid_max)


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


def _shift_component_indices_along_axis(indices, axis, shift, field_shape):
    """Shift a component support tuple by integer cells along the propagation axis."""
    if indices is None:
        return None
    axis_pos = {"x": 2, "y": 1, "z": 0}[axis]
    out = list(indices)
    plane_idx = out[axis_pos]
    if isinstance(plane_idx, slice):
        return None
    plane_new = int(plane_idx) + int(shift)
    if plane_new < 0 or plane_new >= int(field_shape[axis_pos]):
        return None
    out[axis_pos] = plane_new
    return tuple(out)


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
    """Return solve_modes 3D components in the global Cartesian basis.

    solve_modes(...) already emits 3D fields in global (Ex, Ey, Ez, Hx, Hy, Hz)
    order for the requested propagation axis. A second axis remap here corrupts
    y/z-directed launches and monitor projections.
    """
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported axis {axis!r} for 3D remap")
    return Ex, Ey, Ez, Hx, Hy, Hz


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


def _mirror_correlation(arr, axis):
    arr = np.asarray(arr, dtype=np.complex128)
    if arr.size == 0:
        return 0.0
    flipped = np.flip(arr, axis=axis)
    denom = float(np.sum(np.abs(arr) ** 2))
    if denom <= 1e-18:
        return 0.0
    return float(np.real(np.sum(arr * np.conjugate(flipped))) / denom)


def _detect_transverse_symmetry_axes(eps_profile, threshold=0.995):
    eps_arr = np.asarray(np.real(eps_profile), dtype=float)
    if eps_arr.ndim < 2:
        return ()
    symmetric_axes = []
    for axis in range(eps_arr.ndim):
        corr = _mirror_correlation(eps_arr, axis)
        if corr >= float(threshold):
            symmetric_axes.append(int(axis))
    return tuple(symmetric_axes)


def _enforce_componentwise_parity(component_map, symmetric_axes):
    if not symmetric_axes:
        return {
            name: np.asarray(value, dtype=np.complex128)
            for name, value in component_map.items()
        }

    out = {}
    for name, value in component_map.items():
        arr = np.asarray(value, dtype=np.complex128)
        for axis in symmetric_axes:
            flipped = np.flip(arr, axis=axis)
            overlap = float(np.real(np.sum(arr * np.conjugate(flipped))))
            parity = 1.0 if overlap >= 0.0 else -1.0
            arr = 0.5 * (arr + parity * flipped)
        out[name] = arr
    return out


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
    """Build staggered, windowed injection profiles for 3D.

    Returns
    -------
    profiles : dict
        Mapping component name -> numpy real profile array.
    indices : dict
        Mapping component name -> index tuple for field injection.
    """
    _ = (impedance_neff, omega, dt)
    nz, ny, nx = grid_shape
    dir_sign = 1.0 if direction.startswith("+") else -1.0

    if axis == "x":
        return _build_3d_x(
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            dir_sign,
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
    y_start, y_end = _compute_padded_3d_transverse_bounds(
        center[1], width, resolution, ny
    )
    z_center = center[2] if len(center) > 2 else (nz // 2) * resolution
    z_start, z_end = _compute_padded_3d_transverse_bounds(
        z_center, height, resolution, nz
    )

    ex_z, ex_y = _component_support_slices_3d(
        "Ex", "x", z_start, z_end, y_start, y_end, Ex_s.shape
    )
    ey_z, ey_y = _component_support_slices_3d(
        "Ey", "x", z_start, z_end, y_start, y_end, Ey_s.shape
    )
    ez_z, ez_y = _component_support_slices_3d(
        "Ez", "x", z_start, z_end, y_start, y_end, Ez_s.shape
    )
    hx_z, hx_y = _component_support_slices_3d(
        "Hx", "x", z_start, z_end, y_start, y_end, Hx_s.shape
    )
    hy_z, hy_y = _component_support_slices_3d(
        "Hy", "x", z_start, z_end, y_start, y_end, Hy_s.shape
    )
    hz_z, hz_y = _component_support_slices_3d(
        "Hz", "x", z_start, z_end, y_start, y_end, Hz_s.shape
    )

    # --- indices  (z_slice, y_slice, x_index) ---
    indices = {
        "Ex": (
            ex_z,
            ex_y,
            offset_idx,
        ),
        "Ey": (
            ey_z,
            ey_y,
            center_idx,
        ),
        "Ez": (
            ez_z,
            ez_y,
            center_idx,
        ),
        "Hx": (
            hx_z,
            hx_y,
            center_idx,
        ),
        "Hy": (
            hy_z,
            hy_y,
            offset_idx,
        ),
        "Hz": (
            hz_z,
            hz_y,
            offset_idx,
        ),
    }

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
        axis="x",
    )
    d_area = float(resolution * resolution)
    profiles = _normalize_3d_profiles_by_flux(
        profiles,
        axis="x",
        d_area=d_area,
        direction_sign=dir_sign,
    )

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
    x_start, x_end = _compute_padded_3d_transverse_bounds(
        center[0], width, resolution, nx
    )
    z_center = center[2] if len(center) > 2 else (nz // 2) * resolution
    z_start, z_end = _compute_padded_3d_transverse_bounds(
        z_center, height, resolution, nz
    )

    ex_z, ex_x = _component_support_slices_3d(
        "Ex", "y", z_start, z_end, x_start, x_end, Ex_s.shape
    )
    ey_z, ey_x = _component_support_slices_3d(
        "Ey", "y", z_start, z_end, x_start, x_end, Ey_s.shape
    )
    ez_z, ez_x = _component_support_slices_3d(
        "Ez", "y", z_start, z_end, x_start, x_end, Ez_s.shape
    )
    hx_z, hx_x = _component_support_slices_3d(
        "Hx", "y", z_start, z_end, x_start, x_end, Hx_s.shape
    )
    hy_z, hy_x = _component_support_slices_3d(
        "Hy", "y", z_start, z_end, x_start, x_end, Hy_s.shape
    )
    hz_z, hz_x = _component_support_slices_3d(
        "Hz", "y", z_start, z_end, x_start, x_end, Hz_s.shape
    )

    # --- indices  (z_slice, y_index, x_slice) ---
    indices = {
        "Ex": (
            ex_z,
            center_idx,
            ex_x,
        ),
        "Ey": (
            ey_z,
            offset_idx,
            ey_x,
        ),
        "Ez": (
            ez_z,
            center_idx,
            ez_x,
        ),
        "Hx": (
            hx_z,
            offset_idx,
            hx_x,
        ),
        "Hy": (
            hy_z,
            center_idx,
            hy_x,
        ),
        "Hz": (
            hz_z,
            offset_idx,
            hz_x,
        ),
    }

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
        axis="y",
    )
    d_area = float(resolution * resolution)
    profiles = _normalize_3d_profiles_by_flux(
        profiles,
        axis="y",
        d_area=d_area,
        direction_sign=dir_sign,
    )
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
    x_start, x_end = _compute_padded_3d_transverse_bounds(
        center[0], width, resolution, nx
    )
    y_start, y_end = _compute_padded_3d_transverse_bounds(
        center[1], height, resolution, ny
    )

    e_z_idx = int(np.clip(center_idx, 0, nz - 1))
    h_z_idx = int(np.clip(offset_idx, 0, max(nz - 2, 0)))
    ez_z_idx = int(np.clip(center_idx, 0, max(nz - 2, 0)))
    hz_z_idx = int(np.clip(offset_idx, 0, nz - 1))

    ex_y, ex_x = _component_support_slices_3d(
        "Ex", "z", y_start, y_end, x_start, x_end, Ex_s.shape
    )
    ey_y, ey_x = _component_support_slices_3d(
        "Ey", "z", y_start, y_end, x_start, x_end, Ey_s.shape
    )
    ez_y, ez_x = _component_support_slices_3d(
        "Ez", "z", y_start, y_end, x_start, x_end, Ez_s.shape
    )
    hx_y, hx_x = _component_support_slices_3d(
        "Hx", "z", y_start, y_end, x_start, x_end, Hx_s.shape
    )
    hy_y, hy_x = _component_support_slices_3d(
        "Hy", "z", y_start, y_end, x_start, x_end, Hy_s.shape
    )
    hz_y, hz_x = _component_support_slices_3d(
        "Hz", "z", y_start, y_end, x_start, x_end, Hz_s.shape
    )

    # --- indices  (z_index, y_slice, x_slice) ---
    indices = {
        "Ex": (
            e_z_idx,
            ex_y,
            ex_x,
        ),
        "Ey": (
            e_z_idx,
            ey_y,
            ey_x,
        ),
        "Ez": (
            ez_z_idx,
            ez_y,
            ez_x,
        ),
        "Hx": (
            h_z_idx,
            hx_y,
            hx_x,
        ),
        "Hy": (
            h_z_idx,
            hy_y,
            hy_x,
        ),
        "Hz": (
            hz_z_idx,
            hz_y,
            hz_x,
        ),
    }

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
        axis="z",
    )
    d_area = float(resolution * resolution)
    profiles = _normalize_3d_profiles_by_flux(
        profiles,
        axis="z",
        d_area=d_area,
        direction_sign=dir_sign,
    )

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
    axis=None,
):
    """Crop all six staggered profiles and multiply by 2D Tukey windows."""
    row_axis, col_axis = _profile_transverse_axes_3d(axis) if axis else (None, None)

    profiles = {}
    for name, field in staggered.items():
        ze, te_end = z_end, t_end
        if row_axis is not None and col_axis is not None:
            ze, te_end = _component_support_stops_3d(
                name,
                row_axis=row_axis,
                col_axis=col_axis,
                row_start=z_start,
                row_stop=z_end,
                col_start=t_start,
                col_stop=t_end,
            )
        fe = min(ze, field.shape[0])
        te = min(te_end, field.shape[1])
        h_cells = max(0, fe - z_start)
        w_cells = max(0, te - t_start)
        window = _make_tukey_window_2d(h_cells, w_cells, alpha=alpha, use_jax=use_jax)
        profiles[name] = dir_sign * _crop_and_window_2d(
            field, z_start, fe, t_start, te, window
        )
    return profiles


def _profile_transverse_axes_3d(axis: str) -> tuple[str, str]:
    """Return physical axes for row/column dimensions of a 3D source profile."""
    mapping = {
        "x": ("z", "y"),
        "y": ("z", "x"),
        "z": ("y", "x"),
    }
    try:
        return mapping[str(axis)]
    except KeyError as exc:
        raise ValueError(f"Unsupported 3D source axis {axis!r}") from exc


def _support_stop_for_offset(start: int, stop: int, offset: float) -> int:
    start = int(start)
    stop = int(stop)
    if float(offset) == 0.5 and (stop - start) > 1:
        return stop - 1
    return stop


def _component_support_stops_3d(
    component: str,
    *,
    row_axis: str,
    col_axis: str,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> tuple[int, int]:
    offsets = component_axis_offsets_3d(component)
    return (
        _support_stop_for_offset(row_start, row_stop, offsets[row_axis]),
        _support_stop_for_offset(col_start, col_stop, offsets[col_axis]),
    )


def _component_support_slices_3d(
    component: str,
    axis: str,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    field_shape: tuple[int, int],
) -> tuple[slice, slice]:
    row_axis, col_axis = _profile_transverse_axes_3d(axis)
    row_stop, col_stop = _component_support_stops_3d(
        component,
        row_axis=row_axis,
        col_axis=col_axis,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
    )
    return (
        slice(row_start, min(row_stop, int(field_shape[0]))),
        slice(col_start, min(col_stop, int(field_shape[1]))),
    )


def _modal_power_3d_from_profiles(profiles, axis, d_area, direction_sign=1.0):
    """Compute 3D modal power from profiles on a cross-section."""
    if axis == "x":
        required = ("Ey", "Ez", "Hy", "Hz")
    elif axis == "y":
        required = ("Ex", "Ez", "Hx", "Hz")
    elif axis == "z":
        required = ("Ex", "Ey", "Hx", "Hy")
    else:
        return 0.0
    if any(profiles.get(name) is None for name in required):
        return 0.0

    def _profile(name):
        value = profiles.get(name)
        if value is None:
            return np.zeros_like(np.asarray(profiles[required[0]], dtype=np.complex128))
        return value

    ex = _profile("Ex")
    ey = _profile("Ey")
    ez = _profile("Ez")
    hx = _profile("Hx")
    hy = _profile("Hy")
    hz = _profile("Hz")

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
    return float(0.5 * float(direction_sign) * np.real(np.sum(s_axis) * float(d_area)))


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


def _phase_reference_3d_profiles(
    profiles,
    indices,
    *,
    axis,
    dx,
    dy,
    dz,
    omega,
    k_num,
    ref_coord,
):
    """Return modal profiles phase-shifted to one propagation-axis reference plane."""
    out = {}
    for name, value in profiles.items():
        arr = np.asarray(value, dtype=np.complex128)
        idx = None if indices is None else indices.get(name)
        axis_idx = _axis_index_from_component_indices(idx, axis)
        coord = _component_axis_coord(name, axis_idx, axis, dx, dy, dz)
        delay = _numeric_phase_delay(float(omega), float(k_num), coord - ref_coord)
        out[name] = arr * np.exp(-1j * float(omega) * delay)
    return out


def _normalize_3d_profiles_by_phase_referenced_flux(
    profiles,
    indices,
    *,
    axis,
    d_area,
    direction_sign,
    dx,
    dy,
    dz,
    omega,
    k_num,
    ref_coord,
    eps=1e-18,
):
    """Normalize source profiles using the Yee-stagger phase-referenced flux."""
    referenced = _phase_reference_3d_profiles(
        profiles,
        indices,
        axis=axis,
        dx=dx,
        dy=dy,
        dz=dz,
        omega=omega,
        k_num=k_num,
        ref_coord=ref_coord,
    )
    flux = _modal_power_3d_from_profiles(
        referenced,
        axis=axis,
        d_area=d_area,
        direction_sign=direction_sign,
    )
    if (not np.isfinite(flux)) or abs(flux) <= eps:
        return profiles, 1.0

    scale = float(np.sqrt(1.0 / max(abs(flux), eps)))
    scale = float(np.clip(scale, 1e-6, 1e6))
    for key, value in profiles.items():
        profiles[key] = np.asarray(value) * scale
    return profiles, scale


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


def _backward_3d_mode_from_forward(profiles):
    """Return the backward-going counterpart of a forward 3D modal field set."""
    out = {}
    for key, value in profiles.items():
        arr = np.asarray(value, dtype=np.complex128)
        out[key] = -arr if key.startswith("H") else arr.copy()
    return out


def _make_3d_mode_basis_profiles(profiles, axis, d_area=1.0, direction_sign=1.0):
    """Build unit-flux forward/backward 3D basis fields from one solved mode."""
    forward = {
        key: np.asarray(value, dtype=np.complex128) for key, value in profiles.items()
    }
    forward = _normalize_3d_profiles_by_flux(
        forward,
        axis=axis,
        d_area=d_area,
        direction_sign=direction_sign,
    )
    backward = _backward_3d_mode_from_forward(forward)
    return forward, backward


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
    eps = component_permittivity_at(fields, comp, idx)
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


class ModeSource(RuntimeStateProxy):
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
        "_k_num_axis",
        "_omega_launch",
        "_phase_ref_coord",
        "_phase_plane_coord",
        "_discrete_launch_max_shift",
        "_launch_power_scale",
        "_launch_dt",
        "_discrete_mode",
        "_profiles_are_runtime_oriented",
        "_initialized",
        "_resolution",
        "_is_3d",
        "_grid_shape",
        "_axis",
        "_eps_profile_2d",
        "_snapped_region",
        "_y_start",
        "_y_end",
        "_x_start",
        "_x_end",
    }

    def __setattr__(self, name, value):
        if self._set_runtime_attr(name, value):
            return
        if (
            name in {"signal", "signal_quadrature"}
            and "_signal_quadrature" in self.__dict__
        ):
            object.__setattr__(self, "_signal_quadrature", None)
            object.__setattr__(self, "_signal_quadrature_signature", None)
        object.__setattr__(self, name, value)

    def __init__(
        self,
        grid,
        center,
        width,
        wavelength,
        pol,
        signal,
        direction="+x",
        height=None,
        signal_quadrature=None,
        profile_frequencies=None,
        power=1.0,
        source_time=None,
        num_freqs=None,
        mode_neff=None,
        mode_e_field=None,
        mode_h_field=None,
        mode_eps_profile_full=None,
        mode_crop_slices=None,
        mode_index=0,
        mode_target_neff=None,
        mode_num_modes=None,
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
        self.signal_quadrature = signal_quadrature
        self.source_time = source_time
        self.profile_frequencies = profile_frequencies
        self.num_freqs = None if num_freqs is None else int(num_freqs)
        self.mode_neff = mode_neff
        self.mode_e_field = mode_e_field
        self.mode_h_field = mode_h_field
        self.mode_eps_profile_full = mode_eps_profile_full
        self.mode_crop_slices = mode_crop_slices
        self.mode_index = int(mode_index)
        self.mode_target_neff = mode_target_neff
        self.mode_num_modes = None if mode_num_modes is None else int(mode_num_modes)
        power_value = float(power)
        if not np.isfinite(power_value) or power_value < 0.0:
            raise ValueError(
                f"ModeSource power must be a non-negative finite value, got {power!r}."
            )
        self.power = power_value
        self._signal_quadrature = None
        self._signal_quadrature_signature = None
        self.direction, self._direction_axis, self._direction_sign = _parse_direction(
            direction
        )
        self._state = _ModeSourceState()

    def source_spectrum(self, freqs, *, normalize: bool = True) -> np.ndarray | None:
        """Return this source's analytic spectrum when source-time metadata exists."""
        freq_arr = np.asarray(freqs, dtype=float)
        source_time = getattr(self, "source_time", None)
        if source_time is None:
            return None
        if normalize and hasattr(source_time, "dft_normalization_spectrum"):
            spectrum = source_time.dft_normalization_spectrum(freq_arr)
            freq0 = float(
                getattr(source_time, "freq0", LIGHT_SPEED / float(self.wavelength))
            )
            if np.isfinite(freq0) and freq0 > 0.0:
                spectrum = np.asarray(spectrum, dtype=np.complex128) * np.sqrt(
                    np.maximum(freq_arr, 0.0) / freq0
                )
            return spectrum
        if not hasattr(source_time, "spectrum"):
            return None
        return source_time.spectrum(freq_arr, normalize=normalize)

    def copy(self, *, update=None):
        """Return a configuration copy of this mode source."""
        import copy

        copied = copy.deepcopy(self)
        if update:
            for key, value in dict(update).items():
                setattr(copied, key, value)
        return copied

    def calibrated_to_measured_power(self, measured_power, *, target_power=None):
        """Return a copy whose requested power compensates a reference measurement.

        ``measured_power`` is the source-normalized power measured from an otherwise
        identical straight reference run. Since modal source fields scale as
        ``sqrt(power)``, the measured power scales linearly with ``power``.
        """
        measured = float(measured_power)
        if not np.isfinite(measured) or measured <= 0.0:
            raise ValueError(
                "measured_power must be a positive finite value, "
                f"got {measured_power!r}."
            )
        target = self.power if target_power is None else float(target_power)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError(
                f"target_power must be a non-negative finite value, got {target_power!r}."
            )
        return self.copy(update={"power": self.power * target / measured})

    def shifted(self, offset):
        copied = self.copy()
        offset = tuple(float(v) for v in offset)
        copied.center = tuple(
            a + b for a, b in zip(copied.center, offset, strict=False)
        )
        return copied

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
        self._discrete_mode = None
        self._profiles_are_runtime_oriented = False
        self._snapped_region = snap_mode_source_region(
            center=tuple(float(v) for v in self.center),
            width=float(self.width),
            height=None if self.height is None else float(self.height),
            axis=axis,
            direction_sign=self._direction_sign,
            grid_shape=self._grid_shape,
            resolution=resolution,
            is_3d=is_3d,
        )

        # 1. Get center index for injection plane
        if axis == "x":
            center_idx = int(self._snapped_region.plane_index)
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
            center_idx = int(self._snapped_region.plane_index)
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
            center_idx = int(self._snapped_region.plane_index)
            if self.direction == "+z":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(nz - 2, center_idx + 1)

            eps_profile = permittivity[center_idx, :, :]
            self._eps_profile_2d = eps_profile

        # 2. Solve for mode fields
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        self._omega_launch = float(omega) if is_3d else None
        dL = dz if is_3d else (dy if axis == "x" else dx)
        solver_direction = self.direction
        if is_3d and axis == "y":
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

        precomputed_e = getattr(self, "mode_e_field", None)
        precomputed_h = getattr(self, "mode_h_field", None)
        if precomputed_e is not None and precomputed_h is not None:
            E_mode = np.asarray(precomputed_e, dtype=np.complex128)
            H_mode = np.asarray(precomputed_h, dtype=np.complex128)
            if E_mode.shape[0] != 3 or H_mode.shape[0] != 3:
                raise ValueError(
                    "ModeSource precomputed mode_e_field and mode_h_field must "
                    "have component axis length 3."
                )
            neff_val = np.asarray(
                [
                    (
                        self.mode_neff
                        if self.mode_neff is not None
                        else np.sqrt(max(np.real(np.max(eps_profile_arr)), 1e-12))
                    )
                ],
                dtype=np.complex128,
            )
            self._neff = neff_val[0]
        else:
            if is_3d:
                discrete_mode = self._setup_discrete_3d_mode_from_micromode(
                    eps_profile=eps_profile,
                    permittivity=permittivity,
                    center_idx=center_idx,
                    offset_idx=offset_idx,
                    axis=axis,
                    grid_shape=(nz, ny, nx),
                    resolution=resolution,
                    omega=omega,
                    dt=dt,
                    solver_direction=solver_direction,
                    target_neff=(
                        self.mode_target_neff
                        if self.mode_target_neff is not None
                        else target_neff
                    ),
                )
                if discrete_mode is not None:
                    self._compute_dt_physical(axis, is_3d, dx, dy, dz, dt=dt)
                    self._k_num_axis = float(discrete_mode.k_num_axis)
                    self._phase_ref_coord = float(discrete_mode.phase_reference_coord)
                    self._phase_plane_coord = float(discrete_mode.phase_plane_coord)
                    self._initialized = True
                    return

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
            self._phase_plane_coord = float(self._snapped_region.plane_coord)
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

    def get_snapped_region(self):
        """Return the canonical snapped source region after initialization."""
        return self._snapped_region

    def _snapped_support_bounds(self) -> dict[str, tuple[float, float]] | None:
        snapped = self._snapped_region
        if snapped is None:
            return None
        return {
            axis: snapped.axis_bounds(axis)
            for axis in ("x", "y", "z")[: int(snapped.ndim)]
        }

    def _injection_support_bounds(
        self,
        fields=None,
        *,
        dt: float | None = None,
    ) -> dict[str, tuple[float, float]] | None:
        """Return physical grid-coordinate bounds touched by source injection."""
        fallback = self._snapped_support_bounds()
        if (
            fields is None
            or dt is None
            or not bool(getattr(self, "_is_3d", False))
            or getattr(self, "_omega_launch", None) is None
            or getattr(self, "_k_num_axis", None) is None
        ):
            return fallback

        residuals = self._compute_discrete_3d_phasor_residuals(fields, dt=float(dt))
        if not residuals:
            return fallback

        grid_shape = tuple(int(v) for v in np.asarray(fields.permittivity).shape)
        lows = [int(v) for v in grid_shape]
        highs = [0, 0, 0]
        for residual in residuals:
            bounds = self._component_slices_to_cell_bbox(
                residual.component,
                residual.index,
            )
            for axis_idx, (lo, hi) in enumerate(bounds):
                lows[axis_idx] = min(lows[axis_idx], int(lo))
                highs[axis_idx] = max(highs[axis_idx], int(hi))

        resolution = float(self._resolution or 0.0)
        z_bounds, y_bounds, x_bounds = (
            (lows[axis_idx] * resolution, highs[axis_idx] * resolution)
            for axis_idx in range(3)
        )
        return {"x": x_bounds, "y": y_bounds, "z": z_bounds}

    def _setup_discrete_3d_mode_from_micromode(
        self,
        *,
        eps_profile,
        permittivity,
        center_idx,
        offset_idx,
        axis,
        grid_shape,
        resolution,
        omega,
        dt,
        solver_direction,
        target_neff,
    ):
        """Request a BEAMZ-shaped DiscreteMode from micromode, if available."""
        from beamz.simulation.yee import (
            component_shape_3d,
            sample_voxel_grid_at_component_3d,
            sample_voxel_grid_at_e_component_3d_centered,
        )

        nz, ny, nx = grid_shape
        component_shapes = {
            component: component_shape_3d(component, grid_shape)
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
        permittivity_arr = np.asarray(permittivity)
        permeability_arr = np.ones_like(permittivity_arr, dtype=np.float64)
        component_permittivity = {
            component: np.asarray(
                sample_voxel_grid_at_e_component_3d_centered(
                    permittivity_arr,
                    component,
                    stored_shape=component_shapes[component],
                )
            )
            for component in ("Ex", "Ey", "Ez")
        }
        component_permeability = {
            component: np.asarray(
                sample_voxel_grid_at_component_3d(
                    permeability_arr,
                    component,
                    stored_shape=component_shapes[component],
                )
            )
            for component in ("Hx", "Hy", "Hz")
        }
        transverse_axes = {
            "x": ("z", "y"),
            "y": ("z", "x"),
            "z": ("y", "x"),
        }[axis]
        center = tuple(float(value) for value in self.center)
        if len(center) < 3:
            center = (
                center[0] if len(center) > 0 else 0.5 * nx * float(resolution),
                center[1] if len(center) > 1 else 0.5 * ny * float(resolution),
                0.5 * nz * float(resolution),
            )

        discrete_mode = solve_beamz_mode_plane(
            scalar_permittivity=np.asarray(eps_profile),
            frequency=float(omega) / (2.0 * np.pi),
            resolution=float(resolution),
            dt=None if dt is None else float(dt),
            axis=axis,
            direction=self.direction,
            solver_direction=solver_direction,
            transverse_axes=transverse_axes,
            grid_shape=grid_shape,
            component_shapes=component_shapes,
            component_permittivity=component_permittivity,
            component_permeability=component_permeability,
            center=center,
            width=float(self.width),
            height=float(self.height if self.height is not None else self.width),
            plane_index=int(center_idx),
            offset_index=int(offset_idx),
            mode_index=int(self.mode_index),
            polarization=self.pol,
            target_neff=target_neff,
            num_modes=(
                int(self.mode_num_modes)
                if self.mode_num_modes is not None
                else max(int(self.mode_index) + 1, 3)
            ),
        )
        if discrete_mode is None:
            return None

        profiles = {
            name: np.asarray(value, dtype=np.complex128)
            for name, value in discrete_mode.profiles.items()
        }
        profiles = _scale_profiles_for_power(profiles, self.power)
        indices = dict(discrete_mode.component_indices)
        self._Ex_profile = profiles.get("Ex")
        self._Ey_profile = profiles.get("Ey")
        self._Ez_profile = profiles.get("Ez")
        self._Hx_profile = profiles.get("Hx")
        self._Hy_profile = profiles.get("Hy")
        self._Hz_profile = profiles.get("Hz")
        self._Ex_indices = indices.get("Ex")
        self._Ey_indices = indices.get("Ey")
        self._Ez_indices = indices.get("Ez")
        self._Hx_indices = indices.get("Hx")
        self._Hy_indices = indices.get("Hy")
        self._Hz_indices = indices.get("Hz")
        self._neff = np.complex128(discrete_mode.neff)
        self._impedance_neff = None
        self._launch_power_scale = float(discrete_mode.power_scale)
        self._phase_plane_coord = float(discrete_mode.phase_plane_coord)
        self._phase_ref_coord = float(discrete_mode.phase_reference_coord)
        self._k_num_axis = float(discrete_mode.k_num_axis)
        self._h_component = _dominant_3d_pair(axis, self.pol)[1]
        self._e_component = _dominant_3d_pair(axis, self.pol)[0]
        self._discrete_launch_max_shift = max(
            int(getattr(self, "_discrete_launch_max_shift", 2)),
            12,
        )
        self._profiles_are_runtime_oriented = True
        self._discrete_mode = discrete_mode

        self._jz_profile = self._Hz_profile
        self._my_profile = self._Ez_profile
        return discrete_mode

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
        symmetric_axes = _detect_transverse_symmetry_axes(self._eps_profile_2d)
        if symmetric_axes:
            profiles = _enforce_componentwise_parity(profiles, symmetric_axes)
        d_area = float(resolution * resolution)
        dx = dy = dz = float(resolution)
        ref_coord = _component_axis_coord(
            _dominant_3d_pair(axis, self.pol)[1],
            _axis_index_from_component_indices(
                indices.get(_dominant_3d_pair(axis, self.pol)[1]),
                axis,
            ),
            axis,
            dx,
            dy,
            dz,
        )
        k_num = (
            _solve_numeric_k_axis(omega, dt, resolution, self._neff)
            if dt is not None
            else float(np.real(self._neff)) * float(omega) / LIGHT_SPEED
        )
        profiles, launch_power_scale = _normalize_3d_profiles_by_phase_referenced_flux(
            profiles,
            indices,
            axis=axis,
            d_area=d_area,
            direction_sign=self._direction_sign,
            dx=dx,
            dy=dy,
            dz=dz,
            omega=omega,
            k_num=k_num,
            ref_coord=float(ref_coord),
        )
        self._launch_power_scale = float(launch_power_scale)
        profiles = _scale_profiles_for_power(profiles, self.power)
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
        snapped_y = (
            self._snapped_region.axis_interval("y")
            if self._snapped_region is not None
            else None
        )
        if snapped_y is None:
            center_y_idx = int(round(self.center[1] / resolution))
            half_width_idx = int(round((self.width / 2) / resolution))
            y_start = max(0, center_y_idx - half_width_idx)
            y_end = min(ny, center_y_idx + half_width_idx)
        else:
            y_start = int(snapped_y.start)
            y_end = int(snapped_y.stop)
        y_slice = slice(y_start, y_end)
        self._y_start = y_start
        self._y_end = y_end

        if self.pol == "tm":
            self._ez_indices = (y_slice, center_idx)
            self._h_indices = (y_slice, offset_idx)
            self._h_component = "Hy"

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

            # For x-directed TMz launches the native full-Yee source pair must
            # use opposite J/M handedness so the injected Ez/Hy pair carries
            # power along the requested x direction.
            jz_profile = -dir_sign * Hy_cropped
            my_profile = dir_sign * Ez_cropped
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=-1.0, dl=resolution
            )
            jz_profile = _to_real_profile(jz_profile)
            my_profile = _to_real_profile(my_profile)
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=-1.0, dl=resolution
            )
            jz_profile, my_profile = _scale_pair_for_power(
                jz_profile, my_profile, self.power
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
            jy_profile, mz_profile = _scale_pair_for_power(
                jy_profile, mz_profile, self.power
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
        snapped_x = (
            self._snapped_region.axis_interval("x")
            if self._snapped_region is not None
            else None
        )
        if snapped_x is None:
            center_x_idx = int(round(self.center[0] / resolution))
            half_width_idx = int(round((self.width / 2) / resolution))
            x_start = max(0, center_x_idx - half_width_idx)
            x_end = min(nx, center_x_idx + half_width_idx)
        else:
            x_start = int(snapped_x.start)
            x_end = int(snapped_x.stop)
        x_slice = slice(x_start, x_end)
        self._x_start = x_start
        self._x_end = x_end

        if self.pol == "tm":
            self._ez_indices = (center_idx, x_slice)
            self._h_indices = (offset_idx, x_slice)
            self._h_component = "Hx"

            # +y TM uses the global component pair (Ez, Hx).
            Hx_raw = np.squeeze(H_mode[0])
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

            # Match the rotated x-directed TMz launch on the native full-Yee
            # lattice.  The mode solver's Hx gauge has the opposite sign from
            # the Hy gauge used by x-propagation, so Jz needs the same leading
            # minus sign as the x-directed branch.
            jz_profile = -dir_sign * Hx_cropped
            my_profile = -dir_sign * Ez_cropped
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=1.0, dl=resolution
            )
            jz_profile = _to_real_profile(jz_profile)
            my_profile = _to_real_profile(my_profile)
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=1.0, dl=resolution
            )
            jz_profile, my_profile = _scale_pair_for_power(
                jz_profile, my_profile, self.power
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

            # +y TE uses the global component pair (Ex, Hz).
            Hz_raw = np.squeeze(H_mode[2])
            Ex_raw = np.squeeze(E_mode[0])

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

            jx_profile = dir_sign * Hz_cropped
            mz_profile = -dir_sign * Ex_cropped
            jx_profile, mz_profile = _normalize_2d_pair_by_power(
                jx_profile, mz_profile, signed_flux_sign=-1.0, dl=resolution
            )
            jx_profile = _to_real_profile(jx_profile)
            mz_profile = _to_real_profile(mz_profile)
            jx_profile, mz_profile = _normalize_2d_pair_by_power(
                jx_profile, mz_profile, signed_flux_sign=-1.0, dl=resolution
            )
            jx_profile, mz_profile = _scale_pair_for_power(
                jx_profile, mz_profile, self.power
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
            self._k_num_axis = float(k_num)
            self._dt_physical = _numeric_phase_delay(omega, k_num, delta_s)
        else:
            self._k_num_axis = None
            self._dt_physical = delta_s * float(np.real(self._neff)) / LIGHT_SPEED

        self._phase_ref_coord = float(coord_h)

        if self._direction_sign < 0.0:
            self._dt_physical = -self._dt_physical

    def _get_signal_value(self, time, dt):
        """Interpolate signal value at arbitrary time."""
        return _interpolate_time_signal(self.signal, time, dt)

    def _get_signal_quadrature(self):
        explicit = getattr(self, "signal_quadrature", None)
        if explicit is not None:
            signal = np.asarray(self.signal, dtype=np.float64).reshape(-1)
            quadrature = np.asarray(explicit, dtype=np.float64).reshape(-1)
            if quadrature.shape != signal.shape:
                raise ValueError(
                    "signal_quadrature must have the same shape as signal; "
                    f"got {quadrature.shape} and {signal.shape}"
                )
            return quadrature

        signal = np.asarray(self.signal, dtype=np.float64).reshape(-1)
        signature = (id(self.signal), signal.shape, str(signal.dtype))
        if (
            self._signal_quadrature is None
            or self._signal_quadrature_signature != signature
        ):
            self._signal_quadrature = _analytic_signal_quadrature(signal)
            self._signal_quadrature_signature = signature
        return self._signal_quadrature

    def _get_signal_quadrature_value(self, time, dt):
        """Interpolate the quadrature drive used for complex modal phasors."""
        return _interpolate_time_signal(self._get_signal_quadrature(), time, dt)

    def _get_analytic_signal_value(self, time, dt):
        """Sample the analytic source waveform at an arbitrary time."""
        return complex(
            self._get_signal_value(time, dt),
            self._get_signal_quadrature_value(time, dt),
        )

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

        if self._Ex_profile is not None and self._is_3d:
            self._inject_3d_h(fields, t, dt, resolution)
        else:
            # M=-n×E is injected on the H update at the standard half-step time.
            signal_value_h = self._get_signal_value(t + 0.5 * dt, dt)
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

        if self._Ex_profile is not None and self._is_3d:
            self._inject_3d_e(fields, t, dt, resolution)
        else:
            # J=n×H is evaluated on the E update and needs the physical E/H plane offset.
            # Keep E/H drive samples on the same temporal convention and only apply
            # the physical E/H plane offset correction via _dt_physical.
            signal_time_e = t + 0.5 * dt + self._dt_physical
            signal_value_e = self._get_signal_value(signal_time_e, dt)
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
        if not getattr(self, "_profiles_are_runtime_oriented", False):
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

    def _build_incident_3d_state(
        self,
        fields,
        *,
        t_e,
        t_h,
        dt,
        masked,
    ):
        """Construct the local 3D incident field state used by the discrete source."""
        profiles, indices = self._get_3d_profiles_and_indices()
        dx = dy = dz = float(self._resolution or 0.0)
        axis = self._axis
        if axis is None:
            raise RuntimeError(
                "3D incident state requested before source initialization"
            )
        k_num = self._k_num_axis
        omega = self._omega_launch
        if k_num is None or omega is None:
            raise RuntimeError(
                "3D incident state requested without discrete launch metadata"
            )

        plane_coord = float(self._phase_plane_coord)
        ref_coord = float(self._phase_ref_coord)
        d_axis = {"x": dx, "y": dy, "z": dz}[axis]
        max_shift = int(max(1, self._discrete_launch_max_shift))
        direction_sign = float(self._direction_sign)
        staggered_along_axis = {
            "x": {"Ex", "Hy", "Hz"},
            "y": {"Ey", "Hx", "Hz"},
            "z": {"Ez", "Hx", "Hy"},
        }

        field_arrays = {
            "Ex": np.zeros_like(np.asarray(fields.Ex), dtype=np.float64),
            "Ey": np.zeros_like(np.asarray(fields.Ey), dtype=np.float64),
            "Ez": np.zeros_like(np.asarray(fields.Ez), dtype=np.float64),
            "Hx": np.zeros_like(np.asarray(fields.Hx), dtype=np.float64),
            "Hy": np.zeros_like(np.asarray(fields.Hy), dtype=np.float64),
            "Hz": np.zeros_like(np.asarray(fields.Hz), dtype=np.float64),
        }
        field_shapes = {name: arr.shape for name, arr in field_arrays.items()}

        for comp_name, profile in profiles.items():
            idx = indices.get(comp_name)
            if profile is None or idx is None:
                continue

            base_axis_idx = _axis_index_from_component_indices(idx, axis)
            base_coord = _component_axis_coord(
                comp_name, base_axis_idx, axis, dx, dy, dz
            )
            profile_arr = np.asarray(profile, dtype=np.complex128)
            base_time = float(t_e if comp_name.startswith("E") else t_h)

            for shift in range(-max_shift, max_shift + 1):
                shifted_idx = _shift_component_indices_along_axis(
                    idx, axis, shift, field_shapes[comp_name]
                )
                if shifted_idx is None:
                    continue

                coord = float(base_coord + shift * d_axis)
                if masked:
                    mask_coord = (
                        ref_coord
                        if comp_name in staggered_along_axis[axis]
                        else plane_coord
                    )
                    if direction_sign * (coord - mask_coord) < -1e-12:
                        continue

                delay = _numeric_phase_delay(omega, k_num, coord - ref_coord)
                signal_time = base_time - delay
                amp_re = float(self._get_signal_value(signal_time, dt))
                amp_im = float(self._get_signal_quadrature_value(signal_time, dt))
                if amp_re == 0.0 and amp_im == 0.0:
                    continue

                field_arrays[comp_name][shifted_idx] = field_arrays[comp_name][
                    shifted_idx
                ] + _real_phasor_sample(profile_arr, amp_re, amp_im)

        return field_arrays

    def _build_incident_3d_phasor_state(
        self,
        fields,
        *,
        t_e,
        t_h,
        masked,
    ):
        """Construct a complex carrier phasor for the local 3D incident field."""
        profiles, indices = self._get_3d_profiles_and_indices()
        dx = dy = dz = float(self._resolution or 0.0)
        axis = self._axis
        if axis is None:
            raise RuntimeError(
                "3D incident phasor requested before source initialization"
            )
        k_num = self._k_num_axis
        omega = self._omega_launch
        if k_num is None or omega is None:
            raise RuntimeError(
                "3D incident phasor requested without discrete launch metadata"
            )

        plane_coord = float(self._phase_plane_coord)
        ref_coord = float(self._phase_ref_coord)
        d_axis = {"x": dx, "y": dy, "z": dz}[axis]
        max_shift = int(max(1, self._discrete_launch_max_shift))
        direction_sign = float(self._direction_sign)
        staggered_along_axis = {
            "x": {"Ex", "Hy", "Hz"},
            "y": {"Ey", "Hx", "Hz"},
            "z": {"Ez", "Hx", "Hy"},
        }

        field_arrays = {
            "Ex": np.zeros_like(np.asarray(fields.Ex), dtype=np.complex128),
            "Ey": np.zeros_like(np.asarray(fields.Ey), dtype=np.complex128),
            "Ez": np.zeros_like(np.asarray(fields.Ez), dtype=np.complex128),
            "Hx": np.zeros_like(np.asarray(fields.Hx), dtype=np.complex128),
            "Hy": np.zeros_like(np.asarray(fields.Hy), dtype=np.complex128),
            "Hz": np.zeros_like(np.asarray(fields.Hz), dtype=np.complex128),
        }
        field_shapes = {name: arr.shape for name, arr in field_arrays.items()}

        for comp_name, profile in profiles.items():
            idx = indices.get(comp_name)
            if profile is None or idx is None:
                continue

            base_axis_idx = _axis_index_from_component_indices(idx, axis)
            base_coord = _component_axis_coord(
                comp_name, base_axis_idx, axis, dx, dy, dz
            )
            profile_arr = np.asarray(profile, dtype=np.complex128)
            base_time = float(t_e if comp_name.startswith("E") else t_h)

            for shift in range(-max_shift, max_shift + 1):
                shifted_idx = _shift_component_indices_along_axis(
                    idx, axis, shift, field_shapes[comp_name]
                )
                if shifted_idx is None:
                    continue

                coord = float(base_coord + shift * d_axis)
                if masked:
                    mask_coord = (
                        ref_coord
                        if comp_name in staggered_along_axis[axis]
                        else plane_coord
                    )
                    if direction_sign * (coord - mask_coord) < -1e-12:
                        continue

                delay = _numeric_phase_delay(omega, k_num, coord - ref_coord)
                phase = float(omega) * (base_time - delay)
                field_arrays[comp_name][shifted_idx] = field_arrays[comp_name][
                    shifted_idx
                ] + profile_arr * np.exp(1j * phase)

        return field_arrays

    def _launched_side_component_mask_3d(
        self,
        component: str,
        shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Return a broadcast mask for the component-local launched side."""
        axis = self._axis
        if axis is None:
            raise RuntimeError(
                "3D incident mask requested before source initialization"
            )
        dx = dy = dz = float(self._resolution or 0.0)
        d_axis = {"x": dx, "y": dy, "z": dz}[axis]
        axis_pos = {"z": 0, "y": 1, "x": 2}[axis]
        staggered_along_axis = {
            "x": {"Ex", "Hy", "Hz"},
            "y": {"Ey", "Hx", "Hz"},
            "z": {"Ez", "Hx", "Hy"},
        }
        offset = 1.0 if component in staggered_along_axis[axis] else 0.5
        coord = (np.arange(int(shape[axis_pos]), dtype=np.float64) + offset) * d_axis
        mask_coord = (
            float(self._phase_ref_coord)
            if component in staggered_along_axis[axis]
            else float(self._phase_plane_coord)
        )
        launched = float(self._direction_sign) * (coord - mask_coord) >= -1e-12
        reshape = [1, 1, 1]
        reshape[axis_pos] = int(launched.size)
        return launched.reshape(tuple(reshape))

    def _mask_incident_3d_state_to_launched_side(
        self,
        state: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Keep only the side of an incident state that should enter the grid."""
        out: dict[str, np.ndarray] = {}
        for component, values in state.items():
            arr = np.asarray(values)
            mask = self._launched_side_component_mask_3d(component, arr.shape)
            out[component] = np.where(mask, arr, np.zeros((), dtype=arr.dtype))
        return out

    @staticmethod
    def _component_slices_to_cell_bbox(
        component: str,
        index: tuple[slice, slice, slice],
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        offsets = component_axis_offsets_3d(component)
        bounds: list[tuple[int, int]] = []
        for axis_name, item in zip(("z", "y", "x"), index, strict=True):
            start = int(item.start or 0)
            stop = int(item.stop or start)
            if float(offsets[axis_name]) == 0.5:
                stop += 1
            bounds.append((start, stop))
        return tuple(bounds)  # type: ignore[return-value]

    @staticmethod
    def _crop_local_residual(
        component: str,
        timing: str,
        local_index: tuple[slice, slice, slice],
        residual: np.ndarray,
        *,
        atol: float = 1e-30,
    ) -> _ModeSource3DResidual | None:
        values = np.asarray(residual, dtype=np.complex128)
        if values.size == 0:
            return None
        mask = np.abs(values) > float(atol)
        if not np.any(mask):
            return None
        coords = np.argwhere(mask)
        lo = coords.min(axis=0)
        hi = coords.max(axis=0) + 1
        local_crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
        global_crop = tuple(
            slice(
                int(parent.start or 0) + int(child.start or 0),
                int(parent.start or 0) + int(child.stop or 0),
            )
            for parent, child in zip(local_index, local_crop, strict=True)
        )
        return _ModeSource3DResidual(
            component=component,
            timing=timing,
            index=global_crop,  # type: ignore[arg-type]
            residual=values[local_crop].copy(),
        )

    def _compute_discrete_3d_h_phasor_residuals(
        self,
        fields,
        *,
        dt: float,
    ) -> tuple[_ModeSource3DResidual, ...]:
        """Complex carrier H residuals for the launched-side TF/SF update."""
        full_prev = self._build_incident_3d_phasor_state(
            fields,
            t_e=0.0,
            t_h=-0.5 * float(dt),
            masked=False,
        )
        masked_prev = self._build_incident_3d_phasor_state(
            fields,
            t_e=0.0,
            t_h=-0.5 * float(dt),
            masked=True,
        )
        h_full_next = self._advance_incident_h_3d(fields, full_prev, dt)
        h_target_next = self._mask_incident_3d_state_to_launched_side(h_full_next)
        h_mask_next = self._advance_incident_h_3d(fields, masked_prev, dt)
        delta = {
            component: h_target_next[component] - h_mask_next[component]
            for component in ("Hx", "Hy", "Hz")
        }
        return self._dense_3d_delta_residuals(delta, timing="h")

    def _compute_discrete_3d_e_phasor_residuals(
        self,
        fields,
        *,
        dt: float,
    ) -> tuple[_ModeSource3DResidual, ...]:
        """Complex carrier E residuals for the launched-side TF/SF update."""
        full_prev = self._build_incident_3d_phasor_state(
            fields,
            t_e=0.0,
            t_h=-0.5 * float(dt),
            masked=False,
        )
        masked_prev = self._build_incident_3d_phasor_state(
            fields,
            t_e=0.0,
            t_h=-0.5 * float(dt),
            masked=True,
        )
        h_full_next = self._advance_incident_h_3d(fields, full_prev, dt)
        h_target_next = self._mask_incident_3d_state_to_launched_side(h_full_next)
        e_full_next = self._advance_incident_e_3d(fields, full_prev, h_full_next, dt)
        e_target_next = self._mask_incident_3d_state_to_launched_side(e_full_next)
        e_mask_next = self._advance_incident_e_3d(
            fields,
            masked_prev,
            h_target_next,
            dt,
        )
        delta = {
            component: e_target_next[component] - e_mask_next[component]
            for component in ("Ex", "Ey", "Ez")
        }
        return self._dense_3d_delta_residuals(delta, timing="e")

    def _dense_3d_delta_residuals(
        self,
        delta: dict[str, np.ndarray],
        *,
        timing: str,
    ) -> tuple[_ModeSource3DResidual, ...]:
        out: list[_ModeSource3DResidual] = []
        for component, values in delta.items():
            arr = np.asarray(values, dtype=np.complex128)
            full_index = tuple(slice(0, int(size)) for size in arr.shape)
            residual = self._crop_local_residual(
                component,
                timing,
                full_index,  # type: ignore[arg-type]
                arr,
            )
            if residual is not None:
                out.append(residual)
        return tuple(out)

    def _compute_discrete_3d_phasor_residuals(
        self,
        fields,
        *,
        dt: float,
    ) -> tuple[_ModeSource3DResidual, ...]:
        return (
            *self._compute_discrete_3d_h_phasor_residuals(fields, dt=dt),
            *self._compute_discrete_3d_e_phasor_residuals(fields, dt=dt),
        )

    @staticmethod
    def _expand_3d_residuals(
        residuals: tuple[_ModeSource3DResidual, ...],
        fields,
        components: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        expanded = {
            component: np.zeros(
                tuple(int(v) for v in getattr(fields, component).shape),
                dtype=np.complex128,
            )
            for component in components
        }
        for residual in residuals:
            if residual.component in expanded:
                expanded[residual.component][residual.index] += np.asarray(
                    residual.residual,
                    dtype=np.complex128,
                )
        return expanded

    def _advance_incident_h_3d(self, fields, state, dt):
        """Advance an incident 3D state through the source-free H half-step."""
        from beamz.simulation import ops
        from beamz.simulation.boundaries import (
            has_full_pec_3d,
            initialize_full_pec_3d_state,
            pec_curl_e_to_h_3d,
        )

        ex = jnp.asarray(state["Ex"])
        ey = jnp.asarray(state["Ey"])
        ez = jnp.asarray(state["Ez"])
        hx = jnp.asarray(state["Hx"])
        hy = jnp.asarray(state["Hy"])
        hz = jnp.asarray(state["Hz"])
        boundaries = getattr(fields, "boundaries", None)

        if has_full_pec_3d(boundaries):
            fp_state = initialize_full_pec_3d_state(fields)
            for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                compact = jnp.asarray(state[comp])
                full = jnp.asarray(getattr(fp_state, comp), dtype=compact.dtype)
                full = full.at[:-1, :-1, :-1].set(compact)
                zero = jnp.asarray(0.0, dtype=full.dtype)
                setattr(fp_state, comp, jnp.where(fp_state.masks[comp], zero, full))
            curl_hx, curl_hy, curl_hz = pec_curl_e_to_h_3d(
                fp_state.Ex,
                fp_state.Ey,
                fp_state.Ez,
                self._resolution,
                fp_state.Hx.shape,
                fp_state.Hy.shape,
                fp_state.Hz.shape,
            )
            hx_next = ops.advance_h_field(fp_state.Hx, curl_hx, fp_state.sigma_m_hx, dt)
            hy_next = ops.advance_h_field(fp_state.Hy, curl_hy, fp_state.sigma_m_hy, dt)
            hz_next = ops.advance_h_field(fp_state.Hz, curl_hz, fp_state.sigma_m_hz, dt)
            hx_next = jnp.where(
                fp_state.masks["Hx"], jnp.asarray(0.0, dtype=hx_next.dtype), hx_next
            )
            hy_next = jnp.where(
                fp_state.masks["Hy"], jnp.asarray(0.0, dtype=hy_next.dtype), hy_next
            )
            hz_next = jnp.where(
                fp_state.masks["Hz"], jnp.asarray(0.0, dtype=hz_next.dtype), hz_next
            )
            return {
                "Hx": np.asarray(hx_next[:-1, :-1, :-1]),
                "Hy": np.asarray(hy_next[:-1, :-1, :-1]),
                "Hz": np.asarray(hz_next[:-1, :-1, :-1]),
            }

        curl_hx, curl_hy, curl_hz = ops.curl_e_to_h_3d(ex, ey, ez, self._resolution)
        return {
            "Hx": np.asarray(
                ops.advance_h_field(hx, curl_hx, fields.sigma_m_hx, dt),
            ),
            "Hy": np.asarray(
                ops.advance_h_field(hy, curl_hy, fields.sigma_m_hy, dt),
            ),
            "Hz": np.asarray(
                ops.advance_h_field(hz, curl_hz, fields.sigma_m_hz, dt),
            ),
        }

    def _advance_incident_e_3d(self, fields, state, h_next, dt):
        """Advance an incident 3D state through the source-free E half-step."""
        from beamz.simulation import ops
        from beamz.simulation.boundaries import (
            build_h_boundary_views_for_e_3d,
            full_pec_e_update_coefficients_3d,
            full_pec_update_e_from_h_3d,
            has_full_pec_3d,
            initialize_full_pec_3d_state,
        )

        ex = jnp.asarray(state["Ex"])
        ey = jnp.asarray(state["Ey"])
        ez = jnp.asarray(state["Ez"])
        hx = jnp.asarray(h_next["Hx"])
        hy = jnp.asarray(h_next["Hy"])
        hz = jnp.asarray(h_next["Hz"])
        boundaries = getattr(fields, "boundaries", None)

        if has_full_pec_3d(boundaries):
            fp_state = initialize_full_pec_3d_state(fields)
            for comp in ("Ex", "Ey", "Ez"):
                compact = jnp.asarray(state[comp])
                full = jnp.asarray(getattr(fp_state, comp), dtype=compact.dtype)
                full = full.at[:-1, :-1, :-1].set(compact)
                zero = jnp.asarray(0.0, dtype=full.dtype)
                setattr(fp_state, comp, jnp.where(fp_state.masks[comp], zero, full))
            for comp, arr in (("Hx", hx), ("Hy", hy), ("Hz", hz)):
                compact = jnp.asarray(arr)
                full = jnp.asarray(getattr(fp_state, comp), dtype=compact.dtype)
                full = full.at[:-1, :-1, :-1].set(compact)
                zero = jnp.asarray(0.0, dtype=full.dtype)
                setattr(fp_state, comp, jnp.where(fp_state.masks[comp], zero, full))
            e_decay, e_source = full_pec_e_update_coefficients_3d(fp_state, dt)
            ex_next, ey_next, ez_next = full_pec_update_e_from_h_3d(
                fp_state.Hx,
                fp_state.Hy,
                fp_state.Hz,
                fp_state.Ex,
                fp_state.Ey,
                fp_state.Ez,
                self._resolution,
                e_decay=e_decay,
                e_source=e_source,
                e_mask=(
                    fp_state.masks["Ex"],
                    fp_state.masks["Ey"],
                    fp_state.masks["Ez"],
                ),
            )
            return {
                "Ex": np.asarray(ex_next[:-1, :-1, :-1]),
                "Ey": np.asarray(ey_next[:-1, :-1, :-1]),
                "Ez": np.asarray(ez_next[:-1, :-1, :-1]),
            }

        boundaries = getattr(fields, "boundaries", None)
        boundary_views = build_h_boundary_views_for_e_3d(hx, hy, hz, boundaries)
        curl_hx, curl_hy, curl_hz = ops.curl_h_to_e_3d(
            hx,
            hy,
            hz,
            self._resolution,
            ex_shape=ex.shape,
            ey_shape=ey.shape,
            ez_shape=ez.shape,
            boundary_views=boundary_views,
        )
        return {
            "Ex": np.asarray(
                ops.advance_e_field(
                    ex, curl_hx, fields.sig_x, fields.eps_x, dt, fields.region_x
                ),
            ),
            "Ey": np.asarray(
                ops.advance_e_field(
                    ey, curl_hy, fields.sig_y, fields.eps_y, dt, fields.region_y
                ),
            ),
            "Ez": np.asarray(
                ops.advance_e_field(
                    ez, curl_hz, fields.sig_z, fields.eps_z, dt, fields.region_z
                ),
            ),
        }

    def _compute_discrete_3d_h_delta(self, fields, *, t, dt):
        """Exact discrete H-source residual for the current split launch step."""
        full_prev = self._build_incident_3d_state(
            fields, t_e=float(t), t_h=float(t - 0.5 * dt), dt=dt, masked=False
        )
        masked_prev = self._build_incident_3d_state(
            fields, t_e=float(t), t_h=float(t - 0.5 * dt), dt=dt, masked=True
        )
        h_full_next = self._advance_incident_h_3d(fields, full_prev, dt)
        h_target_next = self._mask_incident_3d_state_to_launched_side(h_full_next)
        h_mask_next = self._advance_incident_h_3d(fields, masked_prev, dt)
        return {
            "Hx": h_target_next["Hx"] - h_mask_next["Hx"],
            "Hy": h_target_next["Hy"] - h_mask_next["Hy"],
            "Hz": h_target_next["Hz"] - h_mask_next["Hz"],
        }

    def _compute_discrete_3d_e_delta(self, fields, *, t, dt):
        """Exact discrete E-source residual for the current split launch step."""
        full_prev = self._build_incident_3d_state(
            fields, t_e=float(t), t_h=float(t - 0.5 * dt), dt=dt, masked=False
        )
        masked_prev = self._build_incident_3d_state(
            fields, t_e=float(t), t_h=float(t - 0.5 * dt), dt=dt, masked=True
        )
        h_full_next = self._advance_incident_h_3d(fields, full_prev, dt)
        h_target_next = self._mask_incident_3d_state_to_launched_side(h_full_next)
        e_full_next = self._advance_incident_e_3d(fields, full_prev, h_full_next, dt)
        e_target_next = self._mask_incident_3d_state_to_launched_side(e_full_next)
        e_mask_next = self._advance_incident_e_3d(
            fields,
            masked_prev,
            h_target_next,
            dt,
        )
        return {
            "Ex": e_target_next["Ex"] - e_mask_next["Ex"],
            "Ey": e_target_next["Ey"] - e_mask_next["Ey"],
            "Ez": e_target_next["Ez"] - e_mask_next["Ez"],
        }

    def _compute_discrete_3d_h_phasor_delta(self, fields, *, dt):
        """Complex carrier residual for compiled 3D ModeSource H injection."""
        return self._expand_3d_residuals(
            self._compute_discrete_3d_h_phasor_residuals(fields, dt=float(dt)),
            fields,
            ("Hx", "Hy", "Hz"),
        )

    def _compute_discrete_3d_e_phasor_delta(self, fields, *, dt):
        """Complex carrier residual for compiled 3D ModeSource E injection."""
        return self._expand_3d_residuals(
            self._compute_discrete_3d_e_phasor_residuals(fields, dt=float(dt)),
            fields,
            ("Ex", "Ey", "Ez"),
        )

    def _inject_3d_h(self, fields, t, dt, resolution):
        """Inject H-field components for 3D source via exact discrete residual."""
        delta = self._compute_discrete_3d_h_delta(fields, t=t, dt=dt)
        fields.Hx = fields.Hx + jnp.asarray(delta["Hx"], dtype=fields.Hx.dtype)
        fields.Hy = fields.Hy + jnp.asarray(delta["Hy"], dtype=fields.Hy.dtype)
        fields.Hz = fields.Hz + jnp.asarray(delta["Hz"], dtype=fields.Hz.dtype)

    def _inject_3d_e(self, fields, t, dt, resolution):
        """Inject E-field components for 3D source via exact discrete residual."""
        delta = self._compute_discrete_3d_e_delta(fields, t=t, dt=dt)
        fields.Ex = fields.Ex + jnp.asarray(delta["Ex"], dtype=fields.Ex.dtype)
        fields.Ey = fields.Ey + jnp.asarray(delta["Ey"], dtype=fields.Ey.dtype)
        fields.Ez = fields.Ez + jnp.asarray(delta["Ez"], dtype=fields.Ez.dtype)

    # -- 2D injection (split, with corrected signs) ---------------------

    def _inject_2d_h(self, fields, signal_h, dt, resolution):
        """Inject magnetic current into H-fields for 2D (after H update)."""
        if self.pol == "tm":
            if self._h_indices is not None and self._my_profile is not None:
                mu_at_source = component_permeability_at(
                    fields,
                    self._h_component,
                    self._h_indices,
                )
                my_term = self._my_profile * signal_h / resolution
                h_injection = -my_term * dt / (MU_0 * mu_at_source)
                if self._h_component == "Hx":
                    fields.Hx = fields.Hx.at[self._h_indices].add(h_injection)
                else:
                    fields.Hy = fields.Hy.at[self._h_indices].add(h_injection)
                fields.apply_tm_xy_pec_masks()
        else:  # TE
            if self._hz_indices is not None and self._mz_profile is not None:
                mu_at_source = component_permeability_at(fields, "Hz", self._hz_indices)
                mz_term = self._mz_profile * signal_h / resolution
                hz_injection = +mz_term * dt / (MU_0 * mu_at_source)
                fields.Hz = fields.Hz.at[self._hz_indices].add(hz_injection)

    def _inject_2d_e(self, fields, signal_e, dt, resolution):
        """Inject electric current into E-fields for 2D (after E update)."""
        if self.pol == "tm":
            if self._ez_indices is not None and self._jz_profile is not None:
                eps_at_source = component_permittivity_at(
                    fields,
                    "Ez",
                    self._ez_indices,
                )
                jz_term = self._jz_profile * signal_e / resolution
                ez_injection = +jz_term * dt / (EPS_0 * eps_at_source)
                fields.Ez = fields.Ez.at[self._ez_indices].add(ez_injection)
                fields.apply_tm_xy_pec_masks()
        else:  # TE
            if self._e_indices is not None:
                j_profile = (
                    self._jx_profile if self._e_component == "Ex" else self._jy_profile
                )
                if j_profile is not None:
                    eps_at_source = component_permittivity_at(
                        fields,
                        self._e_component,
                        self._e_indices,
                    )
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

    def mode_permittivity_plot_data(self):
        """Return permittivity data used by this mode source."""
        from beamz.visual.data import mode_permittivity_plot_data

        return mode_permittivity_plot_data(self)

    def signal_plot_data(self, *, t=None):
        """Return renderer-agnostic source signal data."""
        from beamz.visual.data import source_signal_plot_data

        return source_signal_plot_data(self, t=t)

    def spectrum_plot_data(self, *, t=None, dt=None):
        """Return renderer-agnostic source spectrum data."""
        from beamz.visual.data import source_spectrum_plot_data

        return source_spectrum_plot_data(self, t=t, dt=dt)

    def plot(self, **kwargs):
        """Plot the mode profile using the matplotlib backend."""
        from beamz.visual.mpl import plot_mode_profile

        kwargs.setdefault("show", False)
        return plot_mode_profile(self, **kwargs)

    def show(self, **kwargs):
        """Display the mode profile using the matplotlib backend."""
        kwargs.setdefault("show", True)
        return self.plot(**kwargs)

    def plot_mode(self, **kwargs):
        """Alias for :meth:`plot`."""
        return self.plot(**kwargs)

    def plot_eps(self, **kwargs):
        """Plot the permittivity profile used by the mode solve."""
        from beamz.visual.mpl import plot_mode_permittivity

        kwargs.setdefault("show", False)
        return plot_mode_permittivity(self, **kwargs)

    def show_eps(self, **kwargs):
        """Display the permittivity profile used by the mode solve."""
        kwargs.setdefault("show", True)
        return self.plot_eps(**kwargs)

    def plot_signal(self, **kwargs):
        """Plot the source time dependence."""
        from beamz.visual.mpl import plot_source_signal

        kwargs.setdefault("show", False)
        return plot_source_signal(self, **kwargs)

    def show_signal(self, **kwargs):
        """Display the source time dependence."""
        kwargs.setdefault("show", True)
        return self.plot_signal(**kwargs)

    def plot_spectrum(self, **kwargs):
        """Plot the normalized source spectrum."""
        from beamz.visual.mpl import plot_source_spectrum

        kwargs.setdefault("show", False)
        return plot_source_spectrum(self, **kwargs)

    def show_spectrum(self, **kwargs):
        """Display the normalized source spectrum."""
        kwargs.setdefault("show", True)
        return self.plot_spectrum(**kwargs)

    def to_xarray(self, *, t=None):
        """Return mode profile and signal data as an xarray Dataset."""
        from beamz.data.xarray import mode_dataset

        return mode_dataset(self, t=t)

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
