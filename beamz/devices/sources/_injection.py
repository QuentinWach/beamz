"""Shared helpers for ModeSource injection setup.

Consolidates duplicated staggering, windowing, and index computation logic
that was previously copy-pasted for x vs y propagation in both 2D and 3D.
"""

import logging
import warnings

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0


def jax_tukey_window(M: int, alpha: float = 0.5) -> jnp.ndarray:
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

logger = logging.getLogger(__name__)


def crop_and_window_2d(profile, z_s, z_e, t_s, t_e, window):
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


def make_tukey_window_2d(height_cells, width_cells, alpha=0.3, use_jax=True):
    """Create a 2D Tukey window via outer product of 1D windows."""
    _make = jax_tukey_window if use_jax else _scipy_tukey
    _ones = jnp.ones if use_jax else np.ones

    wz = _make(height_cells, alpha=alpha) if height_cells > 2 else _ones(max(1, height_cells))
    wt = _make(width_cells, alpha=alpha) if width_cells > 2 else _ones(max(1, width_cells))

    if use_jax:
        return wz[:, jnp.newaxis] * wt[jnp.newaxis, :]
    return wz[:, np.newaxis] * wt[np.newaxis, :]


def _scipy_tukey(n, alpha=0.3):
    from scipy.signal.windows import tukey
    return tukey(n, alpha=alpha)


def stagger_half(field, axis):
    """Average adjacent cells along *axis* (0 or 1) for Yee half-grid staggering."""
    if field.shape[axis] <= 1:
        return field
    if axis == 0:
        return 0.5 * (field[:-1, :] + field[1:, :])
    return 0.5 * (field[:, :-1] + field[:, 1:])


def stagger_both(field):
    """Stagger along both axes (for longitudinal H that sits at half-grid in both)."""
    out = field
    if out.shape[1] > 1:
        out = 0.5 * (out[:, :-1] + out[:, 1:])
    if out.shape[0] > 1:
        out = 0.5 * (out[:-1, :] + out[1:, :])
    return out


def compute_transverse_bounds(center_val, extent, resolution, grid_max):
    """Return (start_idx, end_idx) for the injection window along a transverse axis."""
    center_idx = int(round(center_val / resolution))
    half_idx = int(round((extent / 2) / resolution))
    return max(0, center_idx - half_idx), min(grid_max, center_idx + half_idx)


def impedance_correct_e_fields(E_components, H_components, Z_phys, use_jax=True):
    """Scale E-field profiles so E/H ratio matches physical impedance Z_phys.

    Parameters
    ----------
    E_components : list of arrays
        E-field profiles to scale.
    H_components : list of arrays
        H-field profiles used to measure current impedance.
    Z_phys : float
        Target impedance.
    use_jax : bool
        Whether to use JAX operations (True for 3D path) or numpy (False for 2D y-axis path).

    Returns
    -------
    list of arrays
        Corrected E-field profiles.
    """
    _mod = jnp if use_jax else np
    eps = 1e-12

    norm_E = eps
    for e in E_components:
        norm_E = max(norm_E, float(_mod.max(_mod.abs(e))))

    norm_H = eps
    for h in H_components:
        norm_H = max(norm_H, float(_mod.max(_mod.abs(h))))

    if norm_E > eps and norm_H > eps:
        corr = Z_phys / (norm_E / norm_H)
        return [e * corr for e in E_components]
    return list(E_components)


def build_3d_profiles(
    Ex, Ey, Ez, Hx, Hy, Hz,
    axis, direction,
    center, width, height,
    center_idx, offset_idx,
    grid_shape, resolution, neff,
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
    Z_phys = ETA_0 / max(np.real(neff), 1e-6)

    if axis == "x":
        return _build_3d_x(
            Ex, Ey, Ez, Hx, Hy, Hz,
            dir_sign, Z_phys,
            center, width, height,
            center_idx, offset_idx,
            nz, ny, nx, resolution,
        )
    else:
        return _build_3d_y(
            Ex, Ey, Ez, Hx, Hy, Hz,
            dir_sign, Z_phys,
            center, width, height,
            center_idx, offset_idx,
            nz, ny, nx, resolution,
        )


def _build_3d_x(
    Ex, Ey, Ez, Hx, Hy, Hz,
    dir_sign, Z_phys,
    center, width, height,
    center_idx, offset_idx,
    nz, ny, nx, resolution,
):
    # --- stagger ---
    Ey_s = stagger_half(Ey, axis=1)
    Ez_s = stagger_half(Ez, axis=0)
    Ex_s = Ex
    Hx_s = stagger_both(Hx)
    Hy_s = stagger_half(Hy, axis=0)
    Hz_s = stagger_half(Hz, axis=1)

    # --- transverse bounds ---
    y_start, y_end = compute_transverse_bounds(center[1], width, resolution, ny)
    z_center = center[2] if len(center) > 2 else (nz // 2) * resolution
    z_start, z_end = compute_transverse_bounds(z_center, height, resolution, nz)

    # --- indices  (z_slice, y_slice, x_index) ---
    indices = {
        "Ex": (slice(z_start, min(z_end, Ex_s.shape[0], nz)),
               slice(y_start, min(y_end, Ex_s.shape[1], ny)),
               offset_idx),
        "Ey": (slice(z_start, min(z_end, Ey_s.shape[0], nz)),
               slice(y_start, min(y_end, Ey_s.shape[1], ny - 1)),
               center_idx),
        "Ez": (slice(z_start, min(z_end, Ez_s.shape[0], nz - 1)),
               slice(y_start, min(y_end, Ez_s.shape[1], ny)),
               center_idx),
        "Hx": (slice(z_start, min(z_end, Hx_s.shape[0], nz - 1)),
               slice(y_start, min(y_end, Hx_s.shape[1], ny - 1)),
               center_idx),
        "Hy": (slice(z_start, min(z_end, Hy_s.shape[0], nz - 1)),
               slice(y_start, min(y_end, Hy_s.shape[1], ny)),
               offset_idx),
        "Hz": (slice(z_start, min(z_end, Hz_s.shape[0], nz)),
               slice(y_start, min(y_end, Hz_s.shape[1], ny - 1)),
               offset_idx),
    }

    # --- impedance correction (JAX) ---
    Ex_s, Ey_s, Ez_s = impedance_correct_e_fields(
        [Ex_s, Ey_s, Ez_s], [Hy_s, Hz_s], Z_phys, use_jax=True,
    )

    # --- crop & window ---
    staggered = {"Ex": Ex_s, "Ey": Ey_s, "Ez": Ez_s, "Hx": Hx_s, "Hy": Hy_s, "Hz": Hz_s}
    profiles = _crop_and_window_all(
        staggered, z_start, z_end, y_start, y_end, dir_sign, use_jax=True,
    )

    extra = {"_y_start": y_start, "_y_end": y_end, "_z_start": z_start, "_z_end": z_end,
             "_h_component": "Hy", "_e_component": "Ey"}
    return profiles, indices, extra


def _build_3d_y(
    Ex, Ey, Ez, Hx, Hy, Hz,
    dir_sign, Z_phys,
    center, width, height,
    center_idx, offset_idx,
    nz, ny, nx, resolution,
):
    # --- stagger (y-propagation) ---
    Ex_s = stagger_half(Ex, axis=1)
    Ey_s = Ey
    Ez_s = stagger_half(Ez, axis=0)
    Hx_s = stagger_half(Hx, axis=0)
    Hy_s = stagger_both(Hy)
    Hz_s = stagger_half(Hz, axis=1)

    # --- transverse bounds ---
    x_start, x_end = compute_transverse_bounds(center[0], width, resolution, nx)
    z_center = center[2] if len(center) > 2 else (nz // 2) * resolution
    z_start, z_end = compute_transverse_bounds(z_center, height, resolution, nz)

    # --- indices  (z_slice, y_index, x_slice) ---
    indices = {
        "Ex": (slice(z_start, min(z_end, Ex_s.shape[0], nz)),
               center_idx,
               slice(x_start, min(x_end, Ex_s.shape[1], nx - 1))),
        "Ey": (slice(z_start, min(z_end, Ey_s.shape[0], nz)),
               offset_idx,
               slice(x_start, min(x_end, Ey_s.shape[1], nx))),
        "Ez": (slice(z_start, min(z_end, Ez_s.shape[0], nz - 1)),
               center_idx,
               slice(x_start, min(x_end, Ez_s.shape[1], nx))),
        "Hx": (slice(z_start, min(z_end, Hx_s.shape[0], nz - 1)),
               center_idx,
               slice(x_start, min(x_end, Hx_s.shape[1], nx))),
        "Hy": (slice(z_start, min(z_end, Hy_s.shape[0], nz - 1)),
               center_idx,
               slice(x_start, min(x_end, Hy_s.shape[1], nx - 1))),
        "Hz": (slice(z_start, min(z_end, Hz_s.shape[0], nz)),
               offset_idx,
               slice(x_start, min(x_end, Hz_s.shape[1], nx - 1))),
    }

    # --- impedance correction (numpy path — matches original y-axis code) ---
    Ex_s, Ey_s, Ez_s = impedance_correct_e_fields(
        [Ex_s, Ey_s, Ez_s], [Hx_s, Hz_s], Z_phys, use_jax=False,
    )

    # --- crop & window ---
    staggered = {"Ex": Ex_s, "Ey": Ey_s, "Ez": Ez_s, "Hx": Hx_s, "Hy": Hy_s, "Hz": Hz_s}
    profiles = _crop_and_window_all(
        staggered, z_start, z_end, x_start, x_end, dir_sign, use_jax=False,
    )

    extra = {"_x_start": x_start, "_x_end": x_end, "_z_start": z_start, "_z_end": z_end,
             "_h_component": "Hx", "_e_component": "Ex"}
    return profiles, indices, extra


def _crop_and_window_all(staggered, z_start, z_end, t_start, t_end, dir_sign, use_jax):
    """Crop all six staggered profiles and multiply by a 2D Tukey window."""
    ref = next(iter(staggered.values()))
    pz_end = min(z_end, ref.shape[0])
    pt_end = min(t_end, ref.shape[1])
    h_cells = pz_end - z_start
    w_cells = pt_end - t_start

    window = make_tukey_window_2d(h_cells, w_cells, alpha=0.3, use_jax=use_jax)

    profiles = {}
    for name, field in staggered.items():
        fe = min(z_end, field.shape[0])
        te = min(t_end, field.shape[1])
        profiles[name] = dir_sign * np.real(
            crop_and_window_2d(field, z_start, fe, t_start, te, window)
        )
    return profiles


def inject_3d_fields(fields, profiles, indices, signal_e, signal_h, dt, resolution):
    """Inject all 6 field components into a 3D field object.

    Replaces the old try/except:pass pattern with explicit validation.
    """
    _inject_e_component(fields, "Ex", profiles, indices, "Hx", signal_e, dt, resolution)
    _inject_e_component(fields, "Ey", profiles, indices, "Hz", signal_e, dt, resolution)
    _inject_e_component(fields, "Ez", profiles, indices, "Hy", signal_e, dt, resolution)
    _inject_h_component(fields, "Hx", profiles, indices, "Ex", signal_h, dt, resolution)
    _inject_h_component(fields, "Hy", profiles, indices, "Ez", signal_h, dt, resolution)
    _inject_h_component(fields, "Hz", profiles, indices, "Ey", signal_h, dt, resolution)


def _inject_e_component(fields, comp, profiles, indices, j_source, sig, dt, res):
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
    setattr(fields, comp,
            getattr(fields, comp).at[idx].add(-j_term * sig * dt / (EPS_0 * eps * res)))


def _inject_h_component(fields, comp, profiles, indices, m_source, sig, dt, res):
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
    setattr(fields, comp,
            getattr(fields, comp).at[idx].add(-m_term * sig * dt / (MU_0 * mu_val * res)))


def _match_shape(profile, target_shape):
    """Match profile shape to target field shape, trimming or padding as needed."""
    if profile is None:
        return None
    profile = np.squeeze(profile)
    if profile.shape == target_shape:
        return profile
    if profile.ndim == len(target_shape):
        slices = tuple(slice(0, min(profile.shape[i], target_shape[i]))
                       for i in range(profile.ndim))
        trimmed = profile[slices]
        if trimmed.shape == target_shape:
            return trimmed
        result = np.zeros(target_shape, dtype=profile.dtype)
        result[tuple(slice(0, trimmed.shape[i]) for i in range(trimmed.ndim))] = trimmed
        return result
    return None
