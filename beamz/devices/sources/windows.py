"""Windowing and Yee-grid staggering helpers for mode sources."""

import jax.numpy as jnp
import numpy as np


def _jax_tukey_window(M: int, alpha: float = 0.5) -> jnp.ndarray:
    """JAX-compatible Tukey (tapered cosine) window."""
    if M <= 0:
        return jnp.array([])
    if M == 1:
        return jnp.ones(1)

    n = jnp.arange(M)
    width = alpha * (M - 1) / 2.0
    width = jnp.maximum(width, 1e-10)

    left_taper = 0.5 * (1 + jnp.cos(jnp.pi * (n / width - 1)))
    right_taper = 0.5 * (1 + jnp.cos(jnp.pi * ((n - (M - 1 - width)) / width)))

    return jnp.where(
        n < width, left_taper, jnp.where(n > (M - 1) - width, right_taper, 1.0)
    )


def _scipy_tukey(n, alpha=0.3):
    from scipy.signal.windows import tukey

    return tukey(n, alpha=alpha)


def _crop_and_window_2d(profile, z_s, z_e, t_s, t_e, window):
    """Crop a 2D profile and apply the provided window."""
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
    make_window = _jax_tukey_window if use_jax else _scipy_tukey
    ones = jnp.ones if use_jax else np.ones

    wz = (
        make_window(height_cells, alpha=alpha)
        if height_cells > 2
        else ones(max(1, height_cells))
    )
    wt = (
        make_window(width_cells, alpha=alpha)
        if width_cells > 2
        else ones(max(1, width_cells))
    )

    if use_jax:
        return wz[:, jnp.newaxis] * wt[jnp.newaxis, :]
    return wz[:, np.newaxis] * wt[np.newaxis, :]


def _stagger_half(field, axis):
    """Average adjacent cells along one axis for Yee half-grid staggering."""
    if field.shape[axis] <= 1:
        return field
    if axis == 0:
        return 0.5 * (field[:-1, :] + field[1:, :])
    return 0.5 * (field[:, :-1] + field[:, 1:])


def _stagger_both(field):
    """Stagger along both axes."""
    out = field
    if out.shape[1] > 1:
        out = 0.5 * (out[:, :-1] + out[:, 1:])
    if out.shape[0] > 1:
        out = 0.5 * (out[:-1, :] + out[1:, :])
    return out


def _compute_transverse_bounds(center_val, extent, resolution, grid_max):
    """Return `(start_idx, end_idx)` for an injection window."""
    center_idx = int(round(center_val / resolution))
    half_idx = int(round((extent / 2) / resolution))
    return max(0, center_idx - half_idx), min(grid_max, center_idx + half_idx)
