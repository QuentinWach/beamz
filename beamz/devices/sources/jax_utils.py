"""JAX utilities for differentiable source initialization.

This module provides JAX-compatible replacements for NumPy/SciPy operations
used in source initialization, enabling gradient computation through source
parameters for inverse design optimization.
"""

from functools import partial

import jax
import jax.numpy as jnp


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


def soft_argmax(x: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """Differentiable argmax using softmax weighting.

    Returns a continuous index that approximates argmax but allows gradients.

    Args:
        x: Input array (will be flattened)
        temperature: Softmax temperature (lower = sharper, closer to true argmax)

    Returns:
        Soft index as a scalar
    """
    x_flat = x.flatten()
    weights = jax.nn.softmax(x_flat / temperature)
    indices = jnp.arange(x_flat.size, dtype=jnp.float32)
    return jnp.sum(weights * indices)


def soft_index_value(arr: jnp.ndarray, soft_idx: jnp.ndarray) -> jnp.ndarray:
    """Get value at a soft (continuous) index via linear interpolation.

    Args:
        arr: 1D array to index into
        soft_idx: Continuous index value

    Returns:
        Interpolated value at soft_idx
    """
    arr_flat = arr.flatten()
    idx_low = jnp.floor(soft_idx).astype(jnp.int32)
    idx_high = jnp.minimum(idx_low + 1, arr_flat.size - 1)
    idx_low = jnp.maximum(idx_low, 0)

    frac = soft_idx - jnp.floor(soft_idx)
    return (1 - frac) * arr_flat[idx_low] + frac * arr_flat[idx_high]


def differentiable_phase_alignment(
    field: jnp.ndarray, reference_field: jnp.ndarray = None, temperature: float = 0.01
) -> jnp.ndarray:
    """Align field phase to make it mostly real at peak amplitude.

    Uses soft argmax for differentiability instead of hard argmax.

    Args:
        field: Complex field to align
        reference_field: Field to find peak in (defaults to field itself)
        temperature: Softmax temperature for peak finding

    Returns:
        Phase-aligned field
    """
    if reference_field is None:
        reference_field = field

    # Soft argmax for differentiable peak finding
    abs_field = jnp.abs(reference_field.flatten())
    idx_soft = soft_argmax(abs_field, temperature=temperature)

    # Interpolate phase at soft index
    phases = jnp.angle(reference_field.flatten())
    phase_ref = soft_index_value(phases, idx_soft)

    return field * jnp.exp(-1j * phase_ref)


def stagger_field_yee(
    field: jnp.ndarray, axis: int, direction: str = "forward"
) -> jnp.ndarray:
    """Differentiable Yee grid staggering via averaging.

    Averages adjacent cells along the specified axis for Yee grid interpolation.

    Args:
        field: Field array to stagger
        axis: Axis along which to stagger
        direction: 'forward' or 'backward' staggering

    Returns:
        Staggered field (reduced by 1 along axis if field.shape[axis] > 1)
    """
    if field.shape[axis] <= 1:
        return field

    # Build slice objects for adjacent cells
    slices_low = [slice(None)] * field.ndim
    slices_high = [slice(None)] * field.ndim
    slices_low[axis] = slice(None, -1)
    slices_high[axis] = slice(1, None)

    return 0.5 * (field[tuple(slices_low)] + field[tuple(slices_high)])


def gaussian_spatial_profile(
    position: tuple, width: float, grid_coords: tuple, resolution: float
) -> jnp.ndarray:
    """Generate differentiable Gaussian spatial profile.

    Args:
        position: Center position (x, y) for 2D or (x, y, z) for 3D
        width: Standard deviation of Gaussian
        grid_coords: Tuple of coordinate arrays (X, Y) or (X, Y, Z)
        resolution: Grid resolution (for normalization if needed)

    Returns:
        Gaussian profile array
    """
    if len(position) == 2:
        x0, y0 = position
        X, Y = grid_coords
        dist_sq = (X - x0) ** 2 + (Y - y0) ** 2
    else:
        x0, y0, z0 = position
        X, Y, Z = grid_coords
        dist_sq = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2

    return jnp.exp(-dist_sq / (2 * width**2))


def create_grid_coords_2d(
    x_start: int, x_end: int, y_start: int, y_end: int, resolution: float
) -> tuple:
    """Create 2D grid coordinate arrays for Gaussian profile.

    Args:
        x_start, x_end: X index range
        y_start, y_end: Y index range
        resolution: Grid spacing

    Returns:
        Tuple (X, Y) of meshgrid coordinate arrays
    """
    x_coords = (jnp.arange(x_start, x_end) + 0.5) * resolution
    y_coords = (jnp.arange(y_start, y_end) + 0.5) * resolution
    return jnp.meshgrid(x_coords, y_coords, indexing="xy")


def create_grid_coords_3d(
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    z_start: int,
    z_end: int,
    resolution: float,
) -> tuple:
    """Create 3D grid coordinate arrays for Gaussian profile.

    Args:
        x_start, x_end: X index range
        y_start, y_end: Y index range
        z_start, z_end: Z index range
        resolution: Grid spacing

    Returns:
        Tuple (X, Y, Z) of meshgrid coordinate arrays
    """
    x_coords = (jnp.arange(x_start, x_end) + 0.5) * resolution
    y_coords = (jnp.arange(y_start, y_end) + 0.5) * resolution
    z_coords = (jnp.arange(z_start, z_end) + 0.5) * resolution
    return jnp.meshgrid(x_coords, y_coords, z_coords, indexing="ij")


@partial(jax.jit, static_argnums=(1, 2))
def interpolate_signal(
    signal_array: jnp.ndarray, signal_len: int, time: float, dt: float
) -> jnp.ndarray:
    """Differentiable signal interpolation at arbitrary time.

    Args:
        signal_array: Pre-computed signal values as JAX array
        signal_len: Length of signal array (static for JIT)
        time: Time at which to interpolate
        dt: Time step

    Returns:
        Interpolated signal value
    """
    idx_float = time / dt
    idx_low = jnp.floor(idx_float).astype(jnp.int32)
    idx_high = idx_low + 1
    frac = idx_float - jnp.floor(idx_float)

    # Clamp indices to valid range
    idx_low_safe = jnp.clip(idx_low, 0, signal_len - 1)
    idx_high_safe = jnp.clip(idx_high, 0, signal_len - 1)

    # Check if we're within valid range
    in_range = (idx_low >= 0) & (idx_low < signal_len - 1)

    interp_val = (1.0 - frac) * signal_array[idx_low_safe] + frac * signal_array[
        idx_high_safe
    ]

    return jnp.where(
        in_range,
        interp_val,
        jnp.where(idx_low == signal_len - 1, signal_array[idx_low_safe], 0.0),
    )
