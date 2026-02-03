"""JAX-based autodifferentiation helpers for topology optimization."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d


@partial(jax.jit, static_argnames=["radius"])
def generate_conic_kernel(radius: int):
    """
    Generate a 2D conic kernel (linear decay).
    w(r) = max(0, 1 - r/R)
    """
    radius = int(max(1, radius))
    kernel_size = 2 * radius + 1
    center = radius

    # Create coordinate grids
    y, x = jnp.ogrid[-radius : radius + 1, -radius : radius + 1]

    # Calculate distance from center
    dist = jnp.sqrt(x**2 + y**2)

    # Conic weights: linear decay, 0 outside radius
    weights = jnp.maximum(0.0, 1.0 - dist / radius)

    # Normalize
    weights = weights / jnp.sum(weights)

    return weights


@partial(jax.jit, static_argnames=["radius"])
def masked_conic_filter(values, mask, radius: int, fixed_structure_mask=None):
    """
    Apply conic filter with hard mask boundaries (literature-standard).
    Smoothness comes ONLY from the filter kernel radius.

    This follows standard topology optimization practice (Bendsøe, Sigmund, Hammond):
    - Hard boolean mask defines optimization region
    - Filter kernel radius controls minimum feature size
    - No soft boundary blending or protection zones
    """
    radius = int(max(0, radius))
    if radius <= 0:
        # No filtering - just apply hard mask
        return jnp.where(mask, values, 0.0), jnp.ones_like(mask)

    # Provide fixed structure context for better filter behavior
    # This gives the filter visibility into waveguide geometry without forcing values
    filter_input = values
    if fixed_structure_mask is not None:
        filter_input = jnp.where(fixed_structure_mask, 1.0, values)

    # Generate conic kernel
    kernel = generate_conic_kernel(radius)

    # Pad for convolution
    padded_values = jnp.pad(filter_input, radius, mode="edge")

    # Convolve
    filtered = convolve2d(padded_values, kernel, mode="valid")

    # Apply HARD mask (boolean) - no soft blending
    filtered = jnp.where(mask, filtered, 0.0)

    return filtered, jnp.ones_like(mask)


@jax.jit
def smoothed_heaviside(value, beta, eta):
    """
    Smoothed Heaviside projection using tanh.
    """
    beta = jnp.maximum(beta, 1e-6)
    # Use tanh projection: (tanh(beta*eta) + tanh(beta*(x-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)))
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (value - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den


@partial(jax.jit, static_argnames=["axis"])
def smooth_max(x, axis=None, tau=0.1):
    """
    Smooth maximum approximation: tau * log(sum(exp(x/tau)))
    Also known as LogSumExp.
    """
    return tau * jax.scipy.special.logsumexp(x / tau, axis=axis)


@partial(jax.jit, static_argnames=["axis"])
def smooth_min(x, axis=None, tau=0.1):
    """
    Smooth minimum approximation: -smooth_max(-x)
    """
    return -smooth_max(-x, axis=axis, tau=tau)


@partial(jax.jit, static_argnames=["radius"])
def grayscale_erosion(values, radius, tau=0.05):
    """
    Grayscale erosion using smooth minimum filter with a disk structuring element.
    Uses 2D shifts to implement isotropic erosion.
    """
    radius = int(max(0, radius))
    if radius <= 0:
        return values

    # 2D shift helper
    def shift_2d(arr, dy, dx):
        if dy == 0 and dx == 0:
            return arr

        # Handle y shift
        if dy > 0:
            arr = jnp.pad(arr[:-dy, :], ((dy, 0), (0, 0)), mode="edge")
        elif dy < 0:
            arr = jnp.pad(arr[-dy:, :], ((0, -dy), (0, 0)), mode="edge")

        # Handle x shift
        if dx > 0:
            arr = jnp.pad(arr[:, :-dx], ((0, 0), (dx, 0)), mode="edge")
        elif dx < 0:
            arr = jnp.pad(arr[:, -dx:], ((0, 0), (0, -dx)), mode="edge")

        return arr

    # Generate disk offsets
    # This loop runs at trace time since radius is static
    shifts = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                shifts.append((dy, dx))

    # Create stack of shifted images
    stack = jnp.stack([shift_2d(values, dy, dx) for dy, dx in shifts], axis=0)

    # Compute smooth min over the stack
    eroded = smooth_min(stack, axis=0, tau=tau)

    return eroded


@partial(jax.jit, static_argnames=["radius"])
def grayscale_dilation(values, radius, tau=0.05):
    """
    Grayscale dilation using smooth maximum filter.
    Separable implementation.
    """
    radius = int(max(0, radius))
    if radius <= 0:
        return values

    # Use relationship: Dilation(f) = -Erosion(-f)
    return -grayscale_erosion(-values, radius, tau)


@partial(jax.jit, static_argnames=["radius"])
def grayscale_opening(values, radius, tau=0.05):
    """Opening: Erosion followed by Dilation."""
    return grayscale_dilation(grayscale_erosion(values, radius, tau), radius, tau)


@partial(jax.jit, static_argnames=["radius"])
def grayscale_closing(values, radius, tau=0.05):
    """Closing: Dilation followed by Erosion."""
    return grayscale_erosion(grayscale_dilation(values, radius, tau), radius, tau)


@partial(jax.jit, static_argnames=["radius", "operation"])
def masked_morphological_filter(
    values, mask, radius, operation="openclose", tau=0.05, fixed_structure_mask=None
):
    """
    Apply masked morphological filtering with hard boundaries (literature-standard).

    Args:
        values: Density field
        mask: Design region mask (hard boolean boundary)
        radius: Filter radius in cells
        operation: 'erosion', 'dilation', 'opening', 'closing', 'openclose' (opening then closing)
        tau: Smoothness temperature for differentiable min/max
        fixed_structure_mask: Optional boolean mask of fixed solid structures (e.g. waveguides)
                              used to provide context for filtering without forcing values.
    """
    # Isolate design region values.
    # For morphology, boundaries are important.

    # Pad with fixed structures if provided
    # Treat fixed structures as solid (1.0) to provide context for erosion/dilation
    filter_input = values
    if fixed_structure_mask is not None:
        # We assume values is already density [0,1].
        # We override fixed structure locations with 1.0
        # NOTE: fixed_structure_mask should be a JAX array (tracer or concrete)
        filter_input = jnp.where(fixed_structure_mask, 1.0, values)

    # Apply filter to the padded/context-aware field
    filtered = filter_input

    if operation == "erosion":
        filtered = grayscale_erosion(filtered, radius, tau)
    elif operation == "dilation":
        filtered = grayscale_dilation(filtered, radius, tau)
    elif operation == "opening":
        filtered = grayscale_opening(filtered, radius, tau)
    elif operation == "closing":
        filtered = grayscale_closing(filtered, radius, tau)
    elif operation == "openclose":
        # Opening then Closing is a standard noise removal filter
        filtered = grayscale_closing(
            grayscale_opening(filtered, radius, tau), radius, tau
        )

    # Apply hard mask - no soft blending (literature standard)
    return jnp.where(mask, filtered, 0.0)


@partial(
    jax.jit,
    static_argnames=[
        "radius",
        "filter_type",
        "morphology_operation",
    ],
)
def transform_density(
    density,
    mask,
    beta,
    eta,
    radius,
    filter_type="conic",
    morphology_operation="openclose",
    morphology_tau=0.05,
    fixed_structure_mask=None,
):
    """
    Full density transform: Filter -> Project (literature-standard).
    Returns the physical density [0, 1] with hard boundary masking.

    Args:
        filter_type: 'morphological' or 'conic' (recommended)
        morphology_operation: 'opening', 'closing', 'openclose'
        fixed_structure_mask: Optional mask for fixed structures
    """
    # Apply filter (returns hard-masked result)
    if filter_type == "morphological":
        filtered = masked_morphological_filter(
            density,
            mask,
            radius,
            morphology_operation,
            morphology_tau,
            fixed_structure_mask,
        )
    elif filter_type == "conic":
        # Conic filter (for geometric constraints)
        filtered, _ = masked_conic_filter(density, mask, radius, fixed_structure_mask)
    else:
        raise ValueError(
            f"Unknown filter_type: {filter_type}. Use 'conic' or 'morphological'."
        )

    # Project
    # Note: Filters already apply hard masking, so no additional masking needed
    projected = smoothed_heaviside(filtered, beta, eta)

    return projected


@partial(
    jax.jit,
    static_argnames=[
        "radius",
        "filter_type",
        "morphology_operation",
    ],
)
def compute_parameter_gradient_vjp(
    density,
    grad_physical,
    mask,
    beta,
    eta,
    radius,
    filter_type="conic",
    morphology_operation="openclose",
    morphology_tau=0.05,
    fixed_structure_mask=None,
):
    """
    Compute gradient w.r.t. design density using VJP.
    Supports both morphological and conic filters with hard boundary masking (literature-standard).
    """

    # Define a wrapper for the transform to differentiate
    def transform_wrapper(d):
        return transform_density(
            d,
            mask,
            beta,
            eta,
            radius,
            filter_type,
            morphology_operation,
            morphology_tau,
            fixed_structure_mask,
        )

    # Compute VJP
    _, vjp_fun = jax.vjp(transform_wrapper, density)
    grad_density = vjp_fun(grad_physical)[0]

    return grad_density
