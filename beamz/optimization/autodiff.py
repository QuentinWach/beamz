"""JAX-based autodifferentiation helpers for topology optimization."""

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from functools import partial

@partial(jax.jit, static_argnames=['radius'])
def masked_box_blur(values, mask, radius: int):
    """
    Apply a masked box blur using JAX convolutions.
    """
    radius = int(max(0, radius))
    if radius <= 0:
        return jnp.where(mask, values, 0.0), jnp.where(mask, 1.0, 1.0)
    
    masked_values = jnp.where(mask, values, 0.0)
    float_mask = mask.astype(float)
    
    # Create box kernel
    kernel_size = 2 * radius + 1
    kernel = jnp.ones((kernel_size, kernel_size))
    
    # Pad input to handle edges manually to match numpy 'edge' padding behavior roughly
    # For simplicity and efficiency in JAX, we use standard padding
    padded_values = jnp.pad(masked_values, radius, mode='edge')
    padded_mask = jnp.pad(float_mask, radius, mode='constant', constant_values=0.0)
    
    # Convolve
    weighted_sum = convolve2d(padded_values, kernel, mode='valid')
    weights = convolve2d(padded_mask, kernel, mode='valid')
    
    # Avoid division by zero
    weights = jnp.where(weights == 0.0, 1.0, weights)
    
    blurred = weighted_sum / weights
    blurred = jnp.where(mask, blurred, 0.0)
    weights = jnp.where(mask, weights, 1.0)
    
    return blurred, weights

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

@partial(jax.jit, static_argnames=['axis'])
def smooth_max(x, axis=None, tau=0.1):
    """
    Smooth maximum approximation: tau * log(sum(exp(x/tau)))
    Also known as LogSumExp.
    """
    return tau * jax.scipy.special.logsumexp(x / tau, axis=axis)

@partial(jax.jit, static_argnames=['axis'])
def smooth_min(x, axis=None, tau=0.1):
    """
    Smooth minimum approximation: -smooth_max(-x)
    """
    return -smooth_max(-x, axis=axis, tau=tau)

@partial(jax.jit, static_argnames=['radius'])
def grayscale_erosion(values, radius, tau=0.05):
    """
    Grayscale erosion using smooth minimum filter with a disk structuring element.
    Uses 2D shifts to implement isotropic erosion.
    """
    radius = int(max(0, radius))
    if radius <= 0: return values
    
    # 2D shift helper
    def shift_2d(arr, dy, dx):
        if dy == 0 and dx == 0: return arr
        
        # Handle y shift
        if dy > 0: arr = jnp.pad(arr[:-dy, :], ((dy,0), (0,0)), mode='edge')
        elif dy < 0: arr = jnp.pad(arr[-dy:, :], ((0,-dy), (0,0)), mode='edge')
        
        # Handle x shift
        if dx > 0: arr = jnp.pad(arr[:, :-dx], ((0,0), (dx,0)), mode='edge')
        elif dx < 0: arr = jnp.pad(arr[:, -dx:], ((0,0), (0,-dx)), mode='edge')
        
        return arr

    # Generate disk offsets
    # This loop runs at trace time since radius is static
    shifts = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy*dy + dx*dx <= radius*radius:
                shifts.append((dy, dx))
    
    # Create stack of shifted images
    stack = jnp.stack([shift_2d(values, dy, dx) for dy, dx in shifts], axis=0)
    
    # Compute smooth min over the stack
    eroded = smooth_min(stack, axis=0, tau=tau)
    
    return eroded

@partial(jax.jit, static_argnames=['radius'])
def grayscale_dilation(values, radius, tau=0.05):
    """
    Grayscale dilation using smooth maximum filter.
    Separable implementation.
    """
    radius = int(max(0, radius))
    if radius <= 0: return values
    
    # Use relationship: Dilation(f) = -Erosion(-f)
    return -grayscale_erosion(-values, radius, tau)

@partial(jax.jit, static_argnames=['radius'])
def grayscale_opening(values, radius, tau=0.05):
    """Opening: Erosion followed by Dilation."""
    return grayscale_dilation(grayscale_erosion(values, radius, tau), radius, tau)

@partial(jax.jit, static_argnames=['radius'])
def grayscale_closing(values, radius, tau=0.05):
    """Closing: Dilation followed by Erosion."""
    return grayscale_erosion(grayscale_dilation(values, radius, tau), radius, tau)

@partial(jax.jit, static_argnames=['radius', 'operation'])
def masked_morphological_filter(values, mask, radius, operation='openclose', tau=0.05):
    """
    Apply masked morphological filtering.
    
    Args:
        values: Density field
        mask: Design region mask
        radius: Filter radius in cells
        operation: 'erosion', 'dilation', 'opening', 'closing', 'openclose' (opening then closing)
        tau: Smoothness temperature for differentiable min/max
    """
    # Isolate design region values. 
    # For morphology, boundaries are important.
    # Dilation should expand into the void, but only within mask?
    # Usually in topology optimization, we only care about the result INSIDE the mask.
    # But filtering requires neighborhood.
    # We pad with 0 (void) or 1 (solid) depending on operation to avoid boundary artifacts?
    # Simpler: Just apply to whole array (assuming values outside mask are 0 or don't matter)
    # and then re-apply mask.
    
    # Ensure values outside mask don't interfere overly.
    # If we assume 0 outside, erosion works fine (0 is min).
    # Dilation might pull 0s from outside if we aren't careful? No, max pulls 1s.
    # Let's trust values are 0 outside mask typically.
    
    # Apply filter
    filtered = values
    
    if operation == 'erosion':
        filtered = grayscale_erosion(filtered, radius, tau)
    elif operation == 'dilation':
        filtered = grayscale_dilation(filtered, radius, tau)
    elif operation == 'opening':
        filtered = grayscale_opening(filtered, radius, tau)
    elif operation == 'closing':
        filtered = grayscale_closing(filtered, radius, tau)
    elif operation == 'openclose':
        # Opening then Closing is a standard noise removal filter
        filtered = grayscale_closing(grayscale_opening(filtered, radius, tau), radius, tau)
    
    # Re-apply mask constraints
    return jnp.where(mask, filtered, 0.0)

@partial(jax.jit, static_argnames=['radius', 'filter_type', 'morphology_operation'])
def transform_density(density, mask, beta, eta, radius, filter_type='blur', morphology_operation='openclose', morphology_tau=0.05):
    """
    Full density transform: Filter -> Project.
    Returns the physical density [0, 1].
    
    Args:
        filter_type: 'blur' or 'morphological'
        morphology_operation: 'opening', 'closing', 'openclose'
    """
    if filter_type == 'morphological':
        # Morphological filter
        filtered = masked_morphological_filter(density, mask, radius, morphology_operation, morphology_tau)
    else:
        # Standard box blur
        filtered, _ = masked_box_blur(density, mask, radius)
        
    projected = smoothed_heaviside(filtered, beta, eta)
    return jnp.where(mask, projected, 0.0)

@partial(jax.jit, static_argnames=['radius', 'filter_type', 'morphology_operation'])
def compute_parameter_gradient_vjp(density, grad_physical, mask, beta, eta, radius, filter_type='blur', morphology_operation='openclose', morphology_tau=0.05):
    """
    Compute gradient w.r.t. design density using VJP.
    Supports both blur and morphological filters.
    """
    # Define a wrapper for the transform to differentiate
    def transform_wrapper(d):
        return transform_density(d, mask, beta, eta, radius, filter_type, morphology_operation, morphology_tau)
    
    # Compute VJP
    _, vjp_fun = jax.vjp(transform_wrapper, density)
    grad_density = vjp_fun(grad_physical)[0]
    
    return grad_density
