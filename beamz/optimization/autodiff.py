"""JAX-based autodifferentiation helpers for topology optimization."""

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from functools import partial

@partial(jax.jit, static_argnames=['radius'])
def generate_conic_kernel(radius: int):
    """
    Generate a 2D conic kernel (linear decay).
    w(r) = max(0, 1 - r/R)
    """
    radius = int(max(1, radius))
    kernel_size = 2 * radius + 1
    center = radius
    
    # Create coordinate grids
    y, x = jnp.ogrid[-radius:radius+1, -radius:radius+1]
    
    # Calculate distance from center
    dist = jnp.sqrt(x**2 + y**2)
    
    # Conic weights: linear decay, 0 outside radius
    weights = jnp.maximum(0.0, 1.0 - dist / radius)
    
    # Normalize
    weights = weights / jnp.sum(weights)
    
    return weights

@partial(jax.jit, static_argnames=['radius'])
def masked_conic_filter(values, mask, radius: int):
    """
    Apply a masked conic filter (linear decay).
    Used for geometric constraints (minimum feature size).
    """
    radius = int(max(0, radius))
    if radius <= 0:
        return jnp.where(mask, values, 0.0), jnp.where(mask, 1.0, 1.0)
    
    masked_values = jnp.where(mask, values, 0.0)
    float_mask = mask.astype(float)
    
    # Create conic kernel
    kernel = generate_conic_kernel(radius)
    
    # Pad input
    padded_values = jnp.pad(masked_values, radius, mode='edge')
    padded_mask = jnp.pad(float_mask, radius, mode='constant', constant_values=0.0)
    
    # Convolve
    weighted_sum = convolve2d(padded_values, kernel, mode='valid')
    weights = convolve2d(padded_mask, kernel, mode='valid')
    
    # Avoid division by zero
    weights = jnp.where(weights == 0.0, 1.0, weights)
    
    filtered = weighted_sum / weights
    filtered = jnp.where(mask, filtered, 0.0)
    
    return filtered, weights

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
def masked_morphological_filter(values, mask, radius, operation='openclose', tau=0.05, fixed_structure_mask=None):
    """
    Apply masked morphological filtering.
    
    Args:
        values: Density field
        mask: Design region mask
        radius: Filter radius in cells
        operation: 'erosion', 'dilation', 'opening', 'closing', 'openclose' (opening then closing)
        tau: Smoothness temperature for differentiable min/max
        fixed_structure_mask: Optional boolean mask of fixed solid structures (e.g. waveguides)
                              used to pad the filter input to prevent boundary erosion.
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
    # We only care about the result inside the design region
    return jnp.where(mask, filtered, 0.0)

@partial(jax.jit, static_argnames=['radius', 'filter_type', 'morphology_operation', 'post_smooth_radius'])
def transform_density(density, mask, beta, eta, radius, filter_type='blur', morphology_operation='openclose', morphology_tau=0.05, fixed_structure_mask=None, post_smooth_radius=0):
    """
    Full density transform: Filter -> Project.
    Returns the physical density [0, 1].
    
    Args:
        filter_type: 'blur', 'morphological', or 'conic'
        morphology_operation: 'opening', 'closing', 'openclose'
        fixed_structure_mask: Optional mask for fixed structures (morphological filter only)
        post_smooth_radius: Optional blur after morphology or filter
    """
    if filter_type == 'morphological':
        # Morphological filter
        filtered = masked_morphological_filter(density, mask, radius, morphology_operation, morphology_tau, fixed_structure_mask)
    elif filter_type == 'conic':
        # Conic filter (for geometric constraints)
        filtered, _ = masked_conic_filter(density, mask, radius)
    else:
        # Standard box blur
        filtered, _ = masked_box_blur(density, mask, radius)
    
    # Universal Post-Smoothing: Apply a small blur to smooth edges/artifacts
    if post_smooth_radius > 0:
        smoothed, _ = masked_box_blur(filtered, mask, post_smooth_radius)
        filtered = smoothed
        
    projected = smoothed_heaviside(filtered, beta, eta)
    return jnp.where(mask, projected, 0.0)

@partial(jax.jit, static_argnames=['radius', 'filter_type', 'morphology_operation', 'post_smooth_radius'])
def compute_parameter_gradient_vjp(density, grad_physical, mask, beta, eta, radius, filter_type='blur', morphology_operation='openclose', morphology_tau=0.05, fixed_structure_mask=None, post_smooth_radius=0):
    """
    Compute gradient w.r.t. design density using VJP.
    Supports both blur and morphological filters.
    """
    # Define a wrapper for the transform to differentiate
    def transform_wrapper(d):
        return transform_density(d, mask, beta, eta, radius, filter_type, morphology_operation, morphology_tau, fixed_structure_mask, post_smooth_radius)
    
    # Compute VJP
    _, vjp_fun = jax.vjp(transform_wrapper, density)
    grad_density = vjp_fun(grad_physical)[0]
    
    return grad_density
