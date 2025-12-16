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

@partial(jax.jit, static_argnames=['radius'])
def transform_density(density, mask, beta, eta, radius):
    """
    Full density transform: Blur -> Project.
    Returns the physical density [0, 1].
    """
    blurred, _ = masked_box_blur(density, mask, radius)
    projected = smoothed_heaviside(blurred, beta, eta)
    return jnp.where(mask, projected, 0.0)

@partial(jax.jit, static_argnames=['radius'])
def compute_parameter_gradient_vjp(density, grad_physical, mask, beta, eta, radius):
    """
    Compute gradient w.r.t. design density using VJP.
    
    Args:
        density: Current design parameters (logits or raw density)
        grad_physical: Gradient w.r.t. physical output density (dL/d_phys)
        mask: Design region mask
        beta, eta, radius: Filter parameters
        
    Returns:
        grad_density: Gradient w.r.t. input density
    """
    # Define a wrapper for the transform to differentiate
    def transform_wrapper(d):
        return transform_density(d, mask, beta, eta, radius)
    
    # Compute VJP
    _, vjp_fun = jax.vjp(transform_wrapper, density)
    grad_density = vjp_fun(grad_physical)[0]
    
    return grad_density

