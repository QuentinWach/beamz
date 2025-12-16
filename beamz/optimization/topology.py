"""Topology optimization manager and helpers."""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from beamz.const import LIGHT_SPEED

from .core import Optimizer
from .autodiff import transform_density, compute_parameter_gradient_vjp

class TopologyManager:
    """
    High-level manager for topology optimization.
    
    Handles:
    - Density parameter storage
    - Physical density transformation (JAX-based)
    - Gradient backpropagation (JAX-based)
    - Optimizer stepping
    - Material grid updates
    """
    
    def __init__(
        self,
        design,
        region_mask: np.ndarray,
        optimizer: str = "Adam",
        learning_rate: float = 0.1,
        filter_radius: float = 0.0,
        projection_eta: float = 0.5,
        beta_schedule: tuple[float, float] = (1.0, 20.0),
        eps_min: float = 1.0,
        eps_max: float = 12.0,
        resolution: float = None
    ):
        self.design = design
        self.mask = region_mask.astype(bool)
        self.optimizer = Optimizer(method=optimizer, learning_rate=learning_rate)
        
        # Parameters
        self.filter_radius = filter_radius
        self.projection_eta = projection_eta
        self.beta_start, self.beta_end = beta_schedule
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.resolution = resolution or getattr(design.rasterize(resolution=0.1), "dx") # Fallback resolution check?
        
        # Convert filter radius to cells
        self.filter_radius_cells = int(round(filter_radius / self.resolution)) if self.resolution else 0
        
        # Initialize density parameters (0.5 inside mask)
        self.design_density = np.zeros_like(self.mask, dtype=float)
        self.design_density[self.mask] = 0.5
        
        # History
        self.objective_history = []
    
    def get_current_beta(self, step: int, total_steps: int) -> float:
        """Calculate projection beta for current step."""
        if total_steps <= 1: return self.beta_end
        frac = step / (total_steps - 1)
        return self.beta_start + frac * (self.beta_end - self.beta_start)
    
    def get_physical_density(self, beta: float) -> np.ndarray:
        """Compute physical density from design parameters using JAX transform."""
        import jax.numpy as jnp
        d_jax = jnp.array(self.design_density)
        m_jax = jnp.array(self.mask)
        
        p_jax = transform_density(
            d_jax, m_jax, 
            beta, self.projection_eta, self.filter_radius_cells
        )
        return np.array(p_jax)
    
    def update_design(self, step: int, total_steps: int) -> tuple[float, np.ndarray]:
        """
        Update the design's material grid based on current parameters.
        Returns (current_beta, physical_density).
        """
        beta = self.get_current_beta(step, total_steps)
        physical_density = self.get_physical_density(beta)
        
        # Update grid permittivity
        # Assuming we can access the grid from design or it's passed separately?
        # Ideally the design object has a rasterized grid we modify.
        # We need the base permittivity to mix with.
        # For now, we assume the user handles the base/mixing or we do it if we have the grid.
        
        return beta, physical_density
    
    def apply_gradient(self, grad_eps: np.ndarray, beta: float):
        """
        Apply gradient update:
        1. Convert dJ/dEps -> dJ/dPhysical
        2. Backprop dJ/dPhysical -> dJ/dParams (using JAX)
        3. Optimizer step
        """
        import jax.numpy as jnp
        
        # dJ/dPhysical = dJ/dEps * (eps_max - eps_min)
        grad_physical = grad_eps * (self.eps_max - self.eps_min)
        
        # JAX Backprop
        grad_param_jax = compute_parameter_gradient_vjp(
            jnp.array(self.design_density),
            jnp.array(grad_physical),
            jnp.array(self.mask),
            beta,
            self.projection_eta,
            self.filter_radius_cells
        )
        grad_param = np.array(grad_param_jax)
        
        # Optimizer step (maximize objective -> ascent -> negative grad for minimizer)
        # Assuming we want to MAXIMIZE the objective (e.g. overlap), we pass -grad
        update = self.optimizer.step(-grad_param)
        
        # Apply update
        self.design_density[self.mask] += update[self.mask]
        self.design_density = np.clip(self.design_density, 0.0, 1.0)
        
        return np.max(np.abs(update))


def compute_overlap_gradient(forward_fields_history, adjoint_fields_history, field_key="Ez"):
    """
    Compute the gradient of the overlap integral with respect to epsilon.
    Gradient = Re(E_fwd * E_adj) integrated over time.
    """
    grad = np.zeros_like(forward_fields_history[0], dtype=float)
    
    # We assume histories are aligned in reverse time or we iterate appropriately.
    # Usually: Forward t=0..T, Adjoint t=T..0 (simulated as 0..T in adjoint run)
    # So E_fwd[i] pairs with E_adj[N-1-i] typically, OR
    # if adjoint source was injected backwards in time, E_adj[i] corresponds to E_fwd[N-1-i].
    # In standard FDTD adjoint:
    # dJ/deps ~ integral(E_fwd(t) . E_adj(T-t) dt)
    # If we saved frames sequentially:
    # fwd: 0, 1, ..., N
    # adj: 0, 1, ..., N (where adj step 0 corresponds to real time T)
    # So we pair fwd[i] with adj[N-1-i] ideally?
    # Or if we ran adjoint simulation from t=0 to T, the field at step k represents field at T-k.
    # So fwd[k] and adj[k] should NOT be multiplied?
    # Actually, Convolution theorem -> Integral E(t) E_adj(T-t) dt.
    # If sim_adj[k] is field at step k of adjoint simulation, it corresponds to time T - k*dt.
    # sim_fwd[k] is field at step k*dt.
    # So we want to multiply sim_fwd[k] * sim_adj[k] ?? 
    # Let's check the math. 
    # Gradient = \int E(t) \cdot E_adj(T-t) dt.
    # If adj[k] is the field at step k of the adjoint run, and the adjoint run simulates the time-reversed problem,
    # then adj[k] IS effectively E_adj(T - t_k).
    # So we multiply fwd[k] * adj[k].
    
    n_steps = min(len(forward_fields_history), len(adjoint_fields_history))
    
    for i in range(n_steps):
        # Taking real part of dot product (elementwise multiply for scalar eps)
        # For complex fields (if frequency domain) it's different, but FDTD is time domain real fields.
        # If fields are complex (e.g. envelopes), use Re(E . E).
        # Assuming real fields here from FDTD.
        # FDTD Adjoint sensitivity requires convolution (time-reversal alignment)
        # Forward field at t corresponds to Adjoint field at T-t
        # So we pair fwd[i] with adj[n_steps - 1 - i]
        grad += forward_fields_history[i] * adjoint_fields_history[n_steps - 1 - i]
        
    return grad

def create_optimization_mask(grid, region_structure):
    """Helper to create a boolean mask from a structure on a grid."""
    dx, dy = grid.dx, grid.dy
    mask = np.zeros(grid.permittivity.shape, dtype=bool)
    ys = (np.arange(mask.shape[0]) + 0.5) * dy
    xs = (np.arange(mask.shape[1]) + 0.5) * dx
    
    # Use bounding box for speed
    minx, miny, _, maxx, maxy, _ = region_structure.get_bounding_box()
    
    # This simple rect check works for Rectangles. 
    # For general polygons, we might need rasterization logic.
    mask[(ys[:,None] >= miny) & (ys[:,None] <= maxy) & 
         (xs[None,:] >= minx) & (xs[None,:] <= maxx)] = True
         
    return mask
