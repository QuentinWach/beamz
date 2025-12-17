"""Topology optimization manager and helpers."""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict
from beamz.const import LIGHT_SPEED

from .autodiff import (
    transform_density, compute_parameter_gradient_vjp, 
    harmonic_erosion, harmonic_dilation, 
    constraint_min_linewidth, constraint_min_area
)
from .mma import MMAState, init_mma, mma_update

# Defer imports to avoid circular dependencies if any, 
# or import at top level if safe. design shouldn't depend on optimization.
from beamz.design.core import Design
from beamz.design.materials import Material
from beamz.design.meshing import RegularGrid

class TopologyManager:
    """
    High-level manager for robust topology optimization with MMA.
    
    Handles:
    - Density parameter storage
    - Physical density transformation (JAX-based) for Eroded, Nominal, Dilated variants
    - Gradient backpropagation (JAX-based)
    - MMA Optimizer stepping
    - Geometric Constraints
    """
    
    def __init__(
        self,
        design,
        region_mask: np.ndarray,
        optimizer: str = "MMA", # Default now MMA
        learning_rate: float = 0.1, # Not used for MMA directly, but for compatibility
        filter_radius: float = 0.0,
        simple_smooth_radius: float = 0.0,
        projection_eta: float = 0.5,
        beta_schedule: tuple[float, float] = (1.0, 32.0), # Updated default to 32
        eps_min: float = 1.0,
        eps_max: float = 12.0,
        resolution: float = None,
        filter_type: str = 'conic',
        morphology_operation: str = 'openclose',
        **kwargs
    ):
        """
        Args:
            filter_radius: Primary filter radius in physical units (e.g. microns).
            simple_smooth_radius: Optional post-filter smoothing radius in physical units.
            filter_type: 'conic' (geometric constraints), 'morphological', or 'blur'.
        """
        self.design = design
        self.mask = region_mask.astype(bool)
        self.optimizer_type = optimizer
        
        # Initialize MMA state
        if optimizer == "MMA":
            import jax.numpy as jnp
            # Initial density
            init_density = jnp.zeros(np.sum(self.mask), dtype=float) + 0.5
            self.mma_state = init_mma(init_density)
        else:
            # Fallback to Optax (not recommended for robust mode)
            try:
                import optax
            except ImportError:
                raise ImportError("optax is required for optimization. Install with: pip install optax")
            if optimizer.lower() == "adam":
                self.optax_optimizer = optax.adam(learning_rate=learning_rate)
            elif optimizer.lower() == "sgd":
                self.optax_optimizer = optax.sgd(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unknown optimizer '{optimizer}'. Supported: 'MMA', 'adam', 'sgd'")
            self._opt_state = None
        
        # Parameters
        self.filter_radius = filter_radius
        self.simple_smooth_radius = simple_smooth_radius
        self.projection_eta = projection_eta
        self.beta_start, self.beta_end = beta_schedule
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.resolution = resolution or getattr(design.rasterize(resolution=0.1), "dx") 
        
        # Filter settings
        self.filter_type = filter_type
        self.morphology_operation = morphology_operation
        self.morphology_smooth_tau = kwargs.get('morphology_smooth_tau', 0.01)
        
        # Convert filter radius to cells
        self.filter_radius_cells = int(round(filter_radius / self.resolution)) if self.resolution else 0
        self.smooth_radius_cells = int(round(simple_smooth_radius / self.resolution)) if self.resolution else 0
        
        # Initialize density parameters (0.5 inside mask)
        self.design_density = np.zeros_like(self.mask, dtype=float)
        self.design_density[self.mask] = 0.5
        
        # Store base grid for fixed structure detection
        self.base_grid = design.rasterize(self.resolution)
        self.fixed_structure_mask = get_fixed_structure_mask(
            self.base_grid, self.eps_min, self.eps_max, self.mask
        )
        
        # History
        self.objective_history = []
    
    def get_current_beta(self, step: int, total_steps: int) -> float:
        """Calculate projection beta for current step."""
        if total_steps <= 1: return self.beta_end
        frac = step / (total_steps - 1)
        # Typically Hammond ramps beta: 8 -> 16 -> 32
        # We use linear ramp for now, or piecewise
        return self.beta_start + frac * (self.beta_end - self.beta_start)
    
    def get_physical_density(self, beta: float, variant: str = 'nominal') -> np.ndarray:
        """
        Compute physical density from design parameters using JAX transform.
        Supports 'nominal', 'eroded', 'dilated' variants.
        """
        import jax.numpy as jnp
        d_jax = jnp.array(self.design_density)
        m_jax = jnp.array(self.mask)
        fixed_jax = jnp.array(self.fixed_structure_mask) if self.fixed_structure_mask is not None else None
        
        # Select filter type based on variant
        # Hammond uses Harmonic Erosion/Dilation for robust variants
        current_filter_type = self.filter_type
        
        if variant == 'eroded':
            current_filter_type = 'harmonic_erosion'
        elif variant == 'dilated':
            current_filter_type = 'harmonic_dilation'
        
        p_jax = transform_density(
            d_jax, m_jax, 
            beta, self.projection_eta, self.filter_radius_cells,
            filter_type=current_filter_type,
            morphology_operation=self.morphology_operation,
            morphology_tau=self.morphology_smooth_tau,
            fixed_structure_mask=fixed_jax,
            post_smooth_radius=self.smooth_radius_cells
        )
        return np.array(p_jax)
    
    def update_design(self, step: int, total_steps: int, variant: str = 'nominal') -> tuple[float, np.ndarray]:
        """
        Update the design's material grid based on current parameters and variant.
        Returns (current_beta, physical_density).
        """
        beta = self.get_current_beta(step, total_steps)
        physical_density = self.get_physical_density(beta, variant)
        return beta, physical_density
    
    def apply_gradient_robust(self, 
                              gradients: Dict[str, np.ndarray], 
                              objectives: Dict[str, float],
                              beta: float,
                              constraints_val: np.ndarray = None,
                              constraints_grad: np.ndarray = None):
        """
        Apply robust optimization update using MMA.
        Minimax objective: min max(f_nom, f_ero, f_dil)
        
        Args:
            gradients: Dict of 'nominal', 'eroded', 'dilated' gradients (dJ/dEps).
            objectives: Dict of objective values (lower is better).
            beta: Current projection sharpness.
            constraints_val: Array of constraint values (g <= 0).
            constraints_grad: Gradient of constraints w.r.t parameters.
        """
        import jax.numpy as jnp
        from .mma import mma_update
        
        # 1. Backpropagate gradients to design parameters for each variant
        grad_params = {}
        fixed_jax = jnp.array(self.fixed_structure_mask) if self.fixed_structure_mask is not None else None
        d_jax = jnp.array(self.design_density)
        m_jax = jnp.array(self.mask)
        
        variants = ['nominal', 'eroded', 'dilated']
        # Filter valid variants present in input
        active_variants = [v for v in variants if v in gradients]
        
        for v in active_variants:
            # dJ/dPhysical = dJ/dEps * (eps_max - eps_min)
            # Since we maximize transmission usually, input gradients are likely for maximization.
            # MMA minimizes. So objective = -Transmission.
            # Gradient of objective = - Gradient of Transmission.
            # grad_eps passed in is usually d(Transmission)/dEps.
            # So grad_obj_phys = - grad_eps * delta
            
            grad_physical = -gradients[v] * (self.eps_max - self.eps_min)
            
            # Determine filter type for backprop
            ftype = self.filter_type
            if v == 'eroded': ftype = 'harmonic_erosion'
            elif v == 'dilated': ftype = 'harmonic_dilation'
            
            # JAX Backprop
            grad_p = compute_parameter_gradient_vjp(
                d_jax,
                jnp.array(grad_physical),
                m_jax,
                beta,
                self.projection_eta,
                self.filter_radius_cells,
                filter_type=ftype,
                morphology_operation=self.morphology_operation,
                morphology_tau=self.morphology_smooth_tau,
                fixed_structure_mask=fixed_jax,
                post_smooth_radius=self.smooth_radius_cells
            )
            grad_params[v] = np.array(grad_p)
            
        # 2. Compute Minimax Gradient (Subgradient)
        # For min max(f_i), the gradient is the gradient of the active (max) function.
        # If multiple are active, it's a convex combination (handled by MMA if formulated as bound formulation).
        # Standard approach: Introduce variable z. Minimize z. Subject to f_i(x) - z <= 0.
        # MMA handles this naturally if we add f_i(x) as constraints and z as variable.
        # OR simpler heuristic: Use Softmax or just the max one if MMA formulation doesn't support Minimax directly easily.
        # Hammond uses epigraph formulation: min t subject to f_i <= t.
        # Our MMA implementation currently expects a single objective gradient.
        # Let's use LogSumExp (Softmax) to aggregate:
        # f_smooth = alpha * log(sum(exp(f_i/alpha)))
        # grad = sum( w_i * grad_i ) where w_i = exp(f_i/alpha) / sum(...)
        
        # Convert objectives to minimization form (negative transmission)
        # Objectives passed in are likely Transmission (positive).
        # We minimize -Transmission.
        f_vals = np.array([-objectives[v] for v in active_variants])
        
        # Softmax aggregation
        alpha = 5.0 # Smoothing parameter (Increased to avoid overflow)
        # Shift for stability
        f_shift = f_vals - np.min(f_vals)
        weights = np.exp(f_shift / alpha)
        weights /= np.sum(weights)
        
        # Aggregate gradient (w.r.t parameters)
        # Note: grad_params are already gradients of the MINIMIZATION objective (-Trans)
        grad_agg = np.zeros_like(d_jax)
        for i, v in enumerate(active_variants):
            grad_agg += weights[i] * grad_params[v] # Removed [self.mask] to fix broadcasting
            
        # 3. Handle Geometric Constraints
        # We assume constraints are passed as values and gradients w.r.t parameters
        # If not provided, we calculate internal penalty gradients (like volume) here or assume passed.
        # For this implementation, we assume user passes 'constraints_val' and 'constraints_grad' if any.
        
        constraints_arr = jnp.array(constraints_val) if constraints_val is not None else jnp.array([])
        grad_g_arr = jnp.array(constraints_grad) if constraints_grad is not None else jnp.zeros((0, grad_agg.shape[0]))
        
        # 4. MMA Update
        x_current = jnp.array(self.design_density[self.mask])
        grad_f_current = jnp.array(grad_agg[self.mask])
        
        if self.optimizer_type == "MMA":
            x_new, self.mma_state = mma_update(
                self.mma_state,
                x_current,
                grad_f_current,
                constraints_arr,
                grad_g_arr
            )
            update = np.array(x_new - x_current)
            self.design_density[self.mask] = np.array(x_new)
        else:
            # Fallback Optax
            import jax.numpy as jnp
            if self._opt_state is None:
                self._opt_state = self.optax_optimizer.init(x_current)
            updates, self._opt_state = self.optax_optimizer.update(grad_f_current, self._opt_state)
            update = np.array(updates)
            self.design_density[self.mask] += update
            self.design_density = np.clip(self.design_density, 0.0, 1.0)
            
        return np.max(np.abs(update))

# ... [Keep existing helper functions like compute_overlap_gradient, create_optimization_mask, get_fixed_structure_mask] ...
def compute_overlap_gradient(forward_fields_history, adjoint_fields_history, field_key="Ez"):
    """
    Compute the gradient of the overlap integral with respect to epsilon.
    Gradient = Re(E_fwd * E_adj) integrated over time.
    """
    grad = np.zeros_like(forward_fields_history[0], dtype=float)
    n_steps = min(len(forward_fields_history), len(adjoint_fields_history))
    for i in range(n_steps):
        grad += forward_fields_history[i] * adjoint_fields_history[n_steps - 1 - i]
    return grad

def create_optimization_mask(grid, region_structure):
    # Create temp design to rasterize mask exactly as grid does
    temp_design = Design(width=grid.width, height=grid.height, 
                         material=Material(permittivity=1.0))
    if hasattr(region_structure, 'copy'):
        struct_copy = region_structure.copy()
    else:
        struct_copy = region_structure
    struct_copy.material = Material(permittivity=2.0)
    temp_design.add(struct_copy)
    temp_grid = RegularGrid(temp_design, resolution=grid.dx)
    mask = temp_grid.permittivity > 1.001
    return mask

def get_fixed_structure_mask(grid, eps_min, eps_max, design_mask):
    threshold = eps_min + 0.9 * (eps_max - eps_min)
    high_eps = grid.permittivity >= threshold
    fixed_structures = high_eps & (~design_mask)
    return fixed_structures
