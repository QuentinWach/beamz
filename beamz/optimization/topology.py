"""Topology optimization manager and helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from beamz.const import LIGHT_SPEED

# Defer imports to avoid circular dependencies if any,
# or import at top level if safe. design shouldn't depend on optimization.
from beamz.design.core import Design
from beamz.design.materials import Material
from beamz.design.meshing import RegularGrid

from .autodiff import compute_parameter_gradient_vjp, transform_density


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
        resolution: float = None,
        filter_type: str = "conic",  # 'conic' or 'morphological'
        morphology_operation: str = "openclose",  # 'opening', 'closing', 'openclose'
        **kwargs,
    ):
        """
        Args:
            filter_radius: Filter radius in physical units (e.g. microns).
                           Controls minimum feature size AND boundary smoothness.
                           Recommended: 0.25-0.35 µm for smooth, rounded structures.
            filter_type: 'conic' (recommended, geometric constraints) or 'morphological'.
            morphology_operation: 'opening', 'closing', or 'openclose' (for morphological filter).
        """
        self.design = design
        self.mask = region_mask.astype(bool)

        # Setup optimizer using optax (JAX-native)
        try:
            import optax
        except ImportError:
            raise ImportError(
                "optax is required for optimization. Install with: pip install optax"
            )

        if optimizer.lower() == "adam":
            self.optax_optimizer = optax.adam(learning_rate=learning_rate)
        elif optimizer.lower() == "sgd":
            self.optax_optimizer = optax.sgd(learning_rate=learning_rate)
        else:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Supported: 'adam', 'sgd'"
            )

        # Initialize optimizer state (will be created on first use)
        self._opt_state = None

        # Parameters
        self.filter_radius = filter_radius
        self.projection_eta = projection_eta
        self.beta_start, self.beta_end = beta_schedule
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.resolution = resolution or getattr(
            design.rasterize(resolution=0.1), "dx"
        )  # Fallback resolution check?

        # Filter settings
        self.filter_type = filter_type
        self.morphology_operation = morphology_operation
        self.morphology_smooth_tau = kwargs.get(
            "morphology_smooth_tau", 0.01
        )  # Default good value

        # Convert filter radius to cells
        self.filter_radius_cells = (
            int(round(filter_radius / self.resolution)) if self.resolution else 0
        )

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
        if total_steps <= 1:
            return self.beta_end
        frac = step / (total_steps - 1)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def get_physical_density(self, beta: float) -> np.ndarray:
        """Compute physical density from design parameters using JAX transform."""
        import jax.numpy as jnp

        d_jax = jnp.array(self.design_density)
        m_jax = jnp.array(self.mask)
        fixed_jax = (
            jnp.array(self.fixed_structure_mask)
            if self.fixed_structure_mask is not None
            else None
        )

        p_jax = transform_density(
            d_jax,
            m_jax,
            beta,
            self.projection_eta,
            self.filter_radius_cells,
            filter_type=self.filter_type,
            morphology_operation=self.morphology_operation,
            morphology_tau=self.morphology_smooth_tau,
            fixed_structure_mask=fixed_jax,
        )
        return np.array(p_jax)

    def update_design(self, step: int, total_steps: int) -> tuple[float, np.ndarray]:
        """
        Update the design's material grid based on current parameters.
        Returns (current_beta, physical_density).
        """
        beta = self.get_current_beta(step, total_steps)
        physical_density = self.get_physical_density(beta)

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

        fixed_jax = (
            jnp.array(self.fixed_structure_mask)
            if self.fixed_structure_mask is not None
            else None
        )

        # JAX Backprop
        grad_param_jax = compute_parameter_gradient_vjp(
            jnp.array(self.design_density),
            jnp.array(grad_physical),
            jnp.array(self.mask),
            beta,
            self.projection_eta,
            self.filter_radius_cells,
            filter_type=self.filter_type,
            morphology_operation=self.morphology_operation,
            morphology_tau=self.morphology_smooth_tau,
            fixed_structure_mask=fixed_jax,
        )
        grad_param = np.array(grad_param_jax)

        # Optimizer step (maximize objective -> ascent -> negative grad for minimizer)
        # Convert to JAX array for optax
        import jax.numpy as jnp

        grad_jax = jnp.array(-grad_param)  # Negative because we want to maximize

        # Initialize optimizer state on first call
        if self._opt_state is None:
            params_init = jnp.array(self.design_density)
            self._opt_state = self.optax_optimizer.init(params_init)

        # Compute updates
        updates, self._opt_state = self.optax_optimizer.update(
            grad_jax, self._opt_state
        )
        update = np.array(updates)

        # Apply update
        self.design_density[self.mask] += update[self.mask]
        self.design_density = np.clip(self.design_density, 0.0, 1.0)

        return np.max(np.abs(update))


def compute_overlap_gradient(
    forward_fields_history, adjoint_fields_history, field_key="Ez"
):
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
    """
    Helper to create a boolean mask from a structure on a grid.
    Uses rasterization to ensure exact alignment with how structures are mapped to the grid.
    """
    # Create temp design to rasterize mask exactly as grid does
    temp_design = Design(
        width=grid.width, height=grid.height, material=Material(permittivity=1.0)
    )

    # Copy structure to avoid modifying original
    if hasattr(region_structure, "copy"):
        struct_copy = region_structure.copy()
    else:
        # Fallback if no copy method, reuse (risky if material modified, but we set it)
        struct_copy = region_structure

    # Set to a distinct permittivity to detect it
    struct_copy.material = Material(permittivity=2.0)
    temp_design.add(struct_copy)

    # Rasterize
    temp_grid = RegularGrid(temp_design, resolution=grid.dx)

    # Mask is where permittivity > background
    # Use a safe threshold to include any partial fill
    mask = temp_grid.permittivity > 1.001

    return mask


def get_fixed_structure_mask(grid, eps_min, eps_max, design_mask):
    """
    Identify fixed structures (waveguides) from the base permittivity grid.
    Returns boolean mask where fixed solid material exists outside design region.
    """
    # Threshold for "solid" material. E.g. > 90% of core-clad difference
    threshold = eps_min + 0.9 * (eps_max - eps_min)

    # High permittivity regions
    high_eps = grid.permittivity >= threshold

    # Exclude design region (we only care about fixed structures outside)
    fixed_structures = high_eps & (~design_mask)

    return fixed_structures
