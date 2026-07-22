"""Immutable topology-optimization configuration, state, and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from beamz._cache_tokens import cache_token

# Defer imports to avoid circular dependencies if any,
# or import at top level if safe. design shouldn't depend on optimization.
from beamz.design.core import Design
from beamz.design.materials import Material
from beamz.design.meshing import RegularGrid
from beamz.design.structures import Structure

from .autodiff import compute_parameter_gradient_vjp, transform_density
from .projections import validate_projection_options


def _readonly_array(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, eq=False)
class TopologyState:
    """Store explicit evolving values for a topology-optimization run.

    Parameters
    ----------
    density : array-like
        Latent design density in ``(y, x)`` array order.
    optimizer_state : object, optional
        Backend-specific immutable optimizer state.
    objective_history : tuple of float, optional
        Objective values from completed optimization steps.
    """

    density: np.ndarray
    optimizer_state: Any = field(default=None, repr=False, compare=False)
    objective_history: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "density", _readonly_array(self.density, float))
        object.__setattr__(
            self, "objective_history", tuple(float(v) for v in self.objective_history)
        )

    def with_objective(self, value: float) -> "TopologyState":
        """Return state with one objective value appended to its history.

        Parameters
        ----------
        value : float
            Scalar objective value for the completed iteration.
        """
        return replace(self, objective_history=(*self.objective_history, float(value)))

    def canonical_spec(self):
        """Return evolving values included in state cache identity."""
        return self.density, self.objective_history

    def __eq__(self, other):
        if not isinstance(other, TopologyState):
            return NotImplemented
        return cache_token(self.canonical_spec()) == cache_token(other.canonical_spec())

    def __hash__(self):
        return hash(cache_token(self.canonical_spec()))


@dataclass(frozen=True, slots=True, eq=False)
class TopologySpec:
    """Configure immutable density-based topology optimization.

    Parameters
    ----------
    design : Design
        Base design containing fixed geometry and background material.
    region_mask : array-like of bool
        Cells that optimization may modify, in ``(y, x)`` order.
    optimizer : {"adam", "sgd"}, default="adam"
        Gradient-ascent optimizer.
    learning_rate : float, default=0.1
        Optimizer step size in latent-density units.
    filter_radius : float, default=0
        Physical density-filter radius in metres.
    projection_eta : float, default=0.5
        Threshold of the smooth Heaviside projection.
    projection_type : {"heaviside", "ssp"}, default="heaviside"
        Density projection applied after filtering.
    ssp_smoothing_radius : float, default=0.55
        Hammond SSP smoothing radius in density-grid cell units.
    beta_schedule : tuple, default=(1.0, 20.0)
        Initial and final projection sharpness.
    eps_min : float, default=1.0
        Relative permittivity represented by zero physical density.
    eps_max : float, default=12.0
        Relative permittivity represented by unit physical density.
    resolution : float
        Design-grid cell size in metres; required explicitly.
    filter_type : str, default="conic"
        Spatial density-filter implementation.
    morphology_operation : str, default="openclose"
        Optional smooth morphology applied after filtering.
    morphology_smooth_tau : float, default=0.01
        Smooth min/max temperature used by morphology.

    Notes
    -----
    The region and fixed-structure masks are copied and made read-only. Evolving
    density and optimizer values live in :class:`TopologyState`.
    """

    design: Design
    region_mask: np.ndarray
    optimizer: str = "adam"
    learning_rate: float = 0.1
    filter_radius: float = 0.0
    projection_eta: float = 0.5
    projection_type: str = "heaviside"
    ssp_smoothing_radius: float = 0.55
    beta_schedule: tuple[float, float] = (1.0, 20.0)
    eps_min: float = 1.0
    eps_max: float = 12.0
    resolution: float | None = None
    filter_type: str = "conic"
    morphology_operation: str = "openclose"
    morphology_smooth_tau: float = 0.01
    fixed_structure_mask: np.ndarray = field(init=False, repr=False)
    filter_radius_cells: int = field(init=False)

    def __post_init__(self):
        if self.resolution is None:
            raise ValueError("TopologySpec requires explicit `resolution` (meters).")
        resolution = float(self.resolution)
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution!r}")
        optimizer = str(self.optimizer).lower()
        if optimizer not in {"adam", "sgd"}:
            raise ValueError(
                f"Unknown optimizer {self.optimizer!r}. Supported: 'adam', 'sgd'"
            )
        projection_type = str(self.projection_type).lower()
        ssp_smoothing_radius = float(self.ssp_smoothing_radius)
        validate_projection_options(projection_type, ssp_smoothing_radius)
        mask = _readonly_array(self.region_mask, bool)
        base_grid = self.design.rasterize(resolution)
        fixed = get_fixed_structure_mask(
            base_grid, float(self.eps_min), float(self.eps_max), mask
        )
        object.__setattr__(self, "region_mask", mask)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "projection_type", projection_type)
        object.__setattr__(self, "ssp_smoothing_radius", ssp_smoothing_radius)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(
            self, "beta_schedule", tuple(float(v) for v in self.beta_schedule)
        )
        object.__setattr__(self, "fixed_structure_mask", _readonly_array(fixed, bool))
        object.__setattr__(
            self,
            "filter_radius_cells",
            int(round(float(self.filter_radius) / resolution)),
        )

    def initial_state(self, density: np.ndarray | None = None) -> TopologyState:
        """Create independent optimization state for this specification.

        Parameters
        ----------
        density : array-like, optional
            Initial latent density with the design-grid shape. By default,
            optimizable cells start at 0.5 and other cells at zero.
        """
        if density is None:
            density = np.zeros(self.region_mask.shape, dtype=float)
            density[self.region_mask] = 0.5
        elif np.shape(density) != self.region_mask.shape:
            raise ValueError(
                f"density shape {np.shape(density)} does not match mask "
                f"shape {self.region_mask.shape}"
            )
        return TopologyState(density)

    def beta(self, step: int, total_steps: int) -> float:
        """Return the projection strength for a continuation step.

        Parameters
        ----------
        step : int
            Zero-based optimization step.
        total_steps : int
            Total number of continuation steps.
        """
        start, end = self.beta_schedule
        if total_steps <= 1:
            return end
        return start + step / (total_steps - 1) * (end - start)

    def physical_density(self, state: TopologyState, beta: float) -> np.ndarray:
        """Transform latent density into a detached physical density.

        Parameters
        ----------
        state : TopologyState
            Latent design density and optimizer state.
        beta : float
            Projection strength.
        """
        import jax.numpy as jnp

        density = transform_density(
            jnp.asarray(state.density),
            jnp.asarray(self.region_mask),
            beta,
            self.projection_eta,
            self.filter_radius_cells,
            filter_type=self.filter_type,
            morphology_operation=self.morphology_operation,
            morphology_tau=self.morphology_smooth_tau,
            fixed_structure_mask=jnp.asarray(self.fixed_structure_mask),
            projection_type=self.projection_type,
            ssp_smoothing_radius=self.ssp_smoothing_radius,
        )
        return _readonly_array(density, float)

    def density_for_step(
        self, state: TopologyState, step: int, total_steps: int
    ) -> tuple[float, np.ndarray]:
        """Return continuation beta and physical density for an iteration.

        Parameters
        ----------
        state : TopologyState
            Current latent design state.
        step : int
            Zero-based optimization step.
        total_steps : int
            Total number of continuation steps.
        """
        beta = self.beta(step, total_steps)
        return beta, self.physical_density(state, beta)

    def apply_gradient(
        self, state: TopologyState, grad_eps: np.ndarray, beta: float
    ) -> tuple[TopologyState, float]:
        """Return state advanced by one gradient-ascent optimizer step.

        Parameters
        ----------
        state : TopologyState
            Current latent density and backend optimizer state.
        grad_eps : array-like
            Objective gradient with respect to field-grid permittivity.
        beta : float
            Projection strength used for the current physical density.
        """
        import jax.numpy as jnp

        grad_eps = self.gradient_to_design_grid(grad_eps)
        grad_physical = grad_eps * (self.eps_max - self.eps_min)
        grad_param = compute_parameter_gradient_vjp(
            jnp.asarray(state.density),
            jnp.asarray(grad_physical),
            jnp.asarray(self.region_mask),
            beta,
            self.projection_eta,
            self.filter_radius_cells,
            filter_type=self.filter_type,
            morphology_operation=self.morphology_operation,
            morphology_tau=self.morphology_smooth_tau,
            fixed_structure_mask=jnp.asarray(self.fixed_structure_mask),
            projection_type=self.projection_type,
            ssp_smoothing_radius=self.ssp_smoothing_radius,
        )
        optimizer = self._optimizer()
        opt_state = state.optimizer_state
        if opt_state is None:
            opt_state = optimizer.init(jnp.asarray(state.density))
        updates, opt_state = optimizer.update(-grad_param, opt_state)
        update = np.asarray(updates)
        density = np.asarray(state.density).copy()
        density[self.region_mask] += update[self.region_mask]
        density = np.clip(density, 0.0, 1.0)
        return replace(state, density=density, optimizer_state=opt_state), float(
            np.max(np.abs(update))
        )

    def gradient_to_design_grid(self, grad_eps: np.ndarray) -> np.ndarray:
        """Collocate a field-grid permittivity gradient onto design cells.

        Parameters
        ----------
        grad_eps : array-like
            Permittivity gradient on the Yee/material field grid.
        """
        return fold_high_side_yee_padding_to_shape(grad_eps, self.region_mask.shape)

    def _optimizer(self):
        try:
            import optax
        except ImportError as exc:
            raise ImportError(
                "optax is required for optimization. Install with: pip install optax"
            ) from exc
        factory = optax.adam if self.optimizer == "adam" else optax.sgd
        return factory(learning_rate=self.learning_rate)


def compute_overlap_gradient(
    forward_fields_history,
    adjoint_fields_history,
    field_key="Ez",
    forward_start=0,
    adjoint_start=0,
):
    """
    Compute the gradient of the overlap integral with respect to epsilon.
    Gradient = Re(E_fwd * E_adj) integrated over time.
    """
    if len(forward_fields_history) == 0 or len(adjoint_fields_history) == 0:
        raise ValueError("Field histories must be non-empty.")

    forward_start = max(int(forward_start), 0)
    adjoint_start = max(int(adjoint_start), 0)
    fwd_hist = forward_fields_history[forward_start:]
    adj_hist = adjoint_fields_history[adjoint_start:]

    grad = np.zeros_like(forward_fields_history[0], dtype=float)

    n_steps = min(len(fwd_hist), len(adj_hist))
    if n_steps <= 0:
        return grad

    for i in range(n_steps):
        grad += fwd_hist[i] * adj_hist[n_steps - 1 - i]

    return grad


def fold_high_side_yee_padding_to_shape(
    grad: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Fold high-side Yee padding planes back onto a material-grid shape.

    Some full-Yee field components include the high boundary sample explicitly,
    so their field-derived gradients can be one element larger than the
    cell-centered material grid. Material sampling clips those high-side samples
    to the last material cell; this helper applies the corresponding adjoint by
    accumulating the high-side plane into the last target index.
    """

    out = np.asarray(grad, dtype=float)
    target_shape = tuple(int(v) for v in target_shape)
    if out.shape == target_shape:
        return out
    if out.ndim != len(target_shape):
        raise ValueError(
            f"Gradient ndim {out.ndim} does not match target ndim {len(target_shape)}."
        )

    for axis, target in enumerate(target_shape):
        size = out.shape[axis]
        if size == target:
            continue
        if size != target + 1:
            raise ValueError(
                "Cannot align gradient shape "
                f"{out.shape} to target shape {target_shape}: axis {axis} has "
                f"size {size}, expected {target} or {target + 1}."
            )

        core: list[Any] = [slice(None)] * out.ndim
        core[axis] = slice(0, target)
        folded = out[tuple(core)].copy()

        last: list[Any] = [slice(None)] * out.ndim
        last[axis] = target - 1
        high: list[Any] = [slice(None)] * out.ndim
        high[axis] = target
        folded[tuple(last)] += out[tuple(high)]
        out = folded

    return out


def create_optimization_mask(grid, region_structure):
    """
    Helper to create a boolean mask from a structure on a grid.
    Uses rasterization to ensure exact alignment with how structures are mapped to the grid.
    """
    # Create temp design to rasterize mask exactly as grid does
    temp_design = Design(
        width=grid.width, height=grid.height, material=Material(permittivity=1.0)
    )

    # Set to a distinct permittivity to detect it without mutating the input.
    marker = Material(permittivity=2.0)
    if not isinstance(region_structure, Structure):
        raise TypeError("Optimization regions must be canonical Structure objects.")
    temp_design += region_structure.with_material(marker)

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
