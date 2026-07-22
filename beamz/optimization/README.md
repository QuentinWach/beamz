# Beamz Optimization Module

This module provides tools for gradient-based optimization of electromagnetic devices, with a primary focus on topology optimization using the adjoint method and JAX for autodifferentiation.

## Structure

### `topology.py`
The high-level interface for topology optimization.
- **`TopologySpec`**: Immutable design mask, optimizer, filter, and projection configuration.
- **`TopologyState`**: Immutable density, optimizer buffers, and objective history.
  - **`density_for_step(state, step, total_steps)`**: Returns the continuation beta and physical density.
  - **`apply_gradient(state, grad_eps, beta)`**: Returns updated state and the maximum parameter update.
  - **`physical_density(state, beta)`**: Transforms latent parameters into physical density (0 to 1).
- **Helper Functions**:
  - `compute_overlap_gradient`: Calculates the gradient of the overlap integral (mode matching) using forward and adjoint fields.
  - `create_optimization_mask`: Generates a boolean mask defining the design region.
  - `get_fixed_structure_mask`: Identifies fixed structures (e.g., waveguides) outside the design region to ensure proper connectivity.

### `autodiff.py`
A library of JAX-based differentiable operations used for density filtering and projection.
- **Morphological Filters**:
  - `grayscale_erosion`, `grayscale_dilation`: Differentiable grayscale morphology using smooth min/max approximations (LogSumExp).
  - `grayscale_opening`, `grayscale_closing`: Compound operations for noise removal and feature size control.
  - `masked_morphological_filter`: Applies filters with support for a "fixed structure mask" to prevent erosion at waveguide connections.
- **Conic Filters**:
  - `masked_conic_filter`: A filter with a linear decay kernel (cone), used for enforcing geometric constraints like minimum linewidth and spacing.
- **Blurring**:
  - `masked_box_blur`: Standard box blur implementation.
- **Projection**:
  - `smoothed_heaviside`: A differentiable step function (using `tanh`) to binarize the density field.
  - `transform_density`: Applies the selected filter and projection method.
- **Backpropagation**:
  - `compute_parameter_gradient_vjp`: Uses JAX's vector-Jacobian product (VJP) to automatically compute gradients through the entire filter-project pipeline.

### `projections.py`
Projection methods for filtered topology-optimization densities.
- `smoothed_heaviside`: The default tanh-based projection method.
- `subpixel_smoothed_projection`: Hammond SSP1 for smooth 2D density fields.

## Key Features

1.  **Differentiable Morphology**: Unlike standard blurring, this module supports differentiable morphological operations (erosion, dilation, opening, closing). This allows for strict control over minimum feature sizes and avoids "gray" boundaries often seen with Gaussian blurs.
2.  **Geometric Constraints**: The **conic filter** option provides a method to enforce minimum length scales (linewidth and spacing) by using a cone-shaped kernel, as described in topology optimization literature.
3.  **Connectivity Preservation**: The filtering pipeline includes a mechanism to "pad" the design region with information from fixed structures (like input/output waveguides). This prevents the optimization from creating gaps or disconnecting the device from the external circuit.
4.  **Beta-Continuation**: Supports a beta-schedule for the Heaviside projection, gradually increasing the sharpness of the binarization to avoid getting stuck in local minima while ensuring a final binary design.
5.  **Projection Selection**: Supports `projection_type="heaviside"` for the default tanh projection and `projection_type="ssp"` for Hammond subpixel-smoothed projection.
6.  **JAX Integration**: All heavy lifting for density transformation and gradient chain-rule calculation is handled efficiently by JAX. Uses `optax` for JAX-native optimizer implementations (Adam, SGD).

## Projection Methods

The density transform follows:

```text
design density -> filter -> projection -> physical density
```

Available projection methods:

- `projection_type="heaviside"`: the default tanh smoothed Heaviside projection.
- `projection_type="ssp"`: Hammond SSP1, applied to the already-filtered 2D density field.

SSP is intended for smooth or filtered 2D density inputs. It supports `beta=jnp.inf`
without adding any runtime, optional, or test dependency. The smoothing radius is in
density-grid cell units and defaults to `ssp_smoothing_radius=0.55`.

Example:

```python
import jax.numpy as jnp

from beamz.optimization.autodiff import transform_density

physical_density = transform_density(
    density,
    mask,
    beta=jnp.inf,
    eta=0.5,
    radius=2,
    filter_type="conic",
    projection_type="ssp",
)
```

## Usage Example

```python
from beamz.optimization.topology import TopologySpec, create_optimization_mask

# 1. Setup Design and Mask
mask = create_optimization_mask(grid, opt_region)

# 2. Initialize immutable configuration and explicit state
opt = TopologySpec(
    design=design,
    region_mask=mask,
    resolution=DX,
    filter_type='conic',  # Options: 'morphological', 'conic'
    projection_type='ssp',
    filter_radius=0.15*µm,  # Physical units (e.g. microns)
    ssp_smoothing_radius=0.55,  # Density-grid cell units
)
state = opt.initial_state()

# 3. Optimization Loop
for step in range(STEPS):
    # Get current physical density
    beta, phys_density = opt.density_for_step(state, step, STEPS)
    
    # Update grid permittivity
    grid.permittivity[mask] = EPS_MIN + phys_density[mask] * (EPS_MAX - EPS_MIN)
    
    # ... Run FDTD & Compute Gradient (grad_eps) ...
    
    # Update Parameters
    state, max_update = opt.apply_gradient(state, grad_eps, beta)
```
