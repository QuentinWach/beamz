# Beamz Optimization Module

This module provides tools for gradient-based optimization of electromagnetic devices, with a primary focus on topology optimization using the adjoint method and JAX for autodifferentiation.

## Structure

### `topology.py`
The high-level interface for topology optimization.
- **`TopologyManager` Class**: Manages the optimization lifecycle.
  - **Initialization**: Sets up the design, mask, optimizer (using `optax`), and filter parameters.
  - **`update_design(step)`**: Updates the physical material distribution based on the current optimization step (handling beta-continuation).
  - **`apply_gradient(grad_eps, beta)`**: Backpropagates the gradient from the epsilon (permittivity) domain to the design parameter domain and performs an optimizer step using `optax`.
  - **`get_physical_density(beta)`**: Transforms the latent design parameters into a physical density (0 to 1) using filtering and projection.
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
- **Backpropagation**:
  - `compute_parameter_gradient_vjp`: Uses JAX's vector-Jacobian product (VJP) to automatically compute gradients through the entire filter-project pipeline.

## Key Features

1.  **Differentiable Morphology**: Unlike standard blurring, this module supports differentiable morphological operations (erosion, dilation, opening, closing). This allows for strict control over minimum feature sizes and avoids "gray" boundaries often seen with Gaussian blurs.
2.  **Geometric Constraints**: The **conic filter** option provides a method to enforce minimum length scales (linewidth and spacing) by using a cone-shaped kernel, as described in topology optimization literature.
3.  **Connectivity Preservation**: The filtering pipeline includes a mechanism to "pad" the design region with information from fixed structures (like input/output waveguides). This prevents the optimization from creating gaps or disconnecting the device from the external circuit.
4.  **Beta-Continuation**: Supports a beta-schedule for the Heaviside projection, gradually increasing the sharpness of the binarization to avoid getting stuck in local minima while ensuring a final binary design.
5.  **JAX Integration**: All heavy lifting for density transformation and gradient chain-rule calculation is handled efficiently by JAX. Uses `optax` for JAX-native optimizer implementations (Adam, SGD).

## Usage Example

```python
from beamz.optimization.topology import TopologyManager, create_optimization_mask

# 1. Setup Design and Mask
mask = create_optimization_mask(grid, opt_region)

# 2. Initialize Manager
opt = TopologyManager(
    design=design,
    region_mask=mask,
    resolution=DX,
    filter_type='conic', # Options: 'morphological', 'conic', 'blur'
    filter_radius=0.15*µm,       # Physical units (e.g. microns)
    simple_smooth_radius=0.03*µm # Optional smoothing (physical units)
)

# 3. Optimization Loop
for step in range(STEPS):
    # Get current physical density
    beta, phys_density = opt.update_design(step, STEPS)
    
    # Update grid permittivity
    grid.permittivity[mask] = EPS_MIN + phys_density[mask] * (EPS_MAX - EPS_MIN)
    
    # ... Run FDTD & Compute Gradient (grad_eps) ...
    
    # Update Parameters
    opt.apply_gradient(grad_eps, beta)
```
