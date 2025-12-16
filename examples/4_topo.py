import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
from beamz import *
from beamz.optimization.core import Optimizer
from beamz.devices.sources.mode import solve_modes

# JAX-based filter functions for topology optimization
def masked_box_blur_jax(values, mask, radius):
    """
    Apply a masked box blur using JAX convolutions.
    Equivalent to the numpy version but differentiable.
    """
    radius = int(max(0, radius))
    if radius <= 0:
        return jnp.where(mask, values, 0.0), jnp.where(mask, 1.0, 1.0)
    
    masked_values = jnp.where(mask, values, 0.0)
    float_mask = mask.astype(float)
    
    # Create box kernel
    kernel_size = 2 * radius + 1
    kernel = jnp.ones((kernel_size, kernel_size))
    
    # Pad input to handle edges (mimic mode='edge' roughly or 'constant' 0)
    # Numpy implementation used 'edge' padding for values and 'constant' 0 for mask.
    # JAX convolve2d with 'same' and proper boundary handling is tricky.
    # We will pad manually.
    
    padded_values = jnp.pad(masked_values, radius, mode='edge')
    padded_mask = jnp.pad(float_mask, radius, mode='constant', constant_values=0.0)
    
    # Convolve with 'valid' since we padded
    weighted_sum = convolve2d(padded_values, kernel, mode='valid')
    weights = convolve2d(padded_mask, kernel, mode='valid')
    
    # Avoid division by zero
    weights = jnp.where(weights == 0.0, 1.0, weights)
    
    blurred = weighted_sum / weights
    blurred = jnp.where(mask, blurred, 0.0)
    weights = jnp.where(mask, weights, 1.0)
    
    return blurred, weights

def smoothed_heaviside_jax(value, beta, eta):
    """
    Smoothed Heaviside projection using tanh.
    """
    beta = jnp.maximum(beta, 1e-6)
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (value - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den

def transform_density(density, mask, beta, eta, radius):
    """
    Full density transform: Blur -> Project.
    """
    blurred, _ = masked_box_blur_jax(density, mask, radius)
    projected = smoothed_heaviside_jax(blurred, beta, eta)
    return jnp.where(mask, projected, 0.0)

# Parameters
W = H = 15*µm
WG_W = 0.5*µm
N_CORE, N_CLAD = 2.25, 1.444
EPS_CORE, EPS_CLAD = N_CORE**2, N_CLAD**2
WL = 1.55*µm
PPP = 9
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), points_per_wavelength=PPP)
T_MAX = 30*WL/LIGHT_SPEED
STEPS = 50 # Increased steps for meaningful optimization
LR = 0.1

# Projection parameters
PROJECTION_BETA_START = 1.0
PROJECTION_BETA_END = 20.0
PROJECTION_ETA = 0.5
FILTER_RADIUS = 0.2*µm

# Create the design
design = Design(width=W, height=H, material=Material(permittivity=EPS_CLAD))
design += Rectangle(position=(0*µm,H/2-WG_W/2), width=3.5*µm, height=WG_W, material=Material(permittivity=EPS_CORE))
design += Rectangle(position=(W/2-WG_W/2,H), width=WG_W, height=-3.5*µm, material=Material(permittivity=EPS_CORE))
region_rect = Rectangle(position=(W/2-4*µm,H/2-4*µm), width=8*µm, height=8*µm, material=Material(permittivity=EPS_CORE))
# Note: We don't add region_rect to design yet, we use it to define mask
# Actually, let's add it with base permittivity (CLAD) so it can be updated
design += Rectangle(position=(W/2-4*µm,H/2-4*µm), width=8*µm, height=8*µm, material=Material(permittivity=EPS_CLAD))

# Precompute the material grid
grid = design.rasterize(resolution=DX)

# Mask setup
dx, dy = getattr(grid, "dx", DX), getattr(grid, "dy", DX)
min_cell = float(min(dx, dy))
filter_radius_cells = max(1, int(round(FILTER_RADIUS / min_cell)))

base = grid.permittivity.copy()
mask = np.zeros_like(base, bool)
xs, ys = (np.arange(mask.shape[1])+0.5)*dx, (np.arange(mask.shape[0])+0.5)*dy
minx, miny, _, maxx, maxy, _ = region_rect.get_bounding_box()
mask[(ys[:,None]>=miny)&(ys[:,None]<=maxy)&(xs[None,:]>=minx)&(xs[None,:]<=maxx)] = True

# Define sources
time = np.arange(0, T_MAX, DT)
signal = ramped_cosine(t=time, amplitude=1, frequency=LIGHT_SPEED/WL, t_max=T_MAX, ramp_duration=6*WL/LIGHT_SPEED, phase=0)

input_source = ModeSource(grid=grid, center=(2*µm, H/2), width=WG_W*4, wavelength=WL,
                pol="tm", signal=signal, direction="+x")
                
# Target mode calculation for adjoint source
monitor_y = H - 2*µm
monitor_x_center = W/2
monitor_width = WG_W*4

# 1D Mode Solve for target profile
# Extract eps slice
x_axis = np.arange(0, W, dx)
y_idx_monitor = int(monitor_y / dy)
eps_line = base[y_idx_monitor, :]
# Focus on the waveguide region for mode solving
x_indices = np.where((x_axis >= monitor_x_center - monitor_width/2) & (x_axis <= monitor_x_center + monitor_width/2))[0]
# Actually simpler to just solve for a waveguide cross section
# We construct a theoretical waveguide cross section
eps_1d = np.ones_like(x_axis) * EPS_CLAD
wg_mask = (x_axis >= monitor_x_center - WG_W/2) & (x_axis <= monitor_x_center + WG_W/2)
eps_1d[wg_mask] = EPS_CORE

omega = 2 * np.pi * LIGHT_SPEED / WL
neffs, mode_e, _, _ = solve_modes(eps=eps_1d, omega=omega, dL=dx, m=1, direction="+y", return_fields=True)
target_mode_profile = np.squeeze(mode_e[0][2]) # Ez component of first mode
target_mode_profile = target_mode_profile / np.linalg.norm(target_mode_profile) # Normalize

# Adjoint Source
# We need to spatially match the target mode profile to the source
# The Adjoint source injects the conjugate of the desired output mode
# For a mode source, we can just specify the mode.
# But for precise adjoint, we often want to inject the mode profile weighted by the error.
# Here we just optimize for transmission into the fundamental mode, so we inject the fundamental mode backwards.
back_source = ModeSource(grid=grid, center=(monitor_x_center, monitor_y), width=monitor_width, wavelength=WL,
                pol="tm", signal=signal, direction="-y")

# Optimization state
optimizer = Optimizer("Adam", learning_rate=LR)
design_density = np.full(mask.shape, 0.5) # Initialize density 0.5
design_density[~mask] = 0.0

objective_history = []

print("Starting topology optimization with JAX autodiff...")

for step in range(STEPS):
    # 1. Update density parameters
    beta = PROJECTION_BETA_START + (step/STEPS) * (PROJECTION_BETA_END - PROJECTION_BETA_START)
    
    # Transform density to physical permittivity using JAX (forward pass)
    design_density_jax = jnp.array(design_density)
    mask_jax = jnp.array(mask)
    physical_density_jax = transform_density(design_density_jax, mask_jax, beta, PROJECTION_ETA, filter_radius_cells)
    physical_density = np.array(physical_density_jax)
    
    # Update grid permittivity
    current_eps = base.copy()
    current_eps[mask] = EPS_CLAD + physical_density[mask] * (EPS_CORE - EPS_CLAD)
    np.copyto(grid.permittivity, current_eps)
    
    # 2. Forward Simulation
    print(f"[Step {step+1}/{STEPS}] Running Forward...")
    forward = Simulation(design=design, devices=[input_source],
        boundaries=[PML(edges='all', thickness=1.0*µm)], time=time, resolution=DX)
    
    # We need to record fields manually for adjoint calculation
    forward_ez_history = []
    # Monitor for objective calculation
    # We'll just record the field at the output plane
    monitor_ez_history = []
    
    # Run forward loop
    while forward.step():
        # Store full field for adjoint (subsampled to save memory if needed, but here we store all)
        # Assuming 2D, Ez is (Ny, Nx)
        if forward.current_step % 1 == 0:
            forward_ez_history.append(forward.fields.Ez.copy())
            
        # Record output for objective
        # Sample at monitor line
        monitor_field = forward.fields.Ez[y_idx_monitor, :]
        monitor_ez_history.append(monitor_field)
    
    # Calculate Objective (Mode Overlap)
    # Overlap = |<Psi_target | Psi_sim>|^2
    # We integrate overlap over time
    total_overlap = 0.0
    # Resample target profile to grid if needed (it matches x_axis already)
    
    # Simple power transmission objective: integrate Poynting or just intensity
    # But for mode match, we project onto target mode.
    # We use the final fields or integrate? Usually integrate flux.
    # Here we use the overlap integral approach similar to secondary example.
    
    # Calculate error signal for adjoint source: dJ/dE
    # For maximize mode overlap: J = |integral(E_sim * conj(E_mode))|^2
    # This is slightly complex. Let's simplify: Maximize projection onto mode at output.
    # Or simplified: maximize power in mode 0.
    
    # We'll use the "mode match" score from secondary as objective to print
    # But for gradient, the standard method is:
    # Adjoint source = conj(Mode_profile) if we maximize transmission.
    # back_source is already set up to inject mode 0 backwards.
    
    # 3. Adjoint Simulation
    print(f"              Running Adjoint...")
    adjoint = Simulation(design=design, devices=[back_source],
        boundaries=[PML(edges='all', thickness=1.0*µm)], time=time, resolution=DX)
    
    grad_eps = np.zeros_like(base)
    
    # We iterate backwards in time for the adjoint field interacting with forward field?
    # No, usually adjoint is run forward in time but logically it's backward.
    # The gradient is integral(E_fwd * E_adj).
    # Since we saved forward history, we can pop from it.
    
    idx = len(forward_ez_history) - 1
    while adjoint.step():
        if idx >= 0:
            # E_adj is the field from adjoint simulation
            e_adj = adjoint.fields.Ez
            e_fwd = forward_ez_history[idx]
            
            # Gradient accumulation: Re(E_fwd * E_adj)
            # (Note: exact form depends on formulation, usually integral of E_fwd dot E_adj)
            grad_eps += np.real(e_fwd * e_adj)
            
            idx -= 1
            
    # Calculate objective value for reporting
    # Just sum of intensity at monitor for now
    obj_val = np.sum(np.abs(np.array(monitor_ez_history))**2)
    objective_history.append(obj_val)
    print(f"              Objective: {obj_val:.4e}")
            
    # 4. Backpropagation through filters (JAX)
    # grad_eps is dJ/d(epsilon)
    # We need dJ/d(density)
    # epsilon = eps_clad + density * (eps_core - eps_clad)
    # d(eps)/d(density) = eps_core - eps_clad
    
    grad_physical = grad_eps * (EPS_CORE - EPS_CLAD)
    
    # VJP to backprop through transform
    # Define function wrapper for JAX
    def transform_wrapper(d):
        return transform_density(d, mask_jax, beta, PROJECTION_ETA, filter_radius_cells)
    
    _, vjp_fun = jax.vjp(transform_wrapper, design_density_jax)
    
    # Compute gradient w.r.t design parameters
    # Note: grad_physical should be passed to VJP
    # JAX expects arrays
    grad_param_jax = vjp_fun(jnp.array(grad_physical))[0]
    grad_param = np.array(grad_param_jax)
    
    # 5. Optimizer Step
    # We maximize objective, so we ascend gradient.
    # Optimizer usually minimizes, so we feed negative gradient.
    update = optimizer.step(-grad_param)
    
    # Apply update and clip
    design_density[mask] += update[mask]
    design_density = np.clip(design_density, 0.0, 1.0)
    
    # Visualization
    if step % 5 == 0:
        plt.figure(figsize=(10,4))
        plt.subplot(131)
        plt.imshow(current_eps.T, origin='lower', cmap='gray')
        plt.title(f"Permittivity (Step {step})")
        plt.subplot(132)
        plt.imshow(grad_param.T, origin='lower', cmap='RdBu', vmin=-np.max(np.abs(grad_param)), vmax=np.max(np.abs(grad_param)))
        plt.title("Gradient")
        plt.subplot(133)
        plt.plot(objective_history)
        plt.title("Objective")
        plt.tight_layout()
        plt.savefig(f"topo_opt_step_{step:03d}.png")
        plt.close()

# Final simulation and viz
design_density_jax = jnp.array(design_density)
physical_density_jax = transform_density(design_density_jax, mask_jax, PROJECTION_BETA_END, PROJECTION_ETA, filter_radius_cells)
physical_density = np.array(physical_density_jax)
final_eps = base.copy()
final_eps[mask] = EPS_CLAD + physical_density[mask] * (EPS_CORE - EPS_CLAD)
np.copyto(grid.permittivity, final_eps)

print("Running final validation...")
final_sim = Simulation(design=design, devices=[input_source],
    boundaries=[PML(edges='all', thickness=1.0*µm)], time=time, resolution=DX)
final_sim.run(animate_live="Ez", animation_interval=10, axis_scale=[-6e-5, 6e-5])
