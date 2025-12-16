import numpy as np
import matplotlib.pyplot as plt
from beamz import *
from beamz.optimization.topology import TopologyManager, compute_overlap_gradient, create_optimization_mask

# --- 1. Simulation Setup ---
W = H = 15*µm
WG_W = 0.5*µm
WL = 1.55*µm
N_CORE, N_CLAD = 2.25, 1.444
DX, DT = calc_optimal_fdtd_params(WL, 2.25, points_per_wavelength=9)
STEPS = 50

# Design & Materials
design = Design(width=W, height=H, material=Material(permittivity=N_CLAD**2))
design += Rectangle(position=(0, H/2-WG_W/2), width=3.5*µm, height=WG_W, material=Material(permittivity=N_CORE**2))
design += Rectangle(position=(W/2-WG_W/2, H), width=WG_W, height=-3.5*µm, material=Material(permittivity=N_CORE**2))

# Optimization Region (added as placeholder)
opt_region = Rectangle(position=(W/2-4*µm, H/2-4*µm), width=8*µm, height=8*µm, material=Material(permittivity=N_CLAD**2))
design += opt_region

# Sources
time = np.arange(0, 30*WL/LIGHT_SPEED, DT)
signal = ramped_cosine(time, 1, LIGHT_SPEED/WL, ramp_duration=6*WL/LIGHT_SPEED, t_max=time[-1])
src_fwd = ModeSource(None, center=(2*µm, H/2), width=WG_W*4, wavelength=WL, pol="tm", signal=signal, direction="+x")
src_adj = ModeSource(None, center=(W/2, H-2*µm), width=WG_W*4, wavelength=WL, pol="tm", signal=signal, direction="-y")

# --- 2. Optimization Manager ---
# Rasterize once to get grid and mask
grid = design.rasterize(DX)
mask = create_optimization_mask(grid, opt_region)

opt = TopologyManager(
    design=design,
    region_mask=mask,
    resolution=DX,
    learning_rate=0.1,
    filter_radius=0.2*µm,
    eps_min=N_CLAD**2,
    eps_max=N_CORE**2,
    beta_schedule=(1.0, 20.0)
)

print(f"Starting Topology Optimization ({STEPS} steps)...")
base_eps = grid.permittivity.copy() # Store background (cladding)

# --- 3. Optimization Loop ---
for step in range(STEPS):
    # Update Design
    beta, phys_density = opt.update_design(step, STEPS)
    
    # Mix Density into Permittivity (Linear Interpolation)
    grid.permittivity[:] = base_eps
    grid.permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)
    
    # Forward Simulation
    src_fwd.grid = grid # Update grid ref
    sim_fwd = Simulation(grid, [src_fwd], [PML(edges='all', thickness=1*µm)], time=time, resolution=DX)
    fwd_ez_history = []
    monitor_vals = []
    
    print(f"[{step+1}/{STEPS}] Forward Sim...", end="\r")
    while sim_fwd.step():
        if sim_fwd.current_step % 2 == 0: # Subsample for memory
            fwd_ez_history.append(sim_fwd.fields.Ez.copy())
            # Simple point monitor at output for tracking
            monitor_vals.append(sim_fwd.fields.Ez[int((H-2*µm)/DX), int(W/2/DX)])
            
    obj_val = np.sum(np.abs(np.array(monitor_vals))**2)
    opt.objective_history.append(obj_val)
    
    # Adjoint Simulation
    print(f"[{step+1}/{STEPS}] Adjoint Sim... (Obj: {obj_val:.2e})", end="\r")
    src_adj.grid = grid
    sim_adj = Simulation(grid, [src_adj], [PML(edges='all', thickness=1*µm)], time=time, resolution=DX)
    adj_ez_history = []
    
    while sim_adj.step():
        if sim_adj.current_step % 2 == 0:
            adj_ez_history.append(sim_adj.fields.Ez.copy())
            
    # Compute Gradient (overlap of fwd and adj fields)
    # Note: adj_ez_history is time 0..T, fwd is 0..T. 
    # Adjoint field at time t in simulation corresponds to real time T-t.
    # We multiply aligned indices.
    grad_eps = compute_overlap_gradient(fwd_ez_history, adj_ez_history)
    
    # Step Optimizer
    max_update = opt.apply_gradient(grad_eps, beta)
    
    # Viz
    if step % 5 == 0:
        plt.imsave(f"topo_opt_{step:03d}.png", grid.permittivity.T, cmap='gray', origin='lower')

print(f"\nOptimization Complete. Final Objective: {obj_val:.2e}")
