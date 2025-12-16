import numpy as np
import matplotlib.pyplot as plt
from beamz import *
from beamz.optimization.topology import TopologyManager, compute_overlap_gradient, create_optimization_mask

# --- 1. Simulation Setup ---
W = H = 15*µm
WG_W = 0.5*µm
WL = 1.55*µm
N_CORE, N_CLAD = 2.25, 1.444
DX, DT = calc_optimal_fdtd_params(WL, 2.25, points_per_wavelength=12)
STEPS = 50
MAT_PENALTY = 300.0 # Penalty weight for material usage

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
    filter_radius=0.28*µm,
    eps_min=N_CLAD**2,
    eps_max=N_CORE**2,
    beta_schedule=(1.0, 20.0)
)

print(f"Starting Topology Optimization ({STEPS} steps)...")
base_eps = grid.permittivity.copy() # Store background (cladding)

# Track transmission history
transmission_history = []

# --- 3. Optimization Loop ---
for step in range(STEPS):
    # Update Design
    beta, phys_density = opt.update_design(step, STEPS)
    
    # Mix Density into Permittivity (Linear Interpolation)
    grid.permittivity[:] = base_eps
    grid.permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)
    
    # Forward Simulation
    src_fwd.grid = grid # Update grid ref
    
    # Setup monitors for input and output power measurement
    input_monitor = Monitor(design=grid, start=(2.5*µm, H/2-WG_W*2), end=(2.5*µm, H/2+WG_W*2), 
                           accumulate_power=True, record_fields=False)
    output_monitor = Monitor(design=grid, start=(W/2-WG_W*2, H-2*µm), end=(W/2+WG_W*2, H-2*µm),
                            accumulate_power=True, record_fields=False)
    
    sim_fwd = Simulation(grid, [src_fwd], [PML(edges='all', thickness=1*µm)], time=time, resolution=DX)
    fwd_ez_history = []
    
    print(f"[{step+1}/{STEPS}] Forward Sim...", end="\r")
    while sim_fwd.step():
        if sim_fwd.current_step % 2 == 0: # Subsample for memory
            fwd_ez_history.append(sim_fwd.fields.Ez.copy())
        
        # Record power at monitors
        if input_monitor.should_record(sim_fwd.current_step):
            input_monitor.record_fields(sim_fwd.fields.Ez, sim_fwd.fields.Hx, sim_fwd.fields.Hy, 
                                       sim_fwd.t, DX, DX, sim_fwd.current_step)
        if output_monitor.should_record(sim_fwd.current_step):
            output_monitor.record_fields(sim_fwd.fields.Ez, sim_fwd.fields.Hx, sim_fwd.fields.Hy,
                                        sim_fwd.t, DX, DX, sim_fwd.current_step)
    
    # Calculate transmission percentage
    input_power = np.sum(input_monitor.power_history) if input_monitor.power_history else 1.0
    output_power = np.sum(output_monitor.power_history) if output_monitor.power_history else 0.0
    transmission_pct = (output_power / input_power * 100.0) if input_power > 0 else 0.0
    
    # For objective, use transmission percentage scaled appropriately
    obj_val = transmission_pct  # Use percentage directly as objective
    
    opt.objective_history.append(obj_val)
    transmission_history.append(transmission_pct)
    
    # Adjoint Simulation
    # print(f"[{step+1}/{STEPS}] Adjoint Sim... (Obj: {obj_val:.2e})", end="\r")
    src_adj.grid = grid
    sim_adj = Simulation(grid, [src_adj], [PML(edges='all', thickness=1*µm)], time=time, resolution=DX)
    adj_ez_history = []
    
    while sim_adj.step():
        if sim_adj.current_step % 2 == 0:
            adj_ez_history.append(sim_adj.fields.Ez.copy())
            
    # Compute Gradient (overlap of fwd and adj fields)
    grad_eps = compute_overlap_gradient(fwd_ez_history, adj_ez_history)
    
    # Measure Material Usage (Relative core material amount)
    # phys_density is 0 (cladding) to 1 (core)
    # We sum this to get the total effective "area" of core material used
    core_usage = np.sum(phys_density[mask])
    
    # Apply Penalty
    # We penalize the total core usage.
    # Objective = Transmission - Penalty_Weight * Core_Usage
    # To correspond with the gradient subtraction of MAT_PENALTY in the epsilon domain:
    # grad_eps -= MAT_PENALTY
    # Since eps = eps_min + rho * (eps_max - eps_min), d(rho)/d(eps) = 1/DeltaEps
    # If we want dJ/d(eps) to shift by -MAT_PENALTY, then dJ/d(rho) must shift by -MAT_PENALTY * DeltaEps
    # Thus the penalty term in the objective is: MAT_PENALTY * DeltaEps * Sum(rho)
    delta_eps = opt.eps_max - opt.eps_min
    penalty_val = MAT_PENALTY * delta_eps * core_usage
    
    grad_eps[mask] -= MAT_PENALTY

    # Total Objective (transmission percentage - penalty)
    # Scale penalty to be comparable to transmission percentage
    penalty_scaled = penalty_val / (delta_eps * np.sum(mask))  # Normalize penalty
    total_obj = obj_val - penalty_scaled
    
    # Step Optimizer
    max_update = opt.apply_gradient(grad_eps, beta)
    
    # Calculate fraction for display
    mat_frac = np.mean(phys_density[mask])
    
    print(f" Step {step+1}: Obj={total_obj:.2e} (Trans={transmission_pct:.1f}%) | Mat={mat_frac:.1%} | MaxUp={max_update:.2e}", end="\r")
    
    # Viz
    if step % 5 == 0:
        plt.imsave(f"topo_opt_{step:03d}.png", grid.permittivity.T, cmap='gray', origin='lower')

print(f"\nOptimization Complete. Final Transmission: {transmission_history[-1]:.1f}%")

# Plot transmission vs step (as percentage)
plt.figure(figsize=(10, 6))
steps = np.arange(1, len(transmission_history) + 1)
plt.plot(steps, transmission_history, 'b-', linewidth=2, marker='o', markersize=4)
plt.xlabel('Optimization Step', fontsize=12)
plt.ylabel('Transmission (%)', fontsize=12)
plt.title('Transmission vs Optimization Step', fontsize=14)
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('transmission_vs_step.png', dpi=150, bbox_inches='tight')
print(f"Transmission plot saved to transmission_vs_step.png")
plt.close()