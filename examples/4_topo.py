import numpy as np
import matplotlib.pyplot as plt
from beamz import *
from beamz.optimization.topology import TopologyManager, compute_overlap_gradient, create_optimization_mask

# --- 1. Simulation Setup ---
W = H = 15*µm
WG_W = 0.5*µm
WL = 1.55*µm
N_CORE, N_CLAD = 2.25, 1.444
DX, DT = calc_optimal_fdtd_params(WL, 2.25, points_per_wavelength=15)
STEPS = 50
MAT_PENALTY = 0.0 # Penalty weight for material usage

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
    filter_radius=0.15*µm,
    eps_min=N_CLAD**2,
    eps_max=N_CORE**2,
    beta_schedule=(1.0, 20.0),
    filter_type='conic',
    morphology_operation='openclose',
    morphology_smooth_tau=0.005,
    post_smooth_radius=1
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
    
    # Forward Simulation (only output monitor)
    src_fwd.grid = grid # Update grid ref
    
    # Setup monitors for input and output power measurement
    # Place monitor immediately after source to measure actual injected power
    # This accounts for soft source loading and back-reflection
    monitor_input_flux = Monitor(design=grid, start=(2.5*µm, H/2-WG_W*2), end=(2.5*µm, H/2+WG_W*2), 
                           accumulate_power=True, record_fields=False)
    
    # Output monitor at top waveguide
    output_monitor_fwd = Monitor(design=grid, start=(W/2-WG_W*2, H-2*µm), end=(W/2+WG_W*2, H-2*µm),
                                 accumulate_power=True, record_fields=False)
    
    # Run forward simulation with output monitor
    sim_fwd = Simulation(grid, [src_fwd, monitor_input_flux, output_monitor_fwd], 
                        [PML(edges='all', thickness=1*µm)], time=time, resolution=DX)
    
    print(f"[{step+1}/{STEPS}] Forward Sim...", end="\r")
    results = sim_fwd.run(save_fields=['Ez'], field_subsample=2)
    
    # Extract field history
    fwd_ez_history = results['fields']['Ez'] if results and 'fields' in results else []
    
    # Calculate transmission normalizing by measured input flux
    # Input flux includes forward wave + reflection. 
    # For high transmission structures, reflection is low, so this is a good approximation of injected power.
    measured_input_energy = np.sum(monitor_input_flux.power_history) * DT
    measured_output_energy = np.sum(output_monitor_fwd.power_history) * DT
    
    # Avoid division by zero
    if measured_input_energy <= 0: measured_input_energy = 1.0
    
    transmission_fwd = (measured_output_energy / measured_input_energy * 100.0)
    
    # Backward Simulation (with backward monitor at input location)
    src_adj.grid = grid
    
    # Backward source monitor (near top source)
    monitor_back_flux = Monitor(design=grid, start=(W/2-WG_W*2, H-2.5*µm), end=(W/2+WG_W*2, H-2.5*µm),
                              accumulate_power=True, record_fields=False)
    
    # Backward monitor at original input location (left waveguide)
    backward_monitor = Monitor(design=grid, start=(2.5*µm, H/2-WG_W*2), end=(2.5*µm, H/2+WG_W*2),
                              accumulate_power=True, record_fields=False)
    
    sim_adj = Simulation(grid, [src_adj, monitor_back_flux, backward_monitor], 
                        [PML(edges='all', thickness=1*µm)], time=time, resolution=DX)
    
    adj_results = sim_adj.run(save_fields=['Ez'], field_subsample=2)
    adj_ez_history = adj_results['fields']['Ez'] if adj_results and 'fields' in adj_results else []
    
    # Calculate backward transmission normalizing by measured input flux
    measured_input_energy_back = np.sum(monitor_back_flux.power_history) * DT
    if measured_input_energy_back <= 0: measured_input_energy_back = 1.0
    
    output_energy_back = np.sum(backward_monitor.power_history) * DT
    transmission_back = (output_energy_back / measured_input_energy_back * 100.0)
    
    # Average bidirectional transmission
    transmission_pct = (transmission_fwd + transmission_back) / 2.0
    
    # For objective, use averaged transmission percentage
    obj_val = transmission_pct
    
    opt.objective_history.append(obj_val)
    transmission_history.append(transmission_pct)
            
    # Compute Gradient (overlap of fwd and adj fields)
    grad_eps = compute_overlap_gradient(fwd_ez_history, adj_ez_history)
    
    # Measure Material Usage (Relative core material amount)
    # phys_density is 0 (cladding) to 1 (core)
    # We sum this to get the total effective "area" of core material used
    core_usage = np.sum(phys_density[mask])
    
    # Apply Penalty
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
    
    print(f" Step {step+1}: Obj={total_obj:.2e} (Trans={transmission_pct:.1f}% | Fwd={transmission_fwd:.1f}% Bwd={transmission_back:.1f}%) | Mat={mat_frac:.1%} | MaxUp={max_update:.2e}", end="\r")
    
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

# --- 4. Final Verification & Visualization ---
print("\n--- Running Final Verification Simulation ---")

# 1. Use extended time to ensure full pulse transmission
# Previous time was 30*WL/LIGHT_SPEED. Let's use 60*WL/LIGHT_SPEED.
time_final = np.arange(0, 60*WL/LIGHT_SPEED, DT)
signal_final = ramped_cosine(time_final, 1, LIGHT_SPEED/WL, ramp_duration=6*WL/LIGHT_SPEED, t_max=time_final[-1])

# 2. Update Sources and Monitors
src_fwd = ModeSource(grid, center=(2*µm, H/2), width=WG_W*4, wavelength=WL, pol="tm", signal=signal_final, direction="+x")
if src_fwd._jz_profile is None: src_fwd.initialize(grid.permittivity, DX)

monitor_input = Monitor(design=grid, start=(2.5*µm, H/2-WG_W*2), end=(2.5*µm, H/2+WG_W*2), accumulate_power=True)
monitor_output = Monitor(design=grid, start=(W/2-WG_W*2, H-2*µm), end=(W/2+WG_W*2, H-2*µm), accumulate_power=True)

# 3. Run Simulation
sim_final = Simulation(grid, [src_fwd, monitor_input, monitor_output], 
                       [PML(edges='all', thickness=1*µm)], time=time_final, resolution=DX)

print("Running final simulation with full field capture...")
results_final = sim_final.run(save_fields=['Ez', 'Hx', 'Hy'], field_subsample=1)

# 4. Calculate Transmission
input_E = np.sum(monitor_input.power_history) * DT
output_E = np.sum(monitor_output.power_history) * DT
trans_final = (output_E / input_E * 100.0) if input_E > 0 else 0.0
print(f"Final Verified Transmission: {trans_final:.1f}%")

# 5. Calculate Energy Flow (Time-Integrated Poynting Vector)
print("Calculating energy flow...")
Ez_t = np.array(results_final['fields']['Ez'])
Hx_t = np.array(results_final['fields']['Hx'])
Hy_t = np.array(results_final['fields']['Hy'])

# Handle grid staggering by cropping to common size
# Ez(i,j), Hx(i, j+0.5), Hy(i+0.5, j)
min_x = min(Ez_t.shape[1], Hx_t.shape[1], Hy_t.shape[1])
min_y = min(Ez_t.shape[2], Hx_t.shape[2], Hy_t.shape[2])

Ez_c = Ez_t[:, :min_x, :min_y]
Hx_c = Hx_t[:, :min_x, :min_y]
Hy_c = Hy_t[:, :min_x, :min_y]

# Poynting Vector S = E x H
# For TM (Ez, Hx, Hy): Sx = -Ez * Hy, Sy = Ez * Hx
Sx_t = -Ez_c * Hy_c
Sy_t = Ez_c * Hx_c

# Integrate magnitude of Poynting vector over time: Integral(|S|) dt
# This shows the total energy density flow through each point
S_mag_t = np.sqrt(Sx_t**2 + Sy_t**2)
energy_flow = np.sum(S_mag_t, axis=0) * DT

# 6. Plot
plt.figure(figsize=(10, 8))
# Plot Structure (Permittivity)
# Crop permittivity to match field size
perm_c = grid.permittivity[:min_x, :min_y]

plt.imshow(perm_c.T, cmap='gray', origin='lower', alpha=0.2)
plt.contour(perm_c.T, levels=[(N_CORE**2 + N_CLAD**2)/2], colors='white', linewidths=0.5, origin='lower')

# Plot Energy Flow
# Use 'hot' or 'inferno' for energy
im = plt.imshow(energy_flow.T, cmap='inferno', origin='lower', alpha=0.9, interpolation='bicubic')
plt.colorbar(im, label=r'Time-Integrated Energy Flow $\int |\mathbf{S}| dt$')
plt.title(f'Final Energy Flow Map (T = {trans_final:.1f}%)')
plt.xlabel('x (grid cells)')
plt.ylabel('y (grid cells)')
plt.tight_layout()
plt.savefig('final_energy_flow.png', dpi=150)
print("Energy flow map saved to final_energy_flow.png")
