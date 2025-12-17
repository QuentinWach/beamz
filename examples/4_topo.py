import numpy as np
import matplotlib.pyplot as plt
from beamz import *
from beamz.optimization.topology import TopologyManager, compute_overlap_gradient, create_optimization_mask

# --- 1. Simulation Setup ---
W = H = 7*µm
WG_W = 0.5*µm
WL = 1.55*µm
N_CORE, N_CLAD = 2.25, 1.444
DX, DT = calc_optimal_fdtd_params(WL, 2.25, points_per_wavelength=15)
STEPS = 50 # Increase this for full robust convergence (e.g. 200) for real runs
MAT_PENALTY = 0.1      # Target core material fraction (0.0 to 1.0)
PENALTY_STRENGTH = 200000 # Scaling factor for the penalty gradient

# Design & Materials
design = Design(width=W, height=H, material=Material(permittivity=N_CLAD**2))
design += Rectangle(position=(0, H/2-WG_W/2), width=W/2, height=WG_W, material=Material(permittivity=N_CORE**2))
design += Rectangle(position=(W/2-WG_W/2, 0), width=WG_W, height=H/2, material=Material(permittivity=N_CORE**2))

# Optimization Region
opt_region = Rectangle(position=(W/2-1.5*µm, H/2-1.5*µm), width=3*µm, height=3*µm, material=Material(permittivity=N_CORE**2))
design += opt_region

# Sources
time = np.arange(0, 30*WL/LIGHT_SPEED, DT)
signal = ramped_cosine(time, 1, LIGHT_SPEED/WL, ramp_duration=6*WL/LIGHT_SPEED, t_max=time[-1])
src_fwd = ModeSource(None, center=(1.0*µm, H/2), width=WG_W*4, wavelength=WL, pol="tm", signal=signal, direction="+x")
src_adj = ModeSource(None, center=(W/2, 1.0*µm), width=WG_W*4, wavelength=WL, pol="tm", signal=signal, direction="+y")

# --- 2. Optimization Manager ---
grid = design.rasterize(DX)
mask = create_optimization_mask(grid, opt_region)

# Initialize ROBUST Topology Manager with MMA
opt = TopologyManager(
    design=design,
    region_mask=mask,
    optimizer="MMA", # Use Method of Moving Asymptotes
    resolution=DX,
    filter_radius=0.04*µm,       # Physical units
    simple_smooth_radius=0.02*µm,
    eps_min=N_CLAD**2,
    eps_max=N_CORE**2,
    beta_schedule=(8.0, 32.0),   # Hammond's schedule (sharper start)
    filter_type='conic',
)

print(f"Starting Robust Topology Optimization ({STEPS} steps)...")
base_eps = grid.permittivity.copy() 

transmission_history = []

# --- 3. Robust Optimization Loop ---
# We will simulate 3 variants: 'nominal', 'eroded', 'dilated'
variants = ['nominal', 'eroded', 'dilated']

for step in range(STEPS):
    
    gradients = {}
    objectives = {}
    
    # 3.1 Simulate each variant
    for variant in variants:
        # Update Design for this variant
        beta, phys_density = opt.update_design(step, STEPS, variant=variant)
        
        # Mix Density into Permittivity
        grid.permittivity[:] = base_eps
        grid.permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)
        
        # --- Forward Simulation ---
        src_fwd.grid = grid
        monitor_input_flux = Monitor(design=grid, start=(1.5*µm, H/2-WG_W*2), end=(1.5*µm, H/2+WG_W*2), 
                               accumulate_power=True, record_fields=False)
        output_monitor_fwd = Monitor(design=grid, start=(W/2-WG_W*2, 1.5*µm), end=(W/2+WG_W*2, 1.5*µm),
                                     accumulate_power=True, record_fields=False)
        
        sim_fwd = Simulation(grid, [src_fwd, monitor_input_flux, output_monitor_fwd], 
                            [PML(edges='all', thickness=1*µm)], time=time, resolution=DX)
        
        # Only print progress for nominal to avoid clutter
        if variant == 'nominal':
            print(f"[{step+1}/{STEPS}] Simulating Robust Ensemble...", end="\r")
            
        results = sim_fwd.run(save_fields=['Ez'], field_subsample=2)
        fwd_ez_history = results['fields']['Ez'] if results and 'fields' in results else []
        
        measured_input_energy = np.sum(monitor_input_flux.power_history) * DT
        if measured_input_energy <= 0: measured_input_energy = 1.0
        measured_output_energy = np.sum(output_monitor_fwd.power_history) * DT
        transmission_fwd = (np.abs(measured_output_energy) / np.abs(measured_input_energy) * 100.0)
        
        # --- Backward Simulation ---
        src_adj.grid = grid
        monitor_back_flux = Monitor(design=grid, start=(W/2-WG_W*2, 1.5*µm), end=(W/2+WG_W*2, 1.5*µm),
                                  accumulate_power=True, record_fields=False)
        backward_monitor = Monitor(design=grid, start=(1.5*µm, H/2-WG_W*2), end=(1.5*µm, H/2+WG_W*2),
                                  accumulate_power=True, record_fields=False)
        
        sim_adj = Simulation(grid, [src_adj, monitor_back_flux, backward_monitor], 
                            [PML(edges='all', thickness=1*µm)], time=time, resolution=DX)
        
        adj_results = sim_adj.run(save_fields=['Ez'], field_subsample=2)
        adj_ez_history = adj_results['fields']['Ez'] if adj_results and 'fields' in adj_results else []
        
        measured_input_energy_back = np.sum(monitor_back_flux.power_history) * DT
        if measured_input_energy_back <= 0: measured_input_energy_back = 1.0
        output_energy_back = np.sum(backward_monitor.power_history) * DT
        transmission_back = (np.abs(output_energy_back) / np.abs(measured_input_energy_back) * 100.0)
        
        # Objective: Maximize Transmission -> Minimize -Transmission
        # We store positive transmission for display, but gradient is for maximization
        transmission_pct = (transmission_fwd + transmission_back) / 2.0
        objectives[variant] = transmission_pct # We will flip sign in apply_gradient
        
        # Compute Gradient (overlap of fwd and adj fields)
        # Note: This is d(Overlap)/dEps. 
        # Overlap roughly proportional to Transmission.
        # We assume grad_eps is the gradient we want to follow (ascent).
        gradients[variant] = compute_overlap_gradient(fwd_ez_history, adj_ez_history)
        
        # Explicit Material Penalty per variant (optional, Hammond uses constraints)
        # We can integrate it into the objective or constraints.
        # For now, let's keep the penalty term simple or rely on MMA constraints later.
        # Adding penalty to gradient here for now to maintain previous logic
        current_density = np.mean(phys_density[mask])
        grad_penalty = PENALTY_STRENGTH * (current_density - MAT_PENALTY)
        # Scale penalty to match gradient magnitude roughly?
        # Or just subtract it.
        # Note: apply_gradient_robust expects 'gradients' to be d(Transmission)/dEps.
        # It will flip them to minimize -Trans.
        # If we want to penalize volume, we should subtract penalty from Transmission (Objective = Trans - Penalty).
        # So we should ADD penalty gradient to the ascent gradient? 
        # d(Obj)/dEps = d(Trans)/dEps - d(Penalty)/dEps.
        # grad_eps calculated is d(Trans)/dEps.
        # So yes, subtract d(Penalty)/dEps.
        gradients[variant][mask] -= grad_penalty
        
        # Adjust objective value for tracking
        penalty_val = PENALTY_STRENGTH * 0.5 * (current_density - MAT_PENALTY)**2
        objectives[variant] -= penalty_val

    # 3.2 Update Design using Robust Minimax (MMA)
    # Pass gradients and objectives to manager
    max_update = opt.apply_gradient_robust(gradients, objectives, beta)
    
    # Logging
    nom_trans = objectives['nominal'] # This includes penalty subtraction
    # Reconstruct raw transmission for display
    # (Approximate, as we modified objective above)
    print(f" Step {step+1}: Obj={nom_trans:.2f} (Ero={objectives['eroded']:.1f} Nom={objectives['nominal']:.1f} Dil={objectives['dilated']:.1f}) | MaxUp={max_update:.2e}", end="\r")
    
    transmission_history.append(objectives['nominal'])
    
    # Viz
    if step % 5 == 0:
        plt.imsave(f"topo_opt_{step:03d}.png", grid.permittivity.T, cmap='gray', origin='lower')

print(f"\nOptimization Complete.")

# Plot transmission vs step
plt.figure(figsize=(10, 6))
steps_arr = np.arange(1, len(transmission_history) + 1)
plt.plot(steps_arr, transmission_history, 'b-', linewidth=2, marker='o', markersize=4)
plt.xlabel('Optimization Step')
plt.ylabel('Robust Objective')
plt.title('Robust Optimization Convergence')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('transmission_vs_step.png', dpi=150)
print(f"Convergence plot saved.")
plt.close()

# --- 4. Final Verification (Nominal) ---
print("\n--- Running Final Verification Simulation (Nominal) ---")
# Update design to final nominal
_, phys_density = opt.update_design(STEPS, STEPS, variant='nominal')
grid.permittivity[:] = base_eps
grid.permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)

time_final = np.arange(0, 60*WL/LIGHT_SPEED, DT)
signal_final = ramped_cosine(time_final, 1, LIGHT_SPEED/WL, ramp_duration=6*WL/LIGHT_SPEED, t_max=time_final[-1])

src_fwd = ModeSource(grid, center=(1.0*µm, H/2), width=WG_W*4, wavelength=WL, pol="tm", signal=signal_final, direction="+x")
if src_fwd._jz_profile is None: src_fwd.initialize(grid.permittivity, DX)

monitor_input = Monitor(design=grid, start=(1.5*µm, H/2-WG_W*2), end=(1.5*µm, H/2+WG_W*2), accumulate_power=True)
monitor_output = Monitor(design=grid, start=(W/2-WG_W*2, 1.5*µm), end=(W/2+WG_W*2, 1.5*µm), accumulate_power=True)

sim_final = Simulation(grid, [src_fwd, monitor_input, monitor_output], 
                       [PML(edges='all', thickness=1*µm)], time=time_final, resolution=DX)

print("Running final simulation...")
results_final = sim_final.run(save_fields=['Ez', 'Hx', 'Hy'], field_subsample=1)

input_E = np.sum(monitor_input.power_history) * DT
output_E = np.sum(monitor_output.power_history) * DT
trans_final = (np.abs(output_E) / np.abs(input_E) * 100.0) if np.abs(input_E) > 0 else 0.0
print(f"Final Verified Transmission (Nominal): {trans_final:.1f}%")

# 5. Energy Flow Plot
print("Calculating energy flow...")
Ez_t = np.array(results_final['fields']['Ez'])
Hx_t = np.array(results_final['fields']['Hx'])
Hy_t = np.array(results_final['fields']['Hy'])

min_x = min(Ez_t.shape[1], Hx_t.shape[1], Hy_t.shape[1])
min_y = min(Ez_t.shape[2], Hx_t.shape[2], Hy_t.shape[2])

Ez_c = Ez_t[:, :min_x, :min_y]
Hx_c = Hx_t[:, :min_x, :min_y]
Hy_c = Hy_t[:, :min_x, :min_y]

Sx_t = -Ez_c * Hy_c
Sy_t = Ez_c * Hx_c
S_mag_t = np.sqrt(Sx_t**2 + Sy_t**2)
energy_flow = np.sum(S_mag_t, axis=0) * DT

plt.figure(figsize=(10, 8))
perm_c = grid.permittivity[:min_x, :min_y]
plt.imshow(perm_c.T, cmap='gray', origin='lower', alpha=0.2)
plt.contour(perm_c.T, levels=[(N_CORE**2 + N_CLAD**2)/2], colors='white', linewidths=0.5, origin='lower')
im = plt.imshow(energy_flow.T, cmap='inferno', origin='lower', alpha=0.9, interpolation='bicubic')
plt.colorbar(im, label=r'Time-Integrated Energy Flow')
plt.title(f'Final Energy Flow Map (T = {trans_final:.1f}%)')
plt.xlabel('x (grid cells)')
plt.ylabel('y (grid cells)')
plt.tight_layout()
plt.savefig('final_energy_flow.png', dpi=150)
print("Energy flow map saved.")
