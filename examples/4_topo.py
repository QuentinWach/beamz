import numpy as np
import matplotlib.pyplot as plt
from beamz import *
from beamz.optimization.topology import TopologyManager, compute_overlap_gradient, create_optimization_mask

# --- 1. Simulation Setup ---
W = H = 7*µm
WG_W = 0.55*µm
WL = 1.55*µm
N_CORE, N_CLAD = 2.25, 1.444 # Si3N4, SiO2
DX, DT = calc_optimal_fdtd_params(WL, 2.25, points_per_wavelength=20) # reduce to 9 for faster simulation
STEPS = 50 # reduce to 40 for faster optimization
MAT_PENALTY = 0.3      # Target core material fraction (0.0 to 1.0)
PENALTY_STRENGTH = 1 # Scaling factor for the penalty gradient

# Design & Materials
design = Design(width=W, height=H, material=Material(permittivity=N_CLAD**2))
design += Rectangle(position=(0, H/2-WG_W/2), width=W/2, height=WG_W, material=Material(permittivity=N_CORE**2))
design += Rectangle(position=(W/2-WG_W/2, 0), width=WG_W, height=H/2, material=Material(permittivity=N_CORE**2))

# Optimization Region (added as placeholder)
opt_region = Rectangle(position=(W/2-1.5*µm, H/2-1.5*µm), width=3*µm, height=3*µm, material=Material(permittivity=N_CORE**2))
design += opt_region

# design.show()

# Sources
time = np.arange(0, 15*WL/LIGHT_SPEED, DT)
signal = ramped_cosine(time, 1, LIGHT_SPEED/WL, ramp_duration=3.5*WL/LIGHT_SPEED, t_max=time[-1]/2)

from beamz.devices.sources.signals import plot_signal
plot_signal(signal, time, save_path='signal.png')
src_fwd = ModeSource(None, center=(1.0*µm, H/2), width=WG_W*4, wavelength=WL, pol="tm", signal=signal, direction="+x")
src_adj = ModeSource(None, center=(W/2, 1.0*µm), width=WG_W*4, wavelength=WL, pol="tm", signal=signal, direction="+y")

# --- 2. Optimization Manager ---
# Rasterize once to get grid and mask
grid = design.rasterize(DX)
mask = create_optimization_mask(grid, opt_region)

opt = TopologyManager(
    design=design,
    region_mask=mask,
    resolution=DX,
    learning_rate=0.015,
    filter_radius=0.3*µm,       # Physical units: Controls minimum feature size AND boundary smoothness
    eps_min=N_CLAD**2,
    eps_max=N_CORE**2,
    beta_schedule=(1.0, 20.0),
    filter_type="conic",         # Use conic filter for geometric constraints
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
    monitor_input_flux = Monitor(design=grid, start=(1.5*µm, H/2-WG_W*2), end=(1.5*µm, H/2+WG_W*2), 
                           accumulate_power=True, record_fields=False)
    
    # Output monitor at output waveguide (bottom)
    output_monitor_fwd = Monitor(design=grid, start=(W/2-WG_W*2, 1.5*µm), end=(W/2+WG_W*2, 1.5*µm),
                                 accumulate_power=True, record_fields=False)
    
    # Run forward simulation with output monitor
    sim_fwd = Simulation(grid, [src_fwd, monitor_input_flux, output_monitor_fwd], 
                        [PML(edges='all', thickness=1*µm)], time=time, resolution=DX)
    
    print(f"[{step+1}/{STEPS}] Forward Sim...", end="\r")
    results = sim_fwd.run(save_fields=['Ez'], field_subsample=2)
    
    # Extract field history and ensure NumPy arrays
    fwd_ez_history = [np.array(field) for field in results['fields']['Ez']] if results and 'fields' in results else []
    
    # Calculate transmission normalizing by measured input flux
    # Input flux includes forward wave + reflection. 
    # For high transmission structures, reflection is low, so this is a good approximation of injected power.
    measured_input_energy = np.sum(monitor_input_flux.power_history) * DT
    measured_output_energy = np.sum(output_monitor_fwd.power_history) * DT
    
    # Avoid division by zero
    if measured_input_energy <= 0: measured_input_energy = 1.0
    
    transmission_fwd = (np.abs(measured_output_energy) / np.abs(measured_input_energy) * 100.0)
    
    # Backward Simulation (with backward monitor at input location)
    src_adj.grid = grid
    
    # Backward source monitor (just downstream of source)
    monitor_back_flux = Monitor(design=grid, start=(W/2-WG_W*2, 1.5*µm), end=(W/2+WG_W*2, 1.5*µm),
                              accumulate_power=True, record_fields=False)
    
    # Backward monitor at original input location (left waveguide)
    backward_monitor = Monitor(design=grid, start=(1.5*µm, H/2-WG_W*2), end=(1.5*µm, H/2+WG_W*2),
                              accumulate_power=True, record_fields=False)
    
    sim_adj = Simulation(grid, [src_adj, monitor_back_flux, backward_monitor], 
                        [PML(edges='all', thickness=1*µm)], time=time, resolution=DX)
    
    adj_results = sim_adj.run(save_fields=['Ez'], field_subsample=2)
    adj_ez_history = [np.array(field) for field in adj_results['fields']['Ez']] if adj_results and 'fields' in adj_results else []
    
    # Calculate backward transmission normalizing by measured input flux
    measured_input_energy_back = np.sum(monitor_back_flux.power_history) * DT
    if measured_input_energy_back <= 0: measured_input_energy_back = 1.0
    
    output_energy_back = np.sum(backward_monitor.power_history) * DT
    transmission_back = (np.abs(output_energy_back) / np.abs(measured_input_energy_back) * 100.0)
    
    # Average bidirectional transmission
    transmission_pct = (transmission_fwd + transmission_back) / 2.0
    
    # For objective, use averaged transmission percentage
    obj_val = transmission_pct
    
    opt.objective_history.append(obj_val)
    transmission_history.append(transmission_pct)
            
    # Compute Gradient (overlap of fwd and adj fields)
    grad_eps = compute_overlap_gradient(fwd_ez_history, adj_ez_history)

    # Ensure grad_eps is a NumPy array (not JAX array)
    grad_eps = np.array(grad_eps)

    # Measure Material Usage (Relative core material amount)
    # phys_density is 0 (cladding) to 1 (core)
    current_density = np.mean(phys_density[mask])

    # Quadratic Penalty: Strength * (current - target)^2
    # We want to maximize Obj, so we subtract penalty.
    # The gradient w.r.t. density is roughly proportional to (current - target).
    # We apply this uniform gradient correction to all pixels in the mask.

    # Gradient contribution: push density towards target
    # If current > target, we want to decrease density -> negative gradient contribution
    # If current < target, we want to increase density -> positive gradient contribution
    # grad_correction = -Strength * (current - target)

    grad_penalty = PENALTY_STRENGTH * (current_density - MAT_PENALTY)
    grad_eps[mask] -= grad_penalty

    # Total Objective for display (Transmission - Penalty term)
    # Scaled for readability
    penalty_val = PENALTY_STRENGTH * 0.5 * (current_density - MAT_PENALTY)**2
    total_obj = obj_val - penalty_val
    
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
# We now perform a frequency sweep to verify broadband performance
print("\n--- Running Final Frequency Sweep (1500-1600 nm) ---")

wavelengths = np.linspace(1.2*µm, 1.8*µm, 12)
sweep_transmission = []

# Use extended time to ensure full pulse transmission for all runs
time_sweep = np.arange(0, 15*WL/LIGHT_SPEED, DT)

for i, wl_val in enumerate(wavelengths):
    print(f"Simulating Wavelength: {wl_val/µm:.3f} µm...", end="\r")
    
    # Create signal for this specific wavelength
    signal_sweep = ramped_cosine(time, 1, LIGHT_SPEED/WL, ramp_duration=3.5*WL/LIGHT_SPEED, t_max=time[-1]/2)
    
    # Create source
    src_sweep = ModeSource(grid, center=(1.0*µm, H/2), width=WG_W*4, wavelength=wl_val, pol="tm", signal=signal_sweep, direction="+x")
    # Force re-initialization of mode profile for new wavelength
    src_sweep._jz_profile = None 
    src_sweep.initialize(grid.permittivity, DX)
    
    # Monitors
    mon_in = Monitor(design=grid, start=(1.5*µm, H/2-WG_W*2), end=(1.5*µm, H/2+WG_W*2), accumulate_power=True)
    mon_out = Monitor(design=grid, start=(W/2-WG_W*2, 1.5*µm), end=(W/2+WG_W*2, 1.5*µm), accumulate_power=True)
    
    # Simulation
    sim_sweep = Simulation(grid, [src_sweep, mon_in, mon_out], 
                           [PML(edges='all', thickness=1*µm)], time=time_sweep, resolution=DX)
    
    # Run (no field saving needed for sweep, faster)
    sim_sweep.run(save_fields=[], field_subsample=10)
    
    # Calculate Transmission
    in_E = np.sum(mon_in.power_history) * DT
    out_E = np.sum(mon_out.power_history) * DT
    trans = (np.abs(out_E) / np.abs(in_E) * 100.0) if np.abs(in_E) > 0 else 0.0
    sweep_transmission.append(trans)

print(f"\nSweep Complete.")

# Plot Frequency Sweep
plt.figure(figsize=(10, 6))
plt.plot(wavelengths/µm, sweep_transmission, 'r-o', linewidth=2)
plt.xlabel('Wavelength (µm)', fontsize=12)
plt.ylabel('Transmission (%)', fontsize=12)
plt.title('Transmission Spectrum', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('transmission_spectrum.png', dpi=150)
print(f"Spectrum plot saved to transmission_spectrum.png")

# --- 5. Final Visualization (Center Wavelength) ---
# Re-run simulation at center wavelength (1.55) to generate field plot
print("\n--- Generating Final Field Plot (1.55 µm) ---")
signal_final = ramped_cosine(time, 1, LIGHT_SPEED/WL, ramp_duration=3.5*WL/LIGHT_SPEED, t_max=time[-1]/2)
src_final = ModeSource(grid, center=(1.0*µm, H/2), width=WG_W*4, wavelength=WL, pol="tm", signal=signal_final, direction="+x")
src_final.initialize(grid.permittivity, DX)

mon_in_final = Monitor(design=grid, start=(1.5*µm, H/2-WG_W*2), end=(1.5*µm, H/2+WG_W*2), accumulate_power=True)
mon_out_final = Monitor(design=grid, start=(W/2-WG_W*2, 1.5*µm), end=(W/2+WG_W*2, 1.5*µm), accumulate_power=True)

sim_final = Simulation(grid, [src_final, mon_in_final, mon_out_final], [PML(edges='all', thickness=1*µm)], time=time_sweep, resolution=DX)
results_final = sim_final.run(save_fields=['Ez', 'Hx', 'Hy'], field_subsample=1)

# Calculate final transmission for title
in_E = np.sum(mon_in_final.power_history) * DT
out_E = np.sum(mon_out_final.power_history) * DT
trans_final = (np.abs(out_E) / np.abs(in_E) * 100.0) if np.abs(in_E) > 0 else 0.0

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
plt.colorbar(im, label=r'Time-Integrated Energy Flow $\int |\mathbf{S}| dt$')
plt.title(f'Final Energy Flow Map (1.55 µm)')
plt.xlabel('x (grid cells)')
plt.ylabel('y (grid cells)')
plt.tight_layout()
plt.savefig('final_energy_flow.png', dpi=150)
print("Energy flow map saved to final_energy_flow.png")