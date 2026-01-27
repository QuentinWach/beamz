"""
Wavelength Division Multiplexer (WDM) using Topology Optimization

This example demonstrates inverse design of a wavelength-selective device that routes:
- λ₁ = 1550 nm → Output port 1 (upward)
- λ₂ = 1600 nm → Output port 2 (downward)

Uses the adjoint method with multi-wavelength optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from beamz import *
from beamz.optimization.topology import TopologyManager, compute_overlap_gradient, create_optimization_mask

# ============================================================================
# 1. SIMULATION SETUP
# ============================================================================
print("="*70)
print("Wavelength Division Multiplexer - Topology Optimization")
print("="*70)

# Domain and waveguide parameters
W = H = 12*µm                    # Domain size (increased for larger optimization region)
WG_W = 0.5*µm                    # Waveguide width (single-mode)
WL1, WL2 = 1.55*µm, 1.60*µm     # Target wavelengths
N_CORE, N_CLAD = 2.25, 1.444     # Refractive indices (Si₃N₄/SiO₂)

# Optimization parameters
STEPS = 25                       # Number of optimization iterations (increased for convergence)
CROSSTALK_PENALTY = 0.3          # Weight for crosstalk penalty

# Calculate optimal FDTD parameters using λ₁ as reference
DX, DT = calc_optimal_fdtd_params(WL1, N_CORE, points_per_wavelength=9)
print(f"Resolution: {DX/nm:.2f} nm | Timestep: {DT*1e15:.3f} fs")

# ============================================================================
# 2. DESIGN GEOMETRY
# ============================================================================
print("\n--- Creating Design Geometry ---")

# Background (cladding)
design = Design(width=W, height=H, material=Material(permittivity=N_CLAD**2))

# Input waveguide (horizontal, entering from left)
input_wg = Rectangle(
    position=(0, H/2 - WG_W/2),
    width=W/2,
    height=WG_W,
    material=Material(permittivity=N_CORE**2)
)
design += input_wg

# Output waveguide 1 (vertical, exiting upward)
output_wg1 = Rectangle(
    position=(W/2 - WG_W/2, H/2),
    width=WG_W,
    height=H/2,
    material=Material(permittivity=N_CORE**2)
)
design += output_wg1

# Output waveguide 2 (vertical, exiting downward)
output_wg2 = Rectangle(
    position=(W/2 - WG_W/2, 0),
    width=WG_W,
    height=H/2,
    material=Material(permittivity=N_CORE**2)
)
design += output_wg2

# Optimization region (6µm × 6µm at the junction - larger for better wavelength separation)
opt_region = Rectangle(
    position=(W/2 - 3*µm, H/2 - 3*µm),
    width=6*µm,
    height=6*µm,
    material=Material(permittivity=N_CORE**2)
)
design += opt_region

print(f"Domain: {W/µm:.1f} µm × {H/µm:.1f} µm")
print(f"Waveguide width: {WG_W/µm:.2f} µm")
print(f"Optimization region: 6 µm × 6 µm")

# ============================================================================
# 3. SOURCES AND TIME ARRAY
# ============================================================================
print("\n--- Creating Sources ---")

# Time array (30 wavelengths for steady-state and gradient accumulation)
time = np.arange(0, 30*WL1/LIGHT_SPEED, DT)
print(f"Simulation time: {time[-1]*1e12:.1f} ps ({len(time)} steps)")

# Temporal signals
signal_wl1 = ramped_cosine(time, 1.0, LIGHT_SPEED/WL1, ramp_duration=3*WL1/LIGHT_SPEED, t_max=time[-1])
signal_wl2 = ramped_cosine(time, 1.0, LIGHT_SPEED/WL2, ramp_duration=3*WL2/LIGHT_SPEED, t_max=time[-1])

# Forward sources (at input waveguide)
src_fwd_wl1 = ModeSource(
    None,
    center=(1.0*µm, H/2),
    width=WG_W*4,
    wavelength=WL1,
    pol="tm",
    signal=signal_wl1,
    direction="+x"
)

src_fwd_wl2 = ModeSource(
    None,
    center=(1.0*µm, H/2),
    width=WG_W*4,
    wavelength=WL2,
    pol="tm",
    signal=signal_wl2,
    direction="+x"
)

# Adjoint sources (at output waveguides - positioned deeper for better mode stabilization)
src_adj_wl1 = ModeSource(
    None,
    center=(W/2, H - 2.5*µm),
    width=WG_W*4,
    wavelength=WL1,
    pol="tm",
    signal=signal_wl1,
    direction="-y"
)

src_adj_wl2 = ModeSource(
    None,
    center=(W/2, 2.5*µm),
    width=WG_W*4,
    wavelength=WL2,
    pol="tm",
    signal=signal_wl2,
    direction="+y"
)

# ============================================================================
# 4. TOPOLOGY OPTIMIZATION SETUP
# ============================================================================
print("\n--- Initializing Topology Optimization ---")

# Rasterize design to get grid
grid = design.rasterize(DX)
mask = create_optimization_mask(grid, opt_region)

print(f"Grid size: {grid.width} × {grid.height} pixels")
print(f"Optimization pixels: {np.sum(mask)}")

# Create topology manager
opt = TopologyManager(
    design=design,
    region_mask=mask,
    resolution=DX,
    learning_rate=0.05,              # Moderate increase for better convergence
    filter_radius=0.4*µm,            # Min feature size (maintains ~4 cells at 16 ppw)
    simple_smooth_radius=0.1*µm,     # Anti-pixelation
    eps_min=N_CLAD**2,
    eps_max=N_CORE**2,
    beta_schedule=(1.0, 12.0),       # Gentler binarization for dual-wavelength exploration
    filter_type='conic',
)

# Store background permittivity
base_eps = grid.permittivity.copy()

# Tracking arrays
transmission_wl1_to_out1 = []
transmission_wl2_to_out2 = []
crosstalk_wl1_to_out2 = []
crosstalk_wl2_to_out1 = []
objective_values = []

# ============================================================================
# 5. OPTIMIZATION LOOP
# ============================================================================
print("\n" + "="*70)
print("Starting Optimization")
print("="*70 + "\n")

for step in range(STEPS):
    # ------------------------------------------------------------------------
    # 5.1 Update Design
    # ------------------------------------------------------------------------
    beta, phys_density = opt.update_design(step, STEPS)

    # Update permittivity from physical density
    grid.permittivity[:] = base_eps
    grid.permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)

    # ------------------------------------------------------------------------
    # 5.2 Forward Simulations (2 wavelengths)
    # ------------------------------------------------------------------------

    # === WL1 Forward Simulation ===
    src_fwd_wl1.grid = grid

    # Input flux monitor (measures injected power)
    mon_in_wl1 = Monitor(
        design=grid,
        start=(1.5*µm, H/2 - WG_W*2),
        end=(1.5*µm, H/2 + WG_W*2),
        accumulate_power=True,
        record_fields=False
    )

    # Output monitors
    mon_out1_wl1 = Monitor(
        design=grid,
        start=(W/2 - WG_W*2, H - 1.5*µm),
        end=(W/2 + WG_W*2, H - 1.5*µm),
        accumulate_power=True,
        record_fields=False
    )

    mon_out2_wl1 = Monitor(
        design=grid,
        start=(W/2 - WG_W*2, 1.5*µm),
        end=(W/2 + WG_W*2, 1.5*µm),
        accumulate_power=True,
        record_fields=False
    )

    print(f"[{step+1}/{STEPS}] Running λ₁ forward sim...", end="\r")
    sim_fwd_wl1 = Simulation(
        grid,
        [src_fwd_wl1, mon_in_wl1, mon_out1_wl1, mon_out2_wl1],
        [PML(edges='all', thickness=1*µm)],
        time=time,
        resolution=DX
    )
    results_fwd_wl1 = sim_fwd_wl1.run(save_fields=['Ez'], field_subsample=2)
    fwd_ez_wl1 = results_fwd_wl1['fields']['Ez'] if results_fwd_wl1 and 'fields' in results_fwd_wl1 else []

    # Calculate transmissions
    E_in_wl1 = np.sum(mon_in_wl1.power_history) * DT
    E_out1_wl1 = np.sum(mon_out1_wl1.power_history) * DT
    E_out2_wl1 = np.sum(mon_out2_wl1.power_history) * DT

    if E_in_wl1 <= 0:
        E_in_wl1 = 1.0

    T_wl1_to_out1 = np.abs(E_out1_wl1) / np.abs(E_in_wl1)  # Should be high
    T_wl1_to_out2 = np.abs(E_out2_wl1) / np.abs(E_in_wl1)  # Should be low (crosstalk)

    # === WL2 Forward Simulation ===
    src_fwd_wl2.grid = grid

    mon_in_wl2 = Monitor(
        design=grid,
        start=(1.5*µm, H/2 - WG_W*2),
        end=(1.5*µm, H/2 + WG_W*2),
        accumulate_power=True,
        record_fields=False
    )

    mon_out1_wl2 = Monitor(
        design=grid,
        start=(W/2 - WG_W*2, H - 1.5*µm),
        end=(W/2 + WG_W*2, H - 1.5*µm),
        accumulate_power=True,
        record_fields=False
    )

    mon_out2_wl2 = Monitor(
        design=grid,
        start=(W/2 - WG_W*2, 1.5*µm),
        end=(W/2 + WG_W*2, 1.5*µm),
        accumulate_power=True,
        record_fields=False
    )

    print(f"[{step+1}/{STEPS}] Running λ₂ forward sim...", end="\r")
    sim_fwd_wl2 = Simulation(
        grid,
        [src_fwd_wl2, mon_in_wl2, mon_out1_wl2, mon_out2_wl2],
        [PML(edges='all', thickness=1*µm)],
        time=time,
        resolution=DX
    )
    results_fwd_wl2 = sim_fwd_wl2.run(save_fields=['Ez'], field_subsample=2)
    fwd_ez_wl2 = results_fwd_wl2['fields']['Ez'] if results_fwd_wl2 and 'fields' in results_fwd_wl2 else []

    # Calculate transmissions
    E_in_wl2 = np.sum(mon_in_wl2.power_history) * DT
    E_out1_wl2 = np.sum(mon_out1_wl2.power_history) * DT
    E_out2_wl2 = np.sum(mon_out2_wl2.power_history) * DT

    if E_in_wl2 <= 0:
        E_in_wl2 = 1.0

    T_wl2_to_out1 = np.abs(E_out1_wl2) / np.abs(E_in_wl2)  # Should be low (crosstalk)
    T_wl2_to_out2 = np.abs(E_out2_wl2) / np.abs(E_in_wl2)  # Should be high

    # ------------------------------------------------------------------------
    # 5.3 Adjoint Simulations (2 wavelengths)
    # ------------------------------------------------------------------------

    # === WL1 Adjoint Simulation (source at output 1, top) ===
    src_adj_wl1.grid = grid

    print(f"[{step+1}/{STEPS}] Running λ₁ adjoint sim...", end="\r")
    sim_adj_wl1 = Simulation(
        grid,
        [src_adj_wl1],
        [PML(edges='all', thickness=1*µm)],
        time=time,
        resolution=DX
    )
    results_adj_wl1 = sim_adj_wl1.run(save_fields=['Ez'], field_subsample=2)
    adj_ez_wl1 = results_adj_wl1['fields']['Ez'] if results_adj_wl1 and 'fields' in results_adj_wl1 else []

    # === WL2 Adjoint Simulation (source at output 2, bottom) ===
    src_adj_wl2.grid = grid

    print(f"[{step+1}/{STEPS}] Running λ₂ adjoint sim...", end="\r")
    sim_adj_wl2 = Simulation(
        grid,
        [src_adj_wl2],
        [PML(edges='all', thickness=1*µm)],
        time=time,
        resolution=DX
    )
    results_adj_wl2 = sim_adj_wl2.run(save_fields=['Ez'], field_subsample=2)
    adj_ez_wl2 = results_adj_wl2['fields']['Ez'] if results_adj_wl2 and 'fields' in results_adj_wl2 else []

    # ------------------------------------------------------------------------
    # 5.4 Objective Function
    # ------------------------------------------------------------------------

    # Transmission to correct ports (maximize)
    obj_correct = (T_wl1_to_out1 + T_wl2_to_out2) / 2.0

    # Crosstalk to wrong ports (minimize)
    obj_crosstalk = (T_wl1_to_out2 + T_wl2_to_out1) / 2.0

    # Total objective
    obj_val = obj_correct - CROSSTALK_PENALTY * obj_crosstalk

    # Track history
    opt.objective_history.append(obj_val * 100)  # Convert to percentage
    transmission_wl1_to_out1.append(T_wl1_to_out1 * 100)
    transmission_wl2_to_out2.append(T_wl2_to_out2 * 100)
    crosstalk_wl1_to_out2.append(T_wl1_to_out2 * 100)
    crosstalk_wl2_to_out1.append(T_wl2_to_out1 * 100)
    objective_values.append(obj_val * 100)

    # ------------------------------------------------------------------------
    # 5.5 Gradient Computation
    # ------------------------------------------------------------------------

    # Compute overlap gradients for each wavelength
    grad_wl1 = compute_overlap_gradient(fwd_ez_wl1, adj_ez_wl1)
    grad_wl2 = compute_overlap_gradient(fwd_ez_wl2, adj_ez_wl2)

    # Combine gradients with adaptive weighting (focus on worse-performing wavelength)
    w1 = max(0.1, 1.0 - T_wl1_to_out1)  # Higher weight when transmission is low
    w2 = max(0.1, 1.0 - T_wl2_to_out2)
    grad_eps = (w1 * grad_wl1 + w2 * grad_wl2) / (w1 + w2)

    # ------------------------------------------------------------------------
    # 5.6 Apply Gradient
    # ------------------------------------------------------------------------

    max_update = opt.apply_gradient(grad_eps, beta)

    # ------------------------------------------------------------------------
    # 5.7 Progress Logging
    # ------------------------------------------------------------------------

    mat_frac = np.mean(phys_density[mask])

    print(f"Step {step+1:2d}/{STEPS}: "
          f"Obj={obj_val*100:5.1f}% | "
          f"λ₁→out₁={T_wl1_to_out1*100:5.1f}% λ₂→out₂={T_wl2_to_out2*100:5.1f}% | "
          f"XT={obj_crosstalk*100:5.1f}% | "
          f"Mat={mat_frac:.2f} | "
          f"Up={max_update:.2e}")

    # ------------------------------------------------------------------------
    # 5.8 Visualization (every 5 steps)
    # ------------------------------------------------------------------------

    if step % 5 == 0:
        plt.imsave(f"wdm_opt_{step:03d}.png", grid.permittivity.T, cmap='gray', origin='lower')

# ============================================================================
# 6. POST-OPTIMIZATION ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("Optimization Complete - Running Analysis")
print("="*70)

# ----------------------------------------------------------------------------
# 6.1 Training Curves
# ----------------------------------------------------------------------------
print("\n--- Generating Training Curves ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
steps_arr = np.arange(1, STEPS + 1)

# Objective value
axes[0, 0].plot(steps_arr, objective_values, 'k-', linewidth=2, marker='o', markersize=4)
axes[0, 0].set_xlabel('Optimization Step', fontsize=11)
axes[0, 0].set_ylabel('Objective Value (%)', fontsize=11)
axes[0, 0].set_title('Objective (Transmission - Crosstalk Penalty)', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# λ₁ transmission to output 1
axes[0, 1].plot(steps_arr, transmission_wl1_to_out1, 'b-', linewidth=2, marker='o', markersize=4)
axes[0, 1].set_xlabel('Optimization Step', fontsize=11)
axes[0, 1].set_ylabel('Transmission (%)', fontsize=11)
axes[0, 1].set_title('λ₁ (1550 nm) → Output 1 (Top)', fontsize=12, color='blue')
axes[0, 1].set_ylim(0, 100)
axes[0, 1].grid(True, alpha=0.3)

# λ₂ transmission to output 2
axes[1, 0].plot(steps_arr, transmission_wl2_to_out2, 'r-', linewidth=2, marker='o', markersize=4)
axes[1, 0].set_xlabel('Optimization Step', fontsize=11)
axes[1, 0].set_ylabel('Transmission (%)', fontsize=11)
axes[1, 0].set_title('λ₂ (1600 nm) → Output 2 (Bottom)', fontsize=12, color='red')
axes[1, 0].set_ylim(0, 100)
axes[1, 0].grid(True, alpha=0.3)

# Average crosstalk
avg_crosstalk = [(c1 + c2) / 2 for c1, c2 in zip(crosstalk_wl1_to_out2, crosstalk_wl2_to_out1)]
axes[1, 1].plot(steps_arr, avg_crosstalk, 'orange', linewidth=2, marker='o', markersize=4)
axes[1, 1].set_xlabel('Optimization Step', fontsize=11)
axes[1, 1].set_ylabel('Crosstalk (%)', fontsize=11)
axes[1, 1].set_title('Average Crosstalk', fontsize=12)
axes[1, 1].set_ylim(0, 100)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wdm_training_curves.png', dpi=150, bbox_inches='tight')
print("Saved: wdm_training_curves.png")
plt.close()

# ----------------------------------------------------------------------------
# 6.2 Frequency Sweep (Broadband Performance)
# ----------------------------------------------------------------------------
print("\n--- Running Frequency Sweep ---")

wavelengths_sweep = np.linspace(1.45*µm, 1.65*µm, 15)
trans_to_out1 = []
trans_to_out2 = []

# Extended time for sweep
time_sweep = np.arange(0, 10*WL1/LIGHT_SPEED, DT)

for i, wl in enumerate(wavelengths_sweep):
    print(f"  Wavelength: {wl/µm:.3f} µm ({i+1}/{len(wavelengths_sweep)})", end="\r")

    # Create signal
    signal = ramped_cosine(time_sweep, 1.0, LIGHT_SPEED/wl, ramp_duration=3*wl/LIGHT_SPEED, t_max=time_sweep[-1])

    # Create source
    src = ModeSource(grid, center=(1.0*µm, H/2), width=WG_W*4, wavelength=wl, pol="tm", signal=signal, direction="+x")
    src._jz_profile = None  # Force reinitialization
    src.initialize(grid.permittivity, DX)

    # Monitors
    mon_in = Monitor(design=grid, start=(1.5*µm, H/2-WG_W*2), end=(1.5*µm, H/2+WG_W*2), accumulate_power=True)
    mon_out1 = Monitor(design=grid, start=(W/2-WG_W*2, H-1.5*µm), end=(W/2+WG_W*2, H-1.5*µm), accumulate_power=True)
    mon_out2 = Monitor(design=grid, start=(W/2-WG_W*2, 1.5*µm), end=(W/2+WG_W*2, 1.5*µm), accumulate_power=True)

    # Run simulation
    sim = Simulation(grid, [src, mon_in, mon_out1, mon_out2], [PML(edges='all', thickness=1*µm)],
                     time=time_sweep, resolution=DX)
    sim.run(save_fields=[], field_subsample=10)

    # Calculate transmissions
    E_in = np.sum(mon_in.power_history) * DT
    E_out1 = np.sum(mon_out1.power_history) * DT
    E_out2 = np.sum(mon_out2.power_history) * DT

    if E_in > 0:
        trans_to_out1.append(np.abs(E_out1) / np.abs(E_in) * 100)
        trans_to_out2.append(np.abs(E_out2) / np.abs(E_in) * 100)
    else:
        trans_to_out1.append(0)
        trans_to_out2.append(0)

print("\nSweep complete.                                ")

# Plot transmission spectrum
plt.figure(figsize=(12, 6))
plt.plot(wavelengths_sweep/µm, trans_to_out1, 'b-o', linewidth=2, label='Output 1 (Top)', markersize=6)
plt.plot(wavelengths_sweep/µm, trans_to_out2, 'r-s', linewidth=2, label='Output 2 (Bottom)', markersize=6)

# Highlight target wavelengths
plt.axvline(WL1/µm, color='blue', linestyle='--', alpha=0.5, label=f'λ₁ = {WL1/µm:.2f} µm')
plt.axvline(WL2/µm, color='red', linestyle='--', alpha=0.5, label=f'λ₂ = {WL2/µm:.2f} µm')

plt.xlabel('Wavelength (µm)', fontsize=12)
plt.ylabel('Transmission (%)', fontsize=12)
plt.title('WDM Transmission Spectrum', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('wdm_transmission_spectrum.png', dpi=150, bbox_inches='tight')
print("Saved: wdm_transmission_spectrum.png")
plt.close()

# ----------------------------------------------------------------------------
# 6.3 Final Field Visualization
# ----------------------------------------------------------------------------
print("\n--- Generating Final Field Visualization ---")

# Re-run simulations at target wavelengths with full field recording
print("  Running λ₁ simulation...")
signal_wl1_final = ramped_cosine(time_sweep, 1.0, LIGHT_SPEED/WL1, ramp_duration=6*WL1/LIGHT_SPEED, t_max=time_sweep[-1])
src_wl1_final = ModeSource(grid, center=(1.0*µm, H/2), width=WG_W*4, wavelength=WL1, pol="tm",
                           signal=signal_wl1_final, direction="+x")
src_wl1_final.initialize(grid.permittivity, DX)

sim_wl1_final = Simulation(grid, [src_wl1_final], [PML(edges='all', thickness=1*µm)],
                           time=time_sweep, resolution=DX)
results_wl1_final = sim_wl1_final.run(save_fields=['Ez', 'Hx', 'Hy'], field_subsample=1)

print("  Running λ₂ simulation...")
signal_wl2_final = ramped_cosine(time_sweep, 1.0, LIGHT_SPEED/WL2, ramp_duration=6*WL2/LIGHT_SPEED, t_max=time_sweep[-1])
src_wl2_final = ModeSource(grid, center=(1.0*µm, H/2), width=WG_W*4, wavelength=WL2, pol="tm",
                           signal=signal_wl2_final, direction="+x")
src_wl2_final.initialize(grid.permittivity, DX)

sim_wl2_final = Simulation(grid, [src_wl2_final], [PML(edges='all', thickness=1*µm)],
                           time=time_sweep, resolution=DX)
results_wl2_final = sim_wl2_final.run(save_fields=['Ez', 'Hx', 'Hy'], field_subsample=1)

print("  Calculating energy flow...")

# Calculate time-integrated Poynting vector for WL1
Ez_wl1 = np.array(results_wl1_final['fields']['Ez'])
Hx_wl1 = np.array(results_wl1_final['fields']['Hx'])
Hy_wl1 = np.array(results_wl1_final['fields']['Hy'])

min_x1 = min(Ez_wl1.shape[1], Hx_wl1.shape[1], Hy_wl1.shape[1])
min_y1 = min(Ez_wl1.shape[2], Hx_wl1.shape[2], Hy_wl1.shape[2])

Ez_wl1 = Ez_wl1[:, :min_x1, :min_y1]
Hx_wl1 = Hx_wl1[:, :min_x1, :min_y1]
Hy_wl1 = Hy_wl1[:, :min_x1, :min_y1]

Sx_wl1 = -Ez_wl1 * Hy_wl1
Sy_wl1 = Ez_wl1 * Hx_wl1
S_mag_wl1 = np.sqrt(Sx_wl1**2 + Sy_wl1**2)
energy_flow_wl1 = np.sum(S_mag_wl1, axis=0) * DT

# Calculate time-integrated Poynting vector for WL2
Ez_wl2 = np.array(results_wl2_final['fields']['Ez'])
Hx_wl2 = np.array(results_wl2_final['fields']['Hx'])
Hy_wl2 = np.array(results_wl2_final['fields']['Hy'])

min_x2 = min(Ez_wl2.shape[1], Hx_wl2.shape[1], Hy_wl2.shape[1])
min_y2 = min(Ez_wl2.shape[2], Hx_wl2.shape[2], Hy_wl2.shape[2])

Ez_wl2 = Ez_wl2[:, :min_x2, :min_y2]
Hx_wl2 = Hx_wl2[:, :min_x2, :min_y2]
Hy_wl2 = Hy_wl2[:, :min_x2, :min_y2]

Sx_wl2 = -Ez_wl2 * Hy_wl2
Sy_wl2 = Ez_wl2 * Hx_wl2
S_mag_wl2 = np.sqrt(Sx_wl2**2 + Sy_wl2**2)
energy_flow_wl2 = np.sum(S_mag_wl2, axis=0) * DT

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# WL1 plot (routes to top)
perm_wl1 = grid.permittivity[:min_x1, :min_y1]
axes[0].imshow(perm_wl1.T, cmap='gray', origin='lower', alpha=0.2)
axes[0].contour(perm_wl1.T, levels=[(N_CORE**2 + N_CLAD**2)/2], colors='white', linewidths=0.8, origin='lower')
im1 = axes[0].imshow(energy_flow_wl1.T, cmap='inferno', origin='lower', alpha=0.9, interpolation='bicubic')
axes[0].set_title(f'λ₁ = {WL1/µm:.2f} µm → Output 1 (Top)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('x (grid cells)', fontsize=11)
axes[0].set_ylabel('y (grid cells)', fontsize=11)
plt.colorbar(im1, ax=axes[0], label=r'Energy Flow $\int |\mathbf{S}| dt$', fraction=0.046)

# WL2 plot (routes to bottom)
perm_wl2 = grid.permittivity[:min_x2, :min_y2]
axes[1].imshow(perm_wl2.T, cmap='gray', origin='lower', alpha=0.2)
axes[1].contour(perm_wl2.T, levels=[(N_CORE**2 + N_CLAD**2)/2], colors='white', linewidths=0.8, origin='lower')
im2 = axes[1].imshow(energy_flow_wl2.T, cmap='inferno', origin='lower', alpha=0.9, interpolation='bicubic')
axes[1].set_title(f'λ₂ = {WL2/µm:.2f} µm → Output 2 (Bottom)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('x (grid cells)', fontsize=11)
axes[1].set_ylabel('y (grid cells)', fontsize=11)
plt.colorbar(im2, ax=axes[1], label=r'Energy Flow $\int |\mathbf{S}| dt$', fraction=0.046)

plt.tight_layout()
plt.savefig('wdm_final_fields.png', dpi=150, bbox_inches='tight')
print("Saved: wdm_final_fields.png")
plt.close()

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("WDM Optimization Summary")
print("="*70)
print(f"Final Objective:        {objective_values[-1]:.1f}%")
print(f"λ₁ → Output 1:          {transmission_wl1_to_out1[-1]:.1f}%")
print(f"λ₂ → Output 2:          {transmission_wl2_to_out2[-1]:.1f}%")
print(f"λ₁ → Output 2 (XT):     {crosstalk_wl1_to_out2[-1]:.1f}%")
print(f"λ₂ → Output 1 (XT):     {crosstalk_wl2_to_out1[-1]:.1f}%")
print(f"Average Crosstalk:      {avg_crosstalk[-1]:.1f}%")
print("="*70)
print("\nGenerated files:")
print("  - wdm_opt_*.png (permittivity snapshots)")
print("  - wdm_training_curves.png")
print("  - wdm_transmission_spectrum.png")
print("  - wdm_final_fields.png")
print("\nOptimization complete!")
