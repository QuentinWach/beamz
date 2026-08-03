import matplotlib.pyplot as plt
import numpy as np

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    FieldRecorder,
    FluxMonitor,
    Material,
    ModeSource,
    ModeSpec,
    Rectangle,
    SampledSignal,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    µm,
)
from beamz.optimization.topology import (
    TopologySpec,
    compute_overlap_gradient,
    create_optimization_mask,
)

# --- 1. Simulation Setup ---
W = H = 7 * µm
WG_W = 0.55 * µm
WL = 1.55 * µm
N_CORE, N_CLAD = 2.25, 1.444  # Si3N4, SiO2
DX, DT = calc_optimal_fdtd_params(WL, 2.25, points_per_wavelength=8)
STEPS = 50  # reduce to 40 for faster optimization
MAT_PENALTY = 0.3  # Target core material fraction (0.0 to 1.0)
PENALTY_STRENGTH = 1  # Scaling factor for the penalty gradient
GRADIENT_SCALE_PERCENTILE = 95.0  # Robust EM-gradient normalization percentile
PML_FORMULATION_2D = "sponge"


def monitor_energy(monitor, dt):
    """Return integrated flux magnitude from a signed monitor trace."""

    del dt
    return abs(float(np.real(np.asarray(monitor.flux).reshape(-1)[0])))


def monitor_result(results, name):
    """Return a monitor snapshot from a SimulationResults object."""

    monitor_results = None if results is None else results.monitors
    if not monitor_results:
        raise RuntimeError("Simulation returned no monitor results.")
    if name not in monitor_results:
        available = ", ".join(sorted(monitor_results))
        raise KeyError(f"Monitor result {name!r} not found. Available: {available}")
    return monitor_results[name]


def saved_field_history(results, name):
    """Return saved field frames from a SimulationResults object."""

    fields = None if results is None else results.monitor("fields").fields
    if not fields or name not in fields:
        return []
    data = np.asarray(fields[name])
    if data.ndim == 0:
        return [data]
    return [np.asarray(frame) for frame in data]


def saved_field_array(results, name):
    """Return a saved field array, or an empty array when absent."""

    fields = None if results is None else results.monitor("fields").fields
    if not fields or name not in fields:
        return np.zeros((0,), dtype=float)
    return np.asarray(fields[name])


def transmission_percent(input_monitor, output_monitor, dt):
    input_energy = monitor_energy(input_monitor, dt)
    if input_energy <= 0.0:
        return 0.0
    return monitor_energy(output_monitor, dt) / input_energy * 100.0


def transmission_percent_from_results(results, input_name, output_name, dt):
    """Return transmission using compiled-run monitor snapshots."""

    input_result = monitor_result(results, input_name)
    output_result = monitor_result(results, output_name)
    return transmission_percent(input_result, output_result, dt)


def normalize_gradient_in_mask(grad, region_mask, percentile=95.0):
    """Normalize a masked gradient to a robust unit scale."""

    values = np.asarray(grad[region_mask], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return grad, 1.0

    scale = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(np.abs(finite)))
    if not np.isfinite(scale) or scale <= 0.0:
        return grad, 1.0

    return grad / scale, scale


# Design & Materials
design = Design(width=W, height=H, material=Material(permittivity=N_CLAD**2))
design += Rectangle(
    position=(0, H / 2 - WG_W / 2),
    width=W / 2,
    height=WG_W,
    material=Material(permittivity=N_CORE**2),
)
design += Rectangle(
    position=(W / 2 - WG_W / 2, 0),
    width=WG_W,
    height=H / 2,
    material=Material(permittivity=N_CORE**2),
)

# Optimization Region (added as placeholder)
opt_region = Rectangle(
    position=(W / 2 - 1.5 * µm, H / 2 - 1.5 * µm),
    width=3 * µm,
    height=3 * µm,
    material=Material(permittivity=N_CORE**2),
)
design += opt_region

# Sources
time = np.arange(0, 15 * WL / LIGHT_SPEED, DT)
signal = ramped_cosine(
    time, 1, LIGHT_SPEED / WL, ramp_duration=3.5 * WL / LIGHT_SPEED, t_max=time[-1] / 2
)

src_fwd = ModeSource(
    center=(1.0 * µm, H / 2, 0.0),
    size=(0.0, WG_W * 4, WG_W),
    source_time=SampledSignal(signal, dt=DT, freq0=LIGHT_SPEED / WL),
    direction="+",
    mode_spec=ModeSpec(polarization="tm"),
)
src_adj = ModeSource(
    center=(W / 2, 1.0 * µm, 0.0),
    size=(WG_W * 4, 0.0, WG_W),
    source_time=SampledSignal(signal, dt=DT, freq0=LIGHT_SPEED / WL),
    direction="+",
    mode_spec=ModeSpec(polarization="tm"),
)

# --- 2. Optimization Manager ---
# Rasterize once to get grid and mask
grid = design.rasterize(DX)
mask = create_optimization_mask(grid, opt_region)

opt = TopologySpec(
    design=design,
    region_mask=mask,
    resolution=DX,
    learning_rate=0.01,
    filter_radius=0.3
    * µm,  # Physical units: Controls minimum feature size AND boundary smoothness
    eps_min=N_CLAD**2,
    eps_max=N_CORE**2,
    beta_schedule=(1.0, 20.0),
    filter_type="conic",  # Use conic filter for geometric constraints
)
opt_state = opt.initial_state()

print(f"Starting Topology Optimization ({STEPS} steps)...")
base_eps = grid.permittivity.copy()  # Store background (cladding)

# Track transmission history
transmission_history = []

# --- 3. Optimization Loop ---
for step in range(STEPS):
    # Update Design
    beta, phys_density = opt.density_for_step(opt_state, step, STEPS)

    # Mix Density into Permittivity (Linear Interpolation)
    permittivity = np.array(base_eps, copy=True)
    permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)
    grid = grid.updated_copy(permittivity=permittivity)

    # Forward Simulation (only output monitor)
    # Setup monitors for input and output power measurement
    # Place monitor immediately after source to measure actual injected power
    # This accounts for soft source loading and back-reflection
    monitor_input_flux = FluxMonitor(
        center=(1.5 * µm, H / 2, 0.0),
        size=(0.0, WG_W * 4, WG_W),
        freqs=(LIGHT_SPEED / WL,),
        name="input_flux",
    )

    # Output monitor at output waveguide (bottom)
    output_monitor_fwd = FluxMonitor(
        center=(W / 2, 1.5 * µm, 0.0),
        size=(WG_W * 4, 0.0, WG_W),
        freqs=(LIGHT_SPEED / WL,),
        name="output_flux",
    )

    # Run forward simulation with output monitor
    sim_fwd = Simulation(
        material_grid=grid,
        sources=[src_fwd],
        monitors=[
            monitor_input_flux,
            output_monitor_fwd,
            FieldRecorder(("Ez",), interval=2, name="fields"),
        ],
        boundaries=[PML(edges="all", thickness=1 * µm, formulation=PML_FORMULATION_2D)],
        time=time,
        resolution=DX,
    )

    print(f"[{step + 1}/{STEPS}] Forward Sim...", end="\r")
    results = sim_fwd.run()

    fwd_ez_history = saved_field_history(results, "Ez")
    transmission_fwd = transmission_percent_from_results(
        results, "input_flux", "output_flux", DT
    )

    # Backward Simulation (with backward monitor at input location)
    # Backward source monitor (just downstream of source)
    monitor_back_flux = FluxMonitor(
        center=(W / 2, 1.5 * µm, 0.0),
        size=(WG_W * 4, 0.0, WG_W),
        freqs=(LIGHT_SPEED / WL,),
        name="backward_source_flux",
    )

    # Backward monitor at original input location (left waveguide)
    backward_monitor = FluxMonitor(
        center=(1.5 * µm, H / 2, 0.0),
        size=(0.0, WG_W * 4, WG_W),
        freqs=(LIGHT_SPEED / WL,),
        name="backward_output_flux",
    )

    sim_adj = Simulation(
        material_grid=grid,
        sources=[src_adj],
        monitors=[
            monitor_back_flux,
            backward_monitor,
            FieldRecorder(("Ez",), interval=2, name="fields"),
        ],
        boundaries=[PML(edges="all", thickness=1 * µm, formulation=PML_FORMULATION_2D)],
        time=time,
        resolution=DX,
    )

    adj_results = sim_adj.run()
    adj_ez_history = saved_field_history(adj_results, "Ez")
    transmission_back = transmission_percent_from_results(
        adj_results, "backward_source_flux", "backward_output_flux", DT
    )

    # Average bidirectional transmission
    transmission_pct = (transmission_fwd + transmission_back) / 2.0
    obj_val = transmission_pct

    opt_state = opt_state.with_objective(obj_val)
    transmission_history.append(transmission_pct)

    # Compute Gradient (overlap of fwd and adj fields)
    grad_eps = compute_overlap_gradient(fwd_ez_history, adj_ez_history)

    # Ez is stored on the full TMz Yee lattice, which has one high-side sample
    # more than the material grid on each axis. Fold that padding back before
    # applying design-mask penalties and optimizer updates.
    grad_eps = opt.gradient_to_design_grid(grad_eps)
    grad_eps, grad_scale = normalize_gradient_in_mask(
        grad_eps,
        mask,
        percentile=GRADIENT_SCALE_PERCENTILE,
    )

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
    penalty_val = PENALTY_STRENGTH * 0.5 * (current_density - MAT_PENALTY) ** 2
    total_obj = obj_val - penalty_val

    # Step Optimizer
    opt_state, max_update = opt.apply_gradient(opt_state, grad_eps, beta)

    # Calculate fraction for display
    mat_frac = np.mean(phys_density[mask])

    print(
        f" Step {step + 1}: Obj={total_obj:.2e} "
        f"(Trans={transmission_pct:.1f}% | Fwd={transmission_fwd:.1f}% "
        f"Bwd={transmission_back:.1f}%) | Mat={mat_frac:.1%} | "
        f"GradScale={grad_scale:.1e} | MaxUp={max_update:.2e}",
        end="\r",
    )

    # Viz
    if step % 5 == 0:
        plt.imsave(
            f"topo_opt_{step:03d}.png", grid.permittivity.T, cmap="gray", origin="lower"
        )

print(f"\nOptimization Complete. Final Transmission: {transmission_history[-1]:.1f}%")

# Plot transmission vs step (as percentage)
plt.figure(figsize=(10, 6))
steps = np.arange(1, len(transmission_history) + 1)
plt.plot(steps, transmission_history, "b-", linewidth=2, marker="o", markersize=4)
plt.xlabel("Optimization Step", fontsize=12)
plt.ylabel("Transmission (%)", fontsize=12)
plt.title("Transmission vs Optimization Step", fontsize=14)
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("transmission_vs_step.png", dpi=150, bbox_inches="tight")
print("Transmission plot saved to transmission_vs_step.png")
plt.close()

# --- 4. Final Verification & Visualization ---
# We now perform a frequency sweep to verify broadband performance
print("\n--- Running Final Frequency Sweep (1500-1600 nm) ---")

wavelengths = np.linspace(1.2 * µm, 1.8 * µm, 12)
sweep_transmission = []

# Use extended time to ensure full pulse transmission for all runs
time_sweep = np.arange(0, 15 * WL / LIGHT_SPEED, DT)

for wl_val in wavelengths:
    print(f"Simulating Wavelength: {wl_val / µm:.3f} µm...", end="\r")

    # Create signal for this specific wavelength
    signal_sweep = ramped_cosine(
        time_sweep,
        1,
        LIGHT_SPEED / wl_val,
        ramp_duration=3.5 * wl_val / LIGHT_SPEED,
        t_max=time_sweep[-1] / 2,
    )

    # Create source
    src_sweep = ModeSource(
        center=(1.0 * µm, H / 2, 0.0),
        size=(0.0, WG_W * 4, WG_W),
        source_time=SampledSignal(signal_sweep, dt=DT, freq0=LIGHT_SPEED / wl_val),
        direction="+",
        mode_spec=ModeSpec(polarization="tm"),
    )
    # Monitors
    mon_in = FluxMonitor(
        center=(1.5 * µm, H / 2, 0.0),
        size=(0.0, WG_W * 4, WG_W),
        freqs=(LIGHT_SPEED / wl_val,),
        name="input_flux",
    )
    mon_out = FluxMonitor(
        center=(W / 2, 1.5 * µm, 0.0),
        size=(WG_W * 4, 0.0, WG_W),
        freqs=(LIGHT_SPEED / wl_val,),
        name="output_flux",
    )

    # Simulation
    sim_sweep = Simulation(
        material_grid=grid,
        sources=[src_sweep],
        monitors=[mon_in, mon_out],
        boundaries=[PML(edges="all", thickness=1 * µm, formulation=PML_FORMULATION_2D)],
        time=time_sweep,
        resolution=DX,
    )

    # Run (no field saving needed for sweep, faster)
    sweep_results = sim_sweep.run()
    trans = transmission_percent_from_results(
        sweep_results, "input_flux", "output_flux", DT
    )
    sweep_transmission.append(trans)

print("\nSweep Complete.")

# Plot Frequency Sweep
plt.figure(figsize=(10, 6))
plt.plot(wavelengths / µm, sweep_transmission, "r-o", linewidth=2)
plt.xlabel("Wavelength (µm)", fontsize=12)
plt.ylabel("Transmission (%)", fontsize=12)
plt.title("Transmission Spectrum", fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("transmission_spectrum.png", dpi=150)
print("Spectrum plot saved to transmission_spectrum.png")

# --- 5. Final Visualization (Center Wavelength) ---
# Re-run simulation at center wavelength (1.55) to generate field plot
print("\n--- Generating Final Field Plot (1.55 µm) ---")
signal_final = ramped_cosine(
    time_sweep,
    1,
    LIGHT_SPEED / WL,
    ramp_duration=3.5 * WL / LIGHT_SPEED,
    t_max=time_sweep[-1] / 2,
)
src_final = ModeSource(
    center=(1.0 * µm, H / 2, 0.0),
    size=(0.0, WG_W * 4, WG_W),
    source_time=SampledSignal(signal_final, dt=DT, freq0=LIGHT_SPEED / WL),
    direction="+",
    mode_spec=ModeSpec(polarization="tm"),
)

mon_in_final = FluxMonitor(
    center=(1.5 * µm, H / 2, 0.0),
    size=(0.0, WG_W * 4, WG_W),
    freqs=(LIGHT_SPEED / WL,),
    name="input_flux",
)
mon_out_final = FluxMonitor(
    center=(W / 2, 1.5 * µm, 0.0),
    size=(WG_W * 4, 0.0, WG_W),
    freqs=(LIGHT_SPEED / WL,),
    name="output_flux",
)

sim_final = Simulation(
    material_grid=grid,
    sources=[src_final],
    monitors=[
        mon_in_final,
        mon_out_final,
        FieldRecorder(("Ez", "Hx", "Hy"), interval=1, name="fields"),
    ],
    boundaries=[PML(edges="all", thickness=1 * µm, formulation=PML_FORMULATION_2D)],
    time=time_sweep,
    resolution=DX,
)
results_final = sim_final.run()

trans_final = transmission_percent_from_results(
    results_final, "input_flux", "output_flux", DT
)

print("Calculating energy flow...")
Ez_t = saved_field_array(results_final, "Ez")
Hx_t = saved_field_array(results_final, "Hx")
Hy_t = saved_field_array(results_final, "Hy")

min_x = min(Ez_t.shape[1], Hx_t.shape[1], Hy_t.shape[1])
min_y = min(Ez_t.shape[2], Hx_t.shape[2], Hy_t.shape[2])

Ez_c = Ez_t[:, :min_x, :min_y]
Hx_c = Hx_t[:, :min_x, :min_y]
Hy_c = Hy_t[:, :min_x, :min_y]

Sx_t = -Ez_c * Hy_c
Sy_t = Ez_c * Hx_c
S_mag_t = np.sqrt(Sx_t**2 + Sy_t**2)
energy_flow = np.sum(S_mag_t, axis=0) * DT

# For display, crop out the PML.
# We keep the full interior visible and use a robust color scale so the
# source plane does not dominate the normalization.
pml_cells = max(1, int(round((1 * µm) / DX)))
crop_x = slice(pml_cells, max(pml_cells + 1, energy_flow.shape[0] - pml_cells))
crop_y = slice(pml_cells, max(pml_cells + 1, energy_flow.shape[1] - pml_cells))
display_energy_flow = energy_flow[crop_x, crop_y].copy()
perm_display = grid.permittivity[:min_x, :min_y][crop_x, crop_y]

finite_display = display_energy_flow[np.isfinite(display_energy_flow)]
display_vmax = np.quantile(finite_display, 0.995) if finite_display.size else None
display_map = np.ma.masked_invalid(display_energy_flow.T)

plt.figure(figsize=(10, 8))
plt.imshow(perm_display.T, cmap="gray", origin="lower", alpha=0.2)
plt.contour(
    perm_display.T,
    levels=[(N_CORE**2 + N_CLAD**2) / 2],
    colors="white",
    linewidths=1.5,
    origin="lower",
)
im = plt.imshow(
    display_map,
    cmap="inferno",
    origin="lower",
    alpha=0.9,
    interpolation="bicubic",
    vmax=display_vmax,
)
plt.colorbar(im, label=r"Time-Integrated Energy Flow $\int |\mathbf{S}| dt$")
plt.title(f"Final Energy Flow Map (1.55 µm, T={trans_final:.1f}%)")
plt.xlabel("x (grid cells)")
plt.ylabel("y (grid cells)")
plt.tight_layout()
plt.savefig("final_energy_flow.png", dpi=150)
print("Energy flow map saved to final_energy_flow.png")
