import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

#plt.switch_backend("Agg")
from beamz import *
from beamz import viz
from beamz.optimization.optimizers import Optimizer
from beamz.devices.mode import solve_modes

# Parameters
W = H = 15*µm
WG_W = 0.5*µm
N_CORE, N_CLAD = 2.25, 1.444
EPS_CORE, EPS_CLAD = N_CORE**2, N_CLAD**2
WL = 1.55*µm
STEPS, LR = 2, 0.1
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE,N_CLAD), dims=2, safety_factor=0.99, points_per_wavelength=9)
TIME = 30*WL/LIGHT_SPEED
t = np.arange(0, TIME, DT)

FILTER_RADIUS = 0.1*µm
PROJECTION_BETA_START = 2.0
PROJECTION_BETA_END = 20.0
PROJECTION_ETA = 0.5
TRANSMISSION_WEIGHT = 0.5
MODE_WEIGHT = 0.5
FINAL_PROJECTION_BETA = 32.0
FINAL_PROJECTION_THRESHOLD = 0.5

# Create the design
# Use very small amplitude to prevent divergence in topology optimization
signal = ramped_cosine(t=t,amplitude=1e-6, frequency=LIGHT_SPEED/WL, t_max=TIME, ramp_duration=6*WL/LIGHT_SPEED, phase=0)
design = Design(width=W, height=H, pml_size=2*µm)
design += Rectangle(position=(0*µm,H/2-WG_W/2), width=3.5*µm, height=WG_W, material=Material(permittivity=EPS_CORE))
design += Rectangle(position=(W/2-WG_W/2,H), width=WG_W, height=-3.5*µm, material=Material(permittivity=EPS_CORE))
region = Rectangle(position=(W/2-4*µm,H/2-4*µm), width=8*µm, height=8*µm, material=Material(permittivity=EPS_CLAD))
design += region
design.show()

# Rasterize the design
grid = RegularGrid(design=design, resolution=DX)

# Mask and density initialization for topology updates
base = grid.permittivity.copy()
mask = np.zeros_like(base,bool)
dx, dy = getattr(grid,"dx",DX), getattr(grid,"dy",DX)
min_cell = float(min(dx, dy))
filter_radius_cells = max(1, int(round(FILTER_RADIUS / min_cell)))
xs, ys = (np.arange(mask.shape[1])+0.5)*dx, (np.arange(mask.shape[0])+0.5)*dy
minx, miny, _, maxx, maxy, _ = region.get_bounding_box()
mask[(ys[:,None]>=miny)&(ys[:,None]<=maxy)&(xs[None,:]>=minx)&(xs[None,:]<=maxx)] = True

def masked_box_blur(values, mask, radius):
    """Return masked box blur and the corresponding weight map."""

    radius = int(max(0, radius))
    masked_values = np.where(mask, values, 0.0)

    if radius <= 0:
        weights = np.where(mask, 1.0, 1.0)
        return masked_values, weights

    padded_values = np.pad(masked_values, radius, mode="edge")
    padded_mask = np.pad(mask.astype(float), radius, mode="constant", constant_values=0.0)
    window_shape = (2 * radius + 1, 2 * radius + 1)

    values_view = sliding_window_view(padded_values, window_shape)
    mask_view = sliding_window_view(padded_mask, window_shape)

    weighted_sum = (values_view * mask_view).sum(axis=(-2, -1))
    weights = mask_view.sum(axis=(-2, -1))
    weights = np.where(weights == 0.0, 1.0, weights)

    blurred = weighted_sum / weights
    blurred = np.where(mask, blurred, 0.0)
    weights = np.where(mask, weights, 1.0)
    return blurred, weights


def masked_box_blur_backprop(grad_output, mask, weights, radius):
    """Propagate gradients through a masked box blur."""

    radius = int(max(0, radius))
    grad_output = np.where(mask, grad_output, 0.0)
    weights = np.where(weights == 0.0, 1.0, weights)
    contributions = grad_output / weights

    if radius <= 0:
        return np.where(mask, contributions, 0.0)

    padded = np.pad(contributions, radius, mode="constant", constant_values=0.0)
    window_shape = (2 * radius + 1, 2 * radius + 1)
    contrib_view = sliding_window_view(padded, window_shape)
    grad_input = contrib_view.sum(axis=(-2, -1))
    grad_input = np.where(mask, grad_input, 0.0)
    return grad_input


def projection_beta_schedule(step: int) -> float:
    """Return the projection sharpness beta for the current optimization step."""

    if STEPS <= 1:
        return PROJECTION_BETA_END
    frac = (step - 1) / float(STEPS - 1)
    return PROJECTION_BETA_START + frac * (PROJECTION_BETA_END - PROJECTION_BETA_START)


def smoothed_heaviside(value, beta, eta):
    """Smoothed Heaviside projection yielding near-binary densities."""

    beta = max(beta, 1e-6)
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    if abs(denominator) < 1e-9:
        denominator = 1e-9 if denominator >= 0 else -1e-9
    numerator = np.tanh(beta * eta) + np.tanh(beta * (value - eta))
    projected = numerator / denominator
    derivative = (beta * (1.0 - np.tanh(beta * (value - eta)) ** 2)) / denominator
    return projected, derivative


def apply_density_filters(values, beta):
    """Apply masked blur followed by projection to obtain physical densities."""

    blurred, weights = masked_box_blur(values, mask, filter_radius_cells)
    blurred = np.where(mask, blurred, 0.0)
    blurred = np.clip(blurred, 0.0, 1.0)
    projected, derivative = smoothed_heaviside(blurred, beta, PROJECTION_ETA)
    projected = np.where(mask, projected, 0.0)
    projected = np.clip(projected, 0.0, 1.0)
    derivative = np.where(mask, derivative, 0.0)
    derivative = np.where((projected <= 0.0) | (projected >= 1.0), 0.0, derivative)
    return blurred, projected, weights, derivative


def binarize_density(physical_density):
    """Return a binary density map using a sharp projection."""

    projected, _ = smoothed_heaviside(
        physical_density,
        beta=FINAL_PROJECTION_BETA,
        eta=FINAL_PROJECTION_THRESHOLD,
    )
    binary = (projected >= 0.5).astype(float)
    return np.where(mask, binary, 0.0)


def build_output_monitor(design):
    """Create an output monitor with objective favoring transmission and mode shape."""

    monitor_start = (W/2 - WG_W*2, H - 2.5*µm)
    monitor_end = (W/2 + WG_W*2, H - 2.5*µm)
    monitor = Monitor(
        design=design,
        start=monitor_start,
        end=monitor_end,
        record_fields=True,
        accumulate_power=True,
        record_interval=1,
        max_history_steps=None,
        name="out",
    )

    grid_points = monitor.get_grid_points_2d(dx, dy)
    if not grid_points:
        raise RuntimeError("Monitor line produced no grid points; check resolution or monitor bounds.")
    x_positions = np.array([pt[0] * dx for pt in grid_points], dtype=float)
    center_x = 0.5 * (monitor_start[0] + monitor_end[0])
    relative_x = x_positions - center_x
    # Compute target mode profile using the new numerical mode solver
    # Build 1D permittivity cross-section along the monitor line (fixed y)
    fixed_y = monitor_start[1]
    eps_line = np.array([design.get_material_value(float(xp), float(fixed_y), 0.0)[0] for xp in x_positions], dtype=float)
    # Solve the 1D eigenmode for propagation in +y (monitor is horizontal)
    omega = 2 * np.pi * LIGHT_SPEED / WL
    try:
        neffs, mode_e_fields, mode_h_fields, prop_axis = solve_modes(
            eps=eps_line,
            omega=omega,
            dL=dx,
            npml=0,
            m=1,
            direction="+y",
            return_fields=True,
        )
        if mode_e_fields.size:
            ez_idx_lookup = {0: 0, 1: 1, 2: 2}
            ez_idx = ez_idx_lookup.get(prop_axis, 0)
            ez_component = mode_e_fields[0][ez_idx]
            ez_line = np.squeeze(ez_component)
            if ez_line.ndim > 1:
                ez_line = ez_line.reshape(-1)
            mode_vec = np.asarray(ez_line, dtype=complex)
        else:
            mode_vec = np.ones_like(eps_line, dtype=complex)
    except Exception:
        mode_vec = np.ones_like(eps_line, dtype=complex)
    mode_vec = np.asarray(mode_vec, dtype=complex)
    if mode_vec.size == 0:
        mode_vec = np.ones_like(eps_line, dtype=complex)
    elif mode_vec.size != eps_line.size:
        src = np.linspace(0.0, 1.0, mode_vec.size)
        dst = np.linspace(0.0, 1.0, eps_line.size)
        mode_vec = (
            np.interp(dst, src, mode_vec.real, left=mode_vec.real[0], right=mode_vec.real[-1])
            + 1j * np.interp(dst, src, mode_vec.imag, left=mode_vec.imag[0], right=mode_vec.imag[-1])
        )
    # Normalize target samples
    target_samples = np.asarray(mode_vec, dtype=complex)
    target_norm = np.linalg.norm(target_samples)
    if target_norm == 0.0:
        target_norm = 1.0
    target_samples /= target_norm

    def objective_fn(m) -> float:
        if not m.fields.get("Ez"):
            return 0.0
        field_history = [np.asarray(sample, dtype=complex) for sample in m.fields["Ez"]]
        overlaps = []
        ts_norm = np.linalg.norm(target_samples)
        for field_vec in field_history:
            field_norm = np.linalg.norm(field_vec)
            if field_norm == 0.0:
                overlaps.append(0.0)
                continue
            overlap = np.abs(np.vdot(target_samples, field_vec)) / (ts_norm * field_norm)
            overlaps.append(float(overlap))
        mode_score = max(overlaps) if overlaps else 0.0
        power_score = float(np.trapz(np.abs(m.power_history), dx=DT)) if m.power_history else 0.0
        m.mode_score = mode_score
        m.power_score = power_score
        return -(TRANSMISSION_WEIGHT * power_score + MODE_WEIGHT * mode_score)

    monitor.objective_function = objective_fn
    monitor.target_samples = target_samples
    monitor.target_positions = x_positions
    monitor.center_x = center_x
    return monitor


def build_adjoint_source(design, signal, target_positions, target_samples):
    """Construct adjoint source shaped to the desired mode profile."""

    adjoint = ModeSource(
        design=design,
        position=(W/2, H - 2.5*µm),
        width=4 * WG_W,
        wavelength=WL,
        signal=signal,
        direction="-y",
        modes=1,
    )
    if adjoint.mode_profiles:
        profile = adjoint.mode_profiles[0]
        # profile entries are dicts with coordinates
        profile_positions = np.array([float(pt.get("x", 0.0)) for pt in profile], dtype=float)
        real_interp = np.interp(profile_positions, target_positions, target_samples.real, left=0.0, right=0.0)
        imag_interp = np.interp(profile_positions, target_positions, target_samples.imag, left=0.0, right=0.0)
        
        # Store original H-field ratios for maintaining unidirectionality
        first_point = profile[0]
        e_field_name = None
        h_field_names = []
        for key in first_point.keys():
            if key.startswith('E') and key not in ['x', 'y', 'z']:
                e_field_name = key
            elif key.startswith('H') and key not in ['x', 'y', 'z']:
                h_field_names.append(key)
        
        for idx, point in enumerate(profile):
            target_component = MODE_WEIGHT * (real_interp[idx] + 1j * imag_interp[idx])
            transmission_component = TRANSMISSION_WEIGHT
            combined = transmission_component + target_component
            
            if not np.isfinite(combined):
                combined = transmission_component
            
            # Scale E-field by target amplitude
            original_e = point.get(e_field_name, 1.0) if e_field_name else 1.0
            if abs(original_e) > 1e-15:
                scale_factor = combined / original_e
            else:
                scale_factor = combined
            
            # Apply same scaling to E and H to maintain unidirectional Poynting vector
            if e_field_name:
                point[e_field_name] = combined
            
            # Scale H-fields proportionally to maintain S = E × H direction
            for h_name in h_field_names:
                original_h = point.get(h_name, 0.0)
                if np.isfinite(original_h) and np.isfinite(scale_factor):
                    point[h_name] = original_h * scale_factor
                else:
                    point[h_name] = 0.0
    return adjoint


def build_forward_source(design, signal):
    """Return a forward mode source that injects the input waveguide mode without mirrors."""

    source = ModeSource(
        design=design,
        position=(2.5*µm, H/2),
        width=WG_W * 4,
        wavelength=WL,
        signal=signal,
        direction="+x",
        modes=1,
    )
    return source
rng = np.random.default_rng(0)
design_density = np.zeros_like(base)
design_density[mask] = rng.random(np.count_nonzero(mask))
objective_history = []
optimizer = Optimizer(method="adam", learning_rate=LR)

# Optimization loop
for step in range(1,STEPS+1):
    current_beta = projection_beta_schedule(step)
    blurred_density, physical_density, blur_weights, projection_derivative = apply_density_filters(
        design_density,
        current_beta,
    )

    # Update the permittivity
    eps = base.copy()
    eps[mask] = EPS_CLAD + physical_density[mask]*(EPS_CORE-EPS_CLAD)
    np.copyto(grid.permittivity, eps)
    
    # Forward simulation using non-reflective source and mode-aware monitor
    print(f"\n[STEP {step}] Running FORWARD simulation...")
    source = build_forward_source(design, signal)
    monitor = build_output_monitor(design)
    
    # Debug: check permittivity for NaN/Inf
    eps_min = float(np.min(grid.permittivity))
    eps_max = float(np.max(grid.permittivity))
    has_nan = not np.all(np.isfinite(grid.permittivity))
    print(f"  Permittivity range: [{eps_min:.3f}, {eps_max:.3f}], has NaN/Inf: {has_nan}")
    if has_nan or eps_min < 0 or eps_max > 100:
        print(f"  ⚠️ WARNING: Permittivity out of reasonable bounds!")
    
    forward = FDTD(design=grid, devices=[source, monitor], time=t)
    # Use explicit axis_scale to make field amplitudes visible (in V/µm)
    # New normalization: |E| ~ 1e6 V/m × signal(1e-6) × viz_scale(1e-6) = 1 V/µm
    fres = forward.run(live=True, save_memory_mode=True, accumulate_power=True, save_fields=["Ez"], fields_to_cache=["Ez"])
    
    # Check if forward fields are reasonable
    if fres.get("Ez"):
        last_ez = fres["Ez"][-1] if fres["Ez"] else None
        if last_ez is not None:
            ez_max = float(np.max(np.abs(last_ez)))
            ez_has_nan = not np.all(np.isfinite(last_ez))
            print(f"  Forward Ez_max: {ez_max:.3e}, has NaN/Inf: {ez_has_nan}")
            if ez_has_nan or ez_max > 1e3:
                print(f"  ⚠️ WARNING: Forward simulation may have diverged!")
    
    # Debug: check accumulated power statistics
    if forward.power_accumulated is not None:
        power_min = float(np.min(forward.power_accumulated))
        power_max = float(np.max(forward.power_accumulated))
        power_mean = float(np.mean(forward.power_accumulated))
        power_nonzero = np.count_nonzero(forward.power_accumulated)
        print(f"  Forward power: min={power_min:.3e}, max={power_max:.3e}, mean={power_mean:.3e}, nonzero={power_nonzero}/{forward.power_accumulated.size}")
        
        # Show power distribution percentiles for debugging
        nonzero = forward.power_accumulated[forward.power_accumulated > 0]
        if len(nonzero) > 0:
            p50 = np.percentile(nonzero, 50)
            p95 = np.percentile(nonzero, 95)
            p99 = np.percentile(nonzero, 99)
            print(f"  Power percentiles: p50={p50:.3e}, p95={p95:.3e}, p99={p99:.3e}, max={power_max:.3e}")
            print(f"  Ratio p99/max: {(p99/power_max)*100:.1f}% (shows if max is outlier)")
    
    forward.plot_power(db_colorbar=True)
    forward_power_path = f"forward_power_step{step:03d}.png"
    forward.fig.savefig(forward_power_path, dpi=200, bbox_inches="tight")
    plt.close(forward.fig)
    ffields = list(fres.get("Ez",[]))
    
    # Adjoint simulation, computing the overlap gradient
    print(f"[STEP {step}] Running ADJOINT simulation...")
    adj_source = build_adjoint_source(design, signal, monitor.target_positions, monitor.target_samples)
    
    # Debug: check adjoint source profile to verify field amplitudes
    if adj_source.mode_profiles:
        adj_profile = adj_source.mode_profiles[0]
        adj_profile_sample = adj_profile[0]
        adj_field_names = [k for k in adj_profile_sample.keys() if k not in ['x', 'y', 'z']]
        print(f"  Adjoint source field components: {adj_field_names}")
        # Get max amplitudes across the profile
        for fname in adj_field_names:
            amplitudes = [abs(pt.get(fname, 0.0)) for pt in adj_profile]
            max_amp = max(amplitudes) if amplitudes else 0.0
            print(f"    {fname} max amplitude: {max_amp:.3e}")
        
        # Debug: Check field signs at center to verify unidirectionality (disabled for production)
        # center_idx = len(adj_profile) // 2
        # if center_idx < len(adj_profile):
        #     center_pt = adj_profile[center_idx]
        #     print(f"  Center point fields (for unidirectional check):")
        #     for fname in adj_field_names:
        #         val = center_pt.get(fname, 0.0)
        #         print(f"    {fname} = {val.real:.3e} + {val.imag:.3e}j")
    
    adj = FDTD(design=grid, devices=[adj_source], time=t)
    # Note: FDTD's live animation won't be used; we use manual viz below
    # But initialize with live=False to avoid creating unused figure
    adj.initialize_simulation(save=False, live=False, accumulate_power=True, save_memory_mode=True, fields_to_cache=None)
    grad = np.zeros_like(base)
    num_ffields = len(ffields)
    print(f"  Forward fields available: {num_ffields}")
    
    # Manual visualization context for adjoint Ez field (V/µm)
    ez_extent = (0.0, float(design.width), 0.0, float(design.height))
    manual_viz_ctx = None
    viz_stride = 2  # match forward animation cadence
    
    # Track max field magnitudes during adjoint simulation
    adj_ez_max_overall = 0.0
    for step_idx in range(adj.num_steps):
        if not ffields or not adj.step(): 
            break
        ez_field = adj.backend.to_numpy(adj.Ez)
        grad += np.real(ez_field*np.conj(ffields.pop()))
        
        if step_idx % viz_stride == 0:
            ez_real = np.real(ez_field)  # ensure float for visualization
            manual_viz_ctx = viz.animate_manual_field(
                ez_real * 1.0e-6,
                context=manual_viz_ctx,
                axis_scale=None,
                extent=ez_extent,
                cmap='RdBu',
                percentile=99,
                title=f"Adjoint Ez (step {step_idx})",
                units='V/µm',
                pause=0.002,
                auto_interval=4,
                smoothing=0.25,
                design=design,
            )
        
        # Sample field magnitude every 100 steps
        if step_idx % 100 == 0:
            adj_ez_max = float(np.max(np.abs(ez_field)))
            adj_ez_max_overall = max(adj_ez_max_overall, adj_ez_max)
    
    # Check gradient for issues
    grad_max = float(np.max(np.abs(grad)))
    grad_has_nan = not np.all(np.isfinite(grad))
    print(f"  Adjoint Ez_max: {adj_ez_max_overall:.3e}, gradient_max: {grad_max:.3e}, has NaN/Inf: {grad_has_nan}")
    
    adj.finalize_simulation()
    
    # Debug: check accumulated power statistics
    if adj.power_accumulated is not None:
        power_min = float(np.min(adj.power_accumulated))
        power_max = float(np.max(adj.power_accumulated))
        power_mean = float(np.mean(adj.power_accumulated))
        power_nonzero = np.count_nonzero(adj.power_accumulated)
        print(f"  Adjoint power: min={power_min:.3e}, max={power_max:.3e}, mean={power_mean:.3e}, nonzero={power_nonzero}/{adj.power_accumulated.size}")
        
        # Show power distribution percentiles for debugging
        nonzero = adj.power_accumulated[adj.power_accumulated > 0]
        if len(nonzero) > 0:
            p50 = np.percentile(nonzero, 50)
            p95 = np.percentile(nonzero, 95)
            p99 = np.percentile(nonzero, 99)
            print(f"  Power percentiles: p50={p50:.3e}, p95={p95:.3e}, p99={p99:.3e}, max={power_max:.3e}")
            print(f"  Ratio p99/max: {(p99/power_max)*100:.1f}% (shows if max is outlier)")
    else:
        print(f"  ⚠️ WARNING: Adjoint power_accumulated is None!")
    
    adj.plot_power(db_colorbar=True)
    adj_power_path = f"adjoint_power_step{step:03d}.png"
    adj.fig.savefig(adj_power_path, dpi=200, bbox_inches="tight")
    plt.close(adj.fig)

    grad_plot = np.zeros_like(grad)
    grad_plot[mask] = grad[mask]
    grad_scale = np.max(np.abs(grad_plot)) or 1.0
    grad_fig, grad_ax = plt.subplots(figsize=(6, 6))
    grad_im = grad_ax.imshow(grad_plot, origin="lower", extent=(0, design.width, 0, design.height), cmap="RdBu", aspect="equal", vmin=-grad_scale, vmax=grad_scale)
    plt.colorbar(grad_im, ax=grad_ax, orientation="vertical", label="Adjoint Gradient")
    grad_ax.set_title(f"Overlap Gradient Step {step}")
    grad_ax.set_xlabel("x (m)")
    grad_ax.set_ylabel("y (m)")
    grad_path = f"overlap_gradient_step{step:03d}.png"
    grad_fig.savefig(grad_path, dpi=200, bbox_inches="tight")
    plt.close(grad_fig)

    # Apply blur and projection filtered update with Adam optimizer
    grad_density = grad * (EPS_CORE - EPS_CLAD)
    grad_after_projection = grad_density * projection_derivative
    grad_design = masked_box_blur_backprop(
        grad_after_projection,
        mask,
        blur_weights,
        filter_radius_cells,
    )
    grad_design = np.where(mask, grad_design, 0.0)
    
    # Robust gradient normalization with NaN guards
    grad_design_max = np.abs(grad_design).max()
    if not np.isfinite(grad_design_max) or grad_design_max == 0.0:
        print(f"  ⚠️ WARNING: Gradient is zero or NaN! Skipping update.")
        grad_norm = np.zeros_like(grad_design)
    else:
        grad_norm = grad_design / grad_design_max
    adam_update = optimizer.step(-grad_norm)
    adam_update = np.where(mask, adam_update, 0.0)
    new_design_density = design_density + adam_update
    new_design_density = np.clip(new_design_density, 0.0, 1.0)
    density_delta = new_design_density - design_density
    design_density = np.where(mask, new_design_density, design_density)
    design_density[~mask] = 0.0  # Reset density outside the design region
    update_norm = float(np.linalg.norm(density_delta[mask])) if np.any(mask) else 0.0
    max_update = float(np.max(np.abs(density_delta[mask]))) if np.any(mask) else 0.0
    blurred_density, physical_density, blur_weights, projection_derivative = apply_density_filters(
        design_density,
        current_beta,
    )
    permittivity_grid = base.copy()
    permittivity_grid[mask] = EPS_CLAD + physical_density[mask] * (EPS_CORE - EPS_CLAD)
    permittivity_grid[mask] = np.clip(permittivity_grid[mask], EPS_CLAD, EPS_CORE)
    perm_fig, perm_ax = plt.subplots(figsize=(6, 6))
    perm_im = perm_ax.imshow(
        permittivity_grid,
        origin="lower",
        extent=(0, design.width, 0, design.height),
        cmap="gray",
        aspect="equal",
        vmin=EPS_CLAD,
        vmax=EPS_CORE,
    )
    plt.colorbar(perm_im, ax=perm_ax, orientation="vertical", label="Permittivity")
    perm_ax.set_title(f"Permittivity Step {step}")
    perm_ax.set_xlabel("x (m)")
    perm_ax.set_ylabel("y (m)")
    perm_path = f"permittivity_step{step:03d}.png"
    perm_fig.savefig(perm_path, dpi=200, bbox_inches="tight")
    plt.close(perm_fig)
    obj = float(next(iter(fres.get("objectives",{"out":0}).values())))
    combined_objective = -obj
    transmission = getattr(monitor, "power_score", 0.0)
    mode_score = getattr(monitor, "mode_score", 0.0)
    objective_history.append(combined_objective)
    history_fig, history_ax = plt.subplots(figsize=(6, 4))
    history_ax.plot(range(1, len(objective_history)+1), objective_history, marker="o", linewidth=2)
    history_ax.set_xlabel("Optimization Step")
    history_ax.set_ylabel("Combined Objective")
    history_ax.set_title("Objective History (higher is better)")
    history_ax.grid(True, alpha=0.3)
    history_fig.savefig("objective_history.png", dpi=200, bbox_inches="tight")
    plt.close(history_fig)
    print(
        f"step {step}: obj {combined_objective:.4e} | power {transmission:.4e} | "
        f"mode {mode_score:.3f} | update norm {update_norm:.3e} | max density update {max_update:.3e}"
    )

# Final transmission with binary projection
_, physical_density, _, _ = apply_density_filters(design_density, PROJECTION_BETA_END)
binary_density = binarize_density(physical_density)
eps = base.copy()
eps[mask] = EPS_CLAD + binary_density[mask] * (EPS_CORE - EPS_CLAD)
np.copyto(grid.permittivity, eps)
final_density_fig, final_density_ax = plt.subplots(figsize=(6, 6))
binary_im = final_density_ax.imshow(
    binary_density,
    origin="lower",
    extent=(0, design.width, 0, design.height),
    cmap="viridis",
    aspect="equal",
    vmin=0.0,
    vmax=1.0,
)
plt.colorbar(binary_im, ax=final_density_ax, orientation="vertical", label="Binary Density")
final_density_ax.set_title("Final Binary Density")
final_density_ax.set_xlabel("x (m)")
final_density_ax.set_ylabel("y (m)")
final_density_fig.savefig("final_binary_density.png", dpi=200, bbox_inches="tight")
plt.close(final_density_fig)

final_source = build_forward_source(design, signal)
final_monitor = build_output_monitor(design)
# Use auto-scaling with percentile-based clipping for optimal display
# New normalization: |E| ~ 1e6 V/m × signal(1e-6) × viz_scale(1e-6) = 1 V/µm
final = FDTD(design=grid, devices=[final_source, final_monitor], time=t).run(
    live=True,
    save_memory_mode=True,
    accumulate_power=True,
    save_fields=["Ez"],
    fields_to_cache=None,
)
final_obj = -float(next(iter(final.get("objectives", {"out": 0}).values())))
final_power = getattr(final_monitor, "power_score", 0.0)
final_mode = getattr(final_monitor, "mode_score", 0.0)
print("final objective", final_obj, "power", final_power, "mode", final_mode)
