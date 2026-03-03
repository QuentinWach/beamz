import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRectangle
from pathlib import Path

from beamz import *
from beamz.optimization.topology import (
    TopologyManager,
    compute_overlap_gradient,
    create_optimization_mask,
)


"""
Simplified inverse-design example: 1x2 wavelength demultiplexer (Si on SiO2) in 2D.

Per optimization step:
- 2 forward sims (one per wavelength)
- 4 adjoint sims (top + bottom, per wavelength)

Objective (power-based, simple):
FoM_k = T_target - w_leak*T_leak
Total objective = mean(FoM_k over wavelengths) - material_penalty
"""


# Unit convenience (ASCII-only)
UM = 1e-6


# --- 1) Setup ---
WG_W = 0.55 * UM
OUT_GAP = 0.95 * UM
INV_W = 3.50 * UM
INV_H = 3.50 * UM

W = 8.0 * UM
H = 6.0 * UM
PML_T = 1.2 * UM

WL_SHORT = 1.30 * UM
WL_LONG = 1.55 * UM
WAVELENGTH_CASES = [
    (WL_SHORT, "top"),
    (WL_LONG, "bottom"),
]

N_CORE = 3.48 # Si
N_CLAD = 1.0 # Air
EPS_CORE = N_CORE**2
EPS_CLAD = N_CLAD**2

DX, DT = calc_optimal_fdtd_params(
    wavelength=WL_SHORT,
    n_max=N_CORE,
    points_per_wavelength=12,
)

STEPS = 60
FIELD_SUBSAMPLE = 1

TARGET_DENSITY = 0.85
PENALTY_STRENGTH = 0.05
EMA_ALPHA = 0.20

LEAK_W_START = 0.20
LEAK_W_END = 1.00

GRAD_SCALE_START = 0.80
GRAD_SCALE_END = 0.50
CLIP_PCT = 99.5

# Use +1.0 if overlap gradients are already dJ/deps;
# switch to -1.0 if objective moves the wrong way.
ADJOINT_GRAD_SIGN = 1.0

# Debug outputs cadence.
DEBUG_EVERY = 5

# Beta continuation.
BETA_START = 1.0
BETA_MID = 6.0
BETA_END = 18.0
BETA_STAGE1_FRAC = 0.75

# Small deterministic perturbation to break exact geometric symmetry.
INITIAL_ASYM_NOISE = 1e-2

# Prevent early exact cancellation between the two wavelength objectives.
WL_LONG_RAMP_STEPS = 25


def cleanup_old_wdm_outputs():
    patterns = [
        "wdm_setup_sources_monitors.png",
        "wdm_topo_*.png",
        "wdm_topo_delta_*.png",
        "wdm_density_*.png",
        "wdm_density_delta_*.png",
        "wdm_grad_total_*.png",
        "wdm_grad_short_*.png",
        "wdm_grad_long_*.png",
        "wdm_flux_short_*.png",
        "wdm_flux_long_*.png",
        "wdm_objective_vs_step.png",
        "wdm_routing_vs_step.png",
        "wdm_per_channel_fom_vs_step.png",
        "wdm_final_binary_flux_overlay.png",
        "[0-9]_wdm_*.png",
    ]
    removed = 0
    for pat in patterns:
        for p in Path(".").glob(pat):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def port_center_y(domain_h, wg_w, output_gap):
    center = 0.5 * domain_h
    offset = 0.5 * (wg_w + output_gap)
    return center, center + offset, center - offset


Y_IN, Y_TOP, Y_BOT = port_center_y(H, WG_W, OUT_GAP)
X_INV0 = 0.5 * (W - INV_W)
Y_INV0 = 0.5 * (H - INV_H)
X_INV1 = X_INV0 + INV_W
Y_INV1 = Y_INV0 + INV_H

X_SRC = PML_T + 0.25 * UM
X_MON_OUT = W - PML_T - 0.35 * UM
MON_SPAN = 2.6 * WG_W


def continuation_value(step, total_steps, start, end, power=1.0):
    if total_steps <= 1:
        return end
    frac = step / (total_steps - 1)
    frac = np.clip(frac, 0.0, 1.0) ** power
    return start + frac * (end - start)


def continuation_value_two_stage(
    step,
    total_steps,
    start,
    mid,
    end,
    stage1_frac=0.45,
    power1=1.0,
    power2=1.0,
):
    if total_steps <= 1:
        return end

    frac = np.clip(step / (total_steps - 1), 0.0, 1.0)
    split = np.clip(stage1_frac, 0.05, 0.95)

    if frac <= split:
        local = (frac / split) ** power1
        return start + local * (mid - start)

    local = ((frac - split) / (1.0 - split)) ** power2
    return mid + local * (end - mid)


def build_time_and_signal(wavelength):
    flight_time = (X_MON_OUT - X_SRC) * N_CORE / LIGHT_SPEED
    ramp_duration = 8.0 * wavelength / LIGHT_SPEED
    total_time = 2.2 * flight_time + 2.5 * ramp_duration
    time = np.arange(0.0, total_time, DT)

    signal = ramped_cosine(
        time,
        1.0,
        LIGHT_SPEED / wavelength,
        ramp_duration=ramp_duration,
        t_max=0.45 * total_time,
    )
    return time, signal, total_time, flight_time


def make_vertical_monitor(grid, x, y_center, span, name=None, record_fields=True):
    kwargs = dict(
        design=grid,
        start=(x, y_center - 0.5 * span),
        end=(x, y_center + 0.5 * span),
        accumulate_power=True,
        record_fields=bool(record_fields),
    )
    if name is not None:
        kwargs["name"] = str(name)
    return Monitor(**kwargs)


def build_port_monitors(grid):
    mon_top = make_vertical_monitor(
        grid,
        X_MON_OUT,
        Y_TOP,
        MON_SPAN,
        name="out_top",
        record_fields=True,
    )
    mon_bot = make_vertical_monitor(
        grid,
        X_MON_OUT,
        Y_BOT,
        MON_SPAN,
        name="out_bottom",
        record_fields=True,
    )
    return mon_top, mon_bot


def run_forward(grid, wavelength, time, signal, save_fields=True):
    src = ModeSource(
        grid,
        center=(X_SRC, Y_IN),
        width=3.0 * WG_W,
        wavelength=wavelength,
        pol="tm",
        signal=signal,
        direction="+x",
    )
    mon_top, mon_bot = build_port_monitors(grid)

    sim = Simulation(
        grid,
        [src, mon_top, mon_bot],
        [PML(edges="all", thickness=PML_T)],
        time=time,
        resolution=DX,
    )

    save = ["Ez"] if save_fields else []
    results = sim.run(save_fields=save, field_subsample=FIELD_SUBSAMPLE)

    top_energy = float(np.sum(np.abs(np.asarray(mon_top.power_history, dtype=float))) * DT)
    bot_energy = float(np.sum(np.abs(np.asarray(mon_bot.power_history, dtype=float))) * DT)
    total_out = max(top_energy + bot_energy, 1e-30)

    ez_hist = []
    if save_fields:
        ez_hist = [np.array(frame) for frame in results.get("fields", {}).get("Ez", [])]
        if not ez_hist:
            raise RuntimeError("Forward simulation returned no Ez history.")

    return ez_hist, top_energy, bot_energy, total_out


def run_adjoint(grid, wavelength, target_port, time, signal):
    x_target = X_MON_OUT
    y_target = Y_TOP if target_port == "top" else Y_BOT
    direction = "-x"

    src = ModeSource(
        grid,
        center=(x_target, y_target),
        width=3.0 * WG_W,
        wavelength=wavelength,
        pol="tm",
        signal=signal,
        direction=direction,
    )

    sim = Simulation(
        grid,
        [src],
        [PML(edges="all", thickness=PML_T)],
        time=time,
        resolution=DX,
    )
    results = sim.run(save_fields=["Ez"], field_subsample=FIELD_SUBSAMPLE)

    ez_hist = [np.array(frame) for frame in results.get("fields", {}).get("Ez", [])]
    if not ez_hist:
        raise RuntimeError("Adjoint simulation returned no Ez history.")
    return ez_hist


def run_forward_flux_map(grid, wavelength, time, signal):
    src = ModeSource(
        grid,
        center=(X_SRC, Y_IN),
        width=3.0 * WG_W,
        wavelength=wavelength,
        pol="tm",
        signal=signal,
        direction="+x",
    )
    mon_top, mon_bot = build_port_monitors(grid)

    sim = Simulation(
        grid,
        [src, mon_top, mon_bot],
        [PML(edges="all", thickness=PML_T)],
        time=time,
        resolution=DX,
    )
    results = sim.run(save_fields=["Ez", "Hx", "Hy"], field_subsample=1)

    top_energy = float(np.sum(np.abs(np.asarray(mon_top.power_history, dtype=float))) * DT)
    bot_energy = float(np.sum(np.abs(np.asarray(mon_bot.power_history, dtype=float))) * DT)
    total_out = max(top_energy + bot_energy, 1e-30)
    tx_top = max(0.0, top_energy / total_out)
    tx_bot = max(0.0, bot_energy / total_out)

    ez_hist = [np.array(frame) for frame in results.get("fields", {}).get("Ez", [])]
    hx_hist = [np.array(frame) for frame in results.get("fields", {}).get("Hx", [])]
    hy_hist = [np.array(frame) for frame in results.get("fields", {}).get("Hy", [])]
    n_frames = min(len(ez_hist), len(hx_hist), len(hy_hist))
    if n_frames == 0:
        raise RuntimeError("Forward flux map sim returned no Ez/Hx/Hy history.")

    flux_map = np.zeros_like(ez_hist[0], dtype=float)
    for i in range(n_frames):
        ez_i = ez_hist[i]
        hx_i = hx_hist[i]
        hy_i = hy_hist[i]

        ny = min(ez_i.shape[0], hx_i.shape[0], hy_i.shape[0])
        nx = min(ez_i.shape[1], hx_i.shape[1], hy_i.shape[1])
        if ny <= 0 or nx <= 0:
            continue

        ez_c = ez_i[:ny, :nx]
        hx_c = hx_i[:ny, :nx]
        hy_c = hy_i[:ny, :nx]

        s_x = -ez_c * hy_c
        s_y = ez_c * hx_c
        flux_map[:ny, :nx] += np.sqrt(s_x * s_x + s_y * s_y) * DT

    return tx_top, tx_bot, flux_map


def save_gradient_debug_image(path, grad):
    vmax = np.percentile(np.abs(grad), 99.5)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vis = np.clip(grad / vmax, -1.0, 1.0)
    plt.imsave(path, vis.T, cmap="seismic", vmin=-1.0, vmax=1.0, origin="lower")


def save_flux_debug_image(path, flux):
    scale = np.percentile(flux, 99.0)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    vis = np.clip(flux / scale, 0.0, 1.0)
    plt.imsave(path, vis.T, cmap="inferno", origin="lower")


def save_setup_sources_monitors_plot(grid, path="wdm_setup_sources_monitors.png"):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(
        grid.permittivity.T,
        cmap="gray",
        origin="lower",
        extent=(0.0, W / UM, 0.0, H / UM),
        aspect="equal",
    )

    src_span = 3.0 * WG_W

    # The background uses permittivity.T, so annotate in the same rotated frame.
    # Map physical (x, y) -> plotted (x', y') coordinates.
    def map_xy(x_phys, y_phys):
        x_plot = (y_phys / H) * W
        y_plot = (x_phys / W) * H
        return x_plot / UM, y_plot / UM

    def draw_vline(x, y_center, span, color, label=None, linestyle="-", lw=2.0):
        x0, y0 = map_xy(x, y_center - 0.5 * span)
        x1, y1 = map_xy(x, y_center + 0.5 * span)
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            label=label,
        )

    # Forward mode source.
    draw_vline(X_SRC, Y_IN, src_span, color="cyan", label="Forward mode source", linestyle="-")

    # Adjoint mode sources used during optimization.
    draw_vline(
        X_MON_OUT,
        Y_TOP,
        src_span,
        color="magenta",
        label="Adjoint mode sources",
        linestyle="--",
    )
    draw_vline(X_MON_OUT, Y_BOT, src_span, color="magenta", linestyle="--")
    # Output monitors.
    draw_vline(X_MON_OUT, Y_TOP, MON_SPAN, color="yellow", linestyle=":")
    draw_vline(X_MON_OUT, Y_BOT, MON_SPAN, color="yellow", label="Monitors", linestyle=":")

    # Optimization region.
    r0x, r0y = map_xy(X_INV0, Y_INV0)
    r1x, r1y = map_xy(X_INV0 + INV_W, Y_INV0 + INV_H)
    ax.add_patch(
        MplRectangle(
            (min(r0x, r1x), min(r0y, r1y)),
            abs(r1x - r0x),
            abs(r1y - r0y),
            fill=False,
            edgecolor="lime",
            linewidth=1.8,
            linestyle="--",
            label="Optimization region",
        )
    )

    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("WDM setup: mode sources and monitors")
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close(fig)


# --- 2) Design & optimization setup ---
design = Design(width=W, height=H, material=Material(permittivity=EPS_CLAD))

# Fixed access waveguides outside optimization region.
design += Rectangle(
    position=(0.0, Y_IN - 0.5 * WG_W),
    width=X_INV0,
    height=WG_W,
    material=Material(permittivity=EPS_CORE),
)
design += Rectangle(
    position=(X_INV1, Y_TOP - 0.5 * WG_W),
    width=W - X_INV1,
    height=WG_W,
    material=Material(permittivity=EPS_CORE),
)
design += Rectangle(
    position=(X_INV1, Y_BOT - 0.5 * WG_W),
    width=W - X_INV1,
    height=WG_W,
    material=Material(permittivity=EPS_CORE),
)

opt_region = Rectangle(
    position=(X_INV0, Y_INV0),
    width=INV_W,
    height=INV_H,
    material=Material(permittivity=EPS_CORE),
)
design += opt_region

grid = design.rasterize(DX)
mask = create_optimization_mask(grid, opt_region)

opt = TopologyManager(
    design=design,
    region_mask=mask,
    resolution=DX,
    optimizer="Adam",
    learning_rate=0.003,
    filter_radius=0.05 * UM,
    eps_min=EPS_CLAD,
    eps_max=EPS_CORE,
    beta_schedule=(1.0, 20.0),
    filter_type="conic",
)

if INITIAL_ASYM_NOISE > 0.0:
    rng = np.random.default_rng(7)
    noise = np.zeros_like(opt.design_density, dtype=float)
    noise[mask] = INITIAL_ASYM_NOISE * (rng.random(np.count_nonzero(mask)) - 0.5)
    opt.design_density = np.clip(opt.design_density + noise, 0.0, 1.0)

base_eps = grid.permittivity.copy()
removed_files = cleanup_old_wdm_outputs()
if removed_files > 0:
    print(f"Removed {removed_files} stale WDM output files from previous runs.")
save_setup_sources_monitors_plot(grid, path="wdm_setup_sources_monitors.png")
print("Saved setup plot: wdm_setup_sources_monitors.png")

waveforms = {}
print("Waveform timing check (pulse should clear the full device):")
for wl, _target in WAVELENGTH_CASES:
    t_axis, sig, t_total, t_flight = build_time_and_signal(wl)
    waveforms[wl] = (t_axis, sig)
    print(
        f"  wl={wl / UM:.3f} um | total={t_total * 1e15:.1f} fs | "
        f"flight_est={t_flight * 1e15:.1f} fs | ratio={t_total / t_flight:.2f}x"
    )


# --- 3) Optimization loop ---
objective_history = []
objective_ema_history = []
route_mean_history = []
softmin_history = []
tx_hist = {
    WL_SHORT: {"target": [], "leak": []},
    WL_LONG: {"target": [], "leak": []},
}
fom_hist = {WL_SHORT: [], WL_LONG: []}
ema_objective = None
throughput_ref = {WL_SHORT: None, WL_LONG: None}
prev_phys_plot = None
prev_design_density = None

print(f"Starting simplified 1x2 WDM topology optimization for {STEPS} steps...")
for step in range(STEPS):
    step_id = step + 1
    beta = continuation_value_two_stage(
        step,
        STEPS,
        BETA_START,
        BETA_MID,
        BETA_END,
        stage1_frac=BETA_STAGE1_FRAC,
        power1=1.0,
        power2=1.4,
    )
    phys_density = opt.get_physical_density(beta)

    grid.permittivity[:] = base_eps
    grid.permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)

    leak_weight = continuation_value(step, STEPS, LEAK_W_START, LEAK_W_END, power=1.0)
    grad_scale = continuation_value(step, STEPS, GRAD_SCALE_START, GRAD_SCALE_END, power=1.0)

    per_wl_fom = []
    per_wl_grad = []
    route_mean = 0.0
    step_report = []
    forward_cache = []

    for wl, target_port in WAVELENGTH_CASES:
        t_axis, sig = waveforms[wl]
        fwd_hist, top_energy, bot_energy, total_out = run_forward(
            grid=grid,
            wavelength=wl,
            time=t_axis,
            signal=sig,
            save_fields=True,
        )

        if throughput_ref[wl] is None:
            throughput_ref[wl] = max(total_out, 1e-30)
        norm_scale = throughput_ref[wl]
        target_e = top_energy if target_port == "top" else bot_energy
        leak_e = bot_energy if target_port == "top" else top_energy
        target_tx = target_e / norm_scale
        leak_tx = leak_e / norm_scale

        route_mean += target_tx
        tx_hist[wl]["target"].append(100.0 * target_tx)
        tx_hist[wl]["leak"].append(100.0 * leak_tx)

        forward_cache.append(
            {
                "wl": wl,
                "target_port": target_port,
                "time": t_axis,
                "signal": sig,
                "fwd_hist": fwd_hist,
                "tx_top": top_energy,
                "tx_bot": bot_energy,
                "norm_scale": norm_scale,
                "target_tx": target_tx,
                "leak_tx": leak_tx,
            }
        )

    route_mean /= len(WAVELENGTH_CASES)
    route_mean_history.append(route_mean)

    grad_maps_by_wl = {}

    for item in forward_cache:
        wl = item["wl"]
        target_port = item["target_port"]
        t_axis = item["time"]
        sig = item["signal"]
        fwd_hist = item["fwd_hist"]

        tx_top = item["tx_top"]
        tx_bot = item["tx_bot"]
        norm_scale = item["norm_scale"]
        target_tx = item["target_tx"]
        leak_tx = item["leak_tx"]

        adj_top = run_adjoint(grid=grid, wavelength=wl, target_port="top", time=t_axis, signal=sig)
        adj_bot = run_adjoint(grid=grid, wavelength=wl, target_port="bottom", time=t_axis, signal=sig)

        grad_top = np.array(compute_overlap_gradient(fwd_hist, adj_top))
        grad_bot = np.array(compute_overlap_gradient(fwd_hist, adj_bot))

        if target_port == "top":
            grad_target = grad_top
            grad_leak = grad_bot
        else:
            grad_target = grad_bot
            grad_leak = grad_top

        fom_k = target_tx - leak_weight * leak_tx
        grad_fom_k = ADJOINT_GRAD_SIGN * (grad_target - leak_weight * grad_leak)

        per_wl_fom.append(fom_k)
        per_wl_grad.append(grad_fom_k)
        fom_hist[wl].append(100.0 * fom_k)
        grad_maps_by_wl[wl] = grad_fom_k

        step_report.append(
            f"{wl / UM:.3f}um->{target_port}: "
            f"T={100.0 * target_tx:5.1f}% "
            f"L={100.0 * leak_tx:5.1f}% "
            f"FoM={100.0 * fom_k:5.1f}%"
        )

    # Weight long-wavelength objective in gradually to avoid symmetric early cancellation.
    long_w = min(1.0, step / max(1, WL_LONG_RAMP_STEPS))
    wl_weights = np.array([1.0, long_w], dtype=float)
    wl_weights /= np.sum(wl_weights)

    objective_route = float(np.dot(wl_weights, np.asarray(per_wl_fom, dtype=float)))
    softmin_history.append(objective_route)

    grad_total = np.zeros_like(grid.permittivity, dtype=float)
    for w_i, grad_i in zip(wl_weights, per_wl_grad):
        grad_total += w_i * grad_i

    current_density = np.mean(phys_density[mask])
    grad_penalty = PENALTY_STRENGTH * (current_density - TARGET_DENSITY)
    grad_total[mask] -= grad_penalty

    grad_abs = np.abs(grad_total[mask])
    if grad_abs.size > 0:
        clip_val = np.percentile(grad_abs, CLIP_PCT)
        if clip_val > 0:
            grad_total[mask] = np.clip(grad_total[mask], -clip_val, clip_val)

    grad_total[mask] *= grad_scale

    penalty_value = 0.5 * PENALTY_STRENGTH * (current_density - TARGET_DENSITY) ** 2
    total_objective = objective_route - penalty_value

    max_update = opt.apply_gradient(grad_total, beta)

    objective_history.append(total_objective)
    opt.objective_history.append(total_objective)
    if ema_objective is None:
        ema_objective = total_objective
    else:
        ema_objective = EMA_ALPHA * total_objective + (1.0 - EMA_ALPHA) * ema_objective
    objective_ema_history.append(ema_objective)

    should_debug = (step == 0) or (step_id % DEBUG_EVERY == 0) or (step_id == STEPS)
    if should_debug:
        print(
            f"[{step_id:03d}/{STEPS}] Obj={total_objective:.4f} "
            f"(meanFoM={100.0 * objective_route:.1f}% meanT={100.0 * route_mean:.1f}% "
            f"wL={leak_weight:.2f} wLong={long_w:.2f} s={grad_scale:.2f} "
            f"mat={current_density:.2f} beta={beta:.2f} dmax={max_update:.3e}) | "
            + " | ".join(step_report)
        )

    if should_debug:
        is_baseline_debug = prev_phys_plot is None
        phys_plot = opt.get_physical_density(beta)
        grid.permittivity[:] = base_eps
        grid.permittivity[mask] = opt.eps_min + phys_plot[mask] * (opt.eps_max - opt.eps_min)

        rho_slice = phys_plot[mask]
        rho_min = float(np.min(rho_slice)) if rho_slice.size > 0 else 0.0
        rho_max = float(np.max(rho_slice)) if rho_slice.size > 0 else 0.0
        grad_slice = grad_total[mask]
        grad_rms = float(np.sqrt(np.mean(grad_slice * grad_slice))) if grad_slice.size > 0 else 0.0

        plt.imsave(
            f"wdm_topo_{step_id:03d}.png",
            grid.permittivity.T,
            cmap="gray",
            origin="lower",
        )
        plt.imsave(
            f"wdm_density_{step_id:03d}.png",
            phys_plot.T,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
        )

        if prev_phys_plot is None:
            dphys_rms = 0.0
            dphys_max = 0.0
            delta_phys = np.zeros_like(phys_plot)
        else:
            delta_phys = phys_plot - prev_phys_plot
            d_slice = delta_phys[mask]
            if d_slice.size > 0:
                dphys_rms = float(np.sqrt(np.mean(d_slice * d_slice)))
                dphys_max = float(np.max(np.abs(d_slice)))
            else:
                dphys_rms = 0.0
                dphys_max = 0.0

        if prev_design_density is None:
            dden_rms = 0.0
            dden_max = 0.0
            delta_den = np.zeros_like(opt.design_density)
        else:
            delta_den = opt.design_density - prev_design_density
            d_slice_den = delta_den[mask]
            if d_slice_den.size > 0:
                dden_rms = float(np.sqrt(np.mean(d_slice_den * d_slice_den)))
                dden_max = float(np.max(np.abs(d_slice_den)))
            else:
                dden_rms = 0.0
                dden_max = 0.0

        if not is_baseline_debug:
            save_gradient_debug_image(f"wdm_topo_delta_{step_id:03d}.png", delta_phys)
            save_gradient_debug_image(f"wdm_density_delta_{step_id:03d}.png", delta_den)
        prev_phys_plot = np.array(phys_plot, copy=True)
        prev_design_density = np.array(opt.design_density, copy=True)

        save_gradient_debug_image(f"wdm_grad_total_{step_id:03d}.png", grad_total)
        save_gradient_debug_image(f"wdm_grad_short_{step_id:03d}.png", grad_maps_by_wl[WL_SHORT])
        save_gradient_debug_image(f"wdm_grad_long_{step_id:03d}.png", grad_maps_by_wl[WL_LONG])

        t_short, s_short = waveforms[WL_SHORT]
        t_long, s_long = waveforms[WL_LONG]
        _, _, flux_short = run_forward_flux_map(grid, WL_SHORT, t_short, s_short)
        _, _, flux_long = run_forward_flux_map(grid, WL_LONG, t_long, s_long)
        save_flux_debug_image(f"wdm_flux_short_{step_id:03d}.png", flux_short)
        save_flux_debug_image(f"wdm_flux_long_{step_id:03d}.png", flux_long)

        if is_baseline_debug:
            print(
                f"    design-state: grad_rms={grad_rms:.3e} rho=[{rho_min:.3f},{rho_max:.3f}] "
                "(baseline step; delta plots start next debug step)"
            )
        else:
            print(
                f"    design-change: dphys_rms={dphys_rms:.3e} dphys_max={dphys_max:.3e} "
                f"dden_rms={dden_rms:.3e} dden_max={dden_max:.3e} grad_rms={grad_rms:.3e} "
                f"rho=[{rho_min:.3f},{rho_max:.3f}] "
                f"(saved wdm_topo_delta_{step_id:03d}.png, wdm_density_delta_{step_id:03d}.png)"
            )


# --- 4) Project to binary and run final wavelength-resolved flux maps ---
beta_final = opt.beta_end
phys_final = opt.get_physical_density(beta_final)
phys_binary = (phys_final >= 0.5).astype(float)

grid.permittivity[:] = base_eps
grid.permittivity[mask] = opt.eps_min + phys_binary[mask] * (opt.eps_max - opt.eps_min)

binary_design = (grid.permittivity > 0.5 * (EPS_CLAD + EPS_CORE)).astype(float)

final_flux_maps = {}
print("\nFinal routing check on binary projected design:")
for wl, target_port in WAVELENGTH_CASES:
    t_axis, sig = waveforms[wl]
    tx_top, tx_bot, flux_map = run_forward_flux_map(
        grid=grid,
        wavelength=wl,
        time=t_axis,
        signal=sig,
    )
    final_flux_maps[wl] = flux_map
    target_tx = tx_top if target_port == "top" else tx_bot
    leak_tx = tx_bot if target_port == "top" else tx_top
    print(
        f"  wl={wl / UM:.3f} um target={target_port}: "
        f"target={100.0 * target_tx:.2f}% leak={100.0 * leak_tx:.2f}%"
    )

red_flux = final_flux_maps[WL_SHORT]
blue_flux = final_flux_maps[WL_LONG]
red_scale = np.percentile(red_flux, 99.0)
blue_scale = np.percentile(blue_flux, 99.0)
red_scale = red_scale if red_scale > 0 else 1.0
blue_scale = blue_scale if blue_scale > 0 else 1.0
red_norm = np.clip(red_flux / red_scale, 0.0, 1.0)
blue_norm = np.clip(blue_flux / blue_scale, 0.0, 1.0)

plot_mask = binary_design.T > 0.5
red_layer = np.clip(red_norm.T, 0.0, 1.0) ** 0.72
blue_layer = np.clip(blue_norm.T, 0.0, 1.0) ** 0.72

ny_plot, nx_plot = plot_mask.shape
overlay_rgb = np.zeros((ny_plot, nx_plot, 3), dtype=float)
overlay_rgb[..., 0] = red_layer
overlay_rgb[..., 2] = blue_layer

eroded = np.zeros_like(plot_mask, dtype=bool)
eroded[1:-1, 1:-1] = (
    plot_mask[1:-1, 1:-1]
    & plot_mask[:-2, 1:-1]
    & plot_mask[2:, 1:-1]
    & plot_mask[1:-1, :-2]
    & plot_mask[1:-1, 2:]
)
outline = plot_mask & (~eroded)
overlay_rgb[outline] = 1.0

plt.imsave("wdm_final_binary_flux_overlay.png", overlay_rgb, origin="lower")


# --- 5) Plots ---
steps = np.arange(1, STEPS + 1)

plt.figure(figsize=(9, 5))
plt.plot(steps, 100.0 * np.array(objective_history), "k-", linewidth=1.5, label="Total objective")
plt.plot(
    steps,
    100.0 * np.array(objective_ema_history),
    color="tab:orange",
    linewidth=2.2,
    label=f"Objective EMA (alpha={EMA_ALPHA:.2f})",
)
plt.plot(steps, 100.0 * np.array(route_mean_history), "g--", linewidth=1.4, label="Mean target T")
plt.plot(steps, 100.0 * np.array(softmin_history), "b--", linewidth=1.4, label="Mean route FoM")
plt.xlabel("Optimization step")
plt.ylabel("FoM (%)")
plt.title("1x2 WDM objective progress (simplified)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("wdm_objective_vs_step.png", dpi=160)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(steps, tx_hist[WL_SHORT]["target"], "b-", linewidth=2, label="1.31 um -> top (target)")
plt.plot(steps, tx_hist[WL_SHORT]["leak"], "b--", linewidth=1.8, label="1.31 um leak")
plt.plot(steps, tx_hist[WL_LONG]["target"], "r-", linewidth=2, label="1.55 um -> bottom (target)")
plt.plot(steps, tx_hist[WL_LONG]["leak"], "r--", linewidth=1.8, label="1.55 um leak")
plt.xlabel("Optimization step")
plt.ylabel("Transmission (%)")
plt.title("Port routing and leakage")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("wdm_routing_vs_step.png", dpi=160)
plt.close()

plt.figure(figsize=(9, 4.8))
plt.plot(steps, fom_hist[WL_SHORT], "b-", linewidth=2, label="FoM 1.31 um")
plt.plot(steps, fom_hist[WL_LONG], "r-", linewidth=2, label="FoM 1.55 um")
plt.xlabel("Optimization step")
plt.ylabel("Per-wavelength FoM (%)")
plt.title("Per-channel FoM = T_target - wL*T_leak")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("wdm_per_channel_fom_vs_step.png", dpi=160)
plt.close()

print(
    "Saved: wdm_objective_vs_step.png, wdm_routing_vs_step.png, "
    "wdm_per_channel_fom_vs_step.png, wdm_final_binary_flux_overlay.png, "
    "wdm_topo_*.png, wdm_density_*.png, wdm_grad_*.png, wdm_flux_*.png"
)
