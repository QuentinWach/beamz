import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from beamz import *
from beamz.optimization.topology import (
    TopologyManager,
    compute_overlap_gradient,
    create_optimization_mask,
)


"""
Inverse-design example: 1x2 wavelength demultiplexer (Si on SiO2) in 2D.

Inspired by examples/4_topo.py with a multi-wavelength objective:
- lambda_short -> top output
- lambda_long  -> bottom output

Per optimization step, this script runs:
- 2 forward sims (one for each wavelength)
- 4 adjoint sims (target + leakage port for each wavelength)
"""


# Unit convenience (ASCII-only)
UM = 1e-6


# --- 1) Setup ---
# Geometry targets requested:
# - waveguide width ~= 0.55 um
# - inverse-design region ~= 3.5 x 3.5 um
# - output edge-to-edge gap ~= 0.8 um
WG_W = 0.55 * UM
OUT_GAP = 0.80 * UM
INV_W = 3.50 * UM
INV_H = 3.50 * UM

# Simulation window (kept compact but with room for source/monitors/PML)
W = 8.0 * UM
H = 6.0 * UM
PML_T = 1.0 * UM

# Two target wavelengths for WDM routing
WL_SHORT = 1.31 * UM
WL_LONG = 1.55 * UM
WAVELENGTH_CASES = [
    (WL_SHORT, "top"),
    (WL_LONG, "bottom"),
]

# Si core on SiO2 cladding (effective-index 2D model)
N_CORE = 3.48
N_CLAD = 1.444
EPS_CORE = N_CORE**2
EPS_CLAD = N_CLAD**2

# Grid/time discretization from shortest wavelength
DX, DT = calc_optimal_fdtd_params(
    wavelength=WL_SHORT,
    n_max=N_CORE,
    points_per_wavelength=14,
)

STEPS = 80
FIELD_SUBSAMPLE = 2
TARGET_DENSITY = 0.5
PENALTY_STRENGTH = 0.8
EMA_ALPHA = 0.20

# Continuation schedule endpoints for multi-objective WDM optimization.
# alpha_xtalk: leak suppression weight in FoM_k = T_target - alpha*T_leak
ALPHA_XTALK_START = 0.15
ALPHA_XTALK_END = 1.00
# gamma_softmin: fairness sharpness over wavelengths. Low=mean-like, high=worst-like.
GAMMA_SOFTMIN_START = 2.0
GAMMA_SOFTMIN_END = 14.0
# Gradient step scale (acts like learning-rate decay without rebuilding optax optimizer).
GRAD_SCALE_START = 1.00
GRAD_SCALE_END = 0.45
# Gradient clipping schedule (%tile of |grad| inside design region).
CLIP_PCT_START = 99.8
CLIP_PCT_END = 98.8


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
X_MON_IN = X_SRC + 0.35 * UM
X_MON_OUT = W - PML_T - 0.35 * UM
MON_SPAN = 2.6 * WG_W


def build_time_and_signal(wavelength):
    # Travel estimate in slow-medium limit (conservative enough for this setup).
    flight_time = (X_MON_OUT - X_SRC) * N_CORE / LIGHT_SPEED
    ramp_duration = 8.0 * wavelength / LIGHT_SPEED

    # Keep simulation long enough so the pulse is emitted and crosses full device.
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


def make_vertical_monitor(grid, x, y_center, span):
    return Monitor(
        design=grid,
        start=(x, y_center - 0.5 * span),
        end=(x, y_center + 0.5 * span),
        accumulate_power=True,
        record_fields=False,
    )


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

    mon_in = make_vertical_monitor(grid, X_MON_IN, Y_IN, MON_SPAN)
    mon_top = make_vertical_monitor(grid, X_MON_OUT, Y_TOP, MON_SPAN)
    mon_bot = make_vertical_monitor(grid, X_MON_OUT, Y_BOT, MON_SPAN)

    sim = Simulation(
        grid,
        [src, mon_in, mon_top, mon_bot],
        [PML(edges="all", thickness=PML_T)],
        time=time,
        resolution=DX,
    )

    save = ["Ez"] if save_fields else []
    results = sim.run(save_fields=save, field_subsample=FIELD_SUBSAMPLE)

    in_energy = np.sum(mon_in.power_history) * DT
    top_energy = np.sum(mon_top.power_history) * DT
    bot_energy = np.sum(mon_bot.power_history) * DT

    if np.abs(in_energy) < 1e-30:
        in_energy = 1.0

    top_tx = np.abs(top_energy) / np.abs(in_energy)
    bot_tx = np.abs(bot_energy) / np.abs(in_energy)

    ez_hist = []
    if save_fields:
        ez_hist = [np.array(frame) for frame in results.get("fields", {}).get("Ez", [])]
        if not ez_hist:
            raise RuntimeError("Forward simulation returned no Ez history.")

    return ez_hist, top_tx, bot_tx


def run_adjoint(grid, wavelength, target_port, time, signal):
    y_target = Y_TOP if target_port == "top" else Y_BOT
    src = ModeSource(
        grid,
        center=(X_MON_OUT, y_target),
        width=3.0 * WG_W,
        wavelength=wavelength,
        pol="tm",
        signal=signal,
        direction="-x",
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


def opposite_port(port_name):
    return "bottom" if port_name == "top" else "top"


def continuation_value(step, total_steps, start, end, power=1.0):
    if total_steps <= 1:
        return end
    frac = step / (total_steps - 1)
    frac = np.clip(frac, 0.0, 1.0) ** power
    return start + frac * (end - start)


def softmin_weights(values, gamma):
    # Stable soft-min weighting: higher weight on lower-performing channels.
    vals = np.array(values, dtype=float)
    logits = -gamma * vals
    logits -= np.max(logits)
    expv = np.exp(logits)
    return expv / np.sum(expv)


def run_forward_flux_map(grid, wavelength, time, signal):
    """
    Run a final forward simulation and accumulate time-integrated |S| map.
    For 2D TM fields in this codebase: Sx ~ -Ez*Hy, Sy ~ Ez*Hx.
    """
    src = ModeSource(
        grid,
        center=(X_SRC, Y_IN),
        width=3.0 * WG_W,
        wavelength=wavelength,
        pol="tm",
        signal=signal,
        direction="+x",
    )

    mon_in = make_vertical_monitor(grid, X_MON_IN, Y_IN, MON_SPAN)
    mon_top = make_vertical_monitor(grid, X_MON_OUT, Y_TOP, MON_SPAN)
    mon_bot = make_vertical_monitor(grid, X_MON_OUT, Y_BOT, MON_SPAN)

    sim = Simulation(
        grid,
        [src, mon_in, mon_top, mon_bot],
        [PML(edges="all", thickness=PML_T)],
        time=time,
        resolution=DX,
    )
    results = sim.run(save_fields=["Ez", "Hx", "Hy"], field_subsample=1)

    in_energy = np.sum(mon_in.power_history) * DT
    top_energy = np.sum(mon_top.power_history) * DT
    bot_energy = np.sum(mon_bot.power_history) * DT
    if np.abs(in_energy) < 1e-30:
        in_energy = 1.0

    ez_hist = [np.array(frame) for frame in results.get("fields", {}).get("Ez", [])]
    hx_hist = [np.array(frame) for frame in results.get("fields", {}).get("Hx", [])]
    hy_hist = [np.array(frame) for frame in results.get("fields", {}).get("Hy", [])]
    n_frames = min(len(ez_hist), len(hx_hist), len(hy_hist))
    if n_frames == 0:
        raise RuntimeError("Final forward simulation returned no Ez/Hx/Hy history.")

    flux_map = np.zeros_like(ez_hist[0], dtype=float)
    for i in range(n_frames):
        ez_i = ez_hist[i]
        hx_i = hx_hist[i]
        hy_i = hy_hist[i]

        # Yee-grid field components can differ by one cell in x/y; use overlap area.
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

    tx_top = np.abs(top_energy) / np.abs(in_energy)
    tx_bot = np.abs(bot_energy) / np.abs(in_energy)
    return tx_top, tx_bot, flux_map


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

# Optimization region placeholder.
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
    learning_rate=0.01,
    filter_radius=0.05 * UM,
    eps_min=EPS_CLAD,
    eps_max=EPS_CORE,
    beta_schedule=(1.0, 25.0),
    filter_type="conic",
)

base_eps = grid.permittivity.copy()

# Precompute waveforms and print pulse/flight diagnostics.
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
_ema_objective = None

print(f"Starting 1x2 WDM topology optimization for {STEPS} steps...")
for step in range(STEPS):
    beta, phys_density = opt.update_design(step, STEPS)

    # Apply current physical density to the optimization mask.
    grid.permittivity[:] = base_eps
    grid.permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)

    alpha_xtalk = continuation_value(
        step, STEPS, ALPHA_XTALK_START, ALPHA_XTALK_END, power=1.0
    )
    gamma_softmin = continuation_value(
        step, STEPS, GAMMA_SOFTMIN_START, GAMMA_SOFTMIN_END, power=1.2
    )
    grad_scale = continuation_value(
        step, STEPS, GRAD_SCALE_START, GRAD_SCALE_END, power=1.0
    )
    clip_pct = continuation_value(
        step, STEPS, CLIP_PCT_START, CLIP_PCT_END, power=1.0
    )

    per_wl_fom = []
    per_wl_grad = []
    route_mean = 0.0
    step_report = []

    # Two forward + four adjoint simulations per step:
    # - 1 adjoint for target-port transmission gradient
    # - 1 adjoint for wrong-port (crosstalk) gradient
    for wl, target_port in WAVELENGTH_CASES:
        t_axis, sig = waveforms[wl]

        fwd_hist, tx_top, tx_bot = run_forward(
            grid=grid,
            wavelength=wl,
            time=t_axis,
            signal=sig,
            save_fields=True,
        )

        target_tx = tx_top if target_port == "top" else tx_bot
        leak_tx = tx_bot if target_port == "top" else tx_top
        route_mean += target_tx

        tx_hist[wl]["target"].append(100.0 * target_tx)
        tx_hist[wl]["leak"].append(100.0 * leak_tx)
        fom_k = target_tx - alpha_xtalk * leak_tx
        fom_hist[wl].append(100.0 * fom_k)

        adj_target = run_adjoint(
            grid=grid,
            wavelength=wl,
            target_port=target_port,
            time=t_axis,
            signal=sig,
        )

        leak_port = opposite_port(target_port)
        adj_leak = run_adjoint(
            grid=grid,
            wavelength=wl,
            target_port=leak_port,
            time=t_axis,
            signal=sig,
        )
        grad_target = np.array(compute_overlap_gradient(fwd_hist, adj_target))
        grad_leak = np.array(compute_overlap_gradient(fwd_hist, adj_leak))
        grad_fom_k = grad_target - alpha_xtalk * grad_leak

        per_wl_fom.append(fom_k)
        per_wl_grad.append(grad_fom_k)

        step_report.append(
            f"{wl / UM:.3f}um->{target_port}: T={100.0 * target_tx:5.1f}% "
            f"L={100.0 * leak_tx:5.1f}% FoM={100.0 * fom_k:5.1f}%"
        )

    route_mean /= len(WAVELENGTH_CASES)
    route_mean_history.append(route_mean)

    # Fairness: soft-min over wavelengths (approaches worst-channel objective as gamma increases).
    soft_w = softmin_weights(per_wl_fom, gamma_softmin)
    _fom_arr = np.array(per_wl_fom, dtype=float)
    _logits = -gamma_softmin * _fom_arr
    _logits_max = np.max(_logits)
    _lse = _logits_max + np.log(np.sum(np.exp(_logits - _logits_max)))
    objective_route = -_lse / gamma_softmin
    softmin_history.append(objective_route)

    grad_total = np.zeros_like(grid.permittivity, dtype=float)
    for w_i, grad_i in zip(soft_w, per_wl_grad):
        grad_total += w_i * grad_i

    # Material-usage regularization.
    current_density = np.mean(phys_density[mask])
    grad_penalty = PENALTY_STRENGTH * (current_density - TARGET_DENSITY)
    grad_total[mask] -= grad_penalty

    # Clip tail gradients in the design region for stability.
    grad_abs = np.abs(grad_total[mask])
    if grad_abs.size > 0:
        clip_val = np.percentile(grad_abs, clip_pct)
        if clip_val > 0:
            grad_total[mask] = np.clip(grad_total[mask], -clip_val, clip_val)

    # Step-size continuation (effective LR decay).
    grad_total[mask] *= grad_scale

    penalty_value = 0.5 * PENALTY_STRENGTH * (current_density - TARGET_DENSITY) ** 2
    total_objective = objective_route - penalty_value

    objective_history.append(total_objective)
    opt.objective_history.append(total_objective)
    if _ema_objective is None:
        _ema_objective = total_objective
    else:
        _ema_objective = EMA_ALPHA * total_objective + (1.0 - EMA_ALPHA) * _ema_objective
    objective_ema_history.append(_ema_objective)

    max_update = opt.apply_gradient(grad_total, beta)

    print(
        f"[{step + 1:02d}/{STEPS}] Obj={total_objective:.4f} "
        f"(softmin={100.0 * objective_route:.1f}% meanT={100.0 * route_mean:.1f}% "
        f"a={alpha_xtalk:.2f} g={gamma_softmin:.1f} s={grad_scale:.2f} "
        f"mat={current_density:.2f} beta={beta:.2f} dmax={max_update:.3e}) | "
        + " | ".join(step_report)
    )

    if step % 5 == 0 or step == STEPS - 1:
        plt.imsave(
            f"wdm_topo_{step:03d}.png",
            grid.permittivity.T,
            cmap="gray",
            origin="lower",
        )


# --- 4) Project to binary and run final wavelength-resolved flux maps ---
# Enforce a binary projected design for final validation/visualization.
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

# Normalize final flux maps robustly for overlay.
red_flux = final_flux_maps[WL_SHORT]
blue_flux = final_flux_maps[WL_LONG]
red_scale = np.percentile(red_flux, 99.5)
blue_scale = np.percentile(blue_flux, 99.5)
red_scale = red_scale if red_scale > 0 else 1.0
blue_scale = blue_scale if blue_scale > 0 else 1.0
red_norm = np.clip(red_flux / red_scale, 0.0, 1.0)
blue_norm = np.clip(blue_flux / blue_scale, 0.0, 1.0)

red_cmap = LinearSegmentedColormap.from_list(
    "pure_red_overlay", [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.95)]
)
blue_cmap = LinearSegmentedColormap.from_list(
    "pure_blue_overlay", [(0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.95)]
)

extent_um = [0.0, W / UM, 0.0, H / UM]
plt.figure(figsize=(8.2, 6.0))
plt.imshow(
    binary_design.T,
    cmap="gray",
    origin="lower",
    interpolation="nearest",
    extent=extent_um,
    vmin=0.0,
    vmax=1.0,
)
plt.imshow(
    red_norm.T,
    cmap=red_cmap,
    origin="lower",
    interpolation="bilinear",
    extent=extent_um,
    vmin=0.0,
    vmax=1.0,
)
plt.imshow(
    blue_norm.T,
    cmap=blue_cmap,
    origin="lower",
    interpolation="bilinear",
    extent=extent_um,
    vmin=0.0,
    vmax=1.0,
)
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title("Binary projected design with integrated flux overlay (red: 1.31 um, blue: 1.55 um)")
plt.tight_layout()
plt.savefig("wdm_final_binary_flux_overlay.png", dpi=220)
plt.close()


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
plt.plot(steps, 100.0 * np.array(softmin_history), "b--", linewidth=1.4, label="Soft-min route FoM")
plt.xlabel("Optimization step")
plt.ylabel("FoM (%)")
plt.title("1x2 WDM objective progress (fairness + continuation)")
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
plt.title("Per-channel FoM = T_target - alpha(step) * T_leak")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("wdm_per_channel_fom_vs_step.png", dpi=160)
plt.close()

# Additional ideas (not enabled here):
# 1) Reflection penalty:
#    Add a backward-power term and corresponding adjoint source near the input port,
#    then optimize FoM_k = T_target - a*T_leak - b*R.
# 2) Epigraph / explicit max-min:
#    Introduce scalar t and constrain each channel FoM_k >= t, then maximize t.
# 3) Multi-frequency per channel:
#    Replace each single wavelength with a small band sample set and average FoM.
# 4) Robust fabrication objective:
#    Evaluate FoM on eroded/nominal/dilated projections and optimize worst-case average.

print(
    "Saved: wdm_objective_vs_step.png, wdm_routing_vs_step.png, "
    "wdm_per_channel_fom_vs_step.png, wdm_final_binary_flux_overlay.png, "
    "wdm_topo_*.png"
)
