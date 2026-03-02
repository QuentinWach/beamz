import numpy as np
import matplotlib.pyplot as plt

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
- 2 adjoint sims (one for each wavelength / target port)
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

STEPS = 60
FIELD_SUBSAMPLE = 2
TARGET_DENSITY = 0.5
PENALTY_STRENGTH = 0.8


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
    filter_radius=0.22 * UM,
    eps_min=EPS_CLAD,
    eps_max=EPS_CORE,
    beta_schedule=(1.0, 18.0),
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
tx_hist = {
    WL_SHORT: {"target": [], "leak": []},
    WL_LONG: {"target": [], "leak": []},
}

print(f"Starting 1x2 WDM topology optimization for {STEPS} steps...")
for step in range(STEPS):
    beta, phys_density = opt.update_design(step, STEPS)

    # Apply current physical density to the optimization mask.
    grid.permittivity[:] = base_eps
    grid.permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)

    grad_total = np.zeros_like(grid.permittivity, dtype=float)
    objective_step = 0.0
    step_report = []

    # Exactly two forward + two adjoint simulations per step.
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
        objective_step += target_tx

        tx_hist[wl]["target"].append(100.0 * target_tx)
        tx_hist[wl]["leak"].append(100.0 * leak_tx)

        adj_hist = run_adjoint(
            grid=grid,
            wavelength=wl,
            target_port=target_port,
            time=t_axis,
            signal=sig,
        )
        grad_total += np.array(compute_overlap_gradient(fwd_hist, adj_hist))

        step_report.append(
            f"{wl / UM:.3f}um->{target_port}: target={100.0 * target_tx:5.1f}% "
            f"leak={100.0 * leak_tx:5.1f}%"
        )

    objective_step /= len(WAVELENGTH_CASES)
    objective_history.append(objective_step)
    opt.objective_history.append(objective_step)

    # Material-usage regularization.
    current_density = np.mean(phys_density[mask])
    grad_penalty = PENALTY_STRENGTH * (current_density - TARGET_DENSITY)
    grad_total[mask] -= grad_penalty

    penalty_value = 0.5 * PENALTY_STRENGTH * (current_density - TARGET_DENSITY) ** 2
    total_objective = objective_step - penalty_value

    max_update = opt.apply_gradient(grad_total / len(WAVELENGTH_CASES), beta)

    print(
        f"[{step + 1:02d}/{STEPS}] Obj={total_objective:.4f} "
        f"(route={100.0 * objective_step:.1f}% mat={current_density:.2f} "
        f"beta={beta:.2f} dmax={max_update:.3e}) | " + " | ".join(step_report)
    )

    if step % 5 == 0 or step == STEPS - 1:
        plt.imsave(
            f"wdm_topo_{step:03d}.png",
            grid.permittivity.T,
            cmap="gray",
            origin="lower",
        )


# --- 4) Final routing check ---
print("\nFinal routing check:")
for wl, target_port in WAVELENGTH_CASES:
    t_axis, sig = waveforms[wl]
    _, tx_top, tx_bot = run_forward(
        grid=grid,
        wavelength=wl,
        time=t_axis,
        signal=sig,
        save_fields=False,
    )
    target_tx = tx_top if target_port == "top" else tx_bot
    leak_tx = tx_bot if target_port == "top" else tx_top
    print(
        f"  wl={wl / UM:.3f} um target={target_port}: "
        f"target={100.0 * target_tx:.2f}% leak={100.0 * leak_tx:.2f}%"
    )


# --- 5) Plots ---
steps = np.arange(1, STEPS + 1)

plt.figure(figsize=(9, 5))
plt.plot(steps, 100.0 * np.array(objective_history), "k-", linewidth=2)
plt.xlabel("Optimization step")
plt.ylabel("Average routed transmission (%)")
plt.title("1x2 WDM objective progress")
plt.grid(alpha=0.3)
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

print("Saved: wdm_objective_vs_step.png, wdm_routing_vs_step.png, wdm_topo_*.png")
