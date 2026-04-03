import matplotlib.pyplot as plt
import numpy as np

from beamz import *
from beamz.optimization.topology import (
    TopologyManager,
    compute_overlap_gradient,
    create_optimization_mask,
)
UM = 1e-6
PREFIX = "wdm_simple"

WG_W = 0.55 * UM
OUT_GAP = 0.95 * UM
INV_W = 3.50 * UM
INV_H = INV_W
INPUT_WG_LEN = 3.10 * UM
OUTPUT_WG_LEN = 3.10 * UM
W = INPUT_WG_LEN + INV_W + OUTPUT_WG_LEN
H = 7.0 * UM
PML_T = 1.2 * UM

WL_SHORT = 1.30 * UM
WL_LONG = 1.55 * UM
CASES = ((WL_SHORT, "top"), (WL_LONG, "bottom"))

N_CORE = 3.48
N_CLAD = 1.0
EPS_CORE = N_CORE**2
EPS_CLAD = N_CLAD**2

DX, DT = calc_optimal_fdtd_params(wavelength=WL_SHORT, n_max=N_CORE, points_per_wavelength=8)

STEPS = 150
PRINT_EVERY = 5
DEBUG_EVERY = 10
SAVE_DEBUG = True
FIELD_SUBSAMPLE = 1
EMA_ALPHA = 0.20

LEARNING_RATE = 0.0010
POLISH_LR = 0.0001
BETA_START = 12.0
BETA_END = 40.0
POLISH_BETA = 64.0
BLUR_START = 0.18 * UM
BLUR_END = 0.14 * UM
POLISH_BLUR = 0.12 * UM
LOSS_WEIGHT = 0.50
POWER_BOUND_WEIGHT = 4.0
POWER_SUM_TOL = 1.02
BINARITY_START = 0.01
BINARITY_END = 0.18
POLISH_BINARITY = 0.30
GRAD_CLIP_PCT = 99.5
GRAD_HARD_CAP = 50.0
NORMALIZE_PER_WL_GRAD = True
INITIAL_ASYM_NOISE = 1e-3
ENABLE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 20
POLISH_STEPS = 60
MAIN_STEPS = max(STEPS - POLISH_STEPS, 1)
POLISH_START = MAIN_STEPS + 1

Y_IN = 0.5 * H
Y_TOP = Y_IN + 0.5 * (WG_W + OUT_GAP)
Y_BOT = Y_IN - 0.5 * (WG_W + OUT_GAP)
X_INV0 = INPUT_WG_LEN
X_INV1 = X_INV0 + INV_W
Y_INV0 = 0.5 * (H - INV_H)
X_SRC = PML_T + 0.70 * UM
X_MON_REF = X_INV0 - 0.70 * UM
X_MON_IN = X_INV0 - 0.35 * UM
X_MON_OUT = W - PML_T - 0.35 * UM
MON_SPAN = 2.6 * WG_W


def build_waveform(wavelength):
    flight = (X_MON_OUT - X_SRC) * N_CORE / LIGHT_SPEED
    ramp = 12.0 * wavelength / LIGHT_SPEED
    total = 3.6 * flight + 4.5 * ramp
    gate = 1.2 * flight + 0.9 * ramp
    time = np.arange(0.0, total, DT)
    signal = ramped_cosine(
        time,
        1.0,
        LIGHT_SPEED / wavelength,
        ramp_duration=ramp,
        t_max=0.78 * total,
    )
    return {
        "time": time,
        "signal": signal,
        "gate_start": gate,
        "gate_index": int(np.searchsorted(time, gate, side="left")),
        "total": total,
        "flight": flight,
    }


def smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def step_settings(step):
    if step <= MAIN_STEPS:
        frac = 0.0 if MAIN_STEPS <= 1 else (step - 1) / (MAIN_STEPS - 1)
        beta = BETA_START + (BETA_END - BETA_START) * smoothstep01(frac)
        blur = BLUR_START + (BLUR_END - BLUR_START) * smoothstep01(frac)
        w_bin = BINARITY_START + (BINARITY_END - BINARITY_START) * smoothstep01(frac)
        return beta, blur, w_bin, False
    frac = 1.0 if POLISH_STEPS <= 1 else (step - POLISH_START) / (POLISH_STEPS - 1)
    beta = BETA_END + (POLISH_BETA - BETA_END) * smoothstep01(frac)
    blur = BLUR_END + (POLISH_BLUR - BLUR_END) * smoothstep01(frac)
    w_bin = BINARITY_END + (POLISH_BINARITY - BINARITY_END) * smoothstep01(frac)
    return beta, blur, w_bin, True


def modal_specs():
    return [
        PortSpec(
            name="in",
            monitor_name="in_norm",
            reference_monitor="in_ref",
            direction="+x",
            polarization="tm",
            incident_wave="plus",
            scattered_wave="minus",
        ),
        PortSpec(
            name="top",
            monitor_name="out_top",
            direction="+x",
            polarization="tm",
            incident_wave="minus",
            scattered_wave="plus",
        ),
        PortSpec(
            name="bottom",
            monitor_name="out_bottom",
            direction="+x",
            polarization="tm",
            incident_wave="minus",
            scattered_wave="plus",
        ),
    ]
def modal_metrics(sim, wavelength):
    result = sim.get_S_matrix_modal_dft(
        source_port="in",
        ports=modal_specs(),
        output_ports=["top", "bottom"],
        frequencies=np.array([LIGHT_SPEED / wavelength], dtype=float),
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-80.0,
    )
    if not bool(np.asarray(result["diagnostics"]["valid_mask"], dtype=bool)[0]):
        raise RuntimeError("Invalid incident amplitude at source port.")
    smat = result["s_matrix"]
    tx_top = float(np.abs(np.asarray(smat[("top", "in")], dtype=np.complex128)[0]) ** 2)
    tx_bot = float(np.abs(np.asarray(smat[("bottom", "in")], dtype=np.complex128)[0]) ** 2)
    p_in = max(float(np.asarray(result["diagnostics"]["P_in"], dtype=float)[0]), 1e-30)
    p_top = max(tx_top * p_in, 0.0)
    p_bot = max(tx_bot * p_in, 0.0)
    power_sum = (p_top + p_bot) / p_in
    return {
        "p_in": p_in,
        "p_top": p_top,
        "p_bot": p_bot,
        "power_sum": power_sum,
        "loss_est": max(0.0, 1.0 - power_sum),
        "tx_top": tx_top,
        "tx_bot": tx_bot,
    }

def run_forward(grid, wavelength, wave, fields=("Ez",)):
    source = ModeSource(
        grid,
        center=(X_SRC, Y_IN),
        width=3.0 * WG_W,
        wavelength=wavelength,
        pol="tm",
        signal=wave["signal"],
        direction="+x",
    )
    monitors = [
        Monitor(
            design=grid,
            start=(x, y - 0.5 * MON_SPAN),
            end=(x, y + 0.5 * MON_SPAN),
            name=name,
            accumulate_power=True,
            dft_enabled=True,
            dft_frequencies=np.array([LIGHT_SPEED / wavelength], dtype=float),
            dft_components=("Ez", "Hx", "Hy"),
            dft_window="rect",
            dft_t_start=float(wave["gate_start"]),
        )
        for name, x, y in (
            ("in_ref", X_MON_REF, Y_IN),
            ("in_norm", X_MON_IN, Y_IN),
            ("out_top", X_MON_OUT, Y_TOP),
            ("out_bottom", X_MON_OUT, Y_BOT),
        )
    ]
    sim = Simulation(
        grid,
        [source, *monitors],
        [PML(edges="all", thickness=PML_T)],
        time=wave["time"],
        resolution=DX,
    )
    results = sim.run(save_fields=list(fields), field_subsample=FIELD_SUBSAMPLE)
    return modal_metrics(sim, wavelength), results.get("fields", {})

def run_adjoint(grid, wavelength, target_port, wave):
    source = ModeSource(
        grid,
        center=(X_MON_OUT, Y_TOP if target_port == "top" else Y_BOT),
        width=3.0 * WG_W,
        wavelength=wavelength,
        pol="tm",
        signal=wave["signal"],
        direction="-x",
    )
    sim = Simulation(
        grid,
        [source],
        [PML(edges="all", thickness=PML_T)],
        time=wave["time"],
        resolution=DX,
    )
    return [np.array(frame) for frame in sim.run(save_fields=["Ez"], field_subsample=FIELD_SUBSAMPLE)["fields"]["Ez"]]

def flux_map(fields, time, gate_start):
    ez_hist = [np.array(frame) for frame in fields["Ez"]]
    hx_hist = [np.array(frame) for frame in fields["Hx"]]
    hy_hist = [np.array(frame) for frame in fields["Hy"]]
    n = min(len(ez_hist), len(hx_hist), len(hy_hist), len(time))
    flux = np.zeros_like(ez_hist[0], dtype=float)
    for i in range(n):
        if time[i] < gate_start:
            continue
        ny = min(ez_hist[i].shape[0], hx_hist[i].shape[0], hy_hist[i].shape[0])
        nx = min(ez_hist[i].shape[1], hx_hist[i].shape[1], hy_hist[i].shape[1])
        ez = ez_hist[i][:ny, :nx]
        hx = hx_hist[i][:ny, :nx]
        hy = hy_hist[i][:ny, :nx]
        flux[:ny, :nx] += np.sqrt((-ez * hy) ** 2 + (ez * hx) ** 2) * DT
    return flux

def save_flux(path, flux):
    scale = max(float(np.percentile(flux, 99.0)), 1.0)
    plt.imsave(path, np.clip(flux / scale, 0.0, 1.0).T, cmap="inferno", origin="lower")

def save_progress(path, hist):
    steps = np.arange(1, len(hist["objective"]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(steps, 100.0 * np.array(hist["objective"]), "k-", lw=1.6, label="Objective")
    axes[0].plot(steps, 100.0 * np.array(hist["ema"]), color="tab:orange", lw=2.0, label="EMA")
    axes[0].plot(steps, 100.0 * np.array(hist["route"]), "b--", lw=1.4, label="Route FoM")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)
    axes[0].set_ylabel("FoM (%)")
    axes[1].plot(steps, hist["tx"][WL_SHORT]["target"], "b-", lw=2.0, label="1.30 um target")
    axes[1].plot(steps, hist["tx"][WL_SHORT]["leak"], "b--", lw=1.8, label="1.30 um leak")
    axes[1].plot(steps, hist["tx"][WL_LONG]["target"], "r-", lw=2.0, label="1.55 um target")
    axes[1].plot(steps, hist["tx"][WL_LONG]["leak"], "r--", lw=1.8, label="1.55 um leak")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylabel("Power (%)")
    axes[2].plot(steps, hist["fom"][WL_SHORT], "b-", lw=2.0, label="FoM 1.30 um")
    axes[2].plot(steps, hist["fom"][WL_LONG], "r-", lw=2.0, label="FoM 1.55 um")
    axes[2].plot(steps, 100.0 * np.array(hist["power_sum"]), color="tab:purple", lw=1.2, label="Guided power sum")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.3)
    axes[2].set_xlabel("Optimization step")
    axes[2].set_ylabel("Percent (%)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)

def save_overlay(path, short_flux, long_flux, design_mask):
    r = np.clip(short_flux / max(float(np.percentile(short_flux, 99.0)), 1.0), 0.0, 1.0).T ** 0.72
    b = np.clip(long_flux / max(float(np.percentile(long_flux, 99.0)), 1.0), 0.0, 1.0).T ** 0.72
    mask = design_mask.T > 0.5
    base = np.zeros(mask.shape + (3,), dtype=float)
    base[mask] = 0.16
    rgb = np.maximum(base, np.stack([r, np.zeros_like(r), b], axis=-1))
    eroded = np.zeros_like(mask, dtype=bool)
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    )
    rgb[mask & (~eroded)] = 1.0
    plt.imsave(path, rgb, origin="lower")

def set_design(grid, base_eps, mask, density, opt):
    grid.permittivity[:] = base_eps
    grid.permittivity[mask] = opt.eps_min + density[mask] * (opt.eps_max - opt.eps_min)


def score_density(grid, density):
    set_design(grid, base_eps, mask, density, opt)
    route_terms, modal_data = [], {}
    for wl, target_port in CASES:
        modal, _ = run_forward(grid, wl, waves[wl])
        target = modal["p_top"] if target_port == "top" else modal["p_bot"]
        leak = modal["p_bot"] if target_port == "top" else modal["p_top"]
        norm = max(modal["p_in"], 1e-30)
        overflow = max(float(modal["power_sum"]) - 1.0, 0.0)
        route_terms.append(
            target / norm
            - leak / norm
            - LOSS_WEIGHT * modal["loss_est"]
            - POWER_BOUND_WEIGHT * overflow * overflow
        )
        modal_data[wl] = modal
    return float(np.mean(route_terms)), modal_data


def flux_artifacts(grid, density):
    set_design(grid, base_eps, mask, density, opt)
    fluxes, modal = {}, {}
    for wl in (WL_SHORT, WL_LONG):
        modal[wl], fields = run_forward(grid, wl, waves[wl], fields=("Ez", "Hx", "Hy"))
        fluxes[wl] = flux_map(fields, waves[wl]["time"], waves[wl]["gate_start"])
    design_mask = (grid.permittivity > 0.5 * (EPS_CLAD + EPS_CORE)).astype(float)
    return modal, fluxes, design_mask
design = Design(width=W, height=H, material=Material(permittivity=EPS_CLAD))
for x, y in ((0.0, Y_IN), (X_INV1, Y_TOP), (X_INV1, Y_BOT)):
    design += Rectangle(
        position=(x, y - 0.5 * WG_W),
        width=(X_INV0 if x == 0.0 else W - X_INV1),
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
    learning_rate=LEARNING_RATE,
    filter_radius=BLUR_START,
    eps_min=EPS_CLAD,
    eps_max=EPS_CORE,
    beta_schedule=(BETA_START, BETA_END),
    filter_type="conic",
)
if INITIAL_ASYM_NOISE > 0:
    rng = np.random.default_rng(7)
    noise = np.zeros_like(opt.design_density)
    noise[mask] = INITIAL_ASYM_NOISE * (rng.random(np.count_nonzero(mask)) - 0.5)
    opt.design_density = np.clip(opt.design_density + noise, 0.0, 1.0)
base_eps = grid.permittivity.copy()
waves = {wl: build_waveform(wl) for wl, _ in CASES}
for wl, _ in CASES:
    w = waves[wl]
    print(
        f"wl={wl / UM:.3f} um | total={w['total'] * 1e15:.1f} fs | "
        f"flight={w['flight'] * 1e15:.1f} fs | gate={w['gate_start'] * 1e15:.1f} fs"
    )

hist = {
    "objective": [],
    "ema": [],
    "route": [],
    "power_sum": [],
    "tx": {WL_SHORT: {"target": [], "leak": []}, WL_LONG: {"target": [], "leak": []}},
    "fom": {WL_SHORT: [], WL_LONG: []},
}
ema = None
best_projected_score = -np.inf
best_projected_density = None
best_binary_score = -np.inf
best_binary_density = None
best_binary_step = 0

for step in range(1, STEPS + 1):
    beta, blur_radius, binarity_weight, is_polish = step_settings(step)
    if step == POLISH_START:
        import optax

        opt.optax_optimizer = optax.adam(learning_rate=POLISH_LR)
        opt._opt_state = None
    opt.filter_radius = blur_radius
    opt.filter_radius_cells = max(1, int(round(blur_radius / opt.resolution)))
    density = opt.get_physical_density(beta)
    set_design(grid, base_eps, mask, density, opt)
    route_terms, grad_terms, caches = [], [], []
    target_mean = 0.0
    power_sum_mean = 0.0

    for wl, target_port in CASES:
        modal, fields = run_forward(grid, wl, waves[wl], fields=("Ez",))
        ez_hist = [np.array(frame) for frame in fields["Ez"]]
        norm = max(modal["p_in"], 1e-30)
        target_tx = (modal["p_top"] if target_port == "top" else modal["p_bot"]) / norm
        leak_tx = (modal["p_bot"] if target_port == "top" else modal["p_top"]) / norm
        target_mean += target_tx
        power_sum_mean += modal["power_sum"]
        hist["tx"][wl]["target"].append(100.0 * target_tx)
        hist["tx"][wl]["leak"].append(100.0 * leak_tx)
        caches.append(
            {
                "wl": wl,
                "target": target_port,
                "wave": waves[wl],
                "ez_hist": ez_hist,
                "norm": norm,
                "target_tx": target_tx,
                "leak_tx": leak_tx,
                "loss_est": modal["loss_est"],
                "power_sum": modal["power_sum"],
            }
        )

    target_mean /= len(CASES)
    power_sum_mean /= len(CASES)
    hist["power_sum"].append(power_sum_mean)

    report = []
    for cache in caches:
        wl = cache["wl"]
        wave = cache["wave"]
        grad_top = np.array(
            compute_overlap_gradient(cache["ez_hist"], run_adjoint(grid, wl, "top", wave), forward_start=wave["gate_index"])
        )
        grad_bot = np.array(
            compute_overlap_gradient(cache["ez_hist"], run_adjoint(grid, wl, "bottom", wave), forward_start=wave["gate_index"])
        )
        if cache["target"] == "top":
            coeff_top, coeff_bot = (1.0 + LOSS_WEIGHT) / cache["norm"], (LOSS_WEIGHT - 1.0) / cache["norm"]
        else:
            coeff_top, coeff_bot = (LOSS_WEIGHT - 1.0) / cache["norm"], (1.0 + LOSS_WEIGHT) / cache["norm"]
        overflow = max(cache["power_sum"] - 1.0, 0.0)
        fom = (
            cache["target_tx"]
            - cache["leak_tx"]
            - LOSS_WEIGHT * cache["loss_est"]
            - POWER_BOUND_WEIGHT * overflow * overflow
        )
        grad = coeff_top * grad_top + coeff_bot * grad_bot
        route_terms.append(fom)
        grad_terms.append(grad / max(float(np.sqrt(np.mean(grad[mask] ** 2))), 1e-30) if NORMALIZE_PER_WL_GRAD else grad)
        hist["fom"][wl].append(100.0 * fom)
        report.append(
            f"{wl / UM:.3f}um: T={100.0 * cache['target_tx']:5.1f}% "
            f"L={100.0 * cache['leak_tx']:5.1f}% Loss={100.0 * cache['loss_est']:5.1f}% "
            f"Psum={100.0 * cache['power_sum']:5.1f}% FoM={100.0 * fom:5.1f}%"
        )

    grad_total = sum(grad_terms) / len(grad_terms)
    rho = density[mask]
    binarity = float(np.mean((2.0 * rho - 1.0) ** 2)) if rho.size > 0 else 0.0
    if rho.size > 0:
        grad_total[mask] += binarity_weight * (
            4.0 * (2.0 * rho - 1.0) / (max(opt.eps_max - opt.eps_min, 1e-30) * float(rho.size))
        )
    if np.any(mask):
        clip = np.percentile(np.abs(grad_total[mask]), GRAD_CLIP_PCT)
        if clip > 0:
            grad_total[mask] = np.clip(grad_total[mask], -clip, clip)
        grad_total[mask] = np.clip(grad_total[mask], -GRAD_HARD_CAP, GRAD_HARD_CAP)

    route_obj = float(np.mean(route_terms))
    objective = route_obj + binarity_weight * binarity
    max_power_sum = max(cache["power_sum"] for cache in caches)
    if max_power_sum <= POWER_SUM_TOL and route_obj > best_projected_score:
        best_projected_score = route_obj
        best_projected_density = density.copy()
    dmax = opt.apply_gradient(-grad_total, beta)
    hist["route"].append(route_obj)
    hist["objective"].append(objective)
    ema = objective if ema is None else EMA_ALPHA * objective + (1.0 - EMA_ALPHA) * ema
    hist["ema"].append(ema)

    if step == 1 or step % PRINT_EVERY == 0 or step == STEPS:
        print(
            f"[{step:03d}/{STEPS}] Obj={objective:.4f} meanT={100.0 * target_mean:.1f}% "
            f"Psum={100.0 * power_sum_mean:.1f}% bin={binarity:.2f} "
            f"{'polish' if is_polish else 'main'} beta={beta:.1f} dmax={dmax:.3e} | "
            + " | ".join(report)
        )

    should_debug = step == 1 or step % DEBUG_EVERY == 0 or step == STEPS or is_polish
    if SAVE_DEBUG and should_debug:
        binary_density = (density >= 0.5).astype(float)
        binary_score, binary_modal = score_density(grid, binary_density)
        if max(m["power_sum"] for m in binary_modal.values()) <= POWER_SUM_TOL and binary_score > best_binary_score:
            best_binary_score = binary_score
            best_binary_density = binary_density.copy()
            best_binary_step = step
        set_design(grid, base_eps, mask, density, opt)
        plt.imsave(f"{PREFIX}_topo_{step:03d}.png", grid.permittivity.T, cmap="gray", origin="lower")
        plt.imsave(
            f"{PREFIX}_density_{step:03d}.png",
            density.T,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
        )
        _, fluxes, _ = flux_artifacts(grid, density)
        save_flux(f"{PREFIX}_flux_short_{step:03d}.png", fluxes[WL_SHORT])
        save_flux(f"{PREFIX}_flux_long_{step:03d}.png", fluxes[WL_LONG])
        save_progress(f"{PREFIX}_progress_{step:03d}.png", hist)
        save_progress(f"{PREFIX}_progress_latest.png", hist)
        if (
            ENABLE_EARLY_STOP
            and is_polish
            and best_binary_step > 0
            and step - best_binary_step >= EARLY_STOP_PATIENCE
            and binary_score < best_binary_score - 0.03
        ):
            print(f"Stopping polish at step {step:03d}; best binary checkpoint was step {best_binary_step:03d}.")
            break

save_progress(f"{PREFIX}_progress_final.png", hist)

if best_projected_density is None:
    best_projected_density = density.copy()
if best_binary_density is None:
    best_binary_density = (best_projected_density >= 0.5).astype(float)

for label, density, path in (
    ("projected", best_projected_density, f"{PREFIX}_final_projected_flux_overlay.png"),
    ("binary", best_binary_density, f"{PREFIX}_final_binary_flux_overlay.png"),
):
    modal, fluxes, design_mask = flux_artifacts(grid, density)
    save_overlay(
        path,
        fluxes[WL_SHORT],
        fluxes[WL_LONG],
        design_mask,
    )
    print(
        f"{label}: "
        f"short target={100.0 * modal[WL_SHORT]['tx_top']:.2f}% leak={100.0 * modal[WL_SHORT]['tx_bot']:.2f}% | "
        f"long target={100.0 * modal[WL_LONG]['tx_bot']:.2f}% leak={100.0 * modal[WL_LONG]['tx_top']:.2f}%"
    )
