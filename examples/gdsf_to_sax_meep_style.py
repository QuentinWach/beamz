"""3D Meep-style broadband SAX extraction using compiled JAX + in-loop DFT.

Key choices:
- 3D Gaussian-pulse excitation (single broadband run per launched port).
- DFT monitor accumulation happens inside compiled scan (no raw field history).
- Arrival-gated DFT windows per monitor to reduce runtime and transient contamination.
- Optional two-run normalization for cleaner absolute S-parameters.
- Adaptive compiled early-stop based on monitor decay.
"""

import os
import time as pytime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sax

from beamz import *
from beamz.devices.sources.signals import gaussian_pulse
from beamz.visual.helpers import dxdt


def env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return float(default)
    return float(val)


def env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return int(default)
    return int(val)


FAST_MODE = env_bool("BEAMZ_FAST", True)
USE_TWO_RUN_NORM = env_bool("BEAMZ_TWO_RUN_NORM", False)
SHOW_LAYOUT = env_bool("BEAMZ_SHOW_LAYOUT", not FAST_MODE)
SHOW_SIGNAL = env_bool("BEAMZ_SHOW_SIGNAL", not FAST_MODE)
SHOW_FINAL = env_bool("BEAMZ_SHOW_FINAL", not FAST_MODE)
DEBUG_DFT = env_bool("BEAMZ_DEBUG_DFT", False)
CALIBRATE_PORT_SCALE = env_bool("BEAMZ_CALIBRATE_PORT_SCALE", False)
AUTO_SELECT_OUTPUT_WAVE = env_bool("BEAMZ_AUTO_SELECT_OUTPUT_WAVE", True)
POLARIZATION = str(os.getenv("BEAMZ_POLARIZATION", "tm")).strip().lower()
if POLARIZATION not in {"te", "tm"}:
    raise ValueError(
        f"Unsupported BEAMZ_POLARIZATION={POLARIZATION!r}; use 'te' or 'tm'."
    )

PLOT_SOURCE_MODE = env_bool("BEAMZ_PLOT_SOURCE_MODE", True)
MODE_PLOT_PATH = os.getenv("BEAMZ_MODE_PLOT_PATH", "mode_source_fields_3d.png")
WRITE_DEBUG_PLOTS = env_bool("BEAMZ_WRITE_DEBUG_PLOTS", True)
DEBUG_OUT_DIR = Path(
    os.getenv("BEAMZ_DEBUG_OUT_DIR", "benchmarks/results/gdsf_to_sax_debug")
)
PREFLIGHT_DEBUG = env_bool("BEAMZ_PREFLIGHT_DEBUG", True)
PREFLIGHT_STEPS = env_int("BEAMZ_PREFLIGHT_STEPS", 800)
PREFLIGHT_RECORD_INTERVAL = env_int("BEAMZ_PREFLIGHT_RECORD_INTERVAL", 4)
if WRITE_DEBUG_PLOTS:
    DEBUG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode_path = Path(MODE_PLOT_PATH)
    if not mode_path.is_absolute():
        MODE_PLOT_PATH = str(DEBUG_OUT_DIR / mode_path)

WL0 = 1.55 * µm
WL_MIN, WL_MAX = 1.50 * µm, 1.60 * µm
WL_POINTS = env_int("BEAMZ_SWEEP_POINTS", 21)
N_CORE, N_CLAD = 3.48, 1.44
POINTS_PER_WAVELENGTH = env_int("BEAMZ_PPW", 10)
DX, DT = dxdt(
    WL0,
    n_max=N_CORE,
    points_per_wavelength=POINTS_PER_WAVELENGTH,
    dims=3,
)

INPUT_EXTENSION, OUTPUT_EXTENSION, Y_MARGIN = 4.0 * µm, 4.0 * µm, 3.0 * µm
CORE_THICKNESS = 0.22 * µm
PML_XY = env_float("BEAMZ_PML_XY", 1.0 * WL0)
PML_RIGHT = env_float("BEAMZ_PML_RIGHT", 1.5 * WL0)
PML_Z = env_float("BEAMZ_PML_Z", 0.6 * WL0)
Z_BUFFER_FROM_PML = env_float("BEAMZ_Z_BUFFER_FROM_PML", 0.5 * µm)
CLAD_BELOW = max(env_float("BEAMZ_CLAD_BELOW", 1.5 * µm), PML_Z + Z_BUFFER_FROM_PML)
CLAD_ABOVE = max(env_float("BEAMZ_CLAD_ABOVE", 1.5 * µm), PML_Z + Z_BUFFER_FROM_PML)
DEVICE_DEPTH = CLAD_BELOW + CORE_THICKNESS + CLAD_ABOVE
CORE_Z0 = CLAD_BELOW
CORE_ZC = CORE_Z0 + 0.5 * CORE_THICKNESS

SOURCE_SPAN_FACTOR = env_float("BEAMZ_SOURCE_SPAN_FACTOR", 1.8)
SOURCE_MIN_SPAN = env_float("BEAMZ_SOURCE_MIN_SPAN", 0.8 * µm)
SOURCE_HEIGHT_FACTOR = env_float("BEAMZ_SOURCE_HEIGHT_FACTOR", 1.0)
SOURCE_MIN_HEIGHT = env_float("BEAMZ_SOURCE_MIN_HEIGHT", 0.8 * µm)
MONITOR_SPAN_FACTOR = env_float("BEAMZ_MONITOR_SPAN_FACTOR", 1.8)
MONITOR_MIN_SPAN = env_float("BEAMZ_MONITOR_MIN_SPAN", 0.8 * µm)
MONITOR_Z_SPAN = max(
    env_float("BEAMZ_MONITOR_Z_SPAN", 0.7 * µm), 3.0 * CORE_THICKNESS
)
MONITOR_Z_MARGIN = env_float("BEAMZ_MONITOR_Z_MARGIN", 0.25 * µm)
MONITOR_Z_MIN = PML_Z + MONITOR_Z_MARGIN
MONITOR_Z_MAX = DEVICE_DEPTH - PML_Z - MONITOR_Z_MARGIN

SOURCE_OFFSET = -1.2 * µm
FORWARD_MONITOR_OFFSET = env_float("BEAMZ_FORWARD_MONITOR_OFFSET", 2.0 * µm)
REFLECTION_MONITOR_BACKOFF = 2.0 * µm
OUTPUT_MONITOR_OFFSET = env_float("BEAMZ_OUTPUT_MONITOR_OFFSET", 1.2 * µm)
OUTPUT_MONITOR_SEARCH_RADIUS = env_float(
    "BEAMZ_OUTPUT_MONITOR_SEARCH_RADIUS", 2.2 * µm
)
OUTPUT_MONITOR_SEARCH_STEPS = env_int("BEAMZ_OUTPUT_MONITOR_SEARCH_STEPS", 17)

SOURCE_PORT, OUTPUT_PORTS = "o1", ["o2", "o3"]
DIR_VEC = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}


def move_along(center_xy, direction, distance):
    vx, vy, _ = DIR_VEC[direction]
    return center_xy[0] + vx * float(distance), center_xy[1] + vy * float(distance)


def outward_direction(inward_direction):
    return ("-" if inward_direction.startswith("+") else "+") + inward_direction[1]


def port_span(port, span_factor, span_min):
    return max(float(span_min), span_factor * float(port["width"]))


def port_plane(port, span_factor, span_min, z_span, offset=0.0):
    cx, cy = move_along(port["center"], port["direction"], offset)
    span = port_span(port, span_factor, span_min)
    z0 = max(float(MONITOR_Z_MIN), CORE_ZC - 0.5 * float(z_span))
    z1 = min(float(MONITOR_Z_MAX), CORE_ZC + 0.5 * float(z_span))
    if z1 <= z0:
        z_mid = 0.5 * (float(MONITOR_Z_MIN) + float(MONITOR_Z_MAX))
        dz = max(float(MONITOR_Z_MAX) - float(MONITOR_Z_MIN), float(DX))
        z0 = z_mid - 0.5 * dz
        z1 = z_mid + 0.5 * dz
    axis = port["direction"][1]

    if axis == "x":
        return (cx, cy - 0.5 * span, z0), (cx, cy + 0.5 * span, z1), span, (z1 - z0)
    return (cx - 0.5 * span, cy, z0), (cx + 0.5 * span, cy, z1), span, (z1 - z0)


def dft_components_for_port(direction, polarization="tm"):
    axis = direction[1]
    pol = str(polarization).lower()
    if pol == "tm":
        return {"x": ("Ez", "Hy"), "y": ("Ez", "Hx"), "z": ("Ey", "Hx")}[axis]
    return {"x": ("Ey", "Hz"), "y": ("Ex", "Hz"), "z": ("Ex", "Hy")}[axis]


def monitor_center(start, end):
    return (
        0.5 * (float(start[0]) + float(end[0])),
        0.5 * (float(start[1]) + float(end[1])),
        0.5 * (float(start[2]) + float(end[2])),
    )


def monitor_plane_status(name, start, end, design_obj):
    x0, x1 = sorted([float(start[0]), float(end[0])])
    y0, y1 = sorted([float(start[1]), float(end[1])])
    z0, z1 = sorted([float(start[2]), float(end[2])])
    ix0 = float(PML_XY)
    ix1 = float(design_obj.width) - float(PML_RIGHT)
    iy0 = float(PML_XY)
    iy1 = float(design_obj.height) - float(PML_XY)
    iz0 = float(PML_Z)
    iz1 = float(design_obj.depth) - float(PML_Z)
    inside = (x0 >= ix0 and x1 <= ix1 and y0 >= iy0 and y1 <= iy1 and z0 >= iz0 and z1 <= iz1)
    return (
        f"{name}: x=[{x0/µm:.2f},{x1/µm:.2f}]um y=[{y0/µm:.2f},{y1/µm:.2f}]um "
        f"z=[{z0/µm:.2f},{z1/µm:.2f}]um inside_non_pml={inside}"
    )


def save_mode_diagnostics(mode_source, out_path):
    eps = np.asarray(getattr(mode_source, "_eps_profile_2d", np.zeros((1, 1))), dtype=float)
    fields = {
        "Ex": np.asarray(getattr(mode_source, "_Ex_profile", 0.0), dtype=float),
        "Ey": np.asarray(getattr(mode_source, "_Ey_profile", 0.0), dtype=float),
        "Ez": np.asarray(getattr(mode_source, "_Ez_profile", 0.0), dtype=float),
        "Hx": np.asarray(getattr(mode_source, "_Hx_profile", 0.0), dtype=float),
        "Hy": np.asarray(getattr(mode_source, "_Hy_profile", 0.0), dtype=float),
        "Hz": np.asarray(getattr(mode_source, "_Hz_profile", 0.0), dtype=float),
    }
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=220)
    ax = axes.ravel()
    eps_im = ax[0].imshow(eps, origin="lower", cmap="viridis", aspect="auto")
    ax[0].set_title("eps(r)")
    fig.colorbar(eps_im, ax=ax[0], fraction=0.046, pad=0.04)

    for i, name in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], start=1):
        arr = np.asarray(fields[name]).squeeze()
        if arr.ndim != 2:
            arr = np.atleast_2d(arr)
        im = ax[i].imshow(np.abs(arr), origin="lower", cmap="magma", aspect="auto")
        ax[i].set_title(name)
        fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)

    ax[7].axis("off")
    neff = float(np.real(getattr(mode_source, "_neff", np.nan)))
    neff_imp = float(np.real(getattr(mode_source, "_impedance_neff", np.nan)))
    ax[7].text(
        0.02,
        0.95,
        (
            f"pol={mode_source.pol}\n"
            f"dir={mode_source.direction}\n"
            f"neff={neff:.6f}\n"
            f"neff_imp={neff_imp:.6f}\n"
            f"width={mode_source.width/µm:.3f} um\n"
            f"height={float(mode_source.height)/µm:.3f} um"
        ),
        va="top",
        ha="left",
        fontsize=10,
    )

    for a in ax[:7]:
        a.set_xlabel("u")
        a.set_ylabel("v")
    fig.suptitle("Mode Source Diagnostics", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def draw_pml_lines(ax, x_max_um, y_max_um, pml_x_um, pml_y_um):
    ax.axvline(pml_x_um, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(x_max_um - pml_x_um, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(pml_y_um, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(y_max_um - pml_y_um, color="white", linestyle="--", linewidth=1.0, alpha=0.8)


def save_design_debug_plots(
    design_obj,
    permittivity,
    source_ctr,
    src_w,
    src_h,
    port_planes,
    out_path,
):
    eps = np.asarray(permittivity, dtype=float)
    core_mask = eps > (0.5 * (N_CORE**2 + N_CLAD**2))

    z_mid = int(np.clip(round(source_ctr[2] / DX), 0, eps.shape[0] - 1))
    y_mid = int(np.clip(round(source_ctr[1] / DX), 0, eps.shape[1] - 1))
    x_src = int(np.clip(round(source_ctr[0] / DX), 0, eps.shape[2] - 1))

    xy = core_mask[z_mid]
    xz = core_mask[:, y_mid, :]
    yz = core_mask[:, :, x_src]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=200)
    axes[0].imshow(
        xy,
        origin="lower",
        cmap="gray_r",
        extent=[0, design_obj.width / µm, 0, design_obj.height / µm],
        aspect="equal",
    )
    axes[0].set_title(f"XY Core Mask (z={source_ctr[2]/µm:.2f} um)")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")

    axes[1].imshow(
        xz,
        origin="lower",
        cmap="gray_r",
        extent=[0, design_obj.width / µm, 0, design_obj.depth / µm],
        aspect="equal",
    )
    axes[1].set_title(f"XZ Core Mask (y={source_ctr[1]/µm:.2f} um)")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("z (um)")

    axes[2].imshow(
        yz,
        origin="lower",
        cmap="gray_r",
        extent=[0, design_obj.height / µm, 0, design_obj.depth / µm],
        aspect="equal",
    )
    axes[2].set_title(f"YZ Core Mask (x={source_ctr[0]/µm:.2f} um)")
    axes[2].set_xlabel("y (um)")
    axes[2].set_ylabel("z (um)")

    # Source window overlays.
    axes[0].plot(
        [source_ctr[0] / µm, source_ctr[0] / µm],
        [(source_ctr[1] - 0.5 * src_w) / µm, (source_ctr[1] + 0.5 * src_w) / µm],
        color="red",
        linewidth=2.0,
    )
    axes[1].plot(
        [source_ctr[0] / µm, source_ctr[0] / µm],
        [(source_ctr[2] - 0.5 * src_h) / µm, (source_ctr[2] + 0.5 * src_h) / µm],
        color="red",
        linewidth=2.0,
    )
    axes[2].plot(
        [(source_ctr[1] - 0.5 * src_w) / µm, (source_ctr[1] + 0.5 * src_w) / µm],
        [(source_ctr[2] - 0.5 * src_h) / µm, (source_ctr[2] + 0.5 * src_h) / µm],
        color="red",
        linewidth=2.0,
    )

    # Monitor window center lines.
    for name, (s, e) in port_planes.items():
        x0, y0, z0 = s
        x1, y1, z1 = e
        axes[0].plot([x0 / µm, x1 / µm], [0.5 * (y0 + y1) / µm, 0.5 * (y0 + y1) / µm], linewidth=1.2, label=name)
        axes[1].plot([x0 / µm, x1 / µm], [0.5 * (z0 + z1) / µm, 0.5 * (z0 + z1) / µm], linewidth=1.2, label=name)
        axes[2].plot([0.5 * (y0 + y1) / µm, 0.5 * (y0 + y1) / µm], [z0 / µm, z1 / µm], linewidth=1.2, label=name)

    draw_pml_lines(
        axes[0],
        design_obj.width / µm,
        design_obj.height / µm,
        PML_XY / µm,
        PML_XY / µm,
    )
    draw_pml_lines(
        axes[1],
        design_obj.width / µm,
        design_obj.depth / µm,
        PML_XY / µm,
        PML_Z / µm,
    )
    draw_pml_lines(
        axes[2],
        design_obj.height / µm,
        design_obj.depth / µm,
        PML_XY / µm,
        PML_Z / µm,
    )
    for ax in axes:
        ax.grid(alpha=0.2, linestyle="--")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def monitor_core_overlap_stats(sim, monitor_obj):
    eps = np.asarray(sim.fields.permittivity)
    return monitor_core_overlap_from_arrays(eps, sim.resolution, monitor_obj)


def monitor_core_overlap_from_arrays(eps, resolution, monitor_obj):
    z_idx, y_idx, x_idx = monitor_obj.get_grid_slice_3d(
        resolution, resolution, resolution, eps.shape
    )
    eps_slice = np.asarray(eps[z_idx, y_idx, x_idx], dtype=float)
    if eps_slice.ndim != 2 or eps_slice.size == 0:
        return 0.0, 0.0, 0.0
    core_mask = eps_slice > (0.5 * (N_CORE**2 + N_CLAD**2))
    clad_mask = np.isclose(eps_slice, N_CLAD**2, rtol=0.0, atol=1e-3)
    return (
        float(np.mean(core_mask)),
        float(np.mean(clad_mask)),
        float(np.max(eps_slice)),
    )


def select_plane_offset_for_core(
    port,
    eps,
    resolution,
    *,
    span_factor,
    span_min,
    z_span,
    offset_center,
    offset_radius,
    n_steps,
):
    offsets = np.linspace(
        float(offset_center) - float(offset_radius),
        float(offset_center) + float(offset_radius),
        int(max(3, n_steps)),
    )
    best_offset = float(offsets[0])
    best_core = -1.0
    best_start = None
    best_end = None
    for off in offsets:
        s, e, _, _ = port_plane(port, span_factor, span_min, z_span, offset=float(off))
        mon = Monitor(start=s, end=e, name="candidate")
        core_frac, _, eps_max = monitor_core_overlap_from_arrays(eps, resolution, mon)
        score = core_frac + (1e-6 * eps_max)
        if score > best_core:
            best_core = score
            best_offset = float(off)
            best_start, best_end = s, e
    return best_offset, best_start, best_end


def signed_sx_timeseries_from_monitor(monitor_obj, dx):
    ex = np.asarray(monitor_obj.fields.get("Ex", []), dtype=float)
    ey = np.asarray(monitor_obj.fields.get("Ey", []), dtype=float)
    ez = np.asarray(monitor_obj.fields.get("Ez", []), dtype=float)
    hy = np.asarray(monitor_obj.fields.get("Hy", []), dtype=float)
    hz = np.asarray(monitor_obj.fields.get("Hz", []), dtype=float)
    ts = np.asarray(monitor_obj.fields.get("t", []), dtype=float)
    if ex.ndim != 3 or ex.shape[0] == 0:
        return ts, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    nt = min(ex.shape[0], ey.shape[0], ez.shape[0], hy.shape[0], hz.shape[0], ts.size)
    sx = ey[:nt] * hz[:nt] - ez[:nt] * hy[:nt]
    signed = np.sum(sx, axis=(1, 2)) * float(dx) * float(dx)
    abs_flux = np.sum(np.abs(sx), axis=(1, 2)) * float(dx) * float(dx)
    return ts[:nt], signed, abs_flux


def estimate_arrival_time(src_xyz, dst_xyz, n_eff, t_launch):
    dist = float(np.linalg.norm(np.asarray(dst_xyz, dtype=float) - np.asarray(src_xyz, dtype=float)))
    return float(t_launch + n_eff * dist / LIGHT_SPEED)


def build_reference_design_like(device_design, src_port):
    """Build a straight-waveguide reference design for incident normalization."""
    ref = Design(
        width=device_design.width,
        height=device_design.height,
        depth=device_design.depth,
        material=Material(N_CLAD**2),
    )
    y0 = float(src_port["center"][1]) - 0.5 * float(src_port["width"])
    ref += Rectangle(
        position=(0.0, y0, CORE_Z0),
        width=float(device_design.width),
        height=float(src_port["width"]),
        depth=CORE_THICKNESS,
        material=Material(N_CORE**2),
    )
    return ref


def run_compiled_until_decay(
    sim,
    stop_monitors,
    *,
    chunk_steps,
    min_steps,
    max_steps,
    lookback_records,
    decay_ratio,
):
    """Run compiled simulation in chunks and stop once monitors decay sufficiently."""
    steps_done = 0
    peak = 0.0
    lookback_records = max(2, int(lookback_records))

    while steps_done < int(max_steps):
        n_chunk = int(min(chunk_steps, max_steps - steps_done))
        if n_chunk <= 0:
            break
        sim.run_compiled(num_steps=n_chunk, progress=False)
        steps_done += n_chunk

        latest_vals = []
        for mon in stop_monitors:
            if len(mon.power_history) == 0:
                continue
            p = np.abs(np.asarray(mon.power_history, dtype=np.float64))
            peak = max(peak, float(np.max(p)))
            latest_vals.extend(p[-lookback_records:].tolist())

        if steps_done < int(min_steps):
            continue
        if peak <= 0.0 or len(latest_vals) == 0:
            continue

        tail_max = float(np.max(np.asarray(latest_vals, dtype=np.float64)))
        if tail_max <= float(decay_ratio) * peak:
            break

    return steps_done


def safe_ratio(num, den, eps=1e-18):
    out = np.zeros_like(num, dtype=np.complex128)
    valid = np.abs(den) > eps
    out[valid] = num[valid] / den[valid]
    return out


def select_output_wave(port_name, waves_dict, auto_select=True):
    a_plus = np.asarray(waves_dict[port_name]["a_plus"], dtype=np.complex128)
    a_minus = np.asarray(waves_dict[port_name]["a_minus"], dtype=np.complex128)
    p_plus = float(np.mean(np.abs(a_plus) ** 2)) if a_plus.size else 0.0
    p_minus = float(np.mean(np.abs(a_minus) ** 2)) if a_minus.size else 0.0
    if auto_select:
        use_plus = p_plus > p_minus
    else:
        use_plus = False
    selected = "a_plus" if use_plus else "a_minus"
    coeff = a_plus if use_plus else a_minus
    return coeff, selected, p_plus, p_minus


def trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def build_mode_source_with_autofit(
    *,
    grid_obj,
    center,
    base_width,
    base_height,
    wavelength,
    pol,
    signal,
    direction,
    port_width,
):
    min_guided_neff = env_float("BEAMZ_MIN_GUIDED_NEFF", N_CLAD + 0.15)
    candidate_scales = [1.0, 0.85, 0.7, 1.2]
    candidates = []
    for s in candidate_scales:
        w = max(float(SOURCE_MIN_SPAN), float(base_width) * float(s), 1.2 * float(port_width))
        h = max(float(SOURCE_MIN_HEIGHT), float(base_height) * float(s), 1.2 * float(port_width))
        candidates.append((w, h))

    best = None
    best_neff = -np.inf
    for w, h in candidates:
        cand = ModeSource(
            grid=grid_obj,
            center=center,
            width=w,
            height=h,
            wavelength=float(wavelength),
            pol=pol,
            signal=signal,
            direction=direction,
        )
        cand.initialize(grid_obj.permittivity, DX, dt=DT)
        neff_val = float(np.real(cand._neff))
        if DEBUG_DFT:
            print(f"[mode candidate] width={w/µm:.3f}um, height={h/µm:.3f}um -> neff={neff_val:.6f}")
        if neff_val > best_neff:
            best = cand
            best_neff = neff_val
        if neff_val >= float(min_guided_neff):
            # First strongly guided candidate wins to avoid oversizing.
            return cand

    return best


def calibration_source_center(cal_design, port, launch_direction):
    y = float(port["center"][1])
    z = CORE_ZC
    x_pad_l = max(float(PML_XY) + 0.8 * float(WL0), 1.5 * float(µm))
    x_pad_r = max(float(PML_RIGHT) + 0.8 * float(WL0), 1.5 * float(µm))
    if launch_direction == "+x":
        x = x_pad_l
    elif launch_direction == "-x":
        x = float(cal_design.width) - x_pad_r
    else:
        x = float(port["center"][0])
    return (x, y, z)


def run_modal_gain_calibration(
    *,
    port_name,
    port,
    monitor_start,
    monitor_end,
    port_direction,
    polarization,
    coeff_key,
    cal_frequency,
    time,
    signal,
    chunk_steps,
    min_steps_env,
    max_steps,
    lookback,
    decay_ratio,
    record_interval,
):
    cal_design = build_reference_design_like(device_design, port)
    cal_grid = cal_design.rasterize(resolution=DX)
    launch_dir = (
        str(port_direction)
        if str(coeff_key) == "a_plus"
        else outward_direction(str(port_direction))
    )
    src_ctr = calibration_source_center(cal_design, port, launch_dir)
    src_w = max(float(SOURCE_MIN_SPAN), float(port["width"]) * float(SOURCE_SPAN_FACTOR))
    src_h = max(float(SOURCE_MIN_HEIGHT), float(SOURCE_HEIGHT_FACTOR) * src_w)
    cal_source = ModeSource(
        grid=cal_grid,
        center=src_ctr,
        width=src_w,
        height=src_h,
        wavelength=float(LIGHT_SPEED / float(cal_frequency)),
        pol=polarization,
        signal=signal,
        direction=launch_dir,
    )
    arrival = estimate_arrival_time(
        src_ctr,
        monitor_center(monitor_start, monitor_end),
        n_group_est,
        t0,
    )
    t_start = max(0.0, arrival - 4.0 * sigma_t)
    t_end = arrival + 10.0 * sigma_t + 10.0 / float(cal_frequency)
    cal_monitor = Monitor(
        start=monitor_start,
        end=monitor_end,
        name=f"cal_{port_name}",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.asarray([float(cal_frequency)], dtype=float),
        dft_t_start=t_start,
        dft_t_end=t_end,
        dft_components=dft_components_for_port(port_direction, polarization=polarization),
        dft_window=os.getenv("BEAMZ_DFT_WINDOW", "hann"),
        record_interval=max(1, int(record_interval)),
        dft_record_every_step=(int(record_interval) <= 1),
    )
    sim_cal = Simulation(
        design=cal_design,
        devices=[cal_source, cal_monitor],
        boundaries=[
            PML(edges=["left", "top", "bottom"], thickness=PML_XY),
            PML(edges="right", thickness=PML_RIGHT),
            PML(edges=["front", "back"], thickness=PML_Z),
        ],
        time=time,
        resolution=DX,
    )
    min_time = t_end + 2.0 / float(cal_frequency)
    min_steps = max(int(min_steps_env), int(np.ceil(min_time / DT)))
    run_compiled_until_decay(
        sim_cal,
        stop_monitors=[cal_monitor],
        chunk_steps=chunk_steps,
        min_steps=min_steps,
        max_steps=max_steps,
        lookback_records=lookback,
        decay_ratio=decay_ratio,
    )
    waves = sim_cal.extract_port_waves_dft(
        ports=[
            PortSpec(
                name=port_name,
                monitor_name=f"cal_{port_name}",
                direction=port_direction,
                polarization=polarization,
            )
        ],
        frequencies=np.asarray([float(cal_frequency)], dtype=float),
        min_incident_db=-80.0,
        return_power=True,
    )
    coeff = np.asarray(waves[port_name][coeff_key], dtype=np.complex128)
    amp = float(np.abs(coeff[0])) if coeff.size else 0.0
    return max(amp, 1e-18)


imported_design, ports = design.io.gdsf.load(
    "mmi1x2",
    n_core=N_CORE,
    n_clad=N_CLAD,
    layer=(1, 0),
    padding=0.0,
)

device_design = Design(
    width=imported_design.width + INPUT_EXTENSION + OUTPUT_EXTENSION,
    height=imported_design.height + 2.0 * Y_MARGIN,
    depth=DEVICE_DEPTH,
    material=Material(N_CLAD**2),
)

for structure in imported_design.structures[1:]:
    shifted = structure.copy().shift(INPUT_EXTENSION, Y_MARGIN, CORE_Z0)
    shifted.z = CORE_Z0
    shifted.depth = CORE_THICKNESS
    device_design += shifted

ports = {
    name: {
        **p,
        "center": (p["center"][0] + INPUT_EXTENSION, p["center"][1] + Y_MARGIN),
    }
    for name, p in ports.items()
}

for name, port in ports.items():
    cx, cy, width = *port["center"], float(port["width"])
    extension = INPUT_EXTENSION if name == SOURCE_PORT else OUTPUT_EXTENSION
    ox, oy = move_along((cx, cy), outward_direction(port["direction"]), extension)
    if port["direction"][1] == "x":
        device_design += Rectangle(
            position=(min(cx, ox), cy - width / 2, CORE_Z0),
            width=extension,
            height=width,
            depth=CORE_THICKNESS,
            material=Material(N_CORE**2),
        )
    else:
        device_design += Rectangle(
            position=(cx - width / 2, min(cy, oy), CORE_Z0),
            width=width,
            height=extension,
            depth=CORE_THICKNESS,
            material=Material(N_CORE**2),
        )

if SHOW_LAYOUT:
    device_design.show()

grid = device_design.rasterize(resolution=DX)
if SHOW_LAYOUT:
    grid.show(field="permittivity")

src = ports[SOURCE_PORT]
source_xy = move_along(src["center"], src["direction"], SOURCE_OFFSET)
source_center = (source_xy[0], source_xy[1], CORE_ZC)
source_width = port_span(src, SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN)
source_height = max(SOURCE_MIN_HEIGHT, SOURCE_HEIGHT_FACTOR * source_width)

o1_fwd_start, o1_fwd_end, _, _ = port_plane(
    src,
    MONITOR_SPAN_FACTOR,
    MONITOR_MIN_SPAN,
    MONITOR_Z_SPAN,
    offset=FORWARD_MONITOR_OFFSET,
)
o1_ref_start, o1_ref_end, _, _ = port_plane(
    src,
    MONITOR_SPAN_FACTOR,
    MONITOR_MIN_SPAN,
    MONITOR_Z_SPAN,
    offset=-REFLECTION_MONITOR_BACKOFF,
)

out_planes = {}
out_offsets = {}
eps_grid = np.asarray(grid.permittivity)
for out in OUTPUT_PORTS:
    best_offset, best_start, best_end = select_plane_offset_for_core(
        ports[out],
        eps_grid,
        DX,
        span_factor=MONITOR_SPAN_FACTOR,
        span_min=MONITOR_MIN_SPAN,
        z_span=MONITOR_Z_SPAN,
        offset_center=OUTPUT_MONITOR_OFFSET,
        offset_radius=OUTPUT_MONITOR_SEARCH_RADIUS,
        n_steps=OUTPUT_MONITOR_SEARCH_STEPS,
    )
    out_offsets[out] = best_offset
    out_planes[out] = (best_start, best_end)
    if DEBUG_DFT:
        cand_mon = Monitor(start=best_start, end=best_end, name=f"{out}_best")
        core_frac, clad_frac, eps_max = monitor_core_overlap_from_arrays(
            eps_grid, DX, cand_mon
        )
        print(
            f"[output monitor search] {out}: offset={best_offset/µm:.3f}um, "
            f"core_frac={core_frac:.3f}, clad_frac={clad_frac:.3f}, eps_max={eps_max:.3f}"
        )

port_planes_debug = {
    "o1_fwd": (o1_fwd_start, o1_fwd_end),
    "o1_ref": (o1_ref_start, o1_ref_end),
    "o2": out_planes["o2"],
    "o3": out_planes["o3"],
}
if WRITE_DEBUG_PLOTS:
    save_design_debug_plots(
        design_obj=device_design,
        permittivity=grid.permittivity,
        source_ctr=source_center,
        src_w=source_width,
        src_h=source_height,
        port_planes=port_planes_debug,
        out_path=DEBUG_OUT_DIR / "design_preflight_views.png",
    )

if DEBUG_DFT:
    print(
        "Non-PML interior extents: "
        f"x=[{PML_XY/µm:.2f},{(device_design.width - PML_RIGHT)/µm:.2f}]um, "
        f"y=[{PML_XY/µm:.2f},{(device_design.height - PML_XY)/µm:.2f}]um, "
        f"z=[{PML_Z/µm:.2f},{(device_design.depth - PML_Z)/µm:.2f}]um"
    )
    print(monitor_plane_status("o1_fwd", o1_fwd_start, o1_fwd_end, device_design))
    print(monitor_plane_status("o1_ref", o1_ref_start, o1_ref_end, device_design))
    for out in OUTPUT_PORTS:
        s_out, e_out = out_planes[out]
        print(monitor_plane_status(out, s_out, e_out, device_design))

freqs = np.linspace(
    LIGHT_SPEED / WL_MAX, LIGHT_SPEED / WL_MIN, WL_POINTS, dtype=np.float32
)
wl = LIGHT_SPEED / freqs
wl_um = wl / µm
fmin, fmax = float(np.min(freqs)), float(np.max(freqs))
fcen, fwidth = 0.5 * (fmin + fmax), max(fmax - fmin, 1e9)

sigma_t = 0.20 / fwidth
t0 = 4.0 * sigma_t

# Arrival-gated DFT windows.
dft_pre_sigma = env_float("BEAMZ_DFT_PRE_SIGMA", 4.0)
dft_post_sigma = env_float("BEAMZ_DFT_POST_SIGMA", 10.0)
dft_extra_cycles = env_float("BEAMZ_DFT_EXTRA_CYCLES", 18.0)
extra_time = dft_extra_cycles / fmin

forward_center = monitor_center(o1_fwd_start, o1_fwd_end)
reflect_center = monitor_center(o1_ref_start, o1_ref_end)
n_group_est = env_float("BEAMZ_NGROUP_EST", 2.2)
arr_fwd = estimate_arrival_time(source_center, forward_center, n_group_est, t0)

x_scatter = float(np.median([ports[p]["center"][0] for p in OUTPUT_PORTS]))
refl_path = max(0.0, 2.0 * (x_scatter - source_center[0]))
arr_ref = float(t0 + n_group_est * refl_path / LIGHT_SPEED)

arr_out = {
    out: estimate_arrival_time(source_center, monitor_center(*out_planes[out]), n_group_est, t0)
    for out in OUTPUT_PORTS
}
arrival_out_max = float(max(arr_out.values()))

window_pre = dft_pre_sigma * sigma_t
window_post = dft_post_sigma * sigma_t + extra_time

dft_window = {
    "o1_fwd": (max(0.0, arr_fwd - window_pre), arr_fwd + window_post),
    "o1_ref": (max(0.0, arr_ref - window_pre), arr_ref + window_post),
}
for out in OUTPUT_PORTS:
    a = arr_out[out]
    dft_window[out] = (max(0.0, a - window_pre), a + window_post)

# Upper bound simulation horizon; adaptive runner stops earlier.
max_arr = max([arr_fwd, arr_ref, *arr_out.values()])
decay_guard_cycles = env_float("BEAMZ_DECAY_GUARD_CYCLES", 8.0)
total_time_max = max_arr + window_post + decay_guard_cycles / fmin
time = np.arange(0.0, total_time_max, DT, dtype=np.float32)
preflight_reach_time = arrival_out_max + 3.0 * sigma_t
preflight_reach_steps = int(np.ceil(preflight_reach_time / DT))

signal = gaussian_pulse(
    time,
    amplitude=1.0,
    center=t0,
    width=sigma_t,
    frequency=fcen,
    phase=0.0,
)
signal = np.asarray(signal, dtype=np.float32)
if SHOW_SIGNAL:
    plot_signal(signal, time)

source = build_mode_source_with_autofit(
    grid_obj=grid,
    center=source_center,
    base_width=source_width,
    base_height=source_height,
    wavelength=float(WL0),
    pol=POLARIZATION,
    signal=signal,
    direction=src["direction"],
    port_width=float(src["width"]),
)
print(
    "Mode source diagnostics: "
    f"pol={POLARIZATION}, dir={src['direction']}, "
    f"neff={float(np.real(source._neff)):.6f}, "
    f"neff_imp={float(np.real(source._impedance_neff)):.6f}, "
    f"width={float(source.width)/µm:.3f}um, height={float(source.height)/µm:.3f}um"
)
if PLOT_SOURCE_MODE:
    save_mode_diagnostics(source, MODE_PLOT_PATH)
    print(f"Saved mode diagnostics plot: {MODE_PLOT_PATH}")

monitor_stride = env_int("BEAMZ_MONITOR_STRIDE", 3)
source_dft_components = dft_components_for_port(
    src["direction"], polarization=POLARIZATION
)

monitor_cfg = dict(
    record_fields=False,
    dft_enabled=True,
    dft_frequencies=freqs,
    dft_window=os.getenv("BEAMZ_DFT_WINDOW", "hann"),
    record_interval=max(monitor_stride, 1),
    dft_record_every_step=(monitor_stride <= 1),
)

# Device monitors.
device_monitors = [
    Monitor(
        start=o1_ref_start,
        end=o1_ref_end,
        name="o1_ref",
        dft_t_start=dft_window["o1_ref"][0],
        dft_t_end=dft_window["o1_ref"][1],
        dft_components=source_dft_components,
        **monitor_cfg,
    )
]
if not USE_TWO_RUN_NORM:
    device_monitors.append(
        Monitor(
            start=o1_fwd_start,
            end=o1_fwd_end,
            name="o1_fwd",
            dft_t_start=dft_window["o1_fwd"][0],
            dft_t_end=dft_window["o1_fwd"][1],
            dft_components=source_dft_components,
            **monitor_cfg,
        )
    )
for out in OUTPUT_PORTS:
    m_start, m_end = out_planes[out]
    comps = dft_components_for_port(
        ports[out]["direction"], polarization=POLARIZATION
    )
    t0_m, t1_m = dft_window[out]
    device_monitors.append(
        Monitor(
            start=m_start,
            end=m_end,
            name=out,
            dft_t_start=t0_m,
            dft_t_end=t1_m,
            dft_components=comps,
            **monitor_cfg,
        )
    )

power_diag_monitors = [
    Monitor(
        start=o1_fwd_start,
        end=o1_fwd_end,
        name="diag_o1_fwd_power",
        record_fields=False,
        accumulate_power=True,
        record_interval=max(1, monitor_stride),
    ),
    Monitor(
        start=out_planes["o2"][0],
        end=out_planes["o2"][1],
        name="diag_o2_power",
        record_fields=False,
        accumulate_power=True,
        record_interval=max(1, monitor_stride),
    ),
    Monitor(
        start=out_planes["o3"][0],
        end=out_planes["o3"][1],
        name="diag_o3_power",
        record_fields=False,
        accumulate_power=True,
        record_interval=max(1, monitor_stride),
    ),
]

sim_device = Simulation(
    design=device_design,
    devices=[source, *device_monitors, *power_diag_monitors],
    boundaries=[
        PML(edges=["left", "top", "bottom"], thickness=PML_XY),
        PML(edges="right", thickness=PML_RIGHT),
        PML(edges=["front", "back"], thickness=PML_Z),
    ],
    time=time,
    resolution=DX,
)

for mon_name, mon_start, mon_end in [
    ("o1_fwd", o1_fwd_start, o1_fwd_end),
    ("o1_ref", o1_ref_start, o1_ref_end),
    ("o2", out_planes["o2"][0], out_planes["o2"][1]),
    ("o3", out_planes["o3"][0], out_planes["o3"][1]),
]:
    probe_mon = Monitor(start=mon_start, end=mon_end, name=f"{mon_name}_overlap")
    core_frac, clad_frac, eps_max = monitor_core_overlap_stats(sim_device, probe_mon)
    print(
        f"[monitor overlap] {mon_name}: core_frac={core_frac:.3f}, "
        f"clad_frac={clad_frac:.3f}, eps_max={eps_max:.3f}"
    )

if PREFLIGHT_DEBUG:
    print("Running preflight diagnostics...")
    pre_steps = int(min(max(8, PREFLIGHT_STEPS), len(time)))
    print(
        "[preflight setup] "
        f"requested_steps={pre_steps}, "
        f"estimated_steps_to_outputs={preflight_reach_steps}"
    )
    if pre_steps < preflight_reach_steps:
        print(
            "[preflight note] this run is too short to reach output monitors; "
            "increase BEAMZ_PREFLIGHT_STEPS for end-to-end flux diagnostics."
        )
    pre_src = ModeSource(
        grid=grid,
        center=source_center,
        width=float(source.width),
        height=float(source.height),
        wavelength=float(WL0),
        pol=POLARIZATION,
        signal=signal,
        direction=src["direction"],
    )
    flux_monitors = [
        Monitor(
            start=o1_fwd_start,
            end=o1_fwd_end,
            name="pre_o1_fwd",
            record_fields=False,
            accumulate_power=True,
            record_interval=max(1, PREFLIGHT_RECORD_INTERVAL),
        ),
        Monitor(
            start=out_planes["o2"][0],
            end=out_planes["o2"][1],
            name="pre_o2",
            record_fields=False,
            accumulate_power=True,
            record_interval=max(1, PREFLIGHT_RECORD_INTERVAL),
        ),
        Monitor(
            start=out_planes["o3"][0],
            end=out_planes["o3"][1],
            name="pre_o3",
            record_fields=False,
            accumulate_power=True,
            record_interval=max(1, PREFLIGHT_RECORD_INTERVAL),
        ),
    ]
    sim_pre = Simulation(
        design=device_design,
        devices=[pre_src, *flux_monitors],
        boundaries=[
            PML(edges=["left", "top", "bottom"], thickness=PML_XY),
            PML(edges="right", thickness=PML_RIGHT),
            PML(edges=["front", "back"], thickness=PML_Z),
        ],
        time=time[:pre_steps],
        resolution=DX,
    )
    t_pre0 = pytime.time()
    sim_pre.run_compiled(num_steps=pre_steps, progress=False)
    pre_wall = pytime.time() - t_pre0
    print(f"[preflight] wall={pre_wall:.2f}s, steps={pre_steps}")

    pre_data = {}
    for mon in flux_monitors:
        tt = np.asarray(mon.power_timestamps, dtype=float)
        pw = np.asarray(mon.power_history, dtype=float)
        pre_data[mon.name] = (tt, pw)
        if tt.size and pw.size:
            p_integ = float(trapz(pw, tt))
            print(
                f"[preflight flux] {mon.name}: "
                f"max_power={np.max(pw):.3e}, "
                f"mean_power={np.mean(pw):.3e}, "
                f"int_power={p_integ:.3e}"
            )

    if (
        "pre_o1_fwd" in pre_data
        and pre_data["pre_o1_fwd"][0].size
        and pre_data["pre_o1_fwd"][1].size
    ):
        p_in = float(trapz(pre_data["pre_o1_fwd"][1], pre_data["pre_o1_fwd"][0]))
        p_o2 = (
            float(trapz(pre_data["pre_o2"][1], pre_data["pre_o2"][0]))
            if "pre_o2" in pre_data and pre_data["pre_o2"][0].size
            else 0.0
        )
        p_o3 = (
            float(trapz(pre_data["pre_o3"][1], pre_data["pre_o3"][0]))
            if "pre_o3" in pre_data and pre_data["pre_o3"][0].size
            else 0.0
        )
        if p_in > 0.0:
            print(
                f"[preflight transmission proxy] (o2+o3)/in = {(p_o2 + p_o3) / p_in:.3f}"
            )

    if WRITE_DEBUG_PLOTS:
        fig, ax = plt.subplots(2, 1, figsize=(8.0, 5.8), dpi=220, sharex=True)
        color_map = {"pre_o1_fwd": "tab:blue", "pre_o2": "tab:orange", "pre_o3": "tab:green"}
        for name in ["pre_o1_fwd", "pre_o2", "pre_o3"]:
            tt, pw = pre_data.get(name, (np.zeros((0,)), np.zeros((0,))))
            if tt.size == 0 or pw.size == 0:
                continue
            t_fs = tt / 1e-15
            ax[0].plot(t_fs, pw, lw=1.7, label=f"{name} power", color=color_map[name])
            cum = np.cumsum(np.maximum(pw, 0.0))
            ax[1].plot(t_fs, cum, lw=1.7, label=f"{name} cumulative", color=color_map[name])
        ax[0].set_ylabel("Monitor Power (a.u.)")
        ax[1].set_ylabel("Cumulative Power")
        ax[1].set_xlabel("Time (fs)")
        ax[0].grid(alpha=0.3)
        ax[1].grid(alpha=0.3)
        if ax[0].lines:
            ax[0].legend(loc="best", fontsize=8)
        if ax[1].lines:
            ax[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(DEBUG_OUT_DIR / "preflight_flux_traces.png", dpi=300)
        plt.close(fig)

        # Lightweight snapshot run (few full-field samples) to verify propagation.
        snap_steps = int(min(pre_steps, max(48, pre_steps // 2)))
        snap_src = ModeSource(
            grid=grid,
            center=source_center,
            width=float(source.width),
            height=float(source.height),
            wavelength=float(WL0),
            pol=POLARIZATION,
            signal=signal,
            direction=src["direction"],
        )
        sim_snap = Simulation(
            design=device_design,
            devices=[snap_src],
            boundaries=[
                PML(edges=["left", "top", "bottom"], thickness=PML_XY),
                PML(edges="right", thickness=PML_RIGHT),
                PML(edges=["front", "back"], thickness=PML_Z),
            ],
            time=time[:snap_steps],
            resolution=DX,
        )
        rec_every = max(1, snap_steps // 8)
        snap_result = sim_snap.run_compiled(
            num_steps=snap_steps,
            record_interval=rec_every,
            record_fields=["Ez"],
            progress=False,
        )
        ez_snaps = np.asarray(
            snap_result.get("fields", {}).get("Ez", np.zeros((0,))),
            dtype=float,
        )
        if ez_snaps.ndim == 4 and ez_snaps.shape[0] > 0:
            ez_last = ez_snaps[-1]
            ez_max = np.max(np.abs(ez_snaps), axis=0)
            z_idx = int(np.clip(round(CORE_ZC / DX), 0, ez_last.shape[0] - 1))
            fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=220)
            im0 = axes[0].imshow(
                ez_last[z_idx],
                origin="lower",
                cmap="RdBu",
                aspect="auto",
            )
            axes[0].set_title(f"Preflight Ez (last, z_idx={z_idx})")
            axes[0].set_xlabel("x idx")
            axes[0].set_ylabel("y idx")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            im1 = axes[1].imshow(
                ez_max[z_idx],
                origin="lower",
                cmap="magma",
                aspect="auto",
            )
            axes[1].set_title("Preflight max|Ez| (z slice)")
            axes[1].set_xlabel("x idx")
            axes[1].set_ylabel("y idx")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(DEBUG_OUT_DIR / "preflight_ez_snapshots.png", dpi=300)
            plt.close(fig)

if DEBUG_DFT:
    prog_dbg = sim_device.compile(num_steps=1)
    for s in prog_dbg.monitor_specs:
        print(
            f"[device spec] {s.name}: dft_enabled={s.dft_enabled}, freq_count={s.freq_count}, "
            f"dft_record_interval={s.dft_record_interval}, dft_t_start={s.dft_t_start:.3e}, dft_t_end={s.dft_t_end:.3e}, "
            f"is_3d={s.is_3d}, dft_points={s.dft_point_count}"
        )

chunk_steps = env_int("BEAMZ_DECAY_CHUNK_STEPS", 64)
min_steps_env = env_int("BEAMZ_DECAY_MIN_STEPS", 256)
max_steps = len(time)
lookback = env_int("BEAMZ_DECAY_LOOKBACK_RECORDS", 12)
decay_ratio = env_float("BEAMZ_DECAY_RATIO", 1e-3)

device_min_time = max(
    dft_window["o1_ref"][1],
    dft_window["o2"][1],
    dft_window["o3"][1],
) + (2.0 / fmin)
device_min_steps = int(np.ceil(device_min_time / DT))
min_steps_device = max(min_steps_env, device_min_steps)

print(
    f"Running 3D compiled device simulation ({WL_POINTS} frequencies, two_run_norm={USE_TWO_RUN_NORM})..."
)
wall_t0 = pytime.time()
stop_steps_device = run_compiled_until_decay(
    sim_device,
    stop_monitors=device_monitors,
    chunk_steps=chunk_steps,
    min_steps=min_steps_device,
    max_steps=max_steps,
    lookback_records=lookback,
    decay_ratio=decay_ratio,
)
device_wall = pytime.time() - wall_t0
print(f"Device run wall-time: {device_wall:.1f} s (steps={stop_steps_device})")

diag_power_data = {}
for mon in power_diag_monitors:
    tt = np.asarray(mon.power_timestamps, dtype=float)
    pw = np.asarray(mon.power_history, dtype=float)
    diag_power_data[mon.name] = (tt, pw)
    if tt.size and pw.size:
        print(
            f"[full-run power] {mon.name}: "
            f"max={np.max(pw):.3e}, mean={np.mean(pw):.3e}, "
            f"int={float(trapz(pw, tt)):.3e}"
        )
if (
    "diag_o1_fwd_power" in diag_power_data
    and diag_power_data["diag_o1_fwd_power"][0].size
    and diag_power_data["diag_o1_fwd_power"][1].size
):
    pin = float(
        trapz(
            diag_power_data["diag_o1_fwd_power"][1],
            diag_power_data["diag_o1_fwd_power"][0],
        )
    )
    p2 = float(
        trapz(diag_power_data["diag_o2_power"][1], diag_power_data["diag_o2_power"][0])
    ) if diag_power_data["diag_o2_power"][0].size else 0.0
    p3 = float(
        trapz(diag_power_data["diag_o3_power"][1], diag_power_data["diag_o3_power"][0])
    ) if diag_power_data["diag_o3_power"][0].size else 0.0
    if pin > 0.0:
        print(f"[full-run transmission proxy] (o2+o3)/in = {(p2 + p3) / pin:.3f}")
    if WRITE_DEBUG_PLOTS:
        fig, ax = plt.subplots(2, 1, figsize=(8.0, 5.8), dpi=220, sharex=True)
        cmap = {
            "diag_o1_fwd_power": "tab:blue",
            "diag_o2_power": "tab:orange",
            "diag_o3_power": "tab:green",
        }
        for nm in ["diag_o1_fwd_power", "diag_o2_power", "diag_o3_power"]:
            tt, pw = diag_power_data[nm]
            if tt.size == 0 or pw.size == 0:
                continue
            t_fs = tt / 1e-15
            ax[0].plot(t_fs, pw, lw=1.6, label=nm, color=cmap[nm])
            ax[1].plot(t_fs, np.cumsum(np.maximum(pw, 0.0)), lw=1.6, label=nm, color=cmap[nm])
        ax[0].set_ylabel("Power (a.u.)")
        ax[1].set_ylabel("Cumulative")
        ax[1].set_xlabel("Time (fs)")
        ax[0].grid(alpha=0.3)
        ax[1].grid(alpha=0.3)
        if ax[0].lines:
            ax[0].legend(loc="best", fontsize=8)
        if ax[1].lines:
            ax[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(DEBUG_OUT_DIR / "fullrun_power_traces.png", dpi=300)
        plt.close(fig)

# Extract modal waves from device run.
device_port_specs = [
    PortSpec(
        name=SOURCE_PORT,
        monitor_name="o1_ref",
        direction=ports[SOURCE_PORT]["direction"],
        polarization=POLARIZATION,
    ),
    PortSpec(
        name="o2",
        monitor_name="o2",
        direction=ports["o2"]["direction"],
        polarization=POLARIZATION,
    ),
    PortSpec(
        name="o3",
        monitor_name="o3",
        direction=ports["o3"]["direction"],
        polarization=POLARIZATION,
    ),
]
waves_dev = sim_device.extract_port_waves_dft(
    ports=device_port_specs,
    frequencies=freqs,
    min_incident_db=-55.0,
    return_power=True,
)

# Incident normalization from a dedicated straight-waveguide reference run.
if USE_TWO_RUN_NORM:
    ref_design = build_reference_design_like(device_design, src)
    ref_grid = ref_design.rasterize(resolution=DX)
    ref_source = ModeSource(
        grid=ref_grid,
        center=source_center,
        width=source_width,
        height=source_height,
        wavelength=float(WL0),
        pol=POLARIZATION,
        signal=signal,
        direction=src["direction"],
    )
    ref_monitor = Monitor(
        start=o1_fwd_start,
        end=o1_fwd_end,
        name="o1_norm",
        dft_t_start=dft_window["o1_fwd"][0],
        dft_t_end=dft_window["o1_fwd"][1],
        dft_components=source_dft_components,
        **monitor_cfg,
    )
    sim_ref = Simulation(
        design=ref_design,
        devices=[ref_source, ref_monitor],
        boundaries=[
            PML(edges=["left", "top", "bottom"], thickness=PML_XY),
            PML(edges="right", thickness=PML_RIGHT),
            PML(edges=["front", "back"], thickness=PML_Z),
        ],
        time=time,
        resolution=DX,
    )
    if DEBUG_DFT:
        prog_ref_dbg = sim_ref.compile(num_steps=1)
        for s in prog_ref_dbg.monitor_specs:
            print(
                f"[ref spec] {s.name}: dft_enabled={s.dft_enabled}, freq_count={s.freq_count}, "
                f"dft_record_interval={s.dft_record_interval}, dft_t_start={s.dft_t_start:.3e}, dft_t_end={s.dft_t_end:.3e}, "
                f"is_3d={s.is_3d}, dft_points={s.dft_point_count}"
            )
    print("Running 3D compiled reference normalization simulation...")
    ref_min_time = dft_window["o1_fwd"][1] + (2.0 / fmin)
    ref_min_steps = max(min_steps_env, int(np.ceil(ref_min_time / DT)))
    ref_t0 = pytime.time()
    stop_steps_ref = run_compiled_until_decay(
        sim_ref,
        stop_monitors=[ref_monitor],
        chunk_steps=chunk_steps,
        min_steps=ref_min_steps,
        max_steps=max_steps,
        lookback_records=lookback,
        decay_ratio=decay_ratio,
    )
    ref_wall = pytime.time() - ref_t0
    print(f"Reference run wall-time: {ref_wall:.1f} s (steps={stop_steps_ref})")
    if DEBUG_DFT:
        print(
            "Reference monitor DFT diagnostics: "
            f"weight_max={float(np.max(np.asarray(ref_monitor._dft_weight_sum, dtype=float))) if hasattr(ref_monitor, '_dft_weight_sum') and np.size(ref_monitor._dft_weight_sum) else 0.0:.3e}, "
            f"|freq_flux|max={float(np.max(np.abs(np.asarray(ref_monitor.frequency_flux_spectrum, dtype=np.complex128)))) if np.size(ref_monitor.frequency_flux_spectrum) else 0.0:.3e}, "
            f"power_records={len(ref_monitor.power_history)}, "
            f"power_max={float(np.max(np.abs(np.asarray(ref_monitor.power_history, dtype=float)))) if len(ref_monitor.power_history) else 0.0:.3e}"
        )

    waves_ref = sim_ref.extract_port_waves_dft(
        ports=[
            PortSpec(
                name="o1_inc",
                monitor_name="o1_norm",
                direction=ports[SOURCE_PORT]["direction"],
                polarization=POLARIZATION,
            )
        ],
        frequencies=freqs,
        min_incident_db=-55.0,
        return_power=True,
    )
    a_incident = np.asarray(waves_ref["o1_inc"]["a_plus"], dtype=np.complex128)
else:
    # Fallback to single-run normalization from forward monitor in device sim.
    single_norm_spec = [
        PortSpec(
            name="o1_inc",
            monitor_name="o1_fwd",
            direction=ports[SOURCE_PORT]["direction"],
            polarization=POLARIZATION,
        )
    ]
    waves_inc = sim_device.extract_port_waves_dft(
        ports=single_norm_spec,
        frequencies=freqs,
        min_incident_db=-55.0,
        return_power=True,
    )
    a_incident = np.asarray(waves_inc["o1_inc"]["a_plus"], dtype=np.complex128)

b_o1 = np.asarray(waves_dev[SOURCE_PORT]["a_minus"], dtype=np.complex128)
b_o2, sel_o2, p2_plus, p2_minus = select_output_wave(
    "o2", waves_dev, auto_select=AUTO_SELECT_OUTPUT_WAVE
)
b_o3, sel_o3, p3_plus, p3_minus = select_output_wave(
    "o3", waves_dev, auto_select=AUTO_SELECT_OUTPUT_WAVE
)
if DEBUG_DFT:
    print(
        "Device wave diagnostics: "
        f"max|b_o1|={np.max(np.abs(b_o1)):.3e}, "
        f"max|b_o2|={np.max(np.abs(b_o2)):.3e}, "
        f"max|b_o3|={np.max(np.abs(b_o3)):.3e}"
    )
print(
    "Output-wave selection: "
    f"o2={sel_o2} (mean|a+|^2={p2_plus:.3e}, mean|a-|^2={p2_minus:.3e}), "
    f"o3={sel_o3} (mean|a+|^2={p3_plus:.3e}, mean|a-|^2={p3_minus:.3e})"
)
for port_name in [SOURCE_PORT, "o2", "o3"]:
    cond = np.asarray(waves_dev[port_name].get("condition_number", []), dtype=float)
    if cond.size:
        print(
            f"[modal conditioning] {port_name}: "
            f"min={np.min(cond):.2e}, median={np.median(cond):.2e}, max={np.max(cond):.2e}"
        )

if CALIBRATE_PORT_SCALE:
    cal_freq = env_float("BEAMZ_CALIBRATION_FREQUENCY_HZ", fcen)
    print(f"Running port-scale calibration at {cal_freq/1e12:.2f} THz ...")
    g_inc = run_modal_gain_calibration(
        port_name="o1_inc_cal",
        port=ports[SOURCE_PORT],
        monitor_start=o1_fwd_start,
        monitor_end=o1_fwd_end,
        port_direction=ports[SOURCE_PORT]["direction"],
        polarization=POLARIZATION,
        coeff_key="a_plus",
        cal_frequency=cal_freq,
        time=time,
        signal=signal,
        chunk_steps=chunk_steps,
        min_steps_env=min_steps_env,
        max_steps=max_steps,
        lookback=lookback,
        decay_ratio=decay_ratio,
        record_interval=max(monitor_stride, 1),
    )
    g_refl = run_modal_gain_calibration(
        port_name="o1_refl_cal",
        port=ports[SOURCE_PORT],
        monitor_start=o1_ref_start,
        monitor_end=o1_ref_end,
        port_direction=ports[SOURCE_PORT]["direction"],
        polarization=POLARIZATION,
        coeff_key="a_minus",
        cal_frequency=cal_freq,
        time=time,
        signal=signal,
        chunk_steps=chunk_steps,
        min_steps_env=min_steps_env,
        max_steps=max_steps,
        lookback=lookback,
        decay_ratio=decay_ratio,
        record_interval=max(monitor_stride, 1),
    )
    g_o2 = run_modal_gain_calibration(
        port_name="o2_cal",
        port=ports["o2"],
        monitor_start=out_planes["o2"][0],
        monitor_end=out_planes["o2"][1],
        port_direction=ports["o2"]["direction"],
        polarization=POLARIZATION,
        coeff_key="a_minus",
        cal_frequency=cal_freq,
        time=time,
        signal=signal,
        chunk_steps=chunk_steps,
        min_steps_env=min_steps_env,
        max_steps=max_steps,
        lookback=lookback,
        decay_ratio=decay_ratio,
        record_interval=max(monitor_stride, 1),
    )
    g_o3 = run_modal_gain_calibration(
        port_name="o3_cal",
        port=ports["o3"],
        monitor_start=out_planes["o3"][0],
        monitor_end=out_planes["o3"][1],
        port_direction=ports["o3"]["direction"],
        polarization=POLARIZATION,
        coeff_key="a_minus",
        cal_frequency=cal_freq,
        time=time,
        signal=signal,
        chunk_steps=chunk_steps,
        min_steps_env=min_steps_env,
        max_steps=max_steps,
        lookback=lookback,
        decay_ratio=decay_ratio,
        record_interval=max(monitor_stride, 1),
    )
    print(
        "Port gain calibration: "
        f"g_inc={g_inc:.3e}, g_refl={g_refl:.3e}, g_o2={g_o2:.3e}, g_o3={g_o3:.3e}"
    )
    a_incident = a_incident / max(g_inc, 1e-18)
    b_o1 = b_o1 / max(g_refl, 1e-18)
    b_o2 = b_o2 / max(g_o2, 1e-18)
    b_o3 = b_o3 / max(g_o3, 1e-18)

s_matrix = {
    ("o1", "o1"): safe_ratio(b_o1, a_incident),
    ("o2", "o1"): safe_ratio(b_o2, a_incident),
    ("o3", "o1"): safe_ratio(b_o3, a_incident),
}

min_incident_db = env_float("BEAMZ_MIN_INCIDENT_DB", -55.0)
max_inc = float(np.max(np.abs(a_incident))) if a_incident.size else 0.0
floor = max(1e-18, max_inc * (10.0 ** (min_incident_db / 20.0)))
valid = np.abs(a_incident) >= floor
print(
    "Incident amplitude diagnostics: "
    f"max={np.max(np.abs(a_incident)):.3e}, "
    f"min={np.min(np.abs(a_incident)):.3e}, "
    f"valid_bins={int(np.count_nonzero(valid))}/{len(valid)}"
)
for key in list(s_matrix.keys()):
    s_matrix[key] = np.where(valid, s_matrix[key], 0.0 + 0.0j)

s_sax = sax.sdict(s_matrix)

power_sum = (
    np.abs(np.asarray(s_sax[("o1", "o1")])) ** 2
    + np.abs(np.asarray(s_sax[("o2", "o1")])) ** 2
    + np.abs(np.asarray(s_sax[("o3", "o1")])) ** 2
)
loss_est = 1.0 - power_sum
power_sum = np.where(valid, power_sum, np.nan)
loss_est = np.where(valid, loss_est, np.nan)

i0 = int(np.argmin(np.abs(wl_um - WL0 / µm)))
print(f"|S11|^2+|S21|^2+|S31|^2 @ {WL0 / µm:.3f}um: {power_sum[i0]:.3f}")
print(f"loss @ {WL0 / µm:.3f}um: {loss_est[i0]:.3f} (valid={bool(valid[i0])})")
for key in [("o1", "o1"), ("o2", "o1"), ("o3", "o1")]:
    s0 = np.asarray(s_sax[key])[i0]
    print(
        f"S[{key[0]},{key[1]}] @ {WL0 / µm:.3f}um: |S|={np.abs(s0):.3f}, "
        f"phase={np.angle(s0):.3f} rad"
    )

# Final diagnostics figure: raw samples only + closure diagnostics.
fig, (ax_s, ax_c) = plt.subplots(2, 1, figsize=(6.0, 5.2), dpi=250, sharex=True)
for key, color in [
    (("o1", "o1"), "black"),
    (("o2", "o1"), "tab:blue"),
    (("o3", "o1"), "tab:orange"),
]:
    vals = np.asarray(s_sax[key], dtype=np.complex128)
    y_db = 20 * np.log10(np.maximum(np.abs(vals), 1e-12))
    y_db = np.where(valid, y_db, np.nan)
    ax_s.plot(wl_um, y_db, "o-", ms=3.0, lw=1.8, color=color, label=rf"$S_{{{key[0][1:]}{key[1][1:]}}}$")

ax_s.set_ylabel("Magnitude (dB)")
ax_s.set_ylim(-45, 1)
ax_s.grid(alpha=0.3)
ax_s.legend(loc="best")
ax_s.set_title("GDSFactory MMI1x2 (3D Compiled DFT, Raw Bins)")

ax_c.plot(wl_um, power_sum, "o-", ms=3.0, lw=1.8, color="tab:green", label="closure: |S11|^2+|S21|^2+|S31|^2")
ax_c.plot(wl_um, loss_est, "o-", ms=3.0, lw=1.8, color="tab:red", label="1 - closure")
ax_c.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6)
ax_c.set_xlabel("Wavelength (um)")
ax_c.set_ylabel("Power")
ax_c.set_xlim(WL_MIN / µm, WL_MAX / µm)
ax_c.grid(alpha=0.3)
ax_c.legend(loc="best")

fig.tight_layout()
out_png = os.getenv("BEAMZ_SAX_PLOT", "sax_splitter_terms_meep_style_3d.png")
fig.savefig(out_png, dpi=300)
if SHOW_FINAL:
    plt.show()
else:
    plt.close(fig)
