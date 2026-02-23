from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle as MplRect

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local workspace package import when running from examples/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamz import (
    LIGHT_SPEED,
    Design,
    Material,
    ModeSource,
    Monitor,
    PML,
    PortSpec,
    Rectangle,
    Simulation,
    design,
    um,
    µm,
)
from beamz.const import BLUE, GREEN, ORANGE, PURPLE, RED
from beamz.devices.sources.signals import gaussian_pulse
from beamz.visual.helpers import dxdt


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(default) if value is None else float(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(default) if value is None else int(value)


WL0 = 1.55 * µm
WL_MIN = 1.50 * µm
WL_MAX = 1.60 * µm
WL_POINTS = env_int("BEAMZ_SWEEP_POINTS", 21)
N_CORE = 3.48
N_CLAD = 1.44
N_AIR = 1.0
POINTS_PER_WAVELENGTH = env_int("BEAMZ_PPW", 10)
POLARIZATION = "tm"

DX, DT = dxdt(
    WL0,
    n_max=N_CORE,
    points_per_wavelength=POINTS_PER_WAVELENGTH,
    dims=3,
)

INPUT_EXTENSION = 4.0 * µm
OUTPUT_EXTENSION = 4.0 * µm
Y_MARGIN = 3.0 * µm

CORE_THICKNESS = 0.22 * µm
CLAD_BELOW = 1.8 * µm
CLAD_ABOVE = 1.4 * µm
AIR_TOP = 1.6 * µm
DEVICE_DEPTH = CLAD_BELOW + CORE_THICKNESS + CLAD_ABOVE + AIR_TOP
CORE_Z0 = CLAD_BELOW
CORE_ZC = CORE_Z0 + 0.5 * CORE_THICKNESS

PML_XY = 1.0 * WL0
PML_RIGHT = 1.5 * WL0
PML_Z = 0.6 * WL0
MONITOR_Z_MARGIN = 0.25 * µm
MONITOR_Z_MIN = PML_Z + MONITOR_Z_MARGIN
MONITOR_Z_MAX = DEVICE_DEPTH - PML_Z - MONITOR_Z_MARGIN

SOURCE_PORT = "o1"
OUTPUT_PORTS = ["o2", "o3"]

SOURCE_OFFSET = -1.2 * µm
FORWARD_MONITOR_OFFSET = 0.6 * µm
REFLECTION_MONITOR_BACKOFF = 2.0 * µm
OUTPUT_MONITOR_OFFSET = 1.2 * µm

SOURCE_SPAN_FACTOR = 1.8
SOURCE_MIN_SPAN = 0.9 * µm
MODAL_MONITOR_SPAN_FACTOR = 1.25
MODAL_MONITOR_MIN_SPAN = 0.55 * µm
MODAL_MONITOR_Z_SPAN = max(0.38 * µm, 1.6 * CORE_THICKNESS)

OUT_DIR = Path("benchmarks/results/gdsf_to_sax_compiled")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DESIGN_PLOT_PATH = OUT_DIR / "design_overview_3d.png"
MODE_PLOT_PATH = OUT_DIR / "mode_source_fields_3d.png"
SPLOT_PATH = OUT_DIR / "sparams_compiled_3d.png"

DIR_VEC = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}


def move_along(center_xy: tuple[float, float], direction: str, distance: float) -> tuple[float, float]:
    vx, vy, _ = DIR_VEC[direction]
    return center_xy[0] + vx * float(distance), center_xy[1] + vy * float(distance)


def outward_direction(inward_direction: str) -> str:
    return ("-" if inward_direction.startswith("+") else "+") + inward_direction[1]


def port_span(port: dict, span_factor: float, span_min: float) -> float:
    return max(float(span_min), float(span_factor) * float(port["width"]))


def port_plane(
    port: dict,
    span_factor: float,
    span_min: float,
    z_span: float,
    offset: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cx, cy = move_along(port["center"], port["direction"], offset)
    span = port_span(port, span_factor, span_min)
    z0 = max(float(MONITOR_Z_MIN), float(CORE_ZC - 0.5 * z_span))
    z1 = min(float(MONITOR_Z_MAX), float(CORE_ZC + 0.5 * z_span))
    axis = port["direction"][1]
    if axis == "x":
        return (cx, cy - 0.5 * span, z0), (cx, cy + 0.5 * span, z1)
    return (cx - 0.5 * span, cy, z0), (cx + 0.5 * span, cy, z1)


def monitor_center(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        0.5 * (float(start[0]) + float(end[0])),
        0.5 * (float(start[1]) + float(end[1])),
        0.5 * (float(start[2]) + float(end[2])),
    )


def estimate_arrival_time(
    source_xyz: tuple[float, float, float],
    monitor_xyz: tuple[float, float, float],
    n_group: float,
    t_launch: float,
) -> float:
    distance = float(
        np.linalg.norm(np.asarray(monitor_xyz, dtype=float) - np.asarray(source_xyz, dtype=float))
    )
    return float(t_launch + n_group * distance / LIGHT_SPEED)


def dft_projection_components_for_port(direction: str) -> tuple[str, str, str, str]:
    axis = direction[1]
    if axis == "x":
        return ("Ey", "Ez", "Hy", "Hz")
    if axis == "y":
        return ("Ex", "Ez", "Hx", "Hz")
    return ("Ex", "Ey", "Hx", "Hy")


def draw_pml_lines(
    ax,
    x_max_um: float,
    y_max_um: float,
    pml_x_um: float,
    pml_y_um: float,
    *,
    color: str = "white",
) -> None:
    ax.axvline(pml_x_um, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axvline(x_max_um - pml_x_um, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhline(pml_y_um, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhline(y_max_um - pml_y_um, color=color, linestyle="--", linewidth=1.0, alpha=0.85)


def save_mode_plot(source: ModeSource, out_path: Path) -> None:
    eps = np.asarray(getattr(source, "_eps_profile_2d", np.zeros((1, 1))), dtype=float)
    fields = {
        "Ex": np.asarray(getattr(source, "_Ex_profile", 0.0), dtype=float),
        "Ey": np.asarray(getattr(source, "_Ey_profile", 0.0), dtype=float),
        "Ez": np.asarray(getattr(source, "_Ez_profile", 0.0), dtype=float),
        "Hx": np.asarray(getattr(source, "_Hx_profile", 0.0), dtype=float),
        "Hy": np.asarray(getattr(source, "_Hy_profile", 0.0), dtype=float),
        "Hz": np.asarray(getattr(source, "_Hz_profile", 0.0), dtype=float),
    }

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=220)
    flat = axes.ravel()
    im0 = flat[0].imshow(eps, origin="lower", cmap="viridis", aspect="auto")
    flat[0].set_title("eps(r)")
    fig.colorbar(im0, ax=flat[0], fraction=0.046, pad=0.04)

    for idx, name in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], start=1):
        arr = np.atleast_2d(np.asarray(fields[name]).squeeze())
        im = flat[idx].imshow(np.abs(arr), origin="lower", cmap="magma", aspect="auto")
        flat[idx].set_title(name)
        fig.colorbar(im, ax=flat[idx], fraction=0.046, pad=0.04)

    flat[7].axis("off")
    neff = float(np.real(getattr(source, "_neff", np.nan)))
    neff_imp = float(np.real(getattr(source, "_impedance_neff", np.nan)))
    flat[7].text(
        0.02,
        0.95,
        (
            f"pol={source.pol}\n"
            f"dir={source.direction}\n"
            f"neff={neff:.6f}\n"
            f"neff_imp={neff_imp:.6f}\n"
            f"width={float(source.width)/µm:.3f} um\n"
            f"height={float(source.height)/µm:.3f} um"
        ),
        va="top",
        ha="left",
        fontsize=10,
    )
    for ax in flat[:7]:
        ax.set_xlabel("u")
        ax.set_ylabel("v")
    fig.suptitle("Mode Source Fields", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_design_overview_plot(
    design_obj: Design,
    permittivity: np.ndarray,
    source_center: tuple[float, float, float],
    source_width: float,
    source_height: float,
    monitor_planes: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
    out_path: Path,
) -> None:
    eps = np.asarray(permittivity, dtype=float)
    eps_levels = np.asarray([N_AIR**2, N_CLAD**2, N_CORE**2], dtype=float)
    material_names = ["Air", "Cladding", "Core"]
    material_colors = [BLUE, GREEN, ORANGE]

    material_idx = np.argmin(
        np.abs(eps[..., None] - eps_levels[None, None, None, :]),
        axis=-1,
    )
    cmap = ListedColormap(material_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(eps_levels) + 0.5, 1.0), cmap.N)

    z_mid = int(np.clip(round(source_center[2] / DX), 0, eps.shape[0] - 1))
    y_mid = int(np.clip(round(source_center[1] / DX), 0, eps.shape[1] - 1))
    x_src = int(np.clip(round(source_center[0] / DX), 0, eps.shape[2] - 1))

    xy = material_idx[z_mid]
    xz = material_idx[:, y_mid, :]
    yz = material_idx[:, :, x_src]
    outline_levels = np.arange(0.5, len(eps_levels) - 0.5, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=220)
    monitor_colors = {"o1_fwd": BLUE, "o1_ref": PURPLE, "o2": GREEN, "o3": ORANGE}

    axes[0].imshow(
        xy,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[0, design_obj.width / um, 0, design_obj.height / um],
        aspect="equal",
    )
    axes[0].set_title(f"XY Slice (z={source_center[2]/um:.2f} um)")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[0].contour(
        xy,
        levels=outline_levels,
        colors="black",
        linewidths=0.9,
        origin="lower",
        extent=[0, design_obj.width / um, 0, design_obj.height / um],
    )

    axes[1].imshow(
        xz,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[0, design_obj.width / um, 0, design_obj.depth / um],
        aspect="equal",
    )
    axes[1].set_title(f"XZ Slice (y={source_center[1]/um:.2f} um)")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("z (um)")
    axes[1].contour(
        xz,
        levels=outline_levels,
        colors="black",
        linewidths=0.9,
        origin="lower",
        extent=[0, design_obj.width / um, 0, design_obj.depth / um],
    )

    axes[2].imshow(
        yz,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[0, design_obj.height / um, 0, design_obj.depth / um],
        aspect="equal",
    )
    axes[2].set_title(f"YZ Slice (x={source_center[0]/um:.2f} um)")
    axes[2].set_xlabel("y (um)")
    axes[2].set_ylabel("z (um)")
    axes[2].contour(
        yz,
        levels=outline_levels,
        colors="black",
        linewidths=0.9,
        origin="lower",
        extent=[0, design_obj.height / um, 0, design_obj.depth / um],
    )

    for name, (start, end) in monitor_planes.items():
        x0, y0, z0 = start
        x1, y1, z1 = end
        color = monitor_colors.get(name, "cyan")
        axes[0].plot([x0 / um, x1 / um], [0.5 * (y0 + y1) / um, 0.5 * (y0 + y1) / um], color=color, lw=1.8)
        axes[1].plot([x0 / um, x1 / um], [0.5 * (z0 + z1) / um, 0.5 * (z0 + z1) / um], color=color, lw=1.8)
        axes[2].add_patch(
            MplRect(
                (min(y0, y1) / um, min(z0, z1) / um),
                abs(y1 - y0) / um,
                abs(z1 - z0) / um,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )

    axes[0].add_patch(
        MplRect((source_center[0] / um - 0.03, (source_center[1] - 0.5 * source_width) / um), 0.06, source_width / um, fill=False, edgecolor=RED, linewidth=2.0)
    )
    axes[1].add_patch(
        MplRect((source_center[0] / um - 0.03, (source_center[2] - 0.5 * source_height) / um), 0.06, source_height / um, fill=False, edgecolor=RED, linewidth=2.0)
    )
    axes[2].add_patch(
        MplRect(((source_center[1] - 0.5 * source_width) / um, (source_center[2] - 0.5 * source_height) / um), source_width / um, source_height / um, fill=False, edgecolor=RED, linewidth=2.0)
    )

    draw_pml_lines(axes[0], design_obj.width / um, design_obj.height / um, PML_XY / um, PML_XY / um)
    draw_pml_lines(axes[1], design_obj.width / um, design_obj.depth / um, PML_XY / um, PML_Z / um)
    draw_pml_lines(axes[2], design_obj.height / um, design_obj.depth / um, PML_XY / um, PML_Z / um)

    for ax in axes:
        ax.grid(alpha=0.2, linestyle="--")

    material_handles = [
        Patch(facecolor=material_colors[i], edgecolor="black", label=f"{material_names[i]} (eps={eps_levels[i]:.3f})")
        for i in range(len(material_names))
    ]
    monitor_handles = [
        Line2D([0], [0], color=monitor_colors[name], lw=1.8, label=name)
        for name in ["o1_fwd", "o1_ref", "o2", "o3"]
        if name in monitor_planes
    ]
    extras = [
        Line2D([0], [0], color=RED, lw=2.0, label="Mode source"),
        Line2D([0], [0], color="white", lw=1.0, linestyle="--", label="PML"),
    ]
    fig.legend(
        handles=[*material_handles, *monitor_handles, *extras],
        loc="lower center",
        ncol=4,
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_mode_source_with_autofit(
    *,
    grid,
    center: tuple[float, float, float],
    base_span: float,
    wavelength: float,
    signal: np.ndarray,
    direction: str,
    polarization: str,
    port_width: float,
) -> ModeSource:
    candidate_scales = [1.0, 1.3, 1.6]
    best_source = None
    best_neff = -np.inf
    for scale in candidate_scales:
        span = max(float(SOURCE_MIN_SPAN), float(base_span) * scale, 1.2 * float(port_width))
        candidate = ModeSource(
            grid=grid,
            center=center,
            width=span,
            height=span,
            wavelength=float(wavelength),
            pol=polarization,
            signal=np.asarray(signal, dtype=np.float32),
            direction=direction,
        )
        candidate.initialize(grid.permittivity, DX, dt=DT)
        neff = float(np.real(getattr(candidate, "_neff", np.nan)))
        if np.isfinite(neff) and neff > best_neff:
            best_neff = neff
            best_source = candidate
        if np.isfinite(neff) and neff >= (N_CLAD + 0.15):
            return candidate
    if best_source is None:
        raise RuntimeError("Unable to build a valid mode source.")
    return best_source


if __name__ == "__main__":
    imported_design, imported_ports = design.io.gdsf.load(
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

    # Add top air region above the cladding stack.
    device_design += Rectangle(
        position=(0.0, 0.0, DEVICE_DEPTH - AIR_TOP),
        width=float(device_design.width),
        height=float(device_design.height),
        depth=float(AIR_TOP),
        material=Material(N_AIR**2),
    )

    for structure in imported_design.structures[1:]:
        shifted = structure.copy().shift(INPUT_EXTENSION, Y_MARGIN, CORE_Z0)
        shifted.z = CORE_Z0
        shifted.depth = CORE_THICKNESS
        device_design += shifted

    ports = {
        name: {
            **port,
            "center": (port["center"][0] + INPUT_EXTENSION, port["center"][1] + Y_MARGIN),
        }
        for name, port in imported_ports.items()
    }

    for name, port in ports.items():
        cx, cy = port["center"]
        width = float(port["width"])
        extension = INPUT_EXTENSION if name == SOURCE_PORT else OUTPUT_EXTENSION
        ox, oy = move_along((cx, cy), outward_direction(port["direction"]), extension)
        if port["direction"][1] == "x":
            device_design += Rectangle(
                position=(min(cx, ox), cy - width / 2.0, CORE_Z0),
                width=extension,
                height=width,
                depth=CORE_THICKNESS,
                material=Material(N_CORE**2),
            )
        else:
            device_design += Rectangle(
                position=(cx - width / 2.0, min(cy, oy), CORE_Z0),
                width=width,
                height=extension,
                depth=CORE_THICKNESS,
                material=Material(N_CORE**2),
            )

    grid = device_design.rasterize(resolution=DX)
    nz, ny, nx = grid.permittivity.shape
    print(f"Created 3D mesh: {nx} x {ny} x {nz} cells")

    src_port = ports[SOURCE_PORT]
    source_xy = move_along(src_port["center"], src_port["direction"], SOURCE_OFFSET)
    source_center = (source_xy[0], source_xy[1], CORE_ZC)
    source_span = port_span(src_port, SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN)

    o1_fwd_start, o1_fwd_end = port_plane(
        src_port,
        MODAL_MONITOR_SPAN_FACTOR,
        MODAL_MONITOR_MIN_SPAN,
        MODAL_MONITOR_Z_SPAN,
        offset=FORWARD_MONITOR_OFFSET,
    )
    o1_ref_start, o1_ref_end = port_plane(
        src_port,
        MODAL_MONITOR_SPAN_FACTOR,
        MODAL_MONITOR_MIN_SPAN,
        MODAL_MONITOR_Z_SPAN,
        offset=-REFLECTION_MONITOR_BACKOFF,
    )

    out_planes: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    for out_name in OUTPUT_PORTS:
        out_port_for_placement = dict(ports[out_name])
        out_port_for_placement["direction"] = outward_direction(out_port_for_placement["direction"])
        out_planes[out_name] = port_plane(
            out_port_for_placement,
            MODAL_MONITOR_SPAN_FACTOR,
            MODAL_MONITOR_MIN_SPAN,
            MODAL_MONITOR_Z_SPAN,
            offset=OUTPUT_MONITOR_OFFSET,
        )

    freqs = np.linspace(LIGHT_SPEED / WL_MAX, LIGHT_SPEED / WL_MIN, WL_POINTS, dtype=np.float32)
    wl = LIGHT_SPEED / freqs
    wl_um = wl / µm
    fmin = float(np.min(freqs))
    fmax = float(np.max(freqs))
    fcen = 0.5 * (fmin + fmax)
    fwidth = max(fmax - fmin, 1e9)

    sigma_t = 0.20 / fwidth
    t0 = 4.0 * sigma_t
    n_group_est = 2.2
    window_pre = 4.0 * sigma_t
    window_post = 10.0 * sigma_t + 18.0 / fmin

    arr_fwd = estimate_arrival_time(source_center, monitor_center(o1_fwd_start, o1_fwd_end), n_group_est, t0)
    x_scatter = float(np.median([ports[name]["center"][0] for name in OUTPUT_PORTS]))
    refl_path = max(0.0, 2.0 * (x_scatter - source_center[0]))
    arr_ref = float(t0 + n_group_est * refl_path / LIGHT_SPEED)
    arr_o2 = estimate_arrival_time(source_center, monitor_center(*out_planes["o2"]), n_group_est, t0)
    arr_o3 = estimate_arrival_time(source_center, monitor_center(*out_planes["o3"]), n_group_est, t0)

    dft_window = {
        "o1_fwd": (max(0.0, arr_fwd - window_pre), arr_fwd + window_post),
        "o1_ref": (max(0.0, arr_ref - window_pre), arr_ref + window_post),
        "o2": (max(0.0, arr_o2 - window_pre), arr_o2 + window_post),
        "o3": (max(0.0, arr_o3 - window_pre), arr_o3 + window_post),
    }

    total_time = max(v[1] for v in dft_window.values()) + 2.0 / fmin
    time_axis = np.arange(0.0, total_time, DT, dtype=np.float32)
    signal = np.asarray(
        gaussian_pulse(
            time_axis,
            amplitude=1.0,
            center=t0,
            width=sigma_t,
            frequency=fcen,
            phase=0.0,
        ),
        dtype=np.float32,
    )

    source = build_mode_source_with_autofit(
        grid=grid,
        center=source_center,
        base_span=source_span,
        wavelength=float(WL0),
        signal=signal,
        direction=src_port["direction"],
        polarization=POLARIZATION,
        port_width=float(src_port["width"]),
    )
    print(
        "Mode source diagnostics: "
        f"pol={POLARIZATION}, dir={src_port['direction']}, "
        f"neff={float(np.real(source._neff)):.6f}, "
        f"neff_imp={float(np.real(source._impedance_neff)):.6f}, "
        f"width={float(source.width)/µm:.3f}um, height={float(source.height)/µm:.3f}um"
    )

    monitor_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_window="hann",
        record_interval=2,
        dft_record_every_step=False,
    )

    monitors = [
        Monitor(
            start=o1_fwd_start,
            end=o1_fwd_end,
            name="o1_fwd",
            dft_t_start=dft_window["o1_fwd"][0],
            dft_t_end=dft_window["o1_fwd"][1],
            dft_components=dft_projection_components_for_port(src_port["direction"]),
            **monitor_cfg,
        ),
        Monitor(
            start=o1_ref_start,
            end=o1_ref_end,
            name="o1_ref",
            dft_t_start=dft_window["o1_ref"][0],
            dft_t_end=dft_window["o1_ref"][1],
            dft_components=dft_projection_components_for_port(src_port["direction"]),
            **monitor_cfg,
        ),
    ]
    for out_name in OUTPUT_PORTS:
        start, end = out_planes[out_name]
        monitors.append(
            Monitor(
                start=start,
                end=end,
                name=out_name,
                dft_t_start=dft_window[out_name][0],
                dft_t_end=dft_window[out_name][1],
                dft_components=dft_projection_components_for_port(ports[out_name]["direction"]),
                **monitor_cfg,
            )
        )

    sim = Simulation(
        design=device_design,
        devices=[source, *monitors],
        boundaries=[
            PML(edges=["left", "top", "bottom"], thickness=PML_XY),
            PML(edges="right", thickness=PML_RIGHT),
            PML(edges=["front", "back"], thickness=PML_Z),
        ],
        time=time_axis,
        resolution=DX,
    )

    save_design_overview_plot(
        device_design,
        grid.permittivity,
        source_center=source_center,
        source_width=float(source.width),
        source_height=float(source.height),
        monitor_planes={
            "o1_fwd": (o1_fwd_start, o1_fwd_end),
            "o1_ref": (o1_ref_start, o1_ref_end),
            "o2": out_planes["o2"],
            "o3": out_planes["o3"],
        },
        out_path=DESIGN_PLOT_PATH,
    )
    save_mode_plot(source, MODE_PLOT_PATH)

    print(f"Running one 3D compiled simulation ({WL_POINTS} frequencies)...")
    wall_t0 = time.perf_counter()
    sim.run_compiled(num_steps=len(time_axis), progress=True)
    sim.fields.Ez.block_until_ready()
    wall_s = time.perf_counter() - wall_t0
    steps = len(time_axis)
    voxels = int(np.prod(sim.fields.permittivity.shape))
    gcups = (6.0 * float(voxels) * float(steps)) / max(wall_s, 1e-12) / 1e9
    print(
        f"Simulation wall-time: {wall_s:.2f} s | steps={steps} | "
        f"voxels={voxels} | GCUPS={gcups:.4f}"
    )

    port_specs = {
        "o1": PortSpec(
            name="o1",
            monitor_name="o1_ref",
            reference_monitor="o1_fwd",
            direction=ports["o1"]["direction"],
            polarization=POLARIZATION,
        ),
        "o2": PortSpec(
            name="o2",
            monitor_name="o2",
            direction=ports["o2"]["direction"],
            polarization=POLARIZATION,
        ),
        "o3": PortSpec(
            name="o3",
            monitor_name="o3",
            direction=ports["o3"]["direction"],
            polarization=POLARIZATION,
        ),
    }
    modal = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=port_specs,
        output_ports=["o1", "o2", "o3"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-50.0,
    )

    s = modal["s_matrix"]
    diag = modal["diagnostics"]
    valid = np.asarray(diag["valid_mask"], dtype=bool)
    s11 = np.asarray(s[("o1", "o1")], dtype=np.complex128)
    s21 = np.asarray(s[("o2", "o1")], dtype=np.complex128)
    s31 = np.asarray(s[("o3", "o1")], dtype=np.complex128)
    closure = np.abs(s11) ** 2 + np.abs(s21) ** 2 + np.abs(s31) ** 2
    closure = np.where(valid, closure, np.nan)
    loss = np.where(valid, 1.0 - closure, np.nan)

    idx0 = int(np.argmin(np.abs(wl_um - WL0 / µm)))
    print(f"|S11|^2+|S21|^2+|S31|^2 @ {WL0 / µm:.3f}um: {closure[idx0]:.3f}")
    print(f"loss @ {WL0 / µm:.3f}um: {loss[idx0]:.3f} (valid={bool(valid[idx0])})")
    for key in [("o1", "o1"), ("o2", "o1"), ("o3", "o1")]:
        sval = np.asarray(s[key], dtype=np.complex128)[idx0]
        print(
            f"S[{key[0]},{key[1]}] @ {WL0 / µm:.3f}um: "
            f"|S|={np.abs(sval):.3f}, phase={np.angle(sval):.3f} rad"
        )

    if np.any(valid):
        closure_valid = closure[valid]
        print(
            "Closure diagnostics: "
            f"min={np.nanmin(closure_valid):.3f}, "
            f"max={np.nanmax(closure_valid):.3f}, "
            f"mean={np.nanmean(closure_valid):.3f}"
        )

    fig, (ax_s, ax_c) = plt.subplots(2, 1, figsize=(6.5, 5.8), dpi=280, sharex=True)
    for key, color, label in [
        (("o1", "o1"), "black", r"$S_{11}$"),
        (("o2", "o1"), "tab:blue", r"$S_{21}$"),
        (("o3", "o1"), "tab:orange", r"$S_{31}$"),
    ]:
        values = np.asarray(s[key], dtype=np.complex128)
        db = 20.0 * np.log10(np.maximum(np.abs(values), 1e-12))
        db = np.where(valid, db, np.nan)
        ax_s.plot(wl_um, db, "o-", ms=3.0, lw=1.6, color=color, label=label)
    ax_s.set_ylabel("Magnitude (dB)")
    ax_s.set_ylim(-45.0, 1.0)
    ax_s.set_title("GDSFactory MMI1x2 (3D compiled DFT)")
    ax_s.grid(alpha=0.3)
    ax_s.legend(loc="best")

    ax_c.plot(wl_um, closure, "o-", ms=3.0, lw=1.8, color="tab:green", label="|S11|^2+|S21|^2+|S31|^2")
    ax_c.plot(wl_um, loss, "o-", ms=3.0, lw=1.8, color="tab:red", label="1 - closure")
    ax_c.axhline(1.0, color="k", lw=1.0, linestyle="--", alpha=0.6)
    ax_c.set_xlabel("Wavelength (um)")
    ax_c.set_ylabel("Power")
    ax_c.set_xlim(WL_MIN / µm, WL_MAX / µm)
    ax_c.grid(alpha=0.3)
    ax_c.legend(loc="best")

    fig.tight_layout()
    fig.savefig(SPLOT_PATH, dpi=300)
    plt.close(fig)

    print(f"Saved design overview: {DESIGN_PLOT_PATH}")
    print(f"Saved mode fields: {MODE_PLOT_PATH}")
    print(f"Saved S-parameter plot: {SPLOT_PATH}")
