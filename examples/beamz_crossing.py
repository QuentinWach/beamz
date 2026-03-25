"""Standard 3D BeamZ crossing example."""

from __future__ import annotations

import argparse
import time as pytime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    dxdt,
    µm,
)
from beamz.design.io import gdsf
from beamz.devices.sources.signals import gaussian_pulse

N_CORE, N_CLAD = 3.47, 1.44
CORE_T, CLAD_BELOW, CLAD_ABOVE = 0.22 * µm, 0.50 * µm, 0.50 * µm
PML_UM, EXT_UM, PORT_MARGIN_UM = 1.0 * µm, 1.5 * µm, 0.5 * µm
PORT_OVERLAP_UM, SOURCE_OFFSET_UM, MONITOR_OFFSET_UM = 0.10 * µm, 0.10 * µm, 0.30 * µm
Z_PAD_UM, EXTRA_DECAY_UOC = 1.10 * µm, 120.0


def outward(direction: str) -> str:
    return ("-" if direction.startswith("+") else "+") + direction[1:]


def move(center: tuple[float, float], direction: str, distance: float) -> tuple[float, float]:
    x, y = center
    return {
        "+x": (x + distance, y),
        "-x": (x - distance, y),
        "+y": (x, y + distance),
        "-y": (x, y - distance),
    }[direction]


def wave_selectors(direction: str) -> tuple[str, str]:
    return ("plus", "minus") if direction.startswith("+") else ("minus", "plus")


def load_crossing(name: str):
    try:
        from ubcpdk import PDK, cells

        PDK.activate()
        return getattr(cells, name)() if hasattr(cells, name) else cells.ebeam_crossing4()
    except Exception:
        import gdsfactory as gf

        try:
            gf.gpdk.PDK.activate()
        except Exception:
            pass
        return gf.get_component(name) if hasattr(gf.components, name) else gf.components.crossing()


def port_plane(port: dict[str, object], offset: float, span: float, z0: float, z1: float):
    cx, cy = move(port["center"], port["direction"], offset)
    if port["direction"].endswith("x"):
        return (cx, cy - 0.5 * span, z0), (cx, cy + 0.5 * span, z1)
    return (cx - 0.5 * span, cy, z0), (cx + 0.5 * span, cy, z1)


def add_port_extensions(design: Design, ports: dict[str, dict[str, object]], core_z0: float) -> None:
    edge = {"+x": design.width, "-x": 0.0, "+y": design.height, "-y": 0.0}
    for port in ports.values():
        cx, cy = port["center"]
        w, d_out = float(port["width"]), outward(port["direction"])
        sx, sy = move((cx, cy), d_out, -PORT_OVERLAP_UM)
        if d_out.endswith("x"):
            x1 = edge[d_out]
            design += Rectangle(
                position=(min(sx, x1), cy - 0.5 * w, core_z0),
                width=abs(x1 - sx),
                height=w,
                depth=CORE_T,
                material=Material(N_CORE**2),
            )
        else:
            y1 = edge[d_out]
            design += Rectangle(
                position=(cx - 0.5 * w, min(sy, y1), core_z0),
                width=w,
                height=abs(y1 - sy),
                depth=CORE_T,
                material=Material(N_CORE**2),
            )


def save_overview(path: Path, eps: np.ndarray, design: Design, source_plane, monitor_planes) -> None:
    z_idx = int(np.clip(round((CLAD_BELOW + Z_PAD_UM + 0.5 * CORE_T) / design.depth * (eps.shape[0] - 1)), 0, eps.shape[0] - 1))
    y_idx = eps.shape[1] // 2
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=240)
    axes[0].imshow(eps[z_idx], origin="lower", extent=[0, design.width / µm, 0, design.height / µm], cmap="viridis", aspect="equal")
    axes[1].imshow(eps[:, y_idx, :], origin="lower", extent=[0, design.width / µm, 0, design.depth / µm], cmap="viridis", aspect="auto")
    for ax, title, ylabel in ((axes[0], "XY overview", "y (um)"), (axes[1], "XZ overview", "z (um)")):
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel(ylabel)
    for name, plane in {"source": source_plane, **monitor_planes}.items():
        (x0, y0, z0), (x1, y1, z1) = plane
        color = "red" if name == "source" else "white"
        axes[0].plot([x0 / µm, x1 / µm], [y0 / µm, y1 / µm], color=color, lw=1.5)
        axes[1].plot([0.5 * (x0 + x1) / µm, 0.5 * (x0 + x1) / µm], [z0 / µm, z1 / µm], color=color, lw=1.5)
        axes[0].text(0.5 * (x0 + x1) / µm, 0.5 * (y0 + y1) / µm + 0.08, name, color=color, fontsize=7, ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=320)
    plt.close(fig)


def save_sparams(path: Path, wavelengths_um: np.ndarray, s_matrix: dict[tuple[str, str], np.ndarray]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.0), dpi=240)
    for port, color in zip(("o1", "o2", "o3", "o4"), ("black", "tab:blue", "tab:green", "tab:orange")):
        mag_db = 20.0 * np.log10(np.maximum(np.abs(np.asarray(s_matrix[(port, "o1")], dtype=np.complex128)), 1e-12))
        ax.plot(wavelengths_um, mag_db, lw=2.0, label=f"S[{port},o1]", color=color)
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title("Crossing S-parameters")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=320)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standard BeamZ crossing example.")
    parser.add_argument("--component", default="ebeam_crossing4", help="Crossing component name from the active PDK.")
    parser.add_argument("--wl0-nm", type=float, default=1550.0, help="Center wavelength in nm.")
    parser.add_argument("--wl-min-nm", type=float, default=1530.0, help="Sweep minimum wavelength in nm.")
    parser.add_argument("--wl-max-nm", type=float, default=1570.0, help="Sweep maximum wavelength in nm.")
    parser.add_argument("--num-freqs", type=int, default=51, help="Number of DFT frequency points.")
    parser.add_argument("--points-per-wavelength", type=int, default=10, help="Grid resolution in points per wavelength.")
    parser.add_argument("--run-after-sources-uoc", type=float, default=EXTRA_DECAY_UOC, help="Post-source run time in um/c units.")
    parser.add_argument("--quiet-run", action="store_true", help="Disable compiled-run progress output.")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/beamz_crossing"), help="Directory for the saved plots.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_freqs < 2 or args.wl_min_nm >= args.wl_max_nm or not (args.wl_min_nm <= args.wl0_nm <= args.wl_max_nm):
        raise ValueError("Invalid wavelength or frequency arguments.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sparam_png, overview_png = args.out_dir / "beamz_crossing_sparams.png", args.out_dir / "beamz_crossing_overview.png"

    wl0, wl_min, wl_max = args.wl0_nm * 1e-9, args.wl_min_nm * 1e-9, args.wl_max_nm * 1e-9
    freqs = np.linspace(LIGHT_SPEED / wl_max, LIGHT_SPEED / wl_min, args.num_freqs, dtype=float)
    f0, fwidth = LIGHT_SPEED / wl0, float(np.ptp(freqs))
    dx, dt = dxdt(wl0, n_max=N_CORE, dims=3, safety_factor=0.999, points_per_wavelength=args.points_per_wavelength)

    imported, ports = gdsf.load(load_crossing(args.component), layer=(1, 0), n_core=N_CORE, n_clad=N_CLAD, padding=0.0)
    pad_xy = EXT_UM + PML_UM + 0.5 * µm
    depth = CLAD_BELOW + CORE_T + CLAD_ABOVE + 2.0 * Z_PAD_UM
    core_z0 = Z_PAD_UM + CLAD_BELOW
    design = Design(width=imported.width + 2.0 * pad_xy, height=imported.height + 2.0 * pad_xy, depth=depth, material=Material(N_CLAD**2))
    for structure in imported.structures[1:]:
        design += structure.copy().shift(pad_xy, pad_xy, core_z0)
    ports = {name: {**port, "center": (port["center"][0] + pad_xy, port["center"][1] + pad_xy)} for name, port in ports.items()}
    add_port_extensions(design, ports, core_z0)
    grid = design.rasterize(resolution=dx)

    span = max(max(float(port["width"]) for port in ports.values()) + 2.0 * PORT_MARGIN_UM, 1.5 * µm)
    z0, z1 = core_z0 - 0.25 * µm, core_z0 + CORE_T + 0.75 * µm
    source_port = ports["o1"]
    source_plane = port_plane(source_port, SOURCE_OFFSET_UM, span, z0, z1)
    monitor_planes = {name: port_plane(port, MONITOR_OFFSET_UM, span, z0, z1) for name, port in ports.items()}
    signal_sigma, signal_t0 = 0.20 / max(fwidth, 1e-30), 4.0 * 0.20 / max(fwidth, 1e-30)
    source_end = signal_t0 + 6.0 * signal_sigma
    t_total = source_end + args.run_after_sources_uoc * µm / LIGHT_SPEED
    time = np.arange(0.0, t_total, dt)

    source = ModeSource(
        grid=grid,
        center=(0.5 * (source_plane[0][0] + source_plane[1][0]), 0.5 * (source_plane[0][1] + source_plane[1][1]), 0.5 * (z0 + z1)),
        width=span,
        height=z1 - z0,
        wavelength=wl0,
        pol="te",
        signal=gaussian_pulse(time, 1.0, signal_t0, signal_sigma, f0, 0.0),
        direction=source_port["direction"],
    )
    monitor_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        dft_window="hann",
        dft_record_every_step=True,
    )
    monitors = [Monitor(start=start, end=end, name=name, **monitor_cfg) for name, (start, end) in monitor_planes.items()]
    sim = Simulation(design=design, devices=[source, *monitors], boundaries=[PML("all", thickness=PML_UM)], time=time, resolution=dx)

    print(
        f"Running crossing: component={args.component}, freqs={args.num_freqs}, dx={dx/µm:.4f}um, "
        f"grid={tuple(np.asarray(grid.permittivity).shape)}, steps={len(time)}, total={t_total*1e15:.2f}fs"
    )
    bench_t0 = pytime.perf_counter()
    sim.run_compiled(num_steps=len(time), progress=not args.quiet_run, record_fields=[])
    wall = max(pytime.perf_counter() - bench_t0, 1e-12)
    voxels = int(np.prod(np.asarray(grid.permittivity).shape))
    print(f"Simulation stats: voxels={voxels:,}, wall={wall:.2f}s, MCUPS={voxels * len(time) / wall / 1e6:.2f}")

    specs = []
    for name, port in ports.items():
        inc, scat = wave_selectors(str(port["direction"]))
        specs.append(
            PortSpec(
                name=name,
                monitor_name=name,
                direction=port["direction"],
                polarization="te",
                mode_index=0,
                incident_wave=inc,
                scattered_wave=scat,
            )
        )
    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=specs,
        output_ports=["o1", "o2", "o3", "o4"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-35.0,
    )
    s_matrix = {key: np.asarray(val, dtype=np.complex128) for key, val in result["s_matrix"].items()}
    wavelengths_um = LIGHT_SPEED / np.asarray(result["diagnostics"]["frequencies"], dtype=float) / µm
    i0 = int(np.argmin(np.abs(wavelengths_um - args.wl0_nm / 1000.0)))
    for port in ("o1", "o2", "o3", "o4"):
        mag = abs(s_matrix[(port, "o1")][i0])
        print(f"S[{port},o1] @ {wavelengths_um[i0]:.4f}um: {20*np.log10(max(mag, 1e-12)):.2f} dB")
    save_sparams(sparam_png, wavelengths_um, s_matrix)
    save_overview(overview_png, np.asarray(grid.permittivity, dtype=float), design, source_plane, monitor_planes)
    print(f"Saved S-parameter plot: {sparam_png}")
    print(f"Saved overview plot: {overview_png}")


if __name__ == "__main__":
    main()
