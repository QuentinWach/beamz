"""Tiny standalone 3D BeamZ crossing example.

Workflow:
1. Define fixed hyperparameters.
2. Build the 3D crossing geometry and extend the ports.
3. Define the broadband source and DFT monitors.
4. Build the simulation.
5. Save an overview plot of the design, source, and monitors.
6. Run the simulation with adaptive monitor-decay stopping.
7. Extract and plot the S-parameters.
"""

from __future__ import annotations

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

OUT_DIR = Path("benchmarks/results/tiny_beamz_crossing")
COMPONENT_NAME = "ebeam_crossing4"
WL0 = 1550.0e-9
WL_MIN = 1530.0e-9
WL_MAX = 1570.0e-9
NUM_FREQS = 51
PPW = 10
N_CORE, N_CLAD = 3.47, 1.44
LAYER = (1, 0)
CORE_T = 0.22 * µm
CLAD_BELOW = 0.50 * µm
CLAD_ABOVE = 0.50 * µm
PML_XY = 1.0 * µm
PML_Z = 1.0 * µm
XY_MARGIN = 0.50 * µm
Z_PADDING = 1.10 * µm
EXTENSION = 1.50 * µm
PORT_OVERLAP = 0.10 * µm
PORT_MARGIN = 0.50 * µm
SOURCE_OFFSET = 0.10 * µm
MONITOR_OFFSET = 0.30 * µm
RUN_AFTER_SOURCES_UOC = 90.0


def outward(direction: str) -> str:
    return ("-" if direction.startswith("+") else "+") + direction[1:]


def positive_axis(direction: str) -> str:
    return "+" + direction[1:]


def incoming_wave(direction: str) -> str:
    return "plus" if direction.startswith("+") else "minus"


def outgoing_wave(direction: str) -> str:
    return "minus" if direction.startswith("+") else "plus"


def move(center: tuple[float, float], direction: str, distance: float) -> tuple[float, float]:
    x, y = center
    return {
        "+x": (x + distance, y),
        "-x": (x - distance, y),
        "+y": (x, y + distance),
        "-y": (x, y - distance),
    }[direction]


def port_plane(port: dict[str, object], span: float, z_span: float, z_center: float, offset: float):
    cx, cy = move(port["center"], port["direction"], offset)
    z0, z1 = z_center - 0.5 * z_span, z_center + 0.5 * z_span
    if port["direction"].endswith("x"):
        return (cx, cy - 0.5 * span, z0), (cx, cy + 0.5 * span, z1)
    return (cx - 0.5 * span, cy, z0), (cx + 0.5 * span, cy, z1)


def line_center(line):
    a, b = line
    return tuple(0.5 * (float(a[i]) + float(b[i])) for i in range(len(a)))


def wave_dominance_db(a_plus: np.ndarray, a_minus: np.ndarray, selector: str, mask: np.ndarray) -> float:
    sel = np.asarray(a_plus if selector == "plus" else a_minus, dtype=np.complex128)
    opp = np.asarray(a_minus if selector == "plus" else a_plus, dtype=np.complex128)
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    p_sel = float(np.mean(np.abs(sel[valid]) ** 2))
    p_opp = float(np.mean(np.abs(opp[valid]) ** 2))
    return 10.0 * np.log10(max(p_sel, 1e-18) / max(p_opp, 1e-18))


def build_pulse(freqs: np.ndarray, dt: float, max_output_distance_um: float):
    df = max(float(np.ptp(freqs)), 1e-12)
    fmin = max(float(np.min(freqs)), 1e-12)
    sigma = 0.20 / max(df, 1e9)
    peak = 4.0 * sigma
    source_end = peak + 6.0 * sigma
    min_tail_uoc = max(RUN_AFTER_SOURCES_UOC, 6.0 * max_output_distance_um)
    tail = max(min_tail_uoc * 1e-6 / LIGHT_SPEED, 96.0 / fmin)
    tail_cap = max(180.0 * 1e-6 / LIGHT_SPEED, 192.0 / fmin)
    time = np.arange(0.0, source_end + tail_cap, dt)
    signal = np.asarray(gaussian_pulse(time, 1.0, peak, sigma, LIGHT_SPEED / WL0, 0.0), dtype=np.float32)
    return time, signal, source_end + tail


def build_crossing_design():
    try:
        from ubcpdk import PDK, cells

        PDK.activate()
        component = getattr(cells, COMPONENT_NAME)()
        component_label = f"ubcpdk.cells.{COMPONENT_NAME}"
    except Exception:
        import gdsfactory as gf

        try:
            gf.gpdk.PDK.activate()
        except Exception:
            pass
        component = gf.get_component(COMPONENT_NAME)
        component_label = f"gf.get_component('{COMPONENT_NAME}')"

    imported, raw_ports = gdsf.load(component, layer=LAYER, n_core=N_CORE, n_clad=N_CLAD, padding=0.0)
    depth = 2.0 * Z_PADDING + CLAD_BELOW + CORE_T + CLAD_ABOVE
    core_z0 = Z_PADDING + CLAD_BELOW
    design = Design(
        width=imported.width + 2.0 * EXTENSION,
        height=imported.height + 2.0 * EXTENSION,
        depth=depth,
        material=Material(N_CLAD**2),
    )
    for structure in imported.structures[1:]:
        shifted = structure.copy().shift(EXTENSION, EXTENSION, core_z0)
        shifted.z = core_z0
        shifted.depth = CORE_T
        design += shifted
    ports = {
        name: {
            **port,
            "center": (float(port["center"][0] + EXTENSION), float(port["center"][1] + EXTENSION)),
            "width": float(port["width"]),
            "z_center": float(core_z0 + 0.5 * CORE_T),
        }
        for name, port in raw_ports.items()
    }
    edge = {"+x": design.width, "-x": 0.0, "+y": design.height, "-y": 0.0}
    for port in ports.values():
        cx, cy = port["center"]
        w, d_out = float(port["width"]), outward(port["direction"])
        sx, sy = move((cx, cy), d_out, -PORT_OVERLAP)
        if d_out.endswith("x"):
            x1 = edge[d_out]
            design += Rectangle((min(sx, x1), cy - 0.5 * w, core_z0), abs(x1 - sx), w, CORE_T, Material(N_CORE**2))
        else:
            y1 = edge[d_out]
            design += Rectangle((cx - 0.5 * w, min(sy, y1), core_z0), w, abs(y1 - sy), CORE_T, Material(N_CORE**2))
    design.unify_polygons()
    return component_label, design, ports, imported, core_z0


def save_overview(path: Path, eps: np.ndarray, design: Design, source_plane, monitor_planes) -> None:
    z_idx = int(np.clip(round((Z_PADDING + CLAD_BELOW + 0.5 * CORE_T) / design.depth * (eps.shape[0] - 1)), 0, eps.shape[0] - 1))
    y_idx = eps.shape[1] // 2
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=260)
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


def save_sparams(path: Path, wl_um: np.ndarray, s_matrix: dict[tuple[str, str], np.ndarray]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.5), dpi=320)
    colors = {"o1": "black", "o2": "tab:blue", "o3": "tab:orange", "o4": "tab:green"}
    for port in ("o1", "o2", "o3", "o4"):
        y_db = 20.0 * np.log10(np.maximum(np.abs(np.asarray(s_matrix[(port, "o1")], dtype=np.complex128)), 1e-12))
        ax.plot(wl_um, y_db, "o-", lw=2.0, ms=4.0, color=colors[port], label=rf"$|S_{{{port[1:]}1}}|$")
    ax.set_xlim(float(np.min(wl_um)), float(np.max(wl_um)))
    ax.set_ylim(-55.0, 0.0)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Crossing S-Parameters")
    ax.grid(which="major", alpha=0.25, lw=0.6)
    ax.minorticks_on()
    ax.grid(which="minor", alpha=0.12, lw=0.4)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=320)
    plt.close(fig)


def run_until_decay(sim: Simulation, stop_monitors: list[Monitor], time: np.ndarray, min_time_s: float) -> int:
    total_steps = len(time)
    dt = float(time[1] - time[0])
    chunk_steps = max(64, min(512, int(np.ceil(total_steps / 24.0))))
    min_steps = int(np.ceil(max(0.0, min_time_s) / max(dt, 1e-30)))
    steps_done = 0
    peak = 0.0
    print(
        "Compiled run mode: adaptive monitor-decay stop "
        f"(chunk_steps={chunk_steps}, min_steps={min_steps}, max_steps={total_steps}, decay_ratio=1.0e-03)"
    )
    while steps_done < total_steps:
        this_chunk = min(chunk_steps, total_steps - steps_done)
        sim.run_compiled(num_steps=this_chunk, progress=False)
        steps_done += this_chunk
        histories = [np.abs(np.asarray(m.power_history, dtype=np.float64)) for m in stop_monitors if len(m.power_history)]
        if histories:
            peak = max(peak, max(float(np.max(h)) for h in histories))
            tail = max(float(np.max(h[-12:])) for h in histories)
        else:
            tail = np.inf
        print(f"\r● Progress: {100.0 * steps_done / total_steps:.0f}% ({steps_done}/{total_steps} steps)", end="", flush=True)
        if steps_done >= min_steps and peak > 0.0 and np.isfinite(tail) and tail <= 1e-3 * peak:
            break
    print()
    return steps_done


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    component_label, design, ports, _imported, _core_z0 = build_crossing_design()
    source_port, output_ports = "o1", ["o2", "o3", "o4"]
    dx, dt = dxdt(WL0, n_max=N_CORE, dims=3, safety_factor=0.999, points_per_wavelength=PPW)
    grid = design.rasterize(resolution=dx)
    freqs = np.linspace(LIGHT_SPEED / WL_MAX, LIGHT_SPEED / WL_MIN, NUM_FREQS, dtype=np.float32)
    wl_um = LIGHT_SPEED / freqs / µm

    src = ports[source_port]
    source_direction = src["direction"]
    span = max(float(src["width"]) + 2.0 * PORT_MARGIN, float(src["width"]) + 0.1 * µm)
    z_center = float(src["z_center"])
    z_span = CLAD_BELOW + CORE_T + CLAD_ABOVE
    source_plane = port_plane(src, span, z_span, z_center, SOURCE_OFFSET)
    fwd_plane = port_plane(src, span, z_span, z_center, MONITOR_OFFSET)
    source_center = line_center(source_plane)
    out_planes = {}
    max_output_distance_um = 0.0
    for port_name in output_ports:
        out_port = {**ports[port_name], "direction": outward(ports[port_name]["direction"])}
        plane = port_plane(out_port, span, z_span, z_center, SOURCE_OFFSET)
        out_planes[port_name] = plane
        c_out = line_center(plane)
        max_output_distance_um = max(max_output_distance_um, float(np.hypot(c_out[0] - source_center[0], c_out[1] - source_center[1])) / µm)

    time, signal, decay_min_time_s = build_pulse(freqs, dt, max_output_distance_um)
    source = ModeSource(
        grid=grid,
        center=source_center,
        width=span,
        height=z_span,
        wavelength=WL0,
        pol="te",
        signal=signal,
        direction=source_direction,
    )
    monitor_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        dft_window="none",
        dft_record_every_step=True,
    )
    m_fwd = Monitor(start=fwd_plane[0], end=fwd_plane[1], name="o1_fwd", **monitor_cfg)
    out_monitors = [
        Monitor(start=out_planes[p][0], end=out_planes[p][1], name=f"{p}_cand0", **monitor_cfg)
        for p in output_ports
    ]
    sim = Simulation(
        design=design,
        devices=[source, m_fwd, *out_monitors],
        boundaries=[PML(edges=["left", "right", "top", "bottom"], thickness=PML_XY), PML(edges=["front", "back"], thickness=PML_Z)],
        time=time,
        resolution=dx,
    )

    print(
        "Running crossing modal DFT extraction: "
        f"component={component_label}, source={source_port}, outputs={output_ports}, "
        f"pol=te, freq_points={NUM_FREQS}, steps={len(time)}, dx={dx/µm:.4f}um"
    )
    print(f"Workload: grid={grid.permittivity.shape}, voxels={int(np.prod(np.asarray(grid.permittivity).shape)):,}, updates~{int(np.prod(np.asarray(grid.permittivity).shape))*len(time):.3e}")
    save_overview(OUT_DIR / "beamz_crossing_overview.png", np.asarray(grid.permittivity, dtype=float), design, source_plane, {"o1_fwd": fwd_plane, **{f"{p}_cand0": out_planes[p] for p in output_ports}})

    wall_t0 = pytime.perf_counter()
    executed_steps = run_until_decay(sim, [m_fwd, *out_monitors], time, decay_min_time_s)
    wall_s = max(pytime.perf_counter() - wall_t0, 1e-12)
    num_voxels = int(np.prod(np.asarray(grid.permittivity).shape))
    print(
        "Simulation stats: "
        f"steps={executed_steps}, voxels={num_voxels:,}, sim_time={(executed_steps - 1) * dt * 1e15:.2f}fs, "
        f"wall={wall_s:.2f}s, step_rate={executed_steps / wall_s:.2f} steps/s, MCUPS={num_voxels * executed_steps / wall_s / 1e6:.2f}"
    )

    specs = [
        PortSpec(
            name="o1",
            monitor_name="o1_fwd",
            direction=positive_axis(source_direction),
            polarization="te",
            mode_index=0,
            incident_wave=incoming_wave(source_direction),
            scattered_wave=outgoing_wave(source_direction),
        )
    ]
    for port_name in output_ports:
        direction = ports[port_name]["direction"]
        specs.append(
            PortSpec(
                name=port_name,
                monitor_name=f"{port_name}_cand0",
                direction=positive_axis(direction),
                polarization="te",
                mode_index=0,
                incident_wave=incoming_wave(direction),
                scattered_wave=outgoing_wave(direction),
            )
        )
    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=specs,
        output_ports=["o1", "o2", "o3", "o4"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-45.0,
    )
    valid = np.asarray(result["diagnostics"]["valid_mask"], dtype=bool)
    for spec in specs:
        waves = result["diagnostics"]["waves"][spec.name]
        dom = wave_dominance_db(waves["a_plus"], waves["a_minus"], spec.scattered_wave if spec.name != "o1" else spec.incident_wave, valid)
        print(f"{spec.name} wave dominance: {dom:.2f} dB")
    s_matrix = {key: np.asarray(val, dtype=np.complex128) for key, val in result["s_matrix"].items()}
    i0 = int(np.argmin(np.abs(wl_um - WL0 / µm)))
    for port_name in ("o1", "o2", "o3", "o4"):
        mag = abs(s_matrix[(port_name, "o1")][i0])
        print(f"S[{port_name},o1] @ {wl_um[i0]:.4f}um: {20.0 * np.log10(max(mag, 1e-12)):.2f} dB")

    save_sparams(OUT_DIR / "beamz_crossing_sparams.png", wl_um, s_matrix)
    print(f"Saved S-parameter plot: {OUT_DIR / 'beamz_crossing_sparams.png'}")
    print(f"Saved overview plot: {OUT_DIR / 'beamz_crossing_overview.png'}")


if __name__ == "__main__":
    main()
