import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamz import *

WL0 = 1.55 * µm
N_CORE, N_CLAD = 3.48, 1.44  # Silicon (Si) core, Silicon Oxide (SiO2) cladding
DX, DT = dxdt(
    WL0, n_max=N_CORE, safety_factor=0.999, points_per_wavelength=12, dims=2
)
WAVELENGTHS = np.linspace(1.50 * µm, 1.60 * µm, 41)
FREQUENCIES = LIGHT_SPEED / WAVELENGTHS
BASE_TIME_MULT = 60
SOURCE_OFFSET = 1.20 * µm
MONITOR_OFFSET = 1.80 * µm
SOURCE_SPAN_FACTOR = 4.0
SOURCE_MIN_SPAN = 1.00 * µm
MONITOR_SPAN_FACTOR = 1.6
MONITOR_MIN_SPAN = 0.80 * µm
FULL_NXN = False  # Set True to excite all ports and build a full square matrix.


def _show_plot():
    backend = str(plt.get_backend()).lower()
    if "agg" in backend:
        plt.close()
    else:
        plt.show()


def _inward_offset(port, distance):
    """Move a point inward from a port along BeamZ direction."""
    cx, cy = port["center"]
    direction = port["direction"]
    sign = 1.0 if direction.startswith("+") else -1.0
    axis = direction[1]
    if axis == "x":
        return cx + sign * distance, cy
    return cx, cy + sign * distance


def _line_monitor_for_port(port, offset, span_factor, min_span):
    """Create a monitor line orthogonal to the local propagation axis."""
    cx, cy = _inward_offset(port, offset)
    span = max(float(min_span), float(span_factor) * float(port["width"]))
    axis = port["direction"][1]
    if axis == "x":
        start = (cx, cy - span / 2)
        end = (cx, cy + span / 2)
    else:
        start = (cx - span / 2, cy)
        end = (cx + span / 2, cy)
    return start, end, span


def _estimate_required_time(ports_dict, source_ports, output_ports, n_eff=2.0):
    """Estimate a runtime long enough for the pulse to reach all requested monitors."""
    max_distance = 0.0
    for src_name in source_ports:
        src_center = _inward_offset(ports_dict[src_name], SOURCE_OFFSET)
        for out_name in output_ports:
            out_center = _inward_offset(ports_dict[out_name], MONITOR_OFFSET)
            max_distance = max(max_distance, float(np.hypot(
                out_center[0] - src_center[0], out_center[1] - src_center[1]
            )))

    propagation_time = n_eff * max_distance / LIGHT_SPEED
    return max(
        BASE_TIME_MULT * WL0 / LIGHT_SPEED,
        2.2 * propagation_time + 10.0 * WL0 / LIGHT_SPEED,
    )


def _plot_design_and_ports(design_obj, ports_dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Imported gdsfactory Cell (BeamZ Geometry)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_xlim(0, design_obj.width / µm)
    ax.set_ylim(0, design_obj.height / µm)
    ax.set_aspect("equal", adjustable="box")

    for structure in design_obj.structures[1:]:
        if not hasattr(structure, "vertices") or not structure.vertices:
            continue
        verts = np.asarray(structure.vertices)
        ax.fill(
            verts[:, 0] / µm,
            verts[:, 1] / µm,
            color="tab:blue",
            alpha=0.35,
            edgecolor="tab:blue",
            linewidth=0.8,
        )

    for name, port in ports_dict.items():
        px, py = port["center"]
        ax.scatter(px / µm, py / µm, s=45, color="crimson", zorder=3)
        ax.text(px / µm, py / µm + 0.12, f"{name} ({port['direction']})", ha="center")

    plt.tight_layout()
    _show_plot()


def _plot_rasterized_permittivity(grid_obj, design_obj):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Rasterized Permittivity Grid")
    im = ax.imshow(
        np.asarray(grid_obj.permittivity),
        origin="lower",
        extent=[0, design_obj.width / µm, 0, design_obj.height / µm],
        cmap="viridis",
        aspect="auto",
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    fig.colorbar(im, ax=ax, label="epsilon_r")
    plt.tight_layout()
    _show_plot()


def _plot_source_signal(time_values, signal_values):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_title("Injected Ramped-Cosine Source Signal")
    ax.plot(time_values * 1e15, signal_values, lw=1.6, color="tab:orange")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("amplitude")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    _show_plot()


def _plot_run_setup(design_obj, grid_obj, ports_dict, source_port_name, source_center, monitors):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Simulation Setup for Excitation at {source_port_name}")
    ax.imshow(
        np.asarray(grid_obj.permittivity),
        origin="lower",
        extent=[0, design_obj.width / µm, 0, design_obj.height / µm],
        cmap="Greys",
        alpha=0.5,
        aspect="auto",
    )
    sx, sy = source_center
    ax.scatter(sx / µm, sy / µm, marker="*", s=160, color="gold", edgecolor="black")
    ax.text(sx / µm, sy / µm + 0.16, f"source: {source_port_name}", ha="center")

    for mon in monitors:
        x0, y0 = mon.start
        x1, y1 = mon.end
        ax.plot([x0 / µm, x1 / µm], [y0 / µm, y1 / µm], lw=2.0, label=f"mon {mon.name}")

    for port_name, port in ports_dict.items():
        px, py = port["center"]
        ax.scatter(px / µm, py / µm, s=30, color="crimson")
        ax.text(px / µm, py / µm - 0.16, port_name, ha="center", fontsize=9)

    ax.set_xlim(0, design_obj.width / µm)
    ax.set_ylim(0, design_obj.height / µm)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    _show_plot()


def _monitor_trace_and_spectrum(monitor, dt):
    field = np.asarray(monitor.fields["Ez"])
    trace = field.mean(axis=1) if field.ndim > 1 else field
    freqs = np.fft.rfftfreq(len(trace), d=dt)
    spectrum = np.fft.rfft(trace)
    return trace, freqs, spectrum


def _plot_monitor_results(monitors, dt, source_port_name):
    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 4))
    ax_t.set_title(f"Monitor Traces ({source_port_name} excited)")
    ax_f.set_title("Monitor Spectra |FFT(Ez)|")

    for mon in monitors:
        trace, freqs, spectrum = _monitor_trace_and_spectrum(mon, dt)
        t_axis = np.arange(len(trace)) * dt * 1e15
        ax_t.plot(t_axis, trace, label=mon.name)
        ax_f.plot(freqs * 1e-12, np.abs(spectrum), label=mon.name)

    ax_t.set_xlabel("time (fs)")
    ax_t.set_ylabel("mean Ez")
    ax_t.grid(alpha=0.25)
    ax_t.legend()
    ax_f.set_xlabel("frequency (THz)")
    ax_f.set_ylabel("|FFT|")
    ax_f.grid(alpha=0.25)
    ax_f.legend()
    plt.tight_layout()
    _show_plot()


def _plot_final_smatrix(s_sax_dict, wavelengths, output_ports, input_ports, center_idx):
    matrix = np.zeros((len(output_ports), len(input_ports)))
    for i, p_out in enumerate(output_ports):
        for j, p_in in enumerate(input_ports):
            key = (p_out, p_in)
            if key in s_sax_dict:
                matrix[i, j] = np.abs(np.asarray(s_sax_dict[key])[center_idx])

    fig, (ax_hm, ax_sp) = plt.subplots(1, 2, figsize=(12, 4))
    ax_hm.set_title(f"|S| at {wavelengths[center_idx] / µm:.3f} um")
    im = ax_hm.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(1.0, np.max(matrix)))
    ax_hm.set_xticks(np.arange(len(input_ports)))
    ax_hm.set_yticks(np.arange(len(output_ports)))
    ax_hm.set_xticklabels(input_ports)
    ax_hm.set_yticklabels(output_ports)
    ax_hm.set_xlabel("input port")
    ax_hm.set_ylabel("output port")
    fig.colorbar(im, ax=ax_hm, label="|S|")

    ax_sp.set_title("Broadband S-parameter Magnitudes")
    for p_out in output_ports:
        for p_in in input_ports:
            key = (p_out, p_in)
            if key in s_sax_dict:
                ax_sp.plot(
                    wavelengths / µm,
                    np.abs(np.asarray(s_sax_dict[key])),
                    label=f"S[{p_out},{p_in}]",
                )
    ax_sp.set_xlabel("wavelength (um)")
    ax_sp.set_ylabel("|S|")
    ax_sp.grid(alpha=0.25)
    ax_sp.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    _show_plot()


device_design, ports = design.io.gdsf.load(
    "mmi1x2",
    n_core=N_CORE,
    n_clad=N_CLAD,
    layer=(1, 0),
    padding=3.0,
)
_plot_design_and_ports(device_design, ports)

grid = device_design.rasterize(resolution=DX)
_plot_rasterized_permittivity(grid, device_design)

port_names = sorted(ports.keys())
excited_ports = port_names if FULL_NXN else ["o1"]
TIME = _estimate_required_time(
    ports_dict=ports,
    source_ports=excited_ports,
    output_ports=port_names,
    n_eff=max(N_CORE, N_CLAD),
)
print(f"Using simulation time: {TIME * 1e15:.1f} fs")

time = np.arange(0, TIME, DT)
signal = ramped_cosine(
    time,
    amplitude=1.0,
    frequency=LIGHT_SPEED / WL0,
    ramp_duration=WL0 * 6 / LIGHT_SPEED,
    t_max=TIME * 0.30,
)
_plot_source_signal(time, signal)

s_full = {}
s_power = {}

for source_port in excited_ports:
    source_meta = ports[source_port]
    src_center = _inward_offset(source_meta, SOURCE_OFFSET)
    _monitor_start, _monitor_end, source_span = _line_monitor_for_port(
        source_meta, SOURCE_OFFSET, SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN
    )
    source = ModeSource(
        grid=grid,
        center=src_center,
        width=source_span,
        wavelength=WL0,
        pol="tm",
        signal=signal,
        direction=source_meta["direction"],
    )

    monitors = []
    for port_name in port_names:
        start, end, _ = _line_monitor_for_port(
            ports[port_name], MONITOR_OFFSET, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN
        )
        monitors.append(
            Monitor(start=start, end=end, name=port_name, record_fields=True)
        )

    _plot_run_setup(
        device_design,
        grid,
        ports,
        source_port_name=source_port,
        source_center=src_center,
        monitors=monitors,
    )

    sim = Simulation(
        design=device_design,
        devices=[source, *monitors],
        boundaries=[PML(edges="all", thickness=1.0 * WL0)],
        time=time,
        resolution=DX,
    )
    sim.run()
    _plot_monitor_results(monitors, DT, source_port_name=source_port)

    energy_by_port = {
        mon.name: float(np.sum(np.asarray(mon.power_history)) * DT) for mon in monitors
    }
    src_energy = max(energy_by_port.get(source_port, 0.0), 1e-30)
    for out_name in port_names:
        s_power[(out_name, source_port)] = energy_by_port.get(out_name, 0.0) / src_energy

    s_column = sim.get_S_matrix(
        input_ports=port_names,
        output_ports=port_names,
        source_port=source_port,
        frequencies=FREQUENCIES,
        field_component="Ez",
        reduction="mean",
        as_sax=False,
    )
    s_full.update(s_column)

s_sax = sax.sdict(s_full)
center_idx = int(np.argmin(np.abs(WAVELENGTHS - WL0)))
print(f"Computed SAX S-matrix entries: {len(s_sax)} at {len(WAVELENGTHS)} wavelengths")
for to_port, from_port in sorted(s_sax.keys()):
    value = np.asarray(s_sax[(to_port, from_port)])[center_idx]
    power_est = float(np.abs(value) ** 2)
    power_ratio = s_power.get((to_port, from_port), np.nan)
    print(
        f"S[{to_port},{from_port}] @ {WL0 / µm:.3f}um: "
        f"|S|={np.abs(value):.3f}, |S|^2={power_est:.3f}, "
        f"P-ratio={power_ratio:.3f}, phase={np.angle(value):.3f} rad"
    )

_plot_final_smatrix(
    s_sax,
    WAVELENGTHS,
    output_ports=port_names,
    input_ports=excited_ports,
    center_idx=center_idx,
)
