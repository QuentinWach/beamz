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
    ModeSource,
    Monitor,
    PML,
    PortSpec,
    Simulation,
    dxdt,
    µm,
)
from beamz.design.io import gdsf
from beamz.devices.sources.signals import gaussian_band_pulse

# Fixed example hyperparameters. Edit these directly when using the example as a
# starting point for another component or sweep.
OUT_DIR = Path("benchmarks/results/tiny_beamz_crossing")
COMPONENT_NAME = "ebeam_crossing4"
NUM_FREQS = 51
PPW = 10
WL0, WL_MIN, WL_MAX = 1550.0e-9, 1530.0e-9, 1570.0e-9
N_CORE, N_CLAD = 3.47, 1.44
LAYER = (1, 0)
CORE_T = 0.22 * µm
CLAD_BELOW = 0.50 * µm
CLAD_ABOVE = 0.50 * µm
PML_XY, PML_Z = 1.5 * µm, 1.0 * µm
XY_MARGIN = 0.50 * µm
Z_PADDING = 1.10 * µm
EXTENSION = 2.50 * µm
PORT_OVERLAP = 0.10 * µm
PORT_MARGIN = 0.50 * µm
SOURCE_OFFSET = 0.10 * µm
DISTANCE_SOURCE_TO_MONITORS = 0.20 * µm
OUTPUT_MONITOR_OFFSETS = {"o2": 0.70 * µm, "o3": 0.10 * µm, "o4": 0.70 * µm}
RUN_AFTER_SOURCES_UOC = 90.0

def positive_axis_direction(direction: str) -> str:
    return "+" + str(direction)[1:]


def incoming_wave(direction: str) -> str:
    return "plus" if str(direction).startswith("+") else "minus"


def outgoing_wave(direction: str) -> str:
    return "minus" if str(direction).startswith("+") else "plus"


def move_along(center: tuple[float, float], direction: str, distance: float):
    x, y = center
    return {
        "+x": (x + distance, y),
        "-x": (x - distance, y),
        "+y": (x, y + distance),
        "-y": (x, y - distance),
    }[str(direction)]


def port_plane(
    port: dict,
    *,
    span: float,
    z_span: float,
    z_center: float,
    offset: float = 0.0,
):
    cx, cy = move_along(port["center"], port["direction"], offset)
    z0 = float(z_center) - 0.5 * float(z_span)
    z1 = float(z_center) + 0.5 * float(z_span)
    if str(port["direction"]).endswith("x"):
        return (cx, cy - 0.5 * float(span), z0), (cx, cy + 0.5 * float(span), z1)
    return (cx - 0.5 * float(span), cy, z0), (cx + 0.5 * float(span), cy, z1)


def line_center(line):
    a, b = line
    return tuple(0.5 * (float(a[i]) + float(b[i])) for i in range(len(a)))

def plot_simulation_overview(
    out_path: Path,
    eps_grid: np.ndarray,
    *,
    width: float,
    height: float,
    depth: float,
    z_focus: float,
    source_plane,
    monitor_planes,
):
    eps_grid = np.asarray(eps_grid, dtype=float)
    if eps_grid.ndim == 3:
        z_idx = int(np.clip(round((z_focus / max(depth, 1e-30)) * (eps_grid.shape[0] - 1)), 0, eps_grid.shape[0] - 1))
        eps_view = eps_grid[z_idx]
    else:
        eps_view = eps_grid

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=220)
    im = ax.imshow(
        eps_view,
        origin="lower",
        extent=[0.0, width / µm, 0.0, height / µm],
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="Permittivity", fraction=0.046, pad=0.04)

    def _plot_plane(line, label, color):
        (x0, y0, _), (x1, y1, _) = line
        ax.plot([x0 / µm, x1 / µm], [y0 / µm, y1 / µm], color=color, lw=2.0, label=label)

    _plot_plane(source_plane, "source", "white")
    for name, plane in monitor_planes.items():
        _plot_plane(plane, name, "tab:red")

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Simulation overview")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_sparameters_db(out_path: Path, wl_um: np.ndarray, s_matrix: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=220)
    for (out_port, in_port), values in sorted(s_matrix.items()):
        arr = np.asarray(values, dtype=np.complex128)
        ax.plot(wl_um, 20.0 * np.log10(np.maximum(np.abs(arr), 1e-12)), lw=2.0, label=f"S[{out_port},{in_port}]")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("S-parameters")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def wave_dominance_db(a_plus: np.ndarray, a_minus: np.ndarray, selector: str, mask: np.ndarray) -> float:
    # Report how cleanly a monitor separates the selected traveling wave from
    # the opposite-going component.
    sel = np.asarray(a_plus if selector == "plus" else a_minus, dtype=np.complex128)
    opp = np.asarray(a_minus if selector == "plus" else a_plus, dtype=np.complex128)
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid): return float("nan")
    p_sel = float(np.mean(np.abs(sel[valid]) ** 2))
    p_opp = float(np.mean(np.abs(opp[valid]) ** 2))
    return 10.0 * np.log10(max(p_sel, 1e-18) / max(p_opp, 1e-18))


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m"


def expected_mode_components(axis: str, pol: str) -> tuple[str, str]:
    pol_key = str(pol).lower()
    if pol_key == "te":
        mapping = {"x": ("Ey", "Hz"), "y": ("Ex", "Hz"), "z": ("Ex", "Hy")}
    else:
        mapping = {"x": ("Ez", "Hy"), "y": ("Ez", "Hx"), "z": ("Ey", "Hx")}
    return mapping[str(axis)]


def save_mode_profile_plot(
    *,
    label: str,
    mode_src: ModeSource,
    grid_eps: np.ndarray,
    dx: float,
    out_path: Path,
) -> None:
    axis = mode_src.direction[1]
    e_expected, h_expected = expected_mode_components(axis, mode_src.pol)
    eps2d = np.asarray(getattr(mode_src, "_eps_profile_2d", np.array([])))
    if eps2d.ndim == 2 and eps2d.size > 0:
        profile_map = {
            "Ex": getattr(mode_src, "_Ex_profile", None),
            "Ey": getattr(mode_src, "_Ey_profile", None),
            "Ez": getattr(mode_src, "_Ez_profile", None),
            "Hx": getattr(mode_src, "_Hx_profile", None),
            "Hy": getattr(mode_src, "_Hy_profile", None),
            "Hz": getattr(mode_src, "_Hz_profile", None),
        }
        fig, ax = plt.subplots(2, 4, figsize=(10.8, 5.4), dpi=250)
        ax = ax.ravel()
        im_eps = ax[0].imshow(eps2d, origin="lower", cmap="viridis", aspect="equal")
        ax[0].set_title(f"{label}: eps")
        fig.colorbar(im_eps, ax=ax[0], fraction=0.046, pad=0.04)
        for i, name in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], start=1):
            arr = profile_map[name]
            if arr is None:
                ax[i].axis("off")
                continue
            a2 = np.asarray(arr).squeeze()
            if a2.ndim != 2:
                a2 = np.atleast_2d(a2)
            im = ax[i].imshow(np.abs(a2), origin="lower", cmap="magma", aspect="equal")
            ax[i].set_title(f"{label}: |{name}|")
            fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        ax[7].axis("off")
        ax[7].text(
            0.02,
            0.95,
            (
                f"pol={mode_src.pol}\n"
                f"dir={mode_src.direction}\n"
                f"axis={axis}\n"
                f"expected E/H={e_expected}/{h_expected}\n"
                f"neff={float(np.real(getattr(mode_src, '_neff', np.nan))):.5f}\n"
                f"width={float(mode_src.width)/µm:.3f}um\n"
                f"height={float(getattr(mode_src, 'height', 0.0) or 0.0)/µm:.3f}um"
            ),
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    if grid_eps.ndim == 3:
        zc = int(np.clip(round(float(mode_src.center[2]) / dx), 0, grid_eps.shape[0] - 1))
        yc = int(np.clip(round(float(mode_src.center[1]) / dx), 0, grid_eps.shape[1] - 1))
        xc = int(np.clip(round(float(mode_src.center[0]) / dx), 0, grid_eps.shape[2] - 1))
        eps_profile = np.asarray(grid_eps[zc, :, xc] if axis == "x" else grid_eps[zc, yc, :], dtype=float)
    else:
        if axis == "x":
            x_idx = int(np.clip(round(float(mode_src.center[0]) / dx), 0, grid_eps.shape[1] - 1))
            eps_profile = np.asarray(grid_eps[:, x_idx], dtype=float)
        else:
            y_idx = int(np.clip(round(float(mode_src.center[1]) / dx), 0, grid_eps.shape[0] - 1))
            eps_profile = np.asarray(grid_eps[y_idx, :], dtype=float)

    profiles = {
        "jz": getattr(mode_src, "_jz_profile", None),
        "jy": getattr(mode_src, "_jy_profile", None),
        "jx": getattr(mode_src, "_jx_profile", None),
        "my": getattr(mode_src, "_my_profile", None),
        "mz": getattr(mode_src, "_mz_profile", None),
    }
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.2, 2.8), dpi=260)
    u = np.arange(eps_profile.size, dtype=float) * dx / µm
    ax0.plot(u, eps_profile, color="tab:blue", lw=1.6)
    ax0.set_title(f"{label}: eps profile")
    ax0.set_xlabel("transverse coordinate (um)")
    ax0.set_ylabel("permittivity")
    ax0.grid(alpha=0.3)

    plotted = False
    for name, arr in profiles.items():
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.complex128).reshape(-1)
        if a.size == 0:
            continue
        uu = np.arange(a.size, dtype=float) * dx / µm
        ax1.plot(uu, np.abs(a), lw=1.5, label=f"|{name}|")
        plotted = True
    if not plotted:
        ax1.text(0.02, 0.7, "No mode profile data", transform=ax1.transAxes)
    ax1.set_title(
        f"{label}: mode profile ({mode_src.pol}, {mode_src.direction})\n"
        f"expected {e_expected}/{h_expected}, neff={float(np.real(getattr(mode_src, '_neff', np.nan))):.4f}"
    )
    ax1.set_xlabel("transverse coordinate (um)")
    ax1.set_ylabel("normalized magnitude")
    ax1.grid(alpha=0.3)
    if plotted:
        ax1.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# 1. Import the GDSFactory/PDK component, extrude it to 3D, pad the domain,
# and extend the ports into uniform straight sections.
OUT_DIR.mkdir(parents=True, exist_ok=True)
try:
    prepared = gdsf.prepare_component(
        COMPONENT_NAME,
        layer=LAYER,
        n_core=N_CORE,
        n_clad=N_CLAD,
        core_thickness=CORE_T,
        clad_below=CLAD_BELOW,
        clad_above=CLAD_ABOVE,
        xy_padding=EXTENSION,
        z_padding=Z_PADDING,
        extension=EXTENSION,
        port_overlap=PORT_OVERLAP,
    )
except ValueError as exc:
    print(
        f"{exc} Install or activate the matching gdsfactory PDK before running "
        f"{Path(__file__).name}."
    )
    raise SystemExit(0) from exc
component_label, design, ports = prepared["component_label"], prepared["design"], prepared["ports"]
source_port, output_ports = "o1", ["o2", "o3", "o4"]
dx, dt = dxdt(WL0, n_max=N_CORE, dims=3, safety_factor=0.999, points_per_wavelength=PPW)
grid = design.rasterize(resolution=dx)
freqs = np.linspace(LIGHT_SPEED / WL_MAX, LIGHT_SPEED / WL_MIN, NUM_FREQS, dtype=np.float32)
wl_um = LIGHT_SPEED / freqs / µm

# 2. Place the source and monitors directly from the imported port metadata.
# BeamZ's current raw port extraction is most stable with the source on the
# source port and fixed monitor planes on the device side of the ports:
#   - source at 0.10 um inward from o1
#   - source monitor at 0.30 um inward from o1
#   - weak ports o2/o4 at 0.70 um inward
#   - through port o3 at 0.10 um inward
src = ports[source_port]
source_direction = src["direction"]
span = max(float(src["width"]) + 2.0 * PORT_MARGIN, float(src["width"]) + 0.1 * µm)
z_center = float(src["z_center"])
z_span = CLAD_BELOW + CORE_T + CLAD_ABOVE
source_plane = port_plane(
    src,
    span=span,
    z_span=z_span,
    z_center=z_center,
    offset=SOURCE_OFFSET,
)
fwd_plane = port_plane(
    src,
    span=span,
    z_span=z_span,
    z_center=z_center,
    offset=SOURCE_OFFSET + DISTANCE_SOURCE_TO_MONITORS,
)
source_center = line_center(source_plane)
out_planes = {
    port_name: port_plane(
        ports[port_name],
        span=span,
        z_span=z_span,
        z_center=z_center,
        offset=OUTPUT_MONITOR_OFFSETS[port_name],
    )
    for port_name in output_ports
}
print("Plane positions (um inward from imported ports):")
print(f"  o1_fwd: {(SOURCE_OFFSET + DISTANCE_SOURCE_TO_MONITORS) / µm:.2f}")
print(f"  source: {SOURCE_OFFSET / µm:.2f}")
for port_name in output_ports:
    print(f"  {port_name}: {OUTPUT_MONITOR_OFFSETS[port_name] / µm:.2f}")
runtime_output_distance_um = 0.0
for port_name in output_ports:
    c_out = line_center(out_planes[port_name])
    runtime_output_distance_um = max(
        runtime_output_distance_um,
        float(np.hypot(c_out[0] - source_center[0], c_out[1] - source_center[1])) / µm,
    )

# 3. Generate the broadband Gaussian pulse and build the source / DFT monitors.
pulse = gaussian_band_pulse(
    freqs,
    carrier_frequency=LIGHT_SPEED / WL0,
    dt=dt,
    run_after_sources_uoc=RUN_AFTER_SOURCES_UOC,
    max_output_distance_um=runtime_output_distance_um,
)
source = ModeSource(
    grid=grid,
    center=source_center,
    width=span,
    height=z_span,
    wavelength=WL0,
    pol="te",
    signal=pulse.signal,
    direction=source_direction,
)
source.initialize(grid.permittivity, dx)
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
    Monitor(
        start=out_planes[p][0],
        end=out_planes[p][1],
        name=p,
        **monitor_cfg,
    )
    for p in output_ports
]
decay_monitors = [m_fwd, *out_monitors]

# Create one diagnostic modal basis plot per source/monitor location before
# time stepping so monitor placement issues are visible immediately.
save_mode_profile_plot(
    label="source_o1",
    mode_src=source,
    grid_eps=np.asarray(grid.permittivity),
    dx=dx,
    out_path=OUT_DIR / "beamz_crossing_mode_source_o1.png",
)
mode_plot_paths = [OUT_DIR / "beamz_crossing_mode_source_o1.png"]
for port_name in output_ports:
    plane_center = line_center(out_planes[port_name])
    mode_probe = ModeSource(
        grid=grid,
        center=plane_center,
        width=span,
        height=z_span,
        wavelength=WL0,
        pol="te",
        signal=np.zeros((1,), dtype=np.float32),
        direction=gdsf.outward_direction(ports[port_name]["direction"]),
    )
    mode_probe.initialize(grid.permittivity, dx)
    out_path = OUT_DIR / f"beamz_crossing_mode_{port_name}.png"
    save_mode_profile_plot(
        label=f"monitor_{port_name}",
        mode_src=mode_probe,
        grid_eps=np.asarray(grid.permittivity),
        dx=dx,
        out_path=out_path,
    )
    mode_plot_paths.append(out_path)
print("Saved mode profile plots:")
for path in mode_plot_paths:
    print(f"  - {path}")

# 4. Feed the design, source, monitors, boundaries, and time array into the
# simulation object.
sim = Simulation(
    design=design,
    sources=[source],
    monitors=[m_fwd, *out_monitors],
    boundaries=[
        PML(edges=["left", "right", "top", "bottom"], thickness=PML_XY), 
        PML(edges=["front", "back"], thickness=PML_Z)],
    time=pulse.time,
    resolution=dx,
)
sim.show()

# 5. Save a compact overview plot of the rasterized structure with the source
# and monitor planes overlaid.
print(f"Workload: grid={grid.permittivity.shape}, voxels={int(np.prod(np.asarray(grid.permittivity).shape)):,}, updates~{int(np.prod(np.asarray(grid.permittivity).shape))*len(pulse.time):.3e}")
estimated_updates = float(int(np.prod(np.asarray(grid.permittivity).shape)) * len(pulse.time))
print(
    "Estimated runtime: "
    f"100 MCUPS ~ {format_duration(estimated_updates / 100e6)}, "
    f"250 MCUPS ~ {format_duration(estimated_updates / 250e6)}, "
    f"500 MCUPS ~ {format_duration(estimated_updates / 500e6)}"
)
plot_simulation_overview(
    OUT_DIR / "beamz_crossing_overview.png",
    np.asarray(grid.permittivity, dtype=float),
    width=design.width,
    height=design.height,
    depth=design.depth,
    z_focus=Z_PADDING + CLAD_BELOW + 0.5 * CORE_T,
    source_plane=source_plane,
    monitor_planes={
        "o1_fwd": fwd_plane,
        **out_planes,
    },
)

# 6. Run in compiled chunks until the monitor power has decayed sufficiently
# after the pulse leaves the device.
wall_t0 = pytime.perf_counter()
executed_steps = sim.run_compiled_until_decay(
    decay_monitors,
    min_time_s=pulse.source_end_time + pulse.tail_time,
    progress=True,
)
wall_s = max(pytime.perf_counter() - wall_t0, 1e-12)
num_voxels = int(np.prod(np.asarray(grid.permittivity).shape))
print(
    "Simulation stats: "
    f"steps={executed_steps}, voxels={num_voxels:,}, sim_time={(executed_steps - 1) * dt * 1e15:.2f}fs, "
    f"wall={wall_s:.2f}s, step_rate={executed_steps / wall_s:.2f} steps/s, MCUPS={num_voxels * executed_steps / wall_s / 1e6:.2f}"
)

# 7. Define one modal port per monitor plane and extract the broadband S-matrix
# directly from the in-simulation DFT accumulators.
source_spec = PortSpec(
    name="o1",
    monitor_name="o1_fwd",
    direction=positive_axis_direction(source_direction),
    polarization="te",
    mode_index=0,
    incident_wave=incoming_wave(source_direction),
    scattered_wave=outgoing_wave(source_direction),
)
selected_specs = [source_spec]
for port_name in output_ports:
    direction = ports[port_name]["direction"]
    selected_specs.append(
        PortSpec(
            name=port_name,
            monitor_name=port_name,
            direction=positive_axis_direction(direction),
            polarization="te",
            mode_index=0,
            incident_wave=incoming_wave(direction),
            scattered_wave=outgoing_wave(direction),
        )
    )
result = sim.get_S_matrix_modal_dft(
    source_port="o1",
    ports=selected_specs,
    output_ports=["o1", *output_ports],
    frequencies=freqs,
    as_sax=False,
    return_diagnostics=True,
    min_incident_db=-45.0,
)
valid = np.asarray(result["diagnostics"]["valid_mask"], dtype=bool)
source_waves = result["diagnostics"]["waves"]["o1"]
source_dom = wave_dominance_db(source_waves["a_plus"], source_waves["a_minus"], source_spec.incident_wave, valid)
print(f"o1 wave dominance: {source_dom:.2f} dB")

selected_monitor_planes = {"o1_fwd": fwd_plane}
selected_s = {("o1", "o1"): np.asarray(result["s_matrix"][("o1", "o1")], dtype=np.complex128)}
for port_name in output_ports:
    waves = result["diagnostics"]["waves"][port_name]
    dom = wave_dominance_db(
        waves["a_plus"],
        waves["a_minus"],
        outgoing_wave(ports[port_name]["direction"]),
        valid,
    )
    selected_monitor_planes[port_name] = out_planes[port_name]
    selected_s[(port_name, "o1")] = np.asarray(
        result["s_matrix"][(port_name, "o1")], dtype=np.complex128
    )
    print(
        f"{port_name} fixed plane at {OUTPUT_MONITOR_OFFSETS[port_name] / µm:.2f} um inward "
        f"(dominance={dom:.2f} dB)"
    )
s_matrix = selected_s
i0 = int(np.argmin(np.abs(wl_um - WL0 / µm)))
for port_name in ("o1", "o2", "o3", "o4"):
    mag = abs(s_matrix[(port_name, "o1")][i0])
    print(f"S[{port_name},o1] @ {wl_um[i0]:.4f}um: {20.0 * np.log10(max(mag, 1e-12)):.2f} dB")

# Overwrite the overview with the selected monitor planes so the saved figure
# matches the final S-matrix extraction path.
plot_simulation_overview(
    OUT_DIR / "beamz_crossing_overview.png",
    np.asarray(grid.permittivity, dtype=float),
    width=design.width,
    height=design.height,
    depth=design.depth,
    z_focus=Z_PADDING + CLAD_BELOW + 0.5 * CORE_T,
    source_plane=source_plane,
    monitor_planes=selected_monitor_planes,
)

# 8. Save the final S-parameter plot using the same helper style as the full
# example so regression checks remain straightforward.
plot_sparameters_db(OUT_DIR / "beamz_crossing_sparams.png", wl_um, s_matrix)
