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
from beamz.visual.example_plots import plot_simulation_overview, plot_sparameters_db

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
PML_XY, PML_Z = 1.0 * µm, 1.0 * µm
XY_MARGIN = 0.50 * µm
Z_PADDING = 1.10 * µm
EXTENSION = 1.50 * µm
PORT_OVERLAP = 0.10 * µm
PORT_MARGIN = 0.50 * µm
SOURCE_OFFSET = 0.10 * µm
MONITOR_OFFSET = 0.30 * µm
RUN_AFTER_SOURCES_UOC = 90.0


def wave_dominance_db(a_plus: np.ndarray, a_minus: np.ndarray, selector: str, mask: np.ndarray) -> float:
    sel = np.asarray(a_plus if selector == "plus" else a_minus, dtype=np.complex128)
    opp = np.asarray(a_minus if selector == "plus" else a_plus, dtype=np.complex128)
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    p_sel = float(np.mean(np.abs(sel[valid]) ** 2))
    p_opp = float(np.mean(np.abs(opp[valid]) ** 2))
    return 10.0 * np.log10(max(p_sel, 1e-18) / max(p_opp, 1e-18))


OUT_DIR.mkdir(parents=True, exist_ok=True)
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
component_label, design, ports = prepared["component_label"], prepared["design"], prepared["ports"]
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
source_plane = gdsf.port_plane(src, span=span, z_span=z_span, z_center=z_center, offset=SOURCE_OFFSET)
fwd_plane = gdsf.port_plane(src, span=span, z_span=z_span, z_center=z_center, offset=MONITOR_OFFSET)
source_center = gdsf.line_center(source_plane)
out_planes = {}
max_output_distance_um = 0.0
for port_name in output_ports:
    out_port = {**ports[port_name], "direction": gdsf.outward_direction(ports[port_name]["direction"])}
    plane = gdsf.port_plane(out_port, span=span, z_span=z_span, z_center=z_center, offset=SOURCE_OFFSET)
    out_planes[port_name] = plane
    c_out = gdsf.line_center(plane)
    max_output_distance_um = max(max_output_distance_um, float(np.hypot(c_out[0] - source_center[0], c_out[1] - source_center[1])) / µm)

pulse = gaussian_band_pulse(
    freqs,
    carrier_frequency=LIGHT_SPEED / WL0,
    dt=dt,
    run_after_sources_uoc=RUN_AFTER_SOURCES_UOC,
    max_output_distance_um=max_output_distance_um,
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
    time=pulse.time,
    resolution=dx,
)

print(f"Workload: grid={grid.permittivity.shape}, voxels={int(np.prod(np.asarray(grid.permittivity).shape)):,}, updates~{int(np.prod(np.asarray(grid.permittivity).shape))*len(pulse.time):.3e}")
plot_simulation_overview(
    OUT_DIR / "beamz_crossing_overview.png",
    np.asarray(grid.permittivity, dtype=float),
    width=design.width,
    height=design.height,
    depth=design.depth,
    z_focus=Z_PADDING + CLAD_BELOW + 0.5 * CORE_T,
    source_plane=source_plane,
    monitor_planes={"o1_fwd": fwd_plane, **{f"{p}_cand0": out_planes[p] for p in output_ports}},
)

wall_t0 = pytime.perf_counter()
executed_steps = sim.run_compiled_until_decay(
    [m_fwd, *out_monitors],
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

specs = [
    PortSpec(
        name="o1",
        monitor_name="o1_fwd",
        direction=gdsf.positive_axis_direction(source_direction),
        polarization="te",
        mode_index=0,
        incident_wave=gdsf.incoming_wave(source_direction),
        scattered_wave=gdsf.outgoing_wave(source_direction),
    )
]
for port_name in output_ports:
    direction = ports[port_name]["direction"]
    specs.append(
        PortSpec(
            name=port_name,
            monitor_name=f"{port_name}_cand0",
            direction=gdsf.positive_axis_direction(direction),
            polarization="te",
            mode_index=0,
            incident_wave=gdsf.incoming_wave(direction),
            scattered_wave=gdsf.outgoing_wave(direction),
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

plot_sparameters_db(OUT_DIR / "beamz_crossing_sparams.png", wl_um, s_matrix)