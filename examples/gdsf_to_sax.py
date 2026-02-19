"""
WARNING: This example likely give the correct ratios, but the absolute loss
is not accounted for. This requires further development of the core tooling.
"""
import matplotlib.pyplot as plt
import numpy as np
import sax
from beamz import *
from beamz.visual.helpers import dxdt

# Parameters
WL0 = 1.55 * µm
N_CORE, N_CLAD = 3.48, 1.44
WL_MIN, WL_MAX = 1.30 * µm, 1.80 * µm
WL_POINTS = 25
POINTS_PER_WAVELENGTH = 9
DX, DT = dxdt(
    WL0,
    n_max=N_CORE,
    safety_factor=0.999,
    points_per_wavelength=POINTS_PER_WAVELENGTH,
    dims=2,
)
INPUT_EXTENSION = 4.0 * µm
OUTPUT_EXTENSION = 4.0 * µm
Y_MARGIN = 3.0 * µm
PML_BASE = 1.0 * WL0
PML_RIGHT = 1.5 * WL0
SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN = 4.0, 1.0 * µm
MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN = 1.6, 0.8 * µm
SOURCE_OFFSET = -1.2 * µm
FORWARD_MONITOR_OFFSET = -0.4 * µm
REFLECTION_MONITOR_BACKOFF = 2.0 * µm
OUTPUT_MONITOR_OFFSET = 0.6 * µm
source_port, output_ports = "o1", ["o2", "o3"]
DIR_VEC = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}

def move_along(center, direction, distance):
    vx, vy = DIR_VEC[direction]
    return center[0] + vx * float(distance), center[1] + vy * float(distance)

def outward_direction(inward_direction):
    return ("-" if inward_direction.startswith("+") else "+") + inward_direction[1]

def snap_to_grid(value):
    return round(float(value) / DX) * DX

# Load the GDS cell with no extra padding, then rebuild a compact domain
imported_design, ports = design.io.gdsf.load(
    "mmi1x2",
    n_core=N_CORE,
    n_clad=N_CLAD,
    layer=(1, 0),
    padding=0.0,
)
design_obj = Design(
    width=imported_design.width + INPUT_EXTENSION + OUTPUT_EXTENSION,
    height=imported_design.height + 2.0 * Y_MARGIN,
    depth=0,
    material=Material(N_CLAD**2),
)
for structure in imported_design.structures[1:]:
    design_obj += structure.copy().shift(INPUT_EXTENSION, Y_MARGIN)
ports = {
    name: {
        **port,
        "center": (port["center"][0] + INPUT_EXTENSION, port["center"][1] + Y_MARGIN),
    }
    for name, port in ports.items()
}

# Add straight waveguide extensions from imported ports to the simulation edges (into PML)
for name, port in ports.items():
    cx, cy = port["center"]
    width = float(port["width"])
    extension = INPUT_EXTENSION if name == source_port else OUTPUT_EXTENSION
    ox, oy = move_along((cx, cy), outward_direction(port["direction"]), extension)
    if port["direction"][1] == "x":
        x0 = min(cx, ox)
        design_obj += Rectangle(
            position=(x0, cy - width / 2),
            width=extension,
            height=width,
            material=Material(N_CORE**2),
            depth=0,
        )
    else:
        y0 = min(cy, oy)
        design_obj += Rectangle(
            position=(cx - width / 2, y0),
            width=width,
            height=extension,
            material=Material(N_CORE**2),
            depth=0,
        )
design_obj.show()

# Rasterize and define source/monitor lines directly at imported port centers
grid = design_obj.rasterize(resolution=DX)
grid.show(field="permittivity")

def port_line(port, span_factor, span_min, offset=0.0):
    cx, cy = move_along(port["center"], port["direction"], offset)
    span = max(float(span_min), span_factor * float(port["width"]))
    if port["direction"][1] == "x":
        return (cx, cy - span / 2), (cx, cy + span / 2), span
    return (cx - span / 2, cy), (cx + span / 2, cy), span

max_dist = max(np.hypot(
    ports[source_port]["center"][0] - ports[out]["center"][0],
    ports[source_port]["center"][1] - ports[out]["center"][1],
) for out in output_ports)
n_eff = max(N_CORE, N_CLAD)
travel_time = 2.2 * n_eff * max_dist / LIGHT_SPEED
src = ports[source_port]
_, _, src_span = port_line(src, SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN)
source_center = move_along(src["center"], src["direction"], SOURCE_OFFSET)
o1_fwd_start, o1_fwd_end, _ = port_line(
    src, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN, offset=FORWARD_MONITOR_OFFSET
)
o1_ref_start, o1_ref_end, _ = port_line(
    src, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN, offset=-REFLECTION_MONITOR_BACKOFF
)

# Place output monitors on a mirrored, grid-snapped pair to avoid sampling asymmetry.
src_y = ports[source_port]["center"][1]
dy = 0.5 * (
    abs(ports[output_ports[0]]["center"][1] - src_y)
    + abs(ports[output_ports[1]]["center"][1] - src_y)
)
x_out = snap_to_grid(
    move_along(ports[output_ports[0]]["center"], ports[output_ports[0]]["direction"], -OUTPUT_MONITOR_OFFSET)[0]
)
span_out = max(
    float(MONITOR_MIN_SPAN),
    MONITOR_SPAN_FACTOR
    * 0.5
    * (float(ports[output_ports[0]]["width"]) + float(ports[output_ports[1]]["width"])),
)
y_out = {
    output_ports[0]: snap_to_grid(src_y + dy),
    output_ports[1]: snap_to_grid(src_y - dy),
}
output_lines = {
    out: ((x_out, y_out[out] - span_out / 2), (x_out, y_out[out] + span_out / 2))
    for out in output_ports
}

wl = np.linspace(WL_MIN, WL_MAX, WL_POINTS)
freqs = LIGHT_SPEED / wl
RAMP_CYCLES = 8
SETTLE_CYCLES = 25
AVG_CYCLES = 12
f0 = LIGHT_SPEED / WL0
ramp_time0 = RAMP_CYCLES / f0
settle_time0 = SETTLE_CYCLES / f0
avg_time0 = AVG_CYCLES / f0
steady_start0 = ramp_time0 + travel_time + settle_time0
time0 = np.arange(0, steady_start0 + avg_time0, DT)
signal0 = ramped_cosine(
    time0,
    amplitude=1.0,
    frequency=f0,
    ramp_duration=ramp_time0,
    t_max=float(time0[-1]) + 2.0 * ramp_time0,
)
plot_signal(signal0, time0)

port_specs = {
    source_port: PortSpec(
        name=source_port,
        monitor_name="o1_ref",
        reference_monitor="o1_fwd",
        direction=ports[source_port]["direction"],
        polarization="tm",
    ),
    "o2": PortSpec(
        name="o2",
        monitor_name="o2",
        direction=ports["o2"]["direction"],
        polarization="tm",
    ),
    "o3": PortSpec(
        name="o3",
        monitor_name="o3",
        direction=ports["o3"]["direction"],
        polarization="tm",
    ),
}

s_sparse = {
    ("o1", "o1"): np.zeros(WL_POINTS, dtype=np.complex128),
    ("o2", "o1"): np.zeros(WL_POINTS, dtype=np.complex128),
    ("o3", "o1"): np.zeros(WL_POINTS, dtype=np.complex128),
}
guided_out_ratio = np.zeros(WL_POINTS, dtype=float)
loss = np.zeros(WL_POINTS, dtype=float)
for i, (wl_i, f_i) in enumerate(zip(wl, freqs)):
    ramp_time = RAMP_CYCLES / f_i
    settle_time = SETTLE_CYCLES / f_i
    avg_time = AVG_CYCLES / f_i
    steady_start = ramp_time + travel_time + settle_time
    time = np.arange(0, steady_start + avg_time, DT)
    signal = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=f_i,
        ramp_duration=ramp_time,
        t_max=float(time[-1]) + 2.0 * ramp_time,
    )
    source = ModeSource(
        grid=grid,
        center=source_center,
        width=src_span,
        wavelength=float(wl_i),
        pol="tm",
        signal=signal,
        direction=src["direction"],
    )
    monitors = [
        Monitor(start=o1_fwd_start, end=o1_fwd_end, name="o1_fwd", record_fields=True),
        Monitor(start=o1_ref_start, end=o1_ref_end, name="o1_ref", record_fields=True),
    ]
    for out in output_ports:
        m_start, m_end = output_lines[out]
        monitors.append(Monitor(start=m_start, end=m_end, name=out, record_fields=True))
    sim = Simulation(
        design=design_obj,
        devices=[source, *monitors],
        boundaries=[
            PML(edges=["left", "top", "bottom"], thickness=PML_BASE),
            PML(edges="right", thickness=PML_RIGHT),
        ],
        time=time,
        resolution=DX,
    )
    print(f"CW sweep {i + 1}/{WL_POINTS} at {wl_i / µm:.4f} um")
    sim.run_fast(progress=False)
    modal_result = sim.get_S_matrix_modal_cw(
        source_port=source_port,
        ports=port_specs,
        output_ports=[source_port, *output_ports],
        frequency=float(f_i),
        steady_start_time=steady_start,
        avg_cycles=AVG_CYCLES,
        mode_strategy="per_frequency",
        as_sax=False,
        return_diagnostics=True,
    )
    s_col = modal_result["s_matrix"]
    s_sparse[("o1", "o1")][i] = s_col[("o1", "o1")]
    s_sparse[("o2", "o1")][i] = s_col[("o2", "o1")]
    s_sparse[("o3", "o1")][i] = s_col[("o3", "o1")]
    guided_out_ratio[i] = modal_result["diagnostics"]["power_sum"]
    loss[i] = modal_result["diagnostics"]["loss_est"]

s_sax = sax.sdict(s_sparse)
wl_um = wl / µm
power_sum = (
    np.abs(np.asarray(s_sax[("o1", "o1")])) ** 2
    + np.abs(np.asarray(s_sax[("o2", "o1")])) ** 2
    + np.abs(np.asarray(s_sax[("o3", "o1")])) ** 2
)
idx0 = int(np.argmin(np.abs(wl_um - WL0 / µm)))
print(f"|S11|^2+|S21|^2+|S31|^2 @ {WL0 / µm:.3f}um: {power_sum[idx0]:.3f}")
print(f"guided_out_ratio @ {WL0 / µm:.3f}um: {guided_out_ratio[idx0]:.3f}")
print(f"loss @ {WL0 / µm:.3f}um: {loss[idx0]:.3f}")

for key in [("o1", "o1"), ("o2", "o1"), ("o3", "o1")]:
    s_vals = np.asarray(s_sax[key])
    s0 = s_vals[np.argmin(np.abs(wl_um - WL0 / µm))]
    print(f"S[{key[0]},{key[1]}] @ {WL0 / µm:.3f}um: |S|={np.abs(s0):.3f}, phase={np.angle(s0):.3f} rad")

plt.figure(figsize=(5, 3), dpi=300)
for key, color in [(("o1", "o1"), "black"), (("o2", "o1"), "tab:blue"), (("o3", "o1"), "tab:orange")]:
    y_db = 20 * np.log10(np.maximum(np.abs(np.asarray(s_sax[key])), 1e-12))
    plt.plot(wl_um, y_db, "o-", linewidth=1.5, ms=2.5, color=color, label=rf"$S_{{{key[0][1:]}{key[1][1:]}}}$")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Magnitude (dB)")
plt.title("GDSFactory MMI1x2")
plt.grid(alpha=0.3)
plt.xlim(WL_MIN / µm, WL_MAX / µm)
plt.ylim(-40, 0)
plt.legend()
plt.tight_layout()
plt.savefig("sax_splitter_terms.png", dpi=300)
plt.show()
