import matplotlib.pyplot as plt
import numpy as np
import sax

from beamz import *
from beamz.visual.helpers import dxdt

# Parameters
WL0 = 1.55 * µm
N_CORE, N_CLAD = 3.48, 1.44
WAVELENGTHS = np.linspace(1.50 * µm, 1.60 * µm, 41)
DX, DT = dxdt(WL0, n_max=N_CORE, safety_factor=0.999, points_per_wavelength=12, dims=2)
BASE_TIME_MULT = 25
INPUT_EXTENSION = 2.0 * µm
OUTPUT_EXTENSION = 4.0 * µm
Y_MARGIN = 2.0 * µm
PML_BASE = 1.0 * WL0
PML_RIGHT = 1.5 * WL0
SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN = 4.0, 1.0 * µm
MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN = 1.6, 0.8 * µm

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
for port in ports.values():
    cx, cy = port["center"]
    width = float(port["width"])
    outward = -1.0 if port["direction"].startswith("+") else 1.0
    extension = OUTPUT_EXTENSION if outward > 0 else INPUT_EXTENSION
    if port["direction"][1] == "x":
        x0 = cx if outward > 0 else cx - extension
        design_obj += Rectangle(
            position=(x0, cy - width / 2),
            width=extension,
            height=width,
            material=Material(N_CORE**2),
            depth=0,
        )
    else:
        y0 = cy if outward > 0 else cy - extension
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
source_port, output_ports = "o1", ["o1", "o2", "o3"]

def port_line(port, span_factor, span_min):
    cx, cy = port["center"]
    span = max(float(span_min), span_factor * float(port["width"]))
    if port["direction"][1] == "x":
        return (cx, cy - span / 2), (cx, cy + span / 2), span
    return (cx - span / 2, cy), (cx + span / 2, cy), span

max_dist = max(np.hypot(
    ports[source_port]["center"][0] - ports[out]["center"][0],
    ports[source_port]["center"][1] - ports[out]["center"][1],
) for out in output_ports)
TIME = max(
    BASE_TIME_MULT * WL0 / LIGHT_SPEED,
    2.2 * max(N_CORE, N_CLAD) * max_dist / LIGHT_SPEED + 10.0 * WL0 / LIGHT_SPEED,
)
time = np.arange(0, TIME, DT)
signal = ramped_cosine(
    time,
    amplitude=1.0,
    frequency=LIGHT_SPEED / WL0,
    ramp_duration=WL0 * 6 / LIGHT_SPEED,
    t_max=TIME / 3,
)
plot_signal(signal, time)

src = ports[source_port]
_, _, src_span = port_line(src, SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN)
source = ModeSource(
    grid=grid,
    center=src["center"],
    width=src_span,
    wavelength=WL0,
    pol="tm",
    signal=signal,
    direction=src["direction"],
)
monitors = []
for out in output_ports:
    start, end, _ = port_line(ports[out], MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN)
    monitors.append(Monitor(start=start, end=end, name=out, record_fields=True))

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
sim.run(
    animate_live="Ez",
    animation_interval=10,
    axis_scale=[-5e-5, 5e-5],
    cmap="twilight_zero",
    clean_visualization=True,
)

s_sparse = sim.get_S_matrix(
    input_ports=[source_port],
    output_ports=output_ports,
    source_port=source_port,
    frequencies=LIGHT_SPEED / WAVELENGTHS,
    field_component="Ez",
    reduction="mean",
    as_sax=False,
)
s_sax = sax.sdict(s_sparse)
wl_um = LIGHT_SPEED / np.maximum(sim.s_matrix_frequencies, 1e-30) / µm

for key in [("o1", "o1"), ("o2", "o1"), ("o3", "o1")]:
    s_vals = np.asarray(s_sax[key])
    s0 = s_vals[np.argmin(np.abs(wl_um - WL0 / µm))]
    print(f"S[{key[0]},{key[1]}] @ {WL0 / µm:.3f}um: |S|={np.abs(s0):.3f}, phase={np.angle(s0):.3f} rad")

plt.figure(figsize=(7, 4))
for key, color in [(("o1", "o1"), "black"), (("o2", "o1"), "tab:blue"), (("o3", "o1"), "tab:orange")]:
    y_db = 20 * np.log10(np.maximum(np.abs(np.asarray(s_sax[key])), 1e-12))
    plt.plot(wl_um, y_db, color=color, label=rf"$S_{{{key[0][1:]},{key[1][1:]}}}$")
plt.xlabel("wavelength (um)")
plt.ylabel("magnitude (dB)")
plt.title("SAX Splitter Terms (o1 excitation)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
