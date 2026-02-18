import matplotlib.pyplot as plt
import numpy as np
import sax
from beamz.visual.helpers import dxdt
from beamz import *

# Parameters
WL0 = 1.55 * µm
N_CORE, N_CLAD = 3.48, 1.44
DX, DT = dxdt(WL0, n_max=N_CORE, safety_factor=0.999, points_per_wavelength=12, dims=2)
WAVELENGTHS = np.linspace(1.50 * µm, 1.60 * µm, 41)
BASE_TIME_MULT = 45
SOURCE_OFFSET, MONITOR_OFFSET = 1.2 * µm, 1.8 * µm
SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN = 4.0, 1.0 * µm
MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN = 1.6, 0.8 * µm

# Load and visualize the gdsfactory cell
design_obj, ports = design.io.gdsf.load("mmi1x2", n_core=N_CORE, n_clad=N_CLAD, layer=(1, 0), padding=3.0)
design_obj.show()

# Rasterize the design
grid = design_obj.rasterize(resolution=DX)
grid.show(field="permittivity")

# Define the source and output ports
source_port, output_ports = "o1", ["o1", "o2", "o3"]

# Define the inward point function
def inward_point(port, offset):
    sign = 1.0 if port["direction"].startswith("+") else -1.0
    axis = port["direction"][1]
    cx, cy = port["center"]
    return (cx + sign * offset, cy) if axis == "x" else (cx, cy + sign * offset)

# Define the port line function
def port_line(port, offset, span_factor, span_min):
    cx, cy = inward_point(port, offset)
    span = max(float(span_min), span_factor * float(port["width"]))
    if port["direction"][1] == "x":
        return (cx, cy - span / 2), (cx, cy + span / 2), span
    return (cx - span / 2, cy), (cx + span / 2, cy), span

# Calculate the maximum distance between the source and output ports
max_dist = max(
    np.hypot(
        inward_point(ports[source_port], SOURCE_OFFSET)[0]
        - inward_point(ports[out], MONITOR_OFFSET)[0],
        inward_point(ports[source_port], SOURCE_OFFSET)[1]
        - inward_point(ports[out], MONITOR_OFFSET)[1],
    )
    for out in output_ports
)
TIME = max(
    BASE_TIME_MULT * WL0 / LIGHT_SPEED,
    2.2 * max(N_CORE, N_CLAD) * max_dist / LIGHT_SPEED + 10.0 * WL0 / LIGHT_SPEED,
)

# Define the time and signal
time = np.arange(0, TIME, DT)
signal = ramped_cosine(time, amplitude=1.0, frequency=LIGHT_SPEED/WL0,
    ramp_duration=WL0*6/LIGHT_SPEED, t_max=TIME / 4)
plot_signal(signal, time)

# Place the source
src = ports[source_port]
src_center = inward_point(src, SOURCE_OFFSET)
_, _, src_span = port_line(src, SOURCE_OFFSET, SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN)
source = ModeSource(
    grid=grid,
    center=src_center,
    width=src_span,
    wavelength=WL0,
    pol="tm",
    signal=signal,
    direction=src["direction"]
)

# Define the monitors
monitors = []
for out in output_ports:
    start, end, _ = port_line(
        ports[out], MONITOR_OFFSET, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN
    )
    monitors.append(Monitor(start=start, end=end, name=out, record_fields=True))

# Define the simulation
sim = Simulation(
    design=design_obj,
    devices=[source, *monitors],
    boundaries=[PML(edges="all", thickness=1.0 * WL0)],
    time=time,
    resolution=DX
)

# Run the simulation
sim.run(animate_live="Ez", animation_interval=10, axis_scale=[-5e-5, 5e-5],
    cmap="twilight_zero", clean_visualization=True)

# Extract sparse SAX terms for o1 excitation
s_sparse = sim.get_S_matrix(
    input_ports=[source_port],
    output_ports=output_ports,
    source_port=source_port,
    frequencies=None,
    field_component="Ez",
    reduction="mean",
    as_sax=False,
)

# Extract the SAX terms
s_sax = sax.sdict(s_sparse)

# Convert the frequencies to wavelengths
wl_um = LIGHT_SPEED / np.maximum(sim.s_matrix_frequencies, 1e-30) / µm

for key in [("o1", "o1"), ("o2", "o1"), ("o3", "o1")]:
    s_vals = np.asarray(s_sax[key])
    mask = np.isfinite(wl_um)
    w = wl_um[mask]
    s_vals = s_vals[mask]
    s0 = s_vals[np.argmin(np.abs(w - WL0 / µm))]
    print(
        f"S[{key[0]},{key[1]}] @ {WL0 / µm:.3f}um: "
        f"|S|={np.abs(s0):.3f}, phase={np.angle(s0):.3f} rad"
    )

# Plot S_ij curves in dB
plt.figure(figsize=(7, 4))
for key, color in [
    (("o1", "o1"), "black"),
    (("o2", "o1"), "tab:blue"),
    (("o3", "o1"), "tab:orange"),
]:
    s_vals = np.asarray(s_sax[key])
    mask = (
        np.isfinite(wl_um)
        & (wl_um >= WAVELENGTHS.min() / µm)
        & (wl_um <= WAVELENGTHS.max() / µm)
    )
    x = wl_um[mask]
    y_db = 20 * np.log10(np.maximum(np.abs(s_vals[mask]), 1e-12))
    order = np.argsort(x)
    x_dense = np.linspace(WAVELENGTHS.min() / µm, WAVELENGTHS.max() / µm, 400)
    y_dense = np.interp(x_dense, x[order], y_db[order])
    plt.plot(x_dense, y_dense, color=color, label=rf"$S_{{{key[0][1:]},{key[1][1:]}}}$")

plt.xlabel("wavelength (um)")
plt.ylabel("magnitude (dB)")
plt.title("SAX Splitter Terms (o1 excitation)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
