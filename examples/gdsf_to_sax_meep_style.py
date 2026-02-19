"""Meep-style broadband SAX extraction with one short Gaussian pulse.

Workflow:
1. Import gdsfactory `mmi1x2` geometry into BeamZ.
2. Run one broadband pulse simulation (single source port).
3. Use DFT-enabled mode monitors for source/output ports.
4. Extract S11/S21/S31 via modal decomposition.
"""

import os
import time as pytime
import matplotlib.pyplot as plt
import numpy as np
import sax
from beamz import *
from beamz.devices.sources.signals import gaussian_pulse
from beamz.visual.helpers import dxdt
try: from scipy.interpolate import PchipInterpolator
except Exception: PchipInterpolator = None

WL0 = 1.55 * µm
WL_MIN, WL_MAX = 1.50 * µm, 1.60 * µm
WL_POINTS = int(os.getenv("BEAMZ_SWEEP_POINTS", "21"))
N_CORE, N_CLAD = 3.48, 1.44
POINTS_PER_WAVELENGTH = int(os.getenv("BEAMZ_PPW", "10"))
DX, DT = dxdt(WL0, n_max=N_CORE, points_per_wavelength=POINTS_PER_WAVELENGTH, dims=2)
INPUT_EXTENSION, OUTPUT_EXTENSION, Y_MARGIN = 4.0 * µm, 4.0 * µm, 3.0 * µm
PML_BASE, PML_RIGHT = 1.0 * WL0, 1.5 * WL0
SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN = 4.0, 1.0 * µm
MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN = 2.4, 1.0 * µm
SOURCE_OFFSET = -1.2 * µm
FORWARD_MONITOR_OFFSET = -0.4 * µm
REFLECTION_MONITOR_BACKOFF = 2.0 * µm
OUTPUT_MONITOR_OFFSET = 0.6 * µm
SOURCE_PORT, OUTPUT_PORTS = "o1", ["o2", "o3"]
DIR_VEC = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}


def move_along(center, direction, distance):
    vx, vy = DIR_VEC[direction]
    return center[0] + vx * float(distance), center[1] + vy * float(distance)


def outward_direction(inward_direction):
    return ("-" if inward_direction.startswith("+") else "+") + inward_direction[1]


def port_line(port, span_factor, span_min, offset=0.0):
    cx, cy = move_along(port["center"], port["direction"], offset)
    span = max(float(span_min), span_factor * float(port["width"]))
    if port["direction"][1] == "x":
        return (cx, cy - span / 2), (cx, cy + span / 2), span
    return (cx - span / 2, cy), (cx + span / 2, cy), span


imported_design, ports = design.io.gdsf.load("mmi1x2", n_core=N_CORE, n_clad=N_CLAD, layer=(1, 0), padding=0.0)
device_design = Design(
    width=imported_design.width + INPUT_EXTENSION + OUTPUT_EXTENSION,
    height=imported_design.height + 2.0 * Y_MARGIN,
    depth=0,
    material=Material(N_CLAD**2),
)
for structure in imported_design.structures[1:]:
    device_design += structure.copy().shift(INPUT_EXTENSION, Y_MARGIN)
ports = {name: {**p, "center": (p["center"][0] + INPUT_EXTENSION, p["center"][1] + Y_MARGIN)} for name, p in ports.items()}

for name, port in ports.items():
    cx, cy, width = *port["center"], float(port["width"])
    extension = INPUT_EXTENSION if name == SOURCE_PORT else OUTPUT_EXTENSION
    ox, oy = move_along((cx, cy), outward_direction(port["direction"]), extension)
    if port["direction"][1] == "x":
        device_design += Rectangle(position=(min(cx, ox), cy - width / 2), width=extension, height=width, material=Material(N_CORE**2), depth=0)
    else:
        device_design += Rectangle(position=(cx - width / 2, min(cy, oy)), width=width, height=extension, material=Material(N_CORE**2), depth=0)

device_design.show()
grid = device_design.rasterize(resolution=DX)
grid.show(field="permittivity")

src = ports[SOURCE_PORT]
source_center = move_along(src["center"], src["direction"], SOURCE_OFFSET)
_, _, src_span = port_line(src, SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN, offset=0.0)
o1_fwd_start, o1_fwd_end, _ = port_line(src, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN, offset=FORWARD_MONITOR_OFFSET)
o1_ref_start, o1_ref_end, _ = port_line(src, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN, offset=-REFLECTION_MONITOR_BACKOFF)
max_dist = max(np.hypot(src["center"][0] - ports[o]["center"][0], src["center"][1] - ports[o]["center"][1]) for o in OUTPUT_PORTS)
travel_time = 2.3 * max(N_CORE, N_CLAD) * max_dist / LIGHT_SPEED

out_lines = {}
for out in OUTPUT_PORTS:
    out_lines[out] = port_line(ports[out], MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN, offset=-OUTPUT_MONITOR_OFFSET)[:2]

freqs = np.linspace(LIGHT_SPEED / WL_MAX, LIGHT_SPEED / WL_MIN, WL_POINTS)
wl = LIGHT_SPEED / freqs
fmin, fmax = float(np.min(freqs)), float(np.max(freqs))
df = float(np.min(np.diff(np.sort(freqs)))) if len(freqs) > 1 else fmin
fcen, fwidth = 0.5 * (fmin + fmax), max(fmax - fmin, 1e9)
sigma_t = 0.20 / fwidth
t0 = 4.0 * sigma_t
t_end_pulse = t0 + 4.0 * sigma_t
ringdown = max(4.0 / fmin, 0.5 / max(df, 1e6))
total_time = t_end_pulse + 1.5 * travel_time + ringdown
time = np.arange(0.0, total_time, DT)
signal = gaussian_pulse(time, amplitude=1.0, center=t0, width=sigma_t, frequency=fcen, phase=0.0)
plot_signal(signal, time)

source = ModeSource(grid=grid, center=source_center, width=src_span, wavelength=float(WL0), pol="tm", signal=signal, direction=src["direction"])
monitor_stride = int(os.getenv("BEAMZ_MONITOR_STRIDE", "3"))
monitor_cfg = dict(
    record_fields=False,
    dft_enabled=True,
    dft_frequencies=freqs,
    dft_t_start=0.0,
    dft_t_end=float(total_time),
    dft_components=("Ez", "Hx", "Hy"),
    dft_window="rect",
    record_interval=max(monitor_stride, 1),
    dft_record_every_step=(monitor_stride <= 1),
)
monitors = [
    Monitor(start=o1_fwd_start, end=o1_fwd_end, name="o1_fwd", **monitor_cfg),
    Monitor(start=o1_ref_start, end=o1_ref_end, name="o1_ref", **monitor_cfg),
]
for out in OUTPUT_PORTS:
    m_start, m_end = out_lines[out]
    monitors.append(Monitor(start=m_start, end=m_end, name=out, **monitor_cfg))

port_specs = {
    SOURCE_PORT: PortSpec(
        name=SOURCE_PORT,
        monitor_name="o1_ref",
        reference_monitor="o1_fwd",
        direction=ports[SOURCE_PORT]["direction"],
        polarization="tm",
    ),
    "o2": PortSpec(name="o2", monitor_name="o2", direction=ports["o2"]["direction"], polarization="tm"),
    "o3": PortSpec(name="o3", monitor_name="o3", direction=ports["o3"]["direction"], polarization="tm"),
}

sim = Simulation(
    design=device_design,
    devices=[source, *monitors],
    boundaries=[PML(edges=["left", "top", "bottom"], thickness=PML_BASE), PML(edges="right", thickness=PML_RIGHT)],
    time=time,
    resolution=DX,
)
print(f"Running Meep-style pulse simulation ({WL_POINTS} wavelengths, one run)...")
wall_t0 = pytime.time()
sim.run_fast(progress=False)
print(f"Simulation wall-time: {pytime.time() - wall_t0:.1f} s")

result = sim.get_S_matrix_modal_dft(
    source_port=SOURCE_PORT,
    ports=port_specs,
    output_ports=[SOURCE_PORT, *OUTPUT_PORTS],
    frequencies=freqs,
    as_sax=False,
    return_diagnostics=True,
    min_incident_db=-55.0,
)
s_sax = sax.sdict(result["s_matrix"])
diag = result["diagnostics"]
valid = np.asarray(diag["valid_mask"], dtype=bool)
wl_um = wl / µm

power_sum = np.abs(np.asarray(s_sax[("o1", "o1")])) ** 2 + np.abs(np.asarray(s_sax[("o2", "o1")])) ** 2 + np.abs(np.asarray(s_sax[("o3", "o1")])) ** 2
i0 = int(np.argmin(np.abs(wl_um - WL0 / µm)))
print(f"|S11|^2+|S21|^2+|S31|^2 @ {WL0 / µm:.3f}um: {power_sum[i0]:.3f}")
print(f"loss @ {WL0 / µm:.3f}um: {diag['loss_est'][i0]:.3f} (valid={bool(valid[i0])})")
for key in [("o1", "o1"), ("o2", "o1"), ("o3", "o1")]:
    s0 = np.asarray(s_sax[key])[i0]
    print(f"S[{key[0]},{key[1]}] @ {WL0 / µm:.3f}um: |S|={np.abs(s0):.3f}, phase={np.angle(s0):.3f} rad")

plt.figure(figsize=(5.5, 3.4), dpi=250)
wl_dense = np.linspace(np.min(wl_um), np.max(wl_um), max(700, 20 * len(wl_um)))
for key, color in [(("o1", "o1"), "black"), (("o2", "o1"), "tab:blue"), (("o3", "o1"), "tab:orange")]:
    y_db = 20 * np.log10(np.maximum(np.abs(np.asarray(s_sax[key], dtype=np.complex128)), 1e-12))
    y_db = np.where(valid, y_db, np.nan)
    finite = np.isfinite(y_db)
    if np.count_nonzero(finite) >= 4:
        x = wl_um[finite]
        y = y_db[finite]
        if x[0] > x[-1]:
            x = x[::-1]
            y = y[::-1]
        y_dense = PchipInterpolator(x, y)(wl_dense) if PchipInterpolator is not None else np.interp(wl_dense, x, y)
        plt.plot(wl_dense, y_dense, "-", linewidth=2.2, color=color, label=rf"$S_{{{key[0][1:]}{key[1][1:]}}}$")
        plt.plot(wl_um[finite], y_db[finite], "o", ms=2.6, color=color, alpha=0.45)
    else:
        plt.plot(wl_um, y_db, "o-", linewidth=2.2, ms=3.0, color=color, label=rf"$S_{{{key[0][1:]}{key[1][1:]}}}$")
plt.title("GDSFactory MMI1x2 (Meep-style Gaussian Pulse)")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Magnitude (dB)")
plt.grid(alpha=0.3)
plt.ylim(-40, 0)
plt.xlim(WL_MIN / µm, WL_MAX / µm)
plt.legend()
plt.tight_layout()
plt.savefig("sax_splitter_terms_meep_style.png", dpi=300)
plt.show()
