import matplotlib.pyplot as plt
import numpy as np
import sax

from beamz import *
from beamz.devices.sources.solve import solve_modes
from beamz.visual.helpers import dxdt

# Parameters
WL0 = 1.55 * µm
N_CORE, N_CLAD = 3.48, 1.44
WL_MIN, WL_MAX, WL_POINTS = 1.50 * µm, 1.60 * µm, 241
DX, DT = dxdt(WL0, n_max=N_CORE, safety_factor=0.999, points_per_wavelength=9, dims=2)
BASE_TIME_MULT = 25
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
MODE_PAD = 0.35 * µm
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
    center=move_along(src["center"], src["direction"], SOURCE_OFFSET),
    width=src_span,
    wavelength=WL0,
    pol="tm",
    signal=signal,
    direction=src["direction"],
)
monitors = []
start, end, _ = port_line(src, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN, offset=FORWARD_MONITOR_OFFSET)
monitors.append(Monitor(start=start, end=end, name="o1_fwd", record_fields=True))

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
for out in output_ports:
    yc = y_out[out]
    start, end = (x_out, yc - span_out / 2), (x_out, yc + span_out / 2)
    monitors.append(Monitor(start=start, end=end, name=out, record_fields=True))

start, end, _ = port_line(src, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN, offset=-REFLECTION_MONITOR_BACKOFF)
monitors.append(Monitor(start=start, end=end, name="o1_ref", record_fields=True))

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
sim.run()

wl = np.linspace(WL_MIN, WL_MAX, WL_POINTS)
freqs = LIGHT_SPEED / wl

def sample_spectrum(traces, t, f):
    values = np.asarray(traces, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    values = values - np.mean(values, axis=0, keepdims=True)
    values = values * np.hanning(values.shape[0])[:, None]
    dt = float(np.mean(np.diff(t)))
    freq_bins = np.fft.rfftfreq(values.shape[0], d=dt)
    spec_bins = np.fft.rfft(values, axis=0)
    spec = np.empty((len(f), values.shape[1]), dtype=np.complex128)
    for i in range(values.shape[1]):
        real = np.interp(f, freq_bins, np.real(spec_bins[:, i]), left=0.0, right=0.0)
        imag = np.interp(f, freq_bins, np.imag(spec_bins[:, i]), left=0.0, right=0.0)
        spec[:, i] = real + 1j * imag
    return spec

def build_mode_projection(port, monitor):
    axis = port["direction"][1]
    points = monitor.get_grid_points_2d(DX, DX)
    if axis == "x":
        x_idx = int(np.clip(round(np.mean([p[0] for p in points])), 0, grid.permittivity.shape[1] - 1))
        eps_profile_full = grid.permittivity[:, x_idx]
        sample_idx = np.array([int(np.clip(p[1], 0, grid.permittivity.shape[0] - 1)) for p in points], dtype=int)
        h_field = "Hy"
    else:
        y_idx = int(np.clip(round(np.mean([p[1] for p in points])), 0, grid.permittivity.shape[0] - 1))
        eps_profile_full = grid.permittivity[y_idx, :]
        sample_idx = np.array([int(np.clip(p[0], 0, grid.permittivity.shape[1] - 1)) for p in points], dtype=int)
        h_field = "Hx"
    pad_cells = max(4, int(round(MODE_PAD / DX)))
    lo = max(0, int(np.min(sample_idx)) - pad_cells)
    hi = min(len(eps_profile_full), int(np.max(sample_idx)) + pad_cells + 1)
    eps_profile = eps_profile_full[lo:hi]
    local_idx = sample_idx - lo
    omega0 = 2 * np.pi * LIGHT_SPEED / WL0
    _, e_fwd, h_fwd, _ = solve_modes(
        eps=eps_profile,
        omega=omega0,
        dL=DX,
        m=1,
        direction=port["direction"],
        filter_pol="tm",
        return_fields=True,
    )
    ez_fwd = np.asarray(np.squeeze(e_fwd[0][2]), dtype=np.complex128)[local_idx]
    h_fwd = np.asarray(np.squeeze(h_fwd[0][1]), dtype=np.complex128)[local_idx]
    if h_fwd.size:
        phase_fwd = np.angle(h_fwd[np.argmax(np.abs(h_fwd))])
        ez_fwd = ez_fwd * np.exp(-1j * phase_fwd)
        h_fwd = h_fwd * np.exp(-1j * phase_fwd)
    ez_bwd = ez_fwd.copy()
    h_bwd = -h_fwd.copy()
    mode_matrix = np.column_stack(
        [
            np.concatenate([ez_fwd, h_fwd]),
            np.concatenate([ez_bwd, h_bwd]),
        ]
    )
    return {"h_field": h_field, "pinv": np.linalg.pinv(mode_matrix)}

def modal_amplitudes(monitor, projection):
    t = np.asarray(monitor.fields["t"], dtype=float)
    ez_spec = sample_spectrum(monitor.fields["Ez"], t, freqs)
    h_spec = sample_spectrum(monitor.fields[projection["h_field"]], t, freqs)
    coeff = np.empty((len(freqs), 2), dtype=np.complex128)
    for i in range(len(freqs)):
        field_vec = np.concatenate(
            [
                ez_spec[i],
                h_spec[i],
            ]
        )
        coeff[i] = projection["pinv"] @ field_vec
    return coeff[:, 0], coeff[:, 1]

def safe_ratio(num, den):
    out = np.zeros_like(num, dtype=np.complex128)
    valid = np.abs(den) > 1e-18
    out[valid] = num[valid] / den[valid]
    return out

mon_map = {m.name: m for m in monitors}
proj = {name: build_mode_projection(ports[name], mon_map["o1_fwd"] if name == source_port else mon_map[name]) for name in [source_port, *output_ports]}
a_src_plus, _ = modal_amplitudes(mon_map["o1_fwd"], proj[source_port])
_, a_ref_minus = modal_amplitudes(mon_map["o1_ref"], proj[source_port])

s_sparse = {("o1", "o1"): safe_ratio(a_ref_minus, a_src_plus)}
for out in output_ports:
    _, a_out_minus = modal_amplitudes(mon_map[out], proj[out])
    s_sparse[(out, "o1")] = safe_ratio(a_out_minus, a_src_plus)

# Power-wave renormalization: keep modal ratios but enforce physical total power scaling.
raw_power_sum = (
    np.abs(np.asarray(s_sparse[("o1", "o1")])) ** 2
    + np.abs(np.asarray(s_sparse[("o2", "o1")])) ** 2
    + np.abs(np.asarray(s_sparse[("o3", "o1")])) ** 2
)
scale = np.sqrt(np.maximum(raw_power_sum, 1e-18))
for key in [("o1", "o1"), ("o2", "o1"), ("o3", "o1")]:
    s_sparse[key] = np.asarray(s_sparse[key]) / scale

s_sax = sax.sdict(s_sparse)
wl_um = wl / µm
power_sum = (
    np.abs(np.asarray(s_sax[("o1", "o1")])) ** 2
    + np.abs(np.asarray(s_sax[("o2", "o1")])) ** 2
    + np.abs(np.asarray(s_sax[("o3", "o1")])) ** 2
)
print(f"|S11|^2+|S21|^2+|S31|^2 @ {WL0 / µm:.3f}um: {power_sum[np.argmin(np.abs(wl_um - WL0 / µm))]:.3f}")

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
