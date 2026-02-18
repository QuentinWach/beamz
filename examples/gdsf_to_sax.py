import sys; from pathlib import Path; import matplotlib.pyplot as plt; import numpy as np; import sax
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from beamz import *
WL0, N_CORE, N_CLAD = 1.55 * µm, 3.48, 1.44
DX, DT = dxdt(WL0, n_max=N_CORE, safety_factor=0.999, points_per_wavelength=12, dims=2)
WAVELENGTHS = np.linspace(1.50 * µm, 1.60 * µm, 41)
BASE_TIME_MULT, SOURCE_OFFSET, MONITOR_OFFSET = 45, 1.2 * µm, 1.8 * µm
SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN = 4.0, 1.0 * µm, 1.6, 0.8 * µm
design_obj, ports = design.io.gdsf.load("mmi1x2", n_core=N_CORE, n_clad=N_CLAD, layer=(1, 0), padding=3.0); design_obj.show()
grid = design_obj.rasterize(resolution=DX); source_port, output_ports = "o1", ["o1", "o2", "o3"]
inside = lambda p, d: ((p["center"][0] + (1.0 if p["direction"].startswith("+") else -1.0) * d, p["center"][1]) if p["direction"][1] == "x" else (p["center"][0], p["center"][1] + (1.0 if p["direction"].startswith("+") else -1.0) * d))
monline = lambda p, off, fac, mn: ((inside(p, off)[0], inside(p, off)[1] - max(float(mn), fac * float(p["width"])) / 2), (inside(p, off)[0], inside(p, off)[1] + max(float(mn), fac * float(p["width"])) / 2), max(float(mn), fac * float(p["width"]))) if p["direction"][1] == "x" else ((inside(p, off)[0] - max(float(mn), fac * float(p["width"])) / 2, inside(p, off)[1]), (inside(p, off)[0] + max(float(mn), fac * float(p["width"])) / 2, inside(p, off)[1]), max(float(mn), fac * float(p["width"])))
max_dist = max(np.hypot(inside(ports[source_port], SOURCE_OFFSET)[0] - inside(ports[o], MONITOR_OFFSET)[0], inside(ports[source_port], SOURCE_OFFSET)[1] - inside(ports[o], MONITOR_OFFSET)[1]) for o in output_ports)
TIME = max(BASE_TIME_MULT * WL0 / LIGHT_SPEED, 2.2 * max(N_CORE, N_CLAD) * max_dist / LIGHT_SPEED + 10.0 * WL0 / LIGHT_SPEED)
time = np.arange(0, TIME, DT); signal = ramped_cosine(time, amplitude=1.0, frequency=LIGHT_SPEED / WL0, ramp_duration=WL0 * 6 / LIGHT_SPEED, t_max=TIME * 0.30); plot_signal(signal, time)
src = ports[source_port]; src_center = inside(src, SOURCE_OFFSET); _, _, src_span = monline(src, SOURCE_OFFSET, SOURCE_SPAN_FACTOR, SOURCE_MIN_SPAN)
source = ModeSource(grid=grid, center=src_center, width=src_span, wavelength=WL0, pol="tm", signal=signal, direction=src["direction"])
monitors = [Monitor(start=monline(ports[p], MONITOR_OFFSET, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN)[0], end=monline(ports[p], MONITOR_OFFSET, MONITOR_SPAN_FACTOR, MONITOR_MIN_SPAN)[1], name=p, record_fields=True) for p in output_ports]
sim = Simulation(design=design_obj, devices=[source, *monitors], boundaries=[PML(edges="all", thickness=1.0 * WL0)], time=time, resolution=DX); sim.run()
s_sparse = sim.get_S_matrix(input_ports=[source_port], output_ports=output_ports, source_port=source_port, frequencies=None, field_component="Ez", reduction="mean", as_sax=False)
s_sax = sax.sdict(s_sparse); wl_um = LIGHT_SPEED / np.maximum(sim.s_matrix_frequencies, 1e-30) / µm
for k in [("o1", "o1"), ("o2", "o1"), ("o3", "o1")]:
    s = np.asarray(s_sax[k]); m = np.isfinite(wl_um); w = wl_um[m]; s = s[m]; s0 = s[np.argmin(np.abs(w - WL0 / µm))]; print(f"S[{k[0]},{k[1]}] @ {WL0 / µm:.3f}um: |S|={np.abs(s0):.3f}, phase={np.angle(s0):.3f} rad")
plt.figure(figsize=(7, 4))
for k, c in [(("o1", "o1"), "black"), (("o2", "o1"), "tab:blue"), (("o3", "o1"), "tab:orange")]:
    s = np.asarray(s_sax[k]); m = np.isfinite(wl_um) & (wl_um >= WAVELENGTHS.min() / µm) & (wl_um <= WAVELENGTHS.max() / µm); x = wl_um[m]; y = 20 * np.log10(np.maximum(np.abs(s[m]), 1e-12)); o = np.argsort(x); xd = np.linspace(WAVELENGTHS.min() / µm, WAVELENGTHS.max() / µm, 400); yd = np.interp(xd, x[o], y[o]); plt.plot(xd, yd, color=c, label=rf"$S_{{{k[0][1:]},{k[1][1:]}}}$")
plt.xlabel("wavelength (um)"); plt.ylabel("magnitude (dB)"); plt.title("SAX Splitter Terms (o1 excitation)"); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.show()
