from beamz import *
from beamz.design.library import gold, sio2, water
from beamz.devices.sources.signals import gaussian_pulse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Choose one of: "SiO2", "Gold", "Water"
CASE = "SiO2"
FAST = True
SAVE = False

CASES = {
    "SiO2": {
        "material_fn": sio2,
        "name": "SiO2 Sellmeier Slab",
        "wl0": 1.2 * um,
        "domain": (18.0 * um, 6.0 * um),
        "resolution": 0.5 * um,
        "slab_thickness": 3.0 * um,
        "steps": 90,
        "source_width": 0.5 * um,
    },
    "Gold": {
        "material_fn": gold,
        "name": "Gold Drude Slab",
        "wl0": 0.8 * um,
        "domain": (18.0 * um, 6.0 * um),
        "resolution": 0.6 * um,
        "slab_thickness": 2.4 * um,
        "steps": 90,
        "source_width": 0.4 * um,
    },
    "Water": {
        "material_fn": water,
        "name": "Water Debye Slab",
        "wl0": 6.0e-4,
        "domain": (6.0e-3, 2.0e-3),
        "resolution": 6.0e-5,
        "slab_thickness": 1.5e-3,
        "steps": 140,
        "source_width": 1.5e-4,
    },
}

cfg = CASES[CASE]
if FAST:
    cfg = dict(cfg)
    cfg["steps"] = max(40, int(0.6 * cfg["steps"]))

material = cfg["material_fn"]()
width, height = cfg["domain"]
resolution = cfg["resolution"]
wl0 = cfg["wl0"]
slab_thickness = cfg["slab_thickness"]

# Same simulation workflow as the basic examples:
# 1) define design, 2) define source, 3) run simulation, 4) plot results.
dt = 0.95 * resolution / (LIGHT_SPEED * np.sqrt(2.0))
time_steps = np.arange(cfg["steps"], dtype=float) * dt
source_freq = LIGHT_SPEED / wl0
signal = gaussian_pulse(
    time_steps,
    amplitude=1e-8,
    center=0.25 * time_steps[-1],
    width=0.16 * time_steps[-1],
    frequency=source_freq,
    phase=0.0,
)

slab_x0 = 0.45 * width
source_x = 0.16 * width
probe_offset = 0.5 * wl0
x_before = slab_x0 - probe_offset
x_after = slab_x0 + slab_thickness + probe_offset


def run_case(with_slab: bool):
    design = Design(width=width, height=height, material=Material(permittivity=1.0))
    if with_slab:
        design += Rectangle(
            position=(slab_x0, 0.0),
            width=slab_thickness,
            height=height,
            material=material,
        )

    source = GaussianSource(
        position=(source_x, 0.5 * height),
        width=cfg["source_width"],
        signal=signal,
    )
    boundaries = [PML(edges="all", thickness=max(2.0 * resolution, 1.2 * wl0))]

    sim = Simulation(
        design=design,
        devices=[source],
        boundaries=boundaries,
        time=time_steps,
        resolution=resolution,
    )
    result = sim.run(record_interval=1, record_fields=["Ez"], progress=False)
    fields = np.asarray(result["fields"]["Ez"])

    finite = np.isfinite(fields.reshape(fields.shape[0], -1)).all(axis=1)
    if np.any(~finite):
        first_bad = int(np.where(~finite)[0][0])
        fields = fields[:first_bad]
        print(f"Warning: non-finite fields after step {first_bad}; truncating history.")

    y_mid = fields.shape[1] // 2
    x_before_idx = int(np.clip(round(x_before / resolution), 0, fields.shape[2] - 1))
    x_after_idx = int(np.clip(round(x_after / resolution), 0, fields.shape[2] - 1))
    probe_before = fields[:, y_mid, x_before_idx]
    probe_after = fields[:, y_mid, x_after_idx]
    return fields, probe_before, probe_after


print(f"Running showcase case: {cfg['name']}")
fields_ref, ref_before, ref_after = run_case(with_slab=False)
fields_slab, slab_before, slab_after = run_case(with_slab=True)

# Extract transfer function through the slab and compare to analytic material model.
n = min(len(ref_before), len(ref_after), len(slab_before), len(slab_after))
if n < 16:
    raise RuntimeError("Not enough field samples for spectral extraction.")

s1 = slab_before[:n] - np.mean(slab_before[:n])
s2 = slab_after[:n] - np.mean(slab_after[:n])
r1 = ref_before[:n] - np.mean(ref_before[:n])
r2 = ref_after[:n] - np.mean(ref_after[:n])

w = np.hanning(n)
S1 = np.fft.rfft(s1 * w)
S2 = np.fft.rfft(s2 * w)
R1 = np.fft.rfft(r1 * w)
R2 = np.fft.rfft(r2 * w)
f = np.fft.rfftfreq(n, dt)
H = (S2 / (S1 + 1e-30)) / (R2 / (R1 + 1e-30) + 1e-30)

band = (f > 0.55 * source_freq) & (f < 1.45 * source_freq)
if np.sum(band) < 6:
    raise RuntimeError("Passband is too small for extraction; increase steps or adjust pulse.")

f = f[band]
H = H[band]
k0 = 2.0 * np.pi * f / LIGHT_SPEED
n_ext = 1.0 - np.unwrap(np.angle(H)) / (k0 * slab_thickness + 1e-30)
k_ext = -np.log(np.abs(H) + 1e-30) / (k0 * slab_thickness + 1e-30)

n_ref_complex = material.n_complex(frequency=f)
n_ref = np.real(n_ref_complex)
k_ref = np.imag(n_ref_complex)

eps_ext = (n_ext + 1j * k_ext) ** 2
eps_ref = np.asarray(material.epsilon(frequency=f), dtype=complex)
wl_um = (LIGHT_SPEED / f) * 1e6
order = np.argsort(wl_um)
wl_um = wl_um[order]

# Visualization: animation + spectral overlays.
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.34, hspace=0.32)
ax_anim = fig.add_subplot(gs[:, 0])
ax_n = fig.add_subplot(gs[0, 1])
ax_k = fig.add_subplot(gs[0, 2])
ax_er = fig.add_subplot(gs[1, 1])
ax_ei = fig.add_subplot(gs[1, 2])

vmax = float(np.nanmax(np.abs(fields_slab)))
vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
extent = (0.0, width, 0.0, height)
image = ax_anim.imshow(
    fields_slab[0],
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
)
ax_anim.set_title("Ez propagation through slab")
ax_anim.set_xlabel("x (m)")
ax_anim.set_ylabel("y (m)")
fig.colorbar(image, ax=ax_anim, fraction=0.046, pad=0.04)

frame_ids = np.linspace(0, fields_slab.shape[0] - 1, min(90, fields_slab.shape[0]), dtype=int)


def update(frame_idx):
    idx = int(frame_ids[frame_idx])
    image.set_data(fields_slab[idx])
    ax_anim.set_title(f"Ez propagation through slab (frame {idx + 1}/{fields_slab.shape[0]})")
    return (image,)


_ani = FuncAnimation(fig, update, frames=len(frame_ids), interval=60, blit=False, repeat=True)

n_ref_plot = n_ref[order]
n_ext_plot = n_ext[order]
k_ref_plot = k_ref[order]
k_ext_plot = k_ext[order]
eps_ref_plot = eps_ref[order]
eps_ext_plot = eps_ext[order]

ax_n.plot(wl_um, n_ref_plot, lw=1.7, color="#1f77b4", label="Reference")
ax_n.plot(wl_um, n_ext_plot, lw=1.1, color="#ff7f0e", alpha=0.9, label="Extracted")
ax_n.set_xlabel("Wavelength (um)")
ax_n.set_ylabel("n")
ax_n.grid(alpha=0.25)
ax_n.legend(fontsize=8)

ax_k.plot(wl_um, k_ref_plot, lw=1.7, color="#d62728", label="Reference")
ax_k.plot(wl_um, k_ext_plot, lw=1.1, color="#9467bd", alpha=0.9, label="Extracted")
ax_k.set_xlabel("Wavelength (um)")
ax_k.set_ylabel("k")
ax_k.grid(alpha=0.25)
ax_k.legend(fontsize=8)

ax_er.plot(wl_um, np.real(eps_ref_plot), lw=1.7, color="#2ca02c", label="Reference")
ax_er.plot(wl_um, np.real(eps_ext_plot), lw=1.1, color="#8c564b", alpha=0.9, label="Extracted")
ax_er.set_xlabel("Wavelength (um)")
ax_er.set_ylabel("Re(eps_r)")
ax_er.grid(alpha=0.25)
ax_er.legend(fontsize=8)

ax_ei.plot(wl_um, np.imag(eps_ref_plot), lw=1.7, color="#17becf", label="Reference")
ax_ei.plot(wl_um, np.imag(eps_ext_plot), lw=1.1, color="#bcbd22", alpha=0.9, label="Extracted")
ax_ei.set_xlabel("Wavelength (um)")
ax_ei.set_ylabel("Im(eps_r)")
ax_ei.grid(alpha=0.25)
ax_ei.legend(fontsize=8)

rmse_n = float(np.sqrt(np.mean((n_ext - n_ref) ** 2)))
rmse_k = float(np.sqrt(np.mean((k_ext - k_ref) ** 2)))
ax_ei.text(
    0.98,
    0.03,
    f"RMSE(n): {rmse_n:.3g}\\nRMSE(k): {rmse_k:.3g}",
    transform=ax_ei.transAxes,
    ha="right",
    va="bottom",
    family="monospace",
    fontsize=8,
    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
)

fig.suptitle(cfg["name"])
fig.subplots_adjust(top=0.9, wspace=0.34, hspace=0.32)

if SAVE:
    out_path = f"artifacts/dispersion_showcase_{CASE.lower()}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

plt.show()
