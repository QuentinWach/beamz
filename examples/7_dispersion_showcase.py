from beamz import *
from beamz.devices.sources.signals import gaussian_pulse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

_ANIMATION_HOLD = None

# Select one curated material key from design.material_library.
# Supported by this example: "SiO2", "Gold", "Water".
MATERIAL_KEY = "SiO2"
FAST = True
SAVE = False

# Per-material operating point presets (kept minimal).
CASE_PRESETS = {
    "SiO2": {"wl0": 1.2 * um, "resolution": 0.5 * um, "steps": 90},
    "Gold": {"wl0": 0.8 * um, "resolution": 0.6 * um, "steps": 90},
    "Water": {"wl0": 6.0e-4, "resolution": 6.0e-5, "steps": 140},
}


def get_case_params(material_key: str) -> dict:
    if material_key not in CASE_PRESETS:
        choices = ", ".join(sorted(CASE_PRESETS))
        raise ValueError(f"Unsupported MATERIAL_KEY='{material_key}'. Choose one of: {choices}")
    p = dict(CASE_PRESETS[material_key])
    wl0 = p["wl0"]
    p["width"] = 15.0 * wl0
    p["height"] = 5.0 * wl0
    p["slab_thickness"] = 2.5 * wl0
    p["source_width"] = 0.45 * wl0
    if FAST:
        p["steps"] = max(40, int(0.6 * p["steps"]))
    return p


def get_library_material(material_key: str):
    item = material_library[material_key]
    medium = item.medium
    return medium() if callable(medium) else medium


def build_slab_design(material_key: str, *, with_slab: bool, p: dict) -> tuple[Design, float]:
    width = p["width"]
    height = p["height"]
    slab_thickness = p["slab_thickness"]
    slab_x0 = 0.45 * width

    design = Design(width=width, height=height, material=material_library["Vacuum"].medium)
    if with_slab:
        design += Rectangle(
            position=(slab_x0, 0.0),
            width=slab_thickness,
            height=height,
            material=get_library_material(material_key),
        )
    return design, slab_x0


def run_design(design: Design, p: dict, slab_x0: float):
    width = p["width"]
    height = p["height"]
    wl0 = p["wl0"]
    resolution = p["resolution"]

    dt = 0.95 * resolution / (LIGHT_SPEED * np.sqrt(2.0))
    time_steps = np.arange(p["steps"], dtype=float) * dt
    source_freq = LIGHT_SPEED / wl0
    signal = gaussian_pulse(
        time_steps,
        amplitude=1e-8,
        center=0.25 * time_steps[-1],
        width=0.16 * time_steps[-1],
        frequency=source_freq,
        phase=0.0,
    )

    source = GaussianSource(
        position=(0.16 * width, 0.5 * height),
        width=p["source_width"],
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
    probe_offset = 0.5 * wl0
    x_before = slab_x0 - probe_offset
    x_after = slab_x0 + p["slab_thickness"] + probe_offset
    x_before_idx = int(np.clip(round(x_before / resolution), 0, fields.shape[2] - 1))
    x_after_idx = int(np.clip(round(x_after / resolution), 0, fields.shape[2] - 1))
    probe_before = fields[:, y_mid, x_before_idx]
    probe_after = fields[:, y_mid, x_after_idx]

    return {
        "fields": fields,
        "probe_before": probe_before,
        "probe_after": probe_after,
        "dt": dt,
        "source_freq": source_freq,
        "extent": (0.0, width, 0.0, height),
    }


def extract_dispersion(ref: dict, slab: dict, material, thickness: float):
    n = min(
        len(ref["probe_before"]),
        len(ref["probe_after"]),
        len(slab["probe_before"]),
        len(slab["probe_after"]),
    )
    if n < 16:
        raise RuntimeError("Not enough field samples for spectral extraction.")

    s1 = slab["probe_before"][:n] - np.mean(slab["probe_before"][:n])
    s2 = slab["probe_after"][:n] - np.mean(slab["probe_after"][:n])
    r1 = ref["probe_before"][:n] - np.mean(ref["probe_before"][:n])
    r2 = ref["probe_after"][:n] - np.mean(ref["probe_after"][:n])

    w = np.hanning(n)
    S1 = np.fft.rfft(s1 * w)
    S2 = np.fft.rfft(s2 * w)
    R1 = np.fft.rfft(r1 * w)
    R2 = np.fft.rfft(r2 * w)
    f = np.fft.rfftfreq(n, slab["dt"])
    H = (S2 / (S1 + 1e-30)) / (R2 / (R1 + 1e-30) + 1e-30)

    source_freq = slab["source_freq"]
    band = (f > 0.55 * source_freq) & (f < 1.45 * source_freq)
    if np.sum(band) < 6:
        raise RuntimeError("Passband is too small for extraction; increase steps.")
    f = f[band]
    H = H[band]

    k0 = 2.0 * np.pi * f / LIGHT_SPEED
    n_ext = 1.0 - np.unwrap(np.angle(H)) / (k0 * thickness + 1e-30)
    k_ext = -np.log(np.abs(H) + 1e-30) / (k0 * thickness + 1e-30)
    eps_ext = (n_ext + 1j * k_ext) ** 2

    n_ref_complex = material.n_complex(frequency=f)
    n_ref = np.real(n_ref_complex)
    k_ref = np.imag(n_ref_complex)
    eps_ref = np.asarray(material.epsilon(frequency=f), dtype=complex)

    wl_um = (LIGHT_SPEED / f) * 1e6
    order = np.argsort(wl_um)
    return {
        "wl_um": wl_um[order],
        "n_ref": n_ref[order],
        "n_ext": n_ext[order],
        "k_ref": k_ref[order],
        "k_ext": k_ext[order],
        "eps_ref": eps_ref[order],
        "eps_ext": eps_ext[order],
    }


def plot_showcase(case_name: str, slab_fields: np.ndarray, extent, data):
    global _ANIMATION_HOLD
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.34, hspace=0.32)
    ax_anim = fig.add_subplot(gs[:, 0])
    ax_n = fig.add_subplot(gs[0, 1])
    ax_k = fig.add_subplot(gs[0, 2])
    ax_er = fig.add_subplot(gs[1, 1])
    ax_ei = fig.add_subplot(gs[1, 2])

    vmax = float(np.nanmax(np.abs(slab_fields)))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    image = ax_anim.imshow(
        slab_fields[0],
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

    frame_ids = np.linspace(0, slab_fields.shape[0] - 1, min(90, slab_fields.shape[0]), dtype=int)

    def update(frame_idx):
        idx = int(frame_ids[frame_idx])
        image.set_data(slab_fields[idx])
        ax_anim.set_title(f"Ez propagation through slab (frame {idx + 1}/{slab_fields.shape[0]})")
        return (image,)

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        ani = FuncAnimation(fig, update, frames=len(frame_ids), interval=60, blit=False, repeat=True)
        _ANIMATION_HOLD = ani

    wl_um = data["wl_um"]
    ax_n.plot(wl_um, data["n_ref"], lw=1.7, color="#1f77b4", label="Reference")
    ax_n.plot(wl_um, data["n_ext"], lw=1.1, color="#ff7f0e", alpha=0.9, label="Extracted")
    ax_n.set_xlabel("Wavelength (um)")
    ax_n.set_ylabel("n")
    ax_n.grid(alpha=0.25)
    ax_n.legend(fontsize=8)

    ax_k.plot(wl_um, data["k_ref"], lw=1.7, color="#d62728", label="Reference")
    ax_k.plot(wl_um, data["k_ext"], lw=1.1, color="#9467bd", alpha=0.9, label="Extracted")
    ax_k.set_xlabel("Wavelength (um)")
    ax_k.set_ylabel("k")
    ax_k.grid(alpha=0.25)
    ax_k.legend(fontsize=8)

    ax_er.plot(wl_um, np.real(data["eps_ref"]), lw=1.7, color="#2ca02c", label="Reference")
    ax_er.plot(wl_um, np.real(data["eps_ext"]), lw=1.1, color="#8c564b", alpha=0.9, label="Extracted")
    ax_er.set_xlabel("Wavelength (um)")
    ax_er.set_ylabel("Re(eps_r)")
    ax_er.grid(alpha=0.25)
    ax_er.legend(fontsize=8)

    ax_ei.plot(wl_um, np.imag(data["eps_ref"]), lw=1.7, color="#17becf", label="Reference")
    ax_ei.plot(wl_um, np.imag(data["eps_ext"]), lw=1.1, color="#bcbd22", alpha=0.9, label="Extracted")
    ax_ei.set_xlabel("Wavelength (um)")
    ax_ei.set_ylabel("Im(eps_r)")
    ax_ei.grid(alpha=0.25)
    ax_ei.legend(fontsize=8)

    rmse_n = float(np.sqrt(np.mean((data["n_ext"] - data["n_ref"]) ** 2)))
    rmse_k = float(np.sqrt(np.mean((data["k_ext"] - data["k_ref"]) ** 2)))
    ax_ei.text(
        0.98,
        0.03,
        f"RMSE(n): {rmse_n:.3g}\nRMSE(k): {rmse_k:.3g}",
        transform=ax_ei.transAxes,
        ha="right",
        va="bottom",
        family="monospace",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    fig.suptitle(case_name)
    fig.subplots_adjust(top=0.9, wspace=0.34, hspace=0.32)

    if SAVE:
        out_path = f"artifacts/dispersion_showcase_{MATERIAL_KEY.lower()}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Saved figure: {out_path}")

    plt.show()


params = get_case_params(MATERIAL_KEY)
material = get_library_material(MATERIAL_KEY)
case_name = material_library[MATERIAL_KEY].name
print(f"Running showcase case: {case_name}")

design_ref, slab_x0 = build_slab_design(MATERIAL_KEY, with_slab=False, p=params)
design_slab, _ = build_slab_design(MATERIAL_KEY, with_slab=True, p=params)

ref = run_design(design_ref, params, slab_x0)
slab = run_design(design_slab, params, slab_x0)

spectral = extract_dispersion(ref, slab, material, params["slab_thickness"])
plot_showcase(case_name, slab["fields"], slab["extent"], spectral)
