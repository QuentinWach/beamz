from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import Monitor, Simulation


def run_dft_component_recovery():
    n = 4096
    dt = 1.0
    bins = np.array([7, 13, 31, 64, 97, 131, 173, 257], dtype=int)
    freqs = bins / (n * dt)
    rng = np.random.default_rng(7)
    amps = (0.2 + 0.8 * rng.random(freqs.size)) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, freqs.size)
    )

    t = np.arange(n, dtype=float) * dt
    signal = np.zeros((n,), dtype=float)
    for amp, f in zip(amps, freqs):
        signal += np.real(amp * np.exp(1j * 2.0 * np.pi * f * t))

    mon = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 0.0),
        name="proof_dft",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_t_start=float(t[0]),
        dft_t_end=float(t[-1]),
        dft_window="rect",
        dft_components=("Ez",),
    )

    zeros = np.zeros((1, 1), dtype=float)
    for i, ti in enumerate(t):
        mon.record_fields_2d(
            Ez=np.array([[signal[i]]], dtype=float),
            Hx=zeros,
            Hy=zeros,
            t=float(ti),
            dx=1.0,
            dy=1.0,
            step=i,
        )

    recovered = np.asarray(mon.get_dft_component("Ez"), dtype=np.complex128)[:, 0]
    mag_true = np.abs(amps)
    mag_rec = np.abs(recovered)
    phase_true = np.unwrap(np.angle(amps))
    phase_rec = np.unwrap(np.angle(recovered))

    rel_mag_err = np.abs(mag_rec - mag_true) / np.maximum(mag_true, 1e-30)
    phase_err = np.abs(np.angle(recovered / amps))
    return {
        "bins": bins,
        "mag_true": mag_true,
        "mag_rec": mag_rec,
        "phase_true": phase_true,
        "phase_rec": phase_rec,
        "rel_mag_err": rel_mag_err,
        "phase_err": phase_err,
    }


def run_modal_projection_recovery():
    rng = np.random.default_rng(11)
    npoints = 96
    mode_components = {
        "Ex": np.zeros((npoints,), dtype=np.complex128),
        "Ey": rng.normal(size=npoints) + 1j * rng.normal(size=npoints),
        "Ez": rng.normal(size=npoints) + 1j * rng.normal(size=npoints),
        "Hx": np.zeros((npoints,), dtype=np.complex128),
        "Hy": rng.normal(size=npoints) + 1j * rng.normal(size=npoints),
        "Hz": rng.normal(size=npoints) + 1j * rng.normal(size=npoints),
    }
    projection = {
        "axis": "x",
        "d_area": 1.0,
        "components": ("Ey", "Ez", "Hy", "Hz"),
        "mode_components": mode_components,
    }

    ncases = 250
    a_true = (rng.normal(size=ncases) + 1j * rng.normal(size=ncases)) * 0.6
    b_true = (rng.normal(size=ncases) + 1j * rng.normal(size=ncases)) * 0.25
    a_rec = np.zeros((ncases,), dtype=np.complex128)
    b_rec = np.zeros((ncases,), dtype=np.complex128)

    for i in range(ncases):
        field_components = {}
        for name, vec in mode_components.items():
            if name.startswith("H"):
                field_components[name] = a_true[i] * vec - b_true[i] * vec
            else:
                field_components[name] = a_true[i] * vec + b_true[i] * vec
        a_i, b_i = Simulation._project_modal_coefficients_3d(
            field_components, projection, apply_calibration=False
        )
        a_rec[i] = a_i
        b_rec[i] = b_i

    rel_err_a = np.abs(a_rec - a_true) / np.maximum(np.abs(a_true), 1e-30)
    rel_err_b = np.abs(b_rec - b_true) / np.maximum(np.abs(b_true), 1e-30)
    return {
        "a_true": a_true,
        "b_true": b_true,
        "a_rec": a_rec,
        "b_rec": b_rec,
        "rel_err_a": rel_err_a,
        "rel_err_b": rel_err_b,
    }


def make_figure(dft_data, modal_data, out_path: Path):
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.2), dpi=220)

    k = dft_data["bins"]
    axs[0, 0].plot(k, dft_data["mag_true"], "o-", lw=1.8, ms=4, label="target |A|")
    axs[0, 0].plot(k, dft_data["mag_rec"], "s--", lw=1.5, ms=3, label="recovered |A|")
    axs[0, 0].set_title("DFT Monitor Amplitude Recovery")
    axs[0, 0].set_xlabel("FFT bin")
    axs[0, 0].set_ylabel("Magnitude")
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend(loc="best", fontsize=8)

    axs[0, 1].plot(k, dft_data["phase_true"], "o-", lw=1.8, ms=4, label="target phase")
    axs[0, 1].plot(k, dft_data["phase_rec"], "s--", lw=1.5, ms=3, label="recovered phase")
    axs[0, 1].set_title("DFT Monitor Phase Recovery")
    axs[0, 1].set_xlabel("FFT bin")
    axs[0, 1].set_ylabel("Phase (rad)")
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].legend(loc="best", fontsize=8)

    a_true = modal_data["a_true"]
    a_rec = modal_data["a_rec"]
    b_true = modal_data["b_true"]
    b_rec = modal_data["b_rec"]
    axs[1, 0].scatter(np.real(a_true), np.real(a_rec), s=10, alpha=0.75, label="a+ (real)")
    axs[1, 0].scatter(np.real(b_true), np.real(b_rec), s=10, alpha=0.75, label="a- (real)")
    xlim = axs[1, 0].get_xlim()
    ylim = axs[1, 0].get_ylim()
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    axs[1, 0].plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.7)
    axs[1, 0].set_xlim(lo, hi)
    axs[1, 0].set_ylim(lo, hi)
    axs[1, 0].set_title("3D Modal Extraction: True vs Recovered")
    axs[1, 0].set_xlabel("True coefficient (real)")
    axs[1, 0].set_ylabel("Recovered coefficient (real)")
    axs[1, 0].grid(alpha=0.3)
    axs[1, 0].legend(loc="best", fontsize=8)

    rel_err_a = modal_data["rel_err_a"]
    rel_err_b = modal_data["rel_err_b"]
    axs[1, 1].hist(np.log10(np.maximum(rel_err_a, 1e-18)), bins=24, alpha=0.65, label="a+")
    axs[1, 1].hist(np.log10(np.maximum(rel_err_b, 1e-18)), bins=24, alpha=0.65, label="a-")
    axs[1, 1].set_title("Relative Error Histogram (log10)")
    axs[1, 1].set_xlabel("log10(relative error)")
    axs[1, 1].set_ylabel("Count")
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend(loc="best", fontsize=8)

    max_mag_rel = float(np.max(dft_data["rel_mag_err"]))
    max_phase_err = float(np.max(dft_data["phase_err"]))
    max_modal_rel = float(max(np.max(rel_err_a), np.max(rel_err_b)))
    fig.suptitle(
        "DFT Monitor + 3D Modal Extraction Proof\n"
        f"max rel |A| error={max_mag_rel:.2e}, "
        f"max phase error={max_phase_err:.2e} rad, "
        f"max modal rel error={max_modal_rel:.2e}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    out_path = Path("benchmarks/results/gdsf_to_sax_debug/dft_monitor_perfect_proof.png")
    dft_data = run_dft_component_recovery()
    modal_data = run_modal_projection_recovery()
    make_figure(dft_data, modal_data, out_path=out_path)

    print(f"Saved proof figure: {out_path}")
    print(f"Max DFT relative magnitude error: {np.max(dft_data['rel_mag_err']):.3e}")
    print(f"Max DFT phase error (rad): {np.max(dft_data['phase_err']):.3e}")
    print(
        "Max modal relative error: "
        f"{max(np.max(modal_data['rel_err_a']), np.max(modal_data['rel_err_b'])):.3e}"
    )


if __name__ == "__main__":
    main()
