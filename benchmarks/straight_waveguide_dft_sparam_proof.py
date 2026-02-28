from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import (
    Design,
    LIGHT_SPEED,
    Material,
    ModeSource,
    Monitor,
    PML,
    PortSpec,
    Rectangle,
    Simulation,
    dxdt,
    µm,
)


def main():
    out_dir = Path("benchmarks/results/gdsf_to_sax_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "straight_wg_dft_sparam_proof.png"

    wl0 = 1.55 * µm
    wl_min, wl_max = 1.52 * µm, 1.58 * µm
    wl_points = 5
    n_core, n_clad = 2.04, 1.444
    wg_width = 0.56 * µm

    dx, dt = dxdt(
        wl0,
        n_max=n_core,
        dims=2,
        safety_factor=0.98,
        points_per_wavelength=12,
    )

    width = 14.0 * µm
    height = 8.0 * µm
    y0 = 0.5 * (height - wg_width)
    design = Design(width=width, height=height, material=Material(n_clad**2))
    design += Rectangle(
        position=(0.0, y0),
        width=width,
        height=wg_width,
        material=Material(n_core**2),
    )

    grid = design.rasterize(resolution=dx)

    freqs = np.linspace(LIGHT_SPEED / wl_max, LIGHT_SPEED / wl_min, wl_points)
    wl_um = (LIGHT_SPEED / freqs) / µm
    fmin = float(np.min(freqs))
    df = float(np.min(np.diff(np.sort(freqs)))) if wl_points > 1 else float(freqs[0])

    pml_left = 1.2 * wl0
    pml_right = 1.5 * wl0
    pml_y = 1.0 * wl0
    source_x = float(pml_left + 1.3 * µm)
    # Keep source-reference monitors one guided wavelength away from the source plane
    # to reduce near-field contamination in incident-wave normalization.
    fwd_x = float(source_x + 0.94 * µm)
    ref_x = float(source_x - 0.94 * µm)
    out_x = float(width - pml_right - 1.0 * µm)

    span = max(1.4 * wg_width, 0.9 * µm)
    y_center = 0.5 * height

    def vline(x_pos):
        return (x_pos, y_center - 0.5 * span), (x_pos, y_center + 0.5 * span)

    ramp_cycles = 12.0
    settle_cycles = 50.0
    dft_cycles = 2.5 / max(df / fmin, 1e-12)
    n_group_est = 1.8
    travel_time = n_group_est * max(out_x - source_x, 0.0) / LIGHT_SPEED
    ramp_time = ramp_cycles / fmin
    dft_t_start = ramp_time + travel_time + settle_cycles / fmin
    dft_t_end = dft_t_start + dft_cycles / fmin
    total_time = dft_t_end + 15.0 / fmin
    time = np.arange(0.0, total_time, dt)

    envelope = 1.0 - np.exp(-((time / max(ramp_time, 1e-30)) ** 2))
    phase_offsets = 2.0 * np.pi * np.arange(wl_points, dtype=float) / max(wl_points, 1)
    signal = envelope * (
        np.sum(
            np.cos(2.0 * np.pi * freqs[:, None] * time[None, :] + phase_offsets[:, None]),
            axis=0,
        )
        / np.sqrt(max(wl_points, 1))
    )

    source = ModeSource(
        grid=grid,
        center=(source_x, y_center),
        width=span,
        wavelength=wl0,
        pol="tm",
        signal=signal,
        direction="+x",
    )

    monitor_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=("Ez", "Hy"),
        dft_window="rect",
        dft_t_start=dft_t_start,
        dft_t_end=dft_t_end,
        dft_record_every_step=True,
    )
    m_fwd = Monitor(*vline(fwd_x), name="o1_fwd", **monitor_cfg)
    m_ref = Monitor(*vline(ref_x), name="o1_ref", **monitor_cfg)
    m_out = Monitor(*vline(out_x), name="o2_out", **monitor_cfg)

    sim = Simulation(
        design=design,
        devices=[source, m_fwd, m_ref, m_out],
        boundaries=[
            PML(edges=["left", "top", "bottom"], thickness=pml_y),
            PML(edges="right", thickness=pml_right),
        ],
        time=time,
        resolution=dx,
    )

    print(
        "Running straight-waveguide DFT proof: "
        f"{wl_points} tones, steps={len(time)}, dx={dx/µm:.4f}um"
    )
    sim.run_fast(progress=False)

    ports = [
        PortSpec(
            name="o1",
            monitor_name="o1_ref",
            reference_monitor="o1_fwd",
            direction="+x",
            polarization="tm",
        ),
        PortSpec(
            name="o2",
            monitor_name="o2_out",
            direction="+x",
            polarization="tm",
        ),
    ]
    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=ports,
        output_ports=["o1", "o2"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-60.0,
    )

    s11 = np.asarray(result["s_matrix"][("o1", "o1")], dtype=np.complex128)
    s21 = np.asarray(result["s_matrix"][("o2", "o1")], dtype=np.complex128)
    valid = np.asarray(result["diagnostics"]["valid_mask"], dtype=bool)
    modal_closure = np.abs(s11) ** 2 + np.abs(s21) ** 2
    modal_loss = 1.0 - modal_closure

    i0 = int(np.argmin(np.abs(wl_um - wl0 / µm)))
    print(
        f"S11 @ {wl_um[i0]:.4f}um: |S11|={np.abs(s11[i0]):.4e}, "
        f"{20*np.log10(max(np.abs(s11[i0]), 1e-12)):.2f} dB"
    )
    print(
        f"S21 @ {wl_um[i0]:.4f}um: |S21|={np.abs(s21[i0]):.6f}, "
        f"{20*np.log10(max(np.abs(s21[i0]), 1e-12)):.3f} dB"
    )
    print(
        f"Modal closure @ {wl_um[i0]:.4f}um: {modal_closure[i0]:.6f}, "
        f"loss={modal_loss[i0]:.3e}, valid={bool(valid[i0])}"
    )
    if np.any(valid):
        vc = modal_closure[valid]
        print(
            "Modal closure over valid bins: "
            f"min={np.min(vc):.6f}, max={np.max(vc):.6f}, mean={np.mean(vc):.6f}"
        )

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.5, 5.6), dpi=240, sharex=True)
    s11_db = np.where(valid, 20 * np.log10(np.maximum(np.abs(s11), 1e-12)), np.nan)
    s21_db = np.where(valid, 20 * np.log10(np.maximum(np.abs(s21), 1e-12)), np.nan)
    ax0.plot(wl_um, s11_db, "o-", lw=1.8, ms=3.5, color="black", label="S11")
    ax0.plot(wl_um, s21_db, "o-", lw=1.8, ms=3.5, color="tab:blue", label="S21")
    ax0.set_ylabel("Magnitude (dB)")
    ax0.set_ylim(-45, 1.5)
    ax0.grid(alpha=0.3)
    ax0.legend(loc="best")
    ax0.set_title("Straight Waveguide DFT Monitor Scaling Proof")

    ax1.plot(
        wl_um,
        modal_closure,
        "o-",
        lw=1.8,
        ms=3.5,
        color="tab:green",
        label="|S11|^2 + |S21|^2 (modal)",
    )
    ax1.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6)
    ax1.set_xlabel("Wavelength (um)")
    ax1.set_ylabel("Power")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=8)
    ax1.set_xlim(np.min(wl_um), np.max(wl_um))

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved proof figure: {out_png}")


if __name__ == "__main__":
    main()
