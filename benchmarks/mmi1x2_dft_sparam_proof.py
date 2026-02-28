from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import (
    LIGHT_SPEED,
    ModeSource,
    Monitor,
    PML,
    PortSpec,
    Simulation,
    dxdt,
    µm,
)
from beamz.design.io import gdsf


def _monitor_line(ax, x0, y0, y1, label, color):
    ax.plot([x0, x0], [y0, y1], color=color, lw=1.3, alpha=0.95)
    ax.text(x0, y1 + 0.05, label, color=color, fontsize=7, ha="center", va="bottom")


def main():
    out_dir = Path("benchmarks/results/gdsf_to_sax_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "mmi1x2_dft_sparam_proof.png"

    wl0 = 1.55 * µm
    f0 = LIGHT_SPEED / wl0
    n_core, n_clad = 2.04, 1.444

    # Import a compact, deterministic 1x2 MMI splitter from gdsfactory.
    design, ports = gdsf.load(
        "mmi1x2",
        layer=(1, 0),
        n_core=n_core,
        n_clad=n_clad,
        padding=2.0,
    )
    dx, dt = dxdt(
        wl0,
        n_max=n_core,
        dims=2,
        safety_factor=0.98,
        points_per_wavelength=12,
    )
    grid = design.rasterize(resolution=dx)

    y_in = float(ports["o1"]["center"][1])
    y_o2 = float(ports["o2"]["center"][1])
    y_o3 = float(ports["o3"]["center"][1])
    source_x = float(ports["o1"]["center"][0] + 1.5 * µm)
    out_mon_x = float(ports["o2"]["center"][0] - 0.6 * µm)
    d_ref = 1.0 * µm
    span = 0.9 * µm

    def vline(x_pos, y_ctr):
        return (x_pos, y_ctr - 0.5 * span), (x_pos, y_ctr + 0.5 * span)

    t_total = 130.0 / f0
    time = np.arange(0.0, t_total, dt)
    ramp = 20.0 / f0
    signal = (1.0 - np.exp(-((time / ramp) ** 2))) * np.cos(2.0 * np.pi * f0 * time)

    source = ModeSource(
        grid=grid,
        center=(source_x, y_in),
        width=span,
        wavelength=wl0,
        pol="tm",
        signal=signal,
        direction="+x",
    )

    dft_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=[f0],
        dft_components=("Ez", "Hy"),
        dft_window="rect",
        dft_t_start=85.0 / f0,
        dft_t_end=125.0 / f0,
        dft_record_every_step=True,
    )
    m_fwd = Monitor(*vline(source_x + d_ref, y_in), name="o1_fwd", **dft_cfg)
    m_ref = Monitor(*vline(source_x - d_ref, y_in), name="o1_ref", **dft_cfg)
    m_o2 = Monitor(*vline(out_mon_x, y_o2), name="o2_out", **dft_cfg)
    m_o3 = Monitor(*vline(out_mon_x, y_o3), name="o3_out", **dft_cfg)

    sim = Simulation(
        design=design,
        devices=[source, m_fwd, m_ref, m_o2, m_o3],
        boundaries=[
            PML(edges=["left", "right"], thickness=1.2 * wl0),
            PML(edges=["top", "bottom"], thickness=1.0 * wl0),
        ],
        time=time,
        resolution=dx,
    )

    print(
        "Running MMI1x2 DFT proof: "
        f"steps={len(time)}, dx={dx/µm:.4f}um, wl0={wl0/µm:.4f}um"
    )
    sim.run_fast(progress=False)

    portspecs = [
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
        PortSpec(
            name="o3",
            monitor_name="o3_out",
            direction="+x",
            polarization="tm",
        ),
    ]
    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=portspecs,
        output_ports=["o1", "o2", "o3"],
        frequencies=[f0],
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-80.0,
    )

    s11 = complex(result["s_matrix"][("o1", "o1")][0])
    s21 = complex(result["s_matrix"][("o2", "o1")][0])
    s31 = complex(result["s_matrix"][("o3", "o1")][0])

    p11 = abs(s11) ** 2
    p21 = abs(s21) ** 2
    p31 = abs(s31) ** 2
    closure = p11 + p21 + p31
    split_sum = max(p21 + p31, 1e-30)
    ratio_o2 = p21 / split_sum
    ratio_o3 = p31 / split_sum
    balance_db = abs(
        20.0 * np.log10(max(abs(s21), 1e-12)) - 20.0 * np.log10(max(abs(s31), 1e-12))
    )

    print(
        f"S11 @ {wl0/µm:.4f}um: |S11|={abs(s11):.6f}, "
        f"{20*np.log10(max(abs(s11),1e-12)):.2f} dB"
    )
    print(
        f"S21 @ {wl0/µm:.4f}um: |S21|={abs(s21):.6f}, "
        f"{20*np.log10(max(abs(s21),1e-12)):.2f} dB"
    )
    print(
        f"S31 @ {wl0/µm:.4f}um: |S31|={abs(s31):.6f}, "
        f"{20*np.log10(max(abs(s31),1e-12)):.2f} dB"
    )
    print(
        f"Power closure: {closure:.6f}, split ratio: "
        f"o2={ratio_o2:.3f}, o3={ratio_o3:.3f}, balance={balance_db:.3f} dB"
    )

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.2, 6.0), dpi=240)
    eps = np.asarray(grid.permittivity, dtype=float)
    extent = [0.0, design.width / µm, 0.0, design.height / µm]
    ax0.imshow(
        eps,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax0.set_title("MMI1x2 Geometry and DFT Monitor Placement")
    ax0.set_xlabel("x (um)")
    ax0.set_ylabel("y (um)")

    y0, y1 = (y_in - 0.5 * span) / µm, (y_in + 0.5 * span) / µm
    _monitor_line(ax0, (source_x + d_ref) / µm, y0, y1, "o1_fwd", "white")
    _monitor_line(ax0, (source_x - d_ref) / µm, y0, y1, "o1_ref", "cyan")
    _monitor_line(
        ax0,
        out_mon_x / µm,
        (y_o2 - 0.5 * span) / µm,
        (y_o2 + 0.5 * span) / µm,
        "o2_out",
        "orange",
    )
    _monitor_line(
        ax0,
        out_mon_x / µm,
        (y_o3 - 0.5 * span) / µm,
        (y_o3 + 0.5 * span) / µm,
        "o3_out",
        "orange",
    )

    labels = ["|S11|^2", "|S21|^2", "|S31|^2", "closure"]
    vals = [p11, p21, p31, closure]
    colors = ["black", "tab:blue", "tab:green", "tab:red"]
    ax1.bar(labels, vals, color=colors, alpha=0.85)
    ax1.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.7)
    ax1.set_ylabel("Power")
    ax1.set_title(
        f"MMI1x2 Modal DFT Metrics @ {wl0/µm:.4f}um "
        f"(split o2/o3 = {ratio_o2:.3f}/{ratio_o3:.3f}, Δ={balance_db:.3f} dB)"
    )
    ax1.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved proof figure: {out_png}")


if __name__ == "__main__":
    main()
