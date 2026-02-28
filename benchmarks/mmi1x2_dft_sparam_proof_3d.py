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


def make_x_plane(x_pos, y_center, y_span, z_center, z_span, z_min, z_max):
    z0 = max(z_min, z_center - 0.5 * z_span)
    z1 = min(z_max, z_center + 0.5 * z_span)
    if z1 <= z0:
        zmid = 0.5 * (z_min + z_max)
        dz = max(z_max - z_min, 1e-9)
        z0 = zmid - 0.5 * dz
        z1 = zmid + 0.5 * dz
    return (
        (x_pos, y_center - 0.5 * y_span, z0),
        (x_pos, y_center + 0.5 * y_span, z1),
    )


def directional_power_x(monitor, dx, direction):
    sign = 1.0 if str(direction).strip() == "+x" else -1.0
    ey = np.asarray(monitor.get_dft_component("Ey"), dtype=np.complex128)[0]
    ez = np.asarray(monitor.get_dft_component("Ez"), dtype=np.complex128)[0]
    hy = np.asarray(monitor.get_dft_component("Hy"), dtype=np.complex128)[0]
    hz = np.asarray(monitor.get_dft_component("Hz"), dtype=np.complex128)[0]
    s = 0.5 * np.sum(ey * np.conjugate(hz) - ez * np.conjugate(hy)) * (dx * dx)
    return float(sign * np.real(s))


def monitor_inside_non_pml(monitor_bounds, *, width, height, depth, pml_xy, pml_right, pml_z):
    start, end = monitor_bounds
    x0, x1 = sorted([float(start[0]), float(end[0])])
    y0, y1 = sorted([float(start[1]), float(end[1])])
    z0, z1 = sorted([float(start[2]), float(end[2])])
    return (
        x0 >= float(pml_xy)
        and x1 <= float(width - pml_right)
        and y0 >= float(pml_xy)
        and y1 <= float(height - pml_xy)
        and z0 >= float(pml_z)
        and z1 <= float(depth - pml_z)
    )


def main():
    out_dir = Path("benchmarks/results/gdsf_to_sax_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "mmi1x2_dft_sparam_proof_3d.png"
    out_field_png = out_dir / "mmi1x2_dft_sparam_proof_3d_field_slices.png"

    wl0 = 1.55 * µm
    f0 = LIGHT_SPEED / wl0

    n_core, n_clad = 2.04, 1.444
    core_t = 0.45 * µm

    width = 22.0 * µm
    height = 7.0 * µm
    depth = 2.6 * µm
    core_z0 = 0.5 * (depth - core_t)
    core_zc = core_z0 + 0.5 * core_t

    dx, dt = dxdt(
        wl0,
        n_max=n_core,
        dims=3,
        safety_factor=0.96,
        points_per_wavelength=10,
    )

    # Compact 3D 1x2 splitter: input straight + small MMI block + two connected branches.
    y_mid = 0.5 * height
    y_up = y_mid + 0.70 * µm
    y_dn = y_mid - 0.70 * µm
    wg_w = 0.56 * µm
    x_in_end = 7.0 * µm
    x_mmi_end = 11.0 * µm

    design = Design(width=width, height=height, depth=depth, material=Material(n_clad**2))

    design += Rectangle(
        position=(0.0, y_mid - 0.5 * wg_w, core_z0),
        width=x_in_end,
        height=wg_w,
        depth=core_t,
        material=Material(n_core**2),
    )
    design += Rectangle(
        position=(x_in_end, y_mid - 1.15 * µm, core_z0),
        width=(x_mmi_end - x_in_end),
        height=2.3 * µm,
        depth=core_t,
        material=Material(n_core**2),
    )

    design += Rectangle(
        position=(x_mmi_end, y_up - 0.5 * wg_w, core_z0),
        width=width - x_mmi_end,
        height=wg_w,
        depth=core_t,
        material=Material(n_core**2),
    )
    design += Rectangle(
        position=(x_mmi_end, y_dn - 0.5 * wg_w, core_z0),
        width=width - x_mmi_end,
        height=wg_w,
        depth=core_t,
        material=Material(n_core**2),
    )

    grid = design.rasterize(resolution=dx)

    src_w = max(0.95 * µm, 1.7 * wg_w)
    src_h = max(0.80 * µm, 1.8 * core_t)
    source = ModeSource(
        grid=grid,
        center=(2.5 * µm, y_mid, core_zc),
        width=src_w,
        height=src_h,
        wavelength=wl0,
        pol="tm",
        signal=None,
        direction="+x",
    )

    pml_xy = 1.0 * wl0
    pml_right = 1.8 * wl0
    pml_z = 0.5 * wl0
    z_min = pml_z + 0.05 * µm
    z_max = depth - pml_z - 0.05 * µm

    mon_y_span = max(0.90 * µm, 1.6 * wg_w)
    mon_z_span = max(0.85 * µm, 2.0 * core_t)
    fwd_start, fwd_end = make_x_plane(4.0 * µm, y_mid, mon_y_span, core_zc, mon_z_span, z_min, z_max)
    ref_start, ref_end = make_x_plane(2.05 * µm, y_mid, mon_y_span, core_zc, mon_z_span, z_min, z_max)
    o2_start, o2_end = make_x_plane(18.3 * µm, y_up, mon_y_span, core_zc, mon_z_span, z_min, z_max)
    o3_start, o3_end = make_x_plane(18.3 * µm, y_dn, mon_y_span, core_zc, mon_z_span, z_min, z_max)

    t_ramp = 14.0 / f0
    dft_t_start = 60.0 / f0
    dft_t_end = 100.0 / f0
    t_total = 115.0 / f0
    time = np.arange(0.0, t_total, dt)
    envelope = 1.0 - np.exp(-((time / max(t_ramp, 1e-30)) ** 2))
    source.signal = envelope * np.cos(2.0 * np.pi * f0 * time)

    dft_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=[f0],
        dft_components=("Ey", "Ez", "Hy", "Hz"),
        dft_window="hann",
        dft_t_start=dft_t_start,
        dft_t_end=dft_t_end,
        dft_record_every_step=True,
    )
    m_fwd = Monitor(start=fwd_start, end=fwd_end, name="o1_fwd", **dft_cfg)
    m_ref = Monitor(start=ref_start, end=ref_end, name="o1_ref", **dft_cfg)
    m_o2 = Monitor(start=o2_start, end=o2_end, name="o2_out", **dft_cfg)
    m_o3 = Monitor(start=o3_start, end=o3_end, name="o3_out", **dft_cfg)

    flux_y_span = max(1.0 * µm, height - 2.0 * pml_xy - 0.15 * µm)
    flux_z_span = max(0.25 * µm, depth - 2.0 * pml_z - 0.10 * µm)
    in_flux_start, in_flux_end = make_x_plane(
        4.0 * µm, y_mid, flux_y_span, core_zc, flux_z_span, z_min, z_max
    )
    ref_flux_start, ref_flux_end = make_x_plane(
        2.05 * µm, y_mid, flux_y_span, core_zc, flux_z_span, z_min, z_max
    )
    out_flux_start, out_flux_end = make_x_plane(
        18.3 * µm, y_mid, flux_y_span, core_zc, flux_z_span, z_min, z_max
    )
    m_in_flux = Monitor(start=in_flux_start, end=in_flux_end, name="in_flux", **dft_cfg)
    m_ref_flux = Monitor(start=ref_flux_start, end=ref_flux_end, name="ref_flux", **dft_cfg)
    m_out_flux = Monitor(start=out_flux_start, end=out_flux_end, name="out_flux", **dft_cfg)

    sim = Simulation(
        design=design,
        devices=[source, m_fwd, m_ref, m_o2, m_o3, m_in_flux, m_ref_flux, m_out_flux],
        boundaries=[
            PML(edges=["left", "top", "bottom"], thickness=pml_xy),
            PML(edges="right", thickness=pml_right),
            PML(edges=["front", "back"], thickness=pml_z),
        ],
        time=time,
        resolution=dx,
    )

    print(
        "Running compact 3D splitter DFT proof: "
        f"steps={len(time)}, dx={dx/µm:.4f}um, "
        f"domain=({width/µm:.2f},{height/µm:.2f},{depth/µm:.2f})um"
    )
    for name, bounds in [
        ("o1_fwd", (fwd_start, fwd_end)),
        ("o1_ref", (ref_start, ref_end)),
        ("o2_out", (o2_start, o2_end)),
        ("o3_out", (o3_start, o3_end)),
    ]:
        print(
            f"Monitor '{name}' inside non-PML: "
            f"{monitor_inside_non_pml(bounds, width=width, height=height, depth=depth, pml_xy=pml_xy, pml_right=pml_right, pml_z=pml_z)}"
        )

    rec_interval = max(1, len(time) // 6)
    run_result = sim.run_compiled(
        num_steps=len(time),
        record_interval=rec_interval,
        record_fields=["Ez"],
        progress=False,
    )

    specs = [
        PortSpec(name="o1_fwd", monitor_name="o1_fwd", direction="+x", polarization="tm"),
        PortSpec(name="o1_ref", monitor_name="o1_ref", direction="+x", polarization="tm"),
        PortSpec(name="o2_out", monitor_name="o2_out", direction="-x", polarization="tm"),
        PortSpec(name="o3_out", monitor_name="o3_out", direction="-x", polarization="tm"),
    ]
    waves = sim.extract_port_waves_dft(ports=specs, frequencies=[f0], return_power=True)

    a_fwd_plus = complex(np.asarray(waves["o1_fwd"]["a_plus"], dtype=np.complex128)[0])
    a_fwd_minus = complex(np.asarray(waves["o1_fwd"]["a_minus"], dtype=np.complex128)[0])
    if abs(a_fwd_plus) >= abs(a_fwd_minus):
        inc_key, opp_key, a_incident = "a_plus", "a_minus", a_fwd_plus
    else:
        inc_key, opp_key, a_incident = "a_minus", "a_plus", a_fwd_minus

    denom = max(abs(a_incident), 1e-30)
    s11 = complex(np.asarray(waves["o1_ref"][opp_key], dtype=np.complex128)[0] / denom)
    s21 = complex(np.asarray(waves["o2_out"][opp_key], dtype=np.complex128)[0] / denom)
    s31 = complex(np.asarray(waves["o3_out"][opp_key], dtype=np.complex128)[0] / denom)
    s21_same = complex(np.asarray(waves["o2_out"][inc_key], dtype=np.complex128)[0] / denom)
    s31_same = complex(np.asarray(waves["o3_out"][inc_key], dtype=np.complex128)[0] / denom)

    p11, p21, p31 = abs(s11) ** 2, abs(s21) ** 2, abs(s31) ** 2
    closure = p11 + p21 + p31
    split_sum = max(p21 + p31, 1e-30)
    ratio_o2 = p21 / split_sum
    ratio_o3 = p31 / split_sum
    bal_db = abs(
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
        f"o2={ratio_o2:.3f}, o3={ratio_o3:.3f}, balance={bal_db:.3f} dB"
    )
    print(
        "Incident-wave normalization: "
        f"inc_key={inc_key}, |a_plus_fwd|={abs(a_fwd_plus):.3e}, "
        f"|a_minus_fwd|={abs(a_fwd_minus):.3e}, |a_inc|={abs(a_incident):.3e}"
    )
    print(
        "Output wave split: "
        f"o2(|a+|,|a-|)=({abs(complex(np.asarray(waves['o2_out']['a_plus'])[0])):.3e},"
        f"{abs(complex(np.asarray(waves['o2_out']['a_minus'])[0])):.3e}), "
        f"o3(|a+|,|a-|)=({abs(complex(np.asarray(waves['o3_out']['a_plus'])[0])):.3e},"
        f"{abs(complex(np.asarray(waves['o3_out']['a_minus'])[0])):.3e})"
    )
    print(
        "Output normalization sanity: "
        f"|S21(opp)|={abs(s21):.6f}, |S21(same)|={abs(s21_same):.6f}, "
        f"|S31(opp)|={abs(s31):.6f}, |S31(same)|={abs(s31_same):.6f}"
    )

    p_in_flux = directional_power_x(m_in_flux, dx=dx, direction="+x")
    p_ref_flux = directional_power_x(m_ref_flux, dx=dx, direction="-x")
    p_out_flux = directional_power_x(m_out_flux, dx=dx, direction="+x")
    flux_closure = (p_ref_flux + p_out_flux) / max(p_in_flux, 1e-30)
    print(
        "Total-flux closure (wide DFT planes): "
        f"R+T={flux_closure:.6f} "
        f"(R={p_ref_flux/max(p_in_flux,1e-30):.6f}, "
        f"T={p_out_flux/max(p_in_flux,1e-30):.6f})"
    )

    ez_snap = np.asarray(sim.fields.Ez, dtype=float)
    if isinstance(run_result, dict):
        ez_hist = np.asarray(run_result.get("fields", {}).get("Ez", np.zeros((0,))), dtype=float)
        if ez_hist.ndim == 4 and ez_hist.shape[0] > 0:
            peak_idx = int(np.argmax(np.max(np.abs(ez_hist), axis=(1, 2, 3))))
            ez_snap = np.asarray(ez_hist[peak_idx], dtype=float)

    eps = np.asarray(grid.permittivity, dtype=float)
    z_idx = int(np.clip(round(core_zc / dx), 0, eps.shape[0] - 1))
    y_idx = int(np.clip(round(y_mid / dx), 0, eps.shape[1] - 1))
    eps_xy = eps[z_idx]
    eps_xz = eps[:, y_idx, :]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), dpi=260)

    im0 = axes[0, 0].imshow(
        eps_xy,
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=[0.0, width / µm, 0.0, height / µm],
    )
    axes[0, 0].set_title("XY Slice (core plane)")
    axes[0, 0].set_xlabel("x (um)")
    axes[0, 0].set_ylabel("y (um)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    for name, (s, e), color in [
        ("o1_fwd", (fwd_start, fwd_end), "white"),
        ("o1_ref", (ref_start, ref_end), "cyan"),
        ("o2_out", (o2_start, o2_end), "orange"),
        ("o3_out", (o3_start, o3_end), "orange"),
    ]:
        x_um = 0.5 * (s[0] + e[0]) / µm
        y0_um = min(s[1], e[1]) / µm
        y1_um = max(s[1], e[1]) / µm
        axes[0, 0].plot([x_um, x_um], [y0_um, y1_um], color=color, lw=1.6)
        axes[0, 0].text(x_um, y1_um + 0.06, name, color=color, fontsize=7, ha="center")

    im1 = axes[0, 1].imshow(
        eps_xz,
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=[0.0, width / µm, 0.0, depth / µm],
    )
    axes[0, 1].set_title("XZ Slice (center y)")
    axes[0, 1].set_xlabel("x (um)")
    axes[0, 1].set_ylabel("z (um)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    for (s, e), color in [
        ((fwd_start, fwd_end), "white"),
        ((ref_start, ref_end), "cyan"),
        ((o2_start, o2_end), "orange"),
        ((o3_start, o3_end), "orange"),
    ]:
        x_um = 0.5 * (s[0] + e[0]) / µm
        z0_um = min(s[2], e[2]) / µm
        z1_um = max(s[2], e[2]) / µm
        axes[0, 1].plot([x_um, x_um], [z0_um, z1_um], color=color, lw=1.3)

    labels = ["|S11|^2", "|S21|^2", "|S31|^2", "closure"]
    vals = [p11, p21, p31, closure]
    colors = ["black", "tab:blue", "tab:green", "tab:red"]
    axes[1, 0].bar(labels, vals, color=colors, alpha=0.88)
    axes[1, 0].axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.7)
    axes[1, 0].set_ylabel("Power")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].set_title("3D Modal DFT Power Metrics")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.96,
        (
            f"Compact 3D splitter @ {wl0/µm:.4f} um\n"
            f"|S11| = {abs(s11):.6f} ({20*np.log10(max(abs(s11),1e-12)):.2f} dB)\n"
            f"|S21| = {abs(s21):.6f} ({20*np.log10(max(abs(s21),1e-12)):.2f} dB)\n"
            f"|S31| = {abs(s31):.6f} ({20*np.log10(max(abs(s31),1e-12)):.2f} dB)\n"
            f"closure = {closure:.6f}\n"
            f"wide-plane flux R+T = {flux_closure:.6f}\n"
            f"split o2/o3 = {ratio_o2:.3f}/{ratio_o3:.3f}\n"
            f"balance = {bal_db:.3f} dB\n"
            f"inc_key = {inc_key}, |a_inc|={abs(a_incident):.3e}\n"
            f"dx = {dx/µm:.4f} um, steps = {len(time)}"
        ),
        fontsize=9,
        va="top",
        ha="left",
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)
    print(f"Saved proof figure: {out_png}")

    x_idx = int(np.clip(round((x_mmi_end + 1.0 * µm) / dx), 0, ez_snap.shape[2] - 1))
    ez_xy = ez_snap[z_idx, :, :]
    ez_xz = ez_snap[:, y_idx, :]
    ez_yz = ez_snap[:, :, x_idx]

    fig2, ax2 = plt.subplots(1, 3, figsize=(10.0, 3.2), dpi=260)
    im_xy = ax2[0].imshow(
        ez_xy,
        origin="lower",
        cmap="RdBu",
        aspect="auto",
        extent=[0.0, width / µm, 0.0, height / µm],
    )
    ax2[0].set_title(f"Ez XY @ z={z_idx}")
    ax2[0].set_xlabel("x (um)")
    ax2[0].set_ylabel("y (um)")
    fig2.colorbar(im_xy, ax=ax2[0], fraction=0.046, pad=0.04)

    im_xz = ax2[1].imshow(
        ez_xz,
        origin="lower",
        cmap="RdBu",
        aspect="auto",
        extent=[0.0, width / µm, 0.0, depth / µm],
    )
    ax2[1].set_title(f"Ez XZ @ y={y_idx}")
    ax2[1].set_xlabel("x (um)")
    ax2[1].set_ylabel("z (um)")
    fig2.colorbar(im_xz, ax=ax2[1], fraction=0.046, pad=0.04)

    im_yz = ax2[2].imshow(
        ez_yz,
        origin="lower",
        cmap="RdBu",
        aspect="auto",
        extent=[0.0, height / µm, 0.0, depth / µm],
    )
    ax2[2].set_title(f"Ez YZ @ x={x_idx}")
    ax2[2].set_xlabel("y (um)")
    ax2[2].set_ylabel("z (um)")
    fig2.colorbar(im_yz, ax=ax2[2], fraction=0.046, pad=0.04)

    fig2.tight_layout()
    fig2.savefig(out_field_png, dpi=320)
    plt.close(fig2)
    print(f"Saved field-slice figure: {out_field_png}")


if __name__ == "__main__":
    main()
