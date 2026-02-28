"""Minimal 3D gdsfactory MMI1x2 modal S-parameter extraction example.

This example is intentionally compact/fast so it can be used as a working baseline
for 3D DFT monitors + modal S extraction on gdsfactory geometry.
"""

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
from beamz.design.io import gdsf


def outward_direction(direction: str) -> str:
    return ("-" if direction.startswith("+") else "+") + direction[1:]


def move_along(center_xy, direction: str, distance: float):
    if direction == "+x":
        return (center_xy[0] + distance, center_xy[1])
    if direction == "-x":
        return (center_xy[0] - distance, center_xy[1])
    if direction == "+y":
        return (center_xy[0], center_xy[1] + distance)
    if direction == "-y":
        return (center_xy[0], center_xy[1] - distance)
    raise ValueError(f"Unsupported direction {direction!r}")


def make_x_plane(x_pos, y_center, y_span, z_center, z_span, z_min, z_max):
    z0 = max(z_min, z_center - 0.5 * z_span)
    z1 = min(z_max, z_center + 0.5 * z_span)
    if z1 <= z0:
        zmid = 0.5 * (z_min + z_max)
        dz = max(z_max - z_min, 1e-12)
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


def draw_pml_lines_xy(ax, *, width, height, pml_xy, pml_right, color="white"):
    ax.axvline(float(pml_xy) / µm, color=color, lw=1.0, ls="--", alpha=0.85)
    ax.axvline(float(width - pml_right) / µm, color=color, lw=1.0, ls="--", alpha=0.85)
    ax.axhline(float(pml_xy) / µm, color=color, lw=1.0, ls="--", alpha=0.85)
    ax.axhline(float(height - pml_xy) / µm, color=color, lw=1.0, ls="--", alpha=0.85)


def draw_pml_lines_xz(ax, *, width, depth, pml_xy, pml_right, pml_z, color="white"):
    ax.axvline(float(pml_xy) / µm, color=color, lw=1.0, ls="--", alpha=0.85)
    ax.axvline(float(width - pml_right) / µm, color=color, lw=1.0, ls="--", alpha=0.85)
    ax.axhline(float(pml_z) / µm, color=color, lw=1.0, ls="--", alpha=0.85)
    ax.axhline(float(depth - pml_z) / µm, color=color, lw=1.0, ls="--", alpha=0.85)


def monitor_overlap_stats(eps_grid, monitor_obj, dx, dy, dz, *, eps_core, eps_clad):
    z_idx, y_idx, x_idx = monitor_obj.get_grid_slice_3d(dx, dy, dz, eps_grid.shape)
    eps_roi = np.asarray(eps_grid[z_idx, y_idx, x_idx], dtype=float)
    vals = eps_roi.reshape(-1)
    if vals.size == 0:
        return 0.0, 0.0, float("nan")
    core_frac = float(np.mean(np.abs(vals - eps_core) <= np.abs(vals - eps_clad)))
    clad_frac = float(np.mean(np.abs(vals - eps_clad) < np.abs(vals - eps_core)))
    return core_frac, clad_frac, float(np.max(vals))


def main():
    out_dir = Path("benchmarks/results/gdsf_to_sax_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "gdsf_mmi1x2_3d_sparams.png"
    out_field_png = out_dir / "gdsf_mmi1x2_3d_field_slices.png"

    wl0 = 1.55 * µm
    f0 = LIGHT_SPEED / wl0

    # Use moderate index contrast and resolution to keep this example lightweight.
    n_core, n_clad = 2.04, 1.444
    core_t = 0.45 * µm

    input_extension = 2.0 * µm
    output_extension = 4.0 * µm
    y_margin = 1.8 * µm

    clad_below = 1.1 * µm
    clad_above = 1.1 * µm
    depth = clad_below + core_t + clad_above
    core_z0 = clad_below
    core_zc = core_z0 + 0.5 * core_t

    dx, dt = dxdt(
        wl0,
        n_max=n_core,
        dims=3,
        safety_factor=0.96,
        points_per_wavelength=8,
    )

    imported_design, ports = gdsf.load(
        "mmi1x2",
        n_core=n_core,
        n_clad=n_clad,
        layer=(1, 0),
        padding=0.0,
    )

    design = Design(
        width=imported_design.width + input_extension + output_extension,
        height=imported_design.height + 2.0 * y_margin,
        depth=depth,
        material=Material(n_clad**2),
    )

    # Extrude imported 2D geometry into a 3D core layer.
    for structure in imported_design.structures[1:]:
        shifted = structure.copy().shift(input_extension, y_margin, core_z0)
        shifted.z = core_z0
        shifted.depth = core_t
        design += shifted

    ports = {
        name: {
            **p,
            "center": (
                float(p["center"][0] + input_extension),
                float(p["center"][1] + y_margin),
            ),
            "width": float(p["width"]),
        }
        for name, p in ports.items()
    }

    # Extend all ports outward to create uniform monitor/source sections.
    for name, port in ports.items():
        cx, cy = port["center"]
        w = float(port["width"])
        ext = input_extension if name == "o1" else output_extension
        d_out = outward_direction(port["direction"])
        ox, oy = move_along((cx, cy), d_out, ext)
        if d_out.endswith("x"):
            design += Rectangle(
                position=(min(cx, ox), cy - 0.5 * w, core_z0),
                width=abs(ox - cx),
                height=w,
                depth=core_t,
                material=Material(n_core**2),
            )
        else:
            design += Rectangle(
                position=(cx - 0.5 * w, min(cy, oy), core_z0),
                width=w,
                height=abs(oy - cy),
                depth=core_t,
                material=Material(n_core**2),
            )

    grid = design.rasterize(resolution=dx)

    pml_xy = 1.0 * wl0
    pml_right = 1.5 * wl0
    pml_z = 0.5 * wl0
    z_min = pml_z + 0.05 * µm
    z_max = depth - pml_z - 0.05 * µm

    src = ports["o1"]
    src_w = max(0.70 * µm, 1.2 * src["width"])
    src_h = max(0.50 * µm, 1.2 * core_t)
    # Keep source safely inside the non-PML interior.
    source_x_nominal = move_along(src["center"], src["direction"], -0.90 * µm)[0]
    source_x = max(float(source_x_nominal), float(pml_xy + 0.60 * µm))
    source_xy = (source_x, float(src["center"][1]))
    source = ModeSource(
        grid=grid,
        center=(source_xy[0], source_xy[1], core_zc),
        width=src_w,
        height=src_h,
        wavelength=wl0,
        pol="te",
        signal=None,
        direction=src["direction"],
    )

    mon_y_span = max(0.62 * µm, 1.15 * src["width"])
    mon_z_span = max(0.55 * µm, 1.25 * core_t)

    fwd_xy = move_along(src["center"], src["direction"], +0.70 * µm)
    ref_xy = move_along(src["center"], src["direction"], -0.20 * µm)
    fwd_start, fwd_end = make_x_plane(
        fwd_xy[0], fwd_xy[1], mon_y_span, core_zc, mon_z_span, z_min, z_max
    )
    ref_start, ref_end = make_x_plane(
        ref_xy[0], ref_xy[1], mon_y_span, core_zc, mon_z_span, z_min, z_max
    )

    out_planes = {}
    for name in ("o2", "o3"):
        p = ports[name]
        d_out = outward_direction(p["direction"])
        mon_xy = move_along(p["center"], d_out, +0.4 * µm)
        out_planes[name] = make_x_plane(
            mon_xy[0], mon_xy[1], mon_y_span, core_zc, mon_z_span, z_min, z_max
        )

    t_ramp = 12.0 / f0
    dft_t_start = 52.0 / f0
    dft_t_end = 88.0 / f0
    t_total = 98.0 / f0
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
    m_o2 = Monitor(start=out_planes["o2"][0], end=out_planes["o2"][1], name="o2_out", **dft_cfg)
    m_o3 = Monitor(start=out_planes["o3"][0], end=out_planes["o3"][1], name="o3_out", **dft_cfg)

    flux_y_span = max(1.0 * µm, design.height - 2.0 * pml_xy - 0.15 * µm)
    flux_z_span = max(0.25 * µm, design.depth - 2.0 * pml_z - 0.10 * µm)
    in_flux_start, in_flux_end = make_x_plane(
        fwd_xy[0], src["center"][1], flux_y_span, core_zc, flux_z_span, z_min, z_max
    )
    ref_flux_start, ref_flux_end = make_x_plane(
        ref_xy[0], src["center"][1], flux_y_span, core_zc, flux_z_span, z_min, z_max
    )
    out_flux_x = 0.5 * (out_planes["o2"][0][0] + out_planes["o3"][0][0])
    out_flux_start, out_flux_end = make_x_plane(
        out_flux_x, src["center"][1], flux_y_span, core_zc, flux_z_span, z_min, z_max
    )
    m_in_flux = Monitor(start=in_flux_start, end=in_flux_end, name="in_flux", **dft_cfg)
    m_ref_flux = Monitor(start=ref_flux_start, end=ref_flux_end, name="ref_flux", **dft_cfg)
    m_out_flux = Monitor(start=out_flux_start, end=out_flux_end, name="out_flux", **dft_cfg)

    print(
        "Running 3D gdsfactory MMI1x2: "
        f"steps={len(time)}, dx={dx/µm:.4f}um, "
        f"domain=({design.width/µm:.2f},{design.height/µm:.2f},{design.depth/µm:.2f})um"
    )

    for name, bounds in [
        ("o1_fwd", (fwd_start, fwd_end)),
        ("o1_ref", (ref_start, ref_end)),
        ("o2_out", out_planes["o2"]),
        ("o3_out", out_planes["o3"]),
    ]:
        inside = monitor_inside_non_pml(
            bounds,
            width=design.width,
            height=design.height,
            depth=design.depth,
            pml_xy=pml_xy,
            pml_right=pml_right,
            pml_z=pml_z,
        )
        print(f"Monitor '{name}' inside non-PML: {inside}")
    src_x, src_y, src_z = source.center
    src_inside_non_pml = (
        float(src_x) >= float(pml_xy)
        and float(src_x) <= float(design.width - pml_right)
        and float(src_y - 0.5 * source.width) >= float(pml_xy)
        and float(src_y + 0.5 * source.width) <= float(design.height - pml_xy)
        and float(src_z - 0.5 * source.height) >= float(pml_z)
        and float(src_z + 0.5 * source.height) <= float(design.depth - pml_z)
    )
    print(f"Source inside non-PML: {src_inside_non_pml}")
    for name, mon in [
        ("o1_fwd", m_fwd),
        ("o1_ref", m_ref),
        ("o2_out", m_o2),
        ("o3_out", m_o3),
    ]:
        core_frac, clad_frac, eps_max = monitor_overlap_stats(
            np.asarray(grid.permittivity, dtype=float),
            mon,
            dx=dx,
            dy=dx,
            dz=dx,
            eps_core=n_core**2,
            eps_clad=n_clad**2,
        )
        print(
            f"Monitor overlap '{name}': core_frac={core_frac:.3f}, "
            f"clad_frac={clad_frac:.3f}, eps_max={eps_max:.3f}"
        )

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

    rec_interval = max(1, len(time) // 6)
    run_result = sim.run_compiled(
        num_steps=len(time),
        record_interval=rec_interval,
        record_fields=["Ez"],
        progress=False,
    )

    specs = [
        PortSpec(name="o1_fwd", monitor_name="o1_fwd", direction=src["direction"], polarization="te"),
        PortSpec(name="o1_ref", monitor_name="o1_ref", direction=src["direction"], polarization="te"),
        PortSpec(name="o2_out", monitor_name="o2_out", direction=ports["o2"]["direction"], polarization="te"),
        PortSpec(name="o3_out", monitor_name="o3_out", direction=ports["o3"]["direction"], polarization="te"),
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

    p11 = abs(s11) ** 2
    p21 = abs(s21) ** 2
    p31 = abs(s31) ** 2
    closure = p11 + p21 + p31
    split_sum = max(p21 + p31, 1e-30)
    split_o2 = p21 / split_sum
    split_o3 = p31 / split_sum
    balance_db = abs(
        20.0 * np.log10(max(abs(s21), 1e-12)) - 20.0 * np.log10(max(abs(s31), 1e-12))
    )

    p_in_flux = directional_power_x(m_in_flux, dx=dx, direction="+x")
    p_ref_flux = directional_power_x(m_ref_flux, dx=dx, direction="-x")
    p_out_flux = directional_power_x(m_out_flux, dx=dx, direction="+x")
    p_in_abs = max(abs(p_in_flux), 1e-30)
    flux_closure_signed = (p_ref_flux + p_out_flux) / p_in_abs
    flux_closure_unsigned = (abs(p_ref_flux) + abs(p_out_flux)) / p_in_abs

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
        f"Modal closure: {closure:.6f}, split o2/o3={split_o2:.3f}/{split_o3:.3f}, "
        f"balance={balance_db:.3f} dB"
    )
    print(
        "Incident-wave normalization: "
        f"inc_key={inc_key}, |a_plus_fwd|={abs(a_fwd_plus):.3e}, "
        f"|a_minus_fwd|={abs(a_fwd_minus):.3e}, |a_inc|={abs(a_incident):.3e}"
    )
    print(
        "Wide-plane flux closure: "
        f"signed={flux_closure_signed:.6f}, unsigned={flux_closure_unsigned:.6f} "
        f"(R={p_ref_flux/p_in_abs:.6f}, T={p_out_flux/p_in_abs:.6f})"
    )

    eps = np.asarray(grid.permittivity, dtype=float)
    z_idx = int(np.clip(round(core_zc / dx), 0, eps.shape[0] - 1))
    y_idx = int(np.clip(round(src["center"][1] / dx), 0, eps.shape[1] - 1))

    fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.6), dpi=260)

    eps_xy = eps[z_idx]
    im0 = ax[0, 0].imshow(
        eps_xy,
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=[0.0, design.width / µm, 0.0, design.height / µm],
    )
    ax[0, 0].set_title("3D gdsfactory MMI (XY core slice)")
    ax[0, 0].set_xlabel("x (um)")
    ax[0, 0].set_ylabel("y (um)")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)
    draw_pml_lines_xy(
        ax[0, 0],
        width=design.width,
        height=design.height,
        pml_xy=pml_xy,
        pml_right=pml_right,
    )

    for name, (s, e), color in [
        ("o1_fwd", (fwd_start, fwd_end), "white"),
        ("o1_ref", (ref_start, ref_end), "cyan"),
        ("o2_out", out_planes["o2"], "orange"),
        ("o3_out", out_planes["o3"], "orange"),
    ]:
        x_um = 0.5 * (s[0] + e[0]) / µm
        y0_um = min(s[1], e[1]) / µm
        y1_um = max(s[1], e[1]) / µm
        ax[0, 0].plot([x_um, x_um], [y0_um, y1_um], color=color, lw=1.5)
        ax[0, 0].text(x_um, y1_um + 0.04, name, color=color, fontsize=7, ha="center")
    ax[0, 0].plot(
        [src_x / µm, src_x / µm],
        [(src_y - 0.5 * source.width) / µm, (src_y + 0.5 * source.width) / µm],
        color="red",
        lw=1.8,
    )
    ax[0, 0].text(src_x / µm, (src_y + 0.5 * source.width) / µm + 0.04, "source", color="red", fontsize=7, ha="center")

    labels = ["|S11|^2", "|S21|^2", "|S31|^2", "closure"]
    vals = [p11, p21, p31, closure]
    colors = ["black", "tab:blue", "tab:green", "tab:red"]
    ax[0, 1].bar(labels, vals, color=colors, alpha=0.88)
    ax[0, 1].axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.7)
    ax[0, 1].set_ylabel("Power")
    ax[0, 1].set_title("Modal DFT Metrics")
    ax[0, 1].grid(axis="y", alpha=0.25)

    ax[1, 0].axis("off")
    ax[1, 0].text(
        0.02,
        0.96,
        (
            f"MMI1x2 3D @ {wl0/µm:.4f} um\n"
            f"|S11| = {abs(s11):.6f}\n"
            f"|S21| = {abs(s21):.6f}\n"
            f"|S31| = {abs(s31):.6f}\n"
            f"closure = {closure:.6f}\n"
            f"flux signed/unsigned = {flux_closure_signed:.3f}/{flux_closure_unsigned:.3f}\n"
            f"split o2/o3 = {split_o2:.3f}/{split_o3:.3f}\n"
            f"inc_key={inc_key}, |a_inc|={abs(a_incident):.3e}\n"
            f"dx={dx/µm:.4f} um, steps={len(time)}"
        ),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    eps_xz = eps[:, y_idx, :]
    im1 = ax[1, 1].imshow(
        eps_xz,
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=[0.0, design.width / µm, 0.0, design.depth / µm],
    )
    ax[1, 1].set_title("XZ Slice (input y)")
    ax[1, 1].set_xlabel("x (um)")
    ax[1, 1].set_ylabel("z (um)")
    fig.colorbar(im1, ax=ax[1, 1], fraction=0.046, pad=0.04)
    draw_pml_lines_xz(
        ax[1, 1],
        width=design.width,
        depth=design.depth,
        pml_xy=pml_xy,
        pml_right=pml_right,
        pml_z=pml_z,
    )
    ax[1, 1].plot(
        [src_x / µm, src_x / µm],
        [(src_z - 0.5 * source.height) / µm, (src_z + 0.5 * source.height) / µm],
        color="red",
        lw=1.8,
    )
    for name, (s, e), color in [
        ("o1_fwd", (fwd_start, fwd_end), "white"),
        ("o1_ref", (ref_start, ref_end), "cyan"),
        ("o2_out", out_planes["o2"], "orange"),
        ("o3_out", out_planes["o3"], "orange"),
    ]:
        x_um = 0.5 * (s[0] + e[0]) / µm
        z0_um = min(s[2], e[2]) / µm
        z1_um = max(s[2], e[2]) / µm
        ax[1, 1].plot([x_um, x_um], [z0_um, z1_um], color=color, lw=1.3)
        ax[1, 1].text(x_um, z1_um + 0.02, name, color=color, fontsize=6.5, ha="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=320)
    plt.close(fig)
    print(f"Saved S-parameter figure: {out_png}")

    ez_snap = np.asarray(sim.fields.Ez, dtype=float)
    if isinstance(run_result, dict):
        ez_hist = np.asarray(run_result.get("fields", {}).get("Ez", np.zeros((0,))), dtype=float)
        if ez_hist.ndim == 4 and ez_hist.shape[0] > 0:
            peak_idx = int(np.argmax(np.max(np.abs(ez_hist), axis=(1, 2, 3))))
            ez_snap = np.asarray(ez_hist[peak_idx], dtype=float)

    x_probe = int(np.clip(round(0.5 * (out_flux_x + src["center"][0]) / dx), 0, ez_snap.shape[2] - 1))
    ez_xy = ez_snap[z_idx, :, :]
    ez_xz = ez_snap[:, y_idx, :]
    ez_yz = ez_snap[:, :, x_probe]

    fig2, ax2 = plt.subplots(1, 3, figsize=(10.2, 3.2), dpi=260)
    im_xy = ax2[0].imshow(
        ez_xy,
        origin="lower",
        cmap="RdBu",
        aspect="auto",
        extent=[0.0, design.width / µm, 0.0, design.height / µm],
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
        extent=[0.0, design.width / µm, 0.0, design.depth / µm],
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
        extent=[0.0, design.height / µm, 0.0, design.depth / µm],
    )
    ax2[2].set_title(f"Ez YZ @ x={x_probe}")
    ax2[2].set_xlabel("y (um)")
    ax2[2].set_ylabel("z (um)")
    fig2.colorbar(im_yz, ax=ax2[2], fraction=0.046, pad=0.04)

    fig2.tight_layout()
    fig2.savefig(out_field_png, dpi=320)
    plt.close(fig2)
    print(f"Saved field-slice figure: {out_field_png}")


if __name__ == "__main__":
    main()
