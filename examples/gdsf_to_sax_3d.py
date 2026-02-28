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


def directional_power_x(monitor, *, dx, dy, dz, direction, freq_idx=0):
    sign = 1.0 if str(direction).strip() == "+x" else -1.0
    ey = np.asarray(monitor.get_dft_component("Ey"), dtype=np.complex128)[int(freq_idx)]
    ez = np.asarray(monitor.get_dft_component("Ez"), dtype=np.complex128)[int(freq_idx)]
    hy = np.asarray(monitor.get_dft_component("Hy"), dtype=np.complex128)[int(freq_idx)]
    hz = np.asarray(monitor.get_dft_component("Hz"), dtype=np.complex128)[int(freq_idx)]
    del dx  # x-normal plane area is dy*dz
    s = 0.5 * np.sum(ey * np.conjugate(hz) - ez * np.conjugate(hy)) * (dy * dz)
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


def save_mode_source_plot(mode_source, out_path):
    eps = np.asarray(getattr(mode_source, "_eps_profile_2d", np.zeros((1, 1))), dtype=float)
    fields = {
        "Ex": np.asarray(getattr(mode_source, "_Ex_profile", 0.0), dtype=float),
        "Ey": np.asarray(getattr(mode_source, "_Ey_profile", 0.0), dtype=float),
        "Ez": np.asarray(getattr(mode_source, "_Ez_profile", 0.0), dtype=float),
        "Hx": np.asarray(getattr(mode_source, "_Hx_profile", 0.0), dtype=float),
        "Hy": np.asarray(getattr(mode_source, "_Hy_profile", 0.0), dtype=float),
        "Hz": np.asarray(getattr(mode_source, "_Hz_profile", 0.0), dtype=float),
    }

    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.6), dpi=220)
    ax = axes.ravel()
    for a in ax:
        a.set_box_aspect(1.0)
    im_eps = ax[0].imshow(eps, origin="lower", cmap="viridis", aspect="equal")
    ax[0].set_title("eps cross-section")
    fig.colorbar(im_eps, ax=ax[0], fraction=0.046, pad=0.04)

    for i, name in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], start=1):
        arr = np.asarray(fields[name]).squeeze()
        if arr.ndim != 2:
            arr = np.atleast_2d(arr)
        im = ax[i].imshow(np.abs(arr), origin="lower", cmap="magma", aspect="equal")
        ax[i].set_title(f"|{name}|")
        fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)

    ax[7].axis("off")
    ax[7].text(
        0.02,
        0.95,
        (
            f"pol={mode_source.pol}\n"
            f"dir={mode_source.direction}\n"
            f"neff={float(np.real(getattr(mode_source, '_neff', np.nan))):.6f}\n"
            f"neff_imp={float(np.real(getattr(mode_source, '_impedance_neff', np.nan))):.6f}\n"
            f"width={float(mode_source.width)/µm:.3f} um\n"
            f"height={float(mode_source.height)/µm:.3f} um"
        ),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    out_dir = Path("benchmarks/results/gdsf_to_sax_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "gdsf_mmi1x2_3d_sparams.png"
    out_field_png = out_dir / "gdsf_mmi1x2_3d_ey_field_slices.png"
    out_mode_png = out_dir / "gdsf_mmi1x2_3d_mode_source.png"
    out_signal_png = out_dir / "gdsf_mmi1x2_3d_signal.png"

    wl0 = 1.55 * µm
    f0 = LIGHT_SPEED / wl0
    wl_min = 1.50 * µm
    wl_max = 1.60 * µm
    nfreq = 31
    freqs = np.linspace(LIGHT_SPEED / wl_max, LIGHT_SPEED / wl_min, nfreq, dtype=float)

    # Use moderate index contrast and resolution to keep this example lightweight.
    n_core, n_clad = 2.04, 1.444
    core_t = 0.45 * µm

    input_extension = 5.0 * µm
    output_extension = 4.0 * µm
    y_margin = 2.6 * µm

    clad_below = 2.2 * µm
    clad_above = 2.2 * µm
    depth = clad_below + core_t + clad_above
    core_z0 = clad_below
    core_zc = core_z0 + 0.5 * core_t

    dx, dt = dxdt(
        wl0,
        n_max=n_core,
        dims=3,
        safety_factor=0.96,
        points_per_wavelength=20,
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
    port_overlap = 0.20 * µm
    for name, port in ports.items():
        cx, cy = port["center"]
        w = float(port["width"])
        ext = input_extension if name == "o1" else output_extension
        d_out = outward_direction(port["direction"])
        sx, sy = move_along((cx, cy), d_out, -port_overlap)
        ox, oy = move_along((cx, cy), d_out, ext)
        if d_out.endswith("x"):
            design += Rectangle(
                position=(min(sx, ox), cy - 0.5 * w, core_z0),
                width=abs(ox - sx),
                height=w,
                depth=core_t,
                material=Material(n_core**2),
            )
        else:
            design += Rectangle(
                position=(cx - 0.5 * w, min(sy, oy), core_z0),
                width=w,
                height=abs(oy - sy),
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
    # Larger source/monitor windows to stabilize 3D mode solving/projection.
    src_w = max(1.60 * µm, 3.2 * src["width"])
    src_h = max(2.20 * µm, 4.8 * core_t)
    max_src_w = max(0.2 * µm, design.height - 2.0 * pml_xy - 0.10 * µm)
    max_src_h = max(0.2 * µm, design.depth - 2.0 * pml_z - 0.10 * µm)
    src_w = min(src_w, max_src_w)
    src_h = min(src_h, max_src_h)
    x_port = float(src["center"][0])
    x_straight_min = float(pml_xy + 0.60 * µm)
    x_straight_max = float(x_port - 0.60 * µm)
    if x_straight_max <= x_straight_min:
        raise RuntimeError(
            "Input straight section too short for source/monitor placement. "
            "Increase input_extension."
        )
    source_x = np.clip(x_port - 2.0 * µm, x_straight_min + 0.3 * µm, x_straight_max - 0.3 * µm)
    source_xy = (float(source_x), float(src["center"][1]))
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

    mon_y_span = max(1.60 * µm, 3.2 * src["width"])
    mon_z_span = max(2.20 * µm, 4.8 * core_t)

    # Place source and input monitors on the long straight input section (before taper/MMI).
    fwd_x = float(np.clip(source_x + 0.70 * µm, x_straight_min + 0.1 * µm, x_straight_max - 0.1 * µm))
    ref_x = float(np.clip(source_x - 0.80 * µm, x_straight_min + 0.1 * µm, x_straight_max - 0.1 * µm))
    fwd_xy = (fwd_x, float(src["center"][1]))
    ref_xy = (ref_x, float(src["center"][1]))
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
        mon_xy = move_along(p["center"], d_out, +1.0 * µm)
        out_planes[name] = make_x_plane(
            mon_xy[0], mon_xy[1], mon_y_span, core_zc, mon_z_span, z_min, z_max
        )

    # Relatively short Gaussian pulse to launch broadband energy through the device.
    pulse_t0 = 18.0 / f0
    pulse_sigma = 4.0 / f0

    # Place DFT windows around estimated pulse arrival at each monitor plane.
    # A conservative estimate based on core index is sufficient for this sanity test.
    v_est = LIGHT_SPEED / n_core
    t_fwd_center = pulse_t0 + abs(fwd_x - source_x) / max(v_est, 1e-30)
    # Approximate dominant reflection from the MMI/taper entrance near input port x-position.
    t_ref_center = pulse_t0 + (abs(x_port - source_x) + abs(x_port - ref_x)) / max(v_est, 1e-30)
    t_o2_center = pulse_t0 + abs(float(out_planes["o2"][0][0]) - float(source_x)) / max(v_est, 1e-30)
    t_o3_center = pulse_t0 + abs(float(out_planes["o3"][0][0]) - float(source_x)) / max(v_est, 1e-30)
    dft_half = 10.0 / f0

    def centered_window(tc):
        return max(0.0, tc - dft_half), tc + dft_half

    dft_fwd_t_start, dft_fwd_t_end = centered_window(t_fwd_center)
    dft_ref_t_start, dft_ref_t_end = centered_window(t_ref_center)
    dft_o2_t_start, dft_o2_t_end = centered_window(t_o2_center)
    dft_o3_t_start, dft_o3_t_end = centered_window(t_o3_center)

    settle_tail = 24.0 / f0
    t_total = max(dft_ref_t_end, dft_o2_t_end, dft_o3_t_end) + settle_tail
    time = np.arange(0.0, t_total, dt)

    source.signal = np.exp(-0.5 * ((time - pulse_t0) / max(pulse_sigma, 1e-30)) ** 2) * np.cos(
        2.0 * np.pi * f0 * (time - pulse_t0)
    )
    fig_sig, ax_sig = plt.subplots(1, 1, figsize=(6.6, 2.5), dpi=240)
    ax_sig.plot(time / 1e-15, source.signal, color="black", lw=1.6)
    ax_sig.set_title("Mode Source Signal (Short Gaussian Pulse)")
    ax_sig.set_xlabel("time (fs)")
    ax_sig.set_ylabel("amplitude")
    ax_sig.grid(alpha=0.3)
    fig_sig.tight_layout()
    fig_sig.savefig(out_signal_png, dpi=320)
    plt.close(fig_sig)
    print(
        f"Saved signal figure: {out_signal_png} "
        f"(t0={pulse_t0*f0:.1f}/f0, sigma={pulse_sigma*f0:.1f}/f0)"
    )

    dft_cfg_common = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=("Ey", "Ez", "Hy", "Hz"),
        dft_window="hann",
        dft_record_every_step=True,
    )
    dft_fwd_cfg = dict(dft_cfg_common, dft_t_start=dft_fwd_t_start, dft_t_end=dft_fwd_t_end)
    dft_ref_cfg = dict(dft_cfg_common, dft_t_start=dft_ref_t_start, dft_t_end=dft_ref_t_end)
    dft_o2_cfg = dict(dft_cfg_common, dft_t_start=dft_o2_t_start, dft_t_end=dft_o2_t_end)
    dft_o3_cfg = dict(dft_cfg_common, dft_t_start=dft_o3_t_start, dft_t_end=dft_o3_t_end)

    m_fwd = Monitor(start=fwd_start, end=fwd_end, name="o1_fwd", **dft_fwd_cfg)
    m_ref = Monitor(start=ref_start, end=ref_end, name="o1_ref", **dft_ref_cfg)
    m_o2 = Monitor(start=out_planes["o2"][0], end=out_planes["o2"][1], name="o2_out", **dft_o2_cfg)
    m_o3 = Monitor(start=out_planes["o3"][0], end=out_planes["o3"][1], name="o3_out", **dft_o3_cfg)

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
    m_in_flux = Monitor(start=in_flux_start, end=in_flux_end, name="in_flux", **dft_fwd_cfg)
    m_ref_flux = Monitor(start=ref_flux_start, end=ref_flux_end, name="ref_flux", **dft_ref_cfg)
    m_out_flux = Monitor(start=out_flux_start, end=out_flux_end, name="out_flux", **dft_o2_cfg)

    print(
        "Running 3D gdsfactory MMI1x2: "
        f"steps={len(time)}, dx={dx/µm:.4f}um, "
        f"domain=({design.width/µm:.2f},{design.height/µm:.2f},{design.depth/µm:.2f})um"
    )
    print(
        "DFT freq sweep: "
        f"{(LIGHT_SPEED/np.max(freqs))/µm:.4f}..{(LIGHT_SPEED/np.min(freqs))/µm:.4f} um "
        f"({len(freqs)} points)"
    )
    print(
        "DFT windows (/f0): "
        f"fwd=[{dft_fwd_t_start*f0:.1f},{dft_fwd_t_end*f0:.1f}], "
        f"ref=[{dft_ref_t_start*f0:.1f},{dft_ref_t_end*f0:.1f}], "
        f"o2=[{dft_o2_t_start*f0:.1f},{dft_o2_t_end*f0:.1f}], "
        f"o3=[{dft_o3_t_start*f0:.1f},{dft_o3_t_end*f0:.1f}] "
        f"(pulse_t0={pulse_t0*f0:.1f}, steps={len(time)})"
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
    print(
        "Input straight placement: "
        f"x_straight=[{x_straight_min/µm:.3f},{x_straight_max/µm:.3f}]um, "
        f"source_x={source_x/µm:.3f}um, fwd_x={fwd_x/µm:.3f}um, ref_x={ref_x/µm:.3f}um, "
        f"port_x={x_port/µm:.3f}um"
    )
    print(
        "Port directions (gdsfactory): "
        f"o1={ports['o1']['direction']}, o2={ports['o2']['direction']}, o3={ports['o3']['direction']}"
    )
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
    source.initialize(grid.permittivity, dx, dt=dt)
    save_mode_source_plot(source, out_mode_png)
    print(f"Saved mode-source figure: {out_mode_png}")

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
        record_fields=["Ey"],
        progress=False,
    )

    # Multimode extraction: excite fundamental input mode and project multiple
    # output/reflection mode indices to quantify guided power not in TE0.
    requested_modes = 3
    s_dft = None
    modes_eval = requested_modes
    last_extract_error = None
    port_defs = {
        "o1": {"monitor": "o1_ref", "direction": outward_direction(src["direction"])},
        "o2": {"monitor": "o2_out", "direction": outward_direction(ports["o2"]["direction"])},
        "o3": {"monitor": "o3_out", "direction": outward_direction(ports["o3"]["direction"])},
    }
    source_key = "o1_m0"
    for modes_try in range(requested_modes, 0, -1):
        specs_try = []
        output_try = []
        for port_name in ("o1", "o2", "o3"):
            for mode_idx in range(modes_try):
                key = f"{port_name}_m{mode_idx}"
                kwargs = {}
                if port_name == "o1" and mode_idx == 0:
                    kwargs["reference_monitor"] = "o1_fwd"
                specs_try.append(
                    PortSpec(
                        name=key,
                        monitor_name=port_defs[port_name]["monitor"],
                        direction=port_defs[port_name]["direction"],
                        polarization="te",
                        mode_index=mode_idx,
                        **kwargs,
                    )
                )
                output_try.append(key)
        try:
            s_dft = sim.get_S_matrix_modal_dft(
                source_port=source_key,
                ports=specs_try,
                output_ports=output_try,
                frequencies=freqs,
                as_sax=False,
                return_diagnostics=True,
                min_incident_db=-35.0,
            )
            modes_eval = modes_try
            break
        except Exception as exc:
            last_extract_error = exc

    if s_dft is None:
        raise RuntimeError(
            f"Multimode S extraction failed up to mode {requested_modes - 1}: {last_extract_error}"
        )
    if modes_eval < requested_modes:
        print(
            f"Multimode extraction fallback: requested {requested_modes} modes, using {modes_eval} modes."
        )

    s_matrix = s_dft["s_matrix"]
    diagnostics = s_dft["diagnostics"]
    waves = diagnostics["waves"]

    def s_col(dst_key):
        return np.asarray(
            s_matrix.get((dst_key, source_key), np.zeros(freqs.size, dtype=np.complex128)),
            dtype=np.complex128,
        )

    s11_spec = s_col("o1_m0")
    s21_spec = s_col("o2_m0")
    s31_spec = s_col("o3_m0")
    valid_mask = np.asarray(diagnostics.get("valid_mask", np.ones_like(freqs, dtype=bool)), dtype=bool)

    p11_spec = np.abs(s11_spec) ** 2
    p21_spec = np.abs(s21_spec) ** 2
    p31_spec = np.abs(s31_spec) ** 2
    closure_spec = p11_spec + p21_spec + p31_spec
    split_sum_spec = np.maximum(p21_spec + p31_spec, 1e-30)
    split_o2_spec = p21_spec / split_sum_spec
    split_o3_spec = p31_spec / split_sum_spec
    balance_db_spec = np.abs(
        20.0 * np.log10(np.maximum(np.abs(s21_spec), 1e-12))
        - 20.0 * np.log10(np.maximum(np.abs(s31_spec), 1e-12))
    )
    p_ref_multimode_spec = np.zeros(freqs.size, dtype=float)
    p_out_multimode_spec = np.zeros(freqs.size, dtype=float)
    p_ref_multimode_guided_spec = np.zeros(freqs.size, dtype=float)
    p_out_multimode_guided_spec = np.zeros(freqs.size, dtype=float)
    guided_neff_threshold = n_clad + 1e-3
    for port_name in ("o1", "o2", "o3"):
        for mode_idx in range(modes_eval):
            key = f"{port_name}_m{mode_idx}"
            s_mode = s_col(key)
            p_mode = np.abs(s_mode) ** 2
            neff_mode = np.asarray(
                waves.get(key, {}).get("mode_neff", np.full(freqs.size, np.nan)),
                dtype=float,
            )
            is_guided = np.isfinite(neff_mode) & (neff_mode > guided_neff_threshold)
            if port_name == "o1":
                p_ref_multimode_spec += p_mode
                p_ref_multimode_guided_spec += np.where(is_guided, p_mode, 0.0)
            else:
                p_out_multimode_spec += p_mode
                p_out_multimode_guided_spec += np.where(is_guided, p_mode, 0.0)
    closure_multimode_spec = p_ref_multimode_spec + p_out_multimode_spec
    closure_multimode_guided_spec = p_ref_multimode_guided_spec + p_out_multimode_guided_spec

    wl_spec = LIGHT_SPEED / freqs
    wl0_idx = int(np.argmin(np.abs(wl_spec - wl0)))
    wl_c = float(wl_spec[wl0_idx])
    s11 = complex(s11_spec[wl0_idx])
    s21 = complex(s21_spec[wl0_idx])
    s31 = complex(s31_spec[wl0_idx])
    p11 = float(p11_spec[wl0_idx])
    p21 = float(p21_spec[wl0_idx])
    p31 = float(p31_spec[wl0_idx])
    closure = float(closure_spec[wl0_idx])
    closure_multimode = float(closure_multimode_spec[wl0_idx])
    closure_multimode_guided = float(closure_multimode_guided_spec[wl0_idx])
    split_o2 = float(split_o2_spec[wl0_idx])
    split_o3 = float(split_o3_spec[wl0_idx])
    balance_db = float(balance_db_spec[wl0_idx])
    a_fwd_plus = np.asarray(waves[source_key]["a_plus"], dtype=np.complex128)
    a_fwd_minus = np.asarray(waves[source_key]["a_minus"], dtype=np.complex128)
    a_incident = np.asarray(waves[source_key].get("a_incident", a_fwd_plus), dtype=np.complex128)
    a_incident_c = complex(a_incident[wl0_idx])
    a_fwd_plus_c = complex(a_fwd_plus[wl0_idx])
    a_fwd_minus_c = complex(a_fwd_minus[wl0_idx])

    p_in_flux = directional_power_x(m_in_flux, dx=dx, dy=dx, dz=dx, direction="+x", freq_idx=wl0_idx)
    p_ref_flux = directional_power_x(m_ref_flux, dx=dx, dy=dx, dz=dx, direction="-x", freq_idx=wl0_idx)
    p_out_flux = directional_power_x(m_out_flux, dx=dx, dy=dx, dz=dx, direction="+x", freq_idx=wl0_idx)
    p_in_abs = max(abs(p_in_flux), 1e-30)
    flux_closure_signed = (p_ref_flux + p_out_flux) / p_in_abs
    flux_closure_unsigned = (abs(p_ref_flux) + abs(p_out_flux)) / p_in_abs

    print(
        f"S11 @ {wl_c/µm:.4f}um: |S11|={abs(s11):.6f}, "
        f"{20*np.log10(max(abs(s11),1e-12)):.2f} dB"
    )
    print(
        f"S21 @ {wl_c/µm:.4f}um: |S21|={abs(s21):.6f}, "
        f"{20*np.log10(max(abs(s21),1e-12)):.2f} dB"
    )
    print(
        f"S31 @ {wl_c/µm:.4f}um: |S31|={abs(s31):.6f}, "
        f"{20*np.log10(max(abs(s31),1e-12)):.2f} dB"
    )
    print(
        f"Modal closure: {closure:.6f}, split o2/o3={split_o2:.3f}/{split_o3:.3f}, "
        f"balance={balance_db:.3f} dB"
    )
    print(
        "Multimode closure: "
        f"modes=0..{modes_eval-1}, total={closure_multimode:.6f}, "
        f"guided(neff>{guided_neff_threshold:.3f})={closure_multimode_guided:.6f}"
    )
    source_out_dir = outward_direction(src["direction"])
    print(
        "Wave convention check: "
        f"src_dir={src['direction']}, src_outward={source_out_dir}, "
        f"|a_plus_ref|={abs(a_fwd_plus_c):.3e}, |a_minus_ref|={abs(a_fwd_minus_c):.3e}, "
        f"|a_inc|={abs(a_incident_c):.3e}"
    )
    print(
        "Wide-plane flux closure: "
        f"signed={flux_closure_signed:.6f}, unsigned={flux_closure_unsigned:.6f} "
        f"(R={p_ref_flux/p_in_abs:.6f}, T={p_out_flux/p_in_abs:.6f})"
    )
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size > 0:
        p_sum_valid = np.asarray(closure_spec[valid_idx], dtype=float)
        p_sum_mm_valid = np.asarray(closure_multimode_spec[valid_idx], dtype=float)
        p_sum_mm_guided_valid = np.asarray(closure_multimode_guided_spec[valid_idx], dtype=float)
        s21_db = 20.0 * np.log10(np.maximum(np.abs(s21_spec[valid_idx]), 1e-12))
        s31_db = 20.0 * np.log10(np.maximum(np.abs(s31_spec[valid_idx]), 1e-12))
        print(
            "Sweep summary: "
            f"valid={valid_idx.size}/{len(freqs)}, "
            f"closure[min,max]=[{np.nanmin(p_sum_valid):.3f},{np.nanmax(p_sum_valid):.3f}], "
            f"closure_mm[min,max]=[{np.nanmin(p_sum_mm_valid):.3f},{np.nanmax(p_sum_mm_valid):.3f}], "
            f"closure_mm_guided[min,max]=[{np.nanmin(p_sum_mm_guided_valid):.3f},{np.nanmax(p_sum_mm_guided_valid):.3f}], "
            f"S21[dB min,max]=[{np.nanmin(s21_db):.2f},{np.nanmax(s21_db):.2f}], "
            f"S31[dB min,max]=[{np.nanmin(s31_db):.2f},{np.nanmax(s31_db):.2f}]"
        )

    eps = np.asarray(grid.permittivity, dtype=float)
    z_idx = int(np.clip(round(core_zc / dx), 0, eps.shape[0] - 1))
    y_idx = int(np.clip(round(src["center"][1] / dx), 0, eps.shape[1] - 1))

    fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.6), dpi=260)
    for a in ax.ravel():
        a.set_box_aspect(1.0)

    eps_xy = eps[z_idx]
    im0 = ax[0, 0].imshow(
        eps_xy,
        origin="lower",
        cmap="viridis",
        aspect="equal",
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

    wl_um = wl_spec / µm
    order = np.argsort(wl_um)
    ax[0, 1].plot(wl_um[order], p11_spec[order], "o-", color="black", lw=1.4, label="|S11|^2")
    ax[0, 1].plot(wl_um[order], p21_spec[order], "o-", color="tab:blue", lw=1.4, label="|S21|^2")
    ax[0, 1].plot(wl_um[order], p31_spec[order], "o-", color="tab:green", lw=1.4, label="|S31|^2")
    ax[0, 1].plot(wl_um[order], closure_spec[order], color="tab:red", lw=1.2, ls="--", label="closure m0")
    ax[0, 1].plot(
        wl_um[order],
        closure_multimode_guided_spec[order],
        color="tab:orange",
        lw=1.2,
        ls="-.",
        label=f"closure m0..m{modes_eval-1} (guided)",
    )
    ax[0, 1].plot(
        wl_um[order],
        closure_multimode_spec[order],
        color="tab:brown",
        lw=1.0,
        ls=":",
        label=f"closure m0..m{modes_eval-1} (all)",
    )
    ax[0, 1].axhline(1.0, color="k", ls=":", lw=1.0, alpha=0.75)
    ax[0, 1].axvline(wl0 / µm, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax[0, 1].set_xlabel("wavelength (um)")
    ax[0, 1].set_ylabel("Power")
    ax[0, 1].set_title("Modal DFT Spectrum (Fundamental + Multimode)")
    ax[0, 1].grid(alpha=0.25)
    ax[0, 1].legend(fontsize=7, loc="best")

    ax[1, 0].axis("off")
    ax[1, 0].text(
        0.02,
        0.96,
        (
            f"MMI1x2 3D @ {wl_c/µm:.4f} um (center)\n"
            f"sweep = {wl_min/µm:.4f}..{wl_max/µm:.4f} um ({len(freqs)} pts)\n"
            f"|S11| = {abs(s11):.6f}\n"
            f"|S21| = {abs(s21):.6f}\n"
            f"|S31| = {abs(s31):.6f}\n"
            f"closure m0 = {closure:.6f}\n"
            f"closure mm(guided) = {closure_multimode_guided:.6f}\n"
            f"closure mm(all) = {closure_multimode:.6f}\n"
            f"mode count eval = {modes_eval}\n"
            f"flux signed/unsigned = {flux_closure_signed:.3f}/{flux_closure_unsigned:.3f}\n"
            f"split o2/o3 = {split_o2:.3f}/{split_o3:.3f}\n"
            f"src_outward={source_out_dir}, |a_inc|={abs(a_incident_c):.3e}\n"
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
        aspect="equal",
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

    ey_snap = np.asarray(sim.fields.Ey, dtype=float)
    if isinstance(run_result, dict):
        ey_hist = np.asarray(run_result.get("fields", {}).get("Ey", np.zeros((0,))), dtype=float)
        if ey_hist.ndim == 4 and ey_hist.shape[0] > 0:
            peak_idx = int(np.argmax(np.max(np.abs(ey_hist), axis=(1, 2, 3))))
            ey_snap = np.asarray(ey_hist[peak_idx], dtype=float)

    x_probe = int(np.clip(round(0.5 * (out_flux_x + src["center"][0]) / dx), 0, ey_snap.shape[2] - 1))
    ey_xy = ey_snap[z_idx, :, :]
    ey_xz = ey_snap[:, y_idx, :]
    ey_yz = ey_snap[:, :, x_probe]

    fig2, ax2 = plt.subplots(1, 3, figsize=(10.2, 3.2), dpi=260)
    for a in ax2:
        a.set_box_aspect(1.0)
    im_xy = ax2[0].imshow(
        ey_xy,
        origin="lower",
        cmap="RdBu",
        aspect="equal",
        extent=[0.0, design.width / µm, 0.0, design.height / µm],
    )
    ax2[0].set_title(f"Ey XY @ z={z_idx}")
    ax2[0].set_xlabel("x (um)")
    ax2[0].set_ylabel("y (um)")
    fig2.colorbar(im_xy, ax=ax2[0], fraction=0.046, pad=0.04)

    im_xz = ax2[1].imshow(
        ey_xz,
        origin="lower",
        cmap="RdBu",
        aspect="equal",
        extent=[0.0, design.width / µm, 0.0, design.depth / µm],
    )
    ax2[1].set_title(f"Ey XZ @ y={y_idx}")
    ax2[1].set_xlabel("x (um)")
    ax2[1].set_ylabel("z (um)")
    fig2.colorbar(im_xz, ax=ax2[1], fraction=0.046, pad=0.04)

    im_yz = ax2[2].imshow(
        ey_yz,
        origin="lower",
        cmap="RdBu",
        aspect="equal",
        extent=[0.0, design.height / µm, 0.0, design.depth / µm],
    )
    ax2[2].set_title(f"Ey YZ @ x={x_probe}")
    ax2[2].set_xlabel("y (um)")
    ax2[2].set_ylabel("z (um)")
    fig2.colorbar(im_yz, ax=ax2[2], fraction=0.046, pad=0.04)

    fig2.tight_layout()
    fig2.savefig(out_field_png, dpi=320)
    plt.close(fig2)
    print(f"Saved field-slice figure: {out_field_png}")


if __name__ == "__main__":
    main()
