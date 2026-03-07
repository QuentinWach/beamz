"""Broadband 2D modal S-parameter extraction for a 4-port crossing.

Workflow:
1. Import a crossing from UBC PDK (fallback: gdsfactory generic crossing).
2. Build a BeamZ design and extend each port with a straight waveguide section.
3. Launch a Gaussian pulse at one source port.
4. Use DFT monitors + modal decomposition to extract S11/S21/S31/S41 over frequency.
5. Save a compact-model data file and a dB plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz import (
    LIGHT_SPEED,
    Design,
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


def move_along(center: tuple[float, float], direction: str, distance: float) -> tuple[float, float]:
    if direction == "+x":
        return center[0] + distance, center[1]
    if direction == "-x":
        return center[0] - distance, center[1]
    if direction == "+y":
        return center[0], center[1] + distance
    if direction == "-y":
        return center[0], center[1] - distance
    raise ValueError(f"Unsupported direction {direction!r}")


def parse_layer(layer_str: str) -> tuple[int, int]:
    parts = [p.strip() for p in layer_str.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid layer '{layer_str}'. Use 'layer,datatype', e.g. '1,0'.")
    return int(parts[0]), int(parts[1])


def load_crossing_component():
    """Return (component, label) using UBC PDK if available, else gdsfactory fallback."""
    try:
        from ubcpdk import PDK, cells

        PDK.activate()
        return cells.ebeam_crossing4(), "ubcpdk.cells.ebeam_crossing4"
    except Exception as exc:
        import gdsfactory as gf

        gf.gpdk.PDK.activate()
        print(
            "[beamz_crossing] UBC PDK unavailable, using generic gdsfactory crossing "
            f"fallback. Reason: {type(exc).__name__}: {exc}"
        )
        return gf.components.crossing(), "gdsfactory.components.crossing"


def port_line(
    port: dict,
    span: float,
    offset: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    cx, cy = move_along(port["center"], port["direction"], offset)
    if port["direction"].endswith("x"):
        return (cx, cy - 0.5 * span), (cx, cy + 0.5 * span)
    return (cx - 0.5 * span, cy), (cx + 0.5 * span, cy)


def line_center(line: tuple[tuple[float, float], tuple[float, float]]) -> tuple[float, float]:
    (x0, y0), (x1, y1) = line
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def build_design_with_extensions(
    component,
    *,
    layer: tuple[int, int],
    n_core: float,
    n_clad: float,
    extension: float,
) -> tuple[Design, dict]:
    imported_design, ports = gdsf.load(
        component,
        layer=layer,
        n_core=n_core,
        n_clad=n_clad,
        padding=0.0,
    )

    design = Design(
        width=imported_design.width + 2.0 * extension,
        height=imported_design.height + 2.0 * extension,
        depth=0.0,
        material=Material(n_clad**2),
    )
    for structure in imported_design.structures[1:]:
        design += structure.copy().shift(extension, extension)

    ports = {
        name: {
            **p,
            "center": (
                float(p["center"][0] + extension),
                float(p["center"][1] + extension),
            ),
            "width": float(p["width"]),
        }
        for name, p in ports.items()
    }

    # Extend each port outward so source/monitors can be placed in clean straight sections.
    for port in ports.values():
        cx, cy = port["center"]
        width = float(port["width"])
        d_out = outward_direction(port["direction"])
        ox, oy = move_along((cx, cy), d_out, extension)
        if port["direction"].endswith("x"):
            design += Rectangle(
                position=(min(cx, ox), cy - 0.5 * width),
                width=abs(ox - cx),
                height=width,
                material=Material(n_core**2),
                depth=0,
            )
        else:
            design += Rectangle(
                position=(cx - 0.5 * width, min(cy, oy)),
                width=width,
                height=abs(oy - cy),
                material=Material(n_core**2),
                depth=0,
            )

    return design, ports


def run_crossing(
    *,
    wl0: float,
    wl_min: float,
    wl_max: float,
    num_freqs: int,
    n_core: float,
    n_clad: float,
    polarization: str,
    points_per_wavelength: int,
    layer: tuple[int, int],
    out_dir: Path,
) -> None:
    component, component_label = load_crossing_component()
    polarization = str(polarization).lower()
    if polarization not in {"tm", "te"}:
        raise ValueError("--polarization must be 'tm' or 'te'.")

    extension = 5.0 * µm
    design, ports = build_design_with_extensions(
        component,
        layer=layer,
        n_core=n_core,
        n_clad=n_clad,
        extension=extension,
    )
    source_port = "o1" if "o1" in ports else sorted(ports.keys())[0]
    output_ports = [name for name in sorted(ports.keys()) if name != source_port]
    all_ports = [source_port, *output_ports]

    dx, dt = dxdt(
        wl0,
        n_max=n_core,
        dims=2,
        safety_factor=0.999,
        points_per_wavelength=points_per_wavelength,
    )
    grid = design.rasterize(resolution=dx)

    freqs = np.linspace(LIGHT_SPEED / wl_max, LIGHT_SPEED / wl_min, num_freqs)
    wl = LIGHT_SPEED / freqs
    f0 = LIGHT_SPEED / wl0

    src = ports[source_port]
    source_span = max(1.0 * µm, 3.0 * float(src["width"]))
    monitor_span = max(1.0 * µm, 3.0 * float(src["width"]))

    pml_thickness = 1.0 * wl0
    max_outward_offset = max(0.9 * µm, extension - pml_thickness - 0.35 * µm)
    source_mag = min(1.6 * µm, 0.55 * max_outward_offset)
    fwd_mag = max(0.70 * µm, source_mag - 0.45 * µm)
    ref_mag = min(max_outward_offset - 0.20 * µm, source_mag + 0.65 * µm)
    out_mag = min(max_outward_offset - 0.15 * µm, 0.80 * max_outward_offset)

    source_offset = -source_mag
    fwd_offset = -fwd_mag
    ref_offset = -ref_mag
    source_center = move_along(src["center"], src["direction"], source_offset)
    source_line = port_line(src, source_span, offset=source_offset)
    src_line_center = line_center(source_line)

    fwd_line = port_line(src, monitor_span, offset=fwd_offset)
    ref_line = port_line(src, monitor_span, offset=ref_offset)

    # Build multiple output-monitor placement candidates (farther into straight sections).
    out_mag_candidates = []
    for frac in (0.55, 0.75, 0.95):
        mag = float(np.clip(frac * out_mag, 0.70 * µm, max_outward_offset - 0.08 * µm))
        if not any(abs(mag - m) < 1e-12 for m in out_mag_candidates):
            out_mag_candidates.append(mag)

    out_candidates = {}
    min_center_separation = 1.2 * max(source_span, monitor_span)
    for p in output_ports:
        cand_list = []
        for i, mag in enumerate(out_mag_candidates):
            line = port_line(ports[p], monitor_span, offset=-mag)
            c_out = line_center(line)
            dist = float(np.hypot(c_out[0] - src_line_center[0], c_out[1] - src_line_center[1]))
            if dist < min_center_separation:
                deeper_mag = min(max_outward_offset - 0.08 * µm, mag + (min_center_separation - dist))
                line = port_line(ports[p], monitor_span, offset=-deeper_mag)
                mag = deeper_mag
            cand_list.append(
                {
                    "name": f"{p}_cand{i}",
                    "offset": -mag,
                    "line": line,
                }
            )
        out_candidates[p] = cand_list

    pulse_t0 = 12.0 / f0
    pulse_sigma = 4.0 / f0
    v_est = LIGHT_SPEED / max(n_core, 1e-9)

    def travel_time(p0, p1):
        return np.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])) / max(v_est, 1e-30)

    src_center_xy = tuple(float(v) for v in src["center"])
    src_xy = tuple(float(v) for v in source_center)
    t_fwd = pulse_t0 + travel_time(src_xy, line_center(fwd_line))
    t_ref = pulse_t0 + travel_time(src_xy, src_center_xy) + travel_time(src_center_xy, line_center(ref_line))

    dft_half = 26.0 / f0

    def centered_window(t_center):
        return max(0.0, t_center - dft_half), t_center + dft_half

    dft_fwd_t_start, dft_fwd_t_end = centered_window(t_fwd)
    dft_ref_t_start, dft_ref_t_end = centered_window(t_ref)

    out_windows = {}
    for p in output_ports:
        out_windows[p] = {}
        out_center = tuple(float(v) for v in ports[p]["center"])
        for cand in out_candidates[p]:
            t_out = (
                pulse_t0
                + travel_time(src_xy, src_center_xy)
                + travel_time(src_center_xy, out_center)
                + travel_time(out_center, line_center(cand["line"]))
            )
            out_windows[p][cand["name"]] = centered_window(t_out)

    t_total = max(
        dft_ref_t_end,
        *(
            out_windows[p][cand["name"]][1]
            for p in output_ports
            for cand in out_candidates[p]
        ),
    ) + 18.0 / f0
    time = np.arange(0.0, t_total, dt)
    signal = np.exp(-0.5 * ((time - pulse_t0) / max(pulse_sigma, 1e-30)) ** 2) * np.cos(
        2.0 * np.pi * f0 * (time - pulse_t0)
    )

    source = ModeSource(
        grid=grid,
        center=source_center,
        width=source_span,
        wavelength=wl0,
        pol=polarization,
        signal=signal,
        direction=src["direction"],
    )

    dft_components = (
        ("Ez", "Hx", "Hy")
        if polarization == "tm"
        else ("Ex", "Ey", "Hz")
    )
    monitor_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=dft_components,
        dft_window="hann",
        dft_record_every_step=True,
    )

    m_fwd = Monitor(
        *fwd_line,
        name=f"{source_port}_fwd",
        dft_t_start=dft_fwd_t_start,
        dft_t_end=dft_fwd_t_end,
        **monitor_cfg,
    )
    m_ref = Monitor(
        *ref_line,
        name=f"{source_port}_ref",
        dft_t_start=dft_ref_t_start,
        dft_t_end=dft_ref_t_end,
        **monitor_cfg,
    )
    output_monitors = []
    for p in output_ports:
        for cand in out_candidates[p]:
            w0, w1 = out_windows[p][cand["name"]]
            output_monitors.append(
                Monitor(
                    *cand["line"],
                    name=cand["name"],
                    dft_t_start=w0,
                    dft_t_end=w1,
                    **monitor_cfg,
                )
            )

    sim = Simulation(
        design=design,
        devices=[source, m_fwd, m_ref, *output_monitors],
        boundaries=[PML(edges="all", thickness=pml_thickness)],
        time=time,
        resolution=dx,
    )

    print(
        "Running crossing modal DFT extraction: "
        f"component={component_label}, source={source_port}, outputs={output_ports}, "
        f"pol={polarization}, freq_points={num_freqs}, steps={len(time)}, dx={dx/µm:.4f}um, "
        f"offsets(src/fwd/ref)={source_offset/µm:.2f}/{fwd_offset/µm:.2f}/{ref_offset/µm:.2f}um"
    )
    print(
        "Placement check: "
        f"source_center=({source_center[0]/µm:.2f},{source_center[1]/µm:.2f})um, "
        f"source_line_center=({src_line_center[0]/µm:.2f},{src_line_center[1]/µm:.2f})um"
    )
    for p in output_ports:
        for cand in out_candidates[p]:
            c_out = line_center(cand["line"])
            dist = float(np.hypot(c_out[0] - src_line_center[0], c_out[1] - src_line_center[1]))
            print(
                f"  monitor {cand['name']}: center=({c_out[0]/µm:.2f},{c_out[1]/µm:.2f})um, "
                f"offset={cand['offset']/µm:.2f}um, distance_to_source={dist/µm:.2f}um"
            )
    sim.run_fast(progress=False)

    cond_threshold = 1e8
    max_mode_search = 3

    def source_spec(mode_index: int) -> PortSpec:
        return PortSpec(
            name=source_port,
            monitor_name=f"{source_port}_ref",
            reference_monitor=f"{source_port}_fwd",
            direction=src["direction"],
            polarization=polarization,
            mode_index=mode_index,
        )

    # Choose source mode index.
    source_best = None
    for mode_idx in range(max_mode_search + 1):
        result = sim.get_S_matrix_modal_dft(
            source_port=source_port,
            ports={source_port: source_spec(mode_idx)},
            output_ports=[source_port],
            frequencies=freqs,
            as_sax=False,
            return_diagnostics=True,
            min_incident_db=-45.0,
        )
        waves = result["diagnostics"]["waves"].get(source_port, {})
        neff = np.asarray(waves.get("mode_neff", np.full(freqs.shape, np.nan)), dtype=float)
        cond = np.asarray(waves.get("condition_number", np.full(freqs.shape, np.inf)), dtype=float)
        valid = np.asarray(result["diagnostics"]["valid_mask"], dtype=bool)
        qual = valid & np.isfinite(cond) & (cond < cond_threshold)
        neff_med = float(np.nanmedian(neff[np.isfinite(neff)])) if np.any(np.isfinite(neff)) else -np.inf
        cond_med = float(np.nanmedian(cond[np.isfinite(cond)])) if np.any(np.isfinite(cond)) else np.inf
        qual_frac = float(np.mean(qual)) if qual.size else 0.0
        guided_bonus = 0.4 if neff_med > (n_clad + 1e-3) else 0.0
        score = 3.0 * qual_frac + guided_bonus + neff_med - 0.05 * np.log10(max(cond_med, 1.0))
        candidate = {"mode_index": mode_idx, "result": result, "score": score}
        if source_best is None or score > source_best["score"]:
            source_best = candidate

    source_mode_idx = int(source_best["mode_index"])
    source_result = source_best["result"]
    valid_mask = np.asarray(source_result["diagnostics"]["valid_mask"], dtype=bool)

    s_cols = {
        source_port: np.asarray(source_result["s_matrix"][(source_port, source_port)], dtype=np.complex128)
    }
    port_quality = {}
    source_waves = source_result["diagnostics"]["waves"].get(source_port, {})
    source_neff = np.asarray(source_waves.get("mode_neff", np.full(freqs.shape, np.nan)), dtype=float)
    source_cond = np.asarray(source_waves.get("condition_number", np.full(freqs.shape, np.inf)), dtype=float)
    port_quality[source_port] = (
        valid_mask
        & np.isfinite(source_cond)
        & (source_cond < cond_threshold)
    )

    mode_indices = {source_port: source_mode_idx}
    selected_monitors = {source_port: f"{source_port}_ref"}
    port_diagnostics = {
        source_port: {
            "neff": source_neff,
            "cond": source_cond,
        }
    }

    # Select best monitor placement + mode index for each output port.
    for p in output_ports:
        best = None
        for cand in out_candidates[p]:
            for mode_idx in range(max_mode_search + 1):
                result = sim.get_S_matrix_modal_dft(
                    source_port=source_port,
                    ports={
                        source_port: source_spec(source_mode_idx),
                        p: PortSpec(
                            name=p,
                            monitor_name=cand["name"],
                            direction=outward_direction(ports[p]["direction"]),
                            polarization=polarization,
                            mode_index=mode_idx,
                        ),
                    },
                    output_ports=[p],
                    frequencies=freqs,
                    as_sax=False,
                    return_diagnostics=True,
                    min_incident_db=-45.0,
                )
                s_p = np.asarray(result["s_matrix"][(p, source_port)], dtype=np.complex128)
                waves_p = result["diagnostics"]["waves"].get(p, {})
                neff_p = np.asarray(waves_p.get("mode_neff", np.full(freqs.shape, np.nan)), dtype=float)
                cond_p = np.asarray(waves_p.get("condition_number", np.full(freqs.shape, np.inf)), dtype=float)
                qual = (
                    valid_mask
                    & np.isfinite(cond_p)
                    & (cond_p < cond_threshold)
                )
                qual_frac = float(np.mean(qual)) if qual.size else 0.0
                if np.count_nonzero(qual) >= 4:
                    db = 20.0 * np.log10(np.maximum(np.abs(s_p[qual]), 1e-12))
                    ripple = float(np.nanstd(np.diff(db))) if db.size > 1 else 30.0
                    mag_med = float(np.nanmedian(np.abs(s_p[qual])))
                else:
                    ripple = 30.0
                    mag_med = float(np.nanmedian(np.abs(s_p))) if s_p.size else 0.0
                neff_med = float(np.nanmedian(neff_p[np.isfinite(neff_p)])) if np.any(np.isfinite(neff_p)) else -np.inf
                cond_med = float(np.nanmedian(cond_p[np.isfinite(cond_p)])) if np.any(np.isfinite(cond_p)) else np.inf
                # Prefer guided, well-conditioned, smooth spectra near expected passive range.
                score = (
                    3.0 * qual_frac
                    + (0.4 if neff_med > (n_clad + 1e-3) else 0.0)
                    + neff_med
                    - 0.05 * np.log10(max(cond_med, 1.0))
                    - 0.03 * ripple
                    - 0.6 * max(mag_med - 1.2, 0.0)
                )
                candidate = {
                    "score": score,
                    "monitor_name": cand["name"],
                    "mode_index": mode_idx,
                    "s": s_p,
                    "quality": qual,
                    "neff": neff_p,
                    "cond": cond_p,
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate

        s_cols[p] = np.asarray(best["s"], dtype=np.complex128)
        port_quality[p] = np.asarray(best["quality"], dtype=bool)
        mode_indices[p] = int(best["mode_index"])
        selected_monitors[p] = str(best["monitor_name"])
        port_diagnostics[p] = {
            "neff": np.asarray(best["neff"], dtype=float),
            "cond": np.asarray(best["cond"], dtype=float),
        }

    print(
        "Selected source mode and output monitor/mode: "
        + ", ".join(
            f"{p}={selected_monitors[p]}/m{mode_indices[p]}"
            for p in all_ports
        )
    )

    closure = np.zeros_like(wl, dtype=float)
    for p in all_ports:
        closure += np.abs(s_cols[p]) ** 2

    wl_um = wl / µm
    idx0 = int(np.argmin(np.abs(wl - wl0)))
    print(f"Center wavelength = {wl_um[idx0]:.4f} um")
    for p in all_ports:
        val = complex(s_cols[p][idx0])
        neff_p = port_diagnostics[p]["neff"]
        cond_p = port_diagnostics[p]["cond"]
        quality = bool(port_quality[p][idx0]) if idx0 < len(port_quality[p]) else False
        print(
            f"S[{p},{source_port}] @ {wl_um[idx0]:.4f}um: "
            f"|S|={abs(val):.6f}, {20*np.log10(max(abs(val), 1e-12)):.2f} dB, "
            f"neff={neff_p[idx0]:.4f}, cond={cond_p[idx0]:.2e}, "
            f"quality={quality}, monitor={selected_monitors[p]}, mode=m{mode_indices[p]}"
        )
    print(
        f"Power closure @ {wl_um[idx0]:.4f}um: {closure[idx0]:.6f} "
        f"(source_valid={bool(valid_mask[idx0])})"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "beamz_crossing_sparams.npz"
    np.savez(
        data_path,
        source_port=source_port,
        output_ports=np.asarray(all_ports, dtype=object),
        selected_monitors=np.asarray([selected_monitors[p] for p in all_ports], dtype=object),
        mode_indices=np.asarray([mode_indices[p] for p in all_ports], dtype=int),
        wavelengths_um=wl_um,
        valid_mask=valid_mask.astype(bool),
        closure=closure,
        **{f"quality_{p}": port_quality[p].astype(bool) for p in all_ports},
        **{f"s_{p}_{source_port}": s_cols[p] for p in all_ports},
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.5), dpi=320)
    color_cycle = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    db_min = 0.0
    for i, p in enumerate(all_ports):
        y_db = 20.0 * np.log10(np.maximum(np.abs(s_cols[p]), 1e-12))
        y_db = np.where(valid_mask & port_quality[p], y_db, np.nan)
        finite = np.isfinite(y_db)
        if np.any(finite):
            db_min = min(db_min, float(np.nanmin(y_db[finite])))
        ax.plot(
            wl_um,
            y_db,
            "-",
            color=color_cycle[i % len(color_cycle)],
            lw=2.0,
            label=rf"$|S_{{{p[1:]}{source_port[1:]}}}|$",
        )
    y_floor = min(-40.0, 5.0 * np.floor(db_min / 5.0))
    ax.set_xlim(float(np.min(wl_um)), float(np.max(wl_um)))
    ax.set_ylim(y_floor, 0.5)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"Crossing S-Parameters ({component_label})")
    ax.grid(which="major", alpha=0.25, lw=0.6)
    ax.minorticks_on()
    ax.grid(which="minor", alpha=0.12, lw=0.4)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig_path = out_dir / "beamz_crossing_sparams_db.png"
    fig.savefig(fig_path, dpi=320)
    plt.close(fig)

    print(f"Saved S-parameter data: {data_path}")
    print(f"Saved dB plot: {fig_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wl0-nm", type=float, default=1550.0, help="Center wavelength in nm.")
    parser.add_argument("--wl-min-nm", type=float, default=1500.0, help="Sweep min wavelength in nm.")
    parser.add_argument("--wl-max-nm", type=float, default=1600.0, help="Sweep max wavelength in nm.")
    parser.add_argument(
        "--num-freqs",
        type=int,
        default=51,
        help="Number of DFT frequency points (recommended 11..51).",
    )
    parser.add_argument("--n-core", type=float, default=3.47, help="Core refractive index.")
    parser.add_argument("--n-clad", type=float, default=1.44, help="Cladding refractive index.")
    parser.add_argument(
        "--polarization",
        type=str,
        default="te",
        choices=["te", "tm"],
        help="Modal polarization used for source/ports.",
    )
    parser.add_argument(
        "--points-per-wavelength",
        type=int,
        default=12,
        help="Grid resolution in points per wavelength.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="1,0",
        help="GDS layer,datatype used for core extraction (example: 1,0).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmarks/results/compact_models"),
        help="Output directory for compact-model data and plots.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    if args.num_freqs < 2:
        raise ValueError("--num-freqs must be >= 2.")
    if args.wl_min_nm >= args.wl_max_nm:
        raise ValueError("--wl-min-nm must be smaller than --wl-max-nm.")
    if args.wl0_nm < args.wl_min_nm or args.wl0_nm > args.wl_max_nm:
        raise ValueError("--wl0-nm must be within [wl-min-nm, wl-max-nm].")

    run_crossing(
        wl0=args.wl0_nm * 1e-9,
        wl_min=args.wl_min_nm * 1e-9,
        wl_max=args.wl_max_nm * 1e-9,
        num_freqs=args.num_freqs,
        n_core=args.n_core,
        n_clad=args.n_clad,
        polarization=args.polarization,
        points_per_wavelength=args.points_per_wavelength,
        layer=parse_layer(args.layer),
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
