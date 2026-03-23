"""Broadband 3D modal S-parameter extraction for a 4-port crossing.

Workflow:
1. Import a crossing from UBC PDK (fallback: gdsfactory generic crossing).
2. Build a 3D BeamZ design (explicit layer stack) and extend each port.
3. Launch a Gaussian pulse at one source port.
4. Use DFT monitors + modal decomposition to extract S11/S21/S31/S41 over frequency.
5. Save a compact-model data file and a dB plot.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time as time_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

from beamz import (
    LIGHT_SPEED,
    Design,
    Material,
    ModeSource,
    Monitor,
    PML,
    Polygon,
    PortSpec,
    Simulation,
    dxdt,
    µm,
)
from beamz.design.io import gdsf
from beamz.devices.sources.signals import gaussian_pulse


def outward_direction(direction: str) -> str:
    return ("-" if direction.startswith("+") else "+") + direction[1:]


def positive_axis_direction(direction: str) -> str:
    return "+" + str(direction)[1:]


def incoming_wave_selector(direction: str) -> str:
    return "plus" if str(direction).startswith("+") else "minus"


def outgoing_wave_selector(direction: str) -> str:
    return "minus" if str(direction).startswith("+") else "plus"


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


def parse_layer(layer_str: str) -> tuple[int, int] | None:
    if str(layer_str).strip().lower() == "auto":
        return None
    parts = [p.strip() for p in layer_str.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid layer '{layer_str}'. Use 'layer,datatype', e.g. '1,0'.")
    return int(parts[0]), int(parts[1])


def load_crossing_component(component_name: str = "ebeam_crossing4"):
    """Return (component, label) preferring PDK crossing over generic fallback."""
    try:
        from ubcpdk import PDK, cells

        PDK.activate()
        if hasattr(cells, component_name):
            return getattr(cells, component_name)(), f"ubcpdk.cells.{component_name}"
        if hasattr(cells, "ebeam_crossing4"):
            return cells.ebeam_crossing4(), "ubcpdk.cells.ebeam_crossing4"
    except Exception as exc:
        ubc_exc = exc
    else:
        ubc_exc = None

    import gdsfactory as gf

    if hasattr(gf, "gpdk") and hasattr(gf.gpdk, "PDK"):
        gf.gpdk.PDK.activate()
    else:
        try:
            from gdsfactory.pdk import get_active_pdk

            active_pdk = get_active_pdk()
            if active_pdk is not None and hasattr(active_pdk, "activate"):
                active_pdk.activate()
        except Exception:
            pass
    for name in [component_name, "crossing", "crossing45"]:
        try:
            return gf.get_component(name), f"gf.get_component('{name}')"
        except Exception:
            continue
    print(
        "[beamz_crossing] Could not load requested PDK crossing; using generic fallback. "
        f"UBC load reason: {type(ubc_exc).__name__ if ubc_exc else 'n/a'}: {ubc_exc}"
    )
    return gf.components.crossing(), "gdsfactory.components.crossing"


QUALITY_PRESETS = {
    "fast": {
        "num_freqs": 11,
        "points_per_wavelength": 10,
        "run_after_sources_uoc": 18.0,
        "animation_frames": 0,
    },
    "high": {
        "num_freqs": 51,
        "points_per_wavelength": 20,
        "run_after_sources_uoc": 45.0,
        "animation_frames": 36,
    },
}


def cli_option_present(argv: list[str], *flags: str) -> bool:
    for token in argv:
        for flag in flags:
            if token == flag or token.startswith(flag + "="):
                return True
    return False


def apply_quality_preset(args, argv: list[str]):
    preset = QUALITY_PRESETS[str(args.quality)]
    option_map = {
        "num_freqs": ("--num-freqs",),
        "points_per_wavelength": ("--points-per-wavelength",),
        "run_after_sources_uoc": ("--run-after-sources-uoc",),
        "animation_frames": ("--animation-frames",),
    }
    applied = {}
    for attr, value in preset.items():
        if not cli_option_present(argv, *option_map[attr]):
            setattr(args, attr, value)
            applied[attr] = value
    return applied


def _layer_spec_to_tuple(layer_spec, pdk=None) -> tuple[int, int] | None:
    if layer_spec is None:
        return None
    if isinstance(layer_spec, tuple) and len(layer_spec) == 2:
        return int(layer_spec[0]), int(layer_spec[1])
    if hasattr(layer_spec, "layer") and hasattr(layer_spec, "datatype"):
        return int(layer_spec.layer), int(layer_spec.datatype)
    resolved = layer_spec
    if pdk is not None:
        try:
            resolved = pdk.get_layer(layer_spec)
        except Exception:
            resolved = layer_spec
    if isinstance(resolved, tuple) and len(resolved) == 2:
        return int(resolved[0]), int(resolved[1])
    if hasattr(resolved, "layer") and hasattr(resolved, "datatype"):
        return int(resolved.layer), int(resolved.datatype)
    if isinstance(resolved, int):
        dtype = int(getattr(layer_spec, "datatype", 0))
        layer_num = int(getattr(layer_spec, "layer", resolved))
        return layer_num, dtype
    return None


def resolve_pdk_stack(
    component,
    *,
    layer: tuple[int, int] | None,
    core_t_um: float,
    clad_below_um: float,
    clad_above_um: float,
    use_pdk_stack: bool,
) -> tuple[tuple[int, int], float, float, float, dict[str, object]]:
    layer_resolved = layer
    core_t_out = float(core_t_um)
    clad_below_out = float(clad_below_um)
    clad_above_out = float(clad_above_um)
    meta: dict[str, object] = {"used_pdk_stack": False}

    if not use_pdk_stack:
        if layer_resolved is None:
            layer_resolved = (1, 0)
        meta["selected_layer"] = layer_resolved
        return layer_resolved, core_t_out, clad_below_out, clad_above_out, meta

    try:
        import gdsfactory as gf

        pdk = gf.get_active_pdk() if hasattr(gf, "get_active_pdk") else None
    except Exception:
        pdk = None
    if pdk is None:
        if layer_resolved is None:
            layer_resolved = (1, 0)
        meta["selected_layer"] = layer_resolved
        return layer_resolved, core_t_out, clad_below_out, clad_above_out, meta

    polygons_by_layer = component.get_polygons_points(by="tuple")
    available_layers = set(polygons_by_layer.keys())
    layer_stack = getattr(pdk, "layer_stack", None)
    levels = getattr(layer_stack, "layers", {}) if layer_stack is not None else {}

    core_level = None
    for lname, level in levels.items():
        try:
            name_key = str(lname).lower()
            th = float(getattr(level, "thickness", 0.0))
            mat = str(getattr(level, "material", "")).lower()
        except Exception:
            continue
        if th <= 0:
            continue
        if "core" in name_key:
            core_level = level
            break
        if core_level is None and ("si3n4" in mat or "sin" == mat or "nitride" in mat):
            core_level = level

    core_layer_from_stack = _layer_spec_to_tuple(
        getattr(core_level, "layer", None) if core_level is not None else None,
        pdk=pdk,
    )
    if layer_resolved is None:
        layer_resolved = core_layer_from_stack
        if layer_resolved not in available_layers and layer_resolved is not None:
            layer_num = int(layer_resolved[0])
            same_num = [lt for lt in available_layers if int(lt[0]) == layer_num]
            if same_num:
                layer_resolved = sorted(same_num, key=lambda lt: int(lt[1]))[0]
        if layer_resolved is None:
            layer_resolved = (1, 0) if (1, 0) in available_layers else sorted(available_layers)[0]

    if core_level is not None:
        try:
            core_z0 = float(getattr(core_level, "zmin"))
            core_th = float(getattr(core_level, "thickness"))
            core_t_out = core_th
            core_top = core_z0 + core_th
            oxide_levels = []
            for level in levels.values():
                try:
                    z0 = float(getattr(level, "zmin"))
                    th = float(getattr(level, "thickness"))
                    z1 = z0 + th
                    mat = str(getattr(level, "material", "")).lower()
                except Exception:
                    continue
                if th <= 0:
                    continue
                if any(k in mat for k in ("sio2", "oxide", "silica", "glass")):
                    oxide_levels.append((z0, z1))

            if oxide_levels:
                below_candidates = [z0 for z0, z1 in oxide_levels if z1 <= core_z0 + 1e-9]
                if below_candidates:
                    clad_below_out = max(core_z0 - min(below_candidates), 0.0)

                above_cover = [z1 for z0, z1 in oxide_levels if z0 <= core_top + 1e-9 and z1 >= core_top - 1e-9]
                if above_cover:
                    clad_above_out = max(max(above_cover) - core_top, 0.0)
                else:
                    above_candidates = [z1 for z0, z1 in oxide_levels if z0 >= core_top - 1e-9]
                    if above_candidates:
                        clad_above_out = max(max(above_candidates) - core_top, 0.0)
            meta["used_pdk_stack"] = True
            meta["core_layer_stack"] = core_layer_from_stack
            meta["core_material"] = str(getattr(core_level, "material", ""))
        except Exception:
            pass

    meta["selected_layer"] = layer_resolved
    return layer_resolved, core_t_out, clad_below_out, clad_above_out, meta


def port_line(
    port: dict,
    span: float,
    offset: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    cx, cy = move_along(port["center"], port["direction"], offset)
    if port["direction"].endswith("x"):
        return (cx, cy - 0.5 * span), (cx, cy + 0.5 * span)
    return (cx - 0.5 * span, cy), (cx + 0.5 * span, cy)


def port_plane(
    port: dict,
    *,
    y_span: float,
    z_span: float,
    z_center: float,
    offset: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cx, cy = move_along(port["center"], port["direction"], offset)
    z0 = z_center - 0.5 * z_span
    z1 = z_center + 0.5 * z_span
    if port["direction"].endswith("x"):
        return (cx, cy - 0.5 * y_span, z0), (cx, cy + 0.5 * y_span, z1)
    return (cx - 0.5 * y_span, cy, z0), (cx + 0.5 * y_span, cy, z1)


def line_center(line):
    a, b = line
    if len(a) == 3:
        return 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])
    return 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])


def line_box(line):
    a, b = line
    dim = len(a)
    lo = tuple(min(float(a[i]), float(b[i])) for i in range(dim))
    hi = tuple(max(float(a[i]), float(b[i])) for i in range(dim))
    return lo, hi


def point_clearance_to_box(point, bounds):
    lo, hi = bounds
    coords = tuple(float(v) for v in point)
    return float(
        min(min(coords[i] - lo[i], hi[i] - coords[i]) for i in range(len(coords)))
    )


def line_clearance_to_box(line, bounds):
    lo_line, hi_line = line_box(line)
    lo_box, hi_box = bounds
    return float(
        min(
            min(
                lo_line[i] - lo_box[i],
                hi_box[i] - hi_line[i],
            )
            for i in range(len(lo_line))
        )
    )


def cleanup_crossing_optional_artifacts(
    out_dir: Path,
    *,
    write_plots: bool,
    write_mode_plots: bool,
    write_animation: bool,
) -> None:
    if not write_plots:
        for name in (
            "beamz_crossing_sparams_db.png",
            "beamz_crossing_sparams_db_full.png",
            "beamz_crossing_sparams_raw_vs_corrected.png",
            "beamz_crossing_closure_compare.png",
            "beamz_crossing_overview.png",
            "beamz_crossing_signal.png",
        ):
            (out_dir / name).unlink(missing_ok=True)
    if not write_animation:
        (out_dir / "beamz_crossing_field_propagation.mp4").unlink(missing_ok=True)
    if not write_mode_plots:
        shutil.rmtree(out_dir / "modes", ignore_errors=True)


def safe_complex_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    out = np.zeros_like(np.asarray(num, dtype=np.complex128), dtype=np.complex128)
    den_arr = np.asarray(den, dtype=np.complex128)
    valid = np.abs(den_arr) > float(eps)
    out[valid] = np.asarray(num, dtype=np.complex128)[valid] / den_arr[valid]
    return out


def safe_real_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    num_arr = np.asarray(num, dtype=float)
    den_arr = np.asarray(den, dtype=float)
    out = np.full_like(num_arr, np.nan, dtype=float)
    valid = np.abs(den_arr) > float(eps)
    out[valid] = num_arr[valid] / den_arr[valid]
    return out


def wave_dominance_db(
    selected_wave: np.ndarray,
    opposite_wave: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-18,
) -> float:
    sel = np.asarray(selected_wave, dtype=np.complex128)
    opp = np.asarray(opposite_wave, dtype=np.complex128)
    mask = np.asarray(valid_mask, dtype=bool)
    if sel.shape != opp.shape:
        n = min(sel.size, opp.size)
        sel = sel.reshape(-1)[:n]
        opp = opp.reshape(-1)[:n]
        mask = mask.reshape(-1)[:n]
    if not np.any(mask):
        return float("nan")
    p_sel = np.mean(np.abs(sel[mask]) ** 2)
    p_opp = np.mean(np.abs(opp[mask]) ** 2)
    return float(10.0 * np.log10(max(p_sel, eps) / max(p_opp, eps)))


def select_dominant_wave(
    a_plus: np.ndarray,
    a_minus: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray, float]:
    plus = np.asarray(a_plus, dtype=np.complex128)
    minus = np.asarray(a_minus, dtype=np.complex128)
    mask = np.asarray(valid_mask, dtype=bool)
    if plus.shape != minus.shape:
        n = min(plus.size, minus.size)
        plus = plus.reshape(-1)[:n]
        minus = minus.reshape(-1)[:n]
        mask = mask.reshape(-1)[:n]
    if not np.any(mask):
        mask = np.ones_like(plus, dtype=bool)
    p_plus = float(np.mean(np.abs(plus[mask]) ** 2))
    p_minus = float(np.mean(np.abs(minus[mask]) ** 2))
    if p_plus >= p_minus:
        return "plus", plus, minus, wave_dominance_db(plus, minus, mask)
    return "minus", minus, plus, wave_dominance_db(minus, plus, mask)


def choose_wave_by_selector(
    a_plus: np.ndarray,
    a_minus: np.ndarray,
    selector: str,
) -> tuple[np.ndarray, np.ndarray]:
    plus = np.asarray(a_plus, dtype=np.complex128)
    minus = np.asarray(a_minus, dtype=np.complex128)
    key = str(selector).lower()
    if key == "plus":
        return plus, minus
    if key == "minus":
        return minus, plus
    raise ValueError(f"Unsupported selector {selector!r}; expected 'plus' or 'minus'.")


def dft_directional_power_spectrum(
    sim: Simulation,
    monitor: Monitor,
    direction: str,
    frequencies: np.ndarray,
) -> np.ndarray:
    freqs = np.asarray(frequencies, dtype=float)
    comps = {}
    for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        _, spec = sim._sample_monitor_component_dft(monitor, c, frequencies=freqs)
        comps[c] = np.asarray(spec, dtype=np.complex128)
    n_pts = min(arr.shape[1] for arr in comps.values())
    if n_pts <= 0:
        return np.zeros(freqs.shape, dtype=float)
    ex = comps["Ex"][:, :n_pts]
    ey = comps["Ey"][:, :n_pts]
    ez = comps["Ez"][:, :n_pts]
    hx = comps["Hx"][:, :n_pts]
    hy = comps["Hy"][:, :n_pts]
    hz = comps["Hz"][:, :n_pts]
    axis = str(direction)[1]
    sign = 1.0 if str(direction).startswith("+") else -1.0
    if axis == "x":
        s_axis = 0.5 * np.real(ey * np.conjugate(hz) - ez * np.conjugate(hy))
    elif axis == "y":
        s_axis = 0.5 * np.real(ez * np.conjugate(hx) - ex * np.conjugate(hz))
    elif axis == "z":
        s_axis = 0.5 * np.real(ex * np.conjugate(hy) - ey * np.conjugate(hx))
    else:
        raise ValueError(f"Unsupported direction '{direction}'.")
    d_area = float(sim.resolution) * float(sim.resolution)
    return np.asarray(sign * np.sum(s_axis, axis=1) * d_area, dtype=float)


def save_signal_plot(time: np.ndarray, signal: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.6), dpi=260)
    ax.plot(time / 1e-15, signal, color="black", lw=1.5)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("amplitude")
    ax.set_title("Excitation Signal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_closure_compare_plot(
    wavelengths_um: np.ndarray,
    modal_closure: np.ndarray,
    flux_closure: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
) -> None:
    wl = np.asarray(wavelengths_um, dtype=float)
    modal = np.asarray(modal_closure, dtype=float)
    flux = np.asarray(flux_closure, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    modal = np.where(mask, modal, np.nan)
    flux = np.where(mask, flux, np.nan)
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.4), dpi=280)
    ax.plot(wl, modal, color="tab:blue", lw=1.9, label="Modal closure")
    ax.plot(wl, flux, color="tab:orange", lw=1.9, label="Flux closure")
    ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Closure")
    ax.set_title("Modal vs Flux Closure")
    ax.grid(alpha=0.28)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_raw_vs_corrected_sparams_plot(
    *,
    wavelengths_um: np.ndarray,
    source_port: str,
    all_ports: list[str],
    s_cols_raw: dict[str, np.ndarray],
    s_cols: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    port_quality: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    wl = np.asarray(wavelengths_um, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    color_cycle = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.8), dpi=300)
    for i, p in enumerate(all_ports):
        q = np.asarray(port_quality[p], dtype=bool)
        raw_db = 20.0 * np.log10(np.maximum(np.abs(np.asarray(s_cols_raw[p], dtype=np.complex128)), 1e-12))
        corr_db = 20.0 * np.log10(np.maximum(np.abs(np.asarray(s_cols[p], dtype=np.complex128)), 1e-12))
        raw_db = np.where(mask & q, raw_db, np.nan)
        corr_db = np.where(mask & q, corr_db, np.nan)
        color = color_cycle[i % len(color_cycle)]
        ax.plot(
            wl,
            raw_db,
            color=color,
            lw=1.4,
            ls="--",
            alpha=0.8,
            label=rf"raw $|S_{{{p[1:]}{source_port[1:]}}}|$",
        )
        ax.plot(
            wl,
            corr_db,
            color=color,
            lw=2.0,
            label=rf"corr $|S_{{{p[1:]}{source_port[1:]}}}|$",
        )
    ax.set_xlim(float(np.min(wl)), float(np.max(wl)))
    ax.set_ylim(-55.0, 2.0)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Raw vs Corrected S-Parameters")
    ax.grid(which="major", alpha=0.25, lw=0.6)
    ax.minorticks_on()
    ax.grid(which="minor", alpha=0.12, lw=0.4)
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_mode_profile_plot(
    *,
    label: str,
    mode_src: ModeSource,
    grid_eps: np.ndarray,
    dx: float,
    out_path: Path,
) -> None:
    axis = mode_src.direction[1]
    eps2d = np.asarray(getattr(mode_src, "_eps_profile_2d", np.array([])))
    if eps2d.ndim == 2 and eps2d.size > 0:
        profile_map = {
            "Ex": getattr(mode_src, "_Ex_profile", None),
            "Ey": getattr(mode_src, "_Ey_profile", None),
            "Ez": getattr(mode_src, "_Ez_profile", None),
            "Hx": getattr(mode_src, "_Hx_profile", None),
            "Hy": getattr(mode_src, "_Hy_profile", None),
            "Hz": getattr(mode_src, "_Hz_profile", None),
        }
        fig, ax = plt.subplots(2, 4, figsize=(10.8, 5.4), dpi=250)
        ax = ax.ravel()
        im_eps = ax[0].imshow(eps2d, origin="lower", cmap="viridis", aspect="equal")
        ax[0].set_title(f"{label}: eps")
        fig.colorbar(im_eps, ax=ax[0], fraction=0.046, pad=0.04)
        for i, name in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], start=1):
            arr = profile_map[name]
            if arr is None:
                ax[i].axis("off")
                continue
            a2 = np.asarray(arr).squeeze()
            if a2.ndim != 2:
                a2 = np.atleast_2d(a2)
            im = ax[i].imshow(np.abs(a2), origin="lower", cmap="magma", aspect="equal")
            ax[i].set_title(f"{label}: |{name}|")
            fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        ax[7].axis("off")
        ax[7].text(
            0.02,
            0.95,
            (
                f"pol={mode_src.pol}\n"
                f"dir={mode_src.direction}\n"
                f"axis={axis}\n"
                f"neff={float(np.real(getattr(mode_src, '_neff', np.nan))):.5f}\n"
                f"width={float(mode_src.width)/µm:.3f}um\n"
                f"height={float(getattr(mode_src, 'height', 0.0) or 0.0)/µm:.3f}um"
            ),
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    if grid_eps.ndim == 3:
        zc = int(np.clip(round(float(mode_src.center[2]) / dx), 0, grid_eps.shape[0] - 1))
        yc = int(np.clip(round(float(mode_src.center[1]) / dx), 0, grid_eps.shape[1] - 1))
        xc = int(np.clip(round(float(mode_src.center[0]) / dx), 0, grid_eps.shape[2] - 1))
        eps_profile = np.asarray(grid_eps[zc, :, xc] if axis == "x" else grid_eps[zc, yc, :], dtype=float)
    else:
        if axis == "x":
            x_idx = int(np.clip(round(float(mode_src.center[0]) / dx), 0, grid_eps.shape[1] - 1))
            eps_profile = np.asarray(grid_eps[:, x_idx], dtype=float)
        else:
            y_idx = int(np.clip(round(float(mode_src.center[1]) / dx), 0, grid_eps.shape[0] - 1))
            eps_profile = np.asarray(grid_eps[y_idx, :], dtype=float)

    profiles = {
        "jz": getattr(mode_src, "_jz_profile", None),
        "jy": getattr(mode_src, "_jy_profile", None),
        "jx": getattr(mode_src, "_jx_profile", None),
        "my": getattr(mode_src, "_my_profile", None),
        "mz": getattr(mode_src, "_mz_profile", None),
    }
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.2, 2.8), dpi=260)
    u = np.arange(eps_profile.size, dtype=float) * dx / µm
    ax0.plot(u, eps_profile, color="tab:blue", lw=1.6)
    ax0.set_title(f"{label}: eps profile")
    ax0.set_xlabel("transverse coordinate (um)")
    ax0.set_ylabel("permittivity")
    ax0.grid(alpha=0.3)

    plotted = False
    for name, arr in profiles.items():
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.complex128).reshape(-1)
        if a.size == 0:
            continue
        uu = np.arange(a.size, dtype=float) * dx / µm
        ax1.plot(uu, np.abs(a), lw=1.5, label=f"|{name}|")
        plotted = True
    if not plotted:
        ax1.text(0.02, 0.7, "No mode profile data", transform=ax1.transAxes)
    ax1.set_title(
        f"{label}: mode profile ({mode_src.pol}, {mode_src.direction})\n"
        f"neff={float(np.real(getattr(mode_src, '_neff', np.nan))):.4f}"
    )
    ax1.set_xlabel("transverse coordinate (um)")
    ax1.set_ylabel("normalized magnitude")
    ax1.grid(alpha=0.3)
    if plotted:
        ax1.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_overview_plot(
    *,
    eps: np.ndarray,
    width: float,
    height: float,
    depth: float,
    imported_bbox: tuple[float, float, float, float],
    source_plane: tuple[tuple[float, float, float], tuple[float, float, float]],
    monitor_planes: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
    layer_z: dict[str, tuple[float, float]],
    pml_xy: float = 0.0,
    pml_z: float = 0.0,
    out_path: Path,
) -> None:
    x0, x1, y0, y1 = imported_bbox
    if eps.ndim == 3:
        nz, ny, nx = eps.shape
        dz = depth / max(nz, 1)
        dy = height / max(ny, 1)
        dx = width / max(nx, 1)
        core_z0, core_z1 = layer_z.get("core", (0.0, depth))
        z_core_idx = int(np.clip(round(0.5 * (core_z0 + core_z1) / dz), 0, nz - 1))
        y_mid_idx = int(np.clip(round(0.5 * (y0 + y1) / dy), 0, ny - 1))
        x_mid_idx = int(np.clip(round(0.5 * (x0 + x1) / dx), 0, nx - 1))
        eps_xy = np.asarray(eps[z_core_idx], dtype=float)
        eps_xz = np.asarray(eps[:, y_mid_idx, :], dtype=float)
        eps_yz = np.asarray(eps[:, :, x_mid_idx], dtype=float)
    else:
        dy = height / max(eps.shape[0], 1)
        dx = width / max(eps.shape[1], 1)
        y_mid_idx = int(np.clip(round(0.5 * (y0 + y1) / dy), 0, eps.shape[0] - 1))
        x_mid_idx = int(np.clip(round(0.5 * (x0 + x1) / dx), 0, eps.shape[1] - 1))
        eps_xy = np.asarray(eps, dtype=float)
        eps_x = np.asarray(eps[y_mid_idx, :], dtype=float)
        eps_y = np.asarray(eps[:, x_mid_idx], dtype=float)
        nz_vis = 40
        eps_xz = np.repeat(eps_x[None, :], nz_vis, axis=0)
        eps_yz = np.repeat(eps_y[:, None], nz_vis, axis=1).T

    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.6), dpi=260)
    ax[0].imshow(
        eps_xy,
        origin="lower",
        extent=[0.0, width / µm, 0.0, height / µm],
        cmap="viridis",
        aspect="equal",
    )
    ax[0].set_title("XY overview (core z-slice)")
    ax[0].set_xlabel("x (um)")
    ax[0].set_ylabel("y (um)")
    pml_xy_um = max(0.0, float(pml_xy) / µm)
    pml_z_um = max(0.0, float(pml_z) / µm)
    x_max_um = float(width / µm)
    y_max_um = float(height / µm)
    z_max_um = float(depth / µm)
    if pml_xy_um > 0.0:
        ax[0].axvspan(0.0, min(pml_xy_um, x_max_um), color="tab:orange", alpha=0.12)
        ax[0].axvspan(max(0.0, x_max_um - pml_xy_um), x_max_um, color="tab:orange", alpha=0.12)
        ax[0].axhspan(0.0, min(pml_xy_um, y_max_um), color="tab:orange", alpha=0.12)
        ax[0].axhspan(max(0.0, y_max_um - pml_xy_um), y_max_um, color="tab:orange", alpha=0.12)
        ax[0].text(
            max(0.05, 0.03 * x_max_um),
            max(0.05, 0.03 * y_max_um),
            "PML",
            color="tab:orange",
            fontsize=7,
            weight="bold",
        )
    ax[0].plot([x0 / µm, x1 / µm, x1 / µm, x0 / µm, x0 / µm], [y0 / µm, y0 / µm, y1 / µm, y1 / µm, y0 / µm], "w--", lw=1.2)
    for name, plane in [("source", source_plane), *monitor_planes.items()]:
        (xa, ya, _za), (xb, yb, _zb) = plane
        color = "red" if name == "source" else "white"
        lw = 1.8 if name == "source" else 1.1
        ax[0].plot([xa / µm, xb / µm], [ya / µm, yb / µm], color=color, lw=lw)
        xc, yc = 0.5 * (xa + xb), 0.5 * (ya + yb)
        ax[0].text(xc / µm, yc / µm + 0.08, name, color=color, fontsize=6.5, ha="center")

    ax[1].imshow(
        eps_xz,
        origin="lower",
        extent=[0.0, width / µm, 0.0, depth / µm],
        cmap="viridis",
        aspect="auto",
    )
    ax[1].set_title("XZ")
    ax[1].set_xlabel("x (um)")
    ax[1].set_ylabel("z (um)")
    if pml_xy_um > 0.0:
        ax[1].axvspan(0.0, min(pml_xy_um, x_max_um), color="tab:orange", alpha=0.12)
        ax[1].axvspan(max(0.0, x_max_um - pml_xy_um), x_max_um, color="tab:orange", alpha=0.12)
    if pml_z_um > 0.0:
        ax[1].axhspan(0.0, min(pml_z_um, z_max_um), color="tab:orange", alpha=0.12)
        ax[1].axhspan(max(0.0, z_max_um - pml_z_um), z_max_um, color="tab:orange", alpha=0.12)
        ax[1].text(
            max(0.05, 0.03 * x_max_um),
            max(0.05, 0.03 * z_max_um),
            "PML",
            color="tab:orange",
            fontsize=7,
            weight="bold",
        )
    for lyr, (lz0, lz1) in layer_z.items():
        ax[1].axhspan(lz0 / µm, lz1 / µm, color="white", alpha=0.05)
        ax[1].text(0.15, 0.5 * (lz0 + lz1) / µm, lyr, color="white", fontsize=6.5, va="center")
    for name, plane in [("source", source_plane), *monitor_planes.items()]:
        (xa, _, za), (xb, _, zb) = plane
        xc = 0.5 * (xa + xb)
        color = "red" if name == "source" else "white"
        ax[1].plot([xc / µm, xc / µm], [min(za, zb) / µm, max(za, zb) / µm], color=color, lw=1.1, alpha=0.9)

    ax[2].imshow(
        eps_yz,
        origin="lower",
        extent=[0.0, height / µm, 0.0, depth / µm],
        cmap="viridis",
        aspect="auto",
    )
    ax[2].set_title("YZ")
    ax[2].set_xlabel("y (um)")
    ax[2].set_ylabel("z (um)")
    if pml_xy_um > 0.0:
        ax[2].axvspan(0.0, min(pml_xy_um, y_max_um), color="tab:orange", alpha=0.12)
        ax[2].axvspan(max(0.0, y_max_um - pml_xy_um), y_max_um, color="tab:orange", alpha=0.12)
    if pml_z_um > 0.0:
        ax[2].axhspan(0.0, min(pml_z_um, z_max_um), color="tab:orange", alpha=0.12)
        ax[2].axhspan(max(0.0, z_max_um - pml_z_um), z_max_um, color="tab:orange", alpha=0.12)
        ax[2].text(
            max(0.05, 0.03 * y_max_um),
            max(0.05, 0.03 * z_max_um),
            "PML",
            color="tab:orange",
            fontsize=7,
            weight="bold",
        )
    for lyr, (lz0, lz1) in layer_z.items():
        ax[2].axhspan(lz0 / µm, lz1 / µm, color="white", alpha=0.05)
        ax[2].text(0.15, 0.5 * (lz0 + lz1) / µm, lyr, color="white", fontsize=6.5, va="center")
    for name, plane in [("source", source_plane), *monitor_planes.items()]:
        (_, ya, za), (_, yb, zb) = plane
        yc = 0.5 * (ya + yb)
        color = "red" if name == "source" else "white"
        ax[2].plot([yc / µm, yc / µm], [min(za, zb) / µm, max(za, zb) / µm], color=color, lw=1.1, alpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=320)
    plt.close(fig)


def save_field_animation(
    *,
    field_hist: np.ndarray,
    eps: np.ndarray,
    width: float,
    height: float,
    field_label: str,
    out_path: Path,
    fps: int = 20,
) -> bool:
    if field_hist.ndim not in {3, 4} or field_hist.shape[0] < 2:
        return False
    try:
        from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
    except Exception:
        return False
    if not writers.is_available("ffmpeg"):
        return False

    if eps.ndim == 3:
        nz, ny, nx = eps.shape
        dx_x = width / max(nx, 1)
        dx_y = height / max(ny, 1)
        core_z_idx = int(np.clip(round(0.5 * nz), 0, nz - 1))
    else:
        ny, nx = eps.shape
        dx_x = width / max(nx, 1)
        dx_y = height / max(ny, 1)
        core_z_idx = 0
    y_slice = max(0, min(ny - 1, ny // 2))

    if field_hist.ndim == 4:
        f_xy = np.asarray(field_hist[:, core_z_idx, :, :], dtype=float)
    else:
        f_xy = np.asarray(field_hist, dtype=float)
    f_view = np.asarray(f_xy, dtype=float)
    vmax = float(np.nanpercentile(np.abs(f_view), 99.0))
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=220)
    ax0, ax1 = axes
    im = ax0.imshow(
        f_view[0],
        origin="lower",
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        extent=[0.0, width / µm, 0.0, height / µm],
        aspect="equal",
    )
    ax0.set_title(f"{field_label} field (full XY domain)")
    ax0.set_xlabel("x (um)")
    ax0.set_ylabel("y (um)")
    y_slice_um = y_slice * dx_y / µm
    slice_line = ax0.axhline(y_slice_um, color="black", ls="--", lw=1.0, alpha=0.8)

    x_line = np.arange(nx, dtype=float) * dx_x / µm
    line_vals = np.asarray(f_xy[0, y_slice, :], dtype=float)
    (line_plot,) = ax1.plot(x_line, line_vals, color="tab:blue", lw=1.6)
    ax1.set_xlim(float(np.min(x_line)), float(np.max(x_line)))
    ax1.set_ylim(-vmax, vmax)
    ax1.set_xlabel("x (um)")
    ax1.set_ylabel(field_label)
    ax1.set_title("Mid-cell line slice")
    ax1.grid(alpha=0.3)
    frame_text = ax1.text(0.03, 0.93, "", transform=ax1.transAxes, va="top")

    def _update(i):
        im.set_data(f_view[i])
        line_plot.set_ydata(np.asarray(f_xy[i, y_slice, :], dtype=float))
        frame_text.set_text(f"frame {i+1}/{field_hist.shape[0]}")
        return im, line_plot, slice_line, frame_text

    ani = FuncAnimation(fig, _update, frames=field_hist.shape[0], interval=max(1000 // max(fps, 1), 1), blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    return True


def build_design_with_extensions(
    component,
    *,
    layer: tuple[int, int],
    n_core: float,
    n_clad: float,
    xy_padding: float,
    z_padding: float,
    extension: float,
    port_overlap: float,
    core_t: float,
    clad_below: float,
    clad_above: float,
) -> tuple[Design, dict, tuple[float, float, float, float], dict[str, tuple[float, float]]]:
    imported_design, raw_ports = gdsf.load(
        component,
        layer=layer,
        n_core=n_core,
        n_clad=n_clad,
        padding=0.0,
    )
    depth = 2.0 * z_padding + clad_below + core_t + clad_above
    core_z0 = z_padding + clad_below
    core_z1 = core_z0 + core_t

    design = Design(
        width=imported_design.width + 2.0 * xy_padding,
        height=imported_design.height + 2.0 * xy_padding,
        depth=depth,
        material=Material(n_clad**2),
    )
    core_shapes_2d = []
    passthrough_structures = []
    for structure in imported_design.structures[1:]:
        verts = getattr(structure, "vertices", None)
        if not verts:
            passthrough_structures.append(structure)
            continue
        shell = [(float(x), float(y)) for x, y, *_ in verts]
        holes = [
            [(float(x), float(y)) for x, y, *_ in interior]
            for interior in getattr(structure, "interiors", []) or []
        ]
        poly = ShapelyPolygon(shell=shell, holes=holes)
        if poly.is_empty or not poly.is_valid:
            passthrough_structures.append(structure)
            continue
        core_shapes_2d.append(poly)

    def _face_span_for_port(port: dict) -> tuple[float, float] | None:
        """Return transverse (center, span) from the actual imported core face."""
        face_coord = float(port["center"][0] if port["direction"].endswith("x") else port["center"][1])
        tol = max(1e-12, 0.05 * float(port["width"]))
        transverse_coords = []
        for structure in imported_design.structures[1:]:
            verts = getattr(structure, "vertices", None)
            if not verts:
                continue
            for vx, vy, *_ in verts:
                if port["direction"].endswith("x"):
                    if abs(float(vx) - face_coord) <= tol:
                        transverse_coords.append(float(vy))
                else:
                    if abs(float(vy) - face_coord) <= tol:
                        transverse_coords.append(float(vx))
        if len(transverse_coords) < 2:
            return None
        lo = min(transverse_coords)
        hi = max(transverse_coords)
        if hi <= lo:
            return None
        return 0.5 * (lo + hi), hi - lo

    port_faces = {name: _face_span_for_port(p) for name, p in raw_ports.items()}
    ports = {
        name: {
            **p,
            "center": (
                float(((port_faces[name][0] if port_faces[name] is not None else p["center"][0]) if p["direction"].endswith("y") else p["center"][0]) + xy_padding),
                float(((port_faces[name][0] if port_faces[name] is not None else p["center"][1]) if p["direction"].endswith("x") else p["center"][1]) + xy_padding),
            ),
            "width": float(port_faces[name][1] if port_faces[name] is not None else p["width"]),
            "z_center": float(core_z0 + 0.5 * core_t),
        }
        for name, p in raw_ports.items()
    }

    # Extend each port outward and slightly inward, then merge extensions into the imported
    # core geometry so rasterization sees one continuous shape without stitched seams.
    for port in raw_ports.values():
        cx, cy = map(float, port["center"])
        width = float(port["width"])
        d_out = outward_direction(port["direction"])
        sx, sy = move_along((cx, cy), d_out, -port_overlap)
        ox, oy = move_along((cx, cy), d_out, extension)
        if port["direction"].endswith("x"):
            core_shapes_2d.append(
                shapely_box(
                    min(sx, ox),
                    cy - 0.5 * width,
                    max(sx, ox),
                    cy + 0.5 * width,
                )
            )
        else:
            core_shapes_2d.append(
                shapely_box(
                    cx - 0.5 * width,
                    min(sy, oy),
                    cx + 0.5 * width,
                    max(sy, oy),
                )
            )

    if core_shapes_2d:
        merged_core = unary_union(core_shapes_2d)
        merged_geoms = (
            [merged_core]
            if merged_core.geom_type == "Polygon"
            else list(getattr(merged_core, "geoms", []))
        )
        for geom in merged_geoms:
            shell = [(float(x + xy_padding), float(y + xy_padding), core_z0) for x, y in geom.exterior.coords[:-1]]
            holes = [
                [(float(x + xy_padding), float(y + xy_padding), core_z0) for x, y in interior.coords[:-1]]
                for interior in geom.interiors
            ]
            design += Polygon(
                vertices=shell,
                interiors=holes,
                material=Material(n_core**2),
                depth=core_t,
                z=core_z0,
            )

    for structure in passthrough_structures:
        shifted = structure.copy().shift(xy_padding, xy_padding, core_z0)
        shifted.z = core_z0
        shifted.depth = core_t
        design += shifted

    imported_bbox = (
        float(xy_padding),
        float(xy_padding + imported_design.width),
        float(xy_padding),
        float(xy_padding + imported_design.height),
    )
    layer_z = {
        "pad_bottom": (0.0, z_padding),
        "clad_bottom": (z_padding, core_z0),
        "core": (core_z0, core_z1),
        "clad_top": (core_z1, depth - z_padding),
        "pad_top": (depth - z_padding, depth),
    }
    return design, ports, imported_bbox, layer_z


def prepare_crossing_setup(
    *,
    component_name: str,
    wl0: float,
    wl_min: float,
    wl_max: float,
    num_freqs: int,
    n_core: float,
    n_clad: float,
    polarization: str,
    points_per_wavelength: int,
    layer: tuple[int, int] | None,
    use_pdk_stack: bool,
    z_crop_auto: bool,
    margin_z_above_um: float,
    margin_z_below_um: float,
    extension_um: float,
    port_overlap_um: float,
    core_t_um: float,
    clad_below_um: float,
    clad_above_um: float,
    top_clad_shift_um: float,
    min_bottom_clad_um: float,
    monitor_candidates: int,
    pml_um: float,
    port_margin_um: float,
    source_port_offset_um: float,
    distance_source_to_monitors_um: float,
    run_after_sources_uoc: float,
    write_plots: bool,
    write_mode_plots: bool,
    out_dir: Path,
    source_direction_mode: str,
):
    component, component_label = load_crossing_component(component_name=component_name)
    polarization = str(polarization).lower()
    if polarization not in {"tm", "te"}:
        raise ValueError("--polarization must be 'tm' or 'te'.")
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_dir = out_dir / "modes"
    if write_mode_plots:
        mode_dir.mkdir(parents=True, exist_ok=True)

    layer_resolved, core_t_um_resolved, clad_below_um_resolved, clad_above_um_resolved, stack_meta = resolve_pdk_stack(
        component,
        layer=layer,
        core_t_um=core_t_um,
        clad_below_um=clad_below_um,
        clad_above_um=clad_above_um,
        use_pdk_stack=bool(use_pdk_stack),
    )
    if bool(z_crop_auto):
        clad_below_um_resolved = float(max(0.0, margin_z_below_um))
        clad_above_um_resolved = float(max(0.0, margin_z_above_um))
        stack_meta["z_crop_auto"] = True
    else:
        stack_meta["z_crop_auto"] = False
    shift = max(0.0, float(top_clad_shift_um))
    min_bottom = max(0.0, float(min_bottom_clad_um))
    if shift > 0.0:
        shift_eff = min(shift, max(0.0, clad_below_um_resolved - min_bottom))
        clad_below_um_resolved -= shift_eff
        clad_above_um_resolved += shift_eff
    else:
        shift_eff = 0.0
    print(
        "Stack resolution: "
        f"layer={layer_resolved}, "
        f"core_t={core_t_um_resolved:.3f}um, "
        f"clad_below={clad_below_um_resolved:.3f}um, "
        f"clad_above={clad_above_um_resolved:.3f}um, "
        f"used_pdk_stack={bool(stack_meta.get('used_pdk_stack', False))}, "
        f"z_crop_auto={bool(stack_meta.get('z_crop_auto', False))}, "
        f"top_shift_applied={shift_eff:.3f}um"
    )

    core_t = float(core_t_um_resolved) * µm
    clad_below = float(clad_below_um_resolved) * µm
    clad_above = float(clad_above_um_resolved) * µm
    pml_xy = max(0.0, float(pml_um)) * µm
    pml_z = max(0.0, float(pml_um)) * µm
    domain_guard_xy = 0.0
    domain_guard_z = 0.0
    margin_xy = 0.50 * µm
    extra_z_padding = 0.10 * µm
    extension_requested = float(extension_um) * µm
    port_overlap = max(0.0, float(port_overlap_um)) * µm
    port_margin = max(0.0, float(port_margin_um)) * µm

    stack_height = clad_below + core_t + clad_above
    source_port_offset = max(0.0, float(source_port_offset_um)) * µm
    dist_source_to_mon = max(0.0, float(distance_source_to_monitors_um)) * µm
    extension = max(extension_requested, margin_xy + pml_xy)
    xy_padding = extension
    z_padding = pml_z + extra_z_padding
    design, ports, imported_bbox, layer_z = build_design_with_extensions(
        component,
        layer=layer_resolved,
        n_core=n_core,
        n_clad=n_clad,
        xy_padding=xy_padding,
        z_padding=z_padding,
        extension=extension,
        port_overlap=port_overlap,
        core_t=core_t,
        clad_below=clad_below,
        clad_above=clad_above,
    )
    source_port = "o1" if "o1" in ports else sorted(ports.keys())[0]
    output_ports = [name for name in sorted(ports.keys()) if name != source_port]
    all_ports = [source_port, *output_ports]

    dx, dt = dxdt(
        wl0,
        n_max=n_core,
        dims=3,
        safety_factor=0.999,
        points_per_wavelength=points_per_wavelength,
    )
    grid = design.rasterize(resolution=dx)
    num_voxels = int(np.prod(np.asarray(grid.permittivity).shape))

    freqs = np.linspace(LIGHT_SPEED / wl_max, LIGHT_SPEED / wl_min, num_freqs, dtype=np.float32)
    wl = LIGHT_SPEED / freqs
    f0 = LIGHT_SPEED / wl0

    src = ports[source_port]
    source_direction_mode = str(source_direction_mode).lower()
    if source_direction_mode not in {"inward", "outward"}:
        raise ValueError("source_direction_mode must be one of {'inward', 'outward'}.")
    source_drive_direction = (
        src["direction"]
        if source_direction_mode == "inward"
        else outward_direction(src["direction"])
    )
    source_span = max(float(src["width"]) + 2.0 * port_margin, float(src["width"]) + 0.1 * µm)
    monitor_span = source_span
    z_center = float(src["z_center"])
    source_height = stack_height
    monitor_height = stack_height

    source_probe_port = dict(src)
    source_probe_label = "device-side"

    source_offset = source_port_offset
    fwd_offset = source_port_offset + dist_source_to_mon
    source_xy = move_along(src["center"], source_probe_port["direction"], source_offset)
    source_center = (source_xy[0], source_xy[1], z_center)
    source_plane = port_plane(
        source_probe_port,
        y_span=source_span,
        z_span=source_height,
        z_center=z_center,
        offset=source_offset,
    )
    src_plane_center = line_center(source_plane)

    fwd_plane = port_plane(
        source_probe_port,
        y_span=monitor_span,
        z_span=monitor_height,
        z_center=z_center,
        offset=fwd_offset,
    )
    out_mag_candidates = []
    n_cands = int(np.clip(monitor_candidates, 1, 3))
    out_base = source_port_offset
    out_step = 0.20 * µm
    for i in range(n_cands):
        mag = out_base + float(i) * out_step
        if not any(abs(mag - m) < 1e-12 for m in out_mag_candidates):
            out_mag_candidates.append(mag)

    out_candidates = {}
    for p in output_ports:
        out_dirn = outward_direction(ports[p]["direction"])
        out_port = dict(ports[p])
        out_port["direction"] = out_dirn
        cand_list = []
        for i, mag in enumerate(out_mag_candidates):
            plane = port_plane(
                out_port,
                y_span=monitor_span,
                z_span=monitor_height,
                z_center=z_center,
                offset=mag,
            )
            cand_list.append(
                {
                    "name": f"{p}_cand{i}",
                    "offset": mag,
                    "plane": plane,
                }
            )
        out_candidates[p] = cand_list

    max_output_distance_um = 0.0
    for p in output_ports:
        for cand in out_candidates[p]:
            c_out = line_center(cand["plane"])
            max_output_distance_um = max(
                max_output_distance_um,
                float(np.hypot(c_out[0] - src_plane_center[0], c_out[1] - src_plane_center[1])) / µm,
            )

    df = max(float(np.max(freqs) - np.min(freqs)), 1e-12)
    fwidth = max(df, 1e9)
    # Match Meep's GaussianSource convention more closely:
    # time-domain width = 1 / fwidth and the pulse peak occurs after the
    # default cutoff of 5 widths from t=0.
    pulse_sigma = 1.0 / fwidth
    pulse_t0 = 5.0 * pulse_sigma
    source_end_time = 2.0 * pulse_t0
    uoc_to_s = 1e-6 / LIGHT_SPEED
    requested_run_after_sources_uoc = max(0.0, float(run_after_sources_uoc))
    min_run_after_sources_uoc = max(45.0, 4.0 * max_output_distance_um)
    effective_run_after_sources_uoc = max(
        requested_run_after_sources_uoc,
        min_run_after_sources_uoc,
    )
    run_after_s = effective_run_after_sources_uoc * uoc_to_s
    t_total = source_end_time + run_after_s
    time = np.arange(0.0, t_total, dt)
    signal = np.asarray(
        gaussian_pulse(
            time,
            amplitude=1.0,
            center=pulse_t0,
            width=pulse_sigma,
            frequency=f0,
            phase=0.0,
        ),
        dtype=np.float32,
    )
    signal_path = out_dir / "beamz_crossing_signal.png"
    if write_plots:
        save_signal_plot(time, signal, signal_path)

    source = ModeSource(
        grid=grid,
        center=source_center,
        width=source_span,
        height=source_height,
        wavelength=wl0,
        pol=polarization,
        signal=signal,
        direction=source_drive_direction,
    )

    dft_components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    monitor_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=dft_components,
        dft_window="none",
        dft_record_every_step=True,
    )

    m_fwd = Monitor(
        start=fwd_plane[0],
        end=fwd_plane[1],
        name=f"{source_port}_fwd",
        **monitor_cfg,
    )
    output_monitors = []
    for p in output_ports:
        for cand in out_candidates[p]:
            output_monitors.append(
                Monitor(
                    start=cand["plane"][0],
                    end=cand["plane"][1],
                    name=cand["name"],
                    **monitor_cfg,
                )
            )

    monitor_planes = {
        f"{source_port}_fwd": fwd_plane,
    }
    for p in output_ports:
        for cand in out_candidates[p]:
            monitor_planes[cand["name"]] = cand["plane"]

    inner_xy_bounds = (
        (pml_xy + domain_guard_xy, pml_xy + domain_guard_xy),
        (float(design.width) - pml_xy - domain_guard_xy, float(design.height) - pml_xy - domain_guard_xy),
    )
    inner_xyz_bounds = (
        (pml_xy + domain_guard_xy, pml_xy + domain_guard_xy, pml_z + domain_guard_z),
        (
            float(design.width) - pml_xy - domain_guard_xy,
            float(design.height) - pml_xy - domain_guard_xy,
            float(design.depth) - pml_z - domain_guard_z,
        ),
    )
    placement_clearances = {
        "imported_design_xy": line_clearance_to_box(
            (
                (imported_bbox[0], imported_bbox[2]),
                (imported_bbox[1], imported_bbox[3]),
            ),
            inner_xy_bounds,
        ),
        "source_center": point_clearance_to_box(source_center, inner_xyz_bounds),
        f"{source_port}_source_plane": line_clearance_to_box(source_plane, inner_xyz_bounds),
        f"{source_port}_fwd": line_clearance_to_box(fwd_plane, inner_xyz_bounds),
    }
    for p in output_ports:
        for cand in out_candidates[p]:
            placement_clearances[cand["name"]] = line_clearance_to_box(cand["plane"], inner_xyz_bounds)
    bad_clearances = {
        name: clearance for name, clearance in placement_clearances.items() if clearance <= 1e-9
    }
    if bad_clearances:
        formatted = ", ".join(f"{name}={clearance/µm:.3f}um" for name, clearance in bad_clearances.items())
        raise RuntimeError(f"Placement overlaps or touches PML-safe region boundary: {formatted}")

    overview_path = out_dir / "beamz_crossing_overview.png"
    if write_plots:
        save_overview_plot(
            eps=np.asarray(grid.permittivity, dtype=float),
            width=design.width,
            height=design.height,
            depth=design.depth,
            imported_bbox=imported_bbox,
            source_plane=source_plane,
            monitor_planes=monitor_planes,
            layer_z=layer_z,
            pml_xy=pml_xy,
            pml_z=pml_z,
            out_path=overview_path,
        )

    if write_mode_plots:
        mode_sources = {"source_main": source}
        mode_sources[f"{source_port}_fwd"] = ModeSource(
            grid=grid,
            center=line_center(fwd_plane),
            width=monitor_span,
            height=monitor_height,
            wavelength=wl0,
            pol=polarization,
            signal=np.zeros(8, dtype=float),
            direction=positive_axis_direction(source_drive_direction),
        )
        for p in output_ports:
            out_dirn = positive_axis_direction(ports[p]["direction"])
            for cand in out_candidates[p]:
                mode_sources[cand["name"]] = ModeSource(
                    grid=grid,
                    center=line_center(cand["plane"]),
                    width=monitor_span,
                    height=monitor_height,
                    wavelength=wl0,
                    pol=polarization,
                    signal=np.zeros(8, dtype=float),
                    direction=out_dirn,
                )
        for name, msrc in mode_sources.items():
            try:
                msrc.initialize(grid.permittivity, dx, dt=dt)
                save_mode_profile_plot(
                    label=name,
                    mode_src=msrc,
                    grid_eps=np.asarray(grid.permittivity, dtype=float),
                    dx=dx,
                    out_path=mode_dir / f"{name}_mode.png",
                )
            except Exception as exc:
                print(f"Mode plot skipped for {name}: {type(exc).__name__}: {exc}")

    sim = Simulation(
        design=design,
        devices=[source, m_fwd, *output_monitors],
        boundaries=[
            PML(edges=["left", "right", "top", "bottom"], thickness=pml_xy),
            PML(edges=["front", "back"], thickness=pml_z),
        ],
        time=time,
        resolution=dx,
    )

    print(
        "Running crossing modal DFT extraction: "
        f"component={component_label}, source={source_port}, outputs={output_ports}, "
        f"pol={polarization}, freq_points={num_freqs}, steps={len(time)}, "
        f"dx={dx/µm:.4f}um, depth={design.depth/µm:.2f}um, "
        f"source_dir={source_drive_direction} ({source_direction_mode}), "
        f"offsets(src/mon)={source_offset/µm:.2f}/{fwd_offset/µm:.2f}um"
    )
    print(
        "Domain padding: "
        f"xy_padding={xy_padding/µm:.2f}um, z_padding={z_padding/µm:.2f}um, "
        f"pml=({pml_xy/µm:.2f}um xy, {pml_z/µm:.2f}um z), "
        f"guard=({domain_guard_xy/µm:.2f}um xy, {domain_guard_z/µm:.2f}um z), "
        f"margin_xy={margin_xy/µm:.2f}um, "
        f"extension=requested {extension_requested/µm:.2f}um / used {extension/µm:.2f}um"
    )
    print(
        "Modal planes: "
        f"side={source_probe_label}, y_span={source_span/µm:.2f}um, z_span={source_height/µm:.2f}um, "
        f"source_offset={source_offset/µm:.2f}um, "
        f"monitor_offset={fwd_offset/µm:.2f}um"
    )
    print(
        "PML-safe clearances: "
        + ", ".join(f"{name}={clearance/µm:.2f}um" for name, clearance in placement_clearances.items())
    )
    print(
        "Workload: "
        f"grid={grid.permittivity.shape}, voxels={num_voxels:,}, "
        f"updates~{num_voxels*len(time):.3e}"
    )
    print(
        "Run window: "
        f"requested={requested_run_after_sources_uoc:.2f}um/c, "
        f"used={effective_run_after_sources_uoc:.2f}um/c, "
        f"min_from_path={min_run_after_sources_uoc:.2f}um/c"
    )
    print(
        "Signal timing: "
        f"sigma={pulse_sigma*1e15:.2f}fs, "
        f"peak={pulse_t0*1e15:.2f}fs, "
        f"total={time[-1]*1e15:.2f}fs"
    )
    print(
        "Placement check: "
        f"source_center=({source_center[0]/µm:.2f},{source_center[1]/µm:.2f},{source_center[2]/µm:.2f})um, "
        f"source_plane_center=({src_plane_center[0]/µm:.2f},{src_plane_center[1]/µm:.2f},{src_plane_center[2]/µm:.2f})um"
    )
    for p in output_ports:
        for cand in out_candidates[p]:
            c_out = line_center(cand["plane"])
            dist = float(np.hypot(c_out[0] - src_plane_center[0], c_out[1] - src_plane_center[1]))
            print(
                f"  monitor {cand['name']}: center=({c_out[0]/µm:.2f},{c_out[1]/µm:.2f},{c_out[2]/µm:.2f})um, "
                f"offset={cand['offset']/µm:.2f}um, distance_to_source={dist/µm:.2f}um"
            )

    return {
        "component_label": component_label,
        "design": design,
        "grid": grid,
        "ports": ports,
        "freqs": freqs,
        "wl": wl,
        "time": time,
        "layer_z": layer_z,
        "source_port": source_port,
        "output_ports": output_ports,
        "all_ports": all_ports,
        "layer_resolved": layer_resolved,
        "stack_meta": stack_meta,
        "source_drive_direction": source_drive_direction,
        "source_direction_mode": source_direction_mode,
        "out_candidates": out_candidates,
        "m_fwd": m_fwd,
        "output_monitors": output_monitors,
        "sim": sim,
        "num_voxels": num_voxels,
        "requested_run_after_sources_uoc": requested_run_after_sources_uoc,
        "effective_run_after_sources_uoc": effective_run_after_sources_uoc,
        "min_run_after_sources_uoc": min_run_after_sources_uoc,
        "pulse_sigma_fs": pulse_sigma * 1e15,
        "pulse_peak_time_fs": pulse_t0 * 1e15,
    }


def run_crossing_simulation(
    *,
    sim: Simulation,
    time_points: np.ndarray,
    num_voxels: int,
    polarization: str,
    write_animation: bool,
    animation_frames: int,
    grid,
    layer_z: dict[str, tuple[float, float]],
    design: Design,
    show_progress: bool,
):
    field_hist = np.zeros((0,), dtype=float)
    field_component = "Ey" if polarization == "te" else "Ez"
    eps_grid = None
    capture_z_idx = 0
    total_steps = len(time_points)
    n_anim_frames = max(0, int(animation_frames)) if write_animation else 0
    wall_t0 = time_module.perf_counter()
    if n_anim_frames > 0:
        eps_grid = np.asarray(grid.permittivity, dtype=float)
        if eps_grid.ndim == 3:
            core_z0, core_z1 = layer_z.get("core", (0.0, design.depth))
            dz = design.depth / max(int(eps_grid.shape[0]), 1)
            capture_z_idx = int(
                np.clip(round(0.5 * (core_z0 + core_z1) / max(dz, 1e-30)), 0, eps_grid.shape[0] - 1)
            )
        chunk_size = max(1, int(np.ceil(total_steps / max(n_anim_frames, 1))))
        frame_list = []
        steps_done = 0
        print(
            "Compiled run mode: chunked slice capture "
            f"(target_frames={n_anim_frames}, chunk_size={chunk_size}, field={field_component})"
        )
        while steps_done < total_steps:
            this_chunk = min(chunk_size, total_steps - steps_done)
            sim.run_compiled(num_steps=this_chunk, progress=False)
            steps_done += this_chunk

            field_now = np.asarray(getattr(sim.fields, field_component), dtype=float)
            if field_now.ndim == 3:
                frame = np.asarray(field_now[capture_z_idx, :, :], dtype=np.float32)
            elif field_now.ndim == 2:
                frame = np.asarray(field_now, dtype=np.float32)
            else:
                frame = np.asarray(field_now, dtype=np.float32).reshape(1, 1)
            frame_list.append(frame)

            if show_progress:
                pct = 100.0 * steps_done / max(total_steps, 1)
                print(
                    f"\r● Progress: {pct:.0f}% ({steps_done}/{total_steps} steps)",
                    end="",
                    flush=True,
                )
        if show_progress:
            print()
        if frame_list:
            field_hist = np.stack(frame_list, axis=0)
    else:
        sim.run_compiled(progress=bool(show_progress))

    wall_s = max(time_module.perf_counter() - wall_t0, 1e-12)
    updates_total = float(max(num_voxels, 0) * max(total_steps, 0))
    mcups = updates_total / wall_s / 1e6
    step_rate = float(total_steps) / wall_s
    sim_time_fs = 0.0
    if total_steps > 1:
        sim_time_fs = float(time_points[-1] - time_points[0]) * 1e15
    print(
        "Simulation stats: "
        f"steps={total_steps}, voxels={num_voxels:,}, sim_time={sim_time_fs:.2f}fs, "
        f"wall={wall_s:.2f}s, step_rate={step_rate:.2f} steps/s, MCUPS={mcups:.2f}"
    )

    return {
        "field_hist": field_hist,
        "field_component": field_component,
        "eps_grid": eps_grid,
        "solver_wall_s": wall_s,
        "mcups": mcups,
        "step_rate": step_rate,
    }


def extract_crossing_results(
    *,
    setup: dict[str, object],
    wl0: float,
    polarization: str,
    n_clad: float,
    mode_search_max: int,
    reference_incident: np.ndarray | None,
    reference_reflection: np.ndarray | None,
    wave_dominance_min_db: float,
    strict_normalization_qa: bool,
):
    sim = setup["sim"]
    freqs = setup["freqs"]
    wl = setup["wl"]
    source_port = str(setup["source_port"])
    output_ports = list(setup["output_ports"])
    all_ports = list(setup["all_ports"])
    ports = dict(setup["ports"])
    source_drive_direction = str(setup["source_drive_direction"])
    out_candidates = dict(setup["out_candidates"])
    m_fwd = setup["m_fwd"]
    output_monitors = list(setup["output_monitors"])

    cond_threshold = 1e8
    max_mode_search = int(np.clip(mode_search_max, 0, 3))
    source_drive_port = f"{source_port}_in"

    def source_spec(mode_index: int) -> PortSpec:
        basis_direction = positive_axis_direction(source_drive_direction)
        return PortSpec(
            name=source_drive_port,
            monitor_name=f"{source_port}_fwd",
            direction=basis_direction,
            polarization=polarization,
            mode_index=mode_index,
            incident_wave=incoming_wave_selector(source_drive_direction),
            scattered_wave=outgoing_wave_selector(source_drive_direction),
        )

    print(f"Selecting source mode over m0..m{max_mode_search}")
    source_mode_ports = {}
    source_mode_alias = {}
    for mode_idx in range(max_mode_search + 1):
        alias = f"{source_drive_port}_m{mode_idx}"
        source_mode_alias[mode_idx] = alias
        spec = source_spec(mode_idx)
        source_mode_ports[alias] = PortSpec(
            name=alias,
            monitor_name=spec.monitor_name,
            direction=spec.direction,
            polarization=spec.polarization,
            mode_index=spec.mode_index,
            reference_monitor=spec.reference_monitor,
            incident_wave=spec.incident_wave,
            scattered_wave=spec.scattered_wave,
        )
    source_mode_result = sim.get_S_matrix_modal_dft(
        source_port=source_mode_alias[0],
        ports=source_mode_ports,
        output_ports=[source_mode_alias[0]],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-45.0,
    )
    source_mode_waves = source_mode_result["diagnostics"]["waves"]
    source_best = None
    for mode_idx in range(max_mode_search + 1):
        alias = source_mode_alias[mode_idx]
        waves = source_mode_waves.get(alias, {})
        a_plus = np.asarray(waves.get("a_plus", np.zeros(freqs.shape)), dtype=np.complex128)
        a_minus = np.asarray(waves.get("a_minus", np.zeros(freqs.shape)), dtype=np.complex128)
        inc_key = incoming_wave_selector(source_drive_direction)
        inc_sel, inc_opp = choose_wave_by_selector(a_plus, a_minus, inc_key)
        inc_dom = wave_dominance_db(
            inc_sel,
            inc_opp,
            np.ones(freqs.shape, dtype=bool),
        )
        neff = np.asarray(waves.get("mode_neff", np.full(freqs.shape, np.nan)), dtype=float)
        cond = np.asarray(waves.get("condition_number", np.full(freqs.shape, np.inf)), dtype=float)
        max_inc = float(np.max(np.abs(inc_sel))) if inc_sel.size else 0.0
        abs_floor = max(1e-18, max_inc * (10.0 ** (-45.0 / 20.0)))
        valid = np.abs(inc_sel) >= abs_floor
        qual = valid & np.isfinite(cond) & (cond < cond_threshold)
        neff_med = float(np.nanmedian(neff[np.isfinite(neff)])) if np.any(np.isfinite(neff)) else -np.inf
        cond_med = float(np.nanmedian(cond[np.isfinite(cond)])) if np.any(np.isfinite(cond)) else np.inf
        qual_frac = float(np.mean(qual)) if qual.size else 0.0
        guided_bonus = 0.4 if neff_med > (n_clad + 1e-3) else 0.0
        score = (
            3.0 * qual_frac
            + guided_bonus
            + neff_med
            - 0.05 * np.log10(max(cond_med, 1.0))
            + 0.01 * inc_dom
        )
        print(
            f"  source m{mode_idx}: score={score:.3f}, "
            f"qual={qual_frac:.2f}, neff_med={neff_med:.4f}, cond_med={cond_med:.2e}, "
            f"incident={inc_key}, dom={inc_dom:.2f}dB"
        )
        candidate = {
            "mode_index": mode_idx,
            "score": score,
            "valid_mask": valid,
            "incident_wave": inc_sel,
            "incident_opposite": inc_opp,
            "incident_wave_key": inc_key,
            "incident_dom_db": inc_dom,
        }
        if source_best is None or score > source_best["score"]:
            source_best = candidate

    source_mode_idx = int(source_best["mode_index"])
    valid_mask = np.asarray(source_best["valid_mask"], dtype=bool)
    source_incident = np.asarray(source_best["incident_wave"], dtype=np.complex128)
    source_incident_opposite = np.asarray(source_best["incident_opposite"], dtype=np.complex128)
    source_incident_key = str(source_best["incident_wave_key"])
    source_waves = source_mode_waves.get(source_mode_alias[source_mode_idx], {})
    source_plus = np.asarray(source_waves.get("a_plus", np.zeros(freqs.shape)), dtype=np.complex128)
    source_minus = np.asarray(source_waves.get("a_minus", np.zeros(freqs.shape)), dtype=np.complex128)
    source_refl_wave_key = outgoing_wave_selector(source_drive_direction)
    source_refl_selected, source_refl_opposite = choose_wave_by_selector(
        source_plus,
        source_minus,
        source_refl_wave_key,
    )
    source_refl_dom_db = wave_dominance_db(
        source_refl_selected,
        source_incident,
        valid_mask,
    )
    source_refl = safe_complex_ratio(source_refl_selected, source_incident)
    source_refl = np.where(valid_mask, source_refl, 0.0 + 0.0j)
    source_neff = np.asarray(source_waves.get("mode_neff", np.full(freqs.shape, np.nan)), dtype=float)
    source_cond = np.asarray(source_waves.get("condition_number", np.full(freqs.shape, np.inf)), dtype=float)

    s_cols = {source_port: source_refl}
    port_quality = {
        source_port: valid_mask & np.isfinite(source_cond) & (source_cond < cond_threshold)
    }
    mode_indices = {source_port: source_mode_idx}
    selected_monitors = {source_port: f"{source_port}_fwd"}
    port_diagnostics = {
        source_port: {
            "neff": source_neff,
            "cond": source_cond,
            "a_selected": source_refl_selected,
            "a_opposite": source_incident,
            "wave_key": source_refl_wave_key,
            "wave_dom_db": float(source_refl_dom_db),
        }
    }

    output_search_ports = {source_drive_port: source_spec(source_mode_idx)}
    output_search_meta = []
    for p in output_ports:
        port_direction = str(ports[p]["direction"])
        basis_direction = positive_axis_direction(port_direction)
        scat_key = outgoing_wave_selector(port_direction)
        for cand in out_candidates[p]:
            for mode_idx in range(max_mode_search + 1):
                alias = f"{p}__{cand['name']}__m{mode_idx}"
                output_search_ports[alias] = PortSpec(
                    name=alias,
                    monitor_name=cand["name"],
                    direction=basis_direction,
                    polarization=polarization,
                    mode_index=mode_idx,
                    incident_wave=incoming_wave_selector(port_direction),
                    scattered_wave=scat_key,
                )
                output_search_meta.append(
                    {
                        "alias": alias,
                        "port": p,
                        "monitor_name": cand["name"],
                        "mode_index": mode_idx,
                        "scattered_wave": scat_key,
                    }
                )
    print(
        "Selecting output monitor/mode candidates in one batched extraction: "
        f"ports={len(output_ports)}, combinations={len(output_search_meta)}"
    )
    output_search_result = sim.get_S_matrix_modal_dft(
        source_port=source_drive_port,
        ports=output_search_ports,
        output_ports=[entry["alias"] for entry in output_search_meta],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-45.0,
    )
    output_search_waves = output_search_result["diagnostics"]["waves"]
    for p in output_ports:
        print(
            f"Selecting output port {p} over {len(out_candidates[p])} monitor candidates "
            f"and m0..m{max_mode_search}"
        )
        best = None
        for entry in output_search_meta:
            if entry["port"] != p:
                continue
            waves_p = output_search_waves.get(entry["alias"], {})
            mode_idx = int(entry["mode_index"])
            monitor_name = str(entry["monitor_name"])
            a_plus_p = np.asarray(waves_p.get("a_plus", np.zeros(freqs.shape)), dtype=np.complex128)
            a_minus_p = np.asarray(waves_p.get("a_minus", np.zeros(freqs.shape)), dtype=np.complex128)
            neff_p = np.asarray(waves_p.get("mode_neff", np.full(freqs.shape, np.nan)), dtype=float)
            cond_p = np.asarray(waves_p.get("condition_number", np.full(freqs.shape, np.inf)), dtype=float)
            qual = valid_mask & np.isfinite(cond_p) & (cond_p < cond_threshold)
            wave_key = str(entry["scattered_wave"])
            a_sel, a_opp = choose_wave_by_selector(a_plus_p, a_minus_p, wave_key)
            wave_dom = wave_dominance_db(a_sel, a_opp, qual)
            s_p = safe_complex_ratio(a_sel, source_incident)
            s_p = np.where(qual, s_p, 0.0 + 0.0j)
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
            score = (
                3.0 * qual_frac
                + (0.4 if neff_med > (n_clad + 1e-3) else 0.0)
                + neff_med
                - 0.05 * np.log10(max(cond_med, 1.0))
                - 0.03 * ripple
                - 0.6 * max(mag_med - 1.2, 0.0)
            )
            print(
                f"  {p} {monitor_name} m{mode_idx}: "
                f"score={score:.3f}, qual={qual_frac:.2f}, "
                f"neff_med={neff_med:.4f}, cond_med={cond_med:.2e}, ripple={ripple:.2f}, "
                f"wave={wave_key}, dom={wave_dom:.2f}dB"
            )
            candidate = {
                "score": score,
                "monitor_name": monitor_name,
                "mode_index": mode_idx,
                "s": s_p,
                "quality": qual,
                "neff": neff_p,
                "cond": cond_p,
                "a_selected": a_sel,
                "a_opposite": a_opp,
                "wave_key": wave_key,
                "wave_dom_db": wave_dom,
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
            "a_selected": np.asarray(best["a_selected"], dtype=np.complex128),
            "a_opposite": np.asarray(best["a_opposite"], dtype=np.complex128),
            "wave_key": str(best["wave_key"]),
            "wave_dom_db": float(best["wave_dom_db"]),
        }

    s_cols_raw = {p: np.asarray(v, dtype=np.complex128).copy() for p, v in s_cols.items()}
    ref_ratio = np.ones_like(source_incident, dtype=np.complex128)
    ref_norm_applied = False
    ref_refl_subtracted = False
    if reference_incident is not None:
        ref_incident = np.asarray(reference_incident, dtype=np.complex128)
        if ref_incident.shape == source_incident.shape:
            ref_ratio = safe_complex_ratio(source_incident, ref_incident)
            ref_valid = np.abs(ref_incident) > 1e-18
            valid_mask = valid_mask & ref_valid
            for p in all_ports:
                s_cols[p] = np.asarray(s_cols[p], dtype=np.complex128) * ref_ratio
            ref_norm_applied = True
            print("Applied reference-run incident normalization to device S-parameters.")
        else:
            print(
                "Reference normalization skipped: incident shape mismatch "
                f"(device={source_incident.shape}, reference={ref_incident.shape})."
            )
    if reference_reflection is not None:
        ref_refl = np.asarray(reference_reflection, dtype=np.complex128)
        if ref_refl.shape == s_cols[source_port].shape:
            s_cols[source_port] = np.asarray(s_cols[source_port], dtype=np.complex128) - ref_refl
            ref_refl_subtracted = True
            print("Applied reference-run reflection subtraction on source-port S11.")
        else:
            print(
                "Reference reflection subtraction skipped: shape mismatch "
                f"(device={s_cols[source_port].shape}, reference={ref_refl.shape})."
            )

    print(
        "Selected source mode and output monitor/mode: "
        + ", ".join(
            f"{p}={selected_monitors[p]}/m{mode_indices[p]}/{port_diagnostics[p]['wave_key']}"
            for p in all_ports
        )
    )
    incident_dominance = wave_dominance_db(
        source_incident,
        source_incident_opposite,
        valid_mask,
    )
    port_wave_dominance_db = {
        p: wave_dominance_db(
            port_diagnostics[p]["a_selected"],
            port_diagnostics[p]["a_opposite"],
            valid_mask & port_quality[p],
        )
        for p in all_ports
    }
    print(
        "Wave-direction dominance (selected/opposite power): "
        f"incident={incident_dominance:.2f} dB ({source_incident_key}), "
        + ", ".join(f"{p}={port_wave_dominance_db[p]:.2f} dB" for p in all_ports)
    )
    qa_issues = []
    qa_warnings = []
    dom_threshold = float(wave_dominance_min_db)
    if (not np.isfinite(incident_dominance)) or (incident_dominance < dom_threshold):
        qa_issues.append(
            f"incident dominance {incident_dominance:.2f} dB < threshold {dom_threshold:.2f} dB"
        )
    signal_floor_db = -25.0
    for p in output_ports:
        d = float(port_wave_dominance_db[p])
        mask = np.asarray(valid_mask & port_quality[p], dtype=bool)
        if np.any(mask):
            mag_med = float(np.nanmedian(np.abs(np.asarray(s_cols[p], dtype=np.complex128)[mask])))
        else:
            mag_med = float(np.nanmedian(np.abs(np.asarray(s_cols[p], dtype=np.complex128))))
        mag_db = 20.0 * np.log10(max(mag_med, 1e-12))
        if (not np.isfinite(d)) or (d < dom_threshold):
            msg = (
                f"{p} dominance {d:.2f} dB < threshold {dom_threshold:.2f} dB "
                f"(median |S|={mag_db:.2f} dB)"
            )
            if mag_db > signal_floor_db:
                qa_issues.append(msg)
            else:
                qa_warnings.append(msg)
    if qa_warnings:
        print("Normalization QA warnings:\n  - " + "\n  - ".join(qa_warnings))
    if qa_issues:
        msg = "Normalization QA issues:\n  - " + "\n  - ".join(qa_issues)
        if strict_normalization_qa:
            raise RuntimeError(msg)
        print(msg)

    closure = np.zeros_like(wl, dtype=float)
    for p in all_ports:
        closure += np.abs(s_cols[p]) ** 2

    monitor_objects = {m.name: m for m in [m_fwd, *output_monitors]}
    flux_in = dft_directional_power_spectrum(
        sim,
        monitor_objects[f"{source_port}_fwd"],
        source_drive_direction,
        freqs,
    )
    flux_ref = dft_directional_power_spectrum(
        sim,
        monitor_objects[f"{source_port}_fwd"],
        outward_direction(source_drive_direction),
        freqs,
    )
    flux_out = {}
    for p in output_ports:
        mon_name = selected_monitors[p]
        flux_out[p] = dft_directional_power_spectrum(
            sim,
            monitor_objects[mon_name],
            outward_direction(ports[p]["direction"]),
            freqs,
        )
    flux_total_out = np.asarray(flux_ref, dtype=float)
    for p in output_ports:
        flux_total_out = flux_total_out + np.asarray(flux_out[p], dtype=float)
    flux_closure = safe_real_ratio(flux_total_out, flux_in)
    flux_ref_ratio = safe_real_ratio(flux_ref, flux_in)
    flux_ratio = {p: safe_real_ratio(flux_out[p], flux_in) for p in output_ports}

    wl_um = wl / µm
    idx0 = int(np.argmin(np.abs(wl - wl0)))
    print(f"Center wavelength = {wl_um[idx0]:.4f} um")
    for p in all_ports:
        val = complex(s_cols[p][idx0])
        raw_val = complex(s_cols_raw[p][idx0])
        neff_p = port_diagnostics[p]["neff"]
        cond_p = port_diagnostics[p]["cond"]
        quality = bool(port_quality[p][idx0]) if idx0 < len(port_quality[p]) else False
        raw_db = 20 * np.log10(max(abs(raw_val), 1e-12))
        corr_db = 20 * np.log10(max(abs(val), 1e-12))
        if np.isclose(raw_val, val):
            mag_part = f"|S|={abs(val):.6f}, {corr_db:.2f} dB"
        else:
            mag_part = (
                f"|S|={abs(val):.6f}, {corr_db:.2f} dB "
                f"(raw {raw_db:.2f} dB)"
            )
        print(
            f"S[{p},{source_port}] @ {wl_um[idx0]:.4f}um: "
            f"{mag_part}, neff={neff_p[idx0]:.4f}, cond={cond_p[idx0]:.2e}, "
            f"quality={quality}, monitor={selected_monitors[p]}, mode=m{mode_indices[p]}, "
            f"wave={port_diagnostics[p]['wave_key']}, wave_dom={port_wave_dominance_db[p]:.2f}dB"
        )
    print(
        f"Power closure @ {wl_um[idx0]:.4f}um: {closure[idx0]:.6f} "
        f"(source_valid={bool(valid_mask[idx0])})"
    )
    flux_out_center = ", ".join(
        f"{p}={float(flux_ratio[p][idx0]):.3f}" for p in output_ports
    )
    print(
        f"Flux closure @ {wl_um[idx0]:.4f}um: {float(flux_closure[idx0]):.3f} "
        f"(R={float(flux_ref_ratio[idx0]):.3f}, {flux_out_center})"
    )

    return {
        "source_port": source_port,
        "all_ports": all_ports,
        "output_ports": output_ports,
        "selected_monitors": selected_monitors,
        "mode_indices": mode_indices,
        "port_diagnostics": port_diagnostics,
        "port_quality": port_quality,
        "valid_mask": np.asarray(valid_mask, dtype=bool),
        "s_cols": {p: np.asarray(s_cols[p], dtype=np.complex128) for p in all_ports},
        "s_cols_raw": {p: np.asarray(s_cols_raw[p], dtype=np.complex128) for p in all_ports},
        "source_incident": np.asarray(source_incident, dtype=np.complex128),
        "source_incident_opposite": np.asarray(source_incident_opposite, dtype=np.complex128),
        "source_incident_key": source_incident_key,
        "incident_dominance": float(incident_dominance),
        "ref_ratio": np.asarray(ref_ratio, dtype=np.complex128),
        "ref_norm_applied": bool(ref_norm_applied),
        "ref_refl_subtracted": bool(ref_refl_subtracted),
        "qa_issues": list(qa_issues),
        "qa_warnings": list(qa_warnings),
        "closure": np.asarray(closure, dtype=float),
        "flux_in": np.asarray(flux_in, dtype=float),
        "flux_ref": np.asarray(flux_ref, dtype=float),
        "flux_closure": np.asarray(flux_closure, dtype=float),
        "flux_ref_ratio": np.asarray(flux_ref_ratio, dtype=float),
        "flux_out": {p: np.asarray(flux_out[p], dtype=float) for p in output_ports},
        "flux_ratio": {p: np.asarray(flux_ratio[p], dtype=float) for p in output_ports},
        "port_wave_dominance_db": {
            p: float(port_wave_dominance_db[p])
            for p in all_ports
        },
        "wave_keys": {p: str(port_diagnostics[p]["wave_key"]) for p in all_ports},
        "wl_um": np.asarray(wl_um, dtype=float),
    }


def save_crossing_outputs(
    *,
    setup: dict[str, object],
    results: dict[str, object],
    simulation_state: dict[str, object],
    out_dir: Path,
    write_plots: bool,
    write_mode_plots: bool,
    write_animation: bool,
):
    source_port = str(results["source_port"])
    all_ports = list(results["all_ports"])
    output_ports = list(results["output_ports"])
    s_cols = dict(results["s_cols"])
    s_cols_raw = dict(results["s_cols_raw"])
    valid_mask = np.asarray(results["valid_mask"], dtype=bool)
    port_quality = dict(results["port_quality"])
    selected_monitors = dict(results["selected_monitors"])
    mode_indices = dict(results["mode_indices"])
    port_diagnostics = dict(results["port_diagnostics"])
    wl_um = np.asarray(results["wl_um"], dtype=float)
    closure = np.asarray(results["closure"], dtype=float)
    flux_in = np.asarray(results["flux_in"], dtype=float)
    flux_ref = np.asarray(results["flux_ref"], dtype=float)
    flux_closure = np.asarray(results["flux_closure"], dtype=float)
    flux_ref_ratio = np.asarray(results["flux_ref_ratio"], dtype=float)
    flux_out = dict(results["flux_out"])
    flux_ratio = dict(results["flux_ratio"])
    field_hist = np.asarray(simulation_state["field_hist"], dtype=float)
    field_component = str(simulation_state["field_component"])
    eps_grid = simulation_state["eps_grid"]

    data_path = out_dir / "beamz_crossing_sparams.npz"
    np.savez(
        data_path,
        source_port=source_port,
        output_ports=np.asarray(all_ports, dtype=object),
        selected_layer=np.asarray([setup["layer_resolved"]], dtype=object),
        stack_used=np.asarray([bool(setup["stack_meta"].get("used_pdk_stack", False))], dtype=bool),
        selected_monitors=np.asarray([selected_monitors[p] for p in all_ports], dtype=object),
        mode_indices=np.asarray([mode_indices[p] for p in all_ports], dtype=int),
        wave_keys=np.asarray([port_diagnostics[p]["wave_key"] for p in all_ports], dtype=object),
        wavelengths_um=wl_um,
        valid_mask=valid_mask.astype(bool),
        closure=closure,
        incident_device=np.asarray(results["source_incident"], dtype=np.complex128),
        incident_opposite=np.asarray(results["source_incident_opposite"], dtype=np.complex128),
        incident_wave_key=np.asarray([results["source_incident_key"]], dtype=object),
        incident_dominance_db=np.asarray([results["incident_dominance"]], dtype=float),
        incident_ref_ratio=np.asarray(results["ref_ratio"], dtype=np.complex128),
        ref_norm_applied=np.asarray([results["ref_norm_applied"]], dtype=bool),
        ref_refl_subtracted=np.asarray([results["ref_refl_subtracted"]], dtype=bool),
        requested_run_after_sources_uoc=np.asarray(
            [setup["requested_run_after_sources_uoc"]],
            dtype=float,
        ),
        effective_run_after_sources_uoc=np.asarray(
            [setup["effective_run_after_sources_uoc"]],
            dtype=float,
        ),
        min_run_after_sources_uoc=np.asarray(
            [setup["min_run_after_sources_uoc"]],
            dtype=float,
        ),
        pulse_sigma_fs=np.asarray([setup["pulse_sigma_fs"]], dtype=float),
        pulse_peak_time_fs=np.asarray([setup["pulse_peak_time_fs"]], dtype=float),
        port_wave_dominance_db=np.asarray(
            [results["port_wave_dominance_db"][p] for p in all_ports],
            dtype=float,
        ),
        flux_in=flux_in,
        flux_ref=flux_ref,
        flux_closure=flux_closure,
        flux_ref_ratio=flux_ref_ratio,
        **{f"flux_{p}": np.asarray(flux_out[p], dtype=float) for p in output_ports},
        **{f"flux_ratio_{p}": np.asarray(flux_ratio[p], dtype=float) for p in output_ports},
        **{f"quality_{p}": np.asarray(port_quality[p], dtype=bool) for p in all_ports},
        **{f"s_raw_{p}_{source_port}": np.asarray(s_cols_raw[p], dtype=np.complex128) for p in all_ports},
        **{f"s_{p}_{source_port}": np.asarray(s_cols[p], dtype=np.complex128) for p in all_ports},
    )

    fig_path_limited = out_dir / "beamz_crossing_sparams_db.png"
    fig_path_full = out_dir / "beamz_crossing_sparams_db_full.png"
    fig_path_compare = out_dir / "beamz_crossing_sparams_raw_vs_corrected.png"
    closure_plot_path = out_dir / "beamz_crossing_closure_compare.png"
    anim_path = out_dir / "beamz_crossing_field_propagation.mp4"
    anim_ok = False
    if write_plots:
        color_cycle = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]
        plot_series = {}
        for p in all_ports:
            y_db = 20.0 * np.log10(np.maximum(np.abs(np.asarray(s_cols[p], dtype=np.complex128)), 1e-12))
            y_db = np.where(valid_mask & np.asarray(port_quality[p], dtype=bool), y_db, np.nan)
            plot_series[p] = y_db

        fig_limited, ax_limited = plt.subplots(1, 1, figsize=(5.6, 3.5), dpi=320)
        for i, p in enumerate(all_ports):
            ax_limited.plot(
                wl_um,
                plot_series[p],
                "o-",
                color=color_cycle[i % len(color_cycle)],
                lw=2.0,
                ms=4.5,
                label=rf"$|S_{{{p[1:]}{source_port[1:]}}}|$",
            )
        ax_limited.set_xlim(float(np.min(wl_um)), float(np.max(wl_um)))
        ax_limited.set_ylim(-55.0, 0.0)
        ax_limited.set_xlabel("Wavelength (um)")
        ax_limited.set_ylabel("Magnitude (dB)")
        ax_limited.set_title(f"Crossing S-Parameters ({setup['component_label']})")
        ax_limited.grid(which="major", alpha=0.25, lw=0.6)
        ax_limited.minorticks_on()
        ax_limited.grid(which="minor", alpha=0.12, lw=0.4)
        ax_limited.legend(loc="best", fontsize=9, frameon=False)
        fig_limited.tight_layout()
        fig_limited.savefig(fig_path_limited, dpi=320)
        plt.close(fig_limited)

        fig_full, ax_full = plt.subplots(1, 1, figsize=(5.6, 3.5), dpi=320)
        for i, p in enumerate(all_ports):
            ax_full.plot(
                wl_um,
                plot_series[p],
                "o-",
                color=color_cycle[i % len(color_cycle)],
                lw=2.0,
                ms=4.5,
                label=rf"$|S_{{{p[1:]}{source_port[1:]}}}|$",
            )
        ax_full.set_xlim(float(np.min(wl_um)), float(np.max(wl_um)))
        ax_full.set_xlabel("Wavelength (um)")
        ax_full.set_ylabel("Magnitude (dB)")
        ax_full.set_title(f"Crossing S-Parameters (Full Range, {setup['component_label']})")
        ax_full.grid(which="major", alpha=0.25, lw=0.6)
        ax_full.minorticks_on()
        ax_full.grid(which="minor", alpha=0.12, lw=0.4)
        ax_full.legend(loc="best", fontsize=9, frameon=False)
        fig_full.tight_layout()
        fig_full.savefig(fig_path_full, dpi=320)
        plt.close(fig_full)

        save_raw_vs_corrected_sparams_plot(
            wavelengths_um=wl_um,
            source_port=source_port,
            all_ports=all_ports,
            s_cols_raw=s_cols_raw,
            s_cols=s_cols,
            valid_mask=valid_mask,
            port_quality=port_quality,
            out_path=fig_path_compare,
        )
        save_closure_compare_plot(
            wavelengths_um=wl_um,
            modal_closure=closure,
            flux_closure=flux_closure,
            valid_mask=valid_mask,
            out_path=closure_plot_path,
        )
    if write_animation and eps_grid is not None:
        anim_ok = save_field_animation(
            field_hist=field_hist,
            eps=eps_grid,
            width=setup["design"].width,
            height=setup["design"].height,
            field_label=field_component,
            out_path=anim_path,
            fps=20,
        )

    print(f"Saved S-parameter data: {data_path}")
    if write_plots:
        print(f"Saved dB plot (limited -55..0 dB): {fig_path_limited}")
        print(f"Saved dB plot (full range): {fig_path_full}")
        print(f"Saved closure comparison plot: {closure_plot_path}")
        print(f"Saved overview plot: {out_dir / 'beamz_crossing_overview.png'}")
        print(f"Saved signal plot: {out_dir / 'beamz_crossing_signal.png'}")
    if write_mode_plots:
        print(f"Saved mode plots directory: {out_dir / 'modes'}")
    if write_animation:
        if anim_ok:
            print(f"Saved field animation: {anim_path}")
        else:
            print("Field animation was not saved (no recorded frames or ffmpeg unavailable).")


def run_crossing(
    *,
    component_name: str,
    wl0: float,
    wl_min: float,
    wl_max: float,
    num_freqs: int,
    n_core: float,
    n_clad: float,
    polarization: str,
    points_per_wavelength: int,
    layer: tuple[int, int] | None,
    use_pdk_stack: bool,
    z_crop_auto: bool,
    margin_z_above_um: float,
    margin_z_below_um: float,
    extension_um: float,
    port_overlap_um: float,
    core_t_um: float,
    clad_below_um: float,
    clad_above_um: float,
    top_clad_shift_um: float,
    min_bottom_clad_um: float,
    monitor_candidates: int,
    mode_search_max: int,
    pml_um: float,
    port_margin_um: float,
    source_port_offset_um: float,
    distance_source_to_monitors_um: float,
    run_after_sources_uoc: float,
    animation_frames: int,
    write_plots: bool,
    write_mode_plots: bool,
    write_animation: bool,
    show_progress: bool,
    out_dir: Path,
    wave_dominance_min_db: float = 6.0,
    strict_normalization_qa: bool = True,
    reference_incident: np.ndarray | None = None,
    reference_reflection: np.ndarray | None = None,
    source_direction_mode: str = "inward",
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_crossing_optional_artifacts(
        out_dir,
        write_plots=write_plots,
        write_mode_plots=write_mode_plots,
        write_animation=write_animation,
    )
    setup = prepare_crossing_setup(
        component_name=component_name,
        wl0=wl0,
        wl_min=wl_min,
        wl_max=wl_max,
        num_freqs=num_freqs,
        n_core=n_core,
        n_clad=n_clad,
        polarization=polarization,
        points_per_wavelength=points_per_wavelength,
        layer=layer,
        use_pdk_stack=use_pdk_stack,
        z_crop_auto=z_crop_auto,
        margin_z_above_um=margin_z_above_um,
        margin_z_below_um=margin_z_below_um,
        extension_um=extension_um,
        port_overlap_um=port_overlap_um,
        core_t_um=core_t_um,
        clad_below_um=clad_below_um,
        clad_above_um=clad_above_um,
        top_clad_shift_um=top_clad_shift_um,
        min_bottom_clad_um=min_bottom_clad_um,
        monitor_candidates=monitor_candidates,
        pml_um=pml_um,
        port_margin_um=port_margin_um,
        source_port_offset_um=source_port_offset_um,
        distance_source_to_monitors_um=distance_source_to_monitors_um,
        run_after_sources_uoc=run_after_sources_uoc,
        write_plots=write_plots,
        write_mode_plots=write_mode_plots,
        out_dir=out_dir,
        source_direction_mode=source_direction_mode,
    )
    simulation_state = run_crossing_simulation(
        sim=setup["sim"],
        time_points=setup["time"],
        num_voxels=setup["num_voxels"],
        polarization=polarization,
        write_animation=write_animation,
        animation_frames=animation_frames,
        grid=setup["grid"],
        layer_z=setup["layer_z"],
        design=setup["design"],
        show_progress=show_progress,
    )
    results = extract_crossing_results(
        setup=setup,
        wl0=wl0,
        polarization=polarization,
        n_clad=n_clad,
        mode_search_max=mode_search_max,
        reference_incident=reference_incident,
        reference_reflection=reference_reflection,
        wave_dominance_min_db=wave_dominance_min_db,
        strict_normalization_qa=strict_normalization_qa,
    )
    save_crossing_outputs(
        setup=setup,
        results=results,
        simulation_state=simulation_state,
        out_dir=out_dir,
        write_plots=write_plots,
        write_mode_plots=write_mode_plots,
        write_animation=write_animation,
    )
    return {
        "component_label": setup["component_label"],
        "source_port": results["source_port"],
        "source_drive_direction": setup["source_drive_direction"],
        "source_direction_mode": setup["source_direction_mode"],
        "all_ports": list(results["all_ports"]),
        "selected_layer": setup["layer_resolved"],
        "stack_used": bool(setup["stack_meta"].get("used_pdk_stack", False)),
        "wavelength_um": np.asarray(results["wl_um"], dtype=float),
        "s_cols": dict(results["s_cols"]),
        "s_cols_raw": dict(results["s_cols_raw"]),
        "incident_device": np.asarray(results["source_incident"], dtype=np.complex128),
        "incident_opposite": np.asarray(results["source_incident_opposite"], dtype=np.complex128),
        "incident_wave_key": results["source_incident_key"],
        "incident_dominance_db": float(results["incident_dominance"]),
        "incident_ref_ratio": np.asarray(results["ref_ratio"], dtype=np.complex128),
        "ref_norm_applied": bool(results["ref_norm_applied"]),
        "ref_refl_subtracted": bool(results["ref_refl_subtracted"]),
        "valid_mask": np.asarray(results["valid_mask"], dtype=bool),
        "qa_issues": list(results["qa_issues"]),
        "qa_warnings": list(results["qa_warnings"]),
        "port_quality": {p: np.asarray(results["port_quality"][p], dtype=bool) for p in results["all_ports"]},
        "closure": np.asarray(results["closure"], dtype=float),
        "flux_in": np.asarray(results["flux_in"], dtype=float),
        "flux_ref": np.asarray(results["flux_ref"], dtype=float),
        "flux_closure": np.asarray(results["flux_closure"], dtype=float),
        "flux_ref_ratio": np.asarray(results["flux_ref_ratio"], dtype=float),
        "flux_out": {p: np.asarray(results["flux_out"][p], dtype=float) for p in results["output_ports"]},
        "flux_ratio": {p: np.asarray(results["flux_ratio"][p], dtype=float) for p in results["output_ports"]},
        "port_wave_dominance_db": dict(results["port_wave_dominance_db"]),
        "wave_keys": dict(results["wave_keys"]),
        "selected_monitors": dict(results["selected_monitors"]),
        "mode_indices": dict(results["mode_indices"]),
    }


def evaluate_straight_calibration(
    result: dict[str, object],
    *,
    min_through_db: float,
    max_reflection_db: float,
    max_closure_error: float,
) -> tuple[bool, dict[str, float | str]]:
    source_port = str(result["source_port"])
    all_ports = [str(p) for p in result["all_ports"]]
    s_cols = result["s_cols"]
    valid_mask = np.asarray(result["valid_mask"], dtype=bool)
    port_quality = result["port_quality"]
    closure = np.asarray(result["closure"], dtype=float)
    if source_port not in all_ports:
        raise ValueError("Calibration result missing source port in all_ports.")

    def _masked_db(port_name: str) -> np.ndarray:
        s = np.asarray(s_cols[port_name], dtype=np.complex128)
        q = np.asarray(port_quality[port_name], dtype=bool)
        m = valid_mask & q
        db = 20.0 * np.log10(np.maximum(np.abs(s), 1e-12))
        return np.where(m, db, np.nan)

    refl_db = _masked_db(source_port)
    refl_peak_db = float(np.nanmax(refl_db)) if np.any(np.isfinite(refl_db)) else float("inf")

    output_ports = [p for p in all_ports if p != source_port]
    if not output_ports:
        raise ValueError("Calibration requires at least one output port.")
    through_port = max(
        output_ports,
        key=lambda p: float(
            np.nanmedian(np.abs(np.asarray(s_cols[p], dtype=np.complex128)[valid_mask]))
            if np.any(valid_mask)
            else -np.inf
        ),
    )
    through_db = _masked_db(through_port)
    through_med_db = (
        float(np.nanmedian(through_db)) if np.any(np.isfinite(through_db)) else float("-inf")
    )

    closure_valid = np.where(valid_mask, closure, np.nan)
    closure_err = (
        float(np.nanmax(np.abs(closure_valid - 1.0)))
        if np.any(np.isfinite(closure_valid))
        else float("inf")
    )

    passed = (
        through_med_db >= float(min_through_db)
        and refl_peak_db <= float(max_reflection_db)
        and closure_err <= float(max_closure_error)
    )
    summary = {
        "through_port": through_port,
        "through_median_db": through_med_db,
        "reflection_peak_db": refl_peak_db,
        "closure_max_abs_error": closure_err,
    }
    return passed, summary


def choose_best_calibration_direction(
    summaries: dict[str, tuple[bool, dict[str, float | str]]],
    *,
    min_through_db: float,
    max_reflection_db: float,
    max_closure_error: float,
) -> str:
    def _score(item: tuple[bool, dict[str, float | str]]) -> float:
        passed, summary = item
        through = float(summary.get("through_median_db", -np.inf))
        refl = float(summary.get("reflection_peak_db", np.inf))
        closure = float(summary.get("closure_max_abs_error", np.inf))
        score = through - 0.30 * refl - 8.0 * closure
        if passed:
            score += 100.0
        score -= 10.0 * max(0.0, min_through_db - through)
        score -= 2.0 * max(0.0, refl - max_reflection_db)
        score -= 25.0 * max(0.0, closure - max_closure_error)
        return score

    ranked = sorted(
        summaries.items(),
        key=lambda kv: _score(kv[1]),
        reverse=True,
    )
    return str(ranked[0][0])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality",
        type=str,
        default="fast",
        choices=sorted(QUALITY_PRESETS.keys()),
        help="Runtime preset. 'fast' is for iteration; 'high' restores the previous heavy defaults.",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="ebeam_crossing4",
        help="Preferred crossing component name from active PDK.",
    )
    parser.add_argument("--wl0-nm", type=float, default=1550.0, help="Center wavelength in nm.")
    parser.add_argument("--wl-min-nm", type=float, default=1530.0, help="Sweep min wavelength in nm.")
    parser.add_argument("--wl-max-nm", type=float, default=1570.0, help="Sweep max wavelength in nm.")
    parser.add_argument(
        "--num-freqs",
        type=int,
        default=51,
        help="Number of DFT frequency points (recommended 11..51).",
    )
    parser.add_argument(
        "--n-core",
        type=float,
        default=3.47,
        help="Core refractive index (default Si-like, matching gsim reference).",
    )
    parser.add_argument(
        "--n-clad",
        type=float,
        default=1.44,
        help="Cladding refractive index (default SiO2-like).",
    )
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
        default=20,
        help="Grid resolution in points per wavelength (gsim-like default).",
    )
    parser.add_argument(
        "--no-z-crop-auto",
        action="store_true",
        help="Disable core-centered z-crop style margins (gsim-like behavior is enabled by default).",
    )
    parser.add_argument(
        "--margin-z-above-um",
        type=float,
        default=0.5,
        help="Top z-margin above core when z-crop-auto is enabled (um).",
    )
    parser.add_argument(
        "--margin-z-below-um",
        type=float,
        default=0.5,
        help="Bottom z-margin below core when z-crop-auto is enabled (um).",
    )
    parser.add_argument(
        "--extension-um",
        type=float,
        default=1.5,
        help="Requested straight waveguide extension added beyond each imported port; enlarged automatically if needed for stable source/monitor placement.",
    )
    parser.add_argument(
        "--port-overlap-um",
        type=float,
        default=0.10,
        help="Extra inward overlap of extension waveguides into the imported cell (um).",
    )
    parser.add_argument(
        "--core-thickness-um",
        type=float,
        default=0.22,
        help="Core layer thickness in microns (3D).",
    )
    parser.add_argument(
        "--clad-below-um",
        type=float,
        default=0.5,
        help="Bottom cladding thickness in microns (3D).",
    )
    parser.add_argument(
        "--clad-above-um",
        type=float,
        default=0.5,
        help="Top cladding thickness in microns (3D).",
    )
    parser.add_argument(
        "--top-clad-shift-um",
        type=float,
        default=0.0,
        help=(
            "Transfer this much cladding thickness from bottom to top "
            "to increase top clearance without growing total depth."
        ),
    )
    parser.add_argument(
        "--min-bottom-clad-um",
        type=float,
        default=0.8,
        help="Minimum bottom cladding retained when applying --top-clad-shift-um.",
    )
    parser.add_argument(
        "--monitor-candidates",
        type=int,
        default=1,
        help="Number of output-monitor placement candidates per port (1..3).",
    )
    parser.add_argument(
        "--mode-search-max",
        type=int,
        default=0,
        help="Max mode index for automatic search (0..3).",
    )
    parser.add_argument(
        "--pml-um",
        type=float,
        default=1.0,
        help="PML thickness in microns (Meep-style crossing runs typically use 1.0 um).",
    )
    parser.add_argument(
        "--port-margin-um",
        type=float,
        default=0.5,
        help="Extra monitor/source span added on each side of port width (um).",
    )
    parser.add_argument(
        "--source-port-offset-um",
        type=float,
        default=0.1,
        help="Additional outward shift applied to the source/monitor block beyond the default stable-launch placement (um).",
    )
    parser.add_argument(
        "--distance-source-to-monitors-um",
        type=float,
        default=0.2,
        help="Requested spacing between the source plane and the adjacent modal monitors; a stability floor is applied automatically (um).",
    )
    parser.add_argument(
        "--run-after-sources-uoc",
        type=float,
        default=45.0,
        help="Requested run duration after source center in um/c units; increased automatically when the source-to-monitor path length requires a longer window.",
    )
    parser.add_argument(
        "--source-direction",
        type=str,
        default="auto",
        choices=["auto", "inward", "outward"],
        help=(
            "Source launch direction policy relative to gdsf port mapping. "
            "'auto' uses straight calibration (if enabled) to choose inward/outward."
        ),
    )
    parser.add_argument(
        "--wave-dominance-min-db",
        type=float,
        default=6.0,
        help="Minimum selected/opposite modal power dominance (dB) required per port.",
    )
    parser.add_argument(
        "--no-strict-normalization-qa",
        action="store_true",
        help="Do not fail the run when wave-dominance QA checks fail.",
    )
    parser.add_argument(
        "--quiet-run",
        action="store_true",
        help="Disable compiled-run progress output.",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=36,
        help="Number of field-slice frames to capture for MP4 when --write-animation is enabled.",
    )
    parser.add_argument(
        "--write-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save signal, overview, closure, and S-parameter PNG plots.",
    )
    parser.add_argument(
        "--write-mode-plots",
        action="store_true",
        help="Initialize debug ModeSource probes and save per-placement mode-profile PNGs.",
    )
    parser.add_argument(
        "--write-animation",
        action="store_true",
        help="Capture field slices during the run and save an MP4 animation.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="auto",
        help="GDS layer,datatype used for core extraction (example: 1,0), or 'auto'.",
    )
    parser.add_argument(
        "--no-use-pdk-stack",
        action="store_true",
        help="Disable PDK layer-stack based layer/thickness/cladding resolution.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmarks/results/compact_models"),
        help="Output directory for compact-model data and plots.",
    )
    parser.add_argument(
        "--run-calibration",
        action="store_true",
        help="Run straight-waveguide calibration before extracting crossing S-parameters.",
    )
    parser.add_argument(
        "--calibration-component",
        type=str,
        default="straight",
        help="Component used for calibration sanity check.",
    )
    parser.add_argument(
        "--cal-min-through-db",
        type=float,
        default=-1.5,
        help="Calibration pass threshold: median through must be >= this dB value.",
    )
    parser.add_argument(
        "--cal-max-reflection-db",
        type=float,
        default=-15.0,
        help="Calibration pass threshold: worst reflection must be <= this dB value.",
    )
    parser.add_argument(
        "--cal-max-closure-error",
        type=float,
        default=0.35,
        help="Calibration pass threshold: max |closure-1| over valid frequencies.",
    )
    parser.add_argument(
        "--no-calibration-reference-normalization",
        action="store_true",
        help="Disable applying calibration incident normalization to the device run.",
    )
    parser.add_argument(
        "--no-calibration-reflection-subtraction",
        action="store_true",
        help="Disable calibration-based source reflection subtraction on the device run.",
    )
    return parser


def main():
    argv = sys.argv[1:]
    args = build_argparser().parse_args(argv)
    applied_preset = apply_quality_preset(args, argv)
    if args.num_freqs < 2:
        raise ValueError("--num-freqs must be >= 2.")
    if args.wl_min_nm >= args.wl_max_nm:
        raise ValueError("--wl-min-nm must be smaller than --wl-max-nm.")
    if args.wl0_nm < args.wl_min_nm or args.wl0_nm > args.wl_max_nm:
        raise ValueError("--wl0-nm must be within [wl-min-nm, wl-max-nm].")
    if applied_preset:
        preset_desc = ", ".join(f"{k}={v}" for k, v in applied_preset.items())
        print(f"Applied quality preset '{args.quality}': {preset_desc}")

    reference_incident = None
    reference_reflection = None
    selected_source_direction_mode = (
        args.source_direction if args.source_direction in {"inward", "outward"} else "inward"
    )
    if args.source_direction == "auto" and not args.run_calibration:
        print(
            "source-direction=auto requested without calibration; defaulting to 'inward'. "
            "Use --run-calibration to audit inward/outward automatically."
        )
    if args.run_calibration:
        direction_candidates = (
            ["inward", "outward"] if args.source_direction == "auto" else [args.source_direction]
        )
        cal_runs: dict[str, dict[str, object]] = {}
        cal_summaries: dict[str, tuple[bool, dict[str, float | str]]] = {}
        for dir_mode in direction_candidates:
            cal_out = args.out_dir / "calibration" / dir_mode
            print(
                "Running straight-waveguide calibration gate: "
                f"component={args.calibration_component}, out_dir={cal_out}, "
                f"source_direction={dir_mode}"
            )
            cal_result = run_crossing(
                component_name=args.calibration_component,
                wl0=args.wl0_nm * 1e-9,
                wl_min=args.wl_min_nm * 1e-9,
                wl_max=args.wl_max_nm * 1e-9,
                num_freqs=args.num_freqs,
                n_core=args.n_core,
                n_clad=args.n_clad,
                polarization=args.polarization,
                points_per_wavelength=args.points_per_wavelength,
                layer=parse_layer(args.layer),
                use_pdk_stack=not args.no_use_pdk_stack,
                z_crop_auto=not args.no_z_crop_auto,
                margin_z_above_um=args.margin_z_above_um,
                margin_z_below_um=args.margin_z_below_um,
                extension_um=args.extension_um,
                port_overlap_um=args.port_overlap_um,
                core_t_um=args.core_thickness_um,
                clad_below_um=args.clad_below_um,
                clad_above_um=args.clad_above_um,
                top_clad_shift_um=args.top_clad_shift_um,
                min_bottom_clad_um=args.min_bottom_clad_um,
                monitor_candidates=args.monitor_candidates,
                mode_search_max=args.mode_search_max,
                pml_um=args.pml_um,
                port_margin_um=args.port_margin_um,
                source_port_offset_um=args.source_port_offset_um,
                distance_source_to_monitors_um=args.distance_source_to_monitors_um,
                run_after_sources_uoc=args.run_after_sources_uoc,
                animation_frames=0,
                write_plots=False,
                write_mode_plots=False,
                write_animation=False,
                show_progress=not args.quiet_run,
                out_dir=cal_out,
                wave_dominance_min_db=args.wave_dominance_min_db,
                # Audit run should not abort before we can compare candidates.
                strict_normalization_qa=False,
                source_direction_mode=dir_mode,
            )
            cal_ok, cal_summary = evaluate_straight_calibration(
                cal_result,
                min_through_db=args.cal_min_through_db,
                max_reflection_db=args.cal_max_reflection_db,
                max_closure_error=args.cal_max_closure_error,
            )
            cal_runs[dir_mode] = cal_result
            cal_summaries[dir_mode] = (cal_ok, cal_summary)
            print(
                f"Calibration summary ({dir_mode}): "
                f"through_port={cal_summary['through_port']}, "
                f"through_median_db={cal_summary['through_median_db']:.2f}, "
                f"reflection_peak_db={cal_summary['reflection_peak_db']:.2f}, "
                f"closure_max_abs_error={cal_summary['closure_max_abs_error']:.3f}, "
                f"pass={cal_ok}"
            )

        selected_source_direction_mode = choose_best_calibration_direction(
            cal_summaries,
            min_through_db=args.cal_min_through_db,
            max_reflection_db=args.cal_max_reflection_db,
            max_closure_error=args.cal_max_closure_error,
        )
        print(
            f"Selected source-direction mode from calibration audit: {selected_source_direction_mode}"
        )
        cal_ok, cal_summary = cal_summaries[selected_source_direction_mode]
        cal_result = cal_runs[selected_source_direction_mode]
        if not cal_ok:
            raise RuntimeError(
                "Calibration gate failed for selected direction mode "
                f"'{selected_source_direction_mode}'. "
                "Adjust source/monitor placement and normalization before device extraction."
            )
        if not args.no_calibration_reference_normalization:
            reference_incident = np.asarray(cal_result["incident_device"], dtype=np.complex128)
            if not args.no_calibration_reflection_subtraction:
                cal_source = str(cal_result["source_port"])
                reference_reflection = np.asarray(
                    cal_result["s_cols"][cal_source], dtype=np.complex128
                )
            print(
                "Using calibration-derived reference normalization for device extraction: "
                f"incident=yes, reflection_subtraction={not args.no_calibration_reflection_subtraction}"
            )

    run_crossing(
        component_name=args.component,
        wl0=args.wl0_nm * 1e-9,
        wl_min=args.wl_min_nm * 1e-9,
        wl_max=args.wl_max_nm * 1e-9,
        num_freqs=args.num_freqs,
        n_core=args.n_core,
        n_clad=args.n_clad,
        polarization=args.polarization,
        points_per_wavelength=args.points_per_wavelength,
        layer=parse_layer(args.layer),
        use_pdk_stack=not args.no_use_pdk_stack,
        z_crop_auto=not args.no_z_crop_auto,
        margin_z_above_um=args.margin_z_above_um,
        margin_z_below_um=args.margin_z_below_um,
        extension_um=args.extension_um,
        port_overlap_um=args.port_overlap_um,
        core_t_um=args.core_thickness_um,
        clad_below_um=args.clad_below_um,
        clad_above_um=args.clad_above_um,
        top_clad_shift_um=args.top_clad_shift_um,
        min_bottom_clad_um=args.min_bottom_clad_um,
        monitor_candidates=args.monitor_candidates,
        mode_search_max=args.mode_search_max,
        pml_um=args.pml_um,
        port_margin_um=args.port_margin_um,
        source_port_offset_um=args.source_port_offset_um,
        distance_source_to_monitors_um=args.distance_source_to_monitors_um,
        run_after_sources_uoc=args.run_after_sources_uoc,
        animation_frames=args.animation_frames,
        write_plots=args.write_plots,
        write_mode_plots=args.write_mode_plots,
        write_animation=args.write_animation,
        show_progress=not args.quiet_run,
        out_dir=args.out_dir,
        wave_dominance_min_db=args.wave_dominance_min_db,
        strict_normalization_qa=not args.no_strict_normalization_qa,
        reference_incident=reference_incident,
        reference_reflection=reference_reflection,
        source_direction_mode=selected_source_direction_mode,
    )


if __name__ == "__main__":
    main()
