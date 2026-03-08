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
    extension: float,
    port_overlap: float,
    core_t: float,
    clad_below: float,
    clad_above: float,
) -> tuple[Design, dict, tuple[float, float, float, float], dict[str, tuple[float, float]]]:
    imported_design, ports = gdsf.load(
        component,
        layer=layer,
        n_core=n_core,
        n_clad=n_clad,
        padding=0.0,
    )
    depth = clad_below + core_t + clad_above
    core_z0 = clad_below
    core_z1 = core_z0 + core_t

    design = Design(
        width=imported_design.width + 2.0 * extension,
        height=imported_design.height + 2.0 * extension,
        depth=depth,
        material=Material(n_clad**2),
    )
    for structure in imported_design.structures[1:]:
        shifted = structure.copy().shift(extension, extension, core_z0)
        shifted.z = core_z0
        shifted.depth = core_t
        design += shifted

    ports = {
        name: {
            **p,
            "center": (
                float(p["center"][0] + extension),
                float(p["center"][1] + extension),
            ),
            "width": float(p["width"]),
            "z_center": float(core_z0 + 0.5 * core_t),
        }
        for name, p in ports.items()
    }

    # Extend each port outward and slightly inward to ensure solid overlap (no seam/gap at the interface).
    for port in ports.values():
        cx, cy = port["center"]
        width = float(port["width"])
        d_out = outward_direction(port["direction"])
        sx, sy = move_along((cx, cy), d_out, -port_overlap)
        ox, oy = move_along((cx, cy), d_out, extension)
        if port["direction"].endswith("x"):
            design += Rectangle(
                position=(min(sx, ox), cy - 0.5 * width, core_z0),
                width=abs(ox - sx),
                height=width,
                material=Material(n_core**2),
                depth=core_t,
            )
        else:
            design += Rectangle(
                position=(cx - 0.5 * width, min(sy, oy), core_z0),
                width=width,
                height=abs(oy - sy),
                material=Material(n_core**2),
                depth=core_t,
            )

    imported_bbox = (
        float(extension),
        float(extension + imported_design.width),
        float(extension),
        float(extension + imported_design.height),
    )
    layer_z = {
        "clad_bottom": (0.0, core_z0),
        "core": (core_z0, core_z1),
        "clad_top": (core_z1, depth),
    }
    return design, ports, imported_bbox, layer_z


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
    extension_um: float,
    port_overlap_um: float,
    core_t_um: float,
    clad_below_um: float,
    clad_above_um: float,
    top_clad_shift_um: float,
    min_bottom_clad_um: float,
    monitor_candidates: int,
    mode_search_max: int,
    animation_frames: int,
    show_progress: bool,
    out_dir: Path,
    wave_dominance_min_db: float = 6.0,
    strict_normalization_qa: bool = True,
    reference_incident: np.ndarray | None = None,
    reference_reflection: np.ndarray | None = None,
) -> dict[str, object]:
    component, component_label = load_crossing_component(component_name=component_name)
    polarization = str(polarization).lower()
    if polarization not in {"tm", "te"}:
        raise ValueError("--polarization must be 'tm' or 'te'.")
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_dir = out_dir / "modes"
    mode_dir.mkdir(parents=True, exist_ok=True)

    layer_resolved, core_t_um_resolved, clad_below_um_resolved, clad_above_um_resolved, stack_meta = resolve_pdk_stack(
        component,
        layer=layer,
        core_t_um=core_t_um,
        clad_below_um=clad_below_um,
        clad_above_um=clad_above_um,
        use_pdk_stack=bool(use_pdk_stack),
    )
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
        f"top_shift_applied={shift_eff:.3f}um"
    )

    core_t = float(core_t_um_resolved) * µm
    clad_below = float(clad_below_um_resolved) * µm
    clad_above = float(clad_above_um_resolved) * µm
    extension = float(extension_um) * µm
    port_overlap = max(0.0, float(port_overlap_um)) * µm
    design, ports, imported_bbox, layer_z = build_design_with_extensions(
        component,
        layer=layer_resolved,
        n_core=n_core,
        n_clad=n_clad,
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
    source_span = max(2.2 * µm, 6.0 * float(src["width"]))
    monitor_span = max(2.0 * µm, 5.5 * float(src["width"]))
    z_margin = 0.20 * µm
    max_z_span = max(0.6 * µm, design.depth - 2.0 * z_margin)
    source_height = min(max_z_span, max(2.8 * µm, 10.0 * core_t))
    monitor_height = min(max_z_span, max(2.6 * µm, 9.0 * core_t))
    z_center = float(src["z_center"])

    pml_xy = 1.0 * wl0
    pml_z = 0.8 * wl0
    max_outward_offset = max(0.9 * µm, extension - pml_xy - 0.35 * µm)
    source_mag = min(1.6 * µm, 0.55 * max_outward_offset)
    fwd_mag = max(0.70 * µm, source_mag - 0.45 * µm)
    ref_mag = min(max_outward_offset - 0.20 * µm, source_mag + 0.65 * µm)
    out_mag = min(max_outward_offset - 0.15 * µm, 0.80 * max_outward_offset)

    source_offset = -source_mag
    fwd_offset = -fwd_mag
    ref_offset = -ref_mag
    source_xy = move_along(src["center"], src["direction"], source_offset)
    source_center = (source_xy[0], source_xy[1], z_center)
    source_plane = port_plane(
        src,
        y_span=source_span,
        z_span=source_height,
        z_center=z_center,
        offset=source_offset,
    )
    src_plane_center = line_center(source_plane)

    fwd_plane = port_plane(
        src,
        y_span=monitor_span,
        z_span=monitor_height,
        z_center=z_center,
        offset=fwd_offset,
    )
    ref_plane = port_plane(
        src,
        y_span=monitor_span,
        z_span=monitor_height,
        z_center=z_center,
        offset=ref_offset,
    )

    # Build multiple output-monitor placement candidates (farther into straight sections).
    out_mag_candidates = []
    frac_lut = [0.60, 0.80, 0.95]
    n_cands = int(np.clip(monitor_candidates, 1, 3))
    for frac in frac_lut[:n_cands]:
        mag = float(np.clip(frac * out_mag, 0.70 * µm, max_outward_offset - 0.08 * µm))
        if not any(abs(mag - m) < 1e-12 for m in out_mag_candidates):
            out_mag_candidates.append(mag)

    out_candidates = {}
    min_center_separation = 1.2 * max(source_span, monitor_span)
    for p in output_ports:
        cand_list = []
        for i, mag in enumerate(out_mag_candidates):
            plane = port_plane(
                ports[p],
                y_span=monitor_span,
                z_span=monitor_height,
                z_center=z_center,
                offset=-mag,
            )
            c_out = line_center(plane)
            dist = float(np.hypot(c_out[0] - src_plane_center[0], c_out[1] - src_plane_center[1]))
            if dist < min_center_separation:
                deeper_mag = min(max_outward_offset - 0.08 * µm, mag + (min_center_separation - dist))
                plane = port_plane(
                    ports[p],
                    y_span=monitor_span,
                    z_span=monitor_height,
                    z_center=z_center,
                    offset=-deeper_mag,
                )
                mag = deeper_mag
            cand_list.append(
                {
                    "name": f"{p}_cand{i}",
                    "offset": -mag,
                    "plane": plane,
                }
            )
        out_candidates[p] = cand_list

    pulse_t0 = 12.0 / f0
    pulse_sigma = 4.0 / f0
    v_est = LIGHT_SPEED / max(n_core, 1e-9)

    def travel_time(p0, p1):
        return np.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])) / max(v_est, 1e-30)

    src_center_xy = tuple(float(v) for v in src["center"])
    src_xy = (float(source_center[0]), float(source_center[1]))
    fwd_center = line_center(fwd_plane)
    ref_center = line_center(ref_plane)
    t_fwd = pulse_t0 + travel_time(src_xy, (fwd_center[0], fwd_center[1]))
    t_ref = pulse_t0 + travel_time(src_xy, src_center_xy) + travel_time(src_center_xy, (ref_center[0], ref_center[1]))

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
                + travel_time(out_center, (line_center(cand["plane"])[0], line_center(cand["plane"])[1]))
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
    ).astype(np.float32)
    signal_path = out_dir / "beamz_crossing_signal.png"
    save_signal_plot(time, signal, signal_path)

    source = ModeSource(
        grid=grid,
        center=source_center,
        width=source_span,
        height=source_height,
        wavelength=wl0,
        pol=polarization,
        signal=signal,
        direction=src["direction"],
    )

    dft_components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    monitor_cfg = dict(
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=dft_components,
        dft_window="hann",
        dft_record_every_step=True,
    )

    m_fwd = Monitor(
        start=fwd_plane[0],
        end=fwd_plane[1],
        name=f"{source_port}_fwd",
        dft_t_start=dft_fwd_t_start,
        dft_t_end=dft_fwd_t_end,
        **monitor_cfg,
    )
    m_ref = Monitor(
        start=ref_plane[0],
        end=ref_plane[1],
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
                    start=cand["plane"][0],
                    end=cand["plane"][1],
                    name=cand["name"],
                    dft_t_start=w0,
                    dft_t_end=w1,
                    **monitor_cfg,
                )
            )

    monitor_planes = {
        f"{source_port}_fwd": fwd_plane,
        f"{source_port}_ref": ref_plane,
    }
    for p in output_ports:
        for cand in out_candidates[p]:
            monitor_planes[cand["name"]] = cand["plane"]

    overview_path = out_dir / "beamz_crossing_overview.png"
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

    # Build and save a mode-profile debug plot for every source/monitor placement.
    mode_sources = {"source_main": source}
    mode_sources[f"{source_port}_fwd"] = ModeSource(
        grid=grid,
        center=line_center(fwd_plane),
        width=monitor_span,
        height=monitor_height,
        wavelength=wl0,
        pol=polarization,
        signal=np.zeros(8, dtype=float),
        direction=src["direction"],
    )
    mode_sources[f"{source_port}_ref"] = ModeSource(
        grid=grid,
        center=line_center(ref_plane),
        width=monitor_span,
        height=monitor_height,
        wavelength=wl0,
        pol=polarization,
        signal=np.zeros(8, dtype=float),
        direction=src["direction"],
    )
    for p in output_ports:
        out_dirn = outward_direction(ports[p]["direction"])
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
        devices=[source, m_fwd, m_ref, *output_monitors],
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
        f"offsets(src/fwd/ref)={source_offset/µm:.2f}/{fwd_offset/µm:.2f}/{ref_offset/µm:.2f}um"
    )
    print(
        "Workload: "
        f"grid={grid.permittivity.shape}, voxels={num_voxels:,}, "
        f"updates~{num_voxels*len(time):.3e}"
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
    field_component = "Ey" if polarization == "te" else "Ez"
    eps_grid = np.asarray(grid.permittivity, dtype=float)
    capture_z_idx = 0
    if eps_grid.ndim == 3:
        core_z0, core_z1 = layer_z.get("core", (0.0, design.depth))
        dz = design.depth / max(int(eps_grid.shape[0]), 1)
        capture_z_idx = int(
            np.clip(round(0.5 * (core_z0 + core_z1) / max(dz, 1e-30)), 0, eps_grid.shape[0] - 1)
        )

    field_hist = np.zeros((0,), dtype=float)
    n_anim_frames = max(0, int(animation_frames))
    if n_anim_frames > 0:
        # Avoid storing full 3D volumes every interval: run in chunks and keep one XY z-slice.
        total_steps = len(time)
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

    cond_threshold = 1e8
    max_mode_search = int(np.clip(mode_search_max, 0, 3))

    source_drive_port = f"{source_port}_in"

    def source_spec(mode_index: int) -> PortSpec:
        return PortSpec(
            name=source_drive_port,
            monitor_name=f"{source_port}_fwd",
            direction=src["direction"],
            polarization=polarization,
            mode_index=mode_index,
            incident_wave="auto",
            scattered_wave="minus",
        )

    def source_reflection_spec(mode_index: int) -> PortSpec:
        return PortSpec(
            name=source_port,
            monitor_name=f"{source_port}_ref",
            direction=outward_direction(src["direction"]),
            polarization=polarization,
            mode_index=mode_index,
            scattered_wave="plus",
        )

    # Choose source mode index.
    print(f"Selecting source mode over m0..m{max_mode_search}")
    source_best = None
    for mode_idx in range(max_mode_search + 1):
        result = sim.get_S_matrix_modal_dft(
            source_port=source_drive_port,
            ports={source_drive_port: source_spec(mode_idx)},
            output_ports=[source_drive_port],
            frequencies=freqs,
            as_sax=False,
            return_diagnostics=True,
            min_incident_db=-45.0,
        )
        waves = result["diagnostics"]["waves"].get(source_drive_port, {})
        a_plus = np.asarray(waves.get("a_plus", np.zeros(freqs.shape)), dtype=np.complex128)
        a_minus = np.asarray(waves.get("a_minus", np.zeros(freqs.shape)), dtype=np.complex128)
        inc_key, inc_sel, inc_opp, inc_dom = select_dominant_wave(
            a_plus,
            a_minus,
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
            "result": result,
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

    source_refl_result = sim.get_S_matrix_modal_dft(
        source_port=source_drive_port,
        ports={
            source_drive_port: source_spec(source_mode_idx),
            source_port: source_reflection_spec(source_mode_idx),
        },
        output_ports=[source_port],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-45.0,
    )
    source_refl_waves = source_refl_result["diagnostics"]["waves"].get(source_port, {})
    source_refl_plus = np.asarray(source_refl_waves.get("a_plus", np.zeros(freqs.shape)), dtype=np.complex128)
    source_refl_minus = np.asarray(source_refl_waves.get("a_minus", np.zeros(freqs.shape)), dtype=np.complex128)
    source_refl_wave_key, source_refl_selected, source_refl_opposite, source_refl_dom_db = select_dominant_wave(
        source_refl_plus,
        source_refl_minus,
        valid_mask,
    )
    source_refl = safe_complex_ratio(source_refl_selected, source_incident)
    source_refl = np.where(valid_mask, source_refl, 0.0 + 0.0j)
    source_neff = np.asarray(source_refl_waves.get("mode_neff", np.full(freqs.shape, np.nan)), dtype=float)
    source_cond = np.asarray(source_refl_waves.get("condition_number", np.full(freqs.shape, np.inf)), dtype=float)

    s_cols = {source_port: source_refl}
    port_quality = {}
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
            "a_selected": source_refl_selected,
            "a_opposite": source_refl_opposite,
            "wave_key": source_refl_wave_key,
            "wave_dom_db": float(source_refl_dom_db),
        }
    }

    # Select best monitor placement + mode index for each output port.
    for p in output_ports:
        print(f"Selecting output port {p} over {len(out_candidates[p])} monitor candidates and m0..m{max_mode_search}")
        best = None
        for cand in out_candidates[p]:
            for mode_idx in range(max_mode_search + 1):
                result = sim.get_S_matrix_modal_dft(
                    source_port=source_drive_port,
                    ports={
                        source_drive_port: source_spec(source_mode_idx),
                        p: PortSpec(
                            name=p,
                            monitor_name=cand["name"],
                            direction=outward_direction(ports[p]["direction"]),
                            polarization=polarization,
                            mode_index=mode_idx,
                            scattered_wave="plus",
                        ),
                    },
                    output_ports=[p],
                    frequencies=freqs,
                    as_sax=False,
                    return_diagnostics=True,
                    min_incident_db=-45.0,
                )
                waves_p = result["diagnostics"]["waves"].get(p, {})
                a_plus_p = np.asarray(waves_p.get("a_plus", np.zeros(freqs.shape)), dtype=np.complex128)
                a_minus_p = np.asarray(waves_p.get("a_minus", np.zeros(freqs.shape)), dtype=np.complex128)
                neff_p = np.asarray(waves_p.get("mode_neff", np.full(freqs.shape, np.nan)), dtype=float)
                cond_p = np.asarray(waves_p.get("condition_number", np.full(freqs.shape, np.inf)), dtype=float)
                qual = (
                    valid_mask
                    & np.isfinite(cond_p)
                    & (cond_p < cond_threshold)
                )
                wave_key, a_sel, a_opp, wave_dom = select_dominant_wave(
                    a_plus_p,
                    a_minus_p,
                    qual,
                )
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
                # Prefer guided, well-conditioned, smooth spectra near expected passive range.
                score = (
                    3.0 * qual_frac
                    + (0.4 if neff_med > (n_clad + 1e-3) else 0.0)
                    + neff_med
                    - 0.05 * np.log10(max(cond_med, 1.0))
                    - 0.03 * ripple
                    - 0.6 * max(mag_med - 1.2, 0.0)
                )
                print(
                    f"  {p} {cand['name']} m{mode_idx}: "
                    f"score={score:.3f}, qual={qual_frac:.2f}, "
                    f"neff_med={neff_med:.4f}, cond_med={cond_med:.2e}, ripple={ripple:.2f}, "
                    f"wave={wave_key}, dom={wave_dom:.2f}dB"
                )
                candidate = {
                    "score": score,
                    "monitor_name": cand["name"],
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
    dom_threshold = float(wave_dominance_min_db)
    if (not np.isfinite(incident_dominance)) or (incident_dominance < dom_threshold):
        qa_issues.append(
            f"incident dominance {incident_dominance:.2f} dB < threshold {dom_threshold:.2f} dB"
        )
    for p in all_ports:
        d = float(port_wave_dominance_db[p])
        if (not np.isfinite(d)) or (d < dom_threshold):
            qa_issues.append(f"{p} dominance {d:.2f} dB < threshold {dom_threshold:.2f} dB")
    if qa_issues:
        msg = "Normalization QA issues:\n  - " + "\n  - ".join(qa_issues)
        if strict_normalization_qa:
            raise RuntimeError(msg)
        print(msg)

    closure = np.zeros_like(wl, dtype=float)
    for p in all_ports:
        closure += np.abs(s_cols[p]) ** 2

    monitor_objects = {m.name: m for m in [m_fwd, m_ref, *output_monitors]}
    flux_in = dft_directional_power_spectrum(
        sim,
        monitor_objects[f"{source_port}_fwd"],
        src["direction"],
        freqs,
    )
    flux_ref = dft_directional_power_spectrum(
        sim,
        monitor_objects[f"{source_port}_ref"],
        outward_direction(src["direction"]),
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
        neff_p = port_diagnostics[p]["neff"]
        cond_p = port_diagnostics[p]["cond"]
        quality = bool(port_quality[p][idx0]) if idx0 < len(port_quality[p]) else False
        print(
            f"S[{p},{source_port}] @ {wl_um[idx0]:.4f}um: "
            f"|S|={abs(val):.6f}, {20*np.log10(max(abs(val), 1e-12)):.2f} dB, "
            f"neff={neff_p[idx0]:.4f}, cond={cond_p[idx0]:.2e}, "
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

    data_path = out_dir / "beamz_crossing_sparams.npz"
    np.savez(
        data_path,
        source_port=source_port,
        output_ports=np.asarray(all_ports, dtype=object),
        selected_layer=np.asarray([layer_resolved], dtype=object),
        stack_used=np.asarray([bool(stack_meta.get("used_pdk_stack", False))], dtype=bool),
        selected_monitors=np.asarray([selected_monitors[p] for p in all_ports], dtype=object),
        mode_indices=np.asarray([mode_indices[p] for p in all_ports], dtype=int),
        wave_keys=np.asarray([port_diagnostics[p]["wave_key"] for p in all_ports], dtype=object),
        wavelengths_um=wl_um,
        valid_mask=valid_mask.astype(bool),
        closure=closure,
        incident_device=source_incident,
        incident_opposite=source_incident_opposite,
        incident_wave_key=np.asarray([source_incident_key], dtype=object),
        incident_dominance_db=np.asarray([incident_dominance], dtype=float),
        incident_ref_ratio=ref_ratio,
        ref_norm_applied=np.asarray([ref_norm_applied], dtype=bool),
        ref_refl_subtracted=np.asarray([ref_refl_subtracted], dtype=bool),
        port_wave_dominance_db=np.asarray(
            [port_wave_dominance_db[p] for p in all_ports],
            dtype=float,
        ),
        flux_in=flux_in,
        flux_ref=flux_ref,
        flux_closure=flux_closure,
        flux_ref_ratio=flux_ref_ratio,
        **{f"flux_{p}": np.asarray(flux_out[p], dtype=float) for p in output_ports},
        **{f"flux_ratio_{p}": np.asarray(flux_ratio[p], dtype=float) for p in output_ports},
        **{f"quality_{p}": port_quality[p].astype(bool) for p in all_ports},
        **{f"s_raw_{p}_{source_port}": s_cols_raw[p] for p in all_ports},
        **{f"s_{p}_{source_port}": s_cols[p] for p in all_ports},
    )

    color_cycle = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    plot_series = {}
    for p in all_ports:
        y_db = 20.0 * np.log10(np.maximum(np.abs(s_cols[p]), 1e-12))
        y_db = np.where(valid_mask & port_quality[p], y_db, np.nan)
        plot_series[p] = y_db

    # Fixed-axis dB plot requested for compact model review.
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
    ax_limited.set_title(f"Crossing S-Parameters ({component_label})")
    ax_limited.grid(which="major", alpha=0.25, lw=0.6)
    ax_limited.minorticks_on()
    ax_limited.grid(which="minor", alpha=0.12, lw=0.4)
    ax_limited.legend(loc="best", fontsize=9, frameon=False)
    fig_limited.tight_layout()
    fig_path_limited = out_dir / "beamz_crossing_sparams_db.png"
    fig_limited.savefig(fig_path_limited, dpi=320)
    plt.close(fig_limited)

    # Full-range dB plot without y-limit clipping.
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
    ax_full.set_title(f"Crossing S-Parameters (Full Range, {component_label})")
    ax_full.grid(which="major", alpha=0.25, lw=0.6)
    ax_full.minorticks_on()
    ax_full.grid(which="minor", alpha=0.12, lw=0.4)
    ax_full.legend(loc="best", fontsize=9, frameon=False)
    fig_full.tight_layout()
    fig_path_full = out_dir / "beamz_crossing_sparams_db_full.png"
    fig_full.savefig(fig_path_full, dpi=320)
    plt.close(fig_full)

    closure_plot_path = out_dir / "beamz_crossing_closure_compare.png"
    save_closure_compare_plot(
        wavelengths_um=wl_um,
        modal_closure=closure,
        flux_closure=flux_closure,
        valid_mask=valid_mask,
        out_path=closure_plot_path,
    )

    anim_path = out_dir / "beamz_crossing_field_propagation.mp4"
    anim_ok = save_field_animation(
        field_hist=field_hist,
        eps=eps_grid,
        width=design.width,
        height=design.height,
        field_label=field_component,
        out_path=anim_path,
        fps=20,
    )

    print(f"Saved S-parameter data: {data_path}")
    print(f"Saved dB plot (limited -55..0 dB): {fig_path_limited}")
    print(f"Saved dB plot (full range): {fig_path_full}")
    print(f"Saved closure comparison plot: {closure_plot_path}")
    print(f"Saved overview plot: {overview_path}")
    print(f"Saved signal plot: {signal_path}")
    print(f"Saved mode plots directory: {mode_dir}")
    if anim_ok:
        print(f"Saved field animation: {anim_path}")
    else:
        print("Field animation was not saved (no recorded frames or ffmpeg unavailable).")

    return {
        "component_label": component_label,
        "source_port": source_port,
        "all_ports": list(all_ports),
        "selected_layer": layer_resolved,
        "stack_used": bool(stack_meta.get("used_pdk_stack", False)),
        "wavelength_um": np.asarray(wl_um, dtype=float),
        "s_cols": {p: np.asarray(s_cols[p], dtype=np.complex128) for p in all_ports},
        "s_cols_raw": {p: np.asarray(s_cols_raw[p], dtype=np.complex128) for p in all_ports},
        "incident_device": np.asarray(source_incident, dtype=np.complex128),
        "incident_opposite": np.asarray(source_incident_opposite, dtype=np.complex128),
        "incident_wave_key": source_incident_key,
        "incident_dominance_db": float(incident_dominance),
        "incident_ref_ratio": np.asarray(ref_ratio, dtype=np.complex128),
        "ref_norm_applied": bool(ref_norm_applied),
        "ref_refl_subtracted": bool(ref_refl_subtracted),
        "valid_mask": np.asarray(valid_mask, dtype=bool),
        "qa_issues": list(qa_issues),
        "port_quality": {p: np.asarray(port_quality[p], dtype=bool) for p in all_ports},
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
        "selected_monitors": dict(selected_monitors),
        "mode_indices": dict(mode_indices),
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=2.0,
        help="Core refractive index (default Si3N4-like).",
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
        default=8,
        help="Grid resolution in points per wavelength.",
    )
    parser.add_argument(
        "--extension-um",
        type=float,
        default=4.0,
        help="Port extension length on each side in microns.",
    )
    parser.add_argument(
        "--port-overlap-um",
        type=float,
        default=0.35,
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
        default=1.2,
        help="Bottom cladding thickness in microns (3D).",
    )
    parser.add_argument(
        "--clad-above-um",
        type=float,
        default=1.2,
        help="Top cladding thickness in microns (3D).",
    )
    parser.add_argument(
        "--top-clad-shift-um",
        type=float,
        default=0.8,
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
        default=2,
        help="Number of output-monitor placement candidates per port (1..3).",
    )
    parser.add_argument(
        "--mode-search-max",
        type=int,
        default=1,
        help="Max mode index for automatic search (0..3).",
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
        help="Number of field-slice frames to capture for MP4 (0 disables animation capture).",
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
    args = build_argparser().parse_args()
    if args.num_freqs < 2:
        raise ValueError("--num-freqs must be >= 2.")
    if args.wl_min_nm >= args.wl_max_nm:
        raise ValueError("--wl-min-nm must be smaller than --wl-max-nm.")
    if args.wl0_nm < args.wl_min_nm or args.wl0_nm > args.wl_max_nm:
        raise ValueError("--wl0-nm must be within [wl-min-nm, wl-max-nm].")

    reference_incident = None
    reference_reflection = None
    if args.run_calibration:
        cal_out = args.out_dir / "calibration"
        print(
            "Running straight-waveguide calibration gate: "
            f"component={args.calibration_component}, out_dir={cal_out}"
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
            extension_um=args.extension_um,
            port_overlap_um=args.port_overlap_um,
            core_t_um=args.core_thickness_um,
            clad_below_um=args.clad_below_um,
            clad_above_um=args.clad_above_um,
            top_clad_shift_um=args.top_clad_shift_um,
            min_bottom_clad_um=args.min_bottom_clad_um,
            monitor_candidates=args.monitor_candidates,
            mode_search_max=args.mode_search_max,
            animation_frames=0,
            show_progress=not args.quiet_run,
            out_dir=cal_out,
            wave_dominance_min_db=args.wave_dominance_min_db,
            strict_normalization_qa=not args.no_strict_normalization_qa,
        )
        cal_ok, cal_summary = evaluate_straight_calibration(
            cal_result,
            min_through_db=args.cal_min_through_db,
            max_reflection_db=args.cal_max_reflection_db,
            max_closure_error=args.cal_max_closure_error,
        )
        print(
            "Calibration summary: "
            f"through_port={cal_summary['through_port']}, "
            f"through_median_db={cal_summary['through_median_db']:.2f}, "
            f"reflection_peak_db={cal_summary['reflection_peak_db']:.2f}, "
            f"closure_max_abs_error={cal_summary['closure_max_abs_error']:.3f}"
        )
        if not cal_ok:
            raise RuntimeError(
                "Calibration gate failed. "
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
        extension_um=args.extension_um,
        port_overlap_um=args.port_overlap_um,
        core_t_um=args.core_thickness_um,
        clad_below_um=args.clad_below_um,
        clad_above_um=args.clad_above_um,
        top_clad_shift_um=args.top_clad_shift_um,
        min_bottom_clad_um=args.min_bottom_clad_um,
        monitor_candidates=args.monitor_candidates,
        mode_search_max=args.mode_search_max,
        animation_frames=args.animation_frames,
        show_progress=not args.quiet_run,
        out_dir=args.out_dir,
        wave_dominance_min_db=args.wave_dominance_min_db,
        strict_normalization_qa=not args.no_strict_normalization_qa,
        reference_incident=reference_incident,
        reference_reflection=reference_reflection,
    )


if __name__ == "__main__":
    main()
