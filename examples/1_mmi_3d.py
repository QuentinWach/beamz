from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local workspace package import when running from examples/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamz import (
    LIGHT_SPEED,
    Material,
    ModeSource,
    Monitor,
    PML,
    Rectangle,
    Simulation,
    Taper,
    Design,
    ramped_cosine,
    um,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D MMI benchmark with mode export, mid-run Ez snapshot, and normalized flux plots."
    )
    parser.add_argument("--domain-um", type=str, default="6,4,1.5")
    parser.add_argument("--resolution-nm", type=float, default=25.0)
    parser.add_argument("--courant-factor", type=float, default=0.99)
    parser.add_argument("--sim-time-fs", type=float, default=200.0)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results/mmi_3d")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-design", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def parse_triplet(spec: str) -> tuple[float, float, float]:
    vals = tuple(float(v.strip()) for v in spec.split(","))
    if len(vals) != 3:
        raise ValueError("domain-um must be 'x,y,z'")
    return vals


def build_design(
    x: float,
    y: float,
    z: float,
    n_core: float,
    n_clad: float,
    wg_w: float,
    wg_t: float,
    mmi_w: float,
    mmi_l: float,
    taper_l: float,
    out_offset: float,
) -> tuple[Design, dict[str, float]]:
    design = Design(width=x, height=y, depth=z, material=Material(n_clad**2))

    y_mid = 0.5 * y
    z_core0 = 0.5 * (z - wg_t)
    mmi_x0 = 0.5 * x - 0.5 * mmi_l
    mmi_x1 = mmi_x0 + mmi_l
    taper_l = min(taper_l, 0.6 * mmi_l)

    design += Rectangle(
        position=(0.0, y_mid - 0.5 * wg_w, z_core0),
        width=max(mmi_x0, 2 * wg_w),
        height=wg_w,
        depth=wg_t,
        material=Material(n_core**2),
    )
    design += Taper(
        position=(mmi_x0, y_mid, z_core0),
        input_width=wg_w,
        output_width=mmi_w,
        length=taper_l,
        material=Material(n_core**2),
        depth=wg_t,
    )
    design += Rectangle(
        position=(mmi_x0 + taper_l, y_mid - 0.5 * mmi_w, z_core0),
        width=max(mmi_l - taper_l, 2 * wg_w),
        height=mmi_w,
        depth=wg_t,
        material=Material(n_core**2),
    )
    design += Rectangle(
        position=(mmi_x1, y_mid + out_offset - 0.5 * wg_w, z_core0),
        width=max(x - mmi_x1, 2 * wg_w),
        height=wg_w,
        depth=wg_t,
        material=Material(n_core**2),
    )
    design += Rectangle(
        position=(mmi_x1, y_mid - out_offset - 0.5 * wg_w, z_core0),
        width=max(x - mmi_x1, 2 * wg_w),
        height=wg_w,
        depth=wg_t,
        material=Material(n_core**2),
    )

    geom = {
        "y_mid": y_mid,
        "z_core0": z_core0,
        "z_core_center": z_core0 + 0.5 * wg_t,
        "mmi_x0": mmi_x0,
        "mmi_x1": mmi_x1,
    }
    return design, geom


def create_source(
    grid,
    signal: np.ndarray,
    wl: float,
    x_src: float,
    y_mid: float,
    z_center: float,
    wg_w: float,
    z: float,
) -> ModeSource:
    return ModeSource(
        grid=grid,
        center=(x_src, y_mid, z_center),
        width=wg_w * 3.6,
        height=min(0.8 * um, 0.7 * z),
        wavelength=wl,
        pol="tm",
        signal=signal,
        direction="+x",
    )


def export_mode_fields(source: ModeSource, out_dir: Path) -> dict[str, object]:
    comps = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    payload: dict[str, np.ndarray] = {}
    meta: dict[str, object] = {
        "direction": str(source.direction),
        "pol": str(source.pol),
        "neff_real": float(np.real(source._neff)),
        "neff_imag": float(np.imag(source._neff)),
        "impedance_neff": float(
            np.real(source._impedance_neff) if source._impedance_neff is not None else np.nan
        ),
        "components": {},
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for i, comp in enumerate(comps):
        arr = getattr(source, f"_{comp}_profile", None)
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype=np.float32)
        payload[f"{comp}_profile"] = arr_np
        meta["components"][comp] = {
            "shape": list(arr_np.shape),
            "l2_norm": float(np.linalg.norm(arr_np)),
            "max_abs": float(np.max(np.abs(arr_np))) if arr_np.size else 0.0,
        }
        ax = axes[i]
        im = ax.imshow(np.abs(arr_np), origin="lower", cmap="magma", aspect="auto")
        ax.set_title(comp)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Mode Fields | neff={meta['neff_real']:.6f}")
    fig.tight_layout()
    mode_png = out_dir / "mode_fields.png"
    fig.savefig(mode_png, dpi=170)
    plt.close(fig)

    mode_npz = out_dir / "mode_fields.npz"
    np.savez(mode_npz, **payload)
    mode_json = out_dir / "mode_fields.json"
    mode_json.write_text(json.dumps(meta, indent=2))

    return {"png": str(mode_png), "npz": str(mode_npz), "json": str(mode_json), **meta}


def export_ez_snapshot(sim: Simulation, out_dir: Path, step_label: str) -> dict[str, object]:
    ez = np.asarray(sim.fields.Ez, dtype=np.float32)
    if ez.ndim == 3:
        z_idx = ez.shape[0] // 2
        plane = ez[z_idx]
    else:
        z_idx = 0
        plane = ez

    snap_png = out_dir / f"ez_snapshot_{step_label}.png"
    snap_npz = out_dir / f"ez_snapshot_{step_label}.npz"
    np.savez(
        snap_npz,
        ez_plane=plane,
        ez_shape=np.asarray(ez.shape, dtype=np.int32),
        step=np.asarray(sim.current_step, dtype=np.int32),
        time_s=np.asarray(sim.t, dtype=np.float64),
        z_index=np.asarray(z_idx, dtype=np.int32),
    )

    fig, ax = plt.subplots(figsize=(8, 5.2))
    im = ax.imshow(plane, origin="lower", cmap="RdBu", aspect="auto")
    ax.set_title(f"Ez @ step={sim.current_step} (t={sim.t * 1e15:.3f} fs), z_idx={z_idx}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Ez")
    fig.tight_layout()
    fig.savefig(snap_png, dpi=180)
    plt.close(fig)

    return {
        "png": str(snap_png),
        "npz": str(snap_npz),
        "step": int(sim.current_step),
        "time_fs": float(sim.t * 1e15),
    }


def export_flux_plots(monitor: Monitor, dt: float, out_dir: Path) -> dict[str, object]:
    power = np.asarray(monitor.power_history, dtype=np.float64)
    t_s = np.asarray(monitor.power_timestamps, dtype=np.float64)
    if t_s.size == 0 and power.size > 0:
        t_s = np.arange(power.size, dtype=np.float64) * dt
    if power.size == 0:
        return {"png": "", "csv": "", "npz": ""}

    cumulative = np.cumsum(np.maximum(power, 0.0)) * dt
    denom = cumulative[-1] if cumulative[-1] > 0 else 1.0
    cumulative_norm = cumulative / denom
    inst_norm = power / (np.max(np.abs(power)) + 1e-30)

    flux_csv = out_dir / "flux_time_series.csv"
    arr = np.column_stack((t_s, t_s * 1e15, power, inst_norm, cumulative, cumulative_norm))
    header = "time_s,time_fs,power,instant_norm,cumulative_flux,cumulative_flux_norm"
    np.savetxt(flux_csv, arr, delimiter=",", header=header, comments="")

    flux_npz = out_dir / "flux_time_series.npz"
    np.savez(
        flux_npz,
        time_s=t_s,
        time_fs=t_s * 1e15,
        power=power,
        instant_norm=inst_norm,
        cumulative_flux=cumulative,
        cumulative_flux_norm=cumulative_norm,
    )

    flux_png = out_dir / "flux_cumulative_normalized.png"
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax[0].plot(t_s * 1e15, inst_norm, lw=1.2, color="tab:blue")
    ax[0].set_ylabel("Instantaneous Flux (norm.)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(t_s * 1e15, cumulative_norm, lw=1.6, color="tab:orange")
    ax[1].set_ylabel("Cumulative Flux (norm.)")
    ax[1].set_xlabel("Time (fs)")
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Output Flux Over Time (Normalized)")
    fig.tight_layout()
    fig.savefig(flux_png, dpi=170)
    plt.close(fig)

    return {"png": str(flux_png), "csv": str(flux_csv), "npz": str(flux_npz)}


def tcups(sim: Simulation, steps: int, elapsed_s: float) -> float:
    vox = int(np.prod(sim.fields.permittivity.shape))
    return (6.0 * vox * steps) / max(elapsed_s, 1e-30) / 1e12


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_um, y_um, z_um = parse_triplet(args.domain_um)
    x, y, z = x_um * um, y_um * um, z_um * um
    wl = 1.55 * um
    n_core, n_clad = 2.04, 1.444
    wg_w, wg_t = 0.42 * um, 0.22 * um
    mmi_w, mmi_l = 1.35 * um, 2.9 * um
    taper_l, out_offset = 0.70 * um, 0.42 * um

    dx = float(args.resolution_nm) / 1e9
    dt = float(args.courant_factor) * dx / (LIGHT_SPEED * np.sqrt(3.0))
    steps = max(1, int(np.floor((float(args.sim_time_fs) * 1e-15) / dt)))
    t = dt * np.arange(steps, dtype=np.float64)
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=LIGHT_SPEED / wl,
        ramp_duration=6 * wl / LIGHT_SPEED,
        t_max=t[-1] * 0.5,
    )

    design, geom = build_design(
        x=x,
        y=y,
        z=z,
        n_core=n_core,
        n_clad=n_clad,
        wg_w=wg_w,
        wg_t=wg_t,
        mmi_w=mmi_w,
        mmi_l=mmi_l,
        taper_l=taper_l,
        out_offset=out_offset,
    )
    if args.show_design:
        design.show()

    grid = design.rasterize(resolution=dx)
    mode_source = create_source(
        grid=grid,
        signal=signal,
        wl=wl,
        x_src=0.85 * um,
        y_mid=geom["y_mid"],
        z_center=geom["z_core_center"],
        wg_w=wg_w,
        z=z,
    )
    mode_source.initialize(grid.permittivity, dx, dt=dt)
    mode_export = export_mode_fields(mode_source, out_dir)

    mon_margin_y = 0.30 * um
    mon_margin_z = 0.10 * um
    flux_monitor = Monitor(
        design=design,
        start=(x - 1.00 * um, mon_margin_y, mon_margin_z),
        plane_normal="x",
        plane_position=x - 1.00 * um,
        size=(max(y - 2 * mon_margin_y, 0.2 * um), max(z - 2 * mon_margin_z, 0.1 * um)),
        record_fields=False,
        accumulate_power=True,
        record_interval=1,
        name="mmi_flux_out",
    )

    pml_thickness = 0.8 * um
    sim = Simulation(
        design=design,
        devices=[mode_source, flux_monitor],
        boundaries=[PML(edges="all", thickness=pml_thickness)],
        time=t,
        resolution=dx,
    )

    t0 = time.perf_counter()
    sim.run_compiled(num_steps=steps, progress=bool(args.progress))
    elapsed = time.perf_counter() - t0
    sim.fields.Ez.block_until_ready()

    mid_steps = max(1, steps // 2)
    mid_source = create_source(
        grid=grid,
        signal=signal,
        wl=wl,
        x_src=0.85 * um,
        y_mid=geom["y_mid"],
        z_center=geom["z_core_center"],
        wg_w=wg_w,
        z=z,
    )
    sim_mid = Simulation(
        design=design,
        devices=[mid_source],
        boundaries=[PML(edges="all", thickness=pml_thickness)],
        time=t,
        resolution=dx,
    )
    sim_mid.run_compiled(num_steps=mid_steps, progress=False)
    snapshot = export_ez_snapshot(sim_mid, out_dir, f"mid_step_{mid_steps}")

    flux_export = export_flux_plots(flux_monitor, dt, out_dir)

    summary = {
        "domain_um": [x_um, y_um, z_um],
        "resolution_nm": float(args.resolution_nm),
        "courant_factor": float(args.courant_factor),
        "sim_time_fs": float(args.sim_time_fs),
        "resolved_steps": int(steps),
        "elapsed_s": float(elapsed),
        "s_per_step": float(elapsed / steps),
        "tcups": float(tcups(sim, steps, elapsed)),
        "grid_shape": list(sim.fields.permittivity.shape),
        "output_dir": str(out_dir),
        "mode_export": mode_export,
        "mid_snapshot": snapshot,
        "flux_export": flux_export,
    }
    summary_path = out_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("3D MMI benchmark complete")
    print(f"grid_shape={summary['grid_shape']}")
    print(f"resolved_steps={summary['resolved_steps']}")
    print(f"elapsed_s={summary['elapsed_s']:.6f}")
    print(f"s_per_step={summary['s_per_step']:.6e}")
    print(f"tcups={summary['tcups']:.6e}")
    print(f"mode_fields_png={mode_export['png']}")
    print(f"mid_ez_snapshot_png={snapshot['png']}")
    print(f"flux_cumulative_plot_png={flux_export['png']}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
