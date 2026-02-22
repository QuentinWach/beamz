"""3D benchmark comparing legacy and v0.3 compiled FDTD loops.

Defaults are tuned to push an Apple M4 (24 GB unified memory, 10 CPU cores):
- 3D grid sizing auto-selected from memory budget
- 10-thread hint for CPU backends
- enough timesteps to amortize compilation
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

# Ensure local workspace imports when launched from benchmarks/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Hint host threading before importing JAX/BeamZ.
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("JAX_NUM_GENERATED_CPU_KERNELS", "10")

import numpy as np

from beamz import (
    LIGHT_SPEED,
    Monitor,
    PML,
    Design,
    GaussianSource,
    Material,
    ModeSource,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)


@dataclass
class BenchmarkConfig:
    grid_nx: int
    grid_ny: int
    grid_nz: int
    steps: int | None
    points_per_wavelength: int
    memory_gb: float
    saturation_factor: float
    domain_um: tuple[float, float, float] | None = None
    resolution_nm: float | None = None
    courant_factor: float | None = None
    sim_time_fs: float | None = None
    scenario: str = "gaussian_box"
    source_kind: str = "auto"
    with_xy_monitor: bool = False
    monitor_record_interval: int = 1
    monitor_plane_z_um: float | None = None


def parse_modes(spec: str) -> tuple[str, ...]:
    alias = {
        "python": "python",
        "legacy_python_step": "python",
        "split": "split_jit",
        "split_jit": "split_jit",
        "split-jit": "split_jit",
        "legacy_split_jit": "split_jit",
        "compiled": "compiled",
        "compiled_v3_scan": "compiled",
    }
    if not spec or spec.strip().lower() == "all":
        return ("python", "split_jit", "compiled")

    selected: set[str] = set()
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "all":
            return ("python", "split_jit", "compiled")
        mode = alias.get(token)
        if mode is None:
            raise ValueError(
                f"Unknown mode '{raw}'. Valid modes: python, split_jit, compiled, all."
            )
        selected.add(mode)

    order = ("python", "split_jit", "compiled")
    return tuple(m for m in order if m in selected)


def parse_triplet_floats(spec: str, name: str) -> tuple[float, float, float]:
    try:
        vals = tuple(float(x.strip()) for x in spec.split(","))
    except Exception as exc:
        raise ValueError(f"Invalid {name} '{spec}'. Use 'a,b,c'.") from exc
    if len(vals) != 3:
        raise ValueError(f"Invalid {name} '{spec}'. Use exactly 3 comma-separated values.")
    return vals


def parse_triplet_ints(spec: str, name: str) -> tuple[int, int, int]:
    try:
        vals = tuple(int(x.strip()) for x in spec.split(","))
    except Exception as exc:
        raise ValueError(f"Invalid {name} '{spec}'. Use 'a,b,c'.") from exc
    if len(vals) != 3:
        raise ValueError(f"Invalid {name} '{spec}'. Use exactly 3 comma-separated values.")
    if any(v <= 0 for v in vals):
        raise ValueError(f"{name} values must be > 0.")
    return vals


def _safe_div(num: float, den: float) -> float:
    if den <= 0 or (not np.isfinite(num)) or (not np.isfinite(den)):
        return float("nan")
    return num / den


def _hlo_op_counts(hlo_text: str) -> dict[str, int]:
    """Count real HLO ops from instruction lines (ignore metadata substrings)."""
    counts = {
        "fusion": 0,
        "scatter": 0,
        "dynamic-update-slice": 0,
        "slice": 0,
        "copy": 0,
        "while": 0,
    }
    pat = re.compile(r"=\s+.*?\b([a-z][a-z0-9-]*)\(")
    for line in hlo_text.splitlines():
        code = line.split("metadata=", 1)[0]
        m = pat.search(code)
        if not m:
            continue
        op = m.group(1)
        if op in counts:
            counts[op] += 1
    return counts


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return s.strip("_") or "unnamed"


def _index_to_str(idx) -> str:
    if isinstance(idx, slice):
        start = "" if idx.start is None else str(int(idx.start))
        stop = "" if idx.stop is None else str(int(idx.stop))
        step = "" if idx.step is None else str(int(idx.step))
        return f"{start}:{stop}:{step}"
    return str(int(idx))


def _index_tuple_to_str(idx_tuple) -> str:
    if idx_tuple is None:
        return ""
    return ",".join(_index_to_str(v) for v in idx_tuple)


def _export_monitor_artifacts(
    sim: Simulation,
    out_dir: Path,
    save_plot: bool,
) -> dict[str, object]:
    monitors = [d for d in sim.devices if isinstance(d, Monitor)]
    monitor_info: list[dict[str, object]] = []
    max_records = 0

    for i, mon in enumerate(monitors):
        name = _slug(mon.name or f"monitor_{i}")
        ts = np.asarray(mon.power_timestamps, dtype=np.float64)
        power = np.asarray(mon.power_history, dtype=np.float64)
        if ts.size == 0 and power.size > 0:
            ts = np.arange(power.size, dtype=np.float64) * float(sim.dt)
        if power.size == 0:
            continue

        step_idx = np.arange(power.size, dtype=np.int32)
        csv_path = out_dir / f"{name}_power.csv"
        npz_path = out_dir / f"{name}_power.npz"

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx", "time_s", "time_fs", "power"])
            for j in range(power.size):
                t_s = float(ts[j]) if j < ts.size else float(j * sim.dt)
                writer.writerow([int(step_idx[j]), t_s, t_s * 1e15, float(power[j])])

        np.savez(
            npz_path,
            sample_idx=step_idx,
            time_s=ts,
            time_fs=ts * 1e15,
            power=power,
        )

        png_path = out_dir / f"{name}_power.png"
        wrote_plot = False
        if save_plot:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7.0, 3.5))
                ax.plot(ts * 1e15, power, lw=1.6)
                ax.set_xlabel("Time (fs)")
                ax.set_ylabel("Integrated |S| Power")
                ax.set_title(f"Monitor Power: {mon.name or name}")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(png_path, dpi=160)
                plt.close(fig)
                wrote_plot = True
            except Exception:
                wrote_plot = False

        max_records = max(max_records, int(power.size))
        monitor_info.append(
            {
                "name": mon.name or name,
                "plane_normal": getattr(mon, "plane_normal", ""),
                "plane_position": float(getattr(mon, "plane_position", 0.0)),
                "records": int(power.size),
                "power_mean": float(np.mean(power)),
                "power_max": float(np.max(power)),
                "power_min": float(np.min(power)),
                "energy_trapz": float(np.trapezoid(power, ts)) if ts.size >= 2 else 0.0,
                "csv": str(csv_path),
                "npz": str(npz_path),
                "png": str(png_path) if wrote_plot else "",
            }
        )

    return {
        "monitor_count": len(monitors),
        "monitor_records_max": max_records,
        "monitors": monitor_info,
    }


def _export_mode_source_artifacts(sim: Simulation, out_dir: Path) -> dict[str, object]:
    mode_sources = [d for d in sim.devices if isinstance(d, ModeSource)]
    entries: list[dict[str, object]] = []

    for i, src in enumerate(mode_sources):
        base = out_dir / f"mode_source_{i}"
        payload: dict[str, np.ndarray] = {}
        comp_summary: dict[str, dict[str, object]] = {}
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            prof = getattr(src, f"_{comp}_profile", None)
            idx = getattr(src, f"_{comp}_indices", None)
            if prof is None:
                continue
            arr = np.asarray(prof, dtype=np.float32)
            payload[f"{comp}_profile"] = arr
            comp_summary[comp] = {
                "shape": list(arr.shape),
                "l2_norm": float(np.linalg.norm(arr)),
                "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
                "index": _index_tuple_to_str(idx),
            }

        meta = {
            "direction": str(getattr(src, "direction", "")),
            "axis": str(getattr(src, "_axis", "")),
            "pol": str(getattr(src, "pol", "")),
            "wavelength_m": float(getattr(src, "wavelength", np.nan)),
            "neff_real": float(np.real(getattr(src, "_neff", np.nan))),
            "neff_imag": float(np.imag(getattr(src, "_neff", np.nan))),
            "impedance_neff": float(
                np.real(getattr(src, "_impedance_neff", np.nan))
                if getattr(src, "_impedance_neff", None) is not None
                else np.nan
            ),
            "dt_physical_s": float(getattr(src, "_dt_physical", 0.0)),
            "components": comp_summary,
        }

        npz_path = Path(str(base) + ".npz")
        np.savez(npz_path, **payload)
        json_path = Path(str(base) + ".json")
        json_path.write_text(json.dumps(meta, indent=2))

        entries.append(
            {
                "source_index": i,
                "direction": meta["direction"],
                "pol": meta["pol"],
                "neff_real": meta["neff_real"],
                "neff_imag": meta["neff_imag"],
                "impedance_neff": meta["impedance_neff"],
                "npz": str(npz_path),
                "json": str(json_path),
            }
        )

    return {"mode_source_count": len(mode_sources), "mode_sources": entries}


def _resolve_snapshot_step(snapshot_step: str, steps: int) -> int:
    spec = str(snapshot_step).strip().lower()
    if spec in {"mid", "middle", "half"}:
        return max(1, int(steps // 2))
    try:
        val = int(spec)
    except ValueError:
        return max(1, int(steps // 2))
    return int(max(1, min(steps, val)))


def _export_ez_snapshot_artifact(
    sim: Simulation,
    out_dir: Path,
    *,
    label: str,
) -> dict[str, object]:
    ez = np.asarray(sim.fields.Ez, dtype=np.float32)
    if ez.ndim == 3:
        z_idx = int(ez.shape[0] // 2)
        plane = ez[z_idx]
        plane_axis = "z"
        plane_index = z_idx
    elif ez.ndim == 2:
        plane = ez
        plane_axis = "2d"
        plane_index = 0
    else:
        plane = ez.reshape((1, -1))
        plane_axis = "flat"
        plane_index = 0

    npz_path = out_dir / f"ez_snapshot_{label}.npz"
    np.savez(
        npz_path,
        ez_plane=plane,
        ez_shape=np.asarray(ez.shape, dtype=np.int32),
        step=np.asarray(sim.current_step, dtype=np.int32),
        time_s=np.asarray(sim.t, dtype=np.float64),
        plane_axis=np.asarray(plane_axis),
        plane_index=np.asarray(plane_index, dtype=np.int32),
    )

    png_path = out_dir / f"ez_snapshot_{label}.png"
    wrote_plot = False
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        im = ax.imshow(plane, cmap="RdBu", origin="lower")
        ax.set_title(
            f"Ez Snapshot ({label}) step={sim.current_step} t={sim.t * 1e15:.3f} fs"
        )
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Ez")
        fig.tight_layout()
        fig.savefig(png_path, dpi=170)
        plt.close(fig)
        wrote_plot = True
    except Exception:
        wrote_plot = False

    return {
        "label": label,
        "step": int(sim.current_step),
        "time_s": float(sim.t),
        "plane_axis": plane_axis,
        "plane_index": int(plane_index),
        "npz": str(npz_path),
        "png": str(png_path) if wrote_plot else "",
    }


def export_physics_artifacts(
    sim: Simulation,
    out_dir: Path,
    *,
    scenario: str,
    source_kind: str,
    save_power_plot: bool,
    export_mode_profiles: bool,
    ez_snapshot: dict[str, object] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor_data = _export_monitor_artifacts(sim, out_dir, save_plot=save_power_plot)
    mode_data = (
        _export_mode_source_artifacts(sim, out_dir)
        if export_mode_profiles
        else {"mode_source_count": 0, "mode_sources": []}
    )

    summary = {
        "scenario": scenario,
        "source_kind": source_kind,
        **monitor_data,
        **mode_data,
    }
    if ez_snapshot is not None:
        summary["ez_snapshot"] = ez_snapshot
    (out_dir / "physics_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .lower()
        )
    except Exception:
        return "unknown"


def _append_csv(
    csv_path: Path,
    cfg: BenchmarkConfig,
    steps: int,
    scenario: str,
    source_kind: str,
    resolution_nm_used: float,
    courant_factor_used: float,
    sim_time_fs_used: float,
    domain_um_used: tuple[float, float, float],
    estimated_working_set_gb: float,
    t_py: float,
    t_split: float,
    t_compiled: float,
    tcups_py: float,
    tcups_split: float,
    tcups_compiled: float,
    py_runs_s: list[float],
    split_runs_s: list[float],
    compiled_runs_s: list[float],
    repeats: int,
    warmup_jit: bool,
    modes: tuple[str, ...],
    compiled_loop_kind: str,
    e_shell_split: bool,
    h_shell_split: bool,
    source_single_slab_dense: bool,
    hlo_stats: dict[str, int] | None = None,
    physics_artifact_dir: str | None = None,
    physics_monitor_count: int | None = None,
    physics_monitor_records_max: int | None = None,
    physics_mode_source_count: int | None = None,
    physics_ez_snapshot_step: int | None = None,
    physics_ez_snapshot_png: str | None = None,
    physics_ez_snapshot_npz: str | None = None,
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "timestamp_utc",
        "hostname",
        "platform",
        "python_version",
        "git_commit",
        "scenario",
        "source_kind",
        "grid_n",
        "grid_nx",
        "grid_ny",
        "grid_nz",
        "num_cells",
        "domain_x_um",
        "domain_y_um",
        "domain_z_um",
        "resolution_nm",
        "courant_factor",
        "sim_time_fs",
        "steps",
        "repeats",
        "warmup_jit",
        "modes",
        "compiled_loop_kind",
        "compiled_e_shell_split",
        "compiled_h_shell_split",
        "compiled_source_single_slab_dense",
        "ppw",
        "target_memory_gb",
        "saturation_factor",
        "estimated_working_set_gb",
        "legacy_python_step_s",
        "legacy_split_jit_s",
        "compiled_v3_scan_s",
        "legacy_python_step_s_per_step",
        "legacy_split_jit_s_per_step",
        "compiled_v3_scan_s_per_step",
        "legacy_python_step_tcups",
        "legacy_split_jit_tcups",
        "compiled_v3_scan_tcups",
        "compiled_vs_python_speedup_x",
        "compiled_vs_split_jit_speedup_x",
        "legacy_python_step_runs_s",
        "legacy_split_jit_runs_s",
        "compiled_v3_scan_runs_s",
        "compiled_hlo_text_len",
        "compiled_hlo_fusion_count",
        "compiled_hlo_scatter_count",
        "compiled_hlo_dynamic_update_slice_count",
        "compiled_hlo_slice_count",
        "compiled_hlo_copy_count",
        "compiled_hlo_while_count",
        "physics_artifact_dir",
        "physics_monitor_count",
        "physics_monitor_records_max",
        "physics_mode_source_count",
        "physics_ez_snapshot_step",
        "physics_ez_snapshot_png",
        "physics_ez_snapshot_npz",
    ]

    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            existing_headers = next(reader, None)
        if existing_headers != headers:
            with csv_path.open("r", newline="") as f:
                old_rows = list(csv.DictReader(f))
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for old_row in old_rows:
                    migrated = {h: old_row.get(h, "") for h in headers}
                    writer.writerow(migrated)

    repo_root = Path(__file__).resolve().parents[1]
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "git_commit": _git_commit(repo_root),
        "scenario": scenario,
        "source_kind": source_kind,
        "grid_n": cfg.grid_nx if (cfg.grid_nx == cfg.grid_ny == cfg.grid_nz) else "",
        "grid_nx": cfg.grid_nx,
        "grid_ny": cfg.grid_ny,
        "grid_nz": cfg.grid_nz,
        "num_cells": int(cfg.grid_nx * cfg.grid_ny * cfg.grid_nz),
        "domain_x_um": float(domain_um_used[0]),
        "domain_y_um": float(domain_um_used[1]),
        "domain_z_um": float(domain_um_used[2]),
        "resolution_nm": float(resolution_nm_used),
        "courant_factor": float(courant_factor_used),
        "sim_time_fs": float(sim_time_fs_used),
        "steps": steps,
        "repeats": int(repeats),
        "warmup_jit": int(bool(warmup_jit)),
        "modes": ";".join(modes),
        "compiled_loop_kind": compiled_loop_kind,
        "compiled_e_shell_split": int(bool(e_shell_split)),
        "compiled_h_shell_split": int(bool(h_shell_split)),
        "compiled_source_single_slab_dense": int(bool(source_single_slab_dense)),
        "ppw": cfg.points_per_wavelength,
        "target_memory_gb": cfg.memory_gb,
        "saturation_factor": cfg.saturation_factor,
        "estimated_working_set_gb": estimated_working_set_gb,
        "legacy_python_step_s": t_py,
        "legacy_split_jit_s": t_split,
        "compiled_v3_scan_s": t_compiled,
        "legacy_python_step_s_per_step": _safe_div(t_py, steps),
        "legacy_split_jit_s_per_step": _safe_div(t_split, steps),
        "compiled_v3_scan_s_per_step": _safe_div(t_compiled, steps),
        "legacy_python_step_tcups": tcups_py,
        "legacy_split_jit_tcups": tcups_split,
        "compiled_v3_scan_tcups": tcups_compiled,
        "compiled_vs_python_speedup_x": _safe_div(t_py, t_compiled),
        "compiled_vs_split_jit_speedup_x": _safe_div(t_split, t_compiled),
        "legacy_python_step_runs_s": ";".join(f"{x:.9f}" for x in py_runs_s),
        "legacy_split_jit_runs_s": ";".join(f"{x:.9f}" for x in split_runs_s),
        "compiled_v3_scan_runs_s": ";".join(f"{x:.9f}" for x in compiled_runs_s),
        "compiled_hlo_text_len": None,
        "compiled_hlo_fusion_count": None,
        "compiled_hlo_scatter_count": None,
        "compiled_hlo_dynamic_update_slice_count": None,
        "compiled_hlo_slice_count": None,
        "compiled_hlo_copy_count": None,
        "compiled_hlo_while_count": None,
        "physics_artifact_dir": physics_artifact_dir or "",
        "physics_monitor_count": (
            int(physics_monitor_count) if physics_monitor_count is not None else ""
        ),
        "physics_monitor_records_max": (
            int(physics_monitor_records_max)
            if physics_monitor_records_max is not None
            else ""
        ),
        "physics_mode_source_count": (
            int(physics_mode_source_count) if physics_mode_source_count is not None else ""
        ),
        "physics_ez_snapshot_step": (
            int(physics_ez_snapshot_step) if physics_ez_snapshot_step is not None else ""
        ),
        "physics_ez_snapshot_png": physics_ez_snapshot_png or "",
        "physics_ez_snapshot_npz": physics_ez_snapshot_npz or "",
    }
    if hlo_stats is not None:
        row["compiled_hlo_text_len"] = int(hlo_stats.get("text_len", 0))
        row["compiled_hlo_fusion_count"] = int(hlo_stats.get("fusion", 0))
        row["compiled_hlo_scatter_count"] = int(hlo_stats.get("scatter", 0))
        row["compiled_hlo_dynamic_update_slice_count"] = int(
            hlo_stats.get("dynamic-update-slice", 0)
        )
        row["compiled_hlo_slice_count"] = int(hlo_stats.get("slice", 0))
        row["compiled_hlo_copy_count"] = int(hlo_stats.get("copy", 0))
        row["compiled_hlo_while_count"] = int(hlo_stats.get("while", 0))

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def estimate_working_set_gb(nx: int, ny: int, nz: int, saturation_factor: float) -> float:
    """Crude working-set estimator for 3D FDTD in float32.

    Uses ~24 arrays worth of storage per voxel (fields + materials + coeffs + intermediates),
    then multiplies by a saturation factor to approximate temporary buffers and overhead.
    """
    voxels = float(nx * ny * nz)
    bytes_per_voxel = 24.0 * 4.0
    return voxels * bytes_per_voxel * saturation_factor / 1e9


def choose_grid_n(memory_gb: float, saturation_factor: float) -> int:
    candidates = [384, 352, 320, 288, 256, 224, 192]
    budget = memory_gb * 0.8
    for n in candidates:
        if estimate_working_set_gb(n, n, n, saturation_factor) <= budget:
            return n
    return 160


def make_simulation(
    cfg: BenchmarkConfig,
) -> tuple[
    Simulation,
    int,
    float,
    float,
    tuple[float, float, float],
    float,
    float,
    str,
]:
    n_bg = 1.0

    if cfg.resolution_nm is not None:
        dx = float(cfg.resolution_nm) / 1e9
    else:
        wl_ref = 1.55 * um
        dx, _ = calc_optimal_fdtd_params(
            wl_ref,
            n_bg,
            dims=3,
            safety_factor=0.95,
            points_per_wavelength=cfg.points_per_wavelength,
        )

    if cfg.courant_factor is not None:
        courant_used = float(cfg.courant_factor)
        dt = courant_used * dx / (LIGHT_SPEED * np.sqrt(3.0))
    else:
        wl_ref = 1.55 * um
        _, dt = calc_optimal_fdtd_params(
            wl_ref,
            n_bg,
            dims=3,
            safety_factor=0.95,
            points_per_wavelength=cfg.points_per_wavelength,
        )
        courant_used = float(dt * LIGHT_SPEED * np.sqrt(3.0) / dx)

    if cfg.domain_um is not None:
        domain_um = tuple(float(v) for v in cfg.domain_um)
        width, height, depth = (domain_um[0] * um, domain_um[1] * um, domain_um[2] * um)
    else:
        width = cfg.grid_nx * dx
        height = cfg.grid_ny * dx
        depth = cfg.grid_nz * dx
        domain_um = (width / um, height / um, depth / um)

    design = Design(
        width=width,
        height=height,
        depth=depth,
        material=Material(permittivity=n_bg**2),
    )

    if cfg.sim_time_fs is not None:
        sim_time_fs_used = float(cfg.sim_time_fs)
        steps = int(np.floor((sim_time_fs_used * 1e-15) / dt))
    else:
        if cfg.steps is None:
            raise ValueError("steps must be provided when sim_time_fs is not set.")
        steps = int(cfg.steps)
        sim_time_fs_used = float(steps * dt * 1e15)
    steps = max(1, steps)

    t_arr = dt * np.arange(steps, dtype=np.float64)
    wl_eff = cfg.points_per_wavelength * dx
    freq = LIGHT_SPEED / wl_eff
    signal = ramped_cosine(
        t_arr,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t_arr[-1] * 0.5,
    )

    if cfg.scenario == "fdtdx_coupler":
        # Silicon-on-silica corner/coupler-style setup inspired by FDTDx examples.
        eps_sio2 = 1.45**2
        eps_si = 3.5**2
        h_sub = min(0.5 * um, depth * 0.45)
        h_si = min(0.4 * um, max(depth - h_sub - 2 * dx, 0.1 * um))
        z_si = h_sub

        # Substrate
        design += Rectangle(
            position=(0.0, 0.0, 0.0),
            width=width,
            height=height,
            depth=h_sub,
            material=Material(permittivity=eps_sio2),
        )

        # Waveguides + coupling region
        wg_w = min(0.4 * um, 0.12 * min(width, height))
        y_mid = 0.5 * height
        x_corner = 0.56 * width
        y_corner = 0.56 * height
        coupling_w = min(1.6 * um, 0.35 * width)
        coupling_h = min(1.6 * um, 0.45 * height)

        # Input waveguide along +x
        design += Rectangle(
            position=(0.0, y_mid - 0.5 * wg_w, z_si),
            width=min(x_corner + 0.5 * coupling_w, width),
            height=wg_w,
            depth=h_si,
            material=Material(permittivity=eps_si),
        )

        # Output waveguide along -y (turning corner)
        design += Rectangle(
            position=(x_corner - 0.5 * wg_w, 0.0, z_si),
            width=wg_w,
            height=min(y_corner + 0.5 * coupling_h, height),
            depth=h_si,
            material=Material(permittivity=eps_si),
        )

        # Coupling block (acts as silicon coupling element proxy)
        design += Rectangle(
            position=(
                max(0.0, x_corner - 0.5 * coupling_w),
                max(0.0, y_corner - 0.5 * coupling_h),
                z_si,
            ),
            width=min(coupling_w, width),
            height=min(coupling_h, height),
            depth=h_si,
            material=Material(permittivity=eps_si),
        )

    source_kind = cfg.source_kind.strip().lower()
    if source_kind == "auto":
        source_kind = "mode" if cfg.scenario == "fdtdx_coupler" else "gaussian"
    if source_kind not in {"none", "gaussian", "mode"}:
        raise ValueError(
            f"Unsupported source_kind='{cfg.source_kind}'. Use auto|none|gaussian|mode."
        )

    devices = []
    if source_kind == "gaussian":
        if cfg.scenario == "fdtdx_coupler":
            gauss_center = (0.16 * width, y_mid, z_si + 0.5 * h_si)
            gauss_width = min(0.20 * um, 0.20 * min(width, height))
        else:
            gauss_center = (0.35 * width, 0.5 * height, 0.5 * depth)
            gauss_width = wl_eff / 6
        devices = [
            GaussianSource(
                position=gauss_center,
                width=gauss_width,
                signal=signal,
            )
        ]
    elif source_kind == "mode":
        grid = design.rasterize(resolution=dx)
        if cfg.scenario == "fdtdx_coupler":
            mode_center = (0.16 * width, y_mid, z_si + 0.5 * h_si)
            mode_width = min(1.2 * um, 0.35 * height)
            mode_height = min(1.2 * um, 0.8 * depth)
        else:
            mode_center = (0.20 * width, 0.5 * height, 0.5 * depth)
            mode_width = min(1.2 * um, 0.35 * height)
            mode_height = min(1.2 * um, 0.35 * depth)
        devices = [
            ModeSource(
                grid=grid,
                center=mode_center,
                width=mode_width,
                height=mode_height,
                wavelength=1.55 * um,
                pol="tm",
                signal=signal,
                direction="+x",
            )
        ]

    if bool(cfg.with_xy_monitor):
        z_um_req = float(cfg.monitor_plane_z_um) if cfg.monitor_plane_z_um is not None else (
            float(depth / um) * 0.5
        )
        z_pos = float(np.clip(z_um_req * um, 0.0, max(depth - dx, 0.0)))
        devices.append(
            Monitor(
                design=design,
                start=(0.0, 0.0, z_pos),
                plane_normal="z",
                plane_position=z_pos,
                size=(width, height),
                record_fields=False,
                accumulate_power=True,
                record_interval=max(1, int(cfg.monitor_record_interval)),
                name="xy_power_monitor",
            )
        )

    # Keep PML proportional to physical domain size to avoid degenerate cases.
    pml_thickness = max(4 * dx, 0.08 * min(width, height, depth))

    sim = Simulation(
        design=design,
        devices=devices,
        boundaries=[PML(edges="all", thickness=pml_thickness)],
        time=t_arr,
        resolution=dx,
    )
    return sim, steps, dx, dt, domain_um, courant_used, sim_time_fs_used, source_kind


def _tcups(sim: Simulation, steps: int, elapsed_s: float) -> float:
    if elapsed_s <= 0 or not np.isfinite(elapsed_s):
        return float("nan")
    voxels = int(np.prod(sim.fields.permittivity.shape))
    updates = 6 * voxels * steps
    return updates / elapsed_s / 1e12


def run_legacy_python_step(sim: Simulation, steps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(steps):
        if not sim.step():
            break
    sim.fields.Ez.block_until_ready()
    t1 = time.perf_counter()
    return t1 - t0


def run_legacy_split_jit(sim: Simulation, steps: int) -> float:
    # Reconstruct the old run_fast-style path: split JIT kernels + Python source dispatch.
    jit_step_h = sim._create_jit_step_h()
    jit_step_e = sim._create_jit_step_e()

    hx, hy, hz = jit_step_h(
        sim.fields.Ex,
        sim.fields.Ey,
        sim.fields.Ez,
        sim.fields.Hx,
        sim.fields.Hy,
        sim.fields.Hz,
    )
    ex, ey, ez = jit_step_e(sim.fields.Ex, sim.fields.Ey, sim.fields.Ez, hx, hy, hz)
    ex.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(steps):
        sim._inject_legacy_sources()

        sim.fields.Hx, sim.fields.Hy, sim.fields.Hz = jit_step_h(
            sim.fields.Ex,
            sim.fields.Ey,
            sim.fields.Ez,
            sim.fields.Hx,
            sim.fields.Hy,
            sim.fields.Hz,
        )

        sim._inject_h_sources()

        sim.fields.Ex, sim.fields.Ey, sim.fields.Ez = jit_step_e(
            sim.fields.Ex,
            sim.fields.Ey,
            sim.fields.Ez,
            sim.fields.Hx,
            sim.fields.Hy,
            sim.fields.Hz,
        )

        sim._inject_e_sources()

        sim.t += sim.dt
        sim.current_step += 1

    sim.fields.Ez.block_until_ready()
    t1 = time.perf_counter()
    return t1 - t0


def run_compiled(sim: Simulation, steps: int) -> float:
    t0 = time.perf_counter()
    sim.run_compiled(num_steps=steps, progress=False)
    sim.fields.Ez.block_until_ready()
    t1 = time.perf_counter()
    return t1 - t0


def run_repeated(
    sim_base: Simulation,
    steps: int,
    repeats: int,
    runner,
    *,
    warmup: bool = False,
) -> tuple[float, list[float]]:
    """Run benchmark mode repeatedly from identical initial state and return median time."""
    repeats = max(1, int(repeats))
    if warmup:
        # Warm caches (JAX/XLA/etc.) once so timing reflects steady-state kernel runtime.
        sim_warm = copy.deepcopy(sim_base)
        _ = runner(sim_warm, steps)

    times: list[float] = []
    for _ in range(repeats):
        sim = copy.deepcopy(sim_base)
        times.append(float(runner(sim, steps)))
    return float(np.median(np.asarray(times, dtype=np.float64))), times


def compiled_hlo_stats(sim: Simulation, steps: int) -> dict[str, int]:
    """Collect simple operation counts from compiled v0.3 HLO text."""
    import jax
    import jax.numpy as jnp
    from beamz.simulation.compiled import EngineState, MonitorState, monitor_state_size

    def _compiled_inputs():
        program = sim.compile(num_steps=steps)
        if program._compiled_scan is None:
            program._build_scan()

        if program.monitor_specs:
            max_records = max(1, monitor_state_size(program.monitor_specs, steps))
            monitor_state = MonitorState(
                powers=jnp.zeros((len(program.monitor_specs), max_records), dtype=jnp.float32),
                timestamps=jnp.zeros((len(program.monitor_specs), max_records), dtype=jnp.float32),
                counts=jnp.zeros((len(program.monitor_specs),), dtype=jnp.int32),
            )
        else:
            monitor_state = MonitorState(
                powers=jnp.zeros((0, 0), dtype=jnp.float32),
                timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
                counts=jnp.zeros((0,), dtype=jnp.int32),
            )

        engine_state = EngineState(
            ex=sim.fields.Ex,
            ey=sim.fields.Ey,
            ez=sim.fields.Ez,
            hx=sim.fields.Hx,
            hy=sim.fields.Hy,
            hz=sim.fields.Hz,
            t=jnp.asarray(sim.t, dtype=jnp.float32),
            current_step=jnp.asarray(sim.current_step, dtype=jnp.int32),
        )
        coeffs = program._update_coefficients()
        return program, engine_state, monitor_state, coeffs

    program, engine_state, monitor_state, coeffs = _compiled_inputs()
    hlo_text = (
        program._compiled_scan
        .lower(engine_state, monitor_state, coeffs)
        .compile()
        .as_text()
        .lower()
    )
    return {"text_len": len(hlo_text), **_hlo_op_counts(hlo_text)}


def dump_compiled_ir_artifacts(
    sim: Simulation,
    steps: int,
    out_dir: Path,
    *,
    include_dot: bool = True,
    include_optimized_hlo: bool = True,
) -> dict[str, int]:
    """Write JAXPR/HLO artifacts for deep debugging and return optimized-HLO stats."""
    import jax
    import jax.numpy as jnp
    from beamz.simulation.compiled import EngineState, MonitorState, monitor_state_size

    out_dir.mkdir(parents=True, exist_ok=True)

    program = sim.compile(num_steps=steps)
    if program._compiled_scan is None:
        program._build_scan()

    if program.monitor_specs:
        max_records = max(1, monitor_state_size(program.monitor_specs, steps))
        monitor_state = MonitorState(
            powers=jnp.zeros((len(program.monitor_specs), max_records), dtype=jnp.float32),
            timestamps=jnp.zeros((len(program.monitor_specs), max_records), dtype=jnp.float32),
            counts=jnp.zeros((len(program.monitor_specs),), dtype=jnp.int32),
        )
    else:
        monitor_state = MonitorState(
            powers=jnp.zeros((0, 0), dtype=jnp.float32),
            timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
            counts=jnp.zeros((0,), dtype=jnp.int32),
        )

    engine_state = EngineState(
        ex=sim.fields.Ex,
        ey=sim.fields.Ey,
        ez=sim.fields.Ez,
        hx=sim.fields.Hx,
        hy=sim.fields.Hy,
        hz=sim.fields.Hz,
        t=jnp.asarray(sim.t, dtype=jnp.float32),
        current_step=jnp.asarray(sim.current_step, dtype=jnp.int32),
    )
    coeffs = program._update_coefficients()
    lowered = program._compiled_scan.lower(engine_state, monitor_state, coeffs)
    hlo_comp = lowered.compiler_ir(dialect="hlo")

    # Graph-level view before XLA optimization.
    jaxpr = jax.make_jaxpr(program._compiled_scan)(engine_state, monitor_state, coeffs)
    (out_dir / "compiled_jaxpr.txt").write_text(str(jaxpr))

    # Pre-optimization HLO and optional graph.
    hlo_unopt = hlo_comp.as_hlo_text()
    (out_dir / "compiled_hlo_unoptimized.txt").write_text(hlo_unopt)
    if include_dot:
        (out_dir / "compiled_hlo_unoptimized.dot").write_text(hlo_comp.as_hlo_dot_graph())

    opt_text = ""
    if include_optimized_hlo:
        opt_text = lowered.compile().as_text()
        (out_dir / "compiled_hlo_optimized.txt").write_text(opt_text)

    stats_text = (opt_text or hlo_unopt).lower()
    stats = {"text_len": len(stats_text), **_hlo_op_counts(stats_text)}
    (out_dir / "compiled_hlo_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def hlo_diagnostics(hlo_stats: dict[str, int]) -> list[str]:
    """Return actionable diagnostics from simple HLO op counts."""
    notes: list[str] = []
    if hlo_stats.get("scatter", 0) > 0:
        notes.append(
            "Scatter ops detected: remove `.at[idx].add/set` in hot path when possible."
        )
    if hlo_stats.get("dynamic-update-slice", 0) > 0:
        notes.append(
            "dynamic-update-slice present: source/monitor updates likely creating extra memory passes."
        )
    if hlo_stats.get("slice", 0) > 120:
        notes.append(
            "High slice count: likely many shifted views/pads; target fewer stencil passes."
        )
    if hlo_stats.get("copy", 0) > 20:
        notes.append(
            "High copy count: investigate layout conversions and temporary arrays."
        )
    if hlo_stats.get("while", 0) > 2:
        notes.append(
            "Multiple while loops in HLO: nested loop primitives may be adding overhead."
        )
    if not notes:
        notes.append("No major red flags from coarse op counts; profile runtime kernels next.")
    return notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--grid-n", type=int, default=0)
    parser.add_argument(
        "--grid-shape",
        type=str,
        default=None,
        help="Non-cubic grid shape as 'nx,ny,nz'. Overrides --grid-n.",
    )
    parser.add_argument(
        "--domain-um",
        type=str,
        default=None,
        help="Physical domain in micrometers as 'x_um,y_um,z_um'.",
    )
    parser.add_argument(
        "--resolution-nm",
        type=float,
        default=None,
        help="Explicit Yee-grid resolution in nanometers.",
    )
    parser.add_argument(
        "--sim-time-fs",
        type=float,
        default=None,
        help="Physical simulation duration in femtoseconds (overrides --steps).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs per mode; report median time.",
    )
    parser.add_argument("--memory-gb", type=float, default=24.0)
    parser.add_argument("--ppw", type=int, default=10)
    parser.add_argument(
        "--saturation-factor",
        type=float,
        default=4.0,
        help="Higher = more conservative memory estimate.",
    )
    parser.add_argument(
        "--courant-factor",
        type=float,
        default=None,
        help="Explicit Courant factor for dt = courant * dx / (c*sqrt(3)).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="gaussian_box",
        choices=("gaussian_box", "fdtdx_coupler"),
        help="Benchmark scenario: simple Gaussian box or silicon coupler-style setup.",
    )
    parser.add_argument(
        "--source-kind",
        type=str,
        default="auto",
        choices=("auto", "none", "gaussian", "mode"),
        help="Source model override. 'auto' uses scenario default.",
    )
    parser.add_argument(
        "--source-sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run source isolation matrix across none, gaussian, mode on same scenario.",
    )
    parser.add_argument(
        "--with-xy-monitor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach a compiled XY-plane power monitor for physics sanity outputs.",
    )
    parser.add_argument(
        "--monitor-record-interval",
        type=int,
        default=1,
        help="Record every N steps for compiled monitor accumulation.",
    )
    parser.add_argument(
        "--monitor-plane-z-um",
        type=float,
        default=None,
        help="XY monitor z position in um (default: mid-plane).",
    )
    parser.add_argument(
        "--physics-output-dir",
        type=str,
        default=None,
        help="Directory for physics artifacts (monitor power traces/plots and mode profiles).",
    )
    parser.add_argument(
        "--physics-save-power-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write PNG power traces for monitor outputs when physics output is enabled.",
    )
    parser.add_argument(
        "--physics-export-mode-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export ModeSource modal profiles and metadata when physics output is enabled.",
    )
    parser.add_argument(
        "--physics-export-ez-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export an Ez field snapshot image/npz for physics validation.",
    )
    parser.add_argument(
        "--physics-snapshot-step",
        type=str,
        default="mid",
        help="Snapshot step for Ez export (integer, or 'mid').",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="benchmarks/results/compiled_3d_results.csv",
        help="CSV file to append benchmark results to.",
    )
    parser.add_argument(
        "--hlo-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect compiled HLO op counts for CSV tracking.",
    )
    parser.add_argument(
        "--warmup-jit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one unmeasured warmup pass for split-jit and compiled modes.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="python,split_jit,compiled",
        help="Comma-separated benchmark modes: python, split_jit, compiled (or all).",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Directory for JAX profiler trace output (compiled mode only).",
    )
    parser.add_argument(
        "--dump-ir-dir",
        type=str,
        default=None,
        help="Directory to write compiled JAXPR/HLO artifacts (compiled mode only).",
    )
    parser.add_argument(
        "--ir-dot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When dumping IR, also emit HLO dot graph.",
    )
    parser.add_argument(
        "--hlo-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print action-oriented diagnostics from HLO op counts.",
    )
    parser.add_argument(
        "--compiled-loop-kind",
        type=str,
        default="auto",
        choices=("auto", "fori_loop", "scan"),
        help="Compiled engine loop primitive override (auto keeps env/default).",
    )
    parser.add_argument(
        "--enable-e-shell-split",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override BEAMZ_ENABLE_E_SHELL_SPLIT for this benchmark run.",
    )
    parser.add_argument(
        "--enable-h-shell-split",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override BEAMZ_ENABLE_H_SHELL_SPLIT for this benchmark run.",
    )
    parser.add_argument(
        "--source-single-slab-dense",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override BEAMZ_SOURCE_SINGLE_SLAB_DENSE for compiled source injection.",
    )
    args = parser.parse_args()
    if args.compiled_loop_kind != "auto":
        os.environ["BEAMZ_COMPILED_LOOP_KIND"] = args.compiled_loop_kind
    if args.enable_e_shell_split is not None:
        os.environ["BEAMZ_ENABLE_E_SHELL_SPLIT"] = "1" if args.enable_e_shell_split else "0"
    if args.enable_h_shell_split is not None:
        os.environ["BEAMZ_ENABLE_H_SHELL_SPLIT"] = "1" if args.enable_h_shell_split else "0"
    if args.source_single_slab_dense is not None:
        os.environ["BEAMZ_SOURCE_SINGLE_SLAB_DENSE"] = (
            "1" if args.source_single_slab_dense else "0"
        )

    compiled_loop_kind = os.environ.get("BEAMZ_COMPILED_LOOP_KIND", "scan").strip().lower()
    if compiled_loop_kind in {"fori", "fori-loop"}:
        compiled_loop_kind = "fori_loop"
    e_shell_split = os.environ.get("BEAMZ_ENABLE_E_SHELL_SPLIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    h_shell_split = os.environ.get("BEAMZ_ENABLE_H_SHELL_SPLIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    source_single_slab_dense = os.environ.get(
        "BEAMZ_SOURCE_SINGLE_SLAB_DENSE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}

    modes = parse_modes(args.modes)
    if not modes:
        raise ValueError("No benchmark modes selected. Choose at least one mode.")

    domain_um = parse_triplet_floats(args.domain_um, "domain-um") if args.domain_um else None
    if args.grid_shape:
        nx, ny, nz = parse_triplet_ints(args.grid_shape, "grid-shape")
    elif args.grid_n:
        nx = ny = nz = int(args.grid_n)
    elif (domain_um is not None) and (args.resolution_nm is not None):
        nx = max(1, int(round((domain_um[0] * 1000.0) / float(args.resolution_nm))))
        ny = max(1, int(round((domain_um[1] * 1000.0) / float(args.resolution_nm))))
        nz = max(1, int(round((domain_um[2] * 1000.0) / float(args.resolution_nm))))
    else:
        n_auto = choose_grid_n(args.memory_gb, args.saturation_factor)
        nx = ny = nz = int(n_auto)

    cfg = BenchmarkConfig(
        grid_nx=nx,
        grid_ny=ny,
        grid_nz=nz,
        steps=args.steps,
        points_per_wavelength=args.ppw,
        memory_gb=args.memory_gb,
        saturation_factor=args.saturation_factor,
        domain_um=domain_um,
        resolution_nm=args.resolution_nm,
        courant_factor=args.courant_factor,
        sim_time_fs=args.sim_time_fs,
        scenario=args.scenario,
        source_kind=args.source_kind,
        with_xy_monitor=bool(args.with_xy_monitor),
        monitor_record_interval=max(1, int(args.monitor_record_interval)),
        monitor_plane_z_um=args.monitor_plane_z_um,
    )

    print("3D FDTD benchmark")
    print(f"target_memory_gb={cfg.memory_gb:.1f}")
    print(f"grid_shape={cfg.grid_nx}x{cfg.grid_ny}x{cfg.grid_nz}")
    est_ws = estimate_working_set_gb(cfg.grid_nx, cfg.grid_ny, cfg.grid_nz, cfg.saturation_factor)
    print(f"estimated_working_set_gb~{est_ws:.2f}")
    if cfg.domain_um is not None:
        print(f"domain_um={cfg.domain_um[0]:.6g},{cfg.domain_um[1]:.6g},{cfg.domain_um[2]:.6g}")
    if cfg.resolution_nm is not None:
        print(f"resolution_nm={cfg.resolution_nm:.6g}")
    if cfg.courant_factor is not None:
        print(f"courant_factor={cfg.courant_factor:.6g}")
    if cfg.sim_time_fs is not None:
        print(f"sim_time_fs={cfg.sim_time_fs:.6g}")
    print(f"steps={'auto' if cfg.sim_time_fs is not None else cfg.steps}")
    print(f"repeats={max(1, int(args.repeats))}")
    print(f"warmup_jit={bool(args.warmup_jit)}")
    print(f"modes={','.join(modes)}")
    print(f"scenario={cfg.scenario}")
    print(f"source_kind={cfg.source_kind}")
    print(f"source_sweep={bool(args.source_sweep)}")
    print(f"with_xy_monitor={cfg.with_xy_monitor}")
    if cfg.with_xy_monitor:
        print(f"monitor_record_interval={cfg.monitor_record_interval}")
        if cfg.monitor_plane_z_um is not None:
            print(f"monitor_plane_z_um={cfg.monitor_plane_z_um:.6g}")
    print(f"physics_output_dir={args.physics_output_dir or ''}")
    print(f"physics_save_power_plot={bool(args.physics_save_power_plot)}")
    print(f"physics_export_mode_profiles={bool(args.physics_export_mode_profiles)}")
    print(f"physics_export_ez_snapshot={bool(args.physics_export_ez_snapshot)}")
    print(f"physics_snapshot_step={args.physics_snapshot_step}")
    print(f"compiled_loop_kind={compiled_loop_kind}")
    print(f"e_shell_split={e_shell_split}")
    print(f"h_shell_split={h_shell_split}")
    print(f"source_single_slab_dense={source_single_slab_dense}")
    if e_shell_split or h_shell_split:
        print("warning: shell-split is currently slower on M4 in our measurements.")
    if args.source_sweep and (cfg.source_kind != "auto"):
        print("note: --source-sweep overrides explicit --source-kind.")

    source_cases = ("none", "gaussian", "mode") if args.source_sweep else (cfg.source_kind,)
    repeats = max(1, int(args.repeats))
    csv_path = Path(args.csv)
    physics_run_dir: Path | None = None
    if args.physics_output_dir:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        physics_run_dir = Path(args.physics_output_dir) / stamp
        physics_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"physics_run_dir={physics_run_dir}")

    for source_req in source_cases:
        cfg_case = copy.deepcopy(cfg)
        cfg_case.source_kind = source_req

        (
            sim_base,
            steps,
            dx_used,
            dt_used,
            domain_um_used,
            courant_used,
            sim_time_fs_used,
            source_kind_used,
        ) = make_simulation(cfg_case)

        print(f"\n=== Case: source_kind={source_kind_used} ===")
        print(
            f"resolved_domain_um={domain_um_used[0]:.6g},{domain_um_used[1]:.6g},{domain_um_used[2]:.6g}"
        )
        print(f"resolved_resolution_nm={dx_used * 1e9:.6g}")
        print(f"resolved_dt_fs={dt_used * 1e15:.6g}")
        print(f"resolved_courant_factor={courant_used:.6g}")
        print(f"resolved_steps={steps}")

        t_py, py_runs = float("nan"), []
        t_split, split_runs = float("nan"), []
        t_compiled, compiled_runs = float("nan"), []
        if "python" in modes:
            t_py, py_runs = run_repeated(
                sim_base, steps, repeats, run_legacy_python_step, warmup=False
            )
        if "split_jit" in modes:
            t_split, split_runs = run_repeated(
                sim_base,
                steps,
                repeats,
                run_legacy_split_jit,
                warmup=bool(args.warmup_jit),
            )
        if "compiled" in modes:
            if args.profile_dir:
                import jax

                profile_path = Path(args.profile_dir) / f"source_{source_kind_used}"
                profile_path.mkdir(parents=True, exist_ok=True)
                with jax.profiler.trace(str(profile_path)):
                    t_compiled, compiled_runs = run_repeated(
                        sim_base, steps, repeats, run_compiled, warmup=bool(args.warmup_jit)
                    )
            else:
                t_compiled, compiled_runs = run_repeated(
                    sim_base, steps, repeats, run_compiled, warmup=bool(args.warmup_jit)
                )
        hlo_stats = (
            compiled_hlo_stats(copy.deepcopy(sim_base), steps)
            if args.hlo_stats and ("compiled" in modes)
            else None
        )

        tcups_py = _tcups(sim_base, steps, t_py)
        tcups_split = _tcups(sim_base, steps, t_split)
        tcups_compiled = _tcups(sim_base, steps, t_compiled)

        print("Results")
        if "python" in modes:
            print(
                f"legacy_python_step (median): {t_py:.6f}s, {_safe_div(t_py, steps):.6e}s/step, {tcups_py:.6e} TCUPS"
            )
        if "split_jit" in modes:
            print(
                f"legacy_split_jit   (median): {t_split:.6f}s, {_safe_div(t_split, steps):.6e}s/step, {tcups_split:.6e} TCUPS"
            )
        if "compiled" in modes:
            print(
                f"compiled_v3_scan   (median): {t_compiled:.6f}s, {_safe_div(t_compiled, steps):.6e}s/step, {tcups_compiled:.6e} TCUPS"
            )
        if repeats > 1:
            print("Per-Run Times (s)")
            if "python" in modes:
                print(f"legacy_python_step: {[round(x, 6) for x in py_runs]}")
            if "split_jit" in modes:
                print(f"legacy_split_jit:   {[round(x, 6) for x in split_runs]}")
            if "compiled" in modes:
                print(f"compiled_v3_scan:   {[round(x, 6) for x in compiled_runs]}")

        if "compiled" in modes and (("python" in modes) or ("split_jit" in modes)):
            print("Speedups")
            if "python" in modes:
                print(f"compiled / legacy_python_step: {_safe_div(t_py, t_compiled):.2f}x")
            if "split_jit" in modes:
                print(f"compiled / legacy_split_jit:   {_safe_div(t_split, t_compiled):.2f}x")
        if hlo_stats is not None:
            print("Compiled HLO Stats")
            print(
                " ".join(
                    [
                        f"text_len={hlo_stats['text_len']}",
                        f"fusion={hlo_stats['fusion']}",
                        f"scatter={hlo_stats['scatter']}",
                        f"dynamic-update-slice={hlo_stats['dynamic-update-slice']}",
                        f"slice={hlo_stats['slice']}",
                        f"copy={hlo_stats['copy']}",
                        f"while={hlo_stats['while']}",
                    ]
                )
            )
            if args.hlo_diagnostics:
                print("Compiled HLO Diagnostics")
                for note in hlo_diagnostics(hlo_stats):
                    print(f"- {note}")

        if args.dump_ir_dir and ("compiled" in modes):
            ir_dir = Path(args.dump_ir_dir) / f"source_{source_kind_used}"
            ir_stats = dump_compiled_ir_artifacts(
                copy.deepcopy(sim_base),
                steps,
                ir_dir,
                include_dot=bool(args.ir_dot),
                include_optimized_hlo=True,
            )
            print(f"IR artifacts written: {ir_dir}")
            print(
                " ".join(
                    [
                        f"text_len={ir_stats['text_len']}",
                        f"fusion={ir_stats['fusion']}",
                        f"scatter={ir_stats['scatter']}",
                        f"dynamic-update-slice={ir_stats['dynamic-update-slice']}",
                        f"slice={ir_stats['slice']}",
                        f"copy={ir_stats['copy']}",
                        f"while={ir_stats['while']}",
                    ]
                )
            )

        physics_summary: dict[str, object] | None = None
        physics_artifact_dir: str | None = None
        ez_snapshot_info: dict[str, object] | None = None
        if physics_run_dir is not None:
            physics_case_dir = physics_run_dir / f"{cfg_case.scenario}_source_{source_kind_used}"
            physics_case_dir.mkdir(parents=True, exist_ok=True)
            physics_sim = copy.deepcopy(sim_base)
            physics_elapsed = run_compiled(physics_sim, steps)
            if bool(args.physics_export_ez_snapshot):
                snap_step = _resolve_snapshot_step(args.physics_snapshot_step, steps)
                snap_sim = copy.deepcopy(sim_base)
                snap_elapsed = run_compiled(snap_sim, snap_step)
                ez_snapshot_info = _export_ez_snapshot_artifact(
                    snap_sim,
                    physics_case_dir,
                    label=f"step_{snap_step}",
                )
            else:
                snap_elapsed = 0.0
            physics_summary = export_physics_artifacts(
                physics_sim,
                physics_case_dir,
                scenario=cfg_case.scenario,
                source_kind=source_kind_used,
                save_power_plot=bool(args.physics_save_power_plot),
                export_mode_profiles=bool(args.physics_export_mode_profiles),
                ez_snapshot=ez_snapshot_info,
            )
            physics_artifact_dir = str(physics_case_dir)
            print(
                "Physics Artifacts "
                + f"(compiled_validation_s={physics_elapsed:.6f}, "
                + f"snapshot_s={snap_elapsed:.6f}, "
                + f"monitor_count={physics_summary.get('monitor_count', 0)}, "
                + f"mode_source_count={physics_summary.get('mode_source_count', 0)})"
            )
            if ez_snapshot_info is not None:
                print(
                    "  "
                    + f"ez_snapshot: step={ez_snapshot_info.get('step', -1)} "
                    + f"time_fs={float(ez_snapshot_info.get('time_s', 0.0)) * 1e15:.3f} "
                    + f"png={ez_snapshot_info.get('png', '')}"
                )
            for mode_meta in physics_summary.get("mode_sources", []):
                print(
                    "  "
                    + f"mode_source[{mode_meta.get('source_index', 0)}]: "
                    + f"neff={mode_meta.get('neff_real', np.nan):.6f}"
                    + f"+{mode_meta.get('neff_imag', np.nan):.3e}j "
                    + f"pol={mode_meta.get('pol', '')} "
                    + f"direction={mode_meta.get('direction', '')}"
                )

        _append_csv(
            csv_path=csv_path,
            cfg=cfg_case,
            steps=steps,
            scenario=cfg_case.scenario,
            source_kind=source_kind_used,
            resolution_nm_used=dx_used * 1e9,
            courant_factor_used=courant_used,
            sim_time_fs_used=sim_time_fs_used,
            domain_um_used=domain_um_used,
            estimated_working_set_gb=est_ws,
            t_py=t_py,
            t_split=t_split,
            t_compiled=t_compiled,
            tcups_py=tcups_py,
            tcups_split=tcups_split,
            tcups_compiled=tcups_compiled,
            py_runs_s=py_runs,
            split_runs_s=split_runs,
            compiled_runs_s=compiled_runs,
            repeats=repeats,
            warmup_jit=bool(args.warmup_jit),
            modes=modes,
            compiled_loop_kind=compiled_loop_kind,
            e_shell_split=e_shell_split,
            h_shell_split=h_shell_split,
            source_single_slab_dense=source_single_slab_dense,
            hlo_stats=hlo_stats,
            physics_artifact_dir=physics_artifact_dir,
            physics_monitor_count=(
                int(physics_summary.get("monitor_count", 0))
                if physics_summary is not None
                else None
            ),
            physics_monitor_records_max=(
                int(physics_summary.get("monitor_records_max", 0))
                if physics_summary is not None
                else None
            ),
            physics_mode_source_count=(
                int(physics_summary.get("mode_source_count", 0))
                if physics_summary is not None
                else None
            ),
            physics_ez_snapshot_step=(
                int(ez_snapshot_info.get("step", 0))
                if ez_snapshot_info is not None
                else None
            ),
            physics_ez_snapshot_png=(
                str(ez_snapshot_info.get("png", "")) if ez_snapshot_info is not None else None
            ),
            physics_ez_snapshot_npz=(
                str(ez_snapshot_info.get("npz", "")) if ez_snapshot_info is not None else None
            ),
        )
        print(f"CSV appended: {csv_path}")


if __name__ == "__main__":
    main()
