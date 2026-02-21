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
import os
import platform
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
    PML,
    Design,
    GaussianSource,
    Material,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)


@dataclass
class BenchmarkConfig:
    grid_n: int
    steps: int
    points_per_wavelength: int
    memory_gb: float
    saturation_factor: float


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


def _safe_div(num: float, den: float) -> float:
    if den <= 0 or (not np.isfinite(num)) or (not np.isfinite(den)):
        return float("nan")
    return num / den


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
    hlo_stats: dict[str, int] | None = None,
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "timestamp_utc",
        "hostname",
        "platform",
        "python_version",
        "git_commit",
        "grid_n",
        "steps",
        "repeats",
        "warmup_jit",
        "modes",
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
        "grid_n": cfg.grid_n,
        "steps": steps,
        "repeats": int(repeats),
        "warmup_jit": int(bool(warmup_jit)),
        "modes": ";".join(modes),
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


def estimate_working_set_gb(n: int, saturation_factor: float) -> float:
    """Crude working-set estimator for 3D FDTD in float32.

    Uses ~24 arrays worth of storage per voxel (fields + materials + coeffs + intermediates),
    then multiplies by a saturation factor to approximate temporary buffers and overhead.
    """
    voxels = float(n**3)
    bytes_per_voxel = 24.0 * 4.0
    return voxels * bytes_per_voxel * saturation_factor / 1e9


def choose_grid_n(memory_gb: float, saturation_factor: float) -> int:
    candidates = [384, 352, 320, 288, 256, 224, 192]
    budget = memory_gb * 0.8
    for n in candidates:
        if estimate_working_set_gb(n, saturation_factor) <= budget:
            return n
    return 160


def make_simulation(cfg: BenchmarkConfig) -> tuple[Simulation, int]:
    wl = 1.55 * um
    n_bg = 1.0
    dx, dt = calc_optimal_fdtd_params(
        wl,
        n_bg,
        dims=3,
        safety_factor=0.95,
        points_per_wavelength=cfg.points_per_wavelength,
    )

    n = cfg.grid_n
    width = n * dx
    height = n * dx
    depth = n * dx

    design = Design(
        width=width,
        height=height,
        depth=depth,
        material=Material(permittivity=n_bg**2),
    )

    t_arr = np.arange(0, cfg.steps * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t_arr,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t_arr[-1] * 0.5,
    )

    source = GaussianSource(
        position=(0.35 * width, 0.5 * height, 0.5 * depth),
        width=wl / 6,
        signal=signal,
    )

    sim = Simulation(
        design=design,
        devices=[source],
        boundaries=[PML(edges="all", thickness=1.0 * wl)],
        time=t_arr,
        resolution=dx,
    )
    return sim, cfg.steps


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
    import jax.numpy as jnp
    from beamz.simulation.compiled import EngineState, MonitorState, monitor_state_size

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

    hlo_text = (
        program._compiled_scan
        .lower(engine_state, monitor_state, program._update_coefficients())
        .compile()
        .as_text()
        .lower()
    )
    return {
        "text_len": len(hlo_text),
        "fusion": hlo_text.count("fusion"),
        "scatter": hlo_text.count("scatter"),
        "dynamic-update-slice": hlo_text.count("dynamic-update-slice"),
        "slice": hlo_text.count("slice"),
        "copy": hlo_text.count("copy"),
        "while": hlo_text.count("while"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--grid-n", type=int, default=0)
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
    args = parser.parse_args()
    modes = parse_modes(args.modes)
    if not modes:
        raise ValueError("No benchmark modes selected. Choose at least one mode.")

    grid_n = args.grid_n or choose_grid_n(args.memory_gb, args.saturation_factor)
    cfg = BenchmarkConfig(
        grid_n=grid_n,
        steps=args.steps,
        points_per_wavelength=args.ppw,
        memory_gb=args.memory_gb,
        saturation_factor=args.saturation_factor,
    )

    print("3D FDTD benchmark")
    print(f"target_memory_gb={cfg.memory_gb:.1f}")
    print(f"grid_n={cfg.grid_n}")
    est_ws = estimate_working_set_gb(cfg.grid_n, cfg.saturation_factor)
    print(f"estimated_working_set_gb~{est_ws:.2f}")
    print(f"steps={cfg.steps}")
    print(f"repeats={max(1, int(args.repeats))}")
    print(f"warmup_jit={bool(args.warmup_jit)}")
    print(f"modes={','.join(modes)}")

    sim_base, steps = make_simulation(cfg)
    repeats = max(1, int(args.repeats))

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

            profile_path = Path(args.profile_dir)
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

    print("\nResults")
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
        print("\nPer-Run Times (s)")
        if "python" in modes:
            print(f"legacy_python_step: {[round(x, 6) for x in py_runs]}")
        if "split_jit" in modes:
            print(f"legacy_split_jit:   {[round(x, 6) for x in split_runs]}")
        if "compiled" in modes:
            print(f"compiled_v3_scan:   {[round(x, 6) for x in compiled_runs]}")

    if "compiled" in modes and (("python" in modes) or ("split_jit" in modes)):
        print("\nSpeedups")
        if "python" in modes:
            print(f"compiled / legacy_python_step: {_safe_div(t_py, t_compiled):.2f}x")
        if "split_jit" in modes:
            print(f"compiled / legacy_split_jit:   {_safe_div(t_split, t_compiled):.2f}x")
    if hlo_stats is not None:
        print("\nCompiled HLO Stats")
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

    csv_path = Path(args.csv)
    _append_csv(
        csv_path=csv_path,
        cfg=cfg,
        steps=steps,
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
        hlo_stats=hlo_stats,
    )
    print(f"\nCSV appended: {csv_path}")


if __name__ == "__main__":
    main()
