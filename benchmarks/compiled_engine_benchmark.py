"""3D benchmark comparing legacy and v0.3 compiled FDTD loops.

Defaults are tuned to push an Apple M4 (24 GB unified memory, 10 CPU cores):
- 3D grid sizing auto-selected from memory budget
- 10-thread hint for CPU backends
- enough timesteps to amortize compilation
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--grid-n", type=int, default=0)
    parser.add_argument("--memory-gb", type=float, default=24.0)
    parser.add_argument("--ppw", type=int, default=10)
    parser.add_argument(
        "--saturation-factor",
        type=float,
        default=4.0,
        help="Higher = more conservative memory estimate.",
    )
    args = parser.parse_args()

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
    print(
        f"estimated_working_set_gb~{estimate_working_set_gb(cfg.grid_n, cfg.saturation_factor):.2f}"
    )
    print(f"steps={cfg.steps}")

    sim_base, steps = make_simulation(cfg)

    # Benchmark each mode from identical initial state.
    sim_py = copy.deepcopy(sim_base)
    sim_split = copy.deepcopy(sim_base)
    sim_compiled = copy.deepcopy(sim_base)

    t_py = run_legacy_python_step(sim_py, steps)
    t_split = run_legacy_split_jit(sim_split, steps)
    t_compiled = run_compiled(sim_compiled, steps)

    tcups_py = _tcups(sim_py, steps, t_py)
    tcups_split = _tcups(sim_split, steps, t_split)
    tcups_compiled = _tcups(sim_compiled, steps, t_compiled)

    print("\nResults")
    print(f"legacy_python_step: {t_py:.6f}s, {t_py/steps:.6e}s/step, {tcups_py:.6e} TCUPS")
    print(f"legacy_split_jit:   {t_split:.6f}s, {t_split/steps:.6e}s/step, {tcups_split:.6e} TCUPS")
    print(f"compiled_v3_scan:   {t_compiled:.6f}s, {t_compiled/steps:.6e}s/step, {tcups_compiled:.6e} TCUPS")

    print("\nSpeedups")
    print(f"compiled / legacy_python_step: {t_py / t_compiled:.2f}x")
    print(f"compiled / legacy_split_jit:   {t_split / t_compiled:.2f}x")


if __name__ == "__main__":
    main()
