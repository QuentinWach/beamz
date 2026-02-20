"""Benchmark script for v0.3 compiled FDTD engine."""

from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np

# Ensure local workspace imports when launched from benchmarks/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def run_benchmark(steps: int = 600):
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl,
        1.0,
        dims=2,
        safety_factor=0.95,
        points_per_wavelength=10,
    )

    domain = 8 * wl
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    t = np.arange(0, steps * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.4,
    )
    src = GaussianSource(position=(domain / 2, domain / 2), width=wl / 6, signal=signal)

    sim = Simulation(
        design=design,
        devices=[src],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    t0 = time.perf_counter()
    sim.run_compiled(num_steps=steps, progress=False)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    ny, nx = sim.fields.Ez.shape
    voxels = ny * nx
    updates = 6 * voxels * steps
    tcups = updates / elapsed / 1e12

    print(f"steps: {steps}")
    print(f"grid: {ny}x{nx}")
    print(f"elapsed_s: {elapsed:.6f}")
    print(f"step_s: {elapsed / steps:.6e}")
    print(f"tcups: {tcups:.6e}")


if __name__ == "__main__":
    run_benchmark()
