from __future__ import annotations

import time

import numpy as np

from beamz import (
    PML,
    Design,
    Material,
    ModeSource,
    Rectangle,
    Simulation,
    ramped_cosine,
)
from beamz.const import LIGHT_SPEED, µm
from beamz.visual.helpers import calc_optimal_fdtd_params

WL = 1.55 * µm
N_AIR = 1.0
N_CLAD = 1.44
N_CORE = 3.48
DOMAIN_WIDTH = 6.5 * µm
DOMAIN_HEIGHT = 6.5 * µm
DOMAIN_DEPTH = 4.0 * µm
PML_THICKNESS = 0.75 * WL
NUM_STEPS = 600


def build_design() -> Design:
    design = Design(
        width=DOMAIN_WIDTH,
        height=DOMAIN_HEIGHT,
        depth=DOMAIN_DEPTH,
        material=Material(N_AIR**2),
    )
    design += Rectangle(
        position=(0.0, 0.0, 0.0),
        width=DOMAIN_WIDTH,
        height=DOMAIN_HEIGHT,
        depth=2.0 * µm,
        material=Material(N_CLAD**2),
    )
    design += Rectangle(
        position=(0.0, 3.0 * µm, 2.0 * µm),
        width=DOMAIN_WIDTH,
        height=0.5 * µm,
        depth=0.22 * µm,
        material=Material(N_CORE**2),
    )
    return design


def main() -> None:
    dx, dt = calc_optimal_fdtd_params(
        WL,
        N_CORE,
        dims=3,
        safety_factor=0.999,
        points_per_wavelength=16,
        width=DOMAIN_WIDTH,
        height=DOMAIN_HEIGHT,
        depth=DOMAIN_DEPTH,
    )
    time_steps = np.arange(NUM_STEPS, dtype=np.float64) * dt
    signal = ramped_cosine(
        time_steps,
        amplitude=1.0,
        frequency=LIGHT_SPEED / WL,
        ramp_duration=WL * 6.0 / LIGHT_SPEED,
        t_max=time_steps[-1] * 0.5,
    )

    t0 = time.perf_counter()
    design = build_design()
    grid = design.rasterize(resolution=dx)
    raster_s = time.perf_counter() - t0

    source = ModeSource(
        grid=grid,
        center=(3.25 * µm, 3.25 * µm, 2.11 * µm),
        width=3.5 * µm,
        height=0.8 * µm,
        wavelength=WL,
        pol="tm",
        signal=signal,
        direction="+x",
    )

    t0 = time.perf_counter()
    source.initialize(grid.permittivity, dx)
    source_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[],
        boundaries=[PML(edges="all", thickness=PML_THICKNESS)],
        time=time_steps,
        resolution=dx,
    )
    setup_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim.run_compiled(
        num_steps=NUM_STEPS,
        progress=False,
        record_fields=[],
        store_snapshots=False,
    )
    run_s = max(time.perf_counter() - t0, 1e-12)

    shape = tuple(int(v) for v in np.asarray(grid.permittivity).shape)
    cells = int(np.prod(shape))
    neff = float(np.real(np.asarray(source._neff)))
    sim_time_fs = float((NUM_STEPS - 1) * dt * 1e15)

    print("3D waveguide ModeSource benchmark")
    print(f"grid={shape[0]} x {shape[1]} x {shape[2]} cells ({cells:,} total)")
    print(f"dx={dx / µm:.4f} um, dt={dt * 1e18:.4f} as, steps={NUM_STEPS}")
    print(f"mode_neff={neff:.6f}, simulated_time={sim_time_fs:.2f} fs")
    print(
        "timing: "
        f"raster={raster_s:.3f}s, mode_source={source_s:.3f}s, "
        f"setup={setup_s:.3f}s, run={run_s:.3f}s"
    )
    print(
        "throughput: "
        f"{NUM_STEPS / run_s:.2f} steps/s, "
        f"{cells * NUM_STEPS / run_s / 1e6:.2f} MCUPS"
    )


if __name__ == "__main__":
    main()
