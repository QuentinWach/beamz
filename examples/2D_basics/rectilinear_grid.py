"""Run and time equivalent uniform and rectilinear BeamZ simulations."""

from __future__ import annotations

import json
import statistics
import time

import jax
import numpy as np

import beamz as bz
from beamz.design import MaterialGrid


def _edges(extent: float, count: int, stretch: float) -> np.ndarray:
    unit = np.linspace(0.0, 1.0, count + 1)
    warped = unit + stretch * np.sin(2.0 * np.pi * unit) / (2.0 * np.pi)
    return extent * warped


def _simulation(grid: bz.RectilinearGrid, steps: int) -> bz.Simulation:
    values = np.ones((grid.shape[1], grid.shape[0]), dtype=np.float32)
    materials = MaterialGrid(
        values,
        np.zeros_like(values),
        values,
        grid.minimum_spacing,
        values.shape,
        grid=grid,
    )
    dt = grid.cfl_time_step(0.7, active_axes=("x", "y"))
    source = bz.GaussianSource(
        position=(0.28 * grid.extent[0], 0.5 * grid.extent[1]),
        width=0.06 * grid.extent[1],
        signal=np.sin(np.linspace(0.0, 8.0 * np.pi, steps)),
    )
    monitor = bz.FieldMonitor(
        center=(0.72 * grid.extent[0], 0.5 * grid.extent[1], 0.0),
        size=(0.0, grid.extent[1], 0.0),
        freqs=np.asarray([2.0e14]),
        name="transmission",
    )
    return bz.Simulation(
        material_grid=materials,
        sources=[source],
        monitors=[monitor],
        time=np.arange(steps, dtype=np.float64) * dt,
        normalize_source=None,
    )


def _warm_runtime(simulation: bz.Simulation, repeats: int = 5) -> float:
    warm = simulation.advance()
    jax.block_until_ready(warm.state.ez)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        run = simulation.advance()
        jax.block_until_ready(run.state.ez)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    nx, ny, steps = 64, 48, 80
    extent = (0.96 * bz.um, 0.72 * bz.um)
    uniform = bz.RectilinearGrid.from_spacing(
        (nx, ny, 1),
        (extent[0] / nx, extent[1] / ny, bz.um),
    )
    rectilinear = bz.RectilinearGrid(
        _edges(extent[0], nx, 0.12),
        _edges(extent[1], ny, -0.10),
        np.asarray([0.0, bz.um]),
    )
    simulations = {
        "uniform": _simulation(uniform, steps),
        "rectilinear": _simulation(rectilinear, steps),
    }
    report = {}
    for name, simulation in simulations.items():
        program = simulation.compile()
        metric_bytes = sum(int(value.nbytes) for value in program.metrics)
        elapsed = _warm_runtime(simulation)
        monitor = program.monitors[0]
        integrated_length = (
            float(np.sum(monitor.integration_weights))
            if monitor.integration_weights.size
            else float(monitor.power_scale) * int(monitor.dft_point_count)
        )
        report[name] = {
            "metric_kind": program.config.metric_kind,
            "update_path": (
                "legacy_scalar"
                if program.config.metric_kind == "isotropic_uniform"
                else "separable_metrics"
            ),
            "shape_yx": list(simulation.material_grid.shape),
            "minimum_spacing_nm": simulation.material_grid.resolution / bz.nm,
            "metric_bytes": metric_bytes,
            "median_80_step_runtime_ms": 1e3 * elapsed,
            "monitor_weight_sum_um": integrated_length / bz.um,
        }
    report["rectilinear_over_uniform_runtime"] = (
        report["rectilinear"]["median_80_step_runtime_ms"]
        / report["uniform"]["median_80_step_runtime_ms"]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
