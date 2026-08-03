from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import beamz as bz
from beamz.design import MaterialGrid
from beamz.simulation.execute import compiled_source_batches
from beamz.simulation.kernels import CompiledStepContext, select_update_kernel


def _simulation(grid: bz.RectilinearGrid) -> bz.Simulation:
    values = np.ones((grid.shape[1], grid.shape[0]), dtype=np.float32)
    materials = MaterialGrid(
        values,
        np.zeros_like(values),
        values,
        grid.minimum_spacing,
        values.shape,
        grid=grid,
    )
    return bz.Simulation(
        material_grid=materials,
        time=np.asarray([0.0, 1e-16]),
        normalize_source=None,
    )


def _selected_kind(program) -> str:
    config = program.config
    context = CompiledStepContext(
        config=config,
        boundary=program.boundary,
        source_batches=compiled_source_batches(program.sources),
        metrics=program.metrics,
        resolution=float(config.resolution),
        dt=float(config.dt),
        dt_scalar=jnp.asarray(config.dt, dtype=jnp.float32),
        is_3d=bool(config.is_3d),
    )
    return select_update_kernel(context).kind


def test_uniform_grid_selects_unchanged_scalar_kernel_and_allocates_no_metrics():
    program = _simulation(bz.RectilinearGrid.from_spacing((32, 24, 1), 30e-9)).compile()

    assert program.config.metric_kind == "isotropic_uniform"
    assert _selected_kind(program) == "physical_tm_xy"
    assert sum(int(value.nbytes) for value in program.metrics) == 0


def test_rectilinear_grid_selects_metric_kernel_with_linear_metric_storage():
    x_edges = np.linspace(0.0, 1.0, 33) ** 1.1
    y_edges = np.linspace(0.0, 1.0, 25) ** 1.05
    program = _simulation(
        bz.RectilinearGrid(x_edges, y_edges, np.asarray([0.0, 1.0]))
    ).compile()

    assert program.config.metric_kind == "rectilinear"
    assert _selected_kind(program) == "rectilinear_physical_tm_xy"
    metric_elements = sum(int(value.size) for value in program.metrics)
    assert metric_elements == 2 * 32 + 1 + 2 * 24 + 1
    assert metric_elements < 4 * (32 * 24)


def test_uniform_3d_flux_monitor_keeps_scalar_integration_path():
    grid = bz.RectilinearGrid.from_spacing((4, 3, 2), 1.0)
    values = np.ones(grid.shape_zyx, dtype=np.float32)
    materials = MaterialGrid(
        values,
        np.zeros_like(values),
        values,
        1.0,
        values.shape,
        grid=grid,
    )
    monitor = bz.FluxMonitor(
        center=(2.0, 1.5, 1.0),
        size=(4.0, 3.0, 0.0),
        freqs=np.asarray([1.0]),
    )

    spec = (
        bz.Simulation(
            material_grid=materials,
            monitors=[monitor],
            time=np.asarray([0.0, 1e-16]),
        )
        .compile()
        .monitors[0]
    )

    assert spec.integration_weights.size == 0
    assert spec.power_scale == 2.0
