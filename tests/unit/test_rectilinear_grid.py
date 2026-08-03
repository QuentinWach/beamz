from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import LIGHT_SPEED, Grid, GridSpec, RectilinearGrid
from beamz.design import MaterialGrid
from beamz.lattice import component_shapes
from beamz.simulation.model import DerivativeMetricPlan, ShardingLayout
from beamz.simulation.sharding import lower_derivative_metrics


def test_grid_alias_and_edge_arrays_are_immutable():
    assert Grid is RectilinearGrid
    grid = RectilinearGrid(
        np.asarray([0.0, 1.0, 3.0]),
        np.asarray([-1.0, 2.0]),
        np.asarray([4.0, 5.0]),
    )
    assert grid.shape == (2, 1, 1)
    with pytest.raises(ValueError):
        grid.x_edges[0] = -2.0


def test_metric_kinds_distinguish_isotropic_axis_uniform_and_rectilinear():
    isotropic = RectilinearGrid.from_spacing((2, 3, 4), 0.5)
    axis_uniform = RectilinearGrid.from_spacing((2, 3, 4), (0.5, 0.75, 1.0))
    rectilinear = RectilinearGrid(
        np.asarray([0.0, 0.5, 1.25]),
        axis_uniform.y_edges,
        axis_uniform.z_edges,
    )
    assert isotropic.metric_kind == "isotropic_uniform"
    assert axis_uniform.metric_kind == "axis_uniform"
    assert rectilinear.metric_kind == "rectilinear"
    with pytest.raises(ValueError, match="one uniform"):
        _ = axis_uniform.uniform_spacing


def test_physical_metrics_and_coordinate_snapping():
    grid = RectilinearGrid(
        np.asarray([0.0, 1.0, 3.0]),
        np.asarray([0.0, 2.0, 5.0]),
        np.asarray([0.0, 4.0, 6.0]),
    )
    np.testing.assert_allclose(grid.centers("x"), [0.5, 2.0])
    np.testing.assert_allclose(
        grid.cell_volume(),
        np.asarray(
            [
                [[8.0, 16.0], [12.0, 24.0]],
                [[4.0, 8.0], [6.0, 12.0]],
            ]
        ),
    )
    assert grid.coord_to_edge_index("x", 2.2, snap="lower") == 1
    assert grid.coord_to_edge_index("x", 2.2, snap="upper") == 2
    assert grid.coord_to_edge_index("x", 2.2, snap="nearest") == 2


def test_cfl_uses_minimum_spacing_on_each_active_axis():
    grid = RectilinearGrid(
        np.asarray([0.0, 2.0, 5.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 4.0]),
    )
    expected_3d = 0.99 / (LIGHT_SPEED * np.sqrt(1 / 2.0**2 + 1 / 1.0**2 + 1 / 4.0**2))
    expected_2d = 0.99 / (LIGHT_SPEED * np.sqrt(1 / 2.0**2 + 1 / 1.0**2))
    assert grid.cfl_time_step(0.99) == pytest.approx(expected_3d)
    assert grid.cfl_time_step(0.99, active_axes=("x", "y")) == pytest.approx(
        expected_2d
    )
    assert GridSpec.uniform(123.0).resolve_time_step(grid, dims=2) == pytest.approx(
        expected_2d
    )


def test_material_grid_realizes_uniform_legacy_inputs_and_validates_explicit_grid():
    values = np.ones((2, 3))
    uniform = MaterialGrid(values, values, values, 0.5, values.shape)
    assert uniform.grid == RectilinearGrid.from_spacing((3, 2, 1), 0.5)

    rectilinear = RectilinearGrid(
        np.asarray([2.0, 2.25, 3.0, 4.0]),
        np.asarray([-1.0, -0.5, 1.0]),
        np.asarray([4.0, 5.0]),
    )
    explicit = MaterialGrid(
        values,
        values,
        values,
        999.0,
        values.shape,
        grid=rectilinear,
    )
    assert explicit.grid is rectilinear
    assert explicit.origin == rectilinear.origin
    assert explicit.resolution == pytest.approx(0.25)


def test_rectilinear_metrics_extend_across_sharding_padding():
    logical = component_shapes((3, 3, 4))
    padded = {
        name: (*shape[:-1], int(np.ceil(shape[-1] / 3)) * 3)
        for name, shape in logical.items()
    }
    layout = ShardingLayout(
        enabled=True,
        axis_name="x",
        axis=2,
        num_devices=3,
        backend="cpu",
        logical_shapes=logical,
        padded_shapes=padded,
    )
    metrics = DerivativeMetricPlan(
        jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        jnp.asarray([10.0, 20.0, 30.0]),
        jnp.asarray([100.0, 200.0, 300.0]),
        jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        jnp.asarray([10.0, 20.0, 30.0, 40.0]),
        jnp.asarray([100.0, 200.0, 300.0, 400.0]),
    )

    lowered = lower_derivative_metrics(metrics, layout)

    np.testing.assert_array_equal(lowered.e_to_h_x, [1, 2, 3, 4, 4])
    np.testing.assert_array_equal(lowered.h_to_e_x, [1, 2, 3, 4, 5, 5, 5])
    np.testing.assert_array_equal(lowered.e_to_h_y, metrics.e_to_h_y)
    np.testing.assert_array_equal(lowered.h_to_e_z, metrics.h_to_e_z)


def test_heterogeneous_rectilinear_material_grid_requires_direct_yee_values():
    grid = RectilinearGrid(
        np.asarray([0.0, 0.2, 1.0, 2.0]),
        np.asarray([0.0, 0.5, 1.5]),
        np.asarray([0.0, 1.0]),
    )
    permittivity = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="Missing: eps_z"):
        MaterialGrid(
            permittivity,
            np.zeros_like(permittivity),
            np.ones_like(permittivity),
            grid.minimum_spacing,
            permittivity.shape,
            grid=grid,
        )
