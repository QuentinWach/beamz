from __future__ import annotations

import numpy as np
import pytest

from beamz import LIGHT_SPEED, Grid, GridSpec, RectilinearGrid
from beamz.design import MaterialGrid


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
