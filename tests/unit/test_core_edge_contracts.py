"""Small edge contracts shared by the lattice and modal-analysis core."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from beamz.analysis.modal_projection.diagnostics import (
    _modal_projection_reconstruction_diagnostics_from_matrix,
    _modal_projection_reconstruction_residual,
)
from beamz.devices.monitors.monitors import (
    _line_integral_scale_2d,
    _line_normal_2d,
)
from beamz.lattice import (
    _uniform_axis_centers,
    common_grid_shape_3d,
    component_axis_offsets_3d,
    component_shape_3d,
    linear_interpolation_plan,
    plane_axes_3d,
    plane_sample_area,
)


def test_lattice_geometry_rejects_unsupported_components_axes_and_ranks():
    with pytest.raises(ValueError, match="Unsupported component"):
        component_axis_offsets_3d("Et")
    with pytest.raises(ValueError, match="three dimensions"):
        component_shape_3d("Ex", (2, 3))
    with pytest.raises(ValueError, match="Unsupported plane normal"):
        plane_axes_3d("time")


def test_common_grid_shape_uses_material_fallback_and_rejects_non_3d_fields():
    fields = SimpleNamespace(permittivity=np.ones((2, 3, 4)))
    assert common_grid_shape_3d(fields) == (2, 3, 4)

    fields = SimpleNamespace(permittivity=np.ones((2, 3)))
    with pytest.raises(ValueError, match="rank-3"):
        common_grid_shape_3d(fields)


def test_linear_interpolation_handles_empty_singleton_and_clamped_targets():
    with pytest.raises(ValueError, match="cannot be empty"):
        linear_interpolation_plan([], [0.0])

    low, high, weight_low, weight_high = linear_interpolation_plan(
        [2.0], [-1.0, 2.0, 10.0]
    )
    np.testing.assert_array_equal(low, [0, 0, 0])
    np.testing.assert_array_equal(high, [0, 0, 0])
    np.testing.assert_array_equal(weight_low, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(weight_high, [0.0, 0.0, 0.0])

    low, high, weight_low, weight_high = linear_interpolation_plan(
        [0.0, 2.0], [-1.0, 1.0, 3.0]
    )
    np.testing.assert_array_equal(low, [0, 0, 1])
    np.testing.assert_array_equal(high, [0, 1, 1])
    np.testing.assert_allclose(weight_low, [1.0, 0.5, 1.0])
    np.testing.assert_allclose(weight_high, [0.0, 0.5, 0.0])


def test_plane_coordinates_and_area_have_explicit_degenerate_fallbacks():
    assert _uniform_axis_centers(2.0, 4.0, 0).shape == (0,)
    np.testing.assert_array_equal(_uniform_axis_centers(2.0, 4.0, 1), [3.0])
    assert plane_sample_area(([1.0], [2.0]), fallback_step=0.25) == 0.25**2


def test_monitor_line_orientation_and_measure_cover_degenerate_geometry():
    assert _line_normal_2d(None, (1.0, 1.0)) is None
    assert _line_normal_2d((0.0, 0.0), (1.0, 1.0)) is None
    assert _line_normal_2d((0.0, 1.0), (0.0, -1.0)) == ("x", -1.0)
    assert _line_normal_2d((1.0, 0.0), (-1.0, 0.0)) == ("y", 1.0)
    assert _line_integral_scale_2d("x", 2.0, 3.0) == 3.0
    assert _line_integral_scale_2d("y", 2.0, 3.0) == 2.0
    assert _line_integral_scale_2d("diagonal", 2.0, 4.0) == 3.0


def test_modal_reconstruction_residual_handles_invalid_and_exact_systems():
    assert np.isnan(_modal_projection_reconstruction_residual([1.0], {}, [1.0]))
    projection = {"mode_matrix": np.eye(2, dtype=complex)}
    assert np.isnan(
        _modal_projection_reconstruction_residual([], projection, [1.0, 1.0])
    )
    assert np.isnan(
        _modal_projection_reconstruction_residual([0.0, 0.0], projection, [0.0, 0.0])
    )
    assert (
        _modal_projection_reconstruction_residual([2.0, 3.0], projection, [2.0, 3.0])
        == 0.0
    )


def test_modal_reconstruction_diagnostics_balance_e_and_h_components():
    empty = _modal_projection_reconstruction_diagnostics_from_matrix(
        [1.0], np.zeros((0, 0)), [1.0]
    )
    assert np.isnan(empty["residual"])

    matrix = np.eye(4, dtype=complex)
    diagnostics = _modal_projection_reconstruction_diagnostics_from_matrix(
        [2.0, 3.0, 4.0, 5.0],
        matrix,
        [1.0, 1.0, 1.0, 1.0],
        component_slices=(
            ("Ex", 0, 2),
            ("Hx", 2, 4),
            ("ignored", 4, 4),
        ),
    )

    assert diagnostics["residual"] > 0.0
    assert diagnostics["residual_e_scaled"] > 0.0
    assert diagnostics["residual_h_scaled"] > 0.0
    assert diagnostics["residual_balanced"] < diagnostics["residual"]
