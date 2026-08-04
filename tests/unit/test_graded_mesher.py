from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from beamz.design.grid import RectilinearGrid
from beamz.design.mesher import _GradedMesher


def _max_ratio(edges: np.ndarray) -> float:
    widths = np.diff(edges)
    if widths.size < 2:
        return 1.0
    return float(np.max(np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])))


def test_graded_mesher_preserves_constraints_and_smooths_large_step_change():
    coords = np.asarray([0.0, 4.0, 6.0, 10.0])
    limits = np.asarray([1.0, 0.2, 1.0])

    edges = _GradedMesher(max_scale=1.2).make_axis_edges(coords, limits)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    local_limits = np.select(
        (centers < 4.0, centers < 6.0), (limits[0], limits[1]), default=limits[2]
    )

    np.testing.assert_allclose(edges[[0, -1]], coords[[0, -1]])
    for coordinate in coords[1:-1]:
        assert np.any(edges == coordinate)
    assert np.all(widths <= local_limits * (1.0 + 1e-12))
    assert _max_ratio(edges) <= 1.2 * (1.0 + 1e-12)
    assert np.any((widths > 0.2) & (widths < 0.9))


def test_graded_mesher_propagates_a_snapped_sliver_into_neighboring_cells():
    coords = np.asarray([0.0, 1.0, 1.01, 2.0])
    edges = _GradedMesher(max_scale=1.25).make_axis_edges(coords, [0.2, 0.2, 0.2])

    assert np.any(edges == 1.0)
    assert np.any(edges == 1.01)
    assert _max_ratio(edges) <= 1.25 * (1.0 + 1e-12)


def test_short_mandatory_interval_is_not_split_into_a_new_sliver():
    coordinates = np.asarray([0.0, 2.0, 2.02163663779975054])
    limits = np.asarray([0.021484375, 0.5])

    edges = _GradedMesher(max_scale=1.125).make_axis_edges(coordinates, limits)

    assert np.count_nonzero(edges >= coordinates[-2]) == 2
    assert _max_ratio(edges) <= 1.125 * (1.0 + 1e-12)


def test_constrained_terminal_interval_grades_into_long_neighbor():
    coordinates = np.asarray([0.0, 1.0, 3.0, 3.021484375])
    limits = np.asarray([0.5, 0.0234375, 0.02])

    edges = _GradedMesher(max_scale=1.0625).make_axis_edges(coordinates, limits)
    widths = np.diff(edges)
    owners = np.searchsorted(coordinates[1:], 0.5 * (edges[:-1] + edges[1:]))

    assert np.all(widths <= limits[owners] * (1.0 + 1e-12))
    assert _max_ratio(edges) <= 1.0625 * (1.0 + 1e-12)


def test_graded_mesher_is_deterministic_and_validates_inputs():
    mesher = _GradedMesher(max_scale=1.3)
    first = mesher.make_axis_edges([0.0, 1.0, 3.0], [0.1, 0.5])
    second = mesher.make_axis_edges([0.0, 1.0, 3.0], [0.1, 0.5])

    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="strictly increasing"):
        mesher.make_axis_edges([0.0, 1.0, 1.0], [0.1, 0.2])
    with pytest.raises(ValueError, match="one value"):
        mesher.make_axis_edges([0.0, 1.0, 2.0], [0.1])
    with pytest.raises(ValueError, match="between 1 and 2"):
        _GradedMesher(max_scale=2.0)


def test_graded_mesher_budget_fails_before_realization_allocation(monkeypatch):
    mesher = _GradedMesher(max_scale=1.2)

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("axis realization allocation must not start")

    monkeypatch.setattr(_GradedMesher, "_realize", unexpected_allocation)

    with pytest.raises(ValueError, match="x axis before allocation") as error:
        mesher.make_axis_edges(
            [0.0, 1.0],
            [1e-9],
            max_cells=100,
            axis_name="x",
        )

    message = str(error.value)
    assert "temporary edge/owner arrays" in message
    assert "smallest requested spacing" in message
    assert "[0, 1]" in message


def test_grid_quality_report_identifies_worst_adjacent_pair():
    grid = RectilinearGrid(
        np.asarray([0.0, 0.125, 0.375, 0.875]),
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([0.0, 1.0]),
    )

    report = grid.quality_report()

    assert report.x.cell_count == 3
    assert report.x.minimum_spacing == pytest.approx(0.125)
    assert report.x.maximum_spacing == pytest.approx(0.5)
    assert report.x.max_adjacent_ratio == pytest.approx(2.0)
    assert report.x.worst_pair_index == 0
    assert report.y.max_adjacent_ratio == 1.0
    assert report.max_adjacent_ratio == pytest.approx(2.0)
    assert report.satisfies_max_scale(2.0, active_axes=("x", "y"))
    assert not report.satisfies_max_scale(1.5, active_axes=("x", "y"))


@settings(max_examples=500, deadline=None)
@given(
    interval_lengths=st.lists(
        st.floats(min_value=0.02, max_value=2.0, allow_nan=False),
        min_size=1,
        max_size=8,
    ),
    interval_limits=st.data(),
    max_scale=st.floats(min_value=1.05, max_value=1.5, allow_nan=False),
)
def test_graded_mesher_randomized_quality_invariants(
    interval_lengths: list[float],
    interval_limits: st.DataObject,
    max_scale: float,
):
    """Every generated axis must honor hard boundaries, caps, and grading."""
    limits = np.asarray(
        interval_limits.draw(
            st.lists(
                st.floats(min_value=0.02, max_value=0.6, allow_nan=False),
                min_size=len(interval_lengths),
                max_size=len(interval_lengths),
            )
        )
    )
    coordinates = np.concatenate(([0.0], np.cumsum(interval_lengths)))

    edges = _GradedMesher(max_scale=max_scale).make_axis_edges(coordinates, limits)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    interval_indices = np.searchsorted(coordinates[1:], centers, side="right")
    interval_indices = np.minimum(interval_indices, limits.size - 1)

    assert np.all(np.diff(edges) > 0.0)
    assert all(np.any(edges == coordinate) for coordinate in coordinates)
    assert np.all(widths <= limits[interval_indices] * (1.0 + 1e-11))
    assert _max_ratio(edges) <= max_scale * (1.0 + 1e-11)
