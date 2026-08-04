from __future__ import annotations

from types import MappingProxyType

import pytest

from beamz.devices._placement import SnappedInterval, SnappedRegion


def test_snapped_interval_snapshots_rectilinear_edges():
    edges = [0.0, 0.2, 1.0]
    interval = SnappedInterval(0, 2, 0.2, edges)  # type: ignore[arg-type]

    edges[1] = 0.5

    assert interval.edges == (0.0, 0.2, 1.0)


def test_snapped_region_snapshots_and_freezes_intervals():
    intervals = {"X": SnappedInterval(0, 2, 0.2)}
    region = SnappedRegion(2, "Y", 1, 0.3, intervals)

    intervals.clear()

    assert isinstance(region.intervals, MappingProxyType)
    assert region.normal_axis == "y"
    assert region.axis_interval("x") == SnappedInterval(0, 2, 0.2)
    with pytest.raises(TypeError):
        region.intervals["x"] = SnappedInterval(1, 2, 0.2)  # type: ignore[index]


def test_snapped_region_rejects_invalid_interval_values():
    with pytest.raises(TypeError, match="SnappedInterval"):
        SnappedRegion(2, "y", 1, 0.3, {"x": object()})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="axis"):
        SnappedRegion(2, "y", 1, 0.3, {"time": SnappedInterval(0, 1, 0.2)})
