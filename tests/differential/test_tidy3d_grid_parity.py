"""Behavioral parity checks against pinned Tidy3D automatic grids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import beamz as bz
from tests.differential.tidy3d_grid_adapter import (
    TIDY3D_REFERENCE_VERSION,
    beamz_grid,
    normalized_spacing_profile,
    parity_cases,
)
from tests.validation.tolerances import Tolerance

REFERENCE_PATH = Path(__file__).with_name("tidy3d_grid_references.json")
REFERENCE = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
CASES = parity_cases()
CELL_COUNT_PARITY = Tolerance(
    name="tidy3d_grid_cell_count",
    absolute=0.0,
    relative=0.04,
    rationale=(
        "Independent graded-grid edge placement may differ by a few cells while "
        "retaining equivalent resolution and computational cost."
    ),
)
PROFILE_ERROR_LIMIT = 0.08


def _reference_edges(case_name: str, axis: str) -> np.ndarray:
    return np.asarray(REFERENCE["cases"][case_name][axis], dtype=np.float64)


def _max_adjacent_ratio(edges: np.ndarray) -> float:
    widths = np.diff(edges)
    if widths.size < 2:
        return 1.0
    return float(
        max(np.max(widths[1:] / widths[:-1]), np.max(widths[:-1] / widths[1:]))
    )


def test_tidy3d_reference_manifest_is_pinned_and_complete():
    assert REFERENCE["schema"] == "beamz.tidy3d-grid-reference/v1"
    assert REFERENCE["tidy3d_version"] == TIDY3D_REFERENCE_VERSION
    assert REFERENCE["length_unit"] == "um"
    assert set(REFERENCE["cases"]) == {case.name for case in CASES}


@pytest.mark.parametrize(
    ("case", "axis"),
    [(case, axis) for case in CASES for axis in "xy"],
    ids=[f"{case.name}-{axis}" for case in CASES for axis in "xy"],
)
def test_automatic_grid_has_tidy3d_cell_count_and_spacing_profile(
    case, axis, validation_metrics
):
    beamz_edges = (
        np.asarray(getattr(beamz_grid(case), f"{axis}_edges"), dtype=np.float64) / bz.um
    )
    tidy3d_edges = _reference_edges(case.name, axis)

    np.testing.assert_allclose(
        beamz_edges[[0, -1]], tidy3d_edges[[0, -1]], rtol=0.0, atol=1e-12
    )
    validation_metrics.check(
        f"{axis}-axis cell count",
        beamz_edges.size - 1,
        tidy3d_edges.size - 1,
        tolerance=CELL_COUNT_PARITY,
        unit="cells",
        backend="beamz-vs-tidy3d-2.12.0",
    )

    beamz_profile = normalized_spacing_profile(beamz_edges)
    tidy3d_profile = normalized_spacing_profile(tidy3d_edges)
    profile_error = float(
        np.mean(np.abs(beamz_profile - tidy3d_profile)) / np.mean(tidy3d_profile)
    )
    validation_metrics.check_upper(
        f"{axis}-axis normalized spacing-profile error",
        profile_error,
        PROFILE_ERROR_LIMIT,
        unit="relative L1",
        backend="beamz-vs-tidy3d-2.12.0",
        metadata={"samples": int(beamz_profile.size)},
    )

    assert _max_adjacent_ratio(beamz_edges) <= case.max_scale * (1.0 + 1e-12)
    assert _max_adjacent_ratio(tidy3d_edges) <= case.max_scale * (1.0 + 1e-12)


def test_explicit_snapping_points_are_grid_edges_in_both_meshers():
    case = next(case for case in CASES if case.name == "snapped_rectangle")
    grid = beamz_grid(case)
    for point in case.snapping_points:
        for axis_index, axis in enumerate("xy"):
            coordinate = point[axis_index]
            if coordinate is None:
                continue
            beamz_edges = np.asarray(getattr(grid, f"{axis}_edges")) / bz.um
            tidy3d_edges = _reference_edges(case.name, axis)
            assert np.any(np.isclose(beamz_edges, coordinate, rtol=0.0, atol=1e-12))
            assert np.any(np.isclose(tidy3d_edges, coordinate, rtol=0.0, atol=1e-12))
