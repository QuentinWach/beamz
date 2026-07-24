from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.differential.case_schema import (
    ORACLE_PRIORITY,
    SCHEMA_VERSION,
    DifferentialCase,
    load_case,
)
from tests.validation.tolerances import TOLERANCES

CASES = tuple(
    sorted((Path(__file__).parents[1] / "differential" / "cases").glob("*.json"))
)


def test_solver_neutral_case_catalog_is_valid_and_unique():
    cases = [load_case(path) for path in CASES]

    assert cases
    assert len({case.name for case in cases}) == len(cases)
    assert ORACLE_PRIORITY == {
        "analytical": 0,
        "invariant": 1,
        "solver_consensus": 2,
        "regression": 3,
    }
    for case in cases:
        assert case.resolution_m < case.wavelength_m
        assert all(item.tolerance in TOLERANCES for item in case.observables)


def test_case_schema_rejects_unknown_root_keys():
    payload = json.loads(CASES[0].read_text(encoding="utf-8"))
    payload["solver_specific_array_layout"] = "forbidden"

    with pytest.raises(ValueError, match="case keys"):
        DifferentialCase.from_dict(payload)


def test_case_schema_rejects_consensus_without_beamz_adapter():
    payload = json.loads(CASES[0].read_text(encoding="utf-8"))
    payload["adapters"] = ["meep", "fdtdx"]

    with pytest.raises(ValueError, match="include 'beamz'"):
        DifferentialCase.from_dict(payload)


def test_case_schema_version_is_explicit():
    assert SCHEMA_VERSION == "beamz.differential/v1"
