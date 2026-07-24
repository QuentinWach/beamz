from __future__ import annotations

import pytest

from scripts.check_coverage_policy import evaluate_policy


def _summary(*, covered_lines, statements, covered_branches, branches):
    return {
        "covered_lines": covered_lines,
        "num_statements": statements,
        "covered_branches": covered_branches,
        "num_branches": branches,
    }


def test_coverage_policy_combines_statements_and_branch_outcomes():
    coverage = {
        "totals": _summary(
            covered_lines=80,
            statements=100,
            covered_branches=20,
            branches=40,
        ),
        "files": {
            "beamz/core.py": {
                "summary": _summary(
                    covered_lines=18,
                    statements=20,
                    covered_branches=8,
                    branches=10,
                )
            }
        },
    }
    policy = {
        "global_minimum": 70.0,
        "groups": {
            "core": {"minimum": 85.0, "files": ["beamz/core.py"]},
        },
    }

    global_result, core_result = evaluate_policy(coverage, policy)

    assert global_result.percent == pytest.approx(100 * 100 / 140)
    assert global_result.passes
    assert core_result.percent == pytest.approx(100 * 26 / 30)
    assert core_result.passes


def test_coverage_policy_reports_threshold_failures_without_rounding_them_up():
    coverage = {
        "totals": _summary(
            covered_lines=79,
            statements=100,
            covered_branches=0,
            branches=0,
        ),
        "files": {},
    }
    policy = {"global_minimum": 80.0, "groups": {}}

    (result,) = evaluate_policy(coverage, policy)

    assert result.percent == 79.0
    assert not result.passes


def test_coverage_policy_rejects_stale_group_paths():
    coverage = {
        "totals": _summary(
            covered_lines=1,
            statements=1,
            covered_branches=0,
            branches=0,
        ),
        "files": {},
    }
    policy = {
        "global_minimum": 0.0,
        "groups": {"core": {"minimum": 0.0, "files": ["beamz/missing.py"]}},
    }

    with pytest.raises(ValueError, match="references missing file"):
        evaluate_policy(coverage, policy)


def test_coverage_policy_rejects_empty_measurement_sets():
    empty = _summary(
        covered_lines=0,
        statements=0,
        covered_branches=0,
        branches=0,
    )
    coverage = {"totals": empty, "files": {}}
    policy = {"global_minimum": 0.0, "groups": {}}

    with pytest.raises(ValueError, match="no measurable"):
        evaluate_policy(coverage, policy)
