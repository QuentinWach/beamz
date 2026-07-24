"""Pytest integration for structured BeamZ validation metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.validation.metrics import (
    EvidenceCaseResult,
    ValidationMetric,
    ValidationRecorder,
    validation_report,
    write_validation_report,
)

_METRICS_KEY = pytest.StashKey[list[ValidationMetric]]()
_CASES_KEY = pytest.StashKey[dict[str, dict[str, str]]]()
_ACTIVE_CASES: dict[str, dict[str, str]] = {}


def pytest_addoption(parser) -> None:
    group = parser.getgroup("beamz validation")
    group.addoption(
        "--validation-report",
        metavar="PATH",
        help="write structured BeamZ validation measurements to PATH as JSON",
    )


def pytest_configure(config) -> None:
    global _ACTIVE_CASES
    config.stash[_METRICS_KEY] = []
    config.stash[_CASES_KEY] = {}
    _ACTIVE_CASES = config.stash[_CASES_KEY]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item) -> None:
    """Inventory every selected validation-directory case before it executes."""
    tests_root = Path(__file__).resolve().parents[1]
    relative = Path(item.path).resolve().relative_to(tests_root)
    if relative.parts[0] not in {"validation", "differential"}:
        return
    evidence_class = (
        "invariant"
        if relative.parts[0] == "validation" and relative.parts[1] == "invariants"
        else "validation"
    )
    _ACTIVE_CASES[item.nodeid] = {
        "evidence_class": evidence_class,
        "outcome": "skipped",
    }


def pytest_runtest_logreport(report) -> None:
    """Retain the final visible outcome for each inventoried evidence case."""
    case = _ACTIVE_CASES.get(report.nodeid)
    if case is None:
        return
    if report.failed:
        case["outcome"] = "failed"
    elif report.when == "call":
        case["outcome"] = (
            "xfailed"
            if report.skipped and hasattr(report, "wasxfail")
            else report.outcome
        )


@pytest.fixture
def validation_metrics(request) -> ValidationRecorder:
    """Record scalar validation evidence owned by the requesting test."""
    return ValidationRecorder(
        case_id=request.node.nodeid,
        sink=request.config.stash[_METRICS_KEY],
    )


def pytest_sessionfinish(session, exitstatus) -> None:
    destination = session.config.getoption("validation_report")
    if not destination:
        return
    seed = str(getattr(session.config.option, "randomly_seed", "unavailable"))
    cases = tuple(
        EvidenceCaseResult(
            case_id=case_id,
            evidence_class=payload["evidence_class"],
            outcome=payload["outcome"],
        )
        for case_id, payload in sorted(session.config.stash[_CASES_KEY].items())
    )
    report = validation_report(
        session.config.stash[_METRICS_KEY],
        exit_status=exitstatus,
        random_seed=seed,
        cases=cases,
    )
    write_validation_report(report, Path(destination))
