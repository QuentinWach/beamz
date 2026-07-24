"""Pytest integration for structured BeamZ validation metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.validation.metrics import (
    ValidationMetric,
    ValidationRecorder,
    validation_report,
    write_validation_report,
)

_METRICS_KEY = pytest.StashKey[list[ValidationMetric]]()


def pytest_addoption(parser) -> None:
    group = parser.getgroup("beamz validation")
    group.addoption(
        "--validation-report",
        metavar="PATH",
        help="write structured BeamZ validation measurements to PATH as JSON",
    )


def pytest_configure(config) -> None:
    config.stash[_METRICS_KEY] = []


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
    report = validation_report(
        session.config.stash[_METRICS_KEY],
        exit_status=exitstatus,
        random_seed=seed,
    )
    write_validation_report(report, Path(destination))
