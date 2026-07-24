import json

import pytest

from tests.validation.metrics import (
    SCHEMA_VERSION,
    EvidenceCaseResult,
    ValidationMetric,
    ValidationRecorder,
    validation_report,
    write_validation_report,
)
from tests.validation.tolerances import TOLERANCES, Tolerance, get_tolerance


def test_named_tolerances_are_reviewable_and_immutable():
    assert {
        "exact",
        "kernel_float32",
        "kernel_float64",
        "analytical_coarse",
        "analytical_fine",
        "normalized_power_balance",
        "second_order_convergence",
        "waveguide_neff",
        "sparameter_reciprocity",
        "gradient_float32",
        "gradient_float64",
        "cross_solver",
        "material_wavefront_coarse",
        "material_wavelength_coarse",
        "strict_bound",
    } == set(TOLERANCES)
    assert all(tolerance.rationale for tolerance in TOLERANCES.values())

    with pytest.raises(TypeError):
        TOLERANCES["anything"] = TOLERANCES["exact"]


def test_unknown_tolerance_lists_valid_names():
    with pytest.raises(KeyError, match="analytical_coarse"):
        get_tolerance("wishful")


def test_tolerance_allows_strict_zero_slack_but_requires_a_rationale():
    strict = Tolerance("strict", 0.0, 0.0, "an exact inequality")
    assert strict.error_limit(1e-15) == 0.0
    with pytest.raises(ValueError, match="rationale"):
        Tolerance("mystery", 1.0, 0.0, "")


def test_validation_metric_reports_absolute_and_relative_error():
    metric = ValidationMetric(
        case_id="fresnel_air_glass",
        quantity="reflectance",
        measured=0.041,
        reference=0.04,
        tolerance=get_tolerance("analytical_coarse"),
    )

    assert metric.absolute_error == pytest.approx(0.001)
    assert metric.relative_error == pytest.approx(0.025)
    assert metric.error_limit == pytest.approx(0.002)
    assert metric.passed
    assert "rel_error=2.500%" in metric.diagnostic()


def test_validation_recorder_preserves_failed_measurement_before_asserting():
    recorded = []
    recorder = ValidationRecorder("case", recorded)

    with pytest.raises(AssertionError, match="measured=0.2"):
        recorder.check(
            "power balance",
            0.2,
            0.0,
            tolerance="normalized_power_balance",
            backend="cpu",
        )

    assert len(recorded) == 1
    assert not recorded[0].passed


def test_validation_recorder_supports_honest_upper_and_lower_bounds():
    recorded = []
    recorder = ValidationRecorder("cpml", recorded)

    upper = recorder.check_upper("reflection", -52.0, -40.0, unit="dB")
    lower = recorder.check_lower("mode overlap", 0.995, 0.99)

    assert upper.passed
    assert upper.margin == pytest.approx(12.0)
    assert upper.comparison == "less_equal"
    assert upper.as_dict()["comparison"] == "less_equal"
    assert lower.passed
    assert lower.margin == pytest.approx(0.005)
    assert lower.comparison == "greater_equal"
    with pytest.raises(AssertionError, match="upper_bound=-40"):
        recorder.check_upper("bad reflection", -35.0, -40.0, unit="dB")
    with pytest.raises(AssertionError, match="upper_bound=5e-16"):
        recorder.check_upper("group delay", 6e-16, 5e-16, unit="s")


def test_validation_metric_rejects_unknown_comparison():
    with pytest.raises(ValueError, match="comparison"):
        ValidationMetric(
            case_id="case",
            quantity="quantity",
            measured=1.0,
            reference=1.0,
            tolerance=get_tolerance("exact"),
            comparison="approximately-ish",
        )


def test_validation_report_is_json_serializable_and_inventories_cases(tmp_path):
    metric = ValidationMetric(
        case_id="kernel",
        quantity="adjoint residual",
        measured=1e-7,
        reference=0.0,
        tolerance=get_tolerance("kernel_float32"),
        backend="cpu",
        metadata={"seed": 7},
    )
    cases = (
        EvidenceCaseResult("kernel", "invariant", "passed"),
        EvidenceCaseResult("material", "validation", "passed"),
    )
    report = validation_report(
        [metric],
        exit_status=0,
        random_seed="123",
        cases=cases,
    )
    destination = tmp_path / "nested" / "validation.json"

    write_validation_report(report, destination)
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["summary"] == {
        "cases": 2,
        "exit_status": 0,
        "failed": 0,
        "failed_cases": 0,
        "metrics": 1,
        "metricless_cases": 1,
        "passed": 1,
        "passed_cases": 2,
        "skipped_cases": 0,
        "xfailed_cases": 0,
    }
    assert loaded["cases"] == [
        {
            "case_id": "kernel",
            "evidence_class": "invariant",
            "metrics": 1,
            "outcome": "passed",
        },
        {
            "case_id": "material",
            "evidence_class": "validation",
            "metrics": 0,
            "outcome": "passed",
        },
    ]
    assert loaded["metricless_case_ids"] == ["material"]
    assert loaded["metrics"][0]["metadata"] == {"seed": 7}
    assert loaded["metrics"][0]["relative_error"] is None


def test_validation_metric_rejects_non_json_metadata():
    with pytest.raises(ValueError, match="finite JSON"):
        ValidationMetric(
            case_id="case",
            quantity="quantity",
            measured=1.0,
            reference=1.0,
            tolerance=get_tolerance("exact"),
            metadata={"not-portable": float("nan")},
        )
