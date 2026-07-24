from __future__ import annotations

import json

import pytest

from scripts.render_validation_report import main, render_report
from tests.validation.metrics import ValidationMetric, validation_report
from tests.validation.tolerances import get_tolerance


def _sample_report():
    convergence = ValidationMetric(
        case_id="curl<script>",
        quantity="second-order convergence",
        measured=1.996,
        reference=2.0,
        tolerance=get_tolerance("second_order_convergence"),
        metadata={
            "grid_sizes": [12, 24, 48],
            "l2_errors": [0.05, 0.0125, 0.003125],
        },
    )
    threshold = ValidationMetric(
        case_id="cpml",
        quantity="reflection",
        measured=-52.0,
        reference=-40.0,
        tolerance=get_tolerance("exact"),
        unit="dB",
        comparison="less_equal",
    )
    return validation_report(
        [convergence, threshold],
        exit_status=0,
        random_seed="123",
    )


def test_html_report_renders_metrics_bounds_and_convergence_chart():
    rendered = render_report(_sample_report())

    assert "<!doctype html>" in rendered
    assert "2</strong><span>metrics recorded" in rendered
    assert "upper bound" in rendered
    assert "Measured grid convergence" in rendered
    assert '<svg viewBox="0 0 640 250"' in rendered
    assert "curl&lt;script&gt;" in rendered
    assert "curl<script>" not in rendered


def test_html_report_rejects_unknown_schema():
    report = _sample_report()
    report["schema_version"] = "wishful/v99"

    with pytest.raises(ValueError, match="beamz.validation/v2"):
        render_report(report)


def test_html_report_cli_writes_standalone_artifact(tmp_path):
    source = tmp_path / "validation.json"
    destination = tmp_path / "site" / "index.html"
    source.write_text(json.dumps(_sample_report()), encoding="utf-8")

    assert main([str(source), str(destination)]) == 0
    assert destination.read_text(encoding="utf-8").endswith("</html>\n")
