"""Render a portable BeamZ validation JSON report as standalone HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "beamz.validation/v2"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude < 1e-4 or magnitude >= 1e5):
        return f"{number:.5e}"
    return f"{number:.8g}"


def _comparison_label(metric: Mapping[str, Any]) -> str:
    return {
        "close": "reference",
        "less_equal": "upper bound",
        "greater_equal": "lower bound",
    }.get(str(metric.get("comparison", "close")), "reference")


def _scale(
    value: float,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> float:
    fraction = (value - source_min) / max(source_max - source_min, 1e-30)
    return target_min + fraction * (target_max - target_min)


def _convergence_chart(metrics: Sequence[Mapping[str, Any]]) -> str:
    for metric in metrics:
        metadata = metric.get("metadata", {})
        sizes = metadata.get("grid_sizes") if isinstance(metadata, Mapping) else None
        errors = metadata.get("l2_errors") if isinstance(metadata, Mapping) else None
        if (
            not isinstance(sizes, list)
            or not isinstance(errors, list)
            or len(sizes) != len(errors)
            or len(sizes) < 3
        ):
            continue
        points = [
            (float(size), float(error))
            for size, error in zip(sizes, errors, strict=True)
        ]
        if any(size <= 0.0 or error <= 0.0 for size, error in points):
            continue
        width, height = 640.0, 250.0
        left, right, top, bottom = 70.0, 25.0, 25.0, 45.0
        xs = [math.log2(size) for size, _ in points]
        ys = [math.log10(error) for _, error in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        plot_x = [_scale(value, xmin, xmax, left, width - right) for value in xs]
        plot_y = [_scale(value, ymin, ymax, height - bottom, top) for value in ys]

        polyline = " ".join(
            f"{x:.2f},{y:.2f}" for x, y in zip(plot_x, plot_y, strict=True)
        )
        labels = "".join(
            (
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5"/>'
                f'<text x="{x:.2f}" y="{height - 18:.2f}" '
                f'text-anchor="middle">{_escape(int(size))}³</text>'
                f'<text x="{x + 8:.2f}" y="{y - 9:.2f}">'
                f"{_escape(f'{error:.3g}')}</text>"
            )
            for (size, error), x, y in zip(points, plot_x, plot_y, strict=True)
        )
        return f"""
        <section class="panel chart-panel">
          <h2>Measured grid convergence</h2>
          <p>Complete 3-D Yee curl L2 error; both factor-two refinement rates
             are recorded as validation metrics.</p>
          <svg viewBox="0 0 {width:.0f} {height:.0f}" role="img"
               aria-label="Yee curl L2 error decreases over three grid refinements">
            <line class="axis" x1="{left}" y1="{height - bottom}"
                  x2="{width - right}" y2="{height - bottom}"/>
            <line class="axis" x1="{left}" y1="{top}"
                  x2="{left}" y2="{height - bottom}"/>
            <polyline points="{polyline}"/>
            {labels}
            <text class="axis-label" x="{width / 2}" y="{height - 2}"
                  text-anchor="middle">material cells per axis</text>
            <text class="axis-label" x="15" y="{height / 2}"
                  transform="rotate(-90 15 {height / 2})"
                  text-anchor="middle">L2 error (log scale)</text>
          </svg>
        </section>"""
    return ""


def render_report(report: Mapping[str, Any]) -> str:
    """Validate and render one complete validation report."""
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"expected {SCHEMA_VERSION!r}, got {report.get('schema_version')!r}"
        )
    raw_metrics = report.get("metrics")
    if not isinstance(raw_metrics, list):
        raise ValueError("validation report metrics must be a list")
    metrics = [metric for metric in raw_metrics if isinstance(metric, Mapping)]
    if len(metrics) != len(raw_metrics):
        raise ValueError("every validation metric must be an object")

    summary = report.get("summary", {})
    environment = report.get("environment", {})
    cases = len({str(metric.get("case_id", "")) for metric in metrics})
    passed = int(summary.get("passed", 0))
    failed = int(summary.get("failed", 0))
    status = (
        "PASS" if failed == 0 and int(summary.get("exit_status", 1)) == 0 else "FAIL"
    )
    status_class = "pass" if status == "PASS" else "fail"

    rows = []
    for metric in metrics:
        tolerance = metric.get("tolerance", {})
        tolerance_name = (
            tolerance.get("name", "—") if isinstance(tolerance, Mapping) else "—"
        )
        row_class = "metric-pass" if metric.get("passed") else "metric-fail"
        reference_label = _comparison_label(metric)
        rows.append(
            f"""
            <tr class="{row_class}">
              <td><span class="status-dot" aria-hidden="true"></span>
                  {_escape("pass" if metric.get("passed") else "fail")}</td>
              <td><code>{_escape(metric.get("case_id", ""))}</code></td>
              <td>{_escape(metric.get("quantity", ""))}</td>
              <td class="number">{_escape(_number(metric.get("measured")))}</td>
              <td><span class="subtle">{_escape(reference_label)}</span><br>
                  <span class="number">{_escape(_number(metric.get("reference")))}</span></td>
              <td class="number">{_escape(_number(metric.get("absolute_error")))}</td>
              <td class="number">{_escape(_number(metric.get("relative_error")))}</td>
              <td class="number">{_escape(_number(metric.get("margin")))}</td>
              <td><code>{_escape(tolerance_name)}</code></td>
              <td>{_escape(metric.get("resolution", ""))}</td>
              <td>{_escape(metric.get("backend", ""))}</td>
            </tr>"""
        )
    table_body = "".join(rows) or (
        '<tr><td colspan="11" class="empty">No validation metrics were recorded.</td></tr>'
    )
    chart = _convergence_chart(metrics)
    generated_at = report.get("generated_at", "unknown")
    commit = report.get("beamz_commit", "unknown")
    environment_items = "".join(
        f"<dt>{_escape(key)}</dt><dd>{_escape(value)}</dd>"
        for key, value in sorted(environment.items())
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BeamZ validation evidence</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #08111f; --panel: #101d30; --line: #2a3a51;
      --text: #edf5ff; --muted: #9db0c8; --good: #46d39a;
      --bad: #ff6f7d; --accent: #72a7ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text);
            font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; }}
    main {{ width: min(1500px, calc(100% - 32px)); margin: 32px auto 64px; }}
    h1 {{ margin: 0; font-size: clamp(28px, 5vw, 48px); letter-spacing: -.04em; }}
    h2 {{ margin: 0 0 8px; font-size: 20px; }}
    p {{ color: var(--muted); }}
    code {{ font-size: 12px; }}
    .eyebrow {{ color: var(--accent); font-weight: 700; letter-spacing: .12em;
                text-transform: uppercase; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr));
              gap: 12px; margin: 24px 0; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line);
                     border-radius: 14px; }}
    .card {{ padding: 18px; }}
    .card strong {{ display: block; font-size: 30px; }}
    .card span {{ color: var(--muted); }}
    .status.pass {{ color: var(--good); }} .status.fail {{ color: var(--bad); }}
    .panel {{ margin-top: 16px; padding: 20px; overflow: hidden; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 5px 14px; }}
    dt {{ color: var(--muted); }} dd {{ margin: 0; overflow-wrap: anywhere; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1260px; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
          letter-spacing: .08em; text-align: left; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px;
              vertical-align: top; }}
    td.number, .number {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .subtle {{ color: var(--muted); font-size: 11px; }}
    .status-dot {{ display: inline-block; width: 8px; height: 8px;
                   border-radius: 50%; margin-right: 5px; background: var(--bad); }}
    .metric-pass .status-dot {{ background: var(--good); }}
    .metric-fail {{ background: color-mix(in srgb, var(--bad) 8%, transparent); }}
    svg {{ width: 100%; max-width: 760px; color: var(--muted); }}
    svg polyline {{ fill: none; stroke: var(--accent); stroke-width: 3; }}
    svg circle {{ fill: var(--good); }} svg text {{ fill: currentColor; font-size: 12px; }}
    svg .axis {{ stroke: var(--line); }} svg .axis-label {{ fill: var(--muted); }}
    footer {{ margin-top: 20px; color: var(--muted); }}
    @media (max-width: 760px) {{
      .cards {{ grid-template-columns: repeat(2, 1fr); }}
      main {{ width: min(100% - 20px, 1500px); margin-top: 20px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Scientific trust artifact</div>
    <h1>BeamZ validation evidence</h1>
    <p>Measured observables, independent references, named tolerances, and
       execution provenance. Generated {_escape(generated_at)}.</p>
  </header>
  <section class="cards" aria-label="Validation summary">
    <div class="card"><strong class="status {status_class}">{status}</strong>
      <span>suite status</span></div>
    <div class="card"><strong>{len(metrics)}</strong><span>metrics recorded</span></div>
    <div class="card"><strong>{passed}</strong><span>metrics passing</span></div>
    <div class="card"><strong>{cases}</strong><span>validation cases</span></div>
  </section>
  {chart}
  <section class="panel">
    <h2>Reproducibility</h2>
    <dl><dt>BeamZ commit</dt><dd><code>{_escape(commit)}</code></dd>
        {environment_items}</dl>
  </section>
  <section class="panel">
    <h2>All measurements</h2>
    <p>Margins are signed distances to the applicable equality or bound gate;
       non-negative values pass.</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Status</th><th>Case</th><th>Quantity</th>
          <th>Measured</th><th>Reference / bound</th><th>Absolute error</th>
          <th>Relative error</th><th>Margin</th><th>Tolerance</th>
          <th>Resolution</th><th>Backend</th></tr></thead>
        <tbody>{table_body}</tbody>
      </table>
    </div>
  </section>
  <footer>Schema {_escape(SCHEMA_VERSION)} · static, dependency-free artifact</footer>
</main>
</body>
</html>
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="validation JSON produced by pytest")
    parser.add_argument("output", type=Path, help="standalone HTML destination")
    args = parser.parse_args(argv)

    report = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("validation report root must be a JSON object")
    rendered = render_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
