from __future__ import annotations

import argparse
import html
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a simple SVG coverage badge from coverage.xml."
    )
    parser.add_argument("coverage_xml", type=Path)
    parser.add_argument("output_svg", type=Path)
    return parser.parse_args()


def pick_color(percent: int) -> str:
    if percent >= 90:
        return "#2ea44f"
    if percent >= 80:
        return "#97CA00"
    if percent >= 70:
        return "#a4a61d"
    if percent >= 60:
        return "#dfb317"
    if percent >= 50:
        return "#fe7d37"
    return "#e05d44"


def text_width(text: str) -> int:
    return max(1, len(text)) * 7 + 10


def read_coverage_percent(path: Path) -> int:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    line_rate = float(root.attrib["line-rate"])
    return int(round(line_rate * 100))


def render_badge(label: str, value: str, color: str) -> str:
    label_width = text_width(label)
    value_width = text_width(value)
    total_width = label_width + value_width
    label_x = label_width / 2
    value_x = label_width + value_width / 2

    label = html.escape(label)
    value = html.escape(value)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{label}: {value}">
<linearGradient id="smooth" x2="0" y2="100%">
  <stop offset="0" stop-color="#fff" stop-opacity=".7"/>
  <stop offset=".1" stop-color="#aaa" stop-opacity=".1"/>
  <stop offset=".9" stop-color="#000" stop-opacity=".3"/>
  <stop offset="1" stop-color="#000" stop-opacity=".5"/>
</linearGradient>
<clipPath id="round">
  <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
</clipPath>
<g clip-path="url(#round)">
  <rect width="{label_width}" height="20" fill="#555"/>
  <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
  <rect width="{total_width}" height="20" fill="url(#smooth)"/>
</g>
<g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
  <text x="{label_x}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
  <text x="{label_x}" y="14">{label}</text>
  <text x="{value_x}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
  <text x="{value_x}" y="14">{value}</text>
</g>
</svg>
"""


def main() -> None:
    args = parse_args()
    percent = read_coverage_percent(args.coverage_xml)
    svg = render_badge("coverage", f"{percent}%", pick_color(percent))
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
