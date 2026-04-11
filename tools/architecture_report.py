#!/usr/bin/env python3
"""Generate a browser-friendly architecture report for the BEAMZ package."""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "beamz"
OUTPUT = ROOT / "docs" / "architecture" / "index.html"
AUX_PARTS = {"examples", "tests"}
FACADE_BASENAMES = {"core", "compiled", "meshing", "mode", "monitors"}
STDLIB_IMPORT_ROOTS = frozenset(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


@dataclass
class ModuleInfo:
    name: str
    path: Path
    rel_path: str
    package: str
    group: str
    total_lines: int
    code_lines: int
    top_level_functions: int
    top_level_classes: int
    internal_imports: list[str]
    external_import_roots: list[str]
    is_auxiliary: bool


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def package_group(module: str) -> str:
    prefixes = [
        "beamz.visual.scene",
        "beamz.devices.sources",
        "beamz.devices.monitors",
        "beamz.design",
        "beamz.optimization",
        "beamz.simulation",
        "beamz.visual",
        "beamz.devices",
        "beamz",
    ]
    for prefix in prefixes:
        if module == prefix or module.startswith(prefix + "."):
            return prefix
    return "beamz"


def module_name_for(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def is_auxiliary(path: Path) -> bool:
    return any(part in AUX_PARTS for part in path.parts)


def count_code_lines(lines: list[str]) -> int:
    return sum(1 for line in lines if (s := line.strip()) and not s.startswith("#"))


def resolve_from_import(current_module: str, node: ast.ImportFrom) -> str | None:
    package_parts = current_module.split(".")[:-1]
    if node.level:
        if node.level - 1 > len(package_parts):
            return None
        anchor = package_parts[: len(package_parts) - (node.level - 1)]
    else:
        anchor = []
    suffix = node.module.split(".") if node.module else []
    full = ".".join(anchor + suffix)
    return full or None


def match_internal_module(candidate: str, modules: set[str]) -> str | None:
    parts = candidate.split(".")
    for end in range(len(parts), 0, -1):
        matched = ".".join(parts[:end])
        if matched in modules:
            return matched
    return None


def scan_modules() -> tuple[list[ModuleInfo], list[ModuleInfo]]:
    all_paths = sorted(PACKAGE_ROOT.rglob("*.py"))
    module_names = {module_name_for(path) for path in all_paths}
    runtime: list[ModuleInfo] = []
    auxiliary: list[ModuleInfo] = []

    for path in all_paths:
        module = module_name_for(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        tree = ast.parse(path.read_text(encoding="utf-8"))

        top_level_functions = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body
        )
        top_level_classes = sum(isinstance(node, ast.ClassDef) for node in tree.body)

        internal_imports: set[str] = set()
        external_roots: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    matched = match_internal_module(alias.name, module_names)
                    if matched is not None:
                        if matched != module:
                            internal_imports.add(matched)
                    else:
                        external_roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                full = resolve_from_import(module, node)
                if not full:
                    continue
                if full.startswith("beamz"):
                    for alias in node.names:
                        if alias.name == "*":
                            matched = match_internal_module(full, module_names)
                        else:
                            matched = match_internal_module(f"{full}.{alias.name}", module_names)
                            if matched is None:
                                matched = match_internal_module(full, module_names)
                        if matched is not None and matched != module:
                            internal_imports.add(matched)
                else:
                    external_roots.add(full.split(".")[0])

        info = ModuleInfo(
            name=module,
            path=path,
            rel_path=str(path.relative_to(ROOT)),
            package=module.rsplit(".", 1)[0],
            group=package_group(module),
            total_lines=len(lines),
            code_lines=count_code_lines(lines),
            top_level_functions=top_level_functions,
            top_level_classes=top_level_classes,
            internal_imports=sorted(internal_imports),
            external_import_roots=sorted(external_roots),
            is_auxiliary=is_auxiliary(path),
        )
        (auxiliary if info.is_auxiliary else runtime).append(info)

    return runtime, auxiliary


def build_tree(modules: list[ModuleInfo]) -> dict:
    root = {"name": "beamz", "kind": "package", "children": {}, "total_lines": 0, "code_lines": 0}

    for info in modules:
        rel_parts = Path(info.rel_path).parts
        cursor = root
        cursor["total_lines"] += info.total_lines
        cursor["code_lines"] += info.code_lines
        for part in rel_parts[1:-1]:
            children = cursor["children"]
            if part not in children:
                children[part] = {
                    "name": part,
                    "kind": "package",
                    "children": {},
                    "total_lines": 0,
                    "code_lines": 0,
                }
            cursor = children[part]
            cursor["total_lines"] += info.total_lines
            cursor["code_lines"] += info.code_lines
        cursor["children"][rel_parts[-1]] = {
            "name": rel_parts[-1],
            "kind": "module",
            "module": info.name,
            "group": info.group,
            "total_lines": info.total_lines,
            "code_lines": info.code_lines,
            "classes": info.top_level_classes,
            "functions": info.top_level_functions,
        }

    def normalize(node: dict) -> dict:
        if node["kind"] == "package":
            children = [normalize(child) for _, child in sorted(node["children"].items())]
            return {
                "name": node["name"],
                "kind": node["kind"],
                "total_lines": node["total_lines"],
                "code_lines": node["code_lines"],
                "children": children,
            }
        return node

    return normalize(root)


def package_stats(modules: list[ModuleInfo]) -> list[dict]:
    groups = defaultdict(list)
    for info in modules:
        groups[info.group].append(info)

    inbound = Counter()
    outbound = Counter()
    for info in modules:
        for target in info.internal_imports:
            src_group = info.group
            dst_group = package_group(target)
            if src_group != dst_group:
                outbound[src_group] += 1
                inbound[dst_group] += 1

    stats = []
    for group, members in groups.items():
        stats.append(
            {
                "group": group,
                "files": len(members),
                "total_lines": sum(m.total_lines for m in members),
                "code_lines": sum(m.code_lines for m in members),
                "imports_out": outbound[group],
                "imports_in": inbound[group],
            }
        )
    return sorted(stats, key=lambda item: (-item["total_lines"], item["group"]))


def package_edges(modules: list[ModuleInfo]) -> list[dict]:
    edges = Counter()
    for info in modules:
        for target in info.internal_imports:
            src_group = info.group
            dst_group = package_group(target)
            if src_group != dst_group:
                edges[(src_group, dst_group)] += 1
    return [
        {"source": source, "target": target, "weight": weight}
        for (source, target), weight in sorted(edges.items(), key=lambda item: (-item[1], item[0]))
    ]


def module_graph(modules: list[ModuleInfo]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for info in modules:
        grouped[info.group].append(info)

    result: dict[str, dict] = {}
    for group, members in grouped.items():
        member_names = {m.name for m in members}
        edges = []
        for info in members:
            for target in info.internal_imports:
                if target in member_names and target != info.name:
                    edges.append({"source": info.name, "target": target, "weight": 1})
        result[group] = {
            "nodes": [
                {
                    "id": info.name,
                    "label": info.name.split(".")[-1],
                    "title": info.name,
                    "total_lines": info.total_lines,
                    "code_lines": info.code_lines,
                }
                for info in sorted(members, key=lambda item: (-item.total_lines, item.name))
            ],
            "edges": edges,
        }
    return result


def hotspot_table(modules: list[ModuleInfo]) -> list[dict]:
    inbound = Counter()
    for info in modules:
        for target in info.internal_imports:
            inbound[target] += 1

    rows = []
    for info in modules:
        rows.append(
            {
                "module": info.name,
                "path": info.rel_path,
                "group": info.group,
                "total_lines": info.total_lines,
                "code_lines": info.code_lines,
                "classes": info.top_level_classes,
                "functions": info.top_level_functions,
                "fan_in": inbound[info.name],
                "fan_out": len(info.internal_imports),
            }
        )
    return sorted(rows, key=lambda item: (-item["total_lines"], -item["fan_in"], item["module"]))


def dependency_roots(modules: list[ModuleInfo]) -> list[dict]:
    counts = Counter()
    for info in modules:
        counts.update(
            name
            for name in info.external_import_roots
            if name not in STDLIB_IMPORT_ROOTS
        )
    rows = [{"name": name, "count": count} for name, count in counts.most_common(16)]
    return rows


def facade_map(modules: list[ModuleInfo]) -> list[dict]:
    module_lookup = {info.name: info for info in modules}
    rows = []
    for info in sorted(modules, key=lambda item: item.rel_path):
        basename = info.path.stem
        if basename not in FACADE_BASENAMES:
            continue
        peers = []
        for target in info.internal_imports:
            target_info = module_lookup.get(target)
            if target_info is None or target_info.package != info.package:
                continue
            peers.append(
                {
                    "module": target,
                    "label": target.split(".")[-1],
                    "total_lines": target_info.total_lines,
                }
            )
        if peers:
            rows.append(
                {
                    "module": info.name,
                    "path": info.rel_path,
                    "helpers": sorted(peers, key=lambda item: (item["label"])),
                }
            )
    return rows


def summary(modules: list[ModuleInfo], auxiliary: list[ModuleInfo]) -> dict:
    return {
        "runtime_modules": len(modules),
        "runtime_total_lines": sum(m.total_lines for m in modules),
        "runtime_code_lines": sum(m.code_lines for m in modules),
        "runtime_packages": len({m.group for m in modules}),
        "giant_modules": sum(1 for m in modules if m.total_lines >= 500),
        "auxiliary_modules": len(auxiliary),
        "auxiliary_total_lines": sum(m.total_lines for m in auxiliary),
    }


def build_report() -> dict:
    runtime_modules, auxiliary_modules = scan_modules()
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_head": git_head(),
        "summary": summary(runtime_modules, auxiliary_modules),
        "tree": build_tree(runtime_modules),
        "packages": package_stats(runtime_modules),
        "package_edges": package_edges(runtime_modules),
        "module_graph": module_graph(runtime_modules),
        "hotspots": hotspot_table(runtime_modules)[:20],
        "dependencies": dependency_roots(runtime_modules),
        "facades": facade_map(runtime_modules),
    }


def html_template(report_json: str) -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BEAMZ Architecture Report</title>
  <style>
    :root {
      --bg: #f5f6f1;
      --panel: #ffffff;
      --ink: #12201b;
      --muted: #61736b;
      --line: #d6ddd8;
      --accent: #2d6a4f;
      --accent-soft: #dceee4;
      --accent-2: #355070;
      --danger: #b56576;
      --shadow: 0 14px 30px rgba(18, 32, 27, 0.08);
      --radius: 18px;
      --mono: "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
      --sans: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(45, 106, 79, 0.08), transparent 25rem),
        radial-gradient(circle at top right, rgba(53, 80, 112, 0.08), transparent 24rem),
        var(--bg);
      color: var(--ink);
    }
    .page {
      width: min(1380px, calc(100vw - 32px));
      margin: 28px auto 40px;
      display: grid;
      gap: 18px;
    }
    .hero, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 28px 30px 24px;
      display: grid;
      gap: 18px;
    }
    .eyebrow {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }
    h1, h2, h3 {
      margin: 0;
      font-weight: 600;
      line-height: 1.12;
    }
    h1 { font-size: clamp(34px, 5vw, 52px); }
    h2 { font-size: 24px; }
    h3 { font-size: 16px; }
    p {
      margin: 0;
      line-height: 1.5;
      color: var(--muted);
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 14px;
      color: var(--muted);
    }
    .chip {
      display: inline-flex;
      padding: 8px 10px;
      border-radius: 999px;
      background: #f0f4f2;
      border: 1px solid var(--line);
      font-family: var(--mono);
      font-size: 12px;
      color: var(--ink);
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }
    .stat {
      padding: 16px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff, #fafcfb);
    }
    .stat .label {
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .stat .value {
      font-size: 30px;
      font-weight: 600;
      line-height: 1;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
    }
    .panel {
      padding: 20px;
      display: grid;
      gap: 16px;
      min-width: 0;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
    }
    .legend {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 12px;
      color: var(--muted);
    }
    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }
    .bars {
      display: grid;
      gap: 12px;
    }
    .bar-row {
      display: grid;
      gap: 6px;
    }
    .bar-meta {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      font-size: 13px;
    }
    .bar {
      height: 11px;
      border-radius: 999px;
      background: #eef2ef;
      overflow: hidden;
      border: 1px solid #e6ece8;
    }
    .bar > span {
      display: block;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), #5f8f75);
    }
    .tree {
      font-family: var(--mono);
      font-size: 12px;
      display: grid;
      gap: 6px;
      max-height: 720px;
      overflow: auto;
      padding-right: 4px;
    }
    details.node {
      border-left: 1px dashed #d2d8d4;
      padding-left: 12px;
      margin-left: 6px;
    }
    .tree .leaf, summary {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 10px;
      align-items: center;
      cursor: default;
      list-style: none;
      padding: 4px 0;
    }
    summary::-webkit-details-marker { display: none; }
    .name {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .badge {
      border-radius: 999px;
      padding: 3px 8px;
      background: #f0f4f2;
      border: 1px solid var(--line);
      font-size: 11px;
      color: var(--muted);
    }
    .code-badge {
      background: #eef5f9;
      border-color: #d7e6f2;
      color: #355070;
    }
    .table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .table th, .table td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    .table th {
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .mono { font-family: var(--mono); }
    .helpers {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .helper-pill {
      padding: 6px 8px;
      border-radius: 10px;
      background: #f5faf7;
      border: 1px solid #dde9e1;
      font-size: 12px;
      font-family: var(--mono);
    }
    .graph-wrap {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, #fcfdfc, #f6f8f7);
      overflow: hidden;
      min-height: 520px;
      position: relative;
    }
    svg.graph {
      width: 100%;
      height: 100%;
      min-height: 520px;
      display: block;
    }
    .controls {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    select {
      font: inherit;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
    }
    .note {
      font-size: 12px;
      color: var(--muted);
    }
    @media (max-width: 1020px) {
      .grid-2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Architecture Snapshot</div>
      <div class="section-head">
        <div>
          <h1>BEAMZ Module Map</h1>
          <p>Runtime package structure, internal import graph, facade-helper splits, and hotspot stats in one browser page.</p>
        </div>
      </div>
      <div class="meta">
        <span class="chip" id="meta-head"></span>
        <span class="chip" id="meta-time"></span>
        <span class="chip">Runtime-only scan; bundled examples/tests excluded from graphs</span>
      </div>
      <div class="stats" id="summary-stats"></div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <div class="section-head">
          <div>
            <h2>Package Weight</h2>
            <p>File count, LOC, and cross-package coupling by runtime package group.</p>
          </div>
        </div>
        <div class="bars" id="package-bars"></div>
      </div>
      <div class="panel">
        <div class="section-head">
          <div>
            <h2>Facade Map</h2>
            <p>Public modules that now front smaller helper modules.</p>
          </div>
        </div>
        <table class="table" id="facade-table"></table>
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <div class="section-head">
          <div>
            <h2>Package Graph</h2>
            <p>Cross-package import flow, weighted by runtime internal imports.</p>
          </div>
          <div class="legend">
            <span><i class="dot" style="background:#2d6a4f"></i>Node size = total LOC</span>
            <span><i class="dot" style="background:#355070"></i>Edge width = import count</span>
          </div>
        </div>
        <div class="graph-wrap"><svg class="graph" id="package-graph"></svg></div>
      </div>
      <div class="panel">
        <div class="section-head">
          <div>
            <h2>Package Tree</h2>
            <p>Collapsible runtime module tree with total and code LOC.</p>
          </div>
        </div>
        <div class="tree" id="tree"></div>
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <div>
          <h2>Module Graph</h2>
          <p>Internal edges inside a selected package group.</p>
        </div>
        <div class="controls">
          <label class="note" for="module-group">Package</label>
          <select id="module-group"></select>
        </div>
      </div>
      <div class="graph-wrap"><svg class="graph" id="module-graph"></svg></div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <div class="section-head">
          <div>
            <h2>Hotspots</h2>
            <p>Largest runtime modules and their local coupling profile.</p>
          </div>
        </div>
        <table class="table" id="hotspot-table"></table>
      </div>
      <div class="panel">
        <div class="section-head">
          <div>
            <h2>External Dependency Roots</h2>
            <p>Top non-BEAMZ import roots across runtime modules.</p>
          </div>
        </div>
        <div class="bars" id="dependency-bars"></div>
      </div>
    </section>
  </div>

  <script>
    const report = __REPORT_JSON__;

    function fmt(n) {
      return new Intl.NumberFormat("en-US").format(n);
    }

    function setSummary() {
      document.getElementById("meta-head").textContent = `git ${report.git_head}`;
      document.getElementById("meta-time").textContent = `generated ${report.generated_at}`;
      const items = [
        ["Runtime modules", report.summary.runtime_modules],
        ["Runtime packages", report.summary.runtime_packages],
        ["Runtime LOC", report.summary.runtime_total_lines],
        ["Code LOC", report.summary.runtime_code_lines],
        ["Modules ≥ 500 LOC", report.summary.giant_modules],
        ["Auxiliary modules omitted", report.summary.auxiliary_modules],
      ];
      const container = document.getElementById("summary-stats");
      container.innerHTML = "";
      for (const [label, value] of items) {
        const card = document.createElement("div");
        card.className = "stat";
        card.innerHTML = `<div class="label">${label}</div><div class="value">${fmt(value)}</div>`;
        container.appendChild(card);
      }
    }

    function renderPackageBars() {
      const maxLines = Math.max(...report.packages.map(pkg => pkg.total_lines), 1);
      const container = document.getElementById("package-bars");
      container.innerHTML = "";
      for (const pkg of report.packages) {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <div class="bar-meta">
            <strong class="mono">${pkg.group}</strong>
            <span>${fmt(pkg.files)} files • ${fmt(pkg.total_lines)} LOC • in ${fmt(pkg.imports_in)} / out ${fmt(pkg.imports_out)}</span>
          </div>
          <div class="bar"><span style="width:${(pkg.total_lines / maxLines) * 100}%"></span></div>
        `;
        container.appendChild(row);
      }
    }

    function renderFacadeTable() {
      const table = document.getElementById("facade-table");
      table.innerHTML = `
        <thead>
          <tr>
            <th>Facade</th>
            <th>Helpers</th>
          </tr>
        </thead>
        <tbody>
          ${report.facades.map(item => `
            <tr>
              <td>
                <div class="mono">${item.module}</div>
                <div class="note">${item.path}</div>
              </td>
              <td>
                <div class="helpers">
                  ${item.helpers.map(helper => `<span class="helper-pill">${helper.label} · ${fmt(helper.total_lines)} LOC</span>`).join("")}
                </div>
              </td>
            </tr>
          `).join("")}
        </tbody>
      `;
    }

    function renderTreeNode(node) {
      if (node.kind === "module") {
        return `
          <div class="leaf">
            <span class="name">${node.name}</span>
            <span class="badge">${fmt(node.total_lines)} LOC</span>
            <span class="badge code-badge">${fmt(node.code_lines)} code</span>
          </div>
        `;
      }
      const open = node.name === "beamz" || node.name === "simulation" || node.name === "devices";
      return `
        <details class="node" ${open ? "open" : ""}>
          <summary>
            <span class="name">${node.name}</span>
            <span class="badge">${fmt(node.total_lines)} LOC</span>
            <span class="badge code-badge">${fmt(node.code_lines)} code</span>
          </summary>
          ${node.children.map(renderTreeNode).join("")}
        </details>
      `;
    }

    function renderTree() {
      document.getElementById("tree").innerHTML = renderTreeNode(report.tree);
    }

    function polar(i, n, radius, cx, cy) {
      const angle = (-Math.PI / 2) + (i / Math.max(n, 1)) * Math.PI * 2;
      return { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius };
    }

    function drawGraph(svgId, nodes, edges, nodeMetricKey) {
      const svg = document.getElementById(svgId);
      const rect = svg.getBoundingClientRect();
      const width = Math.max(880, rect.width || 880);
      const height = Math.max(520, rect.height || 520);
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.innerHTML = "";

      const cx = width / 2;
      const cy = height / 2;
      const radius = Math.min(width, height) * 0.33;
      const ordered = [...nodes].sort((a, b) => (b[nodeMetricKey] - a[nodeMetricKey]) || a.id.localeCompare(b.id));
      const maxMetric = Math.max(...ordered.map(node => node[nodeMetricKey]), 1);
      const positions = new Map();

      ordered.forEach((node, index) => {
        positions.set(node.id, polar(index, ordered.length, radius, cx, cy));
      });

      const edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      svg.appendChild(edgeLayer);
      svg.appendChild(nodeLayer);

      const maxWeight = Math.max(...edges.map(edge => edge.weight), 1);
      for (const edge of edges) {
        const src = positions.get(edge.source);
        const dst = positions.get(edge.target);
        if (!src || !dst) continue;

        const mx = (src.x + dst.x) / 2;
        const my = (src.y + dst.y) / 2;
        const curveX = cx + (mx - cx) * 0.6;
        const curveY = cy + (my - cy) * 0.6;

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", `M ${src.x} ${src.y} Q ${curveX} ${curveY} ${dst.x} ${dst.y}`);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", "rgba(53, 80, 112, 0.34)");
        path.setAttribute("stroke-width", String(1 + (edge.weight / maxWeight) * 5));
        edgeLayer.appendChild(path);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", String(curveX));
        label.setAttribute("y", String(curveY));
        label.setAttribute("fill", "#61736b");
        label.setAttribute("font-size", "11");
        label.setAttribute("text-anchor", "middle");
        label.textContent = String(edge.weight);
        edgeLayer.appendChild(label);
      }

      for (const node of ordered) {
        const pos = positions.get(node.id);
        const r = 14 + Math.sqrt(node[nodeMetricKey] / maxMetric) * 22;
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", String(pos.x));
        circle.setAttribute("cy", String(pos.y));
        circle.setAttribute("r", String(r));
        circle.setAttribute("fill", "#dceee4");
        circle.setAttribute("stroke", "#2d6a4f");
        circle.setAttribute("stroke-width", "2");
        g.appendChild(circle);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", String(pos.x));
        label.setAttribute("y", String(pos.y + 4));
        label.setAttribute("text-anchor", "middle");
        label.setAttribute("fill", "#12201b");
        label.setAttribute("font-size", ordered.length > 12 ? "11" : "12");
        label.setAttribute("font-family", "SFMono-Regular, ui-monospace, monospace");
        label.textContent = node.label;
        g.appendChild(label);

        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = `${node.title || node.id}\n${fmt(node.total_lines || node[nodeMetricKey])} LOC`;
        g.appendChild(title);

        nodeLayer.appendChild(g);
      }
    }

    function renderPackageGraph() {
      const nodes = report.packages.map(pkg => ({
        id: pkg.group,
        label: pkg.group.replace("beamz.", ""),
        title: pkg.group,
        total_lines: pkg.total_lines,
      }));
      const edges = report.package_edges.map(edge => ({
        source: edge.source,
        target: edge.target,
        weight: edge.weight,
      }));
      drawGraph("package-graph", nodes, edges, "total_lines");
    }

    function renderModuleGraph() {
      const select = document.getElementById("module-group");
      const groups = Object.keys(report.module_graph).sort((a, b) => a.localeCompare(b));
      select.innerHTML = groups.map(group => `<option value="${group}">${group}</option>`).join("");
      const preferred = groups.includes("beamz.simulation") ? "beamz.simulation" : groups[0];
      select.value = preferred;

      function paint() {
        const group = select.value;
        const graph = report.module_graph[group];
        drawGraph("module-graph", graph.nodes, graph.edges, "total_lines");
      }

      select.addEventListener("change", paint);
      paint();
    }

    function renderHotspots() {
      const table = document.getElementById("hotspot-table");
      table.innerHTML = `
        <thead>
          <tr>
            <th>Module</th>
            <th>LOC</th>
            <th>Fan-in / Fan-out</th>
            <th>Surface</th>
          </tr>
        </thead>
        <tbody>
          ${report.hotspots.map(row => `
            <tr>
              <td>
                <div class="mono">${row.module}</div>
                <div class="note">${row.path}</div>
              </td>
              <td>${fmt(row.total_lines)} / <span class="note">${fmt(row.code_lines)} code</span></td>
              <td>${fmt(row.fan_in)} / ${fmt(row.fan_out)}</td>
              <td>${fmt(row.classes)} classes • ${fmt(row.functions)} functions</td>
            </tr>
          `).join("")}
        </tbody>
      `;
    }

    function renderDependencies() {
      const maxCount = Math.max(...report.dependencies.map(dep => dep.count), 1);
      const container = document.getElementById("dependency-bars");
      container.innerHTML = "";
      for (const dep of report.dependencies) {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <div class="bar-meta">
            <strong class="mono">${dep.name}</strong>
            <span>${fmt(dep.count)} modules</span>
          </div>
          <div class="bar"><span style="width:${(dep.count / maxCount) * 100}%; background:linear-gradient(90deg, #355070, #6d8aa7);"></span></div>
        `;
        container.appendChild(row);
      }
    }

    setSummary();
    renderPackageBars();
    renderFacadeTable();
    renderTree();
    renderPackageGraph();
    renderModuleGraph();
    renderHotspots();
    renderDependencies();
    window.addEventListener("resize", () => {
      renderPackageGraph();
      const select = document.getElementById("module-group");
      if (select.options.length) {
        const graph = report.module_graph[select.value];
        drawGraph("module-graph", graph.nodes, graph.edges, "total_lines");
      }
    });
  </script>
</body>
</html>
""".replace("__REPORT_JSON__", report_json)


def main() -> None:
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html_template(json.dumps(report)), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
