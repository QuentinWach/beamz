#!/usr/bin/env python3
"""Enforce BeamZ's global and risk-weighted branch-inclusive coverage floors."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CoverageResult:
    name: str
    covered: int
    total: int
    minimum: float

    @property
    def percent(self) -> float:
        return 100.0 * self.covered / self.total if self.total else 100.0

    @property
    def passes(self) -> bool:
        return self.percent + 1e-12 >= self.minimum


def _covered_and_total(summary: dict[str, Any]) -> tuple[int, int]:
    covered = int(summary["covered_lines"]) + int(summary["covered_branches"])
    total = int(summary["num_statements"]) + int(summary["num_branches"])
    return covered, total


def evaluate_policy(
    coverage: dict[str, Any], policy: dict[str, Any]
) -> tuple[CoverageResult, ...]:
    """Calculate every policy row, rejecting stale or malformed file groups."""

    results = []
    covered, total = _covered_and_total(coverage["totals"])
    results.append(
        CoverageResult(
            name="global",
            covered=covered,
            total=total,
            minimum=float(policy["global_minimum"]),
        )
    )

    files = coverage["files"]
    for name, group in policy["groups"].items():
        group_covered = 0
        group_total = 0
        configured_files = tuple(group["files"])
        if not configured_files:
            raise ValueError(f"Coverage group {name!r} contains no files.")
        for filename in configured_files:
            if filename not in files:
                raise ValueError(
                    f"Coverage group {name!r} references missing file {filename!r}."
                )
            file_covered, file_total = _covered_and_total(files[filename]["summary"])
            group_covered += file_covered
            group_total += file_total
        results.append(
            CoverageResult(
                name=name,
                covered=group_covered,
                total=group_total,
                minimum=float(group["minimum"]),
            )
        )
    return tuple(results)


def _print_results(results: tuple[CoverageResult, ...], metric: str) -> None:
    print(f"Coverage policy ({metric})")
    print(f"{'scope':<24} {'actual':>8} {'minimum':>9} {'result':>8}")
    for result in results:
        status = "PASS" if result.passes else "FAIL"
        print(
            f"{result.name:<24} {result.percent:>7.2f}% "
            f"{result.minimum:>8.2f}% {status:>8}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage", type=Path, help="coverage.py JSON report")
    parser.add_argument("policy", type=Path, help="BeamZ coverage policy JSON")
    args = parser.parse_args()

    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    try:
        results = evaluate_policy(coverage, policy)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    _print_results(results, str(policy["metric"]))
    return 0 if all(result.passes for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
