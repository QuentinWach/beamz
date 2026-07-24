"""Structured validation measurements and JSON report serialization."""

from __future__ import annotations

import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import jax
import numpy as np

import beamz
from tests.validation.tolerances import Tolerance, get_tolerance

SCHEMA_VERSION = "beamz.validation/v2"


@dataclass(frozen=True, slots=True)
class ValidationMetric:
    """One scalar BeamZ observable compared with an independent reference."""

    case_id: str
    quantity: str
    measured: float
    reference: float
    tolerance: Tolerance
    unit: str = ""
    resolution: str = ""
    backend: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    comparison: str = "close"

    def __post_init__(self) -> None:
        if not self.case_id or not self.quantity:
            raise ValueError("case_id and quantity must be non-empty")
        if not math.isfinite(self.measured) or not math.isfinite(self.reference):
            raise ValueError("measured and reference values must be finite")
        if self.comparison not in {"close", "less_equal", "greater_equal"}:
            raise ValueError(
                "comparison must be 'close', 'less_equal', or 'greater_equal'"
            )
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must contain finite JSON values") from error
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def absolute_error(self) -> float:
        return abs(self.measured - self.reference)

    @property
    def relative_error(self) -> float:
        scale = max(abs(self.reference), self.tolerance.relative_floor)
        if scale == 0:
            return 0.0 if self.absolute_error == 0 else math.inf
        return self.absolute_error / scale

    @property
    def error_limit(self) -> float:
        return self.tolerance.error_limit(self.reference)

    @property
    def passed(self) -> bool:
        if self.comparison == "less_equal":
            return self.measured <= self.reference + self.error_limit
        if self.comparison == "greater_equal":
            return self.measured >= self.reference - self.error_limit
        return self.absolute_error <= self.error_limit

    @property
    def margin(self) -> float:
        """Return signed distance to the applicable gate; non-negative passes."""
        if self.comparison == "less_equal":
            return self.reference + self.error_limit - self.measured
        if self.comparison == "greater_equal":
            return self.measured - (self.reference - self.error_limit)
        return self.error_limit - self.absolute_error

    def diagnostic(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        relative_error = (
            f"{self.relative_error:.3%}"
            if math.isfinite(self.relative_error)
            else "n/a (zero reference)"
        )
        comparison = {
            "close": "reference",
            "less_equal": "upper_bound",
            "greater_equal": "lower_bound",
        }[self.comparison]
        return (
            f"{self.case_id}: {self.quantity} measured={self.measured:.12g}{unit}, "
            f"{comparison}={self.reference:.12g}{unit}, "
            f"abs_error={self.absolute_error:.6g}, "
            f"rel_error={relative_error}, "
            f"limit={self.error_limit:.6g} "
            f"margin={self.margin:.6g} "
            f"({self.tolerance.name}: {self.tolerance.rationale})"
        )

    def as_dict(self) -> dict[str, Any]:
        relative_error = self.relative_error
        return {
            "case_id": self.case_id,
            "quantity": self.quantity,
            "comparison": self.comparison,
            "measured": self.measured,
            "reference": self.reference,
            "tolerance": asdict(self.tolerance),
            "unit": self.unit,
            "resolution": self.resolution,
            "backend": self.backend,
            "metadata": dict(self.metadata),
            "absolute_error": self.absolute_error,
            "relative_error": (
                relative_error if math.isfinite(relative_error) else None
            ),
            "error_limit": self.error_limit,
            "margin": self.margin,
            "passed": self.passed,
        }


class ValidationRecorder:
    """Collect and assert metrics for one pytest case."""

    def __init__(self, case_id: str, sink: list[ValidationMetric]) -> None:
        self.case_id = case_id
        self._sink = sink

    def check(
        self,
        quantity: str,
        measured: float,
        reference: float,
        *,
        tolerance: str | Tolerance,
        unit: str = "",
        resolution: str = "",
        backend: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ValidationMetric:
        """Record a metric and fail with its complete numerical diagnostic."""
        return self._check(
            quantity,
            measured,
            reference,
            tolerance=tolerance,
            unit=unit,
            resolution=resolution,
            backend=backend,
            metadata=metadata,
            comparison="close",
        )

    def check_upper(
        self,
        quantity: str,
        measured: float,
        upper_bound: float,
        *,
        tolerance: str | Tolerance = "exact",
        unit: str = "",
        resolution: str = "",
        backend: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ValidationMetric:
        """Record and enforce an inclusive scalar upper bound."""
        return self._check(
            quantity,
            measured,
            upper_bound,
            tolerance=tolerance,
            unit=unit,
            resolution=resolution,
            backend=backend,
            metadata=metadata,
            comparison="less_equal",
        )

    def check_lower(
        self,
        quantity: str,
        measured: float,
        lower_bound: float,
        *,
        tolerance: str | Tolerance = "exact",
        unit: str = "",
        resolution: str = "",
        backend: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ValidationMetric:
        """Record and enforce an inclusive scalar lower bound."""
        return self._check(
            quantity,
            measured,
            lower_bound,
            tolerance=tolerance,
            unit=unit,
            resolution=resolution,
            backend=backend,
            metadata=metadata,
            comparison="greater_equal",
        )

    def _check(
        self,
        quantity: str,
        measured: float,
        reference: float,
        *,
        tolerance: str | Tolerance,
        unit: str,
        resolution: str,
        backend: str,
        metadata: dict[str, Any] | None,
        comparison: str,
    ) -> ValidationMetric:
        """Construct, retain, and assert one comparison metric."""
        resolved = get_tolerance(tolerance) if isinstance(tolerance, str) else tolerance
        metric = ValidationMetric(
            case_id=self.case_id,
            quantity=quantity,
            measured=float(measured),
            reference=float(reference),
            tolerance=resolved,
            unit=unit,
            resolution=resolution,
            backend=backend or jax.default_backend(),
            metadata={} if metadata is None else dict(metadata),
            comparison=comparison,
        )
        self._sink.append(metric)
        assert metric.passed, metric.diagnostic()
        return metric


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def validation_report(
    metrics: list[ValidationMetric],
    *,
    exit_status: int,
    random_seed: str,
) -> dict[str, Any]:
    """Build the bounded, portable validation report payload."""
    passed = sum(metric.passed for metric in metrics)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "beamz_commit": _git_commit(),
        "environment": {
            "beamz": beamz.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "random_seed": random_seed,
        },
        "summary": {
            "exit_status": int(exit_status),
            "metrics": len(metrics),
            "passed": passed,
            "failed": len(metrics) - passed,
        },
        "metrics": [metric.as_dict() for metric in metrics],
    }


def write_validation_report(report: dict[str, Any], destination: Path) -> None:
    """Write a stable, human-readable JSON artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
