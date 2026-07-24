"""Portable performance records with controlled-hardware regression policy."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "beamz.performance/v1"


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """One reproducible compile/warm-runtime/memory measurement."""

    beamz_commit: str
    beamz_version: str
    python_version: str
    jax_version: str
    jaxlib_version: str
    device: str
    precision: str
    grid_dimensions: tuple[int, ...]
    timesteps: int
    boundaries: tuple[str, ...]
    sources: tuple[str, ...]
    monitors: tuple[str, ...]
    trace_lower_s: float
    compile_s: float
    warm_runtime_samples_s: tuple[float, ...]
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        text_fields = (
            self.beamz_commit,
            self.beamz_version,
            self.python_version,
            self.jax_version,
            self.jaxlib_version,
            self.device,
            self.precision,
        )
        if not all(str(value).strip() for value in text_fields):
            raise ValueError("benchmark identity fields must be non-empty")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        if len(self.grid_dimensions) not in {2, 3} or any(
            int(value) <= 0 for value in self.grid_dimensions
        ):
            raise ValueError("grid_dimensions must contain two or three positive sizes")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive")
        durations = (
            self.trace_lower_s,
            self.compile_s,
            *self.warm_runtime_samples_s,
        )
        if len(self.warm_runtime_samples_s) < 3:
            raise ValueError("at least three warm runtime samples are required")
        if any(not math.isfinite(value) or value <= 0.0 for value in durations):
            raise ValueError("benchmark durations must be positive and finite")
        if self.peak_memory_bytes <= 0:
            raise ValueError("peak_memory_bytes must be positive")

    @property
    def material_cells(self) -> int:
        return math.prod(self.grid_dimensions)

    @property
    def median_warm_runtime_s(self) -> float:
        return float(statistics.median(self.warm_runtime_samples_s))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": SCHEMA_VERSION,
                "material_cells": self.material_cells,
                "median_warm_runtime_s": self.median_warm_runtime_s,
            }
        )
        json.dumps(payload, allow_nan=False)
        return payload


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Relative changes and optional controlled-machine gate result."""

    warm_runtime_change: float
    compile_change: float
    memory_change: float
    controlled_hardware: bool
    passed: bool | None


def _relative_change(current: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ValueError("baseline benchmark values must be positive")
    return current / baseline - 1.0


def compare_benchmarks(
    baseline: BenchmarkRecord,
    current: BenchmarkRecord,
    *,
    controlled_hardware: bool,
    runtime_limit: float = 0.05,
    memory_limit: float = 0.05,
    compile_limit: float = 0.10,
) -> BenchmarkComparison:
    """Compare records; gate only when the machine is explicitly controlled."""
    if (
        baseline.device != current.device
        or baseline.precision != current.precision
        or baseline.grid_dimensions != current.grid_dimensions
        or baseline.timesteps != current.timesteps
    ):
        raise ValueError("benchmark records do not describe the same workload")
    runtime_change = _relative_change(
        current.median_warm_runtime_s,
        baseline.median_warm_runtime_s,
    )
    compile_change = _relative_change(current.compile_s, baseline.compile_s)
    memory_change = _relative_change(
        float(current.peak_memory_bytes),
        float(baseline.peak_memory_bytes),
    )
    passed = (
        runtime_change <= runtime_limit
        and memory_change <= memory_limit
        and compile_change <= compile_limit
        if controlled_hardware
        else None
    )
    return BenchmarkComparison(
        warm_runtime_change=runtime_change,
        compile_change=compile_change,
        memory_change=memory_change,
        controlled_hardware=controlled_hardware,
        passed=passed,
    )
