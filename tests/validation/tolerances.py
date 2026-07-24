"""Named, reviewable tolerances for BeamZ validation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Combined absolute/relative error gate with a required rationale."""

    name: str
    absolute: float
    relative: float
    rationale: str
    relative_floor: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tolerance name must be non-empty")
        if self.absolute < 0 or self.relative < 0 or self.relative_floor < 0:
            raise ValueError("tolerance limits must be non-negative")
        if self.absolute == 0 and self.relative == 0:
            raise ValueError("at least one tolerance limit must be positive")
        if not self.rationale.strip():
            raise ValueError("tolerance rationale must be non-empty")

    def error_limit(self, reference: float) -> float:
        """Return the inclusive absolute-error limit at ``reference``."""
        scale = max(abs(float(reference)), self.relative_floor)
        return max(self.absolute, self.relative * scale)


_TOLERANCES = {
    "exact": Tolerance(
        name="exact",
        absolute=1e-12,
        relative=1e-12,
        rationale="Floating-point allowance for closed-form helper identities.",
    ),
    "kernel_float64": Tolerance(
        name="kernel_float64",
        absolute=1e-11,
        relative=1e-11,
        rationale="Roundoff-scale gate for small deterministic float64 kernels.",
    ),
    "kernel_float32": Tolerance(
        name="kernel_float32",
        absolute=1e-6,
        relative=1e-6,
        rationale="Small-array float32 JAX kernel accumulation and reduction error.",
    ),
    "analytical_coarse": Tolerance(
        name="analytical_coarse",
        absolute=0.0,
        relative=0.05,
        rationale="Initial documented gate for coarse analytical physics cases.",
        relative_floor=1e-12,
    ),
    "analytical_fine": Tolerance(
        name="analytical_fine",
        absolute=0.0,
        relative=0.02,
        rationale="Target gate after a case demonstrates grid refinement.",
        relative_floor=1e-12,
    ),
    "normalized_power_balance": Tolerance(
        name="normalized_power_balance",
        absolute=0.02,
        relative=0.0,
        rationale="Two percentage-point closure error for normalized power budgets.",
    ),
    "gradient_float64": Tolerance(
        name="gradient_float64",
        absolute=1e-8,
        relative=0.01,
        rationale="One-percent directional-derivative target for float64 gradients.",
        relative_floor=1e-10,
    ),
    "gradient_float32": Tolerance(
        name="gradient_float32",
        absolute=2e-5,
        relative=0.05,
        rationale="Realistic five-percent gate for float32 directional derivatives.",
        relative_floor=1e-6,
    ),
    "cross_solver": Tolerance(
        name="cross_solver",
        absolute=0.0,
        relative=0.05,
        rationale="Initial observable-level consensus gate between independent solvers.",
        relative_floor=1e-12,
    ),
}

TOLERANCES = MappingProxyType(_TOLERANCES)


def get_tolerance(name: str) -> Tolerance:
    """Resolve a named tolerance and list valid names on failure."""
    try:
        return TOLERANCES[name]
    except KeyError as error:
        valid = ", ".join(sorted(TOLERANCES))
        raise KeyError(f"unknown tolerance {name!r}; choose one of: {valid}") from error
