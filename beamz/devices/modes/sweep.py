"""Small helpers for mode sweeps and overlap-based mode tracking."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .result import Result


@dataclass(frozen=True)
class Sweep:
    """A sequence of solved mode results over one scalar sweep parameter."""

    values: np.ndarray
    results: tuple[Result, ...]
    parameter_name: str = "parameter"

    def __post_init__(self) -> None:
        """Validate sweep lengths and mode-count consistency."""
        # Sweep values become a fixed one-dimensional float axis for all summary
        # arrays and data-frame exports.
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if len(values) != len(self.results):
            raise ValueError("values and results must have the same length")
        if not self.results:
            raise ValueError("at least one result is required")

        # Mode tracking and stacked metrics assume every sweep step exposes the
        # same number of modes.
        mode_counts = {
            int(result.n_complex.sizes["mode_index"]) for result in self.results
        }
        if len(mode_counts) != 1:
            raise ValueError("all sweep results must have the same number of modes")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "results", tuple(self.results))

    @property
    def num_modes(self) -> int:
        """Return the shared number of modes in the sweep."""
        return int(self.results[0].n_complex.sizes["mode_index"])

    @property
    def n_eff(self) -> np.ndarray:
        """Return real effective indices arranged by sweep step and mode."""
        # Each Result stores one or more frequencies; sweep helpers currently
        # summarize the first frequency for every step.
        return np.vstack(
            [np.asarray(result.n_eff.values)[0] for result in self.results]
        )

    @property
    def n_complex(self) -> np.ndarray:
        """Return complex effective indices arranged by sweep step and mode."""
        # Preserve the imaginary part here so loss/k_eff can be derived without
        # re-reading individual Result objects.
        return np.vstack(
            [np.asarray(result.n_complex.values)[0] for result in self.results]
        )

    @property
    def pol_fraction(self) -> dict[str, np.ndarray]:
        """Return TE/TM fractions arranged by sweep step and mode."""
        # Keep TE and TM arrays parallel so callers can index by [step, mode]
        # without touching xarray internals.
        return {
            "te": np.vstack(
                [
                    np.asarray(result.pol_fraction["te"].values)[0]
                    for result in self.results
                ]
            ),
            "tm": np.vstack(
                [
                    np.asarray(result.pol_fraction["tm"].values)[0]
                    for result in self.results
                ]
            ),
        }

    @property
    def pol_fraction_waveguide(self) -> dict[str, np.ndarray]:
        """Return waveguide TE/TM fractions arranged by sweep step and mode."""
        return {
            "te": np.vstack(
                [
                    np.asarray(result.pol_fraction_waveguide["te"].values)[0]
                    for result in self.results
                ]
            ),
            "tm": np.vstack(
                [
                    np.asarray(result.pol_fraction_waveguide["tm"].values)[0]
                    for result in self.results
                ]
            ),
        }

    def to_dataframe(self):
        """Return one row per sweep value and mode."""

        import pandas as pd

        # Precompute stacked arrays once; the nested loop below only packages
        # scalar values into tabular records.
        rows = []
        pol = self.pol_fraction
        wg_pol = self.pol_fraction_waveguide
        for step_index, value in enumerate(self.values):
            for mode_index in range(self.num_modes):
                rows.append(
                    {
                        self.parameter_name: value,
                        "mode_index": mode_index,
                        "n_eff": self.n_eff[step_index, mode_index],
                        "k_eff": self.n_complex[step_index, mode_index].imag,
                        "te_fraction": pol["te"][step_index, mode_index],
                        "tm_fraction": pol["tm"][step_index, mode_index],
                        "wg_te_fraction": wg_pol["te"][step_index, mode_index],
                        "wg_tm_fraction": wg_pol["tm"][step_index, mode_index],
                    }
                )
        return pd.DataFrame(rows)


def track_modes_by_overlap(
    results: Iterable[Result],
    *,
    kind: str = "electric",
) -> tuple[Result, ...]:
    """Reorder sweep results so adjacent steps follow the same modal branch.

    Each result is compared to the previous tracked result. The mode assignment
    that maximizes the sum of normalized overlap magnitudes is chosen. This is
    intended for modest mode counts, where trying every assignment is clearer
    and less fragile than a greedy local choice.
    """

    tracked = tuple(results)
    if not tracked:
        return ()

    # The exhaustive assignment search is factorial, so keep it explicitly
    # limited to small mode sets where it is predictable and readable.
    mode_count = int(tracked[0].n_complex.sizes["mode_index"])
    if mode_count > 8:
        raise ValueError("exhaustive overlap tracking is limited to at most 8 modes")
    reordered = [tracked[0]]
    for result in tracked[1:]:
        if int(result.n_complex.sizes["mode_index"]) != mode_count:
            raise ValueError("all results must have the same number of modes")

        # Compare the next raw result against the previously tracked result, then
        # choose the permutation with the largest total normalized overlap.
        overlaps = np.abs(reordered[-1].overlap_matrix(result, kind=kind).values)
        best_order = max(
            permutations(range(mode_count)),
            key=lambda order: _assignment_score(overlaps, order),
        )
        reordered.append(_reorder_result_modes(result, best_order))
    return tuple(reordered)


def _assignment_score(overlaps: np.ndarray, order: tuple[int, ...]) -> float:
    """Score a proposed mode assignment by total overlap magnitude."""
    # order maps tracked mode_index -> source mode_index in the candidate result.
    return float(
        sum(
            overlaps[mode_index, source_index]
            for mode_index, source_index in enumerate(order)
        )
    )


def _reorder_result_modes(result: Result, order: tuple[int, ...]) -> Result:
    """Return a result with all mode-indexed arrays reordered together."""
    # Reset mode coordinates after reordering so the output uses dense
    # branch-tracking indices instead of the source result's raw order.
    mode_coord = np.arange(len(order))
    n_complex = result.n_complex.isel(mode_index=list(order)).assign_coords(
        mode_index=mode_coord
    )

    # Every field component carries the same mode_index dimension and must move
    # in lockstep with n_complex.
    field_components = {
        name: data_array.isel(mode_index=list(order)).assign_coords(
            mode_index=mode_coord
        )
        for name, data_array in result.field_components.items()
    }

    # Optional dispersion arrays are mode-indexed too; preserve them when present.
    n_group = None
    if result.n_group is not None:
        n_group = result.n_group.isel(mode_index=list(order)).assign_coords(
            mode_index=mode_coord
        )
    dispersion = None
    if result.dispersion is not None:
        dispersion = result.dispersion.isel(mode_index=list(order)).assign_coords(
            mode_index=mode_coord
        )
    return Result(
        n_complex=n_complex,
        field_components=field_components,
        n_group=n_group,
        dispersion=dispersion,
        solver_info=result.solver_info,
    )
