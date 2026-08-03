"""Deterministic one-dimensional grading for rectilinear FDTD grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _positive_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values.")
    return array


def _cell_count(value: float) -> int:
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(value))
    return max(1, int(np.ceil(value - tolerance)))


@dataclass(frozen=True, slots=True)
class _SpacingProfile:
    """Piecewise-linear spacing profile and its integrated cell density."""

    coordinates: np.ndarray
    spacings: np.ndarray
    cumulative_density: np.ndarray

    @classmethod
    def from_interval(
        cls,
        lower: float,
        upper: float,
        cap: float,
        left: float,
        right: float,
        slope: float,
    ) -> _SpacingProfile:
        candidates = [lower, upper]
        if slope > 0.0:
            candidates.extend(
                (
                    lower + (cap - left) / slope,
                    upper - (cap - right) / slope,
                    0.5 * (lower + upper + (right - left) / slope),
                )
            )
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
        coordinates = []
        for value in sorted(candidates):
            clipped = float(np.clip(value, lower, upper))
            if not coordinates or clipped - coordinates[-1] > tolerance:
                coordinates.append(clipped)
        if coordinates[-1] != upper:
            coordinates[-1] = upper
        x = np.asarray(coordinates, dtype=np.float64)
        h = np.minimum(
            cap,
            np.minimum(left + slope * (x - lower), right + slope * (upper - x)),
        )
        h = np.maximum(h, np.finfo(float).tiny)
        density = np.zeros(x.size, dtype=np.float64)
        for index, (x0, x1, h0, h1) in enumerate(
            zip(x[:-1], x[1:], h[:-1], h[1:], strict=True), start=1
        ):
            delta = x1 - x0
            gradient = (h1 - h0) / delta
            if abs(gradient) <= 32.0 * np.finfo(float).eps:
                integral = delta / h0
            else:
                integral = np.log(h1 / h0) / gradient
            density[index] = density[index - 1] + integral
        return cls(x, h, density)

    @property
    def total_density(self) -> float:
        return float(self.cumulative_density[-1])

    def coordinate_at_density(self, density: np.ndarray) -> np.ndarray:
        values = np.asarray(density, dtype=np.float64)
        result = np.empty_like(values)
        for output_index, target in np.ndenumerate(values):
            if target <= 0.0:
                result[output_index] = self.coordinates[0]
                continue
            if target >= self.cumulative_density[-1]:
                result[output_index] = self.coordinates[-1]
                continue
            segment = int(
                np.searchsorted(self.cumulative_density, target, side="right") - 1
            )
            x0, x1 = self.coordinates[segment : segment + 2]
            h0, h1 = self.spacings[segment : segment + 2]
            local_density = target - self.cumulative_density[segment]
            gradient = (h1 - h0) / (x1 - x0)
            if abs(gradient) <= 32.0 * np.finfo(float).eps:
                offset = local_density * h0
            else:
                offset = h0 * np.expm1(gradient * local_density) / gradient
            result[output_index] = x0 + offset
        return result


@dataclass(frozen=True, slots=True)
class GradedMesher:
    """Generate smoothly graded cell edges across constrained 1D intervals.

    Parameters
    ----------
    max_scale : float, default=1.3
        Maximum permitted ratio between the widths of adjacent cells.
    transition_safety : float, default=0.85
        Fraction of the theoretical spacing-profile slope used before the exact
        cell-ratio verification pass. Values below one leave room for cell-count
        quantization at snapped interval boundaries.
    max_refinement_passes : int, default=64
        Maximum number of local cell-count corrections for one slope attempt.
    max_slope_passes : int, default=12
        Maximum number of progressively gentler spacing-envelope attempts.

    Notes
    -----
    ``interval_coords`` passed to :meth:`make_axis_edges` are mandatory grid
    boundaries. ``max_spacings[i]`` is an upper bound on cell width within the
    corresponding interval. The result preserves every boundary, never exceeds
    an interval limit, and verifies the adjacent-cell ratio before returning.
    """

    max_scale: float = 1.3
    transition_safety: float = 0.85
    max_refinement_passes: int = 64
    max_slope_passes: int = 12

    def __post_init__(self) -> None:
        max_scale = float(self.max_scale)
        transition_safety = float(self.transition_safety)
        if not np.isfinite(max_scale) or max_scale <= 1.0 or max_scale >= 2.0:
            raise ValueError("max_scale must be finite and strictly between 1 and 2.")
        if (
            not np.isfinite(transition_safety)
            or transition_safety <= 0.0
            or transition_safety > 1.0
        ):
            raise ValueError("transition_safety must be in the interval (0, 1].")
        if int(self.max_refinement_passes) <= 0 or int(self.max_slope_passes) <= 0:
            raise ValueError("Mesher pass limits must be positive integers.")
        object.__setattr__(self, "max_scale", max_scale)
        object.__setattr__(self, "transition_safety", transition_safety)
        object.__setattr__(
            self, "max_refinement_passes", int(self.max_refinement_passes)
        )
        object.__setattr__(self, "max_slope_passes", int(self.max_slope_passes))

    @staticmethod
    def _boundary_spacings(
        lengths: np.ndarray, caps: np.ndarray, slope: float
    ) -> np.ndarray:
        boundaries = np.empty(caps.size + 1, dtype=np.float64)
        boundaries[0] = caps[0]
        boundaries[-1] = caps[-1]
        if caps.size > 1:
            boundaries[1:-1] = np.minimum(caps[:-1], caps[1:])
        for index in range(caps.size):
            boundaries[index + 1] = min(
                boundaries[index + 1],
                caps[index],
                boundaries[index] + slope * lengths[index],
            )
        for index in range(caps.size - 1, -1, -1):
            boundaries[index] = min(
                boundaries[index],
                caps[index],
                boundaries[index + 1] + slope * lengths[index],
            )
        return boundaries

    @staticmethod
    def _profiles(
        coords: np.ndarray,
        caps: np.ndarray,
        slope: float,
    ) -> list[_SpacingProfile]:
        lengths = np.diff(coords)
        # An exact snapped interval always contains at least one cell. Treat its
        # length as an additional local spacing constraint so a tiny interval
        # grades smoothly into its neighbors instead of creating a sliver cell.
        effective_caps = np.minimum(caps, lengths)
        boundary_spacings = GradedMesher._boundary_spacings(
            lengths, effective_caps, slope
        )
        return [
            _SpacingProfile.from_interval(
                coords[index],
                coords[index + 1],
                effective_caps[index],
                boundary_spacings[index],
                boundary_spacings[index + 1],
                slope,
            )
            for index in range(caps.size)
        ]

    @staticmethod
    def _realize(
        profiles: list[_SpacingProfile], counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pieces = []
        owners = []
        for interval, (profile, count) in enumerate(zip(profiles, counts, strict=True)):
            density = np.linspace(
                0.0, profile.total_density, int(count) + 1, dtype=np.float64
            )
            edges = profile.coordinate_at_density(density)
            edges[[0, -1]] = profile.coordinates[[0, -1]]
            pieces.append(edges if interval == 0 else edges[1:])
            owners.extend([interval] * int(count))
        return np.concatenate(pieces), np.asarray(owners, dtype=np.int64)

    def make_axis_edges(self, interval_coords, max_spacings) -> np.ndarray:
        """Return graded edges satisfying interval and neighbor constraints."""
        coords = np.asarray(interval_coords, dtype=np.float64)
        if (
            coords.ndim != 1
            or coords.size < 2
            or not np.all(np.isfinite(coords))
            or np.any(np.diff(coords) <= 0.0)
        ):
            raise ValueError(
                "interval_coords must be finite, one-dimensional, and strictly increasing."
            )
        caps = _positive_array(max_spacings, "max_spacings")
        if caps.size != coords.size - 1:
            raise ValueError(
                "max_spacings must contain one value per coordinate interval."
            )

        theoretical_slope = 2.0 * (self.max_scale - 1.0) / (self.max_scale + 1.0)
        slope = theoretical_slope * self.transition_safety
        target_tolerance = 128.0 * np.finfo(float).eps

        for _slope_pass in range(self.max_slope_passes):
            profiles = self._profiles(coords, caps, slope)
            counts = np.asarray(
                [_cell_count(profile.total_density) for profile in profiles],
                dtype=np.int64,
            )
            for _refinement_pass in range(self.max_refinement_passes):
                edges, owners = self._realize(profiles, counts)
                widths = np.diff(edges)
                target_limits = caps[owners]
                if np.any(widths > target_limits * (1.0 + target_tolerance)):
                    violating = np.unique(owners[widths > target_limits])
                    counts[violating] += 1
                    continue
                if widths.size < 2:
                    return edges
                ratios = np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])
                violating_pairs = np.flatnonzero(
                    ratios > self.max_scale * (1.0 + target_tolerance)
                )
                if violating_pairs.size == 0:
                    return edges
                internal_violation = False
                intervals_to_refine = set()
                for pair in violating_pairs:
                    left_owner, right_owner = owners[pair : pair + 2]
                    if left_owner == right_owner:
                        internal_violation = True
                        break
                    larger_cell = pair if widths[pair] > widths[pair + 1] else pair + 1
                    intervals_to_refine.add(int(owners[larger_cell]))
                if internal_violation:
                    break
                for interval in intervals_to_refine:
                    counts[interval] += 1
            slope *= 0.65

        raise RuntimeError(
            "Could not construct a graded axis within the configured pass limits. "
            "Relax max_scale, remove near-coincident snapped boundaries, or increase "
            "the mesher pass limits."
        )


__all__ = ["GradedMesher"]
