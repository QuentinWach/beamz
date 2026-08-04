"""Shared field operations for BeamZ-native modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from beamz.const import LIGHT_SPEED

if TYPE_CHECKING:
    from .discrete import ComponentIndex

_TANGENTIAL_COMPONENTS = {
    "x": ("Ey", "Ez", "Hz", "Hy"),
    "y": ("Ez", "Ex", "Hx", "Hz"),
    "z": ("Ex", "Ey", "Hy", "Hx"),
}
_STAGGERED_ALONG_AXIS = {
    "x": {"Ex", "Hy", "Hz"},
    "y": {"Ey", "Hx", "Hz"},
    "z": {"Ez", "Hx", "Hy"},
}


def _axis_index(indices: ComponentIndex | None, axis: str) -> int | None:
    if indices is None:
        return None
    value = indices[{"x": 2, "y": 1, "z": 0}[axis]]
    return None if isinstance(value, slice) else int(value)


def _axis_coordinate(
    component: str,
    index: int | None,
    axis: str,
    resolution: float,
    grid=None,
) -> float:
    if index is None:
        return 0.0
    offset = 1.0 if component in _STAGGERED_ALONG_AXIS[axis] else 0.5
    if grid is not None:
        coordinates = (
            np.asarray(grid.axis_edges(axis), dtype=float)
            if offset == 1.0
            else np.asarray(grid.centers(axis), dtype=float)
        )
        return float(coordinates[int(np.clip(index, 0, coordinates.size - 1))])
    return (int(index) + offset) * float(resolution)


def _numeric_wave_number(
    omega: float,
    dt: float | None,
    resolution: float,
    neff: complex,
) -> float:
    neff_real = max(float(np.real(neff)), 1e-30)
    physical = float(omega) * neff_real / LIGHT_SPEED
    if dt is None:
        return physical
    courant = LIGHT_SPEED * float(dt) / (neff_real * float(resolution))
    if not np.isfinite(courant) or courant <= 1e-30:
        return physical
    rhs = np.sin(0.5 * float(omega) * float(dt)) / courant
    numeric = (2.0 / float(resolution)) * np.arcsin(np.clip(rhs, -1.0, 1.0))
    return float(numeric) if np.isfinite(numeric) and numeric > 0.0 else physical


def _phase_delay(omega: float, wave_number: float, distance: float) -> float:
    return float(wave_number) * float(distance) / max(abs(float(omega)), 1e-30)


def _modal_overlap(
    fields: Mapping[str, np.ndarray],
    mode: Mapping[str, np.ndarray],
    axis: str,
    measure,
    direction_sign: float = 1.0,
) -> np.complex128:
    names = _TANGENTIAL_COMPONENTS[axis]
    def paired_term(e_name, h_name, sign):
        entries = tuple(
            (profiles, name)
            for profiles, name in (
            (fields, e_name),
            (mode, h_name),
            (mode, e_name),
            (fields, h_name),
            )
        )
        present = [
            np.asarray(profiles[name], dtype=np.complex128).reshape(-1)
            for profiles, name in entries
            if name in profiles and np.asarray(profiles[name]).size
        ]
        if not present:
            return np.complex128(0.0)
        size = min(array.size for array in present)
        if size <= 0:
            return np.complex128(0.0)
        arrays = [
            np.asarray(profiles[name], dtype=np.complex128).reshape(-1)[:size]
            if name in profiles
            else np.zeros(size, dtype=np.complex128)
            for profiles, name in entries
        ]
        ef, hm, em, hf = (array[:size] for array in arrays)
        if isinstance(measure, Mapping):
            weights = np.asarray(measure[e_name], dtype=float).reshape(-1)[:size]
        else:
            raw = np.asarray(measure, dtype=float)
            weights = raw if raw.ndim == 0 else raw.reshape(-1)[:size]
        return np.complex128(
            np.sum(weights * sign * (ef * np.conjugate(hm) + np.conjugate(em) * hf))
        )

    overlap = paired_term(names[0], names[2], 1.0) + paired_term(
        names[1], names[3], -1.0
    )
    return np.complex128(0.25 * float(direction_sign) * overlap)


def _modal_power(
    profiles: Mapping[str, np.ndarray],
    axis: str,
    measure,
    direction_sign: float = 1.0,
) -> float:
    return float(
        np.real(_modal_overlap(profiles, profiles, axis, measure, direction_sign))
    )


def _normalize_profiles(
    profiles: dict[str, np.ndarray],
    axis: str,
    measure,
    direction_sign: float = 1.0,
    *,
    max_scale: float | None = None,
) -> float:
    power = _modal_power(profiles, axis, measure, direction_sign)
    if np.isfinite(power) and abs(power) > np.finfo(float).tiny:
        scale = np.sqrt(1.0 / abs(power))
        if max_scale is not None:
            scale = np.clip(scale, 1.0 / max_scale, max_scale)
        for name, values in profiles.items():
            profiles[name] = np.asarray(values, dtype=np.complex128) * scale
    return power
