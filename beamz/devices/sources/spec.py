from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from beamz.devices.sources.profiles import _parse_direction


def _freeze_arraylike(signal):
    if isinstance(signal, np.ndarray):
        arr = np.asarray(signal).copy()
        arr.setflags(write=False)
        return arr
    if isinstance(signal, jnp.ndarray):
        return signal
    if isinstance(signal, list):
        arr = np.asarray(signal).copy()
        arr.setflags(write=False)
        return arr
    return signal


def _normalize_center(center, grid=None):
    if isinstance(center, (tuple, list)):
        return tuple(float(v) for v in center)
    if grid is not None:
        return (float(center), float(grid.height) / 2.0)
    return (float(center),)


@dataclass(frozen=True, slots=True)
class GaussianSourceSpec:
    position: tuple[float, ...]
    width: float
    signal: object

    def __post_init__(self):
        object.__setattr__(self, "position", tuple(float(v) for v in self.position))
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "signal", _freeze_arraylike(self.signal))
        if self.width <= 0:
            raise ValueError("width must be positive")
        if len(self.position) not in {2, 3}:
            raise ValueError("position must have length 2 or 3")


@dataclass(frozen=True, slots=True)
class ModeSourceSpec:
    center: tuple[float, ...]
    width: float
    height: float | None
    wavelength: float
    pol: str
    signal: object
    direction: str
    direction_axis: str
    direction_sign: float

    def __post_init__(self):
        object.__setattr__(self, "center", tuple(float(v) for v in self.center))
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "height", None if self.height is None else float(self.height))
        object.__setattr__(self, "wavelength", float(self.wavelength))
        pol = str(self.pol).lower()
        if pol not in {"te", "tm"}:
            raise ValueError(f"pol must be 'te' or 'tm', got {self.pol!r}")
        object.__setattr__(self, "pol", pol)
        object.__setattr__(self, "signal", _freeze_arraylike(self.signal))
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive")


def build_gaussian_source_spec(position, width, signal) -> GaussianSourceSpec:
    return GaussianSourceSpec(
        position=tuple(position),
        width=width,
        signal=signal,
    )


def build_mode_source_spec(
    *,
    grid=None,
    center=None,
    width=None,
    wavelength=None,
    pol=None,
    signal=None,
    direction="+x",
    height=None,
) -> ModeSourceSpec:
    parsed_direction, axis, sign = _parse_direction(direction)
    return ModeSourceSpec(
        center=_normalize_center(center, grid=grid),
        width=width,
        height=height,
        wavelength=wavelength,
        pol=pol,
        signal=signal,
        direction=parsed_direction,
        direction_axis=axis,
        direction_sign=sign,
    )
