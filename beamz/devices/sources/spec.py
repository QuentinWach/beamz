from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import jax.numpy as jnp
import numpy as np

from beamz.devices.sources.profiles import _parse_direction


def _freeze_arraylike(signal):
    if isinstance(signal, jnp.ndarray):
        signal = np.asarray(signal)
    if isinstance(signal, (np.ndarray, list, tuple)):
        arr = np.asarray(signal).copy()
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("signal arrays must be non-empty 1D arrays")
        if not np.all(np.isfinite(arr)):
            raise ValueError("signal arrays must contain only finite values")
        arr.setflags(write=False)
        return arr
    if signal is not None:
        raise TypeError("signal must be a 1D array-like")
    return None


def _normalize_center(center, grid=None):
    if center is None:
        raise ValueError("center is required")
    if isinstance(center, (tuple, list)):
        vals = tuple(float(v) for v in center)
        if len(vals) not in {2, 3}:
            raise ValueError("center must have length 2 or 3")
        return vals
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
        if not isfinite(self.width) or self.width <= 0:
            raise ValueError("width must be positive")
        if len(self.position) not in {2, 3}:
            raise ValueError("position must have length 2 or 3")
        if self.signal is None:
            raise ValueError("signal is required")

    def to_dict(self):
        return {
            "type": "GaussianSourceSpec",
            "position": list(self.position),
            "width": float(self.width),
            "signal": np.asarray(self.signal).tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            position=tuple(data["position"]),
            width=data["width"],
            signal=data["signal"],
        )


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
        if len(self.center) not in {2, 3}:
            raise ValueError("center must have length 2 or 3")
        if not isfinite(self.width) or self.width <= 0:
            raise ValueError("width must be positive")
        if self.height is not None and ((not isfinite(self.height)) or self.height <= 0):
            raise ValueError("height must be positive when provided")
        if not isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if self.signal is None:
            raise ValueError("signal is required")
        if self.direction_sign not in {-1.0, 1.0}:
            raise ValueError("direction_sign must be +/-1")

    def to_dict(self):
        return {
            "type": "ModeSourceSpec",
            "center": list(self.center),
            "width": float(self.width),
            "height": self.height,
            "wavelength": float(self.wavelength),
            "pol": self.pol,
            "signal": np.asarray(self.signal).tolist(),
            "direction": self.direction,
            "direction_axis": self.direction_axis,
            "direction_sign": float(self.direction_sign),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            center=tuple(data["center"]),
            width=data["width"],
            height=data.get("height"),
            wavelength=data["wavelength"],
            pol=data["pol"],
            signal=data["signal"],
            direction=data["direction"],
            direction_axis=data["direction_axis"],
            direction_sign=data["direction_sign"],
        )


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


def source_spec_to_dict(spec):
    if isinstance(spec, GaussianSourceSpec):
        return spec.to_dict()
    if isinstance(spec, ModeSourceSpec):
        return spec.to_dict()
    raise TypeError(f"unsupported source spec type: {type(spec).__name__}")


def source_spec_from_dict(data):
    kind = data.get("type")
    if kind == "GaussianSourceSpec":
        return GaussianSourceSpec.from_dict(data)
    if kind == "ModeSourceSpec":
        return ModeSourceSpec.from_dict(data)
    raise ValueError(f"unknown source spec type: {kind!r}")


def source_to_spec(source):
    if isinstance(source, (GaussianSourceSpec, ModeSourceSpec)):
        return source
    spec = getattr(source, "spec", None)
    if isinstance(spec, (GaussianSourceSpec, ModeSourceSpec)):
        return spec
    raise TypeError("source must be a source spec or spec-backed source facade")
