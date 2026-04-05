from __future__ import annotations

from typing import Any

import jax
import numpy as np


def to_host(value: Any, *, dtype=None, copy: bool = False) -> np.ndarray:
    """Convert a device or Python value to a host NumPy array."""
    arr = np.asarray(jax.device_get(value), dtype=dtype)
    return arr.copy() if copy else arr


def to_scalar(value: Any, cast=None):
    """Extract a Python scalar from a device or Python value."""
    scalar = to_host(value).item()
    return cast(scalar) if cast is not None else scalar


def stack_host(values: list[Any], *, empty_shape: tuple[int, ...] = (0,), dtype=None) -> np.ndarray:
    """Stack host arrays, returning an empty array when no values were recorded."""
    if not values:
        return np.zeros(empty_shape, dtype=dtype)
    arrays = [to_host(value, dtype=dtype) for value in values]
    return np.stack(arrays)
