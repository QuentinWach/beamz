from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np


def readonly_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    arr = np.array(value, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


def readonly_1d_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    return readonly_array(np.atleast_1d(np.asarray(value, dtype=dtype)), dtype=dtype)


def immutable_snapshot(value: Any) -> Any:
    """Recursively copy configuration into immutable Python/NumPy containers."""
    if value is None or isinstance(
        value, bool | int | float | complex | str | bytes | slice
    ):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {immutable_snapshot(k): immutable_snapshot(v) for k, v in value.items()}
        )
    if isinstance(value, tuple | list):
        snapshot = tuple(immutable_snapshot(v) for v in value)
        if isinstance(value, tuple) and all(
            old is new for old, new in zip(value, snapshot, strict=True)
        ):
            return value
        return snapshot
    if is_dataclass(value):
        params = getattr(type(value), "__dataclass_params__", None)
        if not getattr(params, "frozen", False):
            raise TypeError(f"{type(value).__name__} must be a frozen dataclass.")
        updates = {
            item.name: immutable_snapshot(getattr(value, item.name))
            for item in fields(value)
            if item.init
        }
        if all(getattr(value, name) is snapshot for name, snapshot in updates.items()):
            return value
        return replace(cast(Any, value), **updates)
    try:
        arr = np.asarray(value)
    except Exception as exc:
        raise TypeError(
            f"Cannot create an immutable snapshot of {type(value).__name__}."
        ) from exc
    if arr.dtype == object:
        raise TypeError(f"Cannot snapshot object-valued {type(value).__name__}.")
    return readonly_array(arr)


def canonical_tuple(value: Any, *, dtype=float) -> tuple:
    if value is None:
        return ()
    return tuple(dtype(v) for v in value)


def finite_tuple(value: Any, *, name: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value)
    if any(not np.isfinite(item) for item in values):
        raise ValueError(f"{name} must be finite.")
    return values


def nonnegative_finite_extents(value: Any, *, name: str) -> tuple[float, ...]:
    values = finite_tuple(value, name=name)
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} must contain non-negative finite extents.")
    return values


def normalize_source_signal(signal: Any, *, name: str = "signal") -> Any:
    if callable(signal):
        raise TypeError(
            f"{name} cannot be a bare callable in compiled Beamz runs. "
            "Use an immutable array or source-time specification."
        )
    if signal is None or isinstance(signal, bool | int | float | complex | str):
        return signal
    try:
        arr = np.asarray(signal)
    except Exception:
        return signal
    if arr.dtype == object:
        return signal
    return readonly_array(arr)
