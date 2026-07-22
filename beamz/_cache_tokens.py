from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import fields as dataclass_fields
from typing import Any, TypeAlias, cast

import numpy as np

PrimitiveToken: TypeAlias = bool | int | float | str | None
HashToken: TypeAlias = PrimitiveToken | tuple["HashToken", ...]


def object_kind(obj: object) -> str:
    return f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"


def array_cache_token(value: object) -> HashToken | None:
    arr = np.asarray(value)
    if arr.dtype == object:
        return None
    digest = hashlib.blake2b(
        np.ascontiguousarray(arr).tobytes(), digest_size=16
    ).hexdigest()
    return ("array", tuple(int(v) for v in arr.shape), str(arr.dtype), digest)


def cache_token(value: object) -> HashToken:
    if isinstance(value, np.generic):
        value = cast(object, value.item())
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, slice):
        return ("slice", tuple(int(v) for v in value.indices(2**63 - 1)))
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        token = array_cache_token(value)
        if token is not None:
            return token
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return tuple(
            sorted(
                ((cache_token(k), cache_token(v)) for k, v in mapping.items()),
                key=repr,
            )
        )
    if isinstance(value, tuple | list):
        return tuple(cache_token(v) for v in cast(Iterable[object], value))
    if isinstance(value, set | frozenset):
        return tuple(
            sorted(
                (cache_token(v) for v in cast(Iterable[object], value)),
                key=repr,
            )
        )
    if callable(value):
        return (
            "callable",
            str(getattr(value, "__module__", "")),
            str(getattr(value, "__qualname__", repr(value))),
        )
    for hook in ("cache_spec", "canonical_spec"):
        method = getattr(value, hook, None)
        if callable(method):
            return (object_kind(value), hook, cache_token(method()))
    fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        field_names = tuple(
            field.name
            for field in dataclass_fields(cast(Any, value))
            if not field.name.startswith("_")
            and field.metadata.get("beamz_cache", True)
        )
        return (
            object_kind(value),
            tuple(
                (name, cache_token(cast(object, getattr(value, name))))
                for name in field_names
            ),
        )
    return (object_kind(value), repr(value))
