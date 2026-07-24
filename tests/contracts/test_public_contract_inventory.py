"""Behavioral contracts shared by every public configuration object."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass
from types import ModuleType

import numpy as np
import pytest

import beamz as bz
from tests.contracts.public_api_registry import (
    CONFIGURATION_CASES,
    CONSTANT_EXPORTS,
    FUNCTION_EXPORTS,
    MODULE_EXPORTS,
    RUNTIME_EXPORTS,
    registered_export_names,
)


def _assert_canonical_immutable(value, *, path: str) -> None:
    if isinstance(value, np.ndarray):
        assert not value.flags.writeable, f"{path} contains a writeable array"
        return
    assert not isinstance(value, list | dict | set), (
        f"{path} contains mutable {type(value).__name__}"
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_canonical_immutable(key, path=f"{path}.<key>")
            _assert_canonical_immutable(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, tuple | frozenset):
        for index, item in enumerate(value):
            _assert_canonical_immutable(item, path=f"{path}[{index}]")
        return
    if is_dataclass(value):
        for item in fields(value):
            _assert_canonical_immutable(
                getattr(value, item.name), path=f"{path}.{item.name}"
            )


@pytest.mark.contract
def test_top_level_registry_classifies_every_export_exactly_once():
    registered = registered_export_names()
    assert len(registered) == len(set(registered)), "public API categories overlap"
    assert set(registered) == set(bz.__all__)


@pytest.mark.contract
def test_registry_categories_match_runtime_kinds():
    assert all(
        isinstance(getattr(bz, name), ModuleType) for name in MODULE_EXPORTS
    )
    assert all(callable(getattr(bz, name)) for name in FUNCTION_EXPORTS)
    assert all(isinstance(getattr(bz, name), type) for name in RUNTIME_EXPORTS)
    assert all(not callable(getattr(bz, name)) for name in CONSTANT_EXPORTS)


@pytest.mark.contract
@pytest.mark.parametrize("case", CONFIGURATION_CASES, ids=lambda case: case.name)
def test_public_configuration_is_frozen_and_canonically_immutable(case):
    public_type = getattr(bz, case.name)
    instance = case.factory()

    assert type(instance) is public_type
    assert is_dataclass(instance)
    assert public_type.__dataclass_params__.frozen

    first_field = fields(instance)[0]
    with pytest.raises(FrozenInstanceError):
        setattr(instance, first_field.name, getattr(instance, first_field.name))

    _assert_canonical_immutable(instance, path=case.name)


@pytest.mark.contract
@pytest.mark.parametrize("case", CONFIGURATION_CASES, ids=lambda case: case.name)
def test_public_configuration_copy_preserves_canonical_immutability(case):
    instance = case.factory()

    updated_copy = getattr(instance, "updated_copy", None)
    if updated_copy is not None:
        copied = updated_copy()
        assert type(copied) is type(instance)
        _assert_canonical_immutable(copied, path=f"{case.name}.updated_copy")
