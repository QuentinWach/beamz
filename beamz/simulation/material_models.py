"""Compiled material model interfaces for the v0.3 engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import jax.numpy as jnp


@dataclass(frozen=True)
class CompiledMaterialSpec:
    """Static material configuration consumed by compiled engine steps.

    model_kind:
        String model identifier. v0.3 ships with "linear".
    model_ids:
        Integer grid for future voxel-wise model routing.
    params:
        Optional packed model parameter tensor for future ADE/nonlinear models.
    """

    model_kind: str = "linear"
    model_ids: jnp.ndarray | None = None
    params: jnp.ndarray | None = None


class MaterialState(NamedTuple):
    """Runtime material state carried inside the scan loop."""

    aux: jnp.ndarray


class CompiledMaterialModel(Protocol):
    """Material model interface for compiled FDTD loops."""

    def init_state(self, spec: CompiledMaterialSpec) -> MaterialState:
        """Initialize per-step material state."""

    def update(
        self,
        material_state: MaterialState,
        ex: jnp.ndarray,
        ey: jnp.ndarray,
        ez: jnp.ndarray,
        t_idx: jnp.ndarray,
    ) -> tuple[MaterialState, jnp.ndarray | None]:
        """Update material state and optionally return additive polarization/current term."""


class LinearNondispersiveModel:
    """Default model: no internal dynamics, no extra source term."""

    def init_state(self, spec: CompiledMaterialSpec) -> MaterialState:
        del spec
        return MaterialState(aux=jnp.zeros((0,), dtype=jnp.float32))

    def update(
        self,
        material_state: MaterialState,
        ex: jnp.ndarray,
        ey: jnp.ndarray,
        ez: jnp.ndarray,
        t_idx: jnp.ndarray,
    ) -> tuple[MaterialState, jnp.ndarray | None]:
        del ex, ey, ez, t_idx
        return material_state, None


def create_material_model(spec: CompiledMaterialSpec) -> CompiledMaterialModel:
    """Factory for compiled material models."""
    if spec.model_kind == "linear":
        return LinearNondispersiveModel()
    raise ValueError(f"Unsupported material model kind: {spec.model_kind!r}")
