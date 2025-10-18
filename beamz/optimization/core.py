"""Optimization utilities for topology updates.

This module provides a small wrapper that keeps track of optimizer state
across iterations.  The intent is to hide method-specific bookkeeping from the
application layer so that examples can simply call ``step(gradient)`` and feed
the returned update back into their design variables.

Only the Adam optimizer is implemented at the moment, but the structure allows
additional optimizers to be registered transparently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, MutableMapping, Optional

import numpy as np


UpdateFn = Callable[[np.ndarray, np.ndarray, MutableMapping[str, np.ndarray]], np.ndarray]


def _adam_update(
    grad: np.ndarray,
    learning_rate: float,
    state: MutableMapping[str, np.ndarray],
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Single Adam update step.

    Args:
        grad: Gradient array (same shape as parameters).
        learning_rate: Scalar step size.
        state: Mutable mapping storing ``m`` (first moment), ``v`` (second
            moment) and ``t`` (step counter).

    Returns:
        Update array to be added to the parameters.
    """

    if "m" not in state:
        state["m"] = np.zeros_like(grad)
    if "v" not in state:
        state["v"] = np.zeros_like(grad)
    state["t"] = state.get("t", 0) + 1

    m = state["m"]
    v = state["v"]
    t = state["t"]

    m *= beta1
    m += (1.0 - beta1) * grad

    v *= beta2
    v += (1.0 - beta2) * (grad ** 2)

    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)

    return -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


_OPTIMIZER_REGISTRY: Dict[str, UpdateFn] = {
    "adam": _adam_update,
}


@dataclass
class Optimizer:
    """General optimizer wrapper supporting multiple methods.

    Example
    -------
    >>> opt = Optimizer(method="adam", learning_rate=1e-2)
    >>> update = opt.step(gradient)
    >>> density += update
    """

    method: str = "adam"
    learning_rate: float = 1e-2
    options: Mapping[str, float] = field(default_factory=dict)
    _state: MutableMapping[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        method_key = self.method.lower()
        if method_key not in _OPTIMIZER_REGISTRY:
            available = ", ".join(sorted(_OPTIMIZER_REGISTRY))
            raise ValueError(f"Unknown optimizer method '{self.method}'. Available: {available}.")
        self._update_fn = _OPTIMIZER_REGISTRY[method_key]

    def step(self, gradient: np.ndarray) -> np.ndarray:
        """Compute the parameter update for a given gradient."""

        if not isinstance(gradient, np.ndarray):
            raise TypeError("Gradient must be a numpy.ndarray")
        if gradient.size == 0:
            return np.zeros_like(gradient)

        return self._update_fn(gradient, self.learning_rate, self._state, **self.options)

    def reset(self) -> None:
        """Reset internal optimizer state."""

        self._state.clear()


__all__ = ["Optimizer"]


