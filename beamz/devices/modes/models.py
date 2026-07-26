"""Internal validated inputs for the native mode solver."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index
from typing import Literal, SupportsIndex, cast

import numpy as np

from beamz.devices._immutable import readonly_array

BoundaryCondition = Literal["pec", "pmc"]


@dataclass(frozen=True)
class PmlSpec:
    num_cells: tuple[int, int] = (0, 0)
    sigma_max: float = 2.0
    kappa_min: float = 1.0
    kappa_max: float = 3.0
    order: int = 3

    def __post_init__(self) -> None:
        if len(self.num_cells) != 2:
            raise ValueError("num_cells must contain two non-negative integers")
        object.__setattr__(
            self,
            "num_cells",
            tuple(_integer("num_cells", value, 0) for value in self.num_cells),
        )
        for name in ("sigma_max", "kappa_min", "kappa_max"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.kappa_max < self.kappa_min:
            raise ValueError("kappa_max must be greater than or equal to kappa_min")
        object.__setattr__(self, "order", _integer("order", self.order, 1))

    @classmethod
    def from_num_cells(cls, num_cells: tuple[int, int]) -> PmlSpec:
        return cls(num_cells=num_cells)

    def as_dict(self):
        return {
            "num_cells": self.num_cells,
            "sigma_max": self.sigma_max,
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
            "order": self.order,
        }

    def profile_dict(self):
        return {
            "sigma_max": self.sigma_max,
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
            "order": self.order,
        }


@dataclass(frozen=True)
class BoundarySpec:
    low: tuple[BoundaryCondition, BoundaryCondition] = ("pec", "pec")

    def __post_init__(self) -> None:
        if len(self.low) != 2:
            raise ValueError("low must contain two boundary conditions")
        low = tuple(str(value).lower() for value in self.low)
        if set(low).difference({"pec", "pmc"}):
            raise ValueError("boundary conditions must be 'pec' or 'pmc'")
        object.__setattr__(self, "low", low)

    @property
    def dmin_pmc(self) -> tuple[bool, bool]:
        return self.low[0] == "pmc", self.low[1] == "pmc"

    @property
    def dmin_pml(self) -> tuple[bool, bool]:
        return self.low[0] == "pec", self.low[1] == "pec"

    def as_dict(self):
        return {"low": self.low}


@dataclass(frozen=True)
class Grid:
    x_edges: tuple[float, ...]
    y_edges: tuple[float, ...]
    normal_axis: Literal[0, 1, 2] = 2
    normal_coordinate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x_edges", "y_edges"):
            values = tuple(float(value) for value in getattr(self, name))
            array = np.asarray(values)
            if len(values) < 2:
                raise ValueError(f"{name} must contain at least two values")
            if not np.all(np.isfinite(array)) or np.any(np.diff(array) <= 0):
                raise ValueError(f"{name} must be finite and strictly increasing")
            object.__setattr__(self, name, values)
        if self.normal_axis not in {0, 1, 2}:
            raise ValueError("normal_axis must be 0, 1, or 2")

    @property
    def shape(self):
        return len(self.x_edges) - 1, len(self.y_edges) - 1


@dataclass(frozen=True)
class Materials:
    grid: Grid
    eps_tensor: np.ndarray
    mu_tensor: np.ndarray

    def __post_init__(self) -> None:
        eps = readonly_array(self.eps_tensor, dtype=np.complex128)
        mu = readonly_array(self.mu_tensor, dtype=np.complex128)
        expected = (3, 3, *self.grid.shape)
        if eps.shape != expected or mu.shape != expected:
            raise ValueError(f"material tensors must have shape {expected}")
        if not np.all(np.isfinite(eps)) or not np.all(np.isfinite(mu)):
            raise ValueError("material tensors must contain finite values")
        object.__setattr__(self, "eps_tensor", eps)
        object.__setattr__(self, "mu_tensor", mu)

    @classmethod
    def from_components(
        cls,
        *,
        x_edges: Sequence[float],
        y_edges: Sequence[float],
        eps_xx: np.ndarray,
        eps_yy: np.ndarray | None = None,
        eps_zz: np.ndarray | None = None,
        eps_xy: np.ndarray | None = None,
        eps_xz: np.ndarray | None = None,
        eps_yx: np.ndarray | None = None,
        eps_yz: np.ndarray | None = None,
        eps_zx: np.ndarray | None = None,
        eps_zy: np.ndarray | None = None,
        mu_xx: np.ndarray | None = None,
        mu_yy: np.ndarray | None = None,
        mu_zz: np.ndarray | None = None,
        mu_xy: np.ndarray | None = None,
        mu_xz: np.ndarray | None = None,
        mu_yx: np.ndarray | None = None,
        mu_yz: np.ndarray | None = None,
        mu_zx: np.ndarray | None = None,
        mu_zy: np.ndarray | None = None,
        normal_axis: Literal[0, 1, 2] = 2,
        normal_coordinate: float = 0.0,
    ) -> Materials:
        grid = Grid(
            tuple(x_edges), tuple(y_edges), normal_axis, float(normal_coordinate)
        )
        eps = _tensor(
            "eps",
            grid.shape,
            (
                eps_xx,
                eps_xx if eps_yy is None else eps_yy,
                eps_xx if eps_zz is None else eps_zz,
            ),
            (eps_xy, eps_xz, eps_yx, eps_yz, eps_zx, eps_zy),
        )
        ones = np.ones(grid.shape, dtype=np.complex128)
        mu = _tensor(
            "mu",
            grid.shape,
            (
                ones if mu_xx is None else mu_xx,
                ones if mu_yy is None else mu_yy,
                ones if mu_zz is None else mu_zz,
            ),
            (mu_xy, mu_xz, mu_yx, mu_yz, mu_zx, mu_zy),
        )
        return cls(grid, eps, mu)

    @property
    def shape(self):
        return self.grid.shape

    @property
    def is_diagonal(self):
        off_diagonal = ~np.eye(3, dtype=bool)
        return bool(
            np.all(np.abs(self.eps_tensor[off_diagonal]) <= 1e-12)
            and np.all(np.abs(self.mu_tensor[off_diagonal]) <= 1e-12)
        )

    def flat_eps_tensor(self):
        return self.eps_tensor.reshape(3, 3, -1)

    def flat_mu_tensor(self):
        return self.mu_tensor.reshape(3, 3, -1)


def _tensor(label, shape, diagonal, off_diagonal):
    tensor = np.zeros((3, 3, *shape), dtype=np.complex128)
    for axis, values in enumerate(diagonal):
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != shape:
            raise ValueError(f"{label}_{'xyz'[axis] * 2} must have shape {shape}")
        tensor[axis, axis] = array
    for (row, col), suffix, values in zip(
        ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
        ("xy", "xz", "yx", "yz", "zx", "zy"),
        off_diagonal,
        strict=True,
    ):
        if values is None:
            continue
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != shape:
            raise ValueError(f"{label}_{suffix} must have shape {shape}")
        tensor[row, col] = array
    return tensor


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must contain integers")
    try:
        result = index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if result < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return int(result)
