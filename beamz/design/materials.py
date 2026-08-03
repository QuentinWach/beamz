"""Canonical material values shared by designs and imported raster scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MaterialProtocol(Protocol):
    """Runtime contract accepted by designs and structures."""

    max_permittivity: float | None

    def get_sample(self, x=0, y=0, z=None) -> tuple[Any, Any, Any]: ...


_UNSET = object()


def _tensor(
    value: Any,
    name: str,
    *,
    positive_definite: bool,
) -> tuple[float, float, float, float, float, float]:
    """Normalize scalar, diagonal, packed, or matrix constitutive input."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        scalar = float(array)
        packed = (scalar, scalar, scalar, 0.0, 0.0, 0.0)
    elif array.shape == (3,):
        packed = (float(array[0]), float(array[1]), float(array[2]), 0.0, 0.0, 0.0)
    elif array.shape == (6,):
        packed = tuple(float(component) for component in array)
    elif array.shape == (3, 3):
        if not np.allclose(array, array.T, rtol=1e-12, atol=1e-14):
            raise ValueError(f"{name} must be symmetric.")
        packed = (
            float(array[0, 0]),
            float(array[1, 1]),
            float(array[2, 2]),
            float(array[0, 1]),
            float(array[0, 2]),
            float(array[1, 2]),
        )
    else:
        raise ValueError(
            f"{name} must be a scalar, three diagonal values, six packed "
            "symmetric values, or a symmetric 3x3 matrix."
        )
    if not np.isfinite(packed).all():
        raise ValueError(f"{name} must contain finite values.")
    xx, yy, zz, xy, xz, yz = packed
    matrix = np.asarray(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 64.0 * np.finfo(float).eps * scale
    if positive_definite and np.any(eigenvalues <= tolerance):
        raise ValueError(f"{name} must be positive definite.")
    if not positive_definite and np.any(eigenvalues < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite.")
    return packed  # type: ignore[return-value]


def _compact(tensor: tuple[float, ...]):
    diagonal = tensor[:3]
    if tensor[3:] == (0.0, 0.0, 0.0):
        if diagonal[0] == diagonal[1] == diagonal[2]:
            return diagonal[0]
        return diagonal
    return tensor


@dataclass(frozen=True, init=False)
class Material:
    """Describe one homogeneous scalar or symmetric-tensor material.

    ``permittivity`` and ``permeability`` are the BeamZ design names;
    ``epsilon_r`` and ``mu_r`` are equivalent aliases accepted by raster
    workflows. Each coefficient accepts a scalar, three diagonal values, six
    packed symmetric values ``(xx, yy, zz, xy, xz, yz)``, or a symmetric 3×3
    matrix.
    """

    _epsilon_r: tuple[float, float, float, float, float, float]
    _mu_r: tuple[float, float, float, float, float, float]
    _conductivity: tuple[float, float, float, float, float, float]

    def __init__(
        self,
        permittivity: Any = _UNSET,
        permeability: Any = _UNSET,
        conductivity: Any = 0.0,
        *,
        epsilon_r: Any = _UNSET,
        mu_r: Any = _UNSET,
    ) -> None:
        if permittivity is not _UNSET and epsilon_r is not _UNSET:
            raise TypeError("Pass only one of permittivity or epsilon_r.")
        if permeability is not _UNSET and mu_r is not _UNSET:
            raise TypeError("Pass only one of permeability or mu_r.")
        epsilon = (
            1.0
            if permittivity is epsilon_r is _UNSET
            else (epsilon_r if epsilon_r is not _UNSET else permittivity)
        )
        mu = (
            1.0
            if permeability is mu_r is _UNSET
            else (mu_r if mu_r is not _UNSET else permeability)
        )
        object.__setattr__(
            self,
            "_epsilon_r",
            _tensor(epsilon, "permittivity", positive_definite=True),
        )
        object.__setattr__(
            self,
            "_mu_r",
            _tensor(mu, "permeability", positive_definite=True),
        )
        object.__setattr__(
            self,
            "_conductivity",
            _tensor(conductivity, "conductivity", positive_definite=False),
        )

    @property
    def permittivity(self):
        """Return scalar, diagonal, or packed relative permittivity."""
        return _compact(self._epsilon_r)

    @property
    def epsilon_r(self):
        """Return relative permittivity packed as xx, yy, zz, xy, xz, yz."""
        return self._epsilon_r

    @property
    def permeability(self):
        """Return scalar, diagonal, or packed relative permeability."""
        return _compact(self._mu_r)

    @property
    def mu_r(self):
        """Return relative permeability packed as xx, yy, zz, xy, xz, yz."""
        return self._mu_r

    @property
    def conductivity(self):
        """Return scalar, diagonal, or packed electrical conductivity."""
        return _compact(self._conductivity)

    @property
    def max_permittivity(self) -> float:
        """Return the largest principal relative permittivity."""
        xx, yy, zz, xy, xz, yz = self._epsilon_r
        return float(np.linalg.eigvalsh(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))[-1])

    def get_sample(self, x=0, y=0, z=None):
        """Return permittivity, permeability, and conductivity."""
        del x, y, z
        return self.permittivity, self.permeability, self.conductivity

    def to_dict(self) -> dict[str, list[float]]:
        """Return the serialized representation consumed by the Rust scene."""
        return {
            "epsilon_r": list(self._epsilon_r),
            "mu_r": list(self._mu_r),
            "conductivity": list(self._conductivity),
        }

    def cache_spec(self):
        """Return the constitutive values that determine simulation behavior."""
        return self._epsilon_r, self._mu_r, self._conductivity

    def __repr__(self) -> str:
        return (
            f"Material(permittivity={self.permittivity!r}, "
            f"permeability={self.permeability!r}, "
            f"conductivity={self.conductivity!r})"
        )

    def copy(self):
        """Return this immutable material unchanged."""
        return self

    def updated_copy(self, **changes):
        """Return a material with selected coefficients replaced."""
        allowed = {
            "permittivity",
            "permeability",
            "conductivity",
            "epsilon_r",
            "mu_r",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise TypeError(f"Unknown Material field(s): {', '.join(sorted(unknown))}.")
        values = {
            "permittivity": self.permittivity,
            "permeability": self.permeability,
            "conductivity": self.conductivity,
        }
        if "epsilon_r" in changes:
            if "permittivity" in changes:
                raise TypeError("Pass only one of permittivity or epsilon_r.")
            values["permittivity"] = changes.pop("epsilon_r")
        if "mu_r" in changes:
            if "permeability" in changes:
                raise TypeError("Pass only one of permeability or mu_r.")
            values["permeability"] = changes.pop("mu_r")
        values.update(changes)
        return type(self)(**values)
