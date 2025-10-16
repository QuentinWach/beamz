from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Mapping

import numpy as np

Axis = Literal[0, 1, 2]
DirectionSign = Literal["+", "-"]
FieldComponent = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]


def _as_complex_field(array: np.ndarray) -> np.ndarray:
    """Return a copy of the input coerced to complex128 for safe arithmetic."""
    return np.asarray(array, dtype=np.complex128)


def _component_labels(prefix: str) -> tuple[str, str, str]:
    return (f"{prefix}x", f"{prefix}y", f"{prefix}z")


@dataclass(frozen=True)
class TFSFPlaneSource:
    """Total-field/scattered-field representation of fields on a plane."""

    electric: np.ndarray  # shape (3, ...)
    magnetic: np.ndarray  # shape (3, ...)
    axis: Axis
    direction: DirectionSign

    def __post_init__(self) -> None:
        object.__setattr__(self, "electric", _as_complex_field(self.electric))
        object.__setattr__(self, "magnetic", _as_complex_field(self.magnetic))

        if self.electric.shape != self.magnetic.shape:
            raise ValueError(
                f"E and H must share the same shape, got {self.electric.shape} and {self.magnetic.shape}"
            )
        if self.electric.ndim < 1 or self.electric.shape[0] != 3:
            raise ValueError("E and H must expose three vector components along axis 0")
        if self.axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {self.axis}")
        if self.direction not in ("+", "-"):
            raise ValueError(f"direction must be '+' or '-', got {self.direction!r}")

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "+" else -1.0

    def _normal_vector(self) -> np.ndarray:
        normal = np.zeros(3, dtype=np.float64)
        normal[self.axis] = self.sign
        reshape = (3,) + (1,) * (self.electric.ndim - 1)
        return normal.reshape(reshape)

    def _cross_with_normal(self, field: np.ndarray) -> np.ndarray:
        """Compute n × field along the first axis."""
        normal = self._normal_vector()
        return np.cross(normal, field, axisa=0, axisb=0, axisc=0)

    def _surface_currents(self) -> Mapping[str, np.ndarray]:
        electric_labels = _component_labels("E")
        magnetic_labels = _component_labels("H")

        current_electric = self._cross_with_normal(self.magnetic)
        current_magnetic = -self._cross_with_normal(self.electric)

        currents: Dict[str, np.ndarray] = {}
        for idx, name in enumerate(electric_labels):
            component = name[-1].upper()
            currents[f"J{component}"] = current_electric[idx]
        for idx, name in enumerate(magnetic_labels):
            component = name[-1].upper()
            currents[f"M{component}"] = current_magnetic[idx]
        return currents

    def electric_updates(self) -> Dict[FieldComponent, np.ndarray]:
        """Return current densities mapped to electric field components."""
        currents = self._surface_currents()
        updates: Dict[FieldComponent, np.ndarray] = {}
        for comp in ("x", "y", "z"):
            key = f"J{comp.upper()}"
            if key in currents:
                updates[f"E{comp}"] = currents[key]
        return updates

    def magnetic_updates(self) -> Dict[FieldComponent, np.ndarray]:
        """Return magnetic sheet currents mapped to magnetic field components."""
        currents = self._surface_currents()
        updates: Dict[FieldComponent, np.ndarray] = {}
        for comp in ("x", "y", "z"):
            key = f"M{comp.upper()}"
            if key in currents:
                updates[f"H{comp}"] = currents[key]
        return updates

    def poynting_density(self) -> np.ndarray:
        """Compute 0.5 Re(E × H*) dotted with the normal direction."""
        cross = np.cross(self.electric, np.conjugate(self.magnetic), axisa=0, axisb=0, axisc=0)
        density = 0.5 * np.real(cross[self.axis])
        return density * self.sign

    def total_power(self, cell_area: float = 1.0) -> float:
        """Return the integrated power through the plane."""
        return float(np.sum(self.poynting_density()) * cell_area)


__all__ = ["TFSFPlaneSource"]
