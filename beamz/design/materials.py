from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


_VALID_INTERPOLATIONS = {"linear", "nearest"}


def _require_finite_positive(name, value):
    value = float(value)
    if (not isfinite(value)) or value <= 0:
        raise ValueError(f"{name} must be a finite positive value, got {value!r}")
    return value


def _require_finite_nonnegative(name, value):
    value = float(value)
    if (not isfinite(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative value, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    permittivity: float = 1.0
    permeability: float = 1.0
    conductivity: float = 0.0

    def __post_init__(self):
        object.__setattr__(
            self, "permittivity", _require_finite_positive("permittivity", self.permittivity)
        )
        object.__setattr__(
            self, "permeability", _require_finite_positive("permeability", self.permeability)
        )
        object.__setattr__(
            self, "conductivity", _require_finite_nonnegative("conductivity", self.conductivity)
        )


class Material:
    """Dispersionless homogeneous material."""

    def __init__(self, permittivity=1.0, permeability=1.0, conductivity=0.0):
        object.__setattr__(
            self,
            "spec",
            MaterialSpec(
                permittivity=permittivity,
                permeability=permeability,
                conductivity=conductivity,
            ),
        )

    def __getattr__(self, name):
        spec = self.__dict__.get("spec")
        if spec is not None and hasattr(spec, name):
            return getattr(spec, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name == "spec":
            object.__setattr__(self, name, value)
            return
        raise AttributeError(f"{type(self).__name__!s} is immutable")

    def get_sample(self):
        return self.permittivity, self.permeability, self.conductivity

    def copy(self):
        return Material(
            permittivity=self.permittivity,
            permeability=self.permeability,
            conductivity=self.conductivity,
        )


@dataclass(frozen=True, slots=True)
class CustomMaterialSpec:
    permittivity_func: object = None
    permeability_func: object = None
    conductivity_func: object = None
    permittivity_grid: object = None
    permeability_grid: object = None
    conductivity_grid: object = None
    bounds: tuple | None = None
    interpolation: str = "linear"
    default_permittivity: float = 1.0
    default_permeability: float = 1.0
    default_conductivity: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "bounds", _normalize_bounds(self.bounds))
        interpolation = str(self.interpolation).lower()
        if interpolation not in _VALID_INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {sorted(_VALID_INTERPOLATIONS)}, got {self.interpolation!r}"
            )
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(
            self,
            "permittivity_grid",
            _freeze_grid(self.permittivity_grid),
        )
        object.__setattr__(
            self,
            "permeability_grid",
            _freeze_grid(self.permeability_grid),
        )
        object.__setattr__(
            self,
            "conductivity_grid",
            _freeze_grid(self.conductivity_grid),
        )
        object.__setattr__(
            self,
            "default_permittivity",
            _require_finite_positive("default_permittivity", self.default_permittivity),
        )
        object.__setattr__(
            self,
            "default_permeability",
            _require_finite_positive("default_permeability", self.default_permeability),
        )
        object.__setattr__(
            self,
            "default_conductivity",
            _require_finite_nonnegative("default_conductivity", self.default_conductivity),
        )
        for name in ("permittivity_func", "permeability_func", "conductivity_func"):
            func = getattr(self, name)
            if func is not None and not callable(func):
                raise TypeError(f"{name} must be callable when provided")
        if self.bounds is None and any(
            grid is not None
            for grid in (self.permittivity_grid, self.permeability_grid, self.conductivity_grid)
        ):
            raise ValueError("bounds are required when using grid-backed custom materials")


@dataclass(slots=True)
class CustomMaterialState:
    permittivity_interpolator: object = None
    permeability_interpolator: object = None
    conductivity_interpolator: object = None


def _normalize_bounds(bounds):
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise ValueError(f"bounds must be ((x_min, x_max), (y_min, y_max)), got {bounds}")
    x0, x1 = bounds[0]
    y0, y1 = bounds[1]
    if x0 >= x1:
        raise ValueError(f"Invalid x bounds: x_min={x0} >= x_max={x1}")
    if y0 >= y1:
        raise ValueError(f"Invalid y bounds: y_min={y0} >= y_max={y1}")
    return ((float(x0), float(x1)), (float(y0), float(y1)))


def _freeze_grid(grid):
    if grid is None:
        return None
    arr = np.asarray(grid).copy()
    if arr.ndim != 2:
        raise ValueError("material grids must be 2D arrays")
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("material grids must be non-empty and finite")
    arr.setflags(write=False)
    return arr


class CustomMaterial:
    """Spatially varying material with immutable config and mutable interpolator cache."""

    _SPEC_FIELDS = frozenset(CustomMaterialSpec.__dataclass_fields__.keys())
    _STATE_MAP = {
        "_permittivity_interpolator": "permittivity_interpolator",
        "_permeability_interpolator": "permeability_interpolator",
        "_conductivity_interpolator": "conductivity_interpolator",
    }

    def __init__(
        self,
        permittivity_func=None,
        permeability_func=None,
        conductivity_func=None,
        permittivity_grid=None,
        permeability_grid=None,
        conductivity_grid=None,
        bounds=None,
        interpolation="linear",
    ):
        object.__setattr__(
            self,
            "spec",
            CustomMaterialSpec(
                permittivity_func=permittivity_func,
                permeability_func=permeability_func,
                conductivity_func=conductivity_func,
                permittivity_grid=permittivity_grid,
                permeability_grid=permeability_grid,
                conductivity_grid=conductivity_grid,
                bounds=bounds,
                interpolation=interpolation,
            ),
        )
        object.__setattr__(self, "state", CustomMaterialState())
        self._rebuild_interpolators()

    def __getattr__(self, name):
        spec = self.__dict__.get("spec")
        if spec is not None and hasattr(spec, name):
            return getattr(spec, name)
        state = self.__dict__.get("state")
        if state is not None and name in self._STATE_MAP:
            return getattr(state, self._STATE_MAP[name])
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in {"spec", "state"}:
            object.__setattr__(self, name, value)
            return
        if name in self._STATE_MAP and "state" in self.__dict__:
            setattr(self.state, self._STATE_MAP[name], value)
            return
        raise AttributeError(
            f"{type(self).__name__!s} configuration is immutable; use update_grid(...)"
        )

    @property
    def permittivity(self):
        if self.permittivity_grid is not None:
            return f"grid({np.min(self.permittivity_grid):.3f}-{np.max(self.permittivity_grid):.3f})"
        if self.permittivity_func is not None:
            return "function"
        return self.default_permittivity

    @property
    def permeability(self):
        if self.permeability_grid is not None:
            return f"grid({np.min(self.permeability_grid):.3f}-{np.max(self.permeability_grid):.3f})"
        if self.permeability_func is not None:
            return "function"
        return self.default_permeability

    @property
    def conductivity(self):
        if self.conductivity_grid is not None:
            return f"grid({np.min(self.conductivity_grid):.3f}-{np.max(self.conductivity_grid):.3f})"
        if self.conductivity_func is not None:
            return "function"
        return self.default_conductivity

    def _create_grid_interpolator(self, property_name):
        try:
            from scipy.interpolate import RegularGridInterpolator

            grid = getattr(self.spec, f"{property_name}_grid")
            if grid is None or self.bounds is None:
                return None

            x_coords = np.linspace(self.bounds[0][0], self.bounds[0][1], grid.shape[1])
            y_coords = np.linspace(self.bounds[1][0], self.bounds[1][1], grid.shape[0])
            return RegularGridInterpolator(
                (y_coords, x_coords),
                grid,
                method=self.interpolation,
                bounds_error=False,
                fill_value=getattr(self.spec, f"default_{property_name}"),
            )
        except ImportError:
            print("Warning: scipy not available, using nearest neighbor interpolation")
            return None

    def _rebuild_interpolators(self):
        self.state.permittivity_interpolator = self._create_grid_interpolator("permittivity")
        self.state.permeability_interpolator = self._create_grid_interpolator("permeability")
        self.state.conductivity_interpolator = self._create_grid_interpolator("conductivity")

    def get_permittivity(self, x, y, z=None):
        if self.permittivity_func is not None:
            return self.permittivity_func(x, y, z) if z is not None else self.permittivity_func(x, y)
        if self.state.permittivity_interpolator is not None:
            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self.state.permittivity_interpolator(points)
        return self.default_permittivity

    def get_permeability(self, x, y, z=None):
        if self.permeability_func is not None:
            return self.permeability_func(x, y, z) if z is not None else self.permeability_func(x, y)
        if self.state.permeability_interpolator is not None:
            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self.state.permeability_interpolator(points)
        return self.default_permeability

    def get_conductivity(self, x, y, z=None):
        if self.conductivity_func is not None:
            return self.conductivity_func(x, y, z) if z is not None else self.conductivity_func(x, y)
        if self.state.conductivity_interpolator is not None:
            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self.state.conductivity_interpolator(points)
        return self.default_conductivity

    def get_sample(self, x=0, y=0, z=None):
        return (
            self.get_permittivity(x, y, z),
            self.get_permeability(x, y, z),
            self.get_conductivity(x, y, z),
        )

    def update_grid(self, property_name, new_grid):
        if property_name not in {"permittivity", "permeability", "conductivity"}:
            raise ValueError(f"Unknown property: {property_name}")
        spec = self.spec
        object.__setattr__(
            self,
            "spec",
            CustomMaterialSpec(
                permittivity_func=spec.permittivity_func,
                permeability_func=spec.permeability_func,
                conductivity_func=spec.conductivity_func,
                permittivity_grid=(
                    new_grid if property_name == "permittivity" else spec.permittivity_grid
                ),
                permeability_grid=(
                    new_grid if property_name == "permeability" else spec.permeability_grid
                ),
                conductivity_grid=(
                    new_grid if property_name == "conductivity" else spec.conductivity_grid
                ),
                bounds=spec.bounds,
                interpolation=spec.interpolation,
                default_permittivity=spec.default_permittivity,
                default_permeability=spec.default_permeability,
                default_conductivity=spec.default_conductivity,
            ),
        )
        self._rebuild_interpolators()

    def copy(self):
        return CustomMaterial(
            permittivity_func=self.permittivity_func,
            permeability_func=self.permeability_func,
            conductivity_func=self.conductivity_func,
            permittivity_grid=(
                None if self.permittivity_grid is None else np.asarray(self.permittivity_grid).copy()
            ),
            permeability_grid=(
                None if self.permeability_grid is None else np.asarray(self.permeability_grid).copy()
            ),
            conductivity_grid=(
                None if self.conductivity_grid is None else np.asarray(self.conductivity_grid).copy()
            ),
            bounds=self.bounds,
            interpolation=self.interpolation,
        )
