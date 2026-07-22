from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from beamz._cache_tokens import HashToken, cache_token


@runtime_checkable
class MaterialProtocol(Protocol):
    """Runtime contract accepted by designs and structures."""

    max_permittivity: float | None

    def get_sample(self, x=0, y=0, z=None) -> tuple[Any, Any, Any]: ...


def _readonly_optional_array(value):
    if value is None:
        return None
    arr = np.array(value, copy=True)
    arr.setflags(write=False)
    return arr


def _normalize_bounds(bounds):
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise ValueError(
            f"bounds must be ((x_min, x_max), (y_min, y_max)), got {bounds}"
        )
    normalized = tuple(tuple(float(v) for v in axis) for axis in bounds)
    if normalized[0][0] >= normalized[0][1]:
        raise ValueError(
            f"Invalid x bounds: x_min={normalized[0][0]} >= x_max={normalized[0][1]}"
        )
    if normalized[1][0] >= normalized[1][1]:
        raise ValueError(
            f"Invalid y bounds: y_min={normalized[1][0]} >= y_max={normalized[1][1]}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class Material:
    """Describe a homogeneous, isotropic electromagnetic material.

    Values are relative to vacuum except for electrical conductivity. Instances
    are immutable and safe to use in simulation cache keys.

    Parameters
    ----------
    permittivity : float, default=1.0
        Relative electric permittivity, εᵣ.
    permeability : float, default=1.0
        Relative magnetic permeability, μᵣ.
    conductivity : float, default=0.0
        Electrical conductivity in siemens per metre.

    Examples
    --------
    >>> import beamz as bz
    >>> silicon = bz.Material(permittivity=3.48**2)
    >>> silica = bz.Material(permittivity=1.44**2)
    """

    permittivity: float = 1.0
    permeability: float = 1.0
    conductivity: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "permittivity", float(self.permittivity))
        object.__setattr__(self, "permeability", float(self.permeability))
        object.__setattr__(self, "conductivity", float(self.conductivity))

    @property
    def max_permittivity(self) -> float:
        return self.permittivity

    def get_sample(self, x=0, y=0, z=None):
        """Return ``(permittivity, permeability, conductivity)``."""
        del x, y, z
        return self.permittivity, self.permeability, self.conductivity

    def copy(self):
        """Return this immutable material unchanged."""
        return self

    def updated_copy(self, **changes):
        """Return a material with selected fields functionally replaced.

        Parameters
        ----------
        **changes
            Material field names and replacement values.
        """
        return replace(self, **changes)


@dataclass(frozen=True, slots=True, eq=False)
class CustomMaterial:
    """Describe a spatially varying electromagnetic material.

    Callable-backed materials require an explicit semantic ``cache_key``. The
    callable itself is deliberately excluded from equality and cache identity:
    Python closures cannot be fingerprinted reliably from their code object or
    qualified name. Callers must create a new material with a new key whenever
    callable behavior changes.

    Parameters
    ----------
    permittivity_func : callable, optional
        Function returning relative permittivity at ``(x, y)`` or ``(x, y, z)``.
    permeability_func : callable, optional
        Function returning relative permeability at a spatial coordinate.
    conductivity_func : callable, optional
        Function returning conductivity in siemens per metre at a coordinate.
    cache_key : hashable, optional
        Semantic identity for callable behavior. Required when any material
        function is provided and must change when that behavior changes.
    permittivity_grid : array-like, optional
        Read-only 2D samples of relative permittivity.
    permeability_grid : array-like, optional
        Read-only 2D samples of relative permeability.
    conductivity_grid : array-like, optional
        Read-only 2D samples of conductivity in siemens per metre.
    bounds : tuple, optional
        Spatial grid bounds ``((x_min, x_max), (y_min, y_max))`` in metres.
    interpolation : str, default="linear"
        SciPy interpolation method used for grid-backed properties.
    default_permittivity : float, default=1.0
        Relative permittivity outside a sampled grid.
    default_permeability : float, default=1.0
        Relative permeability outside a sampled grid.
    default_conductivity : float, default=0.0
        Conductivity outside a sampled grid, in siemens per metre.
    max_permittivity : float, optional
        Upper bound used by automatic meshing. Inferred for grid-backed and
        constant materials; required for callable permittivity when using
        :meth:`GridSpec.auto <beamz.design.meshing.GridSpec.auto>`.

    Raises
    ------
    TypeError
        If a configured function is not callable or lacks ``cache_key``.
    ValueError
        If bounds are malformed or non-increasing.

    Examples
    --------
    >>> import beamz as bz
    >>> graded = bz.CustomMaterial(
    ...     permittivity_func=lambda x, y: 2.0 + x / bz.um,
    ...     cache_key=("linear-x", 1),
    ...     max_permittivity=3.0,
    ... )
    """

    permittivity_func: Any = None
    permeability_func: Any = None
    conductivity_func: Any = None
    cache_key: HashToken | None = None
    permittivity_grid: Any = field(default=None, repr=False, compare=False, hash=False)
    permeability_grid: Any = field(default=None, repr=False, compare=False, hash=False)
    conductivity_grid: Any = field(default=None, repr=False, compare=False, hash=False)
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    interpolation: str = "linear"
    default_permittivity: float = 1.0
    default_permeability: float = 1.0
    default_conductivity: float = 0.0
    max_permittivity: float | None = None
    _permittivity_interpolator: Any = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        metadata={"beamz_cache": False},
    )
    _permeability_interpolator: Any = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        metadata={"beamz_cache": False},
    )
    _conductivity_interpolator: Any = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        metadata={"beamz_cache": False},
    )

    def __post_init__(self):
        property_names = ("permittivity", "permeability", "conductivity")
        configured_functions = tuple(
            f"{name}_func"
            for name in property_names
            if getattr(self, f"{name}_func") is not None
        )
        for name in configured_functions:
            value = getattr(self, name)
            if not callable(value):
                raise TypeError(f"CustomMaterial.{name} must be callable or None.")
        if configured_functions and self.cache_key is None:
            raise TypeError(
                "Callable-backed CustomMaterial requires an explicit cache_key=... "
                "that changes whenever the callable's physical behavior changes."
            )
        object.__setattr__(
            self,
            "cache_key",
            None if self.cache_key is None else cache_token(self.cache_key),
        )
        grids = tuple(
            _readonly_optional_array(getattr(self, f"{name}_grid"))
            for name in property_names
        )
        for name, grid in zip(property_names, grids, strict=True):
            object.__setattr__(self, f"{name}_grid", grid)
        object.__setattr__(self, "bounds", _normalize_bounds(self.bounds))
        object.__setattr__(self, "interpolation", str(self.interpolation).lower())
        object.__setattr__(
            self, "default_permittivity", float(self.default_permittivity)
        )
        object.__setattr__(
            self, "default_permeability", float(self.default_permeability)
        )
        object.__setattr__(
            self, "default_conductivity", float(self.default_conductivity)
        )
        if self.bounds is None and any(grid is not None for grid in grids):
            raise ValueError("Grid-backed CustomMaterial requires bounds=...")

        inferred_max = None
        if self.permittivity_func is None:
            inferred_max = self.default_permittivity
            permittivity_grid = grids[0]
            if permittivity_grid is not None:
                inferred_max = max(
                    inferred_max, float(np.max(np.real(permittivity_grid)))
                )
        max_permittivity = self.max_permittivity
        if max_permittivity is None:
            max_permittivity = inferred_max
        if max_permittivity is not None:
            max_permittivity = float(max_permittivity)
            if not np.isfinite(max_permittivity) or max_permittivity <= 0.0:
                raise ValueError("max_permittivity must be positive and finite.")
            if inferred_max is not None and max_permittivity < inferred_max:
                raise ValueError(
                    "max_permittivity must include sampled and default permittivity."
                )
        object.__setattr__(self, "max_permittivity", max_permittivity)

        for name, grid in zip(property_names, grids, strict=True):
            if grid is not None:
                object.__setattr__(
                    self, f"_{name}_interpolator", self._grid_interpolator(name)
                )

    @property
    def permittivity(self):
        """Return a compact description of the permittivity model."""
        if self.permittivity_grid is not None:
            return f"grid({np.min(self.permittivity_grid):.3f}-{np.max(self.permittivity_grid):.3f})"
        if self.permittivity_func is not None:
            return "function"
        return self.default_permittivity

    @property
    def permeability(self):
        """Return a compact description of the permeability model."""
        if self.permeability_grid is not None:
            return f"grid({np.min(self.permeability_grid):.3f}-{np.max(self.permeability_grid):.3f})"
        if self.permeability_func is not None:
            return "function"
        return self.default_permeability

    @property
    def conductivity(self):
        """Return a compact description of the conductivity model."""
        if self.conductivity_grid is not None:
            return f"grid({np.min(self.conductivity_grid):.3f}-{np.max(self.conductivity_grid):.3f})"
        if self.conductivity_func is not None:
            return "function"
        return self.default_conductivity

    def _grid_interpolator(self, property_name):
        from scipy.interpolate import RegularGridInterpolator

        grid = getattr(self, f"{property_name}_grid")
        if grid is None:
            return None
        bounds = self.bounds
        assert bounds is not None
        x_coords = np.linspace(bounds[0][0], bounds[0][1], grid.shape[1])
        y_coords = np.linspace(bounds[1][0], bounds[1][1], grid.shape[0])
        return RegularGridInterpolator(
            (y_coords, x_coords),
            grid,
            method=self.interpolation,
            bounds_error=False,
            fill_value=getattr(self, f"default_{property_name}"),
        )

    def get_permittivity(self, x, y, z=None):
        """Evaluate relative permittivity at a physical coordinate.

        Parameters
        ----------
        x, y : float or array-like
            Cartesian coordinates in metres.
        z : float or array-like, optional
            Cartesian z coordinate in metres. Supplying it calls a configured
            material function with three coordinates.
        """
        if self.permittivity_func is not None:
            return (
                self.permittivity_func(x, y, z)
                if z is not None
                else self.permittivity_func(x, y)
            )
        if self._permittivity_interpolator is not None:
            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._permittivity_interpolator(points)
        return self.default_permittivity

    def get_permeability(self, x, y, z=None):
        """Evaluate relative permeability at a physical coordinate.

        Parameters
        ----------
        x, y : float or array-like
            Cartesian coordinates in metres.
        z : float or array-like, optional
            Cartesian z coordinate in metres. Supplying it calls a configured
            material function with three coordinates.
        """
        if self.permeability_func is not None:
            return (
                self.permeability_func(x, y, z)
                if z is not None
                else self.permeability_func(x, y)
            )
        if self._permeability_interpolator is not None:
            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._permeability_interpolator(points)
        return self.default_permeability

    def get_conductivity(self, x, y, z=None):
        """Evaluate conductivity at a physical coordinate.

        Parameters
        ----------
        x, y : float or array-like
            Cartesian coordinates in metres.
        z : float or array-like, optional
            Cartesian z coordinate in metres. Supplying it calls a configured
            material function with three coordinates.
        """
        if self.conductivity_func is not None:
            return (
                self.conductivity_func(x, y, z)
                if z is not None
                else self.conductivity_func(x, y)
            )
        if self._conductivity_interpolator is not None:
            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._conductivity_interpolator(points)
        return self.default_conductivity

    def get_sample(self, x=0, y=0, z=None):
        """Evaluate all electromagnetic properties at a coordinate.

        Parameters
        ----------
        x, y : float or array-like, default=0
            Cartesian coordinates in metres.
        z : float or array-like, optional
            Cartesian z coordinate in metres.
        """
        return (
            self.get_permittivity(x, y, z),
            self.get_permeability(x, y, z),
            self.get_conductivity(x, y, z),
        )

    def canonical_spec(self):
        """Return the immutable values defining cache identity."""
        function_fields = tuple(
            name
            for name in (
                "permittivity_func",
                "permeability_func",
                "conductivity_func",
            )
            if getattr(self, name) is not None
        )
        return (
            ("function_fields", function_fields),
            ("function_cache_key", self.cache_key),
            ("permittivity_grid", self.permittivity_grid),
            ("permeability_grid", self.permeability_grid),
            ("conductivity_grid", self.conductivity_grid),
            ("bounds", self.bounds),
            ("interpolation", self.interpolation),
            ("default_permittivity", self.default_permittivity),
            ("default_permeability", self.default_permeability),
            ("default_conductivity", self.default_conductivity),
            ("max_permittivity", self.max_permittivity),
        )

    def _identity_token(self):
        return cache_token(self.canonical_spec())

    def __eq__(self, other):
        if not isinstance(other, CustomMaterial):
            return NotImplemented
        return self._identity_token() == other._identity_token()

    def __hash__(self):
        return hash(self._identity_token())

    def update_grid(self, property_name, new_grid):
        """Return a material with one sampled property grid replaced.

        Parameters
        ----------
        property_name : {"permittivity", "permeability", "conductivity"}
            Electromagnetic property whose sampled grid is replaced.
        new_grid : array-like
            New 2D samples in ``(y, x)`` order. The result owns a read-only copy.
        """
        if property_name not in {"permittivity", "permeability", "conductivity"}:
            raise ValueError(f"Unknown property: {property_name}")
        return replace(self, **{f"{property_name}_grid": new_grid})

    def updated_copy(self, **changes):
        """Return a material with selected configuration fields replaced.

        Parameters
        ----------
        **changes
            Configuration field names and replacement values. Replacing a
            callable also requires a new explicit ``cache_key``.
        """
        function_fields = {
            "permittivity_func",
            "permeability_func",
            "conductivity_func",
        }
        if function_fields.intersection(changes) and "cache_key" not in changes:
            raise TypeError(
                "Updating CustomMaterial callable fields requires an explicit "
                "cache_key=... for the new callable behavior."
            )
        return replace(self, **changes)

    def copy(self):
        """Return this immutable custom material unchanged."""
        return self


# ================================

# PoleResidue: A dispersive medium described by the pole-residue pair model.

# Lorentz: A dispersive medium described by the Lorentz model.

# Sellmeier: A dispersive medium described by the Sellmeier model.

# Drude: A dispersive medium described by the Drude model.

# Debye: A dispersive medium described by the Debye model.
