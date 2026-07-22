"""Immutable boundary-condition specifications consumed by simulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from beamz.const import µm

BoundaryEdges = tuple[str, ...] | str


def normalize_edges(edges: BoundaryEdges) -> BoundaryEdges:
    """Freeze an edge selection while retaining the dimensional ``"all"`` sentinel."""
    if edges == "all":
        return "all"
    return tuple(edges) if isinstance(edges, (list, tuple, set)) else (str(edges),)


def edges_for_dimension(edges: BoundaryEdges, is_3d: bool) -> tuple[str, ...]:
    """Resolve an edge selection after the compiler knows the domain dimension."""
    if edges != "all":
        return tuple(edges)
    planar = ("left", "right", "top", "bottom")
    return (*planar, "front", "back") if is_3d else planar


@dataclass(frozen=True, slots=True)
class PEC:
    """Request perfect-electric-conductor behavior on domain edges.

    Parameters
    ----------
    edges : "all", str, or sequence of str, default="all"
        Boundary edges to constrain, such as ``"left"`` or ``("top", "bottom")``.
    thickness : float, default=0.0
        Compatibility field. PEC boundaries are enforced at zero thickness.

    Examples
    --------
    >>> boundary = PEC(edges=("top", "bottom"))

    Notes
    -----
    PEC forces tangential electric fields to zero on the selected domain faces and
    is reflective. Use ``PML`` or ``Absorber`` for open-domain truncation.
    """

    edges: BoundaryEdges = "all"
    thickness: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", normalize_edges(self.edges))
        object.__setattr__(self, "thickness", 0.0)

    def updated_copy(self, **changes):
        """Return a validated boundary with selected fields replaced.

        Parameters
        ----------
        **changes : object
            Dataclass fields to replace.

        Returns
        -------
        PEC
            New immutable boundary specification.

        Raises
        ------
        TypeError or ValueError
            If a field is unknown or a replacement is invalid.
        """
        return replace(self, **changes)

    def _get_edges_for_dimensionality(self, is_3d):
        """Compatibility adapter for older low-level callers."""
        return list(edges_for_dimension(self.edges, bool(is_3d)))


@dataclass(frozen=True, slots=True)
class PML:
    """Request a graded absorbing layer on selected domain edges.

    Parameters
    ----------
    edges : "all", str, or sequence of str, default="all"
        Domain faces covered by the layer.
    thickness : float, default=1 um
        Physical layer thickness in metres.
    sigma_max : float, optional
        Explicit maximum conductivity. When omitted, BeamZ derives it from
        ``target_reflection`` and the layer thickness.
    m : int, default=3
        Polynomial grading order.
    formulation : {"sponge", "cpml"}, default="sponge"
        Absorber formulation. CPML adds convolutional recurrence state and is
        generally preferred for broadband or oblique incidence.
    kappa_max : float, default=2.0
        Maximum CPML coordinate-stretching factor.
    alpha_max : float, optional
        Maximum CPML complex-frequency-shift coefficient.
    target_reflection : float, default=1e-6
        Reflection target used to derive automatic conductivity.

    Examples
    --------
    >>> boundary = PML(thickness=1e-6, formulation="cpml")

    Notes
    -----
    Geometry should normally be kept clear of the absorber or extruded through it
    along the boundary normal to avoid material discontinuities inside the layer.
    """

    _DEFAULT_CPML_ALPHA_NORMALIZED: ClassVar[float] = 0.1
    _DEFAULT_3D_CPML_ALPHA_NORMALIZED: ClassVar[float] = 0.05

    edges: BoundaryEdges = "all"
    thickness: float = 1 * µm
    sigma_max: float | None = None
    m: int = 3
    formulation: str = "sponge"
    kappa_max: float = 2.0
    alpha_max: float | None = None
    target_reflection: float = 1e-6

    def __post_init__(self) -> None:
        thickness = float(self.thickness)
        if not np.isfinite(thickness) or thickness < 0.0:
            raise ValueError("PML thickness must be a non-negative finite value.")
        formulation = str(self.formulation).lower()
        if formulation not in {"sponge", "cpml"}:
            raise ValueError(
                f"Unsupported boundary formulation {self.formulation!r}. "
                "Expected one of: 'sponge', 'cpml'."
            )
        object.__setattr__(self, "edges", normalize_edges(self.edges))
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "sigma_max", None if self.sigma_max is None else float(self.sigma_max))  # fmt: skip
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "formulation", formulation)
        object.__setattr__(self, "kappa_max", float(self.kappa_max))
        object.__setattr__(self, "alpha_max", None if self.alpha_max is None else float(self.alpha_max))  # fmt: skip
        object.__setattr__(self, "target_reflection", float(self.target_reflection))

    def updated_copy(self, **changes):
        """Return a validated boundary with selected fields replaced.

        Parameters
        ----------
        **changes : object
            Dataclass fields to replace.

        Returns
        -------
        PML
            New immutable absorbing-boundary specification.

        Raises
        ------
        TypeError or ValueError
            If a field is unknown or a replacement is invalid.
        """
        return replace(self, **changes)

    def _get_edges_for_dimensionality(self, is_3d):
        """Compatibility adapter for older low-level callers."""
        return list(edges_for_dimension(self.edges, bool(is_3d)))


@dataclass(frozen=True, slots=True)
class Absorber:
    """Request a graded-conductivity sponge on selected domain edges.

    Parameters
    ----------
    edges : "all", str, or sequence of str, default="all"
        Domain faces covered by the sponge.
    thickness : float, default=1 um
        Physical sponge thickness in metres.
    sigma_max : float, optional
        Explicit maximum conductivity. When omitted, BeamZ derives it from the
        reflection target.
    m : int, default=3
        Polynomial conductivity-grading order.
    target_reflection : float, default=1e-6
        Reflection target used to derive automatic conductivity.

    Examples
    --------
    >>> boundary = Absorber(edges=("left", "right"), thickness=1e-6)

    Notes
    -----
    ``Absorber`` is the explicit sponge-only counterpart of
    ``PML(formulation="sponge")``. Use CPML when convolutional matching is needed.
    """

    formulation: ClassVar[str] = "sponge"
    # Neutral CPML values let the shared profile compiler remain data-driven; they are
    # never used by the sponge branch.
    kappa_max: ClassVar[float] = 1.0
    alpha_max: ClassVar[None] = None
    _DEFAULT_CPML_ALPHA_NORMALIZED: ClassVar[float] = 0.0
    _DEFAULT_3D_CPML_ALPHA_NORMALIZED: ClassVar[float] = 0.0

    edges: BoundaryEdges = "all"
    thickness: float = 1 * µm
    sigma_max: float | None = None
    m: int = 3
    target_reflection: float = 1e-6

    def __post_init__(self) -> None:
        thickness = float(self.thickness)
        if not np.isfinite(thickness) or thickness < 0.0:
            raise ValueError("Absorber thickness must be a non-negative finite value.")
        object.__setattr__(self, "edges", normalize_edges(self.edges))
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "sigma_max", None if self.sigma_max is None else float(self.sigma_max))  # fmt: skip
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "target_reflection", float(self.target_reflection))

    def updated_copy(self, **changes):
        """Return a validated boundary with selected fields replaced.

        Parameters
        ----------
        **changes : object
            Dataclass fields to replace.

        Returns
        -------
        Absorber
            New immutable sponge specification.

        Raises
        ------
        TypeError or ValueError
            If a field is unknown or a replacement is invalid.
        """
        return replace(self, **changes)

    def _get_edges_for_dimensionality(self, is_3d):
        """Compatibility adapter for older low-level callers."""
        return list(edges_for_dimension(self.edges, bool(is_3d)))


Boundary = PEC | PML | Absorber


def normalize_boundaries(boundaries) -> tuple[Boundary, ...]:
    """Freeze boundary input and preserve Beamz's historical all-PEC default."""
    resolved = tuple(boundaries) if boundaries else (PEC(),)
    if not all(isinstance(boundary, (PEC, PML, Absorber)) for boundary in resolved):
        raise TypeError(
            "boundaries must contain only PEC, PML, or Absorber specifications"
        )
    return resolved


__all__ = ["Absorber", "Boundary", "PEC", "PML"]
