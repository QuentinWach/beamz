"""Solver-neutral rectilinear grid cases shared by BeamZ and Tidy3D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import beamz as bz

TIDY3D_REFERENCE_VERSION = "2.12.0"


@dataclass(frozen=True, slots=True)
class RectangleSpec:
    lower: tuple[float, float]
    size: tuple[float, float]
    refractive_index: float


@dataclass(frozen=True, slots=True)
class RingSpec:
    center: tuple[float, float]
    inner_radius: float
    outer_radius: float
    refractive_index: float


@dataclass(frozen=True, slots=True)
class OverrideSpec:
    center: tuple[float, float]
    size: tuple[float, float]
    dl: tuple[float | None, float | None]
    enforced: bool = False


@dataclass(frozen=True, slots=True)
class GridParityCase:
    name: str
    size: tuple[float, float]
    background_index: float
    wavelength: float = 1.55
    min_steps_per_wvl: float = 10.0
    max_scale: float = 1.2
    min_steps_per_sim_size: float = 10.0
    rectangles: tuple[RectangleSpec, ...] = ()
    rings: tuple[RingSpec, ...] = ()
    overrides: tuple[OverrideSpec, ...] = ()
    snapping_points: tuple[tuple[float | None, float | None], ...] = ()


def parity_cases() -> tuple[GridParityCase, ...]:
    """Return representative material, gap, ring, override, and snap cases."""
    n_clad, n_core = 1.444, 2.04
    ring_width, ring_height = 16.0, 13.0
    waveguide_width, gap, radius, bus_y = 0.6, 0.25, 4.0, 2.5
    ring_center = (
        ring_width / 2,
        bus_y + waveguide_width / 2 + gap + radius + waveguide_width / 2,
    )
    gap_center_y = bus_y + waveguide_width / 2 + gap / 2
    return (
        GridParityCase(
            name="homogeneous",
            size=(4.0, 3.0),
            background_index=n_clad,
        ),
        GridParityCase(
            name="high_index_rectangle",
            size=(6.0, 3.0),
            background_index=n_clad,
            rectangles=(RectangleSpec((2.0, 1.0), (2.0, 1.0), 3.48),),
        ),
        GridParityCase(
            name="coupled_rectangles",
            size=(6.0, 4.0),
            background_index=n_clad,
            rectangles=(
                RectangleSpec((0.8, 1.5), (1.7, 1.0), n_core),
                RectangleSpec((2.65, 1.5), (2.55, 1.0), n_core),
            ),
        ),
        GridParityCase(
            name="ring_bus_override",
            size=(ring_width, ring_height),
            background_index=n_clad,
            rectangles=(
                RectangleSpec(
                    (0.0, bus_y - waveguide_width / 2),
                    (ring_width, waveguide_width),
                    n_core,
                ),
            ),
            rings=(
                RingSpec(
                    ring_center,
                    radius - waveguide_width / 2,
                    radius + waveguide_width / 2,
                    n_core,
                ),
            ),
            overrides=(
                OverrideSpec(
                    (ring_center[0], gap_center_y),
                    (2.0, 1.5),
                    (0.02, None),
                ),
            ),
        ),
        GridParityCase(
            name="snapped_rectangle",
            size=(5.0, 3.0),
            background_index=n_clad,
            rectangles=(RectangleSpec((1.25, 0.8), (2.2, 1.4), n_core),),
            snapping_points=((1.111, None), (None, 2.357)),
        ),
        GridParityCase(
            name="enforced_coarse_override",
            size=(5.0, 3.0),
            background_index=n_clad,
            rectangles=(RectangleSpec((1.0, 0.8), (3.0, 1.4), 3.48),),
            overrides=(OverrideSpec((2.5, 1.5), (1.0, 1.0), (0.2, None), True),),
        ),
    )


def _material(index: float) -> bz.Material:
    return bz.Material(permittivity=float(index) ** 2)


def beamz_grid(case: GridParityCase) -> bz.RectilinearGrid:
    """Build one case with BeamZ's automatic graded mesher."""
    design = bz.Design(
        width=case.size[0] * bz.um,
        height=case.size[1] * bz.um,
        background=_material(case.background_index),
    )
    for rectangle in case.rectangles:
        design += bz.Rectangle(
            position=tuple(value * bz.um for value in rectangle.lower),
            width=rectangle.size[0] * bz.um,
            height=rectangle.size[1] * bz.um,
            material=_material(rectangle.refractive_index),
        )
    for ring in case.rings:
        design += bz.Ring(
            position=tuple(value * bz.um for value in ring.center),
            inner_radius=ring.inner_radius * bz.um,
            outer_radius=ring.outer_radius * bz.um,
            material=_material(ring.refractive_index),
        )
    overrides = tuple(
        bz.MeshOverride(
            center=tuple(value * bz.um for value in override.center),
            size=tuple(value * bz.um for value in override.size),
            dl=tuple(None if value is None else value * bz.um for value in override.dl),
            enforced=override.enforced,
        )
        for override in case.overrides
    )
    snapping_points = tuple(
        tuple(None if value is None else value * bz.um for value in point)
        for point in case.snapping_points
    )
    return bz.GridSpec.graded(
        wavelength=case.wavelength * bz.um,
        min_steps_per_wvl=case.min_steps_per_wvl,
        max_scale=case.max_scale,
        min_steps_per_sim_size=case.min_steps_per_sim_size,
        overrides=overrides,
        snapping_points=snapping_points,
    ).realize(design)


def tidy3d_edges(case: GridParityCase, td: Any) -> dict[str, np.ndarray]:
    """Build one case with a caller-provided Tidy3D module."""
    offset = np.asarray(case.size) / 2
    background = td.Medium(permittivity=case.background_index**2)
    structures = []
    for rectangle in case.rectangles:
        center = np.asarray(rectangle.lower) + np.asarray(rectangle.size) / 2 - offset
        structures.append(
            td.Structure(
                geometry=td.Box(
                    center=(*center, 0.0),
                    size=(*rectangle.size, td.inf),
                ),
                medium=td.Medium(permittivity=rectangle.refractive_index**2),
            )
        )
    for ring in case.rings:
        center = np.asarray(ring.center) - offset
        structures.extend(
            (
                td.Structure(
                    geometry=td.Cylinder(
                        center=(*center, 0.0),
                        radius=ring.outer_radius,
                        length=td.inf,
                        axis=2,
                    ),
                    medium=td.Medium(permittivity=ring.refractive_index**2),
                ),
                td.Structure(
                    geometry=td.Cylinder(
                        center=(*center, 0.0),
                        radius=ring.inner_radius,
                        length=td.inf,
                        axis=2,
                    ),
                    medium=background,
                ),
            )
        )
    overrides = []
    for override in case.overrides:
        center = np.asarray(override.center) - offset
        overrides.append(
            td.MeshOverrideStructure(
                geometry=td.Box(
                    center=(*center, 0.0),
                    size=(*override.size, td.inf),
                ),
                dl=(*override.dl, None),
                enforce=override.enforced,
                shadow=override.enforced,
            )
        )
    snapping_points = tuple(
        (
            None if point[0] is None else point[0] - offset[0],
            None if point[1] is None else point[1] - offset[1],
            None,
        )
        for point in case.snapping_points
    )
    simulation = td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=(*case.size, 0.0),
        medium=background,
        structures=structures,
        grid_spec=td.GridSpec.auto(
            wavelength=case.wavelength,
            min_steps_per_wvl=case.min_steps_per_wvl,
            max_scale=case.max_scale,
            min_steps_per_sim_size=case.min_steps_per_sim_size,
            override_structures=overrides,
            snapping_points=snapping_points,
        ),
        boundary_spec=td.BoundarySpec.all_sides(td.PECBoundary()),
        run_time=1e-12,
    )
    return {
        axis: np.asarray(getattr(simulation.grid.boundaries, axis), dtype=np.float64)
        - float(getattr(simulation.grid.boundaries, axis)[0])
        for axis in "xy"
    }


def normalized_spacing_profile(edges: np.ndarray, samples: int = 2001) -> np.ndarray:
    """Sample a piecewise-constant cell-width profile on normalized position."""
    values = np.asarray(edges, dtype=np.float64)
    normalized = (values - values[0]) / (values[-1] - values[0])
    sample_points = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    indices = np.searchsorted(normalized[1:], sample_points, side="right")
    indices = np.minimum(indices, values.size - 2)
    return np.diff(values)[indices] / (values[-1] - values[0])


__all__ = [
    "GridParityCase",
    "TIDY3D_REFERENCE_VERSION",
    "beamz_grid",
    "normalized_spacing_profile",
    "parity_cases",
    "tidy3d_edges",
]
