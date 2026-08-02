"""BeamZ ``Design`` integration for the native rasterization engine."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from beamz._helpers import env_bool
from beamz.design.discretization import MaterialGrid
from beamz.lattice import normalize_polarization_2d

from .engine import RasterOptions, rasterize
from .importers._beamz import from_beamz
from .schema import Grid


def _cell_count(extent: float, spacing: float) -> int:
    ratio = float(extent) / float(spacing)
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(ratio))
    return max(1, math.ceil(ratio - tolerance))


def _grid_kind(design: Any, grid_type: str) -> str:
    if not isinstance(grid_type, str):
        raise TypeError("grid_type must be 'auto', '2d', or '3d'.")
    value = grid_type.lower()
    if value == "2d":
        return "2d"
    if value == "3d":
        return "3d"
    if value == "auto":
        return "3d" if design.is_3d else "2d"
    raise ValueError(f"Unknown grid_type {grid_type!r}.")


def _rasterize_design(
    design: Any,
    resolution: float,
    *,
    grid_type: str = "auto",
    force_recompute: bool = False,
    progress: bool = False,
    resolution_z: float | None = None,
    quality: str = "balanced",
    smoothing: str = "farjadpour_diagonal",
    polarization: str = "tm",
    cache_directory: str | Path | None = None,
):
    """Rasterize a BeamZ design into its immutable solver material grid.

    Simulation conversion retains compatible staggered Yee arrays on a uniform
    grid. Mode paths that cannot reproduce those supports reject them explicitly.
    Standalone rasterization also supports nonuniform grids. Spatial coefficient
    arrays enter simulation through ``MaterialGrid`` without geometry resampling.
    """

    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Resolution must be finite and positive.")
    kind = _grid_kind(design, grid_type)
    resolution_z = resolution if resolution_z is None else float(resolution_z)
    if not np.isfinite(resolution_z) or resolution_z <= 0:
        raise ValueError("resolution_z must be finite and positive.")

    if kind == "3d" and not math.isclose(
        resolution_z,
        resolution,
        rel_tol=32.0 * np.finfo(float).eps,
        abs_tol=0.0,
    ):
        raise ValueError(
            "BeamZ's current FDTD solver requires the same spacing on x, y, and z. "
            "Use the standalone rasterizer for nonuniform grids."
        )

    nx = _cell_count(design.width, resolution)
    ny = _cell_count(design.height, resolution)
    if kind == "3d":
        nz = _cell_count(design.depth, resolution_z)
        z_max = nz * resolution_z
        scene = from_beamz(
            design,
            padded_size=(nx * resolution, ny * resolution, z_max),
        )
    else:
        nz = 1
        z_max = 1.0
        scene = from_beamz(
            design,
            two_dimensional_depth=z_max,
            padded_size=(nx * resolution, ny * resolution, 0.0),
        )
    native_grid = Grid.uniform(
        (0.0, 0.0, 0.0),
        (nx * resolution, ny * resolution, z_max),
        (nx, ny, nz),
    )

    if progress:
        print(
            f"Rasterizing structures ({len(design.structures)}) with native Rust engine"
        )
    if cache_directory is None and env_bool("BEAMZ_RASTER_CACHE", False):
        cache_directory = (
            Path(os.getenv("BEAMZ_RASTER_CACHE_DIR", ".beamz_cache/raster")) / "native"
        )
    if force_recompute:
        cache_directory = None

    polarization = normalize_polarization_2d(polarization)
    result = rasterize(
        scene,
        native_grid,
        options=RasterOptions(
            quality=quality,
            smoothing=smoothing,
            components="all" if kind == "3d" else f"two_dimensional_{polarization}",
        ),
        cache_directory=cache_directory,
    )
    return MaterialGrid.from_raster_result(
        result,
        dimensions=3 if kind == "3d" else 2,
        polarization=polarization,
    )
