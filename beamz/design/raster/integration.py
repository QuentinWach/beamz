"""BeamZ ``Design`` integration for the native rasterization engine."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from beamz._helpers import env_bool
from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
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
    resolution: float | RectilinearGrid,
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

    kind = _grid_kind(design, grid_type)
    if isinstance(resolution, RectilinearGrid):
        native_grid = resolution
        if resolution_z is not None:
            raise ValueError("resolution_z cannot be combined with a realized grid.")
        dimensions = 3 if kind == "3d" else 2
        if dimensions == 2 and native_grid.shape[2] != 1:
            raise ValueError("A 2D design requires exactly one z cell.")
        if not np.allclose(native_grid.origin, (0.0, 0.0, 0.0), rtol=0.0, atol=0.0):
            raise ValueError("Design raster grids must start at the design origin.")
        design_extent = (
            (float(design.width), float(design.height), float(design.depth))
            if dimensions == 3
            else (float(design.width), float(design.height), native_grid.extent[2])
        )
        for axis, (actual, required) in enumerate(
            zip(native_grid.extent, design_extent, strict=True)
        ):
            tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(required))
            padding = actual - required
            maximum_padding = float(np.max(native_grid.cell_widths(axis)))
            if padding < -tolerance or padding >= maximum_padding + tolerance:
                raise ValueError(
                    "Design raster grids must cover the design with less than one "
                    "terminal cell of padding per active axis."
                )
        representative_resolution = min(native_grid.min_spacings[:dimensions])
        if dimensions == 3:
            scene = from_beamz(design, padded_size=native_grid.extent)
        else:
            scene = from_beamz(
                design,
                two_dimensional_depth=native_grid.extent[2],
                padded_size=(native_grid.extent[0], native_grid.extent[1], 0.0),
            )
    else:
        resolution = float(resolution)
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Resolution must be finite and positive.")
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
                "Scalar design rasterization requires the same x, y, and z spacing."
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
        representative_resolution = resolution

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
        resolution=representative_resolution,
    )
