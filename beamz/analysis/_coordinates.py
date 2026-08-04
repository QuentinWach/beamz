"""Shared coordinate reconstruction for detached monitor analysis."""

from __future__ import annotations

import numpy as np

from beamz.devices._placement import SnappedRegion, snap_plane_region_grid
from beamz.lattice import common_grid_shape_3d, yee_plane_coordinates_3d


def monitor_plane_coordinates_3d(
    metadata,
    monitor,
    *,
    region: SnappedRegion | None = None,
) -> tuple[SnappedRegion, np.ndarray, np.ndarray]:
    """Reconstruct the exact compiled coordinates for a 3D monitor plane."""
    grid = getattr(metadata, "grid", None)
    if grid is not None:
        region = region or snap_plane_region_grid(
            center=monitor.center,
            size=monitor.size,
            plane_normal=monitor.plane_normal,
            grid=grid,
        )
        coord0, coord1 = yee_plane_coordinates_3d(
            monitor.center,
            monitor.size,
            monitor.plane_normal,
            region,
            grid=grid,
        )
    else:
        if region is None:
            fields = getattr(metadata, "fields", None)
            base_shape = common_grid_shape_3d(fields)
            resolution = float(metadata.resolution)
            region = monitor.get_snapped_region(
                dx=resolution,
                dy=resolution,
                dz=resolution,
                field_shape=base_shape,
            )
            if region is None:
                raise ValueError("3D analysis plane requires a snapped monitor region.")
        coord0, coord1 = yee_plane_coordinates_3d(
            monitor.center,
            monitor.size,
            monitor.plane_normal,
            region,
        )
    return region, np.asarray(coord0, dtype=float), np.asarray(coord1, dtype=float)


__all__ = ["monitor_plane_coordinates_3d"]
