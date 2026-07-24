import numpy as np
import pytest
from shapely.geometry import Point

from beamz import Design, Material
from beamz.optimization.polygonize import (
    density_to_polygons,
    density_to_shapely_geometry,
)

EPS_CORE = 3.48**2
EPS_CLAD = 1.0


def _iter_polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_polygons(part)


def test_density_to_shapely_geometry_preserves_holes_and_islands():
    density = np.zeros((60, 60), dtype=float)
    density[10:50, 10:50] = 1.0
    density[20:40, 20:40] = 0.0
    density[27:33, 27:33] = 1.0

    geometry = density_to_shapely_geometry(
        density,
        level=0.5,
        dx=1.0,
        min_area=1.0,
    )

    polygons = list(_iter_polygons(geometry))
    assert len(polygons) == 2
    assert sum(len(poly.interiors) for poly in polygons) == 1
    assert geometry.covers(Point(12.0, 12.0))
    assert not geometry.covers(Point(24.0, 24.0))
    assert geometry.covers(Point(30.0, 30.0))
    assert geometry.area == pytest.approx(40 * 40 - 20 * 20 + 6 * 6, abs=1.0)


def test_density_to_shapely_geometry_handles_boundary_touching_regions():
    density = np.zeros((40, 40), dtype=float)
    density[:, :20] = 1.0
    density[12:28, 6:14] = 0.0

    geometry = density_to_shapely_geometry(
        density,
        level=0.5,
        dx=1.0,
        min_area=1.0,
    )

    polygons = list(_iter_polygons(geometry))
    assert len(polygons) == 1
    assert len(polygons[0].interiors) == 1
    assert geometry.covers(Point(3.0, 3.0))
    assert not geometry.covers(Point(10.0, 20.0))
    assert not geometry.covers(Point(30.0, 20.0))


def test_density_to_shapely_geometry_respects_physical_offsets():
    density = np.zeros((12, 12), dtype=float)
    density[3:9, 2:7] = 1.0

    geometry = density_to_shapely_geometry(
        density,
        level=0.5,
        x0=10.0,
        y0=20.0,
        dx=0.5,
        min_area=0.25,
    )

    assert geometry.covers(Point(11.25, 22.25))
    assert not geometry.covers(Point(10.75, 21.25))
    assert not geometry.covers(Point(13.75, 22.25))


def test_density_to_polygons_rerasterizes_with_preserved_voids():
    density = np.zeros((60, 60), dtype=float)
    density[8:52, 8:52] = 1.0
    density[18:42, 18:42] = 0.0
    density[26:34, 26:34] = 1.0

    design = Design(width=60.0, height=60.0, material=Material(permittivity=EPS_CLAD))
    for polygon in density_to_polygons(
        density,
        material=Material(permittivity=EPS_CORE),
        level=0.5,
        dx=1.0,
        min_area=1.0,
    ):
        design += polygon
    design = design.unified_polygons()

    grid = design.rasterize(
        1.0,
        aa_mode="stratified_jitter",
        aa_samples=64,
        force_recompute=True,
    )

    assert grid.permittivity[12, 12] == pytest.approx(EPS_CORE)
    assert grid.permittivity[24, 24] == pytest.approx(EPS_CLAD)
    assert grid.permittivity[30, 30] == pytest.approx(EPS_CORE)


def test_cropped_density_requires_true_cell_edge_origin():
    dx = 0.5
    density = np.zeros((16, 20), dtype=float)
    density[5:11, 6:15] = 1.0

    full_geometry = density_to_shapely_geometry(
        density,
        level=0.5,
        x0=0.0,
        y0=0.0,
        dx=dx,
        min_area=0.25,
    )

    i0, i1 = 4, 12
    j0, j1 = 5, 16
    cropped = density[i0:i1, j0:j1]
    cropped_geometry = density_to_shapely_geometry(
        cropped,
        level=0.5,
        x0=j0 * dx,
        y0=i0 * dx,
        dx=dx,
        min_area=0.25,
    )

    symmetric_diff = full_geometry.symmetric_difference(cropped_geometry)
    assert symmetric_diff.area == pytest.approx(0.0, abs=1e-9)
