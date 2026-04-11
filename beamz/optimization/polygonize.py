from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import GeometryCollection
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from skimage.measure import find_contours

from beamz.design.structures import Polygon


@dataclass
class _ContourNode:
    polygon: ShapelyPolygon
    children: list[int] = field(default_factory=list)


def _iter_shapely_polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_shapely_polygons(part)


def _build_parent_map(polygons):
    parents = [None] * len(polygons)
    for idx, polygon in enumerate(polygons):
        point = polygon.representative_point()
        best_parent = None
        best_area = None
        for parent_idx in range(idx + 1, len(polygons)):
            candidate = polygons[parent_idx]
            if not candidate.covers(point):
                continue
            area = candidate.area
            if best_area is None or area < best_area:
                best_parent = parent_idx
                best_area = area
        parents[idx] = best_parent
    return parents


def _build_occupied_geometry(nodes, node_idx):
    geometry = nodes[node_idx].polygon
    if not nodes[node_idx].children:
        return geometry
    child_geometry = unary_union(
        [
            _build_occupied_geometry(nodes, child_idx)
            for child_idx in nodes[node_idx].children
        ]
    )
    return geometry.difference(child_geometry).buffer(0)


def density_to_shapely_geometry(
    density,
    *,
    level=0.5,
    x0=0.0,
    y0=0.0,
    dx=1.0,
    dy=None,
    min_area=0.0,
):
    array = np.asarray(density, dtype=float)
    if array.ndim != 2:
        raise ValueError("density_to_shapely_geometry expects a 2D array")

    dy = float(dx if dy is None else dy)
    dx = float(dx)
    x0 = float(x0)
    y0 = float(y0)
    min_area = float(max(0.0, min_area))

    padded = np.pad(array, 1, mode="constant", constant_values=level - 1.0)
    contours = find_contours(
        padded,
        level=level,
        fully_connected="low",
        positive_orientation="low",
    )

    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        coords = [
            (x0 + (col - 0.5) * dx, y0 + (row - 0.5) * dy)
            for row, col in contour
        ]
        geometry = ShapelyPolygon(coords).buffer(0)
        for polygon in _iter_shapely_polygons(geometry):
            if polygon.area >= min_area and len(polygon.exterior.coords) >= 4:
                polygons.append(polygon)

    if not polygons:
        return GeometryCollection()

    polygons.sort(key=lambda polygon: polygon.area)
    parents = _build_parent_map(polygons)
    nodes = [_ContourNode(polygon=polygon) for polygon in polygons]
    roots = []
    for idx, parent_idx in enumerate(parents):
        if parent_idx is None:
            roots.append(idx)
        else:
            nodes[parent_idx].children.append(idx)

    geometry = unary_union(
        [_build_occupied_geometry(nodes, root_idx) for root_idx in roots]
    ).buffer(0)
    if min_area <= 0.0:
        return geometry

    filtered_parts = [
        polygon for polygon in _iter_shapely_polygons(geometry) if polygon.area >= min_area
    ]
    if not filtered_parts:
        return GeometryCollection()
    return unary_union(filtered_parts).buffer(0)


def shapely_geometry_to_polygons(geometry, *, material, min_area=0.0):
    polygons = []
    min_area = float(max(0.0, min_area))
    for polygon in _iter_shapely_polygons(geometry):
        if polygon.area < min_area:
            continue
        interiors = []
        for ring in polygon.interiors:
            hole = ShapelyPolygon(ring)
            if hole.area >= min_area:
                interiors.append(list(ring.coords[:-1]))
        polygons.append(
            Polygon(
                vertices=list(polygon.exterior.coords[:-1]),
                interiors=interiors,
                material=material,
            )
        )
    return polygons


def density_to_polygons(
    density,
    *,
    material,
    level=0.5,
    x0=0.0,
    y0=0.0,
    dx=1.0,
    dy=None,
    min_area=0.0,
):
    geometry = density_to_shapely_geometry(
        density,
        level=level,
        x0=x0,
        y0=y0,
        dx=dx,
        dy=dy,
        min_area=min_area,
    )
    return shapely_geometry_to_polygons(
        geometry,
        material=material,
        min_area=min_area,
    )
