from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from ..schema import (
    ExtrudedPolygon,
    Material,
    Object,
    Polygon,
    Scene,
    TaperedExtrudedPolygon,
)

_BUILTIN_MATERIALS_NEAR_1550_NM = {
    "air": Material(1.0),
    "si": Material(3.48**2),
    "silicon": Material(3.48**2),
    "sio2": Material(1.444**2),
    "oxide": Material(1.444**2),
    "sin": Material(2.0**2),
    "si3n4": Material(2.0**2),
}


def _polygons_by_layer(component: Any) -> Mapping[tuple[int, int], Any]:
    try:
        return component.get_polygons_points(by="tuple")
    except TypeError:
        return component.get_polygons_points(by_spec=True)


def _region_polygons(
    component: Any,
    level: Any,
    polygons_by_layer: Mapping[tuple[int, int], Any],
    unit_scale: float,
) -> list[Polygon]:
    layer_expression = getattr(level, "layer", None)
    get_shapes = getattr(layer_expression, "get_shapes", None)
    if callable(get_shapes):
        region: Any = get_shapes(component)
        bias = float(getattr(level, "bias", 0.0) or 0.0)
        database_unit = float(component.kcl.dbu)
        if bias:
            region = region.sized(int(round(bias / database_unit)))
        scale = database_unit * unit_scale
        result = []
        for polygon in region.each():
            exterior = tuple(
                (float(point.x) * scale, float(point.y) * scale)
                for point in polygon.each_point_hull()
            )
            holes = tuple(
                tuple(
                    (float(point.x) * scale, float(point.y) * scale)
                    for point in polygon.each_point_hole(hole)
                )
                for hole in range(polygon.holes())
            )
            result.append(Polygon(exterior, holes))
        return result

    layer = (
        (int(layer_expression[0]), int(layer_expression[1]))
        if isinstance(layer_expression, (tuple, list)) and len(layer_expression) == 2
        else None
    )
    if layer is None or layer not in polygons_by_layer:
        return []
    return [
        Polygon(
            tuple(
                (float(x) * unit_scale, float(y) * unit_scale)
                for x, y in np.asarray(points, dtype=float)[:, :2]
            )
        )
        for points in polygons_by_layer[layer]
    ]


def _z_to_bias_profile(value: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalize tabulated bias or approximate a callable with fixed policy."""
    if callable(value):
        samples: dict[float, float] = {}
        knots = {0.0, 1.0}

        def evaluate(z: float) -> float:
            if z not in samples:
                bias = float(cast(Any, value(float(z))))
                if not np.isfinite(bias):
                    raise ValueError("z_to_bias callable returned a non-finite bias.")
                samples[z] = bias
            return samples[z]

        def refine(z0: float, z1: float, depth: int) -> None:
            b0, b1 = evaluate(z0), evaluate(z1)
            probes = (
                z0 + 0.25 * (z1 - z0),
                z0 + 0.5 * (z1 - z0),
                z0 + 0.75 * (z1 - z0),
            )
            error = max(
                abs(evaluate(z) - (b0 + (z - z0) / (z1 - z0) * (b1 - b0)))
                for z in probes
            )
            if error <= 1e-3:
                return
            if depth >= 10:
                raise ValueError(
                    "z_to_bias callable did not converge within the importer policy."
                )
            middle = probes[1]
            knots.add(middle)
            refine(z0, middle, depth + 1)
            refine(middle, z1, depth + 1)

        refine(0.0, 1.0, 0)
        z = np.asarray(sorted(knots), dtype=float)
        bias = np.asarray([samples[value] for value in z], dtype=float)
        return z, bias

    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("z_to_bias must be a (normalized_z, bias) pair or callable.")
    z = np.asarray(value[0], dtype=float)
    bias = np.asarray(value[1], dtype=float)
    if z.ndim != 1 or bias.ndim != 1 or len(z) < 2 or len(z) != len(bias):
        raise ValueError(
            "z_to_bias coordinate and bias arrays must have equal length >= 2."
        )
    if (
        not np.isfinite(z).all()
        or not np.isfinite(bias).all()
        or np.any(np.diff(z) <= 0.0)
        or not np.isclose(z[0], 0.0)
        or not np.isclose(z[-1], 1.0)
        or z[0] < 0.0
        or z[-1] > 1.0
    ):
        raise ValueError(
            "z_to_bias coordinates must be finite, strictly increasing, "
            "and span normalized z=0 through z=1."
        )
    return z, bias


def _offset_polygon(polygon: Polygon, distance: float) -> list[Polygon]:
    if distance == 0.0:
        return [polygon]
    from shapely.geometry import Polygon as ShapelyPolygon

    buffered = ShapelyPolygon(polygon.exterior, polygon.holes).buffer(distance)
    geometries = getattr(buffered, "geoms", (buffered,))
    return [
        Polygon(
            tuple((float(x), float(y)) for x, y in geometry.exterior.coords[:-1]),
            tuple(
                tuple((float(x), float(y)) for x, y in ring.coords[:-1])
                for ring in geometry.interiors
            ),
        )
        for geometry in geometries
        if not geometry.is_empty
    ]


def from_gdsfactory(
    component: Any,
    layer_stack: Any,
    *,
    material_map: Mapping[str, Material | float] | None = None,
    use_builtin_materials: bool = False,
    background: Material | None = None,
    unit_scale: float = 1e-6,
) -> Scene:
    """Convert a GDSFactory component and PDK layer stack.

    GDSFactory layout and layer-stack coordinates are conventionally expressed
    in micrometres, hence the default ``unit_scale=1e-6``. PDK material names
    must be resolved through ``material_map``. Set ``use_builtin_materials=True``
    only to opt into nondispersive Si/SiO2/SiN approximations near 1.55 µm.
    """

    unit_scale = float(unit_scale)
    if not np.isfinite(unit_scale) or unit_scale <= 0.0:
        raise ValueError("unit_scale must be finite and positive.")
    if not isinstance(use_builtin_materials, (bool, np.bool_)):
        raise TypeError("use_builtin_materials must be a boolean.")
    background = Material() if background is None else background
    material_lookup = dict(
        _BUILTIN_MATERIALS_NEAR_1550_NM if use_builtin_materials else {}
    )
    material_lookup.update(
        {
            str(name).strip().lower(): (
                value if isinstance(value, Material) else Material(float(value))
            )
            for name, value in (material_map or {}).items()
        }
    )

    polygons = _polygons_by_layer(component)
    materials = [background]
    material_ids = {background: 0}

    def add_material(material: Material) -> int:
        if material not in material_ids:
            material_ids[material] = len(materials)
            materials.append(material)
        return material_ids[material]

    objects = []
    object_id = 1
    for level_name, level in getattr(layer_stack, "layers", {}).items():
        level_polygons = _region_polygons(component, level, polygons, unit_scale)
        if not level_polygons:
            continue
        sidewall = float(getattr(level, "sidewall_angle", 0.0) or 0.0)
        material_name = str(getattr(level, "material", "")).lower()
        if material_name not in material_lookup:
            raise ValueError(
                f"No BeamZ raster material is configured for PDK material "
                f"{material_name!r} on layer {level_name!r}. Pass material_map=... "
                "with the intended wavelength-dependent model, or explicitly set "
                "use_builtin_materials=True for approximate 1.55 µm constants."
            )
        z_min = float(level.zmin) * unit_scale
        z_max = z_min + float(level.thickness) * unit_scale
        if z_max <= z_min:
            continue
        material_id = add_material(material_lookup[material_name])
        z_to_bias = getattr(level, "z_to_bias", None)
        for polygon in level_polygons:
            if z_to_bias is not None:
                normalized_z, bias = _z_to_bias_profile(z_to_bias)
                geometries = []
                for index in range(len(normalized_z) - 1):
                    segment_min = z_min + normalized_z[index] * (z_max - z_min)
                    segment_max = z_min + normalized_z[index + 1] * (z_max - z_min)
                    low_bias = bias[index] * unit_scale
                    high_bias = bias[index + 1] * unit_scale
                    angle = np.degrees(
                        np.arctan2(-(high_bias - low_bias), segment_max - segment_min)
                    )
                    for offset in _offset_polygon(polygon, low_bias):
                        geometries.append(
                            ExtrudedPolygon(offset, segment_min, segment_max)
                            if low_bias == high_bias
                            else TaperedExtrudedPolygon(
                                offset,
                                segment_min,
                                segment_max,
                                sidewall_angle_degrees=float(angle),
                            )
                        )
            else:
                geometries = [
                    TaperedExtrudedPolygon(
                        polygon,
                        z_min,
                        z_max,
                        sidewall_angle_degrees=sidewall,
                        width_to_z=float(getattr(level, "width_to_z", 0.0) or 0.0),
                    )
                    if sidewall != 0.0
                    else ExtrudedPolygon(polygon, z_min, z_max)
                ]
            for geometry in geometries:
                objects.append(
                    Object(
                        id=object_id,
                        material_id=material_id,
                        priority=-int(getattr(level, "mesh_order", 0) or 0),
                        geometry=geometry,
                    )
                )
                object_id += 1

    if not objects:
        available = sorted(polygons)
        raise ValueError(
            "No physical layer-stack levels matched component polygons. "
            f"Component layers: {available}."
        )
    return Scene(tuple(materials), tuple(objects), 0)
