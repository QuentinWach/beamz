from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Any, Iterable

from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

from beamz.const import BLUE, GREEN, ORANGE, PURPLE, RED
from beamz.design.core import (
    _find_rings_to_preserve,
    _material_key as _design_material_key,
    _shapely_to_polygons,
    _to_shapely,
)

from ._scene import CameraSpec, ClipPlaneSpec, MaterialSpec, Object3D, SceneSpec


_STRUCTURE_PALETTE = (BLUE, RED, GREEN, ORANGE, PURPLE)


@dataclass(slots=True)
class _Bounds:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            (self.min_z + self.max_z) / 2.0,
        )

    @property
    def size(self) -> tuple[float, float, float]:
        return (
            max(self.max_x - self.min_x, 1e-12),
            max(self.max_y - self.min_y, 1e-12),
            max(self.max_z - self.min_z, 1e-12),
        )

    @property
    def diagonal(self) -> float:
        sx, sy, sz = self.size
        return (sx * sx + sy * sy + sz * sz) ** 0.5


def looks_like_beamz_design(value: Any) -> bool:
    return all(
        hasattr(value, name)
        for name in ("structures", "sources", "monitors", "width", "height")
    )


def looks_like_beamz_simulation(value: Any) -> bool:
    return hasattr(value, "design") and looks_like_beamz_design(value.design)


def beamz_to_scene(value: Any) -> SceneSpec:
    if looks_like_beamz_simulation(value):
        return simulation_to_scene(value)
    if looks_like_beamz_design(value):
        return design_to_scene(value)
    raise TypeError("Expected a BeamZ-like design or simulation object.")


def design_to_scene(design: Any) -> SceneSpec:
    bounds = _design_bounds(design)
    objects: list[Object3D] = [_domain_box(design, bounds)]
    objects.extend(_structure_objects(design))
    objects.extend(_monitor_objects(getattr(design, "monitors", [])))
    objects.extend(_source_objects(getattr(design, "sources", [])))
    return _build_scene(
        title="BEAMZ Design",
        bounds=bounds,
        objects=objects,
        metadata={
            "object_type": type(design).__name__,
            "is_3d": bool(getattr(design, "is_3d", False)),
            "num_structures": len(getattr(design, "structures", [])),
            "num_sources": len(getattr(design, "sources", [])),
            "num_monitors": len(getattr(design, "monitors", [])),
        },
    )


def simulation_to_scene(simulation: Any) -> SceneSpec:
    design = getattr(simulation, "design", None)
    if design is None or not looks_like_beamz_design(design):
        raise TypeError(
            "simulation_to_scene() expects a BeamZ simulation with a design."
        )

    bounds = _design_bounds(design)
    objects: list[Object3D] = [_domain_box(design, bounds)]
    objects.extend(_structure_objects(design))
    objects.extend(_monitor_objects(_simulation_monitors(simulation)))
    objects.extend(_source_objects(_simulation_sources(simulation)))
    objects.extend(_boundary_objects(simulation, bounds))
    objects.extend(_simulation_planes(simulation, bounds))

    metadata = {
        "object_type": type(simulation).__name__,
        "resolution": getattr(simulation, "resolution", None),
        "is_3d": bool(getattr(simulation, "is_3d", False)),
        "plane_2d": getattr(simulation, "plane_2d", "xy"),
        "dt": getattr(simulation, "dt", None),
        "num_steps": getattr(simulation, "num_steps", None),
        "num_devices": len(getattr(simulation, "devices", [])),
        "num_boundaries": len(getattr(simulation, "boundaries", [])),
    }
    return _build_scene(
        title="BEAMZ Simulation Setup",
        bounds=bounds,
        objects=objects,
        metadata=metadata,
    )


def _design_bounds(design: Any) -> _Bounds:
    width = float(getattr(design, "width", 1.0))
    height = float(getattr(design, "height", 1.0))
    depth = float(getattr(design, "depth", 0.0) or 0.0)
    max_z = depth if depth > 0 else max(min(width, height) * 0.1, 1e-9)
    return _Bounds(0.0, 0.0, 0.0, width, height, max_z)


def _material_spec(structure: Any, color: str) -> MaterialSpec:
    material = getattr(structure, "material", None)
    permittivity = (
        getattr(material, "permittivity", 1.0) if material is not None else 1.0
    )
    opacity = 0.0 if _is_air_like_material(material) else 1.0
    wireframe = bool(getattr(structure, "is_pml", False))
    return MaterialSpec(color=color, opacity=opacity, wireframe=wireframe)


def _structure_metadata(structure: Any) -> dict[str, Any]:
    material = getattr(structure, "material", None)
    metadata = {
        "kind": "structure",
        "type": type(structure).__name__,
        "depth": getattr(structure, "depth", 0.0),
        "z": getattr(structure, "z", 0.0),
    }
    if material is not None:
        metadata["material"] = {
            "permittivity": getattr(material, "permittivity", None),
            "permeability": getattr(material, "permeability", None),
            "conductivity": getattr(material, "conductivity", None),
        }
    return metadata


def _structure_objects(design: Any) -> list[Object3D]:
    objects: list[Object3D] = []
    color_by_material_key: dict[tuple[Any, ...], str] = {}
    merged_structures = _merged_structures_for_view(getattr(design, "structures", []))
    for index, structure in enumerate(merged_structures):
        material_key = _design_material_key(getattr(structure, "material", None))
        color = color_by_material_key.get(material_key)
        if color is None:
            color = _get_deterministic_color(len(color_by_material_key))
            color_by_material_key[material_key] = color
        label = (
            f"PML {index + 1}"
            if bool(getattr(structure, "is_pml", False))
            else f"{type(structure).__name__} {index + 1}"
        )
        vertices = getattr(structure, "vertices", None) or []
        interiors = getattr(structure, "interiors", None) or []
        depth = float(getattr(structure, "depth", 0.0) or 0.0)
        z0 = float(getattr(structure, "z", 0.0) or 0.0)
        if vertices:
            geometry = {
                "vertices": [[float(x), float(y)] for x, y, *_ in vertices],
                "holes": [
                    [[float(x), float(y)] for x, y, *_ in hole]
                    for hole in interiors
                    if hole
                ],
                "depth": depth,
                "z0": z0,
            }
            objects.append(
                Object3D(
                    kind="poly_extrusion",
                    label=label,
                    geometry=geometry,
                    material=_material_spec(structure, str(color)),
                    metadata={
                        **_structure_metadata(structure),
                        "material_key": list(_structure_material_key(structure)),
                    },
                )
            )
            continue

        if hasattr(structure, "position") and hasattr(structure, "radius"):
            px, py, *rest = getattr(structure, "position")
            pz = rest[0] if rest else z0 + depth / 2.0
            objects.append(
                Object3D(
                    kind="sphere",
                    label=label,
                    geometry={
                        "center": [float(px), float(py), float(pz)],
                        "radius": float(getattr(structure, "radius")),
                    },
                    material=_material_spec(structure, str(color)),
                    metadata={
                        **_structure_metadata(structure),
                        "material_key": list(_structure_material_key(structure)),
                    },
                )
            )
            continue

        if (
            hasattr(structure, "position")
            and hasattr(structure, "width")
            and hasattr(structure, "height")
        ):
            px, py, *rest = getattr(structure, "position")
            pz = rest[0] if rest else z0
            geometry = {
                "center": [
                    float(px) + float(getattr(structure, "width")) / 2.0,
                    float(py) + float(getattr(structure, "height")) / 2.0,
                    float(pz) + depth / 2.0,
                ],
                "size": [
                    float(getattr(structure, "width")),
                    float(getattr(structure, "height")),
                    max(depth, 1e-12),
                ],
            }
            objects.append(
                Object3D(
                    kind="box",
                    label=label,
                    geometry=geometry,
                    material=_material_spec(structure, str(color)),
                    metadata={
                        **_structure_metadata(structure),
                        "material_key": list(_structure_material_key(structure)),
                    },
                )
            )
    return objects


def _get_deterministic_color(index: int) -> str:
    if index < len(_STRUCTURE_PALETTE):
        return _STRUCTURE_PALETTE[index]
    saturation, value = 0.6, 0.7
    hue = (index * 0.618034) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _is_air_like_material(material: Any) -> bool:
    if material is None:
        return True
    return abs(float(getattr(material, "permittivity", 1.0)) - 1.0) < 0.1


def _merged_structures_for_view(structures: Iterable[Any]) -> list[Any]:
    material_groups: dict[tuple[Any, ...], list[tuple[int, Any, Any]]] = {}
    passthrough: list[tuple[int, Any]] = []

    for index, structure in enumerate(structures):
        shape = _viewer_structure_shape(structure)
        if shape is None or getattr(structure, "material", None) is None:
            passthrough.append((index, structure))
            continue
        material_groups.setdefault(_viewer_structure_group_key(structure), []).append(
            (index, structure, shape)
        )

    merged_items: list[tuple[float, Any]] = list(passthrough)
    for group_key, entries in material_groups.items():
        group_pairs = [(structure, shape) for _, structure, shape in entries]
        structures_to_remove = [structure for _, structure, _ in entries]
        rings_to_preserve = _find_rings_to_preserve(
            {group_key: group_pairs}, structures_to_remove
        )
        ring_ids = {id(structure) for structure in rings_to_preserve}
        merge_entries = [
            (index, structure, shape)
            for index, structure, shape in entries
            if id(structure) not in ring_ids
        ]

        for index, structure, _ in entries:
            if id(structure) in ring_ids:
                merged_items.append((index, structure))

        if len(merge_entries) <= 1:
            merged_items.extend((index, structure) for index, structure, _ in merge_entries)
            continue

        merged_geometry = unary_union([shape for _, _, shape in merge_entries])
        merged_polygons = _shapely_to_polygons(
            merged_geometry,
            merge_entries[0][1].material,
            merge_entries[0][1],
        )
        if not merged_polygons:
            merged_items.extend((index, structure) for index, structure, _ in merge_entries)
            continue

        display_order = max(index for index, _, _ in merge_entries)
        representative_color = getattr(merge_entries[0][1], "color", None)
        for offset, polygon in enumerate(merged_polygons):
            polygon.color = representative_color
            merged_items.append((display_order + offset * 1e-3, polygon))

    return [structure for _, structure in sorted(merged_items, key=lambda item: item[0])]


def _viewer_structure_group_key(structure: Any) -> tuple[Any, ...]:
    return (
        _design_material_key(getattr(structure, "material", None)),
        round(float(getattr(structure, "depth", 0.0) or 0.0), 12),
        round(float(getattr(structure, "z", 0.0) or 0.0), 12),
        bool(getattr(structure, "is_pml", False)),
    )


def _viewer_structure_shape(structure: Any) -> Any | None:
    shape = _to_shapely(structure)
    if shape is not None:
        return shape
    if (
        hasattr(structure, "position")
        and hasattr(structure, "width")
        and hasattr(structure, "height")
    ):
        px, py, *_ = getattr(structure, "position")
        width = float(getattr(structure, "width", 0.0) or 0.0)
        height = float(getattr(structure, "height", 0.0) or 0.0)
        if width <= 0 or height <= 0:
            return None
        return shapely_box(float(px), float(py), float(px) + width, float(py) + height)
    return None


def _structure_material_key(structure: Any) -> tuple[Any, ...]:
    material = getattr(structure, "material", None)
    return (
        type(material).__name__ if material is not None else None,
        round(float(getattr(material, "permittivity", 1.0)), 9)
        if material is not None
        else None,
        round(float(getattr(material, "permeability", 1.0)), 9)
        if material is not None
        else None,
        round(float(getattr(material, "conductivity", 0.0)), 9)
        if material is not None
        else None,
        bool(getattr(structure, "is_pml", False)),
    )


def _monitor_objects(monitors: Iterable[Any]) -> list[Object3D]:
    objects: list[Object3D] = []
    for index, monitor in enumerate(monitors):
        label = getattr(monitor, "name", None) or f"Monitor {index + 1}"
        if bool(getattr(monitor, "is_3d", False)):
            objects.append(_monitor_plane_object(monitor, label))
        else:
            start = getattr(monitor, "start", (0.0, 0.0))
            end = getattr(monitor, "end", start)
            geometry = {
                "points": [
                    [float(start[0]), float(start[1]), 0.0],
                    [float(end[0]), float(end[1]), 0.0],
                ]
            }
            objects.append(
                Object3D(
                    kind="line",
                    label=label,
                    geometry=geometry,
                    material=MaterialSpec(color="#dc2626", opacity=1.0),
                    metadata={
                        "kind": "monitor",
                        "type": getattr(monitor, "monitor_type", "line"),
                    },
                )
            )
    return objects


def _monitor_plane_object(monitor: Any, label: str) -> Object3D:
    start = getattr(monitor, "start", (0.0, 0.0, 0.0))
    end = getattr(monitor, "end", None)
    plane_normal = str(getattr(monitor, "plane_normal", "z")).lower()
    if end is not None:
        dx = abs(float(end[0]) - float(start[0]))
        dy = abs(float(end[1]) - float(start[1]))
        dz = abs(float(end[2]) - float(start[2]))
        center = [
            (float(start[0]) + float(end[0])) / 2.0,
            (float(start[1]) + float(end[1])) / 2.0,
            (float(start[2]) + float(end[2])) / 2.0,
        ]
        if plane_normal == "x":
            size = [max(dy, 1e-12), max(dz, 1e-12)]
        elif plane_normal == "y":
            size = [max(dx, 1e-12), max(dz, 1e-12)]
        else:
            size = [max(dx, 1e-12), max(dy, 1e-12)]
    else:
        plane_position = float(getattr(monitor, "plane_position", 0.0))
        size_attr = getattr(monitor, "size", (1.0, 1.0))
        size = [float(size_attr[0]), float(size_attr[1])]
        position = getattr(monitor, "position", None)
        if position is not None and len(position) >= 3:
            center = [float(position[0]), float(position[1]), float(position[2])]
        else:
            center = [0.0, 0.0, 0.0]
            axis = {"x": 0, "y": 1, "z": 2}.get(str(plane_normal), 2)
            center[axis] = plane_position
    geometry = {
        "center": center,
        "size": size,
        "normal": _normal_from_axis(plane_normal),
    }
    return Object3D(
        kind="plane",
        label=label,
        geometry=geometry,
        material=MaterialSpec(color="#dc2626", opacity=0.22),
        metadata={
            "kind": "monitor",
            "type": getattr(monitor, "monitor_type", "plane"),
            "plane_normal": getattr(monitor, "plane_normal", None),
        },
    )


def _source_objects(sources: Iterable[Any]) -> list[Object3D]:
    objects: list[Object3D] = []
    for index, source in enumerate(sources):
        if (
            hasattr(source, "position")
            and hasattr(source, "width")
            and not hasattr(source, "center")
        ):
            position = list(getattr(source, "position"))
            while len(position) < 3:
                position.append(0.0)
            radius = max(float(getattr(source, "width", 1.0)) * 0.5, 1e-12)
            objects.append(
                Object3D(
                    kind="sphere",
                    label=f"GaussianSource {index + 1}",
                    geometry={
                        "center": [float(v) for v in position[:3]],
                        "radius": radius,
                    },
                    material=MaterialSpec(color="#f59e0b", opacity=0.85),
                    metadata={
                        "kind": "source",
                        "type": type(source).__name__,
                        "width": getattr(source, "width", None),
                    },
                )
            )
            continue

        if hasattr(source, "center") and hasattr(source, "width"):
            center = list(getattr(source, "center"))
            while len(center) < 3:
                center.append(0.0)
            height = float(getattr(source, "height", getattr(source, "width", 1.0)))
            direction = getattr(source, "direction", "+x")
            normal = _normal_from_direction(direction)
            objects.append(
                Object3D(
                    kind="plane",
                    label=f"ModeSource {index + 1}",
                    geometry={
                        "center": [float(v) for v in center[:3]],
                        "size": [float(getattr(source, "width")), height],
                        "normal": normal,
                    },
                    material=MaterialSpec(color="#f59e0b", opacity=0.35),
                    metadata={
                        "kind": "source",
                        "type": type(source).__name__,
                        "direction": direction,
                        "wavelength": getattr(source, "wavelength", None),
                        "polarization": getattr(source, "pol", None),
                    },
                )
            )
            arrow_length = max(
                float(getattr(source, "wavelength", 1.0)),
                float(getattr(source, "width", 1.0)) * 0.5,
            )
            objects.append(
                Object3D(
                    kind="arrow",
                    label=f"{getattr(source, 'direction', '+x')} launch",
                    geometry={
                        "origin": [float(v) for v in center[:3]],
                        "direction": normal,
                        "length": arrow_length,
                    },
                    material=MaterialSpec(color="#d97706", opacity=1.0),
                    metadata={
                        "kind": "source_direction",
                        "source_direction": direction,
                    },
                )
            )
    return objects


def _simulation_planes(simulation: Any, bounds: _Bounds) -> list[Object3D]:
    resolution = getattr(simulation, "resolution", None)
    if resolution is None:
        return []
    plane_2d = str(getattr(simulation, "plane_2d", "xy")).lower()
    center = list(bounds.center)
    if plane_2d == "yz":
        size = [bounds.size[1], bounds.size[2]]
        normal = [1.0, 0.0, 0.0]
    elif plane_2d == "xz":
        size = [bounds.size[0], bounds.size[2]]
        normal = [0.0, 1.0, 0.0]
    else:
        size = [bounds.size[0], bounds.size[1]]
        normal = [0.0, 0.0, 1.0]
    return [
        Object3D(
            kind="plane",
            label="Simulation mid-plane",
            geometry={
                "center": center,
                "size": size,
                "normal": normal,
            },
            material=MaterialSpec(color="#0f766e", opacity=0.06),
            metadata={
                "kind": "simulation",
                "resolution": float(resolution),
                "plane_2d": plane_2d,
            },
        )
    ]


def _domain_box(design: Any, bounds: _Bounds) -> Object3D:
    return Object3D(
        kind="box",
        label="Simulation Domain",
        geometry={"center": list(bounds.center), "size": list(bounds.size)},
        material=MaterialSpec(color="#0f172a", opacity=0.08, wireframe=True),
        metadata={
            "kind": "domain",
            "width": getattr(design, "width", None),
            "height": getattr(design, "height", None),
            "depth": getattr(design, "depth", None),
        },
    )


def _normal_from_direction(direction: str) -> list[float]:
    sign = -1.0 if str(direction).startswith("-") else 1.0
    axis = str(direction)[-1].lower()
    return _normal_from_axis(axis, sign)


def _normal_from_axis(axis: str, sign: float = 1.0) -> list[float]:
    lookup = {
        "x": [sign, 0.0, 0.0],
        "y": [0.0, sign, 0.0],
        "z": [0.0, 0.0, sign],
    }
    return lookup.get(str(axis).lower(), [0.0, 0.0, sign])


def _build_scene(
    *,
    title: str,
    bounds: _Bounds,
    objects: list[Object3D],
    metadata: dict[str, Any],
) -> SceneSpec:
    for index, obj in enumerate(objects):
        obj.metadata = {
            **obj.metadata,
            "display_order": index,
        }
    center = bounds.center
    diagonal = max(bounds.diagonal, 1e-9)
    camera = CameraSpec(
        position=(
            center[0] + diagonal * 0.9,
            center[1] - diagonal * 1.1,
            center[2] + diagonal * 0.7,
        ),
        target=center,
        up=(0.0, 0.0, 1.0),
        fov=40.0,
    )
    clip_planes = [
        ClipPlaneSpec(normal=(1.0, 0.0, 0.0), constant=-center[0], enabled=False),
        ClipPlaneSpec(normal=(0.0, 1.0, 0.0), constant=-center[1], enabled=False),
        ClipPlaneSpec(normal=(0.0, 0.0, 1.0), constant=-center[2], enabled=False),
    ]
    return SceneSpec(
        title=title,
        units="m",
        background="#f8fafc",
        camera=camera,
        clip_planes=clip_planes,
        objects=objects,
        metadata=metadata,
    )


def _simulation_monitors(simulation: Any) -> list[Any]:
    items = list(getattr(getattr(simulation, "design", None), "monitors", []))
    seen = {id(item) for item in items}
    for device in getattr(simulation, "devices", []):
        if not _looks_like_monitor(device):
            continue
        if id(device) in seen:
            continue
        seen.add(id(device))
        items.append(device)
    return items


def _simulation_sources(simulation: Any) -> list[Any]:
    items = list(getattr(getattr(simulation, "design", None), "sources", []))
    seen = {id(item) for item in items}
    for device in getattr(simulation, "devices", []):
        if not _looks_like_source(device):
            continue
        if id(device) in seen:
            continue
        seen.add(id(device))
        items.append(device)
    return items


def _looks_like_monitor(device: Any) -> bool:
    return any(
        hasattr(device, name)
        for name in (
            "monitor_type",
            "should_record",
            "record_fields",
            "record_fields_2d",
        )
    )


def _looks_like_source(device: Any) -> bool:
    return hasattr(device, "width") and (
        hasattr(device, "position") or hasattr(device, "center")
    )


def _boundary_objects(simulation: Any, bounds: _Bounds) -> list[Object3D]:
    objects: list[Object3D] = []
    for boundary in getattr(simulation, "boundaries", []):
        thickness = float(getattr(boundary, "thickness", 0.0) or 0.0)
        if thickness <= 0:
            continue
        is_3d = bool(getattr(simulation, "is_3d", False))
        if hasattr(boundary, "_get_edges_for_dimensionality"):
            edges = boundary._get_edges_for_dimensionality(is_3d)
        else:
            raw_edges = getattr(boundary, "edges", [])
            edges = raw_edges if isinstance(raw_edges, list) else [raw_edges]
        for edge in edges:
            geometry = _boundary_geometry(edge=edge, thickness=thickness, bounds=bounds)
            if geometry is None:
                continue
            objects.append(
                Object3D(
                    kind="box",
                    label=f"{type(boundary).__name__} {edge}",
                    geometry=geometry,
                    material=_boundary_material(boundary),
                    metadata={
                        "kind": "boundary",
                        "type": type(boundary).__name__,
                        "edge": edge,
                        "thickness": thickness,
                    },
                )
            )
    return objects


def _boundary_geometry(
    *,
    edge: str,
    thickness: float,
    bounds: _Bounds,
) -> dict[str, list[float]] | None:
    width, height, depth = bounds.size
    slab_x = min(thickness, width)
    slab_y = min(thickness, height)
    slab_z = min(thickness, depth)

    if edge == "left":
        return {
            "center": [slab_x / 2.0, height / 2.0, depth / 2.0],
            "size": [slab_x, height, depth],
        }
    if edge == "right":
        return {
            "center": [width - slab_x / 2.0, height / 2.0, depth / 2.0],
            "size": [slab_x, height, depth],
        }
    if edge == "bottom":
        return {
            "center": [width / 2.0, slab_y / 2.0, depth / 2.0],
            "size": [width, slab_y, depth],
        }
    if edge == "top":
        return {
            "center": [width / 2.0, height - slab_y / 2.0, depth / 2.0],
            "size": [width, slab_y, depth],
        }
    if edge == "front":
        return {
            "center": [width / 2.0, height / 2.0, slab_z / 2.0],
            "size": [width, height, slab_z],
        }
    if edge == "back":
        return {
            "center": [width / 2.0, height / 2.0, depth - slab_z / 2.0],
            "size": [width, height, slab_z],
        }
    return None


def _boundary_material(boundary: Any) -> MaterialSpec:
    if type(boundary).__name__.lower() == "pml":
        return MaterialSpec(color="#7c3aed", opacity=0.12, wireframe=True)
    return MaterialSpec(color="#475569", opacity=0.1, wireframe=True)
