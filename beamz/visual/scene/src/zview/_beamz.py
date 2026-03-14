from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._scene import CameraSpec, ClipPlaneSpec, MaterialSpec, Object3D, SceneSpec


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
    return all(hasattr(value, name) for name in ("structures", "sources", "monitors", "width", "height"))


def looks_like_beamz_simulation(value: Any) -> bool:
    return hasattr(value, "design") and looks_like_beamz_design(value.design)


def beamz_to_scene(value: Any) -> SceneSpec:
    if looks_like_beamz_simulation(value):
        simulation = value
        design = value.design
        title = "BEAMZ Simulation Setup"
        metadata = {
            "object_type": type(value).__name__,
            "resolution": getattr(simulation, "resolution", None),
            "is_3d": bool(getattr(simulation, "is_3d", False)),
        }
    elif looks_like_beamz_design(value):
        simulation = None
        design = value
        title = "BEAMZ Design"
        metadata = {
            "object_type": type(value).__name__,
            "is_3d": bool(getattr(design, "is_3d", False)),
        }
    else:
        raise TypeError("Expected a BeamZ-like design or simulation object.")

    bounds = _design_bounds(design)
    objects: list[Object3D] = [_domain_box(design, bounds)]
    objects.extend(_structure_objects(design))
    objects.extend(_monitor_objects(design))
    objects.extend(_source_objects(design))

    if simulation is not None:
        objects.extend(_simulation_planes(simulation, bounds))

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


def _design_bounds(design: Any) -> _Bounds:
    width = float(getattr(design, "width", 1.0))
    height = float(getattr(design, "height", 1.0))
    depth = float(getattr(design, "depth", 0.0) or 0.0)
    max_z = depth if depth > 0 else max(min(width, height) * 0.1, 1e-9)
    return _Bounds(0.0, 0.0, 0.0, width, height, max_z)


def _material_spec(structure: Any, fallback: str) -> MaterialSpec:
    color = getattr(structure, "color", None) or fallback
    material = getattr(structure, "material", None)
    permittivity = getattr(material, "permittivity", 1.0) if material is not None else 1.0
    opacity = 0.08 if abs(float(permittivity) - 1.0) < 0.05 else 0.7
    wireframe = bool(getattr(structure, "is_pml", False))
    return MaterialSpec(color=str(color), opacity=opacity, wireframe=wireframe)


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
    for index, structure in enumerate(getattr(design, "structures", [])):
        if bool(getattr(structure, "is_pml", False)):
            label = f"PML {index + 1}"
        else:
            label = f"{type(structure).__name__} {index + 1}"
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
                    material=_material_spec(structure, "#2563eb"),
                    metadata=_structure_metadata(structure),
                )
            )
            continue

        if hasattr(structure, "position") and hasattr(structure, "width") and hasattr(structure, "height"):
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
                    material=_material_spec(structure, "#2563eb"),
                    metadata=_structure_metadata(structure),
                )
            )
    return objects


def _monitor_objects(design: Any) -> list[Object3D]:
    objects: list[Object3D] = []
    for index, monitor in enumerate(getattr(design, "monitors", [])):
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
    if end is not None:
        center = [
            (float(start[0]) + float(end[0])) / 2.0,
            (float(start[1]) + float(end[1])) / 2.0,
            (float(start[2]) + float(end[2])) / 2.0,
        ]
        size = [
            max(abs(float(end[0]) - float(start[0])), 1e-12),
            max(abs(float(end[1]) - float(start[1])), 1e-12),
        ]
    else:
        plane_position = float(getattr(monitor, "plane_position", 0.0))
        plane_normal = getattr(monitor, "plane_normal", "z")
        size_attr = getattr(monitor, "size", (1.0, 1.0))
        size = [float(size_attr[0]), float(size_attr[1])]
        center = [0.0, 0.0, 0.0]
        axis = {"x": 0, "y": 1, "z": 2}.get(str(plane_normal), 2)
        center[axis] = plane_position
    geometry = {
        "center": center,
        "size": size,
        "normal": _normal_from_axis(getattr(monitor, "plane_normal", "z")),
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


def _source_objects(design: Any) -> list[Object3D]:
    objects: list[Object3D] = []
    for index, source in enumerate(getattr(design, "sources", [])):
        if hasattr(source, "position") and hasattr(source, "width") and not hasattr(source, "center"):
            position = list(getattr(source, "position"))
            while len(position) < 3:
                position.append(0.0)
            radius = max(float(getattr(source, "width", 1.0)) * 0.5, 1e-12)
            objects.append(
                Object3D(
                    kind="sphere",
                    label=f"GaussianSource {index + 1}",
                    geometry={"center": [float(v) for v in position[:3]], "radius": radius},
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
            arrow_length = max(float(getattr(source, "wavelength", 1.0)), float(getattr(source, "width", 1.0)) * 0.5)
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
                    metadata={"kind": "source_direction", "source_direction": direction},
                )
            )
    return objects


def _simulation_planes(simulation: Any, bounds: _Bounds) -> list[Object3D]:
    resolution = getattr(simulation, "resolution", None)
    if resolution is None:
        return []
    return [
        Object3D(
            kind="plane",
            label="Simulation mid-plane",
            geometry={
                "center": [bounds.center[0], bounds.center[1], bounds.center[2]],
                "size": [bounds.size[0], bounds.size[1]],
                "normal": [0.0, 0.0, 1.0],
            },
            material=MaterialSpec(color="#0f766e", opacity=0.06),
            metadata={"kind": "simulation", "resolution": float(resolution)},
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
