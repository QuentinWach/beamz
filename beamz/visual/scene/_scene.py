from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import uuid4


ObjectKind = Literal["arrow", "box", "line", "plane", "poly_extrusion", "sphere"]
ProjectionKind = Literal["perspective", "orthographic"]


def _float_list(values: tuple[float, ...] | list[float]) -> list[float]:
    return [float(value) for value in values]


@dataclass(slots=True)
class MaterialSpec:
    color: str = "#4f46e5"
    opacity: float = 1.0
    wireframe: bool = False
    visible: bool = True
    metalness: float = 0.0
    roughness: float = 0.85
    emissive: str = "#000000"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClipPlaneSpec:
    normal: tuple[float, float, float]
    constant: float
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal": _float_list(self.normal),
            "constant": float(self.constant),
            "enabled": bool(self.enabled),
        }


@dataclass(slots=True)
class CameraSpec:
    projection: ProjectionKind = "perspective"
    position: tuple[float, float, float] = (1.5, 1.5, 1.2)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 45.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection": self.projection,
            "position": _float_list(self.position),
            "target": _float_list(self.target),
            "up": _float_list(self.up),
            "fov": float(self.fov),
        }


@dataclass(slots=True)
class Object3D:
    kind: ObjectKind
    label: str
    geometry: dict[str, Any]
    material: MaterialSpec = field(default_factory=MaterialSpec)
    id: str = field(default_factory=lambda: f"obj-{uuid4().hex[:12]}")
    metadata: dict[str, Any] = field(default_factory=dict)
    visible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "geometry": self.geometry,
            "material": self.material.to_dict(),
            "metadata": self.metadata,
            "visible": bool(self.visible),
        }


@dataclass(slots=True)
class SceneSpec:
    objects: list[Object3D]
    schema_version: str = "0.1.0"
    units: str = "m"
    background: str = "#f8fafc"
    camera: CameraSpec = field(default_factory=CameraSpec)
    clip_planes: list[ClipPlaneSpec] = field(default_factory=list)
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "units": self.units,
            "background": self.background,
            "camera": self.camera.to_dict(),
            "clip_planes": [plane.to_dict() for plane in self.clip_planes],
            "objects": [obj.to_dict() for obj in self.objects],
            "title": self.title,
            "metadata": self.metadata,
        }


def scene_from_dict(data: dict[str, Any]) -> SceneSpec:
    camera_data = data.get("camera", {})
    camera = CameraSpec(
        projection=camera_data.get("projection", "perspective"),
        position=tuple(camera_data.get("position", (1.5, 1.5, 1.2))),
        target=tuple(camera_data.get("target", (0.0, 0.0, 0.0))),
        up=tuple(camera_data.get("up", (0.0, 0.0, 1.0))),
        fov=float(camera_data.get("fov", 45.0)),
    )
    clip_planes = [
        ClipPlaneSpec(
            normal=tuple(plane["normal"]),
            constant=float(plane["constant"]),
            enabled=bool(plane.get("enabled", True)),
        )
        for plane in data.get("clip_planes", [])
    ]
    objects = []
    for obj in data.get("objects", []):
        material_data = obj.get("material", {})
        objects.append(
            Object3D(
                id=obj.get("id", f"obj-{uuid4().hex[:12]}"),
                kind=obj["kind"],
                label=obj.get("label", obj["kind"]),
                geometry=obj.get("geometry", {}),
                material=MaterialSpec(
                    color=material_data.get("color", "#4f46e5"),
                    opacity=float(material_data.get("opacity", 1.0)),
                    wireframe=bool(material_data.get("wireframe", False)),
                    visible=bool(material_data.get("visible", True)),
                    metalness=float(material_data.get("metalness", 0.0)),
                    roughness=float(material_data.get("roughness", 0.85)),
                    emissive=material_data.get("emissive", "#000000"),
                ),
                metadata=obj.get("metadata", {}),
                visible=bool(obj.get("visible", True)),
            )
        )
    return SceneSpec(
        schema_version=data.get("schema_version", "0.1.0"),
        units=data.get("units", "m"),
        background=data.get("background", "#f8fafc"),
        camera=camera,
        clip_planes=clip_planes,
        objects=objects,
        title=data.get("title"),
        metadata=data.get("metadata", {}),
    )
