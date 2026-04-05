from __future__ import annotations

from dataclasses import dataclass

from beamz.design.geometry_ops import freeze_interiors, freeze_vertices
from beamz.design.materials import material_to_spec


def structure_kind(spec):
    if (
        spec.length is not None
        and spec.input_width is not None
        and spec.output_width is not None
    ):
        return "Taper"
    if (
        spec.inner_radius is not None
        and spec.outer_radius is not None
        and spec.angle is not None
    ):
        return "CircularBend"
    if spec.inner_radius is not None and spec.outer_radius is not None:
        return "Ring"
    if spec.radius is not None and spec.depth and spec.depth > 0 and not spec.vertices:
        return "Sphere"
    if spec.radius is not None:
        return "Circle"
    if spec.width is not None and spec.height is not None and len(spec.vertices) == 4:
        return "Rectangle"
    return "Polygon"


@dataclass(frozen=True, slots=True)
class StructureSpec:
    vertices: tuple[tuple[float, float, float], ...] = ()
    interiors: tuple[tuple[tuple[float, float, float], ...], ...] = ()
    material: object = None
    color: str | None = None
    optimize: bool = False
    depth: float = 0.0
    z: float = 0.0
    position: tuple[float, float, float] | None = None
    width: float | None = None
    height: float | None = None
    radius: float | None = None
    points: int | None = None
    inner_radius: float | None = None
    outer_radius: float | None = None
    angle: float | None = None
    rotation: float | None = None
    input_width: float | None = None
    output_width: float | None = None
    length: float | None = None
    is_pml: bool = False

    def __post_init__(self):
        object.__setattr__(self, "vertices", freeze_vertices(self.vertices))
        object.__setattr__(self, "interiors", freeze_interiors(self.interiors))
        object.__setattr__(self, "material", material_to_spec(self.material))
        object.__setattr__(self, "optimize", bool(self.optimize))
        object.__setattr__(self, "depth", float(self.depth))
        object.__setattr__(self, "z", float(self.z))
        object.__setattr__(self, "is_pml", bool(self.is_pml))
        if self.position is not None:
            object.__setattr__(self, "position", tuple(float(v) for v in self.position))
        for name in (
            "width",
            "height",
            "radius",
            "inner_radius",
            "outer_radius",
            "angle",
            "rotation",
            "input_width",
            "output_width",
            "length",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.points is not None:
            object.__setattr__(self, "points", int(self.points))

    def to_dict(self):
        from beamz.design.materials import material_spec_to_dict

        return {
            "type": "StructureSpec",
            "shape_type": structure_kind(self),
            "vertices": [list(vertex) for vertex in self.vertices],
            "interiors": [[list(vertex) for vertex in path] for path in self.interiors],
            "material": material_spec_to_dict(self.material),
            "color": self.color,
            "optimize": bool(self.optimize),
            "depth": float(self.depth),
            "z": float(self.z),
            "position": None if self.position is None else list(self.position),
            "width": self.width,
            "height": self.height,
            "radius": self.radius,
            "points": self.points,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "angle": self.angle,
            "rotation": self.rotation,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "length": self.length,
            "is_pml": bool(self.is_pml),
        }

    @classmethod
    def from_dict(cls, data):
        from beamz.design.materials import material_spec_from_dict

        return cls(
            vertices=data.get("vertices", ()),
            interiors=data.get("interiors", ()),
            material=material_spec_from_dict(data["material"]),
            color=data.get("color"),
            optimize=data.get("optimize", False),
            depth=data.get("depth", 0.0),
            z=data.get("z", 0.0),
            position=data.get("position"),
            width=data.get("width"),
            height=data.get("height"),
            radius=data.get("radius"),
            points=data.get("points"),
            inner_radius=data.get("inner_radius"),
            outer_radius=data.get("outer_radius"),
            angle=data.get("angle"),
            rotation=data.get("rotation"),
            input_width=data.get("input_width"),
            output_width=data.get("output_width"),
            length=data.get("length"),
            is_pml=data.get("is_pml", False),
        )
