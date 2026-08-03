"""Geometry-to-material rasterization for BeamZ and imported scenes."""

from .engine import CompiledScene, RasterOptions, compile_scene, rasterize
from .result import RasterResult
from .schema import (
    Box,
    Cylinder,
    ExtrudedPolygon,
    Grid,
    Material,
    Mesh,
    MeshInspection,
    Object,
    Polygon,
    Scene,
    Sphere,
    TaperedExtrudedPolygon,
    inspect_mesh,
)

__all__ = [
    "Box",
    "CompiledScene",
    "Cylinder",
    "ExtrudedPolygon",
    "Grid",
    "Material",
    "Mesh",
    "MeshInspection",
    "Object",
    "Polygon",
    "RasterOptions",
    "RasterResult",
    "Scene",
    "Sphere",
    "TaperedExtrudedPolygon",
    "compile_scene",
    "inspect_mesh",
    "rasterize",
]
