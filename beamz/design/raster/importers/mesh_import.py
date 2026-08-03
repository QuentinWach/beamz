from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import Material, Mesh, Object, Scene


def _cell_indices(
    values: Any,
    *,
    width: int,
    point_count: int,
    cell_type: str,
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 2 or result.shape[1] < width:
        raise ValueError(
            f"{cell_type} cells must have at least {width} vertex indices."
        )
    result = result[:, :width]
    if np.issubdtype(result.dtype, np.floating):
        if not np.isfinite(result).all() or np.any(result != np.floor(result)):
            raise ValueError(f"{cell_type} cell indices must be finite integers.")
    elif not np.issubdtype(result.dtype, np.integer):
        raise TypeError(f"{cell_type} cell indices must use an integer numeric dtype.")
    if np.any(result < 0) or np.any(result >= point_count):
        raise ValueError(f"{cell_type} cell index is out of range.")
    return result.astype(np.uint32, copy=False)


def _cell_tags(values: Any, *, cell_count: int) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (cell_count,):
        raise ValueError("Gmsh physical tags must match their cell block length.")
    if np.issubdtype(result.dtype, np.floating):
        if not np.isfinite(result).all() or np.any(result != np.floor(result)):
            raise ValueError("Gmsh physical tags must be finite integers.")
    elif not np.issubdtype(result.dtype, np.integer):
        raise TypeError("Gmsh physical tags must use an integer numeric dtype.")
    return result


def from_mesh_arrays(
    vertices: Any,
    triangles: Any,
    *,
    material: Material,
    background: Material | None = None,
) -> Scene:
    background = Material() if background is None else background
    mesh = Mesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles),
    )
    return Scene(
        materials=(background, material),
        objects=(Object(mesh, material_id=1, id=1),),
        background_material=0,
    )


def _oriented_tetra_faces(
    points: np.ndarray, tetra: np.ndarray
) -> Iterator[tuple[int, int, int]]:
    center = points[tetra].mean(axis=0)
    for face in (
        tetra[[1, 2, 3]],
        tetra[[0, 3, 2]],
        tetra[[0, 1, 3]],
        tetra[[0, 2, 1]],
    ):
        vertices = points[face]
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        if np.dot(normal, vertices.mean(axis=0) - center) < 0:
            face = face[[0, 2, 1]]
        yield int(face[0]), int(face[1]), int(face[2])


def _tetra_boundary(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    counts: Counter[tuple[int, int, int]] = Counter()
    oriented: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tetra in tetrahedra:
        for face in _oriented_tetra_faces(points, tetra):
            ordered = sorted(face)
            key = (ordered[0], ordered[1], ordered[2])
            counts[key] += 1
            oriented[key] = face
    return np.asarray(
        [oriented[key] for key, count in counts.items() if count == 1],
        dtype=np.uint32,
    )


def from_mesh(
    path: str | Path,
    *,
    material: Material | None = None,
    materials: dict[str | int, Material] | None = None,
    background: Material | None = None,
    unit_scale: float = 1.0,
) -> Scene:
    """Import surface triangles or tetrahedral Gmsh physical regions via meshio."""

    background = Material() if background is None else background
    unit_scale = float(unit_scale)
    if not np.isfinite(unit_scale) or unit_scale <= 0.0:
        raise ValueError("unit_scale must be finite and positive.")
    try:
        import meshio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("Install BeamZ with meshio to import mesh files.") from exc

    data = meshio.read(path)
    points = np.asarray(data.points[:, :3], dtype=np.float64) * unit_scale
    material_lookup = materials or {}
    field_names = {
        (int(values[0]), int(values[1]) if len(values) > 1 else -1): name
        for name, values in getattr(data, "field_data", {}).items()
        if len(values) >= 1
    }

    surface_regions: dict[int, list[np.ndarray]] = defaultdict(list)
    surface_blocks: list[np.ndarray] = []
    tetra_blocks: list[tuple[np.ndarray, np.ndarray | None]] = []
    physical_data = data.cell_data.get("gmsh:physical", [])
    for index, block in enumerate(data.cells):
        if block.type == "triangle":
            triangles = _cell_indices(
                block.data,
                width=3,
                point_count=len(points),
                cell_type="triangle",
            )
            tags = (
                _cell_tags(physical_data[index], cell_count=len(triangles))
                if index < len(physical_data)
                else None
            )
            if tags is None:
                surface_blocks.append(triangles)
            else:
                for tag in np.unique(tags):
                    surface_regions[int(tag)].append(triangles[tags == tag])
        elif block.type in {"tetra", "tetra10"}:
            tetrahedra = _cell_indices(
                block.data,
                width=4,
                point_count=len(points),
                cell_type=block.type,
            )
            tags = (
                _cell_tags(physical_data[index], cell_count=len(tetrahedra))
                if index < len(physical_data)
                else None
            )
            tetra_blocks.append((tetrahedra, tags))

    if tetra_blocks:
        tetra_regions: dict[int, list[np.ndarray]] = defaultdict(list)
        for tetrahedra, tags in tetra_blocks:
            if tags is None:
                tetra_regions[0].append(tetrahedra)
            else:
                for tag in np.unique(tags):
                    tetra_regions[int(tag)].append(tetrahedra[tags == tag])
        regions = {
            tag: [_tetra_boundary(points, np.concatenate(blocks))]
            for tag, blocks in tetra_regions.items()
        }
        physical_dimension = 3
    else:
        regions = surface_regions
        if surface_blocks:
            regions[0].extend(surface_blocks)
        physical_dimension = 2

    if not regions:
        raise ValueError("The mesh contains no triangle or tetrahedral cells.")

    scene_materials = [background]
    objects = []
    for object_id, (tag, blocks) in enumerate(sorted(regions.items()), start=1):
        name = field_names.get(
            (tag, physical_dimension),
            field_names.get((tag, -1), tag),
        )
        region_material = material_lookup.get(name, material_lookup.get(tag, material))
        if region_material is None:
            raise ValueError(
                f"No material configured for mesh physical region {name!r}."
            )
        scene_materials.append(region_material)
        objects.append(
            Object(
                Mesh(points, np.concatenate(blocks)),
                material_id=len(scene_materials) - 1,
                priority=object_id,
                id=object_id,
            )
        )
    return Scene(tuple(scene_materials), tuple(objects), 0)
