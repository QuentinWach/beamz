from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..schema import Mesh, MeshInspection, inspect_mesh


@dataclass(frozen=True, slots=True)
class MeshRepairOptions:
    """Explicit, conservative triangle-mesh repair policy."""

    tolerance: float | None = None
    merge_vertices: bool = True
    remove_degenerate: bool = True
    remove_duplicate_faces: bool = True
    remove_unreferenced_vertices: bool = True
    fill_small_holes: bool = True
    fix_normals: bool = True


@dataclass(frozen=True, slots=True)
class MeshRepairReport:
    """Auditable record of every mesh-repair mutation."""

    before: MeshInspection
    after: MeshInspection | None
    tolerance: float
    merged_vertices: int
    removed_degenerate_triangles: int
    removed_duplicate_triangles: int
    removed_unreferenced_vertices: int
    filled_triangles: int
    winding_or_normals_fixed: bool
    valid_for_rasterization: bool
    actions: tuple[str, ...]
    remaining_issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MeshRepairResult:
    mesh: Mesh
    report: MeshRepairReport


def _repair_tolerance(vertices: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        result = float(requested)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError("Mesh repair tolerance must be finite and positive.")
        return result
    extent = float(np.max(np.ptp(vertices, axis=0)))
    coordinate_scale = max(
        float(np.max(np.abs(vertices))),
        extent,
        np.finfo(np.float64).tiny,
    )
    return max(extent * 1e-12, 32.0 * np.finfo(np.float64).eps * coordinate_scale)


def _remaining_issues(
    report: MeshInspection | None, error: str | None
) -> tuple[str, ...]:
    if report is None:
        return (error or "native mesh inspection failed",)
    issues = []
    for name in (
        "boundary_edges",
        "nonmanifold_edges",
        "inconsistent_edges",
        "degenerate_triangles",
        "self_intersections",
    ):
        count = getattr(report, name)
        if count:
            issues.append(f"{name}={count}")
    if report.topology_is_valid and not report.valid_for_rasterization:
        issues.append("zero enclosed volume")
    return tuple(issues)


def _nondegenerate_faces(mesh: Any, tolerance: float) -> np.ndarray:
    faces = np.asarray(mesh.faces)
    points = np.asarray(mesh.vertices)[faces]
    first, second = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    third = points[:, 2] - points[:, 1]
    longest = np.sqrt(
        np.maximum.reduce(
            tuple(np.einsum("ij,ij->i", edge, edge) for edge in (first, second, third))
        )
    )
    distinct = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    return distinct & (
        np.linalg.norm(np.cross(first, second), axis=1) > tolerance * longest
    )


def repair_mesh(
    vertices: Any,
    triangles: Any,
    *,
    options: MeshRepairOptions | None = None,
) -> MeshRepairResult:
    """Repair common mesh defects and return the mesh plus an exact audit report.

    This operation is always explicit. It can weld numerically coincident
    vertices, remove duplicate or degenerate faces, fill triangular or
    quadrilateral holes, and consistently orient connected components. The
    strict Rust topology and self-intersection validator runs before and after.
    Defects that cannot be repaired remain visible in ``remaining_issues``.
    """

    options = MeshRepairOptions() if options is None else options
    source = Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(triangles))
    source_vertices, source_triangles = source.validated_arrays()
    before = inspect_mesh(source_vertices, source_triangles)
    tolerance = _repair_tolerance(source_vertices, options.tolerance)
    try:
        import trimesh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Install BeamZ with the 'mesh' extra to repair triangle meshes."
        ) from exc

    actions: list[str] = []
    warnings: list[str] = []
    repaired = trimesh.Trimesh(
        vertices=source_vertices,
        faces=source_triangles,
        process=False,
        validate=False,
    )

    removed_unreferenced = 0
    if options.remove_unreferenced_vertices:
        original = len(repaired.vertices)
        repaired.remove_unreferenced_vertices()
        removed_unreferenced = original - len(repaired.vertices)

    merged_vertices = 0
    if options.merge_vertices:
        original = len(repaired.vertices)
        digits = min(300, max(-300, int(np.ceil(-np.log10(tolerance)))))
        repaired.merge_vertices(digits_vertex=digits)
        merged_vertices = original - len(repaired.vertices)
        if merged_vertices:
            actions.append(f"merged {merged_vertices} coincident vertices")

    removed_degenerate = 0
    if options.remove_degenerate:
        mask = _nondegenerate_faces(repaired, tolerance)
        removed_degenerate = int(len(mask) - np.count_nonzero(mask))
        if removed_degenerate:
            repaired.update_faces(mask)
            actions.append(f"removed {removed_degenerate} degenerate triangles")

    removed_duplicates = 0
    if options.remove_duplicate_faces:
        mask = repaired.unique_faces()
        removed_duplicates = int(len(mask) - np.count_nonzero(mask))
        if removed_duplicates:
            repaired.update_faces(mask)
            actions.append(f"removed {removed_duplicates} duplicate triangles")

    if options.remove_unreferenced_vertices:
        original = len(repaired.vertices)
        repaired.remove_unreferenced_vertices()
        removed_unreferenced += original - len(repaired.vertices)
        if removed_unreferenced:
            actions.append(f"removed {removed_unreferenced} unreferenced vertices")

    filled_triangles = 0
    if options.fill_small_holes:
        original = len(repaired.faces)
        try:
            trimesh.repair.fill_holes(repaired)
        except (IndexError, RuntimeError, ValueError) as exc:
            warnings.append(f"hole filling failed: {exc}")
        filled_triangles = len(repaired.faces) - original
        if filled_triangles:
            actions.append(f"filled small holes with {filled_triangles} triangles")

    normals_fixed = False
    if options.fix_normals:
        was_consistent = bool(repaired.is_winding_consistent)
        try:
            trimesh.repair.fix_normals(repaired, multibody=True)
            normals_fixed = not was_consistent or before.signed_volume < 0.0
        except (IndexError, RuntimeError, ValueError) as exc:
            warnings.append(f"normal repair failed: {exc}")
        if normals_fixed:
            actions.append("oriented connected components outward")

    repaired_vertices = np.asarray(repaired.vertices, dtype=np.float64).copy()
    repaired_triangles = np.asarray(repaired.faces, dtype=np.uint32).copy()
    repaired_vertices.setflags(write=False)
    repaired_triangles.setflags(write=False)
    mesh = Mesh(repaired_vertices, repaired_triangles)

    after = None
    inspection_error = None
    try:
        after = inspect_mesh(repaired_vertices, repaired_triangles)
    except ValueError as exc:
        inspection_error = str(exc)
    valid = after is not None and after.valid_for_rasterization
    remaining_items = list(_remaining_issues(after, inspection_error))
    if after is not None and inspection_error is not None:
        remaining_items.append(inspection_error)
    remaining_items.extend(warnings)
    report = MeshRepairReport(
        before=before,
        after=after,
        tolerance=tolerance,
        merged_vertices=merged_vertices,
        removed_degenerate_triangles=removed_degenerate,
        removed_duplicate_triangles=removed_duplicates,
        removed_unreferenced_vertices=removed_unreferenced,
        filled_triangles=filled_triangles,
        winding_or_normals_fixed=normals_fixed,
        valid_for_rasterization=valid,
        actions=tuple(actions),
        remaining_issues=tuple(remaining_items),
    )
    return MeshRepairResult(mesh, report)
