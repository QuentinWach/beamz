"""Reflection-symmetry validation and reduced-lattice construction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.devices.boundaries import PEC, PMC, PML, Absorber, edges_for_dimension
from beamz.lattice import component_shapes, grid_axes_in_physical_frame_2d

_PHYSICAL_AXES = ("x", "y", "z")
_COMPONENT_FOR_YEE = {
    "eps_x": "Ex",
    "eps_y": "Ey",
    "eps_z": "Ez",
    "sig_x": "Ex",
    "sig_y": "Ey",
    "sig_z": "Ez",
    "mu_hx": "Hx",
    "mu_hy": "Hy",
    "mu_hz": "Hz",
}


def symmetry_cut_edges(
    symmetry: tuple[int, int, int], *, is_3d: bool, plane_2d: str
) -> dict[str, int]:
    """Return retained-half high faces and their requested parity."""
    if is_3d:
        edge_for_physical = {"x": "right", "y": "top", "z": "back"}
    else:
        grid_x, grid_y, _ = grid_axes_in_physical_frame_2d(plane_2d)
        edge_for_physical = {grid_x: "right", grid_y: "top"}
    return {
        edge_for_physical[axis]: int(symmetry[index])
        for index, axis in enumerate(_PHYSICAL_AXES)
        if int(symmetry[index]) and axis in edge_for_physical
    }


def symmetry_boundaries(
    boundaries,
    symmetry: tuple[int, int, int],
    *,
    is_3d: bool,
    plane_2d: str,
):
    """Remove absorbers from cut faces and install parity boundary conditions."""
    cuts = symmetry_cut_edges(symmetry, is_3d=is_3d, plane_2d=plane_2d)
    if not cuts:
        return tuple(boundaries)
    cut_names = set(cuts)
    trimmed = []
    for boundary in boundaries:
        if isinstance(boundary, (PML, Absorber)):
            edges = tuple(
                edge
                for edge in edges_for_dimension(boundary.edges, is_3d)
                if edge not in cut_names
            )
            boundary = replace(boundary, edges=edges)
        trimmed.append(boundary)
    pec = tuple(edge for edge, parity in cuts.items() if parity == -1)
    pmc = tuple(edge for edge, parity in cuts.items() if parity == 1)
    if pec:
        trimmed.append(PEC(edges=pec))
    if pmc:
        trimmed.append(PMC(edges=pmc))
    return tuple(trimmed)


def _storage_axis_for_physical(
    physical_axis: str, *, is_3d: bool, plane_2d: str
) -> int:
    if is_3d:
        return {"x": 2, "y": 1, "z": 0}[physical_axis]
    grid_x, grid_y, _ = grid_axes_in_physical_frame_2d(plane_2d)
    return {grid_x: 1, grid_y: 0}[physical_axis]


def _grid_axis_for_physical(
    physical_axis: str, *, is_3d: bool, plane_2d: str
) -> str:
    if is_3d:
        return physical_axis
    grid_x, grid_y, _ = grid_axes_in_physical_frame_2d(plane_2d)
    return {grid_x: "x", grid_y: "y"}[physical_axis]


def _assert_mirrored(values, axis: int, *, name: str) -> None:
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[axis] <= 1:
        return
    scale = max(float(np.max(np.abs(array))), 1.0)
    if not np.allclose(
        array,
        np.flip(array, axis=axis),
        rtol=1e-7,
        atol=1e-9 * scale,
    ):
        raise ValueError(
            f"Simulation symmetry requires {name} to be mirror symmetric "
            f"along storage axis {axis}."
        )


def _crop(values, target_shape: tuple[int, ...], *, prefix: int = 0):
    array = np.asarray(values)
    if array.ndim == 0:
        return array
    slices = (slice(None),) * prefix + tuple(
        slice(0, size) for size in target_shape
    )
    return array[slices]


def reduce_material_grid(
    material_grid: MaterialGrid,
    symmetry: tuple[int, int, int],
    *,
    is_3d: bool,
    plane_2d: str,
) -> MaterialGrid:
    """Validate reflection symmetry and retain the low-coordinate half per axis."""
    if not any(symmetry):
        return material_grid
    if material_grid.uses_full_permittivity:
        raise ValueError(
            "Simulation symmetry does not yet support off-diagonal permittivity."
        )
    assert material_grid.grid is not None
    grid = material_grid.grid
    storage_axes: list[int] = []
    reduced_grid_axes: set[str] = set()
    for index, physical_axis in enumerate(_PHYSICAL_AXES):
        if not symmetry[index]:
            continue
        storage_axis = _storage_axis_for_physical(
            physical_axis, is_3d=is_3d, plane_2d=plane_2d
        )
        grid_axis = _grid_axis_for_physical(
            physical_axis, is_3d=is_3d, plane_2d=plane_2d
        )
        edges = np.asarray(grid.axis_edges(grid_axis), dtype=float)
        cell_count = int(edges.size - 1)
        if cell_count % 2:
            raise ValueError(
                f"Simulation symmetry along {physical_axis} requires an even cell "
                f"count; got {cell_count}."
            )
        center = 0.5 * float(edges[0] + edges[-1])
        tolerance = max(
            1e-12 * max(float(np.max(np.abs(edges))), 1.0),
            64.0 * np.finfo(float).eps,
        )
        if not np.allclose(
            edges + edges[::-1], 2.0 * center, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                f"Simulation grid must be mirror symmetric along {physical_axis}."
            )
        storage_axes.append(storage_axis)
        reduced_grid_axes.add(grid_axis)

    cell_fields = {
        "permittivity": material_grid.permittivity,
        "conductivity": material_grid.conductivity,
        "permeability": material_grid.permeability,
    }
    for name, values in cell_fields.items():
        for axis in storage_axes:
            _assert_mirrored(values, axis, name=name)
    for name, values in material_grid.yee_materials.items():
        for axis in storage_axes:
            _assert_mirrored(values, axis, name=name)
    for family, mapping in (
        ("tensor", material_grid.tensors),
        ("Yee tensor", material_grid.yee_tensors),
    ):
        for name, values in mapping.items():
            for axis in storage_axes:
                _assert_mirrored(values, axis + 1, name=f"{family} {name}")

    new_edges = {}
    for axis_index, axis in enumerate(_PHYSICAL_AXES):
        edges = np.asarray(grid.axis_edges(axis))
        new_edges[axis] = (
            edges[: grid.shape[axis_index] // 2 + 1]
            if axis in reduced_grid_axes
            else edges
        )
    reduced_grid = RectilinearGrid(
        new_edges["x"], new_edges["y"], new_edges["z"]
    )
    reduced_shape = (
        reduced_grid.shape_zyx
        if is_3d
        else (reduced_grid.shape[1], reduced_grid.shape[0])
    )
    reduced_component_shapes = component_shapes(
        reduced_shape, material_grid.polarization or "tm"
    )
    yee_materials = {
        name: _crop(values, reduced_component_shapes[_COMPONENT_FOR_YEE[name]])
        for name, values in material_grid.yee_materials.items()
    }
    tensors = {
        name: _crop(values, reduced_shape, prefix=1)
        for name, values in material_grid.tensors.items()
    }
    yee_tensors = {}
    for name, values in material_grid.yee_tensors.items():
        target = (
            tuple(size + 1 for size in reduced_shape)
            if name == "eps_node"
            else reduced_component_shapes[_COMPONENT_FOR_YEE[name]]
        )
        yee_tensors[name] = _crop(values, target, prefix=1)
    return MaterialGrid(
        permittivity=_crop(material_grid.permittivity, reduced_shape),
        conductivity=_crop(material_grid.conductivity, reduced_shape),
        permeability=_crop(material_grid.permeability, reduced_shape),
        resolution=material_grid.resolution,
        shape=reduced_shape,
        yee_materials=yee_materials,
        tensors=tensors,
        smoothing=material_grid.smoothing,
        origin=reduced_grid.origin,
        polarization=material_grid.polarization,
        yee_tensors=yee_tensors,
        grid=reduced_grid,
    )
