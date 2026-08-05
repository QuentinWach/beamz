"""Shared extraction and solving for finite 3D mode planes."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from typing import Literal, cast, overload

import numpy as np

from beamz.design.grid import RectilinearGrid
from beamz.devices._placement import snap_centered_extent
from beamz.lattice import (
    component_shape_3d,
    sample_voxel_grid_at_component_3d,
    sample_voxel_grid_at_e_component_3d_centered,
)

from .discrete import AxisName, DiscreteMode, ModePlaneSpec, solve_beamz_mode
from .solver import solve_grid

_AXIS_POS_3D = {"z": 0, "y": 1, "x": 2}
_TRANSVERSE_AXES_3D: dict[str, tuple[AxisName, AxisName]] = {
    "x": ("z", "y"),
    "y": ("z", "x"),
    "z": ("y", "x"),
}
MODE_PLANE_APERTURE_PAD_CELLS = 2
MODE_PLANE_APERTURE_WINDOW_ALPHA = 0.2
SignedAxis = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
_SIGNED_AXES = frozenset({"+x", "-x", "+y", "-y", "+z", "-z"})


@overload
def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 1,
    direction: SignedAxis = "+x",
    filter_pol: Literal["te", "tm"] | None = None,
    return_fields: Literal[True] = True,
    target_neff: float | None = None,
    *,
    eps_yy: np.ndarray | None = None,
    eps_zz: np.ndarray | None = None,
    mu_xx: np.ndarray | None = None,
    mu_yy: np.ndarray | None = None,
    mu_zz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]: ...


@overload
def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 1,
    direction: SignedAxis = "+x",
    filter_pol: Literal["te", "tm"] | None = None,
    return_fields: Literal[False] = False,
    target_neff: float | None = None,
    *,
    eps_yy: np.ndarray | None = None,
    eps_zz: np.ndarray | None = None,
    mu_xx: np.ndarray | None = None,
    mu_yy: np.ndarray | None = None,
    mu_zz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...


def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 1,
    direction: SignedAxis = "+x",
    filter_pol: Literal["te", "tm"] | None = None,
    return_fields: bool = False,
    target_neff: float | None = None,
    *,
    eps_yy: np.ndarray | None = None,
    eps_zz: np.ndarray | None = None,
    mu_xx: np.ndarray | None = None,
    mu_yy: np.ndarray | None = None,
    mu_zz: np.ndarray | None = None,
    grid_edges: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Solve a one- or two-dimensional BeamZ material profile."""
    eps_array = np.asarray(eps, dtype=np.complex128)
    if eps_array.ndim not in {1, 2}:
        raise ValueError("solve_modes expects a 1D or 2D permittivity array")
    if npml < 0:
        raise ValueError("npml must be non-negative")
    if m <= 0:
        raise ValueError("m must be positive")
    if filter_pol not in {None, "te", "tm"}:
        raise ValueError("filter_pol must be 'te', 'tm', or None")

    signed_axis = str(direction).lower()
    if signed_axis not in _SIGNED_AXES:
        raise ValueError(
            "direction must be one of '+x', '-x', '+y', '-y', '+z', or '-z'"
        )
    axis = signed_axis[1]
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]

    is_plane = eps_array.ndim == 2
    # BeamZ stores 3D arrays in (z, y, x) order, so a normal slice retains the
    # reverse of the solver's Cartesian tangential-axis order.
    plane = eps_array[:, None] if not is_plane else eps_array.T
    if grid_edges is None:
        edges = tuple(
            tuple(np.arange(size + 1, dtype=float) * float(dL) / 1e-6)
            for size in plane.shape
        )
    else:
        supplied = tuple(np.asarray(values, dtype=float) for values in grid_edges)
        if len(supplied) != eps_array.ndim:
            raise ValueError("grid_edges must provide one edge array per profile axis")
        supplied = supplied if not is_plane else supplied[::-1]
        if not is_plane:
            supplied = (*supplied, np.asarray([0.0, float(dL)]))
        if any(
            values.size != size + 1
            for values, size in zip(supplied, plane.shape, strict=True)
        ):
            raise ValueError("grid_edges lengths must match the permittivity profile")
        edges = tuple(tuple(values / 1e-6) for values in supplied)

    def solver_plane(value):
        if value is None:
            return None
        array = np.asarray(value, dtype=np.complex128)
        if array.shape != eps_array.shape:
            raise ValueError(
                f"tensor material component has shape {array.shape}, expected {eps_array.shape}"
            )
        return array[:, None] if not is_plane else array.T

    result = solve_grid(
        eps_xx=plane,
        eps_yy=solver_plane(eps_yy),
        eps_zz=solver_plane(eps_zz),
        mu_xx=solver_plane(mu_xx),
        mu_yy=solver_plane(mu_yy),
        mu_zz=solver_plane(mu_zz),
        x_edges=edges[0],
        y_edges=edges[1],
        freqs=[float(omega) / (2.0 * np.pi)],
        direction="+" if signed_axis[0] == "+" else "-",
        num_modes=2 * int(m) + 5,
        target_neff=target_neff,
        pml=(int(npml), int(npml)),
        normal_axis=cast(Literal[0, 1, 2], axis_index),
    )
    fields = {
        name: _mode_planes(data, axis) for name, data in result.field_components.items()
    }
    if is_plane:
        fields = {name: values.swapaxes(-2, -1) for name, values in fields.items()}
    order = _mode_order(result.n_complex.values[0], fields, axis, filter_pol)[: int(m)]
    neffs = np.asarray(result.n_complex.values[0, order], dtype=np.complex128)
    electric = np.stack([fields[name][order] for name in ("Ex", "Ey", "Ez")], axis=1)
    magnetic = np.stack([fields[name][order] for name in ("Hx", "Hy", "Hz")], axis=1)
    if axis_index == 1:
        magnetic = -magnetic
    if eps_array.ndim == 1:
        electric = electric[..., 0]
        magnetic = magnetic[..., 0]

    if return_fields:
        return neffs, electric, magnetic, axis_index

    dominant = np.argmax(
        np.linalg.norm(electric.reshape(len(order), 3, -1), axis=2), axis=1
    )
    vectors = [
        np.ravel(electric[index, component]) for index, component in enumerate(dominant)
    ]
    return neffs, np.column_stack(vectors)


def _mode_planes(data_array, normal_axis: str) -> np.ndarray:
    selected = data_array.isel(f=0)
    values = np.take(
        np.asarray(selected.values),
        0,
        axis=selected.dims.index(normal_axis),
    )
    return np.moveaxis(values, -1, 0)


def _mode_order(neffs, fields, axis, polarization):
    descending = np.argsort(np.real(neffs))[::-1]
    if polarization is None:
        return descending
    first, second = {
        "x": ("Ey", "Ez"),
        "y": ("Ex", "Ez"),
        "z": ("Ex", "Ey"),
    }[axis]
    axes = tuple(range(1, fields[first].ndim))
    first_power = np.sum(np.abs(fields[first]) ** 2, axis=axes)
    second_power = np.sum(np.abs(fields[second]) ** 2, axis=axes)
    fraction = first_power / np.maximum(first_power + second_power, np.finfo(float).eps)
    matching = fraction >= 0.5 if polarization == "te" else fraction < 0.5
    return np.concatenate(
        (descending[matching[descending]], descending[~matching[descending]])
    )


def mode_plane_outer_pad_cells(width, height, resolution) -> int:
    """Return the shared finite-domain padding for 3D modal solves."""
    extent = max(float(width), float(width if height is None else height))
    res = max(float(resolution), 1e-30)
    return int(np.clip(np.ceil(0.5 * extent / res), 8, 48))


def _axis_counts_from_grid_shape(grid_shape):
    nz, ny, nx = (int(v) for v in grid_shape)
    return {"z": nz, "y": ny, "x": nx}


def _center_by_axis(center, grid_shape, resolution):
    counts = _axis_counts_from_grid_shape(grid_shape)
    values = tuple(float(v) for v in center)
    return {
        "x": values[0] if len(values) > 0 else 0.5 * counts["x"] * float(resolution),
        "y": values[1] if len(values) > 1 else 0.5 * counts["y"] * float(resolution),
        "z": values[2] if len(values) > 2 else 0.5 * counts["z"] * float(resolution),
    }


def _plane_extent_by_axis(axis, width, height):
    height = float(width if height is None else height)
    width = float(width)
    if axis == "x":
        return {"y": width, "z": height}
    if axis == "y":
        return {"x": width, "z": height}
    if axis == "z":
        return {"x": width, "y": height}
    raise ValueError(f"Unsupported mode-plane axis {axis!r}")


def _ensure_min_interval(start, stop, limit, min_cells=2):
    start = max(0, min(int(start), int(limit)))
    stop = max(start, min(int(stop), int(limit)))
    need = max(1, int(min_cells))
    while stop - start < need and (start > 0 or stop < int(limit)):
        if start > 0:
            start -= 1
        if stop - start >= need:
            break
        if stop < int(limit):
            stop += 1
    return start, stop


def _local_mode_plane_spec(
    eps_profile,
    *,
    axis,
    grid_shape,
    center,
    width,
    height,
    plane_index,
    offset_index,
    resolution,
    snapped_region=None,
    aperture_pad_cells=2,
    material_origin_zyx=(0, 0, 0),
    grid=None,
):
    """Build a compact local mode plane and metadata to shift it globally."""
    axis = str(axis).lower()
    counts = _axis_counts_from_grid_shape(grid_shape)
    centers = _center_by_axis(center, grid_shape, resolution)
    if snapped_region is not None:
        for name in ("x", "y", "z"):
            with suppress(Exception):
                centers[name] = float(snapped_region.axis_coord(name))

    origin = {"z": 0, "y": 0, "x": 0}
    material_origin: dict[str, int] = dict(
        zip(("z", "y", "x"), (int(value) for value in material_origin_zyx), strict=True)
    )
    local_counts = dict(counts)
    crop_slices = []
    pad = max(0, int(aperture_pad_cells))
    extents = _plane_extent_by_axis(axis, width, height)

    for transverse_axis in _TRANSVERSE_AXES_3D[axis]:
        interval = None
        if snapped_region is not None:
            try:
                interval = snapped_region.axis_interval(transverse_axis)
            except Exception:
                interval = None
        if interval is None:
            interval = snap_centered_extent(
                centers[transverse_axis],
                extents[transverse_axis],
                float(resolution),
                counts[transverse_axis],
                min_cells=2,
            )
        start = int(interval.start) - pad
        stop = int(interval.stop) + pad
        start, stop = _ensure_min_interval(start, stop, counts[transverse_axis])
        origin[transverse_axis] = int(start)
        local_counts[transverse_axis] = int(stop) - int(start)
        crop_slices.append(
            slice(
                int(start) - material_origin[transverse_axis],
                int(stop) - material_origin[transverse_axis],
            )
        )

    normal_origin = min(int(plane_index), int(offset_index))
    normal_stop = max(int(plane_index), int(offset_index)) + 2
    normal_origin, normal_stop = _ensure_min_interval(
        normal_origin,
        normal_stop,
        counts[axis],
    )
    origin[axis] = int(normal_origin)
    local_counts[axis] = int(normal_stop) - int(normal_origin)
    local_plane_index = int(plane_index) - int(normal_origin)
    local_offset_index = int(offset_index) - int(normal_origin)

    eps_local = np.asarray(eps_profile)[tuple(crop_slices)]
    local_grid_shape = (
        int(local_counts["z"]),
        int(local_counts["y"]),
        int(local_counts["x"]),
    )
    if grid is None:
        axis_origins = {
            name: float(origin[name]) * float(resolution) for name in ("x", "y", "z")
        }
        local_grid = None
    else:
        axis_origins = {
            name: float(grid.axis_edges(name)[origin[name]]) for name in ("x", "y", "z")
        }
        local_grid = RectilinearGrid(
            *(
                np.asarray(grid.axis_edges(name))[
                    origin[name] : origin[name] + local_counts[name] + 1
                ]
                - axis_origins[name]
                for name in ("x", "y", "z")
            )
        )
    local_center = tuple(
        float(centers[name]) - axis_origins[name] for name in ("x", "y", "z")
    )
    component_shapes = {
        component: component_shape_3d(component, local_grid_shape)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    return {
        "scalar_permittivity": eps_local,
        "grid_shape": local_grid_shape,
        "component_shapes": component_shapes,
        "center": local_center,
        "plane_index": int(local_plane_index),
        "offset_index": int(local_offset_index),
        "grid": local_grid,
        "axis_origins": axis_origins,
        "origin_zyx": (
            int(origin["z"]),
            int(origin["y"]),
            int(origin["x"]),
        ),
    }


def solve_mode_plane_3d(
    permittivity,
    permeability,
    *,
    material_tensors=None,
    yee_materials=None,
    frequency,
    resolution,
    dt,
    axis,
    direction,
    grid_shape,
    center,
    width,
    height,
    plane_index,
    offset_index,
    mode_index,
    polarization,
    target_neff,
    num_modes,
    snapped_region,
    solver_direction=None,
    material_origin_zyx=(0, 0, 0),
    solver=solve_beamz_mode,
    grid=None,
) -> DiscreteMode:
    """Solve one finite 3D mode plane and return globally indexed fields."""
    axis_index = _AXIS_POS_3D[axis]
    profile_index = int(plane_index) - int(material_origin_zyx[axis_index])
    eps_profile = np.take(permittivity, profile_index, axis=axis_index)
    local_plane = _local_mode_plane_spec(
        eps_profile,
        axis=axis,
        grid_shape=grid_shape,
        center=center,
        width=width,
        height=height,
        plane_index=plane_index,
        offset_index=offset_index,
        resolution=resolution,
        snapped_region=snapped_region,
        aperture_pad_cells=mode_plane_outer_pad_cells(width, height, resolution),
        material_origin_zyx=material_origin_zyx,
        grid=grid,
    )
    target = target_neff
    if target is None:
        target = 0.98 * np.sqrt(
            max(float(np.max(np.real(local_plane["scalar_permittivity"]))), 1e-12)
        )
    sampling_plane = dict(local_plane)
    sampling_plane["origin_zyx"] = tuple(
        int(global_offset) - int(region_offset)
        for global_offset, region_offset in zip(
            local_plane["origin_zyx"], material_origin_zyx, strict=True
        )
    )
    component_permittivity, component_permeability = _local_component_materials(
        permittivity,
        permeability,
        sampling_plane,
        yee_materials=yee_materials,
    )
    diagonal_permittivity, diagonal_permeability = _local_diagonal_materials(
        material_tensors,
        local_plane,
        axis=axis,
        plane_index=plane_index,
        material_origin_zyx=material_origin_zyx,
    )
    mode = solver(
        ModePlaneSpec(
            scalar_permittivity=np.asarray(
                local_plane["scalar_permittivity"], dtype=np.complex128
            ),
            frequency=float(frequency),
            resolution=float(resolution),
            dt=None if dt is None else float(dt),
            axis=axis,
            direction=direction,
            solver_direction=solver_direction or direction,
            transverse_axes=_TRANSVERSE_AXES_3D[axis],
            grid_shape=local_plane["grid_shape"],
            component_permittivity=component_permittivity,
            component_permeability=component_permeability,
            diagonal_permittivity=diagonal_permittivity,
            diagonal_permeability=diagonal_permeability,
            center=local_plane["center"],
            width=float(width),
            height=float(height),
            plane_index=int(local_plane["plane_index"]),
            offset_index=int(local_plane["offset_index"]),
            mode_index=int(mode_index),
            polarization=polarization,
            target_neff=target,
            num_modes=int(num_modes),
            aperture_pad_cells=MODE_PLANE_APERTURE_PAD_CELLS,
            aperture_window_alpha=MODE_PLANE_APERTURE_WINDOW_ALPHA,
            grid=local_plane["grid"],
        )
    )
    return _shift_discrete_mode_to_global(
        mode,
        origin_zyx=local_plane["origin_zyx"],
        axis=axis,
        resolution=resolution,
        axis_offset=local_plane["axis_origins"][axis],
    )


def _local_diagonal_materials(
    tensors,
    local_plane,
    *,
    axis,
    plane_index,
    material_origin_zyx,
):
    if not tensors:
        return {}, {}
    axis_position = {"z": 0, "y": 1, "x": 2}
    normal_axis = axis_position[str(axis)]
    profile_index = int(plane_index) - int(material_origin_zyx[normal_axis])
    transverse_axes = _TRANSVERSE_AXES_3D[str(axis)]
    origin = dict(zip(("z", "y", "x"), local_plane["origin_zyx"], strict=True))
    material_origin = dict(zip(("z", "y", "x"), material_origin_zyx, strict=True))
    counts = _axis_counts_from_grid_shape(local_plane["grid_shape"])
    crop = tuple(
        slice(
            int(origin[name]) - int(material_origin[name]),
            int(origin[name]) - int(material_origin[name]) + int(counts[name]),
        )
        for name in transverse_axes
    )

    def profiles(property_name):
        tensor = np.asarray(tensors[property_name])
        if tensor.ndim != 4 or tensor.shape[0] not in (1, 3, 6):
            raise ValueError(
                f"{property_name} tensor must have compact shape (1|3|6, z, y, x)"
            )
        return {
            component: np.take(
                tensor[0 if tensor.shape[0] == 1 else index],
                profile_index,
                axis=normal_axis,
            )[crop]
            for index, component in enumerate(("xx", "yy", "zz"))
        }

    return profiles("epsilon"), profiles("mu")


def _local_component_materials(
    permittivity, permeability, local_plane, *, yee_materials=None
):
    """Sample local voxel materials on the six Yee component lattices."""
    origin = tuple(int(value) for value in local_plane["origin_zyx"])
    grid_shape = tuple(int(value) for value in local_plane["grid_shape"])
    shapes = local_plane["component_shapes"]
    if yee_materials:

        def crop(name, component):
            values = np.asarray(yee_materials[name])
            selection = tuple(
                slice(start, start + count)
                for start, count in zip(origin, shapes[component], strict=True)
            )
            result = values[selection]
            if result.shape != shapes[component]:
                raise ValueError(
                    f"Local {name} crop has shape {result.shape}, expected "
                    f"{shapes[component]}."
                )
            return result

        required = {"eps_x", "eps_y", "eps_z", "mu_hx", "mu_hy", "mu_hz"}
        missing = required - set(yee_materials)
        if missing:
            raise ValueError(
                f"Mode plane is missing Yee materials: {', '.join(sorted(missing))}."
            )
        return (
            {
                component: crop(name, component)
                for component, name in zip(
                    ("Ex", "Ey", "Ez"),
                    ("eps_x", "eps_y", "eps_z"),
                    strict=True,
                )
            },
            {
                component: crop(name, component)
                for component, name in zip(
                    ("Hx", "Hy", "Hz"),
                    ("mu_hx", "mu_hy", "mu_hz"),
                    strict=True,
                )
            },
        )
    region = tuple(
        slice(start, start + count)
        for start, count in zip(origin, grid_shape, strict=True)
    )

    eps_grid = np.asarray(permittivity)
    mu_grid = np.asarray(permeability)
    if mu_grid.ndim == 0:
        mu_grid = np.full(eps_grid.shape, mu_grid.item(), dtype=mu_grid.dtype)
    else:
        mu_grid = np.broadcast_to(mu_grid, eps_grid.shape)
    eps_local = eps_grid[region]
    mu_local = mu_grid[region]
    component_permittivity = {
        component: np.asarray(
            sample_voxel_grid_at_e_component_3d_centered(
                eps_local,
                component,
                stored_shape=shapes[component],
            )
        )
        for component in ("Ex", "Ey", "Ez")
    }
    component_permeability = {
        component: np.asarray(
            sample_voxel_grid_at_component_3d(
                mu_local,
                component,
                stored_shape=shapes[component],
            )
        )
        for component in ("Hx", "Hy", "Hz")
    }
    return component_permittivity, component_permeability


def _shift_3d_index(index, origin_zyx):
    out = []
    for item, offset in zip(index, origin_zyx, strict=True):
        offset = int(offset)
        if isinstance(item, slice):
            start = None if item.start is None else int(item.start) + offset
            stop = None if item.stop is None else int(item.stop) + offset
            out.append(slice(start, stop, item.step))
        else:
            out.append(int(item) + offset)
    return tuple(out)


def _shift_discrete_mode_to_global(
    discrete_mode: DiscreteMode, *, origin_zyx, axis, resolution, axis_offset=None
) -> DiscreteMode:
    """Shift a mode result solved on a local crop back to global indices."""
    component_indices = {
        name: _shift_3d_index(index, origin_zyx)
        for name, index in discrete_mode.component_indices.items()
    }
    if axis_offset is None:
        axis_offset = float(origin_zyx[_AXIS_POS_3D[str(axis).lower()]]) * float(
            resolution
        )
    return replace(
        discrete_mode,
        component_indices=component_indices,
        phase_reference_coord=float(discrete_mode.phase_reference_coord) + axis_offset,
        phase_plane_coord=float(discrete_mode.phase_plane_coord) + axis_offset,
    )
