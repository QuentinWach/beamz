"""Shared Yee-lattice geometry, coordinates, sampling, and material colocation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0, µm

_COMPONENT_AXIS_OFFSETS_3D = {
    "Ex": {"z": 0.0, "y": 0.0, "x": 0.5},
    "Ey": {"z": 0.0, "y": 0.5, "x": 0.0},
    "Ez": {"z": 0.5, "y": 0.0, "x": 0.0},
    "Hx": {"z": 0.5, "y": 0.5, "x": 0.0},
    "Hy": {"z": 0.5, "y": 0.0, "x": 0.5},
    "Hz": {"z": 0.0, "y": 0.5, "x": 0.5},
}


def component_axis_offsets_3d(component: str) -> dict[str, float]:
    """Return physical Yee offsets, in grid-cell units, for a 3D component."""
    try:
        return dict(_COMPONENT_AXIS_OFFSETS_3D[component])
    except KeyError as exc:
        raise ValueError(f"Unsupported component {component!r}") from exc


def component_shape_3d(
    component: str, grid_shape: tuple[int, ...]
) -> tuple[int, int, int]:
    """Return the complete stored 3D Yee support for a field component."""
    # Material cells are the common basis; complete supports retain both domain walls
    # instead of introducing a second expanded representation only for PEC cavities.
    if len(grid_shape) != 3:
        raise ValueError(f"A 3D grid shape must have three dimensions: {grid_shape!r}")
    offsets = component_axis_offsets_3d(component)
    shape = tuple(
        int(size) + int(offsets[axis] == 0.0)
        for axis, size in zip(("z", "y", "x"), grid_shape, strict=True)
    )
    return shape[0], shape[1], shape[2]


def component_coordinates_3d_um(
    component: str,
    grid_shape: tuple[int, ...],
    dx_um: float,
) -> dict[str, np.ndarray]:
    """Return Beamz raw Yee sample coordinates for a stored 3D component."""

    # Derive coordinates from the same shape/offset contract used by storage allocation.
    shape = component_shape_3d(component, grid_shape)
    offsets = component_axis_offsets_3d(component)
    return {
        "z": (np.arange(shape[0], dtype=np.float64) + offsets["z"]) * dx_um,
        "y": (np.arange(shape[1], dtype=np.float64) + offsets["y"]) * dx_um,
        "x": (np.arange(shape[2], dtype=np.float64) + offsets["x"]) * dx_um,
    }


def component_coordinates_rectilinear(
    component: str,
    grid,
    *,
    plane: str | None = None,
    polarization: str = "tm",
) -> dict[str, np.ndarray]:
    """Return exact physical Yee coordinates from rectilinear grid edges."""

    def samples(axis: str, offset: float) -> np.ndarray:
        edges = np.asarray(grid.axis_edges(axis), dtype=np.float64)
        return edges if offset == 0.0 else 0.5 * (edges[:-1] + edges[1:])

    if plane is None:
        offsets = component_axis_offsets_3d(component)
        return {axis: samples(axis, offsets[axis]) for axis in ("z", "y", "x")}

    normalized_plane = str(plane).lower()
    canonical = canonical_component_2d(component, normalized_plane, polarization)
    if canonical is None:
        return {}
    row_offset, column_offset = {
        "Ez": (0.0, 0.0),
        "Hx": (0.5, 0.0),
        "Hy": (0.0, 0.5),
        "Ex": (0.0, 0.5),
        "Ey": (0.5, 0.0),
        "Hz": (0.5, 0.5),
    }[canonical]
    row_axis, column_axis = {
        "xy": ("y", "x"),
        "yz": ("z", "y"),
        "xz": ("z", "x"),
    }[normalized_plane]
    return {
        row_axis: samples("y", row_offset),
        column_axis: samples("x", column_offset),
    }


def grid_axes_in_physical_frame_2d(plane: str) -> tuple[str, str, str]:
    """Map stored rectilinear ``(x, y, z)`` axes to physical axes for a 2D plane."""
    try:
        return {
            "xy": ("x", "y", "z"),
            "xz": ("x", "z", "y"),
            "yz": ("y", "z", "x"),
        }[str(plane).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported 2D plane {plane!r}.") from exc


def grid_vector_to_physical_2d(
    values: tuple[float, float, float], plane: str
) -> tuple[float, float, float]:
    """Permute a stored-grid vector into public physical ``(x, y, z)`` order."""
    stored = tuple(float(value) for value in values)
    if len(stored) != 3:
        raise ValueError("A stored-grid vector must contain three values.")
    physical = dict.fromkeys("xyz", 0.0)
    for axis, value in zip(grid_axes_in_physical_frame_2d(plane), stored, strict=True):
        physical[axis] = value
    return physical["x"], physical["y"], physical["z"]


def physical_vector_to_grid_2d(
    values: tuple[float, float, float], plane: str
) -> tuple[float, float, float]:
    """Permute a public physical vector into stored-grid ``(x, y, z)`` order."""
    physical = dict(zip("xyz", map(float, values), strict=True))
    grid_x, grid_y, grid_z = grid_axes_in_physical_frame_2d(plane)
    return physical[grid_x], physical[grid_y], physical[grid_z]


def in_plane_vector_2d(
    values: tuple[float, float, float], plane: str
) -> tuple[float, float]:
    """Select the public offsets corresponding to stored grid ``(x, y)`` order."""
    physical = dict(zip("xyz", map(float, values), strict=True))
    grid_x, grid_y, _ = grid_axes_in_physical_frame_2d(plane)
    return physical[grid_x], physical[grid_y]


def coordinates_in_public_frame(
    coordinates: Mapping[str, Any],
    coordinate_offset: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """Translate solver-local axis coordinates into the public coordinate frame.

    Simulation devices are lowered according to ``local = public + offset``.
    Analysis therefore applies the inverse translation to exact Yee coordinates.
    """

    offsets = dict(zip("xyz", map(float, coordinate_offset), strict=True))
    return {
        axis: np.asarray(values, dtype=np.float64) - offsets.get(axis, 0.0)
        for axis, values in coordinates.items()
    }


_FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def common_grid_shape_3d(fields) -> tuple[int, int, int]:
    """Return the largest logical Yee component support for a field container."""
    shapes = getattr(fields, "_logical_component_shapes", None)
    if not shapes:
        shapes = getattr(fields, "component_shapes", None)
    if not shapes:
        shapes = {
            name: tuple(getattr(fields, name).shape)
            for name in _FIELD_COMPONENTS
            if hasattr(fields, name)
        }
    rank3 = [
        tuple(int(v) for v in shape) for shape in shapes.values() if len(shape) == 3
    ]
    if not rank3:
        fallback = getattr(fields, "grid_shape", np.asarray(fields.permittivity).shape)
        if len(fallback) != 3:
            raise ValueError("A common 3D Yee shape requires rank-3 fields.")
        return int(fallback[0]), int(fallback[1]), int(fallback[2])
    maxima = tuple(max(shape[axis] for shape in rank3) for axis in range(3))
    return maxima[0], maxima[1], maxima[2]


def plane_axes_3d(normal_axis: str) -> tuple[str, str]:
    """Return tangential axes in stored plane-array order."""
    try:
        return {
            "x": ("z", "y"),
            "y": ("z", "x"),
            "z": ("y", "x"),
        }[str(normal_axis).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported plane normal {normal_axis!r}.") from exc


def _uniform_axis_centers(lower: float, upper: float, count: int) -> np.ndarray:
    count = max(0, int(count))
    if count == 0:
        return np.zeros((0,), dtype=np.float64)
    lower, upper = float(lower), float(upper)
    if count == 1:
        return np.asarray([0.5 * (lower + upper)], dtype=np.float64)
    step = (upper - lower) / float(count)
    return lower + (np.arange(count, dtype=np.float64) + 0.5) * step


def yee_plane_coordinates_3d(center, size, normal_axis: str, region, grid=None):
    """Return tangential cell-center coordinates for a snapped physical plane."""
    center = tuple(float(value) for value in center)
    size = tuple(float(value) for value in size)
    axis0, axis1 = plane_axes_3d(normal_axis)
    axis_position = {"x": 0, "y": 1, "z": 2}
    coordinates = []
    for axis in (axis0, axis1):
        index = axis_position[axis]
        interval = region.axis_interval(axis)
        if interval is None:
            raise ValueError(f"Plane is missing its {axis!r} tangential interval.")
        if grid is not None:
            coordinates.append(
                np.asarray(grid.centers(axis))[int(interval.start) : int(interval.stop)]
            )
        else:
            lower = center[index] - 0.5 * size[index]
            upper = center[index] + 0.5 * size[index]
            coordinates.append(
                _uniform_axis_centers(
                    lower, upper, int(interval.stop) - int(interval.start)
                )
            )
    return coordinates[0], coordinates[1]


def linear_interpolation_plan(source, target):
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if source.size == 0:
        raise ValueError("Yee interpolation source coordinates cannot be empty.")
    if source.size == 1:
        indices = np.zeros(target.size, dtype=np.int32)
        return (
            indices,
            indices,
            np.ones(target.size, dtype=np.float32),
            np.zeros(target.size, dtype=np.float32),
        )
    high = np.clip(np.searchsorted(source, target, side="right"), 1, source.size - 1)
    low = high - 1
    low[target <= source[0]] = 0
    high[target <= source[0]] = 0
    low[target >= source[-1]] = source.size - 1
    high[target >= source[-1]] = source.size - 1
    denominator = source[high] - source[low]
    alpha = np.divide(
        target - source[low],
        denominator,
        out=np.zeros_like(target),
        where=(high != low) & (np.abs(denominator) > 0.0),
    )
    weight_high = alpha.astype(np.float32)
    weight_low = (1.0 - alpha).astype(np.float32)
    weight_low[high == low] = 1.0
    weight_high[high == low] = 0.0
    return low.astype(np.int32), high.astype(np.int32), weight_low, weight_high


def _component_plane_plan_3d(
    component: str,
    *,
    normal_axis: str,
    plane_position: float,
    coordinates: tuple[np.ndarray, np.ndarray],
    resolution: float,
    grid_shape: tuple[int, int, int],
    field_shape: tuple[int, int, int],
    grid=None,
):
    axis0, axis1 = plane_axes_3d(normal_axis)
    if grid is None:
        component_coords = {
            axis: values * µm
            for axis, values in component_coordinates_3d_um(
                component,
                grid_shape,
                float(resolution / µm),
            ).items()
        }
    else:
        offsets = component_axis_offsets_3d(component)
        component_coords = {
            axis: (
                np.asarray(grid.axis_edges(axis))
                if offsets[axis] == 0.0
                else np.asarray(grid.centers(axis))
            )
            for axis in ("x", "y", "z")
        }
    lengths: dict[str, int] = dict(zip(("z", "y", "x"), field_shape, strict=True))
    source0 = np.asarray(component_coords[axis0])[: lengths[axis0]]
    source1 = np.asarray(component_coords[axis1])[: lengths[axis1]]
    source_normal = np.asarray(component_coords[normal_axis])[: lengths[normal_axis]]
    grid0, grid1 = np.meshgrid(*coordinates, indexing="ij")
    target0, target1 = grid0.reshape(-1), grid1.reshape(-1)
    target_normal = np.full_like(target0, float(plane_position))
    plans = (
        linear_interpolation_plan(source0, target0),
        linear_interpolation_plan(source1, target1),
        linear_interpolation_plan(source_normal, target_normal),
    )
    corner_bits = np.asarray(
        [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)],
        dtype=np.int32,
    )
    indices = []
    weights = []
    for dimension, (low, high, weight_low, weight_high) in enumerate(plans):
        bit = corner_bits[:, dimension][None, :]
        indices.append(np.where(bit == 0, low[:, None], high[:, None]))
        weights.append(np.where(bit == 0, weight_low[:, None], weight_high[:, None]))
    index_map = {
        axis0: indices[0],
        axis1: indices[1],
        normal_axis: indices[2],
    }
    flat_indices = np.asarray(
        np.ravel_multi_index(
            (index_map["z"], index_map["y"], index_map["x"]),
            dims=field_shape,
        ),
        dtype=np.int32,
    )
    combined_weights = np.prod(np.stack(weights), axis=0).astype(np.float32)
    nonzero = np.abs(combined_weights) > 4.0 * np.finfo(np.float32).eps
    combined_weights = np.where(nonzero, combined_weights, 0.0)
    row_sum = np.sum(combined_weights, axis=1, keepdims=True)
    combined_weights = np.divide(
        combined_weights,
        row_sum,
        out=np.zeros_like(combined_weights),
        where=np.abs(row_sum) > 0.0,
    )
    compact_width = int(np.max(np.sum(nonzero, axis=1), initial=0))
    if 0 < compact_width < combined_weights.shape[1]:
        order = np.argsort(~nonzero, axis=1, kind="stable")[:, :compact_width]
        flat_indices = np.take_along_axis(flat_indices, order, axis=1)
        combined_weights = np.take_along_axis(combined_weights, order, axis=1)
    return flat_indices, combined_weights


def plane_sample_area(coordinates, fallback_step: float) -> float:
    def step(values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size > 1:
            spacing = float(np.median(np.abs(np.diff(values))))
            if np.isfinite(spacing) and spacing > 0.0:
                return spacing * float(values.size) / float(values.size - 1)
        return float(fallback_step)

    return float(step(coordinates[0]) * step(coordinates[1]))


@dataclass(frozen=True, slots=True)
class YeePlaneQuadrature:
    """Compiled field colocation and area integration for one Yee-grid plane."""

    normal_axis: int
    sample_area: float
    integration_weights: np.ndarray
    coordinates: tuple[np.ndarray, np.ndarray]
    plans: Mapping[str, tuple[np.ndarray, np.ndarray]]

    @property
    def point_count(self) -> int:
        return int(self.coordinates[0].size * self.coordinates[1].size)

    def plan(self, component: str):
        return self.plans[component]

    def sample(self, fields: Mapping[str, np.ndarray]):
        return tuple(
            np.sum(
                np.asarray(fields[name]).reshape(-1)[self.plans[name][0]]
                * self.plans[name][1],
                axis=-1,
            )
            for name in _FIELD_COMPONENTS
        )


def compile_yee_plane_quadrature_3d(
    *,
    center,
    size,
    normal_axis: str,
    region,
    resolution: float,
    grid_shape: tuple[int, int, int],
    component_shapes: Mapping[str, tuple[int, int, int]],
    grid=None,
) -> YeePlaneQuadrature:
    """Compile the canonical plane colocation used by monitors and sources."""
    normal_axis = str(normal_axis).lower()
    coordinates = yee_plane_coordinates_3d(center, size, normal_axis, region, grid=grid)
    plane_position = float(tuple(center)[{"x": 0, "y": 1, "z": 2}[normal_axis]])
    plans = {}
    for component in _FIELD_COMPONENTS:
        shape = component_shapes[component]
        plans[component] = _component_plane_plan_3d(
            component,
            normal_axis=normal_axis,
            plane_position=plane_position,
            coordinates=coordinates,
            resolution=float(resolution),
            grid_shape=grid_shape,
            field_shape=(int(shape[0]), int(shape[1]), int(shape[2])),
            grid=grid,
        )
    if grid is None:
        sample_area = plane_sample_area(coordinates, float(resolution))
        integration_weights = np.empty((0,), dtype=np.float64)
    else:
        axis0, axis1 = plane_axes_3d(normal_axis)
        interval0 = region.axis_interval(axis0)
        interval1 = region.axis_interval(axis1)
        widths0 = np.asarray(grid.cell_widths(axis0))[
            int(interval0.start) : int(interval0.stop)
        ]
        widths1 = np.asarray(grid.cell_widths(axis1))[
            int(interval1.start) : int(interval1.stop)
        ]
        integration_weights = (widths0[:, None] * widths1[None, :]).reshape(-1)
        sample_area = float(np.mean(integration_weights))
    return YeePlaneQuadrature(
        normal_axis={"x": 0, "y": 1, "z": 2}[normal_axis],
        sample_area=sample_area,
        integration_weights=integration_weights,
        coordinates=coordinates,
        plans=plans,
    )


def yee_flux(
    samples, normal_axis: int, *, normal_sign=1.0, measure: Any = 1.0, phasor=False
):
    """Integrate signed Poynting flux from six colocated Yee field samples."""
    ex, ey, ez, hx, hy, hz = (jnp.asarray(value) for value in samples)
    if phasor:
        hx, hy, hz = jnp.conjugate(hx), jnp.conjugate(hy), jnp.conjugate(hz)
    sx = ey * hz - ez * hy
    sy = ez * hx - ex * hz
    sz = ex * hy - ey * hx
    if int(normal_axis) < 0:
        flux = jnp.sqrt(jnp.abs(sx) ** 2 + jnp.abs(sy) ** 2 + jnp.abs(sz) ** 2)
    else:
        flux = (sx, sy, sz)[int(normal_axis)] * float(normal_sign)
    total = jnp.sum(flux * jnp.asarray(measure, dtype=flux.dtype))
    return 0.5 * jnp.real(total) if phasor else total


def curl_e_to_h_3d(ex, ey, ez, resolution):
    """Differentiate the complete electric Yee supports onto H sites."""
    return (
        ((ez[:, 1:, :] - ez[:, :-1, :]) - (ey[1:, :, :] - ey[:-1, :, :])) / resolution,
        ((ex[1:, :, :] - ex[:-1, :, :]) - (ez[:, :, 1:] - ez[:, :, :-1])) / resolution,
        ((ey[:, :, 1:] - ey[:, :, :-1]) - (ex[:, 1:, :] - ex[:, :-1, :])) / resolution,
    )


def _rectilinear_inverse_distances(grid, axis: str, *, backward: bool) -> np.ndarray:
    widths = np.asarray(grid.cell_widths(axis), dtype=np.float64)
    if not backward:
        return 1.0 / widths
    inverse = np.empty(widths.size + 1, dtype=np.float64)
    inverse[0], inverse[-1] = 1.0 / widths[0], 1.0 / widths[-1]
    if widths.size > 1:
        inverse[1:-1] = 2.0 / (widths[:-1] + widths[1:])
    return inverse


def curl_e_to_h_3d_metric(ex, ey, ez, grid):
    """Differentiate electric fields using exact rectilinear cell widths."""
    ix = _rectilinear_inverse_distances(grid, "x", backward=False)
    iy = _rectilinear_inverse_distances(grid, "y", backward=False)
    iz = _rectilinear_inverse_distances(grid, "z", backward=False)
    return (
        metric_adjacent_difference(ez, 1, iy) - metric_adjacent_difference(ey, 0, iz),
        metric_adjacent_difference(ex, 0, iz) - metric_adjacent_difference(ez, 2, ix),
        metric_adjacent_difference(ey, 2, ix) - metric_adjacent_difference(ex, 1, iy),
    )


def adjacent_difference(array, axis, resolution):
    resolution = jnp.asarray(resolution, dtype=array.dtype)
    moved = jnp.moveaxis(array, axis, 0)
    if resolution.ndim == 1:
        resolution = resolution.reshape((resolution.size,) + (1,) * (moved.ndim - 1))
    return jnp.moveaxis((moved[1:] - moved[:-1]) / resolution, 0, axis)


def metric_adjacent_difference(array, axis, inverse_distance):
    """Apply an adjacent difference scaled by a precomputed inverse distance."""
    values = jnp.asarray(array)
    inverse_distance = jnp.asarray(inverse_distance, dtype=values.dtype)
    moved = jnp.moveaxis(values, axis, 0)
    if inverse_distance.ndim == 1:
        inverse_distance = inverse_distance.reshape(
            (inverse_distance.size,) + (1,) * (moved.ndim - 1)
        )
    return jnp.moveaxis(
        (moved[1:] - moved[:-1]) * inverse_distance,
        0,
        axis,
    )


def _pad_with_boundary_ghosts(
    array, axis, metallic_edges, *, logical_size: int | None = None
):
    logical_size = int(array.shape[axis]) if logical_size is None else int(logical_size)
    if logical_size <= 0 or logical_size > int(array.shape[axis]):
        raise ValueError(
            f"Logical boundary size {logical_size} is invalid for shape {array.shape}."
        )
    physical_region = [slice(None)] * array.ndim
    physical_region[axis] = slice(0, logical_size)
    physical = array[tuple(physical_region)]
    storage_region = [slice(None)] * array.ndim
    storage_region[axis] = slice(logical_size, None)
    storage_padding = array[tuple(storage_region)]
    shape = list(physical.shape)
    shape[axis] = 1
    zero = jnp.zeros(tuple(shape), dtype=array.dtype)
    low_edge, high_edge = (("front", "back"), ("bottom", "top"), ("left", "right"))[
        axis
    ]
    low = (
        zero if low_edge in metallic_edges else jnp.take(physical, jnp.array([0]), axis)
    )
    high = (
        zero
        if high_edge in metallic_edges
        else jnp.take(physical, jnp.array([logical_size - 1]), axis)
    )
    return jnp.concatenate((low, physical, high, storage_padding), axis=axis)


def build_h_boundary_views_for_e_3d(
    hx, hy, hz, metallic_edges=frozenset(), *, logical_shapes=None
):
    """Create the six ghost-padded H views consumed by the 3D E curl."""
    return {
        name: _pad_with_boundary_ghosts(
            field,
            axis,
            metallic_edges,
            logical_size=(
                None if logical_shapes is None else logical_shapes[component][axis]
            ),
        )
        for name, component, field, axis in (
            ("hz_y", "Hz", hz, 1),
            ("hy_z", "Hy", hy, 0),
            ("hx_z", "Hx", hx, 0),
            ("hz_x", "Hz", hz, 2),
            ("hy_x", "Hy", hy, 2),
            ("hx_y", "Hx", hx, 1),
        )
    }


def curl_h_to_e_3d(
    hx,
    hy,
    hz,
    resolution,
    ex_shape=None,
    ey_shape=None,
    ez_shape=None,
    *,
    boundary_views,
):
    """Differentiate H onto E sites using explicit boundary ghost views."""
    if ex_shape is None:
        ex_shape = (hz.shape[0], hz.shape[1] + 1, hz.shape[2])
    if ey_shape is None:
        ey_shape = (hx.shape[0] + 1, hx.shape[1], hx.shape[2])
    if ez_shape is None:
        ez_shape = (hy.shape[0], hy.shape[1], hy.shape[2] + 1)
    curls = (
        adjacent_difference(boundary_views["hz_y"], 1, resolution)
        - adjacent_difference(boundary_views["hy_z"], 0, resolution),
        adjacent_difference(boundary_views["hx_z"], 0, resolution)
        - adjacent_difference(boundary_views["hz_x"], 2, resolution),
        adjacent_difference(boundary_views["hy_x"], 2, resolution)
        - adjacent_difference(boundary_views["hx_y"], 1, resolution),
    )
    expected = (ex_shape, ey_shape, ez_shape)
    if any(curl.shape != shape for curl, shape in zip(curls, expected, strict=True)):
        raise ValueError(
            f"curl(H) shapes {tuple(curl.shape for curl in curls)} do not match {expected}"
        )
    return curls


def curl_h_to_e_3d_metric(
    hx,
    hy,
    hz,
    grid,
    ex_shape=None,
    ey_shape=None,
    ez_shape=None,
    *,
    boundary_views,
):
    """Differentiate magnetic fields using exact dual-grid distances."""
    if ex_shape is None:
        ex_shape = (hz.shape[0], hz.shape[1] + 1, hz.shape[2])
    if ey_shape is None:
        ey_shape = (hx.shape[0] + 1, hx.shape[1], hx.shape[2])
    if ez_shape is None:
        ez_shape = (hy.shape[0], hy.shape[1], hy.shape[2] + 1)
    ix = _rectilinear_inverse_distances(grid, "x", backward=True)
    iy = _rectilinear_inverse_distances(grid, "y", backward=True)
    iz = _rectilinear_inverse_distances(grid, "z", backward=True)
    curls = (
        metric_adjacent_difference(boundary_views["hz_y"], 1, iy)
        - metric_adjacent_difference(boundary_views["hy_z"], 0, iz),
        metric_adjacent_difference(boundary_views["hx_z"], 0, iz)
        - metric_adjacent_difference(boundary_views["hz_x"], 2, ix),
        metric_adjacent_difference(boundary_views["hy_x"], 2, ix)
        - metric_adjacent_difference(boundary_views["hx_y"], 1, iy),
    )
    expected = (ex_shape, ey_shape, ez_shape)
    if any(curl.shape != shape for curl, shape in zip(curls, expected, strict=True)):
        raise ValueError(
            f"curl(H) shapes {tuple(curl.shape for curl in curls)} do not match {expected}"
        )
    return curls


def advance_h_field(field, curl, sigma_m, dt):
    """Advance one magnetic Yee component through a source-free half-step."""
    denominator = 1.0 + sigma_m * dt / (2.0 * MU_0)
    return (
        field * (1.0 - sigma_m * dt / (2.0 * MU_0)) / denominator
        - (dt / MU_0) / denominator * curl
    )


def advance_e_field(field, curl, conductivity, permittivity, dt, region):
    """Advance one electric Yee component through a source-free half-step."""
    denominator = 1.0 + conductivity * dt / (2.0 * EPS_0 * permittivity)
    updated = (
        field[region]
        * (1.0 - conductivity * dt / (2.0 * EPS_0 * permittivity))
        / denominator
    )
    updated += (dt / (EPS_0 * permittivity)) / denominator * curl[region]
    return field.at[region].set(updated)


def component_index_arrays_3d(
    component: str,
    grid_shape: tuple[int, ...],
    *,
    stored_shape: tuple[int, ...] | None = None,
    region: tuple[slice, slice, slice] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return voxel indices sampled by a 3D Yee component on a cell-centered raster."""

    # Stored-shape overrides support padded/cropped views while grid_shape still bounds voxels.
    shape = (
        tuple(int(v) for v in stored_shape)
        if stored_shape is not None
        else component_shape_3d(component, grid_shape)
    )
    offsets = component_axis_offsets_3d(component)
    axes = ("z", "y", "x")
    region = region or (slice(None), slice(None), slice(None))

    indices: list[np.ndarray] = []
    for axis, dim, grid_dim, axis_region in zip(
        axes, shape, grid_shape, region, strict=False
    ):
        coord = np.arange(dim, dtype=np.float64) + offsets[axis]
        idx = np.floor(coord).astype(np.int32)
        idx = np.clip(idx, 0, int(grid_dim) - 1)
        indices.append(idx[axis_region])

    return (indices[0], indices[1], indices[2])


def sample_voxel_grid_at_component_3d(
    grid,
    component: str,
    *,
    stored_shape: tuple[int, ...] | None = None,
    region: tuple[slice, slice, slice] | None = None,
):
    """Sample a cell-centered 3D raster on a Yee component lattice."""

    # Electric interfaces need symmetric averaging; magnetic ownership uses one voxel.
    if component in {"Ex", "Ey", "Ez"}:
        return sample_voxel_grid_at_e_component_3d_centered(
            grid,
            component,
            stored_shape=stored_shape,
            region=region,
        )

    z_idx, y_idx, x_idx = component_index_arrays_3d(
        component,
        tuple(int(v) for v in np.asarray(grid).shape),
        stored_shape=stored_shape,
        region=region,
    )
    sampled = jnp.asarray(grid)
    sampled = jnp.take(sampled, jnp.asarray(z_idx), axis=0)
    sampled = jnp.take(sampled, jnp.asarray(y_idx), axis=1)
    sampled = jnp.take(sampled, jnp.asarray(x_idx), axis=2)
    return sampled


def sample_voxel_grid_at_e_component_3d_centered(
    grid,
    component: str,
    *,
    stored_shape: tuple[int, ...] | None = None,
    region: tuple[slice, slice, slice] | None = None,
):
    """Sample a cell-centered 3D raster onto a staggered E site by symmetric averaging."""

    # 1. Reject H components because magnetic material sampling intentionally follows a
    # different ownership convention from centered electric interpolation.
    if component not in {"Ex", "Ey", "Ez"}:
        raise ValueError(
            "sample_voxel_grid_at_e_component_3d_centered only supports Ex/Ey/Ez"
        )

    # 2. Resolve the requested stored support, Yee offsets, and optional update region before sampling any axis.
    grid_shape = tuple(int(v) for v in np.asarray(grid).shape)
    shape = (
        tuple(int(v) for v in stored_shape)
        if stored_shape is not None
        else component_shape_3d(component, grid_shape)
    )
    offsets = component_axis_offsets_3d(component)
    axes = ("z", "y", "x")
    region = region or (slice(None), slice(None), slice(None))

    sampled = jnp.asarray(grid)

    # 3. Walk axes in storage order, averaging bracketing cell centers only where the E
    # site lies between them and using direct ownership on center-aligned axes.
    for axis_index, (axis, dim, grid_dim, axis_region) in enumerate(
        zip(axes, shape, grid_shape, region, strict=False)
    ):
        coord = np.arange(dim, dtype=np.float64) + offsets[axis]
        if offsets[axis] == 0.0:
            hi = np.floor(coord).astype(np.int32)
            lo = hi - 1
            lo = np.clip(lo, 0, int(grid_dim) - 1)
            hi = np.clip(hi, 0, int(grid_dim) - 1)
            lo = lo[axis_region]
            hi = hi[axis_region]
            sampled_lo = jnp.take(sampled, jnp.asarray(lo), axis=axis_index)
            sampled_hi = jnp.take(sampled, jnp.asarray(hi), axis=axis_index)
            sampled = 0.5 * (sampled_lo + sampled_hi)
        elif offsets[axis] == 0.5:
            idx = np.floor(coord).astype(np.int32)
            idx = np.clip(idx, 0, int(grid_dim) - 1)
            idx = idx[axis_region]
            sampled = jnp.take(sampled, jnp.asarray(idx), axis=axis_index)
        else:
            raise ValueError(f"Unsupported Yee offset {offsets[axis]!r}")

    # 4. The sequential takes preserve the requested component-region shape while retaining the source array's JAX dtype and placement.
    return sampled


def component_shape_2d(
    component: str,
    grid_shape: tuple[int, ...],
    plane: str = "xy",
    polarization: str = "tm",
) -> tuple[int, int]:
    """Return a public component shape on the selected canonical 2D lattice."""
    if len(grid_shape) != 2:
        raise ValueError(f"A 2D grid shape must have two dimensions: {grid_shape!r}")
    ny, nx = (int(v) for v in grid_shape)
    canonical = canonical_component_2d(component, plane, polarization)
    if canonical is None:
        return (1, 1)
    return {
        "tm": {
            "Ez": (ny + 1, nx + 1),
            "Hx": (ny, nx + 1),
            "Hy": (ny + 1, nx),
        },
        "te": {
            "Ex": (ny + 1, nx),
            "Ey": (ny, nx + 1),
            "Hz": (ny, nx),
        },
    }[normalize_polarization_2d(polarization)][canonical]


def component_shapes(
    grid_shape: tuple[int, ...], polarization: str = "tm"
) -> dict[str, tuple[int, ...]]:
    """Return every canonical field support for a material-grid shape."""
    if len(grid_shape) == 3:
        return {
            component: component_shape_3d(component, grid_shape)
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
    return {
        component: component_shape_2d(component, grid_shape, "xy", polarization)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }


def normalize_polarization_2d(value: str) -> Literal["tm", "te"]:
    """Normalize a 2D electromagnetic polarization name."""

    polarization = str(value).strip().lower()
    if polarization not in {"tm", "te"}:
        raise ValueError("polarization must be 'tm' or 'te'.")
    return "tm" if polarization == "tm" else "te"


def canonical_component_2d(
    component: str, plane: str, polarization: str = "tm"
) -> str | None:
    """Map a public plane component to the selected canonical xy lattice."""
    try:
        mappings = {
            "tm": {
                "xy": {"Ez": "Ez", "Hx": "Hx", "Hy": "Hy"},
                "yz": {"Ex": "Ez", "Hy": "Hx", "Hz": "Hy"},
                "xz": {"Ey": "Ez", "Hx": "Hx", "Hz": "Hy"},
            },
            "te": {
                "xy": {"Ex": "Ex", "Ey": "Ey", "Hz": "Hz"},
                "yz": {"Ey": "Ex", "Ez": "Ey", "Hx": "Hz"},
                "xz": {"Ex": "Ex", "Ez": "Ey", "Hy": "Hz"},
            },
        }
        return mappings[normalize_polarization_2d(polarization)][
            str(plane).lower()
        ].get(component)
    except KeyError as exc:
        raise ValueError(f"Unsupported plane {plane!r}") from exc


def public_component_2d(
    component: str, plane: str, polarization: str = "tm"
) -> tuple[str, float]:
    """Map a canonical 2D component back to its public label and orientation."""
    try:
        mappings = {
            "tm": {
                "xy": {"Ez": ("Ez", 1.0), "Hx": ("Hx", 1.0), "Hy": ("Hy", 1.0)},
                "yz": {"Ez": ("Ex", 1.0), "Hx": ("Hy", 1.0), "Hy": ("Hz", 1.0)},
                "xz": {"Ez": ("Ey", -1.0), "Hx": ("Hx", 1.0), "Hy": ("Hz", 1.0)},
            },
            "te": {
                "xy": {"Ex": ("Ex", 1.0), "Ey": ("Ey", 1.0), "Hz": ("Hz", 1.0)},
                "yz": {"Ex": ("Ey", 1.0), "Ey": ("Ez", 1.0), "Hz": ("Hx", 1.0)},
                "xz": {"Ex": ("Ex", 1.0), "Ey": ("Ez", 1.0), "Hz": ("Hy", -1.0)},
            },
        }
        return mappings[normalize_polarization_2d(polarization)][str(plane).lower()][
            component
        ]
    except KeyError as exc:
        raise ValueError(
            f"Component {component!r} is not active in canonical 2D plane {plane!r}"
        ) from exc


def component_coordinates_2d_um(
    component: str,
    grid_shape: tuple[int, ...],
    dx_um: float,
    plane: str,
    polarization: str = "tm",
) -> dict[str, np.ndarray]:
    """Return public-axis coordinates for a canonical 2D Yee component."""
    shape = component_shape_2d(component, grid_shape, plane, polarization)
    canonical = canonical_component_2d(component, plane, polarization)
    row_axis, col_axis = {"xy": ("y", "x"), "yz": ("z", "y"), "xz": ("z", "x")}[plane]
    offsets = (
        (0.0, 0.0)
        if canonical is None
        else {
            "Ez": (0.0, 0.0),
            "Hx": (0.5, 0.0),
            "Hy": (0.0, 0.5),
            "Ex": (0.0, 0.5),
            "Ey": (0.5, 0.0),
            "Hz": (0.5, 0.5),
        }[canonical]
    )
    return {
        row_axis: (np.arange(shape[0], dtype=np.float64) + offsets[0]) * dx_um,
        col_axis: (np.arange(shape[1], dtype=np.float64) + offsets[1]) * dx_um,
    }


def sample_voxel_grid_at_component_2d(
    grid,
    component: str,
    plane: str,
    polarization: str = "tm",
    *,
    stored_shape: tuple[int, ...] | None = None,
    region: tuple[slice, slice] | None = None,
):
    """Sample a material raster on the selected canonical 2D Yee support."""
    grid_np = np.asarray(grid)
    canonical = canonical_component_2d(component, plane, polarization)
    if canonical is None:
        return jnp.asarray(grid_np[:1, :1])
    shape = component_shape_2d(
        component,
        tuple(int(v) for v in grid_np.shape),
        plane,
        polarization,
    )
    if stored_shape is not None and tuple(int(v) for v in stored_shape) != shape:
        raise ValueError(f"stored_shape={stored_shape!r} does not match {shape!r}")
    region = region or (slice(None), slice(None))
    ny, nx = (int(v) for v in grid_np.shape)
    y_len, x_len = shape
    y_offset, x_offset = {
        "Ez": (0.0, 0.0),
        "Hx": (0.5, 0.0),
        "Hy": (0.0, 0.5),
        "Ex": (0.0, 0.5),
        "Ey": (0.5, 0.0),
        "Hz": (0.5, 0.5),
    }[canonical]
    y_idx = np.clip(np.floor(np.arange(y_len) + y_offset).astype(np.int32), 0, ny - 1)
    x_idx = np.clip(np.floor(np.arange(x_len) + x_offset).astype(np.int32), 0, nx - 1)
    sampled = jnp.asarray(grid)
    sampled = jnp.take(sampled, jnp.asarray(y_idx), axis=0)
    sampled = jnp.take(sampled, jnp.asarray(x_idx), axis=1)
    return sampled[region]


class MaterialCoefficients:
    """Read-only attribute view of material arrays on Yee component supports."""

    __slots__ = ("_values",)

    def __init__(self, **values):
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def items(self):
        return self._values.items()


def _total_conductivity(fields):
    # Sponge PML contributes physical conductivity; CPML damping lives in psi state.
    base_sigma = fields.conductivity
    if not (fields.has_pml and not fields.has_cpml and hasattr(fields, "pml_data")):
        return base_sigma
    keys = (
        ("sigma_x", "sigma_y", "sigma_z")
        if fields.permittivity.ndim == 3
        else ("sigma_x", "sigma_y")
    )
    sigma_pml = jnp.zeros_like(base_sigma)
    for key in keys:
        if key in fields.pml_data:
            sigma_pml = sigma_pml + fields.pml_data[key]
    material_grid = getattr(fields, "material_grid", None)
    if (
        getattr(material_grid, "metric_kind", "isotropic_uniform")
        != "isotropic_uniform"
    ):
        return base_sigma + sigma_pml
    return jnp.maximum(base_sigma, sigma_pml)


def _mu_3d(fields):
    if fields.permeability.ndim == 0:
        return (fields.permeability,) * 3
    return tuple(
        sample_voxel_grid_at_component_3d(
            fields.permeability,
            component,
            stored_shape=tuple(getattr(fields, component).shape),  # fmt: skip
        )
        for component in ("Hx", "Hy", "Hz")
    )


def _base_coefficients(total_sigma, e_terms, sigma_m, mu_terms):
    data = {"total_conductivity": total_sigma}
    for axis, (eps, sig, region) in zip("xyz", e_terms, strict=True):
        data[f"eps_{axis}"] = eps
        data[f"sig_{axis}"] = sig
        data[f"region_{axis}"] = region
    data.update(
        eps_ex=data["eps_x"],
        eps_ey=data["eps_y"],
        eps_ez=data["eps_z"],
        sigma_m_hx=sigma_m[0],
        sigma_m_hy=sigma_m[1],
        sigma_m_hz=sigma_m[2],
        mu_hx=mu_terms[0],
        mu_hy=mu_terms[1],
        mu_hz=mu_terms[2],
    )
    return data


def _material_slice_for_e_3d(permittivity, conductivity, orientation):
    """Collocate material values with one staggered 3D electric component."""
    component = {"x": "Ex", "y": "Ey", "z": "Ez"}[orientation]
    region = (slice(None), slice(None), slice(None))
    eps = sample_voxel_grid_at_e_component_3d_centered(permittivity, component)
    sig = (
        jnp.asarray(conductivity)
        if jnp.asarray(conductivity).ndim == 0
        else sample_voxel_grid_at_e_component_3d_centered(conductivity, component)
    )
    return eps, sig, region


def _magnetic_conductivity_terms_3d(
    conductivity, permeability, hx_shape, hy_shape, hz_shape
):
    """Collocate equivalent magnetic conductivity with the three H supports."""
    conductivity = jnp.asarray(conductivity)
    if conductivity.ndim == 0 and float(conductivity) == 0.0:
        zero = jnp.zeros((), dtype=conductivity.dtype)
        return zero, zero, zero
    sigma_base = conductivity * permeability * MU_0 / EPS_0
    return tuple(
        sample_voxel_grid_at_component_3d(
            sigma_base,
            component,
            stored_shape=shape,
        )
        for component, shape in zip(
            ("Hx", "Hy", "Hz"),
            (hx_shape, hy_shape, hz_shape),
            strict=True,
        )
    )


def build_material_coefficients(fields):
    """Collocate every material array once before runtime execution."""

    total_sigma = _total_conductivity(fields)
    if fields.permittivity.ndim == 3:
        e_terms = tuple(
            _material_slice_for_e_3d(fields.permittivity, total_sigma, axis)
            for axis in "xyz"
        )
        sigma_m = _magnetic_conductivity_terms_3d(
            total_sigma,
            fields.permeability,
            fields.Hx.shape,
            fields.Hy.shape,
            fields.Hz.shape,  # fmt: skip
        )
        return MaterialCoefficients(
            **_base_coefficients(total_sigma, e_terms, sigma_m, _mu_3d(fields))
        )

    # The runtime schema always contains six components; only one polarization's
    # three supports carry full arrays in a 2D program.
    polarization = normalize_polarization_2d(getattr(fields, "polarization_2d", "tm"))
    inactive_eps = jnp.asarray(fields.permittivity)[:1, :1]
    inactive_sig = (
        total_sigma
        if jnp.asarray(total_sigma).ndim == 0
        else jnp.asarray(total_sigma)[:1, :1]
    )
    region = (slice(None), slice(None))
    active_e = ("Ez",) if polarization == "tm" else ("Ex", "Ey")
    e_terms = tuple(
        (
            sample_voxel_grid_at_component_2d(
                fields.permittivity, component, "xy", polarization
            ),
            (
                total_sigma
                if jnp.asarray(total_sigma).ndim == 0
                else sample_voxel_grid_at_component_2d(
                    total_sigma, component, "xy", polarization
                )
            ),
            region,
        )
        if component in active_e
        else (inactive_eps, inactive_sig, region)
        for component in ("Ex", "Ey", "Ez")
    )
    if jnp.asarray(total_sigma).ndim == 0 and float(jnp.asarray(total_sigma)) == 0.0:
        sigma_m = (total_sigma,) * 3
    else:
        base = (
            jnp.asarray(total_sigma) * jnp.asarray(fields.permeability) * MU_0 / EPS_0
        )
        active_h = ("Hx", "Hy") if polarization == "tm" else ("Hz",)
        sigma_m = tuple(
            sample_voxel_grid_at_component_2d(base, component, "xy", polarization)
            if component in active_h
            else jnp.asarray(base)[:1, :1]
            for component in ("Hx", "Hy", "Hz")
        )
    mu_terms = (
        (fields.permeability,) * 3
        if fields.permeability.ndim == 0
        else tuple(
            sample_voxel_grid_at_component_2d(
                fields.permeability, component, "xy", polarization
            )
            if component in (("Hx", "Hy") if polarization == "tm" else ("Hz",))
            else jnp.asarray(fields.permeability)[:1, :1]
            for component in ("Hx", "Hy", "Hz")
        )
    )
    return MaterialCoefficients(
        **_base_coefficients(total_sigma, e_terms, sigma_m, mu_terms)
    )


def attach_material_coefficients(fields, materials) -> None:
    fields.materials = materials
    for name, value in materials.items():
        setattr(fields, name, value)


def material_for_component(materials, component: str):
    mapping = {
        "Ex": "eps_ex",
        "Ey": "eps_ey",
        "Ez": "eps_ez",
        "Hx": "mu_hx",
        "Hy": "mu_hy",
        "Hz": "mu_hz",
    }
    try:
        return getattr(materials, mapping[component])
    except KeyError as exc:
        raise ValueError(f"Unsupported field component {component!r}") from exc


def component_material_at(fields, component: str, index):
    materials = getattr(fields, "materials", fields)
    material = material_for_component(materials, component)
    return material if jnp.asarray(material).ndim == 0 else material[index]
