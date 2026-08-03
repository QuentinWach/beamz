from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class SnappedInterval:
    start: int
    stop: int
    step: float
    edges: tuple[float, ...] | None = None

    @property
    def lower(self) -> float:
        if self.edges is not None:
            return float(self.edges[int(self.start)])
        return float(self.start) * float(self.step)

    @property
    def upper(self) -> float:
        if self.edges is not None:
            return float(self.edges[int(self.stop)])
        return float(self.stop) * float(self.step)

    @property
    def center(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def size(self) -> float:
        return max(0, int(self.stop) - int(self.start)) * float(self.step)

    def as_slice(self) -> slice:
        return slice(int(self.start), int(self.stop))


@dataclass(frozen=True)
class SnappedRegion:
    ndim: int
    normal_axis: str
    plane_index: int
    plane_coord: float
    intervals: dict[str, SnappedInterval] = field(default_factory=dict)
    companion_index: int | None = None
    companion_coord: float | None = None

    def axis_interval(self, axis: str) -> SnappedInterval | None:
        return self.intervals.get(str(axis).lower())

    def axis_coord(self, axis: str) -> float:
        axis = str(axis).lower()
        if axis == self.normal_axis:
            return float(self.plane_coord)
        interval = self.axis_interval(axis)
        if interval is None:
            return 0.0
        return float(interval.center)

    def axis_bounds(self, axis: str) -> tuple[float, float]:
        axis = str(axis).lower()
        if axis == self.normal_axis:
            coord = float(self.plane_coord)
            return coord, coord
        interval = self.axis_interval(axis)
        if interval is None:
            return 0.0, 0.0
        return float(interval.lower), float(interval.upper)

    @property
    def start(self) -> tuple[float, ...]:
        return tuple(self.axis_bounds(axis)[0] for axis in _AXES[: self.ndim])

    @property
    def end(self) -> tuple[float, ...]:
        return tuple(self.axis_bounds(axis)[1] for axis in _AXES[: self.ndim])

    @property
    def center(self) -> tuple[float, ...]:
        return tuple(self.axis_coord(axis) for axis in _AXES[: self.ndim])

    @property
    def size(self) -> tuple[float, ...]:
        return tuple(
            self.axis_bounds(axis)[1] - self.axis_bounds(axis)[0]
            for axis in _AXES[: self.ndim]
        )


def snap_cell_center(coord: float, step: float, count: int) -> tuple[int, float]:
    if count <= 0:
        return 0, 0.0
    idx = int(np_round_half_even((float(coord) / float(step)) - 0.5))
    idx = max(0, min(idx, count - 1))
    return idx, (float(idx) + 0.5) * float(step)


def snap_edge_interval(
    lower: float,
    upper: float,
    step: float,
    count: int,
    *,
    min_cells: int = 1,
) -> SnappedInterval:
    if count <= 0:
        return SnappedInterval(0, 0, float(step))

    lo = min(float(lower), float(upper))
    hi = max(float(lower), float(upper))
    eps = 1e-12 * max(abs(lo), abs(hi), abs(float(step)), 1.0)

    start = int(math.floor((lo + eps) / float(step)))
    stop = int(math.ceil((hi - eps) / float(step)))

    start = max(0, min(start, count))
    stop = max(0, min(stop, count))

    need = max(1, int(min_cells))
    if stop - start < need:
        center = 0.5 * (lo + hi)
        cell = int(math.floor(center / float(step)))
        cell = max(0, min(cell, count - 1))
        start = cell
        stop = min(count, cell + need)
        start = max(0, stop - need)

    return SnappedInterval(start=int(start), stop=int(stop), step=float(step))


def snap_rectilinear_cell_center(coord: float, edges) -> tuple[int, float]:
    """Snap a coordinate to the nearest physical cell center."""
    edge_array = np.asarray(edges, dtype=np.float64)
    centers = 0.5 * (edge_array[:-1] + edge_array[1:])
    if centers.size == 0:
        return 0, 0.0
    index = int(np.argmin(np.abs(centers - float(coord))))
    return index, float(centers[index])


def snap_rectilinear_edge_interval(
    lower: float,
    upper: float,
    edges,
    *,
    min_cells: int = 1,
) -> SnappedInterval:
    """Snap physical bounds to the overlapping rectilinear cells."""
    edge_array = np.asarray(edges, dtype=np.float64)
    count = max(0, edge_array.size - 1)
    if count == 0:
        return SnappedInterval(0, 0, 0.0, tuple(edge_array))
    lo, hi = sorted((float(lower), float(upper)))
    start = int(np.searchsorted(edge_array, lo, side="right") - 1)
    stop = int(np.searchsorted(edge_array, hi, side="left"))
    start = max(0, min(start, count))
    stop = max(0, min(stop, count))
    need = max(1, int(min_cells))
    if stop - start < need:
        cell, _ = snap_rectilinear_cell_center(0.5 * (lo + hi), edge_array)
        start = max(0, min(cell, count - need))
        stop = min(count, start + need)
    representative = float(np.min(np.diff(edge_array)))
    return SnappedInterval(
        int(start),
        int(stop),
        representative,
        tuple(float(value) for value in edge_array),
    )


def snap_axis_aligned_line_region_grid(
    start: tuple[float, ...],
    end: tuple[float, ...],
    grid,
) -> SnappedRegion | None:
    """Snap a 2D line using realized x/y cell edges."""
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    tolerance = 1e-12 * max(*map(abs, (x0, x1, y0, y1)), grid.minimum_spacing, 1.0)
    if abs(x0 - x1) <= tolerance:
        index, coordinate = snap_rectilinear_cell_center(0.5 * (x0 + x1), grid.x_edges)
        return SnappedRegion(
            2,
            "x",
            index,
            coordinate,
            {"y": snap_rectilinear_edge_interval(y0, y1, grid.y_edges)},
        )
    if abs(y0 - y1) <= tolerance:
        index, coordinate = snap_rectilinear_cell_center(0.5 * (y0 + y1), grid.y_edges)
        return SnappedRegion(
            2,
            "y",
            index,
            coordinate,
            {"x": snap_rectilinear_edge_interval(x0, x1, grid.x_edges)},
        )
    return None


def snap_plane_region_grid(*, center, size, plane_normal: str, grid) -> SnappedRegion:
    """Snap a center/size plane using realized rectilinear edges."""
    center = tuple(float(value) for value in center)
    size = tuple(float(value) for value in size)
    axis = str(plane_normal).lower()
    edges = {"x": grid.x_edges, "y": grid.y_edges, "z": grid.z_edges}
    axis_index = {"x": 0, "y": 1, "z": 2}
    plane_index, plane_coord = snap_rectilinear_cell_center(
        center[axis_index[axis]], edges[axis]
    )
    intervals = {}
    for tangential in _AXES:
        if tangential == axis:
            continue
        index = axis_index[tangential]
        half = 0.5 * size[index]
        intervals[tangential] = snap_rectilinear_edge_interval(
            center[index] - half,
            center[index] + half,
            edges[tangential],
        )
    return SnappedRegion(3, axis, plane_index, plane_coord, intervals)


def snap_centered_extent(
    center: float,
    extent: float,
    step: float,
    count: int,
    *,
    min_cells: int = 1,
) -> SnappedInterval:
    half = 0.5 * float(extent)
    return snap_edge_interval(
        float(center) - half,
        float(center) + half,
        step,
        count,
        min_cells=min_cells,
    )


def snap_mode_source_region(
    *,
    center: tuple[float, ...],
    width: float,
    height: float | None,
    axis: str,
    direction_sign: float,
    grid_shape: tuple[int, ...],
    resolution: float,
    is_3d: bool,
) -> SnappedRegion:
    axis = str(axis).lower()
    counts = _counts_by_axis(grid_shape, is_3d)
    plane_index, plane_coord = snap_cell_center(
        center[_axis_pos(axis)], resolution, counts[axis]
    )

    companion_max = max(counts[axis] - 2, 0)
    if direction_sign > 0.0:
        companion_index = max(0, plane_index - 1)
    else:
        companion_index = min(companion_max, plane_index + 1)
    companion_coord = (float(companion_index) + 1.0) * float(resolution)

    intervals: dict[str, SnappedInterval] = {}
    if axis == "x":
        intervals["y"] = snap_centered_extent(center[1], width, resolution, counts["y"])
        if is_3d:
            z_center = (
                center[2] if len(center) > 2 else 0.5 * counts["z"] * float(resolution)
            )
            intervals["z"] = snap_centered_extent(
                z_center,
                float(height if height is not None else width),
                resolution,
                counts["z"],
            )
    elif axis == "y":
        intervals["x"] = snap_centered_extent(center[0], width, resolution, counts["x"])
        if is_3d:
            z_center = (
                center[2] if len(center) > 2 else 0.5 * counts["z"] * float(resolution)
            )
            intervals["z"] = snap_centered_extent(
                z_center,
                float(height if height is not None else width),
                resolution,
                counts["z"],
            )
    else:
        intervals["x"] = snap_centered_extent(center[0], width, resolution, counts["x"])
        intervals["y"] = snap_centered_extent(
            center[1],
            float(height if height is not None else width),
            resolution,
            counts["y"],
        )

    return SnappedRegion(
        ndim=3 if is_3d else 2,
        normal_axis=axis,
        plane_index=int(plane_index),
        plane_coord=float(plane_coord),
        intervals=intervals,
        companion_index=int(companion_index),
        companion_coord=float(companion_coord),
    )


def snap_axis_aligned_line_region(
    start: tuple[float, ...],
    end: tuple[float, ...],
    dx: float,
    dy: float,
    shape: tuple[int, ...],
) -> SnappedRegion | None:
    if len(start) < 2 or len(end) < 2 or len(shape) != 2:
        raise ValueError(
            "A 2D line region requires two coordinates and a rank-2 shape."
        )
    ny, nx = shape
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    tol = 1e-9 * max(abs(dx), abs(dy), 1.0)

    if abs(x0 - x1) <= tol:
        plane_index, plane_coord = snap_cell_center(0.5 * (x0 + x1), dx, nx)
        return SnappedRegion(
            ndim=2,
            normal_axis="x",
            plane_index=int(plane_index),
            plane_coord=float(plane_coord),
            intervals={"y": snap_edge_interval(y0, y1, dy, ny)},
        )
    if abs(y0 - y1) <= tol:
        plane_index, plane_coord = snap_cell_center(0.5 * (y0 + y1), dy, ny)
        return SnappedRegion(
            ndim=2,
            normal_axis="y",
            plane_index=int(plane_index),
            plane_coord=float(plane_coord),
            intervals={"x": snap_edge_interval(x0, x1, dx, nx)},
        )
    return None


def snap_plane_region(
    *,
    start: tuple[float, ...],
    end: tuple[float, ...] | None,
    plane_normal: str,
    size: tuple[float, ...] | None,
    dx: float,
    dy: float,
    dz: float,
    shape: tuple[int, ...],
) -> SnappedRegion:
    if len(start) != 3 or (end is not None and len(end) != 3) or len(shape) != 3:
        raise ValueError("A 3D plane region requires rank-3 coordinates and shape.")
    if size is not None and len(size) != 2:
        raise ValueError("A 3D plane size requires two tangential extents.")
    nz, ny, nx = shape
    counts = {"x": nx, "y": ny, "z": nz}
    steps = {"x": float(dx), "y": float(dy), "z": float(dz)}
    axis = str(plane_normal).lower()

    if end is None:
        x0, y0, z0 = (float(v) for v in start)
        span0, span1 = (
            (float(size[0]), float(size[1])) if size is not None else (0.0, 0.0)
        )
        if axis == "x":
            plane_pos = x0
            bounds = {
                "y": (y0, y0 + span0),
                "z": (z0, z0 + span1),
            }
        elif axis == "y":
            plane_pos = y0
            bounds = {
                "x": (x0, x0 + span0),
                "z": (z0, z0 + span1),
            }
        else:
            plane_pos = z0
            bounds = {
                "x": (x0, x0 + span0),
                "y": (y0, y0 + span1),
            }
    else:
        x0, y0, z0 = (float(v) for v in start)
        x1, y1, z1 = (float(v) for v in end)
        if axis == "x":
            plane_pos = 0.5 * (x0 + x1)
            bounds = {"y": (y0, y1), "z": (z0, z1)}
        elif axis == "y":
            plane_pos = 0.5 * (y0 + y1)
            bounds = {"x": (x0, x1), "z": (z0, z1)}
        else:
            plane_pos = 0.5 * (z0 + z1)
            bounds = {"x": (x0, x1), "y": (y0, y1)}

    plane_index, plane_coord = snap_cell_center(plane_pos, steps[axis], counts[axis])
    intervals = {
        name: snap_edge_interval(lo, hi, steps[name], counts[name])
        for name, (lo, hi) in bounds.items()
    }
    return SnappedRegion(
        ndim=3,
        normal_axis=axis,
        plane_index=int(plane_index),
        plane_coord=float(plane_coord),
        intervals=intervals,
    )


def line_region_points(region: SnappedRegion) -> list[tuple[int, int]]:
    if region.normal_axis == "x":
        interval = region.axis_interval("y")
        if interval is None:
            return []
        return [
            (int(region.plane_index), int(y_idx))
            for y_idx in range(int(interval.start), int(interval.stop))
        ]
    interval = region.axis_interval("x")
    if interval is None:
        return []
    return [
        (int(x_idx), int(region.plane_index))
        for x_idx in range(int(interval.start), int(interval.stop))
    ]


def plane_region_slices(
    region: SnappedRegion,
) -> tuple[int | slice, int | slice, int | slice]:
    if region.normal_axis == "x":
        y_interval = region.axis_interval("y")
        z_interval = region.axis_interval("z")
        return (
            z_interval.as_slice() if z_interval is not None else slice(0, 0),
            y_interval.as_slice() if y_interval is not None else slice(0, 0),
            int(region.plane_index),
        )
    if region.normal_axis == "y":
        x_interval = region.axis_interval("x")
        z_interval = region.axis_interval("z")
        return (
            z_interval.as_slice() if z_interval is not None else slice(0, 0),
            int(region.plane_index),
            x_interval.as_slice() if x_interval is not None else slice(0, 0),
        )
    x_interval = region.axis_interval("x")
    y_interval = region.axis_interval("y")
    return (
        int(region.plane_index),
        y_interval.as_slice() if y_interval is not None else slice(0, 0),
        x_interval.as_slice() if x_interval is not None else slice(0, 0),
    )


def np_round_half_even(value: float) -> int:
    return int(round(float(value)))


def _axis_pos(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[str(axis).lower()]


def _counts_by_axis(grid_shape: tuple[int, ...], is_3d: bool) -> dict[str, int]:
    if is_3d:
        nz, ny, nx = (int(v) for v in grid_shape)
        return {"x": nx, "y": ny, "z": nz}
    ny, nx = (int(v) for v in grid_shape)
    return {"x": nx, "y": ny, "z": 1}
