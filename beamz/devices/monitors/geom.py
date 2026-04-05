"""Geometry helpers for monitor setup and sampling."""

import numpy as np
from matplotlib.patches import Rectangle as MatplotlibRectangle

from beamz.devices.monitors.spec import MonitorSpec


def determine_3d_mode(start, end, design):
    """Determine if a monitor should operate in 3D mode."""
    if end is not None and len(end) == 3:
        return True
    if len(start) == 3:
        return True
    if end is not None and len(start) == 2 and len(end) == 2:
        return False
    if design and hasattr(design, "is_3d") and design.is_3d:
        return True
    return False


def init_2d_monitor(monitor, start, end):
    """Initialize 2D line monitor state."""
    if end is None:
        end = start
    monitor.start = start
    monitor.end = end
    monitor.position = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    monitor.monitor_type = "line"


def init_3d_monitor(monitor, start, end, plane_normal, plane_position, size):
    """Initialize 3D plane monitor state."""
    if len(start) == 2:
        start = (start[0], start[1], 0.0)
    monitor.start = start

    if end is not None:
        if len(end) == 2:
            end = (end[0], end[1], start[2])
        monitor.end = end

        if plane_normal is None:
            dx = abs(end[0] - start[0])
            dy = abs(end[1] - start[1])
            dz = abs(end[2] - start[2])
            min_dim_idx = np.argmin([dx, dy, dz])
            if min_dim_idx == 0:
                monitor.plane_normal = "x"
            elif min_dim_idx == 1:
                monitor.plane_normal = "y"
            else:
                monitor.plane_normal = "z"
        else:
            monitor.plane_normal = plane_normal

        if monitor.plane_normal == "x":
            monitor.size = (abs(end[1] - start[1]), abs(end[2] - start[2]))
            monitor.plane_position = start[0]
        elif monitor.plane_normal == "y":
            monitor.size = (abs(end[0] - start[0]), abs(end[2] - start[2]))
            monitor.plane_position = start[1]
        else:
            monitor.size = (abs(end[0] - start[0]), abs(end[1] - start[1]))
            monitor.plane_position = start[2]

        monitor.start = (
            min(start[0], end[0]),
            min(start[1], end[1]),
            min(start[2], end[2]),
        )
    else:
        monitor.end = None
        monitor.plane_normal = plane_normal or "z"
        if plane_position == 0 and start is not None and len(start) >= 3:
            if monitor.plane_normal == "z":
                monitor.plane_position = start[2]
            elif monitor.plane_normal == "y":
                monitor.plane_position = start[1]
            elif monitor.plane_normal == "x":
                monitor.plane_position = start[0]
            else:
                monitor.plane_position = plane_position
        else:
            monitor.plane_position = plane_position

        if size is None:
            if monitor.design:
                if monitor.plane_normal == "z":
                    size = (monitor.design.width, monitor.design.height)
                elif monitor.plane_normal == "y":
                    size = (
                        monitor.design.width,
                        monitor.design.depth or monitor.design.width,
                    )
                else:
                    size = (
                        monitor.design.height,
                        monitor.design.depth or monitor.design.height,
                    )
            else:
                size = (1e-6, 1e-6)
        monitor.size = size

    monitor.monitor_type = "plane"
    monitor.position = get_plane_center(monitor)
    monitor.vertices = generate_plane_vertices(monitor)


def generate_plane_vertices(monitor):
    """Generate vertices for the monitor plane for 3D visualization."""
    if monitor.plane_normal == "z" or (
        hasattr(monitor, "end") and monitor.end is not None
    ):
        x_min, y_min = monitor.start[0], monitor.start[1]
        x_max = x_min + monitor.size[0]
        y_max = y_min + monitor.size[1]
        z = monitor.plane_position
        return [
            (x_min, y_min, z),
            (x_max, y_min, z),
            (x_max, y_max, z),
            (x_min, y_max, z),
        ]

    if monitor.plane_normal == "y":
        x_min, z_min = monitor.start[0], monitor.start[2]
        x_max = x_min + monitor.size[0]
        z_max = z_min + monitor.size[1]
        y = monitor.plane_position
        return [
            (x_min, y, z_min),
            (x_max, y, z_min),
            (x_max, y, z_max),
            (x_min, y, z_max),
        ]

    y_min, z_min = monitor.start[1], monitor.start[2]
    y_max = y_min + monitor.size[0]
    z_max = z_min + monitor.size[1]
    x = monitor.plane_position
    return [
        (x, y_min, z_min),
        (x, y_max, z_min),
        (x, y_max, z_max),
        (x, y_min, z_max),
    ]


def get_plane_center(monitor):
    """Get center position of a 3D plane monitor."""
    if monitor.plane_normal == "z" or (
        hasattr(monitor, "end") and monitor.end is not None
    ):
        return (
            monitor.start[0] + monitor.size[0] / 2,
            monitor.start[1] + monitor.size[1] / 2,
            monitor.plane_position,
        )
    if monitor.plane_normal == "y":
        return (
            monitor.start[0] + monitor.size[0] / 2,
            monitor.plane_position,
            monitor.start[2] + monitor.size[1] / 2,
        )
    return (
        monitor.plane_position,
        monitor.start[1] + monitor.size[0] / 2,
        monitor.start[2] + monitor.size[1] / 2,
    )


def grid_points_2d(monitor, dx, dy):
    """Get grid points for a 2D line monitor."""
    return grid_points_2d_for_spec(monitor.spec, dx, dy)


def grid_points_2d_for_spec(spec, dx, dy):
    """Get grid points for a 2D line monitor spec."""
    if not isinstance(spec, MonitorSpec):
        raise TypeError("grid_points_2d_for_spec expects a MonitorSpec")
    start_x_grid = int(round(spec.start[0] / dx))
    start_y_grid = int(round(spec.start[1] / dy))
    end_x_grid = int(round(spec.end[0] / dx))
    end_y_grid = int(round(spec.end[1] / dy))

    if abs(end_x_grid - start_x_grid) > abs(end_y_grid - start_y_grid):
        num_points = abs(end_x_grid - start_x_grid) + 1
        x_indices = np.linspace(start_x_grid, end_x_grid, num_points, dtype=int)
        y_indices = np.linspace(start_y_grid, end_y_grid, num_points, dtype=int)
    else:
        num_points = abs(end_y_grid - start_y_grid) + 1
        x_indices = np.linspace(start_x_grid, end_x_grid, num_points, dtype=int)
        y_indices = np.linspace(start_y_grid, end_y_grid, num_points, dtype=int)

    return list(zip(x_indices, y_indices))


def grid_slice_3d(monitor, dx, dy, dz, field_shape):
    """Get grid slice for a 3D plane monitor."""
    return grid_slice_3d_for_spec(monitor.spec, dx, dy, dz, field_shape)


def grid_slice_3d_for_spec(spec, dx, dy, dz, field_shape):
    """Get grid slice for a 3D plane monitor spec."""
    if not isinstance(spec, MonitorSpec):
        raise TypeError("grid_slice_3d_for_spec expects a MonitorSpec")
    base_nz, base_ny, base_nx = field_shape
    del base_nx, base_ny, base_nz

    if spec.plane_normal == "z":
        z_idx = int(round(spec.plane_position / dz))
        x_start = int(round(spec.start[0] / dx))
        x_end = int(round((spec.start[0] + spec.size[0]) / dx))
        y_start = int(round(spec.start[1] / dy))
        y_end = int(round((spec.start[1] + spec.size[1]) / dy))
        return z_idx, slice(y_start, y_end), slice(x_start, x_end)
    if spec.plane_normal == "y":
        y_idx = int(round(spec.plane_position / dy))
        x_start = int(round(spec.start[0] / dx))
        x_end = int(round((spec.start[0] + spec.size[0]) / dx))
        z_start = int(round(spec.start[2] / dz))
        z_end = int(round((spec.start[2] + spec.size[1]) / dz))
        return slice(z_start, z_end), y_idx, slice(x_start, x_end)

    x_idx = int(round(spec.plane_position / dx))
    y_start = int(round(spec.start[1] / dy))
    y_end = int(round((spec.start[1] + spec.size[0]) / dy))
    z_start = int(round(spec.start[2] / dz))
    z_end = int(round((spec.start[2] + spec.size[1]) / dz))
    return slice(z_start, z_end), slice(y_start, y_end), x_idx


def add_to_plot(
    monitor, ax, facecolor="none", edgecolor="navy", alpha=1, linestyle="-"
):
    """Add monitor visualization to a 2D plot."""
    if monitor.monitor_type == "line":
        color = edgecolor if facecolor == "none" else facecolor
        ax.plot(
            (monitor.start[0], monitor.end[0]),
            (monitor.start[1], monitor.end[1]),
            lw=4,
            color=color,
            label="Monitor",
            alpha=alpha,
        )
        ax.plot(
            (monitor.start[0], monitor.end[0]),
            (monitor.start[1], monitor.end[1]),
            lw=1,
            color=edgecolor,
            linestyle=linestyle,
        )
        return

    if monitor.plane_normal == "z" or (
        hasattr(monitor, "end") and monitor.end is not None
    ):
        rect = MatplotlibRectangle(
            (monitor.start[0], monitor.start[1]),
            monitor.size[0],
            monitor.size[1],
            fill=(facecolor != "none"),
            facecolor=facecolor,
            alpha=alpha * 0.3,
            edgecolor=edgecolor,
            linestyle=linestyle,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            monitor.position[0],
            monitor.position[1],
            "Monitor\n(3D plane)",
            ha="center",
            va="center",
            fontsize=8,
            color=edgecolor,
        )


def to_polygon(monitor):
    """Convert monitor to a polygon for 3D visualization."""
    if not hasattr(monitor, "vertices") or not monitor.vertices:
        return None

    from beamz.design.materials import Material
    from beamz.design.structures import Polygon

    polygon = Polygon(
        vertices=monitor.vertices,
        material=Material(permittivity=1.0, permeability=1.0, conductivity=0.0),
        color="rgba(0,0,255,0.3)",
        depth=0.001,
    )
    return polygon
