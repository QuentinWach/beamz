"""Matplotlib rendering backend for BeamZ plot-data payloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from beamz.const import LIGHT_SPEED, µm
from beamz.visual.helpers import get_si_scale_and_label


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _mpl_types():
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle, PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    return {
        "Circle": Circle,
        "FFMpegWriter": FFMpegWriter,
        "FuncAnimation": FuncAnimation,
        "LinearSegmentedColormap": LinearSegmentedColormap,
        "MplPath": MplPath,
        "PathPatch": PathPatch,
        "Rectangle": Rectangle,
    }


def get_twilight_zero_cmap():
    """Return the historical BeamZ diverging field colormap."""
    LinearSegmentedColormap = _mpl_types()["LinearSegmentedColormap"]
    colors = [
        (1.0, 1.0, 1.0),
        (0.2, 0.3, 0.8),
        (0.1, 0.1, 0.5),
        (0.1, 0.1, 0.1),
        (0.5, 0.1, 0.1),
        (0.8, 0.3, 0.2),
        (1.0, 1.0, 1.0),
    ]
    return LinearSegmentedColormap.from_list("twilight_zero", colors, N=256)


def resolve_cmap(cmap):
    if cmap == "twilight_zero":
        return get_twilight_zero_cmap()
    return cmap


def resolve_cmap_limits(cmap_limits="dynamic", *, vmin=None, vmax=None):
    """Normalize colormap scaling options to matplotlib ``vmin``/``vmax``."""
    explicit_limits = vmin is not None or vmax is not None
    if cmap_limits is None:
        cmap_limits = "dynamic"

    if isinstance(cmap_limits, str):
        if cmap_limits.lower() != "dynamic":
            raise ValueError("cmap_limits must be 'dynamic' or a (vmin, vmax) pair.")
        return vmin, vmax

    if explicit_limits:
        raise ValueError("Use either cmap_limits or vmin/vmax, not both.")

    try:
        low, high = cmap_limits
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "cmap_limits must be 'dynamic' or a (vmin, vmax) pair."
        ) from exc

    return (
        None if low is None else float(low),
        None if high is None else float(high),
    )


def _maybe_show(fig, *, show):
    if show:
        _pyplot().show()
    return fig


def _draw_polygon(ax, payload):
    vertices = payload["vertices"]
    if not vertices:
        return None

    types = _mpl_types()
    MplPath = types["MplPath"]
    PathPatch = types["PathPatch"]

    coords = []
    codes = []
    coords.extend(vertices)
    coords.append(vertices[0])
    codes.append(MplPath.MOVETO)
    if len(vertices) > 1:
        codes.extend([MplPath.LINETO] * (len(vertices) - 1))
    codes.append(MplPath.CLOSEPOLY)

    for hole in payload.get("interiors", []):
        if not hole:
            continue
        coords.extend(hole)
        coords.append(hole[0])
        codes.append(MplPath.MOVETO)
        if len(hole) > 1:
            codes.extend([MplPath.LINETO] * (len(hole) - 1))
        codes.append(MplPath.CLOSEPOLY)

    style = payload["style"]
    patch = PathPatch(
        MplPath(np.asarray(coords), np.asarray(codes)),
        facecolor=style.get("facecolor", "none"),
        edgecolor=style.get("edgecolor", "black"),
        alpha=style.get("alpha", 1.0),
        linestyle=style.get("linestyle", "-"),
    )
    ax.add_patch(patch)
    return patch


def _draw_source(ax, payload):
    style = payload["style"]
    Circle = _mpl_types()["Circle"]
    if payload["shape"] == "gaussian":
        circle = Circle(
            tuple(payload["position"][:2]),
            radius=payload["radius"],
            facecolor=style.get("facecolor", "none"),
            edgecolor=style.get("edgecolor", "orange"),
            linewidth=2,
            alpha=style.get("alpha", 0.8),
            linestyle=style.get("linestyle", "-"),
        )
        ax.add_patch(circle)
        ax.add_patch(
            Circle(
                tuple(payload["position"][:2]),
                radius=max(payload["radius"] * 0.1, 1e-9),
                facecolor=style.get("edgecolor", "orange"),
                edgecolor="none",
                alpha=style.get("alpha", 0.8),
            )
        )
        return circle

    if payload["shape"] != "mode":
        return None

    center = payload["center"]
    half_width = (payload.get("width") or 0.5e-6) / 2.0
    if payload["direction"] in {"+x", "-x"}:
        x = [center[0], center[0]]
        y = [center[1] - half_width, center[1] + half_width]
    else:
        x = [center[0] - half_width, center[0] + half_width]
        y = [center[1], center[1]]

    (line,) = ax.plot(
        x,
        y,
        color=style.get("edgecolor", "crimson"),
        linewidth=3,
        alpha=style.get("alpha", 0.8),
        solid_capstyle="round",
    )

    arrow_length = (payload.get("wavelength") or 0.5e-6) * 0.5
    dx, dy = 0.0, 0.0
    if payload["direction"] == "+x":
        dx = arrow_length
    elif payload["direction"] == "-x":
        dx = -arrow_length
    elif payload["direction"] == "+y":
        dy = arrow_length
    elif payload["direction"] == "-y":
        dy = -arrow_length

    end_x = center[0] + dx
    end_y = center[1] + dy
    ax.plot(
        [center[0], end_x],
        [center[1], end_y],
        color=style.get("edgecolor", "crimson"),
        linewidth=2,
        alpha=style.get("alpha", 0.8),
    )
    marker = {"+x": ">", "-x": "<", "+y": "^", "-y": "v"}.get(payload["direction"], "o")
    ax.plot(
        [end_x],
        [end_y],
        marker=marker,
        markersize=7,
        color=style.get("edgecolor", "crimson"),
        alpha=style.get("alpha", 0.8),
        linestyle="none",
    )
    return line


def _draw_monitor(ax, payload):
    style = payload["style"]
    if payload["shape"] == "line":
        x0, y0 = payload["start"][:2]
        x1, y1 = payload["end"][:2]
        color = style.get("edgecolor", "navy")
        (line,) = ax.plot(
            [x0, x1], [y0, y1], lw=4, color=color, alpha=style.get("alpha", 1.0)
        )
        ax.plot(
            [x0, x1],
            [y0, y1],
            lw=1,
            color=color,
            linestyle=style.get("linestyle", "-"),
        )
        return line

    Rectangle = _mpl_types()["Rectangle"]
    x0, y0 = payload["start"][:2]
    width, height = payload["size"][:2]
    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fill=style.get("facecolor", "none") != "none",
        facecolor=style.get("facecolor", "none"),
        edgecolor=style.get("edgecolor", "navy"),
        alpha=style.get("alpha", 1.0) * 0.3,
        linestyle=style.get("linestyle", "-"),
        linewidth=2,
    )
    ax.add_patch(rect)
    if payload.get("position") is not None:
        ax.text(
            payload["position"][0],
            payload["position"][1],
            payload["name"],
            ha="center",
            va="center",
            fontsize=8,
            color=style.get("edgecolor", "navy"),
        )
    return rect


def _draw_boundaries(ax, layout, line_color="gray", line_opacity=0.5):
    Rectangle = _mpl_types()["Rectangle"]
    for boundary in layout.get("boundaries", []):
        for rect in boundary["rectangles"]:
            ax.add_patch(
                Rectangle(
                    rect["origin"],
                    rect["width"],
                    rect["height"],
                    facecolor="none",
                    edgecolor=line_color,
                    linestyle=":",
                    alpha=line_opacity,
                )
            )


def _scaled_point(point, scale):
    return tuple(float(v) * scale for v in point)


def _scale_design_payload_for_axes(design_payload, scale):
    """Scale design plot payload coordinates for already-scaled axes."""
    payload = dict(design_payload)
    payload["xlim"] = _scaled_point(payload["xlim"], scale)
    payload["ylim"] = _scaled_point(payload["ylim"], scale)

    structures = []
    for structure in payload.get("structures", []):
        item = dict(structure)
        item["vertices"] = [_scaled_point(vertex, scale) for vertex in item["vertices"]]
        item["interiors"] = [
            [_scaled_point(vertex, scale) for vertex in interior]
            for interior in item.get("interiors", [])
        ]
        item["position"] = _scaled_point(item.get("position", (0.0, 0.0)), scale)
        structures.append(item)
    payload["structures"] = structures

    sources = []
    for source in payload.get("sources", []):
        item = dict(source)
        if "position" in item:
            item["position"] = _scaled_point(item["position"], scale)
        if "center" in item:
            item["center"] = _scaled_point(item["center"], scale)
        for key in ("radius", "width", "height", "wavelength"):
            if item.get(key) is not None:
                item[key] = float(item[key]) * scale
        sources.append(item)
    payload["sources"] = sources

    monitors = []
    for monitor in payload.get("monitors", []):
        item = dict(monitor)
        for key in ("start", "end", "position"):
            if key in item:
                item[key] = _scaled_point(item[key], scale)
        if "size" in item:
            item["size"] = _scaled_point(item["size"], scale)
        monitors.append(item)
    payload["monitors"] = monitors
    return payload


def _scale_layout_payload_for_axes(layout, scale):
    payload = dict(layout)
    payload["design"] = _scale_design_payload_for_axes(layout["design"], scale)
    boundaries = []
    for boundary in payload.get("boundaries", []):
        item = dict(boundary)
        rects = []
        for rect in item.get("rectangles", []):
            rect_item = dict(rect)
            rect_item["origin"] = _scaled_point(rect_item["origin"], scale)
            rect_item["width"] = float(rect_item["width"]) * scale
            rect_item["height"] = float(rect_item["height"]) * scale
            rects.append(rect_item)
        item["rectangles"] = rects
        boundaries.append(item)
    payload["boundaries"] = boundaries
    return payload


def _draw_simulation_overlay(ax, sim, *, scale, line_color="gray", line_opacity=0.5):
    layout = _scale_layout_payload_for_axes(sim.to_plot_data(), scale)
    design_payload = layout["design"]
    for structure in design_payload["structures"]:
        overlay = dict(structure)
        style = dict(overlay["style"])
        style["facecolor"] = "none"
        style["edgecolor"] = line_color
        style["alpha"] = line_opacity
        overlay["style"] = style
        _draw_polygon(ax, overlay)
    for source in design_payload["sources"]:
        _draw_source(ax, source)
    for monitor in design_payload["monitors"]:
        _draw_monitor(ax, monitor)
    _draw_boundaries(ax, layout, line_color=line_color, line_opacity=line_opacity)


def _configure_axes(ax, design_payload):
    unit = design_payload["scale_unit"]
    scale = design_payload["scale_factor"]
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x * scale:.1f}")
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y * scale:.1f}")


def _figure_axes(ax, *, figsize):
    plt = _pyplot()
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def _grid_permittivity_data_array(grid):
    """Return a labeled permittivity DataArray for a rasterized grid."""
    import xarray as xr

    arr = np.asarray(grid.permittivity)
    resolution = float(getattr(grid, "resolution", 1.0))
    design = getattr(grid, "design", None)
    attrs = {
        "component": "permittivity",
        "units": "relative",
    }
    if design is not None:
        attrs.update(
            {
                "design_width": float(getattr(design, "width", np.nan)),
                "design_height": float(getattr(design, "height", np.nan)),
                "design_depth": float(getattr(design, "depth", 0.0) or 0.0),
            }
        )
    if arr.ndim == 3:
        dims = ("z", "y", "x")
    elif arr.ndim == 2:
        dims = ("y", "x")
    else:
        dims = tuple(f"dim_{idx}" for idx in range(arr.ndim))
    coords = {
        dim: (dim, np.arange(size, dtype=float) * resolution, {"units": "m"})
        for dim, size in zip(dims, arr.shape, strict=True)
        if dim in {"x", "y", "z"}
    }
    return xr.DataArray(arr, dims=dims, coords=coords, name="permittivity", attrs=attrs)


def _nearest_coord_index(coord, value):
    values = np.asarray(coord, dtype=float)
    if values.size == 0:
        return 0
    return int(np.argmin(np.abs(values - float(value))))


def _material_category_array(
    eps, *, core_permittivity=None, substrate_permittivity=None
):
    eps = np.real(np.asarray(eps))
    eps_min = float(np.nanmin(eps)) if eps.size else 1.0
    eps_max = float(np.nanmax(eps)) if eps.size else 1.0
    if substrate_permittivity is None:
        unique = np.unique(np.round(eps[np.isfinite(eps)], decimals=8))
        substrate_permittivity = float(unique[1]) if unique.size >= 2 else eps_min
    if core_permittivity is None:
        core_permittivity = eps_max

    sub_mid = 0.5 * (eps_min + float(substrate_permittivity))
    core_mid = 0.5 * (float(substrate_permittivity) + float(core_permittivity))
    return np.where(eps >= core_mid, 2, np.where(eps >= sub_mid, 1, 0))


def _tidy3d_material_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#f5f5f5", "#83abc0", "#d86c96"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    return cmap, norm


def _plot_tidy3d_marker(ax, marker, *, vertical_coord="y"):
    if marker.get("orientation") == "horizontal":
        y = float(marker["y"])
        span = marker.get("x_span", marker.get("span"))
        if span is not None:
            x0, x1 = (float(span[0]), float(span[1]))
            ax.plot(
                [x0, x1],
                [y, y],
                color=marker.get("color", "#f4a51c"),
                lw=float(marker.get("linewidth", 2.0)),
            )
        if marker.get("arrow", False):
            x_mid = float(marker.get("arrow_x", 0.0))
            dy = float(marker.get("arrow_length", 0.55))
            if str(marker.get("direction", "+y")).startswith("-"):
                dy = -abs(dy)
            ax.annotate(
                "",
                xy=(x_mid, y + dy),
                xytext=(x_mid, y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=marker.get("arrow_color", marker.get("color", "#11823b")),
                    lw=float(marker.get("arrow_linewidth", 2.0)),
                ),
            )
        return

    x = float(marker["x"])
    span = marker.get(f"{vertical_coord}_span", marker.get("span"))
    if span is not None:
        y0, y1 = (float(span[0]), float(span[1]))
        ax.plot(
            [x, x],
            [y0, y1],
            color=marker.get("color", "#f4a51c"),
            lw=float(marker.get("linewidth", 2.0)),
        )
    if marker.get("arrow", False):
        y_mid = float(marker.get(f"arrow_{vertical_coord}", marker.get("arrow_y", 0.0)))
        dx = float(marker.get("arrow_length", 0.55))
        if str(marker.get("direction", "+x")).startswith("-"):
            dx = -abs(dx)
        ax.annotate(
            "",
            xy=(x + dx, y_mid),
            xytext=(x, y_mid),
            arrowprops=dict(
                arrowstyle="-|>",
                color=marker.get("arrow_color", marker.get("color", "#11823b")),
                lw=float(marker.get("arrow_linewidth", 2.0)),
            ),
        )


def _tidy3d_origin_for_simulation(sim):
    design = getattr(sim, "design", None)
    offset = getattr(sim, "coordinate_offset", None)
    if offset is not None:
        return tuple(float(v) for v in offset)
    return (
        0.5 * float(getattr(design, "width", 0.0)),
        0.5 * float(getattr(design, "height", 0.0)),
        0.5 * float(getattr(design, "depth", 0.0) or 0.0),
    )


def _tidy3d_material_levels(sim):
    eps = np.asarray(getattr(getattr(sim, "fields", None), "permittivity", ()))
    finite = np.real(eps[np.isfinite(eps)]) if eps.size else np.asarray(())
    if finite.size == 0:
        return None, None
    unique = np.unique(np.round(finite, decimals=8))
    substrate = float(unique[1]) if unique.size >= 3 else None
    core = float(unique[-1])
    return core, substrate


def _field_eps_slice(simulation, *, plane="z", index=None, plane_position=None):
    eps = np.asarray(getattr(getattr(simulation, "fields", None), "permittivity", ()))
    if eps.size == 0:
        return None
    try:
        from beamz.visual.data import _slice_2d

        if plane_position is not None and index is None:
            plane_key = str(plane).lower()
            axis = (
                0 if plane_key in {"xy", "z"} else 1 if plane_key in {"xz", "y"} else 2
            )
            index = int(
                np.clip(
                    round(float(plane_position) / float(simulation.resolution)),
                    0,
                    eps.shape[axis] - 1,
                )
            )
        eps_slice, _plane_label, _selected = _slice_2d(eps, plane=plane, index=index)
        return np.asarray(eps_slice)
    except Exception:
        return None


def _draw_field_eps_overlay(
    ax,
    eps_slice,
    *,
    extent,
    reverse=False,
    alpha=0.2,
    core_permittivity=None,
):
    if eps_slice is None:
        return None
    eps_real = np.real(np.asarray(eps_slice))
    finite = eps_real[np.isfinite(eps_real)]
    if finite.size == 0:
        return None
    eps_min = float(np.nanmin(finite))
    eps_max = (
        float(core_permittivity)
        if core_permittivity is not None
        else float(np.nanmax(finite))
    )
    eps_span = max(eps_max - eps_min, 1e-5)
    eps_fraction = np.clip((eps_real - eps_min) / eps_span, 0.0, 1.0)
    eps_color = eps_fraction if reverse else 1.0 - eps_fraction
    core = eps_real >= 0.5 * eps_max
    if not np.any(core):
        return None
    rgba = np.zeros((*core.shape, 4), dtype=float)
    rgba[core, :3] = eps_color[core, None]
    rgba[core, 3] = np.clip(float(alpha), 0.0, 1.0)
    return ax.imshow(
        rgba,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )


def _tidy3d_field_display_scale_and_units(field):
    """Return plot scale and units for Tidy3D-style field-monitor data."""
    component = str(field)
    family = component[:1].upper()
    if family == "E":
        return µm, "V/um"
    if family == "H":
        return µm, "A/um"
    return 1.0, ""


def _tidy3d_pml_thickness(sim):
    for boundary in getattr(sim, "boundaries", ()) or ():
        thickness = getattr(boundary, "thickness", None)
        if thickness is not None:
            return float(thickness)
    return None


def _normal_axis_from_device(device):
    direction = getattr(device, "direction", None)
    if direction is not None:
        direction = str(direction).lower()
        for axis in ("x", "y", "z"):
            if axis in direction:
                return axis
    plane_normal = getattr(device, "plane_normal", None)
    if plane_normal is not None:
        return str(plane_normal).lower()
    size = getattr(device, "size_spec", None)
    if size is not None:
        return ("x", "y", "z")[int(np.argmin(np.abs(np.asarray(size, dtype=float))))]
    return None


def _device_plane_center(device):
    if hasattr(device, "center"):
        return tuple(float(v) for v in getattr(device, "center"))
    if hasattr(device, "position"):
        return tuple(float(v) for v in getattr(device, "position"))
    start = getattr(device, "start", None)
    end = getattr(device, "end", None)
    if start is not None and end is not None:
        return tuple(
            0.5 * (float(a) + float(b)) for a, b in zip(start, end, strict=False)
        )
    return None


def _device_span_for_axis(device, axis, transverse_axis):
    start = getattr(device, "start", None)
    end = getattr(device, "end", None)
    idx = {"x": 0, "y": 1, "z": 2}[transverse_axis]
    if start is not None and end is not None and len(start) > idx and len(end) > idx:
        return (
            min(float(start[idx]), float(end[idx])),
            max(float(start[idx]), float(end[idx])),
        )

    center = _device_plane_center(device)
    size = getattr(device, "size_spec", None)
    if size is None:
        raw_size = getattr(device, "size", None)
        if raw_size is not None and not callable(raw_size):
            raw_size = tuple(float(v) for v in raw_size)
            if len(raw_size) == 3:
                size = raw_size
            elif len(raw_size) == 2 and axis in {"x", "y", "z"}:
                if axis == "x":
                    size = (0.0, raw_size[0], raw_size[1])
                elif axis == "y":
                    size = (raw_size[0], 0.0, raw_size[1])
                else:
                    size = (raw_size[0], raw_size[1], 0.0)
    if size is None and axis == "x":
        size = (
            0.0,
            getattr(device, "width", 0.0) or 0.0,
            getattr(device, "height", 0.0) or 0.0,
        )
    if center is not None and size is not None and len(size) > idx:
        half = 0.5 * float(size[idx])
        return (float(center[idx]) - half, float(center[idx]) + half)
    return None


def _tidy3d_device_markers(devices, origin, *, color, source=False):
    markers_xy = []
    markers_xz = []
    ox, oy, oz = (float(v) for v in origin)
    for device in devices:
        axis = _normal_axis_from_device(device)
        if axis not in {"x", "y", "z"}:
            continue
        center = _device_plane_center(device)
        if center is None or len(center) < 3:
            continue
        if axis == "x":
            y_span = _device_span_for_axis(device, axis, "y")
            z_span = _device_span_for_axis(device, axis, "z")
            base = {
                "x": (float(center[0]) - ox) / µm,
                "color": color,
                "direction": getattr(device, "direction", "+x"),
            }
            if source:
                base.update({"arrow": True, "arrow_y": (float(center[1]) - oy) / µm})
            if y_span is not None:
                markers_xy.append(
                    {
                        **base,
                        "span": ((y_span[0] - oy) / µm, (y_span[1] - oy) / µm),
                    }
                )
            if z_span is not None:
                marker = {
                    **base,
                    "span": ((z_span[0] - oz) / µm, (z_span[1] - oz) / µm),
                }
                if source:
                    marker["arrow_y"] = (float(center[2]) - oz) / µm
                markers_xz.append(marker)
            continue

        if axis == "z":
            x_span = _device_span_for_axis(device, axis, "x")
            y_span = _device_span_for_axis(device, axis, "y")
            if x_span is not None:
                markers_xy.append(
                    {
                        "orientation": "horizontal",
                        "y": (float(center[1]) - oy) / µm,
                        "span": ((x_span[0] - ox) / µm, (x_span[1] - ox) / µm),
                        "color": color,
                    }
                )
                marker = {
                    "orientation": "horizontal",
                    "y": (float(center[2]) - oz) / µm,
                    "span": ((x_span[0] - ox) / µm, (x_span[1] - ox) / µm),
                    "color": color,
                    "direction": getattr(device, "direction", "-z"),
                }
                if source:
                    marker.update(
                        {
                            "arrow": True,
                            "arrow_x": (float(center[0]) - ox) / µm,
                        }
                    )
                markers_xz.append(marker)
            if y_span is not None:
                markers_xy.append(
                    {
                        "x": (float(center[0]) - ox) / µm,
                        "span": ((y_span[0] - oy) / µm, (y_span[1] - oy) / µm),
                        "color": color,
                    }
                )
            continue

        x_span = _device_span_for_axis(device, axis, "x")
        if x_span is not None:
            marker = {
                "orientation": "horizontal",
                "y": (float(center[1]) - oy) / µm,
                "span": ((x_span[0] - ox) / µm, (x_span[1] - ox) / µm),
                "color": color,
                "direction": getattr(device, "direction", "+y"),
            }
            if source:
                marker.update({"arrow": True, "arrow_x": (float(center[0]) - ox) / µm})
            markers_xy.append(marker)
    return markers_xy, markers_xz


def _plot_tidy3d_simulation_cross_sections(
    sim,
    *,
    z=0.0,
    y=0.0,
    origin=None,
    source_markers=True,
    monitor_markers=True,
    width_ratios=None,
    xlim=None,
    ylim=None,
    zlim=None,
    show=True,
    figsize=(11, 4),
):
    if origin is None:
        origin = _tidy3d_origin_for_simulation(sim)
    z_abs = float(z) + float(origin[2])
    y_abs = float(y) + float(origin[1])
    grid = sim.design.rasterize(resolution=sim.resolution)
    core_eps, substrate_eps = _tidy3d_material_levels(sim)
    xy_markers = []
    xz_markers = []
    if source_markers:
        source_xy, source_xz = _tidy3d_device_markers(
            getattr(sim, "sources", ()),
            origin,
            color="#66bb6a",
            source=True,
        )
        xy_markers.extend(source_xy)
        xz_markers.extend(source_xz)
    if monitor_markers:
        monitor_xy, monitor_xz = _tidy3d_device_markers(
            getattr(sim, "monitors", ()),
            origin,
            color="#f4a51c",
            source=False,
        )
        xy_markers.extend(monitor_xy)
        xz_markers.extend(monitor_xz)

    return plot_tidy3d_cross_sections(
        grid,
        z=z_abs,
        y=y_abs,
        origin=origin,
        substrate_z=float(origin[2]),
        core_permittivity=core_eps,
        substrate_permittivity=substrate_eps,
        pml_thickness=_tidy3d_pml_thickness(sim),
        xy_markers=xy_markers,
        xz_markers=xz_markers,
        figsize=figsize,
        width_ratios=width_ratios,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        show=show,
    )


def plot_tidy3d_cross_sections(
    grid,
    *,
    z=0.0,
    y=None,
    origin=None,
    substrate_z=None,
    core_permittivity=None,
    substrate_permittivity=None,
    pml_thickness=None,
    xy_markers=(),
    xz_markers=(),
    figsize=(11, 4),
    width_ratios=None,
    xlim=None,
    ylim=None,
    zlim=None,
    show=True,
):
    """Plot Tidy3D-like ``xy`` and ``xz`` permittivity cross sections.

    Parameters are in meters except marker coordinates, which are in microns in
    the plotted coordinate system. ``origin`` shifts the displayed coordinates;
    pass ``(width/2, height/2, substrate_top)`` to reproduce the centered
    coordinates common in Tidy3D examples.
    """

    plt = _pyplot()
    da = _grid_permittivity_data_array(grid)
    design = getattr(grid, "design", None)
    width = float(getattr(design, "width", da.sizes.get("x", 1) * grid.resolution))
    height = float(getattr(design, "height", da.sizes.get("y", 1) * grid.resolution))
    depth = float(getattr(design, "depth", da.sizes.get("z", 1) * grid.resolution))
    if y is None:
        y = 0.5 * height
    if origin is None:
        origin = (0.0, 0.0, 0.0)
    ox, oy, oz = (float(v) for v in origin)

    z_index = _nearest_coord_index(da.coords["z"], float(z))
    y_index = _nearest_coord_index(da.coords["y"], float(y))
    xy = _material_category_array(
        da.isel(z=z_index).values,
        core_permittivity=core_permittivity,
        substrate_permittivity=substrate_permittivity,
    )
    if substrate_z is not None and da.ndim == 3:
        # Tidy3D displays material boxes inclusively at their upper boundary.
        # For a slice on the substrate top, show the substrate across the xy
        # plane while preserving any higher-index structure in the selected
        # slice. Keep this to the interface itself so silica does not appear
        # above the waveguide bottom in nearby slices.
        z_tol = 0.5 * float(getattr(grid, "resolution", 0.0) or 0.0)
        if abs(float(z) - float(substrate_z)) <= z_tol:
            xy = np.where(xy >= 2, xy, 1)

    xz = _material_category_array(
        da.isel(y=y_index).values,
        core_permittivity=core_permittivity,
        substrate_permittivity=substrate_permittivity,
    )
    if substrate_z is not None and da.ndim == 3:
        z_coords = np.asarray(da.coords["z"], dtype=float)
        above_substrate = z_coords > float(substrate_z)
        if np.any(above_substrate):
            xz = np.where(above_substrate[:, None] & (xz == 1), 0, xz)
        substrate_mask = z_coords <= float(substrate_z)
        if np.any(substrate_mask):
            xz = np.where(substrate_mask[:, None] & (xz == 0), 1, xz)

    cmap, norm = _tidy3d_material_cmap()
    if width_ratios is None:
        fig, axes = plt.subplots(1, 2, tight_layout=True, figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        import matplotlib as mpl

        gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=width_ratios)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    xy_extent = [
        (0.0 - ox) / µm,
        (width - ox) / µm,
        (0.0 - oy) / µm,
        (height - oy) / µm,
    ]
    xz_extent = [
        (0.0 - ox) / µm,
        (width - ox) / µm,
        (0.0 - oz) / µm,
        (depth - oz) / µm,
    ]
    axes[0].imshow(
        xy,
        origin="lower",
        extent=xy_extent,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    axes[1].imshow(
        xz,
        origin="lower",
        extent=xz_extent,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )

    if pml_thickness is not None and pml_thickness > 0:
        p = float(pml_thickness) / µm
        hatch_style = dict(
            facecolor="#9a9a9a",
            alpha=0.35,
            hatch="xx",
            edgecolor="#777777",
            linewidth=0.0,
        )
        for ax, extent in zip(axes, (xy_extent, xz_extent), strict=True):
            ax.axvspan(extent[0], extent[0] + p, **hatch_style)
            ax.axvspan(extent[1] - p, extent[1], **hatch_style)
            ax.axhspan(extent[2], extent[2] + p, **hatch_style)
            ax.axhspan(extent[3] - p, extent[3], **hatch_style)

    for marker in xy_markers:
        _plot_tidy3d_marker(axes[0], marker, vertical_coord="y")
    for marker in xz_markers:
        _plot_tidy3d_marker(axes[1], marker, vertical_coord="z")

    axes[0].set_title(f"cross section at z={(float(z) - oz) / µm:.2f} (um)", fontsize=9)
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[1].set_title(f"cross section at y={(float(y) - oy) / µm:.2f} (um)", fontsize=9)
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("z (um)")
    for ax in axes:
        ax.set_xlim(xy_extent[0], xy_extent[1])
    if xlim is not None:
        axes[0].set_xlim(*xlim)
        axes[1].set_xlim(*xlim)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    if zlim is not None:
        axes[1].set_ylim(*zlim)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, axes


def mode_field_component_pairs(
    components=("Ey", "Ez"),
    *,
    direction="-x",
    display_components=None,
):
    """Return ``(display_label, solver_component)`` pairs for physical mode fields."""
    if display_components is not None:
        return list(display_components)
    del direction
    return [(name, name) for name in components]


def plot_mode_fields(
    grid,
    *,
    plane_x,
    wavelength,
    polarization=None,
    num_modes=3,
    components=("Ey", "Ez"),
    display_components=None,
    window=None,
    origin=None,
    direction="-x",
    target_neff=None,
    val="abs",
    normalize=False,
    vmin=None,
    vmax=None,
    percentile=99.5,
    figsize=(12, 12),
    show=True,
):
    """Solve and plot mode field components using physical Cartesian labels."""
    from beamz.devices.sources.solve import solve_modes

    plt = _pyplot()
    eps = np.asarray(grid.permittivity)
    dx = float(getattr(grid, "resolution", 1.0))
    x_index = int(np.clip(round(float(plane_x) / dx), 0, eps.shape[2] - 1))
    eps_profile = eps[:, :, x_index]
    if origin is None:
        design = getattr(grid, "design", None)
        origin = (
            0.0,
            0.5 * float(getattr(design, "height", eps.shape[1] * dx)),
            0.0,
        )
    _ox, oy, oz = (float(v) for v in origin)
    if window is None:
        y0, y1 = 0.0, eps.shape[1] * dx
        z0, z1 = 0.0, eps.shape[0] * dx
    else:
        y0, y1, z0, z1 = (float(v) for v in window)
    iy0 = int(np.clip(np.floor(y0 / dx), 0, eps_profile.shape[1] - 1))
    iy1 = int(np.clip(np.ceil(y1 / dx), iy0 + 1, eps_profile.shape[1]))
    iz0 = int(np.clip(np.floor(z0 / dx), 0, eps_profile.shape[0] - 1))
    iz1 = int(np.clip(np.ceil(z1 / dx), iz0 + 1, eps_profile.shape[0]))
    eps_profile = eps_profile[iz0:iz1, iy0:iy1]
    neffs, e_fields, _h_fields, _ = solve_modes(
        eps=eps_profile,
        omega=2.0 * np.pi * LIGHT_SPEED / float(wavelength),
        dL=dx,
        m=int(num_modes),
        direction=direction,
        filter_pol=polarization,
        target_neff=target_neff,
        return_fields=True,
    )
    comp_map = {"Ex": 0, "Ey": 1, "Ez": 2}
    display_components = mode_field_component_pairs(
        components,
        direction=direction,
        display_components=display_components,
    )
    extent = [(y0 - oy) / µm, (y1 - oy) / µm, (z0 - oz) / µm, (z1 - oz) / µm]

    fig, axes = plt.subplots(
        int(num_modes),
        len(display_components),
        figsize=figsize,
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(int(num_modes), len(display_components))
    for mode_index in range(int(num_modes)):
        for col, (display_name, actual_name) in enumerate(display_components):
            arr = np.squeeze(np.asarray(e_fields[mode_index, comp_map[actual_name]]))
            val_key = str(val).lower()
            if val_key in {"abs", "magnitude"}:
                plot_arr = np.abs(arr)
                label = f"|{display_name}|"
                cmap = "magma"
                default_vmin = 0.0
                finite_scale = plot_arr[np.isfinite(plot_arr)]
                if normalize and finite_scale.size:
                    scale = float(np.nanmax(finite_scale))
                    if scale > 0.0:
                        plot_arr = plot_arr / scale
                finite_plot = plot_arr[np.isfinite(plot_arr)]
                if percentile is None:
                    default_vmax = (
                        float(np.nanmax(finite_plot)) if finite_plot.size else 1.0
                    )
                else:
                    default_vmax = float(np.nanpercentile(plot_arr, float(percentile)))
                default_vmax = (
                    default_vmax
                    if np.isfinite(default_vmax) and default_vmax > 0.0
                    else 1.0
                )
            elif val_key in {"real", "re"}:
                plot_arr = np.real(arr)
                label = f"Re({display_name})"
                cmap = "RdBu"
                default_vmin = None
                finite_plot = np.abs(plot_arr[np.isfinite(plot_arr)])
                if percentile is None:
                    default_vmax = (
                        float(np.nanmax(finite_plot)) if finite_plot.size else 1.0
                    )
                else:
                    default_vmax = np.nanpercentile(np.abs(plot_arr), float(percentile))
                default_vmax = (
                    default_vmax
                    if np.isfinite(default_vmax) and default_vmax > 0.0
                    else 1.0
                )
                default_vmin = -default_vmax
            elif val_key in {"imag", "imaginary", "im"}:
                plot_arr = np.imag(arr)
                label = f"Im({display_name})"
                cmap = "RdBu"
                default_vmin = None
                finite_plot = np.abs(plot_arr[np.isfinite(plot_arr)])
                if percentile is None:
                    default_vmax = (
                        float(np.nanmax(finite_plot)) if finite_plot.size else 1.0
                    )
                else:
                    default_vmax = np.nanpercentile(np.abs(plot_arr), float(percentile))
                default_vmax = (
                    default_vmax
                    if np.isfinite(default_vmax) and default_vmax > 0.0
                    else 1.0
                )
                default_vmin = -default_vmax
            else:
                raise ValueError("val must be one of 'abs', 'real', or 'imag'.")
            ax = axes_arr[mode_index, col]
            im = ax.imshow(
                plot_arr,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=default_vmin if vmin is None else float(vmin),
                vmax=default_vmax if vmax is None else float(vmax),
                aspect="equal",
                interpolation="nearest",
            )
            ax.set_title(f"{display_name}, mode_index={mode_index}", fontsize=9)
            ax.set_xlabel("y (um)")
            ax.set_ylabel("z (um)")
            fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.02,
                extend="both",
                label=label,
            )
    _maybe_show(fig, show=show)
    return fig, axes, neffs


def plot_tidy3d_mode_components(*args, **kwargs):
    """Compatibility wrapper for :func:`plot_mode_fields`."""
    return plot_mode_fields(*args, **kwargs)


def plot_tidy3d_field_frame(
    results,
    *,
    field="Ez",
    display_field=None,
    plane="z",
    index=None,
    select="max_energy",
    percentile=99.5,
    origin=None,
    ax=None,
    figsize=(6, 4),
    cmap="RdBu",
    overlay_core=True,
    eps_alpha=0.2,
    core_permittivity=None,
    show=True,
):
    """Plot a centered, symmetric Tidy3D-like field frame from results."""
    ds = results.to_xarray()
    da = ds[field]
    if "t" in da.dims or "frame" in da.dims:
        frame_dim = "t" if "t" in da.dims else "frame"
        if select == "max_energy":
            axes = tuple(dim for dim in da.dims if dim != frame_dim)
            energy = np.sqrt((np.abs(da) ** 2).sum(dim=axes))
            da = da.isel({frame_dim: int(energy.argmax(dim=frame_dim).values)})
        else:
            da = da.isel({frame_dim: int(select)})
    if plane == "z" and "z" in da.dims:
        z_coord = np.asarray(da.coords["z"], dtype=float)
        idx = _nearest_coord_index(
            z_coord, z_coord[len(z_coord) // 2] if index is None else float(index)
        )
        da = da.isel(z=idx)
    arr = np.asarray(da)
    design_width = float(ds.attrs.get("design_width", arr.shape[-1]))
    design_height = float(ds.attrs.get("design_height", arr.shape[-2]))
    if origin is None:
        origin = (0.5 * design_width, 0.5 * design_height, 0.0)
    ox, oy, oz = (float(v) for v in origin)
    extent = [
        (0.0 - ox) / µm,
        (design_width - ox) / µm,
        (0.0 - oy) / µm,
        (design_height - oy) / µm,
    ]
    vmax = np.nanpercentile(np.abs(np.real(arr)), float(percentile))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    fig, ax = _figure_axes(ax, figsize=figsize)
    im = ax.imshow(
        np.real(arr),
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    if overlay_core:
        simulation = getattr(results, "simulation", None)
        if simulation is not None:
            eps_index = None
            if plane == "z" and "z" in da.dims:
                eps_index = idx
            eps_slice = _field_eps_slice(simulation, plane=plane, index=eps_index)
            _draw_field_eps_overlay(
                ax,
                eps_slice,
                extent=extent,
                reverse=False,
                alpha=eps_alpha,
                core_permittivity=core_permittivity,
            )
    label = display_field or field
    fig.colorbar(im, ax=ax, label=f"Re({label})")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    if plane == "z":
        z_value = 0.0 if index is None else float(index) - oz
        ax.set_title(f"cross section at z={z_value / µm:.2f} (um)")
    else:
        ax.set_title(f"cross section at {plane}")
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_tidy3d_dft_field(
    simulation,
    monitor,
    *,
    field="Ey",
    display_field=None,
    frequency=None,
    frequency_index=0,
    val="real",
    origin=None,
    percentile=99.5,
    vmin=None,
    vmax=None,
    ax=None,
    figsize=(6, 4),
    cmap="RdBu",
    overlay_core=True,
    eps_alpha=0.2,
    core_permittivity=None,
    xlim=None,
    ylim=None,
    source_normalize=True,
    show_units=True,
    show=True,
):
    """Plot a Tidy3D-like frequency-domain field from a DFT plane monitor."""

    if not getattr(monitor, "is_3d", False):
        raise ValueError("plot_tidy3d_dft_field expects a 3D plane monitor.")
    axis = str(getattr(monitor, "plane_normal", "z")).lower()
    plane_axes = {
        "x": ("z", "y"),
        "y": ("z", "x"),
        "z": ("y", "x"),
    }
    if axis not in plane_axes:
        raise ValueError("DFT field monitor plane_normal must be one of x, y, or z.")
    axis0, axis1 = plane_axes[axis]

    freqs = np.asarray(monitor.get_dft_frequencies(), dtype=float)
    if freqs.size == 0:
        raise ValueError(
            f"Monitor '{getattr(monitor, 'name', None)}' has no DFT frequencies."
        )
    if frequency is None:
        f_idx = int(frequency_index)
    else:
        f_idx = int(np.argmin(np.abs(freqs - float(frequency))))
    f_idx = int(np.clip(f_idx, 0, freqs.size - 1))

    field_key = str(field)
    vector_components = None
    if field_key in {"E", "H"}:
        candidates = ("Ex", "Ey", "Ez") if field_key == "E" else ("Hx", "Hy", "Hz")
        vector_components = []
        missing = []
        for component in candidates:
            try:
                values = np.asarray(
                    monitor.get_dft_component(component), dtype=np.complex128
                )
            except ValueError:
                missing.append(component)
                continue
            vector_components.append((component, values))
        if not vector_components:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"No DFT data recorded for derived field '{field_key}'. "
                f"Expected at least one of: {missing_text}."
            )
        dft = vector_components[0][1]
    else:
        dft = np.asarray(monitor.get_dft_component(field_key), dtype=np.complex128)
    plane_shape = getattr(monitor, "_compiled_dft_shape_3d", None)
    if plane_shape is None:
        fields = getattr(simulation, "fields", None)
        base_shape = tuple(int(v) for v in np.asarray(fields.permittivity).shape)
        target0, target1 = monitor.get_analysis_plane_coords_3d(
            dx=float(simulation.resolution),
            dy=float(simulation.resolution),
            dz=float(simulation.resolution),
            field_shape=base_shape,
        )
        plane_shape = (int(target0.size), int(target1.size))
    else:
        fields = getattr(simulation, "fields", None)
        base_shape = tuple(int(v) for v in np.asarray(fields.permittivity).shape)
        target0, target1 = monitor.get_analysis_plane_coords_3d(
            dx=float(simulation.resolution),
            dy=float(simulation.resolution),
            dz=float(simulation.resolution),
            field_shape=base_shape,
        )
        target0 = np.asarray(target0, dtype=float)[: int(plane_shape[0])]
        target1 = np.asarray(target1, dtype=float)[: int(plane_shape[1])]

    def component_at_frequency(values):
        return np.asarray(values[f_idx], dtype=np.complex128).reshape(
            tuple(int(v) for v in plane_shape)
        )

    if vector_components is None:
        arr = component_at_frequency(dft)
        component_arrays = None
    else:
        component_arrays = [
            component_at_frequency(values) for _name, values in vector_components
        ]
        arr = component_arrays[0]
    if source_normalize:
        from beamz.simulation.core import _source_spectrum_normalization

        source_norm = _source_spectrum_normalization(
            getattr(simulation, "sources", ()),
            freqs,
            time=getattr(simulation, "time", None),
            monitor=monitor,
        )
        if source_norm is not None:
            norm = np.asarray(source_norm, dtype=np.complex128).reshape(-1)
            if norm.size == freqs.size and abs(norm[f_idx]) > 1e-12:
                arr = arr / norm[f_idx]
                if component_arrays is not None:
                    component_arrays = [
                        component / norm[f_idx] for component in component_arrays
                    ]

    field_scale, field_units = _tidy3d_field_display_scale_and_units(field)
    arr = arr * field_scale
    if component_arrays is not None:
        component_arrays = [component * field_scale for component in component_arrays]
    x_coords = np.asarray(target1, dtype=float)
    y_coords = np.asarray(target0, dtype=float)
    val_key = str(val).lower().replace(" ", "")
    is_power = val_key in {"abs^2", "abs2", "abs_sq", "abssq", "power"}
    if component_arrays is not None and val_key not in {
        "abs",
        "magnitude",
        "abs^2",
        "abs2",
        "abs_sq",
        "abssq",
        "power",
    }:
        raise ValueError(
            f"Derived field '{field_key}' supports val='abs' or val='abs^2'."
        )

    if component_arrays is not None and is_power:
        plot_arr = np.sum(
            [np.abs(component) ** 2 for component in component_arrays], axis=0
        )
        label_prefix = "|"
        label_suffix = r"$^2$"
    elif component_arrays is not None:
        plot_arr = np.sqrt(
            np.sum([np.abs(component) ** 2 for component in component_arrays], axis=0)
        )
        label_prefix = "|"
        label_suffix = ""
    elif val_key in {"real", "re"}:
        plot_arr = np.real(arr)
        label_prefix = "Re"
        label_suffix = ""
    elif val_key in {"imag", "imaginary", "im"}:
        plot_arr = np.imag(arr)
        label_prefix = "Im"
        label_suffix = ""
    elif val_key in {"abs", "magnitude"}:
        plot_arr = np.abs(arr)
        label_prefix = "|"
        label_suffix = ""
    elif is_power:
        plot_arr = np.abs(arr) ** 2
        label_prefix = "|"
        label_suffix = r"$^2$"
    else:
        raise ValueError("val must be one of 'real', 'imag', 'abs', or 'abs^2'.")

    design = getattr(simulation, "design", None)
    if origin is None:
        origin = _tidy3d_origin_for_simulation(simulation)
    ox, oy, oz = (float(v) for v in origin)
    origin_by_axis = {"x": ox, "y": oy, "z": oz}
    dx = (
        float(np.mean(np.diff(x_coords)))
        if x_coords.size > 1
        else float(simulation.resolution)
    )
    dy = (
        float(np.mean(np.diff(y_coords)))
        if y_coords.size > 1
        else float(simulation.resolution)
    )
    extent = [
        (float(x_coords[0]) - 0.5 * dx - origin_by_axis[axis1]) / µm,
        (float(x_coords[-1]) + 0.5 * dx - origin_by_axis[axis1]) / µm,
        (float(y_coords[0]) - 0.5 * dy - origin_by_axis[axis0]) / µm,
        (float(y_coords[-1]) + 0.5 * dy - origin_by_axis[axis0]) / µm,
    ]

    auto_vmax = np.nanpercentile(np.abs(plot_arr), float(percentile))
    auto_vmax = auto_vmax if np.isfinite(auto_vmax) and auto_vmax > 0 else 1.0
    fig, ax = _figure_axes(ax, figsize=figsize)
    eps_reverse = val_key in {"abs", "magnitude"} or is_power
    if val_key in {"abs", "magnitude"} or is_power:
        im = ax.imshow(
            plot_arr,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=0.0 if vmin is None else vmin,
            vmax=auto_vmax if vmax is None else vmax,
            aspect="equal",
            interpolation="nearest",
        )
    else:
        im = ax.imshow(
            plot_arr,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=-auto_vmax if vmin is None else vmin,
            vmax=auto_vmax if vmax is None else vmax,
            aspect="equal",
            interpolation="nearest",
        )

    if overlay_core:
        eps_slice = _field_eps_slice(
            simulation,
            plane=axis,
            plane_position=float(getattr(monitor, "plane_position", 0.0)),
        )
        slice_shape = (
            tuple(np.asarray(eps_slice).shape) if eps_slice is not None else ()
        )
        fallback_extents = {
            axis0: (slice_shape[0] * simulation.resolution if slice_shape else 0.0),
            axis1: (
                slice_shape[1] * simulation.resolution
                if len(slice_shape) > 1
                else 0.0
            ),
        }
        design_attrs = {"x": "width", "y": "height", "z": "depth"}

        def design_extent(axis_name):
            fallback = float(fallback_extents.get(axis_name, 0.0))
            return float(
                getattr(design, design_attrs[axis_name], fallback) or fallback
            )

        _draw_field_eps_overlay(
            ax,
            eps_slice,
            extent=[
                (0.0 - origin_by_axis[axis1]) / µm,
                (design_extent(axis1) - origin_by_axis[axis1]) / µm,
                (0.0 - origin_by_axis[axis0]) / µm,
                (design_extent(axis0) - origin_by_axis[axis0]) / µm,
            ],
            reverse=eps_reverse,
            alpha=eps_alpha,
            core_permittivity=core_permittivity,
        )

    label = display_field or field
    if label_prefix == "|":
        cbar_label = f"|{label}|{label_suffix}"
    else:
        cbar_label = f"{label_prefix}({label})"
    if show_units and field_units:
        if is_power:
            field_units = f"({field_units})^2"
        cbar_label = f"{cbar_label} ({field_units})"
    fig.colorbar(
        im,
        ax=ax,
        label=cbar_label,
        extend="both" if not (val_key in {"abs", "magnitude"} or is_power) else "max",
    )
    ax.set_xlabel(f"{axis1} (um)")
    ax.set_ylabel(f"{axis0} (um)")
    plane_value = (
        float(getattr(monitor, "plane_position", 0.0)) - origin_by_axis[axis]
    )
    ax.set_title(f"cross section at {axis}={plane_value / µm:.2f} (um)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_design(
    design,
    *,
    sources=None,
    monitors=None,
    ax=None,
    figsize=None,
    show=True,
    title="Design Layout",
):
    """Plot a design layout and optional source/monitor overlays."""
    payload = design.to_plot_data(sources=sources, monitors=monitors)
    if figsize is None:
        figsize = (6.0, 6.0 * float(design.height) / max(float(design.width), 1e-30))
    fig, ax = _figure_axes(ax, figsize=figsize)

    for structure in payload["structures"]:
        _draw_polygon(ax, structure)
    for source in payload["sources"]:
        _draw_source(ax, source)
    for monitor in payload["monitors"]:
        _draw_monitor(ax, monitor)

    ax.set_title(title)
    ax.set_xlim(*payload["xlim"])
    ax.set_ylim(*payload["ylim"])
    ax.set_aspect("equal")
    _configure_axes(ax, payload)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_simulation(
    sim,
    *,
    z=None,
    y=None,
    tidy3d=None,
    source_markers=True,
    monitor_markers=True,
    width_ratios=None,
    xlim=None,
    ylim=None,
    zlim=None,
    origin=None,
    ax=None,
    figsize=None,
    show=True,
    title="Simulation Layout",
):
    """Plot a simulation layout with sources, monitors, and boundaries."""
    use_tidy3d = tidy3d
    if use_tidy3d is None:
        use_tidy3d = bool(
            getattr(sim, "is_3d", False) and (z is not None or y is not None)
        )
    if use_tidy3d:
        if ax is not None:
            raise ValueError(
                "Tidy3D-style cross-section plots create two axes; omit ax."
            )
        return _plot_tidy3d_simulation_cross_sections(
            sim,
            z=0.0 if z is None else z,
            y=0.0 if y is None else y,
            origin=origin,
            source_markers=source_markers,
            monitor_markers=monitor_markers,
            width_ratios=width_ratios,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            figsize=(11, 4) if figsize is None else figsize,
            show=show,
        )

    payload = sim.to_plot_data()
    design_payload = payload["design"]
    if figsize is None:
        width = max(float(design_payload["width"]), 1e-30)
        figsize = (6.0, 6.0 * float(design_payload["height"]) / width)
    fig, ax = _figure_axes(ax, figsize=figsize)

    for structure in design_payload["structures"]:
        _draw_polygon(ax, structure)
    for source in design_payload["sources"]:
        _draw_source(ax, source)
    for monitor in design_payload["monitors"]:
        _draw_monitor(ax, monitor)
    _draw_boundaries(ax, payload)

    ax.set_title(title)
    ax.set_xlim(*design_payload["xlim"])
    ax.set_ylim(*design_payload["ylim"])
    ax.set_aspect("equal")
    _configure_axes(ax, design_payload)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_grid(
    grid,
    *,
    field="permittivity",
    z_index=None,
    z_position=None,
    ax=None,
    figsize=None,
    cmap="Grays",
    show=True,
    colorbar=True,
    overlay=False,
):
    """Plot a rasterized grid field or 3D grid slice."""
    payload = grid.to_plot_data(field=field, z_index=z_index, z_position=z_position)
    design = payload["design"]
    if figsize is None:
        figsize = (6.0, 6.0 * design["height"] / max(design["width"], 1e-30))
    fig, ax = _figure_axes(ax, figsize=figsize)
    im = ax.imshow(
        payload["array"],
        origin="lower",
        cmap=cmap,
        extent=payload["extent"],
    )
    if colorbar:
        fig.colorbar(im, ax=ax, label=field)
    if overlay:
        for structure in grid.design.to_plot_data()["structures"]:
            overlay_structure = dict(structure)
            style = dict(overlay_structure["style"])
            style["facecolor"] = "none"
            style["edgecolor"] = "gray"
            style["alpha"] = 0.5
            overlay_structure["style"] = style
            _draw_polygon(ax, overlay_structure)
    ax.set_title("Rasterized Design Grid")
    scale_factor, scale_unit = get_si_scale_and_label(
        max(design["width"], design["height"], design.get("depth", 0.0))
    )
    _configure_axes(
        ax,
        {
            "scale_factor": scale_factor,
            "scale_unit": scale_unit,
        },
    )
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_signal(signals, t, *, ax=None, figsize=(9, 4), show=True, save_path=None):
    """Plot one or more time-domain source signals."""
    from beamz.visual.data import signal_plot_data

    payload = signal_plot_data(signals, t)
    fig, ax = _figure_axes(ax, figsize=figsize)
    for idx, values in enumerate(payload["signals"]):
        kwargs = {"label": f"Signal {idx}"} if len(payload["signals"]) > 1 else {}
        ax.plot(payload["t_scaled"], values, **kwargs)
    ax.set_xlim(*payload["xlim"])
    ax.set_xlabel(f"Time ({payload['time_unit']})")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal")
    if len(payload["signals"]) > 1:
        ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    _maybe_show(fig, show=show)
    return fig, ax


def plot_source_signal(source, *, t=None, ax=None, figsize=(9, 4), show=True):
    """Plot a source object's time dependence."""
    from beamz.visual.data import source_signal_plot_data

    payload = source_signal_plot_data(source, t=t)
    fig, ax = _figure_axes(ax, figsize=figsize)
    ax.plot(payload["t_scaled"], payload["signals"][0])
    ax.set_xlim(*payload["xlim"])
    ax.set_xlabel(f"Time ({payload['time_unit']})")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{payload['source_type']} Signal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_source_spectrum(
    source,
    *,
    t=None,
    dt=None,
    ax=None,
    figsize=(9, 4),
    show=True,
):
    """Plot a source object's normalized spectrum."""
    from beamz.visual.data import source_spectrum_plot_data

    payload = source_spectrum_plot_data(source, t=t, dt=dt)
    fig, ax = _figure_axes(ax, figsize=figsize)
    ax.plot(payload["frequency_scaled"], payload["amplitude"])
    ax.set_xlabel(f"Frequency ({payload['frequency_unit']})")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"{payload['source_type']} Spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_mode_profile(
    source,
    *,
    field=None,
    ax=None,
    figsize=(8, 6),
    show=True,
    save_path=None,
):
    """Plot a ModeSource profile."""
    payload = source.mode_profile_data(field=field)
    fig, ax = _figure_axes(ax, figsize=figsize)
    if payload["is_2d"]:
        im = ax.imshow(
            payload["amplitude"],
            origin="lower",
            cmap="magma",
            aspect="auto",
        )
        fig.colorbar(im, ax=ax, label="Absolute Amplitude")
        ax.set_title(
            f"Mode Source 2D Profile: {payload['title']} (neff={payload['neff']:.4f})"
        )
        if payload["direction"] in ["+x", "-x"]:
            ax.set_xlabel("Y-axis")
            ax.set_ylabel("Z-axis")
        else:
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Z-axis")
    else:
        ax.plot(payload["amplitude"], "k-")
        ax.set_title(
            f"Mode Source 1D Profile: {payload['title']} (neff={payload['neff']:.4f})"
        )
        ax.set_xlabel("Transverse Coordinate (cells)")
        ax.set_ylabel("Absolute Amplitude")
        ax.grid(True)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    _maybe_show(fig, show=show)
    return fig, ax


def plot_mode_permittivity(
    source,
    *,
    ax=None,
    figsize=(8, 6),
    cmap="viridis",
    show=True,
):
    """Plot the permittivity profile used by a ModeSource."""
    from beamz.visual.data import mode_permittivity_plot_data

    payload = mode_permittivity_plot_data(source)
    fig, ax = _figure_axes(ax, figsize=figsize)
    array = np.asarray(payload["array"])
    if array.ndim == 1:
        ax.plot(array, "k-")
        ax.set_xlabel("Coordinate index")
        ax.set_ylabel("Permittivity")
        ax.grid(True, alpha=0.3)
    else:
        im = ax.imshow(array, origin="lower", cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, label="Permittivity")
        ax.set_xlabel("Axis 1 index")
        ax.set_ylabel("Axis 2 index")
    ax.set_title(payload["title"])
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_monitor_field(
    monitor,
    *,
    field="Ez",
    time_index=-1,
    ax=None,
    figsize=(10, 6),
    cmap="RdBu",
    show=True,
):
    """Plot recorded field data from a monitor."""
    if hasattr(monitor, "field_plot_data"):
        payload = monitor.field_plot_data(field=field, time_index=time_index)
    else:
        from beamz.visual.data import monitor_field_plot_data

        payload = monitor_field_plot_data(monitor, field=field, time_index=time_index)
    fig, ax = _figure_axes(ax, figsize=figsize)
    if payload["monitor_type"] == "line":
        ax.plot(payload["x"], np.ravel(payload["array"]), "b-", linewidth=2)
        ax.set_xlabel(payload.get("xlabel", "Position along monitor line"))
        ax.set_ylabel(f"{field} amplitude")
        ax.grid(True, alpha=0.3)
    else:
        im = ax.imshow(
            payload["array"],
            cmap=resolve_cmap(cmap),
            origin="lower",
            aspect="auto",
            extent=payload.get("extent"),
        )
        fig.colorbar(im, ax=ax, label=f"{field} amplitude")
        ax.set_xlabel(payload.get("xlabel", "X index"))
        ax.set_ylabel(payload.get("ylabel", "Y index"))
    ax.set_title(payload["title"])
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_simulation_field(
    results,
    *,
    field="Ez",
    val="real",
    time_index=-1,
    t=None,
    frame=None,
    plane="z",
    index=None,
    method="nearest",
    ax=None,
    figsize=(8, 6),
    cmap="twilight_zero",
    cmap_limits="dynamic",
    vmin=None,
    vmax=None,
    colorbar=True,
    overlay_eps=True,
    eps_alpha=0.2,
    core_permittivity=None,
    overlay=True,
    overlay_color="gray",
    overlay_alpha=0.5,
    show=True,
):
    """Plot a stored simulation field frame from ``SimulationResults``."""
    from beamz.visual.data import simulation_field_plot_data

    payload = simulation_field_plot_data(
        results,
        field=field,
        time_index=time_index,
        t=t,
        frame=frame,
        plane=plane,
        index=index,
        method=method,
    )
    val_key = str(val).lower().replace(" ", "")
    is_power = val_key in {"abs^2", "abs2", "abs_sq", "abssq", "power"}
    arr = np.asarray(payload["array"])
    if val_key in {"real", "re"}:
        plot_arr = np.real(arr)
        label = f"Re({field})"
    elif val_key in {"imag", "imaginary", "im"}:
        plot_arr = np.imag(arr)
        label = f"Im({field})"
    elif val_key in {"abs", "magnitude"}:
        plot_arr = np.abs(arr)
        label = f"|{field}|"
    elif is_power:
        plot_arr = np.abs(arr) ** 2
        label = f"|{field}|^2"
    else:
        raise ValueError("val must be one of 'real', 'imag', 'abs', or 'abs^2'.")
    fig, ax = _figure_axes(ax, figsize=figsize)
    vmin, vmax = resolve_cmap_limits(cmap_limits, vmin=vmin, vmax=vmax)
    im = ax.imshow(
        plot_arr,
        origin="lower",
        cmap=resolve_cmap(cmap),
        extent=payload["extent"],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    if colorbar:
        fig.colorbar(im, ax=ax, label=label)
    if overlay_eps:
        plane_position = index if index is not None else None
        eps_slice = _field_eps_slice(
            results.simulation,
            plane=payload["plane"],
            plane_position=plane_position,
        )
        _draw_field_eps_overlay(
            ax,
            eps_slice,
            extent=payload["extent"],
            reverse=val_key in {"abs", "magnitude"} or is_power,
            alpha=eps_alpha,
            core_permittivity=core_permittivity,
        )
    if overlay and payload["plane"] == "xy":
        _draw_simulation_overlay(
            ax,
            results.simulation,
            scale=payload["scale_factor"],
            line_color=overlay_color,
            line_opacity=overlay_alpha,
        )
    ax.set_xlabel(payload["xlabel"])
    ax.set_ylabel(payload["ylabel"])
    ax.set_title(payload["title"])
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_simulation_permittivity(
    sim,
    *,
    plane="z",
    index=None,
    ax=None,
    figsize=(8, 6),
    cmap="viridis",
    colorbar=True,
    overlay=True,
    overlay_color="gray",
    overlay_alpha=0.5,
    show=True,
):
    """Plot a simulation permittivity slice."""
    from beamz.visual.data import simulation_permittivity_plot_data

    payload = simulation_permittivity_plot_data(sim, plane=plane, index=index)
    fig, ax = _figure_axes(ax, figsize=figsize)
    im = ax.imshow(
        payload["array"],
        origin="lower",
        cmap=resolve_cmap(cmap),
        extent=payload["extent"],
        aspect="auto",
    )
    if colorbar:
        fig.colorbar(im, ax=ax, label="Permittivity")
    if overlay and payload["plane"] == "xy":
        _draw_simulation_overlay(
            ax,
            sim,
            scale=payload["scale_factor"],
            line_color=overlay_color,
            line_opacity=overlay_alpha,
        )
    ax.set_xlabel(payload["xlabel"])
    ax.set_ylabel(payload["ylabel"])
    ax.set_title(payload["title"])
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_monitor_power(
    monitor,
    *,
    log_scale=False,
    db_scale=False,
    ax=None,
    figsize=(10, 6),
    show=True,
):
    """Plot monitor power history."""
    if hasattr(monitor, "power_plot_data"):
        payload = monitor.power_plot_data(log_scale=log_scale, db_scale=db_scale)
    else:
        from beamz.visual.data import monitor_power_plot_data

        payload = monitor_power_plot_data(
            monitor,
            log_scale=log_scale,
            db_scale=db_scale,
        )
    fig, ax = _figure_axes(ax, figsize=figsize)
    ax.plot(payload["time_steps"], payload["power"], "r-", linewidth=2)
    ax.set_yscale(payload["yscale"])
    ax.set_xlabel("Time step")
    ax.set_ylabel(payload["ylabel"])
    ax.set_title(payload["title"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def animate_monitor_fields(
    monitor,
    *,
    field="Ez",
    figsize=(8, 6),
    interval=100,
    save_filename=None,
    show=True,
):
    """Create a matplotlib animation from recorded monitor fields."""
    if (
        not monitor.fields["t"]
        or field not in monitor.fields
        or not monitor.fields[field]
    ):
        raise RuntimeError(f"No data available for field '{field}'.")

    plt = _pyplot()
    FuncAnimation = _mpl_types()["FuncAnimation"]
    fig, ax = plt.subplots(figsize=figsize)

    if monitor.monitor_type == "line":
        (line,) = ax.plot([], [], "b-", linewidth=2)
        ax.set_xlabel("Position along monitor line")
        ax.set_ylabel(f"{field} amplitude")
        all_data = np.concatenate([np.ravel(v) for v in monitor.fields[field]])
        ax.set_xlim(0, len(np.ravel(monitor.fields[field][0])))
        ax.set_ylim(np.min(all_data), np.max(all_data))

        def update(frame):
            field_data = np.ravel(monitor.fields[field][frame])
            line.set_data(range(field_data.size), field_data)
            ax.set_title(f"{field} at t = {monitor.fields['t'][frame]:.2e} s")
            return (line,)

        artists = True
    else:
        field_data = monitor.fields[field][0]
        im = ax.imshow(
            field_data, cmap="RdBu", origin="lower", aspect="auto", animated=True
        )
        fig.colorbar(im, ax=ax, label=f"{field} amplitude")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
        all_data = np.asarray(monitor.fields[field])
        im.set_clim(np.min(all_data), np.max(all_data))

        def update(frame):
            im.set_array(monitor.fields[field][frame])
            ax.set_title(f"{field} at t = {monitor.fields['t'][frame]:.2e} s")
            return (im,)

        artists = True

    anim = FuncAnimation(
        fig,
        update,
        frames=len(monitor.fields["t"]),
        interval=interval,
        blit=artists,
        repeat=True,
    )
    if save_filename:
        anim.save(save_filename, writer="pillow", fps=max(1, 1000 // interval))
    _maybe_show(fig, show=show)
    return anim


def _draw_scale_bar(ax, design_payload, wavelength=None, fontsize=10):
    width = design_payload["width"]
    height = design_payload["height"]
    scale_factor = design_payload["scale_factor"]
    unit = design_payload["scale_unit"]

    if wavelength is not None:
        scale_bar_length_um = np.round(2 * wavelength * 1e6)
        scale_bar_length = scale_bar_length_um * 1e-6
        label_text = f"{int(scale_bar_length_um)} µm"
    else:
        min_dim = min(width, height)
        scale_bar_length_physical = min_dim * 0.18
        if scale_bar_length_physical > 0:
            order = 10 ** np.floor(np.log10(scale_bar_length_physical))
            normalized = scale_bar_length_physical / order
            if normalized <= 1.25:
                nice_value = 1 * order
            elif normalized <= 2.5:
                nice_value = 2 * order
            elif normalized <= 6:
                nice_value = 5 * order
            else:
                nice_value = 10 * order
            scale_bar_length = nice_value
        else:
            scale_bar_length = min_dim * 0.15

        display_value = scale_bar_length * scale_factor
        if display_value >= 1:
            label_text = f"{display_value:.0f} {unit}"
        elif display_value >= 0.1:
            label_text = f"{display_value:.1f} {unit}"
        else:
            label_text = f"{display_value:.2f} {unit}"

    margin_x = width * 0.1
    margin_y = height * 0.1
    x_start = width - scale_bar_length - margin_x
    x_end = width - margin_x
    y_pos = margin_y
    ax.plot([x_start, x_end], [y_pos, y_pos], "w", linewidth=3, solid_capstyle="butt")
    ax.text(
        (x_start + x_end) / 2,
        y_pos - height * 0.02,
        label_text,
        ha="center",
        va="top",
        color="white",
        fontsize=fontsize,
    )


def _snapshot_color_limits(snapshots):
    if not snapshots:
        return None, None

    data_min = np.inf
    data_max = -np.inf
    has_pos = False
    has_neg = False
    for snapshot in snapshots:
        field = np.asarray(snapshot["field"], dtype=np.float64)
        finite = field[np.isfinite(field)]
        if finite.size == 0:
            continue
        data_min = min(data_min, float(np.min(finite)))
        data_max = max(data_max, float(np.max(finite)))
        has_pos = has_pos or bool(np.any(finite > 0.0))
        has_neg = has_neg or bool(np.any(finite < 0.0))

    if not np.isfinite(data_min) or not np.isfinite(data_max):
        return None, None
    if has_pos and has_neg:
        vmax = max(abs(data_min), abs(data_max), 1e-12)
        return -vmax, vmax
    if data_min == data_max:
        pad = max(abs(data_max) * 1e-12, 1e-12)
        return data_min - pad, data_max + pad
    return data_min, data_max


def _snapshot_figsize(snapshot, *, clean_visualization, base_long_edge=10.0):
    if not clean_visualization:
        return (10.0, 8.0)
    extent = snapshot["extent"]
    width = max(float(extent[1]) - float(extent[0]), 1e-12)
    height = max(float(extent[3]) - float(extent[2]), 1e-12)
    if width >= height:
        return (base_long_edge, base_long_edge * (height / width))
    return (base_long_edge * (width / height), base_long_edge)


def snapshot_figure(
    snapshot,
    *,
    cmap="twilight_zero",
    clean_visualization=False,
    interpolation="bicubic",
    figure=None,
    axes=None,
    vmin=None,
    vmax=None,
):
    """Render a simulation snapshot payload."""
    plt = _pyplot()
    layout = snapshot["layout"]
    design_payload = layout["design"]
    field = snapshot["field"]
    extent = snapshot["extent"]
    figsize = _snapshot_figsize(snapshot, clean_visualization=clean_visualization)

    if figure is None or axes is None:
        if clean_visualization:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = figure
        fig.clear()
        fig.set_size_inches(*figsize, forward=True)
        ax = fig.add_axes([0, 0, 1, 1]) if clean_visualization else fig.add_subplot(111)

    im = ax.imshow(
        field,
        origin="lower",
        cmap=resolve_cmap(cmap),
        extent=extent,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(float(extent[0]), float(extent[1]))
    ax.set_ylim(float(extent[2]), float(extent[3]))
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.0)

    for structure in design_payload["structures"]:
        style = dict(structure["style"])
        if not structure.get("is_pml"):
            style["facecolor"] = "none"
            style["edgecolor"] = "gray"
            style["alpha"] = 0.5
        overlay = dict(structure)
        overlay["style"] = style
        _draw_polygon(ax, overlay)
    for source in design_payload["sources"]:
        _draw_source(ax, source)
    for monitor in design_payload["monitors"]:
        overlay = dict(monitor)
        style = dict(overlay["style"])
        style["edgecolor"] = "gray"
        style["alpha"] = 0.5
        overlay["style"] = style
        _draw_monitor(ax, overlay)
    _draw_boundaries(ax, layout)

    if clean_visualization:
        ax.set_axis_off()
        _draw_scale_bar(ax, design_payload)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    else:
        fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            label=f"{snapshot['field_name']} ({snapshot['units']})",
        )
        ax.set_title(
            f"{snapshot['field_name']} at t = {snapshot['time']:.2e} s "
            f"(step {snapshot['step']}/{snapshot['num_steps']})"
        )
        _configure_axes(ax, design_payload)
        fig.tight_layout()
    return fig, ax


def show_snapshots(
    snapshots,
    *,
    field=None,
    frame=None,
    time_index=None,
    cmap="twilight_zero",
    cmap_limits="dynamic",
    clean_visualization=False,
    interpolation="bicubic",
    pause=0.001,
    show=True,
    vmin=None,
    vmax=None,
):
    """Show a sequence of stored snapshot payloads."""
    if not snapshots:
        return None, None
    selected_snapshots = tuple(snapshots)
    if field is not None:
        available_fields = {str(snapshot.get("field_name")) for snapshot in snapshots}
        if str(field) not in available_fields:
            raise ValueError(
                f"Snapshot field {field!r} is not available. "
                f"Available snapshot fields: {sorted(available_fields)}"
            )
        selected_snapshots = tuple(
            snapshot
            for snapshot in selected_snapshots
            if str(snapshot.get("field_name")) == str(field)
        )
    if frame is not None and time_index is not None:
        raise ValueError("Use either frame or time_index, not both.")
    selected_frame = frame if frame is not None else time_index
    if selected_frame is not None:
        selected_snapshots = (selected_snapshots[int(selected_frame)],)
    plt = _pyplot()
    context = {"fig": None, "ax": None}
    vmin, vmax = resolve_cmap_limits(cmap_limits, vmin=vmin, vmax=vmax)
    for snapshot in selected_snapshots:
        fig, ax = snapshot_figure(
            snapshot,
            cmap=cmap,
            clean_visualization=clean_visualization,
            interpolation=interpolation,
            figure=context["fig"],
            axes=context["ax"],
            vmin=vmin,
            vmax=vmax,
        )
        context["fig"], context["ax"] = fig, ax
        if show:
            plt.show(block=False)
            plt.pause(pause)
    return context["fig"], context["ax"]


def save_snapshot_video(
    snapshots,
    *,
    filename,
    fps=30,
    dpi=150,
    cmap="twilight_zero",
    cmap_limits=None,
    clean_visualization=False,
    interpolation="bicubic",
    vmin=None,
    vmax=None,
):
    """Save stored simulation snapshots to a video file."""
    if not snapshots:
        return None

    plt = _pyplot()
    FFMpegWriter = _mpl_types()["FFMpegWriter"]
    output = Path(filename)
    fig, ax = plt.subplots(
        figsize=_snapshot_figsize(snapshots[0], clean_visualization=clean_visualization)
    )
    if cmap_limits is None and vmin is None and vmax is None:
        vmin, vmax = _snapshot_color_limits(snapshots)
    else:
        vmin, vmax = resolve_cmap_limits(cmap_limits, vmin=vmin, vmax=vmax)
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    with writer.saving(fig, str(output), dpi=dpi):
        for snapshot in snapshots:
            snapshot_figure(
                snapshot,
                cmap=cmap,
                clean_visualization=clean_visualization,
                interpolation=interpolation,
                figure=fig,
                axes=ax,
                vmin=vmin,
                vmax=vmax,
            )
            writer.grab_frame()
    plt.close(fig)
    return output


def _field_video_frame_count(results, field):
    if results.fields is None or field not in results.fields:
        available = [] if results.fields is None else sorted(results.fields)
        raise RuntimeError(
            f"Field '{field}' is not stored. Available stored fields: {available}"
        )
    source = results.fields[field]
    if hasattr(source, "dims") and hasattr(source, "sizes"):
        if "t" in source.dims:
            return int(source.sizes["t"])
        if "frame" in source.dims:
            return int(source.sizes["frame"])
    arr = np.asarray(source)
    if arr.ndim in {3, 4}:
        return int(arr.shape[0])
    return 1


def _field_video_global_max_limits(
    results,
    *,
    field,
    frame_count,
    plane,
    index,
    method,
):
    from beamz.visual.data import simulation_field_plot_data

    max_abs = 0.0
    for frame_idx in range(frame_count):
        payload = simulation_field_plot_data(
            results,
            field=field,
            frame=frame_idx,
            plane=plane,
            index=index,
            method=method,
        )
        frame_max = float(np.nanmax(np.abs(np.asarray(payload["array"]))))
        if np.isfinite(frame_max):
            max_abs = max(max_abs, frame_max)
    if max_abs <= 0.0:
        max_abs = 1.0
    return -max_abs, max_abs


def save_field_video(
    results,
    *,
    filename,
    field="Ez",
    fps=30,
    dpi=150,
    cmap="twilight_zero",
    cmap_limits="dynamic",
    clean_visualization=False,
    interpolation="bicubic",
    vmin=None,
    vmax=None,
    plane="z",
    index=None,
    method="nearest",
    overlay=True,
    overlay_color="gray",
    overlay_alpha=0.5,
    colorbar=True,
):
    """Save stored ``SimulationResults`` field frames to a video file."""
    from beamz.visual.data import simulation_field_plot_data

    frame_count = _field_video_frame_count(results, field)
    if frame_count <= 0:
        raise RuntimeError(f"No frames stored for field '{field}'.")

    plt = _pyplot()
    FFMpegWriter = _mpl_types()["FFMpegWriter"]
    output = Path(filename)
    first_payload = simulation_field_plot_data(
        results,
        field=field,
        frame=0,
        plane=plane,
        index=index,
        method=method,
    )
    figsize = _snapshot_figsize(
        {"extent": first_payload["extent"]},
        clean_visualization=clean_visualization,
    )
    if clean_visualization:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
    else:
        fig, ax = plt.subplots(figsize=figsize)
    if isinstance(cmap_limits, str) and cmap_limits.lower() in {
        "max",
        "global",
        "global_max",
        "field_max",
    }:
        if vmin is not None or vmax is not None:
            raise ValueError("Use either cmap_limits='max' or vmin/vmax, not both.")
        vmin, vmax = _field_video_global_max_limits(
            results,
            field=field,
            frame_count=frame_count,
            plane=plane,
            index=index,
            method=method,
        )
    else:
        vmin, vmax = resolve_cmap_limits(cmap_limits, vmin=vmin, vmax=vmax)
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    with writer.saving(fig, str(output), dpi=dpi):
        for frame_idx in range(frame_count):
            payload = (
                first_payload
                if frame_idx == 0
                else simulation_field_plot_data(
                    results,
                    field=field,
                    frame=frame_idx,
                    plane=plane,
                    index=index,
                    method=method,
                )
            )
            ax.clear()
            im = ax.imshow(
                payload["array"],
                origin="lower",
                cmap=resolve_cmap(cmap),
                extent=payload["extent"],
                interpolation=interpolation,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlim(float(payload["extent"][0]), float(payload["extent"][1]))
            ax.set_ylim(float(payload["extent"][2]), float(payload["extent"][3]))
            ax.set_aspect("equal", adjustable="box")
            ax.margins(0.0)
            if colorbar and not clean_visualization:
                if frame_idx == 0:
                    fig.colorbar(im, ax=ax, label=f"{field} amplitude")
            if overlay and payload["plane"] == "xy":
                _draw_simulation_overlay(
                    ax,
                    results.simulation,
                    scale=payload["scale_factor"],
                    line_color=overlay_color,
                    line_opacity=overlay_alpha,
                )
            ax.set_xlabel(payload["xlabel"])
            ax.set_ylabel(payload["ylabel"])
            ax.set_title(payload["title"])
            if clean_visualization:
                ax.set_axis_off()
                fig.subplots_adjust(
                    left=0,
                    right=1,
                    top=1,
                    bottom=0,
                    wspace=0,
                    hspace=0,
                )
            else:
                fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)
    return output


def _modal_dft_x_values(frequencies, x_axis):
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    axis = str(x_axis).lower()
    if axis in {"wavelength", "wavelength_um", "lambda", "lambda_um"}:
        return LIGHT_SPEED / freqs / µm, "Wavelength (um)", True
    if axis in {"frequency", "freq", "hz"}:
        return freqs, "Frequency (Hz)", False
    if axis in {"frequency_thz", "freq_thz", "thz"}:
        return freqs / 1e12, "Frequency (THz)", False
    raise ValueError("x_axis must be 'wavelength', 'frequency', or 'frequency_thz'.")


def _modal_dft_port_sort_key(name):
    text = str(name)
    if "_m" in text:
        prefix, suffix = text.rsplit("_m", 1)
        try:
            return prefix, int(suffix)
        except ValueError:
            return text, 0
    return text, 0


def plot_modal_dft_diagnostics(
    result,
    *,
    source_port=None,
    output_ports=None,
    x_axis="wavelength",
    include_sum=True,
    title="Mode decomposition",
    residual_port=None,
    show=True,
):
    """Plot modal DFT S-parameter powers with projection diagnostics.

    Parameters
    ----------
    result : dict
        Return value from ``Simulation.get_S_matrix_modal_dft(..., as_sax=False,
        return_diagnostics=True)``.
    source_port : str, optional
        Source port name. Defaults to the source recorded in diagnostics.
    output_ports : sequence[str], optional
        Output modal port names. Defaults to diagnostics output ports.
    x_axis : {"wavelength", "frequency", "frequency_thz"}
        Horizontal axis.
    include_sum : bool
        Plot the sum of all selected output ports on the power axis.
    title : str
        Title for the modal-power subplot.
    residual_port : str, optional
        Port whose projection residual and rejected branch are shown. Defaults
        to the first output port.
    show : bool
        Call ``plt.show()`` before returning.

    Returns
    -------
    fig, axes
        Matplotlib figure and the two axes.
    """
    plt = _pyplot()
    diagnostics = result.get("diagnostics", {})
    s_matrix = result.get("s_matrix", {})
    frequencies = np.asarray(diagnostics.get("frequencies", []), dtype=float)
    if frequencies.size == 0:
        raise ValueError("Modal DFT diagnostics are missing the frequency axis.")
    source = source_port or diagnostics.get("source_port")
    if source is None:
        raise ValueError("source_port is required when diagnostics omit it.")
    ports = list(output_ports or diagnostics.get("output_ports", ()))
    if not ports:
        ports = [
            key[0] for key in s_matrix if isinstance(key, tuple) and key[1] == source
        ]
    ports = sorted(ports, key=_modal_dft_port_sort_key)
    if not ports:
        raise ValueError("No output ports available to plot.")

    x_values, x_label, invert_x = _modal_dft_x_values(frequencies, x_axis)
    powers = []
    for port in ports:
        key = (port, source)
        if key not in s_matrix:
            continue
        values = np.asarray(s_matrix[key], dtype=np.complex128)
        if values.size == frequencies.size:
            powers.append((port, np.abs(values) ** 2))
    if not powers:
        raise ValueError("No modal S-matrix powers matched the requested ports.")

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)
    ax_power, ax_diag = axes
    colors = ["#ff7f0e", "#8d99ae", "#ff4f81", "#9467bd", "#17becf", "#bcbd22"]
    if include_sum and len(powers) > 1:
        total = np.sum([power for _, power in powers], axis=0)
        ax_power.plot(x_values, total, color="#1b7837", label="Mode sum")
    for idx, (port, power) in enumerate(powers):
        _, mode_index = _modal_dft_port_sort_key(port)
        ax_power.plot(
            x_values,
            power,
            color=colors[idx % len(colors)],
            label=f"Mode {mode_index}",
        )
    ax_power.set_ylabel("Power in mode (W)")
    ax_power.set_title(title)
    ax_power.legend()

    diag_port = residual_port or powers[0][0]
    wave = diagnostics.get("waves", {}).get(diag_port, {})
    residual = np.asarray(wave.get("projection_residual", []), dtype=float)
    if residual.size == frequencies.size:
        ax_diag.plot(x_values, residual, color="#4c78a8", label="Projection residual")
    p_in = np.asarray(diagnostics.get("P_in", []), dtype=float)
    checks = diagnostics.get("monitor_flux_checks", {}).get(diag_port, {})
    rejected = np.asarray(checks.get("P_rejected", []), dtype=float)
    if rejected.size == frequencies.size and p_in.size == frequencies.size:
        rejected_norm = rejected / np.maximum(p_in, 1e-18)
        ax_diag.plot(x_values, rejected_norm, color="#f58518", label="Rejected branch")
    neff = np.asarray(wave.get("mode_neff", []), dtype=float)
    if neff.size == frequencies.size and np.nanmax(neff) > np.nanmin(neff):
        ax_neff = ax_diag.twinx()
        ax_neff.plot(x_values, neff, color="#54a24b", alpha=0.7, label="n_eff")
        ax_neff.set_ylabel("n_eff")
    ax_diag.set_xlabel(x_label)
    ax_diag.set_ylabel("Diagnostic value")
    ax_diag.legend(loc="best")
    if invert_x:
        ax_diag.set_xlim(float(np.nanmax(x_values)), float(np.nanmin(x_values)))
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


__all__ = [
    "animate_monitor_fields",
    "get_twilight_zero_cmap",
    "mode_field_component_pairs",
    "plot_modal_dft_diagnostics",
    "plot_design",
    "plot_grid",
    "plot_mode_fields",
    "plot_mode_permittivity",
    "plot_mode_profile",
    "plot_monitor_field",
    "plot_monitor_power",
    "plot_simulation_field",
    "plot_simulation_permittivity",
    "plot_signal",
    "plot_simulation",
    "plot_source_signal",
    "plot_source_spectrum",
    "plot_tidy3d_cross_sections",
    "plot_tidy3d_dft_field",
    "plot_tidy3d_field_frame",
    "plot_tidy3d_mode_components",
    "resolve_cmap_limits",
    "resolve_cmap",
    "save_field_video",
    "save_snapshot_video",
    "show_snapshots",
    "snapshot_figure",
]
