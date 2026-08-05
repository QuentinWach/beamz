"""Small matplotlib plotting helpers used by examples and notebooks."""

from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from beamz._helpers import get_si_scale_and_label
from beamz.analysis._coordinates import monitor_plane_coordinates_3d
from beamz.analysis.data import AnalysisData, analysis_inputs, static_fields
from beamz.devices.visualization import visual_spec_from_device
from beamz.simulation.observe import source_field_amplitude_normalization

_UM = 1e-6
_AXIS_INDEX: dict[str, int] = {"z": 0, "y": 1, "x": 2}
_PLANE_AXES: dict[str, tuple[str, str]] = {
    "x": ("z", "y"),
    "y": ("z", "x"),
    "z": ("y", "x"),
}
_DESIGN_EXTENT: dict[str, str] = {
    "x": "width",
    "y": "height",
    "z": "depth",
}


@dataclass(frozen=True, slots=True)
class AxisSlice:
    """An axis-aligned 2D array plus its plotting coordinates."""

    values: np.ndarray
    normal: str
    vertical: str
    horizontal: str
    index: int
    position: float
    extent: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FieldView:
    """Scalar display data derived from a scalar or complex field."""

    values: np.ndarray
    kind: str
    magnitude: bool
    power: bool


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _figure_axes(ax, *, figsize):
    if ax is not None:
        return ax.figure, ax
    return _pyplot().subplots(figsize=figsize)


def _maybe_show(fig, *, show):
    if show:
        _pyplot().show()
    return fig


def _axis_labels(ax, *, width, height, x_label="x", y_label="y"):
    scale, unit = get_si_scale_and_label(max(float(width), float(height), 1e-30))
    ax.set_xlabel(f"{x_label} ({unit})")
    ax.set_ylabel(f"{y_label} ({unit})")
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value * scale:g}")
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value * scale:g}")


def _material_color(structure, index):
    explicit = getattr(structure, "color", None)
    if explicit:
        return explicit
    material = getattr(structure, "material", None)
    key = repr(
        (
            getattr(material, "permittivity", None),
            getattr(material, "permeability", None),
            getattr(material, "conductivity", None),
            index,
        )
    )
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    hue = (seed % 1000) / 1000.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.45, 0.78)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _vertices_2d(vertices):
    return [(float(v[0]), float(v[1])) for v in vertices or ()]


def _draw_polygon(ax, structure, *, index=0, fill=True, alpha=None, edgecolor="black"):
    vertices = _vertices_2d(getattr(structure, "vertices", ()))
    if len(vertices) < 3:
        return None

    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    coords = []
    codes = []
    for path in [
        vertices,
        *[_vertices_2d(p) for p in getattr(structure, "interiors", ())],
    ]:
        if len(path) < 3:
            continue
        coords.append(path[0])
        codes.append(Path.MOVETO)
        coords.extend(path[1:])
        codes.extend([Path.LINETO] * (len(path) - 1))
        coords.append(path[0])
        codes.append(Path.CLOSEPOLY)

    facecolor = _material_color(structure, index) if fill else "none"
    patch = PathPatch(
        Path(np.asarray(coords), np.asarray(codes)),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        alpha=0.22
        if alpha is None and index == 0
        else (0.65 if alpha is None else alpha),
    )
    ax.add_patch(patch)
    return patch


def draw_geometry(ax, design, *, sources=(), monitors=()):
    for index, structure in enumerate(getattr(design, "structures", ())):
        _draw_polygon(ax, structure, index=index)
    for source in sources or ():
        _draw_source(ax, source)
    for monitor in monitors or ():
        _draw_monitor(ax, monitor)


def _draw_source(ax, source):
    spec = visual_spec_from_device(source)
    if spec is None:
        return None
    center = tuple(float(v) for v in spec.center[:2])
    direction = str(spec.direction or "").lower()
    size = np.asarray(spec.size, dtype=float).reshape(-1)
    width = float(np.max(np.abs(size[:2]))) if size.size else 0.25e-6
    color = "crimson"
    if direction in {"+x", "-x"}:
        x = [center[0], center[0]]
        y = [center[1] - 0.5 * width, center[1] + 0.5 * width]
    elif direction in {"+y", "-y"}:
        x = [center[0] - 0.5 * width, center[0] + 0.5 * width]
        y = [center[1], center[1]]
    else:
        from matplotlib.patches import Circle

        circle = Circle(
            (center[0], center[1]),
            radius=0.5 * width,
            fill=False,
            ec=color,
            lw=1.5,
        )
        ax.add_patch(circle)
        return circle
    (line,) = ax.plot(x, y, color=color, linewidth=2.0, solid_capstyle="round")
    return line


def _monitor_rect(spec):
    center = np.asarray(spec.center, dtype=float).reshape(-1)
    size = np.asarray(spec.size, dtype=float).reshape(-1)
    if center.size < 2 or size.size < 2:
        return None
    return (
        float(center[0] - 0.5 * abs(size[0])),
        float(center[1] - 0.5 * abs(size[1])),
        float(abs(size[0])),
        float(abs(size[1])),
    )


def _draw_monitor(ax, monitor):
    spec = visual_spec_from_device(monitor)
    if spec is None:
        return None
    if spec.kind == "monitor-line":
        style = spec.style or {}
        start = style.get("start")
        end = style.get("end")
        if start is None or end is None:
            return None
        (line,) = ax.plot(
            [float(start[0]), float(end[0])],
            [float(start[1]), float(end[1])],
            color="navy",
            linewidth=2.0,
        )
        return line

    rect = _monitor_rect(spec)
    if rect is None:
        return None
    from matplotlib.patches import Rectangle

    patch = Rectangle(
        rect[:2],
        rect[2],
        rect[3],
        facecolor="none",
        edgecolor="navy",
        linestyle="--",
        linewidth=1.5,
    )
    ax.add_patch(patch)
    return patch


def _draw_boundaries(ax, sim):
    from matplotlib.patches import Rectangle

    design = sim.design
    width, height = float(design.width), float(design.height)
    for boundary in getattr(sim, "boundaries", ()) or ():
        thickness = float(getattr(boundary, "thickness", 0.0) or 0.0)
        if thickness <= 0:
            continue
        edges = getattr(boundary, "_get_edges_for_dimensionality", lambda _is_3d: ())(
            bool(getattr(design, "is_3d", False))
        )
        rectangles = {
            "left": (0.0, 0.0, thickness, height),
            "right": (width - thickness, 0.0, thickness, height),
            "bottom": (0.0, 0.0, width, thickness),
            "top": (0.0, height - thickness, width, thickness),
        }
        for edge in edges:
            if edge not in rectangles:
                continue
            ax.add_patch(
                Rectangle(
                    rectangles[edge][:2],
                    *rectangles[edge][2:],
                    facecolor="none",
                    edgecolor="red",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.6,
                )
            )


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
    if figsize is None:
        width = max(float(design.width), 1e-30)
        figsize = (6.0, max(2.0, 6.0 * float(design.height) / width))
    fig, ax = _figure_axes(ax, figsize=figsize)
    draw_geometry(ax, design, sources=sources or (), monitors=monitors or ())
    ax.set_xlim(0.0, float(design.width))
    ax.set_ylim(0.0, float(design.height))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    _axis_labels(ax, width=design.width, height=design.height)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def _index_from_position(position, *, step, count, default):
    if position is None:
        return int(default)
    idx = int(round(float(position) / max(float(step), 1e-30)))
    return int(np.clip(idx, 0, int(count) - 1))


def extract_axis_aligned_slice(
    values,
    *,
    axis,
    step,
    position=None,
    index=None,
    origin=(0.0, 0.0, 0.0),
    display_position=False,
    lengths=None,
    unit=1.0,
) -> AxisSlice:
    """Extract a 3D array plane and attach consistent display coordinates."""
    array = np.asarray(values)
    normal = str(axis).lower()
    if array.ndim != 3 or normal not in _PLANE_AXES:
        raise ValueError("Axis-aligned slices require a 3D array and x, y, or z axis.")
    axis_index = _AXIS_INDEX[normal]
    origin_by_axis: dict[str, float] = dict(
        zip(("x", "y", "z"), map(float, origin), strict=True)
    )
    absolute_position = position
    if position is not None and display_position:
        absolute_position = float(position) + origin_by_axis[normal]
    if index is None:
        index = _index_from_position(
            absolute_position,
            step=step,
            count=array.shape[axis_index],
            default=array.shape[axis_index] // 2,
        )
    index = int(np.clip(index, 0, array.shape[axis_index] - 1))
    vertical, horizontal = _PLANE_AXES[normal]
    if lengths is None:
        lengths = {
            name: array.shape[_AXIS_INDEX[name]] * float(step)
            for name in ("x", "y", "z")
        }
    extent = (
        (0.0 - origin_by_axis[horizontal]) / unit,
        (float(lengths[horizontal]) - origin_by_axis[horizontal]) / unit,
        (0.0 - origin_by_axis[vertical]) / unit,
        (float(lengths[vertical]) - origin_by_axis[vertical]) / unit,
    )
    shown_position = (
        float(position)
        if position is not None and display_position
        else index * float(step) - origin_by_axis[normal]
    )
    return AxisSlice(
        values=np.take(array, index, axis=axis_index),
        normal=normal,
        vertical=vertical,
        horizontal=horizontal,
        index=index,
        position=shown_position,
        extent=extent,
    )


def _field_view(values, *, val="real", vector=False) -> FieldView:
    """Reduce scalar or vector complex data to one plottable scalar array."""
    key = str(val or "real").lower().replace(" ", "")
    power = key in {"abs^2", "abs2", "abs_sq", "abssq", "power", "intensity"}
    magnitude = key in {"abs", "magnitude"} or power
    arrays = [np.asarray(value) for value in values] if vector else [np.asarray(values)]
    if vector:
        if not magnitude:
            raise ValueError("Derived vector fields support val='abs' or val='abs^2'.")
        reduced = np.sum([np.abs(array) ** 2 for array in arrays], axis=0)
        return FieldView(reduced if power else np.sqrt(reduced), key, True, power)
    array = arrays[0]
    transforms = {
        "real": np.real,
        "re": np.real,
        "imag": np.imag,
        "imaginary": np.imag,
        "im": np.imag,
        "abs": np.abs,
        "magnitude": np.abs,
        "phase": np.angle,
    }
    if power:
        reduced = np.abs(array) ** 2
    elif key in transforms:
        reduced = transforms[key](array)
    else:
        raise ValueError(
            "val must be one of 'real', 'imag', 'abs', 'abs^2', or 'phase'."
        )
    return FieldView(np.asarray(reduced), key, magnitude, power)


def plot_field_view(
    ax,
    values,
    *,
    val=None,
    vector=False,
    extent=None,
    cmap: Any = "RdBu",
    vmin=None,
    vmax=None,
    percentile=None,
    aspect=None,
    interpolation="nearest",
    norm=None,
    x_edges=None,
    y_edges=None,
):
    """Plot scalar or complex field data through the shared image primitive."""
    view = (
        values
        if isinstance(values, FieldView)
        else (
            _field_view(values, val="real" if val is None else val, vector=vector)
            if val is not None or vector or np.iscomplexobj(values)
            else FieldView(np.asarray(values), "scalar", False, False)
        )
    )
    array = view.values
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if percentile is not None and vmax is None:
        scale = float(np.nanpercentile(np.abs(array), float(percentile)))
        vmax = scale if np.isfinite(scale) and scale > 0.0 else 1.0
    if vmax is not None and vmin is None:
        vmin = 0.0 if view.magnitude else -float(vmax)
    if (x_edges is None) != (y_edges is None):
        raise ValueError("x_edges and y_edges must be provided together.")
    if x_edges is not None and y_edges is not None:
        x_edges = np.asarray(x_edges, dtype=float)
        y_edges = np.asarray(y_edges, dtype=float)
        if x_edges.shape != (array.shape[1] + 1,) or y_edges.shape != (
            array.shape[0] + 1,
        ):
            raise ValueError(
                "Rectilinear plot edges must have one more entry than their "
                f"field axis; got field {array.shape}, x {x_edges.shape}, "
                f"and y {y_edges.shape}."
            )
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            array,
            shading="flat",
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect(aspect or ("auto" if array.shape[0] == 1 else "equal"))
    else:
        image = ax.imshow(
            array,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            aspect=aspect or ("auto" if array.shape[0] == 1 else "equal"),
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )
    return image, view


def _simulation_origin(sim):
    offset = getattr(sim, "coordinate_offset", None)
    if offset is not None:
        values = tuple(float(v) for v in offset)
        if len(values) == 3:
            return values
    return (0.0, 0.0, 0.0)


def _material_category_array(
    eps, *, core_permittivity=None, substrate_permittivity=None
):
    eps = np.real(np.asarray(eps))
    finite = eps[np.isfinite(eps)]
    if finite.size == 0:
        return np.zeros(eps.shape, dtype=int)
    eps_min = float(np.nanmin(finite))
    eps_max = float(np.nanmax(finite))
    if np.isclose(eps_max, eps_min):
        return np.zeros(eps.shape, dtype=int)
    if substrate_permittivity is None:
        unique = np.unique(np.round(finite, decimals=8))
        substrate_permittivity = float(unique[1]) if unique.size >= 3 else eps_min
    if core_permittivity is None:
        core_permittivity = eps_max

    sub_mid = 0.5 * (eps_min + float(substrate_permittivity))
    core_mid = 0.5 * (float(substrate_permittivity) + float(core_permittivity))
    return np.where(eps >= core_mid, 2, np.where(eps >= sub_mid, 1, 0))


def _material_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#f5f5f5", "#83abc0", "#d86c96"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    return cmap, norm


def _field_permittivity(sim):
    if isinstance(sim, AnalysisData):
        return (
            np.asarray(())
            if sim.materials is None
            else np.asarray(sim.materials.permittivity)
        )
    fields = getattr(sim, "fields", None)
    materials = getattr(fields, "materials", None)
    value = (
        getattr(materials, "permittivity", None)
        if materials is not None
        else getattr(fields, "permittivity", None)
    )
    return np.asarray(()) if value is None else np.asarray(value)


def _material_levels(sim):
    eps = _field_permittivity(sim)
    finite = np.real(eps[np.isfinite(eps)]) if eps.size else np.asarray(())
    if finite.size == 0:
        return None, None
    unique = np.unique(np.round(finite, decimals=8))
    substrate = float(unique[1]) if unique.size >= 3 else None
    core = float(unique[-1])
    return core, substrate


def _pml_thickness(sim):
    for boundary in getattr(sim, "boundaries", ()) or ():
        thickness = getattr(boundary, "thickness", None)
        if thickness is not None:
            return float(thickness)
    return None


def _field_eps_slice(simulation, *, plane="z", index=None, plane_position=None):
    eps = _field_permittivity(simulation)
    if eps.size == 0:
        return None
    try:
        normal = {"xy": "z", "xz": "y", "yz": "x"}.get(
            str(plane).lower(), str(plane).lower()
        )
        return extract_axis_aligned_slice(
            eps,
            axis=normal,
            step=float(simulation.resolution),
            position=plane_position,
            index=index,
        ).values
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
    x_edges=None,
    y_edges=None,
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
    if x_edges is not None and y_edges is not None:
        overlay = ax.pcolormesh(
            x_edges,
            y_edges,
            rgba,
            shading="flat",
        )
        ax.set_aspect("equal")
        return overlay
    return ax.imshow(
        rgba,
        origin="lower",
        extent=(extent[0], extent[1], extent[2], extent[3]),
        aspect="equal",
        interpolation="nearest",
    )


def _field_display_scale_and_units(field):
    family = str(field)[:1].upper()
    if family == "E":
        return _UM, "V/um"
    if family == "H":
        return _UM, "A/um"
    return 1.0, ""


def _device_geometry(device):
    """Return normalized axis, center, size, and direction for an overlay device."""
    spec = visual_spec_from_device(device)
    if spec is None or len(spec.center) < 3:
        return None
    center, size = np.asarray(spec.center), np.asarray(spec.size)
    direction = str(spec.direction or "").lower()
    axis = next((name for name in _PLANE_AXES if direction.endswith(name)), None)
    if axis is None and size.size == 3 and np.any(np.isclose(size, 0.0)):
        axis = ("x", "y", "z")[int(np.argmin(np.abs(size)))]
    if axis is None:
        return None
    if size.size == 2:
        size = {"x": (0.0, *size), "y": (size[0], 0.0, size[1]), "z": (*size, 0.0)}[
            axis
        ]
    return axis, center.astype(float), np.asarray(size, dtype=float), direction


def _draw_device_slice_overlay(ax, device, *, normal, origin, color, source):
    geometry = _device_geometry(device)
    if geometry is None:
        return
    axis, center, size, direction = geometry
    vertical, horizontal = _PLANE_AXES[normal]
    if axis == horizontal:
        segments = [(False, vertical, horizontal, True)]
    elif axis == vertical:
        segments = [(True, horizontal, vertical, True)]
    elif normal == "z":
        segments = [
            (True, horizontal, vertical, True),
            (False, vertical, horizontal, False),
        ]
    else:
        return
    origin_by_axis: dict[str, float] = dict(
        zip(("x", "y", "z"), map(float, origin), strict=True)
    )
    indices = {"x": 0, "y": 1, "z": 2}
    for horizontal_line, span_axis, fixed_axis, arrow in segments:
        span_index, fixed_index = indices[span_axis], indices[fixed_axis]
        half = 0.5 * size[span_index]
        span = (
            center[span_index] - half - origin_by_axis[span_axis],
            center[span_index] + half - origin_by_axis[span_axis],
        )
        fixed = (center[fixed_index] - origin_by_axis[fixed_axis]) / _UM
        span = tuple(value / _UM for value in span)
        x, y = (span, (fixed, fixed)) if horizontal_line else ((fixed, fixed), span)
        ax.plot(x, y, color=color, lw=2.0)
        if not (source and arrow):
            continue
        anchor = (center[span_index] - origin_by_axis[span_axis]) / _UM
        delta = -0.55 if direction.startswith("-") else 0.55
        start_xy = (anchor, fixed) if horizontal_line else (fixed, anchor)
        end_xy = (anchor, fixed + delta) if horizontal_line else (fixed + delta, anchor)
        ax.annotate(
            "",
            xy=end_xy,
            xytext=start_xy,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.0},
        )


def overlay_boundaries(ax, extent, thickness, **style):
    """Draw cross-section PML bands without letting side bands cover corners."""
    from matplotlib.patches import Rectangle

    x0, x1, y0, y1 = (float(v) for v in extent)
    width = max(x1 - x0, 0.0)
    height = max(y1 - y0, 0.0)
    if width <= 0.0 or height <= 0.0:
        return ()

    px = min(max(float(thickness), 0.0), 0.5 * width)
    py = min(max(float(thickness), 0.0), 0.5 * height)
    if px <= 0.0 and py <= 0.0:
        return ()

    specs = []
    side_height = max(height - 2.0 * py, 0.0)
    if px > 0.0 and side_height:
        specs.extend(
            ((x0, y0 + py, px, side_height), (x1 - px, y0 + py, px, side_height))
        )
    if py > 0.0:
        specs.extend(((x0, y0, width, py), (x0, y1 - py, width, py)))
    rectangles = tuple(Rectangle(spec[:2], *spec[2:], **style) for spec in specs)
    for rect in rectangles:
        ax.add_patch(rect)
    return rectangles


def _plot_simulation_slices(
    sim,
    *,
    z,
    y,
    origin,
    display_position,
    width_ratios,
    figsize,
    show,
    cmap,
    categorical,
    overlays,
    source_markers=True,
    monitor_markers=True,
    xlim=None,
    ylim=None,
    zlim=None,
):
    """Compose two simulation cross sections from the generic slice primitives."""
    fields = static_fields(sim)
    eps = np.asarray(fields.permittivity)
    if eps.ndim != 3:
        return plot_design(
            sim.design,
            sources=sim.sources,
            monitors=sim.monitors,
            figsize=figsize,
            show=show,
            title="Simulation Layout",
        )
    lengths = {
        axis: float(getattr(sim.design, attr)) for axis, attr in _DESIGN_EXTENT.items()
    }
    resolution = float(getattr(sim, "resolution", 1.0))
    steps = {
        "z": float(getattr(fields, "dz", resolution)),
        "y": float(getattr(fields, "dy", resolution)),
    }
    unit = _UM if categorical else 1.0
    slices = [
        extract_axis_aligned_slice(
            eps,
            axis=axis,
            step=steps[axis],
            position=position,
            origin=origin,
            display_position=display_position,
            lengths=lengths,
            unit=unit,
        )
        for axis, position in (("z", z), ("y", y))
    ]
    if categorical:
        core_eps, substrate_eps = _material_levels(sim)
        xy, xz = (
            _material_category_array(
                item.values,
                core_permittivity=core_eps,
                substrate_permittivity=substrate_eps,
            )
            for item in slices
        )
        if abs(float(z)) <= 0.5 * resolution:
            xy = np.where(xy >= 2, xy, 1)
        z_coords = np.arange(eps.shape[0], dtype=float) * resolution
        substrate_height = float(origin[2])
        xz = np.where((z_coords > substrate_height)[:, None] & (xz == 1), 0, xz)
        xz = np.where((z_coords <= substrate_height)[:, None] & (xz == 0), 1, xz)
        values = (xy, xz)
        cmap, norm = _material_cmap()
    else:
        values, norm = (item.values for item in slices), None

    fig, axes = _pyplot().subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": width_ratios} if width_ratios else None,
    )
    axes = list(np.ravel(axes))
    images = [
        plot_field_view(
            ax,
            value,
            extent=item.extent,
            cmap=cmap,
            norm=norm,
            aspect="equal" if categorical or item.normal == "z" else "auto",
        )[0]
        for ax, value, item in zip(axes, values, slices, strict=True)
    ]
    if overlays:
        overlay_simulation_devices(
            axes,
            sim,
            origin,
            source_markers=source_markers,
            monitor_markers=monitor_markers,
        )
        thickness = _pml_thickness(sim)
        if thickness is not None and thickness > 0.0:
            style = dict(
                facecolor="#9a9a9a",
                alpha=0.35,
                hatch="xx",
                edgecolor="#777777",
                linewidth=0.0,
            )
            for ax, item in zip(axes, slices, strict=True):
                overlay_boundaries(ax, item.extent, thickness / _UM, **style)

    if categorical:
        for ax, item in zip(axes, slices, strict=True):
            ax.set(
                title=f"cross section at {item.normal}={item.position / _UM:.2f} (um)",
                xlabel=f"{item.horizontal} (um)",
                ylabel=f"{item.vertical} (um)",
            )
        axes[0].set_xlim(slices[0].extent[:2])
        axes[1].set_xlim(slices[1].extent[:2])
    else:
        for ax, image, item in zip(axes, images, slices, strict=True):
            ax.set_title(f"{item.normal} slice {item.index}")
            _axis_labels(
                ax,
                width=lengths[item.horizontal],
                height=lengths[item.vertical],
                y_label=item.vertical,
            )
            fig.colorbar(image, ax=ax, label="permittivity")
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


def overlay_simulation_devices(
    axes, sim, origin, *, source_markers=True, monitor_markers=True
):
    """Overlay projected sources and monitors on xy/xz slice axes."""
    groups = []
    if source_markers:
        groups.append((getattr(sim, "sources", ()), "#66bb6a", True))
    if monitor_markers:
        groups.append((getattr(sim, "monitors", ()), "#f4a51c", False))
    for ax, normal in zip(axes, ("z", "y"), strict=True):
        for devices, color, source in groups:
            for device in devices:
                _draw_device_slice_overlay(
                    ax,
                    device,
                    normal=normal,
                    origin=origin,
                    color=color,
                    source=source,
                )


def _plot_3d_material_cross_sections(
    sim,
    *,
    z=None,
    y=None,
    width_ratios=None,
    figsize=None,
    show=True,
    cmap="Grays",
):
    fig, axes = _plot_simulation_slices(
        sim,
        z=z,
        y=y,
        origin=(0.0, 0.0, 0.0),
        display_position=False,
        width_ratios=width_ratios,
        figsize=figsize or (10.0, 4.0),
        show=show,
        cmap=cmap,
        categorical=False,
        overlays=False,
    )
    return fig, np.asarray(axes)


def _plot_3d_cross_sections(
    sim,
    *,
    z=0.0,
    y=0.0,
    origin=None,
    source_markers=True,
    monitor_markers=True,
    width_ratios=None,
    figsize=None,
    show=True,
    xlim=None,
    ylim=None,
    zlim=None,
):
    if origin is None:
        origin = _simulation_origin(sim)
    return _plot_simulation_slices(
        sim,
        z=z,
        y=y,
        origin=origin,
        display_position=True,
        width_ratios=width_ratios,
        figsize=figsize or (11.0, 4.0),
        show=show,
        cmap=None,
        categorical=True,
        overlays=True,
        source_markers=source_markers,
        monitor_markers=monitor_markers,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )


def plot_simulation(
    sim,
    *,
    z=None,
    y=None,
    categorical_cross_sections=True,
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
    **_kwargs,
):
    """Plot a simulation layout or styled 3D cross sections."""
    if bool(getattr(sim, "is_3d", False)) and (z is not None or y is not None):
        if ax is not None:
            raise ValueError("3D cross-section plots create their own axes.")
        if not categorical_cross_sections:
            return _plot_3d_material_cross_sections(
                sim,
                z=z,
                y=y,
                width_ratios=width_ratios,
                figsize=figsize,
                show=show,
            )
        return _plot_3d_cross_sections(
            sim,
            z=0.0 if z is None else z,
            y=0.0 if y is None else y,
            origin=origin,
            source_markers=source_markers,
            monitor_markers=monitor_markers,
            width_ratios=width_ratios,
            figsize=figsize,
            show=show,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )

    fig, ax = plot_design(
        sim.design,
        sources=getattr(sim, "sources", ()),
        monitors=getattr(sim, "monitors", ()),
        ax=ax,
        figsize=figsize,
        show=False,
        title=title,
    )
    _draw_boundaries(ax, sim)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def view_simulation_3d(sim, *, mode="auto", open_browser=True, show=False, **kwargs):
    """Return a static view for notebooks without restoring the old viewer stack."""
    del mode, open_browser
    if bool(getattr(sim, "is_3d", False)):
        origin = _simulation_origin(sim)
        if any(abs(value) > 0.0 for value in origin):
            kwargs.setdefault("z", 0.0)
            kwargs.setdefault("y", 0.0)
        else:
            kwargs.setdefault("z", 0.5 * float(getattr(sim.design, "depth", 0.0)))
            kwargs.setdefault("y", 0.5 * float(getattr(sim.design, "height", 0.0)))
    return plot_simulation(sim, show=show, **kwargs)


def _coord_extent_um(da, *, sim=None):
    offset = (
        (0.0, 0.0, 0.0)
        if da.attrs.get("coordinate_frame") == "public"
        else getattr(sim, "coordinate_offset", (0.0, 0.0, 0.0))
    )
    offset_by_dim = {"x": offset[0], "y": offset[1], "z": offset[2]}
    dims = [dim for dim in da.dims if dim in {"x", "y", "z", "s"}]
    if len(dims) < 2:
        width = max(int(da.shape[-1]) - 1, 1)
        height = max(int(da.shape[-2]) - 1, 1) if da.ndim > 1 else 1
        return (0.0, float(width), 0.0, float(height)), ("sample", "sample")
    y_dim, x_dim = dims[-2], dims[-1]

    def axis_extent(dim):
        coord = np.asarray(da.coords[dim], dtype=float)
        if coord.size == 0:
            return (0.0, 1.0)
        values = (coord - float(offset_by_dim.get(dim, 0.0))) / 1e-6
        if values.size == 1:
            return (float(values[0] - 0.5), float(values[0] + 0.5))
        step = float(np.nanmedian(np.diff(values)))
        return (float(values[0] - 0.5 * step), float(values[-1] + 0.5 * step))

    x0, x1 = axis_extent(x_dim)
    y0, y1 = axis_extent(y_dim)
    return (x0, x1, y0, y1), (x_dim, y_dim)


def _plot_image(
    values,
    *,
    ax=None,
    figsize=None,
    extent=None,
    xlabel="x (um)",
    ylabel="y (um)",
    title=None,
    cmap="RdBu",
    colorbar=True,
    show=True,
    vmin=None,
    vmax=None,
    xlim=None,
    ylim=None,
    interpolation="nearest",
):
    fig, ax = _figure_axes(ax, figsize=(6.0, 4.0) if figsize is None else figsize)
    im, _view = plot_field_view(
        ax,
        values,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    if colorbar:
        fig.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def _select_saved_field(da, *, frame=None, z=None, y=None):
    selected = da
    if "t" in selected.dims:
        selected = selected.isel(t=-1 if frame is None else int(frame))
    elif "frame" in selected.dims:
        selected = selected.isel(frame=-1 if frame is None else int(frame))
    if "z" in selected.dims:
        if z is None:
            selected = selected.isel(z=selected.sizes["z"] // 2)
        else:
            selected = selected.sel(z=float(z), method="nearest")
    if selected.ndim > 2 and "y" in selected.dims and y is not None:
        selected = selected.sel(y=float(y), method="nearest")
    while selected.ndim > 2:
        selected = selected.isel(
            {selected.dims[0]: selected.sizes[selected.dims[0]] // 2}
        )
    return selected


def _monitor_frequency_index(data, frequency):
    freqs = np.asarray(data.frequencies, dtype=float)
    if freqs.size == 0:
        raise ValueError(f"Monitor {data.name!r} has no DFT data.")
    if frequency is None:
        return 0, freqs
    return int(np.argmin(np.abs(freqs - float(frequency)))), freqs


def _available_dft_component(data, component):
    try:
        return np.asarray(data.field(component), dtype=np.complex128)
    except ValueError:
        return None


def _monitor_dft_values(data, field_name, *, frequency=None, val="real"):
    field_name = str(field_name or "Ez")
    idx, freqs = _monitor_frequency_index(data, frequency)
    if field_name in {"E", "H"}:
        arrays = [
            comp_values
            for comp in (f"{field_name}{axis}" for axis in "xyz")
            if (comp_values := _available_dft_component(data, comp)) is not None
        ]
        if not arrays:
            raise ValueError(f"No DFT components found for field {field_name!r}.")
        vector_val = (
            val if str(val).lower() in {"abs^2", "power", "intensity"} else "abs"
        )
        return _field_view(
            [array[idx] for array in arrays], val=vector_val, vector=True
        ).values, freqs[idx]

    values = _available_dft_component(data, field_name)
    if values is None:
        raise ValueError(f"No DFT component found for field {field_name!r}.")
    return _field_view(values[idx], val=val).values, freqs[idx]


def _monitor_plane_shape_and_coords(simulation, monitor):
    plane_shape = getattr(monitor, "_compiled_dft_shape_3d", None)
    region, coords0, coords1 = monitor_plane_coordinates_3d(simulation, monitor)
    if plane_shape is None:
        plane_shape = (int(coords0.size), int(coords1.size))
    else:
        plane_shape = tuple(int(v) for v in plane_shape)
        coords0 = coords0[: plane_shape[0]]
        coords1 = coords1[: plane_shape[1]]
    return plane_shape, coords0, coords1, region


def _reshape_dft_plane(values, f_idx, plane_shape):
    shape = tuple(int(v) for v in plane_shape)
    samples = np.asarray(values[f_idx], dtype=np.complex128)
    expected = int(np.prod(shape))
    if samples.size != expected:
        raise ValueError(
            f"Recorded DFT plane has {samples.size} samples, but its exact "
            f"rectilinear coordinates require shape {shape} ({expected} samples)."
        )
    return samples.reshape(shape)


def _dft_plane_components(data, field, frequency_index, plane_shape):
    """Return one or more reshaped component planes for a DFT field request."""
    field = str(field)
    if field not in {"E", "H"}:
        values = np.asarray(data.field(field), dtype=np.complex128)
        return [_reshape_dft_plane(values, frequency_index, plane_shape)], False
    names = tuple(f"{field}{axis}" for axis in "xyz")
    components = [
        _reshape_dft_plane(values, frequency_index, plane_shape)
        for name in names
        if (values := _available_dft_component(data, name)) is not None
    ]
    if not components:
        raise ValueError(
            f"No DFT data recorded for derived field {field!r}. "
            f"Expected at least one of: {', '.join(names)}."
        )
    return components, True


def _normalize_dft_components(components, normalization, frequency_index, count):
    if normalization is None:
        return components
    try:
        normalization = source_field_amplitude_normalization(normalization)
    except Exception:
        normalization = np.asarray(normalization, dtype=np.complex128).reshape(-1)
    if (
        normalization is None
        or normalization.size != count
        or abs(normalization[frequency_index]) <= 1e-12
    ):
        return components
    return [value / normalization[frequency_index] for value in components]


def _coordinate_extent(coords0, coords1, *, origin0, origin1, fallback_step):
    def bounds(coords, origin):
        coords = np.asarray(coords, dtype=float)
        step = (
            float(np.mean(np.diff(coords))) if coords.size > 1 else float(fallback_step)
        )
        return (
            (float(coords[0]) - 0.5 * step - float(origin)) / _UM,
            (float(coords[-1]) + 0.5 * step - float(origin)) / _UM,
        )

    x0, x1 = bounds(coords1, origin1)
    y0, y1 = bounds(coords0, origin0)
    return x0, x1, y0, y1


def _edges_are_uniform(edges) -> bool:
    edges = np.asarray(edges, dtype=float)
    widths = np.diff(edges)
    if widths.size < 2:
        return True
    scale = max(1.0, float(np.max(np.abs(edges), initial=0.0)))
    return bool(
        np.allclose(
            widths,
            widths[0],
            rtol=1e-12,
            atol=8.0 * np.finfo(float).eps * scale,
        )
    )


def _field_colorbar_label(
    field, view, *, display_field=None, units="", show_units=True
):
    label = display_field or field
    if view.power:
        text = f"|{label}|" + r"$^2$"
    elif view.magnitude:
        text = f"|{label}|"
    elif view.kind in {"imag", "imaginary", "im"}:
        text = f"Im({label})"
    elif view.kind == "phase":
        text = f"phase({label})"
    else:
        text = f"Re({label})"
    if show_units and units:
        text += f" ({f'({units})^2' if view.power else units})"
    return text


def _overlay_material_slice(
    ax,
    simulation,
    *,
    normal,
    position,
    origin,
    reverse,
    alpha,
    core_permittivity,
    sample_region=None,
):
    vertical, horizontal = _PLANE_AXES[normal]
    origin_by_axis: dict[str, float] = dict(
        zip(("x", "y", "z"), map(float, origin), strict=True)
    )
    grid = getattr(simulation.coordinates, "grid", None)
    plot_x_edges = None
    plot_y_edges = None
    if grid is not None and sample_region is not None:
        materials = simulation.materials
        eps = (
            np.asarray(materials.permittivity)
            if materials is not None
            else np.asarray(())
        )
        if materials is not None and eps.ndim == 3:
            storage_axes = ("z", "y", "x")
            selectors = []
            for dimension, axis in enumerate(storage_axes):
                data_origin = int(materials.origin[dimension])
                if axis == normal:
                    selector = int(sample_region.plane_index) - data_origin
                    if not 0 <= selector < eps.shape[dimension]:
                        return None
                else:
                    interval = sample_region.axis_interval(axis)
                    if interval is None:
                        return None
                    start = int(interval.start) - data_origin
                    stop = int(interval.stop) - data_origin
                    if start < 0 or stop > eps.shape[dimension] or stop <= start:
                        return None
                    selector = slice(
                        start,
                        stop,
                    )
                selectors.append(selector)
            eps_slice = eps[tuple(selectors)]
        else:
            eps_slice = None
        vertical_interval = sample_region.axis_interval(vertical)
        horizontal_interval = sample_region.axis_interval(horizontal)
        if vertical_interval is None or horizontal_interval is None:
            return None
        x_edges = (
            grid.axis_edges(horizontal)[
                horizontal_interval.start : horizontal_interval.stop + 1
            ]
            - origin_by_axis[horizontal]
        ) / _UM
        y_edges = (
            grid.axis_edges(vertical)[
                vertical_interval.start : vertical_interval.stop + 1
            ]
            - origin_by_axis[vertical]
        ) / _UM
        extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
        if not _edges_are_uniform(x_edges) or not _edges_are_uniform(y_edges):
            plot_x_edges, plot_y_edges = x_edges, y_edges
    else:
        eps_slice = _field_eps_slice(
            simulation, plane=normal, plane_position=float(position)
        )
        if eps_slice is None:
            return None
        shape = np.asarray(eps_slice).shape
        metadata = simulation.coordinates
        lengths: dict[str, float] = {
            "x": metadata.width,
            "y": metadata.height,
            "z": metadata.depth,
        }

        def length(axis, fallback):
            return float(lengths[axis] or fallback)

        extent = (
            (0.0 - origin_by_axis[horizontal]) / _UM,
            (
                length(horizontal, shape[1] * simulation.resolution)
                - origin_by_axis[horizontal]
            )
            / _UM,
            (0.0 - origin_by_axis[vertical]) / _UM,
            (
                length(vertical, shape[0] * simulation.resolution)
                - origin_by_axis[vertical]
            )
            / _UM,
        )
    if eps_slice is None:
        return None
    return _draw_field_eps_overlay(
        ax,
        eps_slice,
        extent=extent,
        reverse=reverse,
        alpha=alpha,
        core_permittivity=core_permittivity,
        x_edges=plot_x_edges,
        y_edges=plot_y_edges,
    )


def plot_dft_field(
    simulation,
    monitor=None,
    *,
    field="Ey",
    display_field=None,
    frequency=None,
    frequency_index=0,
    val="real",
    source_normalization=None,
    source_normalize=True,
    origin=None,
    percentile=99.5,
    vmin=None,
    vmax=None,
    ax=None,
    figsize=(6, 4),
    cmap="RdBu",
    colorbar=True,
    overlay_core=True,
    eps_alpha=0.2,
    core_permittivity=None,
    xlim=None,
    ylim=None,
    show_grid=None,
    grid_color="black",
    grid_linewidth=0.2,
    grid_alpha=0.18,
    show_units=True,
    show=True,
):
    """Plot a frequency-domain scalar or vector field on a monitor plane.

    Nonuniform grid edges are drawn automatically. Pass ``show_grid=False`` to
    hide them, or ``show_grid=True`` to draw cell edges on any grid.
    """
    if not isinstance(simulation, AnalysisData):
        raise TypeError("plot_dft_field requires AnalysisData.")
    data = simulation
    monitor = data.monitor_geometry
    if monitor is None:
        raise ValueError("A monitor geometry is required to plot a DFT field.")
    if not getattr(monitor, "is_3d", False):
        raise ValueError("plot_dft_field expects a 3D plane monitor.")
    axis = str(getattr(monitor, "plane_normal", "z")).lower()
    plane_axes = {"x": ("z", "y"), "y": ("z", "x"), "z": ("y", "x")}
    if axis not in plane_axes:
        raise ValueError("DFT field monitor plane_normal must be one of x, y, or z.")
    axis0, axis1 = plane_axes[axis]

    freqs = np.asarray(data.frequencies, dtype=float)
    selected_frequency = (
        freqs[int(np.clip(frequency_index, 0, max(freqs.size - 1, 0)))]
        if frequency is None and freqs.size
        else frequency
    )
    f_idx, freqs = _monitor_frequency_index(data, selected_frequency)

    field_key = str(field)
    coordinate_context = data.coordinates if data is not None else simulation
    plane_shape, target0, target1, sample_region = _monitor_plane_shape_and_coords(
        coordinate_context, monitor
    )
    components, vector = _dft_plane_components(data, field_key, f_idx, plane_shape)

    if source_normalize:
        components = _normalize_dft_components(
            components, source_normalization, f_idx, freqs.size
        )

    field_scale, field_units = _field_display_scale_and_units(field_key)
    components = [component * field_scale for component in components]
    view = _field_view(components if vector else components[0], val=val, vector=vector)

    if origin is None:
        origin = _simulation_origin(coordinate_context)
    ox, oy, oz = (float(v) for v in origin)
    origin_by_axis = {"x": ox, "y": oy, "z": oz}
    grid = getattr(coordinate_context, "grid", None)
    plot_x_edges = None
    plot_y_edges = None
    grid_x_edges = None
    grid_y_edges = None
    if grid is not None:
        interval0 = sample_region.axis_interval(axis0)
        interval1 = sample_region.axis_interval(axis1)
        if interval0 is None or interval1 is None:
            raise ValueError("A 3D DFT monitor is missing a tangential grid interval.")
        edges0 = grid.axis_edges(axis0)
        edges1 = grid.axis_edges(axis1)
        x_edges = (
            edges1[interval1.start : interval1.stop + 1] - origin_by_axis[axis1]
        ) / _UM
        y_edges = (
            edges0[interval0.start : interval0.stop + 1] - origin_by_axis[axis0]
        ) / _UM
        grid_x_edges, grid_y_edges = x_edges, y_edges
        extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
        if not _edges_are_uniform(x_edges) or not _edges_are_uniform(y_edges):
            plot_x_edges, plot_y_edges = x_edges, y_edges
    else:
        extent = _coordinate_extent(
            target0,
            target1,
            origin0=origin_by_axis[axis0],
            origin1=origin_by_axis[axis1],
            fallback_step=simulation.resolution,
        )
    fig, ax = _figure_axes(ax, figsize=figsize)
    im, _ = plot_field_view(
        ax,
        view,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        percentile=percentile,
        aspect="equal",
        x_edges=plot_x_edges,
        y_edges=plot_y_edges,
    )

    if overlay_core:
        _overlay_material_slice(
            ax,
            simulation,
            normal=axis,
            position=float(getattr(monitor, "plane_position", 0.0)),
            origin=origin,
            reverse=view.magnitude,
            alpha=eps_alpha,
            core_permittivity=core_permittivity,
            sample_region=sample_region if plot_x_edges is not None else None,
        )

    draw_grid = plot_x_edges is not None if show_grid is None else bool(show_grid)
    if draw_grid and grid_x_edges is not None and grid_y_edges is not None:
        ax.vlines(
            grid_x_edges,
            grid_y_edges[0],
            grid_y_edges[-1],
            colors=grid_color,
            linewidths=float(grid_linewidth),
            alpha=float(grid_alpha),
        )
        ax.hlines(
            grid_y_edges,
            grid_x_edges[0],
            grid_x_edges[-1],
            colors=grid_color,
            linewidths=float(grid_linewidth),
            alpha=float(grid_alpha),
        )

    cbar_label = _field_colorbar_label(
        field_key,
        view,
        display_field=display_field,
        units=field_units,
        show_units=show_units,
    )
    if colorbar:
        fig.colorbar(
            im,
            ax=ax,
            label=cbar_label,
            extend="max" if view.magnitude else "both",
        )
    ax.set_xlabel(f"{axis1} (um)")
    ax.set_ylabel(f"{axis0} (um)")
    plane_value = float(getattr(monitor, "plane_position", 0.0)) - origin_by_axis[axis]
    ax.set_title(f"cross section at {axis}={plane_value / _UM:.2f} (um)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def _monitor_extent_um(monitor, values, *, sim=None):
    arr = np.asarray(values)
    if arr.ndim == 1:
        length = max(arr.size - 1, 1)
        return (0.0, float(length), 0.0, 1.0), "sample", "sample"
    start = getattr(monitor, "start", None)
    size = getattr(monitor, "size", None)
    if start is not None and size is not None:
        start = np.asarray(start, dtype=float).reshape(-1)
        size = np.asarray(size, dtype=float).reshape(-1)
        offset = np.asarray(
            getattr(sim, "coordinate_offset", (0.0, 0.0, 0.0)), dtype=float
        ).reshape(-1)
        if start.size >= 2 and size.size >= 2:
            return (
                (
                    float((start[0] - offset[0]) / 1e-6),
                    float((start[0] + size[0] - offset[0]) / 1e-6),
                    float((start[1] - offset[1]) / 1e-6),
                    float((start[1] + size[1] - offset[1]) / 1e-6),
                ),
                "x",
                "y",
            )
    return (0.0, float(arr.shape[-1]), 0.0, float(arr.shape[-2])), "sample", "sample"


def plot_result_field(
    results,
    *args,
    field_monitor_name=None,
    monitor_name=None,
    field_name=None,
    field=None,
    frequency=None,
    f=None,
    val="real",
    frame=None,
    z=None,
    y=None,
    ax=None,
    figsize=None,
    cmap=None,
    show=True,
    colorbar=True,
    vmin=None,
    vmax=None,
    xlim=None,
    ylim=None,
    show_grid=None,
    grid_color="black",
    grid_linewidth=0.2,
    grid_alpha=0.18,
    **_kwargs,
):
    """Plot saved simulation fields or a monitor DFT field."""
    inputs = analysis_inputs(results)
    if args:
        first_arg = str(args[0])
        if first_arg in results.monitors or len(args) > 1:
            monitor_name = first_arg
        else:
            field_name = first_arg
    if len(args) > 1:
        field_name = args[1]
    monitor_name = field_monitor_name or monitor_name
    field_name = field_name or field
    frequency = f if frequency is None else frequency
    image_cmap = cmap or ("magma" if str(val).lower() == "abs^2" else "RdBu")

    if monitor_name is not None:
        data = inputs[str(monitor_name)]
        monitor = data.monitor_geometry
        if getattr(monitor, "is_3d", False) and data.frequencies.size:
            dft_freqs = data.frequencies
            if dft_freqs.size:
                return plot_dft_field(
                    data,
                    field=field_name or "Ez",
                    frequency=frequency,
                    val=val,
                    source_normalization=None,
                    ax=ax,
                    figsize=figsize,
                    cmap=image_cmap,
                    show=show,
                    vmin=vmin,
                    vmax=vmax,
                    xlim=xlim,
                    ylim=ylim,
                    show_grid=show_grid,
                    grid_color=grid_color,
                    grid_linewidth=grid_linewidth,
                    grid_alpha=grid_alpha,
                    colorbar=colorbar,
                )
        values, freq = _monitor_dft_values(
            data,
            field_name or "Ez",
            frequency=frequency,
            val=val,
        )
        extent, x_dim, y_dim = _monitor_extent_um(
            monitor,
            values,
            sim=data.coordinates,
        )
        title = f"{field_name or 'field'} at {freq * 1e-12:g} THz"
    else:
        from beamz.analysis.adapters import to_xarray

        fields = to_xarray(results)
        if fields is None or not getattr(fields, "data_vars", None):
            raise RuntimeError("No saved fields are available to plot.")
        name = field_name or next(iter(fields.data_vars))
        selected = _select_saved_field(fields[str(name)], frame=frame, z=z, y=y)
        values = _field_view(selected, val=val).values
        data = next(iter(inputs.values()))
        extent, (x_dim, y_dim) = _coord_extent_um(selected, sim=data.coordinates)
        title = str(name)
    return _plot_image(
        values,
        ax=ax,
        figsize=figsize,
        extent=extent,
        xlabel=f"{x_dim} (um)",
        ylabel=f"{y_dim} (um)",
        title=title,
        cmap=image_cmap,
        colorbar=colorbar,
        show=show,
        vmin=vmin,
        vmax=vmax,
        xlim=xlim,
        ylim=ylim,
    )


def plot_mode_field_components(
    modes,
    *,
    field_names=("Ey", "Ez"),
    mode_indices=(0,),
    val="abs",
    f=None,
    ax=None,
    figsize=None,
    cmap=None,
    show=True,
):
    """Plot selected electric or magnetic components on their physical grid."""
    import matplotlib.pyplot as plt

    value_kind = str(val).lower()
    fields = tuple(map(str, field_names))
    selected_modes = tuple(map(int, mode_indices))
    f_idx = modes._frequency_index(f)
    shape = (len(selected_modes), len(fields))
    if ax is None:
        figsize = figsize or (4.0 * shape[1], 3.2 * shape[0])
        fig, axes = plt.subplots(*shape, squeeze=False, figsize=figsize)
    else:
        axes = np.asarray(ax, dtype=object).reshape(shape)
        fig = axes.flat[0].figure
    component_index = {name: idx for idx, name in enumerate(("Ex", "Ey", "Ez"))}
    component_index.update({name: idx for idx, name in enumerate(("Hx", "Hy", "Hz"))})
    grid_edges = getattr(modes, "grid_edges", None)
    transverse_axes = getattr(modes, "transverse_axes", None)
    neffs = []
    for row, mode_index in enumerate(selected_modes):
        neff = np.asarray(modes.neffs)[f_idx, mode_index]
        neffs.append(neff)
        for col, field_name in enumerate(fields):
            if field_name not in component_index:
                raise ValueError(f"Unsupported modal field component {field_name!r}.")
            source = modes.e_fields if field_name.startswith("E") else modes.h_fields
            values = np.asarray(source)[f_idx, mode_index, component_index[field_name]]
            default_cmap = (
                "RdBu"
                if value_kind in {"real", "imag"}
                else "twilight"
                if value_kind == "phase"
                else "magma"
            )
            raw_field = np.asarray(values)
            field = (
                raw_field
                if grid_edges is not None and raw_field.ndim == len(grid_edges)
                else np.squeeze(raw_field)
            )
            plot_edges = None
            if grid_edges is not None and len(grid_edges) == field.ndim:
                plot_edges = tuple(
                    np.asarray(edge, dtype=float) / _UM for edge in grid_edges
                )
            if field.ndim == 2 and plot_edges is not None:
                y_edges, x_edges = plot_edges
                y_axis, x_axis = (
                    tuple(transverse_axes)
                    if transverse_axes is not None and len(transverse_axes) == 2
                    else ("axis 0", "axis 1")
                )
                xlabel = f"{x_axis} (um)"
                ylabel = f"{y_axis} (um)"
            elif field.ndim == 1 and plot_edges is not None:
                x_edges = plot_edges[0]
                y_edges = np.asarray([-0.5, 0.5])
                x_axis = (
                    str(transverse_axes[0])
                    if transverse_axes is not None and len(transverse_axes) == 1
                    else "axis 0"
                )
                xlabel = f"{x_axis} (um)"
                ylabel = ""
            else:
                x_edges = y_edges = None
                xlabel = ylabel = "grid index"
            plot_field_view(
                axes[row, col],
                field,
                val=value_kind,
                cmap=cmap or default_cmap,
                x_edges=x_edges,
                y_edges=y_edges,
            )
            axes[row, col].set(
                title=f"{field_name} mode {mode_index}, n_eff={np.real(neff):.4g}",
                xlabel=xlabel,
                ylabel=ylabel,
            )
    fig.suptitle(f"Mode fields at f={float(modes.frequencies[f_idx]):.6g} Hz")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes, np.asarray(neffs)
