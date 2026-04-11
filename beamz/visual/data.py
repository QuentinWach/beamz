"""Pure-data visualization helpers.

This module exposes plotting-oriented data structures without depending on any
rendering backend. Examples can consume these payloads with matplotlib or any
other library.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from beamz.visual.helpers import get_si_scale_and_label


def _as_float_tuple(values):
    return tuple(float(v) for v in values)


def _vertices_2d(vertices):
    return [tuple(float(coord) for coord in vertex[:2]) for vertex in vertices]


def _style_payload(**kwargs):
    return {key: value for key, value in kwargs.items() if value is not None}


def _material_signature(structure):
    material = getattr(structure, "material", None)
    if material is None:
        return None
    return (
        getattr(material, "permittivity", 1.0),
        getattr(material, "permeability", 1.0),
        getattr(material, "conductivity", 0.0),
    )


def _get_deterministic_color(index):
    """Get deterministic colors matching the historical matplotlib output."""
    import colorsys

    from beamz.const import BLUE, GREEN, ORANGE, PURPLE, RED

    predefined_colors = [BLUE, RED, GREEN, ORANGE, PURPLE]
    if index < len(predefined_colors):
        return predefined_colors[index]

    hue = (index * 0.618034) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.7)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def structure_plot_data(
    structure,
    *,
    facecolor=None,
    edgecolor="black",
    alpha=1.0,
    linestyle="-",
):
    """Serialize a structure into renderer-agnostic geometry data."""
    return {
        "kind": "structure",
        "shape": "polygon",
        "vertices": _vertices_2d(getattr(structure, "vertices", ()) or ()),
        "interiors": [
            _vertices_2d(interior)
            for interior in (getattr(structure, "interiors", None) or [])
            if interior
        ],
        "depth": float(getattr(structure, "depth", 0.0) or 0.0),
        "z": float(getattr(structure, "z", 0.0) or 0.0),
        "position": _as_float_tuple(getattr(structure, "position", (0.0, 0.0, 0.0))),
        "style": _style_payload(
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=float(alpha),
            linestyle=linestyle,
        ),
        "material": {
            "permittivity": getattr(getattr(structure, "material", None), "permittivity", None),
            "permeability": getattr(getattr(structure, "material", None), "permeability", None),
            "conductivity": getattr(getattr(structure, "material", None), "conductivity", None),
        },
        "is_pml": bool(getattr(structure, "is_pml", False)),
        "name": type(structure).__name__,
    }


def gaussian_source_plot_data(
    source,
    *,
    facecolor="none",
    edgecolor="orange",
    alpha=0.8,
    linestyle="-",
):
    position = tuple(float(v) for v in source.position)
    return {
        "kind": "source",
        "shape": "gaussian",
        "position": position,
        "radius": float(source.width),
        "style": _style_payload(
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=float(alpha),
            linestyle=linestyle,
        ),
        "name": type(source).__name__,
    }


def mode_source_plot_data(
    source,
    *,
    facecolor="none",
    edgecolor="crimson",
    alpha=0.8,
    linestyle="-",
):
    center = source.center if isinstance(source.center, (tuple, list)) else (source.center, 0.0)
    center = tuple(float(v) for v in center)
    return {
        "kind": "source",
        "shape": "mode",
        "center": center,
        "width": float(source.width) if source.width is not None else None,
        "height": float(source.height) if getattr(source, "height", None) is not None else None,
        "direction": str(source.direction),
        "wavelength": float(source.wavelength) if getattr(source, "wavelength", None) is not None else None,
        "style": _style_payload(
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=float(alpha),
            linestyle=linestyle,
        ),
        "name": type(source).__name__,
    }


def source_plot_data(source, **style):
    if hasattr(source, "center") and hasattr(source, "direction"):
        return mode_source_plot_data(source, **style)
    if hasattr(source, "position") and hasattr(source, "width"):
        return gaussian_source_plot_data(source, **style)
    return {
        "kind": "source",
        "shape": "unknown",
        "name": type(source).__name__,
        "style": _style_payload(**style),
    }


def monitor_plot_data(
    monitor,
    *,
    facecolor="none",
    edgecolor="navy",
    alpha=1.0,
    linestyle="-",
):
    if getattr(monitor, "monitor_type", None) == "line":
        return {
            "kind": "monitor",
            "shape": "line",
            "start": _as_float_tuple(monitor.start),
            "end": _as_float_tuple(monitor.end),
            "style": _style_payload(
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=float(alpha),
                linestyle=linestyle,
            ),
            "name": getattr(monitor, "name", None) or type(monitor).__name__,
        }

    return {
        "kind": "monitor",
        "shape": "plane",
        "plane_normal": getattr(monitor, "plane_normal", None),
        "plane_position": float(getattr(monitor, "plane_position", 0.0) or 0.0),
        "position": _as_float_tuple(getattr(monitor, "position", (0.0, 0.0, 0.0))),
        "start": _as_float_tuple(getattr(monitor, "start", (0.0, 0.0, 0.0))),
        "size": _as_float_tuple(getattr(monitor, "size", (0.0, 0.0))),
        "style": _style_payload(
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=float(alpha),
            linestyle=linestyle,
        ),
        "name": getattr(monitor, "name", None) or type(monitor).__name__,
    }


def boundary_plot_data(boundary, design, *, edgecolor="red", alpha=0.5, linestyle="--"):
    """Return axis-aligned boundary rectangles for manual plotting."""
    rectangles = []
    for edge in boundary._get_edges_for_dimensionality(getattr(design, "is_3d", False)):
        if edge == "left":
            rect = ((0.0, 0.0), float(boundary.thickness), float(design.height))
        elif edge == "right":
            rect = (
                (float(design.width - boundary.thickness), 0.0),
                float(boundary.thickness),
                float(design.height),
            )
        elif edge == "bottom":
            rect = ((0.0, 0.0), float(design.width), float(boundary.thickness))
        elif edge == "top":
            rect = (
                (0.0, float(design.height - boundary.thickness)),
                float(design.width),
                float(boundary.thickness),
            )
        elif edge == "front":
            rect = ((0.0, 0.0), float(design.width), float(design.height))
        elif edge == "back":
            rect = ((0.0, 0.0), float(design.width), float(design.height))
        else:
            continue
        rectangles.append(
            {
                "edge": edge,
                "origin": rect[0],
                "width": rect[1],
                "height": rect[2],
                "style": _style_payload(
                    edgecolor=edgecolor,
                    alpha=float(alpha),
                    linestyle=linestyle,
                    facecolor="none",
                ),
            }
        )
    return {
        "kind": "boundary",
        "type": type(boundary).__name__,
        "thickness": float(boundary.thickness),
        "rectangles": rectangles,
    }


def design_plot_data(
    design,
    *,
    unify_structures=True,
    sources=None,
    monitors=None,
):
    """Serialize a design and overlay objects into plotting data."""
    if unify_structures:
        tmp_design = design.copy()
        tmp_design.unify_polygons()
        structures_to_plot = tmp_design.structures
    else:
        structures_to_plot = design.structures

    material_colors = {}
    color_index = 0
    structure_data = []
    for structure in structures_to_plot:
        if getattr(structure, "is_pml", False):
            structure_data.append(
                structure_plot_data(
                    structure,
                    facecolor="none",
                    edgecolor="red",
                    alpha=0.5,
                    linestyle="--",
                )
            )
            continue

        material_key = _material_signature(structure)
        if material_key not in material_colors:
            material_colors[material_key] = _get_deterministic_color(color_index)
            color_index += 1

        structure_data.append(
            structure_plot_data(
                structure,
                facecolor=material_colors[material_key],
                edgecolor="black",
                alpha=1.0,
                linestyle="-",
            )
        )

    max_dim = max(float(design.width), float(design.height), float(getattr(design, "depth", 0.0) or 0.0))
    scale, unit = get_si_scale_and_label(max_dim)
    return {
        "kind": "design",
        "width": float(design.width),
        "height": float(design.height),
        "depth": float(getattr(design, "depth", 0.0) or 0.0),
        "is_3d": bool(getattr(design, "is_3d", False)),
        "scale_factor": float(scale),
        "scale_unit": unit,
        "xlim": (0.0, float(design.width)),
        "ylim": (0.0, float(design.height)),
        "structures": structure_data,
        "sources": [source_plot_data(source) for source in (sources or [])],
        "monitors": [monitor_plot_data(monitor) for monitor in (monitors or [])],
    }


def simulation_plot_data(sim, *, unify_structures=True):
    """Serialize a simulation layout and all static overlays."""
    return {
        "kind": "simulation",
        "plane_2d": sim.plane_2d,
        "design": design_plot_data(
            sim.design,
            unify_structures=unify_structures,
            sources=sim.sources,
            monitors=sim.monitors,
        ),
        "boundaries": [
            boundary_plot_data(boundary, sim.design) for boundary in (sim.boundaries or [])
        ],
    }


def grid_plot_data(grid, *, field="permittivity", z_index=None, z_position=None):
    """Serialize a rasterized grid or grid slice into plotting data."""
    if getattr(grid, "is_3d", False):
        slice_data = grid.get_2d_slice(z_index=z_index, z_position=z_position)
        array = np.asarray(slice_data[field])
        plane_meta = {
            "z_index": (
                int(z_index)
                if z_index is not None
                else int(array.shape[0] // 2 if array.ndim == 3 else grid.shape[0] // 2)
            )
        }
    else:
        array = np.asarray(getattr(grid, field))
        plane_meta = {}

    return {
        "kind": "grid",
        "field": field,
        "array": array.copy(),
        "extent": (0.0, float(grid.design.width), 0.0, float(grid.design.height)),
        "design": {
            "width": float(grid.design.width),
            "height": float(grid.design.height),
            "depth": float(getattr(grid.design, "depth", 0.0) or 0.0),
        },
        "meta": plane_meta,
    }


def signal_plot_data(signals, t):
    """Prepare signal/time arrays with display scaling metadata."""
    t_seconds = np.asarray(t, dtype=float)
    if t_seconds[-1] < 1e-12:
        scale, unit = 1e15, "fs"
    elif t_seconds[-1] < 1e-9:
        scale, unit = 1e12, "ps"
    elif t_seconds[-1] < 1e-6:
        scale, unit = 1e9, "ns"
    elif t_seconds[-1] < 1e-3:
        scale, unit = 1e6, "µs"
    elif t_seconds[-1] < 1:
        scale, unit = 1e3, "ms"
    else:
        scale, unit = 1.0, "s"

    if isinstance(signals, list):
        series = [np.asarray(signal) for signal in signals]
    else:
        series = [np.asarray(signals)]

    return {
        "kind": "signal",
        "t_seconds": t_seconds,
        "t_scaled": t_seconds * scale,
        "time_scale": float(scale),
        "time_unit": unit,
        "signals": [values.copy() for values in series],
        "xlim": (float(t_seconds[0] * scale), float(t_seconds[-1] * scale)),
    }


def mode_profile_data(mode_source, field=None):
    """Serialize mode-source profile data for manual plotting."""
    del field  # backward-compatible placeholder
    if mode_source._Ez_profile is None and mode_source._jz_profile is None:
        if mode_source.grid is not None and hasattr(mode_source.grid, "permittivity"):
            res = getattr(mode_source.grid, "resolution", 0.05e-6)
            mode_source.initialize(mode_source.grid.permittivity, res)
        else:
            raise RuntimeError(
                "[ModeSource] Source not initialized. Call Simulation or initialize manually."
            )

    if mode_source._Ez_profile is not None:
        profile = mode_source._Ez_profile
        title = "Ez (mode profile)"
    elif mode_source._jz_profile is not None:
        profile = mode_source._jz_profile
        title = "Hz (mode profile)"
    else:
        raise RuntimeError("[ModeSource] No profiles available.")

    profile = np.squeeze(np.asarray(profile))
    return {
        "kind": "mode_profile",
        "profile": profile.copy(),
        "amplitude": np.abs(profile),
        "title": title,
        "direction": getattr(mode_source, "direction", None),
        "neff": float(getattr(mode_source, "_neff", np.nan)),
        "is_2d": bool(profile.ndim == 2),
    }


def monitor_field_plot_data(monitor, *, field="Ez", time_index=-1):
    """Serialize monitor field data for manual plotting."""
    if not monitor.fields["t"]:
        raise RuntimeError("No field data recorded.")
    if field not in monitor.fields:
        raise RuntimeError(
            f"Field '{field}' not available. Available fields: {list(monitor.fields.keys())}"
        )
    if not monitor.fields[field]:
        raise RuntimeError(f"No data for field '{field}'.")

    field_data = np.asarray(monitor.fields[field][time_index])
    t_value = float(monitor.fields["t"][time_index])
    return {
        "kind": "monitor_field",
        "field": field,
        "time": t_value,
        "monitor_type": monitor.monitor_type,
        "array": field_data.copy(),
        "x": np.arange(field_data.shape[-1], dtype=float),
        "title": f"{field} at t = {t_value:.2e} s",
    }


def monitor_power_plot_data(monitor, *, log_scale=False, db_scale=False):
    """Serialize monitor power-history data for manual plotting."""
    if not monitor.power_history:
        raise RuntimeError("No power data recorded.")

    power_data = np.asarray(monitor.power_history, dtype=float)
    ylabel = "Power"
    yscale = "linear"
    if db_scale:
        power_data = 10 * np.log10(np.maximum(power_data, 1e-12))
        ylabel = "Power (dB)"
    elif log_scale:
        yscale = "log"
        ylabel = "Power (log scale)"

    return {
        "kind": "monitor_power",
        "time_steps": np.arange(len(power_data), dtype=float),
        "power": power_data,
        "ylabel": ylabel,
        "yscale": yscale,
        "title": "Power vs Time",
    }


def snapshot_payload(
    *,
    field,
    field_name,
    t,
    step,
    num_steps,
    extent,
    units,
    plane_2d,
    simulation=None,
    layout=None,
):
    """Build a renderer-agnostic snapshot payload for streamed fields."""
    return {
        "kind": "simulation_snapshot",
        "field": np.asarray(field).copy(),
        "field_name": str(field_name),
        "time": float(t),
        "step": int(step),
        "num_steps": int(num_steps),
        "extent": tuple(float(v) for v in extent),
        "units": str(units),
        "plane_2d": str(plane_2d),
        "layout": layout if layout is not None else (
            simulation_plot_data(simulation) if simulation is not None else None
        ),
    }


__all__ = [
    "boundary_plot_data",
    "design_plot_data",
    "grid_plot_data",
    "mode_profile_data",
    "monitor_field_plot_data",
    "monitor_plot_data",
    "monitor_power_plot_data",
    "signal_plot_data",
    "simulation_plot_data",
    "snapshot_payload",
    "source_plot_data",
    "structure_plot_data",
]
