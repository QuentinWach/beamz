"""Pure-data visualization helpers.

This module exposes plotting-oriented data structures without depending on any
rendering backend. Examples can consume these payloads with matplotlib or any
other library.
"""

from __future__ import annotations

import numpy as np

from beamz.visual.helpers import get_si_scale_and_label


def _as_float_tuple(values):
    return tuple(float(v) for v in values)


def _as_real_float(value, default=np.nan):
    if value is None:
        return float(default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(np.real(array.reshape(-1)[0]))


def _vertices_2d(vertices):
    return [tuple(float(coord) for coord in vertex[:2]) for vertex in vertices]


def _style_payload(**kwargs):
    return {key: value for key, value in kwargs.items() if value is not None}


def _frequency_scale_and_label(max_frequency):
    max_frequency = float(max_frequency)
    if max_frequency >= 1e12:
        return 1e-12, "THz"
    if max_frequency >= 1e9:
        return 1e-9, "GHz"
    if max_frequency >= 1e6:
        return 1e-6, "MHz"
    if max_frequency >= 1e3:
        return 1e-3, "kHz"
    return 1.0, "Hz"


def _world_origin(design_or_grid):
    design = getattr(design_or_grid, "design", design_or_grid)
    origin = getattr(design, "world_origin", None)
    if origin is None:
        return (0.0, 0.0, 0.0)
    return tuple(float(v) for v in origin)


def _shift_point(point, origin):
    values = tuple(float(v) for v in point)
    if len(values) == 2:
        return (values[0] + origin[0], values[1] + origin[1])
    return (values[0] + origin[0], values[1] + origin[1], values[2] + origin[2])


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
    origin=(0.0, 0.0, 0.0),
    facecolor=None,
    edgecolor="black",
    alpha=1.0,
    linestyle="-",
):
    """Serialize a structure into renderer-agnostic geometry data."""
    return {
        "kind": "structure",
        "shape": "polygon",
        "vertices": [
            (float(x) + origin[0], float(y) + origin[1])
            for x, y in _vertices_2d(getattr(structure, "vertices", ()) or ())
        ],
        "interiors": [
            [
                (float(x) + origin[0], float(y) + origin[1])
                for x, y in _vertices_2d(interior)
            ]
            for interior in (getattr(structure, "interiors", None) or [])
            if interior
        ],
        "depth": float(getattr(structure, "depth", 0.0) or 0.0),
        "z": float(getattr(structure, "z", 0.0) or 0.0) + origin[2],
        "position": _as_float_tuple(
            _shift_point(getattr(structure, "position", (0.0, 0.0, 0.0)), origin)
        ),
        "style": _style_payload(
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=float(alpha),
            linestyle=linestyle,
        ),
        "material": {
            "permittivity": getattr(
                getattr(structure, "material", None), "permittivity", None
            ),
            "permeability": getattr(
                getattr(structure, "material", None), "permeability", None
            ),
            "conductivity": getattr(
                getattr(structure, "material", None), "conductivity", None
            ),
        },
        "is_pml": bool(getattr(structure, "is_pml", False)),
        "name": type(structure).__name__,
    }


def gaussian_source_plot_data(
    source,
    *,
    origin=(0.0, 0.0, 0.0),
    facecolor="none",
    edgecolor="orange",
    alpha=0.8,
    linestyle="-",
):
    position = _shift_point(source.position, origin)
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


def _source_width_height(source):
    size = getattr(source, "size", None)
    if size is not None and not callable(size):
        values = np.asarray(size, dtype=float).reshape(-1)
        if values.size == 3:
            axes = {"+x": (1, 2), "-x": (1, 2), "+y": (0, 2), "-y": (0, 2)}
            values = values[
                list(axes.get(str(getattr(source, "direction", "")), (0, 1)))
            ]
        if values.size:
            return float(values[0]), float(values[min(1, values.size - 1)])
    width = getattr(source, "width", None)
    height = getattr(source, "height", None)
    return (
        float(width) if width is not None else None,
        float(height) if height is not None else None,
    )


def mode_source_plot_data(
    source,
    *,
    origin=(0.0, 0.0, 0.0),
    facecolor="none",
    edgecolor="crimson",
    alpha=0.8,
    linestyle="-",
):
    snapped = (
        source.get_snapped_region()
        if hasattr(source, "get_snapped_region")
        else getattr(source, "_snapped_region", None)
    )
    if snapped is not None:
        center = _shift_point(snapped.center, origin)
        if snapped.normal_axis == "x":
            width = float(snapped.axis_bounds("y")[1] - snapped.axis_bounds("y")[0])
            height = (
                float(snapped.axis_bounds("z")[1] - snapped.axis_bounds("z")[0])
                if snapped.ndim == 3
                else None
            )
        elif snapped.normal_axis == "y":
            width = float(snapped.axis_bounds("x")[1] - snapped.axis_bounds("x")[0])
            height = (
                float(snapped.axis_bounds("z")[1] - snapped.axis_bounds("z")[0])
                if snapped.ndim == 3
                else None
            )
        else:
            width = float(snapped.axis_bounds("x")[1] - snapped.axis_bounds("x")[0])
            height = float(snapped.axis_bounds("y")[1] - snapped.axis_bounds("y")[0])
    else:
        center = (
            source.center
            if isinstance(source.center, (tuple, list))
            else (source.center, 0.0)
        )
        center = _shift_point(center, origin)
        width, height = _source_width_height(source)
    return {
        "kind": "source",
        "shape": "mode",
        "center": center,
        "width": width,
        "height": height,
        "direction": str(source.direction),
        "wavelength": (
            float(source.wavelength)
            if getattr(source, "wavelength", None) is not None
            else None
        ),
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
    origin=(0.0, 0.0, 0.0),
    facecolor="none",
    edgecolor="navy",
    alpha=1.0,
    linestyle="-",
):
    snapped = None
    if hasattr(monitor, "get_snapped_region"):
        kwargs = {}
        resolution = getattr(monitor, "_resolution", None)
        if resolution is not None:
            kwargs["dx"] = float(resolution)
            kwargs["dy"] = float(resolution)
            if getattr(monitor, "is_3d", False):
                kwargs["dz"] = float(resolution)
                shape = getattr(monitor, "_field_shape", None)
                if shape is not None:
                    kwargs["field_shape"] = shape
        if kwargs:
            try:
                snapped = monitor.get_snapped_region(**kwargs)
            except TypeError:
                snapped = getattr(monitor, "_snapped_region", None)
        else:
            snapped = getattr(monitor, "_snapped_region", None)

    if getattr(monitor, "monitor_type", None) == "line":
        return {
            "kind": "monitor",
            "shape": "line",
            "start": _as_float_tuple(
                _shift_point(
                    snapped.start if snapped is not None else monitor.start, origin
                )
            ),
            "end": _as_float_tuple(
                _shift_point(
                    snapped.end if snapped is not None else monitor.end, origin
                )
            ),
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
        "position": _as_float_tuple(
            _shift_point(
                (
                    snapped.center
                    if snapped is not None
                    else getattr(monitor, "position", (0.0, 0.0, 0.0))
                ),
                origin,
            )
        ),
        "start": _as_float_tuple(
            _shift_point(
                (
                    snapped.start
                    if snapped is not None
                    else getattr(monitor, "start", (0.0, 0.0, 0.0))
                ),
                origin,
            )
        ),
        "size": _as_float_tuple(
            (
                (
                    snapped.axis_bounds("y")[1] - snapped.axis_bounds("y")[0],
                    snapped.axis_bounds("z")[1] - snapped.axis_bounds("z")[0],
                )
                if snapped is not None
                and str(getattr(monitor, "plane_normal", "z")).lower() == "x"
                else (
                    (
                        snapped.axis_bounds("x")[1] - snapped.axis_bounds("x")[0],
                        snapped.axis_bounds("z")[1] - snapped.axis_bounds("z")[0],
                    )
                    if snapped is not None
                    and str(getattr(monitor, "plane_normal", "z")).lower() == "y"
                    else (
                        (
                            snapped.axis_bounds("x")[1] - snapped.axis_bounds("x")[0],
                            snapped.axis_bounds("y")[1] - snapped.axis_bounds("y")[0],
                        )
                        if snapped is not None
                        else getattr(monitor, "size", (0.0, 0.0))
                    )
                )
            )
        ),
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
    origin = _world_origin(design)
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
                "origin": _shift_point(rect[0], origin),
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
    origin = _world_origin(design)
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
                    origin=origin,
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
                origin=origin,
                facecolor=material_colors[material_key],
                edgecolor="black",
                alpha=1.0,
                linestyle="-",
            )
        )

    max_dim = max(
        float(design.width),
        float(design.height),
        float(getattr(design, "depth", 0.0) or 0.0),
    )
    scale, unit = get_si_scale_and_label(max_dim)
    return {
        "kind": "design",
        "width": float(design.width),
        "height": float(design.height),
        "depth": float(getattr(design, "depth", 0.0) or 0.0),
        "is_3d": bool(getattr(design, "is_3d", False)),
        "scale_factor": float(scale),
        "scale_unit": unit,
        "xlim": (origin[0], origin[0] + float(design.width)),
        "ylim": (origin[1], origin[1] + float(design.height)),
        "structures": structure_data,
        "sources": [
            source_plot_data(source, origin=origin) for source in (sources or [])
        ],
        "monitors": [
            monitor_plot_data(monitor, origin=origin) for monitor in (monitors or [])
        ],
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
            boundary_plot_data(boundary, sim.design)
            for boundary in (sim.boundaries or [])
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
        "extent": (
            _world_origin(grid)[0],
            _world_origin(grid)[0] + float(grid.design.width),
            _world_origin(grid)[1],
            _world_origin(grid)[1] + float(grid.design.height),
        ),
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


def source_signal_plot_data(source, t=None):
    """Prepare source time-signal data from a source object."""
    signal = getattr(source, "signal", None)
    if signal is None:
        raise RuntimeError(f"{type(source).__name__} has no signal attribute.")
    if callable(signal):
        if t is None:
            raise ValueError(
                "t must be provided when plotting a callable source signal."
            )
        t_arr = np.asarray(t, dtype=float)
        values = np.asarray([signal(float(ti)) for ti in t_arr])
    else:
        values = np.asarray(signal)
        if t is None:
            t_arr = np.arange(values.shape[0], dtype=float)
        else:
            t_arr = np.asarray(t, dtype=float)
            if t_arr.shape[0] != values.shape[0]:
                raise ValueError(
                    "t and source signal must have the same length: "
                    f"{t_arr.shape[0]} != {values.shape[0]}"
                )
    payload = signal_plot_data(values, t_arr)
    payload["kind"] = "source_signal"
    payload["source_type"] = type(source).__name__
    return payload


def source_spectrum_plot_data(source, t=None, *, dt=None):
    """Prepare a one-sided source spectrum from a source object."""
    signal_payload = source_signal_plot_data(source, t=t)
    values = np.asarray(signal_payload["signals"][0])
    n = values.size
    if n < 2:
        raise ValueError("At least two signal samples are required for a spectrum.")

    if dt is None:
        t_seconds = np.asarray(signal_payload["t_seconds"], dtype=float)
        deltas = np.diff(t_seconds)
        if t is None or not np.all(np.isfinite(deltas)) or np.allclose(deltas, 0.0):
            dt_seconds = 1.0
            frequency_unit = "cycles/sample"
            frequency_scale = 1.0
        else:
            dt_seconds = float(np.median(deltas))
            frequency_scale, frequency_unit = _frequency_scale_and_label(
                1.0 / max(dt_seconds, 1e-30)
            )
    else:
        dt_seconds = float(dt)
        frequency_scale, frequency_unit = _frequency_scale_and_label(
            1.0 / max(dt_seconds, 1e-30)
        )

    if np.iscomplexobj(values):
        spectrum = np.fft.fft(values)
        freqs = np.fft.fftfreq(n, d=dt_seconds)
        keep = freqs >= 0.0
        freqs = freqs[keep]
        spectrum = spectrum[keep]
    else:
        spectrum = np.fft.rfft(values)
        freqs = np.fft.rfftfreq(n, d=dt_seconds)

    amplitude = np.abs(spectrum)
    if np.max(amplitude) > 0:
        amplitude = amplitude / np.max(amplitude)

    return {
        "kind": "source_spectrum",
        "source_type": signal_payload["source_type"],
        "frequency_hz": freqs.copy(),
        "frequency_scaled": freqs * frequency_scale,
        "frequency_scale": float(frequency_scale),
        "frequency_unit": frequency_unit,
        "amplitude": amplitude,
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
        "neff": _as_real_float(getattr(mode_source, "_neff", np.nan)),
        "is_2d": bool(profile.ndim == 2),
    }


def mode_permittivity_plot_data(mode_source):
    """Prepare mode-source permittivity data for plotting."""
    eps = getattr(mode_source, "_eps_profile_2d", None)
    if eps is None:
        grid = getattr(mode_source, "grid", None)
        eps = getattr(grid, "permittivity", None)
    if eps is None:
        raise RuntimeError("No permittivity data available for this ModeSource.")
    eps = np.squeeze(np.asarray(eps))
    if eps.ndim > 2:
        eps = eps[eps.shape[0] // 2]
    return {
        "kind": "mode_permittivity",
        "array": np.asarray(eps).copy(),
        "title": "Mode Source Permittivity",
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
    payload = {
        "kind": "monitor_field",
        "field": field,
        "time": t_value,
        "monitor_type": monitor.monitor_type,
        "array": field_data.copy(),
        "title": f"{field} at t = {t_value:.2e} s",
    }
    if monitor.monitor_type == "line":
        start = np.asarray(getattr(monitor, "start", (0.0, 0.0)), dtype=float)
        end = np.asarray(getattr(monitor, "end", start), dtype=float)
        length = float(np.linalg.norm(end - start))
        scale, unit = get_si_scale_and_label(max(length, 1e-30))
        payload.update(
            {
                "x": np.linspace(0.0, length * scale, field_data.size),
                "xlabel": f"Position along monitor ({unit})",
            }
        )
    else:
        size = tuple(float(v) for v in getattr(monitor, "size", field_data.shape[-2:]))
        if len(size) >= 2 and size[0] > 0.0 and size[1] > 0.0:
            extent = (0.0, size[0], 0.0, size[1])
        else:
            extent = (0.0, field_data.shape[-1], 0.0, field_data.shape[-2])
        scale, unit = get_si_scale_and_label(max(extent[1], extent[3], 1e-30))
        payload.update(
            {
                "extent": tuple(float(v * scale) for v in extent),
                "xlabel": f"Axis 1 ({unit})",
                "ylabel": f"Axis 2 ({unit})",
            }
        )
    return payload


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


def _select_field_frame(values, *, time_index=-1):
    arr = np.asarray(values)
    if arr.ndim < 2:
        raise ValueError(f"Field data must be at least 2D, got shape {arr.shape}.")
    if arr.ndim in {3, 4}:
        return np.asarray(arr[time_index]), int(time_index)
    return arr, None


def _slice_2d(values, *, plane="z", index=None):
    arr = np.squeeze(np.asarray(values))
    if arr.ndim == 2:
        return arr, "xy", None
    if arr.ndim != 3:
        raise ValueError(f"Cannot plot field array with shape {arr.shape}.")

    plane_key = str(plane).lower()
    if plane_key in {"xy", "z"}:
        axis = 0
        plane_label = "xy"
    elif plane_key in {"xz", "y"}:
        axis = 1
        plane_label = "xz"
    elif plane_key in {"yz", "x"}:
        axis = 2
        plane_label = "yz"
    else:
        raise ValueError("plane must be one of 'xy'/'z', 'xz'/'y', or 'yz'/'x'.")

    if index is None:
        index = arr.shape[axis] // 2
    index = int(index)
    if index < 0:
        index += arr.shape[axis]
    if index < 0 or index >= arr.shape[axis]:
        raise IndexError(
            f"Slice index {index} is out of bounds for axis {axis} "
            f"with size {arr.shape[axis]}."
        )

    if axis == 0:
        return arr[index, :, :], plane_label, index
    if axis == 1:
        return arr[:, index, :], plane_label, index
    return arr[:, :, index], plane_label, index


def _coord_edges(values):
    coord = np.asarray(values, dtype=float)
    if coord.size == 0:
        return (0.0, 1.0)
    if coord.size == 1:
        width = 1.0
        return (float(coord[0] - 0.5 * width), float(coord[0] + 0.5 * width))
    deltas = np.diff(coord)
    return (float(coord[0] - 0.5 * deltas[0]), float(coord[-1] + 0.5 * deltas[-1]))


def _plane_axis_and_label(plane):
    plane_key = str(plane).lower()
    if plane_key in {"xy", "z"}:
        return "z", "xy"
    if plane_key in {"xz", "y"}:
        return "y", "xz"
    if plane_key in {"yz", "x"}:
        return "x", "yz"
    raise ValueError("plane must be one of 'xy'/'z', 'xz'/'y', or 'yz'/'x'.")


def _select_xarray_frame(da, *, time_index=-1, t=None, frame=None, method="nearest"):
    if t is not None and "t" in da.dims:
        selected = da.sel(t=float(t), method=method)
        return selected, float(selected.coords["t"]), "t"
    frame_dim = "t" if "t" in da.dims else "frame" if "frame" in da.dims else None
    if frame_dim is None:
        return da, None, None
    idx = int(frame if frame is not None else time_index)
    selected = da.isel({frame_dim: idx})
    if frame_dim == "frame" and frame is None:
        value = idx
    elif frame_dim in selected.coords:
        try:
            value = float(selected.coords[frame_dim])
        except Exception:
            value = idx
    else:
        value = idx
    return selected, value, frame_dim


def _select_xarray_plane(da, *, plane="z", index=None, method="nearest"):
    axis, plane_label = _plane_axis_and_label(plane)
    if axis not in da.dims:
        if da.ndim == 2:
            return da, "xy", None
        raise ValueError(f"Cannot select {plane_label} plane from dims {da.dims}.")
    coord = np.asarray(da.coords[axis], dtype=float) if axis in da.coords else None
    if index is None:
        value = (
            float(coord[coord.size // 2])
            if coord is not None and coord.size
            else da.sizes[axis] // 2
        )
    else:
        value = float(index)
    if coord is not None:
        selected = da.sel({axis: value}, method=method)
        actual = float(selected.coords[axis])
    else:
        selected = da.isel({axis: int(value)})
        actual = int(value)
    return selected, plane_label, actual


def _xarray_2d_extent(da):
    dims = tuple(da.dims)
    if len(dims) != 2:
        raise ValueError(f"Expected 2D field after slicing, got dims {dims}.")
    y_dim, x_dim = dims
    x0, x1 = (
        _coord_edges(da.coords[x_dim]) if x_dim in da.coords else (0.0, da.sizes[x_dim])
    )
    y0, y1 = (
        _coord_edges(da.coords[y_dim]) if y_dim in da.coords else (0.0, da.sizes[y_dim])
    )
    return (x0, x1, y0, y1), x_dim, y_dim


def simulation_field_plot_data(
    results,
    *,
    field="Ez",
    time_index=-1,
    t=None,
    frame=None,
    plane="z",
    index=None,
    method="nearest",
):
    """Serialize stored simulation field data into a 2D plot payload."""
    if results.fields is None or field not in results.fields:
        available = [] if results.fields is None else sorted(results.fields)
        raise RuntimeError(
            f"Field '{field}' is not stored. Available stored fields: {available}"
        )

    source = results.fields[field]
    if hasattr(source, "dims") and hasattr(source, "coords"):
        frame_da, selected_time, time_dim = _select_xarray_frame(
            source,
            time_index=time_index,
            t=t,
            frame=frame,
            method=method,
        )
        plane_da, plane_label, selected_index = _select_xarray_plane(
            frame_da,
            plane=plane,
            index=index,
            method=method,
        )
        extent, xlabel, ylabel = _xarray_2d_extent(plane_da)
        scale, unit = get_si_scale_and_label(
            max(abs(extent[1] - extent[0]), abs(extent[3] - extent[2]), 1e-30)
        )
        title = f"{field}"
        if selected_time is not None:
            title += f" {time_dim or 'frame'} {selected_time:g}"
        if selected_index is not None:
            axis, _ = _plane_axis_and_label(plane)
            title += f" ({plane_label}, {axis}={selected_index * scale:g} {unit})"
        return {
            "kind": "simulation_field",
            "field": field,
            "array": np.asarray(plane_da).copy(),
            "plane": plane_label,
            "slice_index": selected_index,
            "time_index": selected_time,
            "extent": tuple(float(v * scale) for v in extent),
            "scale_factor": float(scale),
            "scale_unit": unit,
            "xlabel": f"{xlabel} ({unit})",
            "ylabel": f"{ylabel} ({unit})",
            "title": title,
        }

    frame, selected_time = _select_field_frame(source, time_index=time_index)
    field_2d, plane_label, selected_index = _slice_2d(
        frame,
        plane=plane,
        index=index,
    )

    design = results.simulation.design
    depth = float(getattr(design, "depth", 0.0) or 0.0)
    if plane_label == "xy":
        extent = (0.0, float(design.width), 0.0, float(design.height))
        xlabel, ylabel = "X", "Y"
    elif plane_label == "xz":
        extent = (0.0, float(design.width), 0.0, depth)
        xlabel, ylabel = "X", "Z"
    else:
        extent = (0.0, float(design.height), 0.0, depth)
        xlabel, ylabel = "Y", "Z"

    scale, unit = get_si_scale_and_label(max(extent[1], extent[3], 1e-30))
    title = f"{field}"
    if selected_time is not None:
        title += f" frame {selected_time}"
    if selected_index is not None:
        title += f" ({plane_label}, index {selected_index})"

    return {
        "kind": "simulation_field",
        "field": field,
        "array": np.asarray(field_2d).copy(),
        "plane": plane_label,
        "slice_index": selected_index,
        "time_index": selected_time,
        "extent": tuple(float(v * scale) for v in extent),
        "scale_factor": float(scale),
        "scale_unit": unit,
        "xlabel": f"{xlabel} ({unit})",
        "ylabel": f"{ylabel} ({unit})",
        "title": title,
    }


def simulation_permittivity_plot_data(
    sim,
    *,
    plane="z",
    index=None,
):
    """Serialize simulation permittivity into a 2D plot payload."""
    field_2d, plane_label, selected_index = _slice_2d(
        np.asarray(sim.fields.permittivity),
        plane=plane,
        index=index,
    )

    design = sim.design
    depth = float(getattr(design, "depth", 0.0) or 0.0)
    if plane_label == "xy":
        extent = (0.0, float(design.width), 0.0, float(design.height))
        xlabel, ylabel = "X", "Y"
    elif plane_label == "xz":
        extent = (0.0, float(design.width), 0.0, depth)
        xlabel, ylabel = "X", "Z"
    else:
        extent = (0.0, float(design.height), 0.0, depth)
        xlabel, ylabel = "Y", "Z"

    scale, unit = get_si_scale_and_label(max(extent[1], extent[3], 1e-30))
    title = "Permittivity"
    if selected_index is not None:
        title += f" ({plane_label}, index {selected_index})"

    return {
        "kind": "simulation_permittivity",
        "field": "permittivity",
        "array": np.asarray(field_2d).copy(),
        "plane": plane_label,
        "slice_index": selected_index,
        "extent": tuple(float(v * scale) for v in extent),
        "scale_factor": float(scale),
        "scale_unit": unit,
        "xlabel": f"{xlabel} ({unit})",
        "ylabel": f"{ylabel} ({unit})",
        "title": title,
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
        "layout": (
            layout
            if layout is not None
            else (simulation_plot_data(simulation) if simulation is not None else None)
        ),
    }


__all__ = [
    "boundary_plot_data",
    "design_plot_data",
    "grid_plot_data",
    "mode_profile_data",
    "mode_permittivity_plot_data",
    "monitor_field_plot_data",
    "monitor_plot_data",
    "monitor_power_plot_data",
    "signal_plot_data",
    "simulation_plot_data",
    "simulation_field_plot_data",
    "simulation_permittivity_plot_data",
    "snapshot_payload",
    "source_plot_data",
    "source_signal_plot_data",
    "source_spectrum_plot_data",
    "structure_plot_data",
]
