from typing import Optional

import numpy as np


def evaluate_objective(monitor) -> Optional[float]:
    """Evaluate the monitor objective function, if any."""
    spec = monitor.spec
    state = monitor.state
    if spec.objective_function is None:
        return None
    try:
        value = spec.objective_function(monitor)
    except Exception as exc:
        print(f"Warning: monitor objective evaluation failed: {exc}")
        return None
    if value is None:
        return None
    try:
        state.objective_value = float(value)
    except (TypeError, ValueError):
        print(f"Warning: monitor objective returned non-numeric value: {value}")
        return None
    return state.objective_value


def save_data(monitor, filename, format="npz"):
    """Save recorded monitor data to disk."""
    state = monitor.state
    if format == "npz":
        np.savez(
            filename,
            fields=state.fields,
            power_history=state.power_history,
            power_timestamps=state.power_timestamps,
            frequency_points=monitor.spec.frequency_points,
            frequency_flux_spectrum=state.frequency_flux_spectrum,
            monitor_info={"type": monitor.monitor_type, "is_3d": monitor.is_3d},
        )
        return
    raise ValueError(f"Unsupported format: {format}")


def load_data(monitor, filename):
    """Load recorded monitor data from disk."""
    state = monitor.state
    data = np.load(filename, allow_pickle=True)
    state.fields = data["fields"].item()
    state.power_history = list(data["power_history"])
    if "power_timestamps" in data:
        state.power_timestamps = list(data["power_timestamps"])
    else:
        state.power_timestamps = list(range(len(state.power_history)))
    if "frequency_flux_spectrum" in data:
        state.frequency_flux_spectrum = np.asarray(
            data["frequency_flux_spectrum"], dtype=np.complex64
        )


def _field_time_index(monitor, time_value=None, time_index=None):
    state = monitor.state
    if time_index is not None:
        idx = int(time_index)
        count = len(state.fields.get("t", []))
        if idx < 0:
            idx += count
        return idx
    if time_value is not None and state.fields["t"]:
        times = np.asarray(state.fields["t"], dtype=float)
        return int(np.argmin(np.abs(times - float(time_value))))
    return -1


def _line_coords(monitor, count):
    start = np.asarray(monitor.start, dtype=float)
    end = np.asarray(monitor.end, dtype=float)
    distance = float(np.linalg.norm(end - start))
    if count <= 1:
        return np.zeros((max(count, 1),), dtype=float)
    return np.linspace(0.0, distance, int(count), dtype=float)


def _plane_spec(monitor):
    normal = str(monitor.plane_normal).lower()
    start = np.asarray(monitor.start, dtype=float)
    end = np.asarray(monitor.end, dtype=float)
    if normal == "z":
        return "xy", (start[0], end[0], start[1], end[1]), "x", "y"
    if normal == "x":
        return "yz", (start[1], end[1], start[2], end[2]), "y", "z"
    if normal == "y":
        return "xz", (start[0], end[0], start[2], end[2]), "x", "z"
    return "xy", (start[0], end[0], start[1], end[1]), "x", "y"


def field_snapshot(monitor, field="Ez", time_value=None, time_index=None):
    """Return the latest or selected monitor field as plotting-ready data."""
    from beamz.visual.data import Slice2D, Trace1D
    from beamz.devices.monitors import record as record_helpers

    idx = _field_time_index(monitor, time_value=time_value, time_index=time_index)
    data = record_helpers.field_at_time(
        monitor,
        field=field,
        time_value=time_value,
        time_index=(idx if idx >= 0 else None),
    )
    if data is None:
        raise ValueError(f"No recorded data for field '{field}'.")

    times = np.asarray(monitor.fields.get("t", []), dtype=float)
    if times.size == 0:
        title = field
    else:
        if idx < 0:
            idx = times.size - 1
        title = f"{field} at t={times[idx]:.3e} s"

    arr = np.asarray(data)
    if monitor.monitor_type == "line" or arr.ndim == 1:
        return Trace1D(
            values=arr.reshape(-1),
            coords=_line_coords(monitor, arr.size),
            coord_label="position",
            value_label=f"{field} amplitude",
            title=title,
            style={"color": "tab:blue", "linewidth": 2},
        )

    plane, extent, x_label, y_label = _plane_spec(monitor)
    return Slice2D(
        values=arr,
        extent=extent,
        value_label=f"{field} amplitude",
        plane=plane,
        title=title,
        x_label=x_label,
        y_label=y_label,
        style={"cmap": "RdBu", "origin": "lower", "aspect": "auto"},
    )


def power_trace(monitor, *, db_scale=False):
    """Return monitor power history as plotting-ready data."""
    from beamz.visual.data import Trace1D

    state = monitor.state
    if not state.power_history:
        raise ValueError("No power data recorded.")
    values = np.asarray(state.power_history, dtype=float)
    if db_scale:
        values = 10.0 * np.log10(np.maximum(values, 1e-12))
        value_label = "Power (dB)"
    else:
        value_label = "Power"
    coords = (
        np.asarray(state.power_timestamps, dtype=float)
        if state.power_timestamps
        else np.arange(values.size, dtype=float)
    )
    return Trace1D(
        values=values,
        coords=coords,
        coord_label="time",
        value_label=value_label,
        title="Power vs time",
        style={"color": "tab:red", "linewidth": 2},
    )


def flux_trace(monitor, normal_direction, field_pair=None):
    """Return signed directional flux history as plotting-ready data."""
    from beamz.visual.data import Trace1D
    from beamz.devices.monitors import record as record_helpers

    values = record_helpers.signed_flux_trace(
        monitor, normal_direction, field_pair=field_pair
    )
    coords = (
        np.asarray(monitor.fields.get("t", []), dtype=float)
        if monitor.fields.get("t")
        else np.arange(len(values), dtype=float)
    )
    return Trace1D(
        values=np.asarray(values, dtype=float),
        coords=coords,
        coord_label="time",
        value_label=f"Flux ({normal_direction})",
        title=f"Signed flux {normal_direction}",
        style={"color": "tab:purple", "linewidth": 2},
    )


def describe(monitor):
    """Return a concise monitor summary."""
    if not monitor.fields["t"]:
        return (
            f"Monitor: {monitor.monitor_type} "
            f"({'3D' if monitor.is_3d else '2D'}), 0 records"
        )
    stats = monitor.get_field_statistics()
    return (
        f"Monitor: {stats['monitor_type']} "
        f"({'3D' if stats['is_3d'] else '2D'}), "
        f"{stats['total_records']} records"
    )


def copy_monitor(monitor):
    """Create a deep copy of the monitor configuration."""
    spec = monitor.spec
    kwargs = dict(
        design=monitor.design,
        start=spec.start,
        record_fields=spec.should_record_fields,
        accumulate_power=spec.accumulate_power,
        live_update=spec.live_update,
        record_interval=spec.record_interval,
        max_history_steps=spec.max_history_steps,
        dft_frequencies=spec.dft_frequencies.copy(),
        dft_t_start=spec.dft_t_start,
        dft_t_end=spec.dft_t_end,
        dft_enabled=spec.dft_enabled,
        dft_components=spec.dft_components,
        dft_record_every_step=spec.dft_record_every_step,
        dft_record_interval=spec.dft_record_interval,
        dft_window=spec.dft_window,
        objective_function=spec.objective_function,
        name=spec.name,
        frequency_points=spec.frequency_points.copy(),
        frequency_record_interval=spec.frequency_record_interval,
    )
    if spec.is_3d:
        if spec.end is not None:
            kwargs["end"] = spec.end
        else:
            kwargs["plane_normal"] = spec.plane_normal
            kwargs["plane_position"] = spec.plane_position
            kwargs["size"] = spec.size
    else:
        kwargs["end"] = spec.end
    return monitor.__class__(**kwargs)
