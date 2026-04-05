from typing import Optional

import numpy as np


def evaluate_objective(monitor) -> Optional[float]:
    """Evaluate the monitor objective function, if any."""
    if monitor.objective_function is None:
        return None
    try:
        value = monitor.objective_function(monitor)
    except Exception as exc:
        print(f"Warning: monitor objective evaluation failed: {exc}")
        return None
    if value is None:
        return None
    try:
        monitor.objective_value = float(value)
    except (TypeError, ValueError):
        print(f"Warning: monitor objective returned non-numeric value: {value}")
        return None
    return monitor.objective_value


def save_data(monitor, filename, format="npz"):
    """Save recorded monitor data to disk."""
    if format == "npz":
        np.savez(
            filename,
            fields=monitor.fields,
            power_history=monitor.power_history,
            power_timestamps=monitor.power_timestamps,
            frequency_points=monitor.frequency_points,
            frequency_flux_spectrum=monitor.frequency_flux_spectrum,
            monitor_info={"type": monitor.monitor_type, "is_3d": monitor.is_3d},
        )
        return
    raise ValueError(f"Unsupported format: {format}")


def load_data(monitor, filename):
    """Load recorded monitor data from disk."""
    data = np.load(filename, allow_pickle=True)
    monitor.fields = data["fields"].item()
    monitor.power_history = list(data["power_history"])
    if "power_timestamps" in data:
        monitor.power_timestamps = list(data["power_timestamps"])
    else:
        monitor.power_timestamps = list(range(len(monitor.power_history)))
    if "frequency_points" in data:
        monitor.frequency_points = np.asarray(data["frequency_points"], dtype=np.float64)
    if "frequency_flux_spectrum" in data:
        monitor.frequency_flux_spectrum = np.asarray(
            data["frequency_flux_spectrum"], dtype=np.complex64
        )


def _field_time_index(monitor, time_value=None, time_index=None):
    if time_index is not None:
        return int(time_index)
    if time_value is not None and monitor.fields["t"]:
        times = np.asarray(monitor.fields["t"], dtype=float)
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

    data = record_helpers.field_at_time(
        monitor, field=field, time_value=time_value, time_index=time_index
    )
    if data is None:
        raise ValueError(f"No recorded data for field '{field}'.")

    idx = _field_time_index(monitor, time_value=time_value, time_index=time_index)
    times = np.asarray(monitor.fields.get("t", []), dtype=float)
    title = field if times.size == 0 else f"{field} at t={times[idx]:.3e} s"

    arr = np.asarray(data)
    if monitor.monitor_type == "line" or arr.ndim == 1:
        return Trace1D(
            values=arr.reshape(-1),
            coords=_line_coords(monitor, arr.size),
            coord_label="position",
            value_label=f"{field} amplitude",
            title=title,
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
    )


def power_trace(monitor, *, db_scale=False):
    """Return monitor power history as plotting-ready data."""
    from beamz.visual.data import Trace1D

    if not monitor.power_history:
        raise ValueError("No power data recorded.")
    values = np.asarray(monitor.power_history, dtype=float)
    if db_scale:
        values = 10.0 * np.log10(np.maximum(values, 1e-12))
        value_label = "Power (dB)"
    else:
        value_label = "Power"
    coords = (
        np.asarray(monitor.power_timestamps, dtype=float)
        if monitor.power_timestamps
        else np.arange(values.size, dtype=float)
    )
    return Trace1D(
        values=values,
        coords=coords,
        coord_label="time",
        value_label=value_label,
        title="Power vs time",
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
    )


def plot_fields(monitor, **kwargs):
    """Delegate field plotting to the visualization layer."""
    from beamz.visual.monitor_plots import plot_monitor_fields

    return plot_monitor_fields(monitor, **kwargs)


def plot_power(monitor, **kwargs):
    """Delegate power plotting to the visualization layer."""
    from beamz.visual.monitor_plots import plot_monitor_power

    return plot_monitor_power(monitor, **kwargs)


def animate_fields(monitor, **kwargs):
    """Delegate field animation to the visualization layer."""
    from beamz.visual.monitor_plots import animate_monitor_fields

    return animate_monitor_fields(monitor, **kwargs)


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
    kwargs = dict(
        design=monitor.design,
        start=monitor.start,
        record_fields=monitor.should_record_fields,
        accumulate_power=monitor.accumulate_power,
        live_update=monitor.live_update,
        record_interval=monitor.record_interval,
        max_history_steps=monitor.max_history_steps,
        dft_frequencies=monitor.dft_frequencies.copy(),
        dft_t_start=monitor.dft_t_start,
        dft_t_end=monitor.dft_t_end,
        dft_enabled=monitor.dft_enabled,
        dft_components=monitor.dft_components,
        dft_record_every_step=monitor.dft_record_every_step,
        dft_record_interval=monitor.dft_record_interval,
        dft_window=monitor.dft_window,
    )
    if monitor.is_3d:
        if hasattr(monitor, "end") and monitor.end is not None:
            kwargs["end"] = monitor.end
        else:
            kwargs["plane_normal"] = monitor.plane_normal
            kwargs["plane_position"] = monitor.plane_position
            kwargs["size"] = monitor.size
    else:
        kwargs["end"] = monitor.end
    return monitor.__class__(**kwargs)
