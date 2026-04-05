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
