"""Field recording, power accumulation, and statistics helpers for monitors."""

import numpy as np


def should_record(monitor, step):
    """Check if this step should be recorded based on interval."""
    return (step - monitor.last_record_step) >= monitor.record_interval


def record_fields_2d(
    monitor, Ez, Hx, Hy, t, dx, dy, step=0, Ex=None, Ey=None, Hz=None
):
    """Record 2D field data."""
    do_record = monitor.should_record(step)
    do_dft = monitor._dft_should_accumulate(step, t)
    if not do_record and not do_dft:
        return
    grid_points = monitor.get_grid_points_2d(dx, dy)
    Ez_values, Hx_values, Hy_values = [], [], []
    Ex_values, Ey_values, Hz_values = [], [], []
    for x_idx, y_idx in grid_points:
        if 0 <= y_idx < Ez.shape[0] and 0 <= x_idx < Ez.shape[1]:
            val = Ez[y_idx, x_idx]
            Ez_values.append(complex(val) if np.iscomplexobj(val) else float(val))
        else:
            Ez_values.append(0.0)
        if 0 <= y_idx < Hx.shape[0] and 0 <= x_idx < Hx.shape[1]:
            val = Hx[y_idx, x_idx]
            Hx_values.append(complex(val) if np.iscomplexobj(val) else float(val))
        else:
            Hx_values.append(0.0)
        if 0 <= y_idx < Hy.shape[0] and 0 <= x_idx < Hy.shape[1]:
            val = Hy[y_idx, x_idx]
            Hy_values.append(complex(val) if np.iscomplexobj(val) else float(val))
        else:
            Hy_values.append(0.0)
        if Ex is not None and 0 <= y_idx < Ex.shape[0] and 0 <= x_idx < Ex.shape[1]:
            val = Ex[y_idx, x_idx]
            Ex_values.append(complex(val) if np.iscomplexobj(val) else float(val))
        else:
            Ex_values.append(0.0)
        if Ey is not None and 0 <= y_idx < Ey.shape[0] and 0 <= x_idx < Ey.shape[1]:
            val = Ey[y_idx, x_idx]
            Ey_values.append(complex(val) if np.iscomplexobj(val) else float(val))
        else:
            Ey_values.append(0.0)
        if Hz is not None and 0 <= y_idx < Hz.shape[0] and 0 <= x_idx < Hz.shape[1]:
            val = Hz[y_idx, x_idx]
            Hz_values.append(complex(val) if np.iscomplexobj(val) else float(val))
        else:
            Hz_values.append(0.0)

    if do_record and monitor.should_record_fields:
        monitor.fields["Ex"].append(Ex_values)
        monitor.fields["Ey"].append(Ey_values)
        monitor.fields["Ez"].append(Ez_values)
        monitor.fields["Hx"].append(Hx_values)
        monitor.fields["Hy"].append(Hy_values)
        monitor.fields["Hz"].append(Hz_values)
        monitor.fields["t"].append(t)

    if do_dft:
        dft_components = monitor.dft_components or ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        vectors = {}
        if "Ex" in dft_components:
            vectors["Ex"] = Ex_values
        if "Ey" in dft_components:
            vectors["Ey"] = Ey_values
        if "Ez" in dft_components:
            vectors["Ez"] = Ez_values
        if "Hx" in dft_components:
            vectors["Hx"] = Hx_values
        if "Hy" in dft_components:
            vectors["Hy"] = Hy_values
        if "Hz" in dft_components:
            vectors["Hz"] = Hz_values
        monitor._update_dft(t, vectors)

    if do_record and monitor.accumulate_power:
        calculate_power_2d(monitor, Ez_values, Hx_values, Hy_values, t, dx, dy)

    if do_record:
        monitor.last_record_step = step
    manage_memory(monitor)

    if (
        do_record
        and monitor.live_update
        and (len(monitor.fields["t"]) % monitor.update_interval == 0)
    ):
        monitor._update_live_plot_2d()


def record_fields_3d(monitor, Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy, dz, step=0):
    """Record 3D field data from plane slice."""
    do_record = monitor.should_record(step)
    do_dft = monitor._dft_should_accumulate(step, t)
    if not do_record and not do_dft:
        return

    def slice_field(arr):
        z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(dx, dy, dz, arr.shape)
        nz, ny, nx = arr.shape

        def clamp(idx, limit):
            if isinstance(idx, int):
                return min(max(0, idx), limit - 1)
            start = max(0, min(idx.start if idx.start is not None else 0, limit - 1))
            stop = max(start, min(idx.stop if idx.stop is not None else limit, limit))
            return slice(start, stop)

        return arr[clamp(z_idx, nz), clamp(y_idx, ny), clamp(x_idx, nx)].copy()

    Ex_slice = slice_field(Ex)
    Ey_slice = slice_field(Ey)
    Ez_slice = slice_field(Ez)
    Hx_slice = slice_field(Hx)
    Hy_slice = slice_field(Hy)
    Hz_slice = slice_field(Hz)

    min_dim0 = min(
        Ex_slice.shape[0],
        Ey_slice.shape[0],
        Ez_slice.shape[0],
        Hx_slice.shape[0],
        Hy_slice.shape[0],
        Hz_slice.shape[0],
    )
    min_dim1 = min(
        Ex_slice.shape[1],
        Ey_slice.shape[1],
        Ez_slice.shape[1],
        Hx_slice.shape[1],
        Hy_slice.shape[1],
        Hz_slice.shape[1],
    )

    Ex_slice = Ex_slice[:min_dim0, :min_dim1]
    Ey_slice = Ey_slice[:min_dim0, :min_dim1]
    Ez_slice = Ez_slice[:min_dim0, :min_dim1]
    Hx_slice = Hx_slice[:min_dim0, :min_dim1]
    Hy_slice = Hy_slice[:min_dim0, :min_dim1]
    Hz_slice = Hz_slice[:min_dim0, :min_dim1]

    if do_record and monitor.should_record_fields:
        monitor.fields["Ex"].append(Ex_slice)
        monitor.fields["Ey"].append(Ey_slice)
        monitor.fields["Ez"].append(Ez_slice)
        monitor.fields["Hx"].append(Hx_slice)
        monitor.fields["Hy"].append(Hy_slice)
        monitor.fields["Hz"].append(Hz_slice)
        monitor.fields["t"].append(t)

    if do_dft:
        dft_components = monitor.dft_components or ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        vectors = {}
        if "Ex" in dft_components:
            vectors["Ex"] = Ex_slice.reshape(-1)
        if "Ey" in dft_components:
            vectors["Ey"] = Ey_slice.reshape(-1)
        if "Ez" in dft_components:
            vectors["Ez"] = Ez_slice.reshape(-1)
        if "Hx" in dft_components:
            vectors["Hx"] = Hx_slice.reshape(-1)
        if "Hy" in dft_components:
            vectors["Hy"] = Hy_slice.reshape(-1)
        if "Hz" in dft_components:
            vectors["Hz"] = Hz_slice.reshape(-1)
        monitor._update_dft(t, vectors)

    if do_record and monitor.accumulate_power:
        calculate_power_3d(monitor, Ex_slice, Ey_slice, Ez_slice, Hx_slice, Hy_slice, Hz_slice, t, dx, dy)

    if do_record:
        monitor.last_record_step = step
    manage_memory(monitor)

    if (
        do_record
        and monitor.live_update
        and (len(monitor.fields["t"]) % monitor.update_interval == 0)
    ):
        monitor._update_live_plot_3d()


def record_fields(monitor, *args, **kwargs):
    """Delegate to 2D or 3D field recording."""
    if monitor.is_3d and len(args) >= 6:
        record_fields_3d(monitor, *args, **kwargs)
    else:
        record_fields_2d(monitor, *args, **kwargs)


def calculate_power_2d(monitor, Ez_values, Hx_values, Hy_values, t, dx, dy):
    """Calculate Poynting vector magnitude for 2D fields."""
    Ez_array = np.array(Ez_values)
    Hx_array = np.array(Hx_values)
    Hy_array = np.array(Hy_values)
    Sx = -Ez_array * Hy_array
    Sy = Ez_array * Hx_array
    power_mag = np.sqrt(Sx**2 + Sy**2)
    total_power = np.sum(power_mag) * dx * dy
    if monitor.power_accumulated is None:
        monitor.power_accumulated = power_mag
    else:
        monitor.power_accumulated += power_mag
    monitor.power_history.append(total_power)
    monitor.power_timestamps.append(float(t))
    monitor.power_accumulation_count += 1


def calculate_power_3d(monitor, Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy):
    """Calculate Poynting vector magnitude for 3D fields."""
    Sx = Ey * Hz - Ez * Hy
    Sy = Ez * Hx - Ex * Hz
    Sz = Ex * Hy - Ey * Hx
    power_mag = np.sqrt(Sx**2 + Sy**2 + Sz**2)
    total_power = np.sum(power_mag) * dx * dy
    if monitor.power_accumulated is None:
        monitor.power_accumulated = power_mag.copy()
    else:
        monitor.power_accumulated += power_mag
    monitor.power_history.append(total_power)
    monitor.power_timestamps.append(float(t))
    monitor.power_accumulation_count += 1


def manage_memory(monitor):
    """Manage history limits for recorded monitor data."""
    if monitor.max_history_steps is None:
        return
    for field_name in monitor.fields:
        if len(monitor.fields[field_name]) > monitor.max_history_steps:
            excess = len(monitor.fields[field_name]) - monitor.max_history_steps
            monitor.fields[field_name] = monitor.fields[field_name][excess:]
    if len(monitor.power_history) > monitor.max_history_steps:
        excess = len(monitor.power_history) - monitor.max_history_steps
        monitor.power_history = monitor.power_history[excess:]
        monitor.power_timestamps = monitor.power_timestamps[excess:]


def field_statistics(monitor):
    """Get statistical information about recorded fields."""
    if not monitor.fields["t"]:
        return {}
    stats = {
        "total_records": len(monitor.fields["t"]),
        "time_span": (
            monitor.fields["t"][-1] - monitor.fields["t"][0]
            if len(monitor.fields["t"]) > 1
            else 0
        ),
        "avg_power": np.mean(monitor.power_history) if monitor.power_history else 0,
        "max_power": np.max(monitor.power_history) if monitor.power_history else 0,
        "monitor_type": monitor.monitor_type,
        "is_3d": monitor.is_3d,
    }
    if monitor.is_3d:
        stats["plane_normal"] = monitor.plane_normal
        stats["plane_position"] = monitor.plane_position
        stats["plane_size"] = monitor.size
    else:
        stats["line_start"] = monitor.start
        stats["line_end"] = monitor.end
    return stats


def field_at_time(monitor, field="Ez", time_value=None, time_index=None):
    """Get field data at a specific time."""
    if not monitor.fields["t"] or field not in monitor.fields:
        return None
    if time_index is not None:
        if 0 <= time_index < len(monitor.fields[field]):
            return monitor.fields[field][time_index]
        return None
    if time_value is not None:
        times = np.array(monitor.fields["t"])
        time_index = np.argmin(np.abs(times - time_value))
        return monitor.fields[field][time_index]
    return monitor.fields[field][-1] if monitor.fields[field] else None


def power_statistics(monitor):
    """Get power statistics from recorded data."""
    if not monitor.power_history:
        return {}
    power_array = np.array(monitor.power_history)
    mean_power = np.mean(power_array)
    return {
        "mean_power": mean_power,
        "max_power": np.max(power_array),
        "min_power": np.min(power_array),
        "std_power": np.std(power_array),
        "total_energy": np.sum(power_array),
        "peak_to_average_ratio": (np.max(power_array) / mean_power) if mean_power > 0 else 0,
    }


def signed_flux_trace(monitor, normal_direction, field_pair=None):
    """Return signed directional flux trace from recorded field components."""
    direction = str(normal_direction).lower()
    if direction not in {"+x", "-x", "+y", "-y"}:
        raise ValueError(
            f"normal_direction must be one of ['+x','-x','+y','-y'], got {normal_direction!r}"
        )
    axis = direction[1]
    dir_sign = 1.0 if direction.startswith("+") else -1.0

    if field_pair is None:
        if axis == "x":
            e_comp, h_comp, base_sign = "Ez", "Hy", -1.0
        else:
            e_comp, h_comp, base_sign = "Ez", "Hx", 1.0
    else:
        if len(field_pair) != 2:
            raise ValueError("field_pair must be a tuple like ('Ez', 'Hy').")
        e_comp, h_comp = field_pair
        base_sign = 1.0

    if e_comp not in monitor.fields or h_comp not in monitor.fields:
        raise ValueError(
            f"Requested components ({e_comp}, {h_comp}) are not recorded by this monitor."
        )
    if not monitor.fields[e_comp] or not monitor.fields[h_comp]:
        raise ValueError(
            f"No recorded data for ({e_comp}, {h_comp}) on monitor '{monitor.name}'."
        )

    e_arr = np.asarray(monitor.fields[e_comp], dtype=np.complex128)
    h_arr = np.asarray(monitor.fields[h_comp], dtype=np.complex128)
    if e_arr.ndim == 1:
        e_arr = e_arr[:, None]
    if h_arr.ndim == 1:
        h_arr = h_arr[:, None]

    n_t = min(e_arr.shape[0], h_arr.shape[0])
    n_p = min(e_arr.shape[1], h_arr.shape[1])
    signed_density = base_sign * np.real(
        e_arr[:n_t, :n_p] * np.conjugate(h_arr[:n_t, :n_p])
    )
    return dir_sign * np.sum(signed_density, axis=1)
