"""Port and monitor selection helpers for simulation analysis."""

from __future__ import annotations

import numpy as np

from beamz.devices.monitors.monitors import Monitor


def safe_ratio(num, den, eps=1e-18):
    out = np.zeros_like(num, dtype=np.complex128)
    valid = np.abs(den) > eps
    out[valid] = num[valid] / den[valid]
    return out


def require_supported_plane(sim, *, function_name, allow_3d):
    if allow_3d:
        if (not sim.is_3d) and sim.plane_2d != "xy":
            raise NotImplementedError(
                f"{function_name} currently supports 2D simulations in the xy plane."
            )
        return
    if sim.is_3d or sim.plane_2d != "xy":
        raise NotImplementedError(
            f"{function_name} currently supports 2D simulations in the xy plane."
        )


def normalize_frequencies(frequencies):
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if freqs.size == 0:
        raise ValueError("frequencies must contain at least one value.")
    if np.any(freqs <= 0):
        raise ValueError("frequencies must be strictly positive.")
    return freqs


def select_wave_component(
    wave_data,
    selector="minus",
    *,
    use_reference=False,
):
    sel = str(selector).lower()
    if sel not in {"plus", "minus", "auto"}:
        raise ValueError(
            f"Unsupported wave selector '{selector}'. "
            "Use one of {'plus', 'minus', 'auto'}."
        )

    if use_reference:
        plus = np.asarray(
            wave_data.get(
                "a_incident_plus",
                wave_data.get("a_incident", wave_data.get("a_plus")),
            ),
            dtype=np.complex128,
        )
        minus = np.asarray(
            wave_data.get("a_incident_minus", wave_data.get("a_minus")),
            dtype=np.complex128,
        )
    else:
        plus = np.asarray(wave_data.get("a_plus"), dtype=np.complex128)
        minus = np.asarray(wave_data.get("a_minus"), dtype=np.complex128)

    if sel == "plus":
        return plus
    if sel == "minus":
        return minus
    return np.where(np.abs(plus) >= np.abs(minus), plus, minus)


def format_s_matrix_output(s_matrix, as_sax):
    """Return S-parameter mapping without requiring optional external packages."""
    if as_sax:
        return dict(s_matrix)
    return s_matrix


def resolve_output_ports(port_map, output_ports):
    if output_ports is None:
        return list(port_map.keys())
    missing = [name for name in output_ports if name not in port_map]
    if missing:
        raise ValueError(f"output_ports contains unknown ports: {missing}")
    return list(output_ports)


def normalize_portspecs(ports, port_cls):
    if isinstance(ports, dict):
        values = list(ports.values())
    else:
        values = list(ports)
    if not values:
        raise ValueError("ports must contain at least one PortSpec.")

    normalized = {}
    for item in values:
        if isinstance(item, port_cls):
            spec = item
        else:
            spec = port_cls(
                name=item["name"],
                monitor_name=item["monitor_name"],
                direction=item["direction"],
                polarization=item["polarization"],
                mode_index=int(item.get("mode_index", 0)),
                reference_monitor=item.get("reference_monitor"),
                incident_wave=str(item.get("incident_wave", "plus")).lower(),
                scattered_wave=str(item.get("scattered_wave", "minus")).lower(),
            )
        if spec.direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
            raise ValueError(f"Unsupported port direction '{spec.direction}'.")
        pol = str(spec.polarization).lower()
        if pol not in {"tm", "te"}:
            raise ValueError(f"Unsupported polarization '{spec.polarization}'.")
        inc_wave = str(spec.incident_wave).lower()
        scat_wave = str(spec.scattered_wave).lower()
        if inc_wave not in {"plus", "minus", "auto"}:
            raise ValueError(
                f"Unsupported incident_wave '{spec.incident_wave}' for port '{spec.name}'."
            )
        if scat_wave not in {"plus", "minus", "auto"}:
            raise ValueError(
                f"Unsupported scattered_wave '{spec.scattered_wave}' for port '{spec.name}'."
            )
        normalized[spec.name] = port_cls(
            name=spec.name,
            monitor_name=spec.monitor_name,
            direction=spec.direction,
            polarization=pol,
            mode_index=int(spec.mode_index),
            reference_monitor=spec.reference_monitor,
            incident_wave=inc_wave,
            scattered_wave=scat_wave,
        )
    return normalized


def named_monitors(devices):
    return {
        device.name: device
        for device in devices
        if isinstance(device, Monitor) and getattr(device, "name", None)
    }


def validate_port_monitors(port_map, monitor_by_name, *, require_dft=False):
    for spec in port_map.values():
        main = monitor_by_name.get(spec.monitor_name)
        if main is None:
            raise ValueError(
                f"Missing monitor '{spec.monitor_name}' for port '{spec.name}'."
            )
        if require_dft and not getattr(main, "dft_enabled", False):
            raise ValueError(
                f"Monitor '{spec.monitor_name}' must be created with dft_enabled=True."
            )
        if spec.reference_monitor:
            ref = monitor_by_name.get(spec.reference_monitor)
            if ref is None:
                raise ValueError(
                    f"Missing reference monitor '{spec.reference_monitor}' for port '{spec.name}'."
                )
            if require_dft and not getattr(ref, "dft_enabled", False):
                raise ValueError(
                    f"Reference monitor '{spec.reference_monitor}' must have dft_enabled=True."
                )


def mode_components_for_port(spec):
    axis = spec.direction[1]
    tm_map = {
        "x": ("Ez", "Hy", 2, 1, -1.0),
        "y": ("Ez", "Hx", 2, 0, 1.0),
        "z": ("Ey", "Hx", 1, 0, -1.0),
    }
    te_map = {
        "x": ("Ey", "Hz", 1, 2, 1.0),
        "y": ("Ex", "Hz", 0, 2, -1.0),
        "z": ("Ex", "Hy", 0, 1, 1.0),
    }
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported port axis '{axis}'.")
    e_comp, h_comp, e_idx, h_idx, sign = (
        tm_map[axis] if spec.polarization == "tm" else te_map[axis]
    )
    proj_components_3d = {
        "x": ("Ey", "Ez", "Hy", "Hz"),
        "y": ("Ex", "Ez", "Hx", "Hz"),
        "z": ("Ex", "Ey", "Hx", "Hy"),
    }[axis]
    return {
        "axis": axis,
        "e_component": e_comp,
        "h_component": h_comp,
        "e_mode_index": e_idx,
        "h_mode_index": h_idx,
        "signed_flux_sign": sign,
        "projection_components": (e_comp, h_comp),
        "projection_components_3d": proj_components_3d,
    }


def build_port_wave_data(
    a_plus,
    a_minus,
    *,
    return_power,
    condition_number=None,
    mode_neff=None,
):
    data = {"a_plus": a_plus, "a_minus": a_minus}
    if condition_number is not None:
        data["condition_number"] = condition_number
    if mode_neff is not None:
        data["mode_neff"] = mode_neff
    if return_power:
        data["P_plus"] = np.abs(a_plus) ** 2
        data["P_minus"] = np.abs(a_minus) ** 2
    return data


def add_reference_power(data):
    data["P_incident"] = np.abs(data["a_incident_plus"]) ** 2
    data["P_incident_plus"] = np.abs(data["a_incident_plus"]) ** 2
    data["P_incident_minus"] = np.abs(data["a_incident_minus"]) ** 2


def guided_output_power(sim, waves, port_map, output_ports):
    p_guided_out = None
    for out_port in output_ports:
        out_spec = port_map[out_port]
        component = np.abs(
            sim._select_wave_component(
                waves[out_port],
                selector=out_spec.scattered_wave,
                use_reference=False,
            )
        ) ** 2
        component = np.atleast_1d(np.asarray(component, dtype=float))
        if p_guided_out is None:
            p_guided_out = np.zeros_like(component, dtype=float)
        p_guided_out += component
    return p_guided_out


def condition_number_summary(waves):
    return {
        name: {
            "monitor": np.asarray(data.get("condition_number", []), dtype=float),
            "reference": np.asarray(
                data.get("reference_condition_number", []), dtype=float
            ),
        }
        for name, data in waves.items()
    }


def incident_valid_mask(a_incident, min_incident_db):
    max_incident = float(np.max(np.abs(a_incident))) if a_incident.size else 0.0
    rel_floor = max_incident * (10.0 ** (float(min_incident_db) / 20.0))
    return np.abs(a_incident) >= max(1e-18, rel_floor)


def resolve_source_frequencies(port_map, source_port, frequencies, *, resolver):
    if frequencies is not None:
        return normalize_frequencies(frequencies)
    if source_port not in port_map:
        raise ValueError(f"source_port '{source_port}' not found in ports.")
    return normalize_frequencies(resolver(port_map[source_port]))


def assemble_s_matrix(
    sim,
    *,
    source_port,
    port_map,
    waves,
    output_ports,
    frequencies,
    as_sax,
    return_diagnostics,
    min_incident_db=None,
    scalar_output=False,
    include_condition_numbers=False,
):
    output_ports = resolve_output_ports(port_map, output_ports)
    source_spec = port_map[source_port]
    a_incident = np.atleast_1d(
        np.asarray(
            sim._select_wave_component(
                waves[source_port],
                selector=source_spec.incident_wave,
                use_reference=bool(source_spec.reference_monitor),
            ),
            dtype=np.complex128,
        )
    )

    valid_mask = (
        incident_valid_mask(a_incident, min_incident_db)
        if min_incident_db is not None
        else None
    )

    s_matrix = {}
    for out_port in output_ports:
        out_spec = port_map[out_port]
        b_out = np.atleast_1d(
            np.asarray(
                sim._select_wave_component(
                    waves[out_port],
                    selector=out_spec.scattered_wave,
                    use_reference=False,
                ),
                dtype=np.complex128,
            )
        )
        ratio = sim._safe_ratio(b_out, a_incident)
        if valid_mask is not None:
            ratio = np.where(valid_mask, ratio, 0.0 + 0.0j)
        s_matrix[(out_port, source_port)] = (
            np.complex128(ratio[0]) if scalar_output else ratio
        )

    freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
    sim.s_matrix_frequencies = freq_arr
    s_output = sim._format_s_matrix_output(s_matrix, as_sax=as_sax)
    if not return_diagnostics:
        return s_output

    p_in = np.abs(a_incident) ** 2
    p_guided_out = guided_output_power(sim, waves, port_map, output_ports)
    power_sum = p_guided_out / np.maximum(p_in, 1e-18)
    loss_est = 1.0 - power_sum

    if valid_mask is not None:
        power_sum = np.where(valid_mask, power_sum, np.nan)
        loss_est = np.where(valid_mask, loss_est, np.nan)

    diagnostics = {
        "source_port": source_port,
        "output_ports": output_ports,
        "waves": waves,
    }
    if scalar_output:
        diagnostics.update(
            {
                "frequency": float(freq_arr[0]),
                "P_in": float(p_in[0]),
                "P_guided_out": float(p_guided_out[0]),
                "power_sum": float(power_sum[0]),
                "loss_est": float(loss_est[0]),
            }
        )
    else:
        diagnostics.update(
            {
                "frequencies": freq_arr,
                "P_in": p_in,
                "P_guided_out": p_guided_out,
                "power_sum": power_sum,
                "loss_est": loss_est,
            }
        )
        if valid_mask is not None:
            diagnostics["valid_mask"] = valid_mask
        if include_condition_numbers:
            diagnostics["condition_numbers"] = condition_number_summary(waves)

    return {"s_matrix": s_output, "diagnostics": diagnostics}
