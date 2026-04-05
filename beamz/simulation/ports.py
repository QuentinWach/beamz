"""Port and monitor selection helpers for simulation analysis."""

from __future__ import annotations

import numpy as np

from beamz.devices.monitors.monitors import Monitor


def safe_ratio(num, den, eps=1e-18):
    out = np.zeros_like(num, dtype=np.complex128)
    valid = np.abs(den) > eps
    out[valid] = num[valid] / den[valid]
    return out


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
