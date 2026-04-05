"""Spectral and demodulation helpers for simulation analysis."""

from __future__ import annotations

import numpy as np


def get_monitor_trace(sim, monitor, field_component="Ez", reduction="mean"):
    """Reduce monitor field snapshots to a 1D time trace."""
    if field_component not in monitor.fields:
        raise ValueError(
            f"Monitor '{monitor.name}' has no field '{field_component}'. "
            f"Available: {sorted(monitor.fields.keys())}"
        )

    raw = monitor.fields[field_component]
    if raw is None or len(raw) == 0:
        raise ValueError(
            f"Monitor '{monitor.name}' has no recorded '{field_component}' data."
        )

    values = np.asarray(raw)
    if values.ndim == 1:
        trace = values
    else:
        flattened = values.reshape(values.shape[0], -1)
        reduction_key = str(reduction).lower()
        if reduction_key == "mean":
            trace = np.mean(flattened, axis=1)
        elif reduction_key == "sum":
            trace = np.sum(flattened, axis=1)
        elif reduction_key == "max_abs":
            trace = np.max(np.abs(flattened), axis=1)
        else:
            raise ValueError(
                f"Unsupported reduction '{reduction}'. "
                "Use one of {'mean', 'sum', 'max_abs'}."
            )

    time_values = np.asarray(monitor.fields.get("t", []), dtype=float)
    if time_values.size < trace.shape[0]:
        if hasattr(sim, "time") and len(sim.time) >= trace.shape[0]:
            time_values = np.asarray(sim.time[: trace.shape[0]], dtype=float)
        else:
            time_values = np.arange(trace.shape[0], dtype=float) * float(sim.dt)

    return np.asarray(trace), np.asarray(time_values)


def resample_complex_matrix(freq_src, values_src, freq_dst):
    """Resample a DFT component matrix to requested frequencies."""
    freq_src = np.atleast_1d(np.asarray(freq_src, dtype=float))
    src = np.asarray(values_src, dtype=np.complex128)
    if src.ndim == 0:
        src = src.reshape(1, 1)
    elif src.ndim == 1:
        if src.shape[0] == freq_src.size:
            src = src[:, None]
        elif freq_src.size == 1:
            src = src.reshape(1, -1)
        else:
            raise ValueError(
                "Cannot infer DFT frequency axis for 1D component array: "
                f"len(values)={src.shape[0]}, nfreq={freq_src.size}"
            )
    else:
        if src.shape[0] != freq_src.size:
            raise ValueError(
                "DFT component matrix must use frequency on axis 0: "
                f"got shape={src.shape}, nfreq={freq_src.size}"
            )
        src = src.reshape(src.shape[0], -1)

    if np.allclose(freq_src, freq_dst, rtol=1e-9, atol=0.0) and src.shape[0] == len(
        freq_dst
    ):
        return src
    out = np.empty((len(freq_dst), src.shape[1]), dtype=np.complex128)
    for col in range(src.shape[1]):
        re = np.interp(
            freq_dst, freq_src, np.real(src[:, col]), left=0.0, right=0.0
        )
        im = np.interp(
            freq_dst, freq_src, np.imag(src[:, col]), left=0.0, right=0.0
        )
        out[:, col] = re + 1j * im
    return out


def monitor_projection_phase(component, frequencies, dt):
    """Phase-align sampled monitor spectra to the modal projection convention."""
    freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
    comp = str(component)
    if comp.startswith("E"):
        return np.exp(-1j * 2.0 * np.pi * freq_arr * float(dt))
    if comp.startswith("H"):
        return np.exp(1j * 2.0 * np.pi * freq_arr * (0.5 * float(dt)))
    return np.ones_like(freq_arr, dtype=np.complex128)


def sample_monitor_component_spectrum(
    sim,
    monitor,
    component,
    frequencies=None,
    window="hann",
):
    if component not in monitor.fields:
        raise ValueError(
            f"Monitor '{monitor.name}' has no field '{component}'. "
            f"Available: {sorted(monitor.fields.keys())}"
        )
    raw = monitor.fields[component]
    if raw is None or len(raw) == 0:
        raise ValueError(
            f"Monitor '{monitor.name}' has no recorded '{component}' data."
        )
    values = np.asarray(raw)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)

    t = np.asarray(monitor.fields.get("t", []), dtype=float)
    n = min(values.shape[0], t.size)
    if n < 2:
        raise ValueError(
            f"Monitor '{monitor.name}' has insufficient samples for FFT extraction."
        )
    values = values[:n]
    t = t[:n]
    values = values - np.mean(values, axis=0, keepdims=True)

    win_key = str(window).lower() if window is not None else "none"
    if win_key in {"hann", "hanning"}:
        w = np.hanning(n)
    elif win_key in {"none", "rect", "rectangular"}:
        w = np.ones(n, dtype=float)
    else:
        raise ValueError(f"Unsupported window '{window}'.")
    values = values * w[:, None]

    dt = float(np.mean(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt inferred from monitor '{monitor.name}'.")
    if np.iscomplexobj(values):
        freq_bins = np.fft.fftfreq(n, d=dt)
        spec_bins = np.fft.fft(values, axis=0)
        keep = freq_bins >= 0
        freq_bins = freq_bins[keep]
        spec_bins = spec_bins[keep]
    else:
        freq_bins = np.fft.rfftfreq(n, d=dt)
        spec_bins = np.fft.rfft(values, axis=0)

    if frequencies is None:
        phase = monitor_projection_phase(component, freq_bins, dt)
        out = spec_bins * phase[:, None]
        return freq_bins, out

    requested = np.atleast_1d(np.asarray(frequencies, dtype=float))
    sampled = np.empty((len(requested), spec_bins.shape[1]), dtype=np.complex128)
    for col in range(spec_bins.shape[1]):
        real_part = np.interp(
            requested, freq_bins, np.real(spec_bins[:, col]), left=0.0, right=0.0
        )
        imag_part = np.interp(
            requested, freq_bins, np.imag(spec_bins[:, col]), left=0.0, right=0.0
        )
        sampled[:, col] = real_part + 1j * imag_part
    phase = monitor_projection_phase(component, requested, dt)
    sampled = sampled * phase[:, None]
    return requested, sampled


def sample_monitor_component_dft(sim, monitor, component, frequencies):
    if not hasattr(monitor, "get_dft_component"):
        raise ValueError(
            f"Monitor '{monitor.name}' does not support DFT accumulation."
        )
    freq_src = np.asarray(monitor.get_dft_frequencies(), dtype=float)
    if freq_src.size == 0:
        raise ValueError(
            f"Monitor '{monitor.name}' has no configured DFT frequencies."
        )
    values_src = np.asarray(monitor.get_dft_component(component), dtype=np.complex128)
    values_src = resample_complex_matrix(freq_src, values_src, freq_src)
    freq_dst = np.atleast_1d(np.asarray(frequencies, dtype=float))
    sampled = resample_complex_matrix(freq_src, values_src, freq_dst)
    phase = monitor_projection_phase(component, freq_dst, sim.dt)
    sampled = sampled * phase[:, None]
    return freq_dst, sampled


def demodulate_monitor_component(
    sim,
    monitor,
    component,
    frequency,
    t_start=None,
    avg_cycles=12,
    window="hann",
):
    """Demodulate one monitor component at a single CW frequency."""
    if component not in monitor.fields:
        raise ValueError(
            f"Monitor '{monitor.name}' has no field '{component}'. "
            f"Available: {sorted(monitor.fields.keys())}"
        )
    raw = monitor.fields[component]
    if raw is None or len(raw) == 0:
        raise ValueError(
            f"Monitor '{monitor.name}' has no recorded '{component}' data."
        )
    values = np.asarray(raw)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)

    t = np.asarray(monitor.fields.get("t", []), dtype=float)
    n = min(values.shape[0], t.size)
    if n < 2:
        raise ValueError(
            f"Monitor '{monitor.name}' has insufficient samples for demodulation."
        )
    values = values[:n]
    t = t[:n]
    f0 = float(frequency)
    if not np.isfinite(f0) or f0 <= 0:
        raise ValueError(f"frequency must be positive, got {frequency!r}")

    if t_start is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = t >= float(t_start)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"Monitor '{monitor.name}' has insufficient post-transient samples."
        )
    t_sel = t[mask]
    v_sel = values[mask]

    if avg_cycles is not None:
        cycles = float(avg_cycles)
        if cycles > 0:
            span = cycles / f0
            t_end = t_sel[0] + span
            keep = t_sel <= t_end
            if np.count_nonzero(keep) >= 2:
                t_sel = t_sel[keep]
                v_sel = v_sel[keep]

    n_sel = t_sel.size
    if n_sel < 2:
        raise ValueError(
            f"Monitor '{monitor.name}' has insufficient samples in demod window."
        )
    win_key = str(window).lower() if window is not None else "none"
    if win_key in {"hann", "hanning"}:
        w = np.hanning(n_sel)
    elif win_key in {"none", "rect", "rectangular"}:
        w = np.ones(n_sel, dtype=float)
    else:
        raise ValueError(f"Unsupported window '{window}'.")

    carrier = np.exp(-1j * 2.0 * np.pi * f0 * t_sel)[:, None]
    denom = max(float(np.sum(w)), 1e-18)
    demod = (2.0 / denom) * np.sum((w[:, None] * v_sel) * carrier, axis=0)
    dt = float(getattr(sim, "dt", 0.0))
    phase = monitor_projection_phase(component, np.asarray([f0]), dt)[0]
    demod = demod * phase
    return np.asarray(demod, dtype=np.complex128)
