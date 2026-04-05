"""DFT accumulation helpers for monitors."""

import numpy as np


def should_accumulate(monitor, step, t):
    if not monitor.dft_enabled or monitor.dft_frequencies.size == 0:
        return False
    if monitor.dft_t_end is not None and float(t) > monitor.dft_t_end:
        return False
    if float(t) < monitor.dft_t_start:
        return False
    return (int(step) % int(monitor.dft_record_interval)) == 0


def weight(monitor, t):
    if (
        monitor.dft_window == "hann"
        and monitor.dft_t_end is not None
        and monitor.dft_t_end > monitor.dft_t_start
    ):
        tau = (float(t) - monitor.dft_t_start) / (
            monitor.dft_t_end - monitor.dft_t_start
        )
        tau = min(max(tau, 0.0), 1.0)
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * tau))
    return 1.0


def init_or_get_accum(monitor, component, npoints):
    arr = monitor._dft_accum.get(component)
    shape = (monitor.dft_frequencies.size, int(npoints))
    if arr is None or arr.shape != shape:
        arr = np.zeros(shape, dtype=np.complex128)
        monitor._dft_accum[component] = arr
    return arr


def current_phase(monitor, t):
    t_now = float(t)
    if monitor._dft_last_t is None:
        monitor._dft_phase = np.exp(-1j * 2.0 * np.pi * monitor.dft_frequencies * t_now)
        monitor._dft_last_t = t_now
        return monitor._dft_phase
    dt = t_now - float(monitor._dft_last_t)
    if abs(dt) > 0.0:
        if (
            monitor._dft_last_dt is None
            or monitor._dft_last_rot is None
            or abs(dt - float(monitor._dft_last_dt)) > 1e-18
        ):
            monitor._dft_last_dt = dt
            monitor._dft_last_rot = np.exp(
                -1j * 2.0 * np.pi * monitor.dft_frequencies * dt
            )
        monitor._dft_phase = monitor._dft_phase * monitor._dft_last_rot
    monitor._dft_last_t = t_now
    return monitor._dft_phase


def update(monitor, t, component_vectors):
    if not component_vectors:
        return
    w = float(monitor._dft_weight(t))
    if w <= 0.0:
        return
    phase = monitor._dft_current_phase(t)
    monitor._dft_weight_sum = monitor._dft_weight_sum + w
    monitor._dft_sample_count += 1
    for comp, vec in component_vectors.items():
        arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
        accum = monitor._init_or_get_dft_accum(comp, arr.size)
        accum += (w * phase)[:, None] * arr[None, :]
        monitor._dft_accum[comp] = accum


def reset(monitor):
    monitor._dft_accum = {}
    monitor._dft_weight_sum = np.zeros(monitor.dft_frequencies.size, dtype=float)
    monitor._dft_sample_count = 0
    monitor._dft_phase = np.ones(monitor.dft_frequencies.size, dtype=np.complex128)
    monitor._dft_last_t = None
    monitor._dft_last_dt = None
    monitor._dft_last_rot = None


def get_frequencies(monitor):
    return np.asarray(monitor.dft_frequencies, dtype=float)


def get_component(monitor, component: str):
    comp = str(component)
    if comp not in monitor._dft_accum:
        raise ValueError(f"No DFT data recorded for component '{comp}'.")
    accum = np.asarray(monitor._dft_accum[comp], dtype=np.complex128)
    nfreq = int(monitor.dft_frequencies.size)
    if nfreq <= 0:
        raise ValueError(f"Monitor '{monitor.name}' has no configured DFT frequencies.")
    if accum.ndim == 0:
        accum = accum.reshape(1, 1)
    elif accum.ndim == 1:
        if accum.shape[0] == nfreq:
            accum = accum[:, None]
        elif nfreq == 1:
            accum = accum.reshape(1, -1)
        else:
            raise ValueError(
                "Cannot infer DFT frequency axis for component "
                f"'{comp}': shape={accum.shape}, nfreq={nfreq}"
            )
    else:
        if accum.shape[0] != nfreq:
            raise ValueError(
                "DFT accumulator must use frequency on axis 0 for component "
                f"'{comp}': shape={accum.shape}, nfreq={nfreq}"
            )
        accum = accum.reshape(nfreq, -1)

    scale = np.maximum(np.asarray(monitor._dft_weight_sum, dtype=float), 1e-18).reshape(
        nfreq, 1
    )
    return (2.0 / scale) * accum
