"""Shared source-time utilities for time-domain source compilation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _interpolate_time_signal(signal, time, dt):
    """Linearly interpolate a real-valued source signal at an arbitrary time."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size <= 0:
        return 0.0

    idx_float = float(time / dt)
    idx_low = int(np.floor(idx_float))
    idx_high = idx_low + 1
    frac = idx_float - idx_low

    if 0 <= idx_low < arr.size - 1:
        return float((1.0 - frac) * arr[idx_low] + frac * arr[idx_high])
    if idx_low == arr.size - 1:
        return float(arr[idx_low])
    return 0.0


def _analytic_signal_quadrature(signal):
    """Return the Hilbert-transform quadrature of a real source waveform."""
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.zeros_like(arr, dtype=np.float64)

    spectrum = np.fft.fft(arr)
    weights = np.zeros((n,), dtype=np.float64)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[n // 2] = 1.0
        weights[1 : n // 2] = 2.0
    else:
        weights[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * weights)
    return np.asarray(np.imag(analytic), dtype=np.float64)


def _real_phasor_sample(profile, in_phase, quadrature):
    """Evaluate Re(profile * analytic_signal) for real-valued FDTD fields."""
    arr = np.asarray(profile, dtype=np.complex128)
    return np.real(arr) * float(in_phase) - np.imag(arr) * float(quadrature)


def _sample_waveform(
    get_signal_value,
    t0: float,
    dt: float,
    num_steps: int,
    offset_fn,
    total_steps: int | None = None,
):
    n = total_steps if total_steps is not None else num_steps
    start = float(t0)
    vals = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        t = start + i * dt
        vals[i] = float(get_signal_value(offset_fn(t, dt), dt))
    return jnp.asarray(vals)


def _analytic_waveform_samples(
    device,
    *,
    t0: float,
    dt: float,
    num_steps: int,
    total_steps: int | None = None,
) -> np.ndarray:
    n = int(total_steps if total_steps is not None else num_steps)
    start = float(t0)
    vals = np.zeros((n,), dtype=np.complex128)
    for i in range(n):
        t = start + i * float(dt)
        vals[i] = complex(
            device._get_signal_value(t, dt),
            device._get_signal_quadrature_value(t, dt),
        )
    return vals


def _partition_weights_by_frequency(
    fft_frequencies: np.ndarray,
    profile_frequencies: np.ndarray,
) -> np.ndarray:
    """Return a smooth frequency partition for broadband modal source profiles."""
    nodes = np.sort(np.unique(np.asarray(profile_frequencies, dtype=float).reshape(-1)))
    if nodes.size == 0:
        raise ValueError("profile_frequencies must contain at least one frequency.")
    if np.any(nodes <= 0.0):
        raise ValueError("profile_frequencies must be strictly positive.")
    abs_freq = np.abs(np.asarray(fft_frequencies, dtype=float).reshape(-1))
    weights = np.zeros((nodes.size, abs_freq.size), dtype=np.float64)
    if nodes.size == 1:
        weights[0, :] = 1.0
        return weights

    for idx, freq in enumerate(nodes):
        if idx == 0:
            right = nodes[idx + 1]
            mask = abs_freq <= right
            weights[idx, mask] = np.where(
                abs_freq[mask] <= freq,
                1.0,
                (right - abs_freq[mask]) / max(right - freq, 1e-30),
            )
            continue
        if idx == nodes.size - 1:
            left = nodes[idx - 1]
            mask = abs_freq >= left
            weights[idx, mask] = np.where(
                abs_freq[mask] >= freq,
                1.0,
                (abs_freq[mask] - left) / max(freq - left, 1e-30),
            )
            continue
        left = nodes[idx - 1]
        right = nodes[idx + 1]
        left_mask = (abs_freq >= left) & (abs_freq <= freq)
        right_mask = (abs_freq >= freq) & (abs_freq <= right)
        weights[idx, left_mask] = (abs_freq[left_mask] - left) / max(freq - left, 1e-30)
        weights[idx, right_mask] = (right - abs_freq[right_mask]) / max(
            right - freq, 1e-30
        )

    total = np.sum(weights, axis=0)
    empty = total <= 1e-30
    if np.any(empty):
        nearest = np.argmin(np.abs(abs_freq[empty, None] - nodes[None, :]), axis=1)
        weights[:, empty] = 0.0
        weights[nearest, np.where(empty)[0]] = 1.0
        total = np.sum(weights, axis=0)
    return weights / np.maximum(total, 1e-30)


def _analytic_subband_waveforms(
    analytic_waveform: np.ndarray,
    *,
    dt: float,
    profile_frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an analytic source waveform into profile-frequency subbands."""
    waveform = np.asarray(analytic_waveform, dtype=np.complex128).reshape(-1)
    if waveform.size == 0:
        return (
            np.asarray(profile_frequencies, dtype=float).reshape(-1),
            np.zeros((0, 0), dtype=np.complex128),
        )
    nodes = np.sort(np.unique(np.asarray(profile_frequencies, dtype=float).reshape(-1)))
    spectrum = np.fft.fft(waveform)
    fft_freqs = np.fft.fftfreq(waveform.size, d=float(dt))
    weights = _partition_weights_by_frequency(fft_freqs, nodes)
    subbands = np.fft.ifft(weights * spectrum[None, :], axis=1)
    return nodes, np.asarray(subbands, dtype=np.complex128)
