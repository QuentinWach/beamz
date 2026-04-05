from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from beamz.const import LIGHT_SPEED


def cosine(t, amplitude, frequency, phase):
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


def sigmoid(t, duration=1, min=0, max=1, t0=0):
    return min + (max - min) * (
        1 / (1 + np.exp(-10 * (t - duration / 2 - t0) / duration))
    )


def ramped_cosine(t, amplitude, frequency, phase=None, ramp_duration=None, t_max=None):
    if phase is None:
        phase = 0
    if ramp_duration is None:
        ramp_duration = t_max / 10
    signal = sigmoid(t, min=0, max=1, duration=ramp_duration, t0=0)
    signal *= cosine(t, amplitude=amplitude, frequency=frequency, phase=phase)
    signal *= sigmoid(t, min=1, max=0, duration=ramp_duration, t0=t_max - ramp_duration)
    return signal


def gaussian(t, amplitude, center, width):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width**2))


def gaussian_pulse(t, amplitude, center, width, frequency, phase):
    return gaussian(t, amplitude, center, width) * cosine(
        t, amplitude, frequency, phase
    )


@dataclass(frozen=True)
class GaussianBandPulse:
    frequencies: np.ndarray
    carrier_frequency: float
    sigma: float
    peak_time: float
    source_end_time: float
    min_tail_uoc: float
    tail_time: float
    tail_cap_time: float
    time: np.ndarray
    signal: np.ndarray


def gaussian_band_pulse(
    frequencies,
    *,
    carrier_frequency,
    dt,
    run_after_sources_uoc,
    max_output_distance_um,
    min_sigma_factor=0.20,
    peak_sigma_multiple=4.0,
    source_tail_sigma_multiple=6.0,
    min_tail_cycles=96.0,
    max_tail_cycles=192.0,
    min_tail_distance_factor=6.0,
    min_tail_uoc=90.0,
    max_tail_uoc=180.0,
):
    """Create a broadband Gaussian pulse and a matching simulation timeline."""
    freqs = np.asarray(frequencies, dtype=float)
    df = max(float(np.ptp(freqs)), 1e-12)
    fmin = max(float(np.min(freqs)), 1e-12)
    sigma = float(min_sigma_factor) / max(df, 1e9)
    peak_time = float(peak_sigma_multiple) * sigma
    source_end_time = peak_time + float(source_tail_sigma_multiple) * sigma
    min_tail_uoc_eff = max(
        float(run_after_sources_uoc),
        float(min_tail_uoc),
        float(min_tail_distance_factor) * float(max_output_distance_um),
    )
    tail_time = max(
        min_tail_uoc_eff * 1e-6 / LIGHT_SPEED, float(min_tail_cycles) / fmin
    )
    tail_cap_time = max(
        float(max_tail_uoc) * 1e-6 / LIGHT_SPEED, float(max_tail_cycles) / fmin
    )
    time = np.arange(0.0, source_end_time + tail_cap_time, float(dt))
    signal = np.asarray(
        gaussian_pulse(time, 1.0, peak_time, sigma, float(carrier_frequency), 0.0),
        dtype=np.float32,
    )
    return GaussianBandPulse(
        frequencies=freqs,
        carrier_frequency=float(carrier_frequency),
        sigma=sigma,
        peak_time=peak_time,
        source_end_time=source_end_time,
        min_tail_uoc=min_tail_uoc_eff,
        tail_time=tail_time,
        tail_cap_time=tail_cap_time,
        time=time,
        signal=signal,
    )


def plot_signal(signals, t, save_path=None):
    """Plot one or more time-domain source signals."""
    t_seconds = np.asarray(t, dtype=float)
    if t_seconds.size == 0:
        raise ValueError("t must contain at least one sample")

    if t_seconds[-1] < 1e-12:
        t_scaled, unit = t_seconds * 1e15, "fs"
    elif t_seconds[-1] < 1e-9:
        t_scaled, unit = t_seconds * 1e12, "ps"
    elif t_seconds[-1] < 1e-6:
        t_scaled, unit = t_seconds * 1e9, "ns"
    elif t_seconds[-1] < 1e-3:
        t_scaled, unit = t_seconds * 1e6, "µs"
    elif t_seconds[-1] < 1:
        t_scaled, unit = t_seconds * 1e3, "ms"
    else:
        t_scaled, unit = t_seconds, "s"

    fig, ax = plt.subplots(figsize=(9, 4))
    if isinstance(signals, list):
        for idx, signal in enumerate(signals):
            ax.plot(t_scaled, np.asarray(signal), label=f"Signal {idx}")
        ax.legend()
    else:
        ax.plot(t_scaled, np.asarray(signals), color="black")
    ax.set_xlim(float(t_scaled[0]), float(t_scaled[-1]))
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return None
    return fig, ax
