import math as m

import matplotlib.pyplot as plt
import numpy as np


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


def plot_signal(signals, t, save_path=None):
    """Create a single signal or a list of signals on the same plot."""
    # Convert time to seconds
    t_seconds = t
    # Determine appropriate time unit and scaling factor
    if t_seconds[-1] < 1e-12:
        t_scaled = t_seconds * 1e15  # Convert to fs
        unit = "fs"
    elif t_seconds[-1] < 1e-9:  # Less than 1 ns
        t_scaled = t_seconds * 1e12  # Convert to ps
        unit = "ps"
    elif t_seconds[-1] < 1e-6:  # Less than 1 µs
        t_scaled = t_seconds * 1e9  # Convert to ns
        unit = "ns"
    elif t_seconds[-1] < 1e-3:  # Less than 1 ms
        t_scaled = t_seconds * 1e6  # Convert to µs
        unit = "µs"
    elif t_seconds[-1] < 1:  # Less than 1 s
        t_scaled = t_seconds * 1e3  # Convert to ms
        unit = "ms"
    else:
        t_scaled = t_seconds
        unit = "s"
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(9, 4))
    if isinstance(signals, list):
        i = 0
        for signal in signals:
            ax.plot(t_scaled, signal, label=f"Signal {i}")
            i += 1
        ax.legend()
    else:
        ax.plot(t_scaled, signals, color="black")
    ax.set_xlim(t_scaled[0], t_scaled[-1])
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
