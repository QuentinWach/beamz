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


# Backward-compatible re-export (canonical location: beamz.visual.source_plots)
from beamz.visual.source_plots import plot_signal  # noqa: E402, F401
