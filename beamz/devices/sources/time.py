"""Temporal waveforms and sampling utilities for source compilation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import numpy as np
from numpy import typing as npt

from beamz.const import LIGHT_SPEED
from beamz.devices._immutable import readonly_array


def interpolate_time_signal(signal, time, dt):
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


def analytic_signal_quadrature(signal):
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


def chebyshev_frequency_nodes(freq0: float, fwidth: float, count: int) -> np.ndarray:
    n = int(count)
    if n <= 1:
        return np.asarray([float(freq0)], dtype=float)
    k = np.arange(n, dtype=float)
    nodes = float(freq0) + 1.5 * float(fwidth) * np.cos(
        (2.0 * k + 1.0) * np.pi / (2.0 * n)
    )
    return np.sort(nodes.astype(float))


def sample_waveform(
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


def sample_source_waveforms(
    source,
    *,
    t0: float,
    dt: float,
    num_steps: int,
    total_steps: int | None = None,
    signal_quadrature=None,
    offset_fn=lambda t, _dt: t,
):
    """Sample in-phase and quadrature source waveforms on the compiled time grid."""
    n = int(total_steps if total_steps is not None else num_steps)
    times = np.asarray(
        [offset_fn(float(t0) + i * float(dt), float(dt)) for i in range(n)],
        dtype=np.float64,
    )

    if hasattr(source, "sample") and signal_quadrature is None:
        signal, quadrature = source.sample(times)
        return (
            jnp.asarray(np.asarray(signal, dtype=np.float64).reshape(-1)),
            jnp.asarray(np.asarray(quadrature, dtype=np.float64).reshape(-1)),
        )

    if callable(source):
        source_fn = cast(Callable[[float], float], source)
        signal = np.asarray(
            [float(source_fn(float(t))) for t in times], dtype=np.float64
        )
        default_quadrature = np.zeros_like(signal)
    else:
        signal_array = np.asarray(source, dtype=np.float64).reshape(-1)
        signal = np.asarray(
            [interpolate_time_signal(signal_array, t, dt) for t in times],
            dtype=np.float64,
        )
        default_quadrature = analytic_signal_quadrature(signal_array)

    if signal_quadrature is None:
        quadrature = (
            np.zeros_like(signal)
            if callable(source)
            else np.asarray(
                [interpolate_time_signal(default_quadrature, t, dt) for t in times],
                dtype=np.float64,
            )
        )
    elif callable(signal_quadrature):
        quadrature_fn = cast(Callable[[float], float], signal_quadrature)
        quadrature = np.asarray(
            [float(quadrature_fn(float(t))) for t in times],
            dtype=np.float64,
        )
    else:
        quadrature_array = np.asarray(signal_quadrature, dtype=np.float64).reshape(-1)
        quadrature = np.asarray(
            [interpolate_time_signal(quadrature_array, t, dt) for t in times],
            dtype=np.float64,
        )

    return jnp.asarray(signal), jnp.asarray(quadrature)


def partition_weights_by_frequency(
    fft_frequencies: np.ndarray,
    profile_frequencies: np.ndarray,
) -> np.ndarray:
    """Return barycentric interpolation weights for broadband source profiles."""
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

    center = 0.5 * (nodes[0] + nodes[-1])
    half_span = 0.5 * (nodes[-1] - nodes[0])
    scaled_nodes = (nodes - center) / half_span
    differences = scaled_nodes[:, None] - scaled_nodes[None, :]
    np.fill_diagonal(differences, 1.0)
    barycentric = 1.0 / np.prod(differences, axis=1)
    barycentric /= np.max(np.abs(barycentric))

    # Avoid unstable high-order extrapolation where the source spectrum is
    # negligible. The extreme mode profile is the safer continuation outside
    # the sampled band.
    samples = np.clip(abs_freq, nodes[0], nodes[-1])
    scaled_samples = (samples - center) / half_span
    delta = scaled_samples[:, None] - scaled_nodes[None, :]
    at_node = np.isclose(delta, 0.0, rtol=0.0, atol=8.0 * np.finfo(float).eps)
    regular = ~np.any(at_node, axis=1)
    if np.any(regular):
        values = barycentric[None, :] / delta[regular]
        values /= np.sum(values, axis=1, keepdims=True)
        weights[:, regular] = values.T
    if np.any(~regular):
        weights[:, ~regular] = at_node[~regular].T.astype(float)
    return weights


def analytic_subband_waveforms(
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
    weights = partition_weights_by_frequency(fft_freqs, nodes)
    subbands = np.fft.ifft(weights * spectrum[None, :], axis=1)
    if nodes.size > 1:
        times = np.arange(waveform.size, dtype=float) * float(dt)
        phase = np.exp(-2j * np.pi * nodes[:, None] * times[None, :])
        total_response = phase @ waveform
        peak = float(np.max(np.abs(total_response), initial=0.0))
        if peak > 0.0 and np.all(np.abs(total_response) > 1e-12 * peak):
            response = (phase @ subbands.T) / total_response[:, None]
            if np.linalg.cond(response) < 1e8:
                correction = np.linalg.solve(response, np.eye(nodes.size))
                subbands = correction.T @ subbands
    return nodes, np.asarray(subbands, dtype=np.complex128)


def cosine(t, amplitude, frequency, phase):
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


def sigmoid(t, duration=1, min=0, max=1, t0=0):
    return min + (max - min) * (
        1 / (1 + np.exp(-10 * (t - duration / 2 - t0) / duration))
    )


def ramped_cosine(t, amplitude, frequency, phase=None, ramp_duration=None, t_max=None):
    """Sample a cosine carrier with smooth sigmoid turn-on and turn-off.

    Parameters
    ----------
    t : array-like
        Physical sample times in seconds.
    amplitude : float
        Carrier amplitude.
    frequency : float
        Carrier frequency in hertz.
    phase : float, optional
        Carrier phase in radians; defaults to zero.
    ramp_duration : float, optional
        Turn-on and turn-off duration in seconds. Defaults to ``t_max / 10``.
    t_max : float
        Center time of the falling ramp in seconds.

    Returns
    -------
    numpy.ndarray
        Real waveform with the same shape as ``t``.

    Raises
    ------
    ValueError
        If ``t_max`` is absent or the ramp duration cannot be derived.
    """
    if phase is None:
        phase = 0
    if ramp_duration is None:
        if t_max is None:
            raise ValueError("t_max is required when ramp_duration is omitted")
        ramp_duration = t_max / 10
    if t_max is None:
        raise ValueError("t_max is required for ramped_cosine")
    signal = sigmoid(t, min=0, max=1, duration=ramp_duration, t0=0)
    signal *= cosine(t, amplitude=amplitude, frequency=frequency, phase=phase)
    signal *= sigmoid(t, min=1, max=0, duration=ramp_duration, t0=t_max - ramp_duration)
    return signal


def gaussian(t, amplitude, center, width):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width**2))


@dataclass(frozen=True)
class SampledSignal:
    """Immutable source-time samples on a uniform time grid.

    Parameters
    ----------
    values : array-like
        Real in-phase samples.
    dt : float
        Time spacing between samples in seconds.
    quadrature : array-like, optional
        Quadrature samples. When omitted, they are derived with a Hilbert
        transform.
    start_time : float, default=0
        Physical time of the first sample.
    freq0 : float, optional
        Carrier frequency in hertz. When omitted, the strongest positive FFT
        bin is used.
    """

    values: np.ndarray
    dt: float
    quadrature: np.ndarray | None = None
    start_time: float = 0.0
    freq0: float | None = None

    def __post_init__(self) -> None:
        values = readonly_array(self.values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("SampledSignal.values cannot be empty.")
        dt = float(self.dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("SampledSignal.dt must be positive and finite.")
        quadrature = self.quadrature
        if quadrature is None:
            quadrature = analytic_signal_quadrature(values)
        quadrature = readonly_array(quadrature, dtype=np.float64).reshape(-1)
        if quadrature.shape != values.shape:
            raise ValueError("SampledSignal quadrature must match values.")
        freq0 = self.freq0
        if freq0 is None:
            spectrum = np.abs(np.fft.rfft(values))
            if spectrum.size <= 1:
                freq0 = 0.0
            else:
                freq0 = float(
                    np.fft.rfftfreq(values.size, dt)[1:][np.argmax(spectrum[1:])]
                )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "quadrature", quadrature)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "start_time", float(self.start_time))
        object.__setattr__(self, "freq0", float(freq0))

    def sample(self, time: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate in-phase and quadrature samples at physical times."""
        requested = np.asarray(time, dtype=float)
        sample_time = self.start_time + np.arange(self.values.size) * self.dt
        signal = np.interp(requested, sample_time, self.values, left=0.0, right=0.0)
        quadrature_values = cast(np.ndarray, self.quadrature)
        quadrature = np.interp(
            requested, sample_time, quadrature_values, left=0.0, right=0.0
        )
        return signal.astype(np.float32), quadrature.astype(np.float32)

    def spectrum(self, freqs: npt.ArrayLike, *, normalize: bool = False) -> np.ndarray:
        """Return the complex discrete-time spectrum at frequencies in hertz."""
        frequencies = np.asarray(freqs, dtype=float)
        time = self.start_time + np.arange(self.values.size) * self.dt
        analytic = self.values + 1j * cast(np.ndarray, self.quadrature)
        spectrum = np.sum(
            analytic[:, None]
            * np.exp(1j * 2.0 * np.pi * time[:, None] * frequencies.reshape(1, -1)),
            axis=0,
        ).reshape(frequencies.shape)
        if normalize:
            scale = float(np.max(np.abs(spectrum), initial=0.0))
            if scale > 0.0:
                spectrum = spectrum / scale
        return np.asarray(spectrum, dtype=np.complex128)

    def dft_normalization_spectrum(self, freqs: npt.ArrayLike) -> np.ndarray:
        """Return normalized samples in BeamZ's monitor convention."""
        return self.spectrum(freqs, normalize=True) / (2.0 * np.pi)


@dataclass(frozen=True)
class GaussianPulse:
    """Define an analytic Gaussian-modulated carrier pulse.

    Parameters
    ----------
    freq0 : float
        Carrier frequency in hertz.
    fwidth : float
        Gaussian frequency width in hertz.
    amplitude : float, default=1.0
        Peak envelope amplitude.
    offset : float, default=4.0
        Pulse-center offset measured in inverse-bandwidth units.
    remove_dc_component : bool, default=True
        Subtract the sampled waveform mean to suppress DC excitation.

    Notes
    -----
    :meth:`sample` returns in-phase and quadrature ``float32`` waveforms.

    Examples
    --------
    >>> pulse = GaussianPulse(freq0=193.5e12, fwidth=20e12)
    >>> signal, quadrature = pulse.sample([0.0, 1e-15, 2e-15])
    """

    freq0: float
    fwidth: float
    amplitude: float = 1.0
    offset: float = 4.0
    remove_dc_component: bool = True

    def _time_width(self) -> float:
        return 1.0 / (2.0 * np.pi * max(float(self.fwidth), 1e-30))

    def spectrum(
        self,
        freqs: npt.ArrayLike,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Return the analytic positive-frequency source spectrum.

        Parameters
        ----------
        freqs : array-like
            Frequencies in hertz.
        normalize : bool, default=False
            Divide by the magnitude of the spectrum at the carrier frequency.

        Returns
        -------
        numpy.ndarray
            Complex analytic spectrum with the same shape as ``freqs``.

        Examples
        --------
        >>> spectrum = pulse.spectrum([190e12, 193.5e12, 200e12], normalize=True)
        """
        freq_arr = np.asarray(freqs, dtype=float)
        fwidth = max(float(self.fwidth), 1e-30)
        width = self._time_width()
        peak = float(self.offset) / fwidth
        df = freq_arr - float(self.freq0)
        spectrum = (
            float(self.amplitude)
            * width
            * np.sqrt(2.0 * np.pi)
            * np.exp(-0.5 * (df / fwidth) ** 2)
            * np.exp(1j * 2.0 * np.pi * df * peak)
        )
        if normalize:
            center = float(self.amplitude) * width * np.sqrt(2.0 * np.pi)
            spectrum = spectrum / max(abs(center), 1e-300)
        return np.asarray(spectrum, dtype=np.complex128)

    def dft_normalization_spectrum(self, freqs: npt.ArrayLike) -> np.ndarray:
        """Return the source spectrum in BeamZ's native monitor normalization.

        Parameters
        ----------
        freqs : array-like
            Frequencies in hertz.

        Returns
        -------
        numpy.ndarray
            Complex normalized spectrum in BeamZ's monitor-DFT convention.

        Notes
        -----
        This method supplies source normalization for frequency-domain monitor
        results. Most users should call ``SimulationResults.renormalize`` rather
        than applying it manually.
        """
        return self.spectrum(freqs, normalize=True) / (2.0 * np.pi)

    def sample(self, time: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Sample in-phase and quadrature waveforms at times in seconds.

        Parameters
        ----------
        time : array-like
            Physical sample times in seconds.

        Returns
        -------
        signal, quadrature : tuple of numpy.ndarray
            In-phase and quadrature ``float32`` samples matching ``time``.

        Notes
        -----
        When ``remove_dc_component`` is enabled, the mean is removed from the
        in-phase samples after evaluating the analytic pulse.
        """
        t = np.asarray(time, dtype=float)
        width = self._time_width()
        peak = float(self.offset) / max(float(self.fwidth), 1e-30)
        envelope = float(self.amplitude) * np.exp(-((t - peak) ** 2) / (2.0 * width**2))
        phase = 2.0 * np.pi * float(self.freq0) * t
        signal = envelope * np.cos(phase)
        quadrature = envelope * np.sin(phase)
        if self.remove_dc_component and signal.size:
            signal = signal - np.mean(signal)
        return signal.astype(np.float32), quadrature.astype(np.float32)


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
    signal_quadrature: np.ndarray

    def __post_init__(self) -> None:
        for name in ("frequencies", "time", "signal", "signal_quadrature"):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))


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
    """Create a Meep-style Gaussian pulse and matching simulation timeline.

    This captures the common broadband FDTD setup pattern:
    - choose the pulse width from the requested frequency span
    - delay the pulse peak by a fixed multiple of sigma
    - extend the time array with both a minimum path-based settling window and
      a larger hard cap for adaptive stop conditions
    """

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
    envelope = gaussian(time, 1.0, peak_time, sigma)
    phase = 2.0 * np.pi * float(carrier_frequency) * time
    signal = np.asarray(envelope * np.cos(phase), dtype=np.float32)
    signal_quadrature = np.asarray(envelope * np.sin(phase), dtype=np.float32)
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
        signal_quadrature=signal_quadrature,
    )
