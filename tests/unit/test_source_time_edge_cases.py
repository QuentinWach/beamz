"""Fast numerical contracts for temporal source utilities and spectra."""

from __future__ import annotations

import numpy as np
import pytest

from beamz import GaussianPulse, SampledSignal, ramped_cosine
from beamz.devices.sources.time import (
    analytic_signal_quadrature,
    analytic_subband_waveforms,
    chebyshev_frequency_nodes,
    gaussian_band_pulse,
    interpolate_time_signal,
    partition_weights_by_frequency,
    sample_source_waveforms,
)


def test_interpolation_and_hilbert_edge_cases_have_explicit_zero_behavior():
    assert interpolate_time_signal([], 0.0, 1.0) == 0.0
    assert interpolate_time_signal([2.0], 0.0, 1.0) == 2.0
    assert interpolate_time_signal([2.0, 4.0], 0.5, 1.0) == 3.0
    assert interpolate_time_signal([2.0, 4.0], -1.0, 1.0) == 0.0
    assert interpolate_time_signal([2.0, 4.0], 2.0, 1.0) == 0.0

    assert analytic_signal_quadrature([]).shape == (0,)
    np.testing.assert_array_equal(analytic_signal_quadrature([2.0]), [0.0])
    for count in (5, 6):
        signal = np.cos(2.0 * np.pi * np.arange(count) / count)
        quadrature = analytic_signal_quadrature(signal)
        assert quadrature.shape == signal.shape
        assert abs(float(np.mean(quadrature))) < 1e-12


def test_source_waveform_sampling_supports_protocol_callable_and_arrays():
    sampled = SampledSignal(
        values=np.array([0.0, 1.0, 0.0]),
        quadrature=np.array([1.0, 0.0, -1.0]),
        dt=1.0,
        freq0=1.0,
    )
    signal, quadrature = sample_source_waveforms(
        sampled, t0=0.0, dt=1.0, num_steps=3
    )
    np.testing.assert_array_equal(signal, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(quadrature, [1.0, 0.0, -1.0])

    signal, quadrature = sample_source_waveforms(
        lambda time: 2.0 * time,
        t0=0.0,
        dt=0.5,
        num_steps=3,
        signal_quadrature=lambda time: -time,
    )
    np.testing.assert_array_equal(signal, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(quadrature, [0.0, -0.5, -1.0])

    signal, quadrature = sample_source_waveforms(
        [0.0, 1.0, 0.0, -1.0],
        t0=0.0,
        dt=1.0,
        num_steps=3,
        signal_quadrature=[1.0, 0.0, -1.0, 0.0],
    )
    np.testing.assert_array_equal(signal, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(quadrature, [1.0, 0.0, -1.0])


def test_frequency_partition_is_normalized_exact_at_nodes_and_clamped():
    with pytest.raises(ValueError, match="at least one"):
        partition_weights_by_frequency(np.array([1.0]), np.array([]))
    with pytest.raises(ValueError, match="strictly positive"):
        partition_weights_by_frequency(np.array([1.0]), np.array([0.0, 1.0]))

    single = partition_weights_by_frequency(
        np.array([-2.0, 0.0, 2.0]), np.array([1.0])
    )
    np.testing.assert_array_equal(single, np.ones((1, 3)))

    nodes = np.array([1.0, 2.0, 3.0])
    frequencies = np.array([-4.0, 1.0, 1.5, 2.0, 4.0])
    weights = partition_weights_by_frequency(frequencies, nodes)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_array_equal(weights[:, 1], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(weights[:, 3], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(weights[:, 0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(weights[:, -1], [0.0, 0.0, 1.0])


def test_analytic_subbands_preserve_shape_and_reconstruct_waveform():
    empty_nodes, empty = analytic_subband_waveforms(
        np.array([], dtype=complex), dt=0.1, profile_frequencies=np.array([1.0, 2.0])
    )
    np.testing.assert_array_equal(empty_nodes, [1.0, 2.0])
    assert empty.shape == (0, 0)

    time = np.arange(32) * 0.01
    waveform = np.exp(2j * np.pi * 3.0 * time) + 0.5 * np.exp(
        2j * np.pi * 6.0 * time
    )
    nodes, subbands = analytic_subband_waveforms(
        waveform, dt=0.01, profile_frequencies=np.array([3.0, 6.0])
    )
    np.testing.assert_array_equal(nodes, [3.0, 6.0])
    assert subbands.shape == (2, waveform.size)
    np.testing.assert_allclose(subbands.sum(axis=0), waveform, rtol=1e-10, atol=1e-10)


def test_ramped_cosine_requires_a_complete_envelope_specification():
    time = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match="t_max"):
        ramped_cosine(time, amplitude=1.0, frequency=1.0)
    with pytest.raises(ValueError, match="t_max"):
        ramped_cosine(
            time, amplitude=1.0, frequency=1.0, ramp_duration=0.1
        )

    implicit = ramped_cosine(
        time, amplitude=1.0, frequency=1.0, phase=None, t_max=1.0
    )
    explicit = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=1.0,
        phase=0.0,
        ramp_duration=0.1,
        t_max=1.0,
    )
    np.testing.assert_allclose(implicit, explicit)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"values": [], "dt": 1.0}, "cannot be empty"),
        ({"values": [1.0], "dt": 0.0}, "positive and finite"),
        ({"values": [1.0], "dt": np.inf}, "positive and finite"),
        (
            {"values": [1.0, 2.0], "quadrature": [1.0], "dt": 1.0},
            "must match",
        ),
    ],
)
def test_sampled_signal_rejects_invalid_grids(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SampledSignal(**kwargs)


def test_sampled_signal_infers_carrier_interpolates_and_normalizes_spectrum():
    singleton = SampledSignal([1.0], dt=0.5)
    assert singleton.freq0 == 0.0

    signal = SampledSignal([0.0, 1.0, 0.0, -1.0], dt=0.25)
    assert signal.freq0 == pytest.approx(1.0)
    values, quadrature = signal.sample([-1.0, 0.25, 2.0])
    np.testing.assert_array_equal(values, [0.0, 1.0, 0.0])
    assert quadrature.shape == values.shape

    spectrum = signal.spectrum(np.array([[0.0, 1.0]]), normalize=True)
    assert spectrum.shape == (1, 2)
    assert np.max(np.abs(spectrum)) == pytest.approx(1.0)
    np.testing.assert_allclose(
        signal.dft_normalization_spectrum([1.0]),
        signal.spectrum([1.0], normalize=True) / (2.0 * np.pi),
    )


def test_gaussian_pulse_dc_policy_and_frequency_nodes_are_deterministic():
    assert chebyshev_frequency_nodes(10.0, 2.0, 1).tolist() == [10.0]
    nodes = chebyshev_frequency_nodes(10.0, 2.0, 4)
    assert np.all(np.diff(nodes) > 0)
    assert np.mean(nodes) == pytest.approx(10.0)

    times = np.linspace(0.0, 2e-13, 21)
    with_dc = GaussianPulse(
        2e14, 2e13, remove_dc_component=False
    ).sample(times)[0]
    without_dc = GaussianPulse(
        2e14, 2e13, remove_dc_component=True
    ).sample(times)[0]
    assert float(np.mean(without_dc)) == pytest.approx(0.0, abs=1e-7)
    assert not np.array_equal(with_dc, without_dc)

    normalized = GaussianPulse(2e14, 2e13).spectrum([2e14], normalize=True)
    np.testing.assert_allclose(np.abs(normalized), [1.0])


def test_gaussian_band_pulse_owns_readonly_arrays_without_running_a_simulation():
    pulse = gaussian_band_pulse(
        [1e14, 2e14],
        carrier_frequency=1.5e14,
        dt=1e-15,
        run_after_sources_uoc=20.0,
        max_output_distance_um=5.0,
    )

    assert pulse.sigma > 0.0
    assert pulse.source_end_time > pulse.peak_time
    assert pulse.tail_cap_time >= pulse.tail_time
    assert pulse.time.shape == pulse.signal.shape == pulse.signal_quadrature.shape
    for array in (
        pulse.frequencies,
        pulse.time,
        pulse.signal,
        pulse.signal_quadrature,
    ):
        assert not array.flags.writeable
