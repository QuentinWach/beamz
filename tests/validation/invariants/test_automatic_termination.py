"""End-to-end validation for bounded simulation execution."""

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    AutoTermination,
    Design,
    FieldMonitor,
    GaussianSource,
    Material,
    Simulation,
    calc_optimal_fdtd_params,
    um,
)

pytestmark = [pytest.mark.compiled, pytest.mark.component]


def test_pulse_convergence_preserves_frequency_domain_result():
    """A radiated pulse may stop early without changing its converged DFT."""
    wavelength = 1.0 * um
    resolution, dt = calc_optimal_fdtd_params(
        wavelength,
        1.0,
        dims=2,
        points_per_wavelength=10,
    )
    num_steps = 420
    time = np.arange(num_steps, dtype=float) * dt
    step = np.arange(num_steps)
    frequency = LIGHT_SPEED / wavelength
    signal = np.exp(-0.5 * ((step - 24) / 7) ** 2) * np.cos(
        2.0 * np.pi * frequency * time
    )
    signal[65:] = 0.0

    monitor = FieldMonitor(
        center=(6 * um, 4 * um, 0.0),
        size=(0.0, 4 * um, 0.0),
        freqs=(frequency,),
        fields=("Ez",),
        name="output",
    )
    simulation = Simulation(
        design=Design(
            width=8 * um,
            height=8 * um,
            material=Material(permittivity=1.0),
        ),
        sources=(
            GaussianSource(
                position=(2 * um, 4 * um),
                width=0.3 * um,
                signal=signal,
            ),
        ),
        monitors=(monitor,),
        boundaries=(PML(thickness=wavelength),),
        time=time,
        resolution=resolution,
    )
    policy = AutoTermination(
        field_decay=1e-5,
        monitor_change=1e-3,
        source_decay=1e-6,
        chunk_steps=30,
        consecutive_checks=3,
        monitor_names=("output",),
    )

    bounded = simulation.run(progress=False, termination=policy)
    reference = simulation.run(progress=False)

    report = bounded.termination
    assert report is not None
    assert report.reason == "converged"
    assert report.converged
    assert report.steps < num_steps
    assert report.source_decay == pytest.approx(0.0)
    assert report.field_decay <= policy.field_decay
    assert report.monitor_change is not None
    assert report.monitor_change <= policy.monitor_change

    bounded_field = bounded["output"].get_dft_component("Ez")
    reference_field = reference["output"].get_dft_component("Ez")
    relative_error = np.linalg.norm(bounded_field - reference_field) / max(
        np.linalg.norm(bounded_field), np.linalg.norm(reference_field)
    )
    assert relative_error < 5e-3
    np.testing.assert_allclose(
        bounded["output"].dft_weight_sum,
        reference["output"].dft_weight_sum,
    )
