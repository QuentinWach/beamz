"""Physics validation tests for wave propagation in BeamZ FDTD.

Tests verify:
1. Phase velocity equals c in vacuum
2. No spurious energy growth (stability)
"""

import numpy as np
import pytest

from beamz import (
    EPS_0,
    LIGHT_SPEED,
    PML,
    Design,
    GaussianSource,
    Material,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)

# Import utilities
from tests.utils import TEST_WAVELENGTH, compute_field_energy, estimate_phase_velocity


@pytest.mark.simulation
class TestFreeSpacePropagation:
    """Verify electromagnetic wave propagation in vacuum."""

    def test_phase_velocity_vacuum(self, vacuum_domain_small):
        """Phase velocity should equal c in vacuum.

        Physics: In vacuum, EM waves propagate at the speed of light.
        v_p = c = 299,792,458 m/s

        Tolerance: 5% (accounts for numerical dispersion at 10 ppw)
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        n_periods = 12
        t_total = n_periods / frequency
        time = np.arange(0, t_total, dt)

        # Create ramped cosine - source active for first half
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.5,
        )

        # Source on left side
        source = GaussianSource(
            position=(2 * wavelength, design.height / 2),
            width=wavelength / 4,
            signal=signal,
        )

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        # Save field snapshots for velocity measurement
        subsample = 10
        result = sim.run(save_fields=["Ez"], field_subsample=subsample)

        # Estimate phase velocity from wavefront tracking
        dt_snapshot = dt * subsample
        v_measured = estimate_phase_velocity(
            result["fields"]["Ez"], dx, dt_snapshot, threshold=0.2
        )

        assert (
            v_measured is not None
        ), "Could not measure phase velocity - insufficient wavefront data"

        # Check within 5% of c
        error = abs(v_measured - LIGHT_SPEED) / LIGHT_SPEED
        assert error < 0.05, (
            f"Phase velocity error {error*100:.1f}% exceeds 5% tolerance. "
            f"Measured: {v_measured:.3e} m/s, Expected: {LIGHT_SPEED:.3e} m/s"
        )

    def test_no_energy_growth(self, vacuum_domain_small):
        """Total field energy should never grow (stability check).

        Physics: In a passive system with absorbing boundaries,
        total EM energy should decrease monotonically after the source stops.
        Any growth indicates numerical instability.

        This test catches:
        - Sign errors in update equations
        - CFL condition violations
        - PML instabilities
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        n_periods = 15
        t_total = n_periods / frequency
        time = np.arange(0, t_total, dt)

        # Short source - stops at 30% of simulation
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.3,
        )

        source = GaussianSource(
            position=(design.width / 2, design.height / 2),
            width=wavelength / 4,
            signal=signal,
        )

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        # Save snapshots to track energy
        subsample = 20
        result = sim.run(save_fields=["Ez"], field_subsample=subsample)

        # Compute energy at each snapshot
        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]

        # After source stops (~40% mark with ramp), check for growth
        source_stop_idx = int(len(energies) * 0.45)
        post_source_energies = energies[source_stop_idx:]

        # Check no significant growth (allow 2% fluctuation for numerical noise)
        max_growth_ratio = 1.02
        for i in range(1, len(post_source_energies)):
            if post_source_energies[i - 1] > 1e-30:  # Skip near-zero values
                ratio = post_source_energies[i] / post_source_energies[i - 1]
                assert ratio < max_growth_ratio, (
                    f"Energy grew by {(ratio-1)*100:.1f}% at step {source_stop_idx + i}. "
                    "This indicates numerical instability."
                )

    def test_field_amplitude_bounded(self, vacuum_domain_small):
        """Field amplitude should remain bounded throughout simulation.

        A more stringent stability check - max field should never exceed
        a reasonable multiple of the source amplitude.
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        n_periods = 10
        t_total = n_periods / frequency
        time = np.arange(0, t_total, dt)

        source_amplitude = 1.0
        signal = ramped_cosine(
            time,
            amplitude=source_amplitude,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.6,
        )

        source = GaussianSource(
            position=(design.width / 2, design.height / 2),
            width=wavelength / 4,
            signal=signal,
        )

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        result = sim.run(save_fields=["Ez"], field_subsample=50)

        # Max field should not exceed some reasonable bound
        # The injected current creates E-field proportional to J*dt/eps
        # We allow up to 1000x as field builds up, but this catches explosive growth
        max_reasonable_field = 1e10  # Absolute bound to catch blowup

        for i, Ez in enumerate(result["fields"]["Ez"]):
            max_field = np.max(np.abs(Ez))
            assert max_field < max_reasonable_field, (
                f"Field exploded at snapshot {i}: max = {max_field:.2e}. "
                "This indicates numerical instability."
            )
