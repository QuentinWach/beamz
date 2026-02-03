"""Physics validation tests for wave behavior in dielectric materials.

Tests verify:
1. Phase velocity = c/n in dielectric materials
2. Wavelength contraction by factor n
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
from tests.utils import TEST_WAVELENGTH, estimate_phase_velocity


@pytest.mark.simulation
class TestWaveInMaterial:
    """Verify wave behavior in dielectric materials."""

    @pytest.mark.parametrize("n_material", [1.5, 2.0])
    def test_phase_velocity_in_dielectric(self, n_material):
        """Phase velocity should equal c/n in a dielectric.

        Physics: In a dielectric with refractive index n,
        v_p = c / n = c / sqrt(epsilon_r)

        Tolerance: 8% (higher tolerance due to material averaging)
        """
        wavelength = TEST_WAVELENGTH
        domain_size = 12 * wavelength

        dx, dt = calc_optimal_fdtd_params(
            wavelength, n_material, dims=2, safety_factor=0.95, points_per_wavelength=12
        )

        design = Design(
            width=domain_size,
            height=domain_size,
            material=Material(permittivity=n_material**2),
        )

        frequency = LIGHT_SPEED / wavelength
        n_periods = 15
        t_total = n_periods / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.5,
        )

        # Source on left side
        source = GaussianSource(
            position=(2 * wavelength, domain_size / 2),
            width=wavelength / (4 * n_material),  # Smaller source in higher index
            signal=signal,
        )

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        subsample = 10
        result = sim.run(save_fields=["Ez"], field_subsample=subsample)

        dt_snapshot = dt * subsample
        v_measured = estimate_phase_velocity(
            result["fields"]["Ez"], dx, dt_snapshot, threshold=0.2
        )

        expected_velocity = LIGHT_SPEED / n_material

        assert v_measured is not None, "Could not measure phase velocity"

        error = abs(v_measured - expected_velocity) / expected_velocity
        assert error < 0.08, (
            f"Phase velocity error {error*100:.1f}% exceeds 8% in n={n_material}. "
            f"Measured: {v_measured:.3e} m/s, Expected: {expected_velocity:.3e} m/s"
        )

    def test_wavelength_contraction(self, dielectric_domain):
        """Wavelength should contract by factor n in a dielectric.

        Physics: lambda_material = lambda_0 / n

        Method: Measure spatial period by finding field zero-crossings
        along propagation direction.

        Tolerance: 10%
        """
        design = dielectric_domain["design"]
        wavelength = dielectric_domain["wavelength"]
        dx = dielectric_domain["dx"]
        dt = dielectric_domain["dt"]
        n = dielectric_domain["n"]

        frequency = LIGHT_SPEED / wavelength
        n_periods = 12
        t_total = n_periods / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.6,
        )

        # Source on left side
        source = GaussianSource(
            position=(2 * wavelength, design.height / 2),
            width=wavelength / (4 * n),
            signal=signal,
        )

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        # Run until wave establishes
        result = sim.run(save_fields=["Ez"], field_subsample=50)

        # Take a snapshot after wave has propagated
        # Use one from middle of simulation
        mid_idx = len(result["fields"]["Ez"]) // 2
        Ez = result["fields"]["Ez"][mid_idx]

        # Get 1D profile through center
        center_row = Ez.shape[0] // 2
        profile = Ez[center_row, :]

        # Find zero crossings to measure wavelength
        zero_crossings = np.where(np.diff(np.sign(profile)))[0]

        if len(zero_crossings) < 4:
            pytest.skip("Insufficient zero crossings to measure wavelength")

        # Measure half-wavelength from consecutive crossings
        half_wavelengths = np.diff(zero_crossings) * dx
        # Full wavelength is average of consecutive half-wavelengths
        # (handles sign alternation)
        avg_half_wl = np.median(half_wavelengths)
        measured_wavelength = 2 * avg_half_wl

        expected_wavelength = wavelength / n

        error = abs(measured_wavelength - expected_wavelength) / expected_wavelength
        assert error < 0.10, (
            f"Wavelength error {error*100:.1f}% exceeds 10%. "
            f"Measured: {measured_wavelength/um:.3f} um, "
            f"Expected: {expected_wavelength/um:.3f} um"
        )

    def test_permittivity_affects_propagation(self):
        """Verify that different permittivity values produce different velocities.

        A sanity check that the material actually affects the simulation.
        """
        wavelength = TEST_WAVELENGTH
        domain_size = 10 * wavelength

        velocities = []

        for eps_r in [1.0, 2.25, 4.0]:  # n = 1, 1.5, 2
            n = np.sqrt(eps_r)
            dx, dt = calc_optimal_fdtd_params(
                wavelength, n, dims=2, safety_factor=0.95, points_per_wavelength=10
            )

            design = Design(
                width=domain_size,
                height=domain_size,
                material=Material(permittivity=eps_r),
            )

            frequency = LIGHT_SPEED / wavelength
            t_total = 10 / frequency
            time = np.arange(0, t_total, dt)

            signal = ramped_cosine(
                time,
                amplitude=1.0,
                frequency=frequency,
                ramp_duration=2 / frequency,
                t_max=t_total * 0.4,
            )

            source = GaussianSource(
                position=(2 * wavelength, domain_size / 2),
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

            subsample = 10
            result = sim.run(save_fields=["Ez"], field_subsample=subsample)

            v = estimate_phase_velocity(
                result["fields"]["Ez"], dx, dt * subsample, threshold=0.2
            )
            if v is not None:
                velocities.append(v)

        # Verify we got measurements and they're monotonically decreasing
        assert len(velocities) >= 2, "Could not measure velocities"
        for i in range(1, len(velocities)):
            assert velocities[i] < velocities[i - 1], (
                f"Higher permittivity should give lower velocity. "
                f"Got velocities: {velocities}"
            )
