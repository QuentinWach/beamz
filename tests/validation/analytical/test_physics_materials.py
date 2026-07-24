"""Physics validation tests for wave behavior in dielectric materials.

Tests verify:
1. Pulse wavefront velocity = c/n in nondispersive dielectric materials
2. Wavelength contraction by factor n
"""

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    FieldRecorder,
    GaussianSource,
    Material,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
)

# Import utilities
from tests.utils import TEST_WAVELENGTH, estimate_phase_velocity


@pytest.mark.simulation
class TestWaveInMaterial:
    """Verify wave behavior in dielectric materials."""

    @pytest.mark.parametrize("n_material", [1.5, 2.0])
    def test_wavefront_velocity_in_dielectric(self, n_material, validation_metrics):
        """Pulse wavefront velocity should equal c/n in a dielectric.

        Physics: In a nondispersive dielectric with refractive index n,
        v_front = c / n = c / sqrt(epsilon_r)

        Tolerance: 10% for the compact threshold-tracking measurement.
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
            sources=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        subsample = 10
        sim = sim.updated_copy(
            monitors=(*sim.monitors, FieldRecorder(("Ez",), subsample))
        )
        result = sim.run()

        dt_snapshot = dt * subsample
        v_measured = estimate_phase_velocity(
            result.monitor("fields").fields["Ez"], dx, dt_snapshot, threshold=0.2
        )

        expected_velocity = LIGHT_SPEED / n_material

        assert v_measured is not None, "Could not measure wavefront velocity"

        validation_metrics.check(
            "dielectric wavefront velocity",
            measured=v_measured,
            reference=expected_velocity,
            tolerance="material_wavefront_coarse",
            unit="m/s",
            resolution=f"{wavelength / dx:.1f} vacuum-wavelength ppw",
            metadata={"refractive_index": n_material, "threshold": 0.2},
        )

    def test_wavelength_contraction(self, dielectric_domain, validation_metrics):
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
            sources=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        # Run until wave establishes
        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 50)))
        result = sim.run()

        # Take a snapshot after wave has propagated
        # Use one from middle of simulation
        mid_idx = len(result.monitor("fields").fields["Ez"]) // 2
        Ez = result.monitor("fields").fields["Ez"][mid_idx]

        # Get 1D profile through center
        center_row = Ez.shape[0] // 2
        profile = Ez[center_row, :]

        # Find zero crossings to measure wavelength
        zero_crossings = np.where(np.diff(np.sign(profile)))[0]

        assert len(zero_crossings) >= 4, (
            "Validation premise failed: the sampled field has fewer than four "
            "zero crossings, so wavelength cannot be measured."
        )

        # Measure half-wavelength from consecutive crossings
        half_wavelengths = np.diff(zero_crossings) * dx
        # Full wavelength is average of consecutive half-wavelengths
        # (handles sign alternation)
        avg_half_wl = np.median(half_wavelengths)
        measured_wavelength = 2 * avg_half_wl

        expected_wavelength = wavelength / n

        validation_metrics.check(
            "dielectric wavelength",
            measured=measured_wavelength,
            reference=expected_wavelength,
            tolerance="material_wavelength_coarse",
            unit="m",
            resolution=f"{wavelength / dx:.1f} vacuum-wavelength ppw",
            metadata={"refractive_index": n, "zero_crossings": len(zero_crossings)},
        )

    def test_permittivity_affects_propagation(self, validation_metrics):
        """Verify each permittivity produces its expected wavefront velocity.

        A sanity check that the material actually affects the simulation.
        """
        wavelength = TEST_WAVELENGTH
        domain_size = 10 * wavelength

        velocities = []

        for eps_r in [1.0, 2.25, 4.0]:  # n = 1, 1.5, 2
            n = np.sqrt(eps_r)
            dx, dt = calc_optimal_fdtd_params(
                wavelength, n, dims=2, safety_factor=0.95, points_per_wavelength=12
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
                width=wavelength / (4 * n),
                signal=signal,
            )

            sim = Simulation(
                design=design,
                sources=[source],
                boundaries=[PML(thickness=wavelength)],
                time=time,
                resolution=dx,
            )

            subsample = 10
            sim = sim.updated_copy(
                monitors=(*sim.monitors, FieldRecorder(("Ez",), subsample))
            )
            result = sim.run()

            v = estimate_phase_velocity(
                result.monitor("fields").fields["Ez"], dx, dt * subsample, threshold=0.2
            )
            assert v is not None, (
                f"Could not measure wavefront velocity for eps_r={eps_r}, n={n}."
            )
            velocities.append(v)
            validation_metrics.check(
                f"wavefront velocity for eps_r={eps_r}",
                measured=v,
                reference=LIGHT_SPEED / n,
                tolerance="material_wavefront_coarse",
                unit="m/s",
                resolution=f"{wavelength / dx:.1f} vacuum-wavelength ppw",
                metadata={
                    "relative_permittivity": eps_r,
                    "refractive_index": n,
                    "threshold": 0.2,
                },
            )

        assert len(velocities) == 3
        for i in range(1, len(velocities)):
            assert velocities[i] < velocities[i - 1], (
                f"Higher permittivity should give lower velocity. "
                f"Got velocities: {velocities}"
            )
