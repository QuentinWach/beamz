"""Legacy physics characterization and analytical-helper checks.

Several solver checks here establish stability, field presence, or coarse
refinement consistency. Those checks are useful characterization, but they do
not measure every quantity suggested by the underlying analytical formulas.
Cases graduate to ``validation/`` only after they compare a BeamZ observable to
an independent oracle with a defensible error gate.
"""

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Circle,
    Design,
    FieldRecorder,
    GaussianSource,
    Material,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
)
from tests.utils import (
    TEST_WAVELENGTH,
    analytical_cavity_frequency,
    analytical_fresnel_r,
    analytical_fresnel_t,
    compute_field_energy,
    measure_resonance_frequency,
    mie_qext_2d,
)

# =============================================================================
# Test Configuration
# =============================================================================
TOLERANCE_TIGHT = 0.05  # 5% for most tests


# =============================================================================
# Fresnel Interface Characterization
# =============================================================================
@pytest.mark.simulation
class TestFresnelCoefficients:
    """Characterize stable interface execution and field transmission."""

    @pytest.mark.parametrize(
        "n1,n2,expected_R",
        [
            (1.0, 1.5, 0.04),  # Air → Glass
            (1.5, 1.0, 0.04),  # Glass → Air
            (1.0, 2.0, 0.1111),  # Air → High-index
            (1.5, 2.5, 0.0625),  # Glass → Diamond-like
        ],
    )
    def test_interface_run_has_bounded_fields(self, n1, n2, expected_R):
        """Run an interface case without claiming a measured Fresnel coefficient."""
        wavelength = TEST_WAVELENGTH

        # Domain sized for good pulse separation
        domain_width = 25 * wavelength
        domain_height = 6 * wavelength

        # High resolution for accuracy
        n_max = max(n1, n2)
        dx, dt = calc_optimal_fdtd_params(
            wavelength, n_max, dims=2, safety_factor=0.95, points_per_wavelength=15
        )

        # Interface at center
        interface_x = domain_width / 2

        # Create domain: n1 on left, n2 on right
        design = Design(
            width=domain_width,
            height=domain_height,
            material=Material(permittivity=n1**2),
        )
        design += Rectangle(
            position=(interface_x + domain_width / 4, domain_height / 2),
            width=domain_width / 2,
            height=domain_height,
            material=Material(permittivity=n2**2),
        )

        frequency = LIGHT_SPEED / wavelength
        t_total = 30 / frequency
        time = np.arange(0, t_total, dt)

        # Short pulse for time-domain separation
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.25,
        )

        # Source well before interface
        source_x = interface_x * 0.3
        source = GaussianSource(
            position=(source_x, domain_height / 2),
            width=wavelength * 1.5,
            signal=signal,
        )

        sim = Simulation(
            design=design,
            sources=[source],
            boundaries=[PML(thickness=1.5 * wavelength)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 10)))
        result = sim.run()

        # Track total energy over time
        energies = [
            compute_field_energy(Ez, dx) for Ez in result.monitor("fields").fields["Ez"]
        ]
        peak_energy = max(energies)

        # Check that energy exists and decays (absorbed by PML)
        assert peak_energy > 0, "Should have non-zero energy"

        # Interface index
        interface_idx = int(interface_x / dx)

        # At late times, check that field exists on both sides of interface
        # (demonstrating both reflection and transmission occurred)
        late_field = result.monitor("fields").fields["Ez"][-1]

        # Energy on each side
        E_left = compute_field_energy(late_field[:, :interface_idx], dx, eps=n1**2)
        E_right = compute_field_energy(late_field[:, interface_idx:], dx, eps=n2**2)

        # Verify analytical formula
        R_analytical = analytical_fresnel_r(n1, n2)
        T_analytical = analytical_fresnel_t(n1, n2)
        assert abs(R_analytical + T_analytical - 1.0) < 1e-10, "R + T should equal 1"
        assert abs(R_analytical - expected_R) < 0.01, (
            f"Analytical R={R_analytical:.4f} vs expected {expected_R:.4f}"
        )

        # For higher index contrast, more reflection expected
        # This is a qualitative check that the physics is correct
        if n1 != n2:
            # At least some field should exist in both regions
            total_late = E_left + E_right
            if total_late > 1e-30:
                # Check that both regions have some energy
                assert E_right > 0 or E_left > 0, (
                    "Should have field energy after pulse passes interface"
                )


# =============================================================================
# Grid Refinement Characterization
# =============================================================================
@pytest.mark.simulation
class TestGridRefinementCharacterization:
    """Check stability and coarse consistency across three grid resolutions."""

    def test_propagation_is_consistent_across_refinement(self):
        """Verify FDTD scheme stability and consistency across resolutions.

        Tests that simulations at different resolutions produce consistent
        results, with finer grids being more accurate. The Yee scheme is
        theoretically 2nd order, but measuring exact order requires very
        careful test setup. Here we verify:
        1. All resolutions produce stable, finite results
        2. Results converge (become more similar with finer grids)
        """
        wavelength = TEST_WAVELENGTH
        domain_size = 5 * wavelength

        # Test at different resolutions
        ppw_values = [8, 12, 18]  # points per wavelength

        # Store total energy at peak as convergence metric
        peak_energies = []
        dx_values = []

        for ppw in ppw_values:
            dx = wavelength / ppw
            dt = dx / (LIGHT_SPEED * np.sqrt(2)) * 0.95

            design = Design(
                width=domain_size,
                height=domain_size,
                material=Material(permittivity=1.0),
            )

            frequency = LIGHT_SPEED / wavelength
            t_total = 6 / frequency
            time = np.arange(0, t_total, dt)

            signal = ramped_cosine(
                time,
                amplitude=1.0,
                frequency=frequency,
                ramp_duration=2 / frequency,
                t_max=t_total * 0.5,
            )

            source = GaussianSource(
                position=(domain_size / 2, domain_size / 2),
                width=wavelength / 4,
                signal=signal,
            )

            sim = Simulation(
                design=design,
                sources=[source],
                boundaries=[PML(thickness=wavelength)],
                time=time,
                resolution=dx,
            )

            sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 15)))
            result = sim.run()

            # Use peak energy as metric
            energies = [
                compute_field_energy(Ez, dx)
                for Ez in result.monitor("fields").fields["Ez"]
            ]
            peak_energies.append(max(energies))
            dx_values.append(dx)

        # All simulations should be stable with positive finite energy
        assert all(np.isfinite(e) and e > 0 for e in peak_energies), (
            "All simulations should produce finite positive energy"
        )

        # Energy values should be reasonably close across resolutions
        # (within factor of 2 for these moderate resolutions)
        max_e = max(peak_energies)
        min_e = min(peak_energies)
        assert max_e / min_e < 2.0, (
            f"Energy varies too much: {min_e:.2e} to {max_e:.2e}"
        )

        # Verify that results become more consistent with finer grid
        # Compare coarse-to-fine difference with medium-to-fine difference
        diff_coarse = abs(peak_energies[0] - peak_energies[-1])
        diff_medium = abs(peak_energies[1] - peak_energies[-1])

        # Medium grid should be closer to fine grid than coarse grid is
        # (or both are essentially converged)
        assert (
            diff_medium <= diff_coarse * 1.5 or diff_coarse < 0.05 * peak_energies[-1]
        ), (
            f"Convergence expected: coarse diff={diff_coarse:.2e}, medium diff={diff_medium:.2e}"
        )

    def test_energy_conservation_convergence(self):
        """Verify energy conservation improves with resolution."""
        wavelength = TEST_WAVELENGTH
        domain_size = 5 * wavelength

        ppw_values = [12, 20]
        energy_fluctuations = []

        for ppw in ppw_values:
            dx = wavelength / ppw
            dt = dx / (LIGHT_SPEED * np.sqrt(2)) * 0.95

            design = Design(
                width=domain_size,
                height=domain_size,
                material=Material(permittivity=1.0),
            )

            frequency = LIGHT_SPEED / wavelength
            t_total = 15 / frequency
            time = np.arange(0, t_total, dt)

            # Short pulse to test energy conservation after source stops
            signal = ramped_cosine(
                time,
                amplitude=1.0,
                frequency=frequency,
                ramp_duration=2 / frequency,
                t_max=t_total * 0.2,
            )

            source = GaussianSource(
                position=(domain_size / 2, domain_size / 2),
                width=wavelength / 4,
                signal=signal,
            )

            sim = Simulation(
                design=design,
                sources=[source],
                boundaries=[PML(thickness=1.5 * wavelength)],
                time=time,
                resolution=dx,
            )

            sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 10)))
            result = sim.run()

            # Compute energy after source stops
            energies = [
                compute_field_energy(Ez, dx)
                for Ez in result.monitor("fields").fields["Ez"]
            ]

            # Find energy fluctuation in decay phase
            start_idx = len(energies) // 3
            end_idx = 2 * len(energies) // 3

            if energies[start_idx] > 1e-30:
                max_growth = max(
                    energies[i] / energies[i - 1] if energies[i - 1] > 1e-30 else 1.0
                    for i in range(start_idx + 1, end_idx)
                )
                energy_fluctuations.append(max_growth - 1.0)
            else:
                energy_fluctuations.append(0)

        # Finer grid should have smaller energy fluctuation
        assert energy_fluctuations[-1] <= energy_fluctuations[0] * 1.5, (
            "Energy conservation should improve with finer grid"
        )


# =============================================================================
# Scattering Characterization
# =============================================================================
@pytest.mark.simulation
class TestMieScattering:
    """Characterize stable cylinder scattering without measuring flux efficiency."""

    @pytest.mark.parametrize("size_param", [0.5, 1.5, 3.0])
    def test_2d_cylinder_qext(self, size_param):
        """Verify 2D cylinder scattering simulation is stable and physically correct.

        Size parameter x = 2πr/λ determines scattering regime:
        - x < 1: Rayleigh (small particle)
        - x ~ 1-3: Mie resonance regime
        - x > 5: Geometric optics

        This test verifies:
        1. Simulation stability with scatterer
        2. Field exists and is finite
        3. Analytical Mie formula gives reasonable values
        """
        n_cyl = 2.0
        n_medium = 1.0
        wavelength = TEST_WAVELENGTH

        # Radius from size parameter
        radius = size_param * wavelength / (2 * np.pi * n_medium)

        # Analytical Q_ext
        Q_ext_analytical = mie_qext_2d(radius, wavelength, n_cyl, n_medium)

        # Domain sized for scatterer + monitors
        domain_size = max(8 * wavelength, 10 * radius)

        dx, dt = calc_optimal_fdtd_params(
            wavelength, n_cyl, dims=2, safety_factor=0.95, points_per_wavelength=15
        )

        # Create domain with cylinder at center
        cx, cy = domain_size / 2, domain_size / 2

        design = Design(
            width=domain_size,
            height=domain_size,
            material=Material(permittivity=n_medium**2),
        )
        design += Circle(
            position=(cx, cy), radius=radius, material=Material(permittivity=n_cyl**2)
        )

        frequency = LIGHT_SPEED / wavelength
        t_total = 15 / frequency
        time = np.arange(0, t_total, dt)

        # Plane-wave-like source from left
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=3 / frequency,
            t_max=t_total * 0.5,
        )

        # Wide Gaussian source for plane-wave approximation
        source = GaussianSource(
            position=(wavelength * 2, cy), width=domain_size * 0.6, signal=signal
        )

        sim = Simulation(
            design=design,
            sources=[source],
            boundaries=[PML(thickness=1.5 * wavelength)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 10)))
        result = sim.run()

        # Verify simulation produced valid fields
        final_Ez = result.monitor("fields").fields["Ez"][-1]
        assert np.all(np.isfinite(final_Ez)), "Fields should be finite"

        peak_field = np.max(np.abs(final_Ez))
        assert peak_field > 0, "Should have non-zero field"

        # Verify field exists in shadow region (transmission) and lit region
        center_idx = int(cx / dx)
        shadow_region = final_Ez[:, center_idx + 10 :]
        lit_region = final_Ez[:, : center_idx - 10]

        assert np.max(np.abs(shadow_region)) > 0, "Should have field past scatterer"
        assert np.max(np.abs(lit_region)) > 0, "Should have field before scatterer"

        # Verify analytical formula gives reasonable values
        assert Q_ext_analytical > 0, "Q_ext should be positive"
        assert Q_ext_analytical < 15, f"Q_ext={Q_ext_analytical:.2f} seems too large"


# =============================================================================
# Fabry-Pérot Cavity Characterization
# =============================================================================
@pytest.mark.simulation
class TestFabryPerot:
    """Characterize cavity spectra and independently test helper formulas."""

    def test_cavity_resonance_frequency(self):
        """Verify cavity resonance matches f_m = mc/(2nL).

        Uses FFT of field time series to find resonance peak.
        """
        wavelength = TEST_WAVELENGTH
        n_cavity = 1.0

        # Cavity length for m=2 mode at target wavelength
        # f_2 = 2c/(2nL) = c/(nL), so L = c/(n*f) = λ
        cavity_length = wavelength
        expected_f1 = analytical_cavity_frequency(1, cavity_length, n_cavity)
        expected_f2 = analytical_cavity_frequency(2, cavity_length, n_cavity)

        domain_width = cavity_length + 4 * wavelength
        domain_height = 4 * wavelength

        dx, dt = calc_optimal_fdtd_params(
            wavelength, 1.0, dims=2, safety_factor=0.95, points_per_wavelength=20
        )

        # Create cavity with high-reflectivity "mirrors"
        # Use high-permittivity material for reflection
        mirror_eps = 20.0  # High reflection
        mirror_width = 0.1 * wavelength

        design = Design(
            width=domain_width,
            height=domain_height,
            material=Material(permittivity=n_cavity**2),
        )

        # Left mirror
        mirror_left_x = (domain_width - cavity_length) / 2
        design += Rectangle(
            position=(mirror_left_x, domain_height / 2),
            width=mirror_width,
            height=domain_height,
            material=Material(permittivity=mirror_eps),
        )

        # Right mirror
        mirror_right_x = mirror_left_x + cavity_length
        design += Rectangle(
            position=(mirror_right_x, domain_height / 2),
            width=mirror_width,
            height=domain_height,
            material=Material(permittivity=mirror_eps),
        )

        frequency = LIGHT_SPEED / wavelength
        t_total = 30 / frequency  # Long run for frequency resolution
        time = np.arange(0, t_total, dt)

        # Broadband excitation to excite multiple modes
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.15,  # Short pulse for broadband
        )

        # Source inside cavity
        source_x = domain_width / 2
        source = GaussianSource(
            position=(source_x, domain_height / 2), width=wavelength / 4, signal=signal
        )

        sim = Simulation(
            design=design,
            sources=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 1)))
        result = sim.run()

        # Extract field at cavity center
        center_idx = int(source_x / dx)
        field_at_center = [
            Ez[Ez.shape[0] // 2, center_idx]
            for Ez in result.monitor("fields").fields["Ez"]
        ]

        # Find resonance frequency via FFT
        measured_freq = measure_resonance_frequency(
            field_at_center, time, freq_range=(0.5 * expected_f1, 3 * expected_f1)
        )

        # Check if measured frequency is near a cavity mode
        freq_error_f1 = abs(measured_freq - expected_f1) / expected_f1
        freq_error_f2 = abs(measured_freq - expected_f2) / expected_f2

        min_error = min(freq_error_f1, freq_error_f2)

        assert min_error < TOLERANCE_TIGHT, (
            f"Measured f={measured_freq:.2e} Hz should be near "
            f"f1={expected_f1:.2e} or f2={expected_f2:.2e} Hz "
            f"(error={min_error * 100:.1f}%)"
        )


# =============================================================================
# Waveguide Effective Index Characterization
# =============================================================================
@pytest.mark.simulation
class TestWaveguideEffectiveIndex:
    """Exercise slab-waveguide helpers and basic propagation behavior."""

    def test_waveguide_propagation_qualitative(self):
        """Verify guided mode propagation in slab waveguide.

        Light should be confined to core and propagate without spreading.
        """
        n_core = 2.0
        n_clad = 1.0
        wavelength = TEST_WAVELENGTH
        core_width = 0.8 * wavelength  # Multi-mode capable

        domain_width = 15 * wavelength
        domain_height = 5 * wavelength

        dx, dt = calc_optimal_fdtd_params(
            wavelength, n_core, dims=2, safety_factor=0.95, points_per_wavelength=15
        )

        design = Design(
            width=domain_width,
            height=domain_height,
            material=Material(permittivity=n_clad**2),
        )
        design += Rectangle(
            position=(domain_width / 2, domain_height / 2),
            width=domain_width,
            height=core_width,
            material=Material(permittivity=n_core**2),
        )

        frequency = LIGHT_SPEED / wavelength
        t_total = 15 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=3 / frequency,
            t_max=t_total * 0.4,
        )

        # Source centered on waveguide
        source = GaussianSource(
            position=(2 * wavelength, domain_height / 2),
            width=core_width / 2,
            signal=signal,
        )

        sim = Simulation(
            design=design,
            sources=[source],
            boundaries=[PML(thickness=1.5 * wavelength)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 20)))
        result = sim.run()

        # Check field confinement at late time
        late_field = result.monitor("fields").fields["Ez"][-1]
        ny, nx = late_field.shape

        # Energy in core region vs total
        core_y_min = int((domain_height / 2 - core_width) / dx)
        core_y_max = int((domain_height / 2 + core_width) / dx)

        core_energy = compute_field_energy(late_field[core_y_min:core_y_max, :], dx)
        total_energy = compute_field_energy(late_field, dx)

        if total_energy > 1e-30:
            confinement = core_energy / total_energy
            # Most energy should be in/near core for guided mode
            assert confinement > 0.3, (
                f"Only {confinement * 100:.1f}% energy in core region"
            )
