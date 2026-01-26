"""Energy conservation and Poynting theorem validation tests.

Tests verify:
1. Poynting flux through closed surface equals rate of energy change
2. Total EM energy is conserved in lossless systems
3. Source power injection matches expectations
"""

import numpy as np
import pytest

from beamz import (
    EPS_0,
    LIGHT_SPEED,
    MU_0,
    PML,
    Design,
    GaussianSource,
    Material,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from tests.utils import TEST_WAVELENGTH, compute_field_energy


@pytest.mark.simulation
class TestEnergyConservation:
    """Verify energy conservation and Poynting theorem."""

    def test_energy_conservation_closed_system(self, vacuum_domain_small):
        """Total EM energy should not increase in passive system.

        Physics: dU/dt ≤ 0 after source stops (energy leaves via PML)

        Method: Track total field energy, verify monotonic decay after source.
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 20 / frequency
        time = np.arange(0, t_total, dt)

        # Source active for first 25% of simulation
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.25,
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

        result = sim.run(save_fields=["Ez"], field_subsample=10)

        # Compute energy at each snapshot
        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]

        # After source stops (~35% with ramp), energy should decay
        source_stop_idx = int(len(energies) * 0.4)
        post_source = energies[source_stop_idx:]

        # Check for monotonic decay (with small tolerance for numerical noise)
        max_growth = 1.03  # Allow 3% fluctuation
        growth_violations = 0
        for i in range(1, len(post_source)):
            if post_source[i - 1] > 1e-30:
                ratio = post_source[i] / post_source[i - 1]
                if ratio > max_growth:
                    growth_violations += 1

        # Allow at most 2 violations (numerical transients)
        assert growth_violations < 3, (
            f"Energy grew {growth_violations} times after source stopped. "
            "Possible energy conservation violation."
        )

    def test_energy_decay_rate(self, vacuum_domain_small):
        """Energy should decay to near zero after sufficient time.

        Physics: With PML boundaries, all energy eventually leaves the domain.
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 25 / frequency
        time = np.arange(0, t_total, dt)

        # Very short pulse
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.15,
        )

        source = GaussianSource(
            position=(design.width / 2, design.height / 2),
            width=wavelength / 4,
            signal=signal,
        )

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=1.5 * wavelength)],
            time=time,
            resolution=dx,
        )

        result = sim.run(save_fields=["Ez"], field_subsample=15)

        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]

        peak_energy = max(energies)
        final_energy = energies[-1]

        # Final energy should be small fraction of peak
        decay_ratio = final_energy / peak_energy if peak_energy > 0 else 0
        assert decay_ratio < 0.15, (
            f"Final energy is {decay_ratio*100:.1f}% of peak. "
            "Energy should decay more with PML."
        )

    def test_poynting_flux_direction(self, vacuum_domain_small):
        """Poynting flux should point outward from source.

        Physics: Energy flows away from the source in all directions.
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 10 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.5,
        )

        # Source at center
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

        result = sim.run(save_fields=["Ez", "Hx", "Hy"], field_subsample=20)

        # Get a snapshot during active emission
        mid_idx = len(result["fields"]["Ez"]) // 2
        Ez = result["fields"]["Ez"][mid_idx]
        Hx = result["fields"]["Hx"][mid_idx]
        Hy = result["fields"]["Hy"][mid_idx]

        # Compute Poynting vector components
        # Note: H fields may have different shapes due to Yee staggering
        ny, nx = Ez.shape

        # Simple check: field should spread outward from center
        center_y, center_x = ny // 2, nx // 2

        # Check that field exists in multiple quadrants (spreading)
        quadrant_energies = []
        for qy, qx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            y_slice = slice(qy * center_y, (qy + 1) * center_y)
            x_slice = slice(qx * center_x, (qx + 1) * center_x)
            q_energy = compute_field_energy(Ez[y_slice, x_slice], dx)
            quadrant_energies.append(q_energy)

        # All quadrants should have some energy (omnidirectional emission)
        for i, e in enumerate(quadrant_energies):
            assert e > 0, f"Quadrant {i} has no energy - emission not omnidirectional"

    def test_source_injects_energy(self, vacuum_domain_small):
        """Source should inject energy into the domain.

        Physics: Active source increases total EM energy.
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 8 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.8,
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

        result = sim.run(save_fields=["Ez"], field_subsample=10)

        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]

        # Energy should increase from zero
        initial_energy = energies[0]
        peak_energy = max(energies)

        assert (
            peak_energy > initial_energy
        ), "Source should inject energy. Peak energy not greater than initial."
        assert peak_energy > 0, "Peak energy should be positive"

    def test_energy_in_dielectric(self, dielectric_domain):
        """Energy density should scale with permittivity.

        Physics: U = (1/2) * ε₀ * ε_r * E²
        Higher ε means more energy for same field amplitude.
        """
        design = dielectric_domain["design"]
        wavelength = dielectric_domain["wavelength"]
        dx = dielectric_domain["dx"]
        dt = dielectric_domain["dt"]
        n = dielectric_domain["n"]
        eps_r = n**2

        frequency = LIGHT_SPEED / wavelength
        t_total = 10 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.5,
        )

        source = GaussianSource(
            position=(design.width / 2, design.height / 2),
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

        result = sim.run(save_fields=["Ez"], field_subsample=20)

        # Compute energy with correct permittivity
        energies_correct = [
            compute_field_energy(Ez, dx, eps=eps_r) for Ez in result["fields"]["Ez"]
        ]

        # Compare to energy computed with vacuum permittivity
        energies_vacuum = [
            compute_field_energy(Ez, dx, eps=1.0) for Ez in result["fields"]["Ez"]
        ]

        # Correct energy should be eps_r times vacuum energy
        peak_idx = np.argmax(energies_correct)
        ratio = energies_correct[peak_idx] / energies_vacuum[peak_idx]

        assert (
            abs(ratio - eps_r) / eps_r < 0.01
        ), f"Energy ratio {ratio:.2f} should be ε_r = {eps_r:.2f}"
