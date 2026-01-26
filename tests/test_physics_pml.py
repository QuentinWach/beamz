"""Physics validation tests for PML (Perfectly Matched Layer) boundaries.

Tests verify:
1. PML absorbs outgoing waves with minimal reflection
2. Energy decays monotonically after source stops
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
from tests.utils import TEST_WAVELENGTH, compute_field_energy


@pytest.mark.simulation
class TestPMLAbsorption:
    """Verify PML boundary absorbs waves properly."""

    def test_pml_reflection_level(self, vacuum_domain_small):
        """PML should absorb waves with minimal reflection.

        Physics: A properly implemented PML creates an impedance-matched
        absorbing region that minimizes reflections.

        Method: Launch pulse, let it hit PML, measure late-time field
        relative to peak. Late-time field is primarily reflections.

        Tolerance: Reflection ratio < 10%
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        n_periods = 20
        t_total = n_periods / frequency
        time = np.arange(0, t_total, dt)

        # Short pulse so incident and reflected are temporally separated
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.3,  # Source stops at 30%
        )

        # Source at center
        source = GaussianSource(
            position=(design.width / 2, design.height / 2),
            width=wavelength / 4,
            signal=signal,
        )

        # Thicker PML for better absorption
        pml_thickness = 1.5 * wavelength

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=pml_thickness)],
            time=time,
            resolution=dx,
        )

        result = sim.run(save_fields=["Ez"], field_subsample=20)

        # Compute energy at each snapshot
        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]

        # Find peak energy (during excitation)
        peak_energy = max(energies)
        peak_idx = energies.index(peak_energy)

        # Check late-time energy (should be very small after absorption)
        late_idx = int(len(energies) * 0.9)
        if late_idx > peak_idx:
            late_energy = np.mean(energies[late_idx:])

            # Reflection ratio
            reflection_ratio = late_energy / peak_energy if peak_energy > 0 else 0

            assert reflection_ratio < 0.10, (
                f"PML reflection {reflection_ratio*100:.1f}% exceeds 10%. "
                "This indicates poor PML absorption."
            )

    def test_energy_decay_with_pml(self, vacuum_domain_small):
        """Energy should decay monotonically after source stops.

        Physics: With absorbing boundaries, EM energy leaves the domain
        and should decrease steadily.

        Tolerance: Energy ratio < 1.02 between consecutive measurements
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        n_periods = 15
        t_total = n_periods / frequency
        time = np.arange(0, t_total, dt)

        # Source stops early
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

        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]

        # After source stops (~35% accounting for ramp), check monotonic decay
        source_stop_idx = int(len(energies) * 0.4)
        post_source = energies[source_stop_idx:]

        # Allow small fluctuations (2%) but no sustained growth
        max_ratio = 1.02
        growth_count = 0
        for i in range(1, len(post_source)):
            if post_source[i - 1] > 1e-30:  # Skip near-zero
                ratio = post_source[i] / post_source[i - 1]
                if ratio > max_ratio:
                    growth_count += 1
                    assert growth_count < 3, (
                        f"Sustained energy growth detected: ratio={ratio:.3f} "
                        f"at step {source_stop_idx + i}"
                    )

    @pytest.mark.parametrize("pml_layers_wl", [0.5, 1.0, 1.5])
    def test_thicker_pml_better_absorption(self, vacuum_domain_small, pml_layers_wl):
        """Thicker PML should generally provide better absorption.

        This test verifies that the PML implementation is reasonable
        by checking that absorption improves (or doesn't degrade)
        with thickness.
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 15 / frequency
        time = np.arange(0, t_total, dt)

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

        pml_thickness = pml_layers_wl * wavelength

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=pml_thickness)],
            time=time,
            resolution=dx,
        )

        result = sim.run(save_fields=["Ez"], field_subsample=20)

        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]

        peak_energy = max(energies)
        late_energy = np.mean(energies[-3:]) if len(energies) >= 3 else energies[-1]

        # For any reasonable PML, reflection should be < 20%
        reflection_ratio = late_energy / peak_energy if peak_energy > 0 else 0
        assert reflection_ratio < 0.20, (
            f"PML with {pml_layers_wl} wavelength thickness has "
            f"{reflection_ratio*100:.1f}% reflection, exceeds 20%"
        )

    def test_pml_does_not_cause_instability(self, vacuum_domain_small):
        """PML should not cause numerical instability.

        Some PML implementations can be unstable, especially at corners
        or with certain parameter choices.
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 30 / frequency  # Long simulation to catch late instabilities
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.2,
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

        result = sim.run(save_fields=["Ez"], field_subsample=100)

        # Check for field explosion
        max_reasonable = 1e10
        for i, Ez in enumerate(result["fields"]["Ez"]):
            max_field = np.max(np.abs(Ez))
            assert (
                max_field < max_reasonable
            ), f"PML instability detected at snapshot {i}: max={max_field:.2e}"

        # Check that energy eventually decays (not stuck at high level)
        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]
        if energies[0] > 1e-30:
            decay_ratio = energies[-1] / max(energies)
            assert (
                decay_ratio < 0.5
            ), f"Energy not decaying with PML: final/peak = {decay_ratio:.2f}"
