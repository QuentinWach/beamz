"""Passive-domain energy-decay invariants for compact simulations."""

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    FieldRecorder,
    GaussianSource,
    Simulation,
    ramped_cosine,
)
from tests.utils import compute_field_energy


@pytest.mark.simulation
class TestPassiveEnergyDecay:
    """Verify that energy does not grow after a source turns off."""

    def test_passive_pml_domain_does_not_gain_energy_after_source_turnoff(
        self, vacuum_domain_small
    ):
        """Recorded electric energy should not grow after source turnoff.

        Physics: energy leaves the domain through the PML after source turnoff.

        Method: track electric-field energy and bound transient growth.
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
            sources=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 10)))
        result = sim.run()

        # Compute energy at each snapshot
        energies = [
            compute_field_energy(Ez, dx) for Ez in result.monitor("fields").fields["Ez"]
        ]

        # After source stops (~35% with ramp), energy should decay
        source_stop_idx = int(len(energies) * 0.4)
        post_source = energies[source_stop_idx:]
        assert len(post_source) >= 2, (
            "Energy-decay premise requires at least two post-source snapshots."
        )

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

    def test_recorded_electric_energy_decays_to_near_zero(self, vacuum_domain_small):
        """Recorded electric energy should decay after sufficient time.

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
            sources=[source],
            boundaries=[PML(thickness=1.5 * wavelength)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 15)))
        result = sim.run()

        energies = [
            compute_field_energy(Ez, dx) for Ez in result.monitor("fields").fields["Ez"]
        ]

        peak_energy = max(energies)
        final_energy = energies[-1]

        # Final energy should be small fraction of peak
        assert peak_energy > 0.0, "Energy-decay premise requires a nonzero pulse."
        decay_ratio = final_energy / peak_energy
        assert decay_ratio < 0.15, (
            f"Final energy is {decay_ratio * 100:.1f}% of peak. "
            "Energy should decay more with PML."
        )
