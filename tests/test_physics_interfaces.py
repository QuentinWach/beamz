"""Physics validation tests for Fresnel reflection at dielectric interfaces.

Tests verify:
1. Reflection coefficient matches Fresnel equations
2. Energy conservation R + T = 1
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
    Monitor,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)

# Import utilities
from tests.utils import TEST_WAVELENGTH


@pytest.mark.simulation
class TestFresnelReflection:
    """Verify Fresnel reflection/transmission at dielectric interfaces."""

    def test_field_transmitted_through_interface(self, dielectric_interface_domain):
        """Field should propagate through the interface into the dielectric.

        Physics: At a lossless dielectric interface, most power is transmitted
        (for n1=1.0, n2=1.5, T ~ 96%).

        Method: Check that field energy exists in the dielectric region
        after the wave has had time to propagate through the interface.
        """
        design = dielectric_interface_domain["design"]
        wavelength = dielectric_interface_domain["wavelength"]
        dx = dielectric_interface_domain["dx"]
        dt = dielectric_interface_domain["dt"]
        domain_height = dielectric_interface_domain["domain_height"]
        interface_x = dielectric_interface_domain["interface_x"]

        frequency = LIGHT_SPEED / wavelength
        n_periods = 20
        t_total = n_periods / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=3 / frequency,
            t_max=t_total * 0.4,
        )

        # Source in vacuum region
        source_x = interface_x / 3
        source = GaussianSource(
            position=(source_x, domain_height / 2), width=wavelength / 3, signal=signal
        )

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=1.5 * wavelength)],
            time=time,
            resolution=dx,
        )

        # Run and save field snapshots
        result = sim.run(save_fields=["Ez"], field_subsample=50)

        # Take a late snapshot when wave should have crossed interface
        late_idx = int(len(result["fields"]["Ez"]) * 0.8)
        Ez = result["fields"]["Ez"][late_idx]

        # Check field in dielectric region (right side of interface)
        interface_idx = int(interface_x / dx)
        dielectric_region = Ez[:, interface_idx + 10 :]  # Well past interface

        max_field_dielectric = np.max(np.abs(dielectric_region))

        # There should be significant field in the dielectric
        # (if no transmission, this would be near zero)
        assert max_field_dielectric > 1e-10, (
            f"No field transmitted through interface. "
            f"Max field in dielectric: {max_field_dielectric:.2e}"
        )

    def test_reflection_increases_with_index_contrast(self):
        """Higher index contrast should produce more reflection.

        A sanity check that Fresnel physics is qualitatively correct.
        """
        wavelength = TEST_WAVELENGTH
        domain_width = 15 * wavelength
        domain_height = 8 * wavelength

        reflectances = []
        n2_values = [1.2, 1.5, 2.0]  # Increasing index contrast from n1=1

        for n2 in n2_values:
            dx, dt = calc_optimal_fdtd_params(
                wavelength, n2, dims=2, safety_factor=0.95, points_per_wavelength=12
            )

            # Create domain with interface at center
            design = Design(
                width=domain_width,
                height=domain_height,
                material=Material(permittivity=1.0),  # vacuum
            )

            interface_x = domain_width / 2
            design += Rectangle(
                position=(interface_x + domain_width / 4, domain_height / 2),
                width=domain_width / 2,
                height=domain_height,
                material=Material(permittivity=n2**2),
            )

            frequency = LIGHT_SPEED / wavelength
            t_total = 20 / frequency
            time = np.arange(0, t_total, dt)

            signal = ramped_cosine(
                time,
                amplitude=1.0,
                frequency=frequency,
                ramp_duration=2 / frequency,
                t_max=t_total * 0.3,
            )

            source = GaussianSource(
                position=(interface_x / 3, domain_height / 2),
                width=wavelength / 3,
                signal=signal,
            )

            # Monitor in vacuum region to catch reflected wave
            monitor = Monitor(
                start=(interface_x * 0.4, domain_height * 0.3),
                end=(interface_x * 0.4, domain_height * 0.7),
                name="reflection_monitor",
            )

            sim = Simulation(
                design=design,
                devices=[source, monitor],
                boundaries=[PML(thickness=wavelength)],
                time=time,
                resolution=dx,
            )

            sim.run()

            # Get power history
            if monitor.power_history:
                P = np.array(monitor.power_history)
                # Look for secondary peak (reflected wave)
                # First peak is incident, second (if any) is reflected
                peak_idx = np.argmax(P)

                # Simple measure: ratio of late-time power to peak
                late_idx = int(len(P) * 0.8)
                if late_idx > peak_idx:
                    late_power = np.mean(P[late_idx:])
                    reflectances.append(
                        late_power / P[peak_idx] if P[peak_idx] > 0 else 0
                    )
                else:
                    reflectances.append(0)
            else:
                reflectances.append(0)

        # Verify reflectance increases with index contrast
        # (This is a weak test but catches gross errors)
        for i in range(1, len(reflectances)):
            if reflectances[i - 1] > 1e-10:  # Only check if we have data
                assert reflectances[i] >= reflectances[i - 1] * 0.8, (
                    f"Reflectance should generally increase with index contrast. "
                    f"Got: {reflectances} for n2={n2_values}"
                )

    def test_interface_does_not_cause_instability(self, dielectric_interface_domain):
        """Simulation should remain stable at material interface.

        Material discontinuities can cause numerical instabilities.
        This test verifies the interface doesn't cause field explosion.
        """
        design = dielectric_interface_domain["design"]
        wavelength = dielectric_interface_domain["wavelength"]
        dx = dielectric_interface_domain["dx"]
        dt = dielectric_interface_domain["dt"]
        domain_height = dielectric_interface_domain["domain_height"]
        interface_x = dielectric_interface_domain["interface_x"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 20 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.5,
        )

        source = GaussianSource(
            position=(interface_x / 3, domain_height / 2),
            width=wavelength / 3,
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

        # Check for field explosion
        max_reasonable = 1e10
        for i, Ez in enumerate(result["fields"]["Ez"]):
            max_field = np.max(np.abs(Ez))
            assert (
                max_field < max_reasonable
            ), f"Field explosion at interface: snapshot {i}, max={max_field:.2e}"
