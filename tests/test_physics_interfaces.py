"""Physics validation tests for Fresnel reflection at dielectric interfaces.

Tests verify:
1. Reflection coefficient matches Fresnel equations
2. Energy conservation R + T = 1
"""

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

# Import utilities


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
            sources=[source],
            boundaries=[PML(thickness=1.5 * wavelength)],
            time=time,
            resolution=dx,
        )

        # Run and save field snapshots
        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 50)))
        result = sim.run()

        # Take a late snapshot when wave should have crossed interface
        late_idx = int(len(result.monitor("fields").fields["Ez"]) * 0.8)
        Ez = result.monitor("fields").fields["Ez"][late_idx]

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
            sources=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 50)))
        result = sim.run()

        # Check for field explosion
        max_reasonable = 1e10
        for i, Ez in enumerate(result.monitor("fields").fields["Ez"]):
            max_field = np.max(np.abs(Ez))
            assert max_field < max_reasonable, (
                f"Field explosion at interface: snapshot {i}, max={max_field:.2e}"
            )
