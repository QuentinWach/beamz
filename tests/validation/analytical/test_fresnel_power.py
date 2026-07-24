"""Reference-normalized Fresnel power validation at normal incidence.

This case launches a source-free, right-going TM packet twice: once through a
uniform vacuum reference and once through a vacuum/dielectric interface.  The
separated packets are projected onto their right/left-going characteristic
fields, so the assertions measure power coefficients rather than field
presence or peak snapshots.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import (
    EPS_0,
    LIGHT_SPEED,
    MU_0,
    PEC,
    PML,
    Design,
    Material,
    Rectangle,
    Simulation,
    um,
)
from tests.utils import analytical_fresnel_r, analytical_fresnel_t

_CASE_ID = "fresnel-normal-vacuum-to-n1p5"


def _simulation(*, right_index: float, resolution: float, time: np.ndarray):
    wavelength = 1.0 * um
    interface_x = 8.0 * wavelength
    width = 16.0 * wavelength
    height = 20.0 * wavelength
    design = Design(
        width=width,
        height=height,
        material=Material(permittivity=1.0),
    )
    if right_index != 1.0:
        design += Rectangle(
            # Rectangle.position is its lower-left corner.
            position=(interface_x, 0.0),
            width=width - interface_x,
            height=height,
            material=Material(permittivity=right_index**2),
        )
    return Simulation(
        design=design,
        sources=[],
        boundaries=[
            PML(
                edges=["left", "right"],
                thickness=1.2 * wavelength,
                formulation="cpml",
            ),
            PEC(edges=["top", "bottom"]),
        ],
        time=time,
        resolution=resolution,
    )


def _right_going_packet_state(simulation: Simulation):
    """Initialize a staggered, approximately monochromatic +x TM packet."""
    wavelength = 1.0 * um
    center_x = 4.0 * wavelength
    envelope_sigma = 0.8 * wavelength
    impedance = np.sqrt(MU_0 / EPS_0)
    state = simulation.initial_state()

    electric_x = (np.arange(state.ez.shape[1]) + 0.5) * simulation.resolution
    magnetic_x = (np.arange(state.hy.shape[1]) + 1.0) * simulation.resolution
    wavenumber = 2.0 * np.pi / wavelength

    def profile(x):
        envelope = np.exp(-0.5 * ((x - center_x) / envelope_sigma) ** 2)
        return envelope * np.cos(wavenumber * (x - center_x))

    ez_line = profile(electric_x)
    hy_line = -profile(magnetic_x) / impedance
    return state._replace(
        ez=jnp.asarray(
            np.broadcast_to(ez_line, state.ez.shape),
            dtype=state.ez.dtype,
        ),
        hy=jnp.asarray(
            np.broadcast_to(hy_line, state.hy.shape),
            dtype=state.hy.dtype,
        ),
    )


def _characteristic_fields(ez, hy, *, refractive_index: float):
    """Collocate TM fields and return the +x and -x electric amplitudes."""
    electric = 0.5 * (np.asarray(ez)[:, 1:] + np.asarray(ez)[:, :-1])
    magnetic = np.asarray(hy)[:, : electric.shape[1]]
    impedance = np.sqrt(MU_0 / EPS_0) / refractive_index
    right_going = 0.5 * (electric - impedance * magnetic)
    left_going = 0.5 * (electric + impedance * magnetic)
    return right_going, left_going


@pytest.mark.simulation
def test_normal_incidence_fresnel_power_coefficients(validation_metrics):
    """Measured R and T agree with Fresnel theory and close the power budget."""
    wavelength = 1.0 * um
    n1 = 1.0
    n2 = 1.5
    points_per_wavelength = 18
    resolution = wavelength / (points_per_wavelength * n2)
    dt = 0.95 * resolution / (LIGHT_SPEED * np.sqrt(2.0))
    time = np.arange(0.0, 7.5 * wavelength / LIGHT_SPEED, dt)

    reference_sim = _simulation(
        right_index=n1,
        resolution=resolution,
        time=time,
    )
    interface_sim = _simulation(
        right_index=n2,
        resolution=resolution,
        time=time,
    )
    reference = reference_sim.advance(
        state=_right_going_packet_state(reference_sim),
        progress=False,
    ).state
    interface = interface_sim.advance(
        state=_right_going_packet_state(interface_sim),
        progress=False,
    ).state

    reference_right, _ = _characteristic_fields(
        reference.ez,
        reference.hy,
        refractive_index=n1,
    )
    _, reflected_left = _characteristic_fields(
        interface.ez,
        interface.hy,
        refractive_index=n1,
    )
    transmitted_right, _ = _characteristic_fields(
        interface.ez,
        interface.hy,
        refractive_index=n2,
    )

    x = (np.arange(reference_right.shape[1]) + 1.0) * resolution
    y = (np.arange(reference_right.shape[0]) + 0.5) * resolution
    central_y = (y > 7.8 * wavelength) & (y < 12.2 * wavelength)
    left = (x > 1.5 * wavelength) & (x < 7.6 * wavelength)
    right = (x > 8.4 * wavelength) & (x < 14.6 * wavelength)
    interior = (x > 1.5 * wavelength) & (x < 14.6 * wavelength)

    reference_energy = np.sum(reference_right[np.ix_(central_y, interior)] ** 2)
    reflected_energy = np.sum(reflected_left[np.ix_(central_y, left)] ** 2)
    transmitted_energy = np.sum(transmitted_right[np.ix_(central_y, right)] ** 2)

    # The packet is spatially compressed by n2.  Integrating electric energy
    # density therefore requires epsilon_r=n2**2 before reference normalization.
    measured_r = reflected_energy / reference_energy
    measured_t = n2**2 * transmitted_energy / reference_energy
    analytical_r = analytical_fresnel_r(n1, n2)
    analytical_t = analytical_fresnel_t(n1, n2)
    metadata = {
        "case": _CASE_ID,
        "n1": n1,
        "n2": n2,
        "steps": len(time),
        "points_per_wavelength_in_n2": points_per_wavelength,
        "measurement": "characteristic packet energy",
    }

    validation_metrics.check(
        "power reflectance R",
        measured=measured_r,
        reference=analytical_r,
        tolerance="analytical_coarse",
        resolution=f"{points_per_wavelength} ppw in n={n2}",
        metadata=metadata,
    )
    validation_metrics.check(
        "power transmittance T",
        measured=measured_t,
        reference=analytical_t,
        tolerance="analytical_coarse",
        resolution=f"{points_per_wavelength} ppw in n={n2}",
        metadata=metadata,
    )
    validation_metrics.check(
        "lossless power closure R+T",
        measured=measured_r + measured_t,
        reference=1.0,
        tolerance="normalized_power_balance",
        resolution=f"{points_per_wavelength} ppw in n={n2}",
        metadata=metadata,
    )
