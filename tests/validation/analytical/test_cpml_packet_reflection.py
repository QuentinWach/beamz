"""Compact, source-free CPML reflection measurements."""

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
    Simulation,
    um,
)


def _packet_reflection_db(
    *,
    refractive_index: float,
    points_per_wavelength: int,
) -> tuple[float, tuple[int, int], int]:
    """Launch a +x eigenpacket and measure its time-separated CPML return."""
    vacuum_wavelength = 1.0 * um
    medium_wavelength = vacuum_wavelength / refractive_index
    resolution = medium_wavelength / points_per_wavelength
    dt = 0.95 * resolution / (LIGHT_SPEED * np.sqrt(2.0))
    time = np.arange(
        0.0,
        12.0 * vacuum_wavelength / LIGHT_SPEED,
        dt,
    )
    impedance = np.sqrt(MU_0 / EPS_0) / refractive_index
    wave_speed = LIGHT_SPEED / refractive_index
    width = 12.0 * medium_wavelength
    height = 26.0 * medium_wavelength

    simulation = Simulation(
        design=Design(
            width=width,
            height=height,
            material=Material(permittivity=refractive_index**2),
        ),
        sources=[],
        boundaries=[
            PML(
                edges=["left", "right"],
                thickness=1.5 * medium_wavelength,
                formulation="cpml",
            ),
            PEC(edges=["top", "bottom"]),
        ],
        time=time,
        resolution=resolution,
    )
    initial = simulation.initial_state()
    electric_x = np.arange(initial.ez.shape[1]) * resolution
    magnetic_x = (np.arange(initial.hy.shape[1]) + 0.5) * resolution
    center_x = 3.0 * medium_wavelength
    sigma = 0.8 * medium_wavelength
    wavenumber = 2.0 * np.pi / medium_wavelength

    def packet(x):
        envelope = np.exp(-0.5 * ((x - center_x) / sigma) ** 2)
        return envelope * np.cos(wavenumber * (x - center_x))

    initial = initial._replace(
        ez=jnp.asarray(
            np.broadcast_to(packet(electric_x), initial.ez.shape),
            dtype=initial.ez.dtype,
        ),
        hy=jnp.asarray(
            np.broadcast_to(
                -packet(magnetic_x + 0.5 * wave_speed * dt) / impedance,
                initial.hy.shape,
            ),
            dtype=initial.hy.dtype,
        ),
    )

    initial_e = 0.5 * (np.asarray(initial.ez)[:, 1:] + np.asarray(initial.ez)[:, :-1])
    incident = 0.5 * (initial_e - impedance * np.asarray(initial.hy))
    final = simulation.advance(state=initial, progress=False).state
    final_e = 0.5 * (np.asarray(final.ez)[:, 1:] + np.asarray(final.ez)[:, :-1])
    reflected = 0.5 * (final_e + impedance * np.asarray(final.hy))

    x = (np.arange(reflected.shape[1]) + 0.5) * resolution
    y = np.arange(reflected.shape[0]) * resolution
    central_y = (y > 11.0 * medium_wavelength) & (y < 15.0 * medium_wavelength)
    interior_x = (x > 1.7 * medium_wavelength) & (x < 10.3 * medium_wavelength)
    incident_energy = np.sum(incident[np.ix_(central_y, interior_x)] ** 2)
    reflected_energy = np.sum(reflected[np.ix_(central_y, interior_x)] ** 2)
    reflection_db = 10.0 * np.log10(reflected_energy / incident_energy)
    return reflection_db, tuple(int(value) for value in initial.ez.shape), len(time)


@pytest.mark.simulation
@pytest.mark.parametrize(
    ("refractive_index", "points_per_wavelength"),
    [(1.0, 10), (1.0, 20), (1.5, 15)],
)
def test_cpml_normal_incidence_packet_reflection_is_below_minus_40_db(
    refractive_index,
    points_per_wavelength,
    validation_metrics,
):
    """Record CPML return loss across resolution and background index."""
    reflection_db, grid_shape, steps = _packet_reflection_db(
        refractive_index=refractive_index,
        points_per_wavelength=points_per_wavelength,
    )
    validation_metrics.check_upper(
        "CPML reflected packet power",
        measured=reflection_db,
        upper_bound=-40.0,
        unit="dB",
        resolution=(
            f"{points_per_wavelength} ppw in n={refractive_index}, 1.5 wavelength CPML"
        ),
        metadata={
            "refractive_index": refractive_index,
            "points_per_wavelength": points_per_wavelength,
            "grid_shape": list(grid_shape),
            "steps": steps,
            "incidence": "normal",
            "polarization": "TM",
            "measurement": "time-separated characteristic packet energy",
        },
    )
