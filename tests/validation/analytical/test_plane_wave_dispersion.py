"""Plane-wave phase velocity, impedance, and preservation measurements."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import EPS_0, LIGHT_SPEED, MU_0, Design, Material, Simulation, um


def _complex_spatial_amplitude(
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    wavenumber: float,
    sample_slice: slice,
) -> complex:
    """Project a real field onto its positive-wavenumber Fourier component."""
    return complex(
        2.0
        * np.mean(
            values[sample_slice] * np.exp(-1j * wavenumber * coordinates[sample_slice])
        )
    )


@pytest.mark.simulation
@pytest.mark.parametrize("plane_2d", ["xy", "xz", "yz"])
@pytest.mark.parametrize("points_per_wavelength", [12, 20])
def test_plane_wave_matches_discrete_dispersion_and_impedance(
    plane_2d,
    points_per_wavelength,
    validation_metrics,
):
    """A propagated Yee eigenmode follows its discrete, not fitted, oracle."""
    wavelength = 1.0 * um
    resolution = wavelength / points_per_wavelength
    dt = 0.95 * resolution / (LIGHT_SPEED * np.sqrt(2.0))
    courant = LIGHT_SPEED * dt / resolution
    wavenumber = 2.0 * np.pi / wavelength
    omega_discrete = (2.0 / dt) * np.arcsin(
        courant * np.sin(0.5 * wavenumber * resolution)
    )
    impedance = np.sqrt(MU_0 / EPS_0)
    steps = 8

    simulation = Simulation(
        design=Design(
            width=8.0 * wavelength,
            height=4.0 * wavelength,
            material=Material(permittivity=1.0),
        ),
        sources=[],
        boundaries=[],
        time=np.arange(steps) * dt,
        resolution=resolution,
        plane_2d=plane_2d,
    )
    initial = simulation.initial_state()
    electric_x = np.arange(initial.ez.shape[1]) * resolution
    magnetic_x = (np.arange(initial.hy.shape[1]) + 0.5) * resolution

    # The stored H field is half a timestep behind E in the leapfrog state.
    electric_line = np.cos(wavenumber * electric_x)
    magnetic_line = (
        -np.cos(wavenumber * magnetic_x + 0.5 * omega_discrete * dt) / impedance
    )
    initial = initial._replace(
        ez=jnp.asarray(
            np.broadcast_to(electric_line, initial.ez.shape),
            dtype=initial.ez.dtype,
        ),
        hy=jnp.asarray(
            np.broadcast_to(magnetic_line, initial.hy.shape),
            dtype=initial.hy.dtype,
        ),
    )
    final = simulation.advance(state=initial, progress=False).state

    row = initial.ez.shape[0] // 2
    # Four exact spatial periods, far beyond the eight-step boundary light cone.
    sample_slice = slice(
        2 * points_per_wavelength,
        6 * points_per_wavelength,
    )
    initial_e = _complex_spatial_amplitude(
        np.asarray(initial.ez)[row],
        electric_x,
        wavenumber=wavenumber,
        sample_slice=sample_slice,
    )
    final_e = _complex_spatial_amplitude(
        np.asarray(final.ez)[row],
        electric_x,
        wavenumber=wavenumber,
        sample_slice=sample_slice,
    )
    final_h = _complex_spatial_amplitude(
        np.asarray(final.hy)[row],
        magnetic_x,
        wavenumber=wavenumber,
        sample_slice=sample_slice,
    )

    phase_advance = -float(np.angle(final_e / initial_e))
    assert 0.0 < phase_advance < np.pi, "test setup must avoid phase wrapping"
    measured_omega = phase_advance / (steps * dt)
    measured_velocity_ratio = measured_omega / (LIGHT_SPEED * wavenumber)
    discrete_velocity_ratio = omega_discrete / (LIGHT_SPEED * wavenumber)
    amplitude_ratio = abs(final_e) / abs(initial_e)
    impedance_ratio = abs(final_e / final_h) / impedance
    metadata = {
        "plane_2d": plane_2d,
        "points_per_wavelength": points_per_wavelength,
        "steps": steps,
        "courant": float(courant),
        "polarization": {"xy": "Ez", "xz": "Ey", "yz": "Ex"}[plane_2d],
    }
    resolution_label = f"{points_per_wavelength} ppw, {plane_2d}"

    validation_metrics.check(
        "phase velocity / c against discrete Yee relation",
        measured=measured_velocity_ratio,
        reference=discrete_velocity_ratio,
        tolerance="kernel_float32",
        resolution=resolution_label,
        metadata=metadata,
    )
    validation_metrics.check(
        "phase velocity / c against continuum",
        measured=measured_velocity_ratio,
        reference=1.0,
        tolerance="analytical_coarse",
        resolution=resolution_label,
        metadata=metadata,
    )
    validation_metrics.check(
        "plane-wave amplitude preservation",
        measured=amplitude_ratio,
        reference=1.0,
        tolerance="kernel_float32",
        resolution=resolution_label,
        metadata=metadata,
    )
    validation_metrics.check(
        "numerical impedance / vacuum impedance",
        measured=impedance_ratio,
        reference=1.0,
        tolerance="kernel_float32",
        resolution=resolution_label,
        metadata=metadata,
    )

    # PEC domain-wall cleanup can generate Hx only inside its short boundary
    # light cone; the measurement region remains purely +x polarized.
    np.testing.assert_array_equal(
        np.asarray(final.hx)[final.hx.shape[0] // 2, sample_slice],
        0.0,
    )
    for inactive in (final.ex, final.ey, final.hz):
        np.testing.assert_array_equal(np.asarray(inactive), 0.0)
