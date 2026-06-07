"""Canonical boundary-physics tests.

These tests use a simple, physically grounded process:

- use small canonical domains with simple, physically interpretable geometry
- measure boundary behavior with detector-like quantities (DFT amplitudes) when
  possible instead of relying on raw snapshots alone
- assert quantitative wall/node or propagation metrics rather than only
  "simulation runs" smoke behavior

BeamZ does not currently expose PMC, periodic, Bloch, or a true uniform plane
wave source. The cases below therefore target the supported subset:

- PEC standing-wave node at a reflecting wall
- mixed PEC/PML channel propagation with quantitative speed and wall-suppression
  checks
"""

from __future__ import annotations

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PEC,
    PML,
    Design,
    GaussianSource,
    Material,
    Monitor,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from tests.utils import estimate_phase_velocity

pytestmark = [pytest.mark.simulation, pytest.mark.integration]


def _point_dft_monitor(name: str, x: float, y: float, frequency: float) -> Monitor:
    return Monitor(
        start=(x, y),
        end=(x, y),
        name=name,
        dft_enabled=True,
        dft_frequencies=np.array([frequency], dtype=float),
        dft_components=("Ez",),
        dft_record_every_step=True,
        dft_window="rect",
    )


def _dft_point_amplitude(monitor: Monitor, component: str = "Ez") -> float:
    arr = np.asarray(monitor.get_dft_component(component), dtype=np.complex128)
    return float(abs(complex(arr[0, 0])))


def _build_pec_standing_wave_case():
    wavelength = 1.0 * um
    width = 8.0 * wavelength
    height = 3.0 * wavelength
    dx, dt = calc_optimal_fdtd_params(
        wavelength,
        1.0,
        dims=2,
        safety_factor=0.95,
        points_per_wavelength=20,
    )
    frequency = LIGHT_SPEED / wavelength
    t_total = 18.0 / frequency
    time = np.arange(0.0, t_total, dt)
    signal = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=frequency,
        ramp_duration=2.0 / frequency,
        t_max=t_total,
    )

    design = Design(width=width, height=height, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(1.2 * wavelength, height / 2),
        width=1.2 * wavelength,
        signal=signal,
    )

    # Native xy TM stores Ez on the full node lattice, including the PEC wall.
    # Sample the wall node directly instead of the legacy near-wall interior cell.
    x_near = width
    x_far = width - 0.25 * wavelength
    near = _point_dft_monitor("near", x_near, height / 2, frequency)
    far = _point_dft_monitor("far", x_far, height / 2, frequency)

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[near, far],
        boundaries=[
            PML(edges=["left", "top", "bottom"], thickness=1.0 * wavelength),
            PEC(edges=["right"]),
        ],
        time=time,
        resolution=dx,
    )
    return sim, near, far


def _build_pec_pml_channel_case(*, with_monitors: bool):
    wavelength = 1.0 * um
    width = 10.0 * wavelength
    height = 4.0 * wavelength
    dx, dt = calc_optimal_fdtd_params(
        wavelength,
        1.0,
        dims=2,
        safety_factor=0.95,
        points_per_wavelength=20,
    )
    frequency = LIGHT_SPEED / wavelength
    t_total = 12.0 / frequency
    time = np.arange(0.0, t_total, dt)
    signal = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=frequency,
        ramp_duration=2.0 / frequency,
        t_max=t_total * 0.5,
    )

    design = Design(width=width, height=height, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(1.5 * wavelength, height / 2),
        width=1.0 * wavelength,
        signal=signal,
    )

    monitors = []
    if with_monitors:
        x_probe = 4.0 * wavelength
        monitors = [
            _point_dft_monitor("mid", x_probe, height / 2, frequency),
            _point_dft_monitor("wall", x_probe, height, frequency),
        ]

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[
            PML(edges=["left", "right"], thickness=1.0 * wavelength),
            PEC(edges=["top", "bottom"]),
        ],
        time=time,
        resolution=dx,
    )
    return sim, dx, dt


def test_pec_standing_wave_has_ez_node_at_reflecting_wall():
    """PEC wall should force a near-wall Ez node in a steady standing wave.

    Compare near-wall and λ/4-away DFT amplitudes and expect the near-wall
    amplitude to be strongly suppressed.
    """

    sim, near, far = _build_pec_standing_wave_case()
    sim.run(progress=False)

    amp_near = _dft_point_amplitude(near)
    amp_far = _dft_point_amplitude(far)

    assert amp_far > 0.0, "Far detector measured zero Ez amplitude."
    ratio = amp_near / amp_far
    assert ratio < 0.2, (
        f"PEC standing wave near/far ratio too large: {ratio:.3f} "
        f"(near={amp_near:.4e}, far={amp_far:.4e})."
    )


def test_pec_pml_channel_propagates_close_to_vacuum_speed():
    """Mixed PEC/PML channel should preserve near-vacuum propagation speed.

    Use a canonical channel geometry with absorbing boundaries along propagation
    and metallic walls transverse to it, then check the measured propagation
    speed quantitatively.
    """

    sim, dx, dt = _build_pec_pml_channel_case(with_monitors=False)
    result = sim.run(save_fields=["Ez"], field_subsample=10, progress=False)

    measured = estimate_phase_velocity(
        result["fields"]["Ez"],
        dx,
        dt * 10,
        threshold=0.2,
    )
    assert measured is not None, "Could not estimate propagation speed."

    rel_err = abs(measured - LIGHT_SPEED) / LIGHT_SPEED
    assert rel_err < 0.05, (
        f"Mixed PEC/PML channel speed error too large: {rel_err:.2%} "
        f"(measured={measured:.3e} m/s, expected={LIGHT_SPEED:.3e} m/s)."
    )


def test_pec_pml_channel_suppresses_tangential_e_field_near_wall():
    """Mixed PEC/PML channel should keep Ez strongly suppressed at the PEC wall."""

    sim, _dx, _dt = _build_pec_pml_channel_case(with_monitors=True)
    sim.run(progress=False)

    mid = next(m for m in sim.monitors if m.name == "mid")
    wall = next(m for m in sim.monitors if m.name == "wall")

    amp_mid = _dft_point_amplitude(mid)
    amp_wall = _dft_point_amplitude(wall)

    assert amp_mid > 0.0, "Centerline detector measured zero Ez amplitude."
    ratio = amp_wall / amp_mid
    assert ratio < 0.1, (
        f"PEC wall suppression too weak: wall/mid ratio={ratio:.3f} "
        f"(wall={amp_wall:.4e}, mid={amp_mid:.4e})."
    )


# Disabled from regular collection: these are large, expensive FDTD integration
# runs. Keep the bodies in place for manual re-enabling when needed.
test_pec_standing_wave_has_ez_node_at_reflecting_wall.__test__ = False
test_pec_pml_channel_propagates_close_to_vacuum_speed.__test__ = False
test_pec_pml_channel_suppresses_tangential_e_field_near_wall.__test__ = False
