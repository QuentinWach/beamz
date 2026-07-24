"""Contracts for independent analytical helpers used by validation tests."""

from __future__ import annotations

import numpy as np
import pytest

from beamz import EPS_0, MU_0, um
from tests.utils import (
    analytical_cavity_frequency,
    analytical_fresnel_r,
    analytical_fresnel_t,
    compute_tm_field_energy,
    fabry_perot_fsr,
    fabry_perot_q_factor,
    mie_qext_2d,
    mie_qext_3d,
    mie_qsca_2d,
    mie_qsca_3d,
    slab_waveguide_neff_te,
    slab_waveguide_neff_tm,
)


def test_tm_energy_helper_includes_electric_and_magnetic_components():
    dx = 0.25
    ez = np.full((3, 4), 2.0)
    hx = np.full((2, 4), 3.0)
    hy = np.full((3, 3), 4.0)

    measured = compute_tm_field_energy(ez, hx, hy, dx, eps=2.5, mu=1.5)
    expected = (
        0.5
        * dx**2
        * (EPS_0 * 2.5 * np.sum(ez**2) + MU_0 * 1.5 * (np.sum(hx**2) + np.sum(hy**2)))
    )

    assert measured == pytest.approx(expected)


@pytest.mark.parametrize(
    ("n1", "n2"),
    [(1.0, 1.5), (1.5, 1.0), (1.0, 3.0), (2.0, 2.5)],
)
def test_lossless_fresnel_oracle_closes_power(n1, n2):
    reflectance = analytical_fresnel_r(n1, n2)
    transmittance = analytical_fresnel_t(n1, n2)

    assert reflectance + transmittance == pytest.approx(1.0, abs=1e-12)


def test_cavity_oracles_have_expected_scaling_and_high_q_limit():
    length = 1.0 * um
    index = 1.5
    fundamental = analytical_cavity_frequency(1, length, index)

    assert analytical_cavity_frequency(2, length, index) == pytest.approx(
        2.0 * fundamental
    )
    assert analytical_cavity_frequency(1, 2.0 * length, index) == pytest.approx(
        0.5 * fundamental
    )
    assert analytical_cavity_frequency(1, length, 2.0 * index) == pytest.approx(
        0.5 * fundamental
    )
    assert fabry_perot_fsr(length, index) == pytest.approx(fundamental)
    assert fabry_perot_q_factor(length, index, 0.99, 0.99) > 100.0


def test_lossless_mie_oracles_have_nonnegative_extinction_and_scattering():
    wavelength = 1.0 * um
    for size_parameter in (0.5, 1.0, 2.0, 5.0):
        radius = size_parameter * wavelength / (2.0 * np.pi)
        for extinction, scattering in (
            (
                mie_qext_2d(radius, wavelength, 2.0, 1.0),
                mie_qsca_2d(radius, wavelength, 2.0, 1.0),
            ),
            (
                mie_qext_3d(radius, wavelength, 2.0, 1.0),
                mie_qsca_3d(radius, wavelength, 2.0, 1.0),
            ),
        ):
            assert extinction > 0.0
            assert 0.0 < scattering <= 1.001 * extinction


@pytest.mark.parametrize("width_factor", [0.5, 1.0, 1.5])
def test_slab_mode_oracles_are_bounded_and_polarization_ordered(width_factor):
    n_core = 2.0
    n_clad = 1.0
    wavelength = 1.0 * um
    width = width_factor * wavelength
    neff_te = slab_waveguide_neff_te(n_core, n_clad, width, wavelength, mode=0)
    neff_tm = slab_waveguide_neff_tm(n_core, n_clad, width, wavelength, mode=0)

    assert neff_te is not None
    assert neff_tm is not None
    assert n_clad < neff_tm < neff_te < n_core


def test_slab_mode_oracle_respects_first_higher_order_cutoff():
    n_core = 2.0
    n_clad = 1.0
    wavelength = 1.0 * um
    normalized_frequency = np.pi / 4.0
    width = (
        2.0
        * normalized_frequency
        / ((2.0 * np.pi / wavelength) * np.sqrt(n_core**2 - n_clad**2))
    )

    assert slab_waveguide_neff_te(n_core, n_clad, width, wavelength, mode=0) is not None
    assert slab_waveguide_neff_te(n_core, n_clad, width, wavelength, mode=1) is None
