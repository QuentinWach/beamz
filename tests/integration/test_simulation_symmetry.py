"""End-to-end contracts for reduced-domain mirror symmetry."""

from __future__ import annotations

import numpy as np
import pytest

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.simulation.kernels import tm_xy_curl_h_to_e_2d

pytestmark = [pytest.mark.simulation, pytest.mark.integration]


def _time_and_impulse(steps: int = 8):
    time = np.arange(steps, dtype=float) * 1e-16
    signal = np.zeros(steps, dtype=float)
    signal[1] = 1.0
    return time, signal


def test_symmetry_request_reduces_each_active_axis_and_replaces_cut_absorbers():
    time, _ = _time_and_impulse(3)
    sim = bz.Simulation(
        domain=(4 * bz.um, 4 * bz.um, 4 * bz.um),
        resolution=1 * bz.um,
        time=time,
        boundaries=(bz.PML(edges="all", formulation="cpml"),),
        symmetry=(1, -1, 1),
    )

    request = sim.to_request()

    assert request.materials.shape == (2, 2, 2)
    assert request.materials.grid.shape == (2, 2, 2)
    assert request.boundaries[0].edges == ("left", "bottom", "front")
    assert any(
        isinstance(boundary, bz.PEC) and boundary.edges == ("top",)
        for boundary in request.boundaries
    )
    assert any(
        isinstance(boundary, bz.PMC)
        and set(boundary.edges) == {"right", "back"}
        for boundary in request.boundaries
    )


def test_symmetry_rejects_asymmetric_material_data():
    epsilon = np.ones((4, 4), dtype=float)
    epsilon[0, 0] = 2.0
    material_grid = MaterialGrid(
        permittivity=epsilon,
        conductivity=np.zeros_like(epsilon),
        permeability=np.ones_like(epsilon),
        resolution=1 * bz.um,
        shape=epsilon.shape,
    )
    time, _ = _time_and_impulse(3)
    sim = bz.Simulation(
        material_grid=material_grid,
        time=time,
        symmetry=(1, 0, 0),
    )

    with pytest.raises(ValueError, match="permittivity.*mirror symmetric"):
        sim.to_request()


def test_pmc_ghost_has_odd_magnetic_parity():
    hx = np.zeros((2, 3), dtype=np.float32)
    hy = np.ones((3, 2), dtype=np.float32)

    ordinary = np.asarray(
        tm_xy_curl_h_to_e_2d(hx, hy, 1.0, (3, 3), frozenset())
    )
    pmc = np.asarray(
        tm_xy_curl_h_to_e_2d(
            hx,
            hy,
            1.0,
            (3, 3),
            frozenset(),
            frozenset({"right"}),
        )
    )

    np.testing.assert_allclose(ordinary[:, -1], 0.0)
    np.testing.assert_allclose(pmc[:, -1], -2.0)


@pytest.mark.parametrize("parity", [1, -1])
def test_reduced_2d_tm_matches_full_domain_for_even_and_odd_sources(parity):
    time, impulse = _time_and_impulse()
    if parity == 1:
        sources = (
            bz.GaussianSource(
                position=(0.0, 0.0),
                width=0.5 * bz.um,
                signal=impulse,
            ),
        )
    else:
        sources = (
            bz.GaussianSource(
                position=(-0.5 * bz.um, 0.0),
                width=0.3 * bz.um,
                signal=impulse,
            ),
            bz.GaussianSource(
                position=(0.5 * bz.um, 0.0),
                width=0.3 * bz.um,
                signal=-impulse,
            ),
        )
    setup = dict(
        domain=(4 * bz.um, 4 * bz.um),
        resolution=0.2 * bz.um,
        time=time,
        sources=sources,
        boundaries=(bz.PEC(),),
    )

    full = bz.Simulation(**setup).advance(performance=False).state
    reduced = bz.Simulation(**setup, symmetry=(parity, 0, 0)).advance(
        performance=False
    ).state

    for name in ("ez", "hx", "hy"):
        full_field = np.asarray(getattr(full, name))
        reduced_field = np.asarray(getattr(reduced, name))
        retained = full_field[:, : reduced_field.shape[1]]
        np.testing.assert_allclose(reduced_field, retained, rtol=2e-6, atol=2e-12)
