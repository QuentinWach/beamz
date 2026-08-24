"""End-to-end contracts for reduced-domain mirror symmetry."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.simulation.kernels import tm_xy_curl_h_to_e_2d
from beamz.simulation.symmetry import expand_field_array

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
        isinstance(boundary, bz.PMC) and set(boundary.edges) == {"right", "back"}
        for boundary in request.boundaries
    )


@pytest.mark.parametrize(
    ("plane", "symmetry"),
    [("xy", (1, -1, 0)), ("xz", (1, 0, -1)), ("yz", (0, 1, -1))],
)
def test_2d_symmetry_tuple_always_uses_physical_axes(plane, symmetry):
    time, _ = _time_and_impulse(3)
    request = bz.Simulation(
        domain=(4 * bz.um, 4 * bz.um),
        resolution=1 * bz.um,
        time=time,
        plane_2d=plane,
        boundaries=(bz.PML(edges="all"),),
        symmetry=symmetry,
    ).to_request()

    assert request.materials.shape == (2, 2)
    assert {edge for boundary in request.boundaries for edge in boundary.edges} >= {
        "right",
        "top",
    }


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

    ordinary = np.asarray(tm_xy_curl_h_to_e_2d(hx, hy, 1.0, (3, 3), frozenset()))
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
    reduced = (
        bz.Simulation(**setup, symmetry=(parity, 0, 0)).advance(performance=False).state
    )

    for name in ("ez", "hx", "hy"):
        full_field = np.asarray(getattr(full, name))
        reduced_field = np.asarray(getattr(reduced, name))
        retained = full_field[:, : reduced_field.shape[1]]
        np.testing.assert_allclose(reduced_field, retained, rtol=2e-6, atol=2e-12)


@pytest.mark.parametrize("parity", [1, -1])
def test_domain_field_recorder_expands_to_full_vector_solution(parity):
    time, impulse = _time_and_impulse()
    sources = (
        (
            bz.GaussianSource(
                position=(0.0, 0.0),
                width=0.5 * bz.um,
                signal=impulse,
            ),
        )
        if parity == 1
        else (
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
    )
    recorder = bz.FieldRecorder(("Ez", "Hx", "Hy"), interval=1)
    setup = dict(
        domain=(4 * bz.um, 4 * bz.um),
        resolution=0.2 * bz.um,
        time=time,
        sources=sources,
        monitors=(recorder,),
        boundaries=(bz.PEC(),),
    )

    full = bz.Simulation(**setup).run(progress=False, performance=False)
    reduced = bz.Simulation(**setup, symmetry=(parity, 0, 0)).run(
        progress=False, performance=False
    )
    expanded = reduced.symmetry_expanded

    assert reduced.metadata.symmetry == (parity, 0, 0)
    assert expanded.metadata.symmetry == (0, 0, 0)
    assert expanded.metadata.fields.grid_shape == full.metadata.fields.grid_shape
    assert expanded.metadata.grid.shape == full.metadata.grid.shape
    for component in recorder.components:
        np.testing.assert_allclose(
            expanded[recorder.name].fields[component],
            full[recorder.name].fields[component],
            rtol=2e-6,
            atol=2e-12,
        )


@pytest.mark.parametrize("parity", [1, -1])
def test_reduced_3d_step_matches_full_domain_vector_update(parity):
    time, _ = _time_and_impulse(3)
    setup = dict(
        domain=(4 * bz.um, 4 * bz.um, 4 * bz.um),
        resolution=1 * bz.um,
        time=time,
        boundaries=(bz.PEC(),),
    )
    full_sim = bz.Simulation(**setup)
    reduced_sim = bz.Simulation(**setup, symmetry=(parity, 0, 0))
    reduced = reduced_sim.initial_state()
    rng = np.random.default_rng(7)
    reduced_ez = rng.normal(size=reduced.ez.shape).astype(np.float32)
    reduced_ez[:, 0, :] = 0.0
    reduced_ez[:, -1, :] = 0.0
    reduced_ez[:, :, 0] = 0.0
    if parity == -1:
        reduced_ez[:, :, -1] = 0.0
    full_ez = expand_field_array(
        reduced_ez,
        "Ez",
        (parity, 0, 0),
        is_3d=True,
        plane_2d="xy",
    )
    reduced = reduced._replace(ez=jnp.asarray(reduced_ez))
    full = full_sim.initial_state()._replace(ez=jnp.asarray(full_ez))

    reduced_next = reduced_sim.step(reduced, backend="jax")
    full_next = full_sim.step(full, backend="jax")

    for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
        reduced_field = np.asarray(getattr(reduced_next, name))
        full_field = np.asarray(getattr(full_next, name))
        retained = full_field[..., : reduced_field.shape[-1]]
        np.testing.assert_allclose(reduced_field, retained, rtol=2e-6, atol=2e-12)


@pytest.mark.parametrize("parity", [1, -1])
def test_reduced_2d_te_step_matches_full_domain_vector_update(parity):
    time, _ = _time_and_impulse(3)
    setup = dict(
        domain=(4 * bz.um, 4 * bz.um),
        resolution=0.5 * bz.um,
        time=time,
        polarization="te",
        boundaries=(bz.PEC(),),
    )
    full_sim = bz.Simulation(**setup)
    reduced_sim = bz.Simulation(**setup, symmetry=(parity, 0, 0))
    reduced = reduced_sim.initial_state()
    rng = np.random.default_rng(11)
    reduced_hz = rng.normal(size=reduced.hz.shape).astype(np.float32)
    full_hz = expand_field_array(
        reduced_hz,
        "Hz",
        (parity, 0, 0),
        is_3d=False,
        plane_2d="xy",
    )
    reduced = reduced._replace(hz=jnp.asarray(reduced_hz))
    full = full_sim.initial_state()._replace(hz=jnp.asarray(full_hz))

    reduced_next = reduced_sim.step(reduced, backend="jax")
    full_next = full_sim.step(full, backend="jax")

    for name in ("ex", "ey", "hz"):
        reduced_field = np.asarray(getattr(reduced_next, name))
        full_field = np.asarray(getattr(full_next, name))
        retained = full_field[:, : reduced_field.shape[1]]
        np.testing.assert_allclose(reduced_field, retained, rtol=2e-6, atol=2e-12)


def test_full_span_2d_flux_monitor_expands_with_physical_flux():
    steps = 40
    time = np.arange(steps, dtype=float) * 1e-16
    impulse = np.zeros(steps, dtype=float)
    impulse[1] = 1.0
    source = bz.GaussianSource(position=(0.0, 0.0), width=0.5 * bz.um, signal=impulse)
    monitor = bz.FluxMonitor(
        center=(0.5 * bz.um, 0.0, 0.0),
        size=(0.0, 4 * bz.um, 1.0),
        freqs=(2e14,),
        name="flux",
    )
    setup = dict(
        domain=(4 * bz.um, 4 * bz.um),
        resolution=0.2 * bz.um,
        time=time,
        sources=(source,),
        monitors=(monitor,),
        boundaries=(bz.PEC(),),
        normalize_source=None,
    )

    full = bz.Simulation(**setup).run(progress=False, performance=False)
    reduced = bz.Simulation(**setup, symmetry=(0, 1, 0)).run(
        progress=False, performance=False
    )
    expanded = reduced.symmetry_expanded

    for component in full["flux"].dft_fields:
        np.testing.assert_allclose(
            expanded["flux"].dft_fields[component],
            full["flux"].dft_fields[component],
            rtol=2e-6,
            atol=2e-12,
        )
    np.testing.assert_allclose(
        expanded["flux"].get_dft_flux(),
        full["flux"].get_dft_flux(),
        rtol=3e-6,
        atol=1e-26,
    )
