"""Convergence of the full-tensor TE FDTD constitutive update."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.design import raster
from beamz.lattice import advance_h_field
from beamz.simulation.kernels import (
    advance_e_centered_tensor,
    te_xy_curl_e_to_h_2d,
    te_xy_curl_h_to_e_2d,
)


def _centroid(field, spacing):
    energy = np.sum(np.asarray(field) ** 2, axis=0)
    coordinates = (np.arange(energy.size) + 0.5) * spacing
    return float(np.sum(coordinates * energy) / np.sum(energy))


def _anisotropic_packet_displacement(cells: int) -> float:
    length = 192e-6
    spacing = length / cells
    epsilon = np.asarray(((2.0, 0.8), (0.8, 3.0)))
    inverse = np.linalg.inv(epsilon)
    effective_epsilon = np.linalg.det(epsilon) / epsilon[0, 0]
    impedance = np.sqrt(MU_0 / (EPS_0 * effective_epsilon))
    x_e = np.arange(cells + 1) * spacing
    x_h = (np.arange(cells) + 0.5) * spacing
    center, width, wavelength = 60e-6, 12e-6, 24e-6

    def packet(x):
        return np.exp(-(((x - center) / width) ** 2)) * np.cos(
            2.0 * np.pi * (x - center) / wavelength
        )

    ey = jnp.asarray(np.tile(packet(x_e), (4, 1)), dtype=jnp.float32)
    ex = jnp.asarray(
        np.tile((-epsilon[0, 1] / epsilon[0, 0]) * packet(x_h), (5, 1)),
        dtype=jnp.float32,
    )
    hz = jnp.asarray(np.tile(packet(x_h) / impedance, (4, 1)), dtype=jnp.float32)
    diagonals = (
        jnp.full_like(ex, inverse[0, 0]),
        jnp.full_like(ey, inverse[1, 1]),
    )
    offdiagonal = jnp.zeros((5, cells + 1, 3), dtype=jnp.float32)
    offdiagonal = offdiagonal.at[..., 0].set(inverse[0, 1])
    start = _centroid(hz, spacing)
    duration = 30e-15
    steps = round(duration / (0.2 * spacing / LIGHT_SPEED))
    dt = duration / steps
    for _ in range(steps):
        curl_hz = te_xy_curl_e_to_h_2d(ex, ey, spacing, hz.shape)
        hz = advance_h_field(hz, curl_hz, 0.0, dt)
        curls = te_xy_curl_h_to_e_2d(hz, spacing, ex.shape, ey.shape, frozenset())
        ex, ey = advance_e_centered_tensor(
            (ex, ey), curls, diagonals, offdiagonal, ("Ex", "Ey"), dt
        )
    return _centroid(hz, spacing) - start


def test_full_tensor_te_update_is_second_order_under_refinement():
    coarse, medium, fine = (
        _anisotropic_packet_displacement(cells) for cells in (96, 192, 384)
    )
    observed_order = math.log2(abs(coarse - medium) / abs(medium - fine))

    assert 1.5 < observed_order < 2.2


def _sloped_interface_simulation(cells: int, time: np.ndarray):
    length = 4.0 * bz.um
    spacing = length / cells
    scene = raster.Scene(
        (raster.Material(), raster.Material(4.0)),
        (
            raster.Object(
                raster.ExtrudedPolygon(
                    raster.Polygon(
                        (
                            (-length, 2.0 * length),
                            (2.0 * length, -length),
                            (2.0 * length, 2.0 * length),
                        )
                    ),
                    0.0,
                    1.0,
                ),
                1,
            ),
        ),
    )
    simulation = bz.Simulation(
        scene=scene,
        raster_grid=raster.Grid.uniform(
            (0.0, 0.0, 0.0), (length, length, 1.0), (cells, cells, 1)
        ),
        polarization="te",
        time=time,
        sources=(),
        normalize_source=None,
        raster_options=raster.RasterOptions(smoothing="farjadpour_full"),
    )
    state = simulation.initial_state()
    coordinates = (np.arange(cells) + 0.5) * spacing
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    initial_hz = np.exp(
        -((x - 1.7 * bz.um) ** 2 + (y - 1.7 * bz.um) ** 2) / (2.0 * (0.18 * bz.um) ** 2)
    )
    state = state._replace(hz=jnp.asarray(initial_hz, dtype=state.hz.dtype))
    return simulation, state


def _sloped_interface_probe(cells: int) -> float:
    length = 4.0 * bz.um
    spacing = length / cells
    steps = cells
    duration = 3.0e-15
    simulation, state = _sloped_interface_simulation(
        cells, np.arange(steps) * (duration / steps)
    )
    coordinates = (np.arange(cells) + 0.5) * spacing
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    final_hz = np.asarray(
        simulation.advance(state=state, progress=False).state.hz,
        dtype=np.float64,
    )
    probe = np.exp(
        -((x - 2.1 * bz.um) ** 2 + (y - 2.1 * bz.um) ** 2) / (2.0 * (0.35 * bz.um) ** 2)
    )
    return float(spacing**2 * np.sum(probe * final_hz**2))


@pytest.mark.simulation
def test_sloped_interface_fdtd_converges_under_grid_refinement():
    coarse, medium, fine = (_sloped_interface_probe(cells) for cells in (48, 96, 192))
    observed_order = math.log2(abs(coarse - medium) / abs(medium - fine))

    assert abs(medium - fine) < abs(coarse - medium)
    assert 1.5 < observed_order < 2.4


def _te_energy(state) -> float:
    electric = np.sum(np.asarray(state.ex, dtype=np.float64) ** 2) + np.sum(
        np.asarray(state.ey, dtype=np.float64) ** 2
    )
    magnetic = np.sum(np.asarray(state.hz, dtype=np.float64) ** 2)
    return float(EPS_0 * electric + MU_0 * magnetic)


@pytest.mark.simulation
def test_sloped_interface_full_tensor_update_is_stable_for_four_thousand_steps():
    cells = 48
    spacing = 4.0 * bz.um / cells
    segment_steps = 500
    segments = 8
    dt = 0.2 * spacing / LIGHT_SPEED
    simulation, state = _sloped_interface_simulation(
        cells, np.arange(segment_steps * segments) * dt
    )
    initial_energy = _te_energy(state)
    energies = []

    for _ in range(segments):
        state = simulation.advance(
            state=state,
            num_steps=segment_steps,
            progress=False,
        ).state
        assert all(
            np.all(np.isfinite(np.asarray(field)))
            for field in (state.ex, state.ey, state.hz)
        )
        energies.append(_te_energy(state))

    assert max(energies) < 2.0 * initial_energy
