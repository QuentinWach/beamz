"""Convergence of the full-tensor TE FDTD constitutive update."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.lattice import advance_h_field
from beamz.simulation.kernels import (
    advance_e_full_tensor,
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
    rows = (
        jnp.stack(
            (
                jnp.full_like(ex, inverse[0, 0]),
                jnp.full_like(ex, inverse[0, 1]),
                jnp.zeros_like(ex),
            ),
            axis=-1,
        ),
        jnp.stack(
            (
                jnp.full_like(ey, inverse[1, 0]),
                jnp.full_like(ey, inverse[1, 1]),
                jnp.zeros_like(ey),
            ),
            axis=-1,
        ),
    )
    start = _centroid(hz, spacing)
    duration = 30e-15
    steps = round(duration / (0.2 * spacing / LIGHT_SPEED))
    dt = duration / steps
    for _ in range(steps):
        curl_hz = te_xy_curl_e_to_h_2d(ex, ey, spacing, hz.shape)
        hz = advance_h_field(hz, curl_hz, 0.0, dt)
        curls = te_xy_curl_h_to_e_2d(hz, spacing, ex.shape, ey.shape, frozenset())
        ex, ey = advance_e_full_tensor((ex, ey), curls, rows, ("Ex", "Ey"), dt)
    return _centroid(hz, spacing) - start


def test_full_tensor_te_update_is_second_order_under_refinement():
    coarse, medium, fine = (
        _anisotropic_packet_displacement(cells) for cells in (96, 192, 384)
    )
    observed_order = math.log2(abs(coarse - medium) / abs(medium - fine))

    assert 1.5 < observed_order < 2.2
