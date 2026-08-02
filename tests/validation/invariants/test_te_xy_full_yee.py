from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices._boundary_compile import compile_metallic_masks
from beamz.lattice import advance_e_field, advance_h_field, component_shapes
from beamz.simulation.kernels import (
    advance_e_full_tensor,
    te_xy_curl_e_to_h_2d,
    te_xy_curl_h_to_e_2d,
)


def test_full_tensor_e_update_applies_cross_component_coupling():
    ex = jnp.zeros((4, 5), dtype=jnp.float32)
    ey = jnp.zeros((3, 6), dtype=jnp.float32)
    curl_ex = jnp.ones_like(ex)
    curl_ey = 2.0 * jnp.ones_like(ey)
    epsilon = np.asarray(((2.0, 0.5), (0.5, 3.0)))
    inverse = np.linalg.inv(epsilon)
    row_x = jnp.stack(
        (
            jnp.full_like(ex, inverse[0, 0]),
            jnp.full_like(ex, inverse[0, 1]),
            jnp.zeros_like(ex),
        ),
        axis=-1,
    )
    row_y = jnp.stack(
        (
            jnp.full_like(ey, inverse[1, 0]),
            jnp.full_like(ey, inverse[1, 1]),
            jnp.zeros_like(ey),
        ),
        axis=-1,
    )

    updated_ex, updated_ey = advance_e_full_tensor(
        (ex, ey),
        (curl_ex, curl_ey),
        (row_x, row_y),
        ("Ex", "Ey"),
        EPS_0,
    )

    np.testing.assert_allclose(updated_ex, inverse[0] @ (1.0, 2.0))
    np.testing.assert_allclose(updated_ey, inverse[1] @ (1.0, 2.0))


def _step(ex, ey, hz, *, dx, dt, edges=frozenset()):
    curl_hz = te_xy_curl_e_to_h_2d(ex, ey, dx, hz.shape)
    hz = advance_h_field(hz, curl_hz, 0.0, dt)
    curl_ex, curl_ey = te_xy_curl_h_to_e_2d(hz, dx, ex.shape, ey.shape, edges)
    region = (slice(None), slice(None))
    ex = advance_e_field(ex, curl_ex, 0.0, 1.0, dt, region)
    ey = advance_e_field(ey, curl_ey, 0.0, 1.0, dt, region)
    return ex, ey, hz


def _packet(direction: str, *, ny=12, nx=192, dx=1e-6):
    x_e = np.arange(nx + 1) * dx
    x_h = (np.arange(nx) + 0.5) * dx
    x0, width, wavelength = 0.5 * nx * dx, 10 * dx, 24 * dx
    sign = 1.0 if direction == "+x" else -1.0
    ey_line = np.exp(-(((x_e - x0) / width) ** 2)) * np.cos(
        2 * np.pi * (x_e - x0) / wavelength
    )
    hz_line = (
        sign
        * np.exp(-(((x_h - x0) / width) ** 2))
        * np.cos(2 * np.pi * (x_h - x0) / wavelength)
        / np.sqrt(MU_0 / EPS_0)
    )
    return (
        jnp.zeros((ny + 1, nx), dtype=jnp.float32),
        jnp.asarray(np.tile(ey_line, (ny, 1)), dtype=jnp.float32),
        jnp.asarray(np.tile(hz_line, (ny, 1)), dtype=jnp.float32),
    )


def _centroid(field):
    energy = np.sum(np.asarray(field) ** 2, axis=0)
    return float(np.sum(np.arange(energy.size) * energy) / np.sum(energy))


def test_canonical_texy_curls_match_maxwell_identities():
    ny, nx = 5, 7
    y = np.arange(ny + 1, dtype=np.float32)[:, None]
    x = np.arange(nx + 1, dtype=np.float32)[None, :]
    ex = y * np.ones((1, nx), dtype=np.float32)
    ey = 2 * x * np.ones((ny, 1), dtype=np.float32)
    curl_hz = te_xy_curl_e_to_h_2d(ex, ey, 1.0, (ny, nx))
    np.testing.assert_allclose(curl_hz, 1.0)

    hz = np.asarray(
        3 * (np.arange(ny)[:, None] + 0.5) + 5 * (np.arange(nx)[None, :] + 0.5),
        dtype=np.float32,
    )
    curl_ex, curl_ey = te_xy_curl_h_to_e_2d(
        hz, 1.0, (ny + 1, nx), (ny, nx + 1), frozenset()
    )
    np.testing.assert_allclose(curl_ex[1:-1], 3.0)
    np.testing.assert_allclose(curl_ey[:, 1:-1], -5.0)


def test_canonical_texy_constant_fields_have_zero_curl():
    ny, nx = 6, 8
    ex = np.ones((ny + 1, nx), dtype=np.float32)
    ey = np.ones((ny, nx + 1), dtype=np.float32)
    hz = np.ones((ny, nx), dtype=np.float32)
    np.testing.assert_allclose(te_xy_curl_e_to_h_2d(ex, ey, 1.0, hz.shape), 0.0)
    curl_ex, curl_ey = te_xy_curl_h_to_e_2d(hz, 1.0, ex.shape, ey.shape, frozenset())
    np.testing.assert_allclose(curl_ex, 0.0)
    np.testing.assert_allclose(curl_ey, 0.0)


def test_canonical_texy_packet_propagates_in_requested_direction():
    dx, dt = 1e-6, 0.25e-6 / LIGHT_SPEED
    plus = _packet("+x", dx=dx)
    minus = _packet("-x", dx=dx)
    plus_start, minus_start = _centroid(plus[1]), _centroid(minus[1])
    for _ in range(32):
        plus = _step(*plus, dx=dx, dt=dt)
        minus = _step(*minus, dx=dx, dt=dt)
    assert _centroid(plus[1]) > plus_start + 2
    assert _centroid(minus[1]) < minus_start - 2


def test_canonical_texy_pec_masks_cover_tangential_electric_samples():
    shape = (5, 7)
    masks = compile_metallic_masks(
        component_shapes(shape, "te"), shape, (), polarization_2d="te"
    )
    assert bool(jnp.all(masks["Ex"][[0, -1], :]))
    assert bool(jnp.all(masks["Ey"][:, [0, -1]]))
    assert not bool(jnp.any(masks["Hz"]))


def test_te_vacuum_packet_has_analytical_wave_speed():
    dx, dt = 1e-6, 0.2e-6 / LIGHT_SPEED
    fields = _packet("+x", dx=dx)
    start = _centroid(fields[1])
    steps = 40
    for _ in range(steps):
        fields = _step(*fields, dx=dx, dt=dt)
    measured_cells = _centroid(fields[1]) - start
    expected_cells = LIGHT_SPEED * steps * dt / dx
    assert measured_cells == pytest.approx(expected_cells, rel=0.2)
