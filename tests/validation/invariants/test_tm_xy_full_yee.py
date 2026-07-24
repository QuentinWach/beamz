from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices._boundary_compile import compile_metallic_masks
from beamz.lattice import advance_e_field, advance_h_field, component_shapes
from beamz.simulation.kernels import (
    tm_xy_curl_e_to_h_2d,
    tm_xy_curl_h_to_e_2d,
)


def _step(ez, hx, hy, *, dx, dt, edges=frozenset()):
    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d(ez, dx, hx.shape, hy.shape, edges)
    hx = advance_h_field(hx, curl_hx, 0.0, dt)
    hy = advance_h_field(hy, curl_hy, 0.0, dt)
    curl_ez = tm_xy_curl_h_to_e_2d(hx, hy, dx, ez.shape, edges)
    ez = advance_e_field(ez, curl_ez, 0.0, 1.0, dt, (slice(None), slice(None)))
    return ez, hx, hy


def _packet(direction: str, *, ny=12, nx=192, dx=1e-6):
    x_e = np.arange(nx + 1) * dx
    x_h = (np.arange(nx) + 0.5) * dx
    x0, width, wavelength = 0.5 * nx * dx, 10 * dx, 24 * dx
    sign = 1.0 if direction == "+x" else -1.0
    ez_line = np.exp(-(((x_e - x0) / width) ** 2)) * np.cos(
        2 * np.pi * (x_e - x0) / wavelength
    )
    hy_line = (
        -sign
        * np.exp(-(((x_h - x0) / width) ** 2))
        * np.cos(2 * np.pi * (x_h - x0) / wavelength)
        / np.sqrt(MU_0 / EPS_0)
    )
    return (
        jnp.asarray(np.tile(ez_line, (ny + 1, 1)), dtype=jnp.float32),
        jnp.zeros((ny, nx + 1), dtype=jnp.float32),
        jnp.asarray(np.tile(hy_line, (ny + 1, 1)), dtype=jnp.float32),
    )


def _centroid(field):
    energy = np.sum(np.asarray(field) ** 2, axis=0)
    return float(np.sum(np.arange(energy.size) * energy) / np.sum(energy))


def test_canonical_tmxy_curls_match_maxwell_identities():
    ny, nx = 5, 7
    y = np.arange(ny + 1, dtype=np.float32)[:, None]
    x = np.arange(nx + 1, dtype=np.float32)[None, :]
    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d(
        y + 2 * x, 1.0, (ny, nx + 1), (ny + 1, nx), frozenset()
    )
    np.testing.assert_allclose(curl_hx, 1.0)
    np.testing.assert_allclose(curl_hy, -2.0)

    hx = (
        3
        * (np.arange(ny, dtype=np.float32)[:, None] + 0.5)
        * np.ones((1, nx + 1), dtype=np.float32)
    )
    hy = (
        5
        * np.ones((ny + 1, 1), dtype=np.float32)
        * (np.arange(nx, dtype=np.float32)[None, :] + 0.5)
    )
    curl_ez = tm_xy_curl_h_to_e_2d(hx, hy, 1.0, (ny + 1, nx + 1), frozenset())
    np.testing.assert_allclose(curl_ez[1:-1, 1:-1], 2.0)


def test_canonical_tmxy_constant_fields_have_zero_curl():
    ny, nx = 6, 8
    ez = np.ones((ny + 1, nx + 1), dtype=np.float32)
    hx = np.ones((ny, nx + 1), dtype=np.float32)
    hy = np.ones((ny + 1, nx), dtype=np.float32)
    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d(ez, 1.0, hx.shape, hy.shape, frozenset())
    curl_ez = tm_xy_curl_h_to_e_2d(hx, hy, 1.0, ez.shape, frozenset())
    np.testing.assert_allclose(curl_hx, 0.0)
    np.testing.assert_allclose(curl_hy, 0.0)
    np.testing.assert_allclose(curl_ez, 0.0)


def test_canonical_tmxy_packet_propagates_in_requested_direction():
    dx, dt = 1e-6, 0.25e-6 / LIGHT_SPEED
    plus = _packet("+x", dx=dx)
    minus = _packet("-x", dx=dx)
    plus_start, minus_start = _centroid(plus[0]), _centroid(minus[0])
    for _ in range(32):
        plus = _step(*plus, dx=dx, dt=dt)
        minus = _step(*minus, dx=dx, dt=dt)
    assert _centroid(plus[0]) > plus_start + 2
    assert _centroid(minus[0]) < minus_start - 2


def test_canonical_tmxy_pec_masks_cover_tangential_wall_samples():
    shape = (5, 7)
    masks = compile_metallic_masks(component_shapes(shape), shape, ())
    assert masks["Ez"].shape == (6, 8)
    assert bool(jnp.all(masks["Ez"][[0, -1], :]))
    assert bool(jnp.all(masks["Ez"][:, [0, -1]]))
    assert bool(jnp.all(masks["Hx"][:, [0, -1]]))
    assert bool(jnp.all(masks["Hy"][[0, -1], :]))
