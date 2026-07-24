"""A complete tiny lossless Yee step must be reversible without sources."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.lattice import advance_e_field, advance_h_field
from beamz.simulation.kernels import (
    tm_xy_curl_e_to_h_2d,
    tm_xy_curl_h_to_e_2d,
)


def _forward_lossless_step(ez, hx, hy, *, resolution, dt):
    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d(
        ez, resolution, hx.shape, hy.shape, frozenset()
    )
    next_hx = advance_h_field(hx, curl_hx, 0.0, dt)
    next_hy = advance_h_field(hy, curl_hy, 0.0, dt)
    curl_ez = tm_xy_curl_h_to_e_2d(next_hx, next_hy, resolution, ez.shape, frozenset())
    next_ez = advance_e_field(
        ez,
        curl_ez,
        0.0,
        1.0,
        dt,
        (slice(None), slice(None)),
    )
    return next_ez, next_hx, next_hy


def _reverse_lossless_step(ez, hx, hy, *, resolution, dt):
    curl_ez = tm_xy_curl_h_to_e_2d(hx, hy, resolution, ez.shape, frozenset())
    previous_ez = advance_e_field(
        ez,
        curl_ez,
        0.0,
        1.0,
        -dt,
        (slice(None), slice(None)),
    )
    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d(
        previous_ez, resolution, hx.shape, hy.shape, frozenset()
    )
    previous_hx = advance_h_field(hx, curl_hx, 0.0, -dt)
    previous_hy = advance_h_field(hy, curl_hy, 0.0, -dt)
    return previous_ez, previous_hx, previous_hy


def test_tiny_lossless_yee_step_round_trips_under_time_reversal(validation_metrics):
    ny, nx = 5, 7
    rng = np.random.default_rng(1307)
    initial = (
        jnp.asarray(rng.normal(size=(ny + 1, nx + 1)), dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(ny, nx + 1)), dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(ny + 1, nx)), dtype=jnp.float32),
    )
    resolution = 1e-6
    courant = 0.05
    dt = courant * resolution / LIGHT_SPEED

    advanced = _forward_lossless_step(*initial, resolution=resolution, dt=dt)
    recovered = _reverse_lossless_step(*advanced, resolution=resolution, dt=dt)
    numerator = np.sqrt(
        sum(
            np.linalg.norm(np.asarray(actual) - np.asarray(reference)) ** 2
            for actual, reference in zip(recovered, initial, strict=True)
        )
    )
    denominator = np.sqrt(
        sum(np.linalg.norm(np.asarray(value)) ** 2 for value in initial)
    )
    residual = float(numerator / denominator)

    validation_metrics.check(
        "lossless one-step time-reversal residual",
        measured=residual,
        reference=0.0,
        tolerance="kernel_float32",
        resolution=f"{ny}x{nx} TMxy cells",
        metadata={"courant": courant, "seed": 1307},
    )
