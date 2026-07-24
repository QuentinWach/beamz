"""Demonstrate that key invariants reject representative operator mutations."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamz.lattice import curl_e_to_h_3d


def _random_complete_e_fields(shape, *, seed):
    nz, ny, nx = shape
    rng = np.random.default_rng(seed)
    return tuple(
        jnp.asarray(rng.normal(size=component_shape), dtype=jnp.float32)
        for component_shape in (
            (nz, ny, nx - 1),
            (nz, ny - 1, nx),
            (nz - 1, ny, nx),
        )
    )


def _divergence_residual(curls, resolution):
    curl_x, curl_y, curl_z = curls
    divergence = (
        jnp.diff(curl_x, axis=2) + jnp.diff(curl_y, axis=1) + jnp.diff(curl_z, axis=0)
    ) / resolution
    return float(jnp.max(jnp.abs(divergence)))


def _drop_d_ey_dz(fields, resolution):
    ex, ey, ez = fields
    curl_x, curl_y, curl_z = curl_e_to_h_3d(ex, ey, ez, resolution)
    dropped = curl_x + (ey[1:, :, :] - ey[:-1, :, :]) / resolution
    return dropped, curl_y, curl_z


def _flip_curl_x(fields, resolution):
    curl_x, curl_y, curl_z = curl_e_to_h_3d(*fields, resolution)
    return -curl_x, curl_y, curl_z


@pytest.mark.parametrize(
    "mutant",
    [_drop_d_ey_dz, _flip_curl_x],
    ids=["dropped-derivative", "wrong-sign"],
)
def test_divergence_identity_is_sensitive_to_curl_operator_mutations(
    mutant, validation_metrics
):
    resolution = 1.0
    fields = _random_complete_e_fields((5, 6, 7), seed=20260724)
    canonical = _divergence_residual(
        curl_e_to_h_3d(*fields, resolution),
        resolution,
    )
    mutated = _divergence_residual(mutant(fields, resolution), resolution)

    validation_metrics.check(
        "canonical divergence(curl(E)) residual",
        measured=canonical,
        reference=0.0,
        tolerance="kernel_float32",
        resolution="5x6x7 complete Yee support at unit spacing",
        metadata={"mutant": mutant.__name__, "seed": 20260724},
    )
    assert mutated > 1e3 * max(canonical, 1e-7)
