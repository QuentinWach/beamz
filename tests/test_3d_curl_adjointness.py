from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.simulation import ops
from beamz.simulation.boundaries import (
    build_h_boundary_views_for_e_3d,
    create_metallic_boundary_masks,
    pec_curl_e_to_h_3d,
    pec_curl_h_to_e_3d,
)
from beamz.simulation.fields import Fields


def _pairing_defect(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    curl_hx,
    curl_hy,
    curl_hz,
    curl_ex,
    curl_ey,
    curl_ez,
) -> float:
    lhs = float(jnp.vdot(ex, curl_hx) + jnp.vdot(ey, curl_hy) + jnp.vdot(ez, curl_hz))
    rhs = float(jnp.vdot(hx, curl_ex) + jnp.vdot(hy, curl_ey) + jnp.vdot(hz, curl_ez))
    denom = abs(lhs) + abs(rhs) + 1e-30
    return abs(lhs - rhs) / denom


def test_compact_3d_pec_curl_pair_is_skew_adjoint_after_compact_fix():
    nz, ny, nx = 6, 8, 18
    fields = Fields(
        np.ones((nz, ny, nx), dtype=np.float32),
        np.zeros((nz, ny, nx), dtype=np.float32),
        np.ones((nz, ny, nx), dtype=np.float32),
        resolution=1.0,
    )
    fields.set_metallic_masks(
        create_metallic_boundary_masks(
            fields,
            ["left", "right", "top", "bottom", "front", "back"],
            is_3d=True,
        )
    )
    rng = np.random.default_rng(0)

    ex = fields._apply_metallic_mask(
        "Ex", jnp.asarray(rng.standard_normal(fields.Ex.shape, dtype=np.float32))
    )
    ey = fields._apply_metallic_mask(
        "Ey", jnp.asarray(rng.standard_normal(fields.Ey.shape, dtype=np.float32))
    )
    ez = fields._apply_metallic_mask(
        "Ez", jnp.asarray(rng.standard_normal(fields.Ez.shape, dtype=np.float32))
    )
    hx = fields._apply_metallic_mask(
        "Hx", jnp.asarray(rng.standard_normal(fields.Hx.shape, dtype=np.float32))
    )
    hy = fields._apply_metallic_mask(
        "Hy", jnp.asarray(rng.standard_normal(fields.Hy.shape, dtype=np.float32))
    )
    hz = fields._apply_metallic_mask(
        "Hz", jnp.asarray(rng.standard_normal(fields.Hz.shape, dtype=np.float32))
    )

    curl_ex, curl_ey, curl_ez = pec_curl_e_to_h_3d(
        ex, ey, ez, 1.0, hx.shape, hy.shape, hz.shape
    )
    curl_hx, curl_hy, curl_hz = pec_curl_h_to_e_3d(
        hx, hy, hz, 1.0, ex.shape, ey.shape, ez.shape
    )

    defect = _pairing_defect(
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        curl_hx,
        curl_hy,
        curl_hz,
        curl_ex,
        curl_ey,
        curl_ez,
    )
    assert defect < 1e-6


def test_compact_3d_default_curl_pair_is_skew_adjoint_after_compact_fix():
    rng = np.random.default_rng(0)
    nz, ny, nx = 6, 8, 18
    ex = jnp.asarray(rng.standard_normal((nz, ny, nx - 1), dtype=np.float32))
    ey = jnp.asarray(rng.standard_normal((nz, ny - 1, nx), dtype=np.float32))
    ez = jnp.asarray(rng.standard_normal((nz - 1, ny, nx), dtype=np.float32))
    hx = jnp.asarray(rng.standard_normal((nz - 1, ny - 1, nx), dtype=np.float32))
    hy = jnp.asarray(rng.standard_normal((nz - 1, ny, nx - 1), dtype=np.float32))
    hz = jnp.asarray(rng.standard_normal((nz, ny - 1, nx - 1), dtype=np.float32))

    curl_ex, curl_ey, curl_ez = ops.curl_e_to_h_3d(ex, ey, ez, 1.0)
    views = build_h_boundary_views_for_e_3d(hx, hy, hz, boundaries=None)
    curl_hx, curl_hy, curl_hz = ops.curl_h_to_e_3d(
        hx,
        hy,
        hz,
        1.0,
        ex_shape=ex.shape,
        ey_shape=ey.shape,
        ez_shape=ez.shape,
        boundary_views=views,
    )

    defect = _pairing_defect(
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        curl_hx,
        curl_hy,
        curl_hz,
        curl_ex,
        curl_ey,
        curl_ez,
    )
    assert defect < 1e-6
