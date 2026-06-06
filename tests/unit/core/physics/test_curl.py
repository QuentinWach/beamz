import jax.numpy as jnp
import numpy as np
import pytest

from beamz.const import EPS_0
from beamz.simulation.boundaries import (
    _cpml_ab_from_profiles,
    build_h_boundary_views_for_e_3d,
    cpml_curl_e_to_h_3d,
    cpml_curl_h_to_e_3d,
    cpml_update_e_from_h_3d,
    cpml_update_e_from_h_3d_shell_split,
    cpml_update_h_from_e_3d,
    full_pec_update_e_from_h_3d,
    full_pec_update_h_from_e_3d,
)
from beamz.simulation.ops import curl_e_to_h_3d, curl_h_to_e_3d

pytestmark = pytest.mark.unit


def test_curl_e_to_h_3d_linear_field_has_constant_z_component():
    nz = ny = nx = 6
    ex = jnp.broadcast_to(
        jnp.arange(ny, dtype=jnp.float32)[None, :, None],
        (nz, ny, nx - 1),
    )
    ey = -jnp.broadcast_to(
        jnp.arange(nx, dtype=jnp.float32)[None, None, :],
        (nz, ny - 1, nx),
    )
    ez = jnp.zeros((nz - 1, ny, nx), dtype=jnp.float32)

    curl_ex, curl_ey, curl_ez = curl_e_to_h_3d(ex, ey, ez, resolution=1.0)

    np.testing.assert_allclose(np.asarray(curl_ex), 0.0)
    np.testing.assert_allclose(np.asarray(curl_ey), 0.0)
    np.testing.assert_allclose(np.asarray(curl_ez), -2.0, atol=1e-6)


def test_curl_h_to_e_3d_linear_field_has_constant_y_component():
    nz = ny = nx = 6
    hx = jnp.broadcast_to(
        jnp.arange(nz - 1, dtype=jnp.float32)[:, None, None],
        (nz - 1, ny - 1, nx),
    )
    hy = jnp.zeros((nz - 1, ny, nx - 1), dtype=jnp.float32)
    hz = -jnp.broadcast_to(
        jnp.arange(nx - 1, dtype=jnp.float32)[None, None, :],
        (nz, ny - 1, nx - 1),
    )

    boundary_views = build_h_boundary_views_for_e_3d(hx, hy, hz, boundaries=[])
    curl_hx, curl_hy, curl_hz = curl_h_to_e_3d(
        hx,
        hy,
        hz,
        resolution=1.0,
        ex_shape=(nz, ny, nx - 1),
        ey_shape=(nz, ny - 1, nx),
        ez_shape=(nz - 1, ny, nx),
        boundary_views=boundary_views,
    )

    np.testing.assert_allclose(np.asarray(curl_hx)[1:-1, 1:-1, :], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(curl_hy)[1:-1, :, 1:-1], 2.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(curl_hz)[:, 1:-1, 1:-1], 0.0, atol=1e-6)


def _reference_full_pec_e_to_h_terms(ex, ey, ez, resolution):
    return (
        ((ez[:, 1:, :] - ez[:, :-1, :]) - (ey[1:, :, :] - ey[:-1, :, :]))
        / resolution,
        ((ex[1:, :, :] - ex[:-1, :, :]) - (ez[:, :, 1:] - ez[:, :, :-1]))
        / resolution,
        ((ey[:, :, 1:] - ey[:, :, :-1]) - (ex[:, 1:, :] - ex[:, :-1, :]))
        / resolution,
    )


def _reference_full_pec_h_to_e_terms(
    hx, hy, hz, resolution, ex_shape, ey_shape, ez_shape
):
    curl_x = jnp.zeros(ex_shape, dtype=hx.dtype)
    curl_y = jnp.zeros(ey_shape, dtype=hx.dtype)
    curl_z = jnp.zeros(ez_shape, dtype=hx.dtype)
    curl_x = curl_x.at[1:-1, 1:-1, :].set(
        (
            (hz[:, 1:, :] - hz[:, :-1, :])[1:-1, :, :]
            - (hy[1:, :, :] - hy[:-1, :, :])[:, 1:-1, :]
        )
        / resolution
    )
    curl_y = curl_y.at[1:-1, :, 1:-1].set(
        (
            (hx[1:, :, :] - hx[:-1, :, :])[:, :, 1:-1]
            - (hz[:, :, 1:] - hz[:, :, :-1])[1:-1, :, :]
        )
        / resolution
    )
    curl_z = curl_z.at[:, 1:-1, 1:-1].set(
        (
            (hy[:, :, 1:] - hy[:, :, :-1])[:, 1:-1, :]
            - (hx[:, 1:, :] - hx[:, :-1, :])[:, :, 1:-1]
        )
        / resolution
    )
    return curl_x, curl_y, curl_z


def test_full_pec_update_h_matches_curl_reference():
    ex = jnp.arange(5 * 6 * 6, dtype=jnp.float32).reshape(5, 6, 6) / 100.0
    ey = jnp.arange(5 * 5 * 7, dtype=jnp.float32).reshape(5, 5, 7) / 90.0
    ez = jnp.arange(4 * 6 * 7, dtype=jnp.float32).reshape(4, 6, 7) / 80.0
    hx = jnp.ones((4, 5, 7), dtype=jnp.float32)
    hy = jnp.ones((4, 6, 6), dtype=jnp.float32) * 2.0
    hz = jnp.ones((5, 5, 6), dtype=jnp.float32) * 3.0
    source = jnp.asarray(0.125, dtype=jnp.float32)
    mask_hx = jnp.zeros_like(hx, dtype=bool).at[:, 0, :].set(True)
    mask_hy = jnp.zeros_like(hy, dtype=bool).at[:, :, 0].set(True)
    mask_hz = jnp.zeros_like(hz, dtype=bool).at[0, :, :].set(True)

    curl_x, curl_y, curl_z = _reference_full_pec_e_to_h_terms(ex, ey, ez, 0.5)
    expected = (
        jnp.where(mask_hx, 0.0, hx - source * curl_x),
        jnp.where(mask_hy, 0.0, hy - source * curl_y),
        jnp.where(mask_hz, 0.0, hz - source * curl_z),
    )

    actual = full_pec_update_h_from_e_3d(
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        0.5,
        h_decay=(jnp.zeros((0, 0, 0), dtype=jnp.float32),) * 3,
        h_source=(source, source, source),
        h_mask=(mask_hx, mask_hy, mask_hz),
    )

    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-6)


def test_full_pec_update_e_matches_curl_reference():
    hx = jnp.arange(4 * 5 * 7, dtype=jnp.float32).reshape(4, 5, 7) / 100.0
    hy = jnp.arange(4 * 6 * 6, dtype=jnp.float32).reshape(4, 6, 6) / 90.0
    hz = jnp.arange(5 * 5 * 6, dtype=jnp.float32).reshape(5, 5, 6) / 80.0
    ex = jnp.ones((5, 6, 6), dtype=jnp.float32)
    ey = jnp.ones((5, 5, 7), dtype=jnp.float32) * 2.0
    ez = jnp.ones((4, 6, 7), dtype=jnp.float32) * 3.0
    region_x = (slice(1, -1), slice(1, -1), slice(None))
    region_y = (slice(1, -1), slice(None), slice(1, -1))
    region_z = (slice(None), slice(1, -1), slice(1, -1))
    source_x = jnp.zeros_like(ex).at[region_x].set(0.11)
    source_y = jnp.zeros_like(ey).at[region_y].set(0.12)
    source_z = jnp.zeros_like(ez).at[region_z].set(0.13)
    mask_ex = jnp.zeros_like(ex, dtype=bool).at[:, 0, :].set(True)
    mask_ey = jnp.zeros_like(ey, dtype=bool).at[:, :, 0].set(True)
    mask_ez = jnp.zeros_like(ez, dtype=bool).at[0, :, :].set(True)

    curl_x, curl_y, curl_z = _reference_full_pec_h_to_e_terms(
        hx, hy, hz, 0.5, ex.shape, ey.shape, ez.shape
    )
    expected = (
        jnp.where(mask_ex, 0.0, ex + source_x * curl_x),
        jnp.where(mask_ey, 0.0, ey + source_y * curl_y),
        jnp.where(mask_ez, 0.0, ez + source_z * curl_z),
    )

    actual = full_pec_update_e_from_h_3d(
        hx,
        hy,
        hz,
        ex,
        ey,
        ez,
        0.5,
        e_decay=(jnp.zeros((0, 0, 0), dtype=jnp.float32),) * 3,
        e_source=(source_x, source_y, source_z),
        e_mask=(mask_ex, mask_ey, mask_ez),
    )

    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-6)


def _cpml_coefficients(
    shapes, *, sigma_value=0.5, kappa_value=1.5, alpha_value=0.1, dt=0.05
):
    a_terms = []
    b_terms = []
    inv_kappa_terms = []
    for shape in shapes:
        sigma = jnp.full(shape, sigma_value, dtype=jnp.float32)
        kappa = jnp.full(shape, kappa_value, dtype=jnp.float32)
        alpha = jnp.full(shape, alpha_value, dtype=jnp.float32)
        a_term, b_term = _cpml_ab_from_profiles(sigma, kappa, alpha, dt)
        a_terms.append(a_term)
        b_terms.append(b_term)
        inv_kappa_terms.append(1.0 / kappa)
    return tuple(a_terms), tuple(b_terms), tuple(inv_kappa_terms)


def test_cpml_curl_e_to_h_3d_updates_psi_terms():
    ex = jnp.arange(3 * 4 * 4, dtype=jnp.float32).reshape(3, 4, 4)
    ey = jnp.arange(3 * 3 * 5, dtype=jnp.float32).reshape(3, 3, 5)
    ez = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5)

    term_shapes = (
        (2, 3, 5),
        (2, 3, 5),
        (2, 4, 4),
        (2, 4, 4),
        (3, 3, 4),
        (3, 3, 4),
    )
    a_terms, b_terms, inv_kappa_terms = _cpml_coefficients(term_shapes)
    psi_init = tuple(jnp.zeros(shape, dtype=jnp.float32) for shape in term_shapes)

    curl_hx, curl_hy, curl_hz, psi_updated = cpml_curl_e_to_h_3d(
        ex,
        ey,
        ez,
        resolution=0.2,
        a_h_terms=a_terms,
        b_h_terms=b_terms,
        inv_kappa_h_terms=inv_kappa_terms,
        psi_h_terms=psi_init,
    )

    assert curl_hx.shape == term_shapes[0]
    assert curl_hy.shape == term_shapes[2]
    assert curl_hz.shape == term_shapes[4]
    assert any(not jnp.allclose(term, 0.0) for term in psi_updated)


def test_cpml_curl_h_to_e_3d_updates_psi_terms():
    hx = jnp.arange(2 * 3 * 5, dtype=jnp.float32).reshape(2, 3, 5)
    hy = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    hz = jnp.arange(3 * 3 * 4, dtype=jnp.float32).reshape(3, 3, 4)

    term_shapes = (
        (3, 4, 4),
        (3, 4, 4),
        (3, 3, 5),
        (3, 3, 5),
        (2, 4, 5),
        (2, 4, 5),
    )
    a_terms, b_terms, inv_kappa_terms = _cpml_coefficients(term_shapes)
    psi_init = tuple(jnp.zeros(shape, dtype=jnp.float32) for shape in term_shapes)

    curl_ex, curl_ey, curl_ez, psi_updated = cpml_curl_h_to_e_3d(
        hx,
        hy,
        hz,
        resolution=0.2,
        a_e_terms=a_terms,
        b_e_terms=b_terms,
        inv_kappa_e_terms=inv_kappa_terms,
        psi_e_terms=psi_init,
    )

    assert curl_ex.shape == term_shapes[0]
    assert curl_ey.shape == term_shapes[2]
    assert curl_ez.shape == term_shapes[4]
    assert any(not jnp.allclose(term, 0.0) for term in psi_updated)


def test_cpml_curl_h_to_e_3d_uses_open_ghosts_on_nonmetal_edges():
    hx = jnp.zeros((2, 3, 5), dtype=jnp.float32)
    hy = jnp.zeros((2, 4, 4), dtype=jnp.float32)
    hz = jnp.ones((3, 3, 4), dtype=jnp.float32)
    term_shapes = (
        (3, 4, 4),
        (3, 4, 4),
        (3, 3, 5),
        (3, 3, 5),
        (2, 4, 5),
        (2, 4, 5),
    )
    zeros = tuple(jnp.zeros(shape, dtype=jnp.float32) for shape in term_shapes)
    ones = tuple(jnp.ones(shape, dtype=jnp.float32) for shape in term_shapes)

    curl_open, _, _, _ = cpml_curl_h_to_e_3d(
        hx,
        hy,
        hz,
        resolution=0.5,
        a_e_terms=zeros,
        b_e_terms=ones,
        inv_kappa_e_terms=ones,
        psi_e_terms=zeros,
        metallic_edges=frozenset(),
    )
    curl_pec, _, _, _ = cpml_curl_h_to_e_3d(
        hx,
        hy,
        hz,
        resolution=0.5,
        a_e_terms=zeros,
        b_e_terms=ones,
        inv_kappa_e_terms=ones,
        psi_e_terms=zeros,
        metallic_edges=frozenset({"bottom", "top"}),
    )

    assert jnp.allclose(curl_open[:, 0, :], 0.0)
    assert jnp.allclose(curl_open[:, -1, :], 0.0)
    assert jnp.all(curl_pec[:, 0, :] > 0.0)
    assert jnp.all(curl_pec[:, -1, :] < 0.0)


def test_cpml_update_h_from_e_3d_matches_curl_update_form():
    ex = jnp.arange(3 * 4 * 4, dtype=jnp.float32).reshape(3, 4, 4) / 10.0
    ey = jnp.arange(3 * 3 * 5, dtype=jnp.float32).reshape(3, 3, 5) / 11.0
    ez = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5) / 12.0
    hx = jnp.ones((2, 3, 5), dtype=jnp.float32)
    hy = jnp.ones((2, 4, 4), dtype=jnp.float32) * 2.0
    hz = jnp.ones((3, 3, 4), dtype=jnp.float32) * 3.0

    term_shapes = (hx.shape, hx.shape, hy.shape, hy.shape, hz.shape, hz.shape)
    a_terms, b_terms, inv_kappa_terms = _cpml_coefficients(term_shapes)
    psi_init = tuple(jnp.zeros(shape, dtype=jnp.float32) for shape in term_shapes)
    h_decay = (
        jnp.full(hx.shape, 0.91, dtype=jnp.float32),
        jnp.full(hy.shape, 0.92, dtype=jnp.float32),
        jnp.full(hz.shape, 0.93, dtype=jnp.float32),
    )
    h_source = (
        jnp.full(hx.shape, 0.11, dtype=jnp.float32),
        jnp.full(hy.shape, 0.12, dtype=jnp.float32),
        jnp.full(hz.shape, 0.13, dtype=jnp.float32),
    )

    curl_hx, curl_hy, curl_hz, psi_ref = cpml_curl_e_to_h_3d(
        ex,
        ey,
        ez,
        resolution=0.2,
        a_h_terms=a_terms,
        b_h_terms=b_terms,
        inv_kappa_h_terms=inv_kappa_terms,
        psi_h_terms=psi_init,
    )
    hx_ref = h_decay[0] * hx - h_source[0] * curl_hx
    hy_ref = h_decay[1] * hy - h_source[1] * curl_hy
    hz_ref = h_decay[2] * hz - h_source[2] * curl_hz

    hx_got, hy_got, hz_got, psi_got = cpml_update_h_from_e_3d(
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        h_decay[0],
        h_source[0],
        h_decay[1],
        h_source[1],
        h_decay[2],
        h_source[2],
        resolution=0.2,
        a_h_terms=a_terms,
        b_h_terms=b_terms,
        inv_kappa_h_terms=inv_kappa_terms,
        psi_h_terms=psi_init,
    )

    np.testing.assert_allclose(np.asarray(hx_got), np.asarray(hx_ref), atol=1e-6)
    np.testing.assert_allclose(np.asarray(hy_got), np.asarray(hy_ref), atol=1e-6)
    np.testing.assert_allclose(np.asarray(hz_got), np.asarray(hz_ref), atol=1e-6)
    for got, ref in zip(psi_got, psi_ref, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-6)


def test_cpml_update_e_from_h_3d_matches_curl_update_form():
    hx = jnp.arange(2 * 3 * 5, dtype=jnp.float32).reshape(2, 3, 5) / 10.0
    hy = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4) / 11.0
    hz = jnp.arange(3 * 3 * 4, dtype=jnp.float32).reshape(3, 3, 4) / 12.0
    ex = jnp.ones((3, 4, 4), dtype=jnp.float32)
    ey = jnp.ones((3, 3, 5), dtype=jnp.float32) * 2.0
    ez = jnp.ones((2, 4, 5), dtype=jnp.float32) * 3.0

    term_shapes = (ex.shape, ex.shape, ey.shape, ey.shape, ez.shape, ez.shape)
    a_terms, b_terms, inv_kappa_terms = _cpml_coefficients(term_shapes)
    psi_init = tuple(jnp.zeros(shape, dtype=jnp.float32) for shape in term_shapes)
    e_decay = (
        jnp.full(ex.shape, 0.81, dtype=jnp.float32),
        jnp.full(ey.shape, 0.82, dtype=jnp.float32),
        jnp.full(ez.shape, 0.83, dtype=jnp.float32),
    )
    e_source = (
        jnp.full(ex.shape, 0.21, dtype=jnp.float32),
        jnp.full(ey.shape, 0.22, dtype=jnp.float32),
        jnp.full(ez.shape, 0.23, dtype=jnp.float32),
    )
    metallic_edges = frozenset({"left", "right", "bottom", "top", "front", "back"})

    curl_ex, curl_ey, curl_ez, psi_ref = cpml_curl_h_to_e_3d(
        hx,
        hy,
        hz,
        resolution=0.2,
        a_e_terms=a_terms,
        b_e_terms=b_terms,
        inv_kappa_e_terms=inv_kappa_terms,
        psi_e_terms=psi_init,
        metallic_edges=metallic_edges,
    )
    ex_ref = e_decay[0] * ex + e_source[0] * curl_ex
    ey_ref = e_decay[1] * ey + e_source[1] * curl_ey
    ez_ref = e_decay[2] * ez + e_source[2] * curl_ez

    ex_got, ey_got, ez_got, psi_got = cpml_update_e_from_h_3d(
        hx,
        hy,
        hz,
        ex,
        ey,
        ez,
        e_decay[0],
        e_source[0],
        e_decay[1],
        e_source[1],
        e_decay[2],
        e_source[2],
        resolution=0.2,
        a_e_terms=a_terms,
        b_e_terms=b_terms,
        inv_kappa_e_terms=inv_kappa_terms,
        psi_e_terms=psi_init,
        metallic_edges=metallic_edges,
    )

    np.testing.assert_allclose(np.asarray(ex_got), np.asarray(ex_ref), atol=1e-6)
    np.testing.assert_allclose(np.asarray(ey_got), np.asarray(ey_ref), atol=1e-6)
    np.testing.assert_allclose(np.asarray(ez_got), np.asarray(ez_ref), atol=1e-6)
    for got, ref in zip(psi_got, psi_ref, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-6)


def test_cpml_shell_split_e_update_empty_source_matches_dense_permittivity_scale():
    hx = jnp.arange(2 * 3 * 5, dtype=jnp.float32).reshape(2, 3, 5) / 10.0
    hy = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4) / 11.0
    hz = jnp.arange(3 * 3 * 4, dtype=jnp.float32).reshape(3, 3, 4) / 12.0
    ex = jnp.ones((3, 4, 4), dtype=jnp.float32)
    ey = jnp.ones((3, 3, 5), dtype=jnp.float32) * 2.0
    ez = jnp.ones((2, 4, 5), dtype=jnp.float32) * 3.0
    eps_x = 1.0 + jnp.arange(ex.size, dtype=jnp.float32).reshape(ex.shape) / 50.0
    eps_y = 1.0 + jnp.arange(ey.size, dtype=jnp.float32).reshape(ey.shape) / 60.0
    eps_z = 1.0 + jnp.arange(ez.size, dtype=jnp.float32).reshape(ez.shape) / 70.0
    dt = jnp.asarray(5.0e-18, dtype=jnp.float32)
    dense_sources = (
        dt / (jnp.asarray(EPS_0, dtype=jnp.float32) * eps_x),
        dt / (jnp.asarray(EPS_0, dtype=jnp.float32) * eps_y),
        dt / (jnp.asarray(EPS_0, dtype=jnp.float32) * eps_z),
    )
    empty3 = jnp.zeros((0, 0, 0), dtype=jnp.float32)

    term_shapes = (ex.shape, ex.shape, ey.shape, ey.shape, ez.shape, ez.shape)
    a_terms, b_terms, inv_kappa_terms = _cpml_coefficients(
        term_shapes, dt=float(dt)
    )
    psi_init = tuple(jnp.zeros(shape, dtype=jnp.float32) for shape in term_shapes)
    shell_kwargs = {
        "e_lossy_shell_x": tuple(),
        "e_lossy_shell_y": tuple(),
        "e_lossy_shell_z": tuple(),
        "e_shell_decay_x": tuple(),
        "e_shell_source_x": tuple(),
        "e_shell_decay_y": tuple(),
        "e_shell_source_y": tuple(),
        "e_shell_decay_z": tuple(),
        "e_shell_source_z": tuple(),
    }

    dense = cpml_update_e_from_h_3d_shell_split(
        hx,
        hy,
        hz,
        ex,
        ey,
        ez,
        dense_sources[0],
        dense_sources[1],
        dense_sources[2],
        empty3,
        empty3,
        empty3,
        dt,
        resolution=0.2,
        a_e_terms=a_terms,
        b_e_terms=b_terms,
        inv_kappa_e_terms=inv_kappa_terms,
        psi_e_terms=psi_init,
        **shell_kwargs,
    )
    sparse = cpml_update_e_from_h_3d_shell_split(
        hx,
        hy,
        hz,
        ex,
        ey,
        ez,
        empty3,
        empty3,
        empty3,
        eps_x,
        eps_y,
        eps_z,
        dt,
        resolution=0.2,
        a_e_terms=a_terms,
        b_e_terms=b_terms,
        inv_kappa_e_terms=inv_kappa_terms,
        psi_e_terms=psi_init,
        **shell_kwargs,
    )

    for got, ref in zip(sparse[:3], dense[:3], strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-6)
    for got, ref in zip(sparse[3], dense[3], strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-6)
