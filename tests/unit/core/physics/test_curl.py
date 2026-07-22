import jax.numpy as jnp
import numpy as np
import pytest

from beamz.lattice import (
    build_h_boundary_views_for_e_3d,
    curl_e_to_h_3d,
    curl_h_to_e_3d,
    yee_flux,
)
from beamz.simulation.kernels import compile_cpml_term, cpml_coefficients
from tests.utils import compiled_grid


def test_yee_flux_integrates_signed_complex_poynting_component():
    ones = np.ones(4, dtype=np.complex128)
    zeros = np.zeros_like(ones)
    samples = (zeros, ones, zeros, zeros, zeros, 2.0 * ones)

    forward = yee_flux(samples, 0, normal_sign=1.0, measure=0.25, phasor=True)
    backward = yee_flux(samples, 0, normal_sign=-1.0, measure=0.25, phasor=True)

    assert float(forward) == 1.0
    assert float(backward) == -1.0


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

    boundary_views = build_h_boundary_views_for_e_3d(hx, hy, hz)
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


def test_fields_do_not_expose_obsolete_low_level_update_methods():
    material = np.ones((5, 6), dtype=np.float32)
    fields = compiled_grid(material, np.zeros_like(material), material, resolution=0.2)

    assert not hasattr(fields, "update_h")
    assert not hasattr(fields, "update_e")


def test_canonical_cpml_term_preserves_float64_when_x64_enabled():
    import jax

    previous_x64 = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
        shape = (2, 3, 4)
        sigma = jnp.full(shape, 0.2, dtype=dtype)
        kappa = jnp.full(shape, 1.4, dtype=dtype)
        alpha = jnp.full(shape, 0.03, dtype=dtype)
        coefficients = cpml_coefficients(sigma, kappa, alpha, np.float64(0.05))
        assert all(value.dtype == dtype for value in coefficients)

        term = compile_cpml_term(
            component="Ez",
            axis=2,
            sign=-1.0,
            sigma=sigma,
            kappa=kappa,
            alpha=alpha,
            dt=np.float64(0.05),
            full_shape=shape,
        )
        assert term.sign == -1.0
        assert all(value.dtype == dtype for value in (term.a, term.b, term.inv_kappa))
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
