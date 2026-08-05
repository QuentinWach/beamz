import jax.numpy as jnp
import numpy as np
import pytest

from beamz import MU_0, RectilinearGrid
from beamz.lattice import (
    build_h_boundary_views_for_e_3d,
    curl_e_to_h_3d,
    curl_h_to_e_3d,
    curl_h_to_e_3d_metric,
    yee_flux,
)
from beamz.simulation.kernels import (
    compile_cpml_term,
    cpml_coefficients,
    fused_update_h_lossy_3d_material_metric,
    tm_xy_curl_e_to_h_2d,
    tm_xy_curl_e_to_h_2d_metric,
    tm_xy_curl_h_to_e_2d_metric,
)
from beamz.simulation.model import DerivativeMetricPlan
from tests.utils import compiled_grid


def test_yee_flux_integrates_signed_complex_poynting_component():
    ones = np.ones(4, dtype=np.complex128)
    zeros = np.zeros_like(ones)
    samples = (zeros, ones, zeros, zeros, zeros, 2.0 * ones)

    forward = yee_flux(samples, 0, normal_sign=1.0, measure=0.25, phasor=True)
    backward = yee_flux(samples, 0, normal_sign=-1.0, measure=0.25, phasor=True)

    assert float(forward) == 1.0
    assert float(backward) == -1.0


def test_yee_flux_accepts_local_rectilinear_quadrature_weights():
    zeros = np.zeros(2, dtype=np.complex128)
    ey = np.ones(2, dtype=np.complex128)
    hz = np.asarray([1.0, 2.0], dtype=np.complex128)
    samples = (zeros, ey, zeros, zeros, zeros, hz)

    flux = yee_flux(
        samples,
        0,
        measure=np.asarray([0.25, 0.75]),
        phasor=True,
    )

    assert float(flux) == pytest.approx(0.875)


pytestmark = pytest.mark.unit


def _metrics_2d(dx, dy):
    empty = jnp.zeros((0,), dtype=jnp.float32)
    dx = np.asarray(dx, dtype=np.float32)
    dy = np.asarray(dy, dtype=np.float32)

    def backward(widths):
        result = np.empty(widths.size + 1, dtype=np.float32)
        result[0] = 1.0 / widths[0]
        result[-1] = 1.0 / widths[-1]
        result[1:-1] = 2.0 / (widths[:-1] + widths[1:])
        return jnp.asarray(result)

    return DerivativeMetricPlan(
        jnp.asarray(1.0 / dx),
        jnp.asarray(1.0 / dy),
        empty,
        backward(dx),
        backward(dy),
        empty,
    )


def test_rectilinear_tm_curl_uses_physical_staggered_distances():
    x_edges = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)
    y_edges = np.asarray([0.0, 2.0, 5.0], dtype=np.float32)
    metrics = _metrics_2d(np.diff(x_edges), np.diff(y_edges))
    ez = jnp.asarray(2.0 * x_edges[None, :] + 3.0 * y_edges[:, None])

    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d_metric(ez, metrics)

    np.testing.assert_allclose(curl_hx, 3.0)
    np.testing.assert_allclose(curl_hy, -2.0)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    hy = jnp.broadcast_to(4.0 * x_centers[None, :], (3, 2))
    hx = jnp.zeros((2, 3), dtype=jnp.float32)
    curl_ez = tm_xy_curl_h_to_e_2d_metric(hx, hy, metrics, ez.shape, frozenset())
    np.testing.assert_allclose(curl_ez[:, 1:-1], 4.0)


def test_uniform_metric_curl_matches_legacy_scalar_algebra():
    ez = jnp.arange(20, dtype=jnp.float32).reshape(4, 5)
    metrics = DerivativeMetricPlan(
        jnp.asarray(2.0),
        jnp.asarray(2.0),
        jnp.zeros((0,), dtype=jnp.float32),
        jnp.asarray(2.0),
        jnp.asarray(2.0),
        jnp.zeros((0,), dtype=jnp.float32),
    )

    expected = tm_xy_curl_e_to_h_2d(ez, 0.5, (3, 5), (4, 4), frozenset())
    actual = tm_xy_curl_e_to_h_2d_metric(ez, metrics)

    for lhs, rhs in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(lhs, rhs)


def test_rectilinear_3d_h_update_differentiates_physical_coordinates():
    x_edges = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)
    y_edges = np.asarray([0.0, 2.0, 5.0], dtype=np.float32)
    z_edges = np.asarray([0.0, 4.0, 6.0], dtype=np.float32)
    empty = jnp.zeros((0,), dtype=jnp.float32)
    metrics = DerivativeMetricPlan(
        jnp.asarray(1.0 / np.diff(x_edges)),
        jnp.asarray(1.0 / np.diff(y_edges)),
        jnp.asarray(1.0 / np.diff(z_edges)),
        empty,
        empty,
        empty,
    )
    ex = jnp.zeros((3, 3, 2), dtype=jnp.float32)
    ey = jnp.zeros((3, 2, 3), dtype=jnp.float32)
    ez = jnp.broadcast_to(jnp.asarray(x_edges)[None, None, :], (2, 3, 3))
    hx = jnp.zeros((2, 2, 3), dtype=jnp.float32)
    hy = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    hz = jnp.zeros((3, 2, 2), dtype=jnp.float32)

    next_hx, next_hy, next_hz = fused_update_h_lossy_3d_material_metric(
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        0.0,
        0.0,
        0.0,
        MU_0,
        metrics,
    )

    np.testing.assert_allclose(next_hx, 0.0)
    np.testing.assert_allclose(next_hy, 1.0)
    np.testing.assert_allclose(next_hz, 0.0)


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


def test_rectilinear_curl_h_to_e_infers_yee_shapes_and_rejects_mismatch():
    nz = ny = nx = 4
    hx = jnp.arange(nz * ny * (nx + 1), dtype=jnp.float32).reshape(nz, ny, nx + 1)
    hy = jnp.zeros((nz, ny + 1, nx), dtype=jnp.float32)
    hz = jnp.zeros((nz + 1, ny, nx), dtype=jnp.float32)
    grid = RectilinearGrid(
        np.asarray([0.0, 0.8, 1.9, 3.3, 5.0]),
        np.asarray([0.0, 1.2, 2.1, 3.5, 5.4]),
        np.asarray([0.0, 0.7, 1.8, 3.2, 5.1]),
    )
    boundary_views = build_h_boundary_views_for_e_3d(hx, hy, hz)

    curls = curl_h_to_e_3d_metric(
        hx,
        hy,
        hz,
        grid,
        boundary_views=boundary_views,
    )

    assert tuple(curl.shape for curl in curls) == (
        (nz + 1, ny + 1, nx),
        (nz + 1, ny, nx + 1),
        (nz, ny + 1, nx + 1),
    )
    assert all(np.all(np.isfinite(np.asarray(curl))) for curl in curls)
    with pytest.raises(ValueError, match=r"curl\(H\) shapes"):
        curl_h_to_e_3d_metric(
            hx,
            hy,
            hz,
            grid,
            ex_shape=(1, 1, 1),
            ey_shape=(1, 1, 1),
            ez_shape=(1, 1, 1),
            boundary_views=boundary_views,
        )


def test_h_boundary_views_insert_high_ghost_before_storage_padding():
    hx = jnp.asarray([[[1.0, 2.0, 3.0, 4.0, 5.0, 0.0]]])
    hy = 2.0 * hx
    hz = 3.0 * hx
    logical_shapes = {name: (1, 1, 5) for name in ("Hx", "Hy", "Hz")}

    views = build_h_boundary_views_for_e_3d(
        hx,
        hy,
        hz,
        frozenset(),
        logical_shapes=logical_shapes,
    )

    np.testing.assert_array_equal(
        np.diff(np.asarray(views["hy_x"]), axis=2).reshape(-1)[:6],
        [0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
    )


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
