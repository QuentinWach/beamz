from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.lattice import (
    advance_e_field,
    advance_h_field,
    curl_e_to_h_3d,
)
from beamz.simulation.kernels import (
    compile_cpml_term,
    correct_cpml_term,
    cpml_coefficients,
)


def _random_yee_e_fields(
    shape: tuple[int, int, int], *, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Use small integers so commuting finite differences cancel exactly."""
    nz, ny, nx = shape
    rng = np.random.default_rng(seed)
    return tuple(
        jnp.asarray(rng.integers(-8, 9, size=component_shape), dtype=jnp.float32)
        for component_shape in (
            (nz, ny, nx - 1),
            (nz, ny - 1, nx),
            (nz - 1, ny, nx),
        )
    )


def test_complete_3d_constant_electric_field_has_zero_curl():
    ex, ey, ez = (
        jnp.full(shape, value, dtype=jnp.float32)
        for shape, value in zip(
            ((5, 6, 6), (5, 5, 7), (4, 6, 7)),
            (2.0, -3.0, 4.0),
            strict=True,
        )
    )

    curls = curl_e_to_h_3d(ex, ey, ez, resolution=0.25)

    for curl in curls:
        np.testing.assert_array_equal(np.asarray(curl), 0.0)


def test_discrete_divergence_of_complete_3d_curl_is_zero():
    resolution = 0.5
    ex, ey, ez = _random_yee_e_fields((6, 7, 8), seed=20260724)

    curl_x, curl_y, curl_z = curl_e_to_h_3d(ex, ey, ez, resolution)
    divergence = (
        jnp.diff(curl_x, axis=2) + jnp.diff(curl_y, axis=1) + jnp.diff(curl_z, axis=0)
    ) / resolution

    np.testing.assert_array_equal(np.asarray(divergence), 0.0)


def test_complete_3d_curl_is_equivariant_under_cyclic_axis_permutation():
    ex, ey, ez = _random_yee_e_fields((5, 6, 7), seed=731)
    curl_x, curl_y, curl_z = curl_e_to_h_3d(ex, ey, ez, resolution=0.4)

    # Physical cyclic relabeling: (x', y', z') = (y, z, x).
    # Stored array order is (z, y, x), hence the (2, 0, 1) transpose.
    transpose = (2, 0, 1)
    permuted_e = (
        jnp.transpose(ey, transpose),
        jnp.transpose(ez, transpose),
        jnp.transpose(ex, transpose),
    )
    permuted_curl = curl_e_to_h_3d(*permuted_e, resolution=0.4)
    expected = (
        jnp.transpose(curl_y, transpose),
        jnp.transpose(curl_z, transpose),
        jnp.transpose(curl_x, transpose),
    )

    for actual, reference in zip(permuted_curl, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


def test_lossy_local_updates_match_independent_scalar_derivation():
    rng = np.random.default_rng(81)
    field = rng.normal(size=(3, 4)).astype(np.float32)
    curl = rng.normal(size=(3, 4)).astype(np.float32)
    conductivity = rng.uniform(0.0, 0.03, size=(3, 4)).astype(np.float32)
    permeability = rng.uniform(1.0, 2.0, size=(3, 4)).astype(np.float32)
    dt = np.float32(2.5e-12)

    magnetic_sigma = conductivity * permeability
    h_alpha = magnetic_sigma * dt / (2.0 * MU_0)
    expected_h = ((1.0 - h_alpha) * field - (dt / MU_0) * curl) / (1.0 + h_alpha)
    actual_h = advance_h_field(field, curl, magnetic_sigma, dt)

    e_alpha = conductivity * dt / (2.0 * EPS_0 * permeability)
    expected_e = ((1.0 - e_alpha) * field + (dt / (EPS_0 * permeability)) * curl) / (
        1.0 + e_alpha
    )
    actual_e = advance_e_field(
        jnp.asarray(field),
        jnp.asarray(curl),
        jnp.asarray(conductivity),
        jnp.asarray(permeability),
        dt,
        (slice(None), slice(None)),
    )

    np.testing.assert_allclose(np.asarray(actual_h), expected_h, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual_e), expected_e, rtol=2e-6, atol=1e-6)


def test_cpml_coefficients_match_independent_recurrence_derivation():
    sigma = np.asarray([0.0, 0.2, 1.3, 2.1], dtype=np.float32)
    kappa = np.asarray([1.0, 1.0, 2.0, 4.0], dtype=np.float32)
    alpha = np.asarray([0.0, 0.03, 0.02, 0.0], dtype=np.float32)
    dt = np.float32(4e-13)

    actual_a, actual_b, actual_inv_kappa = cpml_coefficients(sigma, kappa, alpha, dt)
    expected_b = np.exp(-(sigma / kappa + alpha) * dt / EPS_0)
    denominator = np.maximum((sigma + kappa * alpha) * kappa, 1e-30)
    expected_a = np.nan_to_num(((expected_b - 1.0) * sigma) / denominator)

    np.testing.assert_allclose(actual_a, expected_a, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(actual_b, expected_b, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(actual_inv_kappa, 1.0 / kappa, rtol=1e-7)


def test_cpml_packed_recurrence_updates_only_active_boundary_slabs():
    sigma = np.asarray([0.8, 0.0, 0.0, 1.1], dtype=np.float32)
    kappa = np.asarray([1.7, 1.0, 1.0, 2.2], dtype=np.float32)
    alpha = np.asarray([0.04, 0.0, 0.0, 0.02], dtype=np.float32)
    derivative = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    dt = np.float32(2e-13)
    term = compile_cpml_term(
        component="Ez",
        axis=0,
        sign=-1.0,
        sigma=sigma,
        kappa=kappa,
        alpha=alpha,
        dt=dt,
        full_shape=derivative.shape,
    )
    initial_psi = jnp.zeros(term.slab.shape, dtype=jnp.float32)

    corrected, next_psi = correct_cpml_term(jnp.asarray(derivative), initial_psi, term)

    a, _b, inv_kappa = cpml_coefficients(sigma, kappa, alpha, dt)
    expected_psi = np.asarray(a) * derivative
    expected = derivative * np.asarray(inv_kappa) + expected_psi
    np.testing.assert_allclose(np.asarray(corrected), -expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(next_psi),
        expected_psi[[0, -1]],
        rtol=2e-6,
        atol=1e-7,
    )
