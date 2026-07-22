"""Projection method tests for topology optimization."""

import jax
import jax.numpy as jnp
import pytest

import beamz.optimization as optimization
from beamz.optimization.projections import (
    smoothed_heaviside,
    subpixel_smoothed_projection,
)


@pytest.mark.optimization
class TestSmoothedHeaviside:
    """Tests for the tanh smoothed Heaviside projection."""

    def test_beta_inf_returns_finite_hard_threshold(self):
        values = jnp.array([0.25, 0.5, 0.75])

        projected = smoothed_heaviside(values, beta=jnp.inf, eta=0.5)

        assert jnp.all(jnp.isfinite(projected))
        assert jnp.array_equal(projected, jnp.array([0.0, 0.0, 1.0]))


@pytest.mark.optimization
class TestSubpixelSmoothedProjection:
    """Tests for Hammond SSP1 projection behavior."""

    def test_exported_from_optimization_namespace(self):
        assert optimization.subpixel_smoothed_projection is subpixel_smoothed_projection

    def test_beta_inf_keeps_binary_input_binary(self):
        density = jnp.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )

        projected = subpixel_smoothed_projection(density, beta=jnp.inf, eta=0.5)

        assert jnp.all(jnp.isfinite(projected))
        assert jnp.all((projected == 0.0) | (projected == 1.0))

    def test_rejects_non_2d_input(self):
        density = jnp.ones((2, 2, 2))

        with pytest.raises(ValueError, match="2D"):
            subpixel_smoothed_projection(density, beta=1.0, eta=0.5)

    @pytest.mark.parametrize("value", [0.25, 0.5, 0.75])
    def test_constant_inputs_are_finite(self, value):
        density = jnp.full((4, 4), value)

        projected = subpixel_smoothed_projection(density, beta=5.0, eta=0.5)

        assert jnp.all(jnp.isfinite(projected))
        assert jnp.min(projected) >= -1e-6
        assert jnp.max(projected) <= 1.0 + 1e-6

    def test_beta_inf_smooth_disk_is_finite_and_bounded(self):
        coords = jnp.linspace(-1.0, 1.0, 31)
        yy, xx = jnp.meshgrid(coords, coords, indexing="ij")
        distance = jnp.sqrt(xx**2 + yy**2)
        density = jnp.clip(0.5 + (0.45 - distance) / 0.2, 0.0, 1.0)

        projected = subpixel_smoothed_projection(density, beta=jnp.inf, eta=0.5)

        assert jnp.all(jnp.isfinite(projected))
        assert jnp.min(projected) >= -1e-6
        assert jnp.max(projected) <= 1.0 + 1e-6
        assert projected[15, 15] == 1.0
        assert projected[0, 0] == 0.0

    def test_beta_inf_has_nonzero_interface_gradient(self):
        density = jnp.tile(jnp.linspace(0.0, 1.0, 21), (21, 1))

        def objective(values):
            projected = subpixel_smoothed_projection(
                values.reshape(density.shape), beta=jnp.inf, eta=0.5
            )
            return jnp.sum(projected)

        grad = jax.grad(objective)(density.ravel())

        assert jnp.all(jnp.isfinite(grad))
        assert jnp.max(jnp.abs(grad)) > 1e-6

    def test_away_from_interface_matches_heaviside(self):
        density = jnp.array([[0.1, 0.15, 0.2], [0.1, 0.15, 0.2]])

        projected = subpixel_smoothed_projection(density, beta=4.0, eta=0.5)
        expected = smoothed_heaviside(density, beta=4.0, eta=0.5)

        assert jnp.allclose(projected, expected, atol=1e-6)

    def test_rejects_non_positive_smoothing_radius(self):
        density = jnp.ones((3, 3))

        with pytest.raises(ValueError, match="ssp_smoothing_radius"):
            subpixel_smoothed_projection(
                density, beta=1.0, eta=0.5, ssp_smoothing_radius=0.0
            )
