"""Tests for strict brush-feasible binary design generation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from beamz.optimization import (
    brush_feasibility_errors,
    circular_brush,
    conditional_generator,
    filtered_reward,
    generator_state,
    is_brush_feasible,
    morphological_opening,
    notched_square_brush,
    straight_through_gradient,
)


def test_brush_constructors_are_symmetric():
    circle = circular_brush(10)
    # A five-pixel minimum width/spacing rule uses L = 5 + 2.
    square = notched_square_brush(7)
    assert circle.shape == (10, 10)
    assert circle.area == 80
    np.testing.assert_array_equal(
        np.sum(circle.mask, axis=1),
        np.array([4, 8, 8, 10, 10, 10, 10, 8, 8, 4]),
    )
    assert np.array_equal(circle.mask, np.flip(circle.mask, axis=(0, 1)))
    assert square.shape == (7, 7)
    assert square.area == 45
    assert not square.mask[0, 0]
    assert not square.mask[0, -1]
    assert not square.mask[-1, 0]
    assert not square.mask[-1, -1]
    assert np.all(square.mask[1:-1, 1:-1])


def test_opening_detects_small_solid_and_void_features():
    brush = circular_brush(3)
    solid = np.zeros((15, 15), dtype=bool)
    solid[7, 7] = True
    solid_error, void_error = brush_feasibility_errors(solid, brush)
    assert solid_error[7, 7]
    assert not np.any(void_error)
    assert not is_brush_feasible(solid, brush)

    void_hole = np.ones((15, 15), dtype=bool)
    void_hole[7, 7] = False
    solid_error, void_error = brush_feasibility_errors(void_hole, brush)
    assert not np.any(solid_error)
    assert void_error[7, 7]

    assert is_brush_feasible(np.ones((15, 15)), brush)
    assert is_brush_feasible(np.zeros((15, 15)), brush)
    assert np.array_equal(
        morphological_opening(np.ones((15, 15)), brush),
        np.ones((15, 15), dtype=bool),
    )


def test_conditional_generator_is_binary_and_strictly_feasible():
    brush = circular_brush(5)
    reward = np.random.default_rng(4).normal(size=(24, 24))
    generated = conditional_generator(reward, brush)
    assert generated.steps <= reward.size
    assert set(np.unique(generated.density)) <= {0.0, 1.0}
    assert is_brush_feasible(generated.density, brush)
    assert not np.any(generated.solid_touches & generated.void_touches)


def test_conditional_generator_completes_compatible_initial_touches():
    brush = circular_brush(3)
    reward = np.linspace(-1.0, 1.0, 18 * 22).reshape(18, 22)
    initial_solid = np.zeros_like(reward, dtype=bool)
    initial_void = np.zeros_like(reward, dtype=bool)
    initial_solid[4, 5] = True
    initial_void[13, 16] = True

    scipy_generated = conditional_generator(
        reward,
        brush,
        initial_solid_touches=initial_solid,
        initial_void_touches=initial_void,
        backend="scipy",
    )
    compiled_generated = conditional_generator(
        reward,
        brush,
        initial_solid_touches=initial_solid,
        initial_void_touches=initial_void,
        backend="jax",
    )

    np.testing.assert_array_equal(
        compiled_generated.density,
        scipy_generated.density,
    )
    assert compiled_generated.solid_touches[4, 5]
    assert compiled_generated.void_touches[13, 16]
    assert is_brush_feasible(compiled_generated.density, brush)


def test_conditional_generator_supports_even_notched_square_brush():
    brush = notched_square_brush(12)
    reward = np.random.default_rng(12).normal(size=(32, 32))
    generated = conditional_generator(reward, brush)

    assert set(np.unique(generated.density)) <= {0.0, 1.0}
    assert is_brush_feasible(generated.density, brush)


def test_compiled_generator_matches_scipy_for_even_brush():
    brush = circular_brush(10)
    reward = np.random.default_rng(18).normal(size=(24, 25))
    scipy_generated = conditional_generator(reward, brush, backend="scipy")
    compiled_generated = conditional_generator(reward, brush, backend="jax")
    np.testing.assert_array_equal(
        compiled_generated.density,
        scipy_generated.density,
    )
    np.testing.assert_array_equal(
        compiled_generated.solid_touches,
        scipy_generated.solid_touches,
    )
    np.testing.assert_array_equal(
        compiled_generated.void_touches,
        scipy_generated.void_touches,
    )
    assert compiled_generated.steps == scipy_generated.steps


def test_fixed_context_is_precolored_and_generated_region_is_feasible():
    brush = circular_brush(3)
    reward = np.random.default_rng(21).normal(size=(16, 16))
    generated_region = np.zeros_like(reward, dtype=bool)
    generated_region[3:-3, 3:-3] = True
    fixed_void = ~generated_region
    fixed_solid = np.zeros_like(fixed_void)
    fixed_solid[7:10, :3] = True
    fixed_void[fixed_solid] = False

    scipy_generated = conditional_generator(
        reward,
        brush,
        fixed_solid=fixed_solid,
        fixed_void=fixed_void,
        backend="scipy",
    )
    compiled_generated = conditional_generator(
        reward,
        brush,
        fixed_solid=fixed_solid,
        fixed_void=fixed_void,
        backend="jax",
    )

    np.testing.assert_array_equal(compiled_generated.density, scipy_generated.density)
    assert np.all(compiled_generated.density[fixed_solid] == 1.0)
    assert np.all(compiled_generated.density[fixed_void] == 0.0)
    solid_error, void_error = brush_feasibility_errors(
        compiled_generated.density, brush
    )
    assert not np.any((solid_error | void_error) & generated_region)


def test_conditional_generator_imposes_exact_diagonal_symmetry():
    brush = circular_brush(5)
    reward = np.random.default_rng(31).normal(size=(24, 24))
    reward = 0.5 * (reward + reward.T)

    scipy_generated = conditional_generator(
        reward,
        brush,
        backend="scipy",
        diagonal_symmetry=True,
    )
    compiled_generated = conditional_generator(
        reward,
        brush,
        backend="jax",
        diagonal_symmetry=True,
    )

    np.testing.assert_array_equal(
        compiled_generated.density,
        scipy_generated.density,
    )
    np.testing.assert_array_equal(
        compiled_generated.density,
        compiled_generated.density.T,
    )
    assert is_brush_feasible(compiled_generated.density, brush)


def test_conditional_generator_imposes_exact_xy_reflection_with_fixed_ports():
    brush = circular_brush(3)
    shape = (20, 28)
    rng = np.random.default_rng(32)
    reward = rng.normal(size=shape)
    reward = 0.25 * (
        reward
        + np.flip(reward, axis=0)
        + np.flip(reward, axis=1)
        + np.flip(reward, axis=(0, 1))
    )
    generated_region = np.zeros(shape, dtype=bool)
    generated_region[3:-3, 3:-3] = True
    fixed_solid = np.zeros(shape, dtype=bool)
    fixed_solid[5:8, :3] = True
    fixed_solid[12:15, :3] = True
    fixed_solid[:, -3:] = np.flip(fixed_solid[:, :3], axis=1)
    fixed_void = ~generated_region & ~fixed_solid

    scipy_generated = conditional_generator(
        reward,
        brush,
        fixed_solid=fixed_solid,
        fixed_void=fixed_void,
        backend="scipy",
        reflection_symmetry="xy",
    )
    compiled_generated = conditional_generator(
        reward,
        brush,
        fixed_solid=fixed_solid,
        fixed_void=fixed_void,
        backend="jax",
        reflection_symmetry="xy",
    )

    np.testing.assert_array_equal(
        compiled_generated.density,
        scipy_generated.density,
    )
    np.testing.assert_array_equal(
        compiled_generated.density,
        np.flip(compiled_generated.density, axis=0),
    )
    np.testing.assert_array_equal(
        compiled_generated.density,
        np.flip(compiled_generated.density, axis=1),
    )
    assert np.all(compiled_generated.density[fixed_solid] == 1.0)
    assert np.all(compiled_generated.density[fixed_void] == 0.0)
    solid_error, void_error = brush_feasibility_errors(
        compiled_generated.density, brush
    )
    assert not np.any((solid_error | void_error) & generated_region)


def test_even_brush_reflection_uses_the_dual_touch_lattice():
    brush = circular_brush(10)
    shape = (24, 28)
    reward = np.random.default_rng(33).normal(size=shape)
    reward = 0.25 * (
        reward
        + np.flip(reward, axis=0)
        + np.flip(reward, axis=1)
        + np.flip(reward, axis=(0, 1))
    )

    scipy_generated = conditional_generator(
        reward,
        brush,
        backend="scipy",
        reflection_symmetry="xy",
    )
    compiled_generated = conditional_generator(
        reward,
        brush,
        backend="jax",
        reflection_symmetry="xy",
    )

    np.testing.assert_array_equal(
        compiled_generated.density,
        scipy_generated.density,
    )
    np.testing.assert_array_equal(
        compiled_generated.density,
        np.flip(compiled_generated.density, axis=0),
    )
    np.testing.assert_array_equal(
        compiled_generated.density,
        np.flip(compiled_generated.density, axis=1),
    )
    assert is_brush_feasible(compiled_generated.density, brush)


def test_conditional_generator_reward_selects_uniform_phases():
    brush = circular_brush(5)
    solid = conditional_generator(np.ones((20, 20)), brush)
    void = conditional_generator(-np.ones((20, 20)), brush)
    np.testing.assert_array_equal(solid.density, np.ones((20, 20)))
    np.testing.assert_array_equal(void.density, np.zeros((20, 20)))


def test_generator_states_match_supporting_information_figure_s1():
    """Reproduce the documented A6/A7/A0/E6 traversal through SI Figure S1."""

    brush = notched_square_brush(5)
    solid_touches = np.zeros((6, 8), dtype=bool)
    void_touches = np.zeros((6, 8), dtype=bool)

    initial = generator_state(solid_touches, void_touches, brush)
    assert np.all(initial.valid_solid)
    assert np.all(initial.valid_void)
    assert np.all(initial.possible_solid)
    assert np.all(initial.possible_void)

    # Step 2: void A6 makes A7 a redundant free void touch.
    void_touches[0, 6] = True
    after_a6 = generator_state(solid_touches, void_touches, brush)
    expected_a7 = np.zeros_like(void_touches)
    expected_a7[0, 7] = True
    np.testing.assert_array_equal(after_a6.free_void, expected_a7)
    design_after_a6 = after_a6.existing_void.copy()

    # Step 3: taking A7 changes touch state, but not the colored pixels.
    void_touches[0, 7] = True
    after_a7 = generator_state(solid_touches, void_touches, brush)
    np.testing.assert_array_equal(after_a7.existing_void, design_after_a6)

    # Steps 4 and 5: solid A0 followed by void E6 makes C4 required.
    solid_touches[0, 0] = True
    void_touches[4, 6] = True
    after_e6 = generator_state(solid_touches, void_touches, brush)
    assert after_e6.required_void[2, 4]  # C4
    resolving_not_free = after_e6.resolving_void & ~after_e6.free_void
    assert set(np.where(resolving_not_free)[1]) == {3, 4, 5}
    assert set(np.where(after_e6.free_void)[1]) <= {6, 7}
    assert np.any(after_e6.free_void)

    # Step 6: selecting every free touch resolves C4 without a remaining
    # required-resolving touch, exactly as stated in the SI caption.
    void_touches |= after_e6.free_void
    after_free = generator_state(solid_touches, void_touches, brush)
    assert after_free.existing_void[2, 4]
    assert not np.any(after_free.required_void)
    assert not np.any(after_free.resolving_void)


def test_filtered_reward_has_diagonal_symmetry():
    latent = jnp.arange(49, dtype=jnp.float32).reshape(7, 7)
    reward = filtered_reward(
        latent,
        circular_brush(3),
        beta=2.0,
        diagonal_symmetry=True,
    )
    np.testing.assert_allclose(reward, reward.T)


def test_filtered_reward_uses_raw_paper_convolution_by_default():
    brush = circular_brush(3)
    latent = jnp.ones((7, 7), dtype=jnp.float32) * 0.01
    raw = filtered_reward(latent, brush, beta=2.0)
    normalized = filtered_reward(
        latent,
        brush,
        beta=2.0,
        normalize_kernel=True,
    )
    assert float(raw[3, 3]) > float(normalized[3, 3])
    np.testing.assert_allclose(
        raw[3, 3],
        np.tanh(2.0 * 0.01 * brush.area),
        rtol=1e-6,
    )


def test_straight_through_gradient_matches_estimator_vjp():
    brush = circular_brush(3)
    latent = jnp.linspace(-0.2, 0.3, 36).reshape(6, 6)
    cotangent = jnp.linspace(-1.0, 1.0, 36).reshape(6, 6)

    def scalar(value):
        density_proxy = 0.5 * (
            filtered_reward(
                value,
                brush,
                beta=3.0,
                diagonal_symmetry=True,
            )
            + 1.0
        )
        return jnp.sum(density_proxy * cotangent)

    expected = jax.grad(scalar)(latent)
    actual = straight_through_gradient(
        latent,
        cotangent,
        brush,
        beta=3.0,
        diagonal_symmetry=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
