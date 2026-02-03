"""Topology optimization module tests.

Tests verify:
1. Density transformation pipeline (filter → projection) correctness
2. Gradient computation via finite differences validation
3. VJP (Vector-Jacobian Product) computation correctness
4. Filter bounds preservation [0,1]
5. Projection Heaviside function behavior
6. Integration test with small optimization problem
"""

import numpy as np
import pytest
import jax.numpy as jnp

from beamz.optimization.autodiff import (
    transform_density,
    compute_parameter_gradient_vjp,
    smoothed_heaviside,
    masked_conic_filter,
    masked_morphological_filter,
    generate_conic_kernel,
)
from beamz.optimization.topology import compute_overlap_gradient


@pytest.mark.optimization
class TestDensityTransformationPipeline:
    """Test the density transformation pipeline components."""

    def test_conic_filter_preserves_bounds(self):
        """Conic filter should keep values in [0,1] when input is in [0,1]."""
        # Create random density field [0, 1]
        np.random.seed(42)
        density = np.random.rand(20, 20)
        mask = np.ones((20, 20), dtype=bool)
        radius = 2

        # Convert to JAX arrays
        density_jax = jnp.array(density)
        mask_jax = jnp.array(mask)

        # Apply conic filter
        filtered, _ = masked_conic_filter(density_jax, mask_jax, radius)

        # Check bounds
        assert jnp.min(filtered) >= -1e-6, "Conic filter produced negative values"
        assert jnp.max(filtered) <= 1.0 + 1e-6, "Conic filter exceeded 1.0"

    def test_conic_kernel_normalization(self):
        """Conic kernel should sum to 1 (normalized)."""
        radius = 3
        kernel = generate_conic_kernel(radius)

        kernel_sum = jnp.sum(kernel)
        assert jnp.abs(kernel_sum - 1.0) < 1e-6, f"Kernel sum {kernel_sum} != 1.0"

    def test_heaviside_projection_bounds(self):
        """Heaviside projection should map [0,1] to [0,1] for all beta, eta."""
        values = jnp.linspace(0, 1, 100)
        beta_values = [1.0, 5.0, 20.0]
        eta_values = [0.3, 0.5, 0.7]

        for beta in beta_values:
            for eta in eta_values:
                projected = smoothed_heaviside(values, beta, eta)

                assert jnp.min(projected) >= -1e-6, \
                    f"Projection negative for beta={beta}, eta={eta}"
                assert jnp.max(projected) <= 1.0 + 1e-6, \
                    f"Projection exceeded 1.0 for beta={beta}, eta={eta}"

    def test_heaviside_projection_sharpening(self):
        """Higher beta should create sharper transitions (more binary)."""
        value = 0.5
        eta = 0.5

        # Low beta (smooth)
        result_low = smoothed_heaviside(value, beta=1.0, eta=eta)

        # High beta (sharp)
        result_high = smoothed_heaviside(value, beta=50.0, eta=eta)

        # At threshold (value=eta), both should give ~0.5
        assert jnp.abs(result_low - 0.5) < 0.1, "Low beta should be near 0.5 at threshold"
        assert jnp.abs(result_high - 0.5) < 0.1, "High beta should be near 0.5 at threshold"

        # Test non-threshold values show sharpening
        value_above = 0.6
        result_low_above = smoothed_heaviside(value_above, beta=1.0, eta=eta)
        result_high_above = smoothed_heaviside(value_above, beta=50.0, eta=eta)

        # Higher beta should push value closer to 1
        assert result_high_above > result_low_above, \
            "Higher beta should push values closer to binary"

    def test_morphological_filter_bounds(self):
        """Morphological filter should approximately preserve [0,1] bounds.

        Note: Morphological filters use smooth_min/smooth_max which can produce
        small violations (~0.05) due to LogSumExp approximation. This is acceptable
        for optimization where values get clipped anyway.
        """
        np.random.seed(42)
        density = np.random.rand(15, 15)
        mask = np.ones((15, 15), dtype=bool)
        radius = 2

        density_jax = jnp.array(density)
        mask_jax = jnp.array(mask)

        for operation in ["erosion", "dilation", "opening", "closing", "openclose"]:
            filtered = masked_morphological_filter(
                density_jax, mask_jax, radius, operation=operation
            )

            # Relaxed tolerance for morphological filters (~5% violation acceptable)
            assert jnp.min(filtered) >= -0.1, \
                f"{operation} filter produced values << 0: min = {jnp.min(filtered)}"
            assert jnp.max(filtered) <= 1.1, \
                f"{operation} filter produced values >> 1: max = {jnp.max(filtered)}"

    def test_transform_density_full_pipeline(self):
        """Full density transform should preserve bounds with both filter types."""
        np.random.seed(42)
        density = np.random.rand(10, 10)
        mask = np.ones((10, 10), dtype=bool)
        beta = 5.0
        eta = 0.5
        radius = 1

        density_jax = jnp.array(density)
        mask_jax = jnp.array(mask)

        for filter_type in ["conic", "morphological"]:
            physical = transform_density(
                density_jax,
                mask_jax,
                beta,
                eta,
                radius,
                filter_type=filter_type,
            )

            assert jnp.min(physical) >= -1e-3, \
                f"{filter_type}: transform produced negative values"
            assert jnp.max(physical) <= 1.0 + 1e-3, \
                f"{filter_type}: transform exceeded 1.0"


@pytest.mark.optimization
class TestGradientComputation:
    """Test gradient computation correctness via finite differences."""

    def test_compute_overlap_gradient_shape(self):
        """Overlap gradient should have same shape as field."""
        # Create simple forward and adjoint field histories
        n_steps = 10
        ny, nx = 20, 30

        forward_history = [np.random.randn(ny, nx) for _ in range(n_steps)]
        adjoint_history = [np.random.randn(ny, nx) for _ in range(n_steps)]

        grad = compute_overlap_gradient(forward_history, adjoint_history)

        assert grad.shape == (ny, nx), \
            f"Gradient shape {grad.shape} != field shape {(ny, nx)}"

    def test_compute_overlap_gradient_time_reversal(self):
        """Adjoint fields should be time-reversed in overlap computation."""
        n_steps = 5
        ny, nx = 10, 10

        # Create identifiable forward fields (increasing values)
        forward_history = [np.ones((ny, nx)) * i for i in range(n_steps)]

        # Create adjoint fields (increasing values)
        adjoint_history = [np.ones((ny, nx)) * i for i in range(n_steps)]

        grad = compute_overlap_gradient(forward_history, adjoint_history)

        # Manual computation: sum of forward[i] * adjoint[n-1-i]
        # forward[0]*adjoint[4] + forward[1]*adjoint[3] + ... + forward[4]*adjoint[0]
        # = 0*4 + 1*3 + 2*2 + 3*1 + 4*0 = 0 + 3 + 4 + 3 + 0 = 10
        expected_value = 0*4 + 1*3 + 2*2 + 3*1 + 4*0

        assert np.allclose(grad, expected_value), \
            f"Time reversal incorrect: got {grad[0,0]}, expected {expected_value}"

    def test_vjp_gradient_vs_finite_difference(self):
        """VJP gradient should match finite difference approximation."""
        # Create small test case
        np.random.seed(42)
        density = np.random.rand(8, 8)
        mask = np.ones((8, 8), dtype=bool)
        beta = 2.0
        eta = 0.5
        radius = 1

        density_jax = jnp.array(density)
        mask_jax = jnp.array(mask)

        # Random upstream gradient (dL/d(physical_density))
        grad_physical = jnp.array(np.random.randn(8, 8))

        # Compute VJP gradient
        grad_vjp = compute_parameter_gradient_vjp(
            density_jax,
            grad_physical,
            mask_jax,
            beta,
            eta,
            radius,
            filter_type="conic",
        )

        # Compute finite difference gradient for a few random positions
        epsilon = 1e-4
        test_positions = [(2, 3), (5, 5), (1, 6)]

        for i, j in test_positions:
            # Perturb density
            density_plus = density_jax.at[i, j].add(epsilon)
            density_minus = density_jax.at[i, j].add(-epsilon)

            # Forward pass
            physical_plus = transform_density(
                density_plus, mask_jax, beta, eta, radius, filter_type="conic"
            )
            physical_minus = transform_density(
                density_minus, mask_jax, beta, eta, radius, filter_type="conic"
            )

            # Numerical gradient: dL/d(density[i,j]) = sum_k (dL/d(physical[k]) * d(physical[k])/d(density[i,j]))
            d_physical_d_density = (physical_plus - physical_minus) / (2 * epsilon)
            grad_fd_ij = jnp.sum(grad_physical * d_physical_d_density)

            # Compare with VJP gradient
            grad_vjp_ij = grad_vjp[i, j]

            relative_error = abs(grad_vjp_ij - grad_fd_ij) / (abs(grad_fd_ij) + 1e-8)

            assert relative_error < 0.05, \
                f"VJP gradient at ({i},{j}): {grad_vjp_ij:.6f} vs FD: {grad_fd_ij:.6f}, " \
                f"relative error: {relative_error:.6f}"

    def test_vjp_gradient_morphological_filter(self):
        """VJP should work correctly with morphological filter."""
        np.random.seed(42)
        density = np.random.rand(6, 6)
        mask = np.ones((6, 6), dtype=bool)
        beta = 2.0
        eta = 0.5
        radius = 1

        density_jax = jnp.array(density)
        mask_jax = jnp.array(mask)
        grad_physical = jnp.array(np.random.randn(6, 6))

        # Compute VJP gradient with morphological filter
        grad_vjp = compute_parameter_gradient_vjp(
            density_jax,
            grad_physical,
            mask_jax,
            beta,
            eta,
            radius,
            filter_type="morphological",
            morphology_operation="opening",
        )

        # Check gradient is not all zeros (should have non-trivial gradient)
        assert jnp.max(jnp.abs(grad_vjp)) > 1e-6, \
            "Morphological VJP gradient is all zeros"

        # Check gradient is finite
        assert jnp.all(jnp.isfinite(grad_vjp)), \
            "Morphological VJP gradient contains inf/nan"


@pytest.mark.optimization
class TestMaterialPenalty:
    """Test material penalty and intermediate density constraints."""

    def test_projection_reduces_grayscale(self):
        """Projection with high beta should reduce intermediate densities."""
        # Create field with many intermediate values
        density = jnp.array([[0.3, 0.5, 0.7]])
        mask = jnp.ones((1, 3), dtype=bool)

        # Low beta (should preserve intermediate values)
        physical_low = transform_density(
            density, mask, beta=1.0, eta=0.5, radius=0, filter_type="conic"
        )

        # High beta (should push toward binary)
        physical_high = transform_density(
            density, mask, beta=50.0, eta=0.5, radius=0, filter_type="conic"
        )

        # Count "gray" pixels (0.1 < value < 0.9)
        gray_low = jnp.sum((physical_low > 0.1) & (physical_low < 0.9))
        gray_high = jnp.sum((physical_high > 0.1) & (physical_high < 0.9))

        # High beta should have fewer gray pixels
        assert gray_high <= gray_low, \
            f"High beta should reduce grayscale: {gray_high} vs {gray_low}"


@pytest.mark.optimization
@pytest.mark.slow
class TestOptimizationIntegration:
    """Integration test with small optimization problem."""

    def test_simple_optimization_convergence(self):
        """Small optimization should converge to a better objective."""
        # Create simple 10x10 optimization problem
        np.random.seed(42)
        density = np.random.rand(10, 10) * 0.5 + 0.25  # Start near 0.5
        mask = np.ones((10, 10), dtype=bool)

        density_jax = jnp.array(density)
        mask_jax = jnp.array(mask)

        # Define simple objective: maximize sum of physical density
        # (trivial problem: should converge to all 1s)
        def objective(density):
            physical = transform_density(
                density, mask_jax, beta=2.0, eta=0.5, radius=1, filter_type="conic"
            )
            return jnp.sum(physical)

        # Initial objective
        obj_initial = float(objective(density_jax))

        # Compute gradient
        import jax
        grad_fn = jax.grad(objective)
        grad = grad_fn(density_jax)

        # Take gradient step
        learning_rate = 0.1
        density_updated = density_jax + learning_rate * grad

        # Clip to [0, 1]
        density_updated = jnp.clip(density_updated, 0.0, 1.0)

        # New objective
        obj_updated = float(objective(density_updated))

        # Should improve (objective should increase)
        assert obj_updated > obj_initial, \
            f"Optimization did not improve: {obj_updated} vs {obj_initial}"

        # Gradient should not be all zeros
        assert jnp.max(jnp.abs(grad)) > 1e-6, "Gradient is all zeros"


@pytest.mark.optimization
class TestMaskHandling:
    """Test hard boundary masking behavior."""

    def test_hard_mask_zeros_outside(self):
        """Filter should zero values outside mask (hard boundary)."""
        density = jnp.ones((10, 10))

        # Create mask with hole in center
        mask = jnp.ones((10, 10), dtype=bool)
        mask = mask.at[4:6, 4:6].set(False)

        filtered, _ = masked_conic_filter(density, mask, radius=1)

        # Values outside mask should be exactly zero
        assert jnp.all(filtered[4:6, 4:6] == 0.0), \
            "Filter did not zero masked region"

        # Values inside mask should be non-zero
        assert jnp.all(filtered[0:3, 0:3] > 0.0), \
            "Filter zeroed unmasked region"

    def test_fixed_structure_mask_context(self):
        """Fixed structure mask should provide context without forcing values."""
        # Create density field
        density = jnp.ones((10, 10)) * 0.5
        mask = jnp.ones((10, 10), dtype=bool)

        # Fixed structure on left side
        fixed_mask = jnp.zeros((10, 10), dtype=bool)
        fixed_mask = fixed_mask.at[:, 0:3].set(True)

        # Filter with fixed structure context
        filtered_with_context, _ = masked_conic_filter(
            density, mask, radius=2, fixed_structure_mask=fixed_mask
        )

        # Filter without context
        filtered_no_context, _ = masked_conic_filter(
            density, mask, radius=2, fixed_structure_mask=None
        )

        # Results should differ (context affects filtering)
        assert not jnp.allclose(filtered_with_context, filtered_no_context), \
            "Fixed structure mask had no effect on filtering"
