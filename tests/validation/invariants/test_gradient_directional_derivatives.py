from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.optimization.autodiff import (
    masked_conic_filter,
    masked_morphological_filter,
    transform_density,
)
from beamz.optimization.projections import (
    smoothed_heaviside,
    subpixel_smoothed_projection,
)
from beamz.optimization.topology import fold_high_side_yee_padding_to_shape

pytestmark = pytest.mark.optimization


def _primitive(case: str, mask: jax.Array):
    if case == "heaviside":
        return lambda values: smoothed_heaviside(values, beta=3.0, eta=0.47)
    if case == "conic":
        return lambda values: masked_conic_filter(values, mask, radius=2)[0]
    if case.startswith("morphology_"):
        operation = case.removeprefix("morphology_")
        return lambda values: masked_morphological_filter(
            values,
            mask,
            radius=1,
            operation=operation,
            tau=0.08,
        )
    if case == "ssp":
        return lambda values: subpixel_smoothed_projection(
            values,
            beta=3.0,
            eta=0.47,
            ssp_smoothing_radius=0.65,
        )
    if case == "density_pipeline":
        return lambda values: transform_density(
            values,
            mask,
            beta=3.0,
            eta=0.47,
            radius=2,
            filter_type="conic",
            projection_type="heaviside",
        )
    raise ValueError(case)


@pytest.mark.parametrize(
    "case",
    [
        "heaviside",
        "conic",
        "morphology_erosion",
        "morphology_dilation",
        "morphology_opening",
        "morphology_closing",
        "ssp",
        "density_pipeline",
    ],
)
@pytest.mark.parametrize("seed", [17, 29])
def test_primitive_vjp_matches_random_directional_finite_difference(
    case, seed, validation_metrics
):
    rng = np.random.default_rng(seed)
    density = jnp.asarray(rng.uniform(0.15, 0.85, size=(9, 10)), dtype=jnp.float32)
    direction = jnp.asarray(rng.normal(size=density.shape), dtype=jnp.float32)
    direction /= jnp.linalg.norm(direction)
    upstream = jnp.asarray(rng.normal(size=density.shape), dtype=jnp.float32)
    upstream /= jnp.linalg.norm(upstream)
    mask = jnp.asarray(np.ones(density.shape, dtype=bool))
    mask = mask.at[[0, -1], :].set(False)
    mask = mask.at[:, [0, -1]].set(False)
    function = _primitive(case, mask)

    _output, pullback = jax.vjp(function, density)
    vjp_directional = jnp.vdot(direction, pullback(upstream)[0])
    epsilon = jnp.asarray(8e-4, dtype=density.dtype)
    finite_difference = jnp.vdot(
        (
            function(density + epsilon * direction)
            - function(density - epsilon * direction)
        )
        / (2.0 * epsilon),
        upstream,
    )

    validation_metrics.check(
        f"{case} directional derivative",
        measured=float(finite_difference),
        reference=float(vjp_directional),
        tolerance="gradient_float32",
        metadata={"epsilon": float(epsilon), "seed": seed},
    )


def test_high_side_yee_fold_is_adjoint_of_clipped_material_extension():
    rng = np.random.default_rng(808)
    material = rng.normal(size=(3, 4))
    yee_weights = rng.normal(size=(4, 5))
    # The complete Yee support clips its high-side sample onto the last
    # cell-centered material index along each expanded axis.
    extended = material[
        np.minimum(np.arange(4), 2)[:, None], np.minimum(np.arange(5), 3)
    ]

    forward_inner_product = np.vdot(extended, yee_weights)
    folded = fold_high_side_yee_padding_to_shape(yee_weights, material.shape)
    adjoint_inner_product = np.vdot(material, folded)

    np.testing.assert_allclose(
        adjoint_inner_product,
        forward_inner_product,
        rtol=1e-13,
        atol=1e-13,
    )
