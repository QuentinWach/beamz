"""Projection methods for topology optimization densities."""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp

VALID_PROJECTION_TYPES = ("heaviside", "ssp")


def validate_projection_options(
    projection_type: str, ssp_smoothing_radius: float = 0.55
) -> None:
    """Validate projection configuration shared by transform and manager APIs."""
    if projection_type not in VALID_PROJECTION_TYPES:
        allowed = "', '".join(VALID_PROJECTION_TYPES)
        raise ValueError(
            f"Unknown projection_type: {projection_type}. Use '{allowed}'."
        )

    if ssp_smoothing_radius <= 0:
        raise ValueError(
            f"ssp_smoothing_radius must be positive, got {ssp_smoothing_radius!r}."
        )


@jax.jit
def smoothed_heaviside(value, beta, eta):
    """Smoothed Heaviside projection using tanh."""
    beta = jnp.asarray(beta)
    beta_is_inf = jnp.isinf(beta)
    finite_beta = jnp.where(beta_is_inf, 1.0, jnp.maximum(beta, 1e-6))

    num = jnp.tanh(finite_beta * eta) + jnp.tanh(finite_beta * (value - eta))
    den = jnp.tanh(finite_beta * eta) + jnp.tanh(finite_beta * (1.0 - eta))
    projected = num / den

    hard_threshold = jnp.where(value > eta, 1.0, 0.0)
    return jnp.where(beta_is_inf, hard_threshold, projected)


def subpixel_smoothed_projection(value, beta, eta, ssp_smoothing_radius: float = 0.55):
    """Apply Hammond SSP1 projection to a smooth 2D density field.

    The input should already be smooth or filtered. The smoothing radius is in
    density-grid cell units.
    """
    value = jnp.asarray(value)
    if value.ndim != 2:
        raise ValueError(
            f"subpixel_smoothed_projection expects a 2D array, got shape {value.shape}."
        )
    validate_projection_options("ssp", ssp_smoothing_radius)
    return _subpixel_smoothed_projection_jit(
        value, beta, eta, float(ssp_smoothing_radius)
    )


@partial(jax.jit, static_argnames=["ssp_smoothing_radius"])
def _subpixel_smoothed_projection_jit(value, beta, eta, ssp_smoothing_radius: float):
    projected = smoothed_heaviside(value, beta, eta)

    grad_y, grad_x = cast(tuple[jax.Array, jax.Array], jnp.gradient(value))
    grad_norm_squared = grad_y**2 + grad_x**2
    nonzero_norm = jnp.abs(grad_norm_squared) > 1e-12

    grad_norm = jnp.sqrt(jnp.where(nonzero_norm, grad_norm_squared, 1.0))
    grad_norm_eff = jnp.where(nonzero_norm, grad_norm, 1.0)

    distance = (eta - value) / grad_norm_eff
    needs_smoothing = nonzero_norm & (jnp.abs(distance) < ssp_smoothing_radius)

    relative_distance = cast(
        jax.Array,
        jnp.where(needs_smoothing, distance / ssp_smoothing_radius, 0.0),
    )
    fill_fraction = jnp.where(
        needs_smoothing,
        (
            0.5
            - 15.0 / 16.0 * relative_distance
            + 5.0 / 8.0 * relative_distance**3
            - 3.0 / 16.0 * relative_distance**5
        ),
        1.0,
    )
    fill_fraction_neg = jnp.where(
        needs_smoothing,
        (
            0.5
            + 15.0 / 16.0 * relative_distance
            - 5.0 / 8.0 * relative_distance**3
            + 3.0 / 16.0 * relative_distance**5
        ),
        1.0,
    )

    lower = value - ssp_smoothing_radius * grad_norm_eff * fill_fraction
    upper = value + ssp_smoothing_radius * grad_norm_eff * fill_fraction_neg

    lower_projected = smoothed_heaviside(lower, beta, eta)
    upper_projected = smoothed_heaviside(upper, beta, eta)

    smoothed = (1.0 - fill_fraction) * lower_projected + (
        fill_fraction * upper_projected
    )
    result = jnp.where(needs_smoothing, smoothed, projected)

    is_binary_input = jnp.all((value == 0.0) | (value == 1.0))
    return jnp.where(is_binary_input, projected, result)


def project_density(
    value,
    beta,
    eta,
    projection_type: str = "heaviside",
    ssp_smoothing_radius: float = 0.55,
):
    """Project a filtered density using the selected projection method."""
    validate_projection_options(projection_type, ssp_smoothing_radius)

    if projection_type == "heaviside":
        return smoothed_heaviside(value, beta, eta)
    return subpixel_smoothed_projection(
        value, beta, eta, ssp_smoothing_radius=ssp_smoothing_radius
    )
