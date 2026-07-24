"""Measured grid convergence of the complete three-dimensional Yee curl."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.lattice import component_coordinates_3d_um, curl_e_to_h_3d


def _coordinate(
    component: str,
    axis: str,
    grid_shape: tuple[int, int, int],
    resolution: float,
) -> np.ndarray:
    coordinates = component_coordinates_3d_um(
        component,
        grid_shape,
        resolution,
    )
    z, y, x = np.meshgrid(
        coordinates["z"],
        coordinates["y"],
        coordinates["x"],
        indexing="ij",
    )
    return {"x": x, "y": y, "z": z}[axis]


def _curl_l2_error(cells_per_axis: int) -> float:
    """Differentiate a smooth vector field and compare on native H supports."""
    grid_shape = (cells_per_axis,) * 3
    resolution = 1.0 / cells_per_axis
    wavenumber = 2.0 * np.pi

    # E=(sin(ky), sin(kz), sin(kx)) exercises all six partial derivatives.
    ex = jnp.asarray(
        np.sin(wavenumber * _coordinate("Ex", "y", grid_shape, resolution)),
        dtype=jnp.float32,
    )
    ey = jnp.asarray(
        np.sin(wavenumber * _coordinate("Ey", "z", grid_shape, resolution)),
        dtype=jnp.float32,
    )
    ez = jnp.asarray(
        np.sin(wavenumber * _coordinate("Ez", "x", grid_shape, resolution)),
        dtype=jnp.float32,
    )
    measured = curl_e_to_h_3d(ex, ey, ez, resolution)

    # curl(E)=(-k cos(kz), -k cos(kx), -k cos(ky)).
    references = tuple(
        -wavenumber
        * np.cos(wavenumber * _coordinate(component, axis, grid_shape, resolution))
        for component, axis in (("Hx", "z"), ("Hy", "x"), ("Hz", "y"))
    )
    mean_square_error = np.mean(
        [
            np.mean((np.asarray(actual) - reference) ** 2)
            for actual, reference in zip(measured, references, strict=True)
        ]
    )
    return float(np.sqrt(mean_square_error))


def test_complete_3d_yee_curl_has_second_order_grid_convergence(
    validation_metrics,
):
    """Richardson rates approach two over three independently sampled grids."""
    grid_sizes = (12, 24, 48)
    errors = tuple(_curl_l2_error(size) for size in grid_sizes)
    observed_orders = tuple(
        float(np.log2(coarse_error / fine_error))
        for coarse_error, fine_error in zip(errors[:-1], errors[1:], strict=True)
    )

    assert errors[0] > errors[1] > errors[2]
    for coarse_size, fine_size, order in zip(
        grid_sizes[:-1],
        grid_sizes[1:],
        observed_orders,
        strict=True,
    ):
        validation_metrics.check(
            f"Yee curl convergence order {coarse_size}->{fine_size}",
            measured=order,
            reference=2.0,
            tolerance="second_order_convergence",
            resolution=f"{coarse_size}^3 -> {fine_size}^3 cells",
            metadata={
                "grid_sizes": list(grid_sizes),
                "l2_errors": list(errors),
                "field": "E=(sin(2πy), sin(2πz), sin(2πx))",
            },
        )
