"""Topology optimization helpers for BEAMZ.

The public API in this module intentionally focuses on small, easily testable
building blocks that examples can assemble into a full adjoint workflow.  The
functions have been implemented with numpy only dependencies so that they can
be validated in unit tests without running heavy FDTD simulations.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as _np
from numpy.lib.stride_tricks import sliding_window_view as _sliding_window_view

from .core import Optimizer as _Optimizer

DensityArray = _np.ndarray
MaskArray = _np.ndarray


def _validate_shapes(density: DensityArray, gradient: DensityArray, mask: MaskArray) -> None:
    """Raise ``ValueError`` if ``density``, ``gradient`` and ``mask`` disagree."""

    if density.shape != gradient.shape:
        raise ValueError("Gradient and density must share shape.")
    if mask.shape != density.shape:
        raise ValueError("Mask must share the same shape as density.")


def _masked_box_blur(values: DensityArray, mask: MaskArray, radius: int) -> DensityArray:
    """Apply a simple masked box blur with edge padding."""

    radius = int(max(0, round(radius)))
    if radius < 1:
        return values

    padded_values = _np.pad(values, radius, mode="edge")
    padded_mask = _np.pad(mask.astype(float), radius, mode="constant", constant_values=0.0)
    window_shape = (2 * radius + 1, 2 * radius + 1)

    values_view = _sliding_window_view(padded_values, window_shape)
    mask_view = _sliding_window_view(padded_mask, window_shape)

    weighted_sum = (values_view * mask_view).sum(axis=(-2, -1))
    weights = mask_view.sum(axis=(-2, -1))
    weights = _np.where(weights == 0.0, 1.0, weights)
    return weighted_sum / weights


def apply_density_update(
    density: DensityArray,
    gradient: DensityArray,
    mask: MaskArray,
    *,
    learning_rate: float,
    blur_radius: int = 0,
    blur_fn: Optional[Callable[[DensityArray, MaskArray], DensityArray]] = None,
) -> Tuple[DensityArray, DensityArray, DensityArray, float, float]:
    """Update ``density`` using ``gradient`` constrained to ``mask``.

    Parameters
    ----------
    density:
        The design density array to update (values in ``[0, 1]``).  The input is
        not modified in-place.
    gradient:
        Raw gradient array (typically the overlap integral).  Only values inside
        ``mask`` are considered.
    mask:
        Boolean array highlighting the design region.
    learning_rate:
        Scalar step size applied to the smoothed gradient.
    blur_radius:
        Optional radius (in cells) for a masked box blur.  Ignored if ``blur_fn``
        is provided.
    blur_fn:
        Optional callable taking ``(values, mask)`` and returning a filtered
        array.  When provided it overrides the built-in masked box blur.

    Returns
    -------
    tuple
        ``(new_density, applied_gradient, density_delta, change_norm, max_update)``
        where ``density_delta`` contains the actual change applied inside
        ``mask``.
    """

    _validate_shapes(density, gradient, mask)

    masked_gradient = _np.where(mask, gradient, 0.0).astype(float, copy=False)

    if blur_fn is not None:
        smoothed = blur_fn(masked_gradient, mask)
    elif blur_radius > 0:
        smoothed = _masked_box_blur(masked_gradient, mask, blur_radius)
    else:
        smoothed = masked_gradient

    smoothed = _np.where(mask, smoothed, 0.0)

    updated_density = _np.clip(density + learning_rate * smoothed, 0.0, 1.0)
    updated_density = _np.where(mask, updated_density, density)

    density_delta = updated_density - density
    density_delta = _np.where(mask, density_delta, 0.0)

    if _np.any(mask):
        change_norm = float(_np.linalg.norm(density_delta[mask]))
        max_update = float(_np.max(_np.abs(density_delta[mask])))
    else:
        change_norm = 0.0
        max_update = 0.0

    return updated_density, smoothed, density_delta, change_norm, max_update


def initialize_density_from_region(design_region, resolution: float) -> DensityArray:
    """Return a normalized density grid covering ``design_region``."""

    width = getattr(design_region, "width", None)
    height = getattr(design_region, "height", None)
    if width is None or height is None:
        raise ValueError("Design region must expose width and height attributes.")

    nx = max(1, int(round(width / resolution)))
    ny = max(1, int(round(height / resolution)))

    density = _np.full((ny, nx), 0.5, dtype=float)
    material = getattr(design_region, "material", None)
    if material is not None:
        grid = getattr(material, "permittivity_grid", None)
        if grid is not None:
            density = _resample_to_shape(grid, (ny, nx))
    return density


def blur_density(density: DensityArray, radius: float, *, backend: str = "numpy") -> DensityArray:
    """Apply a separable box blur with a configurable radius."""

    radius_cells = max(0, int(round(radius)))
    if radius_cells <= 0:
        return density

    kernel = _np.ones((2 * radius_cells + 1,), dtype=float)
    kernel /= kernel.size

    blurred = _np.apply_along_axis(lambda row: _np.convolve(row, kernel, mode="same"), axis=1, arr=density)
    blurred = _np.apply_along_axis(lambda col: _np.convolve(col, kernel, mode="same"), axis=0, arr=blurred)
    return blurred


def project_density(density: DensityArray, beta: float, eta: float) -> DensityArray:
    """Sigmoid projection toward binary values."""

    if beta <= 0:
        return density

    exp_term = _np.exp(-beta * (density - eta))
    projected = 1.0 / (1.0 + exp_term)
    return projected


def compute_overlap_gradient(forward_fields, adjoint_fields, *, monitor=None) -> DensityArray:
    """Compute overlap gradient using Ez components as a placeholder."""

    forward_ez = _extract_field(forward_fields, "Ez")
    adjoint_ez = _extract_field(adjoint_fields, "Ez")

    if forward_ez is None or adjoint_ez is None:
        raise ValueError("Forward/adjoint data must contain an 'Ez' entry.")

    gradient = _np.real(_np.conj(adjoint_ez) * forward_ez)

    min_shape = tuple(min(f, a) for f, a in zip(forward_ez.shape, adjoint_ez.shape))
    slices = tuple(slice(0, n) for n in min_shape)
    gradient = gradient[slices]
    return gradient


def run_forward(sim, source, monitor, *, live: bool = False):
    """Run a forward simulation with optional live visualization."""

    sim.sources = [source]
    sim.initialize_simulation(save=True, live=live)
    while sim.step():
        pass
    return sim.finalize_simulation()


def run_adjoint(
    density,
    sim,
    source,
    forward_results,
    monitor=None,
    *,
    live: bool = False,
) -> DensityArray:
    """Placeholder adjoint run returning a zero gradient."""

    sim.sources = [source]
    sim.initialize_simulation(save=True, live=live)
    while sim.step():
        pass
    sim.finalize_simulation()
    return _np.zeros_like(density, dtype=float)


def compute_objective(results, monitor=None) -> float:
    """Simple objective: total Ez energy of the final field snapshot."""

    ez = _extract_field(results, "Ez")
    if ez is None:
        return 0.0
    return float(_np.sum(_np.abs(ez) ** 2))


def apply_optimizer_step(
    density: DensityArray,
    gradient: DensityArray,
    optimizer_state: dict | None,
    *,
    method: str = "adam",
    learning_rate: float = 1e-2,
    **kwargs,
) -> tuple[DensityArray, dict | None]:
    """Update density using the generic :class:`Optimizer` helper."""

    if gradient.shape != density.shape:
        raise ValueError("Gradient and density must share shape.")

    if optimizer_state is None:
        optimizer = _Optimizer(method=method, learning_rate=learning_rate, options=kwargs)
    else:
        optimizer = optimizer_state["optimizer"]

    update = optimizer.step(gradient)
    density_updated = density + update
    density_clamped = _np.clip(density_updated, 0.0, 1.0)

    return density_clamped, {"optimizer": optimizer}


def update_design_region_material(design_region, density: DensityArray) -> None:
    """Push the projected density to the region's CustomMaterial grid."""

    material = getattr(design_region, "material", None)
    if material is None or not hasattr(material, "update_grid"):
        return

    permittivity = _np.interp(density, [0.0, 1.0], [_background_eps(), _target_eps()])
    material.update_grid("permittivity", permittivity)


def _normalize_to_unit_interval(values: _np.ndarray) -> _np.ndarray:
    v_min = values.min()
    v_max = values.max()
    if _np.isclose(v_min, v_max):
        return _np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def _resample_to_shape(grid: _np.ndarray, shape: tuple[int, int]) -> _np.ndarray:
    ny, nx = shape
    y_src = _np.linspace(0.0, 1.0, grid.shape[0])
    y_dst = _np.linspace(0.0, 1.0, ny)
    x_src = _np.linspace(0.0, 1.0, grid.shape[1])
    x_dst = _np.linspace(0.0, 1.0, nx)

    temp = _np.zeros((ny, grid.shape[1]), dtype=float)
    for col in range(grid.shape[1]):
        temp[:, col] = _np.interp(y_dst, y_src, grid[:, col])

    resampled = _np.zeros((ny, nx), dtype=float)
    for row in range(ny):
        resampled[row, :] = _np.interp(x_dst, x_src, temp[row, :])

    return _normalize_to_unit_interval(resampled)


def _background_eps() -> float:
    # Placeholder background permittivity (cladding)
    return 1.444 ** 2


def _target_eps() -> float:
    # Placeholder target permittivity (core)
    return 2.25 ** 2


def _extract_field(container, key: str):
    """Helper to pull Ez-like arrays irrespective of container format."""

    if container is None:
        return None

    if isinstance(container, dict):
        value = container.get(key)
        if isinstance(value, list) and value:
            return value[-1]
        return value

    return getattr(container, key, None)


