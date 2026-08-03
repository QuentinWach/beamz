"""Canonical Yee update kernels and static kernel selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.lattice import (
    adjacent_difference as _adjacent_difference,
)
from beamz.lattice import build_h_boundary_views_for_e_3d
from beamz.simulation.model import (
    BoundaryPlan,
    CpmlPackedSlabSpec,
    CpmlTerm,
    RunConfig,
    SimulationState,
    UpdateCoefficients,
)

# Logical shapes describe physics while storage shapes may contain device padding. Every
# phase must preserve this distinction so padded cells never reach results or monitors.


def fit_array_to_shape(arr, target_shape, *, pad_value=0.0):
    """Crop or high-pad a JAX array to one static component shape."""
    arr = jnp.asarray(arr)
    target_shape = tuple(int(value) for value in target_shape)
    if arr.ndim != len(target_shape):
        raise ValueError(
            f"Cannot fit rank-{arr.ndim} array to rank-{len(target_shape)} shape"
        )
    out = arr[
        tuple(
            slice(0, min(int(size), target))
            for size, target in zip(arr.shape, target_shape, strict=True)
        )
    ]
    padding = tuple(
        (0, max(0, target - int(size)))
        for size, target in zip(out.shape, target_shape, strict=True)
    )
    return (
        jnp.pad(out, padding, constant_values=jnp.asarray(pad_value, dtype=arr.dtype))
        if any(high for _, high in padding)
        else out
    )


def cpml_coefficients(sigma, kappa, alpha, dt):
    """Precompute one CPML recurrence as ``a``, ``b``, and inverse-kappa."""
    dtype = jnp.result_type(sigma, kappa, alpha, jnp.float32)
    sigma = jnp.asarray(sigma, dtype=dtype)
    kappa = jnp.maximum(jnp.asarray(kappa, dtype=dtype), 1.0)
    alpha = jnp.asarray(alpha, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    decay = (sigma / kappa + alpha) * (
        jnp.asarray(dt, dtype=dtype) / jnp.asarray(EPS_0, dtype=dtype)
    )
    b = jnp.exp(-decay)
    a = jnp.nan_to_num(
        ((b - one) * sigma) / jnp.maximum((sigma + kappa * alpha) * kappa, 1e-30)
    )
    return a, b, one / kappa


def advance_h_from_coefficients(field, curl, decay, source):
    return decay * field - source * curl


def advance_e_from_coefficients(field, curl, decay, source):
    return decay * field + source * curl


def apply_zero_mask(field, mask):
    """Apply a compiled PEC mask without changing the field representation."""
    return field if mask is None else jnp.where(mask, 0.0, field)


def _scalar_like(value, dtype):
    return jnp.asarray(value, dtype=dtype)


def _scale_by_axis_metric(values, inverse_distance, axis):
    """Multiply one derivative by a scalar or separable 1-D inverse distance."""
    metric = jnp.asarray(inverse_distance, dtype=values.dtype)
    if metric.ndim == 0:
        return values * metric
    shape = [1] * values.ndim
    shape[int(axis)] = metric.shape[0]
    return values * jnp.reshape(metric, shape)


def _metric_difference(values, axis, inverse_distance):
    return _scale_by_axis_metric(jnp.diff(values, axis=axis), inverse_distance, axis)


def _axis_region(ndim, axis, start, stop):
    region = [slice(None)] * ndim
    region[axis] = slice(start, stop)
    return tuple(region)


def _pack_cpml_slab(arr, slab):
    """Gather the active low/high edge regions of one derivative."""
    axis, low, high = int(slab.axis), int(slab.low), int(slab.high)
    if low <= 0 and high <= 0:
        return jnp.zeros(slab.shape, dtype=arr.dtype)
    parts = []
    if low:
        parts.append(arr[_axis_region(arr.ndim, axis, 0, low)])
    logical_stop = (
        int(arr.shape[axis]) if int(slab.logical_stop) < 0 else int(slab.logical_stop)
    )
    if high:
        parts.append(
            arr[_axis_region(arr.ndim, axis, logical_stop - high, logical_stop)]
        )
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=axis)


def _unpack_cpml_slab(base, packed, slab):
    """Rebuild a derivative with its packed low/high CPML corrections."""
    axis, low, high = int(slab.axis), int(slab.low), int(slab.high)
    if not low and not high:
        return base
    logical_stop = (
        int(base.shape[axis]) if int(slab.logical_stop) < 0 else int(slab.logical_stop)
    )
    parts = []
    if low:
        parts.append(packed[_axis_region(packed.ndim, axis, 0, low)])
    middle_stop = logical_stop - high
    if middle_stop > low:
        parts.append(base[_axis_region(base.ndim, axis, low, middle_stop)])
    if high:
        parts.append(
            packed[_axis_region(packed.ndim, axis, low, low + high)]
        )
    if logical_stop < int(base.shape[axis]):
        parts.append(
            base[_axis_region(base.ndim, axis, logical_stop, int(base.shape[axis]))]
        )
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=axis)


def _active_slab_counts(a, inv_kappa, axis):
    active = (np.abs(np.asarray(a)) > 1e-30) | (
        np.abs(np.asarray(inv_kappa) - 1.0) > 1e-7
    )
    transverse = tuple(index for index in range(active.ndim) if index != axis)
    active = np.any(active, axis=transverse) if transverse else active
    low = next((i for i, value in enumerate(active) if not value), active.size)
    tail = active[low:][::-1]
    high = next((i for i, value in enumerate(tail) if not value), tail.size)
    return int(low), int(high)


def compile_cpml_term(*, component, axis, sign, sigma, kappa, alpha, dt, full_shape):
    """Precompute and pack one dimension-independent CPML recurrence."""
    a, b, inv_kappa = cpml_coefficients(sigma, kappa, alpha, dt)
    low, high = _active_slab_counts(a, inv_kappa, int(axis))
    shape = list(full_shape)
    shape[int(axis)] = low + high
    slab = CpmlPackedSlabSpec(
        int(axis), low, high, tuple(shape), int(full_shape[int(axis)])
    )
    return CpmlTerm(
        str(component),
        int(axis),
        float(sign),
        _pack_cpml_slab(a, slab),
        _pack_cpml_slab(b, slab),
        _pack_cpml_slab(inv_kappa, slab),
        slab,
    )


def correct_cpml_term(derivative, psi, term):
    """Apply one packed recurrence and return the signed corrected derivative."""
    if not term.slab.low and not term.slab.high:
        return term.sign * derivative, psi
    derivative_slab = _pack_cpml_slab(derivative.astype(psi.dtype), term.slab)
    psi = term.b.astype(psi.dtype) * psi + term.a.astype(psi.dtype) * derivative_slab
    corrected = derivative_slab * term.inv_kappa.astype(psi.dtype) + psi
    return term.sign * _unpack_cpml_slab(derivative, corrected, term.slab), psi


def fused_update_h_lossy_3d_material(
    ex, ey, ez, hx, hy, hz, sigma_x, sigma_y, sigma_z, dt, resolution
):
    """Advance all magnetic components directly from collocated loss grids."""
    one = jnp.asarray(1.0, dtype=hx.dtype)
    dt_over_mu = jnp.asarray(dt, dtype=hx.dtype) / jnp.asarray(MU_0, dtype=hx.dtype)
    inv_res = one / jnp.asarray(resolution, dtype=hx.dtype)
    out = []
    differences = (
        (ez[:, 1:, :] - ez[:, :-1, :], ey[1:, :, :] - ey[:-1, :, :]),
        (ex[1:, :, :] - ex[:-1, :, :], ez[:, :, 1:] - ez[:, :, :-1]),
        (ey[:, :, 1:] - ey[:, :, :-1], ex[:, 1:, :] - ex[:, :-1, :]),
    )
    for field, sigma, terms in zip(
        (hx, hy, hz), (sigma_x, sigma_y, sigma_z), differences, strict=True
    ):
        alpha = sigma * (0.5 * dt_over_mu)
        denom = one + alpha
        curl = (
            fit_array_to_shape(terms[0], field.shape)
            - fit_array_to_shape(terms[1], field.shape)
        ) * inv_res
        out.append((one - alpha) / denom * field - (dt_over_mu / denom) * curl)
    return tuple(out)


def fused_update_h_lossy_3d_material_metric(
    ex, ey, ez, hx, hy, hz, sigma_x, sigma_y, sigma_z, dt, metrics
):
    """Advance 3D H fields using statically planned separable grid metrics."""
    one = jnp.asarray(1.0, dtype=hx.dtype)
    dt_over_mu = jnp.asarray(dt, dtype=hx.dtype) / jnp.asarray(MU_0, dtype=hx.dtype)
    differences = (
        (
            _metric_difference(ez, 1, metrics.e_to_h_y),
            _metric_difference(ey, 0, metrics.e_to_h_z),
        ),
        (
            _metric_difference(ex, 0, metrics.e_to_h_z),
            _metric_difference(ez, 2, metrics.e_to_h_x),
        ),
        (
            _metric_difference(ey, 2, metrics.e_to_h_x),
            _metric_difference(ex, 1, metrics.e_to_h_y),
        ),
    )
    out = []
    for field, sigma, terms in zip(
        (hx, hy, hz), (sigma_x, sigma_y, sigma_z), differences, strict=True
    ):
        alpha = sigma * (0.5 * dt_over_mu)
        denom = one + alpha
        curl = fit_array_to_shape(terms[0], field.shape) - fit_array_to_shape(
            terms[1], field.shape
        )
        out.append((one - alpha) / denom * field - (dt_over_mu / denom) * curl)
    return tuple(out)


def fused_update_e_lossy_3d_material(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    conductivity_x,
    inv_permittivity_x,
    conductivity_y,
    inv_permittivity_y,
    conductivity_z,
    inv_permittivity_z,
    dt,
    resolution,
    *,
    boundary_views,
):
    """Advance all electric components directly from collocated material grids."""
    del hx, hy, hz
    one = jnp.asarray(1.0, dtype=ex.dtype)
    dt_over_eps = jnp.asarray(dt, dtype=ex.dtype) / jnp.asarray(EPS_0, dtype=ex.dtype)
    derivative_pairs = (
        (("hz_y", 1), ("hy_z", 0)),
        (("hx_z", 0), ("hz_x", 2)),
        (("hy_x", 2), ("hx_y", 1)),
    )
    out = []
    for field, conductivity, inv_permittivity, pair in zip(
        (ex, ey, ez),
        (conductivity_x, conductivity_y, conductivity_z),
        (inv_permittivity_x, inv_permittivity_y, inv_permittivity_z),
        derivative_pairs,
        strict=True,
    ):
        beta = conductivity * (0.5 * dt_over_eps) * inv_permittivity
        denom = one + beta
        curl = fit_array_to_shape(
            _adjacent_difference(boundary_views[pair[0][0]], pair[0][1], resolution),
            field.shape,
        ) - fit_array_to_shape(
            _adjacent_difference(boundary_views[pair[1][0]], pair[1][1], resolution),
            field.shape,
        )
        out.append(
            (one - beta) / denom * field
            + (dt_over_eps * inv_permittivity / denom) * curl
        )
    return tuple(out)


def fused_update_e_lossy_3d_material_metric(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    conductivity_x,
    inv_permittivity_x,
    conductivity_y,
    inv_permittivity_y,
    conductivity_z,
    inv_permittivity_z,
    dt,
    metrics,
    *,
    boundary_views,
):
    """Advance 3D E fields using staggered center-to-edge distances."""
    del hx, hy, hz
    one = jnp.asarray(1.0, dtype=ex.dtype)
    dt_over_eps = jnp.asarray(dt, dtype=ex.dtype) / jnp.asarray(EPS_0, dtype=ex.dtype)
    derivative_pairs = (
        (("hz_y", 1, metrics.h_to_e_y), ("hy_z", 0, metrics.h_to_e_z)),
        (("hx_z", 0, metrics.h_to_e_z), ("hz_x", 2, metrics.h_to_e_x)),
        (("hy_x", 2, metrics.h_to_e_x), ("hx_y", 1, metrics.h_to_e_y)),
    )
    out = []
    for field, conductivity, inv_permittivity, pair in zip(
        (ex, ey, ez),
        (conductivity_x, conductivity_y, conductivity_z),
        (inv_permittivity_x, inv_permittivity_y, inv_permittivity_z),
        derivative_pairs,
        strict=True,
    ):
        beta = conductivity * (0.5 * dt_over_eps) * inv_permittivity
        denom = one + beta
        curl = fit_array_to_shape(
            _metric_difference(boundary_views[pair[0][0]], pair[0][1], pair[0][2]),
            field.shape,
        ) - fit_array_to_shape(
            _metric_difference(boundary_views[pair[1][0]], pair[1][1], pair[1][2]),
            field.shape,
        )
        out.append(
            (one - beta) / denom * field
            + (dt_over_eps * inv_permittivity / denom) * curl
        )
    return tuple(out)


def cpml_update_h_from_e_3d(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    resolution,
    *,
    terms,
    psi_terms,
    dt,
    magnetic_conductivities,
):
    """Advance 3D H fields and their six packed CPML memories."""
    derivatives = (
        (ez, 1, hx.shape),
        (ey, 0, hx.shape),
        (ex, 0, hy.shape),
        (ez, 2, hy.shape),
        (ey, 2, hz.shape),
        (ex, 1, hz.shape),
    )
    corrected, next_psi = zip(
        *(
            correct_cpml_term(
                fit_array_to_shape(jnp.diff(field, axis=axis) / resolution, shape),
                psi,
                term,
            )
            for (field, axis, shape), psi, term in zip(
                derivatives, psi_terms, terms, strict=True
            )
        ),
        strict=True,
    )
    curls = tuple(corrected[index] + corrected[index + 1] for index in (0, 2, 4))
    one = jnp.asarray(1.0, dtype=hx.dtype)
    dt_over_mu = jnp.asarray(dt, dtype=hx.dtype) / jnp.asarray(MU_0, dtype=hx.dtype)
    updated = []
    for field, curl, sigma in zip(
        (hx, hy, hz), curls, magnetic_conductivities, strict=True
    ):
        alpha = sigma * (0.5 * dt_over_mu)
        updated.append(((one - alpha) * field - dt_over_mu * curl) / (one + alpha))
    return *updated, tuple(next_psi)


def cpml_update_h_from_e_3d_metric(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    metrics,
    *,
    terms,
    psi_terms,
    dt,
    magnetic_conductivities,
):
    """Advance 3D H/CPML using physical cell widths for every derivative."""
    derivatives = (
        (ez, 1, metrics.e_to_h_y, hx.shape),
        (ey, 0, metrics.e_to_h_z, hx.shape),
        (ex, 0, metrics.e_to_h_z, hy.shape),
        (ez, 2, metrics.e_to_h_x, hy.shape),
        (ey, 2, metrics.e_to_h_x, hz.shape),
        (ex, 1, metrics.e_to_h_y, hz.shape),
    )
    corrected, next_psi = zip(
        *(
            correct_cpml_term(
                fit_array_to_shape(_metric_difference(field, axis, metric), shape),
                psi,
                term,
            )
            for (field, axis, metric, shape), psi, term in zip(
                derivatives, psi_terms, terms, strict=True
            )
        ),
        strict=True,
    )
    curls = tuple(corrected[index] + corrected[index + 1] for index in (0, 2, 4))
    one = jnp.asarray(1.0, dtype=hx.dtype)
    dt_over_mu = jnp.asarray(dt, dtype=hx.dtype) / jnp.asarray(MU_0, dtype=hx.dtype)
    updated = []
    for field, curl, sigma in zip(
        (hx, hy, hz), curls, magnetic_conductivities, strict=True
    ):
        alpha = sigma * (0.5 * dt_over_mu)
        updated.append(((one - alpha) * field - dt_over_mu * curl) / (one + alpha))
    return *updated, tuple(next_psi)


def cpml_update_e_from_h_3d(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    resolution,
    *,
    terms,
    psi_terms,
    metallic_edges,
    dt,
    conductivities,
    inverse_permittivities,
    logical_shapes=None,
):
    """Advance 3D E fields and their six packed CPML memories."""
    views = build_h_boundary_views_for_e_3d(
        hx, hy, hz, metallic_edges, logical_shapes=logical_shapes
    )
    derivatives = (
        ("hz_y", 1, ex.shape),
        ("hy_z", 0, ex.shape),
        ("hx_z", 0, ey.shape),
        ("hz_x", 2, ey.shape),
        ("hy_x", 2, ez.shape),
        ("hx_y", 1, ez.shape),
    )
    corrected, next_psi = zip(
        *(
            correct_cpml_term(
                fit_array_to_shape(
                    _adjacent_difference(views[name], axis, resolution), shape
                ),
                psi,
                term,
            )
            for (name, axis, shape), psi, term in zip(
                derivatives, psi_terms, terms, strict=True
            )
        ),
        strict=True,
    )
    curls = tuple(corrected[index] + corrected[index + 1] for index in (0, 2, 4))
    one = jnp.asarray(1.0, dtype=ex.dtype)
    dt_over_eps = jnp.asarray(dt, dtype=ex.dtype) / jnp.asarray(EPS_0, dtype=ex.dtype)
    updated = []
    for field, curl, conductivity, inv_permittivity in zip(
        (ex, ey, ez),
        curls,
        conductivities,
        inverse_permittivities,
        strict=True,
    ):
        beta = conductivity * (0.5 * dt_over_eps) * inv_permittivity
        updated.append(
            ((one - beta) * field + dt_over_eps * inv_permittivity * curl)
            / (one + beta)
        )
    return *updated, tuple(next_psi)


def cpml_update_e_from_h_3d_metric(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    metrics,
    *,
    terms,
    psi_terms,
    metallic_edges,
    dt,
    conductivities,
    inverse_permittivities,
    logical_shapes=None,
):
    """Advance 3D E/CPML using physical center-to-center distances."""
    views = build_h_boundary_views_for_e_3d(
        hx, hy, hz, metallic_edges, logical_shapes=logical_shapes
    )
    derivatives = (
        ("hz_y", 1, metrics.h_to_e_y, ex.shape),
        ("hy_z", 0, metrics.h_to_e_z, ex.shape),
        ("hx_z", 0, metrics.h_to_e_z, ey.shape),
        ("hz_x", 2, metrics.h_to_e_x, ey.shape),
        ("hy_x", 2, metrics.h_to_e_x, ez.shape),
        ("hx_y", 1, metrics.h_to_e_y, ez.shape),
    )
    corrected, next_psi = zip(
        *(
            correct_cpml_term(
                fit_array_to_shape(
                    _metric_difference(views[name], axis, metric), shape
                ),
                psi,
                term,
            )
            for (name, axis, metric, shape), psi, term in zip(
                derivatives, psi_terms, terms, strict=True
            )
        ),
        strict=True,
    )
    curls = tuple(corrected[index] + corrected[index + 1] for index in (0, 2, 4))
    one = jnp.asarray(1.0, dtype=ex.dtype)
    dt_over_eps = jnp.asarray(dt, dtype=ex.dtype) / jnp.asarray(EPS_0, dtype=ex.dtype)
    updated = []
    for field, curl, conductivity, inv_permittivity in zip(
        (ex, ey, ez),
        curls,
        conductivities,
        inverse_permittivities,
        strict=True,
    ):
        beta = conductivity * (0.5 * dt_over_eps) * inv_permittivity
        updated.append(
            ((one - beta) * field + dt_over_eps * inv_permittivity * curl)
            / (one + beta)
        )
    return *updated, tuple(next_psi)


def precompute_h_update_coefficients(sigma_m, dt):
    denom = 1.0 + sigma_m * dt / (2.0 * MU_0)
    decay = (1.0 - sigma_m * dt / (2.0 * MU_0)) / denom
    source = (dt / MU_0) / denom
    return decay.astype(jnp.float32), source.astype(jnp.float32)


def precompute_e_update_coefficients(shape, conductivity, permittivity, dt, region):
    denom = 1.0 + conductivity * dt / (2.0 * EPS_0 * permittivity)
    decay = (
        jnp.ones(shape, dtype=jnp.float32)
        .at[region]
        .set(
            ((1.0 - conductivity * dt / (2.0 * EPS_0 * permittivity)) / denom).astype(
                jnp.float32
            )
        )
    )
    source_value = (dt / (EPS_0 * permittivity)) / denom
    source = (
        jnp.zeros(shape, dtype=jnp.float32)
        .at[region]
        .set(source_value.astype(jnp.float32))
    )
    return decay, source


def tm_xy_curl_e_to_h_2d(ez, resolution, hx_shape, hy_shape, metallic_edges):
    """Differentiate Ez onto the native Hx and Hy supports."""
    del hx_shape, hy_shape, metallic_edges
    resolution = _scalar_like(resolution, ez.dtype)
    return (
        (ez[1:, :] - ez[:-1, :]) / resolution,
        -(ez[:, 1:] - ez[:, :-1]) / resolution,
    )


def _tm_xy_h_derivatives(hx, hy, resolution, metallic_edges):
    """Return padded H derivatives on the complete Ez node lattice."""
    resolution = _scalar_like(resolution, hy.dtype)
    left, right = hy[:, :1], hy[:, -1:]
    bottom, top = hx[:1, :], hx[-1:, :]
    if "left" in metallic_edges:
        left = jnp.zeros_like(left)
    if "right" in metallic_edges:
        right = jnp.zeros_like(right)
    if "bottom" in metallic_edges:
        bottom = jnp.zeros_like(bottom)
    if "top" in metallic_edges:
        top = jnp.zeros_like(top)
    padded_hy = jnp.concatenate((left, hy, right), axis=1)
    padded_hx = jnp.concatenate((bottom, hx, top), axis=0)
    d_hy_dx = (padded_hy[:, 1:] - padded_hy[:, :-1]) / resolution
    d_hx_dy = (padded_hx[1:, :] - padded_hx[:-1, :]) / resolution
    return d_hy_dx, d_hx_dy


def tm_xy_curl_h_to_e_2d(hx, hy, resolution, ez_shape, metallic_edges=frozenset()):
    """Differentiate Hx and Hy onto the native Ez support."""
    d_hy_dx, d_hx_dy = _tm_xy_h_derivatives(hx, hy, resolution, metallic_edges)
    curl = d_hy_dx - d_hx_dy
    if curl.shape != ez_shape:
        raise ValueError(f"curl(H) shape {curl.shape} does not match Ez {ez_shape}")
    return curl


def tm_xy_cpml_curl_e_to_h_2d(
    ez,
    resolution,
    *,
    terms,
    psi_h_terms,
):
    """Correct the two Ez derivatives with canonical 2D CPML memory."""
    resolution = _scalar_like(resolution, ez.dtype)
    derivatives = (
        (ez[1:] - ez[:-1]) / resolution,
        (ez[:, 1:] - ez[:, :-1]) / resolution,
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_h_terms, terms, strict=True)
    )
    return corrected[0][0], corrected[1][0], tuple(item[1] for item in corrected)


def tm_xy_cpml_curl_h_to_e_2d(
    hx,
    hy,
    resolution,
    ez_shape,
    metallic_edges,
    *,
    terms,
    psi_e_terms,
):
    """Correct the two H derivatives with canonical 2D CPML memory."""
    derivatives = _tm_xy_h_derivatives(hx, hy, resolution, metallic_edges)
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_e_terms, terms, strict=True)
    )
    curl = corrected[0][0] + corrected[1][0]
    if curl.shape != ez_shape:
        raise ValueError(
            f"CPML curl(H) shape {curl.shape} does not match Ez {ez_shape}"
        )
    return curl.astype(hx.dtype), tuple(item[1] for item in corrected)


def te_xy_curl_e_to_h_2d(ex, ey, resolution, hz_shape):
    """Differentiate in-plane E onto the canonical Hz support."""
    resolution = _scalar_like(resolution, ex.dtype)
    curl = (ey[:, 1:] - ey[:, :-1]) / resolution - (ex[1:, :] - ex[:-1, :]) / resolution
    if curl.shape != hz_shape:
        raise ValueError(f"curl(E) shape {curl.shape} does not match Hz {hz_shape}")
    return curl


def _te_xy_h_derivatives(hz, resolution, metallic_edges):
    """Return padded Hz derivatives on the complete Ex and Ey supports."""
    resolution = _scalar_like(resolution, hz.dtype)
    bottom, top = hz[:1, :], hz[-1:, :]
    left, right = hz[:, :1], hz[:, -1:]
    if "bottom" in metallic_edges:
        bottom = jnp.zeros_like(bottom)
    if "top" in metallic_edges:
        top = jnp.zeros_like(top)
    if "left" in metallic_edges:
        left = jnp.zeros_like(left)
    if "right" in metallic_edges:
        right = jnp.zeros_like(right)
    padded_y = jnp.concatenate((bottom, hz, top), axis=0)
    padded_x = jnp.concatenate((left, hz, right), axis=1)
    return (
        (padded_y[1:, :] - padded_y[:-1, :]) / resolution,
        (padded_x[:, 1:] - padded_x[:, :-1]) / resolution,
    )


def te_xy_curl_h_to_e_2d(hz, resolution, ex_shape, ey_shape, metallic_edges):
    """Differentiate Hz onto the canonical Ex and Ey supports."""
    d_hz_dy, d_hz_dx = _te_xy_h_derivatives(hz, resolution, metallic_edges)
    curls = d_hz_dy, -d_hz_dx
    if curls[0].shape != ex_shape or curls[1].shape != ey_shape:
        raise ValueError(
            f"curl(H) shapes {(curls[0].shape, curls[1].shape)} do not match "
            f"Ex/Ey {(ex_shape, ey_shape)}"
        )
    return curls


def te_xy_cpml_curl_e_to_h_2d(ex, ey, resolution, *, terms, psi_h_terms):
    """Correct the two in-plane E derivatives used by the Hz update."""
    resolution = _scalar_like(resolution, ex.dtype)
    derivatives = (
        (ey[:, 1:] - ey[:, :-1]) / resolution,
        (ex[1:, :] - ex[:-1, :]) / resolution,
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_h_terms, terms, strict=True)
    )
    return corrected[0][0] + corrected[1][0], tuple(item[1] for item in corrected)


def te_xy_cpml_curl_h_to_e_2d(hz, resolution, metallic_edges, *, terms, psi_e_terms):
    """Correct the Hz derivatives used by the Ex and Ey updates."""
    derivatives = _te_xy_h_derivatives(hz, resolution, metallic_edges)
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_e_terms, terms, strict=True)
    )
    return (corrected[0][0], corrected[1][0]), tuple(item[1] for item in corrected)


def tm_xy_curl_e_to_h_2d_metric(ez, metrics):
    """Differentiate Ez onto H using physical x/y cell widths."""
    return (
        _metric_difference(ez, 0, metrics.e_to_h_y),
        -_metric_difference(ez, 1, metrics.e_to_h_x),
    )


def _tm_xy_h_derivatives_metric(hx, hy, metrics, metallic_edges):
    left, right = hy[:, :1], hy[:, -1:]
    bottom, top = hx[:1, :], hx[-1:, :]
    if "left" in metallic_edges:
        left = jnp.zeros_like(left)
    if "right" in metallic_edges:
        right = jnp.zeros_like(right)
    if "bottom" in metallic_edges:
        bottom = jnp.zeros_like(bottom)
    if "top" in metallic_edges:
        top = jnp.zeros_like(top)
    padded_hy = jnp.concatenate((left, hy, right), axis=1)
    padded_hx = jnp.concatenate((bottom, hx, top), axis=0)
    return (
        _metric_difference(padded_hy, 1, metrics.h_to_e_x),
        _metric_difference(padded_hx, 0, metrics.h_to_e_y),
    )


def tm_xy_curl_h_to_e_2d_metric(hx, hy, metrics, ez_shape, metallic_edges):
    """Differentiate H onto Ez using physical center-to-center distances."""
    d_hy_dx, d_hx_dy = _tm_xy_h_derivatives_metric(hx, hy, metrics, metallic_edges)
    curl = d_hy_dx - d_hx_dy
    if curl.shape != ez_shape:
        raise ValueError(f"curl(H) shape {curl.shape} does not match Ez {ez_shape}")
    return curl


def tm_xy_cpml_curl_e_to_h_2d_metric(ez, metrics, *, terms, psi_h_terms):
    derivatives = (
        _metric_difference(ez, 0, metrics.e_to_h_y),
        _metric_difference(ez, 1, metrics.e_to_h_x),
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_h_terms, terms, strict=True)
    )
    return corrected[0][0], corrected[1][0], tuple(item[1] for item in corrected)


def tm_xy_cpml_curl_h_to_e_2d_metric(
    hx,
    hy,
    metrics,
    ez_shape,
    metallic_edges,
    *,
    terms,
    psi_e_terms,
):
    derivatives = _tm_xy_h_derivatives_metric(hx, hy, metrics, metallic_edges)
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_e_terms, terms, strict=True)
    )
    curl = corrected[0][0] + corrected[1][0]
    if curl.shape != ez_shape:
        raise ValueError(
            f"CPML curl(H) shape {curl.shape} does not match Ez {ez_shape}"
        )
    return curl.astype(hx.dtype), tuple(item[1] for item in corrected)


def te_xy_curl_e_to_h_2d_metric(ex, ey, metrics, hz_shape):
    curl = _metric_difference(ey, 1, metrics.e_to_h_x) - _metric_difference(
        ex, 0, metrics.e_to_h_y
    )
    if curl.shape != hz_shape:
        raise ValueError(f"curl(E) shape {curl.shape} does not match Hz {hz_shape}")
    return curl


def _te_xy_h_derivatives_metric(hz, metrics, metallic_edges):
    bottom, top = hz[:1, :], hz[-1:, :]
    left, right = hz[:, :1], hz[:, -1:]
    if "bottom" in metallic_edges:
        bottom = jnp.zeros_like(bottom)
    if "top" in metallic_edges:
        top = jnp.zeros_like(top)
    if "left" in metallic_edges:
        left = jnp.zeros_like(left)
    if "right" in metallic_edges:
        right = jnp.zeros_like(right)
    padded_y = jnp.concatenate((bottom, hz, top), axis=0)
    padded_x = jnp.concatenate((left, hz, right), axis=1)
    return (
        _metric_difference(padded_y, 0, metrics.h_to_e_y),
        _metric_difference(padded_x, 1, metrics.h_to_e_x),
    )


def te_xy_curl_h_to_e_2d_metric(hz, metrics, ex_shape, ey_shape, metallic_edges):
    d_hz_dy, d_hz_dx = _te_xy_h_derivatives_metric(hz, metrics, metallic_edges)
    curls = d_hz_dy, -d_hz_dx
    if curls[0].shape != ex_shape or curls[1].shape != ey_shape:
        raise ValueError(
            f"curl(H) shapes {(curls[0].shape, curls[1].shape)} do not match "
            f"Ex/Ey {(ex_shape, ey_shape)}"
        )
    return curls


def te_xy_cpml_curl_e_to_h_2d_metric(ex, ey, metrics, *, terms, psi_h_terms):
    derivatives = (
        _metric_difference(ey, 1, metrics.e_to_h_x),
        _metric_difference(ex, 0, metrics.e_to_h_y),
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_h_terms, terms, strict=True)
    )
    return corrected[0][0] + corrected[1][0], tuple(item[1] for item in corrected)


def te_xy_cpml_curl_h_to_e_2d_metric(
    hz, metrics, metallic_edges, *, terms, psi_e_terms
):
    derivatives = _te_xy_h_derivatives_metric(hz, metrics, metallic_edges)
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_e_terms, terms, strict=True)
    )
    return (corrected[0][0], corrected[1][0]), tuple(item[1] for item in corrected)


def as_array(value) -> jnp.ndarray:
    # Make dtype conversion explicit to prevent promotion from changing JAX
    # signatures.
    return cast(jnp.ndarray, value)


def astype_like(value, ref: jnp.ndarray) -> jnp.ndarray:
    # Make dtype conversion explicit to prevent promotion from changing JAX
    # signatures.
    return as_array(value).astype(ref.dtype)


def apply_post_source_boundaries(
    values: tuple[jnp.ndarray, ...],
    metallic_masks: tuple[jnp.ndarray, ...],
) -> tuple[jnp.ndarray, ...]:
    # Apply the precomputed boundary policy so kernels never reinterpret configuration
    # objects.
    return tuple(
        jnp.asarray(apply_zero_mask(value, mask))
        for value, mask in zip(values, metallic_masks, strict=True)
    )


# Each function is one static combination of dimension and boundary physics. Some
# repetition is deliberate: branch-free graphs are smaller and update order stays visible.


def _astype_terms(values, references):
    return tuple(
        value.astype(reference.dtype)
        for value, reference in zip(values, references, strict=True)
    )


def _replace_h(eng, hx, hy, hz, *, cpml_h=None):
    # Place the complete pytree consistently so one state never spans incompatible
    # shardings.
    updates: dict[str, Any] = {
        "hx": astype_like(hx, eng.hx),
        "hy": astype_like(hy, eng.hy),
        "hz": astype_like(hz, eng.hz),
    }
    if cpml_h is not None:
        updates["cpml_psi_h_terms"] = _astype_terms(cpml_h, eng.cpml_psi_h_terms)
    return eng._replace(**updates)


def _replace_e(eng, ex, ey, ez, *, cpml_e=None):
    # Place the complete pytree consistently so one state never spans incompatible
    # shardings.
    updates: dict[str, Any] = {
        "ex": astype_like(ex, eng.ex),
        "ey": astype_like(ey, eng.ey),
        "ez": astype_like(ez, eng.ez),
    }
    if cpml_e is not None:
        updates["cpml_psi_e_terms"] = _astype_terms(cpml_e, eng.cpml_psi_e_terms)
    return eng._replace(**updates)


def update_h_3d_cpml(eng, ctx, coeffs):
    # 1. Advance H and its packed-slab CPML memories together using the coefficients and
    # slab geometry fixed during planning.
    cpml = ctx.boundary.cpml
    hx, hy, hz, psi_h = cpml_update_h_from_e_3d(
        eng.ex,
        eng.ey,
        eng.ez,
        eng.hx,
        eng.hy,
        eng.hz,
        ctx.resolution,
        terms=cpml.h_terms,
        psi_terms=eng.cpml_psi_h_terms,
        dt=ctx.dt_scalar,
        magnetic_conductivities=(
            coeffs.h_sigma_m_x,
            coeffs.h_sigma_m_y,
            coeffs.h_sigma_m_z,
        ),
    )
    # 2. Replace only H and its 3D memory, preserving all other fields and auxiliary state
    # in the immutable engine carry.
    return _replace_h(
        eng,
        hx,
        hy,
        hz,
        cpml_h=psi_h,
    )


def update_e_3d_cpml(eng, ctx, coeffs):
    # 1. Derive inverse permittivity at execution time so the compiled plan stores only
    # the authoritative material grids.
    cpml = ctx.boundary.cpml
    inv_x = jnp.reciprocal(coeffs.e_permittivity_x)
    inv_y = jnp.reciprocal(coeffs.e_permittivity_y)
    inv_z = jnp.reciprocal(coeffs.e_permittivity_z)
    # 2. Advance E while updating psi only inside the packed boundary slabs selected by
    # the planner.
    ex, ey, ez, psi_e = cpml_update_e_from_h_3d(
        eng.hx,
        eng.hy,
        eng.hz,
        eng.ex,
        eng.ey,
        eng.ez,
        ctx.resolution,
        terms=cpml.e_terms,
        psi_terms=eng.cpml_psi_e_terms,
        metallic_edges=cpml.metallic_edges,
        dt=ctx.dt_scalar,
        conductivities=(
            coeffs.e_conductivity_x,
            coeffs.e_conductivity_y,
            coeffs.e_conductivity_z,
        ),
        inverse_permittivities=(inv_x, inv_y, inv_z),
        logical_shapes=ctx.boundary.logical_component_shapes,
    )
    # 3. Replace E and packed 3D memory atomically, leaving unrelated carry fields intact.
    return _replace_e(
        eng,
        ex,
        ey,
        ez,
        cpml_e=psi_e,
    )


def update_h_3d_yee(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    hx, hy, hz = fused_update_h_lossy_3d_material(
        eng.ex,
        eng.ey,
        eng.ez,
        eng.hx,
        eng.hy,
        eng.hz,
        coeffs.h_sigma_m_x,
        coeffs.h_sigma_m_y,
        coeffs.h_sigma_m_z,
        ctx.dt_scalar,
        ctx.resolution,
    )
    return _replace_h(eng, hx, hy, hz)


def update_e_3d_yee(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    boundary_views = build_h_boundary_views_for_e_3d(
        eng.hx,
        eng.hy,
        eng.hz,
        ctx.boundary.cpml.metallic_edges,
        logical_shapes=ctx.boundary.logical_component_shapes,
    )
    ex, ey, ez = fused_update_e_lossy_3d_material(
        eng.hx,
        eng.hy,
        eng.hz,
        eng.ex,
        eng.ey,
        eng.ez,
        coeffs.e_conductivity_x,
        jnp.reciprocal(coeffs.e_permittivity_x),
        coeffs.e_conductivity_y,
        jnp.reciprocal(coeffs.e_permittivity_y),
        coeffs.e_conductivity_z,
        jnp.reciprocal(coeffs.e_permittivity_z),
        ctx.dt_scalar,
        ctx.resolution,
        boundary_views=boundary_views,
    )
    return _replace_e(eng, ex, ey, ez)


def _update_h_tm_from_curls(eng, ctx, coeffs, curl_tm_hx, curl_tm_hy, psi_h=None):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    hx = advance_h_from_coefficients(
        eng.hx, curl_tm_hx, coeffs.h_decay_x, coeffs.h_source_x
    )
    hy = advance_h_from_coefficients(
        eng.hy, curl_tm_hy, coeffs.h_decay_y, coeffs.h_source_y
    )
    return _replace_h(eng, hx, hy, eng.hz, cpml_h=psi_h)


def update_h_2d_tm_xy(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d(
        eng.ez,
        ctx.resolution,
        eng.hx.shape,
        eng.hy.shape,
        ctx.boundary.metallic_edges_2d,
    )
    return _update_h_tm_from_curls(eng, ctx, coeffs, curl_hx, curl_hy)


def update_h_2d_tm_xy_cpml(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    cpml = ctx.boundary.cpml
    curl_hx, curl_hy, psi_h = tm_xy_cpml_curl_e_to_h_2d(
        eng.ez,
        ctx.resolution,
        terms=cpml.h_terms,
        psi_h_terms=eng.cpml_psi_h_terms,
    )
    return _update_h_tm_from_curls(eng, ctx, coeffs, curl_hx, curl_hy, psi_h=psi_h)


def _update_e_tm_from_curl(eng, ctx, coeffs, curl_tm_ez, psi_e=None):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    ez = advance_e_from_coefficients(
        eng.ez, curl_tm_ez, coeffs.e_decay_z, coeffs.e_source_z
    )
    return _replace_e(
        eng,
        eng.ex,
        eng.ey,
        ez,
        cpml_e=psi_e,
    )


def update_e_2d_tm_xy(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    curl_ez = tm_xy_curl_h_to_e_2d(
        eng.hx,
        eng.hy,
        ctx.resolution,
        eng.ez.shape,
        ctx.boundary.metallic_edges_2d,
    )
    return _update_e_tm_from_curl(eng, ctx, coeffs, curl_ez)


def update_e_2d_tm_xy_cpml(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    cpml = ctx.boundary.cpml
    curl_ez, psi_e = tm_xy_cpml_curl_h_to_e_2d(
        eng.hx,
        eng.hy,
        ctx.resolution,
        eng.ez.shape,
        ctx.boundary.metallic_edges_2d,
        terms=cpml.e_terms,
        psi_e_terms=eng.cpml_psi_e_terms,
    )
    return _update_e_tm_from_curl(eng, ctx, coeffs, curl_ez, psi_e=psi_e)


def _update_h_te_from_curl(eng, coeffs, curl_hz, psi_h=None):
    hz = advance_h_from_coefficients(
        eng.hz, curl_hz, coeffs.h_decay_z, coeffs.h_source_z
    )
    return _replace_h(eng, eng.hx, eng.hy, hz, cpml_h=psi_h)


def update_h_2d_te_xy(eng, ctx, coeffs):
    curl_hz = te_xy_curl_e_to_h_2d(eng.ex, eng.ey, ctx.resolution, eng.hz.shape)
    return _update_h_te_from_curl(eng, coeffs, curl_hz)


def update_h_2d_te_xy_cpml(eng, ctx, coeffs):
    curl_hz, psi_h = te_xy_cpml_curl_e_to_h_2d(
        eng.ex,
        eng.ey,
        ctx.resolution,
        terms=ctx.boundary.cpml.h_terms,
        psi_h_terms=eng.cpml_psi_h_terms,
    )
    return _update_h_te_from_curl(eng, coeffs, curl_hz, psi_h=psi_h)


def _update_e_te_from_curls(eng, coeffs, curls, psi_e=None):
    ex = advance_e_from_coefficients(
        eng.ex, curls[0], coeffs.e_decay_x, coeffs.e_source_x
    )
    ey = advance_e_from_coefficients(
        eng.ey, curls[1], coeffs.e_decay_y, coeffs.e_source_y
    )
    return _replace_e(eng, ex, ey, eng.ez, cpml_e=psi_e)


def update_e_2d_te_xy(eng, ctx, coeffs):
    curls = te_xy_curl_h_to_e_2d(
        eng.hz,
        ctx.resolution,
        eng.ex.shape,
        eng.ey.shape,
        ctx.boundary.metallic_edges_2d,
    )
    return _update_e_te_from_curls(eng, coeffs, curls)


def update_e_2d_te_xy_cpml(eng, ctx, coeffs):
    curls, psi_e = te_xy_cpml_curl_h_to_e_2d(
        eng.hz,
        ctx.resolution,
        ctx.boundary.metallic_edges_2d,
        terms=ctx.boundary.cpml.e_terms,
        psi_e_terms=eng.cpml_psi_e_terms,
    )
    return _update_e_te_from_curls(eng, coeffs, curls, psi_e=psi_e)


def update_h_3d_cpml_metric(eng, ctx, coeffs):
    cpml = ctx.boundary.cpml
    hx, hy, hz, psi_h = cpml_update_h_from_e_3d_metric(
        eng.ex,
        eng.ey,
        eng.ez,
        eng.hx,
        eng.hy,
        eng.hz,
        ctx.metrics,
        terms=cpml.h_terms,
        psi_terms=eng.cpml_psi_h_terms,
        dt=ctx.dt_scalar,
        magnetic_conductivities=(
            coeffs.h_sigma_m_x,
            coeffs.h_sigma_m_y,
            coeffs.h_sigma_m_z,
        ),
    )
    return _replace_h(eng, hx, hy, hz, cpml_h=psi_h)


def update_e_3d_cpml_metric(eng, ctx, coeffs):
    cpml = ctx.boundary.cpml
    ex, ey, ez, psi_e = cpml_update_e_from_h_3d_metric(
        eng.hx,
        eng.hy,
        eng.hz,
        eng.ex,
        eng.ey,
        eng.ez,
        ctx.metrics,
        terms=cpml.e_terms,
        psi_terms=eng.cpml_psi_e_terms,
        metallic_edges=cpml.metallic_edges,
        dt=ctx.dt_scalar,
        conductivities=(
            coeffs.e_conductivity_x,
            coeffs.e_conductivity_y,
            coeffs.e_conductivity_z,
        ),
        inverse_permittivities=(
            jnp.reciprocal(coeffs.e_permittivity_x),
            jnp.reciprocal(coeffs.e_permittivity_y),
            jnp.reciprocal(coeffs.e_permittivity_z),
        ),
        logical_shapes=ctx.boundary.logical_component_shapes,
    )
    return _replace_e(eng, ex, ey, ez, cpml_e=psi_e)


def update_h_3d_yee_metric(eng, ctx, coeffs):
    hx, hy, hz = fused_update_h_lossy_3d_material_metric(
        eng.ex,
        eng.ey,
        eng.ez,
        eng.hx,
        eng.hy,
        eng.hz,
        coeffs.h_sigma_m_x,
        coeffs.h_sigma_m_y,
        coeffs.h_sigma_m_z,
        ctx.dt_scalar,
        ctx.metrics,
    )
    return _replace_h(eng, hx, hy, hz)


def update_e_3d_yee_metric(eng, ctx, coeffs):
    boundary_views = build_h_boundary_views_for_e_3d(
        eng.hx,
        eng.hy,
        eng.hz,
        ctx.boundary.cpml.metallic_edges,
        logical_shapes=ctx.boundary.logical_component_shapes,
    )
    ex, ey, ez = fused_update_e_lossy_3d_material_metric(
        eng.hx,
        eng.hy,
        eng.hz,
        eng.ex,
        eng.ey,
        eng.ez,
        coeffs.e_conductivity_x,
        jnp.reciprocal(coeffs.e_permittivity_x),
        coeffs.e_conductivity_y,
        jnp.reciprocal(coeffs.e_permittivity_y),
        coeffs.e_conductivity_z,
        jnp.reciprocal(coeffs.e_permittivity_z),
        ctx.dt_scalar,
        ctx.metrics,
        boundary_views=boundary_views,
    )
    return _replace_e(eng, ex, ey, ez)


def update_h_2d_tm_xy_metric(eng, ctx, coeffs):
    curls = tm_xy_curl_e_to_h_2d_metric(eng.ez, ctx.metrics)
    return _update_h_tm_from_curls(eng, ctx, coeffs, *curls)


def update_e_2d_tm_xy_metric(eng, ctx, coeffs):
    curl = tm_xy_curl_h_to_e_2d_metric(
        eng.hx,
        eng.hy,
        ctx.metrics,
        eng.ez.shape,
        ctx.boundary.metallic_edges_2d,
    )
    return _update_e_tm_from_curl(eng, ctx, coeffs, curl)


def update_h_2d_tm_xy_cpml_metric(eng, ctx, coeffs):
    curl_hx, curl_hy, psi_h = tm_xy_cpml_curl_e_to_h_2d_metric(
        eng.ez,
        ctx.metrics,
        terms=ctx.boundary.cpml.h_terms,
        psi_h_terms=eng.cpml_psi_h_terms,
    )
    return _update_h_tm_from_curls(eng, ctx, coeffs, curl_hx, curl_hy, psi_h=psi_h)


def update_e_2d_tm_xy_cpml_metric(eng, ctx, coeffs):
    curl, psi_e = tm_xy_cpml_curl_h_to_e_2d_metric(
        eng.hx,
        eng.hy,
        ctx.metrics,
        eng.ez.shape,
        ctx.boundary.metallic_edges_2d,
        terms=ctx.boundary.cpml.e_terms,
        psi_e_terms=eng.cpml_psi_e_terms,
    )
    return _update_e_tm_from_curl(eng, ctx, coeffs, curl, psi_e=psi_e)


def update_h_2d_te_xy_metric(eng, ctx, coeffs):
    curl = te_xy_curl_e_to_h_2d_metric(eng.ex, eng.ey, ctx.metrics, eng.hz.shape)
    return _update_h_te_from_curl(eng, coeffs, curl)


def update_e_2d_te_xy_metric(eng, ctx, coeffs):
    curls = te_xy_curl_h_to_e_2d_metric(
        eng.hz,
        ctx.metrics,
        eng.ex.shape,
        eng.ey.shape,
        ctx.boundary.metallic_edges_2d,
    )
    return _update_e_te_from_curls(eng, coeffs, curls)


def update_h_2d_te_xy_cpml_metric(eng, ctx, coeffs):
    curl, psi_h = te_xy_cpml_curl_e_to_h_2d_metric(
        eng.ex,
        eng.ey,
        ctx.metrics,
        terms=ctx.boundary.cpml.h_terms,
        psi_h_terms=eng.cpml_psi_h_terms,
    )
    return _update_h_te_from_curl(eng, coeffs, curl, psi_h=psi_h)


def update_e_2d_te_xy_cpml_metric(eng, ctx, coeffs):
    curls, psi_e = te_xy_cpml_curl_h_to_e_2d_metric(
        eng.hz,
        ctx.metrics,
        ctx.boundary.metallic_edges_2d,
        terms=ctx.boundary.cpml.e_terms,
        psi_e_terms=eng.cpml_psi_e_terms,
    )
    return _update_e_te_from_curls(eng, coeffs, curls, psi_e=psi_e)


@dataclass(frozen=True)
class CompiledStepContext:
    """Static data captured by the compiled step builder."""

    # Select one concrete kernel from static plan flags. Timesteps then avoid tracing Python
    # feature branches into every compiled execution.

    config: RunConfig
    boundary: BoundaryPlan
    source_batches: Any
    metrics: Any
    resolution: float
    dt: float
    dt_scalar: jnp.ndarray
    is_3d: bool


@dataclass(frozen=True)
class StepUpdateKernel:
    """Planner-selected E/H update functions for a compiled program."""

    # Group these values because their positional and shape relationships form one
    # invariant.
    kind: str
    update_h: Callable[
        [SimulationState, CompiledStepContext, UpdateCoefficients], SimulationState
    ]
    update_e: Callable[
        [SimulationState, CompiledStepContext, UpdateCoefficients], SimulationState
    ]


def select_update_kernel(ctx: CompiledStepContext) -> StepUpdateKernel:
    """Select the static update-kernel variant before JAX tracing."""

    # 1. Describe variants in priority order: stateful boundaries must precede generic
    # dimensional kernels or their auxiliary state would be ignored.
    boundary = ctx.boundary
    metric = ctx.config.metric_kind != "isotropic_uniform"
    variants = (
        (
            ctx.is_3d and metric and boundary.cpml.enabled,
            f"{ctx.config.metric_kind}_cpml_3d",
            update_h_3d_cpml_metric,
            update_e_3d_cpml_metric,
        ),  # fmt: skip
        (
            ctx.is_3d and metric,
            f"{ctx.config.metric_kind}_yee_3d",
            update_h_3d_yee_metric,
            update_e_3d_yee_metric,
        ),  # fmt: skip
        (
            ctx.is_3d and boundary.cpml.enabled,
            "cpml_3d",
            update_h_3d_cpml,
            update_e_3d_cpml,
        ),  # fmt: skip
        (ctx.is_3d, "yee_3d", update_h_3d_yee, update_e_3d_yee),
        (
            (not ctx.is_3d)
            and metric
            and ctx.config.polarization_2d == "tm"
            and boundary.cpml.enabled,
            f"{ctx.config.metric_kind}_physical_tm_xy_cpml",
            update_h_2d_tm_xy_cpml_metric,
            update_e_2d_tm_xy_cpml_metric,
        ),  # fmt: skip
        (
            (not ctx.is_3d) and metric and ctx.config.polarization_2d == "tm",
            f"{ctx.config.metric_kind}_physical_tm_xy",
            update_h_2d_tm_xy_metric,
            update_e_2d_tm_xy_metric,
        ),  # fmt: skip
        (
            (not ctx.is_3d)
            and ctx.config.polarization_2d == "tm"
            and boundary.cpml.enabled,
            "physical_tm_xy_cpml",
            update_h_2d_tm_xy_cpml,
            update_e_2d_tm_xy_cpml,
        ),  # fmt: skip
        (
            (not ctx.is_3d) and ctx.config.polarization_2d == "tm",
            "physical_tm_xy",
            update_h_2d_tm_xy,
            update_e_2d_tm_xy,
        ),  # fmt: skip
        (
            (not ctx.is_3d) and metric and boundary.cpml.enabled,
            f"{ctx.config.metric_kind}_physical_te_xy_cpml",
            update_h_2d_te_xy_cpml_metric,
            update_e_2d_te_xy_cpml_metric,
        ),  # fmt: skip
        (
            (not ctx.is_3d) and metric,
            f"{ctx.config.metric_kind}_physical_te_xy",
            update_h_2d_te_xy_metric,
            update_e_2d_te_xy_metric,
        ),  # fmt: skip
        (
            (not ctx.is_3d) and boundary.cpml.enabled,
            "physical_te_xy_cpml",
            update_h_2d_te_xy_cpml,
            update_e_2d_te_xy_cpml,
        ),  # fmt: skip
        (
            not ctx.is_3d,
            "physical_te_xy",
            update_h_2d_te_xy,
            update_e_2d_te_xy,
        ),  # fmt: skip
    )

    # 2. Every supported runtime is canonical 3D, TMxy, or TExy.
    _, kind, update_h, update_e = next(variant for variant in variants if variant[0])
    return StepUpdateKernel(kind, update_h, update_e)
