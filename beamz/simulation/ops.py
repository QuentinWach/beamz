"""Numerical operations shared by FDTD field updates."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from beamz.const import EPS_0, MU_0

ArrayLike = np.ndarray


def curl_e_to_h_2d(ez: ArrayLike, dx: float, dy: float) -> Tuple[ArrayLike, ArrayLike]:
    curl_ex = (ez[:, 1:] - ez[:, :-1]) / dy
    curl_ey = (ez[1:, :] - ez[:-1, :]) / dx
    return curl_ex, curl_ey


def curl_h_to_e_2d(hx: ArrayLike, hy: ArrayLike, dx: float, dy: float, shape: Tuple[int, int]) -> ArrayLike:
    curl = np.zeros(shape, dtype=np.result_type(hx, hy))
    curl[1:-1, 1:-1] = (
        (hy[1:, 1:-1] - hy[:-1, 1:-1]) / dx
        - (hx[1:-1, 1:] - hx[1:-1, :-1]) / dy
    )
    return curl


def magnetic_conductivity_terms_2d(sigma: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    sigma_m_x = sigma[:, :-1] * MU_0 / EPS_0
    sigma_m_y = sigma[:-1, :] * MU_0 / EPS_0
    return sigma_m_x, sigma_m_y


def advance_e_field_2d(ez: ArrayLike, curl: ArrayLike, sigma: ArrayLike, eps_r: ArrayLike, dt: float) -> ArrayLike:
    ez_new = ez.copy()
    interior = (slice(1, -1), slice(1, -1))
    sig = sigma[interior]
    eps = eps_r[interior]
    denom = 1.0 + sig * dt / (2.0 * EPS_0 * eps)
    factor = (1.0 - sig * dt / (2.0 * EPS_0 * eps)) / denom
    source = (dt / (EPS_0 * eps)) / denom
    ez_new[interior] = factor * ez[interior] + source * curl[interior]
    return ez_new


def curl_e_to_h_3d(ex: ArrayLike, ey: ArrayLike, ez: ArrayLike, dx: float, dy: float, dz: float) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    dEz_dy = (ez[:, 1:, :] - ez[:, :-1, :]) / dy
    dEy_dz = (ey[1:, :, :] - ey[:-1, :, :]) / dz
    curl_ex = dEz_dy - dEy_dz

    dEx_dz = (ex[1:, :, :] - ex[:-1, :, :]) / dz
    dEz_dx = (ez[:, :, 1:] - ez[:, :, :-1]) / dx
    curl_ey = dEx_dz - dEz_dx

    dEy_dx = (ey[:, :, 1:] - ey[:, :, :-1]) / dx
    dEx_dy = (ex[:, 1:, :] - ex[:, :-1, :]) / dy
    curl_ez = dEy_dx - dEx_dy
    return curl_ex, curl_ey, curl_ez


def magnetic_conductivity_terms_3d(
    sigma: ArrayLike,
    hx_shape: Tuple[int, int, int],
    hy_shape: Tuple[int, int, int],
    hz_shape: Tuple[int, int, int],
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    dtype = sigma.dtype if isinstance(sigma, np.ndarray) else np.float64
    if sigma.ndim == 3:
        sigma_m_hx = (sigma[:-1, :-1, :] * MU_0 / EPS_0).reshape(hx_shape)
        sigma_m_hy = (sigma[:-1, :, :-1] * MU_0 / EPS_0).reshape(hy_shape)
        sigma_m_hz = (sigma[:, :-1, :-1] * MU_0 / EPS_0).reshape(hz_shape)
    else:
        sigma_m_hx = np.zeros(hx_shape, dtype=dtype)
        sigma_m_hy = np.zeros(hy_shape, dtype=dtype)
        sigma_m_hz = np.zeros(hz_shape, dtype=dtype)
    return sigma_m_hx, sigma_m_hy, sigma_m_hz


def advance_h_field(field: ArrayLike, curl: ArrayLike, sigma_m: ArrayLike, dt: float) -> ArrayLike:
    denom = 1.0 + sigma_m * dt / (2.0 * MU_0)
    factor = (1.0 - sigma_m * dt / (2.0 * MU_0)) / denom
    source = (dt / MU_0) / denom
    return factor * field - source * curl


def curl_h_to_e_3d(hx: ArrayLike, hy: ArrayLike, hz: ArrayLike, dx: float, dy: float, dz: float) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    dHz_dy = (hz[:, 1:, :] - hz[:, :-1, :]) / dy
    dHy_dz = (hy[1:, :, :] - hy[:-1, :, :]) / dz
    curl_hx = dHz_dy[1:-1, :, :] - dHy_dz[:, 1:-1, :]

    dHx_dz = (hx[1:, :, :] - hx[:-1, :, :]) / dz
    dHz_dx = (hz[:, :, 1:] - hz[:, :, :-1]) / dx
    curl_hy = dHx_dz[:, :, 1:-1] - dHz_dx[1:-1, :, :]

    dHy_dx = (hy[:, :, 1:] - hy[:, :, :-1]) / dx
    dHx_dy = (hx[:, 1:, :] - hx[:, :-1, :]) / dy
    curl_hz = dHy_dx[:, 1:-1, :] - dHx_dy[:, :, 1:-1]
    return curl_hx, curl_hy, curl_hz


def material_slice_for_e(eps_r: ArrayLike, sigma: ArrayLike, orientation: str) -> Tuple[ArrayLike, ArrayLike, Tuple]:
    if eps_r.ndim == 3:
        if orientation == "x":
            eps = eps_r[1:-1, 1:-1, :-1]
            sig = sigma[1:-1, 1:-1, :-1]
            region = (slice(1, -1), slice(1, -1), slice(None))
        elif orientation == "y":
            eps = eps_r[1:-1, :-1, 1:-1]
            sig = sigma[1:-1, :-1, 1:-1]
            region = (slice(1, -1), slice(None), slice(1, -1))
        else:
            eps = eps_r[:-1, 1:-1, 1:-1]
            sig = sigma[:-1, 1:-1, 1:-1]
            region = (slice(None), slice(1, -1), slice(1, -1))
    else:
        eps = eps_r
        sig = sigma
        region = None
    return eps, sig, region


def advance_e_field(field: ArrayLike, curl: ArrayLike, eps: ArrayLike, sig: ArrayLike, region: Tuple, dt: float) -> ArrayLike:
    updated = field.copy()
    if region is None:
        region = (...,)
    current = field[region]
    denom = 1.0 + sig * dt / (2.0 * EPS_0 * eps)
    factor = (1.0 - sig * dt / (2.0 * EPS_0 * eps)) / denom
    source = (dt / (EPS_0 * eps)) / denom
    updated[region] = factor * current + source * curl
    return updated
