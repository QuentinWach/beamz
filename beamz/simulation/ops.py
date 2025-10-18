"""Numerical operations for FDTD field updates: curls, field advancement, material handling on staggered Yee grids."""
import numpy as np
from beamz.const import EPS_0, MU_0


def finite_diff(arr, axis, spacing):
    """Compute centered finite difference along specified axis."""
    return np.diff(arr, axis=axis) / spacing


def pad_like(reference_shape, value, axis, forward):
    """Pad array to match reference shape along one axis (restores shape after finite differencing)."""
    pad_width = [(0, 0)] * len(reference_shape)
    if forward: pad_width[axis] = (0, 1)
    else: pad_width[axis] = (1, 0)
    return np.pad(value, pad_width, mode="constant")


def curl_e_to_h_2d(ez, dx, dy):
    """Compute curl of E-field for H update in 2D: ∂H/∂t = -∇×E/μ₀."""
    curl_ex = pad_like(ez.shape, finite_diff(ez, axis=1, spacing=dy), axis=1, forward=False)
    curl_ey = pad_like(ez.shape, -finite_diff(ez, axis=0, spacing=dx), axis=0, forward=False)
    return (curl_ex, curl_ey)


def curl_e_to_h_3d(ex, ey, ez, dx, dy, dz):
    """Compute curl of E-field for H update in 3D: ∂H/∂t = -∇×E/μ₀."""
    curl_ex = pad_like(ex.shape, finite_diff(ez, axis=1, spacing=dy), axis=1, forward=False) \
            - pad_like(ex.shape, finite_diff(ey, axis=0, spacing=dz), axis=0, forward=False)
    curl_ey = pad_like(ey.shape, finite_diff(ex, axis=0, spacing=dz), axis=0, forward=False) \
            - pad_like(ey.shape, finite_diff(ez, axis=2, spacing=dx), axis=2, forward=False)
    curl_ez = pad_like(ez.shape, finite_diff(ey, axis=2, spacing=dx), axis=2, forward=False) \
            - pad_like(ez.shape, finite_diff(ex, axis=1, spacing=dy), axis=1, forward=False)
    return (curl_ex, curl_ey, curl_ez)


def curl_h_to_e_2d(hx, hy, dx, dy, target_shape):
    """Compute curl of H-field for E update in 2D: ∂E/∂t = ∇×H/(ε₀εᵣ)."""
    curl = np.zeros(target_shape)
    curl[1:-1, 1:-1] = ((hy[1:, 1:-1] - hy[:-1, 1:-1]) / dx - (hx[1:-1, 1:] - hx[1:-1, :-1]) / dy)
    return (curl,)


def curl_h_to_e_3d(hx, hy, hz, dx, dy, dz):
    """Compute curl of H-field for E update in 3D: ∂E/∂t = ∇×H/(ε₀εᵣ)."""
    curl_hx = (hz[:, 1:, :] - hz[:, :-1, :]) / dy - (hy[1:, :, :] - hy[:-1, :, :]) / dz
    curl_hy = (hx[1:, :, :] - hx[:-1, :, :]) / dz - (hz[:, :, 1:] - hz[:, :, :-1]) / dx
    curl_hz = (hy[:, :, 1:] - hy[:, :, :-1]) / dx - (hx[:, 1:, :] - hx[:, :-1, :]) / dy
    return (curl_hx, curl_hy, curl_hz)


def magnetic_conductivity_terms_2d(sigma, hx_shape, hy_shape):
    """Compute magnetic conductivity σ_m = σ * μ₀/ε₀ for H-field PML absorption in 2D."""
    if sigma.ndim == 2:
        sigma_m_x = sigma[:, :-1] * MU_0 / EPS_0
        sigma_m_y = sigma[:-1, :] * MU_0 / EPS_0
        return (sigma_m_x.reshape(hx_shape), sigma_m_y.reshape(hy_shape))
    return (np.zeros(hx_shape), np.zeros(hy_shape))


def magnetic_conductivity_terms_3d(sigma, hx_shape, hy_shape, hz_shape):
    """Compute magnetic conductivity σ_m = σ * μ₀/ε₀ for H-field PML absorption in 3D."""
    if sigma.ndim == 3:
        sigma_m_hx = (sigma[:-1, :-1, :] * MU_0 / EPS_0).reshape(hx_shape)
        sigma_m_hy = (sigma[:-1, :, :-1] * MU_0 / EPS_0).reshape(hy_shape)
        sigma_m_hz = (sigma[:, :-1, :-1] * MU_0 / EPS_0).reshape(hz_shape)
        return (sigma_m_hx, sigma_m_hy, sigma_m_hz)
    return (np.zeros(hx_shape), np.zeros(hy_shape), np.zeros(hz_shape))


def advance_h_field(field, curl, sigma_m, dt):
    """Advance H-field one time step via Crank-Nicolson: ∂H/∂t = -∇×E/μ₀ - σ_m*H/μ₀."""
    denom = 1.0 + sigma_m * dt / (2.0 * MU_0)
    factor = (1.0 - sigma_m * dt / (2.0 * MU_0)) / denom
    source = (dt / MU_0) / denom
    return factor * field - source * curl


def advance_e_field(field, curl, sigma, eps_r, dt, region):
    """Advance E-field one time step via Crank-Nicolson: ∂E/∂t = ∇×H/(ε₀εᵣ) - σE/(ε₀εᵣ)."""
    updated = field.copy()
    if region is None: region = (...,)
    current, sig, eps = field[region], sigma[region], eps_r[region]
    denom = 1.0 + sig * dt / (2.0 * EPS_0 * eps)
    factor = (1.0 - sig * dt / (2.0 * EPS_0 * eps)) / denom
    source = (dt / (EPS_0 * eps)) / denom
    updated[region] = factor * current + source * curl
    return updated


def material_slice_for_e_2d(eps_r, sigma):
    """Extract material parameters at staggered Yee grid positions for E-field in 2D."""
    region = (slice(1, -1), slice(1, -1))
    return eps_r[region], sigma[region], region


def material_slice_for_e_3d(eps_r, sigma, orientation):
    """Extract material parameters at staggered Yee grid positions for E-field components in 3D."""
    if orientation == "x": region = (slice(1, -1), slice(1, -1), slice(None))
    elif orientation == "y": region = (slice(1, -1), slice(None), slice(1, -1))
    else: region = (slice(None), slice(1, -1), slice(1, -1))
    return eps_r[region], sigma[region], region
