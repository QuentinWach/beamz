"""Numerical operations for FDTD field updates: curls, field advancement, material handling on staggered Yee grids."""
import numpy as np
from beamz.const import EPS_0, MU_0


def curl_e_to_h_2d(ez, dx, dy):
    """Compute curl of E-field for H update in 2D: ∂H/∂t = -∇×E/μ₀."""
    diff_y = np.diff(ez, axis=1) / dy
    pad_width_y = [(0, 0)] * len(ez.shape)
    pad_width_y[1] = (1, 0)
    curl_ex = np.pad(diff_y, pad_width_y, mode="constant")
    
    diff_x = -np.diff(ez, axis=0) / dx
    pad_width_x = [(0, 0)] * len(ez.shape)
    pad_width_x[0] = (1, 0)
    curl_ey = np.pad(diff_x, pad_width_x, mode="constant")
    return (curl_ex, curl_ey)


def curl_e_to_h_3d(ex, ey, ez, dx, dy, dz):
    """Compute curl of E-field for H update in 3D: ∂H/∂t = -∇×E/μ₀."""
    # curl_ex = ∂Ez/∂y - ∂Ey/∂z
    diff_ez_y = np.diff(ez, axis=1) / dy
    pad_ez_y = [(0, 0)] * 3
    pad_ez_y[1] = (1, 0)
    term1_x = np.pad(diff_ez_y, pad_ez_y, mode="constant")
    
    diff_ey_z = np.diff(ey, axis=0) / dz
    pad_ey_z = [(0, 0)] * 3
    pad_ey_z[0] = (1, 0)
    term2_x = np.pad(diff_ey_z, pad_ey_z, mode="constant")
    curl_ex = term1_x - term2_x
    
    # curl_ey = ∂Ex/∂z - ∂Ez/∂x
    diff_ex_z = np.diff(ex, axis=0) / dz
    pad_ex_z = [(0, 0)] * 3
    pad_ex_z[0] = (1, 0)
    term1_y = np.pad(diff_ex_z, pad_ex_z, mode="constant")
    
    diff_ez_x = np.diff(ez, axis=2) / dx
    pad_ez_x = [(0, 0)] * 3
    pad_ez_x[2] = (1, 0)
    term2_y = np.pad(diff_ez_x, pad_ez_x, mode="constant")
    curl_ey = term1_y - term2_y
    
    # curl_ez = ∂Ey/∂x - ∂Ex/∂y
    diff_ey_x = np.diff(ey, axis=2) / dx
    pad_ey_x = [(0, 0)] * 3
    pad_ey_x[2] = (1, 0)
    term1_z = np.pad(diff_ey_x, pad_ey_x, mode="constant")
    
    diff_ex_y = np.diff(ex, axis=1) / dy
    pad_ex_y = [(0, 0)] * 3
    pad_ex_y[1] = (1, 0)
    term2_z = np.pad(diff_ex_y, pad_ex_y, mode="constant")
    curl_ez = term1_z - term2_z
    
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
    if sigma.ndim < 2: return (np.zeros(hx_shape), np.zeros(hy_shape))
    sigma_m_x = sigma[:, :-1] * MU_0 / EPS_0
    sigma_m_y = sigma[:-1, :] * MU_0 / EPS_0
    return (sigma_m_x.reshape(hx_shape), sigma_m_y.reshape(hy_shape))


def magnetic_conductivity_terms_3d(sigma, hx_shape, hy_shape, hz_shape):
    """Compute magnetic conductivity σ_m = σ * μ₀/ε₀ for H-field PML absorption in 3D."""
    if sigma.ndim < 3: return (np.zeros(hx_shape), np.zeros(hy_shape), np.zeros(hz_shape))
    sigma_m_hx = (sigma[:-1, :-1, :] * MU_0 / EPS_0).reshape(hx_shape)
    sigma_m_hy = (sigma[:-1, :, :-1] * MU_0 / EPS_0).reshape(hy_shape)
    sigma_m_hz = (sigma[:, :-1, :-1] * MU_0 / EPS_0).reshape(hz_shape)
    return (sigma_m_hx, sigma_m_hy, sigma_m_hz)


def advance_h_field(field, curl, sigma_m, dt):
    """Advance H-field one time step via Crank-Nicolson: ∂H/∂t = -∇×E/μ₀ - σ_m*H/μ₀."""
    denom = 1.0 + sigma_m * dt / (2.0 * MU_0)
    factor = (1.0 - sigma_m * dt / (2.0 * MU_0)) / denom
    source = (dt / MU_0) / denom
    return factor * field - source * curl


def advance_e_field(field, curl, sigma, eps_r, dt, region):
    """Advance E-field one time step via Crank-Nicolson: ∂E/∂t = ∇×H/(ε₀εᵣ) - σE/(ε₀εᵣ)."""
    updated = field.copy()
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
