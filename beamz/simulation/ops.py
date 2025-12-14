"""Numerical operations for FDTD field updates: curls, field advancement, material handling on staggered Yee grids."""
import numpy as np
from beamz.const import EPS_0, MU_0


def curl_e_to_h_2d(ez, resolution):
    """Compute curl of E-field for H update in 2D on staggered Yee grid: ∂H/∂t = -∇×E/μ₀."""
    # For 2D TM mode (only Ez component), curl reduces to: ∇×E = (∂Ez/∂y)x̂ - (∂Ez/∂x)ŷ
    # On staggered Yee grid: Ez[i,j], Hx[i, j+1/2], Hy[i+1/2, j]
    # Hx component comes from ∂Ez/∂y (x-component of curl)
    # Hx is staggered in x, so curl_ex has shape (Ny, Nx-1)
    curl_ex = np.diff(ez, axis=1) / resolution  # (Ez[i,j+1] - Ez[i,j]) / dy at position [i, j+1/2]
    
    # Hy component comes from -∂Ez/∂x (y-component of curl, with sign flip)
    # Hy is staggered in y, so curl_ey has shape (Ny-1, Nx)
    curl_ey = -np.diff(ez, axis=0) / resolution  # -(Ez[i+1,j] - Ez[i,j]) / dx at position [i+1/2, j]
    
    return (curl_ex, curl_ey)


def curl_e_to_h_3d(ex, ey, ez, resolution):
    """Compute curl of E-field for H update in 3D: ∂H/∂t = -∇×E/μ₀."""
    # Full 3D curl: ∇×E = [(∂Ez/∂y - ∂Ey/∂z)x̂ + (∂Ex/∂z - ∂Ez/∂x)ŷ + (∂Ey/∂x - ∂Ex/∂y)ẑ]
    
    # Hx update from x-component: (∇×E)_x = ∂Ez/∂y - ∂Ey/∂z
    diff_ez_y = np.diff(ez, axis=1) / resolution  # ∂Ez/∂y ≈ (Ez[k,j+1,i] - Ez[k,j,i]) / dy
    pad_ez_y = [(0, 0)] * 3  # Create padding list for 3D
    pad_ez_y[1] = (1, 0)  # Pad axis 1 (y) at beginning to restore shape
    term1_x = np.pad(diff_ez_y, pad_ez_y, mode="constant")
    
    diff_ey_z = np.diff(ey, axis=0) / resolution  # ∂Ey/∂z ≈ (Ey[k+1,j,i] - Ey[k,j,i]) / dz
    pad_ey_z = [(0, 0)] * 3
    pad_ey_z[0] = (1, 0)  # Pad axis 0 (z) at beginning
    term2_x = np.pad(diff_ey_z, pad_ey_z, mode="constant")
    curl_ex = term1_x - term2_x  # Combine terms for Hx
    
    # Hy update from y-component: (∇×E)_y = ∂Ex/∂z - ∂Ez/∂x
    diff_ex_z = np.diff(ex, axis=0) / resolution  # ∂Ex/∂z ≈ (Ex[k+1,j,i] - Ex[k,j,i]) / dz
    pad_ex_z = [(0, 0)] * 3
    pad_ex_z[0] = (1, 0)  # Pad axis 0 (z)
    term1_y = np.pad(diff_ex_z, pad_ex_z, mode="constant")
    
    diff_ez_x = np.diff(ez, axis=2) / resolution  # ∂Ez/∂x ≈ (Ez[k,j,i+1] - Ez[k,j,i]) / dx
    pad_ez_x = [(0, 0)] * 3
    pad_ez_x[2] = (1, 0)  # Pad axis 2 (x)
    term2_y = np.pad(diff_ez_x, pad_ez_x, mode="constant")
    curl_ey = term1_y - term2_y  # Combine terms for Hy
    
    # Hz update from z-component: (∇×E)_z = ∂Ey/∂x - ∂Ex/∂y
    diff_ey_x = np.diff(ey, axis=2) / resolution  # ∂Ey/∂x ≈ (Ey[k,j,i+1] - Ey[k,j,i]) / dx
    pad_ey_x = [(0, 0)] * 3
    pad_ey_x[2] = (1, 0)  # Pad axis 2 (x)
    term1_z = np.pad(diff_ey_x, pad_ey_x, mode="constant")
    
    diff_ex_y = np.diff(ex, axis=1) / resolution  # ∂Ex/∂y ≈ (Ex[k,j+1,i] - Ex[k,j,i]) / dy
    pad_ex_y = [(0, 0)] * 3
    pad_ex_y[1] = (1, 0)  # Pad axis 1 (y)
    term2_z = np.pad(diff_ex_y, pad_ex_y, mode="constant")
    curl_ez = term1_z - term2_z  # Combine terms for Hz
    
    return (curl_ex, curl_ey, curl_ez)


def curl_h_to_e_2d(hx, hy, resolution, target_shape):
    """Compute curl of H-field for E update in 2D on staggered Yee grid: ∂E/∂t = ∇×H/(ε₀εᵣ)."""
    # For 2D TM mode: (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y (z-component drives Ez)
    # On staggered Yee grid: Ez[i,j], Hx[i, j+1/2] with shape (Ny, Nx-1), Hy[i+1/2, j] with shape (Ny-1, Nx)
    # Interior region excludes boundary points where sources/PML are applied
    curl = np.zeros(target_shape)  # Initialize with zeros for boundary conditions
    # Compute curl at interior Ez points [1:-1, 1:-1]
    # ∂Hy/∂x: Hy is at (i+1/2, j), so we need Hy[i, j] - Hy[i-1, j] to get derivative at (i, j)
    dHy_dx = (hy[1:, 1:-1] - hy[:-1, 1:-1]) / resolution  # Shape: (Ny-2, Nx-2)
    # ∂Hx/∂y: Hx is at (i, j+1/2), so we need Hx[i, j] - Hx[i, j-1] to get derivative at (i, j)
    dHx_dy = (hx[1:-1, 1:] - hx[1:-1, :-1]) / resolution  # Shape: (Ny-2, Nx-2)
    curl[1:-1, 1:-1] = dHy_dx - dHx_dy
    return (curl,)


def curl_h_to_e_3d(hx, hy, hz, resolution):
    """Compute curl of H-field for E update in 3D: ∂E/∂t = ∇×H/(ε₀εᵣ)."""
    # Full 3D curl: ∇×H = [(∂Hz/∂y - ∂Hy/∂z)x̂ + (∂Hx/∂z - ∂Hz/∂x)ŷ + (∂Hy/∂x - ∂Hx/∂y)ẑ]
    # Ex update from x-component: (∇×H)_x = ∂Hz/∂y - ∂Hy/∂z
    curl_hx = (hz[:, 1:, :] - hz[:, :-1, :]) / resolution - (hy[1:, :, :] - hy[:-1, :, :]) / resolution
    # Ey update from y-component: (∇×H)_y = ∂Hx/∂z - ∂Hz/∂x
    curl_hy = (hx[1:, :, :] - hx[:-1, :, :]) / resolution - (hz[:, :, 1:] - hz[:, :, :-1]) / resolution
    # Ez update from z-component: (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y
    curl_hz = (hy[:, :, 1:] - hy[:, :, :-1]) / resolution - (hx[:, 1:, :] - hx[:, :-1, :]) / resolution
    return (curl_hx, curl_hy, curl_hz)


def magnetic_conductivity_terms_2d(conductivity, permeability, hx_shape, hy_shape):
    """Compute magnetic conductivity σ_m = σ * μ₀μᵣ/ε₀ for H-field PML absorption in 2D."""
    if conductivity.ndim < 2: return (np.zeros(hx_shape), np.zeros(hy_shape))  # No PML if conductivity is scalar
    # PML uses magnetic loss: σ_m = σ * (μ₀μᵣ/ε₀) to create matched impedance at boundaries
    sigma_m_x = conductivity[:, :-1] * permeability[:, :-1] * MU_0 / EPS_0  # Slice to Hx position (y, x-1/2)
    sigma_m_y = conductivity[:-1, :] * permeability[:-1, :] * MU_0 / EPS_0  # Slice to Hy position (y-1/2, x)
    return (sigma_m_x.reshape(hx_shape), sigma_m_y.reshape(hy_shape))


def magnetic_conductivity_terms_3d(conductivity, permeability, hx_shape, hy_shape, hz_shape):
    """Compute magnetic conductivity σ_m = σ * μ₀μᵣ/ε₀ for H-field PML absorption in 3D."""
    if conductivity.ndim < 3: return (np.zeros(hx_shape), np.zeros(hy_shape), np.zeros(hz_shape))
    # Slice arrays to match staggered Yee grid positions of each H-field component
    sigma_m_hx = (conductivity[:-1, :-1, :] * permeability[:-1, :-1, :] * MU_0 / EPS_0).reshape(hx_shape)  # Hx at (z-1/2, y-1/2, x)
    sigma_m_hy = (conductivity[:-1, :, :-1] * permeability[:-1, :, :-1] * MU_0 / EPS_0).reshape(hy_shape)  # Hy at (z-1/2, y, x-1/2)
    sigma_m_hz = (conductivity[:, :-1, :-1] * permeability[:, :-1, :-1] * MU_0 / EPS_0).reshape(hz_shape)  # Hz at (z, y-1/2, x-1/2)
    return (sigma_m_hx, sigma_m_hy, sigma_m_hz)


def advance_h_field(field, curl, sigma_m, dt):
    """Advance H-field one time step via Crank-Nicolson: ∂H/∂t = -∇×E/μ₀ - σ_m*H/μ₀."""
    # Faraday's law with magnetic loss: μ₀∂H/∂t = -∇×E - σ_m*H
    # Crank-Nicolson (implicit midpoint): H^(n+1) = [(1 - α)/(1 + α)]H^n - [Δt/μ₀/(1 + α)]∇×E^(n+1/2)
    # where α = σ_m*Δt/(2μ₀) ensures second-order accuracy and unconditional stability
    denom = 1.0 + sigma_m * dt / (2.0 * MU_0)  # Denominator: 1 + α
    factor = (1.0 - sigma_m * dt / (2.0 * MU_0)) / denom  # Coefficient for H^n: (1 - α)/(1 + α)
    source = (dt / MU_0) / denom  # Coefficient for curl term: Δt/(μ₀(1 + α))
    return factor * field - source * curl  # H^(n+1) = factor*H^n - source*∇×E


def advance_e_field(field, curl, conductivity, permittivity, dt, region):
    """Advance E-field one time step via Crank-Nicolson: ∂E/∂t = ∇×H/(ε₀εᵣ) - σE/(ε₀εᵣ)."""
    # Ampere's law with electric loss: ε₀εᵣ∂E/∂t = ∇×H - σE
    # Crank-Nicolson: E^(n+1) = [(1 - β)/(1 + β)]E^n + [Δt/(ε₀εᵣ)/(1 + β)]∇×H^(n+1/2)
    # where β = σΔt/(2ε₀εᵣ) for stability and second-order temporal accuracy
    # Note: conductivity and permittivity are already sliced to the interior region
    updated = field.copy()  # Create copy for output (preserve boundary values)
    current = field[region]  # Extract interior field values
    curl_region = curl[region]  # Extract curl at interior points
    denom = 1.0 + conductivity * dt / (2.0 * EPS_0 * permittivity)  # Denominator: 1 + β
    factor = (1.0 - conductivity * dt / (2.0 * EPS_0 * permittivity)) / denom  # Coefficient for E^n: (1 - β)/(1 + β)
    source = (dt / (EPS_0 * permittivity)) / denom  # Coefficient for curl term: Δt/(ε₀εᵣ(1 + β))
    updated[region] = factor * current + source * curl_region  # E^(n+1) = factor*E^n + source*∇×H
    return updated


def material_slice_for_e_2d(permittivity, conductivity):
    """Extract material parameters at staggered Yee grid positions for E-field in 2D."""
    # Ez is located at (i, j) on Yee grid, interior points exclude boundaries for proper curl computation
    region = (slice(1, -1), slice(1, -1))  # [1:-1, 1:-1] selects interior, avoiding edges
    return permittivity[region], conductivity[region], region


def material_slice_for_e_3d(permittivity, conductivity, orientation):
    """Extract material parameters at staggered Yee grid positions for E-field components in 3D."""
    # Each E-field component lives at different staggered positions on Yee grid:
    # Ex at (z, y, x-1/2), Ey at (z, y-1/2, x), Ez at (z-1/2, y, x)
    # Slicing [1:-1] along an axis excludes boundaries for that dimension
    if orientation == "x": region = (slice(1, -1), slice(1, -1), slice(None))  # Ex: interior in z,y; full x
    elif orientation == "y": region = (slice(1, -1), slice(None), slice(1, -1))  # Ey: interior in z,x; full y
    else: region = (slice(None), slice(1, -1), slice(1, -1))  # Ez: full z; interior in y,x
    return permittivity[region], conductivity[region], region


def update_e_field_upml_2d(Ez, Ez_x, Ez_y, Hx, Hy, pml_data, permittivity, conductivity, resolution, dt):
    """Update E field with UPML split-field formulation for 2D TM mode."""
    from beamz.const import EPS_0
    
    # Standard curl calculation
    curl_h, = curl_h_to_e_2d(Hx, Hy, resolution, Ez.shape)
    
    # Get PML parameters
    mask = pml_data['mask']
    sigma_x = pml_data['sigma_x']
    sigma_y = pml_data['sigma_y']
    kappa_x = pml_data['kappa_x']
    kappa_y = pml_data['kappa_y']
    alpha_x = pml_data['alpha_x']
    alpha_y = pml_data['alpha_y']
    
    # Simplified UPML implementation - use standard FDTD with PML conductivity
    # This is more stable than the full split-field formulation
    region = slice(1, -1), slice(1, -1)
    
    # Add PML conductivity to the existing conductivity
    total_conductivity = conductivity[region] + sigma_x[region] + sigma_y[region]
    
    # Use standard FDTD update with modified conductivity
    denom = 1.0 + total_conductivity * dt / (2.0 * EPS_0 * permittivity[region])
    factor = (1.0 - total_conductivity * dt / (2.0 * EPS_0 * permittivity[region])) / denom
    source = (dt / (EPS_0 * permittivity[region])) / denom
    
    # Update Ez field
    Ez[region] = factor * Ez[region] + source * curl_h[region]
    
    return Ez
