"""Shared utility functions for BeamZ FDTD physics validation tests."""
import numpy as np
from beamz import LIGHT_SPEED, EPS_0, um

# =============================================================================
# Constants
# =============================================================================
TEST_WAVELENGTH = 1.0 * um  # Standard test wavelength (1 micron)
TEST_FREQUENCY = LIGHT_SPEED / TEST_WAVELENGTH


# =============================================================================
# Helper Functions
# =============================================================================
def compute_field_energy(Ez, dx, eps=1.0):
    """Compute total electric field energy in the domain.

    U = (1/2) * eps_0 * eps_r * integral(E^2) dA

    Args:
        Ez: 2D field array
        dx: Grid spacing
        eps: Relative permittivity (scalar or array)

    Returns:
        Total field energy
    """
    return 0.5 * EPS_0 * np.sum(eps * Ez**2) * dx * dx


def estimate_phase_velocity(field_snapshots, dx, dt_snapshot, threshold=0.3):
    """Estimate phase velocity by tracking wavefront position.

    Tracks the rightmost position where field exceeds threshold * max
    at each time step, then fits a line to get velocity.

    Args:
        field_snapshots: List of 2D field arrays over time
        dx: Grid spacing
        dt_snapshot: Time between snapshots
        threshold: Fraction of max amplitude to define wavefront

    Returns:
        Estimated phase velocity (m/s), or None if insufficient data
    """
    positions = []
    times = []

    for t_idx, field in enumerate(field_snapshots):
        # Average over y to get 1D profile
        field_1d = np.abs(field).mean(axis=0)
        max_val = np.max(field_1d)

        if max_val > 1e-20:  # Skip empty fields
            # Find rightmost position above threshold
            above_threshold = np.where(field_1d > threshold * max_val)[0]
            if len(above_threshold) > 0:
                positions.append(above_threshold[-1] * dx)
                times.append(t_idx * dt_snapshot)

    if len(positions) < 5:
        return None

    # Use only middle portion (after source ramps up, before hitting boundary)
    start_idx = len(positions) // 4
    end_idx = 3 * len(positions) // 4

    if end_idx - start_idx < 3:
        return None

    # Linear fit
    coeffs = np.polyfit(times[start_idx:end_idx], positions[start_idx:end_idx], 1)
    return coeffs[0]  # Slope is velocity


# =============================================================================
# Fresnel Coefficient Functions
# =============================================================================
def analytical_fresnel_r(n1, n2):
    """Fresnel reflection coefficient (power) at normal incidence.

    R = ((n1 - n2) / (n1 + n2))^2

    Args:
        n1: Refractive index of incident medium
        n2: Refractive index of transmitted medium

    Returns:
        Power reflection coefficient R
    """
    return ((n1 - n2) / (n1 + n2)) ** 2


def analytical_fresnel_t(n1, n2):
    """Fresnel transmission coefficient (power) at normal incidence.

    T = 4*n1*n2 / (n1 + n2)^2

    Note: T = 1 - R for lossless interface.

    Args:
        n1: Refractive index of incident medium
        n2: Refractive index of transmitted medium

    Returns:
        Power transmission coefficient T
    """
    return 4 * n1 * n2 / (n1 + n2) ** 2


# =============================================================================
# Poynting Vector / Energy Functions
# =============================================================================
def compute_poynting_flux_2d(Ez, Hx, Hy, dx):
    """Compute total Poynting flux (power) in 2D domain.

    S = E x H, integrated over the domain.

    Args:
        Ez: Electric field z-component (2D array)
        Hx: Magnetic field x-component (2D array)
        Hy: Magnetic field y-component (2D array)
        dx: Grid spacing

    Returns:
        Total power (integrated |S|)
    """
    # Handle shape mismatches from Yee grid staggering
    ny, nx = Ez.shape
    # Interpolate H to E locations if needed
    if Hx.shape != Ez.shape:
        # Hx is staggered, average to Ez locations
        Hx_interp = np.zeros_like(Ez)
        Hx_interp[:, :-1] = 0.5 * (Hx[:, :-1] + Hx[:, 1:]) if Hx.shape[1] > 1 else Hx
        Hx = Hx_interp
    if Hy.shape != Ez.shape:
        Hy_interp = np.zeros_like(Ez)
        Hy_interp[:-1, :] = 0.5 * (Hy[:-1, :] + Hy[1:, :]) if Hy.shape[0] > 1 else Hy
        Hy = Hy_interp

    Sx = -Ez * Hy
    Sy = Ez * Hx
    power_density = np.sqrt(Sx**2 + Sy**2)
    return np.sum(power_density) * dx * dx


def compute_directional_flux_2d(Ez, Hx, Hy, dx, direction='x'):
    """Compute directional Poynting flux along a line.

    Args:
        Ez, Hx, Hy: Field components
        dx: Grid spacing
        direction: 'x' for Sx, 'y' for Sy

    Returns:
        Directional flux (can be positive or negative)
    """
    if direction == 'x':
        # Sx = -Ez * Hy (power flowing in +x direction)
        return -np.sum(Ez * Hy) * dx
    else:
        # Sy = Ez * Hx (power flowing in +y direction)
        return np.sum(Ez * Hx) * dx


# =============================================================================
# Analytical Formulas
# =============================================================================
def analytical_dipole_power_2d(omega, I0):
    """Analytical radiated power from 2D dipole (line source).

    For a 2D line source, the radiated power per unit length scales as:
    P ~ omega^2 * I0^2 / (4 * pi * eps0 * c^3)

    This is an approximation - exact 2D formula involves Hankel functions.

    Args:
        omega: Angular frequency (rad/s)
        I0: Current amplitude (integrated over source)

    Returns:
        Approximate radiated power
    """
    return (omega**2 * I0**2) / (4 * np.pi * EPS_0 * LIGHT_SPEED**3)


def analytical_cavity_frequency(m, L, n=1.0):
    """Analytical resonance frequency for 1D cavity.

    f_m = m * c / (2 * n * L)

    Args:
        m: Mode number (1, 2, 3, ...)
        L: Cavity length
        n: Refractive index inside cavity

    Returns:
        Resonance frequency (Hz)
    """
    return m * LIGHT_SPEED / (2 * n * L)
