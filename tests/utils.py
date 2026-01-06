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
