"""Shared utility functions for BeamZ FDTD physics validation tests."""

from dataclasses import fields as dataclass_fields
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, yv

from beamz import EPS_0, LIGHT_SPEED, MU_0, um
from beamz.design.discretization import MaterialGrid
from beamz.devices._boundary_compile import lower_boundaries
from beamz.lattice import component_shapes
from beamz.simulation.compile import _compile_grid
from beamz.simulation.model import DomainSpec, RunSpec, SimulationRequest

# =============================================================================
# Constants
# =============================================================================
TEST_WAVELENGTH = 1.0 * um  # Standard test wavelength (1 micron)
TEST_FREQUENCY = LIGHT_SPEED / TEST_WAVELENGTH


def compiled_grid(
    permittivity,
    conductivity,
    permeability,
    resolution,
    plane_2d="xy",
):
    """Build a compiled lattice directly from test material arrays."""
    material_grid = MaterialGrid(
        permittivity,
        conductivity,
        permeability,
        resolution,
        np.asarray(permittivity).shape,
    )
    shape = material_grid.shape
    is_3d = len(shape) == 3
    size = (
        float(shape[-1]) * resolution,
        float(shape[-2]) * resolution,
        float(shape[0]) * resolution if is_3d else 0.0,
    )
    request = SimulationRequest(
        RunSpec(1.0, 1, 1, 0.0, "scan", False, (False, "auto", None, None)),
        DomainSpec(size, is_3d, plane_2d, (0.0, 0.0, 0.0)),
        material_grid,
        (),
        (),
        (),
    )
    boundary_data = lower_boundaries(
        material_grid, component_shapes(shape), (), size, 1.0
    )
    grid = _compile_grid(request, boundary_data)
    # Boundary-lowering unit tests need a mutable setup object; production compilation
    # publishes the frozen CompiledGrid only after this setup phase is complete.
    return SimpleNamespace(
        **{field.name: getattr(grid, field.name) for field in dataclass_fields(grid)}
    )


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


def compute_tm_field_energy(ez, hx, hy, dx, *, eps=1.0, mu=1.0):
    """Compute total TM electromagnetic energy on native Yee supports.

    The electric and magnetic samples are integrated separately because their
    complete-Yee arrays have different shapes. This is suitable for relative
    decay and stability measurements in homogeneous compact test domains.
    """
    eps_r = float(eps)
    mu_r = float(mu)
    electric = 0.5 * EPS_0 * eps_r * np.sum(np.asarray(ez) ** 2)
    magnetic = (
        0.5 * MU_0 * mu_r * (np.sum(np.asarray(hx) ** 2) + np.sum(np.asarray(hy) ** 2))
    )
    return float((electric + magnetic) * dx * dx)


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
# =============================================================================
# Analytical Formulas
# =============================================================================
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


# =============================================================================
# Mie Theory - 3D Sphere
# =============================================================================
def _riccati_bessel_psi(n, z):
    """Riccati-Bessel function ψ_n(z) = z * j_n(z)."""
    return z * np.sqrt(np.pi / (2 * z)) * jv(n + 0.5, z)


def _riccati_bessel_zeta(n, z):
    """Riccati-Bessel function ζ_n(z) = z * h_n^(1)(z)."""
    psi = _riccati_bessel_psi(n, z)
    chi = -z * np.sqrt(np.pi / (2 * z)) * yv(n + 0.5, z)
    return psi + 1j * chi


def _riccati_bessel_psi_prime(n, z):
    """Derivative of Riccati-Bessel ψ_n(z)."""
    return (n + 1) * np.sqrt(np.pi / (2 * z)) * jv(n + 0.5, z) - z * np.sqrt(
        np.pi / (2 * z)
    ) * jv(n + 1.5, z)


def _riccati_bessel_zeta_prime(n, z):
    """Derivative of Riccati-Bessel ζ_n(z)."""
    j_term = ((n + 1) * jv(n + 0.5, z) - z * jv(n + 1.5, z)) * np.sqrt(np.pi / (2 * z))
    y_term = ((n + 1) * yv(n + 0.5, z) - z * yv(n + 1.5, z)) * np.sqrt(np.pi / (2 * z))
    return j_term + 1j * y_term


def mie_coefficients_3d(x, m, n_max):
    """Compute Mie scattering coefficients a_n and b_n for a sphere.

    Args:
        x: Size parameter x = k * radius = 2π * n_medium * radius / wavelength
        m: Relative refractive index m = n_sphere / n_medium
        n_max: Maximum multipole order

    Returns:
        Tuple (a_n, b_n) arrays of complex coefficients
    """
    an, bn = [], []
    for n in range(1, n_max + 1):
        psi_x = _riccati_bessel_psi(n, x)
        psi_mx = _riccati_bessel_psi(n, m * x)
        zeta_x = _riccati_bessel_zeta(n, x)

        psi_x_prime = _riccati_bessel_psi_prime(n, x)
        psi_mx_prime = _riccati_bessel_psi_prime(n, m * x)
        zeta_x_prime = _riccati_bessel_zeta_prime(n, x)

        # Mie coefficients (Bohren & Huffman conventions)
        an_num = m * psi_mx * psi_x_prime - psi_x * psi_mx_prime
        an_den = m * psi_mx * zeta_x_prime - zeta_x * psi_mx_prime
        an.append(an_num / an_den)

        bn_num = psi_mx * psi_x_prime - m * psi_x * psi_mx_prime
        bn_den = psi_mx * zeta_x_prime - m * zeta_x * psi_mx_prime
        bn.append(bn_num / bn_den)

    return np.array(an), np.array(bn)


def mie_qext_3d(radius, wavelength, n_sphere, n_medium=1.0):
    """Compute extinction efficiency Q_ext for a dielectric sphere.

    Q_ext = C_ext / (π * r²) where C_ext is the extinction cross section.

    Args:
        radius: Sphere radius (same units as wavelength)
        wavelength: Free-space wavelength
        n_sphere: Refractive index of sphere
        n_medium: Refractive index of surrounding medium

    Returns:
        Extinction efficiency Q_ext (dimensionless)
    """
    k = 2 * np.pi * n_medium / wavelength
    m = n_sphere / n_medium
    x = k * radius

    # Number of terms (Wiscombe criterion)
    n_max = int(round(x + 4 * x ** (1 / 3) + 2))
    n_max = max(n_max, 3)

    an, bn = mie_coefficients_3d(x, m, n_max)
    n_arr = np.arange(1, n_max + 1)

    return (2 / x**2) * np.sum((2 * n_arr + 1) * np.real(an + bn))


def mie_qsca_3d(radius, wavelength, n_sphere, n_medium=1.0):
    """Compute scattering efficiency Q_sca for a dielectric sphere.

    Args:
        radius: Sphere radius
        wavelength: Free-space wavelength
        n_sphere: Refractive index of sphere
        n_medium: Refractive index of surrounding medium

    Returns:
        Scattering efficiency Q_sca (dimensionless)
    """
    k = 2 * np.pi * n_medium / wavelength
    m = n_sphere / n_medium
    x = k * radius

    n_max = int(round(x + 4 * x ** (1 / 3) + 2))
    n_max = max(n_max, 3)

    an, bn = mie_coefficients_3d(x, m, n_max)
    n_arr = np.arange(1, n_max + 1)

    return (2 / x**2) * np.sum((2 * n_arr + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2))


# =============================================================================
# Mie Theory - 2D Cylinder (TM polarization)
# =============================================================================
def mie_coefficients_2d(x, m, n_max):
    """Compute Mie scattering coefficients for 2D cylinder (TM polarization).

    For TM polarization (E parallel to cylinder axis), the scattering
    coefficients b_n are computed using Bessel functions.

    Args:
        x: Size parameter x = k * radius
        m: Relative refractive index m = n_cyl / n_medium
        n_max: Maximum multipole order

    Returns:
        Array of complex coefficients b_n for n = 0, 1, ..., n_max
    """
    bn = []
    mx = m * x

    for n in range(n_max + 1):
        # TM coefficients (Ez polarization)
        # b_n = [m*J_n(mx)*J'_n(x) - J_n(x)*J'_n(mx)] / [m*J_n(mx)*H'_n(x) - H_n(x)*J'_n(mx)]
        jn_x = jv(n, x)
        jn_mx = jv(n, mx)

        # Derivatives using recurrence: J'_n(z) = J_{n-1}(z) - n/z * J_n(z)
        if n == 0:
            jn_x_prime = -jv(1, x)
            jn_mx_prime = -jv(1, mx)
        else:
            jn_x_prime = jv(n - 1, x) - n / x * jn_x
            jn_mx_prime = jv(n - 1, mx) - n / mx * jn_mx

        # Hankel function H_n^(1) = J_n + i*Y_n
        hn_x = jn_x + 1j * yv(n, x)
        yn_x = yv(n, x)
        yn_x_prime = -yv(1, x) if n == 0 else yv(n - 1, x) - n / x * yn_x
        hn_x_prime = jn_x_prime + 1j * yn_x_prime

        num = m * jn_mx * jn_x_prime - jn_x * jn_mx_prime
        den = m * jn_mx * hn_x_prime - hn_x * jn_mx_prime

        bn.append(num / den)

    return np.array(bn)


def mie_qext_2d(radius, wavelength, n_cylinder, n_medium=1.0):
    """Compute extinction efficiency Q_ext for 2D dielectric cylinder.

    For a 2D cylinder, Q_ext = C_ext / (2r) where C_ext is the extinction
    width (cross section per unit length).

    Q_ext = (2/x) * Re[b_0 + 2*sum_{n=1}^{n_max} b_n]

    Args:
        radius: Cylinder radius (same units as wavelength)
        wavelength: Free-space wavelength
        n_cylinder: Refractive index of cylinder
        n_medium: Refractive index of surrounding medium

    Returns:
        Extinction efficiency Q_ext (dimensionless)
    """
    k = 2 * np.pi * n_medium / wavelength
    m = n_cylinder / n_medium
    x = k * radius

    # Number of terms
    n_max = int(round(x + 4 * x ** (1 / 3) + 10))
    n_max = max(n_max, 5)

    bn = mie_coefficients_2d(x, m, n_max)

    # Q_ext = (2/x) * Re[b_0 + 2*sum(b_n for n>=1)]
    return (2 / x) * np.real(bn[0] + 2 * np.sum(bn[1:]))


def mie_qsca_2d(radius, wavelength, n_cylinder, n_medium=1.0):
    """Compute scattering efficiency Q_sca for 2D dielectric cylinder.

    Q_sca = (2/x) * [|b_0|^2 + 2*sum_{n=1}^{n_max} |b_n|^2]

    Args:
        radius: Cylinder radius
        wavelength: Free-space wavelength
        n_cylinder: Refractive index of cylinder
        n_medium: Refractive index of surrounding medium

    Returns:
        Scattering efficiency Q_sca (dimensionless)
    """
    k = 2 * np.pi * n_medium / wavelength
    m = n_cylinder / n_medium
    x = k * radius

    n_max = int(round(x + 4 * x ** (1 / 3) + 10))
    n_max = max(n_max, 5)

    bn = mie_coefficients_2d(x, m, n_max)

    return (2 / x) * (np.abs(bn[0]) ** 2 + 2 * np.sum(np.abs(bn[1:]) ** 2))


# =============================================================================
# Waveguide Dispersion Relations
# =============================================================================
def slab_waveguide_neff_te(n_core, n_clad, width, wavelength, mode=0):
    """Solve for effective index of symmetric slab waveguide (TE modes).

    Solves the transcendental equation:
    tan(k_t * d/2) = γ / k_t

    where:
    - k_t = k_0 * sqrt(n_core² - n_eff²)  (transverse wavevector in core)
    - γ = k_0 * sqrt(n_eff² - n_clad²)    (decay constant in cladding)
    - k_0 = 2π / λ

    Args:
        n_core: Core refractive index
        n_clad: Cladding refractive index
        width: Core width (same units as wavelength)
        wavelength: Free-space wavelength
        mode: Mode number (0 = fundamental)

    Returns:
        Effective index n_eff, or None if mode doesn't exist
    """
    k0 = 2 * np.pi / wavelength
    d = width

    # Normalized frequency V
    V = k0 * d / 2 * np.sqrt(n_core**2 - n_clad**2)

    # Mode cutoff: V must be large enough
    if mode * np.pi / 2 > V:
        return None

    def dispersion_eq(neff):
        if neff <= n_clad or neff >= n_core:
            return 1e10
        kt = k0 * np.sqrt(n_core**2 - neff**2)
        gamma = k0 * np.sqrt(neff**2 - n_clad**2)
        # TE: tan(kt*d/2) = gamma/kt
        lhs = np.tan(kt * d / 2)
        rhs = gamma / kt
        return lhs - rhs

    # Search in valid range
    n_min = n_clad + 1e-10
    n_max = n_core - 1e-10

    # Use bisection with careful bracketing
    try:
        # Find bracket by scanning
        n_points = np.linspace(n_min, n_max, 200)
        vals = [dispersion_eq(n) for n in n_points]

        # Find zero crossings
        crossings = []
        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] < 0:
                crossings.append((n_points[i], n_points[i + 1]))

        if mode >= len(crossings):
            return None

        # Get the mode-th crossing (counting from highest neff)
        crossings = sorted(crossings, key=lambda x: -x[0])
        bracket = crossings[mode]

        neff = brentq(dispersion_eq, bracket[0], bracket[1])
        return neff
    except (ValueError, IndexError):
        return None


def slab_waveguide_neff_tm(n_core, n_clad, width, wavelength, mode=0):
    """Solve for effective index of symmetric slab waveguide (TM modes).

    Solves the transcendental equation:
    tan(k_t * d/2) = (n_core² / n_clad²) * γ / k_t

    Args:
        n_core: Core refractive index
        n_clad: Cladding refractive index
        width: Core width (same units as wavelength)
        wavelength: Free-space wavelength
        mode: Mode number (0 = fundamental)

    Returns:
        Effective index n_eff, or None if mode doesn't exist
    """
    k0 = 2 * np.pi / wavelength
    d = width

    V = k0 * d / 2 * np.sqrt(n_core**2 - n_clad**2)
    if mode * np.pi / 2 > V:
        return None

    def dispersion_eq(neff):
        if neff <= n_clad or neff >= n_core:
            return 1e10
        kt = k0 * np.sqrt(n_core**2 - neff**2)
        gamma = k0 * np.sqrt(neff**2 - n_clad**2)
        # TM: tan(kt*d/2) = (n_core/n_clad)² * gamma/kt
        lhs = np.tan(kt * d / 2)
        rhs = (n_core / n_clad) ** 2 * gamma / kt
        return lhs - rhs

    n_min = n_clad + 1e-10
    n_max = n_core - 1e-10

    try:
        n_points = np.linspace(n_min, n_max, 200)
        vals = [dispersion_eq(n) for n in n_points]

        crossings = []
        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] < 0:
                crossings.append((n_points[i], n_points[i + 1]))

        if mode >= len(crossings):
            return None

        crossings = sorted(crossings, key=lambda x: -x[0])
        bracket = crossings[mode]

        neff = brentq(dispersion_eq, bracket[0], bracket[1])
        return neff
    except (ValueError, IndexError):
        return None


# =============================================================================
# Fabry-Pérot Cavity Functions
# =============================================================================
def fabry_perot_fsr(L, n=1.0):
    """Free spectral range of Fabry-Pérot cavity.

    FSR = c / (2 * n * L)

    Args:
        L: Cavity length
        n: Refractive index inside cavity

    Returns:
        Free spectral range (Hz)
    """
    return LIGHT_SPEED / (2 * n * L)


def fabry_perot_q_factor(L, n, R1, R2):
    """Q-factor of Fabry-Pérot cavity.

    Q = F * m where F is finesse and m is mode number.
    For fundamental mode at resonance: Q ≈ 2πnL / (λ * (1-R))

    Args:
        L: Cavity length
        n: Refractive index inside cavity
        R1, R2: Mirror reflectivities

    Returns:
        Q-factor for fundamental mode
    """
    # At resonance, m ~ 2nL/λ, so Q ~ F * 2nL/λ
    # But we can estimate Q directly from photon lifetime
    r = np.sqrt(R1 * R2)
    tau_rt = 2 * n * L / LIGHT_SPEED  # round-trip time
    # Amplitude decays by r per round trip, so intensity by r²
    # Energy decay: E(t) = E0 * exp(-t/τ) where τ = -τ_rt / (2*ln(r))
    if r >= 1:
        return np.inf
    tau = -tau_rt / (2 * np.log(r))
    omega = 2 * np.pi * LIGHT_SPEED / (2 * n * L)  # fundamental
    return omega * tau / 2


# =============================================================================
# DFT and Measurement Helpers
# =============================================================================
def measure_resonance_frequency(field_history, time_array, freq_range=None):
    """Find resonance frequency from FFT of field time series.

    Args:
        field_history: Field values at a point over time
        time_array: Time values
        freq_range: Optional (f_min, f_max) to search within

    Returns:
        Peak frequency (Hz)
    """
    field_arr = np.asarray(field_history).flatten()
    time_arr = np.asarray(time_array)

    dt = time_arr[1] - time_arr[0]
    n = len(field_arr)

    # FFT
    spectrum = np.abs(np.fft.rfft(field_arr))
    freqs = np.fft.rfftfreq(n, dt)

    # Apply frequency range filter
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if not np.any(mask):
            mask = np.ones_like(freqs, dtype=bool)
    else:
        mask = freqs > 0  # Exclude DC

    # Find peak
    masked_spectrum = spectrum.copy()
    masked_spectrum[~mask] = 0
    peak_idx = np.argmax(masked_spectrum)

    return freqs[peak_idx]


# =============================================================================
# Quantitative Validation Helpers
# =============================================================================
def fit_exponential_decay(data, time, start_fraction=0.0, end_fraction=0.8):
    """Fit exponential decay A * exp(-t/tau) to data.

    Args:
        data: 1D array of values (e.g., field envelope)
        time: 1D array of time values
        start_fraction: Fraction of data to skip at start (0-1)
        end_fraction: Fraction of data to use (0-1)

    Returns:
        Tuple (tau, amplitude, r_squared) or (None, None, None) if fit fails
    """
    data = np.asarray(data).flatten()
    time = np.asarray(time).flatten()

    n = len(data)
    start_idx = int(start_fraction * n)
    end_idx = int(end_fraction * n)

    if end_idx - start_idx < 5:
        return None, None, None

    t_fit = time[start_idx:end_idx] - time[start_idx]
    d_fit = data[start_idx:end_idx]

    # Filter positive values only
    valid = d_fit > 1e-20
    if np.sum(valid) < 5:
        return None, None, None

    t_fit = t_fit[valid]
    d_fit = d_fit[valid]

    try:
        # Linear fit to log(data)
        log_d = np.log(d_fit)
        coeffs = np.polyfit(t_fit, log_d, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        tau = -1.0 / slope if slope < 0 else None
        amplitude = np.exp(intercept)

        # Compute R²
        fitted = intercept + slope * t_fit
        ss_res = np.sum((log_d - fitted) ** 2)
        ss_tot = np.sum((log_d - np.mean(log_d)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return tau, amplitude, r_squared
    except (ValueError, np.linalg.LinAlgError):
        return None, None, None
