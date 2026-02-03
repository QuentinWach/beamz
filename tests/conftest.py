"""Shared fixtures and utilities for BeamZ FDTD physics validation tests."""

import numpy as np
import pytest

from beamz import (
    EPS_0,
    LIGHT_SPEED,
    MU_0,
    PML,
    Design,
    GaussianSource,
    Material,
    Monitor,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)

# =============================================================================
# Constants (exposed via fixtures)
# =============================================================================
TEST_WAVELENGTH = 1.0 * um  # Standard test wavelength (1 micron)
TEST_FREQUENCY = LIGHT_SPEED / TEST_WAVELENGTH


# =============================================================================
# Basic Fixtures
# =============================================================================
@pytest.fixture
def test_wavelength():
    """Standard wavelength for physics tests (1 um)."""
    return TEST_WAVELENGTH


@pytest.fixture
def test_frequency():
    """Standard frequency for physics tests."""
    return TEST_FREQUENCY


# =============================================================================
# Domain Fixtures
# =============================================================================
@pytest.fixture
def vacuum_domain_small():
    """Small vacuum domain for quick propagation tests.

    Size: 10 x 10 wavelengths
    Target runtime: < 3 seconds
    """
    wavelength = TEST_WAVELENGTH
    n_max = 1.0
    domain_size = 10 * wavelength

    dx, dt = calc_optimal_fdtd_params(
        wavelength, n_max, dims=2, safety_factor=0.95, points_per_wavelength=10
    )

    design = Design(
        width=domain_size, height=domain_size, material=Material(permittivity=1.0)
    )

    return {
        "design": design,
        "wavelength": wavelength,
        "domain_size": domain_size,
        "dx": dx,
        "dt": dt,
        "n": n_max,
    }


@pytest.fixture
def dielectric_domain():
    """Uniform dielectric domain for material tests.

    n = 1.5 (glass-like)
    """
    wavelength = TEST_WAVELENGTH
    n_material = 1.5
    domain_size = 10 * wavelength

    dx, dt = calc_optimal_fdtd_params(
        wavelength, n_material, dims=2, safety_factor=0.95, points_per_wavelength=12
    )

    design = Design(
        width=domain_size,
        height=domain_size,
        material=Material(permittivity=n_material**2),
    )

    return {
        "design": design,
        "wavelength": wavelength,
        "domain_size": domain_size,
        "dx": dx,
        "dt": dt,
        "n": n_material,
    }


@pytest.fixture
def dielectric_interface_domain():
    """Domain with vacuum/dielectric interface for Fresnel tests.

    Left half: vacuum (n=1)
    Right half: dielectric (n=1.5)
    """
    wavelength = TEST_WAVELENGTH
    n1 = 1.0  # vacuum
    n2 = 1.5  # dielectric

    domain_width = 20 * wavelength
    domain_height = 10 * wavelength

    dx, dt = calc_optimal_fdtd_params(
        wavelength, n2, dims=2, safety_factor=0.95, points_per_wavelength=12
    )

    # Background is vacuum
    design = Design(
        width=domain_width, height=domain_height, material=Material(permittivity=n1**2)
    )

    # Add dielectric in right half
    interface_x = domain_width / 2
    design += Rectangle(
        position=(interface_x + domain_width / 4, domain_height / 2),
        width=domain_width / 2,
        height=domain_height,
        material=Material(permittivity=n2**2),
    )

    return {
        "design": design,
        "wavelength": wavelength,
        "domain_width": domain_width,
        "domain_height": domain_height,
        "interface_x": interface_x,
        "n1": n1,
        "n2": n2,
        "dx": dx,
        "dt": dt,
    }


@pytest.fixture
def waveguide_domain():
    """Slab waveguide domain for ModeSource tests.

    Horizontal waveguide: core n=2.0, cladding n=1.0
    Core width: 0.5 wavelengths (single-mode regime)
    """
    wavelength = TEST_WAVELENGTH
    n_core = 2.0
    n_clad = 1.0
    core_width = 0.5 * wavelength

    domain_width = 15 * wavelength
    domain_height = 6 * wavelength

    dx, dt = calc_optimal_fdtd_params(
        wavelength, n_core, dims=2, safety_factor=0.95, points_per_wavelength=15
    )

    # Background is cladding
    design = Design(
        width=domain_width,
        height=domain_height,
        material=Material(permittivity=n_clad**2),
    )

    # Add waveguide core (horizontal stripe in center)
    design += Rectangle(
        position=(domain_width / 2, domain_height / 2),
        width=domain_width,
        height=core_width,
        material=Material(permittivity=n_core**2),
    )

    return {
        "design": design,
        "wavelength": wavelength,
        "domain_width": domain_width,
        "domain_height": domain_height,
        "core_width": core_width,
        "n_core": n_core,
        "n_clad": n_clad,
        "dx": dx,
        "dt": dt,
    }
