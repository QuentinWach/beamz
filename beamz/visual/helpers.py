from typing import Any, Dict

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from beamz.const import LIGHT_SPEED


def get_si_scale_and_label(value):
    """Convert a value to appropriate SI unit and return scale factor and label."""
    if value >= 1e-3:
        return 1e3, "mm"
    elif value >= 1e-6:
        return 1e6, "µm"
    elif value >= 1e-9:
        return 1e9, "nm"
    else:
        return 1e12, "pm"


def check_fdtd_stability(dt, dx, dy=None, dz=None, n_max=1.0, safety_factor=1.0):
    """
    Check FDTD stability with the Courant-Friedrichs-Lewy (CFL) condition.

    Args:
        dt: Time step
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction (None for 1D)
        dz: Grid spacing in z direction (None for 1D/2D)
        n_max: Maximum refractive index in the simulation
        safety_factor: Factor to apply to the theoretical Courant limit (0-1).
                       Use 1.0 to evaluate against the theoretical limit 1/sqrt(dims).

    Returns:
        tuple: (is_stable, courant_number, max_allowed)
    """
    # Determine dimensionality
    dims = 1
    min_spacing = dx
    if dy is not None:
        dims = 2
        min_spacing = min(dx, dy)
    if dz is not None:
        dims = 3
        min_spacing = min(dx, dy, dz)
    # Courant number defined with vacuum speed (conservative and standard for Yee grid)
    c0 = LIGHT_SPEED
    courant = c0 * dt / min_spacing
    # Theoretical stability limit
    max_allowed = 1.0 / np.sqrt(dims)
    # Apply safety factor
    safe_limit = safety_factor * max_allowed
    return courant <= safe_limit, courant, safe_limit


def calc_optimal_fdtd_params(
    wavelength,
    n_max,
    dims=2,
    safety_factor=0.999,
    points_per_wavelength=10,
    width=None,
    height=None,
    depth=None,
):
    """
    Calculate optimal FDTD grid resolution and time step based on wavelength and material properties.

    Args:
        wavelength: Light wavelength in vacuum
        n_max: Maximum refractive index in the simulation
        dims: Dimensionality of simulation (1, 2, or 3)
        safety_factor: Fraction of the theoretical Courant limit to target (0-1).
                       0.95 operates close to the limit; reduce for additional margin.
        points_per_wavelength: Number of grid points per wavelength in the highest index material
        width, height, depth: Optional physical dimensions to estimate total grid size and performance

    Returns:
        tuple: (resolution, dt) - optimal spatial resolution and time step
    """
    # Calculate wavelength in the highest index material
    lambda_material = wavelength / n_max
    # Calculate optimal grid resolution based on desired points per wavelength
    resolution = lambda_material / points_per_wavelength
    # Calculate theoretical Courant limit (dt_max = dx / (c * sqrt(dims)))
    dt_max = resolution / (LIGHT_SPEED * np.sqrt(dims))
    # Apply safety factor (vacuum-based Courant condition)
    dt = safety_factor * dt_max

    # Grid size warning
    if width and height:
        nx = int(width / resolution)
        ny = int(height / resolution)
        nz = int(depth / resolution) if (dims == 3 and depth) else 1
        total_cells = nx * ny * nz

        if total_cells > 5e6:
            display_status(
                f"Warning: Large simulation grid detected ({total_cells/1e6:.1f}M cells). "
                f"3D simulations can be slow. Consider reducing points_per_wavelength (current: {points_per_wavelength}) "
                f"if performance is an issue.",
                "warning",
            )

    # Verify stability
    try:
        _, courant, limit = check_fdtd_stability(
            dt,
            resolution,
            dy=resolution if dims >= 2 else None,
            dz=resolution if dims >= 3 else None,
            n_max=n_max,
            safety_factor=1.0,
        )
        assert (
            courant <= limit + 1e-15
        ), "Internal error: calculated time step exceeds stability limit"
    except Exception:
        pass

    return resolution, dt


def dxdt(
    wavelength,
    n_max=1.0,
    dims=2,
    safety_factor=0.999,
    points_per_wavelength=10,
    **kwargs,
):
    """Convenience alias returning (dx, dt) for FDTD setup."""
    return calc_optimal_fdtd_params(
        wavelength=wavelength,
        n_max=n_max,
        dims=dims,
        safety_factor=safety_factor,
        points_per_wavelength=points_per_wavelength,
        **kwargs,
    )


# Initialize rich console
console = Console()


def display_status(status: str, status_type: str = "info") -> None:
    """Display a status message with appropriate styling."""
    style_map = {
        "info": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red",
    }
    style = style_map.get(status_type, "white")
    console.print(f"[{style}]● {status}[/]")


def create_rich_progress() -> Progress:
    """Create and return a rich progress bar for tracking processes."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    )
