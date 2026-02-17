"""Dispersive material utilities and ADE update kernels."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0
from beamz.simulation import ops

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class PoleSpec:
    pole_hz: complex
    residue: complex


@dataclass(frozen=True)
class DispersiveGridSpec:
    eps_inf: np.ndarray
    conductivity: np.ndarray
    pole_a: np.ndarray
    pole_c: np.ndarray
    mask: np.ndarray


@dataclass
class ADEState:
    psi_x: jnp.ndarray | None
    psi_y: jnp.ndarray | None
    psi_z: jnp.ndarray | None


def evaluate_epsilon_from_poles(
    frequency_hz: np.ndarray | float,
    eps_inf: float,
    poles: tuple[tuple[complex, complex], ...],
) -> np.ndarray:
    f = np.asarray(frequency_hz, dtype=float)
    eps = np.full_like(f, complex(eps_inf), dtype=complex)
    for pole, residue in poles:
        eps = eps + complex(residue) / (1j * f + complex(pole))
    return eps


def slice_poles_for_component_2d(
    pole_grid: jnp.ndarray,
    component: str,
    plane_2d: str,
) -> jnp.ndarray:
    """Slice pole tensors to match a 2D E-component Yee shape."""
    if pole_grid.shape[0] == 0:
        return pole_grid
    zeros = jnp.zeros_like(pole_grid[0], dtype=float)
    slices = []
    for p in range(int(pole_grid.shape[0])):
        sl, _, _ = ops.material_slice_for_e_2d_component(
            pole_grid[p], zeros, component, plane_2d
        )
        slices.append(sl)
    return jnp.stack(slices, axis=0)


def slice_poles_for_component_3d(
    pole_grid: jnp.ndarray,
    component: str,
) -> jnp.ndarray:
    """Slice pole tensors to match a 3D E-component Yee shape."""
    if pole_grid.shape[0] == 0:
        return pole_grid
    slices = []
    for p in range(int(pole_grid.shape[0])):
        sl, _, _ = ops.material_slice_for_e_3d(pole_grid[p], pole_grid[p], component)
        slices.append(sl)
    return jnp.stack(slices, axis=0)


def _cn_update_region(
    field_region: jnp.ndarray,
    curl_region: jnp.ndarray,
    conductivity: jnp.ndarray,
    permittivity: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    denom = 1.0 + conductivity * (dt / (2.0 * EPS_0 * permittivity))
    factor = (1.0 - conductivity * (dt / (2.0 * EPS_0 * permittivity))) / denom
    source = (dt / (EPS_0 * permittivity)) / denom
    return field_region * factor + source * curl_region


def advance_e_field_dispersive(
    field: jnp.ndarray,
    curl: jnp.ndarray,
    conductivity: jnp.ndarray,
    permittivity: jnp.ndarray,
    dt: float,
    region: tuple[slice, ...],
    psi_state: jnp.ndarray | None,
    pole_a: jnp.ndarray | None,
    pole_c: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Advance one E-component with CN + ADE correction."""
    field_region = field[region]
    curl_region = curl[region]

    if (
        psi_state is None
        or pole_a is None
        or pole_c is None
        or int(pole_a.shape[0]) == 0
    ):
        updated = _cn_update_region(field_region, curl_region, conductivity, permittivity, dt)
        return field.at[region].set(updated), psi_state

    # Canonical ADE ODE in Hz-domain:
    #   dpsi/dt = -2*pi*pole_a*psi + 2*pi*pole_c*E
    alpha = TWO_PI * pole_a
    exp_term = jnp.exp(-alpha * dt)
    small = jnp.abs(alpha) < 1e-14
    alpha_safe = jnp.where(small, 1.0 + 0.0j, alpha)
    drive = jnp.where(
        small,
        TWO_PI * pole_c * dt,
        (1.0 - exp_term) * pole_c / alpha_safe,
    )

    psi_new = exp_term * psi_state + drive * field_region[None, ...]
    # D = eps0*(eps_inf*E + Re(sum psi)); polarization current term enters as -eps0*d/dt(Re(sum psi)).
    dpsi_dt = jnp.real(jnp.sum((psi_new - psi_state) / dt, axis=0))
    curl_eff = curl_region - EPS_0 * dpsi_dt
    updated = _cn_update_region(field_region, curl_eff, conductivity, permittivity, dt)
    return field.at[region].set(updated), psi_new

