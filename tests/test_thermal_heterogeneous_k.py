import jax.numpy as jnp
import numpy as np

from beamz import (
    Design,
    Material,
    Rectangle,
    Simulation,
    ThermalConfig,
    ThermalCoupling,
)


def _harmonic_mean(left, right):
    denom = left + right
    out = np.zeros_like(denom, dtype=float)
    valid = (left > 0) & (right > 0) & (denom > 0)
    np.divide(2.0 * left * right, denom, out=out, where=valid)
    return out


def _two_material_design(nx=40, ny=8, k_left=1.0, k_right=5.0):
    design = Design(
        width=float(nx),
        height=float(ny),
        material=Material(permittivity=1.0, k=k_left, rho=1.0, cp=1.0, T0=300.0),
    )
    design += Rectangle(
        position=(nx / 2.0, 0.0),
        width=nx / 2.0,
        height=float(ny),
        material=Material(permittivity=1.0, k=k_right, rho=1.0, cp=1.0, T0=300.0),
    )
    return design


def test_static_solver_preserves_flux_continuity_for_heterogeneous_k():
    nx, ny = 40, 8
    dx = 1.0
    design = _two_material_design(nx=nx, ny=ny, k_left=1.0, k_right=5.0)

    k_grid = design.get_thermal_grids(dx)[0]
    fixed_mask = np.zeros_like(k_grid, dtype=bool)
    fixed_mask[:, 0] = True
    fixed_mask[:, -1] = True

    fixed_values = np.full_like(k_grid, 300.0, dtype=float)
    fixed_values[:, 0] = 310.0
    fixed_values[:, -1] = 300.0

    params = ThermalConfig(thermal_dt=1.0, tau_avg=1.0, max_iters=8000, tol=1e-8)
    result = design.solve_static_thermal(
        resolution=dx,
        config=params,
        heater_mask=np.zeros_like(k_grid, dtype=bool),
        heater_power=0.0,
        fixed_temp_mask=fixed_mask,
        fixed_temp_value=fixed_values,
    )

    T = result.temperature
    mid = ny // 2
    assert np.all(np.diff(T[mid, :]) <= 1e-6)

    k_face = _harmonic_mean(k_grid[:, :-1], k_grid[:, 1:])
    flux_x = -k_face * (T[:, 1:] - T[:, :-1]) / dx
    flux_mid = flux_x[mid, 2:-2]
    flux_ref = float(np.mean(flux_mid))

    assert abs(flux_ref) > 1e-9
    relative_spread = float(np.std(flux_mid)) / abs(flux_ref)
    assert relative_spread < 0.12

    interface_face = nx // 2 - 1
    left_flux = float(np.mean(flux_x[:, interface_face - 1]))
    right_flux = float(np.mean(flux_x[:, interface_face + 1]))
    mismatch = abs(left_flux - right_flux) / max(abs(flux_ref), 1e-12)
    assert mismatch < 0.15


def test_transient_heterogeneous_k_keeps_balanced_profile_nearly_stationary():
    nx, ny = 40, 8
    dx = 1.0
    k_left, k_right = 1.0, 5.0
    design = _two_material_design(nx=nx, ny=ny, k_left=k_left, k_right=k_right)
    sim = Simulation(
        design=design,
        devices=[],
        boundaries=[],
        time=np.array([0.0, 1.0]),
        resolution=dx,
    )

    thermal = ThermalCoupling(ThermalConfig(thermal_dt=1.0, tau_avg=1.0, T0=300.0))
    thermal.initialize(sim)

    x = np.arange(nx, dtype=float) + 0.5
    interface_x = nx / 2.0
    q = 1.0
    left = 320.0 - q * x / k_left
    t_interface = 320.0 - q * interface_x / k_left
    right = t_interface - q * (x - interface_x) / k_right
    profile = np.where(x < interface_x, left, right)
    thermal.T = jnp.asarray(np.repeat(profile[None, :], ny, axis=0))

    sim.fields.Ez = jnp.zeros_like(sim.fields.Ez)
    T_before = np.asarray(thermal.T)
    thermal.step(sim)
    T_after = np.asarray(thermal.T)

    delta = np.abs(T_after - T_before)
    interface_col = nx // 2
    near_interface = delta[2:-2, interface_col - 3 : interface_col + 3]
    assert float(np.max(near_interface)) < 0.08
