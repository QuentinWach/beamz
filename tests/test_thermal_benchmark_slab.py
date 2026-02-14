import numpy as np

from beamz import Design, Material, ThermalConfig


def _analytical_slab_with_generation(x, k, q_vol, t_left, t_ambient, h, length):
    """Analytical 1D solution for k*T'' + q = 0 with:
    T(0)=t_left and -k*T'(L)=h*(T(L)-t_ambient).
    """
    c1 = (
        q_vol * length
        - h * ((-q_vol * length * length) / (2.0 * k) + (t_left - t_ambient))
    ) / (k + h * length)
    return -q_vol / (2.0 * k) * x * x + c1 * x + t_left


def test_static_slab_robin_matches_analytical_profile():
    # 1D-in-2D slab benchmark:
    # uniform material, uniform heating, left Dirichlet, right Robin.
    nx, ny = 81, 9
    dx = 1.0
    k = 5.0
    q_vol = 1.2
    t_left = 300.0
    t_ambient = 295.0
    h = 0.25

    design = Design(
        width=float(nx),
        height=float(ny),
        material=Material(permittivity=1.0, k=k, rho=1.0, cp=1.0, T0=t_left),
    )

    heater_mask = np.ones((ny, nx), dtype=bool)
    fixed_mask = np.zeros((ny, nx), dtype=bool)
    fixed_mask[:, 0] = True
    fixed_values = np.full((ny, nx), t_left, dtype=float)

    config = ThermalConfig(
        thermal_dt=1.0,
        tau_avg=1.0,
        max_iters=20000,
        tol=1e-8,
        robin_h=h,
        robin_T_ambient=t_ambient,
        robin_sides=("right",),
    )
    result = design.solve_static_thermal(
        resolution=dx,
        config=config,
        heater_mask=heater_mask,
        heater_power=q_vol,
        fixed_temp_mask=fixed_mask,
        fixed_temp_value=fixed_values,
    )

    t_num = np.asarray(result.temperature).mean(axis=0)
    x = np.arange(nx, dtype=float) * dx
    length = (nx - 1) * dx
    t_ref = _analytical_slab_with_generation(
        x=x,
        k=k,
        q_vol=q_vol,
        t_left=t_left,
        t_ambient=t_ambient,
        h=h,
        length=length,
    )

    max_abs_err = float(np.max(np.abs(t_num - t_ref)))
    span = max(float(np.max(t_ref) - np.min(t_ref)), 1e-12)
    rel_err = max_abs_err / span

    # Tight check: solver should stay within ~1% of analytical span.
    assert rel_err < 0.01, f"relative error {rel_err:.4f} exceeds tolerance"
