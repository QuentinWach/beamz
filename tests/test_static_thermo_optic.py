import numpy as np

from beamz import Design, Material, ThermalParams, apply_static_thermal


def test_static_thermo_optic_increases_permittivity():
    material = Material(permittivity=4.0, k=1.0, dn_dT=1e-3, T0=300.0)
    design = Design(width=2.0, height=2.0, material=material)

    params = ThermalParams(thermal_dt=1.0, tau_avg=1.0, steady_state=True, max_iters=200, tol=1e-6)
    mask = np.ones((2, 2), dtype=bool)

    eps_r, T = apply_static_thermal(
        design,
        resolution=1.0,
        params=params,
        heater_mask=mask,
        heater_power=5.0,
    )

    assert np.max(eps_r) > 4.0
