import numpy as np

from beamz import Design, Material, ThermalParams, apply_static_thermal


def test_static_thermal_heater_mask_increases_temperature():
    design = Design(width=3.0, height=3.0, material=Material(permittivity=1.0, k=1.0))
    params = ThermalParams(
        thermal_dt=1.0, tau_avg=1.0, steady_state=True, max_iters=200, tol=1e-6
    )

    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    eps_r, T = apply_static_thermal(
        design,
        resolution=1.0,
        params=params,
        heater_mask=mask,
        heater_power=10.0,
    )

    assert T[1, 1] > T[0, 0]
    assert np.max(T) > 300.0
