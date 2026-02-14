import numpy as np

from beamz import (
    Design,
    Material,
    StaticThermalResult,
    ThermalConfig,
)


def test_static_thermal_heater_mask_increases_temperature():
    design = Design(width=3.0, height=3.0, material=Material(permittivity=1.0, k=1.0))
    params = ThermalConfig(thermal_dt=1.0, tau_avg=1.0, max_iters=200, tol=1e-6)

    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    result = design.solve_static_thermal(
        resolution=1.0,
        config=params,
        heater_mask=mask,
        heater_power=10.0,
    )
    assert isinstance(result, StaticThermalResult)
    eps_r, T = result.permittivity, result.temperature

    assert T[1, 1] > T[0, 0]
    assert np.max(T) > 300.0
