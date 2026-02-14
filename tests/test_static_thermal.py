import numpy as np

from beamz import (
    Design,
    Material,
    StaticThermalConfig,
    StaticThermalResult,
    ThermalScenario,
    ThermalSource,
)


def test_static_thermal_heater_mask_increases_temperature():
    design = Design(width=3.0, height=3.0, material=Material(permittivity=1.0, k=1.0))
    params = StaticThermalConfig(max_iters=200, tol=1e-6)

    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    scenario = ThermalScenario(
        extrusion_depth_m=1.0,
        sources=[ThermalSource(region=mask, power_density_w_m3=10.0)],
    )

    result = design.solve_thermal(
        resolution=1.0,
        scenario=scenario,
        config=params,
    )
    assert isinstance(result, StaticThermalResult)
    eps_r, T = result.permittivity, result.temperature

    assert T[1, 1] > T[0, 0]
    assert np.max(T) > 300.0
