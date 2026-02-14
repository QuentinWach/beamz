import numpy as np

from beamz import (
    Design,
    Material,
    StaticThermalConfig,
    ThermalScenario,
    ThermalSource,
)


def test_static_thermo_optic_increases_permittivity():
    material = Material(permittivity=4.0, k=1.0, dn_dT=1e-3, T0=300.0)
    design = Design(width=2.0, height=2.0, material=material)

    params = StaticThermalConfig(max_iters=200, tol=1e-6)
    mask = np.ones((2, 2), dtype=bool)
    scenario = ThermalScenario(
        extrusion_depth_m=1.0,
        sources=[ThermalSource(region=mask, power_density_w_m3=5.0)],
    )

    result = design.solve_thermal(
        resolution=1.0,
        scenario=scenario,
        config=params,
    )
    eps_r, _T = result.permittivity, result.temperature

    assert np.max(eps_r) > 4.0
