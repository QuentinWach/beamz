import numpy as np
import pytest

from beamz import (
    Design,
    Material,
    Rectangle,
    StaticThermalConfig,
    ThermalScenario,
    ThermalSink,
    ThermalSource,
)


def _build_small_design():
    design = Design(
        width=8.0, height=6.0, material=Material(permittivity=1.0, k=1.0, T0=300.0)
    )
    heater_a = Rectangle(
        position=(2.0, 3.0),
        width=2.0,
        height=1.0,
        material=Material(permittivity=1.0, k=2.0, T0=300.0),
    )
    heater_b = Rectangle(
        position=(5.0, 3.0),
        width=1.0,
        height=1.0,
        material=Material(permittivity=1.0, k=2.0, T0=300.0),
    )
    design += heater_a
    design += heater_b
    return design, heater_a, heater_b


def test_structure_region_matches_callable_region():
    design, heater_a, _heater_b = _build_small_design()
    config = StaticThermalConfig(max_iters=2000, tol=1e-7)

    sink = ThermalSink(region=lambda x, y, z: y <= 0.5, temperature_k=300.0)
    scenario_struct = ThermalScenario(
        extrusion_depth_m=1.0,
        sources=[ThermalSource(region=heater_a, power_density_w_m3=5.0)],
        sinks=[sink],
    )
    scenario_callable = ThermalScenario(
        extrusion_depth_m=1.0,
        sources=[
            ThermalSource(
                region=lambda x, y, z: 2.0 <= x <= 4.0 and 3.0 <= y <= 4.0,
                power_density_w_m3=5.0,
            )
        ],
        sinks=[sink],
    )

    t_struct = design.solve_thermal(
        1.0, scenario=scenario_struct, config=config
    ).temperature
    t_call = design.solve_thermal(
        1.0, scenario=scenario_callable, config=config
    ).temperature
    assert np.allclose(t_struct, t_call, atol=1e-6)


def test_iterable_structure_region_matches_union_callable():
    design, heater_a, heater_b = _build_small_design()
    config = StaticThermalConfig(max_iters=2000, tol=1e-7)
    sink = ThermalSink(region=lambda x, y, z: y <= 0.5, temperature_k=300.0)

    scenario_structs = ThermalScenario(
        extrusion_depth_m=1.0,
        sources=[ThermalSource(region=[heater_a, heater_b], power_density_w_m3=4.0)],
        sinks=[sink],
    )
    scenario_union = ThermalScenario(
        extrusion_depth_m=1.0,
        sources=[
            ThermalSource(
                region=lambda x, y, z: (2.0 <= x <= 4.0 and 3.0 <= y <= 4.0)
                or (5.0 <= x <= 6.0 and 3.0 <= y <= 4.0),
                power_density_w_m3=4.0,
            )
        ],
        sinks=[sink],
    )

    t_structs = design.solve_thermal(
        1.0, scenario=scenario_structs, config=config
    ).temperature
    t_union = design.solve_thermal(
        1.0, scenario=scenario_union, config=config
    ).temperature
    assert np.allclose(t_structs, t_union, atol=1e-6)


def test_power_w_requires_extrusion_depth_in_2d():
    design = Design(
        width=3.0, height=3.0, material=Material(permittivity=1.0, k=1.0, T0=300.0)
    )
    scenario = ThermalScenario(
        sources=[ThermalSource(region=np.ones((3, 3), dtype=bool), power_w=0.01)],
    )
    with pytest.raises(ValueError, match="extrusion_depth_m"):
        design.solve_thermal(
            resolution=1.0, scenario=scenario, config=StaticThermalConfig()
        )


def test_legacy_static_kwargs_raise_migration_hint():
    design = Design(
        width=3.0, height=3.0, material=Material(permittivity=1.0, k=1.0, T0=300.0)
    )
    with pytest.raises(ValueError, match="Static thermal API changed"):
        design.solve_static_thermal(
            resolution=1.0,
            config=StaticThermalConfig(),
            heater_mask=np.ones((3, 3), dtype=bool),
            heater_power=1.0,
        )
