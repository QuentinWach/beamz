import jax.numpy as jnp
import numpy as np

from beamz import (
    ConvectionBC,
    Design,
    Material,
    Simulation,
    StaticThermalConfig,
    ThermalConfig,
    ThermalCoupling,
    ThermalScenario,
    ThermalSink,
    ThermalSource,
)


def test_static_robin_bc_reduces_peak_temperature():
    design = Design(width=10.0, height=10.0, material=Material(permittivity=1.0, k=1.0))
    heater_mask = np.zeros((10, 10), dtype=bool)
    heater_mask[5, 5] = True
    fixed_mask = np.zeros((10, 10), dtype=bool)
    fixed_mask[0, :] = True

    base_cfg = StaticThermalConfig(
        max_iters=2000,
        tol=1e-7,
    )
    robin_cfg = StaticThermalConfig(
        max_iters=2000,
        tol=1e-7,
    )
    base_scenario = ThermalScenario(
        extrusion_depth_m=1.0,
        sources=[ThermalSource(region=heater_mask, power_density_w_m3=50.0)],
        sinks=[ThermalSink(region=fixed_mask, temperature_k=300.0)],
    )
    robin_scenario = ThermalScenario(
        extrusion_depth_m=1.0,
        sources=[ThermalSource(region=heater_mask, power_density_w_m3=50.0)],
        sinks=[ThermalSink(region=fixed_mask, temperature_k=300.0)],
        convection=ConvectionBC(h_w_m2_k=1.0, ambient_temp_k=300.0, sides=("top",)),
    )

    base = design.solve_thermal(
        resolution=1.0,
        scenario=base_scenario,
        config=base_cfg,
    )
    robin = design.solve_thermal(
        resolution=1.0,
        scenario=robin_scenario,
        config=robin_cfg,
    )

    assert float(robin.temperature.max()) < float(base.temperature.max())


def test_transient_robin_bc_cools_selected_boundary():
    material = Material(
        permittivity=1.0, conductivity=0.0, k=0.0, rho=1.0, cp=1.0, dn_dT=0.0, T0=300.0
    )
    design = Design(width=5.0, height=5.0, material=material)
    sim = Simulation(
        design=design,
        devices=[],
        boundaries=[],
        time=np.array([0.0, 1.0]),
        resolution=1.0,
    )

    thermal = ThermalCoupling(
        ThermalConfig(
            thermal_dt=1.0,
            tau_avg=1.0,
            T0=300.0,
            robin_h=0.5,
            robin_T_ambient=300.0,
            robin_sides=("top",),
        )
    )
    thermal.initialize(sim)
    thermal.T = jnp.full_like(thermal.T, 350.0)
    sim.fields.Ez = jnp.zeros_like(sim.fields.Ez)

    thermal.step(sim)
    T_after = np.asarray(thermal.T)

    assert np.all(T_after[-1, :] < 350.0)
    assert np.allclose(T_after[:-1, :], 350.0, atol=1e-6)
