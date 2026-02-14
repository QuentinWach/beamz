import jax.numpy as jnp
import numpy as np

from beamz import Design, Material, Simulation, ThermalConfig, ThermalCoupling


def _make_sim(material, dt=1.0, resolution=1.0):
    design = Design(width=2.0, height=2.0, material=material)
    time = np.array([0.0, dt])
    return Simulation(
        design=design, devices=[], boundaries=[], time=time, resolution=resolution
    )


def test_thermal_coupling_increases_temperature_and_eps():
    material = Material(
        permittivity=4.0,
        conductivity=1.0,
        k=1.0,
        rho=1.0,
        cp=1.0,
        dn_dT=1e-3,
        T0=300.0,
    )
    sim = _make_sim(material, dt=1.0, resolution=1.0)

    thermal = ThermalCoupling(ThermalConfig(thermal_dt=1.0, tau_avg=1.0, T0=300.0))
    thermal.initialize(sim)

    sim.fields.Ez = jnp.ones_like(sim.fields.Ez)
    thermal.step(sim)

    assert float(jnp.max(thermal.T)) > 300.0
    assert float(jnp.max(sim.fields.permittivity)) > 4.0
