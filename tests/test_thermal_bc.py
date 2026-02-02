import numpy as np

import jax.numpy as jnp

from beamz import Design, Material, Simulation, ThermalParams, ThermoPhysics


def _make_sim(material, dt=1.0, resolution=1.0):
    design = Design(width=3.0, height=3.0, material=material)
    time = np.array([0.0, dt])
    return Simulation(design=design, devices=[], boundaries=[], time=time, resolution=resolution)


def test_neumann_bc_conserves_total_temperature_without_sources():
    material = Material(permittivity=1.0, k=1.0, rho=1.0, cp=1.0, dn_dT=0.0, T0=300.0)
    sim = _make_sim(material, dt=1.0, resolution=1.0)

    thermal = ThermoPhysics(ThermalParams(thermal_dt=1.0, tau_avg=1.0, T0=300.0))
    thermal.initialize(sim)

    thermal.T = jnp.array([[300.0, 301.0, 300.0],
                           [299.0, 300.0, 301.0],
                           [300.0, 299.0, 300.0]])

    sim.fields.Ez = jnp.zeros_like(sim.fields.Ez)
    thermal.t_accum = 1.0
    total_before = float(jnp.sum(thermal.T))
    thermal.step(sim)
    total_after = float(jnp.sum(thermal.T))

    assert np.isclose(total_before, total_after, rtol=1e-5, atol=1e-5)
