import numpy as np

import jax.numpy as jnp

from beamz.const import µm
from beamz.design.core import Design
from beamz.design.materials import LinearThermoOpticMaterial
from beamz.multiphysics.thermal import ThermalParams, ThermoPhysics
from beamz.simulation.core import Simulation


def test_thermal_updates_temperature_dependent_permittivity():
    material = LinearThermoOpticMaterial(
        n0=2.0,
        dn_dT=1.0e-3,
        sigma0=1.0,
        k0=1.0,
        rho0=1.0,
        cp0=1.0,
        T_ref=300.0,
    )
    design = Design(width=1 * µm, height=1 * µm, depth=0, material=material)
    time = np.array([0.0, 1.0e-15])
    thermal = ThermoPhysics(
        ThermalParams(thermal_dt=time[1], tau_avg=0.0, steady_state=False)
    )
    sim = Simulation(design=design, time=time, thermal=thermal)

    sim.fields.Ex = jnp.ones_like(sim.fields.Ex)
    sim.fields.Ey = jnp.ones_like(sim.fields.Ey)
    sim.fields.Ez = jnp.ones_like(sim.fields.Ez)

    initial_eps = jnp.array(sim.fields.permittivity)
    sim.thermal.step(sim)
    updated_eps = sim.fields.permittivity

    assert jnp.any(updated_eps > initial_eps)
