import jax.numpy as jnp
import numpy as np

from beamz import Design, Material, Simulation, ThermalParams, ThermoPhysics


def _make_sim(material, dt=1.0, resolution=1.0):
    design = Design(width=3.0, height=3.0, material=material)
    time = np.array([0.0, dt])
    return Simulation(
        design=design, devices=[], boundaries=[], time=time, resolution=resolution
    )


def test_thermal_diffusion_smooths_hotspot():
    material = Material(permittivity=1.0, k=1.0, rho=1.0, cp=1.0, dn_dT=0.0, T0=300.0)
    sim = _make_sim(material, dt=1.0, resolution=1.0)

    thermal = ThermoPhysics(ThermalParams(thermal_dt=1.0, tau_avg=1.0, T0=300.0))
    thermal.initialize(sim)

    # Hotspot at center
    T = jnp.array([[300.0, 300.0, 300.0], [300.0, 310.0, 300.0], [300.0, 300.0, 300.0]])
    thermal.T = T

    # Ensure no heating source
    sim.fields.Ez = jnp.zeros_like(sim.fields.Ez)
    thermal.t_accum = 1.0
    thermal.step(sim)

    T_new = np.asarray(thermal.T)
    assert T_new[1, 1] < 310.0
    assert T_new[1, 0] > 300.0 or T_new[0, 1] > 300.0
