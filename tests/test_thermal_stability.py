import jax.numpy as jnp
import numpy as np

from beamz import Design, Material, Simulation, ThermalConfig, ThermalCoupling


def _make_sim(material, dt=1.0, resolution=1.0):
    design = Design(width=3.0, height=3.0, material=material)
    time = np.array([0.0, dt])
    return Simulation(
        design=design, devices=[], boundaries=[], time=time, resolution=resolution
    )


def test_large_thermal_dt_uses_substeps_and_avoids_blowup():
    material = Material(permittivity=1.0, k=1.0, rho=1.0, cp=1.0, dn_dT=0.0, T0=300.0)
    sim = _make_sim(material, dt=1.0, resolution=1.0)
    thermal = ThermalCoupling(ThermalConfig(thermal_dt=10.0, tau_avg=1.0, T0=300.0))
    thermal.initialize(sim)

    thermal.T = jnp.array(
        [[300.0, 300.0, 300.0], [300.0, 310.0, 300.0], [300.0, 300.0, 300.0]]
    )
    sim.fields.Ez = jnp.zeros_like(sim.fields.Ez)

    for _ in range(10):
        thermal.step(sim)

    T_new = np.asarray(thermal.T)
    assert np.isfinite(T_new).all()
    assert T_new[1, 1] < 310.0


def test_diffusion_remains_bounded_across_many_updates():
    material = Material(permittivity=1.0, k=1.0, rho=1.0, cp=1.0, dn_dT=0.0, T0=300.0)
    sim = _make_sim(material, dt=1.0, resolution=1.0)
    thermal = ThermalCoupling(ThermalConfig(thermal_dt=4.0, tau_avg=1.0, T0=300.0))
    thermal.initialize(sim)

    thermal.T = jnp.array(
        [[300.0, 300.0, 300.0], [300.0, 310.0, 300.0], [300.0, 300.0, 300.0]]
    )
    sim.fields.Ez = jnp.zeros_like(sim.fields.Ez)

    max_history = []
    min_history = []
    for _ in range(80):
        thermal.step(sim)
        T = np.asarray(thermal.T)
        max_history.append(float(np.max(T)))
        min_history.append(float(np.min(T)))

    assert np.isfinite(max_history).all()
    assert max(max_history) <= 310.0 + 1e-5
    assert min(min_history) >= 299.0


def test_e2_ema_stays_bounded_when_dt_exceeds_tau():
    material = Material(
        permittivity=1.0,
        conductivity=1.0,
        k=1.0,
        rho=1.0,
        cp=1.0,
        dn_dT=0.0,
        T0=300.0,
    )
    sim = _make_sim(material, dt=1.0, resolution=1.0)
    thermal = ThermalCoupling(ThermalConfig(thermal_dt=1.0, tau_avg=1e-6, T0=300.0))
    thermal.initialize(sim)

    sim.fields.Ez = jnp.ones_like(sim.fields.Ez)
    thermal.step(sim)
    e2_after_on = np.asarray(thermal.E2_avg)
    assert np.all(e2_after_on >= -1e-9)
    assert np.all(e2_after_on <= 1.0 + 1e-6)

    sim.fields.Ez = jnp.zeros_like(sim.fields.Ez)
    thermal.step(sim)
    e2_after_off = np.asarray(thermal.E2_avg)
    assert np.all(e2_after_off >= -1e-9)
    assert np.all(e2_after_off <= e2_after_on + 1e-6)
