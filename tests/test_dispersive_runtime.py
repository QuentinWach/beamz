import numpy as np
import pytest

from beamz import Design, Simulation, um
from beamz.design.library import gold


def _make_2d_dispersive_sim():
    design = Design(width=3.0 * um, height=2.0 * um, material=gold())
    time = np.linspace(0.0, 1.5e-14, 24)
    sim = Simulation(design=design, resolution=0.2 * um, time=time)
    return sim


def test_dispersive_simulation_step_and_run_fast_consistent():
    sim_step = _make_2d_dispersive_sim()
    sim_fast = _make_2d_dispersive_sim()
    assert sim_step.fields.has_dispersion
    assert sim_fast.fields.has_dispersion

    for _ in range(6):
        assert sim_step.step()

    sim_fast.run_fast(num_steps=6, progress=False)

    assert np.allclose(np.asarray(sim_step.fields.Ex), np.asarray(sim_fast.fields.Ex))
    assert np.allclose(np.asarray(sim_step.fields.Ey), np.asarray(sim_fast.fields.Ey))
    assert np.allclose(np.asarray(sim_step.fields.Ez), np.asarray(sim_fast.fields.Ez))


def test_run_fast_dispersive_updates_ade_state():
    sim = _make_2d_dispersive_sim()
    psi_before = sim.fields.get_ade_state_tuple()
    assert all(p is not None for p in psi_before)

    sim.run_fast(num_steps=5, progress=False)

    psi_after = sim.fields.get_ade_state_tuple()
    for psi in psi_after:
        assert psi is not None
        arr = np.asarray(psi)
        assert np.isfinite(arr).all()


def test_run_jit_scan_runs_for_dispersive_without_sources():
    sim = _make_2d_dispersive_sim()
    assert sim.fields.has_dispersion
    sim.run_jit_scan(num_steps=5, progress=False)
    assert sim.current_step == 5


def test_run_jit_scan_falls_back_for_dispersive_with_sources(monkeypatch):
    sim = _make_2d_dispersive_sim()
    assert sim.fields.has_dispersion

    class _DummySource:
        def inject(self, *args, **kwargs):
            return None

    sim.devices.append(_DummySource())
    calls = {"count": 0}

    def _fake_run_fast(self, *args, **kwargs):
        calls["count"] += 1
        return {"used": "run_fast"}

    monkeypatch.setattr(Simulation, "run_fast", _fake_run_fast)
    out = sim.run_jit_scan(num_steps=5, progress=False)
    assert calls["count"] == 1
    assert out == {"used": "run_fast"}


def test_thermal_with_dispersive_materials_raises():
    class _DummyThermal:
        enabled = True

        def initialize(self, _sim):
            return None

    design = Design(width=2.0 * um, height=2.0 * um, material=gold())
    time = np.linspace(0.0, 8e-15, 16)
    with pytest.raises(NotImplementedError, match="Thermal coupling"):
        Simulation(
            design=design,
            thermal=_DummyThermal(),
            resolution=0.2 * um,
            time=time,
        )


def test_dispersive_3d_step_runs():
    design = Design(width=1.2 * um, height=1.2 * um, depth=1.2 * um, material=gold())
    time = np.linspace(0.0, 5e-15, 10)
    sim = Simulation(design=design, resolution=0.3 * um, time=time)
    assert sim.fields.has_dispersion
    assert sim.step()
