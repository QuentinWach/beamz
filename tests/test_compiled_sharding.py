from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np

from beamz import PML, Design, Material, ShardingConfig, Simulation


def _awkward_3d_sim(steps: int = 2) -> Simulation:
    steps = max(2, int(steps))
    return Simulation(
        design=Design(
            width=13.0,
            height=15.0,
            depth=17.0,
            material=Material(permittivity=1.0),
        ),
        sources=[],
        monitors=[],
        boundaries=[PML(thickness=1.0, formulation="sponge")],
        time=np.arange(steps, dtype=float) * 1e-18,
        resolution=1.0,
    )


def test_sharding_config_export_and_disabled_layout_is_shape_preserving():
    sim = _awkward_3d_sim(steps=1)
    program = sim.compile(num_steps=1, sharding=ShardingConfig(enabled=False))

    assert not program.storage_layout.enabled
    assert program.field_shape_ex == tuple(sim.fields.Ex.shape)
    assert program.storage_shape_ex == tuple(sim.fields.Ex.shape)
    assert program.config.sharding == ShardingConfig(enabled=False)


def test_cpu_fake_device_sharding_matches_unsharded_compiled_run():
    code = r"""
import numpy as np
import jax

from beamz import Design, Material, PML, ShardingConfig, Simulation


def make_sim():
    return Simulation(
        design=Design(
            width=13.0,
            height=15.0,
            depth=17.0,
            material=Material(permittivity=1.0),
        ),
        sources=[],
        monitors=[],
        boundaries=[PML(thickness=1.0, formulation="sponge")],
        time=np.arange(3, dtype=float) * 1e-18,
        resolution=1.0,
    )


def seed_fields(sim, seed):
    rng = np.random.default_rng(seed)
    payload = {}
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(sim.fields, name))
        payload[name] = rng.normal(0.0, 1e-3, size=arr.shape).astype(np.float32)
    for name, arr in payload.items():
        setattr(sim.fields, name, arr)
    return payload


def apply_payload(sim, payload):
    for name, arr in payload.items():
        setattr(sim.fields, name, arr.copy())


assert jax.local_device_count("cpu") >= 2
cfg = ShardingConfig(enabled=True, axis="z", num_devices=2, backend="cpu")
reference = make_sim()
sharded = make_sim()
payload = seed_fields(reference, 1234)
apply_payload(sharded, payload)

program = sharded.compile(num_steps=2, sharding=cfg)
engine_state, _monitor_state = sharded._compiled_runtime_inputs(program)
prepared = program.prepare_engine_state(engine_state)
assert program.storage_layout.enabled
assert program.storage_shape_ex == (18, 15, 12)
assert prepared.ex.shape == (18, 15, 12)
assert prepared.ex.addressable_shards[0].data.shape == (9, 15, 12)
report = sharded.memory_estimate(include_compiled=True, num_steps=2, sharding=cfg)
assert report["compiled"]["config"]["sharding"]["enabled"]
assert report["compiled"]["per_device_total_bytes"] > 0

reference.run_compiled(num_steps=2, progress=False)
sharded.run_compiled(num_steps=2, progress=False, sharding=cfg)

for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
    ref = np.asarray(getattr(reference.fields, name))
    got = np.asarray(getattr(sharded.fields, name))
    assert got.shape == ref.shape, (name, got.shape, ref.shape)
    np.testing.assert_allclose(got, ref, atol=2e-6, rtol=2e-6)
"""
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env.setdefault("JAX_PLATFORMS", "cpu")
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        env=env,
        cwd=os.getcwd(),
        timeout=90,
    )


def test_cpu_fake_device_sharding_matches_unsharded_full_pec_compiled_run():
    code = r"""
import numpy as np
import jax

from beamz import Design, Material, PEC, ShardingConfig, Simulation


def make_sim():
    return Simulation(
        design=Design(
            width=13.0,
            height=15.0,
            depth=17.0,
            material=Material(permittivity=1.0),
        ),
        sources=[],
        monitors=[],
        boundaries=[PEC(edges="all")],
        time=np.arange(3, dtype=float) * 1e-18,
        resolution=1.0,
    )


def seed_fields(sim, seed):
    rng = np.random.default_rng(seed)
    payload = {}
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(sim.fields, name))
        payload[name] = rng.normal(0.0, 1e-3, size=arr.shape).astype(np.float32)
    for name, arr in payload.items():
        setattr(sim.fields, name, arr)
    return payload


def apply_payload(sim, payload):
    for name, arr in payload.items():
        setattr(sim.fields, name, arr.copy())


assert jax.local_device_count("cpu") >= 2
cfg = ShardingConfig(enabled=True, axis="z", num_devices=2, backend="cpu")
reference = make_sim()
sharded = make_sim()
payload = seed_fields(reference, 5678)
apply_payload(sharded, payload)

program = sharded.compile(num_steps=2, sharding=cfg)
engine_state, _monitor_state = sharded._compiled_runtime_inputs(program)
prepared = program.prepare_engine_state(engine_state)
cropped = program.crop_engine_state(prepared)
assert program.full_pec_3d
assert program.storage_layout.enabled
assert program.storage_layout.pec_full_storage
assert program.field_shape_ex == (17, 15, 12)
assert program.storage_shape_ex == (18, 16, 13)
assert prepared.ex.shape == (18, 16, 13)
assert cropped.ex.shape == (17, 15, 12)
assert prepared.ex.addressable_shards[0].data.shape == (9, 16, 13)

reference.run_compiled(num_steps=2, progress=False)
sharded.run_compiled(num_steps=2, progress=False, sharding=cfg)

for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
    ref = np.asarray(getattr(reference.fields, name))
    got = np.asarray(getattr(sharded.fields, name))
    assert got.shape == ref.shape, (name, got.shape, ref.shape)
    np.testing.assert_allclose(got, ref, atol=2e-6, rtol=2e-6)
"""
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env.setdefault("JAX_PLATFORMS", "cpu")
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        env=env,
        cwd=os.getcwd(),
        timeout=90,
    )
