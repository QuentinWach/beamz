from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np

from beamz import PML, Design, Material, Simulation
from beamz.simulation.model import ShardingConfig


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
    fields = sim.compile().grid

    assert not program.sharding.layout.enabled
    assert program.sharding.layout.logical_shapes["Ex"] == tuple(fields.Ex.shape)
    assert program.sharding.layout.padded_shapes["Ex"] == tuple(fields.Ex.shape)
    assert program.config.sharding == ShardingConfig(enabled=False)


def test_cpu_fake_device_sharding_matches_unsharded_compiled_run():
    code = r"""
import numpy as np
import jax
from dataclasses import replace

from beamz import Design, FieldRecorder, Material, PML, Simulation
from beamz.simulation.model import ShardingConfig
from beamz.simulation.execute import runtime_inputs
from beamz.simulation.sharding import prepare_state


def make_sim(component=None):
    design = Design(
            width=13.0,
            height=15.0,
            depth=17.0,
            material=Material(permittivity=1.0),
        )
    monitors = [] if component is None else [FieldRecorder(
        (component,), 1, name="plane",
        center=(design.width / 2, design.height / 2, design.depth / 2),
        size=(design.width, design.height, 0.0),
    )]
    return Simulation(
        design=design,
        sources=[],
        monitors=monitors,
        boundaries=[PML(thickness=1.0, formulation="sponge")],
        time=np.arange(3, dtype=float) * 1e-18,
        resolution=1.0,
    )


def seed_state(sim, seed):
    rng = np.random.default_rng(seed)
    state = sim.initial_state()
    updates = {}
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(state, name.lower()))
        updates[name.lower()] = rng.normal(0.0, 1e-3, size=arr.shape).astype(np.float32)
    return state._replace(**updates)


assert jax.local_device_count("cpu") >= 2
cfg = ShardingConfig(enabled=True, axis="z", num_devices=2, backend="cpu")
reference = make_sim()
sharded = make_sim()
reference_state = seed_state(reference, 1234)
sharded_state = seed_state(sharded, 1234)

program = sharded.compile(num_steps=2, sharding=cfg)
state = runtime_inputs(program, sharded_state, monitor_steps=2)
prepared = prepare_state(
    program,
    state,
    replicated_fields=("powers", "timestamps", "counts", "t", "current_step"),
)
assert program.sharding.layout.enabled
assert prepared.ex.sharding.mesh is program.sharding.mesh
assert program.sharding.layout.padded_shapes["Ex"] == (18, 16, 13)
assert prepared.ex.shape == (18, 16, 13)
assert prepared.ex.addressable_shards[0].data.shape == (9, 16, 13)
report = sharded.memory_estimate(include_compiled=True, num_steps=2, sharding=cfg)
assert report["compiled"]["config"]["sharding"]["enabled"]
assert report["compiled"]["per_device_total_bytes"] > 0

reference_result = reference.advance(num_steps=2, progress=False, state=reference_state)
sharded_result = sharded.advance(num_steps=2, progress=False, sharding=cfg, state=sharded_state)

for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
    ref = np.asarray(getattr(reference_result.state, name.lower()))
    got = np.asarray(getattr(sharded_result.state, name.lower()))
    assert got.shape == ref.shape, (name, got.shape, ref.shape)
    np.testing.assert_allclose(got, ref, atol=2e-6, rtol=2e-6)

# Flattened monitor indices must use padded strides on transverse shard axes.
for axis, component in (("x", "Ex"), ("y", "Ey")):
    reference = make_sim(component)
    sharded = make_sim(component)
    reference_result = reference.advance(
        num_steps=1, progress=False, state=seed_state(reference, 2468)
    )
    sharded_result = sharded.advance(
        num_steps=1,
        progress=False,
        sharding=ShardingConfig(
            enabled=True, axis=axis, num_devices=2, backend="cpu"
        ),
        state=seed_state(sharded, 2468),
    )
    np.testing.assert_allclose(
        sharded_result.results.monitor("plane").fields[component],
        reference_result.results.monitor("plane").fields[component],
        atol=2e-6,
        rtol=2e-6,
    )
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


def test_cpu_fake_device_sharding_matches_unsharded_pec_compiled_run():
    code = r"""
import numpy as np
import jax
from dataclasses import replace

from beamz import Design, Material, PEC, Simulation
from beamz.simulation.model import ShardingConfig
from beamz.simulation.execute import runtime_inputs
from beamz.simulation.sharding import crop_state, prepare_state


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


def seed_state(sim, seed):
    rng = np.random.default_rng(seed)
    state = sim.initial_state()
    updates = {}
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(state, name.lower()))
        updates[name.lower()] = rng.normal(0.0, 1e-3, size=arr.shape).astype(np.float32)
    return state._replace(**updates)


assert jax.local_device_count("cpu") >= 2
cfg = ShardingConfig(enabled=True, axis="z", num_devices=2, backend="cpu")
reference = make_sim()
sharded = make_sim()
reference_state = seed_state(reference, 5678)
sharded_state = seed_state(sharded, 5678)

program = sharded.compile(num_steps=2, sharding=cfg)
state = runtime_inputs(program, sharded_state, monitor_steps=2)
prepared = prepare_state(
    program,
    state,
    replicated_fields=("powers", "timestamps", "counts", "t", "current_step"),
)
cropped = crop_state(program, prepared)
assert program.sharding.layout.enabled
assert program.sharding.layout.logical_shapes["Ex"] == (18, 16, 13)
assert program.sharding.layout.padded_shapes["Ex"] == (18, 16, 13)
assert prepared.ex.shape == (18, 16, 13)
assert cropped.ex.shape == (18, 16, 13)
assert prepared.ex.addressable_shards[0].data.shape == (9, 16, 13)

reference_result = reference.advance(num_steps=2, progress=False, state=reference_state)
sharded_result = sharded.advance(num_steps=2, progress=False, sharding=cfg, state=sharded_state)

for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
    ref = np.asarray(getattr(reference_result.state, name.lower()))
    got = np.asarray(getattr(sharded_result.state, name.lower()))
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


def test_cpu_fake_device_sharded_cpml_uses_slab_auxiliaries():
    code = r"""
import os
import numpy as np
import jax
from dataclasses import replace

from beamz import Design, Material, PML, Simulation
from beamz.simulation.model import ShardingConfig
from beamz.simulation.execute import runtime_inputs


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
        boundaries=[PML(thickness=1.0, formulation="cpml")],
        time=np.arange(3, dtype=float) * 1e-18,
        resolution=1.0,
    )


def seed_state(sim, seed):
    rng = np.random.default_rng(seed)
    state = sim.initial_state()
    updates = {}
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(state, name.lower()))
        updates[name.lower()] = rng.normal(0.0, 1e-3, size=arr.shape).astype(np.float32)
    return state._replace(**updates)


assert jax.local_device_count("cpu") >= 2
cfg = ShardingConfig(enabled=True, axis="z", num_devices=2, backend="cpu")
reference = make_sim()
sharded = make_sim()
reference_state = seed_state(reference, 9012)
sharded_state = seed_state(sharded, 9012)

program = sharded.compile(num_steps=2, sharding=cfg)
assert program.sharding.layout.enabled
assert program.boundary.cpml.enabled
state = runtime_inputs(program, sharded_state, monitor_steps=2)
assert isinstance(state.cpml_psi_h_terms[0], np.ndarray)
assert isinstance(state.cpml_psi_e_terms[0], np.ndarray)
assert tuple(value.shape for value in state.cpml_psi_h_terms) == tuple(
    term.slab.shape for term in program.boundary.cpml.h_terms
)

reference_result = reference.advance(num_steps=2, progress=False, state=reference_state)
sharded_result = sharded.advance(num_steps=2, progress=False, sharding=cfg, state=sharded_state)

for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
    ref = np.asarray(getattr(reference_result.state, name.lower()))
    got = np.asarray(getattr(sharded_result.state, name.lower()))
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
        timeout=120,
    )
