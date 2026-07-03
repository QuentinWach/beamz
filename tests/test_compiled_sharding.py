from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from beamz import PML, Design, Material, ShardingConfig, Simulation
from beamz.simulation.compiled import StorageLayout, _make_storage_fields_proxy


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


def test_storage_proxy_preserves_scalar_loss_terms():
    component_shapes = {
        "Ex": (3, 4, 5),
        "Ey": (3, 3, 6),
        "Ez": (2, 4, 6),
        "Hx": (2, 3, 6),
        "Hy": (2, 4, 5),
        "Hz": (3, 3, 5),
    }
    storage_shapes = {
        name: (4, *shape[1:]) for name, shape in component_shapes.items()
    }
    fields = SimpleNamespace(
        permittivity=jnp.ones((3, 4, 6), dtype=jnp.float32),
        pml_data={},
        sig_x=jnp.asarray(0.0, dtype=jnp.float32),
        sig_y=jnp.asarray(0.0, dtype=jnp.float32),
        sig_z=jnp.asarray(0.0, dtype=jnp.float32),
        sigma_m_hx=jnp.asarray(0.0, dtype=jnp.float32),
        sigma_m_hy=jnp.asarray(0.0, dtype=jnp.float32),
        sigma_m_hz=jnp.asarray(0.0, dtype=jnp.float32),
        eps_x=jnp.ones(component_shapes["Ex"], dtype=jnp.float32),
        eps_y=jnp.ones(component_shapes["Ey"], dtype=jnp.float32),
        eps_z=jnp.ones(component_shapes["Ez"], dtype=jnp.float32),
    )
    for name, shape in component_shapes.items():
        setattr(fields, name, jnp.zeros(shape, dtype=jnp.float32))
    layout = StorageLayout(
        enabled=True,
        pec_full_storage=False,
        logical_base_shape=fields.permittivity.shape,
        axis_name="z",
        axis=0,
        num_devices=2,
        backend="cpu",
        logical_shapes=component_shapes,
        active_shapes=component_shapes,
        storage_shapes=storage_shapes,
        padding={name: ((0, 1), (0, 0), (0, 0)) for name in component_shapes},
        valid_masks={name: None for name in component_shapes},
    )

    proxy = _make_storage_fields_proxy(fields, layout)

    assert proxy.eps_x.shape == storage_shapes["Ex"]
    assert proxy.eps_y.shape == storage_shapes["Ey"]
    assert proxy.eps_z.shape == storage_shapes["Ez"]
    assert jnp.asarray(proxy.sig_x).shape == ()
    assert jnp.asarray(proxy.sig_y).shape == ()
    assert jnp.asarray(proxy.sig_z).shape == ()
    assert jnp.asarray(proxy.sigma_m_hx).shape == ()
    assert jnp.asarray(proxy.sigma_m_hy).shape == ()
    assert jnp.asarray(proxy.sigma_m_hz).shape == ()


def test_storage_proxy_keeps_numpy_material_padding_on_host():
    component_shapes = {
        "Ex": (3, 4, 5),
        "Ey": (3, 3, 6),
        "Ez": (2, 4, 6),
        "Hx": (2, 3, 6),
        "Hy": (2, 4, 5),
        "Hz": (3, 3, 5),
    }
    storage_shapes = {
        name: (4, *shape[1:]) for name, shape in component_shapes.items()
    }
    fields = SimpleNamespace(
        permittivity=np.ones((3, 4, 6), dtype=np.float32),
        pml_data={},
        sig_x=np.asarray(0.0, dtype=np.float32),
        sig_y=np.asarray(0.0, dtype=np.float32),
        sig_z=np.asarray(0.0, dtype=np.float32),
        sigma_m_hx=np.asarray(0.0, dtype=np.float32),
        sigma_m_hy=np.asarray(0.0, dtype=np.float32),
        sigma_m_hz=np.asarray(0.0, dtype=np.float32),
        eps_x=np.ones(component_shapes["Ex"], dtype=np.float32),
        eps_y=np.ones(component_shapes["Ey"], dtype=np.float32),
        eps_z=np.ones(component_shapes["Ez"], dtype=np.float32),
    )
    for name, shape in component_shapes.items():
        setattr(fields, name, np.zeros(shape, dtype=np.float32))
    layout = StorageLayout(
        enabled=True,
        pec_full_storage=False,
        logical_base_shape=fields.permittivity.shape,
        axis_name="z",
        axis=0,
        num_devices=2,
        backend="cpu",
        logical_shapes=component_shapes,
        active_shapes=component_shapes,
        storage_shapes=storage_shapes,
        padding={name: ((0, 1), (0, 0), (0, 0)) for name in component_shapes},
        valid_masks={name: None for name in component_shapes},
    )

    proxy = _make_storage_fields_proxy(fields, layout)

    assert isinstance(proxy.eps_x, np.ndarray)
    assert proxy.eps_x.shape == storage_shapes["Ex"]
    np.testing.assert_array_equal(proxy.eps_x[-1], 1.0)
    assert isinstance(proxy.sig_x, np.ndarray)
    assert proxy.sig_x.shape == ()


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


def test_cpu_fake_device_sharded_cpml_supports_packed_psi():
    code = r"""
import os
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
        boundaries=[PML(thickness=1.0, formulation="cpml")],
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


os.environ["BEAMZ_CPML_PACKED_PSI"] = "true"
assert jax.local_device_count("cpu") >= 2
cfg = ShardingConfig(enabled=True, axis="z", num_devices=2, backend="cpu")
reference = make_sim()
sharded = make_sim()
payload = seed_fields(reference, 9012)
apply_payload(sharded, payload)

program = sharded.compile(num_steps=2, sharding=cfg)
assert program.storage_layout.enabled
assert program.use_cpml_3d
assert program.use_cpml_3d_packed_psi
assert program.cpml3d_h_psi_shapes == tuple(
    spec.shape for spec in program.cpml3d_h_slab_specs
)
assert program.cpml3d_e_psi_shapes == tuple(
    spec.shape for spec in program.cpml3d_e_slab_specs
)
engine_state, _monitor_state = sharded._compiled_runtime_inputs(program)
assert isinstance(engine_state.cpml3d_psi_h_terms[0], np.ndarray)
assert isinstance(engine_state.cpml3d_psi_e_terms[0], np.ndarray)

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
    env["BEAMZ_CPML_PACKED_PSI"] = "true"
    env.setdefault("JAX_PLATFORMS", "cpu")
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        env=env,
        cwd=os.getcwd(),
        timeout=120,
    )
