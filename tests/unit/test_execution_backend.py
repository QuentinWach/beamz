from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import beamz as bz
from beamz.design import MaterialGrid, RectilinearGrid
from beamz.design.raster import Grid, Material, RasterOptions, Scene, rasterize
from beamz.simulation import backend as backend_runtime
from beamz.simulation import sharding as sharding_runtime


class _FakeDevice:
    platform = "gpu"
    device_kind = "NVIDIA H100 80GB HBM3"
    compute_capability = (9, 0)


def _extension(
    *targets, abi_version=backend_runtime.CUDA_ABI_VERSION, complete_streamed=True
):
    targets = set(targets)
    if complete_streamed and "beamz_cuda_streamed" in targets:
        targets.update(backend_runtime.CUDA_STREAMED_TARGETS)
    return SimpleNamespace(
        __version__="0.4.0",
        __abi_version__=abi_version,
        registrations=lambda: {target: object() for target in targets},
    )


def test_auto_preserves_jax_when_cuda_is_not_visible(monkeypatch):
    monkeypatch.setattr(backend_runtime, "_gpu_devices", lambda: ())

    assert backend_runtime.resolve_backend("auto") == "jax"
    with pytest.raises(backend_runtime.CudaBackendUnavailable, match="no visible"):
        backend_runtime.resolve_backend("cuda")


def test_auto_preserves_jax_for_unsupported_2d_simulations(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "auto" else backend,
    )
    simulation = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
    )

    program = simulation.compile(backend="auto")

    assert program.config.backend == "jax"


@pytest.mark.parametrize("backend", ["cuda_streamed", "cuda_hopper"])
@pytest.mark.parametrize("polarization", ["tm", "te"])
def test_explicit_cuda_rejects_2d_simulations(backend, polarization):
    simulation = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        polarization=polarization,
        time=np.arange(3) * 1e-17,
    )

    with pytest.raises(backend_runtime.CudaBackendUnavailable, match="3D simulation"):
        simulation.compile(backend=backend)


def _rectilinear_3d_simulation():
    grid = RectilinearGrid(
        np.asarray((0.0, 0.1, 0.3)),
        np.asarray((0.0, 0.15, 0.3)),
        np.asarray((0.0, 0.1, 0.3)),
    )
    shape = (2, 2, 2)
    materials = MaterialGrid(
        permittivity=np.ones(shape, dtype=np.float32),
        conductivity=np.float32(0.0),
        permeability=np.float32(1.0),
        resolution=0.1,
        shape=shape,
        grid=grid,
    )
    return bz.Simulation(material_grid=materials, time=np.arange(3) * 1e-17)


def test_auto_selects_streamed_cuda_for_rectilinear_3d_simulations(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "auto" else backend,
    )

    program = _rectilinear_3d_simulation().compile(backend="auto")

    assert program.config.backend == "cuda_streamed"


def test_explicit_streamed_cuda_accepts_rectilinear_3d_simulations(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "cuda_streamed" else backend,
    )

    program = _rectilinear_3d_simulation().compile(backend="cuda_streamed")

    assert program.config.backend == "cuda_streamed"
    assert program.config.metric_kind == "rectilinear"


def test_auto_preserves_jax_for_multi_device_sharding(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "auto" else backend,
    )
    monkeypatch.setattr(sharding_runtime, "_jax_devices_for_config", lambda _cfg: ())

    program = _rectilinear_3d_simulation().compile(
        backend="auto", sharding={"num_devices": 2}
    )

    assert program.config.backend == "jax"


def test_explicit_cuda_rejects_multi_device_sharding():
    with pytest.raises(
        backend_runtime.CudaBackendUnavailable, match="multi-device sharding"
    ):
        _rectilinear_3d_simulation().compile(
            backend="cuda_streamed", sharding={"num_devices": 2}
        )


def test_hopper_cuda_rejects_rectilinear_3d_simulations():
    with pytest.raises(
        backend_runtime.CudaBackendUnavailable, match="Hopper-specific kernel"
    ):
        _rectilinear_3d_simulation().compile(backend="cuda_hopper")


def _full_tensor_3d_simulation():
    material_grid = MaterialGrid.from_raster_result(
        rasterize(
            Scene(
                (
                    Material(
                        epsilon_r=(
                            (3.0, 0.2, 0.0),
                            (0.2, 2.0, 0.0),
                            (0.0, 0.0, 1.0),
                        )
                    ),
                )
            ),
            Grid.uniform((0.0, 0.0, 0.0), (0.3, 0.3, 0.3), (3, 3, 3)),
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    return bz.Simulation(material_grid=material_grid, time=np.arange(3) * 1e-17)


def test_auto_preserves_jax_for_full_tensor_permittivity(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "auto" else backend,
    )

    program = _full_tensor_3d_simulation().compile(backend="auto")

    assert program.config.backend == "jax"


def test_explicit_cuda_rejects_full_tensor_permittivity():
    with pytest.raises(
        backend_runtime.CudaBackendUnavailable, match="full-tensor permittivity"
    ):
        _full_tensor_3d_simulation().compile(backend="cuda_streamed")


def test_cuda_status_exposes_complete_diagnostics():
    status = backend_runtime.CudaBackendStatus(
        available=True,
        extension_version="0.3.0",
        abi_version=3,
        targets=("beamz_cuda_streamed",),
        gpu_devices=("NVIDIA H100 80GB HBM3",),
        compute_capabilities=(90,),
    )

    assert status.as_dict() == {
        "available": True,
        "extension_version": "0.3.0",
        "abi_version": 3,
        "targets": ("beamz_cuda_streamed",),
        "gpu_devices": ("NVIDIA H100 80GB HBM3",),
        "compute_capabilities": (90,),
        "reason": None,
    }


def test_cuda_defaults_to_validated_streamed_target_on_sm90(monkeypatch):
    extension = _extension("beamz_cuda_streamed", "beamz_cuda_hopper")
    monkeypatch.setattr(backend_runtime, "_gpu_devices", lambda: (_FakeDevice(),))
    monkeypatch.setattr(backend_runtime, "_load_extension", lambda: extension)
    monkeypatch.setattr(
        backend_runtime,
        "register_cuda_ffi_targets",
        lambda module=None: tuple(sorted(extension.registrations())),
    )

    assert backend_runtime.resolve_backend("cuda") == "cuda_streamed"
    assert backend_runtime.resolve_backend("cuda_streamed") == "cuda_streamed"
    assert backend_runtime.resolve_backend("cuda_hopper") == "cuda_hopper"
    assert backend_runtime.resolve_backend("auto") == "cuda_streamed"


@pytest.mark.parametrize(
    ("extension", "message"),
    [
        (_extension("beamz_cuda_streamed", abi_version=1), "ABI mismatch"),
        (
            _extension("beamz_cuda_streamed", complete_streamed=False),
            "missing required streamed FFI targets",
        ),
    ],
)
def test_incompatible_cuda_extension_falls_back_or_fails_explicitly(
    monkeypatch, extension, message
):
    monkeypatch.setattr(backend_runtime, "_gpu_devices", lambda: (_FakeDevice(),))
    monkeypatch.setattr(backend_runtime, "_load_extension", lambda: extension)
    monkeypatch.setattr(backend_runtime, "_REGISTERED_MODULE", None)

    status = backend_runtime.cuda_backend_status()

    assert not status.available
    assert message in (status.reason or "")
    assert backend_runtime.resolve_backend("auto") == "jax"
    with pytest.raises(backend_runtime.CudaBackendUnavailable, match=message):
        backend_runtime.resolve_backend("cuda_streamed")


def test_explicit_hopper_rejects_pre_sm90(monkeypatch):
    device = _FakeDevice()
    device.compute_capability = (8, 0)
    extension = _extension("beamz_cuda_streamed", "beamz_cuda_hopper")
    monkeypatch.setattr(backend_runtime, "_gpu_devices", lambda: (device,))
    monkeypatch.setattr(backend_runtime, "_load_extension", lambda: extension)
    monkeypatch.setattr(
        backend_runtime,
        "register_cuda_ffi_targets",
        lambda module=None: tuple(sorted(extension.registrations())),
    )

    with pytest.raises(backend_runtime.CudaBackendUnavailable, match="SM90"):
        backend_runtime.resolve_backend("cuda_hopper")
    assert backend_runtime.resolve_backend("cuda") == "cuda_streamed"


def test_hopper_only_extension_is_rejected_as_incomplete(monkeypatch):
    extension = _extension("beamz_cuda_hopper")
    monkeypatch.setattr(backend_runtime, "_gpu_devices", lambda: (_FakeDevice(),))
    monkeypatch.setattr(backend_runtime, "_load_extension", lambda: extension)
    monkeypatch.setattr(
        backend_runtime,
        "register_cuda_ffi_targets",
        lambda module=None: tuple(sorted(extension.registrations())),
    )

    assert backend_runtime.resolve_backend("auto") == "jax"
    with pytest.raises(
        backend_runtime.CudaBackendUnavailable, match="missing required"
    ):
        backend_runtime.resolve_backend("cuda_hopper")


def test_typed_ffi_registrations_use_cuda_api_v1(monkeypatch):
    extension = _extension(
        "beamz_cuda_streamed",
        "beamz_cuda_program",
        "beamz_cuda_hopper",
    )
    registrations = []
    monkeypatch.setattr(backend_runtime, "_REGISTERED_MODULE", None)
    monkeypatch.setattr(
        backend_runtime.jax.ffi,
        "register_ffi_target",
        lambda name, capsule, **kwargs: registrations.append((name, capsule, kwargs)),
    )

    targets = backend_runtime.register_cuda_ffi_targets(extension)

    assert targets == (
        "beamz_cuda_hopper",
        "beamz_cuda_program",
        "beamz_cuda_streamed",
    )
    assert {name for name, _, _ in registrations} == set(targets)
    assert all(
        kwargs == {"platform": "CUDA", "api_version": 1}
        for _, _, kwargs in registrations
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("xla", "jax"),
        ("cuda-streamed", "cuda_streamed"),
        ("hopper", "cuda_hopper"),
    ],
)
def test_backend_aliases(value, expected):
    assert backend_runtime.normalize_backend(value) == expected


def test_backend_participates_in_compiled_program_identity():
    from beamz.simulation.compile import CompiledProgramKey

    simulation = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
    )
    jax_key = CompiledProgramKey.from_request(
        simulation.to_request(num_steps=2, backend="jax")
    )
    cuda_key = CompiledProgramKey.from_request(
        simulation.to_request(num_steps=2, backend="cuda_streamed")
    )

    assert jax_key != cuda_key
    assert jax_key.backend == "jax"
    assert cuda_key.backend == "cuda_streamed"


def test_cuda_policy_participates_in_compiled_program_identity():
    from beamz.simulation.compile import CompiledProgramKey

    simulation = _rectilinear_3d_simulation()
    request = simulation.to_request(num_steps=2, backend="cuda_streamed")
    default_key = CompiledProgramKey.from_request(
        replace(
            request,
            run=replace(request.run, cuda_flags=backend_runtime.CUDA_DEFAULT_FLAGS),
        )
    )
    uncached_key = CompiledProgramKey.from_request(
        replace(
            request,
            run=replace(
                request.run,
                cuda_flags=(
                    backend_runtime.CUDA_DEFAULT_FLAGS
                    & ~backend_runtime.CUDA_GRAPH_CACHE
                ),
            ),
        )
    )
    smaller_cache_key = CompiledProgramKey.from_request(
        replace(
            request,
            run=replace(
                request.run,
                cuda_flags=backend_runtime.CUDA_DEFAULT_FLAGS,
                cuda_graph_cache_capacity=8,
            ),
        )
    )

    assert default_key != uncached_key
    assert default_key != smaller_cache_key


def test_cuda_policy_is_snapshotted_when_program_is_compiled(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "cuda_streamed" else backend,
    )
    monkeypatch.delenv("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY", raising=False)
    simulation = _rectilinear_3d_simulation()
    simulation.clear_compiled_cache()

    first = simulation.compile(backend="cuda_streamed")
    monkeypatch.setenv("BEAMZ_CUDA_DISABLE_GRAPH_CACHE", "1")
    monkeypatch.setenv("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY", "8")
    second = simulation.compile(backend="cuda_streamed")

    assert first.config.cuda_flags & backend_runtime.CUDA_GRAPH_CACHE
    assert not second.config.cuda_flags & backend_runtime.CUDA_GRAPH_CACHE
    assert first.config.cuda_graph_cache_capacity == 32
    assert second.config.cuda_graph_cache_capacity == 8
    assert first is not second


def test_cuda_graph_cache_capacity_policy_is_bounded(monkeypatch):
    monkeypatch.delenv("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY", raising=False)
    assert backend_runtime.cuda_graph_cache_capacity_from_env() == 32

    monkeypatch.setenv("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY", "0")
    assert backend_runtime.cuda_graph_cache_capacity_from_env() == 0

    monkeypatch.setenv("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY", "4097")
    with pytest.raises(ValueError, match="must be an integer from 0 to 4096"):
        backend_runtime.cuda_graph_cache_capacity_from_env()

    monkeypatch.setenv("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY", "many")
    with pytest.raises(ValueError, match="must be an integer from 0 to 4096"):
        backend_runtime.cuda_graph_cache_capacity_from_env()
