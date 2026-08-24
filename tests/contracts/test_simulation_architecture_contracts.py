from __future__ import annotations

import ast
import importlib
import inspect
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, is_dataclass

import jax.numpy as jnp
import numpy as np
import pytest

import beamz.simulation.api as simulation_core
from beamz import (
    PEC,
    PMC,
    PML,
    Absorber,
    Box,
    Circle,
    CircularBend,
    Design,
    FieldMonitor,
    FluxMonitor,
    GaussianSource,
    Material,
    ModeMonitor,
    ModeSource,
    ModeSpec,
    Polygon,
    Rectangle,
    Ring,
    SampledSignal,
    Simulation,
    Sphere,
    Taper,
)
from beamz._cache_tokens import cache_token
from beamz.design.discretization import MaterialGrid
from beamz.devices.sources.compiler import (
    CompiledInjectionPlan,
    InjectionPlanEntry,
    SourceLoweringContext,
    SourceSupport,
    SpatialFieldProfile,
    TemporalWaveform,
    compile_source_specs,
    lower_source,
)
from beamz.devices.visualization import visual_spec_from_device
from beamz.simulation.compile import CompiledProgramKey
from beamz.simulation.execute import _apply_specs
from beamz.simulation.model import SimulationRequest, SimulationState
from beamz.simulation.sharding import sharding_cache_token
from tests.validation.invariants.test_engine_equivalence import (
    TEST_WAVELENGTH,
    _center_index,
    _make_2d_sim,
    _make_3d_sim,
    _PointElectricCurrentSource,
)


def _module_dependencies(module_name):
    tree = ast.parse(inspect.getsource(importlib.import_module(module_name)))
    dependencies = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            dependencies.add(node.module)
        elif isinstance(node, ast.Import):
            dependencies.update(alias.name for alias in node.names)
    return dependencies


@pytest.mark.parametrize(
    ("module_name", "forbidden_prefix"),
    [
        ("beamz.lattice", "beamz.devices"),
        ("beamz.lattice", "beamz.simulation"),
        ("beamz.devices._boundary_compile", "beamz.analysis"),
        ("beamz.devices._boundary_compile", "beamz.simulation"),
        ("beamz.devices.monitors.compiler", "beamz.analysis"),
        ("beamz.devices.monitors.compiler", "beamz.simulation"),
        ("beamz.devices.sources.compiler", "beamz.analysis"),
        ("beamz.devices.sources.compiler", "beamz.simulation"),
        ("beamz.devices.sources.mode_launch", "beamz.devices.monitors"),
        ("beamz.devices.sources.planar_tfsf", "beamz.simulation"),
        ("beamz.devices.sources.specs", "beamz.analysis"),
        ("beamz.simulation.api", "beamz.analysis"),
        ("beamz.simulation.compile", "beamz.analysis"),
        ("beamz.simulation.execute", "beamz.analysis"),
        ("beamz.simulation.observe", "beamz.analysis"),
        ("beamz.simulation.results", "beamz.analysis"),
    ],
)
def test_dependency_layers_do_not_import_upward(module_name, forbidden_prefix):
    assert not any(
        dependency == forbidden_prefix or dependency.startswith(f"{forbidden_prefix}.")
        for dependency in _module_dependencies(module_name)
    )


pytestmark = [pytest.mark.compiled, pytest.mark.component]


def _signal(n=8):
    return np.linspace(0.0, 0.7, n, dtype=np.float32)


def _source(wl, *, x=1.2):
    return GaussianSource(
        position=(x * wl, 1.0 * wl),
        width=0.12 * wl,
        signal=_signal(),
    )


def _dft_monitor(sim, *, freq=2.0e14):
    wl = sim.size[0] / 2.4
    return FieldMonitor(
        center=(1.2 * wl, 1.0 * wl, 0.0),
        size=(wl, 0.0, 0.0),
        name="m",
        freqs=[freq],
        fields=("Ez",),
    )


def _count_material_builds(monkeypatch):
    calls = []
    original = simulation_core.build_material_grid

    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation_core, "build_material_grid", counted)
    return calls


def test_public_device_configs_are_immutable_and_copyable():
    sim, wl = _make_2d_sim(plane_2d="xy", steps=5)
    source = _source(wl)
    monitor = _dft_monitor(sim)
    pml = PML(target_reflection=1e-6)

    assert is_dataclass(source)
    assert is_dataclass(ModeSource)
    assert is_dataclass(FieldMonitor)
    assert is_dataclass(pml)
    assert is_dataclass(PEC())

    with pytest.raises(FrozenInstanceError):
        source.signal = _signal()
    with pytest.raises(FrozenInstanceError):
        monitor.freqs = np.asarray([2.2e14])
    with pytest.raises(FrozenInstanceError):
        pml.target_reflection = 1e-4

    changed_source = source.updated_copy(width=2.0 * source.width)
    changed_monitor = monitor.updated_copy(freqs=[2.2e14])
    changed_pml = pml.updated_copy(target_reflection=1e-4)

    assert changed_source.width == 2.0 * source.width
    assert source.width != changed_source.width
    assert np.asarray(changed_monitor.freqs)[0] == pytest.approx(2.2e14)
    assert np.asarray(monitor.freqs)[0] == pytest.approx(2.0e14)
    assert changed_pml.target_reflection == pytest.approx(1e-4)
    assert pml.target_reflection == pytest.approx(1e-6)


def test_array_inputs_are_copied_and_marked_readonly():
    signal = np.arange(4.0, dtype=np.float32)
    source = GaussianSource(position=(0.0, 0.0), width=1.0, signal=signal)
    signal[0] = 99.0

    assert np.asarray(source.signal)[0] == pytest.approx(0.0)
    assert not source.signal.flags.writeable
    with pytest.raises(ValueError):
        source.signal[0] = 1.0


def test_design_and_geometry_inputs_are_immutable_specs():
    background = Material(permittivity=1.0)
    core = Material(permittivity=12.0)
    box = Box(center=(0.0, 0.0, 0.0), size=(1.0, 0.4, 0.2), material=core)
    rect = Rectangle(position=(0.0, 0.0), width=1.0, height=0.5, material=core)

    assert is_dataclass(background)
    assert is_dataclass(box)
    assert is_dataclass(rect)

    with pytest.raises(FrozenInstanceError):
        background.permittivity = 2.0
    with pytest.raises(FrozenInstanceError):
        box.center = (1.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        rect.material = background

    design = Design(width=2.0, height=2.0, depth=1.0, material=background)
    original = design
    design += box
    assert design.structures[-1] is box
    assert original.structures == ()
    with pytest.raises(FrozenInstanceError):
        design.background = core

    changed = box.updated_copy(material=background)
    assert changed.material is background
    assert box.material is core


def test_simulation_replaces_design_functionally():
    sim, _ = _make_2d_sim(plane_2d="xy", steps=5)
    original = sim.design
    rectangle = Rectangle(
        position=(0.0, 0.0),
        width=0.5,
        height=0.5,
        material=Material(2.0),
    )

    changed = sim.updated_copy(design=sim.design.with_structure(rectangle))

    assert changed.design is not original
    assert original.structures == ()
    assert changed.design.structures == (rectangle,)
    assert sim.design is original


def test_shape_updated_copy_preserves_common_polygon_fields():
    shapes = (
        Circle(position=(0.0, 0.0), radius=0.4, depth=0.2),
        Ring(position=(0.0, 0.0), inner_radius=0.2, outer_radius=0.4, depth=0.2),
        CircularBend(
            position=(0.0, 0.0), inner_radius=0.2, outer_radius=0.4, depth=0.2
        ),
        Taper(
            position=(0.0, 0.0),
            input_width=0.3,
            output_width=0.5,
            length=1.0,
            depth=0.2,
        ),
    )

    for shape in shapes:
        changed = shape.updated_copy(sidewall_angle=7.0, width_to_z=0.25)
        assert changed.sidewall_angle == pytest.approx(7.0)
        assert changed.width_to_z == pytest.approx(0.25)

    with pytest.raises(TypeError):
        shapes[0].updated_copy(not_a_field=True)


@pytest.mark.parametrize(
    "shape",
    [
        Polygon(vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        Rectangle(width=1.0, height=0.5),
        Circle(radius=0.5),
        Ring(inner_radius=0.25, outer_radius=0.5),
        CircularBend(inner_radius=0.25, outer_radius=0.5),
        Taper(input_width=0.25, output_width=0.5, length=1.0),
        Box(size=(1.0, 0.5, 0.25)),
        Sphere(radius=0.5),
    ],
)
def test_all_structure_specs_share_immutable_update_contract(shape):
    material = Material(permittivity=2.0)

    changed = shape.updated_copy(material=material)

    assert changed is not shape
    assert changed.material is material
    assert shape.material is None
    assert not hasattr(shape, "__dict__")


@pytest.mark.parametrize(
    "size",
    [(-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (np.nan, 1.0, 1.0)],
)
def test_box_rejects_invalid_size_before_derived_geometry(size):
    with pytest.raises(ValueError, match="non-negative"):
        Box(size=size)


def test_design_copy_reuses_immutable_spec():
    background = Material(permittivity=1.0)
    core = Material(permittivity=12.0)
    design = Design(width=2.0, height=2.0, depth=1.0, material=background)
    design += Box(center=(0.0, 0.0, 0.0), size=(1.0, 0.5, 0.2), material=core)

    copied = design.copy()

    assert copied is design
    assert copied.background is background
    assert copied.structures == (design.structures[0],)


def test_simulation_snapshots_time_input_as_readonly_array():
    time = np.linspace(0.0, 1.0e-15, 4)
    sim = Simulation(
        design=Design(width=1.0, height=1.0, material=Material(1.0)),
        sources=[],
        monitors=[],
        time=time,
    )
    time[0] = 99.0

    assert sim.time[0] == pytest.approx(0.0)
    assert not sim.time.flags.writeable
    with pytest.raises(ValueError):
        sim.time[0] = 1.0


def test_material_grid_cache_tracks_functional_raster_material_updates():
    simulation_core._MATERIAL_GRID_CACHE.clear()
    design = Design(width=1.0, height=1.0, material=Material(permittivity=1.0))
    grid = design.rasterize(0.5)
    with pytest.raises(TypeError, match="canonical Design"):
        Simulation(
            design=grid,
            resolution=0.5,
            time=np.array([0.0, 1e-16, 2e-16]),
        )


def test_material_grid_is_reused_by_runtime_and_compile_cache(monkeypatch):
    sim, wl = _make_2d_sim(plane_2d="xy", steps=5)
    simulation_core._MATERIAL_GRID_CACHE.clear()
    calls = _count_material_builds(monkeypatch)

    first = sim.compile(num_steps=2)
    assert sim.compile(num_steps=2) is first
    assert sim.initial_state().current_step == 0
    assert len(calls) == 1

    changed = sim.updated_copy(sources=(_source(wl),))
    assert changed.compile(num_steps=2) is not first
    assert len(calls) == 1


def test_material_grid_cache_tracks_only_physical_grid_inputs(monkeypatch):
    sim, _ = _make_2d_sim(plane_2d="xy", steps=5)
    simulation_core._MATERIAL_GRID_CACHE.clear()
    calls = _count_material_builds(monkeypatch)

    sim.to_request(num_steps=1)
    shifted = sim.updated_copy(time=sim.time + sim.dt)
    shifted.to_request(num_steps=1)
    assert len(calls) == 1

    refined = sim.updated_copy(resolution=sim.resolution * 1.1)
    refined.to_request(num_steps=1)
    assert len(calls) == 2


def test_monitor_subclass_config_fields_are_immutable_and_cacheable():
    flux = FluxMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        freqs=[2.0e14],
        name="flux",
    )
    mode = ModeMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        freqs=[2.0e14],
        mode_spec=ModeSpec(polarization="te"),
        name="mode",
    )

    with pytest.raises(FrozenInstanceError):
        flux.center = (1.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        mode.mode_spec = ModeSpec(polarization="tm")

    changed = mode.updated_copy(mode_spec=ModeSpec(polarization="tm"))
    assert changed.mode_spec.polarization == "tm"
    assert mode.mode_spec.polarization == "te"
    assert cache_token(changed) != cache_token(mode)


def test_mode_source_config_is_immutable_and_shifted_copy_is_state_free():
    signal = np.arange(4.0, dtype=np.float32)
    source = ModeSource(
        center=(0.0, 0.5, 0.0),
        size=(0.0, 0.25, 1.0),
        source_time=SampledSignal(signal, dt=1e-16, freq0=2e14),
        direction="+",
        mode_spec=ModeSpec(polarization="te"),
    )
    signal[0] = 99.0

    assert np.asarray(source.source_time.values)[0] == pytest.approx(0.0)
    assert not source.source_time.values.flags.writeable
    with pytest.raises(FrozenInstanceError):
        source.center = (1.0, 0.5)

    shifted = source.shifted((1.0, 2.0, 0.0))
    assert shifted.center == pytest.approx((1.0, 2.5, 0.0))
    assert source.center == pytest.approx((0.0, 0.5, 0.0))
    assert shifted is not source
    assert "_state" not in source.__dict__
    assert "_state" not in shifted.__dict__

    monitor = FieldMonitor(
        center=(0.5, 0.0, 0.0), size=(1.0, 0.0, 0.0), freqs=(2e14,), name="m"
    )
    assert "_state" not in monitor.__dict__


def test_builtin_sources_are_the_canonical_compiler_specs():
    sim, wavelength = _make_2d_sim(plane_2d="xy", steps=3)
    request = sim.updated_copy(sources=(_source(wavelength),)).to_request(num_steps=1)

    assert all(isinstance(source, GaussianSource) for source in request.sources)


def test_builtin_visual_specs_match_overlay_geometry():
    sim, wl = _make_2d_sim(plane_2d="xy", steps=5)
    source = _source(wl)
    monitor = _dft_monitor(sim)

    source_spec = visual_spec_from_device(source)
    monitor_spec = visual_spec_from_device(monitor)

    assert source_spec is not None
    assert source_spec.kind == "source"
    assert source_spec.center == pytest.approx(source.position[:2])
    assert source_spec.size == pytest.approx((source.width, source.width))
    assert monitor_spec is not None
    assert monitor_spec.kind == "monitor-line"
    assert monitor_spec.style["start"] == pytest.approx(monitor.start[:2])
    assert monitor_spec.style["end"] == pytest.approx(monitor.end[:2])
    with pytest.raises(TypeError):
        monitor_spec.style["start"] = (0.0, 0.0)


def _program_key(sim, *, num_steps=2):
    return CompiledProgramKey.from_request(
        sim.to_request(
            num_steps=num_steps,
            loop_kind="scan",
            source_single_slab_dense=False,
            sharding=sharding_cache_token(None),
        )
    )


def test_compiled_program_cache_is_bounded_and_uses_lru_order():
    from beamz.simulation import compile as compile_module

    sim, _ = _make_2d_sim(plane_2d="xy", steps=6)
    sim.clear_compiled_cache()
    built = []

    def compile_factory(request):
        program = object()
        built.append(program)
        return program

    def compile_steps(num_steps):
        from beamz.simulation.compile import compile_program

        return compile_program(
            sim,
            num_steps=num_steps,
            setup_context_factory=lambda _device: nullcontext(),
            compile_factory=compile_factory,
        )

    programs = {steps: compile_steps(steps) for steps in range(1, 5)}
    assert compile_steps(1) is programs[1]

    programs[5] = compile_steps(5)

    assert len(compile_module._PROGRAM_CACHE) == 4
    assert [key.num_steps for key in compile_module._PROGRAM_CACHE] == [3, 4, 1, 5]
    assert programs[2] not in compile_module._PROGRAM_CACHE.values()
    assert compile_steps(2) is not programs[2]
    assert len(built) == 6


def test_public_compiled_cache_clear_clears_all_derived_state():
    from beamz.simulation import compile as compile_module
    from beamz.simulation import execute as execute_module

    sim, _ = _make_2d_sim(plane_2d="xy", steps=5)
    token = cache_token(sim)
    program = sim.compile(num_steps=2)
    execute_module.execution_cache(program)

    assert compile_module._PROGRAM_CACHE
    assert simulation_core._MATERIAL_GRID_CACHE
    assert execute_module._EXECUTION_CACHES

    sim.clear_compiled_cache()

    assert not compile_module._PROGRAM_CACHE
    assert not simulation_core._MATERIAL_GRID_CACHE
    assert not execute_module._EXECUTION_CACHES
    assert cache_token(sim) == token


def test_cached_compiled_plans_are_deeply_immutable():
    sim, _ = _make_2d_sim(plane_2d="xy", steps=5)
    program = sim.compile(num_steps=2)

    with pytest.raises(FrozenInstanceError):
        program.monitors = program.monitors
    with pytest.raises(FrozenInstanceError):
        program.boundary.cpml.enabled = True
    with pytest.raises(TypeError):
        program.sharding.layout.logical_shapes["Ez"] = (1, 1)
    assert sim.compile(num_steps=2) is program


def test_compiled_execution_cache_is_external_to_immutable_plan():
    from beamz.simulation.execute import (
        build_program_scan,
        execution_cache,
        program_is_compiled,
    )

    sim, _ = _make_2d_sim(plane_2d="xy", steps=5)
    program = sim.compile(num_steps=2)
    cache = execution_cache(program)

    assert not hasattr(program, "_cache")
    assert cache.compiled_scan is None
    build_program_scan(program)
    assert execution_cache(program) is cache
    assert callable(cache.compiled_scan)
    assert program_is_compiled(program)


def _contains_identity(value, targets):
    if id(value) in targets:
        return True
    if isinstance(value, tuple | list | set | frozenset):
        return any(_contains_identity(v, targets) for v in value)
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        return any(
            _contains_identity(getattr(value, name), targets)
            for name in fields
            if hasattr(value, name)
        )
    return False


def test_simulation_to_request_carries_first_class_immutable_devices():
    sim, wl = _make_2d_sim(plane_2d="xy", steps=5)
    source = _source(wl)
    monitor = _dft_monitor(sim)
    sim = sim.updated_copy(sources=(source,), monitors=(monitor,))

    request = sim.to_request(
        num_steps=2,
        loop_kind="scan",
        source_single_slab_dense=False,
        sharding=sharding_cache_token(None),
    )
    assert isinstance(request, SimulationRequest)
    assert isinstance(hash(request), int)
    assert isinstance(request.materials, MaterialGrid)
    assert not hasattr(request, "compiled_grid")
    assert request.materials.shape == tuple(
        int(v) for v in np.asarray(request.materials.permittivity).shape
    )
    assert request.materials.resolution == sim.resolution
    assert not hasattr(request, "grid")
    assert request.run.num_steps == 2
    assert request.run.total_steps == sim.num_steps
    assert request.domain.size == tuple(float(v) for v in sim.size)
    assert request.sources and request.monitors
    assert not hasattr(request, "compiled_sources")
    assert not hasattr(request, "compiled_monitors")
    assert not hasattr(request, "compiler_inputs")
    assert not hasattr(request, "_compiler_inputs")
    assert len(request.sources) == 1
    assert isinstance(request.sources[0], GaussianSource)
    assert request.sources[0].position == source.position
    assert request.sources[0] is source
    assert request.monitors == (monitor,)
    assert request.monitors[0] is monitor
    with pytest.raises(FrozenInstanceError):
        request.monitors[0].name = "changed"
    assert not request.monitors[0].freqs.flags.writeable
    assert not _contains_identity(request.materials, {id(sim), id(source), id(monitor)})


def test_simulation_request_material_payloads_are_deeply_immutable():
    sim, _ = _make_2d_sim(
        plane_2d="xy",
        steps=3,
        boundaries=[PML(thickness=0.4 * TEST_WAVELENGTH, formulation="cpml")],
    )
    sim.clear_compiled_cache()
    request = sim.to_request(num_steps=1)

    for values in (
        request.materials.permittivity,
        request.materials.conductivity,
        request.materials.permeability,
    ):
        assert not np.asarray(values).flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 0.0
    grid = sim.compile(num_steps=1).grid
    assert grid.material_grid is request.materials

    # Rebuild the request's material value while leaving the equal compiled
    # program warm. Cache transparency must not depend on Python identity.
    simulation_core._MATERIAL_GRID_CACHE.clear()
    warm_request = sim.to_request(num_steps=1)
    warm_grid = sim.compile(num_steps=1).grid
    assert warm_grid is grid
    assert warm_grid.material_grid is not warm_request.materials
    # A warm compiled-program cache may hold an equal immutable MaterialGrid
    # produced by an earlier request. Value equality is the contract; Python
    # object identity is deliberately not observable across cache lifetimes.
    assert warm_grid.material_grid == warm_request.materials
    assert cache_token(warm_grid.material_grid) == cache_token(warm_request.materials)
    assert warm_grid.pml_data is not None
    with pytest.raises(TypeError):
        warm_grid.pml_data["new"] = np.zeros(1)
    for value in warm_grid.pml_data.values():
        if hasattr(value, "flags"):
            assert not value.flags.writeable


def test_compiled_cache_signature_changes_when_source_or_time_physics_changes():
    sim, wl = _make_2d_sim(plane_2d="xy", steps=5)
    sim = sim.updated_copy(sources=(_source(wl),))

    first = sim.compile(num_steps=2)
    changed = sim.updated_copy(sources=(_source(wl, x=1.35),))

    second = changed.compile(num_steps=2)
    assert second is not first
    shifted = changed.updated_copy(time=changed.time + changed.dt)
    assert shifted.compile(num_steps=2) is not second


def test_mode_source_cache_token_contains_only_canonical_fields():
    source = ModeSource(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        source_time=SampledSignal(np.ones(4), dt=1e-16, freq0=2e14),
        direction="+",
    )
    token = cache_token(source)
    assert "grid" not in repr(token)


def test_compiled_cache_signature_changes_when_source_waveform_replaced():
    sim, wl = _make_2d_sim(plane_2d="xy", steps=5)
    source = _source(wl)
    sim = sim.updated_copy(sources=(source,))

    first = sim.compile(num_steps=2)
    changed = sim.updated_copy(
        sources=(
            source.updated_copy(signal=np.asarray(_signal() + 0.25, dtype=np.float32)),
        )
    )

    assert changed.compile(num_steps=2) is not first


def test_compiled_cache_signature_changes_when_monitor_frequency_replaced():
    sim, _ = _make_2d_sim(plane_2d="xy", steps=5)
    monitor = _dft_monitor(sim)
    sim = sim.updated_copy(monitors=(monitor,))

    first = sim.compile(num_steps=2)
    changed = sim.updated_copy(
        monitors=(monitor.updated_copy(freqs=np.asarray([2.2e14])),)
    )

    assert changed.compile(num_steps=2) is not first


def test_compiled_program_key_changes_when_boundary_configuration_replaced():
    sim, _ = _make_2d_sim(
        plane_2d="xy",
        steps=5,
        boundaries=[PML(thickness=0.4 * TEST_WAVELENGTH, formulation="cpml")],
    )
    first = _program_key(sim)

    sim = sim.updated_copy(
        boundaries=(sim.boundaries[0].updated_copy(target_reflection=1e-4),)
    )

    assert _program_key(sim) != first


def test_compiled_program_key_changes_when_design_material_changes():
    sim, _ = _make_2d_sim(plane_2d="xy", steps=5)
    first = _program_key(sim)

    material = Material(permittivity=1.25)
    sim = sim.updated_copy(design=sim.design.updated_copy(background=material))

    assert _program_key(sim) != first


def test_compiled_cache_signature_ignores_equal_source_identity():
    sim, wl = _make_2d_sim(plane_2d="xy", steps=5)
    sim = sim.updated_copy(sources=(_source(wl),))

    first = sim.compile(num_steps=2)
    equivalent = sim.updated_copy(sources=(_source(wl),))

    assert equivalent.compile(num_steps=2) is first


def test_compiled_cache_signature_ignores_equal_monitor_identity():
    sim, _ = _make_2d_sim(plane_2d="xy", steps=5)
    sim = sim.updated_copy(monitors=(_dft_monitor(sim),))

    first = sim.compile(num_steps=2)
    equivalent = sim.updated_copy(monitors=(_dft_monitor(sim),))

    assert equivalent.compile(num_steps=2) is first


def test_boundaries_are_canonical_immutable_request_specs():
    sim, _ = _make_2d_sim(plane_2d="xy", steps=5)

    boundaries = sim.to_request().boundaries
    assert boundaries == tuple(sim.boundaries)
    assert all(
        isinstance(boundary, (PEC, PMC, PML, Absorber)) for boundary in boundaries
    )
    assert isinstance(hash(boundaries[0]), int)


def test_explicit_absorber_lowers_as_a_sponge_boundary():
    sim, wavelength = _make_2d_sim(plane_2d="xy", steps=2)
    absorber = Absorber(thickness=0.4 * wavelength)
    sim = sim.updated_copy(boundaries=(absorber,))

    assert sim.to_request(num_steps=1).boundaries == (absorber,)
    assert sim.pml_data["formulation"] == "sponge"
    assert sim.compile(num_steps=1).boundary.cpml.enabled is False


def test_compiled_program_key_is_hashable_with_sharding_token():
    sim, _ = _make_2d_sim(plane_2d="xy", steps=5)
    key = CompiledProgramKey.from_request(
        sim.to_request(
            num_steps=2,
            loop_kind="scan",
            source_single_slab_dense=False,
            sharding=sharding_cache_token({"axis": "x", "num_devices": 2}),
        )
    )
    assert key.sharding == (True, "x", 2, None)
    assert isinstance(hash(key), int)


def test_compiled_source_spec_matches_canonical_step_update_2d():
    reference, _ = _make_2d_sim(plane_2d="xy", steps=3)
    compiled, _ = _make_2d_sim(plane_2d="xy", steps=3)
    fields = reference.compile().grid
    source = _PointElectricCurrentSource(
        "Ez",
        _center_index(fields.Ez.shape),
        frequency_scale=2.0e15,
    )
    reference = reference.updated_copy(sources=(source.to_custom_spec(reference),))

    specs = tuple(
        spec
        for spec in compile_source_specs(
            (source.to_custom_spec(compiled),),
            fields,
            dt=compiled.dt,
            resolution=compiled.resolution,
            num_steps=compiled.num_steps,
            t0=float(compiled.time[0]),
            total_steps=compiled.num_steps,
        )
        if spec.timing == "e" and spec.component == "Ez"
    )

    reference_state = reference.step()
    applied = _apply_specs(
        fields.Ez,
        0,
        specs,
    )

    np.testing.assert_allclose(applied, reference_state.ez, atol=1e-12, rtol=0)


def test_source_lowerers_share_one_injection_plan_representation():
    sim, wavelength = _make_2d_sim(plane_2d="xy", steps=3)
    fields = sim.compile().grid
    custom = _PointElectricCurrentSource(
        "Ez", _center_index(fields.Ez.shape), frequency_scale=2.0e15
    ).to_custom_spec(sim)
    ctx = SourceLoweringContext(
        fields=fields,
        resolution=sim.resolution,
        dt=sim.dt,
        t0=float(sim.time[0]),
        num_steps=sim.num_steps,
        total_steps=sim.num_steps,
    )

    for source in (_source(wavelength), custom):
        plan = lower_source(source, ctx)
        assert isinstance(plan, CompiledInjectionPlan)
        assert plan.entries
        assert all(isinstance(entry, InjectionPlanEntry) for entry in plan.entries)
        assert all(
            isinstance(entry.profile, SpatialFieldProfile)
            and isinstance(entry.waveform, TemporalWaveform)
            and isinstance(entry.support, SourceSupport)
            for entry in plan.entries
        )


def test_compiled_step_rejects_multi_step_programs():
    from beamz.simulation.execute import step_program

    sim, _ = _make_2d_sim(plane_2d="xy", steps=3)
    program = sim.compile(num_steps=2)

    with pytest.raises(ValueError, match="num_steps=1"):
        step_program(program, None)


def test_step_recompiles_compiled_source_specs_after_source_replaced():
    mutated, wl = _make_2d_sim(plane_2d="xy", steps=4)
    constant, _ = _make_2d_sim(plane_2d="xy", steps=4)
    mutated_source = _source(wl)
    constant_source = _source(wl)
    mutated = mutated.updated_copy(sources=(mutated_source,))
    constant = constant.updated_copy(sources=(constant_source,))

    mutated_state = mutated.step()
    constant_state = constant.step()
    changed = mutated.updated_copy(
        sources=(
            mutated_source.updated_copy(
                signal=np.asarray(_signal() + 0.5, dtype=np.float32)
            ),
        )
    )
    mutated_state_2 = changed.step(mutated_state)
    constant_state_2 = constant.step(constant_state)

    assert mutated_state_2.current_step == constant_state_2.current_step == 2
    assert not np.allclose(
        np.asarray(mutated_state_2.ez), np.asarray(constant_state_2.ez)
    )


def test_simulation_owns_no_runtime_or_execution_state():
    sim, _ = _make_2d_sim(plane_2d="xy", steps=3)

    assert "_runtime" not in sim.__dict__
    assert "_pml_data" not in sim.__dict__
    assert "_compiled_program" not in sim.__dict__
    assert "engine" not in sim.__dict__
    assert "fields" not in sim.__dict__
    assert "runtime" not in sim.__dict__
    with pytest.raises(FrozenInstanceError):
        sim.sources = ()


def test_compiled_grid_is_deeply_immutable():
    sim, _ = _make_2d_sim(plane_2d="xy", steps=3)
    fields = sim.compile().grid

    with pytest.raises(AttributeError):
        fields.Ez = jnp.zeros_like(fields.Ez)
    with pytest.raises(TypeError):
        fields.metallic_masks["Ez"] = jnp.zeros_like(fields.Ez, dtype=bool)
    with pytest.raises(AttributeError):
        fields.materials.eps_ez = jnp.ones_like(fields.Ez)


@pytest.mark.parametrize("dimensions", (2, 3))
def test_compiled_pipeline_preserves_generic_runtime_invariants(dimensions):
    if dimensions == 2:
        _, wavelength = _make_2d_sim(plane_2d="xy", steps=3)
        sim, _ = _make_2d_sim(
            plane_2d="xy",
            steps=3,
            boundaries=[PML(thickness=0.4 * wavelength, formulation="cpml")],
        )
    else:
        _, wavelength = _make_3d_sim(steps=3)
        sim, _ = _make_3d_sim(
            steps=3,
            boundaries=[PML(thickness=0.4 * wavelength, formulation="cpml")],
        )

    program = sim.compile(num_steps=1)
    initial = sim.initial_state()
    initial_step = int(initial.current_step)
    initial_time = float(initial.t)
    initial_shapes = {
        component: getattr(initial, component.lower()).shape
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    advanced = sim.step(initial)

    assert isinstance(initial, SimulationState)
    assert isinstance(advanced, SimulationState)
    assert initial_step == 0
    assert int(advanced.current_step) == 1
    assert float(advanced.t) > initial_time
    assert program.boundary.cpml.enabled
    assert any(term.size > 0 for term in advanced.cpml_psi_h_terms)
    assert not hasattr(advanced, "fields")
    assert not hasattr(advanced, "material_state")

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        expected_shape = getattr(program.grid, component).shape
        assert initial_shapes[component] == expected_shape
        assert getattr(advanced, component.lower()).shape == expected_shape
