from __future__ import annotations

import inspect

import numpy as np
import pytest

import beamz
import beamz.design as design
import beamz.devices as devices
import beamz.devices.sources as sources
import beamz.simulation as simulation

pytestmark = pytest.mark.unit


STABLE_SIMULATION_EXPORTS = {
    "Absorber",
    "AutoTermination",
    "GaussianPulse",
    "GridSpec",
    "ModeSpec",
    "MonitorResults",
    "PEC",
    "PML",
    "Port",
    "RunTermination",
    "Simulation",
    "SimulationResults",
    "SimulationRun",
    "SimulationState",
    "inf",
}

REMOVED_UNUSED_HOOK_EXPORTS = {
    "BoundaryHookSpec",
    "MonitorHookSpec",
    "SnapshotHookSpec",
    "SolverPhaseProgram",
    "SourceHookSpec",
    "UpdateKernelSpec",
    "register_boundary_lowerer",
    "register_monitor_lowerer",
    "register_update_kernel",
}


def test_simulation_package_all_is_only_core_public_surface():
    assert set(simulation.__all__) == STABLE_SIMULATION_EXPORTS
    for name in STABLE_SIMULATION_EXPORTS:
        assert hasattr(simulation, name), name


def test_top_level_beamz_reexports_only_supported_simulation_api():
    top_level_exports = STABLE_SIMULATION_EXPORTS & set(beamz.__all__)

    assert top_level_exports
    for name in top_level_exports:
        if name == "inf":
            assert getattr(beamz, name) == getattr(simulation, name)
        else:
            assert getattr(beamz, name) is getattr(simulation, name), name


def test_compiled_engine_internals_are_not_public_package_exports():
    for name in (
        "CompiledProgram",
        "ShardingConfig",
        "compile_simulation",
    ):
        assert name not in simulation.__all__
        assert name not in beamz.__all__
        assert not hasattr(simulation, name)


def test_public_surface_excludes_internal_extension_hooks():
    for namespace in (beamz, simulation):
        for name in REMOVED_UNUSED_HOOK_EXPORTS:
            assert not hasattr(namespace, name), name


def test_public_specs_have_consistent_reexports():
    assert beamz.GridSpec is simulation.GridSpec is design.GridSpec
    assert beamz.GaussianPulse is simulation.GaussianPulse is sources.GaussianPulse
    assert beamz.ModeSpec is simulation.ModeSpec is sources.ModeSpec
    assert beamz.PEC is simulation.PEC is devices.PEC
    assert beamz.PML is simulation.PML is devices.PML
    assert beamz.Absorber is simulation.Absorber is devices.Absorber
    assert hasattr(design, "MaterialGrid")
    assert hasattr(design, "build_material_grid")


def test_public_simulation_methods_have_reference_docstrings():
    public_classes = (
        simulation.Simulation,
        simulation.AutoTermination,
        simulation.RunTermination,
        simulation.SimulationState,
        simulation.SimulationRun,
        simulation.SimulationResults,
        simulation.MonitorResults,
        simulation.GridSpec,
        simulation.Port,
        simulation.GaussianPulse,
        simulation.ModeSpec,
        simulation.PEC,
        simulation.PML,
        simulation.Absorber,
    )

    for cls in public_classes:
        assert inspect.getdoc(cls), cls.__name__
        for name, descriptor in vars(cls).items():
            if name.startswith("_"):
                continue
            if isinstance(descriptor, property):
                member = descriptor.fget
            elif isinstance(descriptor, (classmethod, staticmethod)):
                member = descriptor.__func__
            elif inspect.isfunction(descriptor):
                member = descriptor
            else:
                continue
            doc = inspect.getdoc(member) or ""
            assert doc, f"{cls.__name__}.{name}"
            assert "Returns\n-------" in doc, f"{cls.__name__}.{name}"


def test_execution_docstrings_explain_the_state_lifecycle():
    class_docs = "\n".join(
        inspect.getdoc(cls) or ""
        for cls in (
            simulation.Simulation,
            simulation.SimulationState,
            simulation.SimulationRun,
            simulation.SimulationResults,
        )
    )
    for term in ("run", "advance", "step", "donat", "continuation"):
        assert term in class_docs.lower()

    run_doc = inspect.getdoc(simulation.Simulation.run) or ""
    advance_doc = inspect.getdoc(simulation.Simulation.advance) or ""
    assert "normal user-facing execution method" in run_doc
    assert "results plus next state" in advance_doc
    assert "donate_state" in advance_doc
    assert "Examples\n--------" in run_doc
    assert "Examples\n--------" in advance_doc


def test_material_grid_has_array_aware_value_equality():
    first = design.MaterialGrid(
        permittivity=np.ones((2, 2)),
        conductivity=np.zeros((2, 2)),
        permeability=np.ones((2, 2)),
        resolution=0.5,
        shape=(2, 2),
    )
    equivalent = design.MaterialGrid(
        permittivity=np.ones((2, 2)),
        conductivity=np.zeros((2, 2)),
        permeability=np.ones((2, 2)),
        resolution=0.5,
        shape=(2, 2),
    )
    changed = design.MaterialGrid(
        permittivity=np.full((2, 2), 4.0),
        conductivity=np.zeros((2, 2)),
        permeability=np.ones((2, 2)),
        resolution=0.5,
        shape=(2, 2),
    )

    assert first == equivalent
    assert hash(first) == hash(equivalent)
    assert first != changed


def test_lowering_and_mutable_registry_internals_are_not_public():
    import beamz.devices.monitors as monitors
    from beamz.devices import visualization

    for name in (
        "CompiledSourceSpec",
        "SourceLoweringContext",
        "compile_source_specs",
        "has_source_lowerer",
        "lower_source",
    ):
        assert name not in sources.__all__
        assert not hasattr(sources, name)
    for name in ("CompiledMonitorSpec", "compile_monitor_specs"):
        assert name not in monitors.__all__
        assert not hasattr(monitors, name)
    assert not hasattr(visualization, "register_visual_spec_handler")
    assert "RegularGrid" not in design.__all__
    assert "RegularGrid" not in simulation.__all__
    assert "RegularGrid" not in beamz.__all__
