import json

import numpy as np
import pytest

from beamz import Design, Material, Simulation
from beamz.design.materials import (
    CustomMaterial,
    CustomMaterialSpec,
    MaterialSpec,
    material_to_spec,
)
from beamz.design.spec import DesignSpec
from beamz.design.structures import Rectangle, StructureSpec
from beamz.devices.monitors.monitors import Monitor
from beamz.devices.monitors.spec import MonitorSpec
from beamz.devices.sources.gaussian import GaussianSource
from beamz.devices.sources.mode import ModeSource
from beamz.devices.sources.spec import GaussianSourceSpec, ModeSourceSpec
from beamz.simulation.boundaries import Boundary, PML
from beamz.simulation.boundary_specs import BoundarySpec, PMLSpec
from beamz.simulation.spec import SimulationSpec


def test_material_spec_rejects_nonphysical_values():
    with pytest.raises(ValueError):
        MaterialSpec(permittivity=0.0)
    with pytest.raises(ValueError):
        MaterialSpec(permeability=-1.0)
    with pytest.raises(ValueError):
        MaterialSpec(conductivity=-0.1)


def test_custom_material_spec_requires_bounds_for_grids():
    grid = np.ones((4, 4))
    with pytest.raises(ValueError):
        CustomMaterialSpec(permittivity_grid=grid)


def test_custom_material_spec_rejects_invalid_interpolation():
    with pytest.raises(ValueError):
        CustomMaterialSpec(interpolation="cubic")


def test_design_spec_rejects_invalid_structures_and_depth():
    with pytest.raises(TypeError):
        DesignSpec(width=1.0, height=1.0, depth=0.0, structures=("bad",), is_3d=False)
    with pytest.raises(ValueError):
        DesignSpec(width=1.0, height=1.0, depth=-1.0, structures=(), is_3d=False)

    rect = Rectangle(position=(0, 0), width=1.0, height=1.0)
    spec = DesignSpec(width=1.0, height=1.0, depth=0.0, structures=(rect,), is_3d=False)
    assert isinstance(spec.structures[0], StructureSpec)
    assert spec.structures[0] == rect.spec


def test_gaussian_source_spec_requires_signal():
    with pytest.raises(ValueError):
        GaussianSourceSpec(position=(0.0, 0.0), width=1.0, signal=None)


def test_source_specs_reject_callable_signals():
    with pytest.raises(TypeError):
        GaussianSourceSpec(position=(0.0, 0.0), width=1.0, signal=lambda t: t)
    with pytest.raises(TypeError):
        ModeSourceSpec(
            center=(0.0, 0.0),
            width=1.0,
            height=None,
            wavelength=1.55,
            pol="tm",
            signal=lambda t: t,
            direction="+x",
            direction_axis="x",
            direction_sign=1.0,
        )


def test_mode_source_spec_rejects_invalid_center_shape():
    signal = np.ones(8)
    with pytest.raises(ValueError):
        ModeSourceSpec(
            center=(0.0,),
            width=1.0,
            height=None,
            wavelength=1.55,
            pol="tm",
            signal=signal,
            direction="+x",
            direction_axis="x",
            direction_sign=1.0,
        )


def test_boundary_rejects_invalid_edges():
    with pytest.raises(ValueError):
        Boundary(edges=("left", "invalid"), thickness=1.0)
    with pytest.raises(ValueError):
        BoundarySpec(edges=("left", "invalid"), thickness=1.0)


def test_pml_rejects_nonpositive_sigma_max():
    with pytest.raises(ValueError):
        PML(thickness=1.0, sigma_max=0.0)
    with pytest.raises(ValueError):
        PMLSpec(thickness=1.0, sigma_max=0.0)


def test_simulation_spec_validates_resolution_and_time():
    design = type("Design", (), {"is_3d": False, "depth": 0.0})()
    with pytest.raises(ValueError):
        SimulationSpec(
            design=design,
            devices=(),
            boundaries=(),
            resolution=0.0,
            time=np.array([0.0, 1.0]),
            plane_2d="xy",
            is_3d=False,
        )


def test_simulation_runtime_is_initialized_lazily():
    design = Design(width=1.0, height=1.0, material=Material(permittivity=1.0))
    sim = Simulation(
        design=design,
        devices=[],
        boundaries=[],
        resolution=0.1,
        time=np.array([0.0, 1.0, 2.0]),
    )

    assert sim.session.simulation is sim
    assert sim.runtime is sim.session.runtime
    assert sim.runtime.initialized is False
    assert sim.runtime.fields is None

    assert sim.dt == 1.0
    assert sim.session.dt == 1.0
    assert sim.runtime.initialized is True
    assert sim.runtime.fields is not None

    sim.time = np.array([0.0, 0.5, 1.0])
    assert sim.runtime.initialized is False
    assert sim.runtime.fields is None
    with pytest.raises(ValueError):
        SimulationSpec(
            design=design,
            devices=(),
            boundaries=(),
            resolution=1.0,
            time=np.array([0.0, 0.0]),
            plane_2d="xy",
            is_3d=False,
        )


def test_simulation_runtime_initializes_from_spec_graph():
    design = Design(width=1.0, height=1.0, material=Material(permittivity=1.0))
    sim = Simulation(
        design=design,
        devices=[],
        boundaries=[PML(thickness=0.1, sigma_max=2.0, m=3)],
        resolution=0.1,
        time=np.array([0.0, 1.0, 2.0]),
    )

    object.__setattr__(sim, "_design", None)
    object.__setattr__(sim, "_boundaries", ())

    assert sim.runtime.initialized is False
    assert sim.dt == 1.0
    assert sim.runtime.initialized is True
    assert sim.runtime.fields is not None
    assert sim.pml_data is not None
    assert "mask" in sim.pml_data


def test_with_spec_returns_updated_facade_copy():
    material = Material(permittivity=1.0)
    material2 = material.with_spec(permittivity=2.0)
    assert material.permittivity == 1.0
    assert material2.permittivity == 2.0

    design = Design(width=1.0, height=2.0, material=Material(permittivity=1.0))
    design2 = design.with_spec(width=3.0)
    assert design.width == 1.0
    assert design2.width == 3.0
    assert design2.spec.structures[0] == design2.structures[0].spec

    rect = Rectangle(position=(0.0, 0.0), width=1.0, height=2.0)
    rect2 = rect.with_spec(width=4.0)
    assert rect.width == 1.0
    assert rect2.width == 4.0

    signal = np.ones(8)
    monitor = Monitor(start=(0.0, 0.0), end=(1.0, 0.0), name="m0")
    monitor2 = monitor.with_spec(name="m1")
    assert monitor.name == "m0"
    assert monitor2.name == "m1"
    assert monitor2.state.fields["Ez"] == []

    source = GaussianSource(position=(0.0, 0.0), width=1.0, signal=signal)
    source2 = source.with_spec(width=2.0)
    assert source.width == 1.0
    assert source2.width == 2.0

    mode = ModeSource(
        None,
        center=(0.0, 0.0),
        width=1.0,
        wavelength=1.55,
        pol="tm",
        signal=signal,
        direction="+x",
    )
    mode2 = mode.with_spec(width=2.0)
    assert mode.width == 1.0
    assert mode2.width == 2.0

    sim = Simulation(
        design=design,
        devices=[],
        boundaries=[],
        resolution=0.1,
        time=np.array([0.0, 1.0, 2.0]),
    )
    sim2 = sim.with_spec(resolution=0.2)
    assert sim.resolution == 0.1
    assert sim2.resolution == 0.2
    assert sim2.runtime.initialized is False
    assert sim2.spec.design == sim2.design.spec
    assert sim2.spec.devices == tuple(device.spec for device in sim2.devices)
    assert sim2.spec.boundaries == tuple(boundary.spec for boundary in sim2.boundaries)


def test_callable_behavior_is_not_stored_in_specs():
    monitor = Monitor(start=(0.0, 0.0), end=(1.0, 0.0), objective_function=lambda m: 1.0)
    assert not hasattr(monitor.spec, "objective_function")
    assert callable(monitor.objective_function)

    material = CustomMaterial(permittivity_func=lambda x, y: 2.0)
    assert not hasattr(material.spec, "permittivity_func")
    assert callable(material.permittivity_func)


def test_structure_and_simulation_specs_store_nested_specs():
    material = Material(permittivity=2.5)
    rect = Rectangle(position=(0.0, 0.0), width=1.0, height=1.0, material=material)
    assert rect.spec.material == material.spec
    assert material_to_spec(rect.material) == rect.spec.material

    design = Design(width=2.0, height=2.0, material=material)
    monitor = Monitor(start=(0.0, 0.0), end=(1.0, 0.0))
    signal = np.ones(8)
    source = GaussianSource(position=(0.0, 0.0), width=1.0, signal=signal)
    boundary = PML(thickness=1.0)
    sim = Simulation(
        design=design,
        devices=[monitor, source],
        boundaries=[boundary],
        resolution=0.1,
        time=np.array([0.0, 1.0, 2.0]),
    )
    assert sim.spec.design == design.spec
    assert sim.spec.devices == (monitor.spec, source.spec)
    assert sim.spec.boundaries == (boundary.spec,)
    assert isinstance(sim.spec.boundaries[0], PMLSpec)


def test_simulation_accepts_boundary_specs_and_rebuilds_boundary_facades():
    design = Design(width=2.0, height=2.0, material=Material())
    boundary_spec = PMLSpec(thickness=1.0, sigma_max=2.0, m=4)
    sim = Simulation(
        design=design,
        devices=[],
        boundaries=[boundary_spec],
        resolution=0.1,
        time=np.array([0.0, 1.0, 2.0]),
    )

    assert sim.spec.boundaries == (boundary_spec,)
    assert len(sim.boundaries) == 1
    assert isinstance(sim.boundaries[0], PML)
    assert sim.boundaries[0].spec == boundary_spec


def test_spec_roundtrip_dicts_are_json_serializable():
    mat = MaterialSpec(permittivity=2.0, permeability=1.5, conductivity=0.1)
    mat_dict = mat.to_dict()
    assert MaterialSpec.from_dict(json.loads(json.dumps(mat_dict))) == mat

    cmat = CustomMaterialSpec(
        permittivity_grid=[[1.0, 2.0], [3.0, 4.0]],
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    cmat_dict = cmat.to_dict()
    cmat_rt = CustomMaterialSpec.from_dict(json.loads(json.dumps(cmat_dict)))
    assert np.allclose(cmat_rt.permittivity_grid, cmat.permittivity_grid)

    rect = Rectangle(position=(0.0, 0.0), width=1.0, height=2.0, material=Material())
    structure_dict = rect.spec.to_dict()
    structure_rt = StructureSpec.from_dict(json.loads(json.dumps(structure_dict)))
    assert structure_rt == rect.spec

    design = Design(width=2.0, height=3.0, material=Material())
    design_dict = design.spec.to_dict()
    design_rt = DesignSpec.from_dict(json.loads(json.dumps(design_dict)))
    assert design_rt == design.spec

    monitor = Monitor(start=(0.0, 0.0), end=(1.0, 0.0), dft_frequencies=[1.0, 2.0])
    monitor_dict = monitor.spec.to_dict()
    monitor_rt = MonitorSpec.from_dict(json.loads(json.dumps(monitor_dict)))
    assert monitor_rt.start == monitor.spec.start
    assert np.allclose(monitor_rt.dft_frequencies, monitor.spec.dft_frequencies)
    assert np.allclose(monitor_rt.frequency_points, monitor.spec.frequency_points)

    signal = np.linspace(0.0, 1.0, 8)
    gaussian = GaussianSourceSpec(position=(0.0, 0.0), width=1.0, signal=signal)
    gaussian_rt = GaussianSourceSpec.from_dict(json.loads(json.dumps(gaussian.to_dict())))
    assert np.allclose(gaussian_rt.signal, gaussian.signal)

    mode = ModeSourceSpec(
        center=(0.0, 0.0),
        width=1.0,
        height=None,
        wavelength=1.55,
        pol="tm",
        signal=signal,
        direction="+x",
        direction_axis="x",
        direction_sign=1.0,
    )
    mode_rt = ModeSourceSpec.from_dict(json.loads(json.dumps(mode.to_dict())))
    assert np.allclose(mode_rt.signal, mode.signal)

    pml = PML(thickness=1.0, sigma_max=2.0, m=4)
    pml_rt = PML.from_dict(json.loads(json.dumps(pml.to_dict())))
    assert pml_rt == pml
    pml_spec_rt = PMLSpec.from_dict(json.loads(json.dumps(pml.spec.to_dict())))
    assert pml_spec_rt == pml.spec

    sim = Simulation(
        design=design,
        devices=[monitor, GaussianSource(position=(0.0, 0.0), width=1.0, signal=signal)],
        boundaries=[pml],
        resolution=0.1,
        time=np.array([0.0, 1.0, 2.0]),
    )
    sim_rt = SimulationSpec.from_dict(json.loads(json.dumps(sim.spec.to_dict())))
    assert sim_rt.design == sim.spec.design
    assert len(sim_rt.devices) == len(sim.spec.devices)
    assert sim_rt.boundaries == sim.spec.boundaries
    assert np.allclose(sim_rt.time, sim.spec.time)


def test_design_facade_roundtrip_uses_serialized_spec_tree():
    material = Material(permittivity=2.0)
    design = Design(width=2.0, height=3.0, material=material)
    design += Rectangle(position=(0.5, 0.5), width=0.25, height=0.5, material=material)

    payload = json.loads(json.dumps(design.to_dict()))
    restored = Design.from_dict(payload)

    assert restored.spec == design.spec
    assert restored.width == design.width
    assert len(restored.structures) == len(design.structures)
    assert restored.structures[0].spec == design.structures[0].spec
    assert restored.structures[1].spec == design.structures[1].spec


def test_simulation_facade_roundtrip_rebuilds_live_objects():
    design = Design(width=2.0, height=2.0, material=Material())
    signal = np.linspace(0.0, 1.0, 8)
    monitor = Monitor(start=(0.0, 0.0), end=(1.0, 0.0), name="m0")
    source = GaussianSource(position=(0.0, 0.0), width=1.0, signal=signal)
    boundary = PML(thickness=1.0, sigma_max=2.0, m=4)
    sim = Simulation(
        design=design,
        devices=[monitor, source],
        boundaries=[boundary],
        resolution=0.1,
        time=np.array([0.0, 1.0, 2.0]),
    )

    payload = json.loads(json.dumps(sim.to_dict()))
    restored = Simulation.from_dict(payload)

    assert restored.to_dict() == sim.to_dict()
    assert restored.design.spec == sim.design.spec
    assert len(restored.devices) == 2
    assert isinstance(restored.devices[0], Monitor)
    assert isinstance(restored.devices[1], GaussianSource)
    assert restored.devices[0].to_dict() == monitor.to_dict()
    assert restored.devices[1].to_dict() == source.to_dict()
    assert restored.boundaries == sim.boundaries
    assert restored.spec.boundaries == tuple(boundary.spec for boundary in restored.boundaries)
    assert restored.runtime.initialized is False
