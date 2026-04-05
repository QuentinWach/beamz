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
from beamz.devices.sources.gaussian import GaussianSource
from beamz.devices.sources.mode import ModeSource
from beamz.devices.sources.spec import GaussianSourceSpec, ModeSourceSpec
from beamz.simulation.boundaries import Boundary, PML
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


def test_pml_rejects_nonpositive_sigma_max():
    with pytest.raises(ValueError):
        PML(thickness=1.0, sigma_max=0.0)


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

    assert sim.runtime.initialized is False
    assert sim.runtime.fields is None

    assert sim.dt == 1.0
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
    assert sim2.spec.boundaries == tuple(sim2.boundaries)


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
    assert sim.spec.boundaries == (boundary,)
