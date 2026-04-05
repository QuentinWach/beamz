import numpy as np
import pytest

from beamz import Design, Material, Simulation
from beamz.design.materials import CustomMaterialSpec, MaterialSpec
from beamz.design.spec import DesignSpec
from beamz.design.structures import Rectangle
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
    assert spec.structures == (rect,)


def test_gaussian_source_spec_requires_signal():
    with pytest.raises(ValueError):
        GaussianSourceSpec(position=(0.0, 0.0), width=1.0, signal=None)


def test_mode_source_spec_rejects_invalid_z_configuration():
    signal = np.ones(8)
    with pytest.raises(ValueError):
        ModeSourceSpec(
            center=(0.0, 0.0),
            width=1.0,
            height=None,
            wavelength=1.55,
            pol="tm",
            signal=signal,
            direction="+z",
            direction_axis="z",
            direction_sign=1.0,
        )
    with pytest.raises(ValueError):
        ModeSourceSpec(
            center=(0.0, 0.0, 0.0),
            width=1.0,
            height=None,
            wavelength=1.55,
            pol="tm",
            signal=signal,
            direction="+z",
            direction_axis="z",
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
