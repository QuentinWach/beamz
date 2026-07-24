"""Tests for dynamic, differentiable material inputs."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    Design,
    DomainFieldMonitor,
    FieldMonitor,
    GaussianSource,
    Material,
    ModeSpec,
    Port,
    Simulation,
    um,
)
from beamz.optimization import (
    DesignRegion,
    DifferentiablePortProjector,
    DifferentiableSimulation,
    InverseDesignProblem,
    PortSweepResult,
)
from beamz.optimization import problems as problem_module
from beamz.optimization.trainable import coefficients_for_permittivity

pytestmark = pytest.mark.unit


def test_design_region_materializes_storage_order_and_odd_symmetry():
    base = jnp.ones((5, 7))
    region = DesignRegion(
        lower=(1.0, 1.0),
        upper=(6.0, 5.0),
        eps_min=1.0,
        eps_max=4.0,
        symmetry="xy",
    )

    assert region.variable_shape(1.0, base.shape) == (2, 3)
    density = jnp.arange(6, dtype=float).reshape(2, 3) / 5.0
    materialized = np.asarray(region.materialize(base, density, 1.0))
    inserted = materialized[1:5, 1:6]

    np.testing.assert_allclose(inserted, np.flipud(inserted))
    np.testing.assert_allclose(inserted, np.fliplr(inserted))
    np.testing.assert_allclose(materialized[0], 1.0)


@pytest.mark.parametrize("depth", (0.0, 2.0))
def test_dynamic_coefficients_match_compiled_base_material(depth):
    design = Design(
        width=2.0,
        height=2.0,
        depth=depth,
        material=Material(permittivity=2.25),
    )
    sim = Simulation(
        design=design,
        resolution=1.0,
        time=np.array([0.0, 0.1]),
    )
    program = sim.compile()
    dynamic = coefficients_for_permittivity(program, program.grid.permittivity)

    if depth:
        names = ("e_permittivity_x", "e_permittivity_y", "e_permittivity_z")
    else:
        names = ("e_decay_z", "e_source_z")
    for name in names:
        np.testing.assert_allclose(
            getattr(dynamic, name), getattr(program.coefficients, name)
        )


def _small_differentiable_simulation():
    wavelength = 1.0 * um
    frequency = LIGHT_SPEED / wavelength
    resolution = 0.2 * um
    dt = 0.45 * resolution / LIGHT_SPEED
    time = np.arange(80, dtype=float) * dt
    signal = np.sin(2.0 * np.pi * frequency * time) * np.hanning(time.size)
    sim = Simulation(
        design=Design(
            width=2.0 * um,
            height=2.0 * um,
            material=Material(permittivity=1.0),
        ),
        sources=[
            GaussianSource(
                position=(0.4 * um, 1.0 * um),
                width=0.18 * um,
                signal=signal,
            )
        ],
        monitors=[
            FieldMonitor(
                center=(1.6 * um, 1.0 * um, 0.0),
                size=(0.0, 1.2 * um, 0.0),
                freqs=np.array([frequency]),
                fields=("Ez",),
                name="output",
            )
        ],
        resolution=resolution,
        time=time,
        normalize_source=None,
    )
    region = DesignRegion(
        lower=(0.8 * um, 0.6 * um),
        upper=(1.2 * um, 1.4 * um),
        eps_min=1.0,
        eps_max=4.0,
    )
    return DifferentiableSimulation(sim, region)


def test_differentiable_simulation_gradient_matches_finite_difference():
    trainable = _small_differentiable_simulation()
    density = jnp.full(trainable.variable_shape, 0.35)

    def objective(result):
        field = result.field("output", "Ez")
        return jnp.mean(jnp.abs(field) ** 2)

    compiled_value_and_grad = trainable.compile_value_and_grad(objective)
    value, gradient = compiled_value_and_grad(density)
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert np.max(np.abs(gradient)) > 0.0

    index = np.unravel_index(np.argmax(np.abs(np.asarray(gradient))), gradient.shape)
    step = 2e-3
    plus = density.at[index].add(step)
    minus = density.at[index].add(-step)
    finite_difference = (
        objective(trainable.run(plus)) - objective(trainable.run(minus))
    ) / (2.0 * step)

    np.testing.assert_allclose(
        np.asarray(gradient[index]),
        np.asarray(finite_difference),
        rtol=3e-2,
        atol=max(1e-12, 1e-4 * abs(float(finite_difference))),
    )

    # The same compiled executable accepts later density values without
    # rebuilding the transformed objective.
    second_value, second_gradient = compiled_value_and_grad(density + 1e-3)
    assert np.isfinite(second_value)
    assert np.all(np.isfinite(second_gradient))


def test_domain_field_monitor_returns_one_complex_value_per_material_cell():
    wavelength = 1.0 * um
    frequency = LIGHT_SPEED / wavelength
    resolution = 0.25 * um
    dt = 0.4 * resolution / LIGHT_SPEED
    time = np.arange(24, dtype=float) * dt
    sim = Simulation(
        design=Design(
            width=1.0 * um,
            height=0.75 * um,
            material=Material(permittivity=1.0),
        ),
        sources=[
            GaussianSource(
                position=(0.25 * um, 0.25 * um),
                width=0.2 * um,
                signal=np.sin(2.0 * np.pi * frequency * time),
            )
        ],
        monitors=[DomainFieldMonitor([frequency], fields=("Ez",), name="domain")],
        resolution=resolution,
        time=time,
        normalize_source=None,
    )

    results = sim.run()
    field = results["domain"].get_dft_component("Ez")
    assert results.metadata.fields.grid_shape == (3, 4)
    assert field.shape == (1, 12)
    assert np.all(np.isfinite(field))


def test_3d_differentiable_material_path_reaches_monitor_objective():
    resolution = 0.5 * um
    frequency = LIGHT_SPEED / (2.0 * um)
    dt = 0.25 * resolution / LIGHT_SPEED
    time = np.arange(20, dtype=float) * dt
    sim = Simulation(
        design=Design(
            width=1.5 * um,
            height=1.5 * um,
            depth=1.5 * um,
            material=Material(permittivity=1.0),
        ),
        sources=[
            GaussianSource(
                position=(0.5 * um, 0.75 * um, 0.75 * um),
                width=0.3 * um,
                signal=np.sin(2.0 * np.pi * frequency * time),
            )
        ],
        monitors=[
            FieldMonitor(
                center=(1.0 * um, 0.75 * um, 0.75 * um),
                size=(0.0, 1.0 * um, 1.0 * um),
                freqs=[frequency],
                fields=("Ez",),
                name="output",
            )
        ],
        resolution=resolution,
        time=time,
        normalize_source=None,
    )
    trainable = DifferentiableSimulation(
        sim,
        DesignRegion(
            lower=(0.5 * um, 0.5 * um, 0.5 * um),
            upper=(1.0 * um, 1.0 * um, 1.0 * um),
            eps_min=1.0,
            eps_max=2.0,
        ),
    )

    def objective(result):
        return jnp.mean(jnp.abs(result.field("output", "Ez")) ** 2)

    value, gradient = trainable.value_and_grad(
        jnp.full(trainable.variable_shape, 0.3), objective
    )
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert np.max(np.abs(gradient)) > 0.0


def test_inverse_design_problem_stacks_dense_port_sweeps(monkeypatch):
    frequencies = np.array([1.0, 2.0])

    class FakeMonitorResult:
        def get_dft_frequencies(self):
            return np.array([3.0])

        def get_dft_component(self, component):
            assert component == "Ez"
            return np.arange(4, dtype=float).reshape(1, 4)

    class FakeRun:
        metadata = SimpleNamespace(fields=SimpleNamespace(grid_shape=(2, 2)))

        def __getitem__(self, name):
            assert name == "domain"
            return FakeMonitorResult()

    class FakeTrainable:
        variable_shape = (2, 2)

        def __init__(self, simulation, design_region, *, rematerialize):
            del simulation, design_region, rematerialize

        def run_results(self, density):
            assert np.shape(density) == self.variable_shape
            return FakeRun()

    def fake_s_parameters(result, source_port, ports, output_ports):
        del result, ports
        return SimpleNamespace(
            frequencies=frequencies,
            s_matrix={
                (output, source_port): np.full(
                    frequencies.shape,
                    1.0 + output_ports.index(output),
                    dtype=np.complex128,
                )
                for output in output_ports
            },
        )

    monkeypatch.setattr(problem_module, "DifferentiableSimulation", FakeTrainable)
    monkeypatch.setattr(problem_module, "extract_s_parameters", fake_s_parameters)
    ports = (
        Port(
            center=(0.0, 0.5, 0.0),
            size=(0.0, 1.0, 1.0),
            name="p1",
            direction="+",
            mode_spec=ModeSpec(polarization="tm"),
        ),
        Port(
            center=(0.5, 0.0, 0.0),
            size=(1.0, 0.0, 1.0),
            name="p2",
            direction="+",
            mode_spec=ModeSpec(polarization="tm"),
        ),
    )
    problem = InverseDesignProblem(
        {"p1": object(), "p2": object()},
        ports,
        DesignRegion((0.0, 0.0), (2.0, 2.0), 1.0, 2.0),
        field_monitor="domain",
    )

    sweep = problem.run(np.ones((2, 2)))

    assert sweep.s_parameters.shape == (2, 2, 2)
    assert sweep.field_frequencies.shape == (1,)
    np.testing.assert_allclose(sweep.field_frequencies, [3.0])
    assert sweep.fields.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(sweep.s_parameters[:, :, 0], 1.0)
    np.testing.assert_allclose(sweep.s_parameters[:, :, 1], 2.0)
    assert not sweep.s_parameters.flags.writeable
    assert not sweep.field_frequencies.flags.writeable
    assert not sweep.fields.flags.writeable


def test_port_sweep_result_preserves_original_positional_constructor():
    result = PortSweepResult(
        np.asarray([1.0]),
        ("port1",),
        ("port1",),
        np.zeros((1, 1, 1), dtype=np.complex64),
        None,
        {},
    )

    assert result.field_frequencies is None


def test_differentiable_port_projector_uses_physical_wave_branches(monkeypatch):
    frequency = 2.0
    monitor1 = SimpleNamespace(name="p1")
    monitor2 = SimpleNamespace(name="p2")
    inputs = {
        "p1": SimpleNamespace(
            is_3d=False,
            frequencies=np.array([frequency]),
            dt=0.0,
            monitor_geometry=monitor1,
        ),
        "p2": SimpleNamespace(
            is_3d=False,
            frequencies=np.array([frequency]),
            dt=0.0,
            monitor_geometry=monitor2,
        ),
    }

    def fake_projection(data, port, monitor, frequency_arg, cache):
        del data, monitor, cache
        assert frequency_arg == frequency
        return {
            "components": ("Ez", "Hy" if port.axis == "x" else "Hx"),
            "pinv": np.eye(2, dtype=np.complex64),
            "modal_plane_delay_s": 0.0,
        }

    monkeypatch.setattr(problem_module, "analysis_inputs", lambda results: inputs)
    monkeypatch.setattr(problem_module, "build_port_projection", fake_projection)
    ports = (
        Port(
            center=(0.0, 0.5, 0.0),
            size=(0.0, 1.0, 1.0),
            name="p1",
            direction="+",
            mode_spec=ModeSpec(polarization="tm"),
        ),
        Port(
            center=(0.5, 0.0, 0.0),
            size=(1.0, 0.0, 1.0),
            name="p2",
            direction="+",
            mode_spec=ModeSpec(polarization="tm"),
        ),
    )
    projector = DifferentiablePortProjector(object(), ports)
    field_values = {
        ("p1", "Ez"): jnp.array([[2.0 + 0.0j]]),
        ("p1", "Hy"): jnp.array([[1.0 + 0.0j]]),
        ("p2", "Ez"): jnp.array([[3.0 + 0.0j]]),
        ("p2", "Hx"): jnp.array([[0.0 + 0.0j]]),
    }
    result = SimpleNamespace(
        field=lambda monitor, component: field_values[monitor, component]
    )

    s11 = projector.s_parameter(result, source_port="p1", output_port="p1")
    s21 = projector.s_parameter(result, source_port="p1", output_port="p2")

    np.testing.assert_allclose(s11, [0.5])
    np.testing.assert_allclose(s21, [1.5])
