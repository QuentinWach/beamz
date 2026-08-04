"""End-to-end validation for bounded simulation execution."""

import numpy as np
import pytest

from beamz import (
    EPS_0,
    LIGHT_SPEED,
    PML,
    AutoTermination,
    Design,
    FieldMonitor,
    GaussianSource,
    Material,
    Simulation,
    calc_optimal_fdtd_params,
    um,
)
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import Grid as RasterGrid
from beamz.design.raster import Material as RasterMaterial
from beamz.design.raster import RasterOptions, Scene, rasterize
from beamz.simulation.execute import _energy_terms, _field_diagnostics

pytestmark = [pytest.mark.compiled, pytest.mark.component]


def test_full_tensor_energy_includes_offdiagonal_coupling():
    """Integrated energy must use the full supported electric constitutive tensor."""
    permittivity = np.asarray(((3.0, 0.2, 0.0), (0.2, 2.0, 0.0), (0.0, 0.0, 1.0)))
    material_grid = MaterialGrid.from_raster_result(
        rasterize(
            Scene((RasterMaterial(epsilon_r=permittivity),)),
            RasterGrid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2)),
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    simulation = Simulation(
        material_grid=material_grid,
        time=np.asarray([0.0, 1e-16]),
    )
    program = simulation.compile(num_steps=1)
    initial = simulation.initial_state()
    state = initial._replace(
        ex=np.ones_like(initial.ex),
        ey=np.ones_like(initial.ey),
        ez=np.ones_like(initial.ez),
    )

    energy, max_field, finite = _field_diagnostics(state, _energy_terms(program))

    electric_field = np.ones(3)
    expected = 0.5 * EPS_0 * (electric_field @ permittivity @ electric_field)
    assert finite
    assert max_field == pytest.approx(1.0)
    assert energy == pytest.approx(expected, rel=2e-6)


def test_pulse_convergence_preserves_frequency_domain_result():
    """A radiated pulse may stop early without changing its converged DFT."""
    wavelength = 1.0 * um
    resolution, dt = calc_optimal_fdtd_params(
        wavelength,
        1.0,
        dims=2,
        points_per_wavelength=10,
    )
    num_steps = 420
    time = np.arange(num_steps, dtype=float) * dt
    step = np.arange(num_steps)
    frequency = LIGHT_SPEED / wavelength
    signal = np.exp(-0.5 * ((step - 24) / 7) ** 2) * np.cos(
        2.0 * np.pi * frequency * time
    )
    signal[65:] = 0.0

    monitor = FieldMonitor(
        center=(6 * um, 4 * um, 0.0),
        size=(0.0, 4 * um, 0.0),
        freqs=(frequency,),
        fields=("Ez",),
        name="output",
    )
    simulation = Simulation(
        design=Design(
            width=8 * um,
            height=8 * um,
            material=Material(permittivity=1.0),
        ),
        sources=(
            GaussianSource(
                position=(2 * um, 4 * um),
                width=0.3 * um,
                signal=signal,
            ),
        ),
        monitors=(monitor,),
        boundaries=(PML(thickness=wavelength),),
        time=time,
        resolution=resolution,
    )
    policy = AutoTermination(
        field_decay=1e-5,
        monitor_change=1e-3,
        source_decay=1e-6,
        chunk_steps=30,
        consecutive_checks=3,
        monitor_names=("output",),
    )

    bounded = simulation.run(progress=False, termination=policy)
    reference = simulation.run(progress=False)

    report = bounded.termination
    assert report is not None
    assert report.reason == "converged"
    assert report.converged
    assert report.steps < num_steps
    assert report.source_decay == pytest.approx(0.0)
    assert report.field_decay <= policy.field_decay
    assert report.monitor_change is not None
    assert report.monitor_change <= policy.monitor_change

    bounded_field = bounded["output"].get_dft_component("Ez")
    reference_field = reference["output"].get_dft_component("Ez")
    relative_error = np.linalg.norm(bounded_field - reference_field) / max(
        np.linalg.norm(bounded_field), np.linalg.norm(reference_field)
    )
    assert relative_error < 5e-3
    np.testing.assert_allclose(
        bounded["output"].dft_weight_sum,
        reference["output"].dft_weight_sum,
    )
