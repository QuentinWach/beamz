from __future__ import annotations

import numpy as np
import pytest

import beamz as bz
from beamz.const import LIGHT_SPEED


def _te_simulation(*, cpml: bool = False):
    resolution = 0.1 * bz.um
    dt = 0.35 * resolution / (LIGHT_SPEED * np.sqrt(2.0))
    time = np.arange(12) * dt
    design = bz.Design(width=1.0 * bz.um, height=0.8 * bz.um)
    boundaries = [bz.PML(thickness=0.2 * bz.um, formulation="cpml")] if cpml else None
    return bz.Simulation(
        design=design,
        resolution=resolution,
        time=time,
        polarization="te",
        boundaries=boundaries,
        sources=[
            bz.GaussianSource(
                position=(0.5 * bz.um, 0.4 * bz.um),
                width=0.1 * bz.um,
                signal=np.r_[1.0, np.zeros(time.size - 1)],
            )
        ],
        monitors=[
            bz.FieldRecorder(("Ex", "Ey", "Hz"), interval=1, name="fields"),
            bz.FieldMonitor(
                center=(0.5 * bz.um, 0.4 * bz.um, 0.0),
                size=(0.0, 0.4 * bz.um, 1.0),
                freqs=[2.0e14],
                fields=("Ex", "Ey", "Hz"),
                name="dft",
            ),
            bz.FluxMonitor(
                center=(0.5 * bz.um, 0.4 * bz.um, 0.0),
                size=(0.0, 0.4 * bz.um, 1.0),
                freqs=[2.0e14],
                name="flux",
            ),
        ],
    )


def test_simulation_defaults_to_tm_and_rejects_invalid_polarization():
    time = np.arange(2) * 1e-16
    assert bz.Simulation(domain=(1e-6, 1e-6), time=time).polarization == "tm"
    with np.testing.assert_raises_regex(ValueError, "polarization"):
        bz.Simulation(domain=(1e-6, 1e-6), time=time, polarization="linear")
    with pytest.raises(ValueError, match="only to 2D"):
        bz.Simulation(domain=(1e-6, 1e-6, 1e-6), time=time, polarization="te")


def test_te_simulation_uses_only_te_yee_supports_and_records_public_fields():
    simulation = _te_simulation()
    program = simulation.compile()

    assert simulation._material_grid().polarization == "te"
    assert program.config.polarization_2d == "te"
    assert program.grid.Ex.shape == (9, 10)
    assert program.grid.Ey.shape == (8, 11)
    assert program.grid.Hz.shape == (8, 10)
    assert (
        program.grid.Ez.shape
        == program.grid.Hx.shape
        == program.grid.Hy.shape
        == (
            1,
            1,
        )
    )

    results = simulation.run()
    fields = results["fields"].fields
    assert set(fields) == {"Ex", "Ey", "Hz"}
    assert max(float(np.max(np.abs(values))) for values in fields.values()) > 0.0
    assert set(results["dft"].dft_fields) == {"Ex", "Ey", "Hz"}
    assert results.metadata.polarization_2d == "te"
    assert np.all(np.isfinite(results["flux"].power_spectrum))


def test_te_domain_simulation_accepts_a_two_coordinate_gaussian_source():
    resolution = 0.1 * bz.um
    dt = 0.35 * resolution / (LIGHT_SPEED * np.sqrt(2.0))
    simulation = bz.Simulation(
        domain=(1.0 * bz.um, 0.8 * bz.um),
        resolution=resolution,
        time=np.arange(4) * dt,
        polarization="te",
        sources=[
            bz.GaussianSource(
                position=(0.0, 0.0),
                width=0.1 * bz.um,
                signal=[1.0, 0.0, 0.0, 0.0],
            )
        ],
    )

    assert simulation.sources[0].position == (0.5 * bz.um, 0.4 * bz.um)
    assert {source.component for source in simulation.compile().sources} == {"Hz"}


def test_te_cpml_uses_te_staggered_memory_and_runs():
    simulation = _te_simulation(cpml=True)
    program = simulation.compile()

    assert program.boundary.cpml.enabled
    assert len(program.boundary.cpml.h_terms) == 2
    assert len(program.boundary.cpml.e_terms) == 2
    assert "te_xy_cpml" in program.grid.pml_data
    assert "tm_xy_cpml" not in program.grid.pml_data
    run = simulation.advance()
    assert int(run.state.current_step) == simulation.num_steps
    assert len(run.state.cpml_psi_h_terms) == 2
    assert len(run.state.cpml_psi_e_terms) == 2
    assert all(np.all(np.isfinite(value)) for value in run.state.cpml_psi_h_terms)
    assert all(np.all(np.isfinite(value)) for value in run.state.cpml_psi_e_terms)


def test_te_simulation_rejects_tm_mode_monitor():
    monitor = bz.ModeMonitor(
        center=(0.5 * bz.um, 0.4 * bz.um, 0.0),
        size=(0.0, 0.4 * bz.um, 1.0),
        freqs=[2.0e14],
        mode_spec=bz.ModeSpec(polarization="tm"),
    )
    simulation = _te_simulation().updated_copy(monitors=(monitor,))

    with pytest.raises(ValueError, match="does not match"):
        simulation.compile()
