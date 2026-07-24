from pathlib import Path

import jax.numpy as jnp
import numpy as np

import beamz as bz
from beamz.optimization.challenges._common import paper_circular_brush
from beamz.optimization.challenges.wdm import (
    build_problem,
    compare_reference_topology,
    paper_loss,
    wdm_boundary_context,
    wdm_port_rows,
)


def test_wdm_geometry_is_the_full_paper_reconstruction():
    problem, metadata = build_problem(resolution=40 * bz.nm, run_time=400e-15)

    assert problem.variable_shape == (160, 160)
    assert np.isclose(metadata["design_width_m"], 6.4 * bz.um)
    assert np.isclose(metadata["design_height_m"], 6.4 * bz.um)
    assert np.isclose(metadata["non_pml_width_m"], 8.0 * bz.um)
    assert np.isclose(metadata["non_pml_height_m"], 7.2 * bz.um)
    assert np.isclose(metadata["output_offset_m"], 2.6 * bz.um)
    ports = problem.ports
    assert [port.name for port in ports] == ["port1", "port2", "port3"]
    assert [port.direction for port in ports] == ["+", "-", "-"]
    assert all(port.mode_spec.mode_index == 0 for port in ports)
    assert ports[1].center[1] < ports[0].center[1] < ports[2].center[1]
    assert "reconstructed" in metadata["geometry_provenance"]["output_offset"]


def test_wdm_ports_match_the_three_physical_interfaces():
    input_rows, output_rows = wdm_port_rows((160, 160), 40 * bz.nm)

    assert input_rows.sum() == 10
    assert output_rows.sum() == 20
    assert np.array_equal(np.flatnonzero(input_rows), np.arange(75, 85))
    assert np.array_equal(
        np.flatnonzero(output_rows),
        np.r_[np.arange(10, 20), np.arange(140, 150)],
    )


def test_wdm_full_context_has_distinct_left_and_right_targets():
    brush = paper_circular_brush(5)
    context = wdm_boundary_context((320, 320), brush, 20 * bz.nm)
    target = context["boundary_target"]
    input_rows, output_rows = wdm_port_rows((320, 320), 20 * bz.nm)

    assert np.array_equal(target[:, 0], input_rows)
    assert np.array_equal(target[:, -1], output_rows)
    assert not np.any(target[0])
    assert not np.any(target[-1])
    assert np.all(context["fixed_solid"][context["fixed_solid"]] == 1)
    assert not np.any(context["fixed_void"] & context["fixed_solid"])


def test_wdm_objective_routes_each_band_to_the_correct_port():
    zeros = jnp.zeros(6)
    correct_s21 = jnp.r_[jnp.ones(3), jnp.zeros(3)]
    correct_s31 = jnp.r_[jnp.zeros(3), jnp.ones(3)]

    correct = paper_loss(zeros, correct_s21, correct_s31)
    swapped = paper_loss(zeros, correct_s31, correct_s21)
    reflected = paper_loss(jnp.ones(6), correct_s21, correct_s31)

    assert float(correct) < float(swapped)
    assert float(correct) < float(reflected)


def test_wdm_objective_accepts_an_arbitrary_routing_schedule():
    zeros = jnp.zeros(4)
    routes = np.array([2, 3, 2, 3])
    correct_s21 = jnp.asarray([1.0, 0.0, 1.0, 0.0])
    correct_s31 = 1.0 - correct_s21

    correct = paper_loss(zeros, correct_s21, correct_s31, target_ports=routes)
    swapped = paper_loss(zeros, correct_s31, correct_s21, target_ports=routes)

    assert float(correct) < float(swapped)


def test_wdm_objective_requires_routes_for_a_nonpaper_grid():
    with np.testing.assert_raises_regex(ValueError, "target_ports"):
        paper_loss(jnp.zeros(4), jnp.zeros(4), jnp.zeros(4))


def test_wdm_reference_comparison_accepts_full_density(tmp_path):
    reference = (
        Path(__file__).resolve().parents[2]
        / "ceviche-challenges"
        / "img"
        / "wavelength_divison_multiplexer.png"
    )
    if not reference.is_file():
        return
    density = np.zeros((230, 250))
    density[70:160, 80:170] = 1.0
    metrics = compare_reference_topology(
        density,
        reference,
        tmp_path / "overlay.png",
        pml_cells=20,
    )
    assert np.isfinite(metrics["topology_symmetric_contour_distance_nm"])
    assert (tmp_path / "overlay.png").is_file()
