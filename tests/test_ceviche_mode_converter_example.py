from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.optimization.challenges.mode_converter import (
    build_problem,
    compare_reference_topology,
    mode_converter_boundary_context,
    mode_converter_port_context,
    mode_converter_port_rows,
    mode_converter_side_context,
    paper_circular_brush,
    paper_loss,
)


def test_mode_converter_geometry_matches_ceviche_material_masks():
    pytest.importorskip(
        "ceviche_challenges",
        reason="requires the optional local ceviche-challenges checkout",
    )
    from ceviche_challenges import units as u
    from ceviche_challenges.mode_converter import model, prefabs

    resolution = 20 * bz.nm
    problem, metadata = build_problem(
        resolution=resolution,
        run_time=200e-15,
        wavelengths_nm=np.array([1280.0]),
    )
    trainable = problem.differentiable("port1")
    spec = prefabs.mode_converter_spec_12(
        left_wg_width=400 * u.nm,
        left_wg_mode_padding=480 * u.nm,
        left_wg_mode_order=1,
        right_wg_width=400 * u.nm,
        right_wg_mode_padding=480 * u.nm,
        right_wg_mode_order=2,
        wg_length=400 * u.nm,
        padding=200 * u.nm,
        port_pml_offset=40 * u.nm,
        variable_region_size=(1600 * u.nm, 1600 * u.nm),
        cladding_permittivity=2.25,
        slab_permittivity=12.25,
        input_monitor_offset=40 * u.nm,
        pml_width=20,
    )
    params = prefabs.mode_converter_sim_params(
        resolution=20 * u.nm,
        wavelengths=u.Array([1280.0], u.nm),
    )
    ceviche_model = model.ModeConverterModel(params, spec)

    assert problem.variable_shape == (80, 80)
    assert metadata["design_width_m"] == 1.6 * bz.um
    assert metadata["design_height_m"] == 1.6 * bz.um
    for value in (0.0, 1.0):
        density = np.full(problem.variable_shape, value, dtype=np.float32)
        beamz_solid = np.asarray(trainable.permittivity(jnp.asarray(density))) > 7.0
        ceviche_density = ceviche_model.density_bg.copy()
        ceviche_density[ceviche_model.design_region] = value
        np.testing.assert_array_equal(
            beamz_solid,
            (ceviche_density > 0.5).T,
        )


def test_mode_converter_uses_fundamental_input_and_second_output_modes():
    problem, _ = build_problem(
        resolution=20 * bz.nm,
        run_time=200e-15,
        wavelengths_nm=np.array([1280.0]),
    )

    assert problem.ports[0].mode_spec.mode_index == 0
    assert problem.ports[1].mode_spec.mode_index == 1
    assert problem.ports[0].mode_spec.polarization == "tm"
    assert problem.ports[1].mode_spec.polarization == "tm"


def test_mode_converter_side_context_only_conditions_waveguide_interfaces():
    shape = (80, 80)
    brush = paper_circular_brush(5)
    port_context = mode_converter_port_context(shape, brush, 20 * bz.nm)
    side_context = mode_converter_side_context(shape, brush, 20 * bz.nm)
    port_rows = mode_converter_port_rows(shape, 20 * bz.nm)

    assert port_rows.sum() == 20
    assert port_context["canvas_shape"] == (80, 90)
    assert np.all(port_context["generated_pixels"][0, 5:-5])
    assert np.all(port_context["generated_pixels"][-1, 5:-5])
    assert not np.any(port_context["fixed_solid"][:, 5:-5])
    assert not np.any(port_context["fixed_void"][:, 5:-5])

    weight = side_context["boundary_weight"]
    assert side_context["transition_depth"] == 3
    assert np.all(weight[:, :1] == 1.0)
    assert np.all(weight[:, -1:] == 1.0)
    assert np.all(weight[:, 3:-3] == 0.0)
    np.testing.assert_array_equal(weight[0], weight[-1])


def test_mode_converter_full_context_rounds_top_and_bottom_into_oxide():
    shape = (80, 80)
    brush = paper_circular_brush(5)
    context = mode_converter_boundary_context(shape, brush, 20 * bz.nm)
    target = context["boundary_target"]
    weight = context["boundary_weight"]

    assert context["canvas_shape"] == (90, 90)
    assert context["transition_depth"] == 3
    assert np.all(weight[0] == 1.0)
    assert np.all(weight[-1] == 1.0)
    assert not np.any(target[0])
    assert not np.any(target[-1])
    assert np.any(target[:, 0])
    assert np.any(target[:, -1])


def test_mode_converter_paper_loss_rewards_the_specification():
    compliant = paper_loss(
        jnp.asarray([0.01, 0.02]),
        jnp.asarray([0.99, 0.98]),
    )
    violating = paper_loss(
        jnp.asarray([0.3, 0.4]),
        jnp.asarray([0.3, 0.2]),
    )
    assert float(compliant) < float(violating)


def test_mode_converter_reference_comparison_accepts_full_density(tmp_path):
    reference = (
        Path(__file__).resolve().parents[2]
        / "ceviche-challenges"
        / "img"
        / "mode_converter.png"
    )
    if not reference.is_file():
        return
    density = np.zeros((120, 140), dtype=float)
    density[50:70, :] = 1.0
    metrics = compare_reference_topology(
        density,
        reference,
        tmp_path / "overlay.png",
    )

    assert metrics["topology_symmetric_contour_distance_nm"] > 0.0
    assert (tmp_path / "overlay.png").is_file()
