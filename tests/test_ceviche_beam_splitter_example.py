"""Integration checks for the fabrication-aware Ceviche beam splitter example."""

from __future__ import annotations

import numpy as np

import beamz as bz
from beamz.optimization import (
    brush_feasibility_errors,
    conditional_generator,
    filtered_reward,
)
from beamz.optimization.challenges._common import load_multiresolution_latent
from beamz.optimization.challenges.beam_splitter import (
    BOUNDARY_PHASE_REWARD,
    mirror_quarter,
    paper_circular_brush,
    splitter_boundary_context,
    splitter_port_context,
    splitter_port_rows,
    splitter_side_context,
    symmetrize_xy,
)


def test_splitter_boundary_canvas_is_symmetric_connected_and_feasible():
    resolution = 40 * bz.nm
    shape = (50, 80)
    brush = paper_circular_brush(3)
    context = splitter_boundary_context(shape, brush, resolution)
    assert context["transition_depth"] == 2
    assert np.count_nonzero(context["boundary_weight"] == 0.0) > 0.85 * np.prod(shape)
    reward = np.zeros(context["canvas_shape"], dtype=np.float32)
    reward[context["design_slices"]] = (1.0 - context["boundary_weight"]) + context[
        "boundary_weight"
    ] * BOUNDARY_PHASE_REWARD * (2.0 * context["boundary_target"] - 1.0)
    reward[context["fixed_solid"]] = BOUNDARY_PHASE_REWARD
    reward[context["fixed_void"]] = -BOUNDARY_PHASE_REWARD

    generated = conditional_generator(
        reward,
        brush,
        reflection_symmetry="xy",
        backend="jax",
    )
    density = generated.density > 0.5
    design = density[context["design_slices"]]
    port_rows = context["port_rows"][context["design_slices"][0]]

    np.testing.assert_array_equal(density, np.flip(density, axis=0))
    np.testing.assert_array_equal(density, np.flip(density, axis=1))
    assert np.all(density[context["fixed_solid"]])
    assert not np.any(density[context["fixed_void"]])
    solid_error, void_error = brush_feasibility_errors(density, brush)
    assert not np.any(solid_error | void_error)
    assert not np.any(design[0])
    assert not np.any(design[-1])
    assert not np.any(design[:, 0] & ~port_rows)
    assert not np.any(design[:, -1] & ~port_rows)
    assert not np.any(~design[:, 0] & port_rows)
    assert not np.any(~design[:, -1] & port_rows)


def test_selected_checkpoint_step_loads_the_exact_pre_update_latent(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    history = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.savez_compressed(
        checkpoint,
        best_latent=np.zeros((3, 4), dtype=np.float32),
        latent_history=history,
    )

    selected = load_multiresolution_latent(
        checkpoint,
        (3, 4),
        brush_size=100 * bz.nm,
        target_resolution=20 * bz.nm,
        step=2,
    )

    np.testing.assert_array_equal(selected, history[1])


def test_paper_generation_modes_begin_solid_without_reducing_the_domain():
    shape = (20, 28)
    quarter_shape = (shape[0] // 2, shape[1] // 2)
    latent = np.full(quarter_shape, 0.01, dtype=np.float32)
    brush = paper_circular_brush(3)

    quarter_reward = np.asarray(filtered_reward(latent, brush, beta=4.0))
    quarter_generated = conditional_generator(quarter_reward, brush)
    quarter_density = np.asarray(mirror_quarter(quarter_generated.density))

    full_latent = mirror_quarter(latent)
    full_reward = np.asarray(
        symmetrize_xy(filtered_reward(full_latent, brush, beta=4.0))
    )
    full_density = conditional_generator(
        full_reward,
        brush,
        reflection_symmetry="xy",
    ).density

    np.testing.assert_array_equal(quarter_density, np.ones(shape))
    np.testing.assert_array_equal(full_density, np.ones(shape))
    assert splitter_port_rows((50, 80), 40 * bz.nm).sum() == 20


def test_mirrored_quarter_generation_is_globally_brush_feasible():
    rng = np.random.default_rng(43)
    latent = rng.normal(size=(12, 16)).astype(np.float32)
    brush = paper_circular_brush(3)
    reward = np.asarray(filtered_reward(latent, brush, beta=4.0))
    quarter = conditional_generator(reward, brush).density
    density = np.asarray(mirror_quarter(quarter))

    np.testing.assert_array_equal(density, np.flip(density, axis=0))
    np.testing.assert_array_equal(density, np.flip(density, axis=1))
    solid_error, void_error = brush_feasibility_errors(density, brush)
    assert not np.any(solid_error | void_error)


def test_port_context_leaves_top_and_bottom_design_boundaries_free():
    shape = (50, 80)
    brush = paper_circular_brush(3)
    context = splitter_port_context(shape, brush, 40 * bz.nm)
    generated = context["generated_pixels"]

    assert context["canvas_shape"] == (50, 86)
    assert np.all(generated[0, 3:-3])
    assert np.all(generated[-1, 3:-3])
    assert not np.any(context["fixed_solid"][:, 3:-3])
    assert not np.any(context["fixed_void"][:, 3:-3])
    assert context["transition_depth"] == 0


def test_side_conditioning_never_biases_top_or_bottom_edges():
    shape = (50, 80)
    brush = paper_circular_brush(3)
    context = splitter_side_context(shape, brush, 40 * bz.nm)
    weight = context["boundary_weight"]

    assert context["transition_depth"] == 2
    assert np.all(weight[:, :1] == 1.0)
    assert np.all(weight[:, -1:] == 1.0)
    assert np.all(weight[:, 2:-2] == 0.0)
    np.testing.assert_array_equal(weight[0], weight[-1])
