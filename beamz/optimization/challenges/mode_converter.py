"""Optimize the O-band Ceviche Challenges mode converter with BeamZ.

The device converts the fundamental mode of a 400 nm silicon waveguide into
the second-order mode of an identical output waveguide. The physical
parameters and objective follow Schubert et al., *Inverse Design of Photonic
Devices with Strict Foundry Fabrication Constraints*.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

import beamz as bz
from beamz.optimization import (
    DesignRegion,
    InverseDesignProblem,
    brush_feasibility_errors,
    conditional_generator,
    filtered_reward,
)
from beamz.optimization.challenges._common import (
    BOUNDARY_PHASE_REWARD,
    CHECKPOINT_SCHEMA_VERSION,
    SOURCE_FRACTIONAL_BANDWIDTH,
    SOURCE_OFFSET,
    adam_update,
    adjacent_reference_image,
    atomic_savez,
    load_multiresolution_latent,
    paper_circular_brush,
    reference_panel_boundary,
    resize_binary,
    validate_run_time,
)

OPTIMIZATION_WAVELENGTHS_NM = np.array([1265.0, 1270.0, 1275.0, 1285.0, 1290.0, 1295.0])
SPECTRUM_WAVELENGTHS_NM = np.unique(
    np.concatenate(
        (
            np.arange(1260.0, 1300.0 + 0.1, 2.0),
            OPTIMIZATION_WAVELENGTHS_NM,
        )
    )
)
FIELD_WAVELENGTH_NM = 1280.0
GenerationMode = Literal[
    "paper",
    "ports",
    "side-conditioned",
    "conditioned-full",
]

# Pixel bounds of the field panel in ceviche-challenges/img/mode_converter.png.
REFERENCE_PANEL_BOUNDS = (29, 121, 441, 467)
REFERENCE_IMAGE_SIZE = (1376, 609)
REFERENCE_PML_CELLS = 20


def mode_converter_port_rows(
    design_shape: tuple[int, int],
    resolution: float,
) -> np.ndarray:
    """Return the centered 400 nm waveguide pixels at a design interface."""

    ny, _ = (int(value) for value in design_shape)
    waveguide_width = int(round((0.4 * bz.um) / resolution))
    half_low = waveguide_width // 2
    half_high = waveguide_width - half_low
    rows = np.zeros(ny, dtype=bool)
    rows[ny // 2 - half_low : ny // 2 + half_high] = True
    return rows


def mode_converter_port_context(
    design_shape: tuple[int, int],
    brush,
    resolution: float,
) -> dict[str, np.ndarray | tuple[slice, slice] | tuple[int, int] | int]:
    """Build fixed left/right waveguide context without a top/bottom halo."""

    ny, nx = (int(value) for value in design_shape)
    padding = max(brush.shape)
    canvas_shape = (ny, nx + 2 * padding)
    design_slices = (slice(None), slice(padding, padding + nx))
    design_pixels = np.zeros(canvas_shape, dtype=bool)
    design_pixels[design_slices] = True
    fixed_solid = np.zeros(canvas_shape, dtype=bool)
    fixed_void = ~design_pixels
    port_rows = mode_converter_port_rows(design_shape, resolution)
    fixed_solid[port_rows, :padding] = True
    fixed_solid[port_rows, padding + nx :] = True
    fixed_void[fixed_solid] = False
    return {
        "canvas_shape": canvas_shape,
        "design_slices": design_slices,
        "generated_pixels": design_pixels,
        "fixed_solid": fixed_solid,
        "fixed_void": fixed_void,
        "port_rows": port_rows,
        "boundary_weight": np.zeros(design_shape, dtype=float),
        "boundary_target": np.zeros(design_shape, dtype=bool),
        "transition_depth": 0,
    }


def mode_converter_boundary_context(
    design_shape: tuple[int, int],
    brush,
    resolution: float,
) -> dict[str, np.ndarray | tuple[slice, slice] | tuple[int, int] | int]:
    """Build an oxide halo with fixed silicon access waveguides."""

    ny, nx = (int(value) for value in design_shape)
    padding = max(brush.shape)
    canvas_shape = (ny + 2 * padding, nx + 2 * padding)
    design_slices = (
        slice(padding, padding + ny),
        slice(padding, padding + nx),
    )
    design_pixels = np.zeros(canvas_shape, dtype=bool)
    design_pixels[design_slices] = True
    fixed_solid = np.zeros(canvas_shape, dtype=bool)
    fixed_void = ~design_pixels
    port_rows = np.zeros(canvas_shape[0], dtype=bool)
    port_rows[design_slices[0]] = mode_converter_port_rows(
        design_shape,
        resolution,
    )
    fixed_solid[port_rows, :padding] = True
    fixed_solid[port_rows, padding + nx :] = True
    fixed_void[fixed_solid] = False

    transition_depth = min(
        max(1, (max(brush.shape) + 1) // 2),
        ny // 2,
        nx // 2,
    )
    yy, xx = np.indices((ny, nx))
    vertical_weight = np.maximum(
        np.clip((transition_depth - xx) / transition_depth, 0.0, 1.0),
        np.clip(
            (transition_depth - (nx - 1 - xx)) / transition_depth,
            0.0,
            1.0,
        ),
    )
    horizontal_weight = np.maximum(
        np.clip((transition_depth - yy) / transition_depth, 0.0, 1.0),
        np.clip(
            (transition_depth - (ny - 1 - yy)) / transition_depth,
            0.0,
            1.0,
        ),
    )
    boundary_weight = np.maximum(vertical_weight, horizontal_weight)
    port_rows_design = port_rows[design_slices[0]]
    boundary_target = np.broadcast_to(
        port_rows_design[:, None],
        design_shape,
    ).copy()
    boundary_target[horizontal_weight >= vertical_weight] = False
    return {
        "canvas_shape": canvas_shape,
        "design_slices": design_slices,
        "generated_pixels": design_pixels,
        "fixed_solid": fixed_solid,
        "fixed_void": fixed_void,
        "port_rows": port_rows,
        "boundary_weight": boundary_weight,
        "boundary_target": boundary_target,
        "transition_depth": transition_depth,
    }


def mode_converter_side_context(
    design_shape: tuple[int, int],
    brush,
    resolution: float,
) -> dict[str, np.ndarray | tuple[slice, slice] | tuple[int, int] | int]:
    """Condition one brush radius at the two waveguide interfaces."""

    context = mode_converter_port_context(design_shape, brush, resolution)
    ny, nx = design_shape
    transition_depth = min(max(1, (max(brush.shape) + 1) // 2), nx // 2)
    _, xx = np.indices((ny, nx))
    boundary_weight = np.maximum(
        np.clip((transition_depth - xx) / transition_depth, 0.0, 1.0),
        np.clip(
            (transition_depth - (nx - 1 - xx)) / transition_depth,
            0.0,
            1.0,
        ),
    )
    port_rows = mode_converter_port_rows(design_shape, resolution)
    context["boundary_weight"] = boundary_weight
    context["boundary_target"] = np.broadcast_to(
        port_rows[:, None],
        design_shape,
    ).copy()
    context["transition_depth"] = transition_depth
    return context


def compare_reference_topology(
    full_density: np.ndarray,
    reference_image: Path,
    output_path: Path | None = None,
    *,
    pml_cells: int = REFERENCE_PML_CELLS,
) -> dict[str, float]:
    """Compare a BeamZ material contour with the checked-in reference PNG."""

    reference_boundary = reference_panel_boundary(
        reference_image,
        bounds=REFERENCE_PANEL_BOUNDS,
        expected_image_size=REFERENCE_IMAGE_SIZE,
    )
    reference_boundary[[0, -1], :] = False
    reference_boundary[:, [0, -1]] = False

    rendered_density = np.asarray(full_density)[
        pml_cells:-pml_cells,
        pml_cells:-pml_cells,
    ]
    beamz_solid = resize_binary(
        np.flipud(rendered_density > 0.5),
        reference_boundary.shape,
    )
    beamz_boundary = beamz_solid & ~ndimage.binary_erosion(beamz_solid)
    distance_to_reference = ndimage.distance_transform_edt(~reference_boundary)
    distance_to_beamz = ndimage.distance_transform_edt(~beamz_boundary)
    reference_to_beamz = float(np.mean(distance_to_beamz[reference_boundary]))
    beamz_to_reference = float(np.mean(distance_to_reference[beamz_boundary]))
    symmetric_pixels = 0.5 * (reference_to_beamz + beamz_to_reference)
    # The non-PML reference panel is 2.4 um wide.
    nanometers_per_pixel = 2400.0 / reference_boundary.shape[1]

    if output_path is not None:
        overlay = np.zeros((*reference_boundary.shape, 3), dtype=np.uint8)
        overlap = reference_boundary & beamz_boundary
        overlay[reference_boundary] = (255, 74, 74)
        overlay[beamz_boundary] = (68, 210, 255)
        overlay[overlap] = (255, 255, 255)
        Image.fromarray(overlay).save(output_path)

    return {
        "topology_symmetric_contour_distance_px": symmetric_pixels,
        "topology_symmetric_contour_distance_nm": (
            symmetric_pixels * nanometers_per_pixel
        ),
        "topology_reference_to_beamz_px": reference_to_beamz,
        "topology_beamz_to_reference_px": beamz_to_reference,
    }


def build_problem(
    *,
    resolution: float,
    run_time: float,
    wavelengths_nm: np.ndarray = OPTIMIZATION_WAVELENGTHS_NM,
    include_field_monitor: bool = False,
):
    """Build the paper's 1.6 by 1.6 um O-band mode converter."""

    design_size = 1.6 * bz.um
    wg_width = 0.4 * bz.um
    wg_length = round((0.4 * bz.um) / resolution) * resolution
    vertical_padding = 0.2 * bz.um
    mode_padding = 0.48 * bz.um
    port_pml_offset = 0.04 * bz.um
    monitor_offset = 0.04 * bz.um
    pml_thickness = 20.0 * resolution
    eps_oxide, eps_silicon = 2.25, 12.25

    extent_x = 2.0 * pml_thickness + 2.0 * wg_length + design_size
    extent_y = 2.0 * pml_thickness + design_size + 2.0 * vertical_padding
    design_min_x = pml_thickness + wg_length
    design_min_y = pml_thickness + vertical_padding
    design_max_x = design_min_x + design_size
    design_max_y = design_min_y + design_size
    center_y = 0.5 * extent_y

    design = bz.Design(
        width=extent_x,
        height=extent_y,
        material=bz.Material(permittivity=eps_oxide),
    )
    design += bz.Rectangle(
        position=(0.0, center_y - 0.5 * wg_width),
        width=extent_x,
        height=wg_width,
        material=bz.Material(permittivity=eps_silicon),
    )
    design += bz.Rectangle(
        position=(design_min_x, design_min_y),
        width=design_size,
        height=design_size,
        material=bz.Material(permittivity=eps_silicon),
    )

    monitor_frequencies = bz.LIGHT_SPEED / (np.asarray(wavelengths_nm) * bz.nm)
    field_frequency = bz.LIGHT_SPEED / (FIELD_WAVELENGTH_NM * bz.nm)
    aperture = wg_width + 2.0 * mode_padding
    input_mode = bz.ModeSpec(num_modes=3, mode_index=0, polarization="tm")
    output_mode = bz.ModeSpec(num_modes=3, mode_index=1, polarization="tm")
    monitor_left_x = pml_thickness + port_pml_offset + monitor_offset
    monitor_right_x = extent_x - monitor_left_x
    ports = (
        bz.Port(
            center=(monitor_left_x, center_y, 0.0),
            size=(0.0, aperture, wg_width),
            name="port1",
            direction="+",
            mode_spec=input_mode,
        ),
        bz.Port(
            center=(monitor_right_x, center_y, 0.0),
            size=(0.0, aperture, wg_width),
            name="port2",
            direction="-",
            mode_spec=output_mode,
        ),
    )

    dt = 0.95 * resolution / (bz.LIGHT_SPEED * np.sqrt(2.0))
    source_time = bz.GaussianPulse(
        freq0=field_frequency,
        fwidth=SOURCE_FRACTIONAL_BANDWIDTH * field_frequency,
        offset=SOURCE_OFFSET,
    )
    validate_run_time(run_time, source_time)
    time_grid = np.arange(int(np.ceil(run_time / dt)), dtype=float) * dt
    source = bz.ModeSource(
        center=(pml_thickness + port_pml_offset, center_y, 0.0),
        size=ports[0].size,
        source_time=source_time,
        direction="+",
        mode_spec=input_mode,
        power=1.0,
    )
    monitors = [port.to_monitor(monitor_frequencies) for port in ports]
    if include_field_monitor:
        monitors.append(
            bz.DomainFieldMonitor(
                [field_frequency],
                fields=("Ez",),
                name="domain_fields",
            )
        )
    simulation = bz.Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[
            bz.PML(
                edges="all",
                thickness=pml_thickness,
                formulation="cpml",
                target_reflection=1e-8,
            )
        ],
        time=time_grid,
        resolution=resolution,
        normalize_source=0,
    )
    region = DesignRegion(
        lower=(design_min_x, design_min_y),
        upper=(design_max_x, design_max_y),
        eps_min=eps_oxide,
        eps_max=eps_silicon,
    )
    problem = InverseDesignProblem(
        {"port1": simulation},
        ports,
        region,
        field_monitor="domain_fields" if include_field_monitor else None,
        field_component="Ez",
    )
    metadata = {
        "resolution_m": float(resolution),
        "run_time_s": float(run_time),
        "extent_x_m": float(extent_x),
        "extent_y_m": float(extent_y),
        "pml_thickness_m": float(pml_thickness),
        "design_bounds_m": [
            float(design_min_x),
            float(design_min_y),
            float(design_max_x),
            float(design_max_y),
        ],
        "design_width_m": float(design_size),
        "design_height_m": float(design_size),
        "waveguide_width_m": float(wg_width),
        "mode_padding_m": float(mode_padding),
        "vertical_padding_m": float(vertical_padding),
        "eps_oxide": eps_oxide,
        "eps_silicon": eps_silicon,
        "num_time_steps": int(len(time_grid)),
        "source_peak_time_s": float(source_time.offset / source_time.fwidth),
        "monitor_wavelengths_nm": np.asarray(wavelengths_nm).tolist(),
    }
    return problem, metadata


def paper_loss(
    s11,
    s21,
    *,
    reflection_target_db: float = -20.0,
    transmission_target_db: float = -0.5,
):
    """Equation 12 with the mode-converter target window from Table 1."""

    power = jnp.stack((jnp.abs(s11) ** 2, jnp.abs(s21) ** 2), axis=-1)
    cutoff = jnp.asarray(
        (
            10.0 ** (reflection_target_db / 10.0),
            10.0 ** (transmission_target_db / 10.0),
        )
    )
    sign = jnp.asarray((1.0, -1.0))
    valid_width = jnp.asarray((cutoff[0], 1.0 - cutoff[1]))
    signed_distance = sign * (power - cutoff) / jnp.min(valid_width)
    return jnp.sum(jax.nn.softplus(signed_distance) ** 2)


def optimize(
    problem,
    *,
    steps: int,
    learning_rate: float,
    brush_size: float,
    beta: float,
    initial_bias: float,
    initial_noise: float,
    seed: int,
    generation_mode: GenerationMode = "side-conditioned",
    reflection_target_db: float = -20.0,
    transmission_target_db: float = -0.5,
    initial_latent: np.ndarray | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 10,
    snapshot_interval: int = 10,
):
    """Optimize a binary, strictly brush-feasible mode converter."""

    trainable = problem.differentiable("port1")
    resolution = problem.simulations["port1"].resolution
    brush_pixels = max(
        1,
        int(np.ceil(float(brush_size) / float(resolution) - 1e-9)),
    )
    brush = paper_circular_brush(brush_pixels)
    design_shape = tuple(int(value) for value in problem.variable_shape)
    if generation_mode == "conditioned-full":
        boundary = mode_converter_boundary_context(
            design_shape,
            brush,
            resolution,
        )
    elif generation_mode == "side-conditioned":
        boundary = mode_converter_side_context(design_shape, brush, resolution)
    elif generation_mode == "ports":
        boundary = mode_converter_port_context(design_shape, brush, resolution)
    elif generation_mode == "paper":
        boundary = {
            "canvas_shape": design_shape,
            "design_slices": (slice(None), slice(None)),
            "generated_pixels": np.ones(design_shape, dtype=bool),
            "fixed_solid": np.zeros(design_shape, dtype=bool),
            "fixed_void": np.zeros(design_shape, dtype=bool),
            "port_rows": mode_converter_port_rows(design_shape, resolution),
            "boundary_weight": np.zeros(design_shape, dtype=float),
            "boundary_target": np.zeros(design_shape, dtype=bool),
            "transition_depth": 0,
        }
    else:
        raise ValueError(f"Unknown mode-converter generation mode: {generation_mode}.")

    canvas_shape = boundary["canvas_shape"]
    design_slices = boundary["design_slices"]
    fixed_solid = boundary["fixed_solid"]
    fixed_void = boundary["fixed_void"]
    boundary_weight = boundary["boundary_weight"]
    boundary_target = boundary["boundary_target"]

    if initial_latent is None:
        rng = np.random.default_rng(seed)
        latent = jnp.asarray(
            initial_bias + initial_noise * rng.standard_normal(design_shape),
            dtype=jnp.float32,
        )
    else:
        latent = jnp.asarray(initial_latent, dtype=jnp.float32)
        if tuple(latent.shape) != design_shape:
            raise ValueError(
                f"Initial latent has shape {latent.shape}, expected {design_shape}."
            )
    first_moment = jnp.zeros_like(latent)
    second_moment = jnp.zeros_like(latent)

    def generate(value):
        reward = filtered_reward(value, brush, beta=beta)
        if generation_mode == "paper":
            generated = conditional_generator(reward, brush)
            return generated, generated.density, generated.density

        canvas_reward = jnp.zeros(canvas_shape, dtype=reward.dtype)
        if generation_mode in {"side-conditioned", "conditioned-full"}:
            canvas_reward = canvas_reward.at[design_slices].set(
                (1.0 - boundary_weight) * reward
                + boundary_weight
                * BOUNDARY_PHASE_REWARD
                * (2.0 * boundary_target - 1.0)
            )
        else:
            canvas_reward = canvas_reward.at[design_slices].set(reward)
        generated = conditional_generator(
            canvas_reward,
            brush,
            fixed_solid=fixed_solid,
            fixed_void=fixed_void,
        )
        return (
            generated,
            generated.density[design_slices],
            generated.density,
        )

    initial_generated, initial_density, _ = generate(latent)
    projector = problem.port_projector(
        jnp.asarray(initial_density),
        source_port="port1",
    )

    def objective(result):
        s11 = projector.s_parameter(
            result,
            source_port="port1",
            output_port="port1",
        )
        s21 = projector.s_parameter(
            result,
            source_port="port1",
            output_port="port2",
        )
        return paper_loss(
            s11,
            s21,
            reflection_target_db=reflection_target_db,
            transmission_target_db=transmission_target_db,
        )

    value_and_grad_density = trainable.compile_value_and_grad(objective)

    def estimator(value):
        density_proxy = 0.5 * (filtered_reward(value, brush, beta=beta) + 1.0)
        if generation_mode not in {"side-conditioned", "conditioned-full"}:
            return density_proxy
        return (1.0 - boundary_weight) * density_proxy + (
            boundary_weight * boundary_target
        )

    best_loss = np.inf
    best_step = 0
    best_latent = latent
    best_density = initial_density
    best_composite = None
    history: list[float] = []
    generator_steps: list[int] = []
    density_history: list[np.ndarray] = []
    latent_history: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    started = time.perf_counter()

    for step in range(1, steps + 1):
        generated, density_np, composite_density = generate(latent)
        density = jnp.asarray(density_np)
        loss, density_gradient = value_and_grad_density(density)
        _, pullback = jax.vjp(estimator, latent)
        gradient = pullback(density_gradient)[0]
        loss_value = float(loss)
        history.append(loss_value)
        generator_steps.append(generated.steps)
        if step == 1 or step % snapshot_interval == 0 or step == steps:
            density_history.append(np.asarray(density_np, dtype=np.uint8))
            latent_history.append(np.asarray(latent, dtype=np.float32))
            snapshot_steps.append(step)
        if np.isfinite(loss_value) and loss_value < best_loss:
            best_loss = loss_value
            best_step = step
            best_latent = latent
            best_density = density_np
            best_composite = composite_density

        latent, first_moment, second_moment = adam_update(
            latent,
            gradient,
            first_moment,
            second_moment,
            step=step,
            learning_rate=learning_rate,
        )

        if checkpoint_path is not None and (
            step % checkpoint_interval == 0 or step == steps
        ):
            atomic_savez(
                checkpoint_path,
                schema_version=np.asarray(CHECKPOINT_SCHEMA_VERSION),
                current_latent=np.asarray(latent),
                first_moment=np.asarray(first_moment),
                second_moment=np.asarray(second_moment),
                completed_steps=np.asarray(step),
                best_latent=np.asarray(best_latent),
                best_density=np.asarray(best_density),
                best_loss=np.asarray(best_loss),
                best_step=np.asarray(best_step),
                history=np.asarray(history),
                generator_steps=np.asarray(generator_steps),
                density_history=np.asarray(density_history, dtype=np.uint8),
                latent_history=np.asarray(latent_history, dtype=np.float32),
                snapshot_steps=np.asarray(snapshot_steps, dtype=np.int32),
                brush_mask=np.asarray(brush.mask, dtype=np.uint8),
                generation_mode=np.asarray(generation_mode),
            )
        if step == 1 or step % 10 == 0 or step == steps:
            print(
                f"BeamZ step {step:3d}/{steps}: loss={loss_value:.6g}, "
                f"silicon={float(np.mean(density_np)):.3f}, "
                f"generator_steps={generated.steps}"
            )

    _, verified_density, verified_composite = generate(best_latent)
    if not np.array_equal(verified_density, best_density):
        raise RuntimeError("Best latent no longer reproduces its saved design.")
    if best_composite is not None and not np.array_equal(
        verified_composite,
        best_composite,
    ):
        raise RuntimeError("Best latent no longer reproduces its composite design.")
    solid_error, void_error = brush_feasibility_errors(verified_composite, brush)
    generated_pixels = np.asarray(boundary["generated_pixels"])
    brush_feasible = not bool(np.any((solid_error | void_error) & generated_pixels))
    if not brush_feasible:
        raise RuntimeError("Generator returned an infeasible mode converter.")
    return {
        "density": np.asarray(best_density),
        "composite_density": np.asarray(verified_composite),
        "composite_generated_pixels": generated_pixels,
        "fixed_solid": np.asarray(fixed_solid),
        "fixed_void": np.asarray(fixed_void),
        "port_rows": np.asarray(boundary["port_rows"]),
        "boundary_weight": np.asarray(boundary_weight),
        "boundary_target": np.asarray(boundary_target),
        "transition_depth": int(boundary["transition_depth"]),
        "design_slices": design_slices,
        "best_latent": np.asarray(best_latent),
        "current_latent": np.asarray(latent),
        "first_moment": np.asarray(first_moment),
        "second_moment": np.asarray(second_moment),
        "completed_steps": steps,
        "best_step": best_step,
        "loss": float(best_loss),
        "history": np.asarray(history),
        "generator_steps": np.asarray(generator_steps),
        "density_history": np.asarray(density_history, dtype=np.uint8),
        "latent_history": np.asarray(latent_history, dtype=np.float32),
        "snapshot_steps": np.asarray(snapshot_steps, dtype=np.int32),
        "runtime_s": time.perf_counter() - started,
        "brush": brush,
        "brush_size_pixels": brush_pixels,
        "brush_feasible": brush_feasible,
        "initial_generator_steps": initial_generated.steps,
        "initial_silicon_fraction": float(np.mean(initial_density)),
        "generation_mode": generation_mode,
        "reflection_target_db": float(reflection_target_db),
        "transmission_target_db": float(transmission_target_db),
    }


def evaluate(problem, density: np.ndarray):
    """Run a broadband BeamZ solve and return S11, S21, field, and material."""

    density_jax = jnp.asarray(density, dtype=jnp.float32)
    trainable = problem.differentiable("port1")
    result = trainable.run(density_jax)
    projector = problem.port_projector(density_jax, source_port="port1")
    s11 = np.asarray(
        projector.s_parameter(
            result,
            source_port="port1",
            output_port="port1",
        )
    )
    s21 = np.asarray(
        projector.s_parameter(
            result,
            source_port="port1",
            output_port="port2",
        )
    )
    field = np.asarray(result.field("domain_fields", "Ez")[0]).reshape(
        trainable.base_permittivity.shape
    )
    permittivity = np.asarray(trainable.permittivity(density_jax))
    return s11, s21, field, permittivity


def _db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.abs(values) ** 2, 1e-12))


def save_topology_preview(output_dir: Path, composite_density: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "beamz_mode_converter_topology.png"
    density = np.asarray(composite_density)
    fig, axis = plt.subplots(figsize=(7.5, 5.0 * density.shape[0] / density.shape[1]))
    axis.imshow(density, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    axis.contour(
        density,
        levels=[0.5],
        colors="#00c8ff",
        linewidths=1.2,
        origin="lower",
    )
    axis.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def save_results(
    output_dir: Path,
    optimization: dict,
    result,
    metadata: dict,
    args,
) -> dict:
    """Save the BeamZ reproduction figure, arrays, and metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    s11, s21, field, permittivity = result
    wavelengths = SPECTRUM_WAVELENGTHS_NM
    s11_db, s21_db = _db(s11), _db(s21)
    density_full = (permittivity - metadata["eps_oxide"]) / (
        metadata["eps_silicon"] - metadata["eps_oxide"]
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.8))
    field_axis, spectrum_axis = axes
    field_axis.imshow(
        np.abs(field),
        origin="lower",
        cmap="magma",
        interpolation="bilinear",
    )
    field_axis.contour(
        permittivity,
        levels=[0.5 * (metadata["eps_oxide"] + metadata["eps_silicon"])],
        colors="white",
        linewidths=0.9,
        origin="lower",
    )
    field_axis.set_axis_off()
    spectrum_axis.plot(wavelengths, s11_db, color="#e41a1c", label=r"$S_{11}$")
    spectrum_axis.plot(wavelengths, s21_db, color="#377eb8", label=r"$S_{21}$")
    spectrum_axis.set_xlabel("Wavelength (nm)")
    spectrum_axis.set_ylabel("dB")
    spectrum_axis.set_ylim(-41.0, 1.0)
    spectrum_axis.grid(alpha=0.25)
    spectrum_axis.legend(loc="upper right")
    figure.suptitle("Mode converter", fontsize=24)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_dir / "beamz_mode_converter.png", dpi=180)
    figure.savefig(output_dir / "beamz_mode_converter_highres.png", dpi=320)
    plt.close(figure)

    np.savez_compressed(
        output_dir / "beamz_mode_converter.npz",
        design_density=optimization["density"],
        composite_density=optimization["composite_density"],
        wavelengths_nm=wavelengths,
        s11=s11,
        s21=s21,
        field=field,
        permittivity=permittivity,
        loss_history=optimization["history"],
        density_history=optimization["density_history"],
        latent_history=optimization["latent_history"],
        snapshot_steps=optimization["snapshot_steps"],
        generation_mode=np.asarray(optimization["generation_mode"]),
    )

    reference_metrics: dict[str, float] = {}
    if args.reference_image is not None and args.reference_image.is_file():
        reference_metrics = compare_reference_topology(
            density_full,
            args.reference_image,
            output_dir / "topology_reference_overlay.png",
        )
    design_indices = np.asarray(
        [
            int(np.argmin(np.abs(wavelengths - wavelength)))
            for wavelength in OPTIMIZATION_WAVELENGTHS_NM
        ]
    )
    binary_design = optimization["density"] > 0.5
    port_rows = mode_converter_port_rows(
        binary_design.shape,
        metadata["resolution_m"],
    )
    solid_error, void_error = brush_feasibility_errors(
        optimization["composite_density"],
        optimization["brush"],
    )
    composite_error = (solid_error | void_error) & optimization[
        "composite_generated_pixels"
    ]
    summary = {
        "configuration": {
            **metadata,
            "evaluation_run_time_s": args.evaluation_run_time_fs * 1e-15,
            "evaluation_num_time_steps": metadata["evaluation_num_time_steps"],
            "optimization_wavelengths_nm": OPTIMIZATION_WAVELENGTHS_NM.tolist(),
            "spectrum_wavelengths_nm": wavelengths.tolist(),
            "field_wavelength_nm": FIELD_WAVELENGTH_NM,
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "projection_beta": float(args.beta),
            "brush_size_nm": float(args.brush_size_nm),
            "brush_size_pixels": int(optimization["brush_size_pixels"]),
            "seed": int(args.seed),
            "initial_bias": float(args.initial_bias),
            "initial_noise": float(args.initial_noise),
            "initial_silicon_fraction": float(optimization["initial_silicon_fraction"]),
            "generation_mode": optimization["generation_mode"],
            "reflection_target_db": float(optimization["reflection_target_db"]),
            "transmission_target_db": float(optimization["transmission_target_db"]),
            "initialization_checkpoint": (
                None if args.initialize_from is None else str(args.initialize_from)
            ),
            "initialization_step": args.initialize_step,
            "best_step": int(optimization["best_step"]),
            "snapshot_interval": int(args.snapshot_interval),
        },
        "metrics": {
            "best_loss": float(optimization["loss"]),
            "runtime_s": float(optimization["runtime_s"]),
            "minimum_design_band_s21_db": float(np.min(s21_db[design_indices])),
            "maximum_design_band_s11_db": float(np.max(s11_db[design_indices])),
            "minimum_spectrum_s21_db": float(np.min(s21_db)),
            "maximum_spectrum_s11_db": float(np.max(s11_db)),
            "silicon_fraction": float(np.mean(binary_design)),
            "binary_fraction": float(
                np.mean(
                    (optimization["density"] == 0.0) | (optimization["density"] == 1.0)
                )
            ),
            "brush_feasible": bool(optimization["brush_feasible"]),
            "composite_fabrication_error_pixels": int(np.sum(composite_error)),
            "top_design_edge_silicon_pixels": int(np.sum(binary_design[-1])),
            "bottom_design_edge_silicon_pixels": int(np.sum(binary_design[0])),
            "left_interface_extra_silicon_pixels": int(
                np.sum(binary_design[:, 0] & ~port_rows)
            ),
            "right_interface_extra_silicon_pixels": int(
                np.sum(binary_design[:, -1] & ~port_rows)
            ),
            "left_interface_missing_port_pixels": int(
                np.sum(~binary_design[:, 0] & port_rows)
            ),
            "right_interface_missing_port_pixels": int(
                np.sum(~binary_design[:, -1] & port_rows)
            ),
            "mean_generator_steps": float(np.mean(optimization["generator_steps"])),
            **reference_metrics,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution-nm", type=float, default=10.0)
    parser.add_argument("--run-time-fs", type=float, default=800.0)
    parser.add_argument("--evaluation-run-time-fs", type=float, default=1200.0)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--brush-size-nm", type=float, default=100.0)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--initial-bias", type=float, default=0.0075)
    parser.add_argument("--initial-noise", type=float, default=0.00075)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=10,
        help="Store full latent/density snapshots every N steps.",
    )
    parser.add_argument(
        "--reflection-target-db",
        type=float,
        default=-20.0,
    )
    parser.add_argument(
        "--transmission-target-db",
        type=float,
        default=-0.5,
    )
    parser.add_argument(
        "--generation-mode",
        choices=("paper", "ports", "side-conditioned", "conditioned-full"),
        default="side-conditioned",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Warm-start from a different-resolution mode-converter checkpoint.",
    )
    parser.add_argument(
        "--initialize-step",
        type=int,
        help="Use this one-based latent-history step from --initialize-from.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
    )
    default_reference = adjacent_reference_image("mode_converter.png")
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=default_reference,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/ceviche_mode_converter_o_band"),
    )
    args = parser.parse_args()
    if args.initialize_step is not None and args.initialize_from is None:
        parser.error("--initialize-step requires --initialize-from.")
    if args.snapshot_interval < 1:
        parser.error("--snapshot-interval must be positive.")

    optimization_problem, optimization_metadata = build_problem(
        resolution=args.resolution_nm * bz.nm,
        run_time=args.run_time_fs * 1e-15,
    )
    design_shape = tuple(int(value) for value in optimization_problem.variable_shape)
    initial_latent = (
        None
        if args.initialize_from is None
        else load_multiresolution_latent(
            args.initialize_from,
            design_shape,
            brush_size=args.brush_size_nm * bz.nm,
            target_resolution=args.resolution_nm * bz.nm,
            step=args.initialize_step,
        )
    )
    checkpoint_path = args.output_dir / "optimizer_checkpoint.npz"
    optimization = optimize(
        optimization_problem,
        steps=args.steps,
        learning_rate=args.learning_rate,
        brush_size=args.brush_size_nm * bz.nm,
        beta=args.beta,
        initial_bias=args.initial_bias,
        initial_noise=args.initial_noise,
        seed=args.seed,
        generation_mode=args.generation_mode,
        reflection_target_db=args.reflection_target_db,
        transmission_target_db=args.transmission_target_db,
        initial_latent=initial_latent,
        checkpoint_path=checkpoint_path,
        snapshot_interval=args.snapshot_interval,
    )
    topology_preview = save_topology_preview(
        args.output_dir,
        optimization["composite_density"],
    )
    if args.skip_evaluation:
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "topology_preview": str(topology_preview),
                    "completed_steps": int(optimization["completed_steps"]),
                    "best_step": int(optimization["best_step"]),
                    "best_loss": float(optimization["loss"]),
                    "brush_feasible": bool(optimization["brush_feasible"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    evaluation_problem, evaluation_metadata = build_problem(
        resolution=args.resolution_nm * bz.nm,
        run_time=args.evaluation_run_time_fs * 1e-15,
        wavelengths_nm=SPECTRUM_WAVELENGTHS_NM,
        include_field_monitor=True,
    )
    result = evaluate(evaluation_problem, optimization["density"])
    summary = save_results(
        args.output_dir,
        optimization,
        result,
        {
            **optimization_metadata,
            "evaluation_num_time_steps": evaluation_metadata["num_time_steps"],
        },
        args,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
