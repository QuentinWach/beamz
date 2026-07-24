"""Reproduce the paper's O-band wavelength demultiplexer with BeamZ.

The paper explicitly gives the 6.4 um square design region, material system,
wavelength bands, scattering targets, Adam parameters, and 100 nm brush.  The
paper does not publish every simulation-domain coordinate.  Access-waveguide
placement below is therefore reconstructed from the rendered figure and the
authors' smaller public Ceviche WDM prefab; output metadata records that
provenance instead of presenting the reconstruction as an exact source value.

Port 1 is the centered input on the left.  Port 2 is the lower output and
receives the 1265--1275 nm band; port 3 is the upper output and receives the
1285--1295 nm band.
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

LOW_BAND_WAVELENGTHS_NM = np.array([1265.0, 1270.0, 1275.0])
HIGH_BAND_WAVELENGTHS_NM = np.array([1285.0, 1290.0, 1295.0])
OPTIMIZATION_WAVELENGTHS_NM = np.concatenate(
    (LOW_BAND_WAVELENGTHS_NM, HIGH_BAND_WAVELENGTHS_NM)
)
SPECTRUM_WAVELENGTHS_NM = np.unique(
    np.concatenate(
        (
            np.arange(1260.0, 1300.0 + 0.1, 1.0),
            OPTIMIZATION_WAVELENGTHS_NM,
        )
    )
)
FIELD_WAVELENGTHS_NM = np.array([1270.0, 1290.0])
GenerationMode = Literal["paper", "ports", "side-conditioned", "conditioned-full"]

# Bounds of the left panel in ceviche-challenges/img/wavelength_divison_multiplexer.png.
REFERENCE_PANEL_BOUNDS = (29, 121, 412, 467)
REFERENCE_IMAGE_SIZE = (1362, 609)
REFERENCE_PML_CELLS = 20


def wdm_port_rows(
    design_shape: tuple[int, int],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left-input and right-output phase masks at the design interfaces."""

    ny, _ = (int(value) for value in design_shape)
    width = int(round((0.4 * bz.um) / resolution))
    output_offset = int(round((2.6 * bz.um) / resolution))
    half_low = width // 2
    half_high = width - half_low

    input_rows = np.zeros(ny, dtype=bool)
    input_rows[ny // 2 - half_low : ny // 2 + half_high] = True
    output_rows = np.zeros(ny, dtype=bool)
    for center in (ny // 2 - output_offset, ny // 2 + output_offset):
        output_rows[center - half_low : center + half_high] = True
    return input_rows, output_rows


def wdm_boundary_context(
    design_shape: tuple[int, int],
    brush,
    resolution: float,
    *,
    condition_sides: bool = True,
    include_vertical_halo: bool = True,
) -> dict:
    """Add the exterior oxide and the three fixed access waveguides.

    The generator operates on the design plus a one-brush halo.  Thus features
    that meet a crop edge are judged against the material that physically
    exists beyond it.  A radius-wide phase target makes the three ports join
    smoothly while keeping essentially all of the 6.4 um square free.
    """

    ny, nx = (int(value) for value in design_shape)
    padding = max(brush.shape)
    vertical_padding = padding if include_vertical_halo else 0
    canvas_shape = (ny + 2 * vertical_padding, nx + 2 * padding)
    design_slices = (
        slice(vertical_padding, vertical_padding + ny),
        slice(padding, padding + nx),
    )
    generated_pixels = np.zeros(canvas_shape, dtype=bool)
    generated_pixels[design_slices] = True
    fixed_solid = np.zeros(canvas_shape, dtype=bool)
    fixed_void = ~generated_pixels

    input_rows, output_rows = wdm_port_rows(design_shape, resolution)
    canvas_input = np.zeros(canvas_shape[0], dtype=bool)
    canvas_output = np.zeros(canvas_shape[0], dtype=bool)
    canvas_input[design_slices[0]] = input_rows
    canvas_output[design_slices[0]] = output_rows
    fixed_solid[canvas_input, :padding] = True
    fixed_solid[canvas_output, padding + nx :] = True
    fixed_void[fixed_solid] = False

    transition_depth = min(max(1, (max(brush.shape) + 1) // 2), ny // 2, nx // 2)
    yy, xx = np.indices((ny, nx))
    left_weight = np.clip((transition_depth - xx) / transition_depth, 0.0, 1.0)
    right_weight = np.clip(
        (transition_depth - (nx - 1 - xx)) / transition_depth, 0.0, 1.0
    )
    if condition_sides:
        boundary_weight = np.maximum(left_weight, right_weight)
        boundary_target = np.zeros((ny, nx), dtype=bool)
        boundary_target[:, :transition_depth] = input_rows[:, None]
        boundary_target[:, nx - transition_depth :] = output_rows[:, None]
    else:
        boundary_weight = np.zeros((ny, nx), dtype=float)
        boundary_target = np.zeros((ny, nx), dtype=bool)

    if include_vertical_halo:
        top_bottom_weight = np.maximum(
            np.clip((transition_depth - yy) / transition_depth, 0.0, 1.0),
            np.clip(
                (transition_depth - (ny - 1 - yy)) / transition_depth,
                0.0,
                1.0,
            ),
        )
        # At a corner the exterior oxide target takes precedence.
        boundary_target[top_bottom_weight >= boundary_weight] = False
        boundary_weight = np.maximum(boundary_weight, top_bottom_weight)

    return {
        "canvas_shape": canvas_shape,
        "design_slices": design_slices,
        "generated_pixels": generated_pixels,
        "fixed_solid": fixed_solid,
        "fixed_void": fixed_void,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "boundary_weight": boundary_weight,
        "boundary_target": boundary_target,
        "transition_depth": transition_depth if condition_sides else 0,
    }


def compare_reference_topology(
    full_density: np.ndarray,
    reference_image: Path,
    output_path: Path | None = None,
    *,
    pml_cells: int = REFERENCE_PML_CELLS,
) -> dict[str, float]:
    """Return a symmetric contour distance to the authors' rendered design."""

    reference_boundary = reference_panel_boundary(
        reference_image,
        bounds=REFERENCE_PANEL_BOUNDS,
        expected_image_size=REFERENCE_IMAGE_SIZE,
    )
    reference_boundary[[0, -1], :] = False
    reference_boundary[:, [0, -1]] = False
    rendered = np.asarray(full_density)[pml_cells:-pml_cells, pml_cells:-pml_cells]
    beamz_solid = resize_binary(np.flipud(rendered > 0.5), reference_boundary.shape)
    beamz_boundary = beamz_solid & ~ndimage.binary_erosion(beamz_solid)
    to_reference = ndimage.distance_transform_edt(~reference_boundary)
    to_beamz = ndimage.distance_transform_edt(~beamz_boundary)
    reference_to_beamz = float(np.mean(to_beamz[reference_boundary]))
    beamz_to_reference = float(np.mean(to_reference[beamz_boundary]))
    symmetric_pixels = 0.5 * (reference_to_beamz + beamz_to_reference)

    if output_path is not None:
        overlay = np.zeros((*reference_boundary.shape, 3), dtype=np.uint8)
        overlap = reference_boundary & beamz_boundary
        overlay[reference_boundary] = (255, 74, 74)
        overlay[beamz_boundary] = (68, 210, 255)
        overlay[overlap] = (255, 255, 255)
        Image.fromarray(overlay).save(output_path)
    return {
        "topology_symmetric_contour_distance_px": symmetric_pixels,
        "topology_symmetric_contour_distance_nm": symmetric_pixels * 8000.0 / 383.0,
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
    """Build the 6.4 by 6.4 um paper WDM reconstruction.

    The public ``ceviche_challenges.wdm.prefabs.wdm_spec`` is a separate,
    smaller 3.2 um test prefab and cannot directly reproduce the article image.
    """

    design_size = 6.4 * bz.um
    wg_width = 0.4 * bz.um
    wg_length = 0.8 * bz.um
    vertical_padding = 0.4 * bz.um
    mode_padding = 0.4 * bz.um
    output_offset = 2.6 * bz.um
    port_pml_offset = 0.04 * bz.um
    monitor_offset = 0.04 * bz.um
    pml_thickness = 20.0 * resolution
    eps_oxide, eps_silicon = 2.25, 12.25

    extent_x = 2.0 * pml_thickness + 2.0 * wg_length + design_size
    extent_y = 2.0 * pml_thickness + 2.0 * vertical_padding + design_size
    design_min_x = pml_thickness + wg_length
    design_min_y = pml_thickness + vertical_padding
    design_max_x = design_min_x + design_size
    design_max_y = design_min_y + design_size
    center_y = 0.5 * extent_y
    upper_y = center_y + output_offset
    lower_y = center_y - output_offset

    design = bz.Design(
        width=extent_x,
        height=extent_y,
        material=bz.Material(permittivity=eps_oxide),
    )
    design += bz.Rectangle(
        position=(0.0, center_y - 0.5 * wg_width),
        width=design_min_x,
        height=wg_width,
        material=bz.Material(permittivity=eps_silicon),
    )
    for output_y in (upper_y, lower_y):
        design += bz.Rectangle(
            position=(design_max_x, output_y - 0.5 * wg_width),
            width=extent_x - design_max_x,
            height=wg_width,
            material=bz.Material(permittivity=eps_silicon),
        )
    design += bz.Rectangle(
        position=(design_min_x, design_min_y),
        width=design_size,
        height=design_size,
        material=bz.Material(permittivity=eps_silicon),
    )

    frequencies = bz.LIGHT_SPEED / (np.asarray(wavelengths_nm) * bz.nm)
    field_frequencies = bz.LIGHT_SPEED / (FIELD_WAVELENGTHS_NM * bz.nm)
    aperture = wg_width + 2.0 * mode_padding
    mode_spec = bz.ModeSpec(num_modes=3, mode_index=0, polarization="tm")
    left_monitor_x = pml_thickness + port_pml_offset + monitor_offset
    right_monitor_x = extent_x - left_monitor_x
    ports = (
        bz.Port(
            center=(left_monitor_x, center_y, 0.0),
            size=(0.0, aperture, wg_width),
            name="port1",
            direction="+",
            mode_spec=mode_spec,
        ),
        bz.Port(
            center=(right_monitor_x, lower_y, 0.0),
            size=(0.0, aperture, wg_width),
            name="port2",
            direction="-",
            mode_spec=mode_spec,
        ),
        bz.Port(
            center=(right_monitor_x, upper_y, 0.0),
            size=(0.0, aperture, wg_width),
            name="port3",
            direction="-",
            mode_spec=mode_spec,
        ),
    )
    field_frequency = bz.LIGHT_SPEED / (1280.0 * bz.nm)
    source_time = bz.GaussianPulse(
        freq0=field_frequency,
        fwidth=SOURCE_FRACTIONAL_BANDWIDTH * field_frequency,
        offset=SOURCE_OFFSET,
    )
    validate_run_time(run_time, source_time)
    dt = 0.95 * resolution / (bz.LIGHT_SPEED * np.sqrt(2.0))
    time_grid = np.arange(int(np.ceil(run_time / dt)), dtype=float) * dt
    source = bz.ModeSource(
        center=(pml_thickness + port_pml_offset, center_y, 0.0),
        size=ports[0].size,
        source_time=source_time,
        direction="+",
        mode_spec=mode_spec,
        power=1.0,
    )
    monitors = [port.to_monitor(frequencies) for port in ports]
    if include_field_monitor:
        monitors.append(
            bz.DomainFieldMonitor(
                field_frequencies,
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
        "non_pml_width_m": float(2.0 * wg_length + design_size),
        "non_pml_height_m": float(2.0 * vertical_padding + design_size),
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
        "output_offset_m": float(output_offset),
        "mode_padding_m": float(mode_padding),
        "eps_oxide": eps_oxide,
        "eps_silicon": eps_silicon,
        "num_time_steps": int(len(time_grid)),
        "source_peak_time_s": float(source_time.offset / source_time.fwidth),
        "monitor_wavelengths_nm": np.asarray(wavelengths_nm).tolist(),
        "geometry_provenance": {
            "design_size": "article text, 6.4 um square",
            "waveguide_width": "article text and public Ceviche prefab, 0.4 um",
            "materials": "article text and public Ceviche prefab",
            "output_offset": "reconstructed from the published rendered figure",
            "access_domain": "BeamZ reconstruction; not published numerically",
            "public_ceviche_prefab_note": (
                "The repository prefab uses a distinct 3.2 um design region "
                "inside a 5.12 um simulation domain."
            ),
        },
    }
    return problem, metadata


def paper_loss(
    s11,
    s21,
    s31,
    *,
    target_ports: np.ndarray | tuple[int, ...] | list[int] | None = None,
    reflection_target_db: float = -20.0,
    transmission_target_db: float = -3.0,
    crosstalk_target_db: float = -20.0,
):
    """Equation 12 and Table 1 for an arbitrary wavelength routing schedule.

    ``target_ports[i]`` is either 2 or 3 and identifies the desired output at
    sample ``i``.  The six paper samples remain the default for compatibility.
    """

    sample_count = int(s11.shape[0])
    if s21.shape != s11.shape or s31.shape != s11.shape:
        raise ValueError("All WDM S-parameter vectors must have the same shape.")
    if target_ports is None:
        if sample_count != len(OPTIMIZATION_WAVELENGTHS_NM):
            raise ValueError(
                "target_ports is required when the wavelength grid does not use "
                "the six paper optimization samples."
            )
        target_ports_array = np.r_[
            np.full(len(LOW_BAND_WAVELENGTHS_NM), 2),
            np.full(len(HIGH_BAND_WAVELENGTHS_NM), 3),
        ]
    else:
        target_ports_array = np.asarray(target_ports, dtype=int)
    if target_ports_array.shape != (sample_count,):
        raise ValueError(
            f"target_ports has shape {target_ports_array.shape}, expected "
            f"{(sample_count,)}."
        )
    if not np.all(np.isin(target_ports_array, (2, 3))):
        raise ValueError("target_ports entries must be either 2 or 3.")

    power = jnp.stack(
        (jnp.abs(s11) ** 2, jnp.abs(s21) ** 2, jnp.abs(s31) ** 2), axis=-1
    )
    reflection = 10.0 ** (reflection_target_db / 10.0)
    transmission = 10.0 ** (transmission_target_db / 10.0)
    crosstalk = 10.0 ** (crosstalk_target_db / 10.0)
    routes_to_port2 = jnp.asarray(target_ports_array == 2)
    cutoff = jnp.stack(
        (
            jnp.full(sample_count, reflection),
            jnp.where(routes_to_port2, transmission, crosstalk),
            jnp.where(routes_to_port2, crosstalk, transmission),
        ),
        axis=-1,
    )
    sign = jnp.stack(
        (
            jnp.ones(sample_count),
            jnp.where(routes_to_port2, -1.0, 1.0),
            jnp.where(routes_to_port2, 1.0, -1.0),
        ),
        axis=-1,
    )
    valid_width = jnp.where(sign > 0.0, cutoff, 1.0 - cutoff)
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
    generation_mode: GenerationMode = "conditioned-full",
    reflection_target_db: float = -20.0,
    transmission_target_db: float = -3.0,
    crosstalk_target_db: float = -20.0,
    target_ports: np.ndarray | tuple[int, ...] | list[int] | None = None,
    initial_latent: np.ndarray | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 10,
    snapshot_interval: int = 10,
):
    """Optimize a binary WDM while enforcing strict brush feasibility every step."""

    trainable = problem.differentiable("port1")
    resolution = problem.simulations["port1"].resolution
    brush_pixels = max(1, int(np.ceil(float(brush_size) / float(resolution) - 1e-9)))
    brush = paper_circular_brush(brush_pixels)
    design_shape = tuple(int(value) for value in problem.variable_shape)

    if generation_mode == "conditioned-full":
        boundary = wdm_boundary_context(
            design_shape,
            brush,
            resolution,
            condition_sides=True,
            include_vertical_halo=True,
        )
    elif generation_mode == "side-conditioned":
        boundary = wdm_boundary_context(
            design_shape,
            brush,
            resolution,
            condition_sides=True,
            include_vertical_halo=False,
        )
    elif generation_mode == "ports":
        boundary = wdm_boundary_context(
            design_shape,
            brush,
            resolution,
            condition_sides=False,
            include_vertical_halo=True,
        )
    elif generation_mode == "paper":
        input_rows, output_rows = wdm_port_rows(design_shape, resolution)
        boundary = {
            "canvas_shape": design_shape,
            "design_slices": (slice(None), slice(None)),
            "generated_pixels": np.ones(design_shape, dtype=bool),
            "fixed_solid": np.zeros(design_shape, dtype=bool),
            "fixed_void": np.zeros(design_shape, dtype=bool),
            "input_rows": input_rows,
            "output_rows": output_rows,
            "boundary_weight": np.zeros(design_shape, dtype=float),
            "boundary_target": np.zeros(design_shape, dtype=bool),
            "transition_depth": 0,
        }
    else:
        raise ValueError(f"Unknown WDM generation mode: {generation_mode}.")

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
        canvas_reward = jnp.zeros(boundary["canvas_shape"], dtype=reward.dtype)
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
        return generated, generated.density[design_slices], generated.density

    _, initial_density, _ = generate(latent)
    projector = problem.port_projector(
        jnp.asarray(initial_density), source_port="port1"
    )

    def objective(result):
        values = [
            projector.s_parameter(
                result, source_port="port1", output_port=f"port{index}"
            )
            for index in range(1, 4)
        ]
        return paper_loss(
            *values,
            target_ports=target_ports,
            reflection_target_db=reflection_target_db,
            transmission_target_db=transmission_target_db,
            crosstalk_target_db=crosstalk_target_db,
        )

    value_and_grad_density = trainable.compile_value_and_grad(objective)

    def estimator(value):
        proxy = 0.5 * (filtered_reward(value, brush, beta=beta) + 1.0)
        if generation_mode not in {"side-conditioned", "conditioned-full"}:
            return proxy
        return (1.0 - boundary_weight) * proxy + boundary_weight * boundary_target

    best_loss = np.inf
    best_step = 0
    best_latent = latent
    best_density = initial_density
    history: list[float] = []
    generator_steps: list[int] = []
    density_history: list[np.ndarray] = []
    latent_history: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        generated, density_np, _ = generate(latent)
        loss, density_gradient = value_and_grad_density(jnp.asarray(density_np))
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
                f"BeamZ WDM step {step:3d}/{steps}: loss={loss_value:.6g}, "
                f"silicon={float(np.mean(density_np)):.3f}, "
                f"generator_steps={generated.steps}"
            )

    generated, verified_density, composite_density = generate(best_latent)
    if not np.array_equal(verified_density, best_density):
        raise RuntimeError("Best WDM latent no longer reproduces its saved density.")
    solid_error, void_error = brush_feasibility_errors(composite_density, brush)
    generated_pixels = np.asarray(boundary["generated_pixels"])
    fabrication_errors = (solid_error | void_error) & generated_pixels
    if np.any(fabrication_errors):
        raise RuntimeError("Generator returned a WDM with strict-foundry errors.")
    return {
        "density": np.asarray(best_density),
        "composite_density": np.asarray(composite_density),
        "composite_generated_pixels": generated_pixels,
        "input_rows": np.asarray(boundary["input_rows"]),
        "output_rows": np.asarray(boundary["output_rows"]),
        "boundary_weight": np.asarray(boundary_weight),
        "boundary_target": np.asarray(boundary_target),
        "best_latent": np.asarray(best_latent),
        "best_step": best_step,
        "loss": float(best_loss),
        "history": np.asarray(history),
        "generator_steps": np.asarray(
            generator_steps if generator_steps else [generated.steps]
        ),
        "density_history": np.asarray(density_history, dtype=np.uint8),
        "latent_history": np.asarray(latent_history, dtype=np.float32),
        "snapshot_steps": np.asarray(snapshot_steps, dtype=np.int32),
        "current_latent": np.asarray(latent),
        "first_moment": np.asarray(first_moment),
        "second_moment": np.asarray(second_moment),
        "completed_steps": int(steps),
        "runtime_s": time.perf_counter() - started,
        "brush": brush,
        "brush_size_pixels": brush_pixels,
        "brush_feasible": True,
        "initial_silicon_fraction": float(np.mean(initial_density)),
        "generation_mode": generation_mode,
        "reflection_target_db": float(reflection_target_db),
        "transmission_target_db": float(transmission_target_db),
        "crosstalk_target_db": float(crosstalk_target_db),
        "target_ports": (
            np.r_[
                np.full(len(LOW_BAND_WAVELENGTHS_NM), 2),
                np.full(len(HIGH_BAND_WAVELENGTHS_NM), 3),
            ]
            if target_ports is None
            else np.asarray(target_ports, dtype=int)
        ),
        "evaluation_only": False,
        "source_checkpoint": None,
        "source_step": None,
        "source_best_loss": None,
        "latent_flip_y": False,
    }


def reconstruct_checkpoint_design(
    checkpoint: Path,
    problem,
    *,
    brush_size: float,
    beta: float,
    generation_mode: GenerationMode,
    source_step: int | None = None,
    flip_y: bool = False,
    reflection_target_db: float = -20.0,
    transmission_target_db: float = -3.0,
    crosstalk_target_db: float = -20.0,
) -> dict:
    """Regenerate a checkpoint topology without taking an optimizer step.

    This path is deliberately separate from warm-start optimization: it retains
    the source trajectory's step/loss metadata and explicitly records any
    resampling or vertical reflection applied to the selected latent.
    """

    target_shape = tuple(int(value) for value in problem.variable_shape)
    target_resolution = problem.simulations["port1"].resolution
    latent = load_multiresolution_latent(
        checkpoint,
        target_shape,
        brush_size=brush_size,
        target_resolution=target_resolution,
        step=source_step,
    )
    if flip_y:
        latent = np.flip(latent, axis=0).copy()

    with np.load(checkpoint) as source:
        source_history = np.asarray(source.get("history", np.empty(0)), dtype=float)
        source_generator_steps = np.asarray(
            source.get("generator_steps", np.empty(0)), dtype=int
        )
        selected_step = (
            int(source_step)
            if source_step is not None
            else int(source.get("best_step", len(source_history)))
        )
        completed_steps = int(source.get("completed_steps", len(source_history)))
        source_best_loss = float(source.get("best_loss", np.nan))

    reconstructed = optimize(
        problem,
        steps=0,
        learning_rate=0.01,
        brush_size=brush_size,
        beta=beta,
        initial_bias=0.0,
        initial_noise=0.0,
        seed=0,
        generation_mode=generation_mode,
        reflection_target_db=reflection_target_db,
        transmission_target_db=transmission_target_db,
        crosstalk_target_db=crosstalk_target_db,
        initial_latent=latent,
    )
    reconstructed.update(
        {
            "best_latent": np.asarray(latent, dtype=np.float32),
            "current_latent": np.asarray(latent, dtype=np.float32),
            "first_moment": np.zeros_like(latent, dtype=np.float32),
            "second_moment": np.zeros_like(latent, dtype=np.float32),
            "completed_steps": completed_steps,
            "best_step": selected_step,
            # The source loss is not valid after resampling or reflection.  The
            # evaluation spectrum supplies the output design's objective later.
            "loss": np.nan,
            "history": source_history,
            "generator_steps": (
                source_generator_steps
                if source_generator_steps.size
                else np.asarray(reconstructed["generator_steps"])
            ),
            "density_history": reconstructed["density"][None].astype(np.uint8),
            "latent_history": np.asarray(latent, dtype=np.float32)[None],
            "snapshot_steps": np.asarray([selected_step], dtype=np.int32),
            "runtime_s": 0.0,
            "evaluation_only": True,
            "source_checkpoint": str(checkpoint),
            "source_step": selected_step,
            "source_best_loss": source_best_loss,
            "latent_flip_y": bool(flip_y),
        }
    )
    return reconstructed


def evaluate(problem, density: np.ndarray):
    """Run one broadband source and return all three S-parameters and two fields."""

    density_jax = jnp.asarray(density, dtype=jnp.float32)
    trainable = problem.differentiable("port1")
    result = trainable.run(density_jax)
    projector = problem.port_projector(density_jax, source_port="port1")
    scattering = []
    for index in range(1, 4):
        scattering.append(
            np.asarray(
                projector.s_parameter(
                    result, source_port="port1", output_port=f"port{index}"
                )
            )
        )
    fields = np.asarray(result.field("domain_fields", "Ez")).reshape(
        (len(FIELD_WAVELENGTHS_NM), *trainable.base_permittivity.shape)
    )
    return np.stack(scattering), fields, np.asarray(trainable.permittivity(density_jax))


def _db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.abs(values) ** 2, 1e-12))


def save_topology_preview(output_dir: Path, composite_density: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "beamz_wdm_topology.png"
    fig, axis = plt.subplots(figsize=(7.2, 7.2))
    axis.imshow(composite_density, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    axis.contour(composite_density, levels=[0.5], colors="#00c8ff", linewidths=0.8)
    axis.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path


def save_results(
    output_dir: Path, optimization: dict, result, metadata: dict, args
) -> dict:
    """Save a Figure 6/7-style result and machine-readable verification data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    scattering, fields, permittivity = result
    s11, s21, s31 = scattering
    s_db = _db(scattering)
    wavelengths = SPECTRUM_WAVELENGTHS_NM
    density_full = (permittivity - metadata["eps_oxide"]) / (
        metadata["eps_silicon"] - metadata["eps_oxide"]
    )

    figure, axes = plt.subplots(2, 2, figsize=(13.8, 10.0))
    for field_axis, field, wavelength in zip(
        axes[:, 0], fields, FIELD_WAVELENGTHS_NM, strict=True
    ):
        field_axis.imshow(
            np.abs(field), origin="lower", cmap="magma", interpolation="bilinear"
        )
        field_axis.contour(
            permittivity,
            levels=[0.5 * (metadata["eps_oxide"] + metadata["eps_silicon"])],
            colors="white",
            linewidths=0.6,
            origin="lower",
        )
        field_axis.set_title(f"{wavelength:.0f} nm")
        field_axis.set_axis_off()
    spectrum_axis = axes[0, 1]
    spectrum_axis.plot(wavelengths, s_db[0], color="#e41a1c", label=r"$S_{11}$")
    spectrum_axis.plot(wavelengths, s_db[1], color="#377eb8", label=r"$S_{21}$")
    spectrum_axis.plot(wavelengths, s_db[2], color="#4daf4a", label=r"$S_{31}$")
    spectrum_axis.set_xlabel("Wavelength (nm)")
    spectrum_axis.set_ylabel("dB")
    spectrum_axis.set_ylim(-41.0, 1.0)
    spectrum_axis.grid(alpha=0.25)
    spectrum_axis.legend(loc="upper right")
    topology_axis = axes[1, 1]
    topology_axis.imshow(optimization["density"], origin="lower", cmap="gray")
    topology_axis.contour(
        optimization["density"], levels=[0.5], colors="#00c8ff", linewidths=0.7
    )
    topology_axis.set_title("Strict-foundry topology")
    topology_axis.set_axis_off()
    figure.suptitle("Wavelength division multiplexer", fontsize=24)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_dir / "beamz_wdm.png", dpi=180)
    figure.savefig(output_dir / "beamz_wdm_highres.png", dpi=320)
    plt.close(figure)

    np.savez_compressed(
        output_dir / "beamz_wdm.npz",
        design_density=optimization["density"],
        composite_density=optimization["composite_density"],
        wavelengths_nm=wavelengths,
        s11=s11,
        s21=s21,
        s31=s31,
        fields=fields,
        field_wavelengths_nm=FIELD_WAVELENGTHS_NM,
        permittivity=permittivity,
        loss_history=optimization["history"],
        density_history=optimization["density_history"],
        latent_history=optimization["latent_history"],
        snapshot_steps=optimization["snapshot_steps"],
        target_ports=optimization["target_ports"],
        generation_mode=np.asarray(optimization["generation_mode"]),
    )
    reference_metrics = {}
    if args.reference_image is not None and args.reference_image.is_file():
        reference_metrics = compare_reference_topology(
            density_full,
            args.reference_image,
            output_dir / "topology_reference_overlay.png",
        )
    low = np.isin(wavelengths, LOW_BAND_WAVELENGTHS_NM)
    high = np.isin(wavelengths, HIGH_BAND_WAVELENGTHS_NM)
    design_indices = np.asarray(
        [
            int(np.argmin(np.abs(wavelengths - value)))
            for value in OPTIMIZATION_WAVELENGTHS_NM
        ]
    )
    binary = optimization["density"] > 0.5
    input_rows, output_rows = wdm_port_rows(binary.shape, metadata["resolution_m"])
    summary = {
        "configuration": {
            **metadata,
            "evaluation_run_time_s": args.evaluation_run_time_fs * 1e-15,
            "optimization_wavelengths_nm": OPTIMIZATION_WAVELENGTHS_NM.tolist(),
            "optimization_target_ports": np.asarray(
                optimization["target_ports"], dtype=int
            ).tolist(),
            "spectrum_wavelengths_nm": wavelengths.tolist(),
            "optimization_steps_this_run": (
                0 if optimization["evaluation_only"] else int(args.steps)
            ),
            "completed_steps": int(optimization["completed_steps"]),
            "learning_rate": float(args.learning_rate),
            "brush_size_nm": float(args.brush_size_nm),
            "brush_size_pixels": int(optimization["brush_size_pixels"]),
            "seed": int(args.seed),
            "generation_mode": optimization["generation_mode"],
            "initialization_checkpoint": (
                None if args.initialize_from is None else str(args.initialize_from)
            ),
            "initialization_step": args.initialize_step,
            "initialization_flip_y": bool(args.flip_initial_y),
            "best_step": int(optimization["best_step"]),
            "snapshot_interval": int(args.snapshot_interval),
            "evaluation_only": bool(optimization["evaluation_only"]),
            "source_checkpoint": optimization["source_checkpoint"],
            "source_step": optimization["source_step"],
            "source_best_loss": optimization["source_best_loss"],
            "latent_flip_y": bool(optimization["latent_flip_y"]),
        },
        "metrics": {
            "best_loss": float(optimization["loss"]),
            "runtime_s": float(optimization["runtime_s"]),
            "maximum_design_s11_db": float(np.max(s_db[0, design_indices])),
            "minimum_low_band_s21_db": float(np.min(s_db[1, low])),
            "maximum_low_band_s31_db": float(np.max(s_db[2, low])),
            "maximum_high_band_s21_db": float(np.max(s_db[1, high])),
            "minimum_high_band_s31_db": float(np.min(s_db[2, high])),
            "silicon_fraction": float(np.mean(binary)),
            "binary_fraction": float(
                np.mean((optimization["density"] == 0) | (optimization["density"] == 1))
            ),
            "brush_feasible": bool(optimization["brush_feasible"]),
            "top_design_edge_silicon_pixels": int(np.sum(binary[-1])),
            "bottom_design_edge_silicon_pixels": int(np.sum(binary[0])),
            "left_interface_extra_silicon_pixels": int(
                np.sum(binary[:, 0] & ~input_rows)
            ),
            "right_interface_extra_silicon_pixels": int(
                np.sum(binary[:, -1] & ~output_rows)
            ),
            "left_interface_missing_port_pixels": int(
                np.sum(~binary[:, 0] & input_rows)
            ),
            "right_interface_missing_port_pixels": int(
                np.sum(~binary[:, -1] & output_rows)
            ),
            "mean_generator_steps": float(np.mean(optimization["generator_steps"])),
            **reference_metrics,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution-nm",
        type=float,
        default=40.0,
        help=(
            "Grid spacing. The WDM defaults to 40 nm (160 x 160 design cells); "
            "10 nm creates a much larger 640 x 640 generator problem."
        ),
    )
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
    parser.add_argument("--reflection-target-db", type=float, default=-20.0)
    parser.add_argument("--transmission-target-db", type=float, default=-3.0)
    parser.add_argument("--crosstalk-target-db", type=float, default=-20.0)
    parser.add_argument(
        "--generation-mode",
        choices=("paper", "ports", "side-conditioned", "conditioned-full"),
        default=None,
        help=(
            "Fabrication boundary model. Defaults to conditioned-full for a new "
            "run and to the saved mode for --evaluate-checkpoint."
        ),
    )
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--initialize-step", type=int)
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        help=(
            "Regenerate and evaluate a checkpoint topology without taking a "
            "new optimizer step. Preserves source-step provenance."
        ),
    )
    parser.add_argument(
        "--flip-initial-y",
        action="store_true",
        help="Reflect a warm-start latent vertically (useful for port conventions).",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    default_reference = adjacent_reference_image("wavelength_divison_multiplexer.png")
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=default_reference,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/ceviche_wdm_o_band"),
    )
    args = parser.parse_args()
    if (
        args.initialize_step is not None
        and args.initialize_from is None
        and args.evaluate_checkpoint is None
    ):
        parser.error(
            "--initialize-step requires --initialize-from or --evaluate-checkpoint."
        )
    if args.initialize_from is not None and args.evaluate_checkpoint is not None:
        parser.error(
            "--initialize-from and --evaluate-checkpoint are mutually exclusive."
        )
    if args.evaluate_checkpoint is not None and args.skip_evaluation:
        parser.error("--evaluate-checkpoint cannot be combined with --skip-evaluation.")
    if args.steps < 1 and args.evaluate_checkpoint is None:
        parser.error("--steps must be positive.")
    if args.snapshot_interval < 1:
        parser.error("--snapshot-interval must be positive.")
    if args.generation_mode is None:
        if args.evaluate_checkpoint is None:
            args.generation_mode = "conditioned-full"
        else:
            with np.load(args.evaluate_checkpoint) as source:
                args.generation_mode = (
                    str(np.asarray(source["generation_mode"]).item())
                    if "generation_mode" in source.files
                    else "conditioned-full"
                )

    optimization_problem, optimization_metadata = build_problem(
        resolution=args.resolution_nm * bz.nm,
        run_time=args.run_time_fs * 1e-15,
    )
    checkpoint_path = args.output_dir / "optimizer_checkpoint.npz"
    if args.evaluate_checkpoint is not None:
        optimization = reconstruct_checkpoint_design(
            args.evaluate_checkpoint,
            optimization_problem,
            brush_size=args.brush_size_nm * bz.nm,
            beta=args.beta,
            generation_mode=args.generation_mode,
            source_step=args.initialize_step,
            flip_y=args.flip_initial_y,
            reflection_target_db=args.reflection_target_db,
            transmission_target_db=args.transmission_target_db,
            crosstalk_target_db=args.crosstalk_target_db,
        )
    else:
        initial_latent = (
            None
            if args.initialize_from is None
            else load_multiresolution_latent(
                args.initialize_from,
                tuple(int(value) for value in optimization_problem.variable_shape),
                brush_size=args.brush_size_nm * bz.nm,
                target_resolution=args.resolution_nm * bz.nm,
                step=args.initialize_step,
            )
        )
        if initial_latent is not None and args.flip_initial_y:
            initial_latent = np.flip(initial_latent, axis=0).copy()
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
            crosstalk_target_db=args.crosstalk_target_db,
            initial_latent=initial_latent,
            checkpoint_path=checkpoint_path,
            snapshot_interval=args.snapshot_interval,
        )
    topology_preview = save_topology_preview(
        args.output_dir, optimization["composite_density"]
    )
    if args.skip_evaluation:
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "topology_preview": str(topology_preview),
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
    if optimization["evaluation_only"]:
        design_indices = np.asarray(
            [
                int(np.argmin(np.abs(SPECTRUM_WAVELENGTHS_NM - wavelength)))
                for wavelength in OPTIMIZATION_WAVELENGTHS_NM
            ]
        )
        optimization["loss"] = float(
            paper_loss(
                *(jnp.asarray(values[design_indices]) for values in result[0]),
                reflection_target_db=args.reflection_target_db,
                transmission_target_db=args.transmission_target_db,
                crosstalk_target_db=args.crosstalk_target_db,
            )
        )
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
