"""Optimize the O-band Ceviche Challenges beam splitter with BeamZ.

This reproduces the four-port component shown in Figure 6/7 of Schubert et al.,
``Inverse Design of Photonic Devices with Strict Foundry Fabrication
Constraints``.  The physical design is binary at every step, uses a 100 nm
circular brush, and is reflection symmetric about both design-region axes.

Port ordering follows ``ceviche_challenges.beam_splitter``:

* port 1: upper left (excited)
* port 2: upper right
* port 3: lower right
* port 4: lower left

The paper objective requires S11 and S41 <= -20 dB and S21 and S31 >= -3.5 dB
at 1265, 1270, 1275, 1285, 1290, and 1295 nm.
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
    Brush2D,
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
    "paper-quarter",
    "paper-full",
    "paper-ports",
    "side-conditioned",
    "conditioned-full",
]

# Pixel bounds of the left panel in ``ceviche-challenges/img/beam_splitter.png``.
REFERENCE_PANEL_BOUNDS = (29, 121, 496, 467)
REFERENCE_IMAGE_SIZE = (1404, 609)
REFERENCE_PML_CELLS = 20


def mirror_quarter(values):
    """Reflect a lower-left design quarter about both physical axes."""

    lower = jnp.concatenate((values, jnp.flip(values, axis=1)), axis=1)
    return jnp.concatenate((lower, jnp.flip(lower, axis=0)), axis=0)


def symmetrize_xy(values):
    """Average all four reflection orbits on a complete design canvas."""

    values = 0.5 * (values + jnp.flip(values, axis=0))
    return 0.5 * (values + jnp.flip(values, axis=1))


def splitter_port_rows(
    design_shape: tuple[int, int],
    resolution: float,
) -> np.ndarray:
    """Return the paper/Ceviche port pixels on a design-region interface."""

    ny, _ = (int(value) for value in design_shape)
    waveguide_width = int(round((0.4 * bz.um) / resolution))
    center_offset = int(round((0.74 * bz.um) / resolution))
    half_low = waveguide_width // 2
    half_high = waveguide_width - half_low
    rows = np.zeros(ny, dtype=bool)
    for port_center in (ny // 2 - center_offset, ny // 2 + center_offset):
        rows[port_center - half_low : port_center + half_high] = True
    return rows


def splitter_boundary_context(
    design_shape: tuple[int, int],
    brush: Brush2D,
    resolution: float,
) -> dict[str, np.ndarray | tuple[slice, slice] | tuple[int, int] | int]:
    """Build the fixed oxide halo and four silicon port continuations.

    The generated region is the complete 3.2 by 2.0 um design rectangle. A
    brush-width halo supplies the physical phase outside that rectangle:
    oxide everywhere except for the two 400 nm waveguides on each vertical
    interface. This makes brush decisions at the crop boundary aware of the
    material into which the design will be inlaid.
    """

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
    port_rows[design_slices[0]] = splitter_port_rows(design_shape, resolution)
    fixed_solid[port_rows, :padding] = True
    fixed_solid[port_rows, padding + nx :] = True
    fixed_void[fixed_solid] = False

    # Condition only one brush *radius* on the design side. A two-diameter
    # taper made the nominal 3.2 by 2.0 um rectangle behave like a much smaller
    # free region and forced the port flare too far inward. The radius-wide
    # strip is sufficient for the global generator to see the exterior phase
    # while leaving essentially the complete paper-sized rectangle variable.
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
    boundary_target = np.broadcast_to(port_rows_design[:, None], (ny, nx)).copy()
    boundary_target[horizontal_weight >= vertical_weight] = False
    if not np.array_equal(fixed_solid, np.flip(fixed_solid, axis=(0, 1))):
        raise RuntimeError("Splitter fixed-solid context is not XY symmetric.")
    if not np.array_equal(fixed_void, np.flip(fixed_void, axis=(0, 1))):
        raise RuntimeError("Splitter fixed-void context is not XY symmetric.")
    if not np.allclose(boundary_weight, np.flip(boundary_weight, axis=(0, 1))):
        raise RuntimeError("Splitter boundary taper is not XY symmetric.")
    if not np.array_equal(boundary_target, np.flip(boundary_target, axis=(0, 1))):
        raise RuntimeError("Splitter boundary target is not XY symmetric.")
    return {
        "canvas_shape": canvas_shape,
        "design_slices": design_slices,
        "generated_pixels": np.ones(canvas_shape, dtype=bool),
        "fixed_solid": fixed_solid,
        "fixed_void": fixed_void,
        "port_rows": port_rows,
        "boundary_weight": boundary_weight,
        "boundary_target": boundary_target,
        "transition_depth": transition_depth,
    }


def splitter_port_context(
    design_shape: tuple[int, int],
    brush: Brush2D,
    resolution: float,
) -> dict[str, np.ndarray | tuple[slice, slice] | tuple[int, int] | int]:
    """Build only the physical left/right waveguide context.

    Unlike ``splitter_boundary_context``, this paper-reproduction context has
    no top/bottom halo and applies no phase taper inside the design. The
    generator may therefore use the complete design height while its boundary
    touches still see the four fixed Ceviche waveguides.
    """

    ny, nx = (int(value) for value in design_shape)
    padding = max(brush.shape)
    canvas_shape = (ny, nx + 2 * padding)
    design_slices = (slice(None), slice(padding, padding + nx))
    design_pixels = np.zeros(canvas_shape, dtype=bool)
    design_pixels[design_slices] = True
    fixed_solid = np.zeros(canvas_shape, dtype=bool)
    fixed_void = ~design_pixels
    port_rows = splitter_port_rows(design_shape, resolution)
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


def splitter_side_context(
    design_shape: tuple[int, int],
    brush: Brush2D,
    resolution: float,
) -> dict[str, np.ndarray | tuple[slice, slice] | tuple[int, int] | int]:
    """Condition only the two waveguide interfaces, never the top/bottom."""

    context = splitter_port_context(design_shape, brush, resolution)
    ny, nx = design_shape
    transition_depth = min(
        max(1, (max(brush.shape) + 1) // 2),
        nx // 2,
    )
    _, xx = np.indices((ny, nx))
    boundary_weight = np.maximum(
        np.clip((transition_depth - xx) / transition_depth, 0.0, 1.0),
        np.clip(
            (transition_depth - (nx - 1 - xx)) / transition_depth,
            0.0,
            1.0,
        ),
    )
    port_rows = splitter_port_rows(design_shape, resolution)
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
    """Compare a simulated material contour with the checked-in reference PNG."""

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
    # The non-PML panel is 4.0 um wide.
    nanometers_per_pixel = 4000.0 / reference_boundary.shape[1]

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
    """Build the paper's four-port O-band beam splitter."""

    design_width = 3.2 * bz.um
    design_height = 2.0 * bz.um
    wg_width = 0.4 * bz.um
    wg_length = round((0.4 * bz.um) / resolution) * resolution
    wg_separation = 1.08 * bz.um
    mode_padding = 0.48 * bz.um
    port_pml_offset = 0.04 * bz.um
    monitor_offset = 0.04 * bz.um
    pml_thickness = 20.0 * resolution
    eps_oxide, eps_silicon = 2.25, 12.25

    extent_x = 2.0 * pml_thickness + 2.0 * wg_length + design_width
    extent_y = 2.0 * pml_thickness + 2.0 * mode_padding + design_height
    design_min_x = pml_thickness + wg_length
    design_min_y = pml_thickness + mode_padding
    design_max_x = design_min_x + design_width
    design_max_y = design_min_y + design_height
    center_y = 0.5 * extent_y
    upper_y = center_y + 0.5 * wg_separation + 0.5 * wg_width
    lower_y = center_y - 0.5 * wg_separation - 0.5 * wg_width

    design = bz.Design(
        width=extent_x,
        height=extent_y,
        material=bz.Material(permittivity=eps_oxide),
    )
    for center in (lower_y, upper_y):
        design += bz.Rectangle(
            position=(0.0, center - 0.5 * wg_width),
            width=extent_x,
            height=wg_width,
            material=bz.Material(permittivity=eps_silicon),
        )
    design += bz.Rectangle(
        position=(design_min_x, design_min_y),
        width=design_width,
        height=design_height,
        material=bz.Material(permittivity=eps_silicon),
    )

    monitor_frequencies = bz.LIGHT_SPEED / (np.asarray(wavelengths_nm) * bz.nm)
    field_frequency = bz.LIGHT_SPEED / (FIELD_WAVELENGTH_NM * bz.nm)
    aperture = wg_width + 2.0 * mode_padding
    mode_spec = bz.ModeSpec(num_modes=3, mode_index=0, polarization="tm")
    monitor_left_x = pml_thickness + port_pml_offset + monitor_offset
    monitor_right_x = extent_x - monitor_left_x
    ports = (
        bz.Port(
            center=(monitor_left_x, upper_y, 0.0),
            size=(0.0, aperture, wg_width),
            name="port1",
            direction="+",
            mode_spec=mode_spec,
        ),
        bz.Port(
            center=(monitor_right_x, upper_y, 0.0),
            size=(0.0, aperture, wg_width),
            name="port2",
            direction="-",
            mode_spec=mode_spec,
        ),
        bz.Port(
            center=(monitor_right_x, lower_y, 0.0),
            size=(0.0, aperture, wg_width),
            name="port3",
            direction="-",
            mode_spec=mode_spec,
        ),
        bz.Port(
            center=(monitor_left_x, lower_y, 0.0),
            size=(0.0, aperture, wg_width),
            name="port4",
            direction="+",
            mode_spec=mode_spec,
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
        center=(pml_thickness + port_pml_offset, upper_y, 0.0),
        size=ports[0].size,
        source_time=source_time,
        direction="+",
        mode_spec=mode_spec,
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
        "design_width_m": float(design_width),
        "design_height_m": float(design_height),
        "waveguide_width_m": float(wg_width),
        "waveguide_separation_m": float(wg_separation),
        "mode_padding_m": float(mode_padding),
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
    s31,
    s41,
    *,
    reflection_target_db: float = -20.0,
    transmission_target_db: float = -3.5,
):
    """Equation 12 with configurable splitter performance margins."""

    power = jnp.stack(
        (
            jnp.abs(s11) ** 2,
            jnp.abs(s21) ** 2,
            jnp.abs(s31) ** 2,
            jnp.abs(s41) ** 2,
        ),
        axis=-1,
    )
    cutoff = jnp.asarray(
        (
            10.0 ** (reflection_target_db / 10.0),
            10.0 ** (transmission_target_db / 10.0),
            10.0 ** (transmission_target_db / 10.0),
            10.0 ** (reflection_target_db / 10.0),
        )
    )
    sign = jnp.asarray((1.0, -1.0, -1.0, 1.0))
    valid_width = jnp.asarray(
        (
            cutoff[0],
            1.0 - cutoff[1],
            1.0 - cutoff[2],
            cutoff[3],
        )
    )
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
    generation_mode: GenerationMode = "paper-quarter",
    reflection_target_db: float = -20.0,
    transmission_target_db: float = -3.5,
    initial_latent: np.ndarray | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 10,
    snapshot_interval: int = 10,
):
    """Optimize a paper-feasible symmetric splitter design."""

    trainable = problem.differentiable("port1")
    resolution = problem.simulations["port1"].resolution
    brush_pixels = max(
        1,
        int(np.ceil(float(brush_size) / float(resolution) - 1e-9)),
    )
    brush = paper_circular_brush(brush_pixels)
    full_shape = tuple(int(value) for value in problem.variable_shape)
    if full_shape[0] % 2 or full_shape[1] % 2:
        raise ValueError(
            "The symmetric beam-splitter design must have even dimensions."
        )
    quarter_shape = (full_shape[0] // 2, full_shape[1] // 2)
    if generation_mode not in {
        "paper-quarter",
        "paper-full",
        "paper-ports",
        "side-conditioned",
        "conditioned-full",
    }:
        raise ValueError(f"Unknown splitter generation mode: {generation_mode}.")
    if generation_mode == "conditioned-full":
        boundary = splitter_boundary_context(full_shape, brush, resolution)
    elif generation_mode == "side-conditioned":
        boundary = splitter_side_context(full_shape, brush, resolution)
    elif generation_mode == "paper-ports":
        boundary = splitter_port_context(full_shape, brush, resolution)
    else:
        boundary = {
            "canvas_shape": full_shape,
            "design_slices": (slice(None), slice(None)),
            "generated_pixels": np.ones(full_shape, dtype=bool),
            "fixed_solid": np.zeros(full_shape, dtype=bool),
            "fixed_void": np.zeros(full_shape, dtype=bool),
            "port_rows": splitter_port_rows(full_shape, resolution),
            "boundary_weight": np.zeros(full_shape, dtype=float),
            "boundary_target": np.zeros(full_shape, dtype=bool),
            "transition_depth": 0,
        }
    canvas_shape = boundary["canvas_shape"]
    design_slices = boundary["design_slices"]
    generated_pixels = boundary["generated_pixels"]
    fixed_solid = boundary["fixed_solid"]
    fixed_void = boundary["fixed_void"]
    boundary_weight = boundary["boundary_weight"]
    boundary_target = boundary["boundary_target"]

    if initial_latent is None:
        rng = np.random.default_rng(seed)
        latent = jnp.asarray(
            initial_bias + initial_noise * rng.standard_normal(quarter_shape),
            dtype=jnp.float32,
        )
    else:
        latent = jnp.asarray(initial_latent, dtype=jnp.float32)
        if tuple(latent.shape) != quarter_shape:
            raise ValueError(
                f"Initial latent has shape {latent.shape}, expected {quarter_shape}."
            )
    first_moment = jnp.zeros_like(latent)
    second_moment = jnp.zeros_like(latent)

    def generate(value):
        if generation_mode == "paper-quarter":
            reward = filtered_reward(value, brush, beta=beta)
            generated = conditional_generator(reward, brush)
            density = np.asarray(mirror_quarter(jnp.asarray(generated.density)))
            return generated, density, density

        full_latent = mirror_quarter(value)
        reward = symmetrize_xy(filtered_reward(full_latent, brush, beta=beta))
        if generation_mode == "paper-full":
            generated = conditional_generator(
                reward,
                brush,
                reflection_symmetry="xy",
            )
            return generated, generated.density, generated.density
        if generation_mode == "paper-ports":
            canvas_reward = jnp.zeros(canvas_shape, dtype=reward.dtype)
            canvas_reward = canvas_reward.at[design_slices].set(reward)
            generated = conditional_generator(
                canvas_reward,
                brush,
                fixed_solid=fixed_solid,
                fixed_void=fixed_void,
                reflection_symmetry="xy",
            )
            return (
                generated,
                generated.density[design_slices],
                generated.density,
            )
        if generation_mode == "side-conditioned":
            canvas_reward = jnp.zeros(canvas_shape, dtype=reward.dtype)
            canvas_reward = canvas_reward.at[design_slices].set(
                (1.0 - boundary_weight) * reward
                + boundary_weight
                * BOUNDARY_PHASE_REWARD
                * (2.0 * boundary_target - 1.0)
            )
            generated = conditional_generator(
                canvas_reward,
                brush,
                fixed_solid=fixed_solid,
                fixed_void=fixed_void,
                reflection_symmetry="xy",
            )
            return (
                generated,
                generated.density[design_slices],
                generated.density,
            )

        canvas_reward = jnp.zeros(canvas_shape, dtype=reward.dtype)
        canvas_reward = canvas_reward.at[design_slices].set(
            (1.0 - boundary_weight) * reward
            + boundary_weight * BOUNDARY_PHASE_REWARD * (2.0 * boundary_target - 1.0)
        )
        canvas_reward = canvas_reward.at[fixed_solid].set(BOUNDARY_PHASE_REWARD)
        canvas_reward = canvas_reward.at[fixed_void].set(-BOUNDARY_PHASE_REWARD)
        generated = conditional_generator(
            canvas_reward,
            brush,
            reflection_symmetry="xy",
        )
        if not np.all(generated.density[fixed_solid] == 1.0) or not np.all(
            generated.density[fixed_void] == 0.0
        ):
            raise RuntimeError(
                "Boundary-conditioned generation changed a required exterior phase."
            )
        return generated, generated.density[design_slices], generated.density

    initial_generated, initial_density, _ = generate(latent)
    projector = problem.port_projector(
        jnp.asarray(initial_density),
        source_port="port1",
    )

    def objective(result):
        values = [
            projector.s_parameter(
                result,
                source_port="port1",
                output_port=f"port{index}",
            )
            for index in range(1, 5)
        ]
        return paper_loss(
            *values,
            reflection_target_db=reflection_target_db,
            transmission_target_db=transmission_target_db,
        )

    value_and_grad_density = trainable.compile_value_and_grad(objective)

    def estimator(value):
        if generation_mode == "paper-quarter":
            reward = filtered_reward(value, brush, beta=beta)
            return mirror_quarter(0.5 * (reward + 1.0))
        full_latent = mirror_quarter(value)
        reward = symmetrize_xy(filtered_reward(full_latent, brush, beta=beta))
        density_proxy = 0.5 * (reward + 1.0)
        if generation_mode in {"paper-full", "paper-ports"}:
            return density_proxy
        return (1.0 - boundary_weight) * density_proxy + (
            boundary_weight * boundary_target
        )

    best_loss = np.inf
    best_step = 0
    best_latent = latent
    best_density = initial_density
    history: list[float] = []
    generator_steps: list[int] = []
    density_history: list[np.ndarray] = []
    latent_history: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    composite_best = None
    started = time.perf_counter()

    for step in range(1, steps + 1):
        generated, full_density, composite_density = generate(latent)
        density = jnp.asarray(full_density)
        loss, density_gradient = value_and_grad_density(density)
        _, pullback = jax.vjp(estimator, latent)
        gradient = pullback(density_gradient)[0]
        loss_value = float(loss)
        history.append(loss_value)
        generator_steps.append(generated.steps)
        if step == 1 or step % snapshot_interval == 0 or step == steps:
            density_history.append(np.asarray(full_density, dtype=np.uint8))
            latent_history.append(np.asarray(latent, dtype=np.float32))
            snapshot_steps.append(step)
        if np.isfinite(loss_value) and loss_value < best_loss:
            best_loss = loss_value
            best_step = step
            best_latent = latent
            best_density = full_density
            composite_best = composite_density

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
                f"silicon={float(np.mean(full_density)):.3f}, "
                f"generator_steps={generated.steps}"
            )

    if not np.isfinite(best_loss):
        raise RuntimeError("Optimization did not produce a finite objective.")
    _, verified_best, verified_composite = generate(best_latent)
    if not np.array_equal(verified_best, best_density):
        raise RuntimeError("The best latent no longer reproduces the saved design.")
    if composite_best is not None and not np.array_equal(
        verified_composite, composite_best
    ):
        raise RuntimeError("Best latent no longer reproduces its composite design.")
    solid_error, void_error = brush_feasibility_errors(verified_composite, brush)
    brush_feasible = not bool(np.any((solid_error | void_error) & generated_pixels))
    if not brush_feasible:
        raise RuntimeError(
            "Quarter reflection produced a design that is not brush-feasible."
        )
    return {
        "density": np.asarray(best_density),
        "composite_density": np.asarray(verified_composite),
        "composite_generated_pixels": np.asarray(generated_pixels),
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
    density_jax = jnp.asarray(density, dtype=jnp.float32)
    trainable = problem.differentiable("port1")
    result = trainable.run(density_jax)
    projector = problem.port_projector(density_jax, source_port="port1")
    s_parameters = []
    for index in range(1, 5):
        s_parameters.append(
            np.asarray(
                projector.s_parameter(
                    result,
                    source_port="port1",
                    output_port=f"port{index}",
                )
            )
        )
    field = np.asarray(result.field("domain_fields", "Ez")[0]).reshape(
        trainable.base_permittivity.shape
    )
    permittivity = np.asarray(trainable.permittivity(density_jax))
    return (*s_parameters, field, permittivity)


def save_topology_preview(
    output_dir: Path,
    composite_density: np.ndarray,
) -> Path:
    """Save the globally generated topology, including its exterior context."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "beamz_beam_splitter_topology.png"
    density = np.asarray(composite_density)
    fig, axis = plt.subplots(figsize=(8.5, 5.5 * density.shape[0] / density.shape[1]))
    axis.imshow(density, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    axis.contour(
        density,
        levels=[0.5],
        colors="#00c8ff",
        linewidths=1.2,
        origin="lower",
    )
    axis.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(output_path, dpi=180, facecolor="black")
    plt.close(fig)
    return output_path


def save_results(
    output_dir: Path,
    optimization,
    result,
    metadata,
    args,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    s11, s21, s31, s41, field, permittivity = result
    eps_low = metadata["eps_oxide"]
    eps_high = metadata["eps_silicon"]
    full_density = (permittivity - eps_low) / (eps_high - eps_low)
    wavelengths = SPECTRUM_WAVELENGTHS_NM
    s_values = (s11, s21, s31, s41)
    db_values = [
        10.0 * np.log10(np.maximum(np.abs(value) ** 2, 1e-12)) for value in s_values
    ]

    pml_cells = int(round(metadata["pml_thickness_m"] / metadata["resolution_m"]))
    plot_crop = (slice(pml_cells, -pml_cells),) * 2
    plotted_density = full_density[plot_crop]
    plotted_field = np.abs(field[plot_crop])
    plotted_field /= max(float(np.max(plotted_field)), 1e-30)

    fig = plt.figure(figsize=(14.04, 6.09))
    image_axis = fig.add_axes((0.021, 0.233, 0.333, 0.568))
    spectrum_axis = fig.add_axes((0.531, 0.233, 0.449, 0.568))
    image_axis.imshow(
        plotted_field,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    image_axis.contour(
        plotted_density,
        levels=[0.5],
        colors="white",
        linewidths=1.5,
        origin="lower",
    )
    image_axis.set_axis_off()
    image_axis.set_box_aspect(plotted_density.shape[0] / plotted_density.shape[1])

    colors = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3")
    labels = (r"$S_{11}$", r"$S_{21}$", r"$S_{31}$", r"$S_{41}$")
    for values, color, label in zip(db_values, colors, labels, strict=True):
        spectrum_axis.plot(wavelengths, values, color=color, lw=2.8, label=label)
    spectrum_axis.set_xlim(1258.0, 1303.0)
    spectrum_axis.set_ylim(-41.0, 1.0)
    spectrum_axis.set_xticks([1260, 1270, 1280, 1290, 1300])
    spectrum_axis.set_yticks([0, -20, -40])
    spectrum_axis.set_xlabel("Wavelength (nm)", fontsize=23)
    spectrum_axis.set_ylabel("dB", fontsize=23)
    spectrum_axis.tick_params(labelsize=20, width=1.6, length=8)
    spectrum_axis.grid(alpha=0.22)
    spectrum_axis.legend(loc="upper right", fontsize=21, framealpha=0.82)
    for spine in spectrum_axis.spines.values():
        spine.set_linewidth(1.8)
    fig.suptitle("Beam splitter", fontsize=30, y=0.97)
    fig.savefig(output_dir / "beamz_beam_splitter.png", dpi=100)
    fig.savefig(output_dir / "beamz_beam_splitter_highres.png", dpi=220)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "beamz_beam_splitter.npz",
        design_density=optimization["density"],
        best_latent=optimization["best_latent"],
        current_latent=optimization["current_latent"],
        first_moment=optimization["first_moment"],
        second_moment=optimization["second_moment"],
        completed_steps=np.asarray(optimization["completed_steps"]),
        best_step=np.asarray(optimization["best_step"]),
        best_loss=np.asarray(optimization["loss"]),
        full_density=full_density,
        ez_1280nm=field,
        wavelengths_nm=wavelengths,
        s11=s11,
        s21=s21,
        s31=s31,
        s41=s41,
        history=optimization["history"],
        generator_steps=optimization["generator_steps"],
        density_history=optimization["density_history"],
        latent_history=optimization["latent_history"],
        snapshot_steps=optimization["snapshot_steps"],
        brush_mask=optimization["brush"].mask,
        composite_density=optimization["composite_density"],
        composite_generated_pixels=optimization["composite_generated_pixels"],
        fixed_solid=optimization["fixed_solid"],
        fixed_void=optimization["fixed_void"],
        boundary_weight=optimization["boundary_weight"],
        boundary_target=optimization["boundary_target"],
        generation_mode=np.asarray(optimization["generation_mode"]),
    )

    design_indices = [
        int(np.argmin(np.abs(wavelengths - value)))
        for value in OPTIMIZATION_WAVELENGTHS_NM
    ]
    reference_metrics: dict[str, float] = {}
    if args.reference_image is not None and args.reference_image.is_file():
        reference_metrics = compare_reference_topology(
            full_density,
            args.reference_image,
            output_dir / "topology_reference_overlay.png",
            pml_cells=pml_cells,
        )

    design_slices = optimization["design_slices"]
    port_rows = optimization["port_rows"][design_slices[0]]
    binary_design = np.asarray(optimization["density"]) > 0.5
    composite_solid_error, composite_void_error = brush_feasibility_errors(
        optimization["composite_density"],
        optimization["brush"],
    )
    composite_error = (composite_solid_error | composite_void_error) & optimization[
        "composite_generated_pixels"
    ]

    summary = {
        "configuration": {
            **metadata,
            "optimization_run_time_s": float(args.run_time_fs * 1e-15),
            "evaluation_run_time_s": float(args.evaluation_run_time_fs * 1e-15),
            "optimization_wavelengths_nm": OPTIMIZATION_WAVELENGTHS_NM.tolist(),
            "spectrum_wavelengths_nm": wavelengths.tolist(),
            "field_wavelength_nm": FIELD_WAVELENGTH_NM,
            "brush_size_nm": float(args.brush_size_nm),
            "brush_size_pixels": int(optimization["brush_size_pixels"]),
            "projection_beta": float(args.beta),
            "completed_steps": int(optimization["completed_steps"]),
            "best_step": int(optimization["best_step"]),
            "snapshot_interval": int(args.snapshot_interval),
            "learning_rate": float(args.learning_rate),
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
            "boundary_conditioning_depth_pixels": int(optimization["transition_depth"]),
            "boundary_conditioning_depth_nm": float(
                optimization["transition_depth"] * metadata["resolution_m"] / bz.nm
            ),
            "symmetry": "xy",
        },
        "metrics": {
            "best_loss": optimization["loss"],
            "runtime_s": optimization["runtime_s"],
            "minimum_design_band_s21_db": float(np.min(db_values[1][design_indices])),
            "minimum_design_band_s31_db": float(np.min(db_values[2][design_indices])),
            "maximum_design_band_s11_db": float(np.max(db_values[0][design_indices])),
            "maximum_design_band_s41_db": float(np.max(db_values[3][design_indices])),
            "minimum_spectrum_s21_db": float(np.min(db_values[1])),
            "minimum_spectrum_s31_db": float(np.min(db_values[2])),
            "maximum_spectrum_s11_db": float(np.max(db_values[0])),
            "maximum_spectrum_s41_db": float(np.max(db_values[3])),
            "output_imbalance_max_db": float(
                np.max(np.abs(db_values[1] - db_values[2]))
            ),
            "silicon_fraction": float(np.mean(optimization["density"])),
            "binary_fraction": float(
                np.mean(
                    (optimization["density"] == 0.0) | (optimization["density"] == 1.0)
                )
            ),
            "brush_feasible": bool(optimization["brush_feasible"]),
            "xy_symmetric": bool(
                np.array_equal(optimization["density"], optimization["density"][::-1])
                and np.array_equal(
                    optimization["density"],
                    optimization["density"][:, ::-1],
                )
            ),
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
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Stop after writing the optimizer checkpoint for trajectory screening.",
    )
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
        help="Maximum S11/S41 power in dB used by Equation 12.",
    )
    parser.add_argument(
        "--transmission-target-db",
        type=float,
        default=-3.5,
        help="Minimum S21/S31 power in dB used by Equation 12.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=(
            "paper-quarter",
            "paper-full",
            "paper-ports",
            "side-conditioned",
            "conditioned-full",
        ),
        default="paper-quarter",
        help=(
            "Paper reproduction generates on the clipped design domain; "
            "conditioned-full retains the stricter exterior-canvas experiment."
        ),
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Warm-start from a different-resolution beam-splitter checkpoint.",
    )
    parser.add_argument(
        "--initialize-step",
        type=int,
        help=(
            "Use this one-based latent-history step from --initialize-from "
            "instead of its BeamZ-best latent."
        ),
    )
    default_reference = adjacent_reference_image("beam_splitter.png")
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=default_reference,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/ceviche_beam_splitter_o_band"),
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
    full_shape = tuple(int(value) for value in optimization_problem.variable_shape)
    quarter_shape = (full_shape[0] // 2, full_shape[1] // 2)
    initial_latent = (
        None
        if args.initialize_from is None
        else load_multiresolution_latent(
            args.initialize_from,
            quarter_shape,
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
