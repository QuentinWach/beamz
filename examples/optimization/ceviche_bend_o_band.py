"""Reproduce the broadband O-band waveguide bend from Schubert et al.

This is the configuration shown in ``ceviche-challenges/img/waveguide_bend.png``
and Figure 5 of "Inverse Design of Photonic Devices with Strict Foundry
Fabrication Constraints".  Unlike the smaller 1550 nm prefab example, it uses
a 1.6 um design region, 400 nm waveguides, oxide cladding, six O-band design
wavelengths, diagonal symmetry, and a binary 100 nm-filtered material map.

The BeamZ implementation follows the paper's conditional feasible-design
generator, smooth reward transform, straight-through estimator, scattering
window loss, and Adam hyperparameters.  Every forward FDTD simulation therefore
sees a binary design whose solid and void phases both satisfy the brush rule.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
    circular_brush,
    conditional_generator,
    filtered_reward,
    notched_square_brush,
    straight_through_gradient,
)
from beamz.optimization.challenges._common import (
    CHECKPOINT_SCHEMA_VERSION,
    MINIMUM_RUN_AFTER_SOURCE_PEAK_FS,
    SOURCE_FRACTIONAL_BANDWIDTH,
    SOURCE_OFFSET,
    adam_update,
    atomic_savez,
    load_multiresolution_latent,
    reference_panel_boundary,
)
from beamz.optimization.challenges._common import (
    resize_binary as _resize_binary,
)

OPTIMIZATION_WAVELENGTHS_NM = np.array([1265.0, 1270.0, 1275.0, 1285.0, 1290.0, 1295.0])
SPECTRUM_WAVELENGTHS_NM = np.arange(1260.0, 1300.0 + 0.1, 2.0)
FIELD_WAVELENGTH_NM = 1280.0
REFERENCE_PANEL_BOUNDS = (29, 121, 375, 467)
REFERENCE_IMAGE_SIZE = (1343, 609)
REFERENCE_PML_CELLS = 20
REFERENCE_CROPPED_SHAPE = (275, 275)
REFERENCE_DESIGN_BOUNDS = (75, 235)


def _validate_run_time(run_time: float, source_time: bz.GaussianPulse) -> None:
    """Reject time windows that cannot produce a converged bend spectrum."""

    source_peak_time = source_time.offset / source_time.fwidth
    minimum_run_time = source_peak_time + MINIMUM_RUN_AFTER_SOURCE_PEAK_FS * 1e-15
    if run_time + 1e-18 < minimum_run_time:
        run_time_fs = run_time / 1e-15
        source_peak_fs = source_peak_time / 1e-15
        minimum_fs = minimum_run_time / 1e-15
        raise ValueError(
            f"run_time={run_time_fs:.1f} fs is under-converged for the Figure 5 "
            f"source: its pulse peaks at {source_peak_fs:.1f} fs and requires "
            f"at least {minimum_fs:.1f} fs to leave a settling window. A short "
            "run produces a source-localized field and invalid S-parameters."
        )


def _binary_boundary(density: np.ndarray) -> np.ndarray:
    """Return the one-pixel material boundary of a binary density raster."""

    density = np.asarray(density) > 0.5
    boundary = np.zeros_like(density)
    vertical = density[1:, :] != density[:-1, :]
    horizontal = density[:, 1:] != density[:, :-1]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def _load_multiresolution_latent(
    checkpoint: Path,
    shape: tuple[int, int],
    *,
    brush_size: float,
    target_resolution: float,
    brush_shape: str,
) -> np.ndarray:
    """Upsample a BeamZ latent field and preserve its filtered-reward scale."""

    if shape[0] != shape[1]:
        raise ValueError("Multiresolution bend initialization requires square arrays.")
    resized = load_multiresolution_latent(
        checkpoint,
        shape,
        brush_size=brush_size,
        target_resolution=target_resolution,
        brush_shape=brush_shape,
    )
    return 0.5 * (resized + resized.T)


def compare_reference_topology(
    full_density: np.ndarray,
    reference_image: Path,
    output_path: Path | None = None,
    *,
    pml_cells: int | None = REFERENCE_PML_CELLS,
) -> dict[str, float]:
    """Measure contour agreement with the checked-in Figure 5 reference.

    The repository contains the final figure rather than the numerical design
    array.  Its white material contour is nevertheless an objective topology
    reference.  The symmetric contour distance is reported in both figure
    pixels and nanometers, and an overlay is saved for visual diagnosis.
    """

    reference_boundary = reference_panel_boundary(
        reference_image,
        bounds=REFERENCE_PANEL_BOUNDS,
        expected_image_size=REFERENCE_IMAGE_SIZE,
    )

    if pml_cells is not None:
        pml_cells = int(pml_cells)
        if pml_cells < 1 or min(full_density.shape) <= 2 * pml_cells:
            raise ValueError("pml_cells must leave a nonempty rendered domain.")
        rendered_density = full_density[
            pml_cells:-pml_cells,
            pml_cells:-pml_cells,
        ]
    else:
        rendered_density = full_density
    # Ceviche Challenges crops the 20-cell PML before plotting. Matplotlib's
    # ``origin="lower"`` maps array row zero to the bottom of the reference
    # panel, hence the vertical flip before comparing image pixels.
    beamz_boundary = _resize_binary(
        np.flipud(_binary_boundary(rendered_density)),
        reference_boundary.shape,
    )
    # The published antialiased contour is about three pixels wide.
    beamz_boundary = ndimage.binary_dilation(beamz_boundary, iterations=1)
    distance_to_reference = ndimage.distance_transform_edt(~reference_boundary)
    distance_to_beamz = ndimage.distance_transform_edt(~beamz_boundary)
    reference_to_beamz = float(np.mean(distance_to_beamz[reference_boundary]))
    beamz_to_reference = float(np.mean(distance_to_reference[beamz_boundary]))
    symmetric_pixels = 0.5 * (reference_to_beamz + beamz_to_reference)
    # The panel spans the 2.75 um non-PML domain.
    nanometers_per_pixel = 2750.0 / reference_boundary.shape[1]

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


def extract_reference_design(reference_image: Path) -> np.ndarray:
    """Digitize the published 160 x 160 binary design from its white contour.

    This is a validation utility, not an optimization input. It allows the
    exact repository figure to be cross-simulated in BeamZ even though the
    numerical final design was not checked into Ceviche Challenges.
    """

    boundary = reference_panel_boundary(
        reference_image,
        bounds=REFERENCE_PANEL_BOUNDS,
        expected_image_size=REFERENCE_IMAGE_SIZE,
    )
    boundary = ndimage.binary_dilation(
        boundary,
        iterations=1,
    )
    regions, count = ndimage.label(~boundary)
    adjacency = {index: set() for index in range(1, count + 1)}
    for row, column in zip(*np.nonzero(boundary), strict=True):
        neighbors = np.unique(
            regions[
                max(0, row - 3) : row + 4,
                max(0, column - 3) : column + 4,
            ]
        )
        neighbors = neighbors[neighbors > 0]
        for first in neighbors:
            for second in neighbors:
                if first != second:
                    adjacency[int(first)].add(int(second))

    exterior = int(regions[10, 10])
    phase = {exterior: 0}
    pending = [exterior]
    while pending:
        region = pending.pop()
        for neighbor in adjacency[region]:
            expected = 1 - phase[region]
            if neighbor in phase and phase[neighbor] != expected:
                raise RuntimeError("Published contour regions are not two-colorable.")
            if neighbor not in phase:
                phase[neighbor] = expected
                pending.append(neighbor)
    if len(phase) != count:
        raise RuntimeError("Published contour contains an unclassified region.")

    solid = np.isin(
        regions,
        [index for index, value in phase.items() if value == 1],
    )
    void = (regions > 0) & ~solid
    distance_to_solid = ndimage.distance_transform_edt(~solid)
    distance_to_void = ndimage.distance_transform_edt(~void)
    digitized = solid.copy()
    digitized[boundary] = distance_to_solid[boundary] <= distance_to_void[boundary]
    cropped_domain = np.flipud(_resize_binary(digitized, REFERENCE_CROPPED_SHAPE))
    design_min, design_max = REFERENCE_DESIGN_BOUNDS
    return np.asarray(
        cropped_domain[design_min:design_max, design_min:design_max],
        dtype=np.float32,
    )


def build_problem(
    *,
    resolution: float,
    run_time: float,
    wavelengths_nm: np.ndarray = OPTIMIZATION_WAVELENGTHS_NM,
    include_field_monitor: bool = False,
):
    """Build the paper's O-band bend as one broadband BeamZ FDTD simulation."""

    design_size = 1.6 * bz.um
    wg_width = 0.4 * bz.um
    # The remaining geometry follows the Ceviche Challenges bend prefab.  The
    # paper changes the design size, waveguide width, and materials, but does
    # not report a different port/PML layout.
    wg_length = round((0.75 * bz.um) / resolution) * resolution
    padding = 0.4 * bz.um
    pml_thickness = 20.0 * resolution
    port_pml_offset = 0.05 * bz.um
    monitor_offset = 0.05 * bz.um
    mode_padding = 0.75 * bz.um
    eps_oxide, eps_silicon = 2.25, 12.25

    extent = 2.0 * pml_thickness + wg_length + design_size + padding
    design_min = pml_thickness + wg_length
    design_max = design_min + design_size
    wg_center = design_min + 0.5 * design_size
    wg_extent = design_min

    design = bz.Design(
        width=extent,
        height=extent,
        material=bz.Material(permittivity=eps_oxide),
    )
    design += bz.Rectangle(
        position=(0.0, wg_center - 0.5 * wg_width),
        width=wg_extent,
        height=wg_width,
        material=bz.Material(permittivity=eps_silicon),
    )
    design += bz.Rectangle(
        position=(wg_center - 0.5 * wg_width, 0.0),
        width=wg_width,
        height=wg_extent,
        material=bz.Material(permittivity=eps_silicon),
    )
    # The static rectangle fixes the compilation geometry.  DesignRegion
    # replaces these cells with the differentiable binary map at run time.
    design += bz.Rectangle(
        position=(design_min, design_min),
        width=design_size,
        height=design_size,
        material=bz.Material(permittivity=eps_silicon),
    )

    monitor_frequencies = bz.LIGHT_SPEED / (np.asarray(wavelengths_nm) * bz.nm)
    field_frequency = bz.LIGHT_SPEED / (FIELD_WAVELENGTH_NM * bz.nm)
    center_frequency = field_frequency
    aperture = wg_width + 2.0 * mode_padding
    mode_spec = bz.ModeSpec(num_modes=3, mode_index=0, polarization="tm")
    ports = (
        bz.Port(
            center=(
                pml_thickness + port_pml_offset + monitor_offset,
                wg_center,
                0.0,
            ),
            size=(0.0, aperture, wg_width),
            name="port1",
            direction="+",
            mode_spec=mode_spec,
        ),
        bz.Port(
            center=(
                wg_center,
                pml_thickness + port_pml_offset + monitor_offset,
                0.0,
            ),
            size=(aperture, 0.0, wg_width),
            name="port2",
            direction="+",
            mode_spec=mode_spec,
        ),
    )

    dt = 0.95 * resolution / (bz.LIGHT_SPEED * np.sqrt(2.0))
    source_time = bz.GaussianPulse(
        freq0=center_frequency,
        fwidth=SOURCE_FRACTIONAL_BANDWIDTH * center_frequency,
        offset=SOURCE_OFFSET,
    )
    _validate_run_time(run_time, source_time)
    time_grid = np.arange(int(np.ceil(run_time / dt)), dtype=float) * dt
    source = bz.ModeSource(
        center=(pml_thickness + port_pml_offset, wg_center, 0.0),
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
                [field_frequency], fields=("Ez",), name="domain_fields"
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
        lower=(design_min, design_min),
        upper=(design_max, design_max),
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
        "extent_m": float(extent),
        "pml_thickness_m": float(pml_thickness),
        "design_bounds_m": [float(design_min), float(design_max)],
        "design_size_m": float(design_size),
        "waveguide_width_m": float(wg_width),
        "eps_oxide": eps_oxide,
        "eps_silicon": eps_silicon,
        "num_time_steps": int(len(time_grid)),
        "source_peak_time_s": float(source_time.offset / source_time.fwidth),
        "minimum_validated_run_time_s": float(
            source_time.offset / source_time.fwidth
            + MINIMUM_RUN_AFTER_SOURCE_PEAK_FS * 1e-15
        ),
        "monitor_wavelengths_nm": np.asarray(wavelengths_nm).tolist(),
    }
    return problem, metadata


def _paper_loss(
    s11: jax.Array,
    s21: jax.Array,
) -> jax.Array:
    """Equation 12: squared L2 softplus distance from the target window."""

    transmission = jnp.stack((jnp.abs(s11) ** 2, jnp.abs(s21) ** 2), axis=-1)
    lower_bound = jnp.asarray((0.0, 10.0 ** (-0.5 / 10.0)))
    upper_bound = jnp.asarray((10.0 ** (-20.0 / 10.0), 1.0))
    distance_to_lower = jnp.where(
        lower_bound > 0.0,
        lower_bound - transmission,
        -1.0,
    )
    distance_to_upper = jnp.where(
        upper_bound < 1.0,
        transmission - upper_bound,
        -1.0,
    )
    signed_pseudodistance = jnp.maximum(distance_to_lower, distance_to_upper)
    signed_pseudodistance /= jnp.min(upper_bound - lower_bound)
    return jnp.sum(jax.nn.softplus(signed_pseudodistance) ** 2)


def _write_optimizer_checkpoint(
    path: Path,
    *,
    current_latent,
    first_moment,
    second_moment,
    completed_steps,
    best_latent,
    best_density,
    best_loss,
    best_step,
    history,
    generator_steps,
    density_history,
    latent_history,
    snapshot_steps,
    brush_mask,
    brush_shape,
    beta,
    normalize_transform,
    fixed_exterior,
):
    """Atomically save everything needed for an exact Adam continuation."""

    atomic_savez(
        path,
        schema_version=np.asarray(CHECKPOINT_SCHEMA_VERSION),
        current_latent=np.asarray(current_latent),
        first_moment=np.asarray(first_moment),
        second_moment=np.asarray(second_moment),
        completed_steps=np.asarray(completed_steps),
        best_latent=np.asarray(best_latent),
        best_density=np.asarray(best_density),
        best_loss=np.asarray(best_loss),
        best_step=np.asarray(best_step),
        history=np.asarray(history),
        generator_steps=np.asarray(generator_steps),
        density_history=np.asarray(density_history, dtype=np.uint8),
        latent_history=np.asarray(latent_history, dtype=np.float32),
        snapshot_steps=np.asarray(snapshot_steps, dtype=np.int32),
        brush_mask=np.asarray(brush_mask, dtype=np.uint8),
        brush_shape=np.asarray(brush_shape),
        beta=np.asarray(beta),
        normalize_transform=np.asarray(normalize_transform),
        fixed_exterior=np.asarray(fixed_exterior),
    )


def optimize(
    problem,
    *,
    steps: int,
    learning_rate: float,
    brush_size: float,
    brush_shape: str,
    beta: float,
    normalize_transform: bool = False,
    fixed_exterior: bool = False,
    resume_state: dict[str, np.ndarray | float | int] | None = None,
    initial_latent: np.ndarray | None = None,
    initial_bias: float = 1e-3,
    initial_noise: float = 1e-4,
    seed: int = 0,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 10,
    snapshot_interval: int = 10,
):
    if resume_state is not None and initial_latent is not None:
        raise ValueError("resume_state and initial_latent are mutually exclusive.")
    trainable = problem.differentiable("port1")
    resolution = problem.simulations["port1"].resolution
    brush_pixels = max(
        1,
        int(np.ceil(float(brush_size) / float(resolution) - 1e-9)),
    )
    if brush_shape == "circular":
        brush = circular_brush(brush_pixels)
    elif brush_shape == "notched-square":
        if brush_pixels < 3:
            raise ValueError(
                "A notched-square brush must be at least three pixels wide."
            )
        brush = notched_square_brush(brush_pixels)
    else:
        raise ValueError(f"Unsupported brush shape: {brush_shape!r}.")

    design_shape = tuple(int(value) for value in problem.variable_shape)
    if fixed_exterior:
        # Optional stricter BeamZ extension: generate against the actual oxide
        # exterior and fixed 400 nm waveguides. The published challenge model
        # itself inlays only the 160 x 160 design variable, so clipped-domain
        # generation remains the reproduction default.
        boundary_padding = max(brush.shape)
        canvas_shape = tuple(value + 2 * boundary_padding for value in design_shape)
        design_slices = tuple(
            slice(boundary_padding, boundary_padding + value) for value in design_shape
        )
        exterior = np.ones(canvas_shape, dtype=bool)
        exterior[design_slices] = False
        fixed_solid_mask = np.zeros(canvas_shape, dtype=bool)
        waveguide_pixels = int(round((0.4 * bz.um) / resolution))
        center_y = boundary_padding + design_shape[0] // 2
        center_x = boundary_padding + design_shape[1] // 2
        half_low = waveguide_pixels // 2
        half_high = waveguide_pixels - half_low
        fixed_solid_mask[
            center_y - half_low : center_y + half_high,
            :boundary_padding,
        ] = True
        fixed_solid_mask[
            :boundary_padding,
            center_x - half_low : center_x + half_high,
        ] = True
        fixed_void_mask = exterior & ~fixed_solid_mask
    else:
        boundary_padding = 0
        canvas_shape = design_shape
        design_slices = tuple(slice(None) for _ in design_shape)
        fixed_solid_mask = np.zeros(canvas_shape, dtype=bool)
        fixed_void_mask = np.zeros(canvas_shape, dtype=bool)

    if resume_state is None:
        if initial_latent is None:
            rng = np.random.default_rng(seed)
            # The paper only specifies a random, positively biased
            # initialization whose first feasible design is solid. Keep the two
            # unpublished values explicit so reproduction sweeps can vary them.
            latent = jnp.asarray(
                initial_bias
                + initial_noise * rng.standard_normal(problem.variable_shape),
                dtype=jnp.float32,
            )
        else:
            latent = jnp.asarray(initial_latent, dtype=jnp.float32)
        first_moment = jnp.zeros_like(latent)
        second_moment = jnp.zeros_like(latent)
        completed_steps = 0
        best_latent = latent
        best_density = None
        best_loss = np.inf
        best_step = 0
        history: list[float] = []
        generator_steps: list[int] = []
        density_history: list[np.ndarray] = []
        latent_history: list[np.ndarray] = []
        snapshot_steps: list[int] = []
    else:
        latent = jnp.asarray(resume_state["current_latent"], dtype=jnp.float32)
        first_moment = jnp.asarray(resume_state["first_moment"], dtype=jnp.float32)
        second_moment = jnp.asarray(resume_state["second_moment"], dtype=jnp.float32)
        completed_steps = int(resume_state["completed_steps"])
        best_latent = jnp.asarray(
            resume_state.get("best_latent", latent), dtype=jnp.float32
        )
        best_density_value = resume_state.get("best_density")
        best_density = (
            None
            if best_density_value is None
            else np.asarray(best_density_value, dtype=np.float32)
        )
        best_loss = float(resume_state.get("best_loss", np.inf))
        best_step = int(resume_state.get("best_step", 0))
        history = np.asarray(
            resume_state.get("history", np.empty(0)), dtype=float
        ).tolist()
        generator_steps = np.asarray(
            resume_state.get("generator_steps", np.empty(0)), dtype=int
        ).tolist()
        previous_density_history = np.asarray(
            resume_state.get(
                "density_history",
                np.empty((0, *problem.variable_shape), dtype=np.uint8),
            ),
            dtype=np.uint8,
        )
        density_history = [value for value in previous_density_history]
        previous_latent_history = np.asarray(
            resume_state.get(
                "latent_history",
                np.empty((0, *problem.variable_shape), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        latent_history = [value for value in previous_latent_history]
        previous_snapshot_steps = resume_state.get("snapshot_steps")
        if previous_snapshot_steps is None:
            # Legacy checkpoints recorded one density per optimizer step.
            snapshot_steps = list(range(1, len(density_history) + 1))
        else:
            snapshot_steps = np.asarray(
                previous_snapshot_steps, dtype=np.int32
            ).tolist()
        if len(snapshot_steps) != len(density_history):
            raise ValueError(
                "Checkpoint snapshot_steps and density_history lengths differ."
            )
        if latent_history and len(latent_history) != len(snapshot_steps):
            raise ValueError(
                "Checkpoint latent_history and snapshot_steps lengths differ."
            )

    if tuple(latent.shape) != tuple(problem.variable_shape):
        raise ValueError(
            f"Checkpoint latent shape {tuple(latent.shape)} does not match "
            f"design shape {problem.variable_shape}; exact Adam continuation "
            "cannot resample optimizer state."
        )
    if first_moment.shape != latent.shape or second_moment.shape != latent.shape:
        raise ValueError("Checkpoint Adam moments do not match the latent shape.")

    def generate(latent_design):
        reward = filtered_reward(
            latent_design,
            brush,
            beta=beta,
            diagonal_symmetry=True,
            normalize_kernel=normalize_transform,
        )
        # Exterior pixels constrain feasibility but do not compete with the
        # learned interior reward.  A zero exterior reward preserves the
        # paper's fully solid first design while still forbidding brushes from
        # crossing into the wrong exterior phase.
        canvas_reward = jnp.zeros(canvas_shape, dtype=reward.dtype)
        canvas_reward = canvas_reward.at[design_slices].set(reward)
        full_generated = conditional_generator(
            canvas_reward,
            brush,
            fixed_solid=fixed_solid_mask,
            fixed_void=fixed_void_mask,
            diagonal_symmetry=True,
        )
        cropped = full_generated.density[design_slices]
        return (
            type(full_generated)(
                density=cropped,
                solid_touches=full_generated.solid_touches,
                void_touches=full_generated.void_touches,
                steps=full_generated.steps,
            ),
            full_generated.density,
        )

    initial_generated, _ = generate(latent)
    if (
        completed_steps == 0
        and initial_latent is None
        and not fixed_exterior
        and not np.all(initial_generated.density == 1.0)
    ):
        raise RuntimeError(
            "The positively biased initialization did not generate the paper's "
            "fully solid first design; increase --initial-bias or reduce "
            "--initial-noise."
        )
    initial_density = jnp.asarray(initial_generated.density)
    projector = problem.port_projector(initial_density, source_port="port1")

    def objective(result):
        s11 = projector.s_parameter(result, source_port="port1", output_port="port1")
        s21 = projector.s_parameter(result, source_port="port1", output_port="port2")
        return _paper_loss(s11, s21)

    value_and_grad_density = trainable.compile_value_and_grad(objective)
    if best_density is None:
        best_density = initial_generated.density
    started = time.perf_counter()
    for local_step in range(1, steps + 1):
        global_step = completed_steps + local_step
        generated, _ = generate(latent)
        density = jnp.asarray(generated.density)
        loss, density_gradient = value_and_grad_density(density)
        gradient = straight_through_gradient(
            latent,
            density_gradient,
            brush,
            beta=beta,
            diagonal_symmetry=True,
            normalize_kernel=normalize_transform,
        )
        loss_value = float(loss)
        history.append(loss_value)
        generator_steps.append(generated.steps)
        if (
            global_step == 1
            or global_step % snapshot_interval == 0
            or local_step == steps
        ):
            density_history.append(np.asarray(generated.density, dtype=np.uint8))
            latent_history.append(np.asarray(latent, dtype=np.float32))
            snapshot_steps.append(global_step)
        if np.isfinite(loss_value) and loss_value < best_loss:
            best_loss = loss_value
            best_latent = latent
            best_density = generated.density
            best_step = global_step
        latent, first_moment, second_moment = adam_update(
            latent,
            gradient,
            first_moment,
            second_moment,
            step=global_step,
            learning_rate=learning_rate,
        )
        if checkpoint_path is not None and (
            global_step % checkpoint_interval == 0 or local_step == steps
        ):
            _write_optimizer_checkpoint(
                checkpoint_path,
                current_latent=latent,
                first_moment=first_moment,
                second_moment=second_moment,
                completed_steps=global_step,
                best_latent=best_latent,
                best_density=best_density,
                best_loss=best_loss,
                best_step=best_step,
                history=history,
                generator_steps=generator_steps,
                density_history=density_history,
                latent_history=latent_history,
                snapshot_steps=snapshot_steps,
                brush_mask=brush.mask,
                brush_shape=brush_shape,
                beta=beta,
                normalize_transform=normalize_transform,
                fixed_exterior=fixed_exterior,
            )
        if local_step == 1 or global_step % 10 == 0 or local_step == steps:
            print(
                f"BeamZ step {global_step:3d}/{completed_steps + steps}: "
                f"loss={loss_value:.6g}, "
                f"silicon={float(jnp.mean(density)):.3f}, "
                f"generator_steps={generated.steps}"
            )
    if not np.isfinite(best_loss):
        raise RuntimeError("Optimization did not produce a finite objective value.")
    verified_best, verified_full_density = generate(best_latent)
    if not np.array_equal(verified_best.density, best_density):
        raise RuntimeError("Best latent state no longer reproduces the saved design.")
    solid_error, void_error = brush_feasibility_errors(verified_full_density, brush)
    generated_pixels = ~(fixed_solid_mask | fixed_void_mask)
    if np.any((solid_error | void_error) & generated_pixels):
        raise RuntimeError(
            "Best conditional-generator design is not brush-feasible in its "
            "generated region."
        )
    return {
        "density": np.asarray(best_density),
        "best_latent": np.asarray(best_latent),
        "current_latent": np.asarray(latent),
        "first_moment": np.asarray(first_moment),
        "second_moment": np.asarray(second_moment),
        "completed_steps": completed_steps + steps,
        "best_step": best_step,
        "loss": float(best_loss),
        "history": np.asarray(history),
        "generator_steps": np.asarray(generator_steps),
        "density_history": np.asarray(density_history, dtype=np.uint8),
        "latent_history": np.asarray(latent_history, dtype=np.float32),
        "snapshot_steps": np.asarray(snapshot_steps, dtype=np.int32),
        "runtime_s": time.perf_counter() - started,
        "projector": projector,
        "brush": brush,
        "brush_size_pixels": brush_pixels,
        "brush_shape": brush_shape,
        "normalize_transform": normalize_transform,
        "brush_feasible": True,
        "boundary_padding_pixels": boundary_padding,
        "fixed_exterior": fixed_exterior,
    }


def evaluate(problem, optimization):
    density = jnp.asarray(optimization["density"], dtype=jnp.float32)
    trainable = problem.differentiable("port1")
    result = trainable.run(density)
    projector = problem.port_projector(density, source_port="port1")
    s11 = np.asarray(
        projector.s_parameter(result, source_port="port1", output_port="port1")
    )
    s21 = np.asarray(
        projector.s_parameter(result, source_port="port1", output_port="port2")
    )
    field = np.asarray(result.field("domain_fields", "Ez")[0]).reshape(
        trainable.base_permittivity.shape
    )
    permittivity = np.asarray(trainable.permittivity(density))
    return s11, s21, field, permittivity


def save_results(output_dir: Path, optimization, result, metadata, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    s11, s21, field, permittivity = result
    eps_low = metadata["eps_oxide"]
    eps_high = metadata["eps_silicon"]
    full_density = (permittivity - eps_low) / (eps_high - eps_low)
    wavelengths = SPECTRUM_WAVELENGTHS_NM
    reflection_db = 10.0 * np.log10(np.maximum(np.abs(s11) ** 2, 1e-12))
    transmission_db = 10.0 * np.log10(np.maximum(np.abs(s21) ** 2, 1e-12))

    # Ceviche Challenges crops the 20-cell PML before rendering Figure 5.
    pml_cells = int(round(metadata["pml_thickness_m"] / metadata["resolution_m"]))
    plot_crop = (slice(pml_cells, -pml_cells),) * 2
    plotted_density = full_density[plot_crop]
    plotted_field = np.abs(field[plot_crop])
    plotted_field /= max(float(np.max(plotted_field)), 1e-30)

    # Match the composition and typography of the Ceviche challenge image.
    fig = plt.figure(figsize=(13.43, 6.09))
    # Explicit positions mirror the 1343 x 609 reference composition and avoid
    # constrained-layout shifts caused by the square field panel.
    image_axis = fig.add_axes((0.022, 0.233, 0.258, 0.568))
    spectrum_axis = fig.add_axes((0.510, 0.233, 0.470, 0.568))
    image_axis.imshow(plotted_field, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    image_axis.contour(
        plotted_density,
        levels=[0.5],
        colors="white",
        linewidths=1.5,
        origin="lower",
    )
    image_axis.set_axis_off()
    image_axis.set_box_aspect(1)

    spectrum_axis.plot(
        wavelengths, reflection_db, color="#e41a1c", lw=2.8, label=r"$S_{11}$"
    )
    spectrum_axis.plot(
        wavelengths, transmission_db, color="#377eb8", lw=2.8, label=r"$S_{21}$"
    )
    spectrum_axis.set_xlim(1258.0, 1303.2)
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
    fig.suptitle("Waveguide bend", fontsize=30, y=0.97)
    fig.savefig(output_dir / "beamz_waveguide_bend.png", dpi=100)
    fig.savefig(output_dir / "beamz_waveguide_bend_highres.png", dpi=220)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "beamz_waveguide_bend.npz",
        design_density=optimization["density"],
        # ``latent`` remains an alias for older analysis scripts; exact resume
        # uses the complete current Adam state below.
        latent=optimization["best_latent"],
        best_latent=optimization["best_latent"],
        current_latent=optimization["current_latent"],
        first_moment=optimization["first_moment"],
        second_moment=optimization["second_moment"],
        completed_steps=np.asarray(optimization["completed_steps"]),
        best_step=np.asarray(optimization["best_step"]),
        best_loss=np.asarray(optimization["loss"]),
        best_density=optimization["density"],
        full_density=full_density,
        ez_1280nm=field,
        wavelengths_nm=wavelengths,
        s11=s11,
        s21=s21,
        history=optimization["history"],
        generator_steps=optimization["generator_steps"],
        density_history=optimization["density_history"],
        latent_history=optimization["latent_history"],
        snapshot_steps=optimization["snapshot_steps"],
        brush_mask=optimization["brush"].mask,
    )
    design_indices = [
        int(np.argmin(np.abs(wavelengths - value)))
        for value in OPTIMIZATION_WAVELENGTHS_NM
    ]
    reference_metrics: dict[str, float] = {}
    reference_image = args.reference_image
    if reference_image is not None and reference_image.is_file():
        reference_metrics = compare_reference_topology(
            full_density,
            reference_image,
            output_dir / "topology_reference_overlay.png",
        )

    summary = {
        "configuration": {
            **metadata,
            "optimization_run_time_s": float(args.run_time_fs * 1e-15),
            "optimization_wavelengths_nm": OPTIMIZATION_WAVELENGTHS_NM.tolist(),
            "spectrum_wavelengths_nm": wavelengths.tolist(),
            "field_wavelength_nm": FIELD_WAVELENGTH_NM,
            "brush_size_m": float(args.brush_size_nm * bz.nm),
            "projection_beta": float(args.beta),
            "snapshot_interval": int(args.snapshot_interval),
            "normalize_transform": bool(args.normalize_transform),
            "completed_steps": int(optimization["completed_steps"]),
            "best_step": int(optimization["best_step"]),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
            "initial_bias": float(args.initial_bias),
            "initial_noise": float(args.initial_noise),
            "initialization_checkpoint": (
                None if args.initialize_from is None else str(args.initialize_from)
            ),
            "resume_checkpoint": (None if args.resume is None else str(args.resume)),
            "brush_size_pixels": int(optimization["brush_size_pixels"]),
            "brush_shape": optimization["brush_shape"],
            "fixed_exterior": bool(optimization["fixed_exterior"]),
            "digitized_reference_validation": bool(args.evaluate_digitized_reference),
            "boundary_padding_pixels": int(optimization["boundary_padding_pixels"]),
            "guaranteed_notched_minimum_pixels": (
                int(optimization["brush_size_pixels"] - 2)
                if optimization["brush_shape"] == "notched-square"
                else None
            ),
        },
        "metrics": {
            "best_loss": optimization["loss"],
            "runtime_s": optimization["runtime_s"],
            "minimum_design_band_s21_db": float(
                np.min(transmission_db[design_indices])
            ),
            "maximum_design_band_s11_db": float(np.max(reflection_db[design_indices])),
            "minimum_spectrum_s21_db": float(np.min(transmission_db)),
            "maximum_spectrum_s11_db": float(np.max(reflection_db)),
            "silicon_fraction": float(np.mean(optimization["density"])),
            "binary_fraction": float(
                np.mean(
                    (optimization["density"] == 0.0) | (optimization["density"] == 1.0)
                )
            ),
            "brush_feasible": bool(optimization["brush_feasible"]),
            "mean_generator_steps": float(np.mean(optimization["generator_steps"])),
            **reference_metrics,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution-nm", type=float, default=10.0)
    parser.add_argument("--run-time-fs", type=float, default=800.0)
    parser.add_argument(
        "--evaluation-run-time-fs",
        type=float,
        default=1200.0,
        help="Longer final field/spectrum run; does not change the optimizer.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help=(
            "Stop after writing the optimizer checkpoint. Useful for screening "
            "independent trajectories before a separate long-window evaluation."
        ),
    )
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--brush-size-nm",
        "--brush-nm",
        dest="brush_size_nm",
        type=float,
        default=100.0,
        help="Physical brush diameter/width. Figure 5 uses a 100 nm circle.",
    )
    parser.add_argument(
        "--brush-shape",
        choices=("circular", "notched-square"),
        default="circular",
        help="Figure 5 uses circular; SI Figure S5 uses notched-square.",
    )
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=10,
        help="Store density/latent snapshots every N optimizer steps.",
    )
    parser.add_argument(
        "--normalize-transform",
        action="store_true",
        help="Ablation: normalize the convolution kernel (Eq. 11 is unnormalized).",
    )
    parser.add_argument(
        "--fixed-exterior",
        action="store_true",
        help=(
            "BeamZ extension: enforce the circular brush against oxide and "
            "waveguide context outside the variable region. The paper/code "
            "reproduction default uses the clipped 160 x 160 design array."
        ),
    )
    parser.add_argument("--initial-bias", type=float, default=1e-3)
    parser.add_argument("--initial-noise", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    default_reference = (
        Path(__file__).resolve().parents[3]
        / "ceviche-challenges"
        / "img"
        / "waveguide_bend.png"
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=default_reference if default_reference.is_file() else None,
        help=(
            "Published Ceviche Challenges Figure 5 image. When available, save "
            "a contour overlay and quantitative topology-distance metrics."
        ),
    )
    parser.add_argument(
        "--evaluate-digitized-reference",
        action="store_true",
        help=(
            "Skip optimization, recover the published binary contour from "
            "--reference-image, and cross-simulate it in BeamZ. The PNG "
            "antialiasing means this is a validation raster, not the authors' "
            "unpublished exact numerical mask."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="NPZ checkpoint containing the complete current Adam state.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "Warm-start at a different resolution from a BeamZ checkpoint's "
            "best latent field. The field is resampled and Adam moments reset; "
            "the published reference is never used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/ceviche_bend_o_band"),
    )
    args = parser.parse_args()
    if args.snapshot_interval < 1:
        parser.error("--snapshot-interval must be positive.")
    if args.resume is not None and args.initialize_from is not None:
        parser.error("--resume and --initialize-from are mutually exclusive.")

    if args.evaluate_digitized_reference:
        if args.reference_image is None or not args.reference_image.is_file():
            raise ValueError(
                "--evaluate-digitized-reference requires --reference-image."
            )
        density = extract_reference_design(args.reference_image)
        zeros = np.zeros_like(density)
        optimization = {
            "density": density,
            "best_latent": density,
            "current_latent": density,
            "first_moment": zeros,
            "second_moment": zeros,
            "completed_steps": 0,
            "best_step": 0,
            "loss": np.nan,
            "history": np.empty(0),
            "generator_steps": np.asarray([0]),
            "density_history": density[None].astype(np.uint8),
            "latent_history": density[None].astype(np.float32),
            "snapshot_steps": np.asarray([0], dtype=np.int32),
            "runtime_s": 0.0,
            "brush": circular_brush(
                int(round(args.brush_size_nm / args.resolution_nm))
            ),
            "brush_size_pixels": int(round(args.brush_size_nm / args.resolution_nm)),
            "brush_shape": "circular",
            "normalize_transform": False,
            "brush_feasible": False,
            "boundary_padding_pixels": 0,
            "fixed_exterior": False,
        }
    else:
        problem, _ = build_problem(
            resolution=args.resolution_nm * bz.nm,
            run_time=args.run_time_fs * 1e-15,
        )
        initial_latent = (
            None
            if args.initialize_from is None
            else _load_multiresolution_latent(
                args.initialize_from,
                tuple(int(value) for value in problem.variable_shape),
                brush_size=args.brush_size_nm * bz.nm,
                target_resolution=args.resolution_nm * bz.nm,
                brush_shape=args.brush_shape,
            )
        )
        resume_state = None
        if args.resume is not None:
            with np.load(args.resume) as previous:
                required = {
                    "current_latent",
                    "first_moment",
                    "second_moment",
                    "completed_steps",
                }
                missing = sorted(required - set(previous.files))
                if missing:
                    raise ValueError(
                        f"{args.resume} is a legacy latent-only result and "
                        f"cannot exactly resume Adam; missing {missing}."
                    )
                checkpoint_configuration = {
                    "brush_shape": args.brush_shape,
                    "beta": args.beta,
                    "normalize_transform": args.normalize_transform,
                    "fixed_exterior": args.fixed_exterior,
                }
                for name, expected in checkpoint_configuration.items():
                    if name not in previous.files:
                        continue
                    actual = np.asarray(previous[name]).item()
                    if isinstance(expected, float):
                        matches = np.isclose(float(actual), expected)
                    else:
                        matches = actual == expected
                    if not matches:
                        raise ValueError(
                            f"Checkpoint {name}={actual!r} does not match "
                            f"requested {expected!r}; this would not be an exact "
                            "continuation."
                        )
                resume_state = {
                    name: np.asarray(previous[name])
                    for name in (
                        "current_latent",
                        "first_moment",
                        "second_moment",
                        "completed_steps",
                        "best_latent",
                        "best_density",
                        "best_loss",
                        "best_step",
                        "history",
                        "generator_steps",
                        "density_history",
                        "latent_history",
                        "snapshot_steps",
                    )
                    if name in previous.files
                }
        optimization = optimize(
            problem,
            steps=args.steps,
            learning_rate=args.learning_rate,
            brush_size=args.brush_size_nm * bz.nm,
            brush_shape=args.brush_shape,
            beta=args.beta,
            normalize_transform=args.normalize_transform,
            fixed_exterior=args.fixed_exterior,
            resume_state=resume_state,
            initial_latent=initial_latent,
            initial_bias=args.initial_bias,
            initial_noise=args.initial_noise,
            seed=args.seed,
            checkpoint_path=args.output_dir / "optimizer_checkpoint.npz",
            snapshot_interval=args.snapshot_interval,
        )
    if args.skip_evaluation:
        checkpoint_summary = {
            "configuration": {
                "resolution_nm": float(args.resolution_nm),
                "run_time_fs": float(args.run_time_fs),
                "completed_steps": int(optimization["completed_steps"]),
                "learning_rate": float(args.learning_rate),
                "projection_beta": float(args.beta),
                "snapshot_interval": int(args.snapshot_interval),
                "brush_size_nm": float(args.brush_size_nm),
                "brush_shape": str(args.brush_shape),
                "fixed_exterior": bool(args.fixed_exterior),
                "seed": int(args.seed),
                "initial_bias": float(args.initial_bias),
                "initial_noise": float(args.initial_noise),
                "initialization_checkpoint": (
                    None if args.initialize_from is None else str(args.initialize_from)
                ),
                "resume_checkpoint": (
                    None if args.resume is None else str(args.resume)
                ),
                "published_reference_used_for_initialization": False,
                "published_reference_used_in_objective": False,
            },
            "metrics": {
                "best_step": int(optimization["best_step"]),
                "best_loss": float(optimization["loss"]),
                "brush_feasible": bool(optimization["brush_feasible"]),
                "binary_fraction": float(
                    np.mean(
                        (optimization["density"] == 0.0)
                        | (optimization["density"] == 1.0)
                    )
                ),
            },
            "checkpoint": str(args.output_dir / "optimizer_checkpoint.npz"),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "optimizer_summary.json").write_text(
            json.dumps(checkpoint_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(checkpoint_summary, indent=2, sort_keys=True))
        return
    evaluation_problem, metadata = build_problem(
        resolution=args.resolution_nm * bz.nm,
        run_time=args.evaluation_run_time_fs * 1e-15,
        wavelengths_nm=SPECTRUM_WAVELENGTHS_NM,
        include_field_monitor=True,
    )
    result = evaluate(evaluation_problem, optimization)
    if args.evaluate_digitized_reference:
        optimization["loss"] = float(
            _paper_loss(
                jnp.asarray(result[0]),
                jnp.asarray(result[1]),
            )
        )
    summary = save_results(args.output_dir, optimization, result, metadata, args)
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
