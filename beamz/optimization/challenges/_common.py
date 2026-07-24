"""Shared utilities for strict-foundry challenge optimizations."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image

import beamz as bz
from beamz.optimization import (
    Brush2D,
    circular_brush,
    notched_square_brush,
)

SOURCE_FRACTIONAL_BANDWIDTH = 0.16
SOURCE_OFFSET = 4.0
MINIMUM_RUN_AFTER_SOURCE_PEAK_FS = 90.0
BOUNDARY_PHASE_REWARD = 20.0
CHECKPOINT_SCHEMA_VERSION = 2


def validate_run_time(run_time: float, source_time: bz.GaussianPulse) -> None:
    """Reject a time window that ends before the source has settled."""

    source_peak_time = source_time.offset / source_time.fwidth
    minimum_run_time = source_peak_time + MINIMUM_RUN_AFTER_SOURCE_PEAK_FS * 1e-15
    if run_time + 1e-18 < minimum_run_time:
        raise ValueError(
            f"run_time={run_time / 1e-15:.1f} fs is under-converged: the pulse "
            f"peaks at {source_peak_time / 1e-15:.1f} fs and requires at least "
            f"{minimum_run_time / 1e-15:.1f} fs."
        )


def resize_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary raster without introducing intermediate phases."""

    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def reference_panel_boundary(
    image_path: Path,
    *,
    bounds: tuple[int, int, int, int],
    expected_image_size: tuple[int, int],
    threshold: int = 230,
) -> np.ndarray:
    """Extract a bright contour from a versioned reference-figure panel.

    The explicit image-size check turns upstream figure/layout changes into a
    clear error instead of silently computing a meaningless topology metric.
    """

    image = Image.open(image_path).convert("RGB")
    if image.size != expected_image_size:
        raise ValueError(
            f"Reference image has size {image.size}, expected "
            f"{expected_image_size}; panel bounds need recalibration."
        )
    reference = np.asarray(image)
    x0, y0, x1, y1 = bounds
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError(f"Reference panel bounds {bounds} are outside {image.size}.")
    boundary = np.min(reference[y0:y1, x0:x1], axis=-1) >= int(threshold)
    if not np.any(boundary):
        raise ValueError("Reference panel contains no contour at the chosen threshold.")
    return boundary


def adjacent_reference_image(filename: str) -> Path | None:
    """Locate an image in an adjacent local ``ceviche-challenges`` checkout."""

    candidate = (
        Path(__file__).resolve().parents[4] / "ceviche-challenges" / "img" / filename
    )
    return candidate if candidate.is_file() else None


def paper_circular_brush(diameter: int) -> Brush2D:
    """Return the paper-compatible rasterized circular brush."""

    return circular_brush(int(diameter))


def atomic_savez(path: Path, **arrays) -> None:
    """Atomically replace a compressed NumPy checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def adam_update(
    latent,
    gradient,
    first_moment,
    second_moment,
    *,
    step: int,
    learning_rate: float,
):
    """Apply the Adam update and hyperparameters reported in the paper."""

    first_moment = 0.667 * first_moment + 0.333 * gradient
    second_moment = 0.9 * second_moment + 0.1 * gradient**2
    corrected_moment = first_moment / (1.0 - 0.667**step)
    corrected_variance = second_moment / (1.0 - 0.9**step)
    latent = latent - learning_rate * corrected_moment / (
        jnp.sqrt(corrected_variance) + 1e-8
    )
    return latent, first_moment, second_moment


def _brush_for_shape(shape: str, pixels: int) -> Brush2D:
    if shape == "circular":
        return paper_circular_brush(pixels)
    if shape == "notched-square":
        return notched_square_brush(pixels)
    raise ValueError(f"Unsupported brush shape: {shape!r}.")


def load_multiresolution_latent(
    checkpoint: Path,
    shape: tuple[int, int],
    *,
    brush_size: float,
    target_resolution: float,
    brush_shape: str = "circular",
    step: int | None = None,
) -> np.ndarray:
    """Resample a latent while preserving unnormalized filter reward scale."""

    with np.load(checkpoint) as data:
        if step is None:
            if "best_latent" not in data.files:
                raise ValueError(f"{checkpoint} does not contain best_latent.")
            source = np.asarray(data["best_latent"], dtype=np.float32)
        else:
            if step < 1:
                raise ValueError("Checkpoint steps are one-based and must be positive.")
            if "latent_history" not in data.files:
                raise ValueError(f"{checkpoint} does not contain latent_history.")
            history = np.asarray(data["latent_history"], dtype=np.float32)
            if "snapshot_steps" in data.files:
                snapshot_steps = np.asarray(data["snapshot_steps"], dtype=int)
                matches = np.flatnonzero(snapshot_steps == step)
                if not matches.size:
                    raise ValueError(
                        f"Checkpoint does not contain snapshot step {step}; "
                        f"available steps are {snapshot_steps.tolist()}."
                    )
                source = history[int(matches[0])]
            elif step <= len(history):
                source = history[step - 1]
            else:
                raise ValueError(
                    f"Checkpoint contains {len(history)} steps, not step {step}."
                )

    if source.ndim != 2:
        raise ValueError("Multiresolution latent checkpoints must be two-dimensional.")
    scale_y = float(shape[0]) / float(source.shape[0])
    scale_x = float(shape[1]) / float(source.shape[1])
    if not np.isclose(scale_y, scale_x, rtol=1e-6, atol=1e-9):
        raise ValueError(
            "Multiresolution initialization requires the same scale on both axes."
        )
    image = Image.fromarray(source, mode="F")
    resized = np.array(
        image.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
        copy=True,
    )
    source_resolution = float(target_resolution) * scale_y
    source_pixels = max(1, int(np.ceil(float(brush_size) / source_resolution - 1e-9)))
    target_pixels = max(
        1, int(np.ceil(float(brush_size) / float(target_resolution) - 1e-9))
    )
    source_area = _brush_for_shape(brush_shape, source_pixels).area
    target_area = _brush_for_shape(brush_shape, target_pixels).area
    resized *= float(source_area) / float(target_area)
    return resized


__all__ = [
    "MINIMUM_RUN_AFTER_SOURCE_PEAK_FS",
    "BOUNDARY_PHASE_REWARD",
    "CHECKPOINT_SCHEMA_VERSION",
    "SOURCE_FRACTIONAL_BANDWIDTH",
    "SOURCE_OFFSET",
    "atomic_savez",
    "adam_update",
    "adjacent_reference_image",
    "load_multiresolution_latent",
    "paper_circular_brush",
    "reference_panel_boundary",
    "resize_binary",
    "validate_run_time",
]
