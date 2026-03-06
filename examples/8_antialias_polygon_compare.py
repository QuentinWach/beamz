"""Compare single-sample vs legacy grid AA vs stratified-jitter on a polygon raster.

Usage:
    uv run python examples/8_antialias_polygon_compare.py
    uv run python examples/8_antialias_polygon_compare.py --output /tmp/aa_compare.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as PlotCircle
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.patches import Rectangle as PlotRectangle

from beamz import Circle, Design, Material, Polygon


def build_demo_design() -> tuple[Design, Polygon, Circle]:
    """Create polygon+circle geometry to highlight raster AA differences."""
    width, height = 12.0, 8.0
    background = Material(permittivity=1.0)
    polygon_mat = Material(permittivity=12.0)

    design = Design(width=width, height=height, material=background)
    poly = Polygon(
        vertices=[
            (1.1, 1.0),
            (10.9, 1.5),
            (9.0, 6.8),
            (3.4, 6.1),
            (1.6, 3.8),
        ],
        material=polygon_mat,
    )
    circle = Circle(
        position=(2.2, 6.4),
        radius=0.9,
        material=polygon_mat,
    )
    design += poly
    design += circle
    return design, poly, circle


def draw_geometry_panel(ax, design: Design, poly: Polygon, circle: Circle) -> None:
    """Draw the input geometry in physical coordinates."""
    outside_color = "white"
    inside_color = "black"
    ax.add_patch(
        PlotRectangle(
            (0.0, 0.0),
            design.width,
            design.height,
            facecolor=outside_color,
            edgecolor="black",
            linewidth=1.0,
        )
    )
    verts = np.asarray([(v[0], v[1]) for v in poly.vertices], dtype=float)
    ax.add_patch(
        PlotPolygon(
            verts,
            closed=True,
            facecolor=inside_color,
            alpha=1.0,
            edgecolor="white",
            linewidth=1.2,
        )
    )
    ax.add_patch(
        PlotCircle(
            xy=(float(circle.position[0]), float(circle.position[1])),
            radius=float(circle.radius),
            facecolor=inside_color,
            edgecolor="white",
            linewidth=1.2,
        )
    )
    ax.set_title("Input Geometry", fontsize=10)
    ax.set_xlim(0.0, design.width)
    ax.set_ylim(0.0, design.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save the comparison figure.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.35,
        help="Raster cell size in design units.",
    )
    parser.add_argument(
        "--jitter-samples",
        type=int,
        default=2*2*2*2*2*2,
        help="Number of stratified-jitter samples per cell.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Deterministic seed for stratified-jitter sampling.",
    )
    args = parser.parse_args()

    os.environ.setdefault("BEAMZ_RASTER_TIMING", "0")

    design, poly, circle = build_demo_design()
    one_sample_grid = design.rasterize(
        resolution=args.resolution,
        force_recompute=True,
        aa_mode="legacy_grid",
        aa_samples=1,
        aa_seed=0,
    )
    legacy_grid = design.rasterize(
        resolution=args.resolution,
        force_recompute=True,
        aa_mode="legacy_grid",
        aa_samples=2*2*2*2*2*2,
        aa_seed=0,
    )
    jitter_grid = design.rasterize(
        resolution=args.resolution,
        force_recompute=True,
        aa_mode="stratified_jitter",
        aa_samples=args.jitter_samples,
        aa_seed=args.seed,
    )

    eps_one = one_sample_grid.permittivity
    eps_legacy = legacy_grid.permittivity
    eps_jitter = jitter_grid.permittivity
    extent = (0.0, design.width, 0.0, design.height)
    vmin = min(
        float(np.min(eps_one)),
        float(np.min(eps_legacy)),
        float(np.min(eps_jitter)),
    )
    vmax = max(
        float(np.max(eps_one)),
        float(np.max(eps_legacy)),
        float(np.max(eps_jitter)),
    )

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), constrained_layout=True, dpi=120)
    axes = axes.ravel()
    draw_geometry_panel(axes[0], design, poly, circle)

    axes[1].imshow(
        eps_one,
        origin="lower",
        extent=extent,
        cmap="Greys",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[1].set_title("Naive Sampling", fontsize=10)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(
        eps_legacy,
        origin="lower",
        extent=extent,
        cmap="Greys",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[2].set_title(f"Gridded Super-Sampling ({args.jitter_samples} Samples)", fontsize=10)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    axes[3].imshow(
        eps_jitter,
        origin="lower",
        extent=extent,
        cmap="Greys",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[3].set_title(
        f"Stratified Jitter ({args.jitter_samples} Samples)",
        fontsize=10,
    )
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    for idx, (ax, panel) in enumerate(zip(axes, ("a", "b", "c", "d"))):
        ax.text(
            0.02,
            0.98,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="black",
        )

    if args.output:
        fig.savefig(args.output, dpi=220, bbox_inches="tight")
        print(f"Saved comparison figure to: {args.output}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
