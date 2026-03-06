"""Compare legacy grid AA vs stratified-jitter AA on a polygon mesh raster.

Usage:
    uv run python examples/8_antialias_polygon_compare.py
    uv run python examples/8_antialias_polygon_compare.py --output /tmp/aa_compare.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.patches import Rectangle as PlotRectangle

from beamz import Design, Material, Polygon


def build_demo_design() -> tuple[Design, Polygon]:
    """Create a polygon with oblique edges to highlight raster AA differences."""
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
    design += poly
    return design, poly


def draw_polygon_panel(ax, design: Design, poly: Polygon) -> None:
    """Draw the input polygon geometry in physical coordinates."""
    ax.add_patch(
        PlotRectangle(
            (0.0, 0.0),
            design.width,
            design.height,
            facecolor="#f7f7f7",
            edgecolor="black",
            linewidth=1.0,
        )
    )
    verts = np.asarray([(v[0], v[1]) for v in poly.vertices], dtype=float)
    ax.add_patch(
        PlotPolygon(
            verts,
            closed=True,
            facecolor="#4C72B0",
            alpha=0.55,
            edgecolor="#1f2a44",
            linewidth=2.0,
        )
    )
    ax.set_title("Polygon Geometry")
    ax.set_xlim(0.0, design.width)
    ax.set_ylim(0.0, design.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


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
        default=0.2,
        help="Raster cell size in design units.",
    )
    parser.add_argument(
        "--jitter-samples",
        type=int,
        default=16,
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

    design, poly = build_demo_design()
    legacy_grid = design.rasterize(
        resolution=args.resolution,
        force_recompute=True,
        aa_mode="legacy_grid",
        aa_samples=9,
        aa_seed=0,
    )
    jitter_grid = design.rasterize(
        resolution=args.resolution,
        force_recompute=True,
        aa_mode="stratified_jitter",
        aa_samples=args.jitter_samples,
        aa_seed=args.seed,
    )

    eps_legacy = legacy_grid.permittivity
    eps_jitter = jitter_grid.permittivity
    extent = (0.0, design.width, 0.0, design.height)
    vmin = min(float(np.min(eps_legacy)), float(np.min(eps_jitter)))
    vmax = max(float(np.max(eps_legacy)), float(np.max(eps_jitter)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    draw_polygon_panel(axes[0], design, poly)

    axes[1].imshow(
        eps_legacy,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[1].set_title("Permittivity: legacy_grid (3x3)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    im2 = axes[2].imshow(
        eps_jitter,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[2].set_title(
        f"Permittivity: stratified_jitter ({args.jitter_samples} samples)"
    )
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    cbar = fig.colorbar(im2, ax=axes[1:], shrink=0.9)
    cbar.set_label("Relative permittivity")

    if args.output:
        fig.savefig(args.output, dpi=220, bbox_inches="tight")
        print(f"Saved comparison figure to: {args.output}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
