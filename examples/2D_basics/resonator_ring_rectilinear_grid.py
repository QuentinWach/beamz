"""Create and inspect a smoothly graded grid for a 2D ring resonator."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import beamz as bz

WAVELENGTH = 1.55 * bz.um
MIN_STEPS_PER_WVL = 10
MIN_FEATURE_CELLS = 12
MAX_SCALE = 1.08
COUPLING_DX = 30 * bz.nm


def create_design() -> tuple[bz.Design, dict[str, float]]:
    """Return a silicon-nitride-like ring resonator and bus waveguide."""
    params = {
        "width": 16.0 * bz.um,
        "height": 13.0 * bz.um,
        "ring_radius": 4.0 * bz.um,
        "waveguide_width": 0.60 * bz.um,
        "coupling_gap": 0.25 * bz.um,
        "bus_center_y": 2.50 * bz.um,
        "n_core": 2.04,
        "n_clad": 1.444,
    }
    params["ring_center_x"] = 0.5 * params["width"]
    params["ring_center_y"] = (
        params["bus_center_y"]
        + 0.5 * params["waveguide_width"]
        + params["coupling_gap"]
        + params["ring_radius"]
        + 0.5 * params["waveguide_width"]
    )

    core = bz.Material(permittivity=params["n_core"] ** 2)
    clad = bz.Material(permittivity=params["n_clad"] ** 2)
    design = bz.Design(width=params["width"], height=params["height"], background=clad)
    design += bz.Rectangle(
        position=(
            0.0,
            params["bus_center_y"] - 0.5 * params["waveguide_width"],
        ),
        width=params["width"],
        height=params["waveguide_width"],
        material=core,
        color="#f59e0b",
    )
    design += bz.Ring(
        position=(params["ring_center_x"], params["ring_center_y"]),
        inner_radius=params["ring_radius"] - 0.5 * params["waveguide_width"],
        outer_radius=params["ring_radius"] + 0.5 * params["waveguide_width"],
        material=core,
        color="#f59e0b",
    )
    return design, params


def create_grid(
    design: bz.Design, params: dict[str, float]
) -> tuple[bz.GridSpec, bz.RectilinearGrid]:
    """Resolve an automatic grid with an x-only coupling-region override."""
    gap_center_y = (
        params["bus_center_y"]
        + 0.5 * params["waveguide_width"]
        + 0.5 * params["coupling_gap"]
    )
    spec = bz.GridSpec.graded(
        wavelength=WAVELENGTH,
        min_steps_per_wvl=MIN_STEPS_PER_WVL,
        min_feature_cells=MIN_FEATURE_CELLS,
        max_scale=MAX_SCALE,
        overrides=(
            bz.MeshOverride(
                center=(params["ring_center_x"], gap_center_y),
                size=(2.0 * bz.um, 1.5 * bz.um),
                dl=(COUPLING_DX, None),
            ),
        ),
    )
    return spec, spec.realize(design)


def _draw_exact_outlines(ax: plt.Axes, design: bz.Design) -> None:
    """Overlay the analytical geometry so staircasing is easy to assess."""
    for structure in design.structures:
        for vertices in (structure.vertices, *structure.interiors):
            path = np.asarray(vertices)
            closed = np.vstack((path[:, :2], path[0, :2])) / bz.um
            ax.plot(closed[:, 0], closed[:, 1], color="#07111e", linewidth=1.05)


def plot_grid(
    design: bz.Design,
    params: dict[str, float],
    grid: bz.RectilinearGrid,
    material_grid: bz.MaterialGrid,
    output: Path,
) -> None:
    """Plot the structure, coupling cells, and smooth spacing transition."""
    epsilon = np.asarray(material_grid.permittivity)
    x_um = grid.x_edges / bz.um
    y_um = grid.y_edges / bz.um
    dx_nm = np.diff(grid.x_edges) / bz.nm
    dy_nm = np.diff(grid.y_edges) / bz.nm
    quality = grid.quality_report()

    cmap = LinearSegmentedColormap.from_list(
        "ring_materials", ("#0f2742", "#2d7897", "#f59e0b")
    )
    figure = plt.figure(figsize=(15.0, 8.6))
    layout = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.38, 1.0),
        height_ratios=(1.0, 0.72),
        left=0.055,
        right=0.89,
        bottom=0.085,
        top=0.90,
        wspace=0.18,
        hspace=0.30,
    )
    full_ax = figure.add_subplot(layout[:, 0])
    zoom_ax = figure.add_subplot(layout[0, 1])
    spacing_layout = layout[1, 1].subgridspec(1, 2, wspace=0.40)
    x_spacing_ax = figure.add_subplot(spacing_layout[0, 0])
    y_spacing_ax = figure.add_subplot(spacing_layout[0, 1])

    image_options = {
        "shading": "flat",
        "cmap": cmap,
        "vmin": params["n_clad"] ** 2,
        "vmax": params["n_core"] ** 2,
    }
    full_mesh = full_ax.pcolormesh(x_um, y_um, epsilon, **image_options)
    _draw_exact_outlines(full_ax, design)

    # Draw every edge: this example is deliberately tuned to expose the grading.
    for edge in x_um:
        full_ax.axvline(edge, color="white", linewidth=0.22, alpha=0.24)
    for edge in y_um:
        full_ax.axhline(edge, color="white", linewidth=0.22, alpha=0.24)
    full_ax.set(
        xlim=(x_um[0], x_um[-1]),
        ylim=(y_um[0], y_um[-1]),
        xlabel="x (µm)",
        ylabel="y (µm)",
        title="Full device — every grid line",
    )
    full_ax.set_aspect("equal", adjustable="box")
    full_ax.text(
        0.018,
        0.982,
        (
            f"{grid.shape[0]:,} × {grid.shape[1]:,} cells\n"
            f"Δx = {dx_nm.min():.1f}–{dx_nm.max():.1f} nm\n"
            f"Δy = {dy_nm.min():.1f}–{dy_nm.max():.1f} nm\n"
            f"scale range: x {dx_nm.max() / dx_nm.min():.1f}×, "
            f"y {dy_nm.max() / dy_nm.min():.1f}×\n"
            f"worst adjacent ratio = {quality.max_adjacent_ratio:.3f}"
        ),
        transform=full_ax.transAxes,
        va="top",
        color="white",
        fontsize=10,
        linespacing=1.35,
        bbox={"facecolor": "#081524", "alpha": 0.9, "edgecolor": "white"},
    )

    # Show every actual cell around the sensitive bus-to-ring coupling region.
    zoom_ax.pcolormesh(
        x_um,
        y_um,
        epsilon,
        edgecolors=(1.0, 1.0, 1.0, 0.52),
        linewidth=0.32,
        antialiased=True,
        **image_options,
    )
    _draw_exact_outlines(zoom_ax, design)
    cx = params["ring_center_x"] / bz.um
    bus_y = params["bus_center_y"] / bz.um
    ring_bottom = (
        params["ring_center_y"]
        - params["ring_radius"]
        - 0.5 * params["waveguide_width"]
    ) / bz.um
    zoom_ax.set(
        xlim=(cx - 1.65, cx + 1.65),
        ylim=(bus_y - 0.55, ring_bottom + 1.15),
        xlabel="x (µm)",
        ylabel="y (µm)",
        title="Coupling region — every grid cell",
    )
    zoom_ax.set_aspect("equal", adjustable="box")

    x_centers = grid.centers("x") / bz.um
    y_centers = grid.centers("y") / bz.um
    x_spacing_ax.plot(
        x_centers,
        dx_nm,
        color="#c45c06",
        marker="o",
        markersize=2.3,
        linewidth=1.3,
    )
    x_spacing_ax.set(
        xlabel="x coordinate (µm)",
        ylabel="Δx (nm)",
        title="x grading: coupling override",
        xlim=(5.5, 10.5),
    )

    y_spacing_ax.plot(
        y_centers,
        dy_nm,
        color="#157d96",
        marker="o",
        markersize=2.3,
        linewidth=1.3,
    )
    y_spacing_ax.axhline(
        WAVELENGTH / params["n_core"] / MIN_STEPS_PER_WVL / bz.nm,
        color="#344054",
        linestyle="--",
        linewidth=1.0,
        label=f"λ/({MIN_STEPS_PER_WVL} n₍core₎)",
    )
    y_spacing_ax.set(
        xlabel="y coordinate (µm)",
        ylabel="Δy (nm)",
        title="y grading: features + gap",
        xlim=(1.45, 4.45),
    )
    for spacing_ax in (x_spacing_ax, y_spacing_ax):
        spacing_ax.grid(color="#98a2b3", alpha=0.25, linewidth=0.7)
        spacing_ax.tick_params(labelsize=8)
        spacing_ax.title.set_fontsize(10)
    y_spacing_ax.legend(loc="upper right", frameon=False, fontsize=7)

    colorbar_ax = figure.add_axes((0.92, 0.17, 0.022, 0.68))
    colorbar = figure.colorbar(full_mesh, cax=colorbar_ax)
    colorbar.set_label("Relative permittivity εᵣ")
    figure.suptitle(
        f"Geometry-aware graded ring mesh — adjacent scale ≤ {MAX_SCALE}",
        fontsize=15,
        y=0.965,
    )
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    design, params = create_design()
    _spec, grid = create_grid(design, params)
    material_grid = design.rasterize(
        grid,
        quality="balanced",
        smoothing="farjadpour_full",
        polarization="tm",
    )
    output = Path(__file__).with_suffix(".png")
    plot_grid(design, params, grid, material_grid, output)

    report = grid.quality_report()
    print(f"Saved {output}")
    print(f"Grid shape: {grid.shape}")
    print(f"Worst adjacent-cell ratio: {report.max_adjacent_ratio:.4f}")


if __name__ == "__main__":
    main()
