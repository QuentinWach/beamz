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
MAX_SCALE = 1.05


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


def create_grid(design: bz.Design) -> tuple[bz.GridSpec, bz.RectilinearGrid]:
    """Resolve a material- and geometry-aware grid with gentle grading."""
    spec = bz.GridSpec.graded(
        wavelength=WAVELENGTH,
        min_steps_per_wvl=16,
        min_feature_cells=6,
        max_scale=MAX_SCALE,
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
    figure = plt.figure(figsize=(15.0, 8.6), layout="constrained")
    layout = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.38, 1.0),
        height_ratios=(1.0, 0.72),
    )
    full_ax = figure.add_subplot(layout[:, 0])
    zoom_ax = figure.add_subplot(layout[0, 1])
    spacing_ax = figure.add_subplot(layout[1, 1])

    image_options = {
        "shading": "flat",
        "cmap": cmap,
        "vmin": params["n_clad"] ** 2,
        "vmax": params["n_core"] ** 2,
    }
    full_mesh = full_ax.pcolormesh(x_um, y_um, epsilon, **image_options)
    _draw_exact_outlines(full_ax, design)

    # A representative subset avoids hiding the device under 85,000 grid cells.
    for edge in x_um[::8]:
        full_ax.axvline(edge, color="white", linewidth=0.28, alpha=0.24)
    for edge in y_um[::6]:
        full_ax.axhline(edge, color="white", linewidth=0.28, alpha=0.24)
    full_ax.set(
        xlim=(x_um[0], x_um[-1]),
        ylim=(y_um[0], y_um[-1]),
        xlabel="x (µm)",
        ylabel="y (µm)",
        title="Full device — representative grid lines",
    )
    full_ax.set_aspect("equal", adjustable="box")
    full_ax.text(
        0.018,
        0.982,
        (
            f"{grid.shape[0]:,} × {grid.shape[1]:,} cells\n"
            f"Δx = {dx_nm.min():.1f}–{dx_nm.max():.1f} nm\n"
            f"Δy = {dy_nm.min():.1f}–{dy_nm.max():.1f} nm\n"
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

    y_centers = grid.centers("y") / bz.um
    spacing_ax.plot(
        y_centers,
        dy_nm,
        color="#157d96",
        marker="o",
        markersize=2.5,
        label="Δy (one marker per cell)",
        linewidth=1.35,
    )
    spacing_ax.axhline(
        WAVELENGTH / params["n_core"] / 16 / bz.nm,
        color="#344054",
        linestyle="--",
        linewidth=1.0,
        label="λ/(16 n₍core₎)",
    )
    spacing_ax.set(
        xlabel="y coordinate (µm)",
        ylabel="cell width Δy (nm)",
        title=f"Resolved lower transition (max scale = {MAX_SCALE})",
        xlim=(1.45, 4.45),
    )
    spacing_ax.grid(color="#98a2b3", alpha=0.25, linewidth=0.7)
    spacing_ax.legend(loc="upper right", frameon=False, ncols=2, fontsize=9)

    colorbar = figure.colorbar(full_mesh, ax=(full_ax, zoom_ax), shrink=0.8, pad=0.02)
    colorbar.set_label("Relative permittivity εᵣ")
    figure.suptitle(
        "Geometry-aware graded rectilinear mesh for a 2D ring resonator",
        fontsize=15,
    )
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    design, params = create_design()
    _spec, grid = create_grid(design)
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
