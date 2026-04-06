import matplotlib.pyplot as plt
import numpy as np

from beamz import Design, Material, Polygon, Rectangle, um
from beamz.optimization.polygonize import density_to_polygons, density_to_shapely_geometry

UM = 1e-6
W = 8.0 * UM
H = 6.0 * UM
DX = 0.08 * UM
EPS_CORE = 3.48**2
EPS_CLAD = 1.0

WG_W = 0.55 * UM
OUT_GAP = 1.0 * UM
X_BOX0 = 2.2 * UM
X_BOX1 = 5.8 * UM
Y_BOX0 = 1.2 * UM
Y_BOX1 = 4.8 * UM
Y_IN = 0.5 * H
Y_TOP = Y_IN + 0.5 * (WG_W + OUT_GAP)
Y_BOT = Y_IN - 0.5 * (WG_W + OUT_GAP)

FINAL_AA_MODE = "stratified_jitter"
FINAL_AA_SAMPLES = 128
MIN_FEATURE_AREA_CELLS = 0.5


def add_fixed_waveguides(design):
    design += Rectangle(
        position=(0.0, Y_IN - 0.5 * WG_W),
        width=X_BOX0,
        height=WG_W,
        material=Material(permittivity=EPS_CORE),
    )
    design += Rectangle(
        position=(X_BOX1, Y_TOP - 0.5 * WG_W),
        width=W - X_BOX1,
        height=WG_W,
        material=Material(permittivity=EPS_CORE),
    )
    design += Rectangle(
        position=(X_BOX1, Y_BOT - 0.5 * WG_W),
        width=W - X_BOX1,
        height=WG_W,
        material=Material(permittivity=EPS_CORE),
    )


def synthetic_density():
    ny = int(H / DX)
    nx = int(W / DX)
    x = (np.arange(nx) + 0.5) * DX
    y = (np.arange(ny) + 0.5) * DX
    xx, yy = np.meshgrid(x, y)

    cx = 0.5 * (X_BOX0 + X_BOX1)
    cy = 0.5 * (Y_BOX0 + Y_BOX1)
    box_mask = (
        (xx >= X_BOX0)
        & (xx <= X_BOX1)
        & (yy >= Y_BOX0)
        & (yy <= Y_BOX1)
    )

    ridge = np.exp(-((xx - cx) / (0.55 * UM)) ** 2) * np.exp(-((yy - cy) / (1.45 * UM)) ** 2)
    branch_top = np.exp(-(((xx - (cx + 0.9 * UM)) / (0.8 * UM)) ** 2 + ((yy - Y_TOP) / (0.6 * UM)) ** 2))
    branch_bot = np.exp(-(((xx - (cx - 0.7 * UM)) / (0.7 * UM)) ** 2 + ((yy - Y_BOT) / (0.65 * UM)) ** 2))
    ripple = 0.18 * (
        np.sin(2.4 * np.pi * (xx - X_BOX0) / max(X_BOX1 - X_BOX0, 1e-30))
        + np.cos(2.0 * np.pi * (yy - Y_BOX0) / max(Y_BOX1 - Y_BOX0, 1e-30))
    )
    notch = 0.22 * np.exp(-(((xx - (cx + 0.15 * UM)) / (0.35 * UM)) ** 2 + ((yy - cy) / (0.45 * UM)) ** 2))
    hole_top = 0.45 * np.exp(-(((xx - (cx - 0.45 * UM)) / (0.28 * UM)) ** 2 + ((yy - (cy + 0.75 * UM)) / (0.22 * UM)) ** 2))
    hole_mid = 0.50 * np.exp(-(((xx - (cx + 0.10 * UM)) / (0.22 * UM)) ** 2 + ((yy - cy) / (0.20 * UM)) ** 2))
    hole_bot = 0.42 * np.exp(-(((xx - (cx + 0.55 * UM)) / (0.24 * UM)) ** 2 + ((yy - (cy - 0.70 * UM)) / (0.24 * UM)) ** 2))

    density = 0.55 * ridge + 0.42 * branch_top + 0.38 * branch_bot + ripple - notch - hole_top - hole_mid - hole_bot
    density = np.where(box_mask, density, 0.0)
    density = (density - density.min()) / max(density.max() - density.min(), 1e-30)
    density = np.where(box_mask, density, 0.0)
    return np.clip(density, 0.0, 1.0)


def build_final_design_from_density(density):
    final_design = Design(width=W, height=H, material=Material(permittivity=EPS_CLAD))
    add_fixed_waveguides(final_design)
    for poly in density_to_polygons(
        density,
        material=Material(permittivity=EPS_CORE),
        level=0.5,
        x0=X_BOX0,
        y0=Y_BOX0,
        dx=DX,
        min_area=MIN_FEATURE_AREA_CELLS * DX * DX,
    ):
        final_design += poly
    final_design.unify_polygons()
    return final_design


def main():
    density = synthetic_density()
    naive_binary = (density >= 0.5).astype(float)
    contour_geometry = density_to_shapely_geometry(
        density,
        level=0.5,
        x0=X_BOX0,
        y0=Y_BOX0,
        dx=DX,
        min_area=MIN_FEATURE_AREA_CELLS * DX * DX,
    )

    final_design = build_final_design_from_density(density)
    final_grid = final_design.rasterize(
        DX,
        aa_mode=FINAL_AA_MODE,
        aa_samples=FINAL_AA_SAMPLES,
        force_recompute=True,
    )
    rerasterized = np.clip(
        (np.asarray(final_grid.permittivity, dtype=float) - EPS_CLAD)
        / max(EPS_CORE - EPS_CLAD, 1e-30),
        0.0,
        1.0,
    )

    extent = [0.0, W / um, 0.0, H / um]
    level = [0.5]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)

    axes[0].imshow(density.T, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", extent=extent)
    axes[0].contour(
        ((np.arange(density.shape[1]) + 0.5) * DX) / um,
        ((np.arange(density.shape[0]) + 0.5) * DX) / um,
        density,
        levels=level,
        colors="tab:cyan",
        linewidths=1.0,
    )
    axes[0].set_title("Smooth Density + 0.5 Contour")

    axes[1].imshow(naive_binary.T, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", extent=extent)
    axes[1].set_title("Naive Pixel Threshold")

    axes[2].imshow(rerasterized.T, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", extent=extent)
    for poly in getattr(contour_geometry, "geoms", [contour_geometry]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        xy = np.asarray(poly.exterior.coords, dtype=float) / um
        axes[2].plot(xy[:, 0], xy[:, 1], color="white", lw=1.0)
        for ring in poly.interiors:
            hole_xy = np.asarray(ring.coords, dtype=float) / um
            axes[2].plot(hole_xy[:, 0], hole_xy[:, 1], color="white", lw=1.0)
    axes[2].set_title("Marching Squares + AA Rerasterized")

    for ax in axes:
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_aspect("equal")

    out = "density_polygon_demo.png"
    fig.savefig(out, dpi=180, facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
