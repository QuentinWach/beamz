import numpy as np
import matplotlib.pyplot as plt

from beamz import (
    Design,
    Rectangle,
    ThermalParams,
    apply_static_thermal,
    µm,
)
from beamz.design.library import Air, SiO2, Si3N4, Silicon, Gold, TiN


def add_materials(design):
    # Ensure background has nonzero thermal conductivity for diffusion
    if hasattr(design.structures[0].material, "k"):
        design.structures[0].material.k = 0.026
        design.structures[0].material.dn_dT = 2e-5

    # Core region (Silicon) - keep as-is
    si = Silicon()
    si.dn_dT = 3.5e-4
    design += Rectangle(
        position=(2 * µm, 2 * µm, 0),
        width=6 * µm,
        height=2 * µm,
        material=si,
    )
    # SiO2 block - override permittivity explicitly
    sio2 = SiO2()
    sio2.dn_dT = 1.5e-5
    design += Rectangle(
        position=(1 * µm, 6 * µm, 0),
        width=4 * µm,
        height=2 * µm,
        material=sio2,
    )
    # Si3N4 block - override permittivity explicitly
    sin = Si3N4()
    sin.dn_dT = 3.0e-5
    design += Rectangle(
        position=(7 * µm, 6 * µm, 0),
        width=2 * µm,
        height=2 * µm,
        material=sin,
    )
    # Gold pad (heater) - override permittivity to avoid metal dominating εr map
    gold = Gold()
    gold.k = 318.0
    gold.dsigma_dT = 2.0e5
    gold.permittivity = 1.0
    design += Rectangle(
        position=(6 * µm, 1 * µm, 0), width=2 * µm, height=1.5 * µm, material=gold
    )
    # TiN strip - override permittivity for contrast
    tin = TiN()
    tin.k = 30.0
    tin.dsigma_dT = 5.0e4
    tin.permittivity = 6.0
    design += Rectangle(
        position=(1 * µm, 1 * µm, 0), width=3 * µm, height=0.8 * µm, material=tin
    )


def make_heater_mask(design, resolution):
    # Simple heater mask: a gaussian centered near the gold pad
    grid = design.rasterize(resolution)
    ny, nx = grid.permittivity.shape
    xs = np.linspace(0, design.width, nx, endpoint=False)
    ys = np.linspace(0, design.height, ny, endpoint=False)
    X, Y = np.meshgrid(xs, ys)
    x0, y0 = 7 * µm, 1.75 * µm
    sigma = 0.6 * µm
    gaussian = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma * sigma))
    return gaussian > 0.1


def draw_material_outlines(ax, design, color="w", lw=0.8, alpha=0.6):
    for structure in design.structures[1:]:
        if not isinstance(structure, Rectangle):
            continue
        x, y, _ = structure.position
        rect = plt.Rectangle(
            (x / µm, y / µm),
            structure.width / µm,
            structure.height / µm,
            fill=False,
            edgecolor=color,
            linewidth=lw,
            alpha=alpha,
        )
        ax.add_patch(rect)


def add_material_legend(fig):
    labels = [
        ("Air", "lightgray"),
        ("Si", "#1f77b4"),
        ("SiO2", "#2ca02c"),
        ("Si3N4", "#ff7f0e"),
        ("Au (heater)", "#d62728"),
        ("TiN", "#9467bd"),
    ]
    handles = [
        plt.Line2D([0], [0], color=color, lw=6) for _, color in labels
    ]
    fig.legend(
        handles,
        [label for label, _ in labels],
        loc="lower center",
        ncol=3,
        frameon=False,
    )


def plot_maps(props_T0, props_T, temperature, design, resolution, material_id):
    extent = (0, design.width / µm, 0, design.height / µm)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    eps0 = props_T0["permittivity"]
    epsT = props_T["permittivity"]
    sigmaT = props_T["conductivity"]
    kT = props_T["k"]
    delta_eps = epsT - eps0
    delta_T = temperature - props_T0["T0"]
    eps0_plot = eps0
    epsT_plot = epsT

    eps_vmin = np.nanpercentile(eps0_plot, 2)
    eps_vmax = np.nanpercentile(eps0_plot, 98)
    temp_vmin = np.percentile(temperature, 5)
    temp_vmax = np.percentile(temperature, 95)
    if temp_vmax - temp_vmin < 1e-6:
        temp_vmax = temp_vmin + 1.0

    eps_cmap = plt.cm.viridis.copy()

    im0 = axes[0, 0].imshow(
        eps0_plot,
        origin="lower",
        extent=extent,
        vmin=eps_vmin,
        vmax=eps_vmax,
        cmap=eps_cmap,
    )
    axes[0, 0].set_title("Permittivity εr (T0)")
    plt.colorbar(im0, ax=axes[0, 0])
    draw_material_outlines(axes[0, 0], design)

    im1 = axes[0, 1].imshow(
        temperature, origin="lower", extent=extent, vmin=temp_vmin, vmax=temp_vmax
    )
    axes[0, 1].set_title("Temperature (K)")
    plt.colorbar(im1, ax=axes[0, 1])
    draw_material_outlines(axes[0, 1], design)

    im2 = axes[0, 2].imshow(
        epsT_plot,
        origin="lower",
        extent=extent,
        vmin=eps_vmin,
        vmax=eps_vmax,
        cmap=eps_cmap,
    )
    axes[0, 2].set_title("Permittivity εr (T)")
    plt.colorbar(im2, ax=axes[0, 2])
    draw_material_outlines(axes[0, 2], design)

    sigma_plot = np.log10(np.maximum(sigmaT, 1e-3))
    im3 = axes[1, 0].imshow(sigma_plot, origin="lower", extent=extent)
    axes[1, 0].set_title("log10 Conductivity σ (T)")
    plt.colorbar(im3, ax=axes[1, 0])
    draw_material_outlines(axes[1, 0], design)

    im4 = axes[1, 1].imshow(kT, origin="lower", extent=extent)
    axes[1, 1].set_title("Thermal k (T)")
    plt.colorbar(im4, ax=axes[1, 1])
    draw_material_outlines(axes[1, 1], design)

    # Material ID map (categorical)
    material_map = np.where(material_id < 0, 0, material_id + 1)
    cmap = plt.cm.get_cmap("tab20", int(np.max(material_map)) + 1)
    im5 = axes[0, 3].imshow(material_map, origin="lower", extent=extent, cmap=cmap)
    axes[0, 3].set_title("Material Map")
    plt.colorbar(im5, ax=axes[0, 3])
    draw_material_outlines(axes[0, 3], design)

    # Leave the last panel for a delta map
    delta_t_v = np.max(np.abs(delta_T))
    if delta_t_v == 0:
        delta_t_v = 1e-3
    im6 = axes[1, 2].imshow(
        delta_T, origin="lower", extent=extent, vmin=-delta_t_v, vmax=delta_t_v
    )
    axes[1, 2].set_title("ΔT (K)")
    plt.colorbar(im6, ax=axes[1, 2])
    draw_material_outlines(axes[1, 2], design)

    for ax in axes.flat:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

    # Hide unused panel
    axes[1, 3].axis("off")

    fig.suptitle("Material + Thermodynamics Demo", fontsize=14)
    add_material_legend(fig)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


def main():
    width, height = 10 * µm, 10 * µm
    resolution = 0.05 * µm

    design = Design(width=width, height=height, depth=0, material=Air())
    add_materials(design)

    heater_mask = make_heater_mask(design, resolution)
    delta_t_target = 50.0
    k_ref = 1.4
    heater_power = delta_t_target * k_ref / (resolution * resolution)
    fixed_temp_value = 300.0
    fixed_temp_mask = np.zeros_like(heater_mask, dtype=bool)
    fixed_temp_mask[0, :] = True
    fixed_temp_mask[-1, :] = True
    fixed_temp_mask[:, 0] = True
    fixed_temp_mask[:, -1] = True
    params = ThermalParams(
        thermal_dt=1e-12,
        tau_avg=0.0,
        steady_state=True,
        max_iters=8000,
        tol=1e-6,
    )

    eps_r, temperature = apply_static_thermal(
        design,
        resolution,
        params=params,
        heater_mask=heater_mask,
        heater_power=heater_power,
        fixed_temp_mask=fixed_temp_mask,
        fixed_temp_value=fixed_temp_value,
    )

    props_T0 = design.evaluate_materials(resolution, None)
    props_T = design.evaluate_materials(resolution, temperature)

    grid = design.rasterize(resolution)
    material_id = grid.get_material_id_grid()
    plot_maps(props_T0, props_T, temperature, design, resolution, material_id)


if __name__ == "__main__":
    main()
