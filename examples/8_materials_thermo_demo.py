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
    # Core region (Silicon)
    design += Rectangle(position=(2 * µm, 2 * µm, 0), width=6 * µm, height=2 * µm, material=Silicon())
    # SiO2 block
    design += Rectangle(position=(1 * µm, 6 * µm, 0), width=4 * µm, height=2 * µm, material=SiO2())
    # Si3N4 block
    design += Rectangle(position=(7 * µm, 6 * µm, 0), width=2 * µm, height=2 * µm, material=Si3N4())
    # Gold pad (heater)
    design += Rectangle(position=(6 * µm, 1 * µm, 0), width=2 * µm, height=1.5 * µm, material=Gold())
    # TiN strip
    design += Rectangle(position=(1 * µm, 1 * µm, 0), width=3 * µm, height=0.8 * µm, material=TiN())


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
    return gaussian > 0.2


def plot_maps(props_T0, props_T, temperature, design, resolution):
    extent = (0, design.width / µm, 0, design.height / µm)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    im0 = axes[0, 0].imshow(props_T0["permittivity"], origin="lower", extent=extent)
    axes[0, 0].set_title("Permittivity εr (T0)")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(temperature, origin="lower", extent=extent)
    axes[0, 1].set_title("Temperature (K)")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(props_T["permittivity"], origin="lower", extent=extent)
    axes[0, 2].set_title("Permittivity εr (T)")
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(props_T["conductivity"], origin="lower", extent=extent)
    axes[1, 0].set_title("Conductivity σ (T)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(props_T["k"], origin="lower", extent=extent)
    axes[1, 1].set_title("Thermal k (T)")
    plt.colorbar(im4, ax=axes[1, 1])

    # Leave the last panel for a delta map
    delta_eps = props_T["permittivity"] - props_T0["permittivity"]
    im5 = axes[1, 2].imshow(delta_eps, origin="lower", extent=extent)
    axes[1, 2].set_title("Δεr (T - T0)")
    plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

    fig.suptitle("Material + Thermodynamics Demo", fontsize=14)
    fig.tight_layout()
    plt.show()


def main():
    width, height = 10 * µm, 10 * µm
    resolution = 0.05 * µm

    design = Design(width=width, height=height, depth=0, material=Air())
    add_materials(design)

    heater_mask = make_heater_mask(design, resolution)
    heater_power = 2.0
    params = ThermalParams(thermal_dt=1e-12, tau_avg=0.0, steady_state=True, max_iters=5000)

    eps_r, temperature = apply_static_thermal(
        design,
        resolution,
        params=params,
        heater_mask=heater_mask,
        heater_power=heater_power,
    )

    props_T0 = design.evaluate_materials(resolution, None)
    props_T = design.evaluate_materials(resolution, temperature)

    plot_maps(props_T0, props_T, temperature, design, resolution)


if __name__ == "__main__":
    main()
