from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure local workspace package import when running from examples/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamz import (
    LIGHT_SPEED,
    Design,
    Material,
    ModeSource,
    PML,
    Rectangle,
    Simulation,
    Taper,
    calc_optimal_fdtd_params,
    ramped_cosine,
    µm,
)

# 2D MMI example.
# For the benchmark-oriented 3D version with artifact export, use examples/1_mmi_3d.py.


def show_slice(slice_data):
    extent, _, unit = slice_data.scaled_extent()
    style = dict(slice_data.style)
    cmap = style.pop("cmap", "viridis")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(slice_data.values, extent=extent, cmap=cmap, **style)
    ax.set_xlabel(slice_data.x_label or f"{slice_data.plane[0]} ({unit})")
    ax.set_ylabel(slice_data.y_label or f"{slice_data.plane[1]} ({unit})")
    ax.set_title(slice_data.title or slice_data.value_label)
    fig.colorbar(im, ax=ax, label=slice_data.value_label)
    fig.tight_layout()
    plt.show()

# Parameters
X, Y = 20 * µm, 10 * µm
WL = 1.55 * µm
TIME = 40 * WL / LIGHT_SPEED
N_CORE, N_CLAD = 2.04, 1.444
WG_W = 0.565 * µm
H, W, OFFSET = 3.5 * µm, 9 * µm, 1.05 * µm
MMI_TAPER = 1.5 * µm

DX, DT = calc_optimal_fdtd_params(
    WL,
    max(N_CORE, N_CLAD),
    dims=2,
    safety_factor=0.999,
    points_per_wavelength=20,
)

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, material=Material(N_CLAD**2))
mmi_body_start = X / 2 - W / 2 + MMI_TAPER
design += Rectangle(
    position=(0, Y / 2 - WG_W / 2),
    width=mmi_body_start,
    height=WG_W,
    material=Material(N_CORE**2),
)
design += Taper(
    position=(mmi_body_start, Y / 2),
    input_width=WG_W,
    output_width=H,
    length=MMI_TAPER,
    material=Material(N_CORE**2),
)
design += Rectangle(
    position=(mmi_body_start + MMI_TAPER, Y / 2 - H / 2),
    width=W - MMI_TAPER,
    height=H,
    material=Material(N_CORE**2),
)
design += Rectangle(
    position=(X / 2, Y / 2 + OFFSET - WG_W / 2),
    width=X / 2,
    height=WG_W,
    material=Material(N_CORE**2),
)
design += Rectangle(
    position=(X / 2, Y / 2 - OFFSET - WG_W / 2),
    width=X / 2,
    height=WG_W,
    material=Material(N_CORE**2),
)
show_slice(design.slice2d(resolution=DX))

# Rasterize the design
grid = design.rasterize(resolution=DX)
show_slice(grid.slice2d(field="permittivity"))

# Define the source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(
    time_steps,
    amplitude=1.0,
    frequency=LIGHT_SPEED / WL,
    ramp_duration=WL * 6 / LIGHT_SPEED,
    t_max=TIME / 2,
)
source = ModeSource(
    grid=grid,
    center=(3 * µm, Y / 2),
    width=WG_W * 3.5,
    wavelength=WL,
    pol="tm",
    signal=signal,
    direction="+x",
)

# Run the simulation
sim = Simulation(
    design=design,
    devices=[source],
    boundaries=[PML(edges="all", thickness=1.2 * WL)],
    time=time_steps,
    resolution=DX,
)
sim.run(animate_live="Ez", animation_interval=15, clean_visualization=True)
