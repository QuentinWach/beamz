import sys
from pathlib import Path

from beamz import *
import numpy as np
from beamz import calc_optimal_fdtd_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _mpl_helpers import plot_design, plot_grid, run_with_snapshots

WL = 1.55*µm
TIME = 90*WL/LIGHT_SPEED
N_CORE, N_CLAD = 2.04, 1.444 # Si3N4, SiO2
WG_WIDTH = 0.565*µm
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), safety_factor=0.999, points_per_wavelength=20)

# Create the design
design = Design(width=18*µm, height=7*µm, material=Material(N_CLAD**2))
design += Rectangle(position=(0,3.5*µm-WG_WIDTH/2), width=18*µm, height=WG_WIDTH, material=Material(N_CORE**2))
#design += Rectangle(position=(9*µm-WG_WIDTH/2,0), width=WG_WIDTH, height=7*µm, material=Material(N_CORE**2))
plot_design(design)

# Rasterize the design
grid = design.rasterize(resolution=DX)
plot_grid(grid, field="permittivity")

# Create the signal & source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(
    time_steps,
    amplitude=1.0,
    frequency=LIGHT_SPEED / WL,
    phase=0,
    ramp_duration=WL * 20 / LIGHT_SPEED,
    t_max=TIME / 2,
)
source = ModeSource(
    grid=grid,
    center=(design.width/2, design.height/2),
    width=WG_WIDTH * 3.5, # Slightly wider than waveguide to capture mode tails, but not so wide to hit PML/boundaries
    wavelength=WL,
    pol="tm",
    signal=signal,
    direction="+x",
)

# Run the simulation
sim = Simulation(
    design=design, 
    sources=[source], 
    boundaries=[PML(edges='all', thickness=1.2*WL)],
    time=time_steps,
    resolution=DX
)
run_with_snapshots(
    sim,
    snapshot_field="Ez",
    snapshot_interval=20,
    cmap="twilight_zero",
    clean_visualization=True,
)
