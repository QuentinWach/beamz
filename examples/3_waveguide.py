from beamz import *
import numpy as np
from beamz.helpers import calc_optimal_fdtd_params

WL = 1.55*µm
TIME = 90*WL/LIGHT_SPEED
N_CORE, N_CLAD = 2.04, 1.444 # Si3N4, SiO2
WG_WIDTH = 0.565*µm
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), safety_factor=0.999, points_per_wavelength=100)

# Create the design
design = Design(width=18*µm, height=7*µm, material=Material(N_CLAD**2), pml_size=WL)
design += Rectangle(position=(0,3.5*µm-WG_WIDTH/2), width=18*µm, height=WG_WIDTH, material=Material(N_CORE**2))
design.show()

# Rasterize the design
grid = design.rasterize(resolution=DX)
grid.show(field="permittivity")

# Create the signal & source (visualizing modes only)
source = ModeSource(
    grid=grid,
    plane=dict(center=(4*µm, design.height/2), size=(-1*µm, 1*µm)),
    wavelength=WL,
    num_modes=2
)
source.show()

# If you wish to run a full FDTD simulation, instantiate FDTD with this grid
# and add a separate source implementation tailored for the simulation backend.


# Run the simulation
#sim = FDTD(design=grid, devices=[source], time=time_steps)
#sim.run(live=True, save_memory_mode=True, accumulate_power=True)
#sim.plot_power(db_colorbar=True)