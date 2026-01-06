from beamz import *
import numpy as np

# Parameters
X, Y = 20*µm, 10*µm # domain width, height
WL = 1.55*µm # wavelength
TIME = 40*WL/LIGHT_SPEED # total simulation duration
N_CORE, N_CLAD = 2.04, 1.444 # Si3N4, SiO2
WG_W = 0.565*µm # width of the waveguide
H, W, OFFSET = 3.5*µm, 9*µm, 1.05*µm # height, length, offset of the MMI
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.999, points_per_wavelength=20) 

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, material=Material(N_CLAD**2))
design += Rectangle(position=(0, Y/2-WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 + OFFSET - WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 - OFFSET - WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2-W/2, Y/2-H/2), width=W, height=H, material=Material(N_CORE**2))
#design.show()

# Rasterize the design
grid = design.rasterize(resolution=DX)
#grid.show(field="permittivity")

# Define the source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=WL*6/LIGHT_SPEED, t_max=TIME/2)
# Prefer TE polarization and restrict transverse width to single-mode core to avoid exciting higher-order lobes
source = ModeSource(grid=grid, center=(3*µm, Y/2), width=WG_W * 3.5, wavelength=WL, pol="tm", signal=signal, direction="+x")

# Run the simulation and show results
sim = Simulation(design=design, devices=[source], boundaries=[PML(edges='all', thickness=1.2*WL)], time=time_steps, resolution=DX)
sim.run(animate_live="Ez",
    animation_interval=12,
    axis_scale=[-7e-5, 7e-5],
    clean_visualization=True,
    line_color="gray")