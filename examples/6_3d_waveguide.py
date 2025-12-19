import numpy as np
from beamz import Design, Rectangle, Material, ModeSource, Monitor, Simulation, ramped_cosine
from beamz.const import µm, LIGHT_SPEED
from beamz.visual.helpers import calc_optimal_fdtd_params

# Units and constants
WL = 1.55 * µm  # Wavelength
TIME = 100 * WL / LIGHT_SPEED  # Duration
N_AIR = 1.0
N_CLAD = 1.44  # SiO2
N_CORE = 3.48  # Silicon

# Calculate optimal grid parameters
# 3D simulations can be memory intensive, so we use a slightly lower resolution (8 points/WL)
DX, DT = calc_optimal_fdtd_params(WL, N_CORE, dims=3, safety_factor=0.95, points_per_wavelength=8,
                                 width=10*µm, height=4*µm, depth=2*µm)

# 1. Create the Design
# A 10µm long waveguide along X, 4µm wide, 2µm thick
design = Design(width=10*µm, height=4*µm, depth=2*µm, material=Material(N_AIR**2))
design += Rectangle(position=(0, 0, 0), width=10*µm, height=4*µm, depth=1*µm, material=Material(N_CLAD**2))
waveguide = Rectangle(
    position=(0, 1.75*µm, 1*µm), 
    width=10*µm, 
    height=0.5*µm, 
    depth=0.22*µm, 
    material=Material(N_CORE**2)
)
design += waveguide

# 3. Add a Mode Source
# Define the signal
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=WL*6/LIGHT_SPEED, t_max=TIME/2)

# Define the source
# We need to rasterize first to get the grid for ModeSource if we use it directly
grid = design.rasterize(resolution=DX)

source = ModeSource(
    grid=grid,
    center=(1.0*µm, 2.0*µm, 1.0*µm),
    width=3.0*µm,
    wavelength=WL,
    pol="te",
    signal=signal,
    direction="+x"
)

# 4. Add Monitors
# XY plane monitor in the middle of the waveguide thickness
monitor_xy = Monitor(
    start=(0, 0, 1.0*µm),
    size=(10*µm, 4*µm),
    plane_normal="z",
    name="xy_plane"
)
design += monitor_xy

# YZ cross-section monitor near the end
monitor_yz = Monitor(
    start=(4.0*µm, 0, 0),
    size=(4*µm, 2*µm),
    plane_normal="x",
    name="yz_cross"
)
design += monitor_yz

# Show the 3D design
design.show()

# 5. Run the Simulation
sim = Simulation(design=design, devices=[source], time=time_steps, resolution=DX)

# Run with live animation of the Hz field on the XY monitor
sim.run(animate_live="Ez", animation_interval=10, clean_visualization=True)

