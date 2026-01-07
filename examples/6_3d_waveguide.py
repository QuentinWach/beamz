import numpy as np
from beamz import Design, Rectangle, Material, ModeSource, Monitor, Simulation, ramped_cosine
from beamz.const import µm, LIGHT_SPEED
from beamz.visual.helpers import calc_optimal_fdtd_params
from beamz import *

# Units and constants
WL = 1.55 * µm  # Wavelength
TIME = 100 * WL / LIGHT_SPEED  # Duration
N_AIR = 1.0
N_CLAD = 1.44  # SiO2
N_CORE = 3.48  # Silicon

# Calculate optimal grid parameters
# 3D simulations can be memory intensive, so we use a slightly lower resolution (8 points/WL)
DX, DT = calc_optimal_fdtd_params(WL, N_CORE, dims=3, safety_factor=0.999,
    points_per_wavelength=10, width=6.5*µm, height=6.5*µm, depth=4*µm)

# 1. Create the Design
# A 10µm long waveguide along X, 4µm wide, 2µm thick
design = Design(width=6.5*µm, height=6.5*µm, depth=4*µm, material=Material(N_AIR**2))
design += Rectangle(position=(0, 0, 0), width=6.5*µm, height=6.5*µm, depth=2*µm, material=Material(N_CLAD**2))
waveguide = Rectangle(
    position=(0, 3*µm, 2*µm), 
    width=6.5*µm, 
    height=0.5*µm, 
    depth=0.22*µm, 
    material=Material(N_CORE**2)
)
design += waveguide
design.show()

# 3. Add a Mode Source
# Define the signal
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=WL*6/LIGHT_SPEED, t_max=TIME/2)

# Define the source
# We need to rasterize first to get the grid for ModeSource if we use it directly
grid = design.rasterize(resolution=DX)

# Source position: X should be inside waveguide, Y and Z at waveguide center
# Waveguide: Y=[1.75, 2.25] (center=2.0µm), Z=[1.0, 1.22] (center=1.11µm)
# Source width should be comparable to waveguide dimensions to capture the mode
# Using 0.8µm (slightly larger than waveguide height 0.5µm) to capture mode field
source = ModeSource(
    grid=grid,
    center=(1*WL, 3.25*µm, 2.11*µm),  # Z at waveguide center
    width=1.5*µm,  # Closer to waveguide height (0.5µm) to better capture mode
    height=0.8*µm,
    wavelength=WL,
    pol="te",
    signal=signal,
    direction="+x"
)

# Initialize the source to compute mode profiles
source.initialize(grid.permittivity, DX)

# Plot and save all mode field components (Ex, Ey, Ez, Hx, Hy, Hz)
print("Plotting all mode field components...")
source._plot_mode_profile_3d()
print("Mode profile figure saved to mode_profile.png")

# 4. Add Monitors
# XY plane monitor in the middle of the waveguide thickness
monitor_xy = Monitor(
    start=(0, 0, 2.11*µm),
    size=(6.5*µm, 6.5*µm),
    plane_normal="z",
    name="xy_plane"
)
#design += monitor_xy
#design.show()

# 5. Run the Simulation
sim = Simulation(design=design, devices=[source, monitor_xy], 
boundaries=[PML(edges='all', thickness=0.75*WL)], time=time_steps, resolution=DX)

# Run with live animation of the Ez field on the XY monitor
results = sim.run(animate_live="Hy", animation_interval=5, clean_visualization=True, save_video="3d_waveguide.mp4")
