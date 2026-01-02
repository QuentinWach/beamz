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
DX, DT = calc_optimal_fdtd_params(WL, N_CORE, dims=3, safety_factor=0.999,
    points_per_wavelength=14, width=4*µm, height=4*µm, depth=2*µm)

# 1. Create the Design
# A 10µm long waveguide along X, 4µm wide, 2µm thick
design = Design(width=4*µm, height=4*µm, depth=2*µm, material=Material(N_AIR**2))
design += Rectangle(position=(0, 0, 0), width=4*µm, height=4*µm, depth=1*µm, material=Material(N_CLAD**2))
waveguide = Rectangle(
    position=(0, 1.75*µm, 1*µm), 
    width=4*µm, 
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
    center=(1.0*µm, 2.0*µm, 1.11*µm),  # Z at waveguide center
    width=0.8*µm,  # Closer to waveguide height (0.5µm) to better capture mode
    wavelength=WL,
    pol="te",
    signal=signal,
    direction="+x"
)
#source.show(field="Ez")
#source.show(field="Hy")

# 4. Add Monitors
# XY plane monitor in the middle of the waveguide thickness
monitor_xy = Monitor(
    start=(0, 0, 1.11*µm),
    size=(4*µm, 4*µm),
    plane_normal="z",
    name="xy_plane"
)
design += monitor_xy
design.show()

# 5. Run the Simulation
sim = Simulation(design=design, devices=[source, monitor_xy], time=time_steps, resolution=DX)

# Run with live animation of the Ez field on the XY monitor
sim.run(animate_live="Ez", animation_interval=5, clean_visualization=True)

