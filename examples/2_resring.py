from beamz import *
import numpy as np

# Parameters
WL = 1.55*µm
TIME = 120*WL/LIGHT_SPEED
X, Y = 20*µm, 19*µm
N_CORE, N_CLAD = 2.04, 1.444 # Si3N4, SiO2
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.999, points_per_wavelength=20)
RING_RADIUS, WG_WIDTH = 6*µm, 0.5*µm #0.565*µm

# Create the design
design = Design(width=X, height=Y, material=Material(N_CLAD**2))
design += Rectangle(position=(0,WL*2), width=X, height=WG_WIDTH, material=Material(N_CORE**2))
design += Ring(position=(X/2, WL*2+WG_WIDTH+RING_RADIUS+WG_WIDTH/2+0.2*WG_WIDTH), 
               inner_radius=RING_RADIUS-WG_WIDTH/2, outer_radius=RING_RADIUS+WG_WIDTH/2, 
               material=Material(N_CORE**2))
#design.show()

# Rasterize the design
grid = design.rasterize(resolution=DX)
#grid.show(field="permittivity")

# Define the signal & source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0,
                       ramp_duration=WL*6/LIGHT_SPEED, t_max=TIME/2.5)
source = ModeSource(
    grid=grid,
    center=(WL*2, WL*2+WG_WIDTH/2),
    width=WG_WIDTH * 3.5,  # Slightly wider than waveguide to capture mode tails
    wavelength=WL,
    pol="tm",
    signal=signal,
    direction="+x",
)

# Run the simulation
sim = Simulation(
    design=design,
    devices=[source],
    boundaries=[PML(edges='all', thickness=1.2*WL)],
    time=time_steps,
    resolution=DX
)
sim.run(animate_live="Ez", 
    animation_interval=15,
    axis_scale=[-0.9e-4, 0.9e-4],
    #axis_scale=None,
    cmap="twilight_zero", 
    clean_visualization=True,
    save_video="resring.mp4",
    video_fps=40)