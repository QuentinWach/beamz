import numpy as np

import beamz as bz

um = bz.um

# Parameters
WL = 1.55 * um
TIME = 120 * WL / bz.LIGHT_SPEED
X, Y = 20 * um, 19 * um
N_CORE, N_CLAD = 2.04, 1.444  # Si3N4, SiO2
DX, DT = bz.calc_optimal_fdtd_params(
    WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.999, points_per_wavelength=20
)
RING_RADIUS, WG_WIDTH = 6 * um, 0.5 * um  # 0.565 um

# Create the design
design = bz.Design(width=X, height=Y, material=bz.Material(N_CLAD**2))
design += bz.Rectangle(
    position=(0, WL * 2), width=X, height=WG_WIDTH, material=bz.Material(N_CORE**2)
)
design += bz.Ring(
    position=(X / 2, WL * 2 + WG_WIDTH + RING_RADIUS + WG_WIDTH / 2 + 0.2 * WG_WIDTH),
    inner_radius=RING_RADIUS - WG_WIDTH / 2,
    outer_radius=RING_RADIUS + WG_WIDTH / 2,
    material=bz.Material(N_CORE**2),
)
design.show()

# Rasterize the design
grid = design.rasterize(resolution=DX)
grid.show(field="permittivity")

# Define the signal & source
time_steps = np.arange(0, TIME, DT)
signal = bz.ramped_cosine(
    time_steps,
    amplitude=1.0,
    frequency=bz.LIGHT_SPEED / WL,
    phase=0,
    ramp_duration=WL * 6 / bz.LIGHT_SPEED,
    t_max=TIME / 2.5,
)
source = bz.ModeSource(
    grid=grid,
    center=(WL * 2, WL * 2 + WG_WIDTH / 2),
    width=WG_WIDTH * 3.5,  # Slightly wider than waveguide to capture mode tails
    wavelength=WL,
    pol="tm",
    signal=signal,
    direction="+x",
)
source.show_signal(t=time_steps)

# Run the simulation
sim = bz.Simulation(
    design=design,
    sources=[source],
    boundaries=[bz.PML(edges="all", thickness=1.2 * WL)],
    time=time_steps,
    resolution=DX,
)
results = sim.save_video(
    "resring.mp4",
    field="Ez",
    animation_interval=15,
    video_fps=40,
    cmap="twilight_zero",
    save_fields=["Ez"],
    field_subsample=15,
)
