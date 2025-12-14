from beamz import *
import numpy as np

# Parameters
W = H = 15*µm
WG_W = 0.5*µm
N_CORE, N_CLAD = 2.25, 1.444
EPS_CORE, EPS_CLAD = N_CORE**2, N_CLAD**2
WL = 1.55*µm
STEPS, LR = 2, 0.1
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE,N_CLAD), dims=2, safety_factor=0.999, points_per_wavelength=12)
TIME = 30*WL/LIGHT_SPEED
tsteps = np.arange(0, TIME, DT)

# Create the design
# Use very small amplitude to prevent divergence in topology optimization
design = Design(width=W, height=H, material=Material(permittivity=EPS_CLAD))
design += Rectangle(position=(0*µm,H/2-WG_W/2), width=3.5*µm, height=WG_W, material=Material(permittivity=EPS_CORE))
design += Rectangle(position=(W/2-WG_W/2,H), width=WG_W, height=-3.5*µm, material=Material(permittivity=EPS_CORE))
design += Rectangle(position=(W/2-4*µm,H/2-4*µm), width=8*µm, height=8*µm, material=Material(permittivity=EPS_CORE))
#design.show()

grid = design.rasterize(resolution=DX)
#grid.show(field="permittivity")

signal = ramped_cosine(t=tsteps, amplitude=1, frequency=LIGHT_SPEED/WL, t_max=TIME, ramp_duration=6*WL/LIGHT_SPEED, phase=0)
input_source = ModeSource(grid=grid, center=(2.5*µm, H/2), width=WG_W, wavelength=WL,
                pol="tm", signal=signal, direction="+x")
back_source = ModeSource(grid=grid, center=(W/2, H-2.5*µm), width=WG_W, wavelength=WL,
                pol="tm", signal=signal, direction="-y")

sim = Simulation(design=design, devices=[input_source, back_source],
    boundaries=[PML(edges='all', thickness=1.2*WL)], time=tsteps, resolution=DX)
sim.run(animate_live="Ez", animation_interval=5, 
    #axis_scale=[-6e-5, 6e-5],
    cmap="twilight_zero", clean_visualization=True)

# Fill in the topology optimization later ...