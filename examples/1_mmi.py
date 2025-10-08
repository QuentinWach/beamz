from beamz import *
import numpy as np

# Parameters
X, Y = 20*µm, 10*µm # domain width, height
WL = 1.55*µm # wavelength
TIME = 40*WL/LIGHT_SPEED # total simulation duration
N_CORE, N_CLAD = 2.04, 1.444 # Si3N4, SiO2
WG_W = 0.565*µm # width of the waveguide
H, W, OFFSET = 3.5*µm, 9*µm, 1.05*µm # height, length, offset of the MMI
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.999, points_per_wavelength=10) 

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
design += Rectangle(position=(0, Y/2-WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 + OFFSET - WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 - OFFSET - WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2-W/2, Y/2-H/2), width=W, height=H, material=Material(N_CORE**2))

# Define the source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/2)
source = ModeSource(design=design, start=(2*µm, Y/2-1.2*µm), end=(2*µm, Y/2+1.2*µm), wavelength=WL, signal=signal)
design += source
design.show()

# Run the simulation and show results
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, save_memory_mode=True, accumulate_power=True) #axis_scale=[-1000/(µm), 1000/(µm)]
sim.plot_power(db_colorbar=True)