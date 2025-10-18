import os, sys
# Ensure local package is used when running from the repo
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from beamz import *
import numpy as np

# Parameters
X, Y, Z = 20*µm, 10*µm, 5*µm # domain width, height, depth
WL = 1.55*µm # wavelength
TIME = 40*WL/LIGHT_SPEED # total simulation duration
N_CORE, N_CLAD = 2.04, 1.444 # Si3N4, SiO2
WG_W = 0.565*µm # width of the waveguide
H, W, OFFSET = 3.5*µm, 9*µm, 1.05*µm # height, length, offset of the MMI
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=3, safety_factor=0.40) 

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, depth=Z, material=Material(permittivity=1.0, permeability=1.0, conductivity=0.0), pml_size=WL)
design += Rectangle(position=(0, 0, 0), width=X, height=Y, depth=Z/2, material=Material(N_CLAD**2))
design += Rectangle(position=(0, Y/2-WG_W/2, Z/2), width=X/2, height=WG_W, depth=Z/8, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 + OFFSET - WG_W/2, Z/2), width=X/2, height=WG_W, depth=Z/8, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 - OFFSET - WG_W/2, Z/2), width=X/2, height=WG_W, depth=Z/8, material=Material(N_CORE**2))
design += Rectangle(position=(X/2-W/2, Y/2-H/2, Z/2), width=W, height=H, depth=Z/8, material=Material(N_CORE**2))

# Add monitors
monitor = Monitor(start=(0, 0, Z/2+Z/16), end=(X, Y, Z/2+Z/16), record_fields=True, accumulate_power=True, live_update=True, record_interval=2)
design += monitor

# Define the source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/2)
source = ModeSource(design=design, position=(2.5*µm, Y/2, Z/2), 
                    width=2*µm, height=2*µm, direction="+x", signal=signal)
design += source
design.show()

source.show()


# Kick off live visualization for the monitor (updates will occur during run)
monitor.start_live_visualization(field_component='Ez')

# Run the simulation
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)



