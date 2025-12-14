from beamz import *
import numpy as np

WL = 0.6*µm # wavelength of the source
TIME = 25*WL/LIGHT_SPEED # total simulation duration
N_CLAD = 1; N_CORE = 2 # refractive indices of the core and cladding
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.999, points_per_wavelength=8)

# Create the design
design = Design(8*µm, 8*µm, material=Material(N_CLAD**2))
design += Rectangle(width=4*µm, height=4*µm, material=Material(N_CORE**2))

time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
source = GaussianSource(position=(4*µm, 5*µm), width=WL/6, signal=signal)

# Add PML boundaries to simulation (not design)
sim = Simulation(design=design, devices=[source], boundaries=[PML(edges='all', thickness=2*WL)], time=time_steps, resolution=DX)
sim.run(animate_live="Ez", animation_interval=1, clean_visualization=True)