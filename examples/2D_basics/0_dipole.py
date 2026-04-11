import sys
from pathlib import Path

from beamz import *
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _mpl_helpers import run_with_snapshots

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

# Add source and PML boundaries to the simulation.
sim = Simulation(design=design, sources=[source], boundaries=[PML(edges='all', thickness=2*WL)], time=time_steps, resolution=DX)
run_with_snapshots(sim, snapshot_field="Ez", snapshot_interval=1, clean_visualization=True)
