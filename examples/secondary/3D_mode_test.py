import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from beamz import *
import numpy as np

# Small 3D waveguide box with a fundamental mode source
WL = 1.55*µm
X, Y, Z = 6*µm, 3*µm, 2*µm
DX, DT = calc_optimal_fdtd_params(WL, 2.1, dims=3, safety_factor=0.45)
TIME = 20*WL/LIGHT_SPEED
time_steps = np.arange(0, TIME, DT)

# Materials
n_clad = 1.444; n_core = 2.04

design = Design(width=X, height=Y, depth=Z, material=Material(n_clad**2), pml_size=WL/2)

# Core strip in the center along x, finite thickness in z
core_w = 0.7*µm
core_t = 0.4*µm
design += Rectangle(position=(0, Y/2-core_w/2, Z/2-core_t/2), width=X*0.6, height=core_w, depth=core_t, material=Material(n_core**2))

# Monitor at half depth
monitor = Monitor(start=(0, 0, Z/2), end=(X, Y, Z/2), record_fields=True, accumulate_power=True, live_update=True, record_interval=2)
design += monitor

# Mode source at x ~ 1 µm, oriented to +x, with 3D cross-section
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
src = ModeSource(design=design, position=(1.0*µm, Y/2, Z/2), width=1.2*µm, height=0.8*µm, direction="+x", signal=signal, grid_resolution=1200, num_modes=1)
design += src

design.show()
monitor.start_live_visualization('Ez')

sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX, backend="numpy")
sim.run(live=True, axis_scale=[-0.05, 0.05], save=False, save_memory_mode=True, accumulate_power=True)
print("Power records:", len(monitor.power_history))

