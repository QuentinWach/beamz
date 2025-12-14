import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from beamz import *
import numpy as np

# Small 3D domain for a fast smoke test
WL = 0.6*µm
X, Y, Z = 2.0*µm, 2.0*µm, 1.0*µm
N_MAX = 1.0

# Stable grid/time step for 3D
DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=3, safety_factor=0.45)
TIME = 10*WL/LIGHT_SPEED
time_steps = np.arange(0, TIME, DT)

# Homogeneous medium with thinner PML to leave a larger propagation region
design = Design(width=X, height=Y, depth=Z, material=Material(permittivity=1.0), pml_size=WL/4)

# 3D Gaussian source at center
src_pos = (X/2, Y/2, Z/2)
src_width = WL/8
signal = ramped_cosine(time_steps, amplitude=2.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=2*WL/LIGHT_SPEED, t_max=TIME/2)
design += GaussianSource(position=src_pos, width=src_width, signal=signal)

# Add a 3D plane monitor at mid-Z for interactive field visualization
monitor = Monitor(start=(0, 0, Z/3), end=(X, Y, Z/3), 
                  record_fields=True, accumulate_power=True, 
                  live_update=True, record_interval=2)
design += monitor

# Show the 3D design domain (uses Plotly if available, otherwise falls back to 2D)
design.show()
monitor.start_live_visualization(field_component='Ez')

# Run 3D FDTD (lightweight z-slice evolution)
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX, backend="numpy")
# Run with live visualization of Ez (center z-slice for 3D). Use tight color scale based on expected amplitude.
results = sim.run(live=True, axis_scale=[-0.1,0.1], save=False, save_memory_mode=True, accumulate_power=False)

# Report Ez magnitude statistics at the end
Ez = sim.backend.to_numpy(sim.Ez)
Ez_abs_max = float(np.max(np.abs(Ez)))
Ez_abs_mean = float(np.mean(np.abs(Ez)))
print(f"3D Gaussian test: Ez |max| = {Ez_abs_max:.3e}, |mean| = {Ez_abs_mean:.3e}, shape = {Ez.shape}")

# Plot a mid-z slice of Ez for visual confirmation
try:
    mid_z = Ez.shape[0]//2
    sim.plot_field(field="Ez", z_slice=mid_z)
except Exception as e:
    print(f"Plot skipped: {e}")


