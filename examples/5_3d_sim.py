from beamz import *
import numpy as np

WL = 0.6*µm # wavelength of the source
TIME = 50*WL/LIGHT_SPEED # total simulation duration
N_CLAD = 1; N_CORE = 2 # refractive indices of the core and cladding
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=3, safety_factor=0.45, points_per_wavelength=8)

# Create the 3D design
design = Design(4*µm, 4*µm, 2*µm, material=Material(N_CLAD**2))
design += Rectangle(width=2*µm, height=2*µm, depth=1*µm, material=Material(N_CORE**2))

# Add a monitor for the middle x-y plane (z = 1μm)
monitor = Monitor(
    start=(0, 0, 1*µm), 
    end=(4*µm, 4*µm, 1*µm), 
    record_fields=True, 
    live_update=False, 
    record_interval=1
)
design += monitor

# Explicitly rasterize as 3D
design.rasterize(DX, grid_type="3d")

time_steps = np.arange(0, TIME, DT)
# Increased amplitude significantly to ensure visibility in 3D
signal = ramped_cosine(time_steps, amplitude=1e9, frequency=LIGHT_SPEED/WL, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
source = GaussianSource(position=(2*µm, 2*µm, 1*µm), width=WL/6, signal=signal)

# Add PML boundaries to simulation (not design)
sim = Simulation(design=design, devices=[source, monitor], boundaries=[PML(edges='all', thickness=1.0*WL)], time=time_steps, resolution=DX)

# The simulation will now automatically detect the monitor and use it for live animation
sim.run(animate_live="Ez", animation_interval=10, clean_visualization=True)