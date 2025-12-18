from beamz import *
import numpy as np

WL = 0.6*µm # wavelength of the source
TIME = 25*WL/LIGHT_SPEED # total simulation duration
N_CLAD = 1; N_CORE = 2 # refractive indices of the core and cladding
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=3, safety_factor=0.45, points_per_wavelength=8)

# Create the 3D design
design = Design(8*µm, 8*µm, 4*µm, material=Material(N_CLAD**2))
design += Rectangle(width=4*µm, height=4*µm, depth=2*µm, material=Material(N_CORE**2))
design.show()

# Explicitly rasterize as 3D to ensure proper 3D meshing
design.rasterize(DX, grid_type="3d")

time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
source = GaussianSource(position=(4*µm, 4*µm, 2*µm), width=WL/6, signal=signal)


# Add a monitor for the middle x-y plane
monitor = Monitor(
    start=(0, 0, 2*µm), 
    end=(8*µm, 8*µm, 2*µm), 
    record_fields=True, 
    live_update=True, 
    record_interval=2
)

# Add PML boundaries to simulation (not design)
sim = Simulation(design=design, devices=[source, monitor], boundaries=[PML(edges='all', thickness=2*WL)], time=time_steps, resolution=DX)

# The simulation will now automatically detect the monitor and use it for live animation
sim.run(animate_live="Ez", animation_interval=1, clean_visualization=True)

