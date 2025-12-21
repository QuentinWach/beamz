from beamz import *
import numpy as np

WL = 0.6*µm # wavelength of the source
TIME = 50*WL/LIGHT_SPEED # total simulation duration
N_CLAD = 1; N_CORE = 2.5 # refractive index reduced for faster demo (was 5)
# Added design dimensions to params call for grid size warning
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=3, safety_factor=0.95, points_per_wavelength=8,
                                 width=4*µm, height=4*µm, depth=2*µm)

# Create the 3D design
design = Design(4*µm, 4*µm, 4*µm, material=Material(N_CLAD**2))
# Ensure the rectangle is smaller than the design to see interaction
design += Rectangle(position=(0*µm, 0*µm, 0.5*µm), width=2*µm, height=2*µm, depth=1*µm, material=Material(N_CORE**2))

# Add a monitor for the middle x-y plane (z = 1μm)
monitor = Monitor(
    start=(0, 0, 1*µm), 
    end=(4*µm, 4*µm, 1*µm), 
    record_fields=True, 
    live_update=False, 
    record_interval=1
)
design += monitor
design.show()

time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
source = GaussianSource(position=(2.5*µm, 3*µm, 1.2*µm), width=WL/6, signal=signal)

# Add PML boundaries to simulation (not design)
sim = Simulation(design=design, devices=[source, monitor], boundaries=[PML(edges='all', thickness=1.0*WL)], time=time_steps, resolution=DX)

# The simulation will now automatically detect the monitor and use it for live animation
sim.run(animate_live="Hy", animation_interval=2, clean_visualization=False)