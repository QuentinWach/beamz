from beamz import *
import numpy as np
from beamz import dxdt

# Define 10 center wavelengeths we want to simulate
WL = 1.55 * µm
TIME = 60 * WL / LIGHT_SPEED
DX, DT = dxdt(WL, safety_factor=0.999, points_per_wavelength=12)

# Create the design by importing a GDSFactory device
design = design.io.gdsf.load("examples/data/gdsfactory_cell.gds")
design.show()

# Rasterize the design
grid = design.rasterize(resolution=DX)
grid.show(field="permittivity")

# Define 10 signal pulses with different center wavelengths and time-stepping
time = np.arange(0, TIME, DT)
signal = ramped_cosine(time, amplitude=1.0, frequency=LIGHT_SPEED / WL, 
    ramp_duration=WL * 20 / LIGHT_SPEED, t_max=TIME / 2)

# PLace the mode source at the input port
source = ModeSource(grid=grid, center=(design.width/2, design.height/2),
    width=design.width/2, wavelength=WL, pol="tm", signal=signal, direction="+x")

# Setup the simulation
sim = Simulation(
    design=design, # use rastered grid instead of rasterizing again
    devices=[source], 
    boundaries=[PML(edges='all', thickness=1.2*WL)],
    time=time,
    resolution=DX
)

# Run the simulation
sim.run(save_fields=['Ez', 'Hy', 'Hx'], record_interval=1, record_fields=True)

# Calculate the S-matrix using FFT (freq. auto-calc. from time-stepping)
S_matrix = sim.get_S_matrix(input_ports=[], output_ports=[])
S_matrix.show()