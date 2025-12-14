from beamz import *
import numpy as np

WL = 0.6*µm # wavelength of the source
TIME = 40*WL/LIGHT_SPEED # total simulation duration
N_CLAD = 1; N_CORE = 2 # refractive indices of the core and cladding
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD)) # optimal grid size and time step

# Create the design
design = Design(8*µm, 8*µm, 8*µm, material=Material(N_CLAD**2), pml_size=WL*1.5)
design += Rectangle(width=4*µm, height=4*µm, depth=4*µm, material=Material(N_CORE**2))
design.show()