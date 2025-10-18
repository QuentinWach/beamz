from beamz import *
import numpy as np

# Parameters
X, Y = 20*µm, 10*µm # domain size
WL = 1.55*µm # wavelength
TIME = 40*WL/LIGHT_SPEED # total simulation duration
N_CORE = 2.04 # Si3N4
N_CLAD = 1.444 # SiO2
WG_W = 0.565*µm # width of the waveguide
H = 3.5*µm # height of the MMI
W = 9*µm # length of the MMI (in propagation direction)
OFFSET = 1.05*µm # offset of the output waveguides from center of the MMI
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.60) 

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
design += Rectangle(position=(0, Y/2-WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 + OFFSET - WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 - OFFSET - WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2-W/2, Y/2-H/2), width=W, height=H, material=Material(N_CORE**2))
design.show()

# TODO: Make this work:
# When no layers are specified, we merge all shapes of the same material that are touching and then seperate the merged polygons by layer,
# i.e. each polygon is a new layer. Note though that we still make polygons early in the list of structures of the Design lower layers.
design.export_gds("mmi.gds")