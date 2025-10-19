from beamz import *

WL = 0.6*µm # wavelength of the source
N_CLAD = 1; N_CORE = 2 # refractive indices of the core and cladding

# Create the design
design = Design(8*µm, 8*µm, material=Material(N_CLAD**2))
design += Rectangle(width=4*µm, height=4*µm, material=Material(N_CORE**2))
design.show()