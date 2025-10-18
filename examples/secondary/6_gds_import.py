from beamz import *
import numpy as np

# Define materials for different layers
LAYER_MATERIALS = {
    0: Material(permittivity=11.67),
    1: Material(permittivity=2.085)
}
LAYER_PROPERTIES = {
    1: {"depth": 1*µm, "z": 0.0},      # Silicon layer at bottom
    0: {"depth": 0.6*µm, "z": 1*µm}   # Oxide layer in middle
}

# Import a GDS file
gds_design = Design.import_gds("mmi.gds")

# Initialize design with appropriate size
max_width = max(max(v[0] for v in s.vertices) for l in gds_design.layers.values() for s in l)
max_height = max(max(v[1] for v in s.vertices) for l in gds_design.layers.values() for s in l)
total_depth = sum(prop["depth"] for prop in LAYER_PROPERTIES.values())
design = Design(width=max_width, height=max_height, depth=total_depth*1.2, material=Material(1))

# Add structures from each layer with proper materials and z-positions
for layer_num, structures in gds_design.layers.items():
    material = LAYER_MATERIALS.get(layer_num, Material(1))
    properties = LAYER_PROPERTIES.get(layer_num, {"depth": 0.5*µm, "z": 0.0})
    for structure in structures:
        design += Polygon(vertices=structure.vertices, material=material, depth=properties["depth"], z=properties["z"])

# This will show the 3D visualization
design.show()