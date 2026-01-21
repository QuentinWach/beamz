"""
3D MMI (Multimode Interference) 1×2 Waveguide Splitter with VDB Export
========================================================================

This example demonstrates a compact 3D MMI 1×2 waveguide splitter with VDB export
for Blender visualization. It shows wave propagation through an MMI structure that
splits light from one input waveguide into two output waveguides.

Physical Principle:
------------------
Multimode interference (MMI) devices use the self-imaging property of multimode
waveguides. When light enters a wide multimode section, it excites multiple modes
that interfere constructively and destructively as they propagate. At specific
lengths, the input field is replicated at multiple positions (self-images), enabling
power splitting.

For a 1×2 splitter, the MMI length is approximately 3L_π, where L_π is the beat
length between the two lowest-order modes. The outputs are positioned at ±W_MMI/6
from the center, where W_MMI is the MMI width.

Geometry:
---------
- Input waveguide: 0.565µm wide, single-mode
- MMI section: 3.5µm wide × 8µm long
- Output waveguides: Two 0.565µm wide waveguides offset by ±1.0µm
- Core thickness: 0.4µm (Si3N4)
- Total domain: 12µm × 6µm × 2.5µm

Layer Stack (Z-direction):
-------------------------
  2.5µm: Air (top)
  1.7µm: -------- (air/cladding interface)
  1.5µm: Top cladding (SiO2)
  1.3µm: -------- (cladding/core interface)
  1.1µm: Waveguide core center (Si3N4, 0.4µm thick)
  0.9µm: -------- (core/cladding interface)
  0.0µm: Bottom cladding (SiO2)

VDB Export for Blender:
-----------------------
This simulation exports electric field magnitude |E| with logarithmic scaling for
optimal Blender visualization. The VDB (OpenVDB) volume files can be imported into
Blender for high-quality 3D visualization and rendering.

Field Magnitude vs Components:
By default, the simulation exports |E| = √(Ex² + Ey² + Ez²) which shows wave structure
clearly in Blender's volume renderer. Individual components (Ex, Ey, Ez) have signed
values (±) that appear as uniform blobs in Blender because volume density expects
positive values. The magnitude field is always positive and reveals interference patterns.

Logarithmic Scaling:
Electromagnetic fields have huge dynamic ranges (10+ orders of magnitude). Log scaling
compresses this to make weak and strong features visible simultaneously in Blender.
This is automatically enabled for magnitude fields.

Installation:
Before running this example, install the VDB export dependencies:
  pip install beamz[vdb]

Note: On some systems (especially newer Python versions on macOS), pyopenvdb may need
to be installed via conda-forge:
  conda install -c conda-forge openvdb pyopenvdb

To import into Blender:
1. Open Blender
2. File > Import > OpenVDB (.vdb)
3. Navigate to the vdb_mmi_splitter/ directory
4. Select all frame files (Shift+A or Ctrl+A) for animation sequence
5. Click "Import OpenVDB"
6. In Shader Editor:
   - Add Volume Scatter node
   - Connect 'E_mag' attribute to Scatter density
   - Increase density scale (try 5.0-20.0)
   - Switch to Rendered viewport (Z key → Rendered)

For animation:
- The imported volumes will have frame numbers
- Set your timeline to match the number of frames
- The volumes will automatically animate

Legacy Compatibility:
To export individual components (old behavior), use:
  vdb_fields=['Ey']  # Or 'Ex', 'Ez', etc.
However, magnitude fields provide much better visualization in Blender.

Performance:
-----------
- Grid cells: ~280,000
- Time steps: ~600
- Expected runtime: 5-15 minutes on MacBook Air
- Memory usage: ~500 MB peak
- VDB frames: ~60
- Total file size: 50-70 MB

Parameters:
----------
- Wavelength: 1.55µm (telecom wavelength)
- Core material: Si3N4 (n=2.04)
- Cladding: SiO2 (n=1.444)
- Resolution: 8 points per wavelength (fast simulation)
- Simulation time: 50 wavelengths
"""

import numpy as np
from beamz import Design, Rectangle, Material, ModeSource, Simulation, ramped_cosine, PML
from beamz.const import µm, LIGHT_SPEED
from beamz.visual.helpers import calc_optimal_fdtd_params

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Domain dimensions (optimized for speed)
X, Y, Z = 12*µm, 6*µm, 2.5*µm

# Material refractive indices
N_CORE = 2.04   # Si3N4 (silicon nitride)
N_CLAD = 1.444  # SiO2 (silicon dioxide)
N_AIR = 1.0

# Wavelength and simulation time
WL = 1.55*µm
TIME = 50*WL/LIGHT_SPEED  # Shorter duration for quick simulation

# Waveguide geometry
WG_WIDTH = 0.565*µm      # Single-mode waveguide width
WG_THICKNESS = 0.4*µm    # Waveguide thickness
MMI_WIDTH = 3.5*µm       # MMI section width (supports multiple modes)
MMI_LENGTH = 8*µm        # MMI section length (~3L_π for 1×2 splitting)
OUTPUT_OFFSET = 1.0*µm   # Output waveguide offset from center

# Z-positions for layer stack
Z_CORE_BOTTOM = 0.7*µm   # Bottom of waveguide core
Z_CORE_TOP = 1.1*µm      # Top of waveguide core (0.7 + 0.4)
Z_CORE_CENTER = 0.9*µm   # Center of waveguide core
Z_CLAD_TOP = 1.5*µm      # Top of upper cladding

# Calculate optimal FDTD parameters
# Using 8 points per wavelength for fast execution
DX, DT = calc_optimal_fdtd_params(
    WL, N_CORE, dims=3,
    safety_factor=0.999,
    points_per_wavelength=8,
    width=X, height=Y, depth=Z
)

print(f"Grid spacing: DX = {DX/µm:.4f} µm")
print(f"Time step: DT = {DT*1e15:.4f} fs")
print(f"Estimated grid cells: {int(X/DX) * int(Y/DX) * int(Z/DX):,}")
print(f"Estimated time steps: {int(TIME/DT):,}")

# =============================================================================
# DESIGN CONSTRUCTION
# =============================================================================

# 1. Background (air)
design = Design(width=X, height=Y, depth=Z, material=Material(N_AIR**2))

# 2. Bottom cladding (SiO2 layer, from bottom to top of cladding)
design += Rectangle(
    position=(0, 0, 0),
    width=X, height=Y, depth=1.7*µm,
    material=Material(N_CLAD**2)
)

# 3. Top cladding (SiO2 layer above core, thin layer for index contrast)
design += Rectangle(
    position=(0, 0, 1.1*µm),
    width=X, height=Y, depth=0.4*µm,
    material=Material(N_CLAD**2)
)

# 4. Input waveguide core (Si3N4)
# Positioned at X=0 to 3µm, centered in Y
design += Rectangle(
    position=(0, Y/2 - WG_WIDTH/2, Z_CORE_BOTTOM),
    width=3*µm,
    height=WG_WIDTH,
    depth=WG_THICKNESS,
    material=Material(N_CORE**2)
)

# 5. MMI section core (Si3N4)
# Positioned at X=3 to 11µm, centered in Y
design += Rectangle(
    position=(3*µm, Y/2 - MMI_WIDTH/2, Z_CORE_BOTTOM),
    width=MMI_LENGTH,
    height=MMI_WIDTH,
    depth=WG_THICKNESS,
    material=Material(N_CORE**2)
)

# 6. Output waveguide 1 (top, +Y offset)
design += Rectangle(
    position=(11*µm, Y/2 + OUTPUT_OFFSET - WG_WIDTH/2, Z_CORE_BOTTOM),
    width=X - 11*µm,
    height=WG_WIDTH,
    depth=WG_THICKNESS,
    material=Material(N_CORE**2)
)

# 7. Output waveguide 2 (bottom, -Y offset)
design += Rectangle(
    position=(11*µm, Y/2 - OUTPUT_OFFSET - WG_WIDTH/2, Z_CORE_BOTTOM),
    width=X - 11*µm,
    height=WG_WIDTH,
    depth=WG_THICKNESS,
    material=Material(N_CORE**2)
)

# Visualize the design
print("\nDesign structure created. Displaying geometry...")
design.show()

# =============================================================================
# SOURCE CONFIGURATION
# =============================================================================

# Rasterize design to get grid
print("\nRasterizing design...")
grid = design.rasterize(resolution=DX)

# Time signal
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(
    time_steps,
    amplitude=1.0,
    frequency=LIGHT_SPEED/WL,
    ramp_duration=6*WL/LIGHT_SPEED,
    t_max=TIME/2
)

# ModeSource at input waveguide
# Positioned in the input waveguide, at the core center height
source = ModeSource(
    grid=grid,
    center=(1.5*µm, Y/2, Z_CORE_CENTER),  # In input waveguide, at core center
    width=2*µm,      # Larger than waveguide to capture mode
    height=1.0*µm,   # Larger than core to capture mode
    wavelength=WL,
    pol="te",        # TE mode (dominant Ey component)
    signal=signal,
    direction="+x"
)

print("\nMode source configured.")

# =============================================================================
# SIMULATION SETUP AND EXECUTION
# =============================================================================

# Create simulation with PML boundaries
sim = Simulation(
    design=design,
    devices=[source],
    boundaries=[PML(edges='all', thickness=0.75*WL)],
    time=time_steps,
    resolution=DX
)

print("\nStarting simulation with VDB export...")
print(f"This will take approximately 5-15 minutes on a MacBook Air.")
print(f"VDB files will be saved to: vdb_mmi_splitter/")

# Run simulation with VDB export
# New default exports |E| magnitude field with automatic log scaling for best Blender visualization
results = sim.run(
    animate_live="Ey",                 # Live animation of dominant TE field
    animation_interval=10,             # Update every 10 steps for performance
    save_vdb="vdb_mmi_splitter",       # Output directory for VDB files
    # vdb_fields=['|E|'] is the new default - magnitude field with auto log scaling
    # For old behavior, use: vdb_fields=['Ey'] (component field, no log scaling)
    vdb_interval=10,                   # Export every 10 steps (~60 frames)
    vdb_normalize=True,                # Normalize for Blender visualization
    clean_visualization=True
)

print("\nSimulation complete!")
print(f"VDB files saved to: vdb_mmi_splitter/")
print(f"Import into Blender: File > Import > OpenVDB (.vdb)")
print(f"Select all frame_*.vdb files for animation sequence")
