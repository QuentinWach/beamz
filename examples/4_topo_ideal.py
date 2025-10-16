import numpy as np
from beamz import *
from beamz.optimization import topology as topo

# General parameters for the simulation
W, H = 15*µm, 15*µm
WG_W = 0.5*µm
N_CORE, N_CLAD = 2.25, 1.444
WL = 1.55*µm
OPT_STEPS, LR = 50, 0.5
TIME = 15*WL/LIGHT_SPEED
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.95, points_per_wavelength=10)

# Design with a custom inhomogeneous material which we will update during optimization
design = Design(width=W, height=H, pml_size=2*µm)
design += Rectangle(position=(0*µm, H/2-WG_W/2), width=3.5*µm, height=WG_W, material=Material(permittivity=N_CORE**2))
design += Rectangle(position=(W/2-WG_W/2, H), width=WG_W, height=-3.5*µm, material=Material(permittivity=N_CORE**2))
design_material = CustomMaterial(permittivity_func= lambda x, y, z=None: np.random.uniform(N_CLAD**2, N_CORE**2))
design += Rectangle(position=(W/2-4*µm, H/2-4*µm), width=8*µm, height=8*µm, material=design_material)

# Define the signal
t = np.linspace(0, TIME, int(TIME/DT))
signal = ramped_cosine(t=t, amplitude=1.0, frequency=LIGHT_SPEED/WL, t_max=TIME, ramp_duration=WL*5/LIGHT_SPEED, phase=0)

# Rasterize the initial design
grid = RegularGrid(design=design, resolution=DX)

# Mask and density initialization for topology updates
design_region = design.structures[-1]
mask = design_region.make_mask()
density = topo.initialize_density_from_region(design_region, DX)

# Define the objective function
def objective_function(monitor: Monitor, normalize: bool = True) -> float:
    total_power = 0.0
    if monitor.power_history: total_power = sum(monitor.power_history)
    else:
        if monitor.accumulate_power and monitor.power_accumulated is not None:
            total_power = float(np.sum(monitor.power_accumulated))
    if normalize and monitor.power_history: total_power /= len(monitor.power_history)
    return -float(total_power)

# Define the optimizer
optimizer = Optimizer(method="adam", learning_rate=LR)
objective_history = []
for opt_step in range(OPT_STEPS):

    # Add filters and contraints to the design region permittivity density
    blurred = topo.blur_density(density, radius=WL / DX)
    projected = topo.project_density(blurred, beta=2.0, eta=0.5)
    density = np.where(mask, projected, density)
    topo.update_design_region_material(design_region, density)

    # Run the forward FDTD simulation
    forward = FDTD(design=grid, devices=[
        ModeSource(grid=grid, center=(2.5*µm, H/2), width=WG_W*4, wavelength=WL, pol="tm", signal=signal, direction="+x", mode=0),
        Monitor(design=design, start=(W/2-WG_W*2, H-2.5*µm), end=(W/2+WG_W*2, H-2.5*µm), objective_function=objective_function)
    ], time=t) # TODO: Integrate the objective function into the FDTD simulation
    forward_fields, objective_value = forward.run(live=True, axis_scale=[-1, 1], save_memory_mode=True) # TODO: only save the Ez field!!!
    objective_history.append(objective_value)

    # Run the adjoint FDTD simulation step-by-step and accumulate the overlap field
    adjoint = FDTD(design=grid, devices=[ModeSource(grid=grid, center=(W/2, H-2.5*µm),
        width=WG_W*4, wavelength=WL, pol="tm", signal=signal, direction="-y", mode=0)], time=t)
    overlap_gradient = np.zeros_like(forward_fields["Ez"]) # TODO: Initialize with the correct shape!!!
    for step in t:
        adjoint_field = adjoint.step() # Simulate one step of the adjoint FDTD simulation
        overlap_gradient += compute_overlap_gradient(forward_fields, adjoint_field) / len(t) # Accumulate overlap gradient
        forward_fields.pop() # Delete the forward field that was just used to free up memory

    # Update the grid permittivity with the overlap gradient & clip values to the permittivity range
    update = optimizer.step(overlap_gradient)
    density = np.clip(density + update, N_CLAD**2, N_CORE**2)

# Run final simulation

# Show the final design

# Convert and save the design as GDS