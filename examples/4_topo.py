from beamz import *
import numpy as np

# Parameters
W = H = 15*µm
WG_W = 0.5*µm
N_CORE, N_CLAD = 2.25, 1.444
EPS_CORE, EPS_CLAD = N_CORE**2, N_CLAD**2
WL = 1.55*µm
PPP = 9 # hyper parameter that sets the "points (i.e. cells) per wavelength", affecting DX & DT
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), points_per_wavelength=PPP)
T_MAX = 30*WL/LIGHT_SPEED
STEPS, LR = 2, 0.1 # paramters for the optimization of the topology

# Create the design
design = Design(width=W, height=H, material=Material(permittivity=EPS_CLAD))
design += Rectangle(position=(0*µm,H/2-WG_W/2), width=3.5*µm, height=WG_W, material=Material(permittivity=EPS_CORE))
design += Rectangle(position=(W/2-WG_W/2,H), width=WG_W, height=-3.5*µm, material=Material(permittivity=EPS_CORE))
design += Rectangle(position=(W/2-4*µm,H/2-4*µm), width=8*µm, height=8*µm, material=Material(permittivity=EPS_CORE))
#design.show()

# Precompute the material grid
grid = design.rasterize(resolution=DX)
#grid.show(field="permittivity")

# Define the sources
time = np.arange(0, T_MAX, DT)
signal = ramped_cosine(t=time, amplitude=1, frequency=LIGHT_SPEED/WL, t_max=T_MAX, ramp_duration=6*WL/LIGHT_SPEED, phase=0)
input_source = ModeSource(grid=grid, center=(2*µm, H/2), width=WG_W, wavelength=WL,
                pol="tm", signal=signal, direction="+x")
back_source = ModeSource(grid=grid, center=(W/2, H-2*µm), width=WG_W, wavelength=WL,
                pol="tm", signal=signal, direction="-y")


# Define the objective function for the simulations: Transmission - Reflection + Mode Match
#obj_function = lambda: ...

# Initialize the optimizer which will handle all the auto-differentiation and updates of the topology
#opt = Optimizer("Adam", lr=LR)

# Fill in the topology optimization later ...
for step in range(STEPS):
    # forward simulation with input source and output monitor
    # store all forward fields and accumulate and save the objective func
    forward = Simulation(design=design, devices=[input_source],
        boundaries=[PML(edges='all', thickness=1.9*µm)], time=time, resolution=DX)
    forward_field_history, forward_obj_value = forward.run(
        animate_live="Ez", animation_interval=5, axis_scale=[-6e-5, 6e-5],
        clean_visualization=True)

    # backward simulation with output source and input monitor
    # and calculation of the accumulated overlap field while emptying the
    # forward field list
    backward = Simulation(design=design, devices=[back_source],
        boundaries=[PML(edges='all', thickness=1.9*µm)], time=time, resolution=DX)
    _, backward_obj_value, acc_overlap = forward.run(
        animate_live="Ez", animation_interval=5, axis_scale=[-6e-5, 6e-5],
        clean_visualization=True)

    # calculate total objective
    total_objective = forward_obj_value + backward_obj_value

    # update the design space using auto-diff
    # function = tanh(blur(density))
    # dfunction/ddensity = blur(density) x tanh(blur(density)) + dtanh/ddensity(blur(density)) ...
    # optimizer: SGD / Adagrad / Adam / Myon