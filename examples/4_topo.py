import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from beamz import *
from beamz.optimization.topology import apply_density_update

# Define waveguide geometry, refractive indices, and optimization schedule
W = H = 15*µm; WG = 0.5*µm; WL = 1.55*µm
N_CORE, N_CLAD = 2.25, 1.444; EPS_CORE, EPS_CLAD = N_CORE**2, N_CLAD**2
STEPS, LR = 20, 0.2
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.95, points_per_wavelength=10)
TIME = 15*WL/LIGHT_SPEED; t = np.arange(0, TIME, DT)

# Launch a ramped cosine source to excite the fundamental mode
signal = ramped_cosine(t=t, amplitude=1.0, frequency=LIGHT_SPEED/WL, t_max=TIME, ramp_duration=5*WL/LIGHT_SPEED, phase=0)
reg_pos, reg_size = (W/2-4*µm, H/2-4*µm), 8*µm
region_mat = CustomMaterial()

# Seed design with waveguide rails and a tunable design region
design = Design(width=W, height=H, pml_size=2*µm)
design += Rectangle(position=(0, H/2-WG/2), width=3.5*µm, height=WG, material=Material(permittivity=EPS_CORE))
design += Rectangle(position=(W/2-WG/2, H), width=WG, height=-3.5*µm, material=Material(permittivity=EPS_CORE))
design += Rectangle(position=reg_pos, width=reg_size, height=reg_size, material=region_mat)

# Build simulation grid and identify the optimization subregion
grid = RegularGrid(design=design, resolution=DX)
base = grid.permittivity.copy(); dx, dy = grid.dx, grid.dy
x0, x1 = int(np.floor(reg_pos[0]/dx)), int(np.ceil((reg_pos[0]+reg_size)/dx))
y0, y1 = int(np.floor(reg_pos[1]/dy)), int(np.ceil((reg_pos[1]+reg_size)/dy))
slc = (slice(y0, y1), slice(x0, x1))
mask = np.zeros_like(base, bool); mask[slc] = True
rng = np.random.default_rng(0)
density = np.zeros_like(base); density[slc] = rng.random(mask.sum()).reshape(y1-y0, x1-x0)

# Helper operators: smoothing, projection, and objective measurement
blur = lambda f, r=2: f if r <= 0 else sliding_window_view(np.pad(f, r, mode="edge"), (2*r+1, 2*r+1)).mean((-2, -1))
project = lambda f, b=6.0, e=0.5: (np.tanh(b*(f-e)) + np.tanh(b*e)) / (np.tanh(b*e) + np.tanh(b*(1-e)))
objective = lambda m: -float(np.sum(np.abs(m.power_history))) if m.power_history else 0.0
make_source = lambda direction: ModeSource(design=design, position=(2.5*µm, H/2), width=WG*4, wavelength=WL, signal=signal, 
    direction=direction, mode_solver="analytical", num_modes=1)
make_monitor = lambda: Monitor(design=design, start=(W/2-WG*2, H-2.5*µm), end=(W/2+WG*2, H-2.5*µm),
    objective_function=objective, name="out")

# Gradient-based topology optimization loop
for step in range(1, STEPS+1):

    # Smooth and threshold the design variables inside the mask
    filtered = np.where(mask, project(blur(density)), density) 

    # Map density to permittivity
    eps = base.copy(); eps[slc] = region_mat.permittivity_grid = EPS_CLAD + filtered[slc]*(EPS_CORE-EPS_CLAD)
    np.copyto(grid.permittivity, eps)

    # Forward simulation
    fres = FDTD(design=grid, devices=[make_source("+x"), make_monitor()], time=t).run(live=False, save_memory_mode=True, 
        accumulate_power=True, save_fields=["Ez"], fields_to_cache=["Ez"])
    forward_fields = list(fres.get("Ez", []))
    
    # Adjoint simulation
    adj = FDTD(design=grid, devices=[make_source("-y")], time=t); adj.initialize_simulation(save=False, live=False, 
        accumulate_power=False, save_memory_mode=True, fields_to_cache=None)
    for _ in range(adj.num_steps):
        if not forward_fields or not adj.step(): break
        grad += np.real(adj.backend.to_numpy(adj.Ez) * np.conj(forward_fields.pop()))  # Accumulate adjoint gradient
    adj.finalize_simulation()
    
    # Apply normalized gradient descent step
    norm = np.max(np.abs(grad[mask])) or 1.0
    density, *_ = apply_density_update(density, grad/norm, mask, learning_rate=LR, blur_radius=1)  
    density[~mask] = 0.0  # Keep density fixed outside optimization region
    print(f"step {step}: transmission {-float(next(iter(fres.get('objectives', {'out': 0}).values()))):.3e}")

# Re-simulate the optimized geometry for final reporting
filtered = np.where(mask, project(blur(density)), density)
eps = base.copy(); eps[slc] = region_mat.permittivity_grid = EPS_CLAD + filtered[slc]*(EPS_CORE-EPS_CLAD)
np.copyto(grid.permittivity, eps)
final = FDTD(design=grid, devices=[make_source("+x"), make_monitor()], time=t).run(live=False, save_memory_mode=True, 
    accumulate_power=True, save_fields=["Ez"], fields_to_cache=None)
print("final transmission", -float(next(iter(final.get("objectives", {"out": 0}).values()))))