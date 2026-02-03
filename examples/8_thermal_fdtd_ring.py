import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle as PlotRectangle

from beamz import (
    Design,
    Material,
    Rectangle,
    Ring,
    GaussianSource,
    Monitor,
    Simulation,
    PML,
    ThermalParams,
    apply_static_thermal,
    calc_optimal_fdtd_params,
    ramped_cosine,
    LIGHT_SPEED,
    µm,
)

# --- 1. Geometry and Materials ---
WL = 1.55 * µm
W, H = 24 * µm, 20 * µm

N_CORE = 3.48  # Si
N_CLAD = 1.44  # SiO2

# Thermal properties (approximate, consistent with examples/7_thermal_static.py)
oxide = Material(
    permittivity=N_CLAD**2,
    k=1.4,
    rho=2200.0,
    cp=703.0,
    dn_dT=1.0e-5,
    T0=300.0,
)

silicon = Material(
    permittivity=N_CORE**2,
    k=148.0,
    rho=2330.0,
    cp=700.0,
    dn_dT=1.86e-4,
    T0=300.0,
)

# Waveguide + ring parameters
WG_WIDTH = 0.5 * µm
RING_RADIUS = 6.0 * µm
GAP = 0.2 * WG_WIDTH

# Build design (2D, in-plane)
design = Design(width=W, height=H, material=oxide)

bus_y = 2.0 * WL
ring_cx = W / 2
ring_cy = bus_y + WG_WIDTH + RING_RADIUS + WG_WIDTH / 2 + GAP

# Bus waveguide
bus = Rectangle(position=(0, bus_y), width=W, height=WG_WIDTH, material=silicon)
# Ring
ring = Ring(
    position=(ring_cx, ring_cy),
    inner_radius=RING_RADIUS - WG_WIDTH / 2,
    outer_radius=RING_RADIUS + WG_WIDTH / 2,
    material=silicon,
)

design += bus
design += ring

# --- 2. FDTD configuration ---
DX, DT = calc_optimal_fdtd_params(
    WL, n_max=N_CORE, dims=2, points_per_wavelength=12
)
TIME = 120 * WL / LIGHT_SPEED

time = np.arange(0, TIME, DT)

# Rasterize once so thermal + EM share the same grid
grid = design.rasterize(resolution=DX)

# --- 3. Baseline simulation (isothermal) ---
signal = ramped_cosine(
    time,
    amplitude=1.0,
    frequency=LIGHT_SPEED / WL,
    phase=0.0,
    ramp_duration=6 * WL / LIGHT_SPEED,
    t_max=TIME / 2.5,
)
source = GaussianSource(
    position=(2.0 * WL, bus_y + WG_WIDTH / 2),
    width=WG_WIDTH * 1.2,
    signal=signal,
)

mon_in = Monitor(
    design=grid,
    start=(2.5 * WL, bus_y - WG_WIDTH),
    end=(2.5 * WL, bus_y + 2.0 * WG_WIDTH),
    accumulate_power=True,
)
mon_out = Monitor(
    design=grid,
    start=(W - 2.5 * WL, bus_y - WG_WIDTH),
    end=(W - 2.5 * WL, bus_y + 2.0 * WG_WIDTH),
    accumulate_power=True,
)

sim = Simulation(
    design=grid,
    devices=[source, mon_in, mon_out],
    boundaries=[PML(edges="all", thickness=1.2 * WL)],
    time=time,
    resolution=DX,
)

print("Running baseline (isothermal) simulation...")
sim.run(save_fields=[], field_subsample=10)

in_E_base = np.sum(mon_in.power_history) * DT
out_E_base = np.sum(mon_out.power_history) * DT
trans_base = (np.abs(out_E_base) / np.abs(in_E_base)) if np.abs(in_E_base) > 0 else 0.0

# --- 4. Static thermal solve ---
outer_r = RING_RADIUS + WG_WIDTH / 2
inner_r = RING_RADIUS - WG_WIDTH / 2


def heater_mask(x, y, z):
    dx = x - ring_cx
    dy = y - ring_cy
    r2 = dx * dx + dy * dy
    if r2 < inner_r * inner_r or r2 > outer_r * outer_r:
        return False
    return (x >= ring_cx) and (y >= ring_cy)


def fixed_temp_mask(x, y, z):
    edge = 0.5 * µm
    return (x <= edge) or (x >= W - edge) or (y <= edge) or (y >= H - edge)


params = ThermalParams(
    thermal_dt=1e-13,
    tau_avg=1e-13,
    steady_state=True,
    max_iters=6000,
    tol=1e-6,
)

eps_r_thermal, temperature = apply_static_thermal(
    design,
    resolution=DX,
    params=params,
    heater_mask=heater_mask,
    heater_power=3e16,
    fixed_temp_mask=fixed_temp_mask,
    fixed_temp_value=300.0,
)

# --- 5. Update EM permittivity (thermal-shifted) ---
grid.permittivity = np.array(eps_r_thermal, copy=True)

source_hot = GaussianSource(
    position=(2.0 * WL, bus_y + WG_WIDTH / 2),
    width=WG_WIDTH * 1.2,
    signal=signal,
)

mon_in_hot = Monitor(
    design=grid,
    start=(2.5 * WL, bus_y - WG_WIDTH),
    end=(2.5 * WL, bus_y + 2.0 * WG_WIDTH),
    accumulate_power=True,
)
mon_out_hot = Monitor(
    design=grid,
    start=(W - 2.5 * WL, bus_y - WG_WIDTH),
    end=(W - 2.5 * WL, bus_y + 2.0 * WG_WIDTH),
    accumulate_power=True,
)

sim_hot = Simulation(
    design=grid,
    devices=[source_hot, mon_in_hot, mon_out_hot],
    boundaries=[PML(edges="all", thickness=1.2 * WL)],
    time=time,
    resolution=DX,
)

print("Running thermal-shifted simulation...")
sim_hot.run(save_fields=[], field_subsample=10)

in_E_hot = np.sum(mon_in_hot.power_history) * DT
out_E_hot = np.sum(mon_out_hot.power_history) * DT
trans_hot = (np.abs(out_E_hot) / np.abs(in_E_hot)) if np.abs(in_E_hot) > 0 else 0.0

# --- 6. Results + Visualization ---
max_dT = np.max(temperature) - 300.0
print("\n--- Summary ---")
print(f"Baseline transmission: {trans_base:.4f}")
print(f"Thermal transmission:  {trans_hot:.4f}")
print(f"Delta transmission:    {trans_hot - trans_base:+.4f}")
print(f"Max temperature rise:  {max_dT:.2f} K")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Temperature plot
extent = (0, W * 1e6, 0, H * 1e6)
ax0 = axes[0]
im0 = ax0.imshow(temperature, origin="lower", extent=extent, cmap="inferno")
plt.colorbar(im0, ax=ax0, label="Temperature (K)")
ax0.set_title("Thermal Gradient (Static Solve)")
ax0.set_xlabel("X (µm)")
ax0.set_ylabel("Y (µm)")

# Overlay ring and bus outlines
ax0.add_patch(
    Circle((ring_cx * 1e6, ring_cy * 1e6), outer_r * 1e6, fill=False, color="white", lw=1.2)
)
ax0.add_patch(
    Circle((ring_cx * 1e6, ring_cy * 1e6), inner_r * 1e6, fill=False, color="white", lw=1.2)
)
ax0.add_patch(
    PlotRectangle((0, bus_y * 1e6), W * 1e6, WG_WIDTH * 1e6, fill=False, edgecolor="white", lw=1.2)
)

# Transmission comparison
ax1 = axes[1]
ax1.bar(["Baseline", "Thermal"], [trans_base, trans_hot], color=["#4c72b0", "#dd8452"])
ax1.set_ylabel("Transmission (arb.)")
ax1.set_title("Thermal Detuning Impact")
ax1.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
