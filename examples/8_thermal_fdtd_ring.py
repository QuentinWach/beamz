import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle as PlotRectangle

try:
    import tidy3d as _tidy3d  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "This example requires tidy3d for mode solving. Install it with: pip install tidy3d"
    ) from exc

from beamz import (
    Design,
    Material,
    Rectangle,
    Ring,
    ModeSource,
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
TIME = 80 * WL / LIGHT_SPEED

# Rasterize once so thermal + EM share the same grid
grid = design.rasterize(resolution=DX)

# --- 3. Baseline simulation (isothermal) ---
def run_transmission_sweep(eps_r_override, wavelengths):
    if eps_r_override is not None:
        grid.permittivity = np.array(eps_r_override, copy=True)

    trans = []
    for wl_val in wavelengths:
        time = np.arange(0, 80 * wl_val / LIGHT_SPEED, DT)
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=LIGHT_SPEED / wl_val,
            phase=0.0,
            ramp_duration=6 * wl_val / LIGHT_SPEED,
            t_max=time[-1] / 2.5,
        )
        source = ModeSource(
            grid=grid,
            center=(2.0 * WL, bus_y + WG_WIDTH / 2),
            width=WG_WIDTH * 3.0,
            wavelength=wl_val,
            pol="tm",
            signal=signal,
            direction="+x",
        )
        source.initialize(grid.permittivity, DX)

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
        sim.run(save_fields=[], field_subsample=10)

        in_E = np.sum(mon_in.power_history) * DT
        out_E = np.sum(mon_out.power_history) * DT
        trans.append((np.abs(out_E) / np.abs(in_E)) if np.abs(in_E) > 0 else 0.0)

    return np.array(trans)

eps_r_base = np.array(grid.permittivity, copy=True)
wavelengths = np.linspace(1.52 * µm, 1.58 * µm, 7)
print("Running baseline (isothermal) sweep...")
trans_base = run_transmission_sweep(eps_r_base, wavelengths)

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

target_delta_t = 100.0
heater_power_guess = 3e16
_, temperature_guess = apply_static_thermal(
    design,
    resolution=DX,
    params=params,
    heater_mask=heater_mask,
    heater_power=heater_power_guess,
    fixed_temp_mask=fixed_temp_mask,
    fixed_temp_value=300.0,
)
max_dT_guess = float(np.max(temperature_guess) - 300.0)
scale = target_delta_t / max_dT_guess if max_dT_guess > 1e-9 else 1.0
heater_power = heater_power_guess * scale

eps_r_thermal, temperature = apply_static_thermal(
    design,
    resolution=DX,
    params=params,
    heater_mask=heater_mask,
    heater_power=heater_power,
    fixed_temp_mask=fixed_temp_mask,
    fixed_temp_value=300.0,
)

# --- 5. Update EM permittivity (thermal-shifted) ---
print("Running thermal-shifted sweep...")
trans_hot = run_transmission_sweep(eps_r_thermal, wavelengths)

# --- 6. Results + Visualization ---
max_dT = np.max(temperature) - 300.0
print("\n--- Summary ---")
idx_center = len(wavelengths) // 2
print(f"Baseline transmission: {trans_base[idx_center]:.4f}")
print(f"Thermal transmission:  {trans_hot[idx_center]:.4f}")
print(f"Delta transmission:    {trans_hot[idx_center] - trans_base[idx_center]:+.4f}")
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
ax1.plot(wavelengths / µm, trans_base, "-o", label="Baseline", color="#4c72b0")
ax1.plot(wavelengths / µm, trans_hot, "-o", label="Thermal", color="#dd8452")
ax1.set_xlabel("Wavelength (µm)")
ax1.set_ylabel("Transmission (arb.)")
ax1.set_title("Thermal Detuning Impact (Spectrum)")
ax1.grid(True, alpha=0.3)
ax1.legend()

plt.tight_layout()
plt.show()
