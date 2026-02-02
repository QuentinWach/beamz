"""
WDM Demultiplexer — Topology Optimization Example
===================================================
Optimizes a compact device that splits 1300 nm and 1500 nm light from a single
input waveguide into two separate output waveguides using the adjoint method
with streaming (memory-efficient) gradient computation.

Device layout (not to scale):

        ┌──────────────────────────────────┐
        │           PML border             │
        │  ┌────────────────────────────┐  │
        │  │                   ┌──── out1 (1300 nm)
        │  │   ┌────────────┐ │        │  │
  in ───┼──┼───┤ design     ├─┤        │  │
        │  │   │ region     │ │        │  │
        │  │   └────────────┘ │        │  │
        │  │                   └──── out2 (1500 nm)
        │  └────────────────────────────┘  │
        │           PML border             │
        └──────────────────────────────────┘
"""

import numpy as np
import matplotlib.pyplot as plt
from beamz import *
from beamz.optimization.topology import (
    TopologyManager,
    compute_overlap_gradient,
    create_optimization_mask,
    run_adjoint_and_compute_gradient,
)

# ── 1. Parameters ────────────────────────────────────────────────────────────
W = 10 * µm                # Domain width
H = 6 * µm                 # Domain height
WG_W = 0.5 * µm            # Waveguide width
WL_1 = 1.30 * µm           # Channel 1 wavelength
WL_2 = 1.50 * µm           # Channel 2 wavelength
N_CORE = 2.0               # Si3N4 refractive index
N_CLAD = 1.444             # SiO2 refractive index
PML_THICK = 1.0 * µm       # PML thickness

# Use the shorter wavelength for resolution (stricter Nyquist)
DX, DT = calc_optimal_fdtd_params(WL_1, N_CORE, points_per_wavelength=20)
STEPS = 50                  # Optimization steps (reduce for faster testing)
MAT_PENALTY = 0.3           # Target core material fraction
PENALTY_STRENGTH = 1.0      # Penalty gradient scaling
FIELD_SUB = 2               # Field subsampling factor

# Output waveguide vertical positions
OUT1_Y = H / 2 + 1.0 * µm  # Upper output (1300 nm)
OUT2_Y = H / 2 - 1.0 * µm  # Lower output (1500 nm)

# ── 2. Design ────────────────────────────────────────────────────────────────
clad_mat = Material(permittivity=N_CLAD**2)
core_mat = Material(permittivity=N_CORE**2)

design = Design(width=W, height=H, material=clad_mat)

# Input waveguide (left side, vertically centred)
design += Rectangle(
    position=(0, H / 2 - WG_W / 2),
    width=W / 2 - 1.5 * µm,
    height=WG_W,
    material=core_mat,
)

# Output waveguide 1 — upper right (1300 nm channel)
design += Rectangle(
    position=(W / 2 + 1.5 * µm, OUT1_Y - WG_W / 2),
    width=W / 2 - 1.5 * µm,
    height=WG_W,
    material=core_mat,
)

# Output waveguide 2 — lower right (1500 nm channel)
design += Rectangle(
    position=(W / 2 + 1.5 * µm, OUT2_Y - WG_W / 2),
    width=W / 2 - 1.5 * µm,
    height=WG_W,
    material=core_mat,
)

# Optimization (design) region
opt_region = Rectangle(
    position=(W / 2 - 1.5 * µm, H / 2 - 2.0 * µm),
    width=3.0 * µm,
    height=4.0 * µm,
    material=core_mat,
)
design += opt_region

# ── 3. Sources & monitors helpers ────────────────────────────────────────────
# Simulation time — use longest wavelength for period estimate
sim_duration = 15 * WL_2 / LIGHT_SPEED
time_arr = np.arange(0, sim_duration, DT)


def _make_signal(wl):
    """Create a ramped cosine signal for a given wavelength."""
    return ramped_cosine(
        time_arr, 1.0, LIGHT_SPEED / wl,
        ramp_duration=3.5 * wl / LIGHT_SPEED,
        t_max=time_arr[-1] / 2,
    )


# ── 4. Rasterize & create optimisation manager ──────────────────────────────
grid = design.rasterize(DX)
mask = create_optimization_mask(grid, opt_region)

opt = TopologyManager(
    design=design,
    region_mask=mask,
    resolution=DX,
    learning_rate=0.015,
    filter_radius=0.3 * µm,
    eps_min=N_CLAD**2,
    eps_max=N_CORE**2,
    beta_schedule=(1.0, 20.0),
    filter_type="conic",
)

base_eps = grid.permittivity.copy()

# History tracking
trans_history_1300 = []
trans_history_1500 = []

print(f"WDM Demux Optimization — {STEPS} steps, grid {grid.permittivity.shape}")
print(f"  Channel 1: {WL_1/µm:.2f} µm → upper output")
print(f"  Channel 2: {WL_2/µm:.2f} µm → lower output")

# ── 5. Optimization loop ────────────────────────────────────────────────────
for step in range(STEPS):
    # --- Update permittivity from density ---
    beta, phys_density = opt.update_design(step, STEPS)
    grid.permittivity[:] = base_eps
    grid.permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)

    # ── Channel 1: 1300 nm ──────────────────────────────────────────────────
    sig_1 = _make_signal(WL_1)

    # Forward source: +x from left input
    src_fwd_1 = ModeSource(
        grid, center=(1.0 * µm, H / 2), width=WG_W * 4,
        wavelength=WL_1, pol="tm", signal=sig_1, direction="+x",
    )

    # Monitors
    mon_in_1 = Monitor(
        design=grid,
        start=(1.5 * µm, H / 2 - WG_W * 2),
        end=(1.5 * µm, H / 2 + WG_W * 2),
        accumulate_power=True, record_fields=False,
    )
    mon_out1_1 = Monitor(
        design=grid,
        start=(W - 1.5 * µm, OUT1_Y - WG_W * 2),
        end=(W - 1.5 * µm, OUT1_Y + WG_W * 2),
        accumulate_power=True, record_fields=False,
    )

    # Forward sim
    sim_fwd_1 = Simulation(
        grid, [src_fwd_1, mon_in_1, mon_out1_1],
        [PML(edges='all', thickness=PML_THICK)],
        time=time_arr, resolution=DX,
    )
    res_fwd_1 = sim_fwd_1.run(save_fields=['Ez'], field_subsample=FIELD_SUB)
    fwd_ez_1300 = [np.array(f) for f in res_fwd_1['fields']['Ez']]

    in_energy_1 = np.abs(np.sum(mon_in_1.power_history) * DT)
    out_energy_1 = np.abs(np.sum(mon_out1_1.power_history) * DT)
    trans_1 = (out_energy_1 / in_energy_1 * 100.0) if in_energy_1 > 0 else 0.0

    # Adjoint source: -x from upper-right output
    src_adj_1 = ModeSource(
        grid, center=(W - 1.0 * µm, OUT1_Y), width=WG_W * 4,
        wavelength=WL_1, pol="tm", signal=sig_1, direction="-x",
    )
    sim_adj_1 = Simulation(
        grid, [src_adj_1],
        [PML(edges='all', thickness=PML_THICK)],
        time=time_arr, resolution=DX,
    )
    grad_1300 = run_adjoint_and_compute_gradient(
        sim_adj_1, fwd_ez_1300, field_component='Ez', field_subsample=FIELD_SUB,
    )
    assert len(fwd_ez_1300) == 0, "forward history should be emptied"

    # ── Channel 2: 1500 nm ──────────────────────────────────────────────────
    sig_2 = _make_signal(WL_2)

    src_fwd_2 = ModeSource(
        grid, center=(1.0 * µm, H / 2), width=WG_W * 4,
        wavelength=WL_2, pol="tm", signal=sig_2, direction="+x",
    )
    mon_in_2 = Monitor(
        design=grid,
        start=(1.5 * µm, H / 2 - WG_W * 2),
        end=(1.5 * µm, H / 2 + WG_W * 2),
        accumulate_power=True, record_fields=False,
    )
    mon_out2_2 = Monitor(
        design=grid,
        start=(W - 1.5 * µm, OUT2_Y - WG_W * 2),
        end=(W - 1.5 * µm, OUT2_Y + WG_W * 2),
        accumulate_power=True, record_fields=False,
    )

    sim_fwd_2 = Simulation(
        grid, [src_fwd_2, mon_in_2, mon_out2_2],
        [PML(edges='all', thickness=PML_THICK)],
        time=time_arr, resolution=DX,
    )
    res_fwd_2 = sim_fwd_2.run(save_fields=['Ez'], field_subsample=FIELD_SUB)
    fwd_ez_1500 = [np.array(f) for f in res_fwd_2['fields']['Ez']]

    in_energy_2 = np.abs(np.sum(mon_in_2.power_history) * DT)
    out_energy_2 = np.abs(np.sum(mon_out2_2.power_history) * DT)
    trans_2 = (out_energy_2 / in_energy_2 * 100.0) if in_energy_2 > 0 else 0.0

    # Adjoint source: -x from lower-right output
    src_adj_2 = ModeSource(
        grid, center=(W - 1.0 * µm, OUT2_Y), width=WG_W * 4,
        wavelength=WL_2, pol="tm", signal=sig_2, direction="-x",
    )
    sim_adj_2 = Simulation(
        grid, [src_adj_2],
        [PML(edges='all', thickness=PML_THICK)],
        time=time_arr, resolution=DX,
    )
    grad_1500 = run_adjoint_and_compute_gradient(
        sim_adj_2, fwd_ez_1500, field_component='Ez', field_subsample=FIELD_SUB,
    )
    assert len(fwd_ez_1500) == 0, "forward history should be emptied"

    # ── Combine gradients ───────────────────────────────────────────────────
    grad_total = np.array(grad_1300) + np.array(grad_1500)

    # Material penalty
    current_density = np.mean(phys_density[mask])
    grad_penalty = PENALTY_STRENGTH * (current_density - MAT_PENALTY)
    grad_total[mask] -= grad_penalty

    # Step optimiser
    max_update = opt.apply_gradient(grad_total, beta)

    # Track
    trans_history_1300.append(trans_1)
    trans_history_1500.append(trans_2)

    mat_frac = np.mean(phys_density[mask])
    print(
        f" Step {step+1}/{STEPS}: "
        f"T1300={trans_1:.1f}%  T1500={trans_2:.1f}%  "
        f"Mat={mat_frac:.1%}  MaxUp={max_update:.2e}"
    )

    # Periodic permittivity snapshot
    if step % 5 == 0:
        plt.imsave(
            f"demux_eps_{step:03d}.png",
            grid.permittivity.T, cmap='gray', origin='lower',
        )

print(f"\nOptimization complete.")
print(f"  Final T(1300 nm) = {trans_history_1300[-1]:.1f}%")
print(f"  Final T(1500 nm) = {trans_history_1500[-1]:.1f}%")

# ── 6. Transmission vs step plot ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
steps_arr = np.arange(1, STEPS + 1)
plt.plot(steps_arr, trans_history_1300, 'b-o', markersize=3, label='1300 nm (upper)')
plt.plot(steps_arr, trans_history_1500, 'r-s', markersize=3, label='1500 nm (lower)')
plt.xlabel('Optimization Step')
plt.ylabel('Transmission (%)')
plt.title('WDM Demux — Transmission vs Optimization Step')
plt.ylim(0, 100)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('demux_transmission_vs_step.png', dpi=150, bbox_inches='tight')
print("Saved demux_transmission_vs_step.png")
plt.close()

# ── 7. Wavelength sweep ─────────────────────────────────────────────────────
print("\n--- Wavelength Sweep (1200–1600 nm) ---")
wavelengths = np.linspace(1.2 * µm, 1.6 * µm, 15)
sweep_trans_out1 = []   # upper output (target: 1300 nm)
sweep_trans_out2 = []   # lower output (target: 1500 nm)

time_sweep = np.arange(0, 15 * WL_2 / LIGHT_SPEED, DT)

for i, wl in enumerate(wavelengths):
    print(f"  {wl/µm:.3f} µm ...", end="\r")
    sig_sw = ramped_cosine(
        time_sweep, 1.0, LIGHT_SPEED / wl,
        ramp_duration=3.5 * wl / LIGHT_SPEED,
        t_max=time_sweep[-1] / 2,
    )
    src_sw = ModeSource(
        grid, center=(1.0 * µm, H / 2), width=WG_W * 4,
        wavelength=wl, pol="tm", signal=sig_sw, direction="+x",
    )
    src_sw._jz_profile = None
    src_sw.initialize(grid.permittivity, DX)

    mon_in_sw = Monitor(
        design=grid,
        start=(1.5 * µm, H / 2 - WG_W * 2),
        end=(1.5 * µm, H / 2 + WG_W * 2),
        accumulate_power=True, record_fields=False,
    )
    mon_o1_sw = Monitor(
        design=grid,
        start=(W - 1.5 * µm, OUT1_Y - WG_W * 2),
        end=(W - 1.5 * µm, OUT1_Y + WG_W * 2),
        accumulate_power=True, record_fields=False,
    )
    mon_o2_sw = Monitor(
        design=grid,
        start=(W - 1.5 * µm, OUT2_Y - WG_W * 2),
        end=(W - 1.5 * µm, OUT2_Y + WG_W * 2),
        accumulate_power=True, record_fields=False,
    )

    sim_sw = Simulation(
        grid, [src_sw, mon_in_sw, mon_o1_sw, mon_o2_sw],
        [PML(edges='all', thickness=PML_THICK)],
        time=time_sweep, resolution=DX,
    )
    sim_sw.run(save_fields=[], field_subsample=10)

    in_E = np.abs(np.sum(mon_in_sw.power_history) * DT)
    if in_E <= 0:
        in_E = 1.0
    t1 = np.abs(np.sum(mon_o1_sw.power_history) * DT) / in_E * 100.0
    t2 = np.abs(np.sum(mon_o2_sw.power_history) * DT) / in_E * 100.0
    sweep_trans_out1.append(t1)
    sweep_trans_out2.append(t2)

print("\nSweep complete.")

# Spectrum plot
plt.figure(figsize=(10, 6))
plt.plot(wavelengths / µm, sweep_trans_out1, 'b-o', linewidth=2, label='Upper output (1300 nm target)')
plt.plot(wavelengths / µm, sweep_trans_out2, 'r-s', linewidth=2, label='Lower output (1500 nm target)')
plt.axvline(WL_1 / µm, color='blue', linestyle='--', alpha=0.4)
plt.axvline(WL_2 / µm, color='red', linestyle='--', alpha=0.4)
plt.xlabel('Wavelength (µm)')
plt.ylabel('Transmission (%)')
plt.title('WDM Demux — Transmission Spectrum')
plt.ylim(0, 100)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('demux_spectrum.png', dpi=150, bbox_inches='tight')
print("Saved demux_spectrum.png")
plt.close()

# ── 8. Energy flow maps at design wavelengths ────────────────────────────────
for wl_label, wl_val in [("1300nm", WL_1), ("1500nm", WL_2)]:
    print(f"\n--- Energy flow map at {wl_label} ---")
    sig_ef = ramped_cosine(
        time_sweep, 1.0, LIGHT_SPEED / wl_val,
        ramp_duration=3.5 * wl_val / LIGHT_SPEED,
        t_max=time_sweep[-1] / 2,
    )
    src_ef = ModeSource(
        grid, center=(1.0 * µm, H / 2), width=WG_W * 4,
        wavelength=wl_val, pol="tm", signal=sig_ef, direction="+x",
    )
    src_ef._jz_profile = None
    src_ef.initialize(grid.permittivity, DX)

    sim_ef = Simulation(
        grid, [src_ef],
        [PML(edges='all', thickness=PML_THICK)],
        time=time_sweep, resolution=DX,
    )
    res_ef = sim_ef.run(save_fields=['Ez', 'Hx', 'Hy'], field_subsample=1)

    Ez_t = np.array(res_ef['fields']['Ez'])
    Hx_t = np.array(res_ef['fields']['Hx'])
    Hy_t = np.array(res_ef['fields']['Hy'])

    min_x = min(Ez_t.shape[1], Hx_t.shape[1], Hy_t.shape[1])
    min_y = min(Ez_t.shape[2], Hx_t.shape[2], Hy_t.shape[2])
    Ez_c = Ez_t[:, :min_x, :min_y]
    Hx_c = Hx_t[:, :min_x, :min_y]
    Hy_c = Hy_t[:, :min_x, :min_y]

    Sx = -Ez_c * Hy_c
    Sy = Ez_c * Hx_c
    S_mag = np.sqrt(Sx**2 + Sy**2)
    energy_flow = np.sum(S_mag, axis=0) * DT

    plt.figure(figsize=(10, 6))
    perm_c = grid.permittivity[:min_x, :min_y]
    plt.imshow(perm_c.T, cmap='gray', origin='lower', alpha=0.2)
    plt.contour(
        perm_c.T,
        levels=[(N_CORE**2 + N_CLAD**2) / 2],
        colors='white', linewidths=0.5, origin='lower',
    )
    im = plt.imshow(
        energy_flow.T, cmap='inferno', origin='lower',
        alpha=0.9, interpolation='bicubic',
    )
    plt.colorbar(im, label=r'$\int |\mathbf{S}|\, dt$')
    plt.title(f'Energy Flow — {wl_label}')
    plt.xlabel('x (grid cells)')
    plt.ylabel('y (grid cells)')
    plt.tight_layout()
    fname = f'demux_energy_{wl_label}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()

print("\nDone.")
