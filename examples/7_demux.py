"""
WDM Demultiplexer — Topology Optimization Example
===================================================
Optimizes a compact device that splits 1300 nm (O-band) and 1550 nm (C-band)
light from a single input waveguide into two separate output waveguides using
the adjoint method with streaming (memory-efficient) gradient computation.

Device layout (not to scale):

        ┌──────────────────────────────────┐
        │           PML border             │
        │  ┌────────────────────────────┐  │
        │  │                   ┌──── out1 (1300 nm)
        │  │   ┌────────────┐ │        │  │
  in ───┼──┼───┤ design     ├─┤        │  │
        │  │   │ region     │ │        │  │
        │  │   └────────────┘ │        │  │
        │  │                   └──── out2 (1550 nm)
        │  └────────────────────────────┘  │
        │           PML border             │
        └──────────────────────────────────┘
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
from beamz import *
from beamz.optimization.topology import (
    TopologyManager,
    compute_overlap_gradient,
    create_optimization_mask,
    run_adjoint_and_compute_gradient,
)

# ── 1. Parameters ────────────────────────────────────────────────────────────
W = 8 * µm                 # Domain width
H = 7.0 * µm               # Domain height
WG_W = 0.50 * µm            # Waveguide width
WL_1 = 1.30 * µm           # Channel 1 wavelength (O-band)
WL_2 = 1.55 * µm           # Channel 2 wavelength (C-band)
N_CORE = 3.476             # Si (silicon) refractive index
N_CLAD = 1.444             # SiO2 refractive index
PML_THICK = 1.0 * µm       # PML thickness

# Use the shorter wavelength for resolution (stricter Nyquist)
DX, DT = calc_optimal_fdtd_params(WL_1, N_CORE, points_per_wavelength=9)
STEPS = 100                 # Optimization steps (reduce for faster testing)
FIELD_SUB = 1               # Field subsampling factor


# Ceviche-style step-function beta schedule: hold beta constant for many
# iterations so Adam can converge at each binarization level before ramping.
def get_beta(step):
    while step < 70:
        return 7
    else:
        return 35

# Output waveguide vertical positions
OUT1_Y = H / 2 + 1.15*WG_W  # Upper output (1300 nm)
OUT2_Y = H / 2 - 1.15*WG_W  # Lower output (1550 nm)

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

# Output waveguide 2 — lower right (1550 nm channel)
design += Rectangle(
    position=(W / 2 + 1.5 * µm, OUT2_Y - WG_W / 2),
    width=W / 2 - 1.5 * µm,
    height=WG_W,
    material=core_mat,
)

# Optimization (design) region
opt_region = Rectangle(
    position=(W / 2 - 1.5 * µm, H / 2 - 1.5 * µm),
    width=3.0 * µm,
    height=3.0 * µm,
    material=core_mat,
)
design += opt_region


# ── 3. Objective function ────────────────────────────────────────────────────
def compute_objective(trans_1, trans_2):
    """
    WDM Demux figure of merit (following Ceviche convention).

        J = T₁ · T₂ / 100

    where T(λ → port) = |∫ P_out dt| / |∫ P_in dt| × 100%
    is the fraction of input power at wavelength λ that exits through
    the target output port.

    The product couples both channels: if either drops, J drops sharply.
    This works when the beta schedule gives the optimizer enough
    iterations at each binarization level to converge.
    """
    return trans_1 * trans_2 / 100.0


# ── 4. Sources & monitors helpers ────────────────────────────────────────────
# Simulation time — use longest wavelength for period estimate
sim_duration = 50 * WL_2 / LIGHT_SPEED
time_arr = np.arange(0, sim_duration, DT)


def _make_signal(wl):
    """Create a ramped cosine signal for a given wavelength."""
    return ramped_cosine(
        time_arr, 1.0, LIGHT_SPEED / wl,
        ramp_duration=3 * wl / LIGHT_SPEED,
        t_max=time_arr[-1] / 4,
    )


# ── 5. Rasterize & create optimisation manager ──────────────────────────────
grid = design.rasterize(DX)
mask = create_optimization_mask(grid, opt_region)

opt = TopologyManager(
    design=design,
    region_mask=mask,
    resolution=DX,
    learning_rate=0.01,
    filter_radius=0.18 * µm,
    eps_min=N_CLAD**2,
    eps_max=N_CORE**2,
    beta_schedule=(5.0, 200.0),   # range only; actual schedule is step-function
    filter_type="conic",
)

base_eps = grid.permittivity.copy()

# History tracking
trans_history_1300 = []
trans_history_1550 = []
xtalk_history_1300 = []
xtalk_history_1550 = []
obj_history = []

print(f"WDM Demux Optimization — {STEPS} steps, grid {grid.permittivity.shape}")
print(f"  Channel 1: {WL_1/µm:.2f} µm → upper output (port 2)")
print(f"  Channel 2: {WL_2/µm:.2f} µm → lower output (port 3)")
print(f"  Design region: 3.0 × 3.0 µm")
print(f"\nObjective function:")
print(f"  J = T(λ₁→port2) · T(λ₂→port3) / 100")
print(f"  where T = |∫ P_out dt| / |∫ P_in dt| × 100%")
print(f"  Beta schedule: step-function (10 → 100 → 200)\n")

# ── 5b. Preliminary verification sims ────────────────────────────────────────
print("Running preliminary sims to verify setup...\n")

# Save permittivity map with physical axes
plt.figure(figsize=(8, 6))
extent = [0, W / µm, 0, H / µm]
plt.imshow(grid.permittivity.T, cmap='gray', origin='lower', extent=extent)
plt.colorbar(label='Permittivity (ε)')
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.title('Permittivity map — verify geometry')
plt.tight_layout()
plt.savefig('demux_permittivity.png', dpi=150, bbox_inches='tight')
print("  Saved demux_permittivity.png")
plt.close()

for wl_label, wl_val in [("1300nm", WL_1), ("1550nm", WL_2)]:
    print(f"  Preliminary sim at {wl_label}...")
    sig_pre = _make_signal(wl_val)
    src_pre = ModeSource(
        grid, center=(2.0 * µm, H / 2), width=WG_W * 4,
        wavelength=wl_val, pol="tm", signal=sig_pre, direction="+x",
    )
    mon_in_pre = Monitor(
        design=grid,
        start=(2.5 * µm, H / 2 - WG_W * 1.5),
        end=(2.5 * µm, H / 2 + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )
    mon_o1_pre = Monitor(
        design=grid,
        start=(W - 2.5 * µm, OUT1_Y - WG_W * 1.5),
        end=(W - 2.5 * µm, OUT1_Y + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )
    mon_o2_pre = Monitor(
        design=grid,
        start=(W - 2.5 * µm, OUT2_Y - WG_W * 1.5),
        end=(W - 2.5 * µm, OUT2_Y + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )
    sim_pre = Simulation(
        grid, [src_pre, mon_in_pre, mon_o1_pre, mon_o2_pre],
        [PML(edges='all', thickness=PML_THICK)],
        time=time_arr, resolution=DX,
    )
    res_pre = sim_pre.run(save_fields=['Ez'], field_subsample=4)

    # Field snapshot near end of simulation
    ez_fields = res_pre['fields']['Ez']
    ez_last = np.array(ez_fields[-1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: permittivity overlay with Ez field
    ax = axes[0]
    ax.imshow(grid.permittivity.T, cmap='gray', origin='lower',
              extent=extent, alpha=0.3)
    vmax = np.max(np.abs(ez_last)) * 0.8
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(ez_last.T, cmap='RdBu_r', origin='lower',
                   extent=extent, alpha=0.85, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label='Ez')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_title(f'Ez snapshot — {wl_label}')

    # Right: time-averaged |Ez|² (from all saved frames)
    ez_sq_avg = np.zeros_like(ez_last, dtype=float)
    for frame in ez_fields:
        f = np.array(frame, dtype=float)
        ez_sq_avg += f ** 2
    ez_sq_avg /= max(len(ez_fields), 1)

    ax = axes[1]
    ax.imshow(grid.permittivity.T, cmap='gray', origin='lower',
              extent=extent, alpha=0.3)
    im2 = ax.imshow(ez_sq_avg.T, cmap='hot', origin='lower',
                    extent=extent, alpha=0.85)
    fig.colorbar(im2, ax=ax, label='⟨|Ez|²⟩')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_title(f'Time-avg |Ez|² — {wl_label}')

    fig.suptitle(f'Preliminary verification — {wl_label}', fontsize=13)
    fig.tight_layout()
    fname = f'demux_prelim_{wl_label}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"    Saved {fname}")
    plt.close(fig)

    in_E = np.abs(np.sum(mon_in_pre.power_history) * DT)
    o1_E = np.abs(np.sum(mon_o1_pre.power_history) * DT)
    o2_E = np.abs(np.sum(mon_o2_pre.power_history) * DT)
    if in_E > 0:
        print(f"    Input energy:  {in_E:.4e}")
        print(f"    Out1 energy:   {o1_E:.4e}  ({o1_E/in_E*100:.1f}%)")
        print(f"    Out2 energy:   {o2_E:.4e}  ({o2_E/in_E*100:.1f}%)")
    else:
        print(f"    WARNING: no input energy detected — check source placement")

print("\nPreliminary sims done. Check demux_permittivity.png and demux_prelim_*.png")
print("to verify geometry and propagation direction before optimization.\n")

# ── 6. Optimization loop ────────────────────────────────────────────────────
for step in range(STEPS):
    # --- Update permittivity from density (Ceviche-style step-function beta) ---
    beta = get_beta(step)
    phys_density = opt.get_physical_density(beta)
    grid.permittivity[:] = base_eps
    grid.permittivity[mask] = opt.eps_min + phys_density[mask] * (opt.eps_max - opt.eps_min)

    # ── Channel 1: 1300 nm ──────────────────────────────────────────────────
    sig_1 = _make_signal(WL_1)

    # Forward source: +x from left input
    src_fwd_1 = ModeSource(
        grid, center=(2.0 * µm, H / 2), width=WG_W * 4,
        wavelength=WL_1, pol="tm", signal=sig_1, direction="+x",
    )

    # Monitors
    mon_in_1 = Monitor(
        design=grid,
        start=(2.5 * µm, H / 2 - WG_W * 1.5),
        end=(2.5 * µm, H / 2 + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )
    mon_out1_1 = Monitor(
        design=grid,
        start=(W - 2.5 * µm, OUT1_Y - WG_W * 1.5),
        end=(W - 2.5 * µm, OUT1_Y + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )
    mon_xtalk_1 = Monitor(
        design=grid,
        start=(W - 2.5 * µm, OUT2_Y - WG_W * 1.5),
        end=(W - 2.5 * µm, OUT2_Y + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )

    # Forward sim
    sim_fwd_1 = Simulation(
        grid, [src_fwd_1, mon_in_1, mon_out1_1, mon_xtalk_1],
        [PML(edges='all', thickness=PML_THICK)],
        time=time_arr, resolution=DX,
    )
    res_fwd_1 = sim_fwd_1.run(save_fields=['Ez'], field_subsample=FIELD_SUB)
    fwd_ez_1300 = [np.array(f) for f in res_fwd_1['fields']['Ez']]

    in_energy_1 = np.abs(np.sum(mon_in_1.power_history) * DT)
    out_energy_1 = np.abs(np.sum(mon_out1_1.power_history) * DT)
    trans_1 = (out_energy_1 / in_energy_1 * 100.0) if in_energy_1 > 0 else 0.0
    xtalk_energy_1 = np.abs(np.sum(mon_xtalk_1.power_history) * DT)
    xtalk_1 = (xtalk_energy_1 / in_energy_1 * 100.0) if in_energy_1 > 0 else 0.0

    # Adjoint: maximize correct output (OUT1)
    src_adj_1 = ModeSource(
        grid, center=(W - 2.0 * µm, OUT1_Y), width=WG_W * 4,
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

    # ── Channel 2: 1550 nm ──────────────────────────────────────────────────
    sig_2 = _make_signal(WL_2)

    src_fwd_2 = ModeSource(
        grid, center=(2.0 * µm, H / 2), width=WG_W * 4,
        wavelength=WL_2, pol="tm", signal=sig_2, direction="+x",
    )
    mon_in_2 = Monitor(
        design=grid,
        start=(2.5 * µm, H / 2 - WG_W * 1.5),
        end=(2.5 * µm, H / 2 + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )
    mon_out2_2 = Monitor(
        design=grid,
        start=(W - 2.5 * µm, OUT2_Y - WG_W * 1.5),
        end=(W - 2.5 * µm, OUT2_Y + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )
    mon_xtalk_2 = Monitor(
        design=grid,
        start=(W - 2.5 * µm, OUT1_Y - WG_W * 1.5),
        end=(W - 2.5 * µm, OUT1_Y + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )

    sim_fwd_2 = Simulation(
        grid, [src_fwd_2, mon_in_2, mon_out2_2, mon_xtalk_2],
        [PML(edges='all', thickness=PML_THICK)],
        time=time_arr, resolution=DX,
    )
    res_fwd_2 = sim_fwd_2.run(save_fields=['Ez'], field_subsample=FIELD_SUB)
    fwd_ez_1550 = [np.array(f) for f in res_fwd_2['fields']['Ez']]

    in_energy_2 = np.abs(np.sum(mon_in_2.power_history) * DT)
    out_energy_2 = np.abs(np.sum(mon_out2_2.power_history) * DT)
    trans_2 = (out_energy_2 / in_energy_2 * 100.0) if in_energy_2 > 0 else 0.0
    xtalk_energy_2 = np.abs(np.sum(mon_xtalk_2.power_history) * DT)
    xtalk_2 = (xtalk_energy_2 / in_energy_2 * 100.0) if in_energy_2 > 0 else 0.0

    # Adjoint: maximize correct output (OUT2)
    src_adj_2 = ModeSource(
        grid, center=(W - 2.0 * µm, OUT2_Y), width=WG_W * 4,
        wavelength=WL_2, pol="tm", signal=sig_2, direction="-x",
    )
    sim_adj_2 = Simulation(
        grid, [src_adj_2],
        [PML(edges='all', thickness=PML_THICK)],
        time=time_arr, resolution=DX,
    )
    grad_1550 = run_adjoint_and_compute_gradient(
        sim_adj_2, fwd_ez_1550, field_component='Ez', field_subsample=FIELD_SUB,
    )
    assert len(fwd_ez_1550) == 0, "forward history should be emptied"

    # ── Combine gradients via chain rule ─────────────────────────────────────
    g1 = np.array(grad_1300, dtype=float)
    g2 = np.array(grad_1550, dtype=float)

    # Compute objective and its partials w.r.t. each channel's transmission
    obj = compute_objective(trans_1, trans_2)
    dJ_dT1, dJ_dT2 = jax.grad(compute_objective, argnums=(0, 1))(
        float(trans_1), float(trans_2),
    )

    # Chain rule: dJ/dε = (∂J/∂T₁)·g₁ + (∂J/∂T₂)·g₂
    grad_total = float(dJ_dT1) * g1 + float(dJ_dT2) * g2

    # Step optimizer (apply_gradient handles ascent internally)
    max_update = opt.apply_gradient(grad_total, beta)

    # Track
    trans_history_1300.append(trans_1)
    trans_history_1550.append(trans_2)
    xtalk_history_1300.append(xtalk_1)
    xtalk_history_1550.append(xtalk_2)
    obj_history.append(obj)

    mat_frac = np.mean(phys_density[mask])
    print(
        f" Step {step+1}/{STEPS}: "
        f"J={obj:.1f}%  "
        f"T1300={trans_1:.1f}%  T1550={trans_2:.1f}%  "
        f"X1300={xtalk_1:.1f}%  X1550={xtalk_2:.1f}%  "
        f"β={beta}  Mat={mat_frac:.1%}"
    )

    # Periodic permittivity snapshot
    if step % 5 == 0:
        plt.imsave(
            f"demux_eps_{step:03d}.png",
            grid.permittivity.T, cmap='gray', origin='lower',
        )

print(f"\nOptimization complete.")
print(f"  Final J = {obj_history[-1]:.1f}%")
print(f"  Final T(1300 nm) = {trans_history_1300[-1]:.1f}%  (cross-talk {xtalk_history_1300[-1]:.1f}%)")
print(f"  Final T(1550 nm) = {trans_history_1550[-1]:.1f}%  (cross-talk {xtalk_history_1550[-1]:.1f}%)")

# ── 7. Transmission vs step plot ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
steps_arr = np.arange(1, STEPS + 1)
plt.plot(steps_arr, trans_history_1300, 'b-o', markersize=3, label='1300 nm → upper (T)')
plt.plot(steps_arr, trans_history_1550, 'r-s', markersize=3, label='1550 nm → lower (T)')
plt.plot(steps_arr, xtalk_history_1300, 'b--', markersize=2, alpha=0.5, label='1300 nm → lower (xtalk)')
plt.plot(steps_arr, xtalk_history_1550, 'r--', markersize=2, alpha=0.5, label='1550 nm → upper (xtalk)')
plt.plot(steps_arr, obj_history, 'k-^', markersize=4, linewidth=2, label='Objective J')
plt.xlabel('Optimization Step')
plt.ylabel('Transmission (%)')
plt.title('WDM Demux — Transmission & Cross-talk vs Optimization Step')
plt.ylim(0, 100)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('demux_transmission_vs_step.png', dpi=150, bbox_inches='tight')
print("Saved demux_transmission_vs_step.png")
plt.close()


# ── 8. Wavelength sweep ─────────────────────────────────────────────────────
print("\n--- Wavelength Sweep (1200–1650 nm) ---")
wavelengths = np.linspace(1.2 * µm, 1.65 * µm, 15)
sweep_trans_out1 = []   # upper output (target: 1300 nm)
sweep_trans_out2 = []   # lower output (target: 1550 nm)

time_sweep = np.arange(0, 20 * WL_2 / LIGHT_SPEED, DT)

"""
for i, wl in enumerate(wavelengths):
    print(f"  {wl/µm:.3f} µm ...", end="\r")
    sig_sw = ramped_cosine(
        time_sweep, 1.0, LIGHT_SPEED / wl,
        ramp_duration=3.5 * wl / LIGHT_SPEED,
        t_max=time_sweep[-1] / 2,
    )
    src_sw = ModeSource(
        grid, center=(2.0 * µm, H / 2), width=WG_W * 4,
        wavelength=wl, pol="tm", signal=sig_sw, direction="+x",
    )
    src_sw._jz_profile = None
    src_sw.initialize(grid.permittivity, DX)

    mon_in_sw = Monitor(
        design=grid,
        start=(2.5 * µm, H / 2 - WG_W * 1.5),
        end=(2.5 * µm, H / 2 + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )
    mon_o1_sw = Monitor(
        design=grid,
        start=(W - 2.5 * µm, OUT1_Y - WG_W * 1.5),
        end=(W - 2.5 * µm, OUT1_Y + WG_W * 1.5),
        accumulate_power=True, record_fields=False,
    )
    mon_o2_sw = Monitor(
        design=grid,
        start=(W - 2.5 * µm, OUT2_Y - WG_W * 1.5),
        end=(W - 2.5 * µm, OUT2_Y + WG_W * 1.5),
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
plt.plot(wavelengths / µm, sweep_trans_out2, 'r-s', linewidth=2, label='Lower output (1550 nm target)')
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
"""

# ── 9. Energy flow maps at design wavelengths ────────────────────────────────
for wl_label, wl_val in [("1300nm", WL_1), ("1550nm", WL_2)]:
    print(f"\n--- Energy flow map at {wl_label} ---")
    sig_ef = ramped_cosine(
        time_sweep, 1.0, LIGHT_SPEED / wl_val,
        ramp_duration=3.5 * wl_val / LIGHT_SPEED,
        t_max=time_sweep[-1] / 2,
    )
    src_ef = ModeSource(
        grid, center=(2.0 * µm, H / 2), width=WG_W * 4,
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
