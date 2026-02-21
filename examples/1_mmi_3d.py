from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle as MplRect

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local workspace package import when running from examples/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beamz import (
    LIGHT_SPEED,
    Design,
    Material,
    ModeSource,
    Monitor,
    PML,
    Rectangle,
    Simulation,
    Taper,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
    µm,
)

# -----------------------------------------------------------------------------
# 3D extension of examples/1_mmi.py with matching x/y geometry and timespan.
# Also exports:
# 1) design projection plot (XY/XZ/YZ) with mode source marker
# 2) mode fields before run
# 3) mid-run Ez snapshot
# 4) normalized cumulative flux plot
# -----------------------------------------------------------------------------

OUT_DIR = Path("benchmarks/results/mmi_3d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match 2D geometry in x/y; extend to a realistic 3D stack in z:
# substrate + cladding + air with enough buffer to keep core away from PML.
X, Y = 20 * µm, 10 * µm
Z_SUBSTRATE = 3.0 * µm
Z_CLADDING = 2.0 * µm
Z_AIR = 3.0 * µm
Z = Z_SUBSTRATE + Z_CLADDING + Z_AIR
WL = 1.55 * µm
TIME = 40 * WL / LIGHT_SPEED
N_CORE, N_CLAD, N_AIR = 2.04, 1.444, 1.0
WG_W = 0.565 * µm
H, W, OFFSET = 3.5 * µm, 9 * µm, 1.05 * µm
MMI_TAPER = 1.5 * µm
# Silicon-like core thickness inside the cladding region.
WG_T = 0.22 * µm
WG_Z0 = Z_SUBSTRATE + 0.9 * µm
PML_THICKNESS = 1.2 * WL
PML_EDGES = "all"

# Use DX/DT helper directly and set 12 points/wavelength as requested.
DX, DT = calc_optimal_fdtd_params(
    WL,
    max(N_CORE, N_CLAD),
    dims=3,
    safety_factor=0.999,
    points_per_wavelength=12,
    width=X,
    height=Y,
    depth=Z,
)

# Build 3D design with explicit material stack in z.
design = Design(width=X, height=Y, depth=Z, material=Material(N_AIR**2))
z0 = WG_Z0
mmi_body_start = X / 2 - W / 2 + MMI_TAPER

# Bottom substrate
design += Rectangle(
    position=(0, 0, 0),
    width=X,
    height=Y,
    depth=Z_SUBSTRATE,
    material=Material(N_CLAD**2),
)
# Middle cladding slab (core sits inside this region)
design += Rectangle(
    position=(0, 0, Z_SUBSTRATE),
    width=X,
    height=Y,
    depth=Z_CLADDING,
    material=Material(N_CLAD**2),
)

design += Rectangle(
    position=(0, Y / 2 - WG_W / 2, z0),
    width=mmi_body_start,
    height=WG_W,
    depth=WG_T,
    material=Material(N_CORE**2),
)
design += Taper(
    position=(mmi_body_start, Y / 2, z0),
    input_width=WG_W,
    output_width=H,
    length=MMI_TAPER,
    depth=WG_T,
    material=Material(N_CORE**2),
)
design += Rectangle(
    position=(mmi_body_start + MMI_TAPER, Y / 2 - H / 2, z0),
    width=W - MMI_TAPER,
    height=H,
    depth=WG_T,
    material=Material(N_CORE**2),
)
design += Rectangle(
    position=(X / 2, Y / 2 + OFFSET - WG_W / 2, z0),
    width=X / 2,
    height=WG_W,
    depth=WG_T,
    material=Material(N_CORE**2),
)
design += Rectangle(
    position=(X / 2, Y / 2 - OFFSET - WG_W / 2, z0),
    width=X / 2,
    height=WG_W,
    depth=WG_T,
    material=Material(N_CORE**2),
)

grid = design.rasterize(resolution=DX)
time_steps = np.arange(0, TIME, DT, dtype=np.float64)
signal = ramped_cosine(
    time_steps,
    amplitude=1.0,
    frequency=LIGHT_SPEED / WL,
    ramp_duration=WL * 6 / LIGHT_SPEED,
    t_max=TIME / 2,
)

source_center = (PML_THICKNESS + 1.5 * µm, Y / 2, WG_Z0 + WG_T / 2)
source_width_base = WG_W * 3.5
# Include core and surrounding cladding in the mode solve window.
source_height_base = 2.4 * µm
source_size = 1.5 * max(source_width_base, source_height_base)
source_width = source_size
source_height = source_size
source = ModeSource(
    grid=grid,
    center=source_center,
    width=source_width,
    height=source_height,
    wavelength=WL,
    pol="tm",
    signal=signal,
    direction="+x",
)
source.initialize(grid.permittivity, DX, dt=DT)

# -----------------------------------------------------------------------------
# Plot geometry projections and mark source.
# -----------------------------------------------------------------------------
eps = np.asarray(grid.permittivity)
eps_levels = np.asarray([N_AIR**2, N_CLAD**2, N_CORE**2], dtype=np.float32)
material_names = ("Air", "Cladding", "Core")
material_colors = ("C0", "C1", "C2")
material_cmap = ListedColormap(material_colors)
material_norm = BoundaryNorm(np.arange(-0.5, len(eps_levels) + 0.5, 1), material_cmap.N)

eps_expanded = eps[..., None]
material_idx = np.argmin(np.abs(eps_expanded - eps_levels[None, None, None, :]), axis=-1)

z_core_idx = int(np.clip(round((WG_Z0 + 0.5 * WG_T) / DX), 0, eps.shape[0] - 1))
xy = material_idx[z_core_idx]  # y,x material classes at core mid-z
y_mid_idx = int(np.clip(round(source_center[1] / DX), 0, eps.shape[1] - 1))
x_src_idx = int(np.clip(round(source_center[0] / DX), 0, eps.shape[2] - 1))
xz = material_idx[:, y_mid_idx, :]  # z,x material slice
yz = material_idx[:, :, x_src_idx]  # z,y material slice

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
axes[0].imshow(
    xy,
    origin="lower",
    cmap=material_cmap,
    norm=material_norm,
    interpolation="nearest",
    extent=[0, X / um, 0, Y / um],
    aspect="equal",
)
axes[0].set_title(f"XY Material Slice (z={((WG_Z0 + 0.5 * WG_T) / um):.2f} um)")
axes[0].set_xlabel("x (um)")
axes[0].set_ylabel("y (um)")
axes[0].add_patch(
    MplRect(
        (source_center[0] / um - 0.03, source_center[1] / um - source_width / (2 * um)),
        0.06,
        source_width / um,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
    )
)

axes[1].imshow(
    xz,
    origin="lower",
    cmap=material_cmap,
    norm=material_norm,
    interpolation="nearest",
    extent=[0, X / um, 0, Z / um],
    aspect="equal",
)
axes[1].set_title("XZ Material Slice (Side)")
axes[1].set_xlabel("x (um)")
axes[1].set_ylabel("z (um)")
axes[1].add_patch(
    MplRect(
        (source_center[0] / um - 0.03, source_center[2] / um - source_height / (2 * um)),
        0.06,
        source_height / um,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
    )
)

axes[2].imshow(
    yz,
    origin="lower",
    cmap=material_cmap,
    norm=material_norm,
    interpolation="nearest",
    extent=[0, Y / um, 0, Z / um],
    aspect="equal",
)
axes[2].set_title("YZ Material Slice (Input Cross-Section)")
axes[2].set_xlabel("y (um)")
axes[2].set_ylabel("z (um)")
axes[2].add_patch(
    MplRect(
        (
            source_center[1] / um - source_width / (2 * um),
            source_center[2] / um - source_height / (2 * um),
        ),
        source_width / um,
        source_height / um,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
    )
)

for ax in axes:
    ax.grid(alpha=0.2, linestyle="--")
    ax.text(
        0.02,
        0.97,
        "red = ModeSource",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="red",
    )
for ax in (axes[1], axes[2]):
    ax.axhline(Z_SUBSTRATE / um, color="cyan", linestyle="--", linewidth=1.0)
    ax.axhline(
        (Z_SUBSTRATE + Z_CLADDING) / um,
        color="cyan",
        linestyle="--",
        linewidth=1.0,
    )

legend_handles = [
    Patch(
        facecolor=material_colors[i],
        edgecolor="black",
        label=f"{material_names[i]} (eps={eps_levels[i]:.3f})",
    )
    for i in range(len(material_names))
]
legend_handles.append(Line2D([0], [0], color="red", lw=2.0, label="ModeSource window"))
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.02),
)
fig.tight_layout(rect=[0, 0.08, 1, 1])
design_proj_png = OUT_DIR / "design_projections_xyz.png"
fig.savefig(design_proj_png, dpi=170)
plt.close(fig)

# -----------------------------------------------------------------------------
# Export mode profiles before running.
# -----------------------------------------------------------------------------
mode_components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
mode_payload: dict[str, np.ndarray] = {}

fig, axarr = plt.subplots(2, 3, figsize=(12, 7))
for i, comp in enumerate(mode_components):
    arr = getattr(source, f"_{comp}_profile", None)
    ax = axarr.ravel()[i]
    if arr is None:
        ax.set_title(f"{comp} (none)")
        ax.axis("off")
        continue
    arr_np = np.asarray(arr, dtype=np.float32)
    mode_payload[f"{comp}_profile"] = arr_np
    im = ax.imshow(np.abs(arr_np), origin="lower", cmap="magma", aspect="auto")
    ax.set_title(comp)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(f"Mode Fields | neff={float(np.real(source._neff)):.6f}")
fig.tight_layout()
mode_png = OUT_DIR / "mode_fields.png"
fig.savefig(mode_png, dpi=170)
plt.close(fig)

mode_npz = OUT_DIR / "mode_fields.npz"
np.savez(mode_npz, **mode_payload)
mode_json = OUT_DIR / "mode_fields.json"
mode_json.write_text(
    json.dumps(
        {
            "direction": source.direction,
            "pol": source.pol,
            "neff_real": float(np.real(source._neff)),
            "neff_imag": float(np.imag(source._neff)),
            "impedance_neff": (
                float(np.real(source._impedance_neff))
                if source._impedance_neff is not None
                else None
            ),
            "source_center_um": [v / um for v in source_center],
            "source_width_um": float(source_width / um),
            "source_height_um": float(source_height / um),
        },
        indent=2,
    )
)

# -----------------------------------------------------------------------------
# Main run with output flux monitor.
# -----------------------------------------------------------------------------
flux_monitor = Monitor(
    design=design,
    start=(
        X - PML_THICKNESS - 1.4 * um,
        PML_THICKNESS + 0.25 * um,
        PML_THICKNESS + 0.25 * um,
    ),
    plane_normal="x",
    plane_position=X - PML_THICKNESS - 1.4 * um,
    size=(
        Y - 2 * (PML_THICKNESS + 0.25 * um),
        Z - 2 * (PML_THICKNESS + 0.25 * um),
    ),
    record_fields=False,
    accumulate_power=True,
    record_interval=1,
    name="mmi_flux_out",
)

sim = Simulation(
    design=design,
    devices=[source, flux_monitor],
    boundaries=[PML(edges=PML_EDGES, thickness=PML_THICKNESS)],
    time=time_steps,
    resolution=DX,
)

t0 = time.perf_counter()
sim.run_compiled(num_steps=len(time_steps), progress=True)
elapsed_s = time.perf_counter() - t0
sim.fields.Ez.block_until_ready()

# -----------------------------------------------------------------------------
# Mid-run Ez snapshot from a fresh half-length run.
# -----------------------------------------------------------------------------
mid_steps = max(1, len(time_steps) // 2)
mid_source = ModeSource(
    grid=grid,
    center=source_center,
    width=source_width,
    height=source_height,
    wavelength=WL,
    pol="tm",
    signal=signal,
    direction="+x",
)
sim_mid = Simulation(
    design=design,
    devices=[mid_source],
    boundaries=[PML(edges=PML_EDGES, thickness=PML_THICKNESS)],
    time=time_steps,
    resolution=DX,
)
sim_mid.run_compiled(num_steps=mid_steps, progress=False)
ez_mid = np.asarray(sim_mid.fields.Ez, dtype=np.float32)
z_mid_idx = ez_mid.shape[0] // 2
ez_plane = ez_mid[z_mid_idx]

ez_mid_png = OUT_DIR / f"ez_snapshot_mid_step_{mid_steps}.png"
fig, ax = plt.subplots(figsize=(8, 5.2))
im = ax.imshow(ez_plane, origin="lower", cmap="RdBu", aspect="auto")
ax.set_title(f"Ez mid-sim snapshot | step={mid_steps}, z_idx={z_mid_idx}")
ax.set_xlabel("x index")
ax.set_ylabel("y index")
fig.colorbar(im, ax=ax, shrink=0.9, label="Ez")
fig.tight_layout()
fig.savefig(ez_mid_png, dpi=180)
plt.close(fig)

ez_mid_npz = OUT_DIR / f"ez_snapshot_mid_step_{mid_steps}.npz"
np.savez(
    ez_mid_npz,
    ez_plane=ez_plane,
    z_index=np.asarray(z_mid_idx, dtype=np.int32),
    step=np.asarray(mid_steps, dtype=np.int32),
    time_s=np.asarray(float(sim_mid.t), dtype=np.float64),
)

# -----------------------------------------------------------------------------
# Flux over time and normalized cumulative flux.
# -----------------------------------------------------------------------------
power = np.asarray(flux_monitor.power_history, dtype=np.float64)
time_s = np.asarray(flux_monitor.power_timestamps, dtype=np.float64)
if time_s.size == 0 and power.size > 0:
    time_s = np.arange(power.size, dtype=np.float64) * DT

cumulative_flux = np.cumsum(np.maximum(power, 0.0)) * DT
cum_den = cumulative_flux[-1] if cumulative_flux.size > 0 and cumulative_flux[-1] > 0 else 1.0
cumulative_flux_norm = cumulative_flux / cum_den if cumulative_flux.size > 0 else cumulative_flux
instant_norm = power / (np.max(np.abs(power)) + 1e-30) if power.size > 0 else power

flux_csv = OUT_DIR / "flux_time_series.csv"
if power.size > 0:
    np.savetxt(
        flux_csv,
        np.column_stack(
            (
                time_s,
                time_s * 1e15,
                power,
                instant_norm,
                cumulative_flux,
                cumulative_flux_norm,
            )
        ),
        delimiter=",",
        header="time_s,time_fs,power,instant_norm,cumulative_flux,cumulative_flux_norm",
        comments="",
    )

flux_npz = OUT_DIR / "flux_time_series.npz"
np.savez(
    flux_npz,
    time_s=time_s,
    time_fs=time_s * 1e15,
    power=power,
    instant_norm=instant_norm,
    cumulative_flux=cumulative_flux,
    cumulative_flux_norm=cumulative_flux_norm,
)

flux_png = OUT_DIR / "flux_cumulative_normalized.png"
fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
if power.size > 0:
    axs[0].plot(time_s * 1e15, instant_norm, color="tab:blue", lw=1.2)
    axs[1].plot(time_s * 1e15, cumulative_flux_norm, color="tab:orange", lw=1.6)
axs[0].set_ylabel("Instantaneous Flux (norm.)")
axs[1].set_ylabel("Cumulative Flux (norm.)")
axs[1].set_xlabel("Time (fs)")
axs[1].set_ylim(-0.02, 1.02)
for ax in axs:
    ax.grid(alpha=0.3)
fig.suptitle("Output Flux Over Time (Normalized)")
fig.tight_layout()
fig.savefig(flux_png, dpi=170)
plt.close(fig)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
voxels = int(np.prod(sim.fields.permittivity.shape))
tcups = (6.0 * voxels * len(time_steps)) / max(elapsed_s, 1e-30) / 1e12

summary = {
    "domain_um": [X / um, Y / um, Z / um],
    "wavelength_um": WL / um,
    "time_fs": TIME * 1e15,
    "dx_nm": DX * 1e9,
    "dt_fs": DT * 1e15,
    "points_per_wavelength": 12,
    "pml_thickness_um": PML_THICKNESS / um,
    "pml_edges": PML_EDGES,
    "z_stack_um": {
        "substrate": Z_SUBSTRATE / um,
        "cladding": Z_CLADDING / um,
        "air": Z_AIR / um,
        "core_z0": WG_Z0 / um,
        "core_thickness": WG_T / um,
    },
    "resolved_steps": int(len(time_steps)),
    "grid_shape_zyx": list(sim.fields.permittivity.shape),
    "elapsed_s": float(elapsed_s),
    "s_per_step": float(elapsed_s / len(time_steps)),
    "tcups": float(tcups),
    "design_projection_png": str(design_proj_png),
    "mode_fields_png": str(mode_png),
    "mid_ez_snapshot_png": str(ez_mid_png),
    "flux_cumulative_plot_png": str(flux_png),
}
summary_path = OUT_DIR / "benchmark_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print("3D MMI benchmark complete")
print(f"dx_nm={summary['dx_nm']:.6f}, dt_fs={summary['dt_fs']:.6f}, steps={summary['resolved_steps']}")
print(f"grid_shape_zyx={summary['grid_shape_zyx']}")
print(f"elapsed_s={summary['elapsed_s']:.6f}, s_per_step={summary['s_per_step']:.6e}, tcups={summary['tcups']:.6e}")
print(f"design_projection_png={design_proj_png}")
print(f"mode_fields_png={mode_png}")
print(f"mid_ez_snapshot_png={ez_mid_png}")
print(f"flux_cumulative_plot_png={flux_png}")
print(f"summary_json={summary_path}")
