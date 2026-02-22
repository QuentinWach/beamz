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
from beamz.const import BLUE, GREEN, ORANGE, RED

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

# Keep source farther from x-PML to reduce immediate launch truncation losses.
source_center = (PML_THICKNESS + 2.5 * µm, Y / 2, WG_Z0 + WG_T / 2)
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
material_colors = (BLUE, GREEN, ORANGE)
material_cmap = ListedColormap(material_colors)
material_norm = BoundaryNorm(np.arange(-0.5, len(eps_levels) + 0.5, 1), material_cmap.N)

eps_expanded = eps[..., None]
material_idx = np.argmin(
    np.abs(eps_expanded - eps_levels[None, None, None, :]), axis=-1
)

z_core_idx = int(np.clip(round((WG_Z0 + 0.5 * WG_T) / DX), 0, eps.shape[0] - 1))
xy = material_idx[z_core_idx]  # y,x material classes at core mid-z
y_mid_idx = int(np.clip(round(source_center[1] / DX), 0, eps.shape[1] - 1))
x_src_idx = int(np.clip(round(source_center[0] / DX), 0, eps.shape[2] - 1))
xz = material_idx[:, y_mid_idx, :]  # z,x material slice
yz = material_idx[:, :, x_src_idx]  # z,y material slice
outline_levels = np.arange(0.5, len(eps_levels) - 0.5, 1.0)

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
        edgecolor=RED,
        linewidth=2.0,
    )
)
axes[0].contour(
    xy,
    levels=outline_levels,
    colors="black",
    linewidths=1.1,
    origin="lower",
    extent=[0, X / um, 0, Y / um],
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
        (
            source_center[0] / um - 0.03,
            source_center[2] / um - source_height / (2 * um),
        ),
        0.06,
        source_height / um,
        fill=False,
        edgecolor=RED,
        linewidth=2.0,
    )
)
axes[1].contour(
    xz,
    levels=outline_levels,
    colors="black",
    linewidths=1.1,
    origin="lower",
    extent=[0, X / um, 0, Z / um],
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
        edgecolor=RED,
        linewidth=2.0,
    )
)
axes[2].contour(
    yz,
    levels=outline_levels,
    colors="black",
    linewidths=1.1,
    origin="lower",
    extent=[0, Y / um, 0, Z / um],
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
        color=RED,
    )


def draw_pml_lines(
    ax, x_len_um: float, y_len_um: float, pml_um: float, color: str = "black"
):
    ax.axvline(pml_um, color=color, linestyle="--", linewidth=1.0)
    ax.axvline(x_len_um - pml_um, color=color, linestyle="--", linewidth=1.0)
    ax.axhline(pml_um, color=color, linestyle="--", linewidth=1.0)
    ax.axhline(y_len_um - pml_um, color=color, linestyle="--", linewidth=1.0)


pml_um = PML_THICKNESS / um
draw_pml_lines(axes[0], X / um, Y / um, pml_um)
draw_pml_lines(axes[1], X / um, Z / um, pml_um)
draw_pml_lines(axes[2], Y / um, Z / um, pml_um)

legend_handles = [
    Patch(
        facecolor=material_colors[i],
        edgecolor="black",
        label=f"{material_names[i]} (eps={eps_levels[i]:.3f})",
    )
    for i in range(len(material_names))
]
legend_handles.append(Line2D([0], [0], color=RED, lw=2.0, label="ModeSource window"))
legend_handles.append(
    Line2D([0], [0], color="black", lw=1.2, label="Material boundaries")
)
legend_handles.append(
    Line2D([0], [0], color="black", lw=1.0, linestyle="--", label="PML boundaries")
)
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
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
# XY-slice monitor and time-integrated flux map (Poynting accumulation).
# -----------------------------------------------------------------------------
pml_inner_margin = PML_THICKNESS + 0.25 * um
xy_flux_source = ModeSource(
    grid=grid,
    center=source_center,
    width=source_width,
    height=source_height,
    wavelength=WL,
    pol="tm",
    signal=signal,
    direction="+x",
)
sim_xy_flux = Simulation(
    design=design,
    devices=[xy_flux_source],
    boundaries=[PML(edges=PML_EDGES, thickness=PML_THICKNESS)],
    time=time_steps,
    resolution=DX,
)


xy_flux_fields = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
integrated_flux_mag = None
integrated_sx = None
integrated_sy = None
integrated_sz = None
ny_flux = None
nx_flux = None
flux_chunk_steps = 2
steps_done = 0
total_steps = len(time_steps)

# Insertion-loss diagnostics around source launch region (signed Sx on yz planes).
probe_offset = 1.2 * um
x_probe_right = min(source_center[0] + probe_offset, mmi_body_start - 0.4 * um)
x_probe_left = max(source_center[0] - probe_offset, pml_inner_margin + 0.4 * um)
insertion_masks_ready = False
roi_mask = None
mode_mask = None
x_probe_right_idx = None
x_probe_left_idx = None
record_idx = 0

ins_time_buf = np.zeros(total_steps, dtype=np.float64)
ins_right_total_buf = np.zeros(total_steps, dtype=np.float64)
ins_right_mode_buf = np.zeros(total_steps, dtype=np.float64)
ins_right_outside_buf = np.zeros(total_steps, dtype=np.float64)
ins_left_backward_buf = np.zeros(total_steps, dtype=np.float64)
ins_left_forward_buf = np.zeros(total_steps, dtype=np.float64)

while steps_done < total_steps:
    this_chunk = min(flux_chunk_steps, total_steps - steps_done)
    chunk_result = sim_xy_flux.run_compiled(
        num_steps=this_chunk,
        record_interval=1,
        record_fields=list(xy_flux_fields),
        progress=False,
    )
    f = chunk_result["fields"]
    nrec = int(f["Ex"].shape[0])
    for i in range(nrec):
        ex_i = f["Ex"][i]
        ey_i = f["Ey"][i]
        ez_i = f["Ez"][i]
        hx_i = f["Hx"][i]
        hy_i = f["Hy"][i]
        hz_i = f["Hz"][i]

        nz = min(
            ex_i.shape[0],
            ey_i.shape[0],
            ez_i.shape[0],
            hx_i.shape[0],
            hy_i.shape[0],
            hz_i.shape[0],
        )
        ny = min(
            ex_i.shape[1],
            ey_i.shape[1],
            ez_i.shape[1],
            hx_i.shape[1],
            hy_i.shape[1],
            hz_i.shape[1],
        )
        nx = min(
            ex_i.shape[2],
            ey_i.shape[2],
            ez_i.shape[2],
            hx_i.shape[2],
            hy_i.shape[2],
            hz_i.shape[2],
        )

        ex = np.asarray(ex_i[:nz, :ny, :nx], dtype=np.float64)
        ey = np.asarray(ey_i[:nz, :ny, :nx], dtype=np.float64)
        ez = np.asarray(ez_i[:nz, :ny, :nx], dtype=np.float64)
        hx = np.asarray(hx_i[:nz, :ny, :nx], dtype=np.float64)
        hy = np.asarray(hy_i[:nz, :ny, :nx], dtype=np.float64)
        hz = np.asarray(hz_i[:nz, :ny, :nx], dtype=np.float64)

        sx_3d = ey * hz - ez * hy
        sy_3d = ez * hx - ex * hz
        sz_3d = ex * hy - ey * hx
        s_mag_3d = np.sqrt(sx_3d * sx_3d + sy_3d * sy_3d + sz_3d * sz_3d)

        if not insertion_masks_ready:
            z_cells, y_cells, x_cells = sx_3d.shape

            def _idx_for_pos(n_cells: int, pos: float, total: float) -> int:
                if n_cells <= 1:
                    return 0
                return int(np.clip(round(pos / total * (n_cells - 1)), 0, n_cells - 1))

            x_probe_right_idx = _idx_for_pos(x_cells, x_probe_right, X)
            x_probe_left_idx = _idx_for_pos(x_cells, x_probe_left, X)
            y0 = _idx_for_pos(y_cells, pml_inner_margin, Y)
            y1 = _idx_for_pos(y_cells, Y - pml_inner_margin, Y)
            z0 = _idx_for_pos(z_cells, pml_inner_margin, Z)
            z1 = _idx_for_pos(z_cells, Z - pml_inner_margin, Z)
            if y1 <= y0:
                y0, y1 = 0, y_cells
            if z1 <= z0:
                z0, z1 = 0, z_cells

            y_mode_lo = max(pml_inner_margin, source_center[1] - 0.5 * source_width)
            y_mode_hi = min(Y - pml_inner_margin, source_center[1] + 0.5 * source_width)
            z_mode_lo = max(pml_inner_margin, source_center[2] - 0.5 * source_height)
            z_mode_hi = min(
                Z - pml_inner_margin, source_center[2] + 0.5 * source_height
            )
            ym0 = _idx_for_pos(y_cells, y_mode_lo, Y)
            ym1 = _idx_for_pos(y_cells, y_mode_hi, Y)
            zm0 = _idx_for_pos(z_cells, z_mode_lo, Z)
            zm1 = _idx_for_pos(z_cells, z_mode_hi, Z)
            if ym1 <= ym0:
                ym0, ym1 = y0, y1
            if zm1 <= zm0:
                zm0, zm1 = z0, z1

            roi_mask = np.zeros((z_cells, y_cells), dtype=bool)
            roi_mask[z0:z1, y0:y1] = True
            mode_mask = np.zeros((z_cells, y_cells), dtype=bool)
            mode_mask[zm0:zm1, ym0:ym1] = True
            mode_mask &= roi_mask
            insertion_masks_ready = True

        sx_right_plane = sx_3d[:, :, x_probe_right_idx]
        sx_left_plane = sx_3d[:, :, x_probe_left_idx]
        area = DX * DX
        p_right_total = np.sum(np.maximum(sx_right_plane[roi_mask], 0.0)) * area
        p_right_mode = np.sum(np.maximum(sx_right_plane[mode_mask], 0.0)) * area
        p_right_outside = max(0.0, p_right_total - p_right_mode)
        p_left_backward = np.sum(np.maximum(-sx_left_plane[roi_mask], 0.0)) * area
        p_left_forward = np.sum(np.maximum(sx_left_plane[roi_mask], 0.0)) * area
        if record_idx < total_steps:
            t_s = (steps_done + i + 1) * DT
            ins_time_buf[record_idx] = float(t_s)
            ins_right_total_buf[record_idx] = float(p_right_total)
            ins_right_mode_buf[record_idx] = float(p_right_mode)
            ins_right_outside_buf[record_idx] = float(p_right_outside)
            ins_left_backward_buf[record_idx] = float(p_left_backward)
            ins_left_forward_buf[record_idx] = float(p_left_forward)
            record_idx += 1

        # Integrate across the full z-direction so the XY map captures all vertical leakage.
        sx = np.sum(sx_3d, axis=0) * DX
        sy = np.sum(sy_3d, axis=0) * DX
        sz = np.sum(sz_3d, axis=0) * DX
        s_mag = np.sum(s_mag_3d, axis=0) * DX

        if integrated_flux_mag is None:
            ny_flux, nx_flux = ny, nx
            integrated_flux_mag = np.zeros((ny_flux, nx_flux), dtype=np.float64)
            integrated_sx = np.zeros((ny_flux, nx_flux), dtype=np.float64)
            integrated_sy = np.zeros((ny_flux, nx_flux), dtype=np.float64)
            integrated_sz = np.zeros((ny_flux, nx_flux), dtype=np.float64)

        integrated_flux_mag += s_mag * DT
        integrated_sx += sx * DT
        integrated_sy += sy * DT
        integrated_sz += sz * DT

    steps_done += this_chunk
    if steps_done % max(1, total_steps // 8) == 0 or steps_done == total_steps:
        print(f"● XY flux accumulation: {steps_done}/{total_steps} steps")

if integrated_flux_mag is None:
    integrated_flux_mag = np.zeros((1, 1), dtype=np.float64)
    integrated_sx = np.zeros((1, 1), dtype=np.float64)
    integrated_sy = np.zeros((1, 1), dtype=np.float64)
    integrated_sz = np.zeros((1, 1), dtype=np.float64)
    ny_flux = nx_flux = 1

flux_mag_norm = integrated_flux_mag / (np.max(integrated_flux_mag) + 1e-30)
xy_outline = xy[:ny_flux, :nx_flux]
xy_core_outline = (xy_outline == 2).astype(np.float32)
x_extent_flux = nx_flux * DX / um
y_extent_flux = ny_flux * DX / um
xy_flux_png = OUT_DIR / "xy_time_integrated_flux.png"
fig, ax = plt.subplots(figsize=(9, 5.6))
im = ax.imshow(
    flux_mag_norm,
    origin="lower",
    extent=[0, x_extent_flux, 0, y_extent_flux],
    cmap="inferno",
    aspect="equal",
)
ax.contour(
    xy_core_outline,
    levels=[0.5],
    colors="white",
    linewidths=1.2,
    origin="lower",
    extent=[0, x_extent_flux, 0, y_extent_flux],
)
draw_pml_lines(ax, x_extent_flux, y_extent_flux, pml_um, color="white")
ax.set_title("XY Time-Integrated Flux (z-integrated)")
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
cbar = fig.colorbar(im, ax=ax, shrink=0.92)
cbar.set_label("z-integrated |S| (norm.)")
ax.grid(alpha=0.25, linestyle="--")
fig.tight_layout()
fig.savefig(xy_flux_png, dpi=180)
plt.close(fig)

xy_flux_npz = OUT_DIR / "xy_time_integrated_flux.npz"
np.savez(
    xy_flux_npz,
    flux_mag=integrated_flux_mag,
    flux_mag_norm=flux_mag_norm,
    flux_sx=integrated_sx,
    flux_sy=integrated_sy,
    flux_sz=integrated_sz,
    z_integrated=np.asarray(True),
    z_min_um=np.asarray(0.0, dtype=np.float64),
    z_max_um=np.asarray(Z / um, dtype=np.float64),
    dx_um=np.asarray(DX / um, dtype=np.float64),
)

# -----------------------------------------------------------------------------
# Insertion-loss diagnostics and visualization.
# -----------------------------------------------------------------------------
ins_t = ins_time_buf[:record_idx]
ins_right_total = ins_right_total_buf[:record_idx]
ins_right_mode = ins_right_mode_buf[:record_idx]
ins_right_outside = ins_right_outside_buf[:record_idx]
ins_left_backward = ins_left_backward_buf[:record_idx]
ins_left_forward = ins_left_forward_buf[:record_idx]

ins_e_right_total = np.sum(ins_right_total) * DT
ins_e_right_mode = np.sum(ins_right_mode) * DT
ins_e_right_outside = np.sum(ins_right_outside) * DT
ins_e_left_backward = np.sum(ins_left_backward) * DT
ins_e_left_forward = np.sum(ins_left_forward) * DT
ins_e_emitted = ins_e_right_total + ins_e_left_backward

if ins_e_emitted > 0:
    ins_guided_frac = ins_e_right_mode / ins_e_emitted
    ins_radiative_frac = ins_e_right_outside / ins_e_emitted
    ins_reflected_frac = ins_e_left_backward / ins_e_emitted
else:
    ins_guided_frac = 0.0
    ins_radiative_frac = 0.0
    ins_reflected_frac = 0.0

cum_right_total = np.cumsum(ins_right_total) * DT
cum_right_mode = np.cumsum(ins_right_mode) * DT
cum_right_out = np.cumsum(ins_right_outside) * DT
cum_left_back = np.cumsum(ins_left_backward) * DT
cum_emitted = cum_right_total + cum_left_back
cum_den = np.where(cum_emitted > 0.0, cum_emitted, 1.0)
cum_guided_frac = cum_right_mode / cum_den
cum_radiative_frac = cum_right_out / cum_den
cum_reflected_frac = cum_left_back / cum_den

# Suppress unstable early-time ratios before meaningful emitted energy exists.
emit_floor = max(
    1e-30, 1e-6 * float(cum_emitted[-1]) if cum_emitted.size > 0 else 1e-30
)
valid_frac = cum_emitted >= emit_floor
cum_guided_plot = np.where(valid_frac, cum_guided_frac, np.nan)
cum_radiative_plot = np.where(valid_frac, cum_radiative_frac, np.nan)
cum_reflected_plot = np.where(valid_frac, cum_reflected_frac, np.nan)

ins_csv = OUT_DIR / "insertion_flux_timeseries.csv"
if ins_t.size > 0:
    np.savetxt(
        ins_csv,
        np.column_stack(
            (
                ins_t,
                ins_t * 1e15,
                ins_right_total,
                ins_right_mode,
                ins_right_outside,
                ins_left_backward,
                ins_left_forward,
                cum_guided_frac,
                cum_radiative_frac,
                cum_reflected_frac,
            )
        ),
        delimiter=",",
        header=(
            "time_s,time_fs,p_right_total,p_right_mode,p_right_outside,"
            "p_left_backward,p_left_forward,cum_guided_frac,cum_radiative_frac,cum_reflected_frac"
        ),
        comments="",
    )

ins_npz = OUT_DIR / "insertion_flux_timeseries.npz"
np.savez(
    ins_npz,
    time_s=ins_t,
    time_fs=ins_t * 1e15,
    p_right_total=ins_right_total,
    p_right_mode=ins_right_mode,
    p_right_outside=ins_right_outside,
    p_left_backward=ins_left_backward,
    p_left_forward=ins_left_forward,
    cum_guided_frac=cum_guided_frac,
    cum_radiative_frac=cum_radiative_frac,
    cum_reflected_frac=cum_reflected_frac,
    x_probe_right_um=np.asarray(x_probe_right / um, dtype=np.float64),
    x_probe_left_um=np.asarray(x_probe_left / um, dtype=np.float64),
)

ins_png = OUT_DIR / "insertion_diagnostics.png"
fig, axs = plt.subplots(2, 1, figsize=(9.4, 7.0), sharex=True)
if ins_t.size > 0:
    norm_den = np.max(
        np.array(
            [
                np.max(ins_right_total) if ins_right_total.size else 0.0,
                np.max(ins_right_mode) if ins_right_mode.size else 0.0,
                np.max(ins_right_outside) if ins_right_outside.size else 0.0,
                np.max(ins_left_backward) if ins_left_backward.size else 0.0,
            ]
        )
    )
    norm_den = norm_den if norm_den > 0 else 1.0
    t_fs = ins_t * 1e15
    axs[0].plot(
        t_fs, ins_right_total / norm_den, color=BLUE, lw=1.4, label="Right +Sx (total)"
    )
    axs[0].plot(
        t_fs,
        ins_right_mode / norm_den,
        color=GREEN,
        lw=1.2,
        label="Right +Sx (mode window)",
    )
    axs[0].plot(
        t_fs,
        ins_right_outside / norm_den,
        color=ORANGE,
        lw=1.2,
        label="Right +Sx (outside window)",
    )
    axs[0].plot(
        t_fs,
        ins_left_backward / norm_den,
        color=RED,
        lw=1.2,
        label="Left -Sx (backward)",
    )
    axs[1].plot(t_fs, cum_guided_plot, color=GREEN, lw=1.6, label="Guided fraction")
    axs[1].plot(
        t_fs,
        cum_radiative_plot,
        color=ORANGE,
        lw=1.4,
        label="Radiative/outside fraction",
    )
    axs[1].plot(t_fs, cum_reflected_plot, color=RED, lw=1.4, label="Reflected fraction")
axs[0].set_ylabel("Instantaneous Flux (norm.)")
axs[1].set_ylabel("Cumulative Energy Fraction")
axs[1].set_xlabel("Time (fs)")
axs[1].set_ylim(-0.02, 1.02)
for ax in axs:
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
axs[0].set_title("Mode Insertion Diagnostics")
axs[1].text(
    0.01,
    0.03,
    (
        f"Final guided={ins_guided_frac:.3f}, outside={ins_radiative_frac:.3f}, "
        f"reflected={ins_reflected_frac:.3f}"
    ),
    transform=axs[1].transAxes,
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="0.5"),
)
fig.tight_layout()
fig.savefig(ins_png, dpi=170)
plt.close(fig)

# -----------------------------------------------------------------------------
# Flux over time and normalized cumulative flux.
# -----------------------------------------------------------------------------
power = np.asarray(flux_monitor.power_history, dtype=np.float64)
time_s = np.asarray(flux_monitor.power_timestamps, dtype=np.float64)
if time_s.size == 0 and power.size > 0:
    time_s = np.arange(power.size, dtype=np.float64) * DT

cumulative_flux = np.cumsum(np.maximum(power, 0.0)) * DT
cum_den = (
    cumulative_flux[-1] if cumulative_flux.size > 0 and cumulative_flux[-1] > 0 else 1.0
)
cumulative_flux_norm = (
    cumulative_flux / cum_den if cumulative_flux.size > 0 else cumulative_flux
)
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
    axs[0].plot(time_s * 1e15, instant_norm, color=BLUE, lw=1.2)
    axs[1].plot(time_s * 1e15, cumulative_flux_norm, color=ORANGE, lw=1.6)
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
    "insertion_diagnostics_png": str(ins_png),
    "insertion_flux_timeseries_npz": str(ins_npz),
    "insertion_flux_timeseries_csv": str(ins_csv),
    "insertion_probe_x_left_um": float(x_probe_left / um),
    "insertion_probe_x_right_um": float(x_probe_right / um),
    "insertion_guided_fraction": float(ins_guided_frac),
    "insertion_outside_fraction": float(ins_radiative_frac),
    "insertion_reflected_fraction": float(ins_reflected_frac),
    "insertion_energy_right_total": float(ins_e_right_total),
    "insertion_energy_right_mode": float(ins_e_right_mode),
    "insertion_energy_right_outside": float(ins_e_right_outside),
    "insertion_energy_left_backward": float(ins_e_left_backward),
    "insertion_energy_left_forward": float(ins_e_left_forward),
    "insertion_samples": int(record_idx),
    "xy_time_integrated_flux_png": str(xy_flux_png),
    "xy_time_integrated_flux_npz": str(xy_flux_npz),
    "xy_flux_mode": "z_integrated",
}
summary_path = OUT_DIR / "benchmark_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print("3D MMI benchmark complete")
print(
    f"dx_nm={summary['dx_nm']:.6f}, dt_fs={summary['dt_fs']:.6f}, steps={summary['resolved_steps']}"
)
print(f"grid_shape_zyx={summary['grid_shape_zyx']}")
print(
    f"elapsed_s={summary['elapsed_s']:.6f}, s_per_step={summary['s_per_step']:.6e}, tcups={summary['tcups']:.6e}"
)
print(f"design_projection_png={design_proj_png}")
print(f"mode_fields_png={mode_png}")
print(f"mid_ez_snapshot_png={ez_mid_png}")
print(f"flux_cumulative_plot_png={flux_png}")
print(f"insertion_diagnostics_png={ins_png}")
print(
    "insertion_fractions="
    f"guided:{ins_guided_frac:.4f}, outside:{ins_radiative_frac:.4f}, reflected:{ins_reflected_frac:.4f}"
)
print(f"xy_time_integrated_flux_png={xy_flux_png}")
print(f"summary_json={summary_path}")
