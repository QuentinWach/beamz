"""Tiny standalone 3D BeamZ crossing example.

Workflow:
1. Define fixed hyperparameters.
2. Build the 3D crossing geometry and extend the ports.
3. Define the broadband source and DFT monitors.
4. Build the simulation.
5. Save an overview plot of the design, source, and monitors.
6. Run the simulation with adaptive monitor-decay stopping.
7. Extract and plot the S-parameters.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import shutil
import subprocess
import sys
import time as pytime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import beamz.simulation.core as simulation_core
from beamz import (
    LIGHT_SPEED,
    PML,
    ModeMonitor,
    ModeSource,
    Monitor,
    PortSpec,
    Simulation,
    dxdt,
    µm,
)
from beamz.design.io import gdsf
from beamz.devices._placement import snap_plane_region
from beamz.devices.sources.signals import gaussian_band_pulse

# Fixed example hyperparameters. Match the gsim/meep reference geometry and
# domain sizing directly, while using the currently best-performing BeamZ
# source-port spacing from the local #106 benchmark harness.
OUT_DIR = Path("benchmarks/results/tiny_beamz_crossing")
COMPONENT_NAME = "ebeam_crossing4"
UBC_PDK_REQUIREMENT = "ubcpdk==2.7.0"
NUM_FREQS = 16
PPW = 8
WL0, WL_MIN, WL_MAX = 1550.0e-9, 1530.0e-9, 1570.0e-9
N_CORE, N_CLAD = 3.47, 1.44
LAYER = (1, 0)
CORE_T = 0.22 * µm
CLAD_BELOW = 0.50 * µm
CLAD_ABOVE = 0.50 * µm
PML_XY, PML_Z = 1.0 * µm, 1.0 * µm
# For the y-directed weak ports, deeper planes look cleaner visually but snap
# to a less symmetric pair of Yee slices at 8 PPW. Keep the outputs near the
# imported port planes so o2/o4 stay mirror-locked on the raster grid.
OUTPUT_MONITOR_OFFSET = 0.05 * µm
MONITOR_TO_PML_SPACING = 1.00 * µm
Z_PADDING = 0.50 * µm
PORT_OVERLAP = 0.0 * µm
PORT_MARGIN = 0.50 * µm
MODE_PLANE_SIZE_SCALE = 1.8
MONITOR_Z_SPAN = MODE_PLANE_SIZE_SCALE * (CORE_T + 2.0 * PORT_MARGIN)
SOURCE_PORT_OFFSET = 0.10 * µm
DISTANCE_SOURCE_TO_MONITORS = 0.40 * µm
RUN_AFTER_SOURCES_UOC = 90.0
DECAY_RATIO = 1e-4
LOOKBACK_RECORDS = 20
PML_FORMULATION = "sigma"
# Keep this compact regression on vertical sidewalls. The full UBC stack's
# sloped sidewalls make the sponge PML terminal waveguides harder to absorb
# cleanly and reintroduce a measurable top/bottom imbalance.
USE_PDK_LAYER_STACK = False


def ubcpdk_crossing_gds_available(cell: str = COMPONENT_NAME) -> bool:
    spec = importlib.util.find_spec("ubcpdk")
    if spec is None or not spec.submodule_search_locations:
        return False
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    return (package_dir / "gds" / f"{cell}.gds").exists()


def ensure_ubcpdk_available(requirement: str = UBC_PDK_REQUIREMENT) -> None:
    """Install the compatible optional UBC PDK into the active Python env."""
    if ubcpdk_crossing_gds_available():
        return

    print(f"Installing compatible UBC PDK dependency: {requirement}")
    install_commands = []
    if importlib.util.find_spec("pip") is not None:
        install_commands.append([sys.executable, "-m", "pip", "install", requirement])
    uv = shutil.which("uv")
    if uv is not None:
        install_commands.append(
            [uv, "pip", "install", "--python", sys.executable, requirement]
        )
    if not install_commands:
        raise RuntimeError(
            "Could not find `pip` or `uv` to install the UBC PDK. Install it "
            f"manually with `uv pip install --python {sys.executable} "
            f"'{requirement}'` and rerun this example."
        )

    errors = []
    for command in install_commands:
        try:
            subprocess.check_call(command)
            break
        except subprocess.CalledProcessError as exc:
            errors.append(f"{' '.join(command)} -> exit {exc.returncode}")
    else:
        raise RuntimeError(
            "Failed to install the UBC PDK. Tried:\n  "
            + "\n  ".join(errors)
            + "\nInstall it manually with "
            f"`uv pip install --python {sys.executable} '{requirement}'` and "
            "rerun this example."
        )

    importlib.invalidate_caches()
    if not ubcpdk_crossing_gds_available():
        raise RuntimeError(
            "The UBC PDK install command completed, but the packaged "
            f"`gds/{COMPONENT_NAME}.gds` file is still not available from this "
            "Python environment."
        )


def micromode_has_right_handed_y_basis() -> bool:
    """Return whether installed micromode has the y-normal basis fix."""
    try:
        raw_version = importlib.metadata.version("micromode")
    except importlib.metadata.PackageNotFoundError:
        return False

    if raw_version.startswith("0.1.0a"):
        suffix = raw_version.split("0.1.0a", 1)[1]
        digits = ""
        for char in suffix:
            if not char.isdigit():
                break
            digits += char
        if digits:
            return int(digits) >= 4
    return raw_version not in {"0.1.0a1", "0.1.0a2", "0.1.0a3"}


def use_fixed_micromode_y_projection_convention() -> None:
    """Keep this example compatible with micromode's corrected y-normal basis."""
    if not micromode_has_right_handed_y_basis():
        return

    def projection_solver_direction(direction: str, axis: str) -> str:
        direction = str(direction).lower()
        axis = str(axis).lower()
        if axis == "x":
            return ("-" if direction.startswith("+") else "+") + axis
        if axis == "y":
            return direction
        return direction

    simulation_core._projection_solver_direction_3d = projection_solver_direction
    print("Using fixed micromode y projection branch convention.")


def ubcpdk_gds_layer_span(cell: str, layer: tuple[int, int]) -> tuple[float, float]:
    """Return the GDS layer bbox span in meters without touching gdsfactory state."""
    import gdspy

    spec = importlib.util.find_spec("ubcpdk")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("ubcpdk is not installed.")
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    gdspath = package_dir / "gds" / f"{cell}.gds"
    if not gdspath.exists():
        raise FileNotFoundError(f"Could not find '{gdspath.name}' in ubcpdk gds data.")

    lib = gdspy.GdsLibrary(infile=str(gdspath))
    polygons = []
    for top_cell in lib.top_level():
        polygons.extend(top_cell.get_polygons(by_spec=True).get(tuple(layer), []))
    if not polygons:
        raise ValueError(f"Layer {tuple(layer)} not found in '{gdspath.name}'.")
    points = np.vstack([np.asarray(poly)[:, :2] for poly in polygons])
    span_um = np.max(points, axis=0) - np.min(points, axis=0)
    return float(span_um[0]) * µm, float(span_um[1]) * µm


def move_along(center: tuple[float, float], direction: str, distance: float):
    x, y = center
    return {
        "+x": (x + distance, y),
        "-x": (x - distance, y),
        "+y": (x, y + distance),
        "-y": (x, y - distance),
    }[str(direction)]


def port_plane(
    port: dict,
    *,
    span: float,
    z_span: float,
    z_center: float,
    offset: float = 0.0,
):
    cx, cy = move_along(port["center"], port["direction"], offset)
    z0 = float(z_center) - 0.5 * float(z_span)
    z1 = float(z_center) + 0.5 * float(z_span)
    if str(port["direction"]).endswith("x"):
        return (cx, cy - 0.5 * float(span), z0), (cx, cy + 0.5 * float(span), z1)
    return (cx - 0.5 * float(span), cy, z0), (cx + 0.5 * float(span), cy, z1)


def line_center(line):
    a, b = line
    return tuple(0.5 * (float(a[i]) + float(b[i])) for i in range(len(a)))


def reflect_plane_y(line, *, mirror_y: float):
    """Reflect a monitor plane across a horizontal symmetry line."""
    reflected = []
    for point in line:
        reflected.append(
            (
                float(point[0]),
                2.0 * float(mirror_y) - float(point[1]),
                float(point[2]),
            )
        )
    return tuple(reflected)


def signed_port_plane_offset(line, port: dict) -> float:
    center = line_center(line)
    port_center = port["center"]
    direction = str(port["direction"])
    if direction == "+x":
        return float(center[0]) - float(port_center[0])
    if direction == "-x":
        return float(port_center[0]) - float(center[0])
    if direction == "+y":
        return float(center[1]) - float(port_center[1])
    if direction == "-y":
        return float(port_center[1]) - float(center[1])
    raise ValueError(f"Unsupported port direction {direction!r}.")


def mirror_error_y(line_a, line_b, *, mirror_y: float) -> float:
    ca = line_center(line_a)
    cb = line_center(line_b)
    return max(
        abs(float(ca[0]) - float(cb[0])),
        abs(float(ca[1]) + float(cb[1]) - 2.0 * float(mirror_y)),
        abs(float(ca[2]) - float(cb[2])),
    )


def pml_clearances_xy(
    line,
    *,
    width: float,
    height: float,
    pml_xy: float,
) -> dict[str, float]:
    (x0, y0, _), (x1, y1, _) = line
    xmin, xmax = min(float(x0), float(x1)), max(float(x0), float(x1))
    ymin, ymax = min(float(y0), float(y1)), max(float(y0), float(y1))
    return {
        "left": xmin - float(pml_xy),
        "right": float(width) - float(pml_xy) - xmax,
        "bottom": ymin - float(pml_xy),
        "top": float(height) - float(pml_xy) - ymax,
    }


def pml_clearance_for_port(
    line,
    port: dict,
    *,
    width: float,
    height: float,
    pml_xy: float,
) -> float:
    clearances = pml_clearances_xy(line, width=width, height=height, pml_xy=pml_xy)
    return {
        "+x": clearances["left"],
        "-x": clearances["right"],
        "+y": clearances["bottom"],
        "-y": clearances["top"],
    }[str(port["direction"])]


def cell_aligned_xy_padding(
    imported_width: float,
    imported_height: float,
    *,
    dx: float,
) -> tuple[float, int, float]:
    """Return symmetric XY padding whose domain size lands on the Yee lattice."""
    imported_span = max(float(imported_width), float(imported_height))
    snap_allowance = 0.5 * float(dx)
    min_margin = (
        float(MONITOR_TO_PML_SPACING) - float(OUTPUT_MONITOR_OFFSET) + snap_allowance
    )
    min_padding = float(PML_XY) + min_margin
    min_domain = imported_span + 2.0 * min_padding
    cells = max(1, int(math.ceil(min_domain / float(dx) - 1e-12)))
    domain_size = float(cells) * float(dx)
    padding = 0.5 * (domain_size - imported_span)
    return padding, cells, domain_size


def port_mode_geometry(port: dict) -> tuple[float, float, float]:
    width = float(port["width"])
    span = MODE_PLANE_SIZE_SCALE * max(width + 2.0 * PORT_MARGIN, width + 0.1 * µm)
    return span, float(MONITOR_Z_SPAN), float(port["z_center"])


def plot_simulation_overview(
    out_path: Path,
    eps_grid: np.ndarray,
    *,
    width: float,
    height: float,
    depth: float,
    z_focus: float,
    source_plane,
    monitor_planes,
    world_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    eps_grid = np.asarray(eps_grid, dtype=float)
    if eps_grid.ndim == 3:
        z_idx = int(
            np.clip(
                round((z_focus / max(depth, 1e-30)) * (eps_grid.shape[0] - 1)),
                0,
                eps_grid.shape[0] - 1,
            )
        )
        eps_view = eps_grid[z_idx]
    else:
        eps_view = eps_grid

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=220)
    im = ax.imshow(
        eps_view,
        origin="lower",
        extent=[
            world_origin[0] / µm,
            (world_origin[0] + width) / µm,
            world_origin[1] / µm,
            (world_origin[1] + height) / µm,
        ],
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="Permittivity", fraction=0.046, pad=0.04)

    def _plot_plane(line, label, color):
        (x0, y0, _), (x1, y1, _) = line
        ax.plot(
            [
                (x0 + world_origin[0]) / µm,
                (x1 + world_origin[0]) / µm,
            ],
            [
                (y0 + world_origin[1]) / µm,
                (y1 + world_origin[1]) / µm,
            ],
            color=color,
            lw=2.0,
            label=label,
        )

    _plot_plane(source_plane, "source", "white")
    for name, plane in monitor_planes.items():
        _plot_plane(plane, name, "tab:red")

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Simulation overview")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_sparameters_db(out_path: Path, wl_um: np.ndarray, s_matrix: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=220)
    for (out_port, in_port), values in sorted(s_matrix.items()):
        arr = np.asarray(values, dtype=np.complex128)
        ax.plot(
            wl_um,
            20.0 * np.log10(np.maximum(np.abs(arr), 1e-12)),
            lw=2.0,
            label=f"S[{out_port},{in_port}]",
        )
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("S-parameters")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def wave_dominance_db(
    a_plus: np.ndarray, a_minus: np.ndarray, selector: str, mask: np.ndarray
) -> float:
    # Report how cleanly a monitor separates the selected traveling wave from
    # the opposite-going component.
    sel = np.asarray(a_plus if selector == "plus" else a_minus, dtype=np.complex128)
    opp = np.asarray(a_minus if selector == "plus" else a_plus, dtype=np.complex128)
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    p_sel = float(np.mean(np.abs(sel[valid]) ** 2))
    p_opp = float(np.mean(np.abs(opp[valid]) ** 2))
    return 10.0 * np.log10(max(p_sel, 1e-18) / max(p_opp, 1e-18))


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m"


# 1. Import the GDSFactory/PDK component, extrude it to 3D, pad the domain,
# and extend the ports into uniform straight sections.
OUT_DIR.mkdir(parents=True, exist_ok=True)
ensure_ubcpdk_available()
try:
    import micromode

    print(
        "Using micromode "
        f"{importlib.metadata.version('micromode')} from {micromode.__file__} "
        f"(right-handed y basis={micromode_has_right_handed_y_basis()})"
    )
except Exception as exc:
    print(f"Could not report micromode runtime details: {exc}")
use_fixed_micromode_y_projection_convention()
dx, dt = dxdt(
    WL0,
    n_max=N_CORE,
    dims=3,
    safety_factor=0.999,
    points_per_wavelength=PPW,
)
imported_width, imported_height = ubcpdk_gds_layer_span(COMPONENT_NAME, LAYER)
EXTENSION, xy_cells, xy_domain_size = cell_aligned_xy_padding(
    imported_width,
    imported_height,
    dx=dx,
)
XY_MARGIN = EXTENSION - PML_XY
print(
    "XY domain: "
    f"{xy_domain_size / µm:.3f} um = {xy_cells} cells, "
    f"monitor-to-PML target >= {MONITOR_TO_PML_SPACING / µm:.2f} um"
)
try:
    prepared = gdsf.prepare_component(
        COMPONENT_NAME,
        layer=LAYER,
        n_core=N_CORE,
        n_clad=N_CLAD,
        core_thickness=CORE_T,
        clad_below=CLAD_BELOW,
        clad_above=CLAD_ABOVE,
        xy_padding=EXTENSION,
        z_padding=Z_PADDING + PML_Z,
        extension=EXTENSION,
        port_overlap=PORT_OVERLAP,
        use_pdk_layer_stack=USE_PDK_LAYER_STACK,
    )
except (ImportError, ValueError) as exc:
    if not isinstance(exc, ImportError) and (
        "Could not resolve gdsfactory/PDK component" not in str(exc)
    ):
        raise
    raise RuntimeError(
        f"Could not load the UBC PDK crossing '{COMPONENT_NAME}'. Install the "
        "compatible UBC PDK in the same Python environment, for example with "
        f"`uv pip install --python {sys.executable} '{UBC_PDK_REQUIREMENT}'`, "
        "then rerun this example."
    ) from exc
print(f"Loaded crossing component: {prepared['component_label']}")
design, ports = prepared["design"], prepared["ports"]
world_origin = tuple(float(v) for v in prepared.get("world_origin", (0.0, 0.0, 0.0)))
source_port, output_ports = "o1", ["o2", "o3", "o4"]
port_names = (source_port, *output_ports)
grid = design.rasterize(resolution=dx)
freqs = np.linspace(
    LIGHT_SPEED / WL_MAX, LIGHT_SPEED / WL_MIN, NUM_FREQS, dtype=np.float32
)
wl_um = LIGHT_SPEED / freqs / µm

# 2. Place the source and monitors directly from the imported port metadata.
# Keep the Meep-matched source plane at 0.1 um inside the source port, but use
# a slightly deeper 0.5 um source-monitor plane from the local BeamZ S11 sweep.
# Output monitors are also moved farther inward so the weak cross ports see a
# cleaner outgoing guided mode before projection.
src = ports[source_port]
source_direction = src["direction"]
source_span, z_span, source_z_center = port_mode_geometry(src)
overview_z_focus = source_z_center
source_plane = port_plane(
    src,
    span=source_span,
    z_span=z_span,
    z_center=source_z_center,
    offset=SOURCE_PORT_OFFSET,
)
source_center = line_center(source_plane)
monitor_offsets = {source_port: SOURCE_PORT_OFFSET + DISTANCE_SOURCE_TO_MONITORS}
monitor_planes = {
    source_port: port_plane(
        src,
        span=source_span,
        z_span=z_span,
        z_center=source_z_center,
        offset=monitor_offsets[source_port],
    )
}
for port_name in output_ports:
    port = ports[port_name]
    span, monitor_z_span, z_center = port_mode_geometry(port)
    monitor_offsets[port_name] = OUTPUT_MONITOR_OFFSET
    monitor_planes[port_name] = port_plane(
        port,
        span=span,
        z_span=monitor_z_span,
        z_center=z_center,
        offset=monitor_offsets[port_name],
    )
crossing_center = (
    0.5 * float(design.width),
    0.5 * float(design.height),
    float(source_z_center),
)
print(
    "Source alignment to crossing center (um): "
    f"dx={(source_center[0] - crossing_center[0]) / µm:.6e}, "
    f"dy={(source_center[1] - crossing_center[1]) / µm:.6e}"
)
y_output_ports = [
    name for name in output_ports if str(ports[name]["direction"]).endswith("y")
]
if len(y_output_ports) == 2:
    top_port = next(
        name for name in y_output_ports if str(ports[name]["direction"]) == "-y"
    )
    bottom_port = next(
        name for name in y_output_ports if str(ports[name]["direction"]) == "+y"
    )
    top_region = snap_plane_region(
        start=monitor_planes[top_port][0],
        end=monitor_planes[top_port][1],
        plane_normal="y",
        size=None,
        dx=dx,
        dy=dx,
        dz=dx,
        shape=tuple(np.asarray(grid.permittivity).shape),
    )
    monitor_planes[top_port] = (top_region.start, top_region.end)
    monitor_planes[bottom_port] = reflect_plane_y(
        monitor_planes[top_port],
        mirror_y=crossing_center[1],
    )
    y_pair_error = mirror_error_y(
        monitor_planes[top_port],
        monitor_planes[bottom_port],
        mirror_y=crossing_center[1],
    )
    if y_pair_error > 1e-15:
        raise RuntimeError(
            f"Y-port monitor planes are not exactly mirrored: "
            f"error={y_pair_error / µm:.6e} um."
        )
    print(f"Y-port monitor mirror error: {y_pair_error / µm:.6e} um")
print("Plane positions relative to imported ports (um):")
print(f"  source: {SOURCE_PORT_OFFSET / µm:.2f}")
for port_name in port_names:
    actual_offset = signed_port_plane_offset(
        monitor_planes[port_name], ports[port_name]
    )
    print(
        f"  {port_name}: requested={monitor_offsets[port_name] / µm:.2f}, "
        f"final={actual_offset / µm:.6f}"
    )
print("Monitor clearances to inner XY PML boundary (um):")
for port_name in port_names:
    normal_clearance = pml_clearance_for_port(
        monitor_planes[port_name],
        ports[port_name],
        width=design.width,
        height=design.height,
        pml_xy=PML_XY,
    )
    clearances = pml_clearances_xy(
        monitor_planes[port_name],
        width=design.width,
        height=design.height,
        pml_xy=PML_XY,
    )
    print(
        f"  {port_name}: normal={normal_clearance / µm:.6f}, "
        f"top={clearances['top'] / µm:.6f}, "
        f"bottom={clearances['bottom'] / µm:.6f}"
    )
    if port_name in output_ports and normal_clearance < MONITOR_TO_PML_SPACING - 1e-12:
        raise RuntimeError(
            f"Monitor {port_name} is only {normal_clearance / µm:.3f} um from "
            f"the inner PML boundary; expected at least "
            f"{MONITOR_TO_PML_SPACING / µm:.3f} um."
        )
    if min(clearances["top"], clearances["bottom"]) < MONITOR_TO_PML_SPACING - 1e-12:
        raise RuntimeError(
            f"Monitor {port_name} has top/bottom PML clearance below "
            f"{MONITOR_TO_PML_SPACING / µm:.3f} um: "
            f"top={clearances['top'] / µm:.3f} um, "
            f"bottom={clearances['bottom'] / µm:.3f} um."
        )
runtime_output_distance_um = 0.0
for port_name in output_ports:
    c_out = line_center(monitor_planes[port_name])
    runtime_output_distance_um = max(
        runtime_output_distance_um,
        float(np.hypot(c_out[0] - source_center[0], c_out[1] - source_center[1])) / µm,
    )

# 3. Generate the broadband Gaussian pulse and build the source / DFT monitors.
pulse = gaussian_band_pulse(
    freqs,
    carrier_frequency=LIGHT_SPEED / WL0,
    dt=dt,
    run_after_sources_uoc=RUN_AFTER_SOURCES_UOC,
    max_output_distance_um=runtime_output_distance_um,
)
source = ModeSource(
    grid=grid,
    center=source_center,
    width=source_span,
    height=z_span,
    wavelength=WL0,
    pol="te",
    signal=pulse.signal,
    direction=source_direction,
)
source.initialize(grid.permittivity, dx, dt=dt)
monitor_cfg = dict(
    record_fields=False,
    dft_enabled=True,
    dft_frequencies=freqs,
    dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    dft_window="none",
    dft_record_every_step=True,
)
monitors = [
    ModeMonitor(
        start=monitor_planes[p][0],
        end=monitor_planes[p][1],
        name=p,
        direction=ports[p]["direction"],
        polarization="te",
        reference_monitor="o1_ref" if p == source_port else None,
        **monitor_cfg,
    )
    for p in port_names
]
mode_monitors = {m.name: m for m in monitors}


def crossing_port_spec(name: str) -> PortSpec:
    """Build explicit modal wave selectors for this crossing benchmark."""
    monitor = mode_monitors[name]
    direction = str(ports[name]["direction"])
    if direction.endswith("y") and micromode_has_right_handed_y_basis():
        incident_wave = "minus"
        scattered_wave = "plus"
    else:
        port = monitor.to_port()
        incident_wave = port.incident_wave
        scattered_wave = port.scattered_wave
    return PortSpec(
        name=name,
        monitor_name=name,
        direction=direction,
        polarization="te",
        mode_index=int(getattr(monitor, "mode_index", 0)),
        reference_monitor="o1_ref" if name == source_port else None,
        incident_wave=incident_wave,
        scattered_wave=scattered_wave,
    )


modal_ports = [crossing_port_spec(name) for name in port_names]
port_specs = {spec.name: spec for spec in modal_ports}
print("Port wave selectors:")
for name in port_names:
    port = port_specs[name]
    print(
        f"  {name}: direction={port.direction}, "
        f"incident={port.incident_wave}, scattered={port.scattered_wave}"
    )
reference_monitor = Monitor(
    start=source_plane[0],
    end=source_plane[1],
    name="o1_ref",
    **monitor_cfg,
)
all_monitors = [*monitors, reference_monitor]
decay_monitors = monitors

# 4. Feed the design, source, monitors, boundaries, and time array into the
# simulation object.
sim = Simulation(
    design=design,
    sources=[source],
    monitors=all_monitors,
    boundaries=[
        PML(
            edges=["left", "right", "top", "bottom"],
            thickness=PML_XY,
            formulation=PML_FORMULATION,
        ),
        PML(
            edges=["front", "back"],
            thickness=PML_Z,
            formulation=PML_FORMULATION,
        ),
    ],
    time=pulse.time,
    resolution=dx,
)
sim.show()

# 5. Save a compact overview plot of the rasterized structure with the source
# and monitor planes overlaid.
print(
    f"Workload: grid={grid.permittivity.shape}, voxels={int(np.prod(np.asarray(grid.permittivity).shape)):,}, updates~{int(np.prod(np.asarray(grid.permittivity).shape)) * len(pulse.time):.3e}"
)
estimated_updates = float(
    int(np.prod(np.asarray(grid.permittivity).shape)) * len(pulse.time)
)
print(
    "Estimated runtime: "
    f"100 MCUPS ~ {format_duration(estimated_updates / 100e6)}, "
    f"250 MCUPS ~ {format_duration(estimated_updates / 250e6)}, "
    f"500 MCUPS ~ {format_duration(estimated_updates / 500e6)}"
)
plot_simulation_overview(
    OUT_DIR / "beamz_crossing_overview.png",
    np.asarray(grid.permittivity, dtype=float),
    width=design.width,
    height=design.height,
    depth=design.depth,
    z_focus=overview_z_focus,
    source_plane=source_plane,
    monitor_planes=monitor_planes,
    world_origin=world_origin,
)
print(f"Saved overview figure: {OUT_DIR / 'beamz_crossing_overview.png'}")

# 6. Run in compiled chunks until the monitor power has decayed sufficiently
# after the pulse leaves the device.
wall_t0 = pytime.perf_counter()
executed_steps = sim.run_compiled_until_decay(
    decay_monitors,
    min_time_s=pulse.source_end_time + pulse.tail_time,
    lookback_records=LOOKBACK_RECORDS,
    decay_ratio=DECAY_RATIO,
    progress=True,
)
wall_s = max(pytime.perf_counter() - wall_t0, 1e-12)
num_voxels = int(np.prod(np.asarray(grid.permittivity).shape))
print(
    "Simulation stats: "
    f"steps={executed_steps}, voxels={num_voxels:,}, sim_time={(executed_steps - 1) * dt * 1e15:.2f}fs, "
    f"wall={wall_s:.2f}s, step_rate={executed_steps / wall_s:.2f} steps/s, MCUPS={num_voxels * executed_steps / wall_s / 1e6:.2f}"
)

# 7. Extract the broadband S-matrix with explicit modal wave selectors. The y
# ports need the corrected micromode y-basis convention when using 0.1.0a4+.
result = sim.get_S_matrix_modal_dft(
    source_port=port_specs[source_port],
    ports=modal_ports,
    output_ports=modal_ports,
    frequencies=freqs,
    as_sax=False,
    return_diagnostics=True,
    min_incident_db=-45.0,
)
i0 = int(np.argmin(np.abs(wl_um - WL0 / µm)))
valid = np.asarray(result["diagnostics"]["valid_mask"], dtype=bool)
source_waves = result["diagnostics"]["waves"]["o1"]
source_dom = wave_dominance_db(
    source_waves["a_plus"],
    source_waves["a_minus"],
    port_specs[source_port].incident_wave,
    valid,
)
print(f"o1 wave dominance: {source_dom:.2f} dB")
src_cond = np.asarray(
    result["diagnostics"]["condition_numbers"]["o1"]["monitor"], dtype=float
)
src_ref_cond = np.asarray(
    result["diagnostics"]["condition_numbers"]["o1"]["reference"], dtype=float
)
if src_cond.size and src_ref_cond.size:
    print(
        "o1 projection conditioning "
        f"@ {wl_um[i0]:.4f}um: main={src_cond[i0]:.2e}, ref={src_ref_cond[i0]:.2e}"
    )

s_matrix = {
    (port_name, source_port): np.asarray(
        result["s_matrix"][(port_name, source_port)], dtype=np.complex128
    )
    for port_name in port_names
}
for port_name in output_ports:
    waves = result["diagnostics"]["waves"][port_name]
    dom = wave_dominance_db(
        waves["a_plus"],
        waves["a_minus"],
        port_specs[port_name].scattered_wave,
        valid,
    )
    print(
        f"{port_name} at imported port plane offset {monitor_offsets[port_name] / µm:.2f} um "
        f"(dominance={dom:.2f} dB)"
    )
for port_name in port_names:
    mag = abs(s_matrix[(port_name, "o1")][i0])
    print(
        f"S[{port_name},o1] @ {wl_um[i0]:.4f}um: {20.0 * np.log10(max(mag, 1e-12)):.2f} dB"
    )

# Refresh the overview after the run so the latest generated artifact matches
# the final S-matrix extraction path.
plot_simulation_overview(
    OUT_DIR / "beamz_crossing_overview.png",
    np.asarray(grid.permittivity, dtype=float),
    width=design.width,
    height=design.height,
    depth=design.depth,
    z_focus=overview_z_focus,
    source_plane=source_plane,
    monitor_planes=monitor_planes,
    world_origin=world_origin,
)
print(f"Updated overview figure: {OUT_DIR / 'beamz_crossing_overview.png'}")

# 8. Save the final S-parameter plot using the same helper style as the full
# example so regression checks remain straightforward.
plot_sparameters_db(OUT_DIR / "beamz_crossing_sparams.png", wl_um, s_matrix)
