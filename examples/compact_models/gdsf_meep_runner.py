"""MEEP runner script generation.

Generates a self-contained Python script that runs on the cloud MEEP instance.
The script reads a JSON config + GDS file, builds the MEEP simulation, runs it,
and outputs S-parameters as CSV.
"""

from __future__ import annotations

_MEEP_RUNNER_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated MEEP runner script.

Reads simulation config from JSON + layout from GDS, builds geometry,
runs FDTD, and extracts S-parameters via mode decomposition.

Cloud dependencies: meep, gdsfactory, numpy, scipy, shapely
"""

import csv
import cmath
import json
import logging
import math
import os
import sys
import time

import meep as mp
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("meep_runner")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path):
    """Load simulation config from JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def build_materials(config):
    """Build MEEP material objects from config."""
    materials = {}
    for name, props in config["materials"].items():
        n = props["refractive_index"]
        k = props["extinction_coeff"]
        if k > 0:
            materials[name] = mp.Medium(index=n, D_conductivity=2 * cmath.pi * k / n)
        else:
            materials[name] = mp.Medium(index=n)
    return materials


# ---------------------------------------------------------------------------
# Geometry: GDS + layer_stack -> mp.Prism objects
# ---------------------------------------------------------------------------

def load_gds_component(gds_filename):
    """Load GDS file as a gdsfactory Component."""
    import gdsfactory as gf
    gf.gpdk.PDK.activate()
    return gf.import_gds(gds_filename)


def _klayout_poly_to_coords(poly_obj, dbu):
    """Convert a KLayout polygon object to a list of (x, y) coordinate tuples.

    Args:
        poly_obj: KLayout PolygonWithProperties or similar object
        dbu: Database unit scaling factor (converts integer coords to microns)

    Returns:
        list of (float, float) coordinate tuples
    """
    coords = []
    for pt in poly_obj.each_point_hull():
        coords.append((pt.x * dbu, pt.y * dbu))
    return coords


def extract_layer_polygons(component, gds_layer, simplify_tol=0.0):
    """Extract merged Shapely polygons for a GDS layer from a component.

    Uses gdsfactory / KLayout to get polygons for the given (layer, datatype)
    tuple, then converts to Shapely with hole support.

    Handles both old gdsfactory (dict keys are (layer, datatype) tuples,
    values are numpy arrays) and new gdsfactory 9.x (dict keys are integer
    KLayout layer indices, values are PolygonWithProperties objects).

    Args:
        component: gdsfactory Component
        gds_layer: [layer_number, datatype] list
        simplify_tol: Shapely simplification tolerance in um (0=off)

    Returns:
        list of Shapely Polygon objects
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    layer_tuple = tuple(gds_layer)
    raw = component.get_polygons(layers=(layer_tuple,), merge=True)

    # Collect all polygon objects from the dict, regardless of key type.
    # Old gdsfactory: keys are (layer, datatype) tuples, values are ndarray lists
    # New gdsfactory 9.x: keys are integer layer indices, values are
    # PolygonWithProperties lists
    all_objects = []
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                all_objects.extend(v)
            else:
                all_objects.append(v)
    elif raw is not None:
        all_objects = list(raw) if hasattr(raw, '__iter__') else [raw]

    dbu = getattr(getattr(component, 'kcl', None), 'dbu', 0.001)

    shapely_polygons = []
    for obj in all_objects:
        try:
            # KLayout PolygonWithProperties — convert via each_point_hull
            if hasattr(obj, 'each_point_hull'):
                coords = _klayout_poly_to_coords(obj, dbu)
            # Numpy array — legacy path
            elif hasattr(obj, 'shape') and len(obj) >= 3:
                coords = [(float(p[0]), float(p[1])) for p in obj]
            else:
                continue

            if len(coords) < 3:
                continue
            poly = Polygon(coords)
            if poly.is_valid and not poly.is_empty:
                shapely_polygons.append(poly)
        except Exception:
            continue

    if not shapely_polygons:
        return []

    merged = unary_union(shapely_polygons)
    if simplify_tol > 0:
        merged = merged.simplify(simplify_tol, preserve_topology=True)
    if hasattr(merged, "geoms"):
        return list(merged.geoms)
    return [merged] if not merged.is_empty else []


def triangulate_polygon_with_holes(polygon):
    """Triangulate a polygon with holes using Delaunay triangulation.

    Returns list of triangles as coordinate lists, each triangle being
    [[x1,y1], [x2,y2], [x3,y3]].
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import Point

    all_points = list(polygon.exterior.coords[:-1])
    for interior in polygon.interiors:
        all_points.extend(list(interior.coords[:-1]))

    if len(all_points) < 3:
        return [list(polygon.exterior.coords[:-1])]

    points_2d = np.array(all_points)
    tri = Delaunay(points_2d)

    triangles = []
    for simplex in tri.simplices:
        triangle_pts = points_2d[simplex]
        centroid = np.mean(triangle_pts, axis=0)
        if polygon.contains(Point(centroid)):
            triangles.append(triangle_pts.tolist())

    return triangles if triangles else [list(polygon.exterior.coords[:-1])]


def build_background_slabs(config, materials):
    """Build background mp.Block slabs from dielectric entries.

    These infinite-XY slabs fill the simulation cell at each z-range with
    the correct cladding/substrate material.  They must come FIRST in the
    geometry list so that patterned prisms (added later) take precedence.
    """
    slabs = []
    for diel in sorted(config["dielectrics"], key=lambda d: d["zmin"]):
        mat = materials.get(diel["material"])
        if mat is None:
            continue
        zmin = diel["zmin"]
        zmax = diel["zmax"]
        thickness = zmax - zmin
        if thickness <= 0:
            continue
        block = mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, thickness),
            center=mp.Vector3(0, 0, (zmin + zmax) / 2),
            material=mat,
        )
        slabs.append(block)
    return slabs


def build_symmetries(config):
    """Build MEEP symmetry objects from config."""
    direction_map = {"X": mp.X, "Y": mp.Y, "Z": mp.Z}
    symmetries = []
    for sym in config["symmetries"]:
        symmetries.append(
            mp.Mirror(direction_map[sym["direction"]], phase=sym["phase"])
        )
    return symmetries


# Map string component names to MEEP field components
_COMPONENT_MAP = {
    "Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez,
    "Hx": mp.Hx, "Hy": mp.Hy, "Hz": mp.Hz,
}


def _make_time_cap(cap):
    """Wrap a numeric time cap as a callable for until_after_sources lists.

    When until_after_sources receives a list, every element must be callable.
    This wraps a float (time units) into a function that returns True once
    ``cap`` time units have elapsed since the first call (i.e. after sources
    turn off).
    """
    _t0 = [None]
    def _check(sim_obj):
        if _t0[0] is None:
            _t0[0] = sim_obj.meep_time()
        return (sim_obj.meep_time() - _t0[0]) >= cap
    return _check


def _make_wall_time_cap(wall_seconds):
    """Wrap a wall-clock time limit as a callable for until_after_sources lists.

    Returns True once ``wall_seconds`` of real (wall) time have elapsed
    since the first call.  This provides a safety net orthogonal to any
    sim-time stopping condition.
    """
    _deadline = [None]
    def _check(sim_obj):
        if _deadline[0] is None:
            _deadline[0] = time.time() + wall_seconds
        return time.time() >= _deadline[0]
    return _check


def resolve_decay_monitor_point(config):
    """Return center mp.Vector3 for decay monitoring.

    Uses the port named by stopping.decay_monitor_port if set,
    otherwise picks the first non-source port.  Falls back to
    the first port if all are sources.
    """
    stopping = config["stopping"]
    target_name = stopping.get("decay_monitor_port")
    ports = config["ports"]

    if target_name:
        for p in ports:
            if p["name"] == target_name:
                return mp.Vector3(*p["center"])

    # Fallback: first non-source port, or first port
    for p in ports:
        if not p["is_source"]:
            return mp.Vector3(*p["center"])
    if ports:
        return mp.Vector3(*ports[0]["center"])

    return mp.Vector3(0, 0, 0)


def build_geometry(config, materials):
    """Build MEEP geometry from GDS file + layer stack config.

    For each layer in layer_stack:
      1. Extract polygons from the GDS for that layer's gds_layer
      2. Extrude to 3D as mp.Prism with correct z-range and material
      3. Handle polygon holes via Delaunay triangulation
    """
    gds_filename = config["gds_filename"]
    component = load_gds_component(gds_filename)

    accuracy = config["accuracy"]
    simplify_tol = accuracy["simplify_tol"]

    geometry = []
    total_vertices = 0

    for layer_entry in config["layer_stack"]:
        material_name = layer_entry["material"]
        mat = materials.get(material_name, mp.Medium())
        zmin = layer_entry["zmin"]
        zmax = layer_entry["zmax"]
        height = zmax - zmin
        gds_layer = layer_entry["gds_layer"]
        sidewall_angle_deg = layer_entry["sidewall_angle"]
        sw_rad = math.radians(sidewall_angle_deg) if sidewall_angle_deg else 0

        if height <= 0:
            continue

        polygons = extract_layer_polygons(
            component, gds_layer, simplify_tol=simplify_tol,
        )

        for polygon in polygons:
            if polygon.is_empty or not polygon.is_valid:
                continue

            if hasattr(polygon, "interiors") and polygon.interiors:
                # Polygon has holes — triangulate
                triangles = triangulate_polygon_with_holes(polygon)
                for tri_coords in triangles:
                    vertices = [mp.Vector3(p[0], p[1], zmin) for p in tri_coords]
                    total_vertices += len(vertices)
                    prism = mp.Prism(
                        vertices=vertices,
                        height=height,
                        material=mat,
                        sidewall_angle=sw_rad,
                    )
                    geometry.append(prism)
            else:
                # Simple polygon — direct extrusion
                coords = list(polygon.exterior.coords[:-1])
                vertices = [mp.Vector3(p[0], p[1], zmin) for p in coords]
                total_vertices += len(vertices)
                prism = mp.Prism(
                    vertices=vertices,
                    height=height,
                    material=mat,
                    sidewall_angle=sw_rad,
                )
                geometry.append(prism)

    logger.info("Total vertices across all prisms: %d", total_vertices)
    return geometry, component


# ---------------------------------------------------------------------------
# Sources and monitors
# ---------------------------------------------------------------------------

def get_port_z_span(config):
    """Get z-span for ports from layer stack."""
    zmin = min(l["zmin"] for l in config["layer_stack"])
    zmax = max(l["zmax"] for l in config["layer_stack"])
    return zmax - zmin


def build_sources(config):
    """Build MEEP source from config port data.

    The source is offset from the port center by ``source_port_offset``
    along the propagation direction (into the device).  This separates
    the soft source from the port monitor so eigenmode coefficients
    measure the true incident amplitude rather than half of it.
    """
    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    df = fdtd["df"]
    fwidth = config["source"]["fwidth"]
    z_span = get_port_z_span(config)
    port_margin = config["domain"]["port_margin"]
    source_port_offset = config["domain"].get("source_port_offset", 0.1)

    sources = []
    for port in config["ports"]:
        if not port["is_source"]:
            continue

        # Offset source center into the device along propagation direction
        center_list = list(port["center"])
        normal_axis = port["normal_axis"]
        direction = port["direction"]
        if direction == "+":
            center_list[normal_axis] += source_port_offset
        else:
            center_list[normal_axis] -= source_port_offset
        center = mp.Vector3(*center_list)

        size = [0, 0, 0]
        transverse_axis = 1 - normal_axis
        size[transverse_axis] = port["width"] + 2 * port_margin
        size[2] = z_span

        # Propagation axis
        prop_axis = mp.X if normal_axis == 0 else mp.Y

        # eig_kpoint points INTO the device (direction of incoming mode)
        if normal_axis == 0:
            kpoint = mp.Vector3(x=1) if direction == "+" else mp.Vector3(x=-1)
        else:
            kpoint = mp.Vector3(y=1) if direction == "+" else mp.Vector3(y=-1)

        eig_src = mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            center=center,
            size=mp.Vector3(*size),
            eig_band=1,
            direction=prop_axis,
            eig_kpoint=kpoint,
            eig_match_freq=True,
        )
        sources.append(eig_src)

    return sources


def build_monitors(config, sim):
    """Build mode monitors at all ports and return flux regions.

    The source-port monitor is offset further into the device (past
    the source) by ``source_port_offset + distance_source_to_monitors``
    so the forward-going mode from the source passes through it at
    full amplitude.  This matches the gplugins approach.
    """
    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    df = fdtd["df"]
    nfreq = fdtd["num_freqs"]
    z_span = get_port_z_span(config)
    port_margin = config["domain"]["port_margin"]
    source_port_offset = config["domain"].get("source_port_offset", 0.1)
    distance_source_to_monitors = config["domain"].get(
        "distance_source_to_monitors", 0.2
    )

    monitors = {}
    for port in config["ports"]:
        center_list = list(port["center"])
        normal_axis = port["normal_axis"]
        direction = port["direction"]

        # All monitors shift inward from port center (matches gplugins):
        #   source-port monitor: source_port_offset + distance_source_to_monitors
        #   non-source monitors: source_port_offset
        if port["is_source"]:
            offset = source_port_offset + distance_source_to_monitors
        else:
            offset = source_port_offset

        if offset > 0:
            if direction == "+":
                center_list[normal_axis] += offset
            else:
                center_list[normal_axis] -= offset

        center = mp.Vector3(*center_list)

        size = [0, 0, 0]
        transverse_axis = 1 - normal_axis
        size[transverse_axis] = port["width"] + 2 * port_margin
        size[2] = z_span

        flux = sim.add_mode_monitor(
            fcen, df, nfreq,
            mp.ModeRegion(center=center, size=mp.Vector3(*size)),
        )
        monitors[port["name"]] = flux

    return monitors


# ---------------------------------------------------------------------------
# S-parameter extraction
# ---------------------------------------------------------------------------

def _port_kpoint(port):
    """Return positive-axis kpoint vector for eigenmode decomposition.

    Always points along +normal_axis so that:
      alpha[band, freq, 0] = forward (+axis) coefficient
      alpha[band, freq, 1] = backward (-axis) coefficient
    """
    if port["normal_axis"] == 0:
        return mp.Vector3(x=1)
    return mp.Vector3(y=1)


def extract_s_params(config, sim, monitors):
    """Extract S-parameters via mode decomposition.

    Uses MEEP eigenmode coefficients with explicit kpoint_func to anchor
    the forward/backward convention:
      alpha[band, freq, 0] = forward (+normal_axis) coefficient
      alpha[band, freq, 1] = backward (-normal_axis) coefficient

    Port "direction" field = direction of incoming mode along normal_axis:
      "+" -> incoming goes +normal, outgoing (reflected/transmitted) goes -normal
      "-" -> incoming goes -normal, outgoing (reflected/transmitted) goes +normal

    Returns:
        (s_params, debug_data) tuple where debug_data contains eigenmode
        diagnostics for post-run analysis.
    """
    port_names = [p["name"] for p in config["ports"]]
    ports = {p["name"]: p for p in config["ports"]}
    source_port = None
    for p in config["ports"]:
        if p["is_source"]:
            source_port = p["name"]
            break

    if not source_port:
        return {}, {}

    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    nfreq = fdtd["num_freqs"]
    df = fdtd["df"]
    freqs = np.linspace(fcen - df / 2, fcen + df / 2, nfreq)

    debug_data = {
        "eigenmode_info": {},
        "raw_coefficients": {},
        "incident_coefficients": {},
    }

    def _incoming_idx(direction):
        """Alpha index for the incoming (incident) mode at a port."""
        return 0 if direction == "+" else 1

    def _outgoing_idx(direction):
        """Alpha index for the outgoing (reflected/transmitted) mode at a port."""
        return 1 if direction == "+" else 0

    def _collect_eigenmode_debug(port_name, ob, freqs):
        """Collect eigenmode diagnostics from coefficients object."""
        info = {"band": 1}
        try:
            kdom = ob.kdom
            kdom_list = [[float(kdom[i].x), float(kdom[i].y), float(kdom[i].z)]
                         for i in range(len(kdom))]
            info["kdom"] = kdom_list
            info["n_eff"] = [float(np.linalg.norm(kdom_list[i])) / float(freqs[i])
                             if float(freqs[i]) > 0 else 0.0
                             for i in range(min(len(kdom_list), len(freqs)))]
        except Exception:
            info["kdom"] = []
            info["n_eff"] = []
        try:
            cg = ob.cg
            info["group_velocity"] = [float(cg[i]) for i in range(len(cg))]
        except Exception:
            info["group_velocity"] = []
        return info

    # Get incident coefficient at source port for normalization
    src_dir = ports[source_port]["direction"]
    src_kp = _port_kpoint(ports[source_port])
    src_ob = sim.get_eigenmode_coefficients(
        monitors[source_port], [1], eig_parity=mp.NO_PARITY,
        kpoint_func=lambda f, n, kp=src_kp: kp,
    )
    incident_coeffs = src_ob.alpha[0, :, _incoming_idx(src_dir)]

    debug_data["eigenmode_info"][source_port] = _collect_eigenmode_debug(
        source_port, src_ob, freqs)
    debug_data["incident_coefficients"] = {
        "port": source_port,
        "magnitudes": [float(abs(c)) for c in incident_coeffs],
    }
    debug_data["raw_coefficients"][source_port] = {
        "forward_mag": [float(abs(src_ob.alpha[0, i, 0])) for i in range(len(freqs))],
        "backward_mag": [float(abs(src_ob.alpha[0, i, 1])) for i in range(len(freqs))],
    }

    s_params = {}

    for i, port_i in enumerate(port_names):
        for j, port_j in enumerate(port_names):
            if port_j != source_port:
                continue

            s_name = f"S{i+1}{j+1}"

            port_kp = _port_kpoint(ports[port_i])
            ob = sim.get_eigenmode_coefficients(
                monitors[port_i], [1], eig_parity=mp.NO_PARITY,
                kpoint_func=lambda f, n, kp=port_kp: kp,
            )

            # Collect debug info (skip source port — already collected)
            if port_i != source_port:
                debug_data["eigenmode_info"][port_i] = _collect_eigenmode_debug(
                    port_i, ob, freqs)
                nf = len(freqs)
                debug_data["raw_coefficients"][port_i] = {
                    "forward_mag": [
                        float(abs(ob.alpha[0, k, 0])) for k in range(nf)
                    ],
                    "backward_mag": [
                        float(abs(ob.alpha[0, k, 1])) for k in range(nf)
                    ],
                }

            # Outgoing direction: reflected at source port, transmitted at output ports
            port_dir = ports[port_i]["direction"]
            alpha = ob.alpha[0, :, _outgoing_idx(port_dir)]

            s_params[s_name] = alpha / incident_coeffs

    # Power conservation: sum |Sij|^2 per frequency
    power_conservation = np.zeros(len(freqs))
    for s_name, s_vals in s_params.items():
        power_conservation += np.abs(s_vals) ** 2
    debug_data["power_conservation"] = [float(p) for p in power_conservation]

    return s_params, debug_data


def save_results(config, s_params, output_path="s_parameters.csv"):
    """Save S-parameters to CSV file.  Only rank 0 writes."""
    if not mp.am_master():
        return
    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    df = fdtd["df"]
    nfreq = fdtd["num_freqs"]

    freqs = np.linspace(fcen - df / 2, fcen + df / 2, nfreq)
    wavelengths = 1.0 / freqs

    fieldnames = ["wavelength"]
    for name in sorted(s_params.keys()):
        fieldnames.extend([f"{name}_mag", f"{name}_phase"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, wl in enumerate(wavelengths):
            row = {"wavelength": f"{wl:.6f}"}
            for name in sorted(s_params.keys()):
                val = s_params[name][i]
                row[f"{name}_mag"] = f"{abs(val):.6f}"
                row[f"{name}_phase"] = f"{cmath.phase(val) * 180 / cmath.pi:.4f}"
            writer.writerow(row)

    logger.info("S-parameters saved to %s", output_path)


def save_debug_log(config, s_params, debug_data, wall_seconds=0.0,
                   output_path="meep_debug.json"):
    """Save eigenmode diagnostics as JSON for post-run analysis.  Only rank 0 writes."""
    if not mp.am_master():
        return
    fdtd = config["fdtd"]
    domain = config["domain"]
    resolution = config["resolution"]["pixels_per_um"]
    stopping = config["stopping"]

    meep_time = 0.0
    timesteps = 0
    try:
        # These are set by caller if available
        meep_time = debug_data.get("_meep_time", 0.0)
        timesteps = debug_data.get("_timesteps", 0)
    except Exception:
        pass

    log = {
        "metadata": {
            "resolution": resolution,
            "cell_size": debug_data.get("_cell_size", []),
            "meep_time": meep_time,
            "timesteps": timesteps,
            "wall_seconds": wall_seconds,
            "stopping_mode": stopping["mode"],
        },
        "eigenmode_info": debug_data.get("eigenmode_info", {}),
        "incident_coefficients": debug_data.get("incident_coefficients", {}),
        "raw_coefficients": debug_data.get("raw_coefficients", {}),
        "power_conservation": debug_data.get("power_conservation", []),
    }

    with open(output_path, "w") as f:
        json.dump(log, f, indent=2)
    logger.info("Debug log saved to %s", output_path)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def save_geometry_diagnostics(sim, config, cell_center):
    """Save geometry cross-section plots showing epsilon, sources, monitors, PML."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping geometry diagnostics")
        return
    if not mp.am_master():
        # plot2D is collective — all ranks call it, only master saves
        pass

    domain = config["domain"]
    dpml = domain["dpml"]
    z_min = min(l["zmin"] for l in config["layer_stack"])
    z_max = max(l["zmax"] for l in config["layer_stack"])
    z_core = (z_min + z_max) / 2

    # XY cross-section at z=core center
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        xy_plane = mp.Volume(
            center=mp.Vector3(cell_center.x, cell_center.y, z_core),
            size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
        )
        sim.plot2D(ax=ax, output_plane=xy_plane)
        ax.set_title(f"XY cross-section at z={z_core:.3f} um (core center)")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.tight_layout()
        if mp.am_master():
            fig.savefig("meep_geometry_xy.png", dpi=150)
            logger.info("Saved meep_geometry_xy.png")
        plt.close(fig)
    except Exception as e:
        logger.warning("XY geometry plot failed: %s", e)

    # For 3D sims: XZ and YZ cross-sections
    if sim.cell_size.z > 0:
        # XZ at y=center
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            xz_plane = mp.Volume(
                center=mp.Vector3(cell_center.x, cell_center.y, cell_center.z),
                size=mp.Vector3(sim.cell_size.x, 0, sim.cell_size.z),
            )
            sim.plot2D(ax=ax, output_plane=xz_plane)
            ax.set_title(f"XZ cross-section at y={cell_center.y:.3f} um")
            ax.set_xlabel("x (um)")
            ax.set_ylabel("z (um)")
            fig.tight_layout()
            if mp.am_master():
                fig.savefig("meep_geometry_xz.png", dpi=150)
                logger.info("Saved meep_geometry_xz.png")
            plt.close(fig)
        except Exception as e:
            logger.warning("XZ geometry plot failed: %s", e)

        # YZ at x=center
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            yz_plane = mp.Volume(
                center=mp.Vector3(cell_center.x, cell_center.y, cell_center.z),
                size=mp.Vector3(0, sim.cell_size.y, sim.cell_size.z),
            )
            sim.plot2D(ax=ax, output_plane=yz_plane)
            ax.set_title(f"YZ cross-section at x={cell_center.x:.3f} um")
            ax.set_xlabel("y (um)")
            ax.set_ylabel("z (um)")
            fig.tight_layout()
            if mp.am_master():
                fig.savefig("meep_geometry_yz.png", dpi=150)
                logger.info("Saved meep_geometry_yz.png")
            plt.close(fig)
        except Exception as e:
            logger.warning("YZ geometry plot failed: %s", e)


def save_field_snapshot(sim, config, cell_center):
    """Save post-run field snapshot (Ey overlaid on epsilon)."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping field snapshot")
        return

    z_min = min(l["zmin"] for l in config["layer_stack"])
    z_max = max(l["zmax"] for l in config["layer_stack"])
    z_core = (z_min + z_max) / 2

    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        xy_plane = mp.Volume(
            center=mp.Vector3(cell_center.x, cell_center.y, z_core),
            size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
        )
        sim.plot2D(ax=ax, output_plane=xy_plane, fields=mp.Ey)
        ax.set_title(f"Ey field at z={z_core:.3f} um (post-run)")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.tight_layout()
        if mp.am_master():
            fig.savefig("meep_fields_xy.png", dpi=150)
            logger.info("Saved meep_fields_xy.png")
        plt.close(fig)
    except Exception as e:
        logger.warning("Field snapshot failed: %s", e)


def save_animation_field(sim, xy_plane, frame_counter):
    """Save raw 2D field data during time-stepping (no plotting).

    Saves a compressed .npz with the Ey field array and timestamp.
    Rendering with a globally fixed colorbar happens after sim.run().

    Returns:
        Incremented frame counter.
    """
    field_data = np.real(sim.get_array(vol=xy_plane, component=mp.Ey))
    t = sim.meep_time()
    if mp.am_master():
        os.makedirs("frames", exist_ok=True)
        np.savez_compressed(
            f"frames/meep_field_{frame_counter:04d}.npz",
            field=field_data,
            time=t,
        )
    return frame_counter + 1


def render_animation_frames(eps_data, extent):
    """Render saved field .npz files into PNGs with fixed global colorbar.

    Two-pass: first finds the global field maximum across all frames,
    then renders every frame with the same vmin/vmax so field decay is
    clearly visible.

    Call only on master rank after sim.run().
    """
    import glob

    if not HAS_MATPLOTLIB:
        logger.warning(
            "matplotlib not available, .npz field files kept but not rendered"
        )
        return

    npz_files = sorted(glob.glob("frames/meep_field_*.npz"))
    if not npz_files:
        logger.warning("No field data files found to render")
        return

    # Pass 1 — global max
    global_max = 0.0
    for path in npz_files:
        d = np.load(path)
        global_max = max(global_max, float(np.max(np.abs(d["field"]))))
    if global_max == 0:
        global_max = 1.0

    logger.info(
        "Rendering %d frames (Ey global max = %.4g) ...",
        len(npz_files), global_max,
    )

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Pass 2 — render each frame
    for i, path in enumerate(npz_files):
        d = np.load(path)
        field = d["field"]
        t = float(d["time"])

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        # Epsilon background (grayscale)
        ax.imshow(
            eps_data.T, origin="lower", extent=extent,
            cmap="binary", interpolation="none",
        )
        # Field overlay with fixed colorbar
        im = ax.imshow(
            field.T, origin="lower", extent=extent,
            cmap="RdBu", interpolation="spline36", alpha=0.8,
            vmin=-global_max, vmax=global_max,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.06)
        fig.colorbar(im, cax=cax, label="Ey")
        ax.set_title(f"Ey  t={t:.2f}")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.tight_layout()
        fig.savefig(f"frames/meep_frame_{i:04d}.png", dpi=150)
        plt.close(fig)

    # Clean up .npz intermediates
    for path in npz_files:
        os.remove(path)

    logger.info("Rendered %d frames with fixed colorbar", len(npz_files))


def compile_animation_mp4(fps=15):
    """Stitch meep_frame_*.png into meep_animation.mp4 via ffmpeg.

    Falls back gracefully if ffmpeg is not available — frame PNGs
    are still kept as individual files.
    """
    import glob
    import subprocess

    frames = sorted(glob.glob("frames/meep_frame_*.png"))
    if not frames:
        logger.warning("No animation frames found to compile")
        return

    logger.info("Compiling %d frames into meep_animation.mp4 ...", len(frames))
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", "frames/meep_frame_%04d.png",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "meep_animation.mp4",
            ],
            check=True,
            capture_output=True,
        )
        logger.info("Saved meep_animation.mp4")
    except FileNotFoundError:
        logger.warning("ffmpeg not found — frame PNGs saved but MP4 not created")
    except subprocess.CalledProcessError as e:
        logger.warning("ffmpeg failed: %s", e.stderr.decode()[:500])
        logger.info("Frame PNGs are still available in frames/")


def save_epsilon_raw(sim, config, cell_center):
    """Save raw epsilon array as .npy for XY slice at core center."""
    z_min = min(l["zmin"] for l in config["layer_stack"])
    z_max = max(l["zmax"] for l in config["layer_stack"])
    z_core = (z_min + z_max) / 2

    try:
        xy_plane = mp.Volume(
            center=mp.Vector3(cell_center.x, cell_center.y, z_core),
            size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
        )
        eps_data = sim.get_array(vol=xy_plane, component=mp.Dielectric)
        if mp.am_master():
            np.save("meep_epsilon_xy.npy", eps_data)
            logger.info("Saved meep_epsilon_xy.npy (shape=%s)", eps_data.shape)
    except Exception as e:
        logger.warning("Epsilon raw save failed: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run MEEP simulation."""
    config_path = "%%CONFIG_FILENAME%%"
    config = load_config(config_path)

    logger.info("Building materials...")
    materials = build_materials(config)

    logger.info("Building background dielectric slabs...")
    background_slabs = build_background_slabs(config, materials)
    logger.info("Created %d background slabs", len(background_slabs))

    logger.info("Building geometry from GDS + layer stack...")
    geometry, component = build_geometry(config, materials)
    logger.info(
        "Created %d prisms from %d layers",
        len(geometry), len(config['layer_stack']),
    )

    # Background slabs first, then patterned prisms (later objects take precedence)
    geometry = background_slabs + geometry

    logger.info("Building sources...")
    sources = build_sources(config)

    if not sources:
        logger.error("No source port found in config")
        sys.exit(1)

    resolution = config["resolution"]["pixels_per_um"]
    fdtd = config["fdtd"]

    # Compute simulation cell from component bounds + layer z-range
    # Use original component bbox if available (port extension changes GDS bbox)
    component_bbox = config["component_bbox"]
    if component_bbox is not None:
        bbox_left, bbox_bottom, bbox_right, bbox_top = component_bbox
    else:
        bbox = component.dbbox()
        bbox_left, bbox_right = bbox.left, bbox.right
        bbox_bottom, bbox_top = bbox.bottom, bbox.top

    # Use both layers and dielectrics for z-range so that PDKs without
    # explicit box/clad layers (e.g. cspdk) still get enough headroom.
    z_vals = [l["zmin"] for l in config["layer_stack"]] + [
        l["zmax"] for l in config["layer_stack"]
    ]
    for d in config.get("dielectrics", []):
        z_vals.extend((d["zmin"], d["zmax"]))
    z_min = min(z_vals)
    z_max = max(z_vals)

    domain = config["domain"]
    dpml = domain["dpml"]
    margin_xy = domain["margin_xy"]

    # XY: margin_xy is gap between geometry bbox and PML
    cell_x = (bbox_right - bbox_left) + 2 * (margin_xy + dpml)
    cell_y = (bbox_top - bbox_bottom) + 2 * (margin_xy + dpml)
    # Z: margin_z_above/below is already baked into the stack via z_crop,
    #    so only add dpml beyond the stack extent
    cell_z = (z_max - z_min) + 2 * dpml
    cell_center = mp.Vector3(
        (bbox_right + bbox_left) / 2,
        (bbox_top + bbox_bottom) / 2,
        (z_max + z_min) / 2,
    )

    logger.info("Cell size: %.2f x %.2f x %.2f um", cell_x, cell_y, cell_z)
    logger.info("PML: %.2f um, margin_xy: %.2f", dpml, margin_xy)
    logger.info("Resolution: %s pixels/um", resolution)

    accuracy = config["accuracy"]
    diagnostics = config["diagnostics"]
    preview_only = diagnostics["preview_only"]

    # In preview mode, skip expensive subpixel averaging
    eps_avg = False if preview_only else accuracy["eps_averaging"]

    # Symmetries are NOT used for S-parameter extraction runs.
    # MEEP's get_eigenmode_coefficients with add_mode_monitor produces
    # incorrect normalization when the source monitor straddles a mirror
    # symmetry plane (coefficients underestimated by ~2x).  This is
    # consistent with gplugins which also never uses mp.Mirror for
    # S-parameter extraction.  Symmetries are only applied in preview-only
    # mode (geometry validation, no FDTD/S-params).
    cfg_symmetries = config["symmetries"]
    if cfg_symmetries and not preview_only:
        logger.info("Symmetries present in config but IGNORED for S-parameter "
                     "extraction (causes incorrect eigenmode normalization). "
                     "Symmetries are only used in preview-only mode.")
    use_symmetries = build_symmetries(config) if preview_only else []

    sim_kwargs = dict(
        cell_size=mp.Vector3(cell_x, cell_y, cell_z),
        geometry_center=cell_center,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        boundary_layers=[mp.PML(dpml)],
        symmetries=use_symmetries,
        split_chunks_evenly=config["split_chunks_evenly"],
        eps_averaging=eps_avg,
    )
    spx_maxeval = accuracy["subpixel_maxeval"]
    if spx_maxeval > 0:
        sim_kwargs["subpixel_maxeval"] = spx_maxeval
    spx_tol = accuracy["subpixel_tol"]
    if spx_tol != 1e-4:
        sim_kwargs["subpixel_tol"] = spx_tol
    sim = mp.Simulation(**sim_kwargs)

    # --- Diagnostics & preview mode ---
    diag_geometry = diagnostics["save_geometry"]
    diag_fields = diagnostics["save_fields"]
    diag_epsilon = diagnostics["save_epsilon_raw"]

    if diag_geometry or diag_epsilon or preview_only:
        logger.info("Initializing simulation for diagnostics...")
        sim.init_sim()
        if diag_geometry or preview_only:
            save_geometry_diagnostics(sim, config, cell_center)
        if diag_epsilon:
            save_epsilon_raw(sim, config, cell_center)

    if preview_only:
        logger.info("MEEP_PREVIEW_ONLY=1 — skipping simulation run.")
        save_debug_log(config, {}, {"_meep_time": 0, "_timesteps": 0,
                                    "_cell_size": [cell_x, cell_y, cell_z]})
        logger.info("Preview complete.")
        sys.exit(0)

    logger.info("Building monitors...")
    monitors = build_monitors(config, sim)

    stopping = config["stopping"]
    run_after = stopping["run_after_sources"]

    # Build verbose step functions
    step_funcs = []
    verbose_interval = config["verbose_interval"]
    if verbose_interval > 0:
        _wall_start = time.time()
        def _verbose_print(sim_obj):
            elapsed = time.time() - _wall_start
            logger.info("t=%.2f | wall=%.1fs", sim_obj.meep_time(), elapsed)
        step_funcs.append(mp.at_every(verbose_interval, _verbose_print))

    # Animation field capture step function (raw data, no plotting yet)
    diag_animation = diagnostics["save_animation"]
    animation_interval = diagnostics["animation_interval"]
    _frame_counter = [0]  # mutable container for closure
    _anim_plane = None

    if diag_animation:
        z_min_anim = min(l["zmin"] for l in config["layer_stack"])
        z_max_anim = max(l["zmax"] for l in config["layer_stack"])
        z_core_anim = (z_min_anim + z_max_anim) / 2
        _anim_plane = mp.Volume(
            center=mp.Vector3(cell_center.x, cell_center.y, z_core_anim),
            size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
        )

        def _capture_frame(sim_obj):
            _frame_counter[0] = save_animation_field(
                sim_obj, _anim_plane, _frame_counter[0]
            )

        step_funcs.append(mp.at_every(animation_interval, _capture_frame))
        logger.info(
            "Animation: saving field data every %s time units",
            animation_interval,
        )

    stop_mode = stopping["mode"]
    wall_time_max = stopping.get("wall_time_max", 0)
    if wall_time_max > 0:
        logger.info("Wall-clock safety net: %.0f seconds", wall_time_max)

    wall_start = time.time()

    if stop_mode == "dft_decay":
        decay_by = stopping["decay_by"]
        min_time = stopping["dft_min_run_time"]
        logger.info(
            "Running simulation (dft_decay mode: tol=%s, "
            "min=%.1f, max=%.1f)...",
            decay_by, min_time, run_after,
        )
        dft_fn = mp.stop_when_dft_decayed(
            tol=decay_by,
            minimum_run_time=min_time,
            maximum_run_time=run_after,
        )
        if wall_time_max > 0:
            conds = [dft_fn, _make_wall_time_cap(wall_time_max)]
            sim.run(*step_funcs, until_after_sources=conds)
        else:
            sim.run(*step_funcs, until_after_sources=dft_fn)
    elif stop_mode == "energy_decay":
        dt = stopping["decay_dt"]
        decay_by = stopping["decay_by"]
        logger.info(
            "Running simulation (energy_decay mode: dt=%s, "
            "decay_by=%s, cap=%.1f)...",
            dt, decay_by, run_after,
        )
        energy_fn = mp.stop_when_energy_decayed(dt=dt, decay_by=decay_by)
        conds = [energy_fn, _make_time_cap(run_after)]
        if wall_time_max > 0:
            conds.append(_make_wall_time_cap(wall_time_max))
        sim.run(*step_funcs, until_after_sources=conds)
    elif stop_mode == "field_decay":
        dt = stopping["decay_dt"]
        comp_name = stopping["decay_component"]
        comp = _COMPONENT_MAP.get(comp_name, mp.Ey)
        decay_by = stopping["decay_by"]
        monitor_pt = resolve_decay_monitor_point(config)
        logger.info("Running simulation (field_decay mode: component=%s, dt=%s, "
                     "decay_by=%s, cap=%.1f)...", comp_name, dt, decay_by, run_after)

        # Decay condition + numeric time cap (list = OR logic, first wins)
        decay_fn = mp.stop_when_fields_decayed(dt, comp, monitor_pt, decay_by)
        conds = [decay_fn, _make_time_cap(run_after)]
        if wall_time_max > 0:
            conds.append(_make_wall_time_cap(wall_time_max))
        sim.run(*step_funcs, until_after_sources=conds)
    else:
        logger.info("Running simulation (until_after_sources=%.1f)...", run_after)
        if wall_time_max > 0:
            conds = [
                _make_time_cap(run_after),
                _make_wall_time_cap(wall_time_max),
            ]
            sim.run(*step_funcs, until_after_sources=conds)
        else:
            sim.run(*step_funcs, until_after_sources=run_after)

    wall_seconds = time.time() - wall_start
    if wall_time_max > 0 and wall_seconds >= wall_time_max * 0.95:
        logger.warning(
            "Wall-clock limit likely triggered (%.1fs elapsed, limit %.0fs). "
            "Results may be incomplete — consider increasing wall_time_max or "
            "using a coarser resolution.",
            wall_seconds, wall_time_max,
        )

    if diag_fields:
        save_field_snapshot(sim, config, cell_center)

    if diag_animation and _frame_counter[0] > 0 and _anim_plane is not None:
        # get_array is collective — all ranks must call
        eps_data = sim.get_array(vol=_anim_plane, component=mp.Dielectric)
        if mp.am_master():
            _ctr = _anim_plane.center
            _sz = _anim_plane.size
            _extent = [
                _ctr.x - _sz.x / 2, _ctr.x + _sz.x / 2,
                _ctr.y - _sz.y / 2, _ctr.y + _sz.y / 2,
            ]
            render_animation_frames(eps_data, _extent)
            compile_animation_mp4()

    logger.info("Extracting S-parameters...")
    s_params, debug_data = extract_s_params(config, sim, monitors)

    # Attach simulation metadata to debug_data
    debug_data["_meep_time"] = sim.meep_time()
    debug_data["_timesteps"] = sim.timestep()
    debug_data["_cell_size"] = [cell_x, cell_y, cell_z]

    save_results(config, s_params)
    save_debug_log(config, s_params, debug_data, wall_seconds=wall_seconds)
    logger.info("Done!")


if __name__ == "__main__":
    main()
'''


def generate_meep_script(config_filename: str = "sim_config.json") -> str:
    """Generate the Python script that runs on the cloud MEEP instance.

    The script is self-contained and reads a JSON config + GDS file to:
    1. Load GDS layout via gdsfactory
    2. Extract polygons per layer, handling holes via triangulation
    3. Create mp.Prism objects with correct materials and sidewall angles
    4. Create EigenModeSource at the source port
    5. Create mode monitors at all ports
    6. Build mp.Simulation with PML boundaries
    7. Run and extract S-parameters
    8. Save results as CSV

    Args:
        config_filename: Name of the JSON config file (default: sim_config.json)

    Returns:
        Python script as a string
    """
    return _MEEP_RUNNER_TEMPLATE.replace("%%CONFIG_FILENAME%%", config_filename)