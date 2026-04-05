import os
import time

import numpy as np

try:
    from matplotlib.path import Path as MplPath
except Exception:  # pragma: no cover - matplotlib is expected but keep fallback safe
    MplPath = None

from beamz.design.cache import _env_bool
from beamz.design.grids import (
    MaterialGrids,
    fill_background_material,
    get_bbox_indices_3d,
    is_axis_aligned_rectangle,
    iter_raster_structures,
)
from beamz.design.structures import Rectangle
from beamz.visual.helpers import (
    create_rich_progress,
    display_status,
)


def rasterize(mesh):
    """3D rasterization using vectorized shape fills and bounded fallbacks."""
    width, height, depth = mesh.design.width, mesh.design.height, mesh.design.depth
    grid_width = int(width / mesh.resolution_xy)
    grid_height = int(height / mesh.resolution_xy)
    grid_depth = int(depth / mesh.resolution_z) if depth > 0 else 1

    total_start = time.perf_counter()
    cell_size_xy = mesh.resolution_xy
    cell_size_z = mesh.resolution_z

    x_centers = np.linspace(0.5 * cell_size_xy, width - 0.5 * cell_size_xy, grid_width)
    y_centers = np.linspace(
        0.5 * cell_size_xy, height - 0.5 * cell_size_xy, grid_height
    )
    z_centers = (
        np.linspace(0.5 * cell_size_z, depth - 0.5 * cell_size_z, grid_depth)
        if depth > 0
        else [0]
    )

    c = 3e8
    dt_estimate = 0.5 * mesh.resolution / (c * np.sqrt(2))

    grids = MaterialGrids((grid_depth, grid_height, grid_width))
    setup_end = time.perf_counter()
    fill_background_material(grids, mesh.design, mesh._get_all_material_props)

    timing_enabled = _env_bool("BEAMZ_RASTER_TIMING", True)
    voxel_count = int(grid_width) * int(grid_height) * int(grid_depth)
    fast_min_voxels = int(
        float(os.getenv("BEAMZ_RASTER_FAST_MIN_VOXELS", 1_000_000))
    )
    fast_env = os.getenv("BEAMZ_RASTER_FAST_3D")
    if fast_env is None:
        prefer_fast = voxel_count >= fast_min_voxels
    else:
        prefer_fast = _env_bool("BEAMZ_RASTER_FAST_3D", True)

    struct_start = time.perf_counter()
    with create_rich_progress() as progress:
        task = progress.add_task(
            "Rasterizing 3D structures...", total=len(mesh.design.structures)
        )
        progress.update(task, advance=1)

        for structure in iter_raster_structures(mesh.design):

            props = mesh._get_all_material_props(structure.material)

            try:
                bbox = get_bbox_indices_3d(
                    structure,
                    grid_height=grid_height,
                    grid_width=grid_width,
                    grid_depth=grid_depth,
                    cell_size_xy=cell_size_xy,
                    cell_size_z=cell_size_z,
                    margin_cells=1,
                )
                if bbox is None:
                    progress.update(task, advance=1)
                    continue
                min_i, min_j, min_k, max_i, max_j, max_k = bbox

                fast_done = False
                if (
                    prefer_fast
                    and isinstance(structure, Rectangle)
                    and is_axis_aligned_rectangle(structure)
                ):
                    rasterize_rectangle(
                        structure=structure,
                        grids=grids,
                        props=props,
                        grid_height=grid_height,
                        grid_width=grid_width,
                        grid_depth=grid_depth,
                        cell_size_xy=cell_size_xy,
                        cell_size_z=cell_size_z,
                    )
                    fast_done = True

                if not fast_done and prefer_fast:
                    poly_done = rasterize_polygon(
                        mesh,
                        structure=structure,
                        grids=grids,
                        props=props,
                        min_i=min_i,
                        min_j=min_j,
                        min_k=min_k,
                        max_i=max_i,
                        max_j=max_j,
                        max_k=max_k,
                        x_centers=x_centers,
                        y_centers=y_centers,
                        cell_size_xy=cell_size_xy,
                        cell_size_z=cell_size_z,
                    )
                    if poly_done:
                        fast_done = True

                if not fast_done:
                    rasterize_fallback(
                        mesh,
                        structure=structure,
                        grids=grids,
                        props=props,
                        min_i=min_i,
                        min_j=min_j,
                        min_k=min_k,
                        max_i=max_i,
                        max_j=max_j,
                        max_k=max_k,
                        cell_size_xy=cell_size_xy,
                        cell_size_z=cell_size_z,
                        x_centers=x_centers,
                        y_centers=y_centers,
                        z_centers=z_centers,
                    )

            except (AttributeError, TypeError) as exc:
                display_status(
                    f"Warning: Structure {type(structure)} processing failed: {exc}",
                    "warning",
                )

            progress.update(task, advance=1)
    struct_end = time.perf_counter()

    pml_start = time.perf_counter()
    process_pml(
        mesh,
        grids.permittivity,
        grids.permeability,
        grids.conductivity,
        x_centers,
        y_centers,
        z_centers,
        dt_estimate,
    )
    pml_end = time.perf_counter()

    grids.assign_to(mesh)
    total_end = time.perf_counter()

    if timing_enabled:
        display_status(
            (
                "3D raster timing: "
                f"setup={setup_end - total_start:.2f}s, "
                f"structures={struct_end - struct_start:.2f}s, "
                f"pml={pml_end - pml_start:.2f}s, "
                f"total={total_end - total_start:.2f}s"
            ),
            "info",
        )


def process_pml(
    mesh,
    permittivity,
    permeability,
    conductivity,
    x_centers,
    y_centers,
    z_centers,
    dt_estimate,
):
    """Process 3D PML boundaries and add conductivity to the grid."""
    if not hasattr(mesh.design, "boundaries") or not mesh.design.boundaries:
        return

    with create_rich_progress() as progress:
        task = progress.add_task(
            "Processing 3D PML boundaries...", total=len(mesh.design.boundaries)
        )

        for boundary in mesh.design.boundaries:
            for k, z in enumerate(z_centers):
                for i, y in enumerate(y_centers):
                    for j, x in enumerate(x_centers):
                        pml_conductivity = boundary.get_conductivity(
                            x,
                            y,
                            z,
                            dx=mesh.resolution_xy,
                            dt=dt_estimate,
                            eps_avg=permittivity[k, i, j],
                            width=mesh.design.width,
                            height=mesh.design.height,
                            depth=mesh.design.depth,
                        )
                        if pml_conductivity > 0:
                            conductivity[k, i, j] += pml_conductivity

            progress.update(task, advance=1)

def rasterize_rectangle(
    *,
    structure,
    grids,
    props,
    grid_height,
    grid_width,
    grid_depth,
    cell_size_xy,
    cell_size_z,
):
    """Fast fill for axis-aligned rectangular prisms."""
    x0, y0, z0 = structure.position
    x1 = x0 + float(structure.width)
    y1 = y0 + float(structure.height)
    z1 = z0 + float(getattr(structure, "depth", 0.0))
    if z1 <= z0:
        z1 = z0 + cell_size_z

    i0 = max(0, int(np.floor(y0 / cell_size_xy)))
    j0 = max(0, int(np.floor(x0 / cell_size_xy)))
    k0 = max(0, int(np.floor(z0 / cell_size_z))) if grid_depth > 1 else 0
    i1 = min(grid_height, int(np.ceil(y1 / cell_size_xy)))
    j1 = min(grid_width, int(np.ceil(x1 / cell_size_xy)))
    k1 = min(grid_depth, int(np.ceil(z1 / cell_size_z))) if grid_depth > 1 else 1
    if i0 >= i1 or j0 >= j1 or k0 >= k1:
        return

    x_edges0 = np.arange(j0, j1, dtype=float) * cell_size_xy
    x_edges1 = x_edges0 + cell_size_xy
    y_edges0 = np.arange(i0, i1, dtype=float) * cell_size_xy
    y_edges1 = y_edges0 + cell_size_xy
    z_edges0 = np.arange(k0, k1, dtype=float) * cell_size_z
    z_edges1 = z_edges0 + cell_size_z

    fx = (
        np.clip(np.minimum(x_edges1, x1) - np.maximum(x_edges0, x0), 0.0, cell_size_xy)
        / cell_size_xy
    )
    fy = (
        np.clip(np.minimum(y_edges1, y1) - np.maximum(y_edges0, y0), 0.0, cell_size_xy)
        / cell_size_xy
    )
    fz = (
        np.clip(np.minimum(z_edges1, z1) - np.maximum(z_edges0, z0), 0.0, cell_size_z)
        / cell_size_z
    )

    frac = fz[:, None, None] * fy[None, :, None] * fx[None, None, :]
    if not np.any(frac > 0.0):
        return

    full_mask = frac >= (1.0 - 1e-12)
    blend_mask = (frac > 0.0) & ~full_mask

    if np.any(full_mask):
        for name, val in zip(MaterialGrids.NAMES, props):
            arr = getattr(grids, name)[k0:k1, i0:i1, j0:j1]
            arr[full_mask] = val

    if np.any(blend_mask):
        for name, val in zip(MaterialGrids.NAMES, props):
            arr = getattr(grids, name)[k0:k1, i0:i1, j0:j1]
            factors = frac[blend_mask]
            arr[blend_mask] = arr[blend_mask] * (1.0 - factors) + val * factors


def rasterize_polygon(
    mesh,
    *,
    structure,
    grids,
    props,
    min_i,
    min_j,
    min_k,
    max_i,
    max_j,
    max_k,
    x_centers,
    y_centers,
    cell_size_xy,
    cell_size_z,
):
    """Vectorized anti-aliased fill for extruded polygons."""
    if MplPath is None:
        return False
    if not hasattr(structure, "vertices") or not structure.vertices:
        return False
    if getattr(structure, "radius", None) is not None:
        return False
    if float(getattr(structure, "depth", 0.0)) <= 0.0:
        return False
    if min_i >= max_i or min_j >= max_j or min_k >= max_k:
        return False

    verts3 = np.asarray(structure.vertices, dtype=float)
    if verts3.ndim != 2 or verts3.shape[0] < 3:
        return False
    if verts3.shape[1] >= 3 and np.ptp(verts3[:, 2]) > 1e-12:
        return False

    verts = np.asarray([(v[0], v[1]) for v in structure.vertices], dtype=float)
    if verts.ndim != 2 or verts.shape[0] < 3:
        return False

    z0 = float(getattr(structure, "z", np.min(verts3[:, 2])))
    z1 = z0 + float(getattr(structure, "depth", 0.0))
    if z1 <= z0:
        z1 = float(np.max(verts3[:, 2]))
    if z1 <= z0:
        z1 = z0 + float(cell_size_z)

    x_local = x_centers[min_j:max_j]
    y_local = y_centers[min_i:max_i]
    xx, yy = np.meshgrid(x_local, y_local)
    outer_path = MplPath(verts)
    interiors = getattr(structure, "interiors", None) or []

    hole_paths = []
    for hole in interiors:
        iv3 = np.asarray(hole, dtype=float)
        if iv3.ndim == 2 and iv3.shape[1] >= 3 and np.ptp(iv3[:, 2]) > 1e-12:
            return False
        iv = np.asarray([(v[0], v[1]) for v in hole], dtype=float)
        if iv.ndim == 2 and iv.shape[0] >= 3:
            hole_paths.append(MplPath(iv))

    sample_dx, sample_dy = mesh._build_supersample_offsets_xy(cell_size_xy)
    n_samples_xy = float(sample_dx.size)
    inside_count = np.zeros(xx.shape, dtype=float)

    shift_x_map = None
    shift_y_map = None
    rot_map = None
    if mesh.aa_mode == "stratified_jitter":
        shift_x_map = np.empty(xx.shape, dtype=float)
        shift_y_map = np.empty(xx.shape, dtype=float)
        rot_map = np.empty(xx.shape, dtype=np.uint8)
        for i_rel in range(xx.shape[0]):
            cell_i = min_i + i_rel
            for j_rel in range(xx.shape[1]):
                cell_j = min_j + j_rel
                sx, sy, rot90 = mesh._cell_scramble_params_xy(cell_i, cell_j, 0)
                shift_x_map[i_rel, j_rel] = sx
                shift_y_map[i_rel, j_rel] = sy
                rot_map[i_rel, j_rel] = rot90

    for sample_idx in range(sample_dx.size):
        if mesh.aa_mode == "stratified_jitter":
            u0 = sample_dx[sample_idx] / float(cell_size_xy) + 0.5
            v0 = sample_dy[sample_idx] / float(cell_size_xy) + 0.5
            u_rot = np.where(
                rot_map == 0,
                u0,
                np.where(rot_map == 1, v0, np.where(rot_map == 2, 1.0 - u0, 1.0 - v0)),
            )
            v_rot = np.where(
                rot_map == 0,
                v0,
                np.where(rot_map == 1, 1.0 - u0, np.where(rot_map == 2, 1.0 - v0, u0)),
            )
            cell_dx = (np.mod(u_rot + shift_x_map, 1.0) - 0.5) * float(cell_size_xy)
            cell_dy = (np.mod(v_rot + shift_y_map, 1.0) - 0.5) * float(cell_size_xy)
            points = np.column_stack(((xx + cell_dx).ravel(), (yy + cell_dy).ravel()))
        else:
            points = np.column_stack(
                ((xx + sample_dx[sample_idx]).ravel(), (yy + sample_dy[sample_idx]).ravel())
            )
        inside = outer_path.contains_points(points, radius=1e-15).reshape(xx.shape)
        for hole_path in hole_paths:
            inside &= ~hole_path.contains_points(points, radius=1e-15).reshape(xx.shape)
        inside_count += inside.astype(float)
    frac_xy = inside_count / n_samples_xy
    if not np.any(frac_xy > 0.0):
        return True

    z_edges0 = np.arange(min_k, max_k, dtype=float) * float(cell_size_z)
    z_edges1 = z_edges0 + float(cell_size_z)
    frac_z = np.clip(
        np.minimum(z_edges1, z1) - np.maximum(z_edges0, z0),
        0.0,
        float(cell_size_z),
    ) / float(cell_size_z)
    frac = frac_z[:, None, None] * frac_xy[None, :, :]
    if not np.any(frac > 0.0):
        return True

    full_mask = frac >= (1.0 - 1e-12)
    blend_mask = (frac > 0.0) & ~full_mask

    if np.any(full_mask):
        for name, val in zip(MaterialGrids.NAMES, props):
            arr = getattr(grids, name)[min_k:max_k, min_i:max_i, min_j:max_j]
            arr[full_mask] = val

    if np.any(blend_mask):
        for name, val in zip(MaterialGrids.NAMES, props):
            arr = getattr(grids, name)[min_k:max_k, min_i:max_i, min_j:max_j]
            factors = frac[blend_mask]
            arr[blend_mask] = arr[blend_mask] * (1.0 - factors) + val * factors

    return True


def rasterize_fallback(
    mesh,
    *,
    structure,
    grids,
    props,
    min_i,
    min_j,
    min_k,
    max_i,
    max_j,
    max_k,
    cell_size_xy,
    cell_size_z,
    x_centers,
    y_centers,
    z_centers,
):
    """Fallback supersampling path for non-rectilinear 3D structures."""
    if min_i >= max_i or min_j >= max_j or min_k >= max_k:
        return

    sample_dx, sample_dy = mesh._build_supersample_offsets_xy(cell_size_xy)
    offsets_z = mesh._build_supersample_offsets_z(
        cell_size_z=cell_size_z,
        depth_samples=(3 if len(z_centers) > 1 else 1),
    )
    num_samples = sample_dx.size * offsets_z.size

    if hasattr(structure, "point_in_polygon"):
        contains_fn = lambda x, y, z: structure.point_in_polygon(x, y, z)
    else:
        contains_fn = lambda x, y, z: any(
            val != def_val
            for val, def_val in zip(
                mesh.design.get_material_value(x, y, z), [1.0, 1.0, 0.0]
            )
        )

    for k in range(min_k, max_k):
        z_center = z_centers[k]
        for i in range(min_i, max_i):
            y_center = y_centers[i]
            for j in range(min_j, max_j):
                x_center = x_centers[j]
                cell_dx, cell_dy = mesh._scramble_offsets_xy_for_cell(
                    sample_dx=sample_dx,
                    sample_dy=sample_dy,
                    cell_size=cell_size_xy,
                    cell_i=i,
                    cell_j=j,
                    cell_k=k,
                )
                inside = 0
                for z_off in offsets_z:
                    for sample_idx in range(cell_dx.size):
                        if contains_fn(
                            x_center + cell_dx[sample_idx],
                            y_center + cell_dy[sample_idx],
                            z_center + z_off,
                        ):
                            inside += 1
                if inside > 0:
                    grids.blend_at((k, i, j), props, inside / float(num_samples))
