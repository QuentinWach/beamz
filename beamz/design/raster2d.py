import numpy as np
from shapely.geometry import box as shapely_box
from shapely.prepared import prep

from beamz.design.grids import MaterialGrids
from beamz.design.structures import Rectangle
from beamz.visual.helpers import create_rich_progress


def rasterize(mesh):
    """Painters algorithm: rasterize a design into a 2D grid."""
    width, height = mesh.design.width, mesh.design.height
    grid_width, grid_height = int(width / mesh.resolution), int(height / mesh.resolution)
    cell_size = mesh.resolution

    x_centers = np.linspace(0.5 * cell_size, width - 0.5 * cell_size, grid_width)
    y_centers = np.linspace(0.5 * cell_size, height - 0.5 * cell_size, grid_height)

    sample_dx, sample_dy = mesh._build_supersample_offsets_xy(cell_size)
    num_samples = len(sample_dx)

    grids = MaterialGrids((grid_height, grid_width))

    if len(mesh.design.structures) > 0:
        background = mesh.design.structures[0]
        if hasattr(background, "material") and background.material is not None:
            grids.fill_all(mesh._get_all_material_props(background.material))

    with create_rich_progress() as progress:
        task = progress.add_task(
            "Rasterizing structures...", total=len(mesh.design.structures)
        )
        progress.update(task, advance=1)

        for idx in range(1, len(mesh.design.structures)):
            structure = mesh.design.structures[idx]

            if hasattr(structure, "is_pml") and structure.is_pml:
                progress.update(task, advance=1)
                continue
            if not hasattr(structure, "material") or structure.material is None:
                progress.update(task, advance=1)
                continue

            is_custom_material = hasattr(structure.material, "get_permittivity")
            props = (
                None
                if is_custom_material
                else mesh._get_all_material_props(structure.material)
            )

            try:
                bbox_indices = get_bbox_indices(
                    structure, grid_height, grid_width, cell_size
                )
                if bbox_indices is None:
                    progress.update(task, advance=1)
                    continue
                min_i, min_j, max_i, max_j = bbox_indices

                if isinstance(structure, Rectangle) and is_axis_aligned(structure):
                    rasterize_rectangle(
                        mesh,
                        structure,
                        grids,
                        props,
                        is_custom_material,
                        grid_height,
                        grid_width,
                        cell_size,
                        x_centers,
                        y_centers,
                        sample_dx,
                        sample_dy,
                        num_samples,
                    )
                elif hasattr(structure, "radius") and not hasattr(
                    structure, "inner_radius"
                ):
                    rasterize_circle(
                        mesh,
                        structure,
                        grids,
                        props,
                        min_i,
                        min_j,
                        max_i,
                        max_j,
                        cell_size,
                        x_centers,
                        y_centers,
                        sample_dx,
                        sample_dy,
                        num_samples,
                    )
                elif hasattr(structure, "inner_radius") and hasattr(
                    structure, "outer_radius"
                ):
                    rasterize_ring(
                        mesh,
                        structure,
                        grids,
                        props,
                        min_i,
                        min_j,
                        max_i,
                        max_j,
                        cell_size,
                        x_centers,
                        y_centers,
                        sample_dx,
                        sample_dy,
                        num_samples,
                    )
                else:
                    rasterize_polygon(
                        mesh,
                        structure,
                        grids,
                        props,
                        is_custom_material,
                        min_i,
                        min_j,
                        max_i,
                        max_j,
                        cell_size,
                        x_centers,
                        y_centers,
                        sample_dx,
                        sample_dy,
                        num_samples,
                    )

            except (AttributeError, TypeError) as exc:
                print(
                    f"Warning: Structure {type(structure)} doesn't have proper bounding box: {exc}"
                )

            progress.update(task, advance=1)

    grids.assign_to(mesh)


def is_axis_aligned(structure):
    """Check if a Rectangle is axis-aligned (not rotated)."""
    return (
        structure.vertices[0][0] == structure.position[0]
        and structure.vertices[0][1] == structure.position[1]
    )


def get_bbox_indices(structure, grid_height, grid_width, cell_size):
    """Get bounding box grid indices for a structure."""
    bbox = structure.get_bounding_box()
    if bbox is None:
        return None

    if len(bbox) == 6:
        min_x, min_y, _, max_x, max_y, _ = bbox
    elif len(bbox) == 4:
        min_x, min_y, max_x, max_y = bbox
    else:
        raise ValueError(f"Invalid bounding box format: {bbox}")

    min_i = max(0, int(min_y / cell_size) - 1)
    min_j = max(0, int(min_x / cell_size) - 1)
    max_i = min(grid_height, int(np.ceil(max_y / cell_size)) + 1)
    max_j = min(grid_width, int(np.ceil(max_x / cell_size)) + 1)

    if min_i >= grid_height or min_j >= grid_width or max_i <= 0 or max_j <= 0:
        return None
    return min_i, min_j, max_i, max_j


def supersample_cell(
    mesh,
    cx,
    cy,
    sample_dx,
    sample_dy,
    num_samples,
    contains_fn,
    *,
    cell_i=None,
    cell_j=None,
    cell_k=0,
    cell_size=None,
):
    """Count how many configured sample points are inside the shape."""
    local_dx, local_dy = sample_dx, sample_dy
    if cell_i is not None and cell_j is not None:
        local_dx, local_dy = mesh._scramble_offsets_xy_for_cell(
            sample_dx=sample_dx,
            sample_dy=sample_dy,
            cell_size=(mesh.resolution if cell_size is None else cell_size),
            cell_i=cell_i,
            cell_j=cell_j,
            cell_k=cell_k,
        )

    count = 0
    for k in range(num_samples):
        if contains_fn(cx + local_dx[k], cy + local_dy[k]):
            count += 1
    return count


def rasterize_rectangle(
    mesh,
    structure,
    grids,
    props,
    is_custom_material,
    grid_height,
    grid_width,
    cell_size,
    x_centers,
    y_centers,
    sample_dx,
    sample_dy,
    num_samples,
):
    """Exact area coverage for axis-aligned rectangles."""
    del sample_dx, sample_dy, num_samples

    rect_min_j = max(0, int(structure.position[0] / cell_size))
    rect_min_i = max(0, int(structure.position[1] / cell_size))
    rect_max_j = min(
        grid_width,
        int(np.ceil((structure.position[0] + structure.width) / cell_size)),
    )
    rect_max_i = min(
        grid_height,
        int(np.ceil((structure.position[1] + structure.height) / cell_size)),
    )

    sx, sy = structure.position[0], structure.position[1]
    sw, sh = structure.width, structure.height
    j_idx = np.arange(rect_min_j, rect_max_j, dtype=float)
    i_idx = np.arange(rect_min_i, rect_max_i, dtype=float)
    cell_x0 = j_idx * cell_size
    cell_x1 = cell_x0 + cell_size
    cell_y0 = i_idx * cell_size
    cell_y1 = cell_y0 + cell_size

    overlap_x = np.clip(
        np.minimum(cell_x1, sx + sw) - np.maximum(cell_x0, sx), 0.0, cell_size
    )
    overlap_y = np.clip(
        np.minimum(cell_y1, sy + sh) - np.maximum(cell_y0, sy), 0.0, cell_size
    )
    coverage = np.outer(overlap_y, overlap_x) / float(cell_size * cell_size)

    local_i, local_j = np.where(coverage >= 1.0 - 1e-15)
    for idx in range(len(local_i)):
        i = local_i[idx] + rect_min_i
        j = local_j[idx] + rect_min_j
        rect_props = props
        if is_custom_material:
            rect_props = mesh._get_all_material_props(
                structure.material, x_centers[j], y_centers[i]
            )
        grids.set_at((i, j), rect_props)

    boundary_i, boundary_j = np.where((coverage > 0.0) & (coverage < 1.0 - 1e-15))
    for idx in range(len(boundary_i)):
        i = boundary_i[idx] + rect_min_i
        j = boundary_j[idx] + rect_min_j
        rect_props = props
        if is_custom_material:
            rect_props = mesh._get_all_material_props(
                structure.material, x_centers[j], y_centers[i]
            )
        grids.blend_at(
            (i, j), rect_props, float(coverage[boundary_i[idx], boundary_j[idx]])
        )


def rasterize_circle(
    mesh,
    structure,
    grids,
    props,
    min_i,
    min_j,
    max_i,
    max_j,
    cell_size,
    x_centers,
    y_centers,
    sample_dx,
    sample_dy,
    num_samples,
):
    """Fast path for circles using distance-based classification."""
    center_x, center_y = structure.position[0], structure.position[1]
    radius = structure.radius

    j_indices = np.arange(min_j, max_j)
    i_indices = np.arange(min_i, max_i)
    x_grid, y_grid = np.meshgrid(x_centers[j_indices], y_centers[i_indices])
    distances = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

    diag = 0.3536 * cell_size
    fully_inside = distances + diag <= radius
    boundary = (distances - diag <= radius) & ~fully_inside

    local_i, local_j = np.where(fully_inside)
    if len(local_i) > 0:
        grids.set_region((local_i + min_i, local_j + min_j), props)

    boundary_i, boundary_j = np.where(boundary)
    for idx in range(len(boundary_i)):
        i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
        cx, cy = x_centers[j], y_centers[i]
        samples_inside = supersample_cell(
            mesh,
            cx,
            cy,
            sample_dx,
            sample_dy,
            num_samples,
            lambda x, y: np.hypot(x - center_x, y - center_y) <= radius,
            cell_i=i,
            cell_j=j,
            cell_size=cell_size,
        )
        if samples_inside > 0:
            grids.blend_at((i, j), props, samples_inside / num_samples)


def rasterize_ring(
    mesh,
    structure,
    grids,
    props,
    min_i,
    min_j,
    max_i,
    max_j,
    cell_size,
    x_centers,
    y_centers,
    sample_dx,
    sample_dy,
    num_samples,
):
    """Fast path for rings using distance-based classification."""
    center_x, center_y = structure.position[0], structure.position[1]
    inner_radius = structure.inner_radius
    outer_radius = structure.outer_radius

    j_indices = np.arange(min_j, max_j)
    i_indices = np.arange(min_i, max_i)
    x_grid, y_grid = np.meshgrid(x_centers[j_indices], y_centers[i_indices])
    distances = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

    diag = 0.3536 * cell_size
    fully_inside = (distances - diag >= inner_radius) & (
        distances + diag <= outer_radius
    )
    inner_boundary = (distances - diag <= inner_radius) & (
        distances + diag >= inner_radius
    )
    outer_boundary = (distances - diag <= outer_radius) & (
        distances + diag >= outer_radius
    )
    boundary = inner_boundary | outer_boundary

    local_i, local_j = np.where(fully_inside)
    if len(local_i) > 0:
        grids.set_region((local_i + min_i, local_j + min_j), props)

    boundary_i, boundary_j = np.where(boundary)
    for idx in range(len(boundary_i)):
        i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
        cx, cy = x_centers[j], y_centers[i]
        samples_inside = supersample_cell(
            mesh,
            cx,
            cy,
            sample_dx,
            sample_dy,
            num_samples,
            lambda x, y: inner_radius
            <= np.hypot(x - center_x, y - center_y)
            <= outer_radius,
            cell_i=i,
            cell_j=j,
            cell_size=cell_size,
        )
        if samples_inside > 0:
            grids.blend_at((i, j), props, samples_inside / num_samples)


def rasterize_polygon(
    mesh,
    structure,
    grids,
    props,
    is_custom_material,
    min_i,
    min_j,
    max_i,
    max_j,
    cell_size,
    x_centers,
    y_centers,
    sample_dx,
    sample_dy,
    num_samples,
):
    """General path for polygons and complex shapes."""
    polygon = mesh._structure_polygon_2d(structure)
    if polygon is not None:
        prepared_polygon = prep(polygon)
        cell_area = float(cell_size * cell_size)
        for i in range(min_i, max_i):
            cell_y0 = i * cell_size
            cell_y1 = cell_y0 + cell_size
            cy = y_centers[i]
            for j in range(min_j, max_j):
                cell_x0 = j * cell_size
                cell_x1 = cell_x0 + cell_size
                cx = x_centers[j]
                cell = shapely_box(cell_x0, cell_y0, cell_x1, cell_y1)
                if prepared_polygon.contains(cell):
                    cell_props = props
                    if is_custom_material:
                        cell_props = mesh._get_all_material_props(
                            structure.material, cx, cy
                        )
                    grids.set_at((i, j), cell_props)
                    continue
                if not prepared_polygon.intersects(cell):
                    continue

                blend_factor = polygon.intersection(cell).area / cell_area
                if blend_factor <= 0.0:
                    continue
                cell_props = props
                if is_custom_material:
                    cell_props = mesh._get_all_material_props(structure.material, cx, cy)
                grids.blend_at((i, j), cell_props, float(blend_factor))
        return

    if hasattr(structure, "point_in_polygon"):
        contains_func = lambda x, y: structure.point_in_polygon(x, y)
    else:
        contains_func = lambda x, y: any(
            val != def_val
            for val, def_val in zip(
                mesh.design.get_material_value(x, y, z=0), [1.0, 1.0, 0.0]
            )
        )

    if hasattr(structure, "vertices") and len(getattr(structure, "vertices", [])) > 0:
        inside_mask = np.zeros((max_i - min_i, max_j - min_j), dtype=bool)
        boundary_mask = np.zeros((max_i - min_i, max_j - min_j), dtype=bool)
        sample_points = [(0, 0), (-0.4, -0.4), (-0.4, 0.4), (0.4, -0.4), (0.4, 0.4)]

        for i_rel in range(max_i - min_i):
            for j_rel in range(max_j - min_j):
                cx = x_centers[j_rel + min_j]
                cy = y_centers[i_rel + min_i]
                points_inside = 0
                center_inside = False
                if contains_func(cx, cy):
                    center_inside = True
                    points_inside += 1
                for dx_pt, dy_pt in sample_points[1:]:
                    if contains_func(cx + dx_pt * cell_size, cy + dy_pt * cell_size):
                        points_inside += 1
                if center_inside and points_inside == len(sample_points):
                    inside_mask[i_rel, j_rel] = True
                elif points_inside > 0:
                    boundary_mask[i_rel, j_rel] = True

        inside_i, inside_j = np.where(inside_mask)
        for idx in range(len(inside_i)):
            i, j = inside_i[idx] + min_i, inside_j[idx] + min_j
            grids.set_at((i, j), props)

        for mask in (boundary_mask, ~inside_mask & ~boundary_mask):
            boundary_i, boundary_j = np.where(mask)
            for idx in range(len(boundary_i)):
                i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
                cx, cy = x_centers[j], y_centers[i]
                samples_inside = supersample_cell(
                    mesh,
                    cx,
                    cy,
                    sample_dx,
                    sample_dy,
                    num_samples,
                    contains_func,
                    cell_i=i,
                    cell_j=j,
                    cell_size=cell_size,
                )
                if samples_inside > 0:
                    grids.blend_at((i, j), props, samples_inside / num_samples)
        return

    for i in range(min_i, max_i):
        for j in range(min_j, max_j):
            cx, cy = x_centers[j], y_centers[i]
            samples_inside = supersample_cell(
                mesh,
                cx,
                cy,
                sample_dx,
                sample_dy,
                num_samples,
                contains_func,
                cell_i=i,
                cell_j=j,
                cell_size=cell_size,
            )
            if samples_inside > 0:
                grids.blend_at((i, j), props, samples_inside / num_samples)
