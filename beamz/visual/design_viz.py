"""Design visualization - 2D matplotlib rendering and show_design dispatcher."""

import numpy as np

from beamz.visual.helpers import get_si_scale_and_label
from beamz.visual.overlays import _get_deterministic_color


def draw_polygon(
    ax, polygon, facecolor=None, edgecolor="black", alpha=None, linestyle=None
):
    """Draw a polygon (with possible holes) on a Matplotlib axis.
    Projects 3D vertices to 2D for plotting.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if facecolor is None:
        facecolor = getattr(polygon, "color", None) or "#999999"
    if alpha is None:
        alpha = 1.0
    if linestyle is None:
        linestyle = "-"
    if not getattr(polygon, "vertices", None):
        return

    # Exterior path - project to 2D
    all_path_coords = []
    all_path_codes = []
    vertices_2d = (
        polygon._vertices_2d(polygon.vertices)
        if hasattr(polygon, "_vertices_2d")
        else [(v[0], v[1]) for v in polygon.vertices]
    )
    if len(vertices_2d) > 0:
        all_path_coords.extend(vertices_2d)
        all_path_coords.append(vertices_2d[0])
        all_path_codes.append(Path.MOVETO)
        if len(vertices_2d) > 1:
            all_path_codes.extend([Path.LINETO] * (len(vertices_2d) - 1))
        all_path_codes.append(Path.CLOSEPOLY)

    # Interior paths (holes)
    for interior_v_list in getattr(polygon, "interiors", []) or []:
        if interior_v_list and len(interior_v_list) > 0:
            interior_2d = (
                polygon._vertices_2d(interior_v_list)
                if hasattr(polygon, "_vertices_2d")
                else [(v[0], v[1]) for v in interior_v_list]
            )
            all_path_coords.extend(interior_2d)
            all_path_coords.append(interior_2d[0])
            all_path_codes.append(Path.MOVETO)
            if len(interior_2d) > 1:
                all_path_codes.extend([Path.LINETO] * (len(interior_2d) - 1))
            all_path_codes.append(Path.CLOSEPOLY)

    if not all_path_coords or not all_path_codes:
        return

    path = Path(np.array(all_path_coords), np.array(all_path_codes))
    patch = PathPatch(
        path, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linestyle=linestyle
    )
    ax.add_patch(patch)


def determine_if_3d(design):
    """Determine if the design should be visualized in 3D based on structure properties."""
    if design.depth and design.depth > 0:
        for structure in design.structures:
            if hasattr(structure, "is_pml") and structure.is_pml:
                continue
            if hasattr(structure, "depth") and structure.depth and structure.depth > 0:
                return True
            if hasattr(structure, "z") and structure.z and structure.z != 0:
                return True
            position = getattr(structure, "position", None)
            if position is not None and len(position) > 2 and position[2] != 0:
                return True
            if hasattr(structure, "vertices") and structure.vertices:
                for vertex in structure.vertices:
                    if len(vertex) > 2 and vertex[2] != 0:
                        return True
    return False


def show_design(design, unify_structures=True):
    """Display the design visually using 2D matplotlib or 3D plotly."""
    if determine_if_3d(design):
        from beamz.visual.design_3d import show_design_3d

        show_design_3d(design, unify_structures)
    else:
        show_design_2d(design, unify_structures)


def show_design_2d(design, unify_structures=True):
    """Display the design using 2D matplotlib visualization."""
    import matplotlib.pyplot as plt

    max_dim = max(design.width, design.height)
    scale, unit = get_si_scale_and_label(max_dim)
    aspect_ratio = design.width / design.height
    base_size = 5
    if aspect_ratio > 1:
        figsize = (base_size * aspect_ratio, base_size)
    else:
        figsize = (base_size, base_size / aspect_ratio)

    if unify_structures:
        tmp_design = design.copy()
        tmp_design.unify_polygons()
        structures_to_plot = tmp_design.structures
        sources_to_plot = tmp_design.sources
        monitors_to_plot = tmp_design.monitors
    else:
        structures_to_plot = design.structures
        sources_to_plot = design.sources
        monitors_to_plot = design.monitors

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # Assign deterministic colors based on material (group by material, then by order)
    material_colors = {}
    color_index = 0
    for structure in structures_to_plot:
        if hasattr(structure, "is_pml") and structure.is_pml:
            structure.add_to_plot(
                ax, edgecolor="red", linestyle="--", facecolor="none", alpha=0.5
            )
        else:
            # Determine color based on material
            material_key = None
            if hasattr(structure, "material") and structure.material:
                material_key = (
                    getattr(structure.material, "permittivity", 1.0),
                    getattr(structure.material, "permeability", 1.0),
                    getattr(structure.material, "conductivity", 0.0),
                )

            if material_key not in material_colors:
                material_colors[material_key] = _get_deterministic_color(color_index)
                color_index += 1

            # Use deterministic color (override structure's random color)
            structure.add_to_plot(ax, facecolor=material_colors[material_key])

    # Add sources
    for source in sources_to_plot:
        if hasattr(source, "add_to_plot"):
            source.add_to_plot(ax)

    # Add monitors
    for monitor in monitors_to_plot:
        if hasattr(monitor, "add_to_plot"):
            monitor.add_to_plot(ax)

    ax.set_title("Design Layout")
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_xlim(0, design.width)
    ax.set_ylim(0, design.height)

    ax.xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")

    plt.tight_layout()
    plt.show()


def draw_boundary(ax, boundary, design, edgecolor="red", linestyle="--", alpha=0.5):
    """Draw boundary regions on a matplotlib axis."""
    from matplotlib.patches import Rectangle as MatplotlibRectangle

    edges = boundary._get_edges_for_dimensionality(design.is_3d)

    for edge in edges:
        if edge == "left":
            rect = MatplotlibRectangle(
                (0, 0),
                boundary.thickness,
                design.height,
                facecolor="none",
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha,
            )
        elif edge == "right":
            rect = MatplotlibRectangle(
                (design.width - boundary.thickness, 0),
                boundary.thickness,
                design.height,
                facecolor="none",
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha,
            )
        elif edge == "bottom":
            rect = MatplotlibRectangle(
                (0, 0),
                design.width,
                boundary.thickness,
                facecolor="none",
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha,
            )
        elif edge == "top":
            rect = MatplotlibRectangle(
                (0, design.height - boundary.thickness),
                design.width,
                boundary.thickness,
                facecolor="none",
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha,
            )
        elif edge == "front" and design.is_3d:
            # 3D front edge (z=0)
            rect = MatplotlibRectangle(
                (0, 0),
                design.width,
                design.height,
                facecolor="none",
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha,
            )
        elif edge == "back" and design.is_3d:
            # 3D back edge (z=depth)
            rect = MatplotlibRectangle(
                (0, 0),
                design.width,
                design.height,
                facecolor="none",
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha,
            )
        else:
            continue

        ax.add_patch(rect)
