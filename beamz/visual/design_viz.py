"""Design visualization - 2D matplotlib and 3D plotly rendering."""

import numpy as np

from beamz.visual.helpers import display_status, get_si_scale_and_label


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

def draw_pml(ax, pml, facecolor="none", edgecolor="black", alpha=0.5, linestyle="--"):
    """Draw a PML boundary on a Matplotlib axis as dashed lines."""
    from matplotlib.patches import Rectangle as MatplotlibRectangle

    if getattr(pml, "region_type", None) == "rect":
        rect_patch = MatplotlibRectangle(
            (pml.position[0], pml.position[1]),
            pml.width,
            pml.height,
            fill=False,
            edgecolor=edgecolor,
            linestyle=linestyle,
            alpha=alpha,
        )
        ax.add_patch(rect_patch)
    elif getattr(pml, "region_type", None) == "corner":
        # Draw a rectangle representing the corner PML based on orientation
        if pml.orientation == "bottom-left":
            rect_patch = MatplotlibRectangle(
                (pml.position[0] - pml.radius, pml.position[1] - pml.radius),
                pml.radius,
                pml.radius,
                fill=False,
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha,
            )
        elif pml.orientation == "bottom-right":
            rect_patch = MatplotlibRectangle(
                (pml.position[0], pml.position[1] - pml.radius),
                pml.radius,
                pml.radius,
                fill=False,
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha,
            )
        elif pml.orientation == "top-right":
            rect_patch = MatplotlibRectangle(
                (pml.position[0], pml.position[1]),
                pml.radius,
                pml.radius,
                fill=False,
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha,
            )
        elif pml.orientation == "top-left":
            rect_patch = MatplotlibRectangle(
                (pml.position[0] - pml.radius, pml.position[1]),
                pml.radius,
                pml.radius,
                fill=False,
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha,
            )
        else:
            return
        ax.add_patch(rect_patch)

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
            if (
                hasattr(structure, "position")
                and len(structure.position) > 2
                and structure.position[2] != 0
            ):
                return True
            if hasattr(structure, "vertices") and structure.vertices:
                for vertex in structure.vertices:
                    if len(vertex) > 2 and vertex[2] != 0:
                        return True
    return False

def show_design(design, unify_structures=True):
    """Display the design visually using 2D matplotlib or 3D plotly."""
    if determine_if_3d(design):
        show_design_3d(design, unify_structures)
    else:
        show_design_2d(design, unify_structures)

def _get_deterministic_color(index):
    """Get deterministic color: first from const.py, then deterministic generated colors."""
    import colorsys

    from beamz.const import BLUE, GREEN, ORANGE, PURPLE, RED

    predefined_colors = [BLUE, RED, GREEN, ORANGE, PURPLE]

    if index < len(predefined_colors):
        return predefined_colors[index]

    # For indices beyond predefined colors, use deterministic color generation
    saturation, value = 0.6, 0.7
    # Generate hue deterministically based on index (golden ratio for good distribution)
    hue = (index * 0.618034) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

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

def show_design_3d(design, unify_structures=True, max_vertices_for_unification=50):
    """Display the design using 3D plotly visualization."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        display_status(
            "Plotly is required for 3D visualization. Install with: pip install plotly",
            "error",
        )
        display_status("Falling back to 2D visualization...", "warning")
        show_design_2d(design, unify_structures)
        return

    max_dim = max(design.width, design.height, design.depth if design.depth else 0)
    scale, unit = get_si_scale_and_label(max_dim)

    if unify_structures:
        complex_structures = 0
        total_vertices = 0
        for structure in design.structures:
            if hasattr(structure, "vertices") and structure.vertices:
                vertices_count = len(structure.vertices)
                total_vertices += vertices_count
                if vertices_count > max_vertices_for_unification:
                    complex_structures += 1
        if complex_structures > 2 or total_vertices > 200:
            display_status(
                f"Disabling polygon unification for 3D (too complex: {complex_structures} complex structures, \
                                {total_vertices} total vertices)",
                "warning",
            )
            unify_structures = False
        else:
            design.unify_polygons()

    fig = go.Figure()
    default_depth = (
        design.depth if design.depth else min(design.width, design.height) * 0.1
    )

    from beamz.devices.monitors import Monitor
    from beamz.devices.sources import GaussianSource, ModeSource

    material_colors = {}
    color_index = 0

    # Add monitors
    for idx, monitor in enumerate(design.monitors):
        _add_monitor_to_3d_plot(fig, monitor, scale, unit, design=design, index=idx)

    # Add sources
    for idx, source in enumerate(design.sources):
        if isinstance(source, ModeSource):
            _add_mode_source_to_3d_plot(
                fig, source, scale, unit, design=design, index=idx
            )
        elif isinstance(source, GaussianSource):
            _add_gaussian_source_to_3d_plot(fig, source, scale, unit, index=idx)

    # Add structures
    structure_index = 0
    for structure in design.structures:
        if hasattr(structure, "is_pml") and structure.is_pml:
            continue

        struct_depth = getattr(structure, "depth", default_depth)
        struct_z = getattr(structure, "z", 0)
        mesh_data = structure_to_3d_mesh(design, structure, struct_depth, struct_z)
        if not mesh_data:
            continue

        x, y, z = mesh_data["vertices"]
        i, j, k = mesh_data["faces"]

        material_permittivity = 1.0
        if hasattr(structure, "material") and structure.material:
            material_permittivity = getattr(structure.material, "permittivity", 1.0)

        material_key = None
        if hasattr(structure, "material") and structure.material:
            material_key = (
                getattr(structure.material, "permittivity", 1.0),
                getattr(structure.material, "permeability", 1.0),
                getattr(structure.material, "conductivity", 0.0),
            )

        if material_key not in material_colors:
            if (
                hasattr(structure, "color")
                and structure.color
                and structure.color != "none"
            ):
                material_colors[material_key] = structure.color
            else:
                material_colors[material_key] = _get_deterministic_color(color_index)
                color_index += 1

        color = material_colors[material_key]
        is_air_like = abs(material_permittivity - 1.0) < 0.1
        if isinstance(color, str) and color.startswith("#"):
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            if is_air_like:
                face_color = f"rgba({r},{g},{b},0.0)"
                opacity = 0.0
            else:
                face_color = f"rgba({r},{g},{b},1.0)"
                opacity = 1.0
        else:
            face_color = color
            opacity = 0.0 if is_air_like else 1.0

        hovertext = f"{structure.__class__.__name__}"
        if hasattr(structure, "material") and structure.material:
            if hasattr(structure.material, "name"):
                hovertext += f"<br>Material: {structure.material.name}"
            if hasattr(structure.material, "permittivity"):
                hovertext += f"<br>εᵣ = {structure.material.permittivity:.1f}"
            if (
                hasattr(structure.material, "permeability")
                and structure.material.permeability != 1.0
            ):
                hovertext += f"<br>μᵣ = {structure.material.permeability:.1f}"
            if (
                hasattr(structure.material, "conductivity")
                and structure.material.conductivity != 0.0
            ):
                hovertext += f"<br>σ = {structure.material.conductivity:.2e} S/m"

        # Create unique name for legend entry
        structure_name = f"{structure.__class__.__name__} {structure_index + 1}"
        if (
            hasattr(structure, "material")
            and structure.material
            and hasattr(structure.material, "name")
        ):
            structure_name += f" ({structure.material.name})"
        elif (
            hasattr(structure, "material")
            and structure.material
            and hasattr(structure.material, "permittivity")
        ):
            structure_name += f" (εᵣ={structure.material.permittivity:.1f})"

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=face_color,
                opacity=opacity,
                name=structure_name,
                showscale=False,
                hovertemplate=hovertext + "<extra></extra>",
                contour=dict(show=True, color="black", width=5),
                lighting=dict(
                    ambient=0.5, diffuse=0.5, fresnel=0.0, specular=0.5, roughness=1.0
                ),
                lightposition=dict(x=0, y=50, z=100),
                flatshading=True,
                showlegend=True,
            )
        )

        structure_index += 1

    scene = dict(
        xaxis=dict(
            title=dict(text=f"X ({unit})", font=dict(size=14, color="#34495e")),
            range=[0, design.width],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.3)",
            showbackground=True,
            backgroundcolor="rgba(248,249,250,0.8)",
            tickmode="array",
            tickvals=np.linspace(0, design.width, 6),
            ticktext=[f"{val*scale:.1f}" for val in np.linspace(0, design.width, 6)],
            tickfont=dict(size=11, color="#34495e"),
        ),
        yaxis=dict(
            title=dict(text=f"Y ({unit})", font=dict(size=14, color="#34495e")),
            range=[0, design.height],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.3)",
            showbackground=True,
            backgroundcolor="rgba(248,249,250,0.8)",
            tickmode="array",
            tickvals=np.linspace(0, design.height, 6),
            ticktext=[f"{val*scale:.1f}" for val in np.linspace(0, design.height, 6)],
            tickfont=dict(size=11, color="#34495e"),
        ),
        zaxis=dict(
            title=dict(text=f"Z ({unit})", font=dict(size=14, color="#34495e")),
            range=[0, design.depth if design.depth else default_depth],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.3)",
            showbackground=True,
            backgroundcolor="rgba(248,249,250,0.8)",
            tickmode="array",
            tickvals=np.linspace(0, design.depth if design.depth else default_depth, 6),
            ticktext=[
                f"{val*scale:.1f}"
                for val in np.linspace(
                    0, design.depth if design.depth else default_depth, 6
                )
            ],
            tickfont=dict(size=11, color="#34495e"),
        ),
        aspectmode="manual",
        aspectratio=dict(
            x=1,
            y=design.height / design.width if design.width > 0 else 1,
            z=(
                (design.depth if design.depth else default_depth) / design.width
                if design.width > 0
                else 1
            ),
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        ),
    )

    fig.update_layout(
        scene=scene,
        width=900,
        height=700,
        margin=dict(l=60, r=60, t=80, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=10),
        ),
    )

    if any(
        getattr(s, "z", 0) > 0
        for s in design.structures
        if not (hasattr(s, "is_pml") and s.is_pml)
    ):
        ground_x = [0, design.width, design.width, 0]
        ground_y = [0, 0, design.height, design.height]
        ground_z = [0, 0, 0, 0]
        fig.add_trace(
            go.Mesh3d(
                x=ground_x + ground_x,
                y=ground_y + ground_y,
                z=ground_z + [-default_depth * 0.05] * 4,
                i=[0, 0, 4, 4, 0, 1, 2, 3],
                j=[1, 3, 5, 7, 4, 5, 6, 7],
                k=[2, 2, 6, 6, 1, 2, 3, 0],
                color="rgba(220,220,220,0.3)",
                name="Ground Plane",
                showlegend=True,
                hoverinfo="skip",
                lighting=dict(
                    ambient=0.8, diffuse=0.2, fresnel=0.0, specular=0.0, roughness=1.0
                ),
                flatshading=True,
                contour=dict(show=True, color="black", width=5),
            )
        )

    fig.show()

def _add_monitor_to_3d_plot(fig, monitor, scale, unit, design=None, index=0):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    # Ensure monitor has vertices for 3D plot
    if (
        not hasattr(monitor, "vertices") or not monitor.vertices
    ) and design is not None:
        if hasattr(monitor, "_init_3d_monitor"):
            # Try to initialize as 3D if not already
            start = getattr(monitor, "start", (0, 0, 0))
            end = getattr(monitor, "end", None)
            normal = getattr(monitor, "plane_normal", "z")
            pos = getattr(monitor, "plane_position", 0)
            size = getattr(monitor, "size", None)
            monitor._init_3d_monitor(start, end, normal, pos, size)

    if not hasattr(monitor, "vertices") or not monitor.vertices:
        return

    vertices = monitor.vertices
    if len(vertices) < 3:
        return
    if len(vertices) == 4:
        faces_i = [0, 0]
        faces_j = [1, 2]
        faces_k = [2, 3]
    else:
        faces_i, faces_j, faces_k = [], [], []
        for i in range(1, len(vertices) - 1):
            faces_i.append(0)
            faces_j.append(i)
            faces_k.append(i + 1)
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]
    hovertext = f"Monitor ({monitor.monitor_type})"
    if hasattr(monitor, "size"):
        hovertext += f"<br>Size: {monitor.size[0]*scale:.2f} x {monitor.size[1]*scale:.2f} {unit}"
    if hasattr(monitor, "plane_normal"):
        hovertext += f"<br>Normal: {monitor.plane_normal}"
    if hasattr(monitor, "plane_position"):
        hovertext += f"<br>Position: {monitor.plane_position*scale:.2f} {unit}"

    # Create unique name for legend
    monitor_name = f"Monitor {index + 1}"
    if hasattr(monitor, "name") and monitor.name:
        monitor_name += f" ({monitor.name})"
    elif hasattr(monitor, "plane_normal"):
        monitor_name += f" ({monitor.plane_normal}-plane)"

    fig.add_trace(
        go.Mesh3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            i=faces_i,
            j=faces_j,
            k=faces_k,
            color="rgba(255,255,0,0.6)",
            opacity=0.75,
            name=monitor_name,
            hovertemplate=hovertext + "<extra></extra>",
            contour=dict(show=True, color="black", width=8),
            lighting=dict(
                ambient=0.8, diffuse=0.2, fresnel=0.0, specular=0.0, roughness=1.0
            ),
            flatshading=True,
            showlegend=True,
        )
    )


def _add_mode_source_to_3d_plot(fig, source, scale, unit, design=None, index=0):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    # Determine vertices for the source plane
    if hasattr(source, "width") and (hasattr(source, "height") or design is not None):
        # Handle ModeSource with width and optional height
        center = getattr(source, "center", getattr(source, "position", (0, 0, 0)))
        # Ensure center is 3D
        if len(center) == 2:
            z_default = design.depth / 2 if design and design.depth else 0
            center = (center[0], center[1], z_default)

        width = source.width
        default_height = design.depth if design and design.depth else source.width
        height = getattr(source, "height", default_height)

        # Use orientation or direction to determine plane
        direction = getattr(source, "direction", "+x")
        orientation = getattr(
            source,
            "orientation",
            (
                "yz"
                if direction in ["+x", "-x"]
                else "xz" if direction in ["+y", "-y"] else "xy"
            ),
        )

        if orientation == "yz":
            vertices = [
                (center[0], center[1] - width / 2, center[2] - height / 2),
                (center[0], center[1] + width / 2, center[2] - height / 2),
                (center[0], center[1] + width / 2, center[2] + height / 2),
                (center[0], center[1] - width / 2, center[2] + height / 2),
            ]
        elif orientation == "xz":
            vertices = [
                (center[0] - width / 2, center[1], center[2] - height / 2),
                (center[0] + width / 2, center[1], center[2] - height / 2),
                (center[0] + width / 2, center[1], center[2] + height / 2),
                (center[0] - width / 2, center[1], center[2] + height / 2),
            ]
        else:
            vertices = [
                (center[0] - width / 2, center[1] - height / 2, center[2]),
                (center[0] + width / 2, center[1] - height / 2, center[2]),
                (center[0] + width / 2, center[1] + height / 2, center[2]),
                (center[0] - width / 2, center[1] + height / 2, center[2]),
            ]
    elif hasattr(source, "start") and hasattr(source, "end"):
        start = source.start
        end = source.end
        # Ensure they are 3D
        if len(start) == 2:
            start = (start[0], start[1], 0)
        if len(end) == 2:
            end = (end[0], end[1], start[2])

        line_vec = np.array([end[0] - start[0], end[1] - start[1], end[2] - start[2]])
        line_length = np.linalg.norm(line_vec)
        if line_length == 0:
            center = start
            plane_size = getattr(source, "wavelength", 1e-6) * 0.5
            vertices = [
                (center[0] - plane_size / 2, center[1] - plane_size / 2, center[2]),
                (center[0] + plane_size / 2, center[1] - plane_size / 2, center[2]),
                (center[0] + plane_size / 2, center[1] + plane_size / 2, center[2]),
                (center[0] - plane_size / 2, center[1] + plane_size / 2, center[2]),
            ]
        else:
            line_unit = line_vec / line_length
            temp_vec = (
                np.array([0, 0, 1]) if abs(line_unit[2]) < 0.9 else np.array([1, 0, 0])
            )
            perp1 = np.cross(line_unit, temp_vec)
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(line_unit, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            plane_size = max(line_length, getattr(source, "wavelength", 1e-6) * 0.5)
            center = np.array(
                [
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2,
                    (start[2] + end[2]) / 2,
                ]
            )
            vertices = [
                center - perp1 * plane_size / 2 - perp2 * plane_size / 2,
                center + perp1 * plane_size / 2 - perp2 * plane_size / 2,
                center + perp1 * plane_size / 2 + perp2 * plane_size / 2,
                center - perp1 * plane_size / 2 + perp2 * plane_size / 2,
            ]
            vertices = [(v[0], v[1], v[2]) for v in vertices]
    else:
        # Fallback for point sources or unknown types
        center = getattr(source, "position", getattr(source, "center", (0, 0, 0)))
        if len(center) == 2:
            center = (center[0], center[1], 0)
        plane_size = getattr(source, "wavelength", 1e-6) * 0.5
        vertices = [
            (center[0] - plane_size / 2, center[1] - plane_size / 2, center[2]),
            (center[0] + plane_size / 2, center[1] - plane_size / 2, center[2]),
            (center[0] + plane_size / 2, center[1] + plane_size / 2, center[2]),
            (center[0] - plane_size / 2, center[1] + plane_size / 2, center[2]),
        ]

    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]
    faces_i = [0, 0]
    faces_j = [1, 2]
    faces_k = [2, 3]
    hovertext = f"ModeSource"
    if hasattr(source, "wavelength"):
        hovertext += f"<br>Wavelength: {source.wavelength*scale*1e6:.0f} nm"
    if hasattr(source, "direction"):
        hovertext += f"<br>Direction: {source.direction}"
    if hasattr(source, "num_modes"):
        hovertext += f"<br>Modes: {source.num_modes}"
    if hasattr(source, "effective_indices") and len(source.effective_indices) > 0:
        hovertext += f"<br>n_eff: {source.effective_indices[0].real:.3f}"

    # Create unique name for legend
    source_name = f"ModeSource {index + 1}"
    if hasattr(source, "direction"):
        source_name += f" ({source.direction})"

    fig.add_trace(
        go.Mesh3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            i=faces_i,
            j=faces_j,
            k=faces_k,
            color="rgba(220,20,60,0.6)",
            opacity=0.75,
            name=source_name,
            hovertemplate=hovertext + "<extra></extra>",
            contour=dict(show=True, color="darkred", width=8),
            lighting=dict(
                ambient=0.8, diffuse=0.2, fresnel=0.0, specular=0.0, roughness=1.0
            ),
            flatshading=True,
            showlegend=True,
        )
    )

    _add_direction_arrow_to_3d_plot(fig, source, vertices)


def _add_direction_arrow_to_3d_plot(fig, source, plane_vertices):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    center = np.array(
        [
            sum(v[0] for v in plane_vertices) / len(plane_vertices),
            sum(v[1] for v in plane_vertices) / len(plane_vertices),
            sum(v[2] for v in plane_vertices) / len(plane_vertices),
        ]
    )
    arrow_length = source.wavelength * 0.8
    if source.direction == "+x":
        arrow_end = center + np.array([arrow_length, 0, 0])
    elif source.direction == "-x":
        arrow_end = center + np.array([-arrow_length, 0, 0])
    elif source.direction == "+y":
        arrow_end = center + np.array([0, arrow_length, 0])
    elif source.direction == "-y":
        arrow_end = center + np.array([0, -arrow_length, 0])
    elif source.direction == "+z":
        arrow_end = center + np.array([0, 0, arrow_length])
    elif source.direction == "-z":
        arrow_end = center + np.array([0, 0, -arrow_length])
    else:
        arrow_end = center + np.array([arrow_length, 0, 0])

    fig.add_trace(
        go.Scatter3d(
            x=[center[0], arrow_end[0]],
            y=[center[1], arrow_end[1]],
            z=[center[2], arrow_end[2]],
            mode="lines",
            line=dict(color="darkred", width=8),
            name="Propagation Direction",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Cone(
            x=[arrow_end[0]],
            y=[arrow_end[1]],
            z=[arrow_end[2]],
            u=[arrow_end[0] - center[0]],
            v=[arrow_end[1] - center[1]],
            w=[arrow_end[2] - center[2]],
            sizemode="absolute",
            sizeref=arrow_length * 0.3,
            colorscale=[[0, "darkred"], [1, "darkred"]],
            showscale=False,
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _add_gaussian_source_to_3d_plot(fig, source, scale, unit, index=0):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    position = source.position
    radius = source.width * 0.5
    phi = np.linspace(0, 2 * np.pi, 20)
    theta = np.linspace(0, np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    x = position[0] + radius * np.sin(theta) * np.cos(phi)
    y = position[1] + radius * np.sin(theta) * np.sin(phi)
    z = position[2] + radius * np.cos(theta)
    hovertext = f"GaussianSource"
    hovertext += f"<br>Position: ({position[0]*scale:.2f}, {position[1]*scale:.2f}, {position[2]*scale:.2f}) {unit}"
    hovertext += f"<br>Width: {source.width*scale:.2f} {unit}"

    # Create unique name for legend
    source_name = f"GaussianSource {index + 1}"

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[[0, "rgba(255,69,0,0.7)"], [1, "rgba(255,69,0,0.7)"]],
            opacity=0.7,
            name=source_name,
            hovertemplate=hovertext + "<extra></extra>",
            showscale=False,
            showlegend=True,
        )
    )

def structure_to_3d_mesh(design, structure, depth, z_offset=0):
    if not hasattr(structure, "vertices") or not structure.vertices:
        return None
    if depth is None:
        depth = 0.1 * min(design.width, design.height)
    vertices_2d = (
        structure._vertices_2d()
        if hasattr(structure, "_vertices_2d")
        else [(v[0], v[1]) for v in structure.vertices]
    )
    n_vertices = len(vertices_2d)
    if n_vertices < 3:
        return None
    actual_z = z_offset
    if hasattr(structure, "z") and structure.z is not None:
        actual_z = structure.z
    elif hasattr(structure, "position") and len(structure.position) > 2:
        actual_z = structure.position[2]

    interior_paths = getattr(structure, "interiors", [])
    if interior_paths and len(interior_paths) > 0:
        return _triangulate_polygon_with_holes(
            vertices_2d, interior_paths, depth, actual_z
        )

    try:
        triangles = _robust_triangulation(vertices_2d)
    except Exception:
        triangles = _fallback_triangulation(vertices_2d)
    if not triangles:
        return None

    vertices_3d = []
    for x, y in vertices_2d:
        vertices_3d.append([x, y, actual_z])
    for x, y in vertices_2d:
        vertices_3d.append([x, y, actual_z + depth])
    x_coords = [v[0] for v in vertices_3d]
    y_coords = [v[1] for v in vertices_3d]
    z_coords = [v[2] for v in vertices_3d]
    faces_i, faces_j, faces_k = [], [], []
    for tri in triangles:
        faces_i.append(tri[0])
        faces_j.append(tri[2])
        faces_k.append(tri[1])
    for tri in triangles:
        faces_i.append(tri[0] + n_vertices)
        faces_j.append(tri[1] + n_vertices)
        faces_k.append(tri[2] + n_vertices)
    for i in range(n_vertices):
        next_i = (i + 1) % n_vertices
        faces_i.append(i)
        faces_j.append(next_i)
        faces_k.append(next_i + n_vertices)
        faces_i.append(i)
        faces_j.append(next_i + n_vertices)
        faces_k.append(i + n_vertices)
    return {
        "vertices": (x_coords, y_coords, z_coords),
        "faces": (faces_i, faces_j, faces_k),
    }


def _robust_triangulation(vertices_2d):
    if len(vertices_2d) < 3:
        return []
    if len(vertices_2d) == 3:
        return [(0, 1, 2)]
    if len(vertices_2d) == 4:
        return [(0, 1, 2), (0, 2, 3)]
    try:
        import scipy.spatial

        points = np.array(vertices_2d)
        tri = scipy.spatial.Delaunay(points)
        valid_triangles = []
        for triangle in tri.simplices:
            centroid = np.mean(points[triangle], axis=0)
            if _point_in_polygon_2d(centroid[0], centroid[1], vertices_2d):
                v1 = points[triangle[1]] - points[triangle[0]]
                v2 = points[triangle[2]] - points[triangle[0]]
                if np.cross(v1, v2) > 0:
                    valid_triangles.append(tuple(triangle))
                else:
                    valid_triangles.append((triangle[0], triangle[2], triangle[1]))
        return valid_triangles
    except ImportError:
        return _ear_clipping_triangulation(vertices_2d)


def _ear_clipping_triangulation(vertices):
    if len(vertices) < 3:
        return []

    def is_ear(i, j, k, vertices, indices):
        a = np.array(vertices[indices[i]])
        b = np.array(vertices[indices[j]])
        c = np.array(vertices[indices[k]])
        ab = b - a
        cb = b - c
        cross = np.cross(ab, cb)
        if cross <= 0:
            return False
        triangle = [a, b, c]
        for m in range(len(indices)):
            if m not in [i, j, k]:
                p = np.array(vertices[indices[m]])
                if _point_in_triangle(p, a, b, c):
                    return False
        return True

    indices = list(range(len(vertices)))
    triangles = []
    while len(indices) > 3:
        n = len(indices)
        ear_found = False
        for j in range(n):
            i = (j - 1) % n
            k = (j + 1) % n
            if is_ear(i, j, k, vertices, indices):
                triangles.append((indices[i], indices[j], indices[k]))
                indices.pop(j)
                ear_found = True
                break
        if not ear_found:
            break
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    return triangles


def _fallback_triangulation(vertices_2d):
    if len(vertices_2d) < 3:
        return []
    if len(vertices_2d) == 3:
        return [(0, 1, 2)]
    if len(vertices_2d) == 4:
        return [(0, 1, 2), (0, 2, 3)]
    try:
        return _convex_hull_triangulation(vertices_2d)
    except Exception:
        triangles = []
        for i in range(1, len(vertices_2d) - 1):
            triangles.append((0, i, i + 1))
        return triangles


def _convex_hull_triangulation(vertices_2d):
    import scipy.spatial

    points = np.array(vertices_2d)
    hull = scipy.spatial.ConvexHull(points)
    hull_vertices = hull.vertices
    if len(hull_vertices) == len(vertices_2d):
        triangles = []
        for i in range(1, len(vertices_2d) - 1):
            triangles.append((0, i, i + 1))
        return triangles
    else:
        return _decompose_polygon(vertices_2d)


def _decompose_polygon(vertices_2d):
    triangles = []
    n = len(vertices_2d)
    center_idx = 0
    for i in range(1, n - 1):
        next_i = i + 1
        if _is_valid_triangle(vertices_2d, center_idx, i, next_i):
            triangles.append((center_idx, i, next_i))
    return triangles if triangles else [(0, 1, 2)]


def _is_valid_triangle(vertices, i, j, k):
    p1, p2, p3 = vertices[i], vertices[j], vertices[k]
    area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    return area > 1e-10


def _point_in_polygon_2d(x, y, polygon_vertices):
    n = len(polygon_vertices)
    inside = False
    p1x, p1y = polygon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _point_in_triangle(point, a, b, c):
    x, y = point
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denominator) < 1e-10:
        return False
    a_coord = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
    b_coord = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
    c_coord = 1 - a_coord - b_coord
    return a_coord >= 0 and b_coord >= 0 and c_coord >= 0


def _triangulate_polygon_with_holes(exterior_vertices, interior_paths, depth, z_offset):
    n_ext = len(exterior_vertices)
    total_vertices = n_ext
    all_vertices_2d = list(exterior_vertices)
    interior_starts = []
    for interior in interior_paths:
        interior_starts.append(total_vertices)
        for v in interior:
            all_vertices_2d.append((v[0], v[1]))
        total_vertices += len(interior)
    vertices_3d = []
    for x, y in all_vertices_2d:
        vertices_3d.append([x, y, z_offset])
    for x, y in all_vertices_2d:
        vertices_3d.append([x, y, z_offset + depth])
    x_coords = [v[0] for v in vertices_3d]
    y_coords = [v[1] for v in vertices_3d]
    z_coords = [v[2] for v in vertices_3d]
    faces_i, faces_j, faces_k = [], [], []
    if len(interior_paths) == 1 and len(interior_paths[0]) == len(exterior_vertices):
        inner_start = interior_starts[0]
        for i in range(n_ext):
            next_i = (i + 1) % n_ext
            outer_i = i
            outer_next = next_i
            inner_i = inner_start + i
            inner_next = inner_start + next_i
            faces_i.append(outer_i)
            faces_j.append(outer_next)
            faces_k.append(inner_i)
            faces_i.append(outer_next)
            faces_j.append(inner_next)
            faces_k.append(inner_i)
            top_offset = total_vertices
            faces_i.append(outer_i + top_offset)
            faces_j.append(inner_i + top_offset)
            faces_k.append(outer_next + top_offset)
            faces_i.append(outer_next + top_offset)
            faces_j.append(inner_i + top_offset)
            faces_k.append(inner_next + top_offset)
        for i in range(n_ext):
            next_i = (i + 1) % n_ext
            faces_i.append(i)
            faces_j.append(next_i)
            faces_k.append(i + total_vertices)
            faces_i.append(next_i)
            faces_j.append(next_i + total_vertices)
            faces_k.append(i + total_vertices)
        for i in range(len(interior_paths[0])):
            next_i = (i + 1) % len(interior_paths[0])
            inner_i = inner_start + i
            inner_next = inner_start + next_i
            faces_i.append(inner_i + total_vertices)
            faces_j.append(inner_next + total_vertices)
            faces_k.append(inner_i)
            faces_i.append(inner_i)
            faces_j.append(inner_next + total_vertices)
            faces_k.append(inner_next)
    return {
        "vertices": (x_coords, y_coords, z_coords),
        "faces": (faces_i, faces_j, faces_k),
    }

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
