"""3D plotly visualization for designs."""

import numpy as np

from beamz.visual.data import _get_deterministic_color
from beamz.visual.helpers import display_status, get_si_scale_and_label


def show_design_3d(
    design,
    unify_structures=True,
    max_vertices_for_unification=50,
    sources=None,
    monitors=None,
):
    """Display the design using 3D plotly visualization."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise RuntimeError(
            "Plotly is required for 3D visualization. Install with: pip install plotly"
        )

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
    for idx, monitor in enumerate(monitors or []):
        _add_monitor_to_3d_plot(fig, monitor, scale, unit, design=design, index=idx)

    # Add sources
    for idx, source in enumerate(sources or []):
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
                hovertext += f"<br>\u03b5\u1d63 = {structure.material.permittivity:.1f}"
            if (
                hasattr(structure.material, "permeability")
                and structure.material.permeability != 1.0
            ):
                hovertext += f"<br>\u03bc\u1d63 = {structure.material.permeability:.1f}"
            if (
                hasattr(structure.material, "conductivity")
                and structure.material.conductivity != 0.0
            ):
                hovertext += f"<br>\u03c3 = {structure.material.conductivity:.2e} S/m"

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
            structure_name += f" (\u03b5\u1d63={structure.material.permittivity:.1f})"

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
        center = getattr(source, "center", getattr(source, "position", (0, 0, 0)))
        if len(center) == 2:
            z_default = design.depth / 2 if design and design.depth else 0
            center = (center[0], center[1], z_default)

        width = source.width
        default_height = design.depth if design and design.depth else source.width
        height = getattr(source, "height", default_height)

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

    triangles = _triangulate(vertices_2d)
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


def _triangulate(vertices_2d):
    """Triangulate a simple polygon using scipy Delaunay with centroid filtering."""
    n = len(vertices_2d)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]
    if n == 4:
        return [(0, 1, 2), (0, 2, 3)]

    import scipy.spatial

    points = np.array(vertices_2d)
    try:
        tri = scipy.spatial.Delaunay(points)
    except Exception:
        # Degenerate geometry -- fall back to fan triangulation
        return [(0, i, i + 1) for i in range(1, n - 1)]

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
