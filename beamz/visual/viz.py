import numpy as np

from beamz.visual.helpers import display_status, get_si_scale_and_label

# Optional plotting backends are imported inside functions to avoid hard deps


def get_twilight_zero_cmap():
    """Get a custom colormap similar to twilight with black at zero and white at edges.

    Returns:
        matplotlib.colors.Colormap: A custom 7-color diverging colormap with
        white at edges, twilight-like colors in between, and black at center.
    """
    from matplotlib.colors import LinearSegmentedColormap

    # 7 colors total: white -> purple -> blue -> cyan -> black -> yellow -> orange -> red -> white
    # Similar to twilight but with black at center and white at edges
    colors = [
        (1.0, 1.0, 1.0),  # White (edge, negative)
        (0.2, 0.3, 0.8),  # Purple
        (0.1, 0.1, 0.5),  # Blue
        (0.1, 0.1, 0.1),  # Black (center, zero)
        (0.5, 0.1, 0.1),  # Orange
        (0.8, 0.3, 0.2),  # Red
        (1.0, 1.0, 1.0),  # White (edge, positive)
    ]

    return LinearSegmentedColormap.from_list("twilight_zero", colors, N=256)


# Register the custom colormap
def _register_custom_colormaps():
    """Register custom colormaps with matplotlib."""
    import matplotlib.pyplot as plt

    try:
        # Check if already registered
        if "twilight_zero" not in plt.colormaps():
            cmap = get_twilight_zero_cmap()
            plt.colormaps.register(cmap, name="twilight_zero")
    except Exception:
        pass  # If registration fails, we'll create it on-the-fly when needed


# Register on import
_register_custom_colormaps()


def is_jupyter_environment():
    """Detect if code is running in a Jupyter notebook/lab environment.

    Returns:
        bool: True if running in Jupyter, False otherwise
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        # ZMQInteractiveShell is used by Jupyter notebook/lab
        if shell_name == "ZMQInteractiveShell":
            return True
        # Check for Google Colab
        if "google.colab" in str(shell.__class__):
            return True
        return False
    except (ImportError, NameError):
        return False


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


# =============================
# FDTD visualization utilities
# =============================


def plot_fdtd_field(
    fdtd, field: str = "Ez", t: float = None, z_slice: int = None
) -> None:
    """Plot an FDTD field at a given time with proper scaling and units."""
    import matplotlib.pyplot as plt

    if len(fdtd.results["t"]) == 0:
        current_field = getattr(fdtd, field)
        current_t = fdtd.t
    else:
        t_idx = int(np.argmin(np.abs(np.array(fdtd.results["t"]) - t)))
        current_field = fdtd.results[field][t_idx]
        current_t = fdtd.results["t"][t_idx]

    if hasattr(current_field, "device"):
        current_field = fdtd.backend.to_numpy(current_field)

    if fdtd.is_3d and len(current_field.shape) == 3:
        if z_slice is None:
            z_slice = current_field.shape[0] // 2
        current_field = current_field[z_slice, :, :]
        slice_info = f" (z-slice {z_slice})"
    else:
        slice_info = ""

    if np.iscomplexobj(current_field):
        current_field = np.real(current_field)
        field_label = f"Re({field}){slice_info}"
    else:
        field_label = field + slice_info

    scale, unit = get_si_scale_and_label(max(fdtd.design.width, fdtd.design.height))
    grid_height, grid_width = current_field.shape
    aspect_ratio = grid_width / grid_height
    base_size = 6
    figsize = (
        (base_size * aspect_ratio, base_size)
        if aspect_ratio > 1
        else (base_size, base_size / aspect_ratio)
    )

    plt.figure(figsize=figsize)
    plt.imshow(
        current_field,
        origin="lower",
        extent=(0, fdtd.design.width, 0, fdtd.design.height),
        cmap="RdBu",
        aspect="equal",
        interpolation="bicubic",
    )
    plt.colorbar(label=f"{field_label}")
    plt.title(f"{field_label} Field at t = {current_t:.2e} s")
    plt.xlabel(f"X ({unit})")
    plt.ylabel(f"Y ({unit})")
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")

    try:
        tmp_design = fdtd.design.copy()
        tmp_design.unify_polygons()
        overlay_structures = tmp_design.structures
    except Exception:
        overlay_structures = fdtd.design.structures
    for structure in overlay_structures:
        if hasattr(structure, "is_pml") and structure.is_pml:
            structure.add_to_plot(
                plt.gca(),
                edgecolor="black",
                linestyle="--",
                facecolor="none",
                alpha=0.5,
            )
        elif hasattr(structure, "vertices") and getattr(structure, "vertices", None):
            structure.add_to_plot(
                plt.gca(), facecolor="none", edgecolor="black", linestyle="-"
            )
    for source in fdtd.design.sources:
        if hasattr(source, "add_to_plot"):
            source.add_to_plot(plt.gca())
    for monitor in fdtd.design.monitors:
        if hasattr(monitor, "add_to_plot"):
            monitor.add_to_plot(plt.gca(), edgecolor="black")
    plt.tight_layout()
    plt.show()


def animate_fdtd_live(fdtd, field_data=None, field="Ez", axis_scale=None, z_slice=None):
    """Animate FDTD field in real time using matplotlib animation."""
    import matplotlib.pyplot as plt

    if field_data is None:
        field_data = fdtd.backend.to_numpy(getattr(fdtd, field))

    if fdtd.is_3d and len(field_data.shape) == 3:
        if z_slice is None:
            z_slice = field_data.shape[0] // 2
        field_data = field_data[z_slice, :, :]
        slice_info = f" (z-slice {z_slice})"
    else:
        slice_info = ""

    # Always visualize Ez field amplitude for live view
    quantity = "field"
    if quantity == "power":
        # Compute instantaneous power magnitude Sx,Sy (2D) and plot W/µm²
        Ez_np = field_data
        Hx_raw = (
            fdtd.backend.to_numpy(getattr(fdtd, "Hx")) if hasattr(fdtd, "Hx") else None
        )
        Hy_raw = (
            fdtd.backend.to_numpy(getattr(fdtd, "Hy")) if hasattr(fdtd, "Hy") else None
        )
        if np.iscomplexobj(Ez_np):
            Ez_real = np.real(Ez_np)
            Ez_imag = np.imag(Ez_np)
        else:
            Ez_real = Ez_np
            Ez_imag = 0.0
        if Hx_raw is None or Hy_raw is None:
            current_field = np.zeros_like(Ez_real)
        else:
            if np.iscomplexobj(Hx_raw) or np.iscomplexobj(Hy_raw):
                Hx_full = np.zeros_like(Ez_real, dtype=np.complex128)
                Hy_full = np.zeros_like(Ez_real, dtype=np.complex128)
            else:
                Hx_full = np.zeros_like(Ez_real)
                Hy_full = np.zeros_like(Ez_real)
            Hx_full[:, :-1] = Hx_raw
            Hy_full[:-1, :] = Hy_raw
            if (
                np.iscomplexobj(Hx_full)
                or np.iscomplexobj(Hy_full)
                or np.iscomplexobj(Ez_np)
            ):
                Hx_real = np.real(Hx_full)
                Hx_imag = np.imag(Hx_full)
                Hy_real = np.real(Hy_full)
                Hy_imag = np.imag(Hy_full)
                Sx = -Ez_real * Hy_real - Ez_imag * Hy_imag
                Sy = Ez_real * Hx_real + Ez_imag * Hx_imag
            else:
                Sx = -Ez_real * Hy_full
                Sy = Ez_real * Hx_full
            power_si = Sx**2 + Sy**2  # W^2/m^4 (magnitude squared); for visualization
            # Use linear power density magnitude for color scaling (W/m^2)
            power_mag = np.sqrt(power_si)
            # Convert to W/µm² for display
            power_um2 = power_mag * (1.0e-12)
            current_field = power_um2
        if axis_scale is None:
            # Dynamic scaling: compute from current field every frame
            # Use 99th percentile for power to avoid outliers
            ax_min = 0.0
            ax_max = float(
                np.percentile(current_field, 99) or np.max(current_field) or 1e-9
            )
        else:
            ax_min, ax_max = axis_scale
        cbar_label = f"Power Density (W/µm²)"
    else:
        if np.iscomplexobj(field_data):
            field_data = np.real(field_data)
        # Convert Ez from V/m to V/µm for display
        current_field = field_data * 1.0e-6

        if axis_scale is None:
            # Dynamic scaling: compute from current field every frame
            # Ignore fdtd._axis_scale for truly adaptive behavior
            field_abs = np.abs(current_field)
            # Use 99th percentile instead of max to avoid extreme values at source
            # dominating the colormap
            amax = float(np.percentile(field_abs, 99) or 1.0)
            # Ensure at least some visible range
            if amax < 1e-10:
                amax = float(np.max(field_abs) or 1.0)
            ax_min, ax_max = -amax, amax
        else:
            # Fixed scaling: use the provided axis_scale
            amax = float(max(abs(axis_scale[0]), abs(axis_scale[1])))
            if not np.isfinite(amax) or amax <= 0:
                amax = float(np.max(np.abs(current_field)) or 1.0)
            ax_min, ax_max = -amax, amax
        cbar_label = f"{field}{slice_info} (V/µm)"

    if fdtd.fig is not None and plt.fignum_exists(fdtd.fig.number):
        fdtd.im.set_array(current_field)
        fdtd.im.set_clim(vmin=ax_min, vmax=ax_max)

        # Update colorbar by directly modifying its properties (fast method)
        if hasattr(fdtd, "colorbar") and fdtd.colorbar is not None:
            try:
                # Update the colorbar's norm to match the new limits
                fdtd.colorbar.mappable.set_clim(vmin=ax_min, vmax=ax_max)
                # Force colorbar to recompute ticks
                fdtd.colorbar.update_ticks()
                fdtd.colorbar.draw_all()
            except:
                pass

        fdtd.ax.set_title(f"t = {fdtd.t:.2e} s{slice_info}")
        fdtd.fig.canvas.draw_idle()
        fdtd.fig.canvas.flush_events()
        return

    grid_height, grid_width = current_field.shape
    aspect_ratio = grid_width / grid_height
    base_size = 5
    figsize = (
        (base_size * aspect_ratio * 1.2, base_size)
        if aspect_ratio > 1
        else (base_size * 1.2, base_size / aspect_ratio)
    )
    fdtd.fig, fdtd.ax = plt.subplots(figsize=figsize)
    fdtd.im = fdtd.ax.imshow(
        current_field,
        origin="lower",
        extent=(0, fdtd.design.width, 0, fdtd.design.height),
        cmap="RdBu",
        aspect="equal",
        interpolation="bicubic",
        vmin=ax_min,
        vmax=ax_max,
    )
    fdtd.colorbar = plt.colorbar(
        fdtd.im, orientation="vertical", aspect=30, extend="both"
    )
    fdtd.colorbar.set_label(cbar_label)

    try:
        tmp_design = fdtd.design.copy()
        tmp_design.unify_polygons()
        overlay_structures = tmp_design.structures
    except Exception:
        overlay_structures = fdtd.design.structures
    for structure in overlay_structures:
        if hasattr(structure, "is_pml") and structure.is_pml:
            structure.add_to_plot(
                fdtd.ax, edgecolor="black", linestyle="--", facecolor="none", alpha=0.5
            )
        elif hasattr(structure, "vertices") and getattr(structure, "vertices", None):
            structure.add_to_plot(
                fdtd.ax, facecolor="none", edgecolor="black", linestyle="-"
            )
    # Draw sources from both design and fdtd.sources list
    all_sources = list(fdtd.design.sources) if hasattr(fdtd.design, "sources") else []
    if hasattr(fdtd, "sources"):
        all_sources.extend(fdtd.sources)
    for source in all_sources:
        if hasattr(source, "add_to_plot"):
            source.add_to_plot(fdtd.ax)

    for monitor in fdtd.design.monitors:
        if hasattr(monitor, "add_to_plot"):
            monitor.add_to_plot(fdtd.ax, edgecolor="black")

    max_dim = max(fdtd.design.width, fdtd.design.height)
    if max_dim >= 1e-3:
        scale, unit = 1e3, "mm"
    elif max_dim >= 1e-6:
        scale, unit = 1e6, "µm"
    elif max_dim >= 1e-9:
        scale, unit = 1e9, "nm"
    else:
        scale, unit = 1e12, "pm"
    plt.xlabel(f"X ({unit})")
    plt.ylabel(f"Y ({unit})")
    fdtd.ax.xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    fdtd.ax.yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def animate_manual_field(
    field_array,
    context=None,
    *,
    axis_scale=None,
    extent=None,
    cmap="RdBu",
    percentile=99,
    title=None,
    units="V/µm",
    pause=0.002,
    auto_interval=4,
    smoothing=0.25,
    design=None,
    boundaries=None,
    show_structures=True,
    show_sources=True,
    show_monitors=True,
    clean_visualization=False,
    wavelength=None,
    line_color="gray",
    line_opacity=0.5,
    plane_2d="xy",
    interpolation="bicubic",
):
    """Create or update a live Matplotlib view of a 2D field array.

    Args:
        field_array: 2D numeric array to visualise (already converted to desired units).
        context: Optional dict (``{'fig','ax','im','cbar','frame','auto_scale'}``) returned by a previous call.
        axis_scale: Optional tuple/list ``(vmin, vmax)`` for fixed scaling.
        extent: Optional Matplotlib extent tuple ``(xmin, xmax, ymin, ymax)``.
        cmap: Matplotlib colormap to use.
        percentile: Percentile used for auto scaling when ``axis_scale`` not provided.
        title: Optional title string for the plot.
        units: Axis label for the colour bar.
        pause: Seconds to pause after drawing (keeps UI responsive).
        auto_interval: Recompute auto scaling every N frames when ``axis_scale`` is ``None``.
        smoothing: Exponential smoothing factor (0-1) applied to auto scale updates.
        design: Optional FDTD design object to overlay structures, sources, and monitors.
        boundaries: Optional list of boundary objects (PML, ABC, etc.) to visualize.
        show_structures: Boolean to control if design structures are overlaid.
        show_sources: Boolean to control if design sources are overlaid.
        show_monitors: Boolean to control if design monitors are overlaid.
        clean_visualization: If True, hide axes, title, and colorbar (only show field and structures).
        wavelength: Optional wavelength for scale bar calculation (if None, uses design-based calculation).
        line_color: Color for structure and PML boundary outlines (default: 'gray').
        line_opacity: Opacity/transparency of structure and PML boundary outlines (0.0 to 1.0, default: 0.5).
        plane_2d: Plane of simulation ('xy', 'yz', 'xz') to determine axis labels.
        interpolation: Interpolation method for imshow ('nearest', 'bilinear', 'bicubic', etc.).

    Returns:
        context dict containing references to the Matplotlib objects for reuse.
    """
    import matplotlib.pyplot as plt

    data = np.asarray(field_array, dtype=float)
    if data.size == 0:
        return context

    if context is None:
        context = {}

    if axis_scale is None:
        frame = context.get("frame", 0)
        use_cached = ("auto_scale" in context) and (frame % auto_interval != 0)
        if use_cached:
            vmax = context["auto_scale"]
        else:
            abs_data = np.abs(data)
            if abs_data.size > 10:
                vmax = np.percentile(abs_data, percentile)
                # If percentile scaling is too aggressive (e.g. for localized sources), fall back to max
                if vmax < 0.01 * np.max(abs_data):
                    vmax = float(np.max(abs_data))
            else:
                vmax = float(np.max(abs_data) or 1.0)

            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 0.0  # Field is zero

            # If we are transitioning from zero or very small to something larger,
            # or if the current vmax is much smaller than previous auto_scale,
            # reset auto_scale to the current vmax immediately instead of smoothing
            # This makes the visualization much more reactive to the start of a pulse.
            if "auto_scale" in context:
                prev_vmax = context["auto_scale"]
                # If current field is 10x larger than previous scale, or previous scale was 'zero' (1.0 default)
                if (vmax > 5.0 * prev_vmax) or (prev_vmax == 1.0 and vmax > 0):
                    context["auto_scale"] = vmax
                else:
                    context["auto_scale"] = (
                        1.0 - smoothing
                    ) * prev_vmax + smoothing * vmax
            else:
                # First frame: if field is zero, default to 1.0, otherwise use current vmax
                context["auto_scale"] = vmax if vmax > 0 else 1.0

            vmax = context["auto_scale"]
            # Final fallback for visualization
            if vmax <= 0:
                vmax = 1.0

        vmin, vmax = -vmax, vmax
    else:
        vmin, vmax = axis_scale

    if context.get("im") is None:
        fig, ax = plt.subplots()
        # Handle custom colormap
        if cmap == "twilight_zero":
            try:
                actual_cmap = plt.get_cmap("twilight_zero")
            except ValueError:
                actual_cmap = get_twilight_zero_cmap()
        else:
            actual_cmap = cmap

        if extent is not None:
            im = ax.imshow(
                data,
                origin="lower",
                cmap=actual_cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                interpolation=interpolation,
            )
        else:
            im = ax.imshow(
                data,
                origin="lower",
                cmap=actual_cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation=interpolation,
            )

        # Determine field name from title if possible, or generic
        field_name = "Field"
        if title and " at t =" in title:
            field_name = title.split(" at t =")[0]

        if clean_visualization:
            ax.set_axis_off()
            cbar = None
        else:
            cbar = plt.colorbar(
                im, ax=ax, orientation="vertical", label=f"{field_name} ({units})"
            )
            if title:
                ax.set_title(title)

        if design is not None and show_structures:
            try:
                tmp_design = design.copy()
                tmp_design.unify_polygons()
                overlay_structures = tmp_design.structures
            except Exception:
                overlay_structures = getattr(design, "structures", [])
            for structure in overlay_structures or []:
                if hasattr(structure, "is_pml") and structure.is_pml:
                    structure.add_to_plot(
                        ax,
                        edgecolor=line_color,
                        linestyle="--",
                        facecolor="none",
                        alpha=line_opacity,
                    )
                elif hasattr(structure, "vertices") and getattr(
                    structure, "vertices", None
                ):
                    structure.add_to_plot(
                        ax,
                        facecolor="none",
                        edgecolor=line_color,
                        linestyle="-",
                        alpha=line_opacity,
                    )
            if show_sources:
                for source in getattr(design, "sources", []) or []:
                    if hasattr(source, "add_to_plot"):
                        source.add_to_plot(ax)
            if show_monitors:
                for monitor in getattr(design, "monitors", []) or []:
                    if hasattr(monitor, "add_to_plot"):
                        monitor.add_to_plot(
                            ax, edgecolor=line_color, alpha=line_opacity
                        )

        # Draw PML boundaries if provided
        if boundaries:
            for boundary in boundaries:
                draw_boundary(
                    ax,
                    boundary,
                    design,
                    edgecolor=line_color,
                    linestyle=":",
                    alpha=line_opacity,
                )

        if design is not None and not clean_visualization:
            max_dim = max(design.width, design.height)
            scale, unit = get_si_scale_and_label(max_dim)

            # Set axis labels based on plane
            xlabel, ylabel = "X", "Y"
            if plane_2d == "yz":
                xlabel, ylabel = "Y", "Z"
            elif plane_2d == "xz":
                xlabel, ylabel = "X", "Z"

            ax.set_xlabel(f"{xlabel} ({unit})")
            ax.set_ylabel(f"{ylabel} ({unit})")
            ax.xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")

        if clean_visualization and design is not None:
            # Add scale bar in bottom-right corner
            max_dim = max(design.width, design.height)
            scale_factor, unit = get_si_scale_and_label(max_dim)

            # Calculate scale bar length: 2 * wavelength rounded up to next integer µm
            if wavelength is not None:
                # Convert wavelength to µm and calculate 2 * wavelength
                wavelength_um = wavelength * 1e6  # Convert from meters to µm
                scale_bar_length_um = 2 * wavelength_um
                # Round to nearest integer µm
                scale_bar_length_um = np.round(scale_bar_length_um)
                # Convert back to meters
                scale_bar_length = scale_bar_length_um * 1e-6
            else:
                # Fallback: use design-based calculation
                min_dim = min(design.width, design.height)
                scale_bar_fraction = 0.18
                scale_bar_length_physical = min_dim * scale_bar_fraction

                # Round to a nice number (round to nearest, not always down)
                if scale_bar_length_physical > 0:
                    order = 10 ** np.floor(np.log10(scale_bar_length_physical))
                    normalized = scale_bar_length_physical / order
                    if normalized <= 1.25:
                        nice_value = 1 * order
                    elif normalized <= 2.5:
                        nice_value = 2 * order
                    elif normalized <= 6:
                        nice_value = 5 * order
                    else:
                        nice_value = 10 * order
                    scale_bar_length = nice_value
                else:
                    scale_bar_length = min_dim * 0.15

            # Position in bottom-right corner with some margin
            margin_x = design.width * 0.1
            margin_y = design.height * 0.1
            x_start = design.width - scale_bar_length - margin_x
            x_end = design.width - margin_x
            y_pos = margin_y

            # Draw scale bar line (solid white bar, no caps)
            ax.plot(
                [x_start, x_end],
                [y_pos, y_pos],
                "w",
                linewidth=3,
                solid_capstyle="butt",
            )

            # Add text label below the bar
            label_y = y_pos - design.height * 0.02
            # If wavelength-based, always display in µm as integer
            if wavelength is not None:
                scale_bar_length_display_um = scale_bar_length * 1e6  # Convert to µm
                label_text = f"{int(scale_bar_length_display_um)} µm"
            else:
                scale_bar_length_display = scale_bar_length * scale_factor
                if scale_bar_length_display >= 1:
                    label_text = f"{scale_bar_length_display:.0f} {unit}"
                elif scale_bar_length_display >= 0.1:
                    label_text = f"{scale_bar_length_display:.1f} {unit}"
                else:
                    label_text = f"{scale_bar_length_display:.2f} {unit}"

            ax.text(
                (x_start + x_end) / 2,
                label_y,
                label_text,
                ha="center",
                va="top",
                color="white",
                fontsize=10,
            )

        if clean_visualization:
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        else:
            plt.tight_layout()
        plt.show(block=False)
        plt.pause(pause)
        context.update(
            {
                "fig": fig,
                "ax": ax,
                "im": im,
                "cbar": cbar,
                "frame": 1,
                "clean_visualization": clean_visualization,
                "wavelength": wavelength,
            }
        )
        context.setdefault("auto_scale", vmax if axis_scale is None else None)
        return context

    # Update existing plot
    clean_visualization = context.get("clean_visualization", False)
    im = context["im"]
    im.set_data(data)
    im.set_clim(vmin, vmax)
    if title and not clean_visualization:
        context["ax"].set_title(title)
    context["frame"] = context.get("frame", 0) + 1
    if context.get("cbar") is not None:
        context["cbar"].mappable.set_clim(vmin, vmax)
    fig = context["fig"]
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(pause)
    return context


def save_fdtd_animation(
    fdtd,
    field: str = "Ez",
    axis_scale=[-1, 1],
    filename="fdtd_animation.mp4",
    fps=60,
    frame_skip=4,
    clean_visualization=False,
):
    """Save an animation of FDTD results as an mp4 file."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if len(fdtd.results[field]) == 0:
        print(
            "No field data to animate. Make sure to run the simulation with save=True."
        )
        return
    total_frames = len(fdtd.results[field])
    frame_indices = range(0, total_frames, frame_skip)
    grid_height, grid_width = fdtd.results[field][0].shape
    aspect_ratio = grid_width / grid_height
    base_size = 5
    figsize = (
        (base_size * aspect_ratio * 1.2, base_size)
        if aspect_ratio > 1
        else (base_size * 1.2, base_size / aspect_ratio)
    )

    if clean_visualization:
        if aspect_ratio > 1:
            figsize = (base_size * aspect_ratio, base_size)
        else:
            figsize = (base_size, base_size / aspect_ratio)
        fig = plt.figure(figsize=figsize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        max_dim = max(fdtd.design.width, fdtd.design.height)
        scale, unit = get_si_scale_and_label(max_dim)

    im = ax.imshow(
        fdtd.results[field][0],
        origin="lower",
        extent=(0, fdtd.design.width, 0, fdtd.design.height),
        cmap="RdBu",
        aspect="equal",
        interpolation="bicubic",
        vmin=axis_scale[0],
        vmax=axis_scale[1],
    )
    if not clean_visualization:
        colorbar = plt.colorbar(im, orientation="vertical", aspect=30, extend="both")
        colorbar.set_label(f"{field}")

    try:
        tmp_design = fdtd.design.copy()
        tmp_design.unify_polygons()
        overlay_structures = tmp_design.structures
    except Exception:
        overlay_structures = fdtd.design.structures
    for structure in overlay_structures:
        if hasattr(structure, "is_pml") and structure.is_pml:
            structure.add_to_plot(
                ax, edgecolor="black", linestyle="--", facecolor="none", alpha=0.5
            )
        elif hasattr(structure, "vertices") and getattr(structure, "vertices", None):
            structure.add_to_plot(
                ax, facecolor="none", edgecolor="black", linestyle="-"
            )
    for source in fdtd.design.sources:
        if hasattr(source, "add_to_plot"):
            source.add_to_plot(ax)
    for monitor in fdtd.design.monitors:
        if hasattr(monitor, "add_to_plot"):
            monitor.add_to_plot(ax)

    if not clean_visualization:
        plt.xlabel(f"X ({unit})")
        plt.ylabel(f"Y ({unit})")
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
        title = ax.set_title(f't = {fdtd.results["t"][0]:.2e} s')
    else:
        title = None

    def update(frame_idx):
        frame = list(frame_indices)[frame_idx]
        im.set_array(fdtd.results[field][frame])
        if not clean_visualization:
            title.set_text(f't = {fdtd.results["t"][frame]:.2e} s')
            return [im, title]
        return [im]

    frames = len(list(frame_indices))
    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    try:
        from matplotlib.animation import FFMpegWriter

        writer = FFMpegWriter(fps=fps)
        if clean_visualization:
            ani.save(filename, writer=writer, dpi=300)
        else:
            ani.save(filename, writer=writer, dpi=100)
        print(
            f"Animation saved to {filename} (using {frames} of {total_frames} frames)"
        )
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Make sure FFmpeg is installed on your system.")
    plt.close(fig)


def plot_fdtd_power(
    fdtd,
    cmap: str = "hot",
    vmin: float = None,
    vmax: float = None,
    db_colorbar: bool = False,
):
    """Plot time-integrated power distribution from FDTD fields."""
    import matplotlib.pyplot as plt

    if fdtd.power_accumulated is not None:
        power = fdtd.power_accumulated
        print("Using accumulated power data")
    elif (
        len(fdtd.results["Ez"]) > 0
        and len(fdtd.results["Hx"]) > 0
        and len(fdtd.results["Hy"]) > 0
    ):
        print("Calculating power from saved field data")
        power = np.zeros((fdtd.nx, fdtd.ny))
        for t_idx in range(len(fdtd.results["t"])):
            Ez = fdtd.results["Ez"][t_idx]
            Hx_raw = fdtd.results["Hx"][t_idx]
            Hy_raw = fdtd.results["Hy"][t_idx]
            is_complex = (
                np.iscomplexobj(Ez)
                or np.iscomplexobj(Hx_raw)
                or np.iscomplexobj(Hy_raw)
            )
            if np.iscomplexobj(Ez):
                Ez_real = np.real(Ez)
                Ez_imag = np.imag(Ez)
            else:
                Ez_real = Ez
                Ez_imag = np.zeros_like(Ez)
            if is_complex:
                Hx = np.zeros_like(Ez, dtype=np.complex128)
                Hy = np.zeros_like(Ez, dtype=np.complex128)
            else:
                Hx = np.zeros_like(Ez_real)
                Hy = np.zeros_like(Ez_real)
            Hx[:, :-1] = Hx_raw
            Hy[:-1, :] = Hy_raw
            if is_complex:
                Hx_real = np.real(Hx)
                Hx_imag = np.imag(Hx)
                Hy_real = np.real(Hy)
                Hy_imag = np.imag(Hy)
                Sx = -Ez_real * Hy_real - Ez_imag * Hy_imag
                Sy = Ez_real * Hx_real + Ez_imag * Hx_imag
            else:
                Sx = -Ez_real * Hy
                Sy = Ez_real * Hx
            power_mag = Sx**2 + Sy**2
            power += power_mag
        power /= len(fdtd.results["t"])
    else:
        print(
            "No field data to calculate power. Make sure to run the simulation with save=True or accumulate_power=True."
        )
        return

    # Normalize power using 99th percentile to avoid source-dominated colormaps
    # This makes propagated power visible by clipping source peaks
    power_sorted = np.sort(power.flatten())
    nonzero_power = power_sorted[power_sorted > 0]
    if len(nonzero_power) > 100:  # Need sufficient data points
        p99 = np.percentile(nonzero_power, 99)
        power_clipped = np.clip(power, 0, p99)
        if p99 > 0:
            power_normalized = power_clipped / p99
        else:
            power_normalized = power
    else:
        # Fallback to max normalization for small datasets
        power_max = np.max(power)
        if power_max > 0 and np.isfinite(power_max):
            power_normalized = power / power_max
        else:
            power_normalized = power

    scale, unit = get_si_scale_and_label(max(fdtd.design.width, fdtd.design.height))
    aspect_ratio = power.shape[1] / power.shape[0]
    base_size = 8
    figsize = (
        (base_size * aspect_ratio, base_size)
        if aspect_ratio > 1
        else (base_size, base_size / aspect_ratio)
    )

    fdtd.fig, fdtd.ax = plt.subplots(figsize=figsize)
    # Use normalized power for display to avoid numerical precision issues with tiny values
    display_power = power_normalized if vmin is None and vmax is None else power
    fdtd.im = fdtd.ax.imshow(
        display_power,
        origin="lower",
        extent=(0, fdtd.design.width, 0, fdtd.design.height),
        cmap=cmap,
        aspect="equal",
        interpolation="bicubic",
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = plt.colorbar(fdtd.im, orientation="vertical", aspect=30, extend="both")
    if db_colorbar:
        # dB scale now works on normalized power (0 to 1)
        def db_formatter(x, pos):
            if x <= 0:
                return "-∞ dB"
            ratio = max(x, 1e-10)  # x is already normalized to max=1
            db_val = 10 * np.log10(ratio)
            return f"{db_val:.1f} dB"

        colorbar.formatter = plt.FuncFormatter(db_formatter)
        colorbar.update_ticks()
        colorbar.set_label("Relative Power (dB)")
    else:
        colorbar.set_label("Normalized Power")

    try:
        tmp_design = fdtd.design.copy()
        tmp_design.unify_polygons()
        overlay_structures = tmp_design.structures
    except Exception:
        overlay_structures = fdtd.design.structures
    for structure in overlay_structures:
        if hasattr(structure, "is_pml") and structure.is_pml:
            structure.add_to_plot(
                fdtd.ax, edgecolor="white", linestyle="--", facecolor="none", alpha=0.5
            )
        elif hasattr(structure, "vertices") and getattr(structure, "vertices", None):
            structure.add_to_plot(
                fdtd.ax, facecolor="none", edgecolor="white", linestyle="-"
            )
    # Sources are not shown in power plot to avoid visual clutter
    for monitor in fdtd.design.monitors:
        if hasattr(monitor, "add_to_plot"):
            monitor.add_to_plot(fdtd.ax, edgecolor="white")

    plt.xlabel(f"X ({unit})")
    plt.ylabel(f"Y ({unit})")
    fdtd.ax.xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    fdtd.ax.yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    plt.title("Time-Averaged Power Distribution")
    plt.tight_layout()
    plt.show()


def close_fdtd_figure(fdtd):
    """Close and reset the current FDTD Matplotlib figure safely."""
    import matplotlib.pyplot as plt

    if fdtd is None:
        return
    if getattr(fdtd, "fig", None) is not None:
        try:
            plt.close(fdtd.fig)
        finally:
            fdtd.fig = None
            fdtd.ax = None
            fdtd.im = None


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


class VideoRecorder:
    """Context manager for recording simulation frames to MP4 video.

    This class collects frames during simulation and exports them as an MP4 video
    when the recording is finished.

    Usage:
        recorder = VideoRecorder('output.mp4', fps=30, dpi=150)
        # During simulation loop:
        recorder.add_frame(field_array, extent, design, ...)
        # After simulation:
        recorder.save()
    """

    def __init__(
        self,
        filename="simulation.mp4",
        fps=30,
        dpi=150,
        cmap="twilight_zero",
        axis_scale=None,
        clean_visualization=False,
        wavelength=None,
        line_color="gray",
        line_opacity=0.5,
        interpolation="bicubic",
    ):
        """Initialize the video recorder.

        Args:
            filename: Output filename for the MP4 video
            fps: Frames per second for the video
            dpi: Resolution (dots per inch) for the video frames
            cmap: Matplotlib colormap name
            axis_scale: Tuple (min, max) for fixed color scale, or None for auto-scaling
            clean_visualization: If True, hide axes, title, and colorbar
            wavelength: Wavelength for scale bar calculation
            line_color: Color for structure outlines
            line_opacity: Opacity for structure outlines
            interpolation: Interpolation method for imshow ('nearest', 'bilinear', 'bicubic', etc.)
        """
        self.filename = filename
        self.fps = fps
        self.dpi = dpi
        self.cmap = cmap
        self.axis_scale = axis_scale
        self.clean_visualization = clean_visualization
        self.wavelength = wavelength
        self.line_color = line_color
        self.line_opacity = line_opacity
        self.interpolation = interpolation

        self.frames = []
        self.times = []
        self.field_name = None
        self.units = "V/µm"
        self.extent = None
        self.design = None
        self.boundaries = None
        self.plane_2d = "xy"

        # For auto-scaling across all frames
        self._global_vmax = 0.0

    def add_frame(
        self,
        field_array,
        t,
        step,
        num_steps,
        field_name="Ez",
        units="V/µm",
        extent=None,
        design=None,
        boundaries=None,
        plane_2d="xy",
    ):
        """Add a frame to the video recording.

        Args:
            field_array: 2D numpy array of field values
            t: Current simulation time
            step: Current simulation step number
            num_steps: Total number of simulation steps
            field_name: Name of the field component (e.g., 'Ez')
            units: Units string for display
            extent: Matplotlib extent tuple (xmin, xmax, ymin, ymax)
            design: Design object for structure overlay
            boundaries: List of boundary objects (PML, etc.)
            plane_2d: Simulation plane ('xy', 'yz', 'xz')
        """
        import numpy as np

        # Store metadata on first frame
        if len(self.frames) == 0:
            self.field_name = field_name
            self.units = units
            self.extent = extent
            self.design = design
            self.boundaries = boundaries
            self.plane_2d = plane_2d

        # Store frame data (make a copy to avoid reference issues)
        self.frames.append(np.asarray(field_array, dtype=float).copy())
        self.times.append((t, step, num_steps))

        # Track global max for auto-scaling
        if self.axis_scale is None:
            frame_max = np.percentile(np.abs(field_array), 99)
            if frame_max > self._global_vmax:
                self._global_vmax = frame_max

    def save(self):
        """Render all frames and save as MP4 video."""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.animation import FFMpegWriter

        # Store current backend and switch to Agg for video rendering
        original_backend = plt.get_backend()
        plt.switch_backend("Agg")

        if len(self.frames) == 0:
            print("No frames to save.")
            return

        # Handle 3D fields by extracting 2D slice
        # Check if first frame is 3D
        first_frame = np.asarray(self.frames[0])
        if first_frame.ndim == 3:
            # Extract middle slice based on plane_2d setting
            plane_2d = getattr(self, "plane_2d", "xy")
            if plane_2d == "xy":
                # Extract middle z-slice: frame[z_mid, :, :]
                z_mid = first_frame.shape[0] // 2
                first_frame = first_frame[z_mid, :, :]
            elif plane_2d == "xz":
                # Extract middle y-slice: frame[:, y_mid, :]
                y_mid = first_frame.shape[1] // 2
                first_frame = first_frame[:, y_mid, :]
            elif plane_2d == "yz":
                # Extract middle x-slice: frame[:, :, x_mid]
                x_mid = first_frame.shape[2] // 2
                first_frame = first_frame[:, :, x_mid]
            else:
                # Default to middle z-slice for xy plane
                z_mid = first_frame.shape[0] // 2
                first_frame = first_frame[z_mid, :, :]

            # Convert all frames to 2D slices
            converted_frames = []
            for frame in self.frames:
                frame_arr = np.asarray(frame)
                if frame_arr.ndim == 3:
                    if plane_2d == "xy":
                        z_mid = frame_arr.shape[0] // 2
                        converted_frames.append(frame_arr[z_mid, :, :])
                    elif plane_2d == "xz":
                        y_mid = frame_arr.shape[1] // 2
                        converted_frames.append(frame_arr[:, y_mid, :])
                    elif plane_2d == "yz":
                        x_mid = frame_arr.shape[2] // 2
                        converted_frames.append(frame_arr[:, :, x_mid])
                    else:
                        z_mid = frame_arr.shape[0] // 2
                        converted_frames.append(frame_arr[z_mid, :, :])
                else:
                    converted_frames.append(frame_arr)
            self.frames = converted_frames

        # Determine color scale
        if self.axis_scale is not None:
            vmin, vmax = self.axis_scale
        else:
            vmax = self._global_vmax if self._global_vmax > 0 else 1.0
            vmin = -vmax

        # Setup figure - ensure dimensions are divisible by 2 for video encoding
        grid_height, grid_width = first_frame.shape
        aspect_ratio = grid_width / grid_height
        base_size = 6

        if self.clean_visualization:
            if aspect_ratio > 1:
                figsize = (base_size * aspect_ratio, base_size)
            else:
                figsize = (base_size, base_size / aspect_ratio)
        else:
            if aspect_ratio > 1:
                figsize = (base_size * aspect_ratio * 1.2, base_size)
            else:
                figsize = (base_size * 1.2, base_size / aspect_ratio)

        # Ensure pixel dimensions are even (required by libx264)
        fig_width_px = int(figsize[0] * self.dpi)
        fig_height_px = int(figsize[1] * self.dpi)
        fig_width_px = fig_width_px + (fig_width_px % 2)
        fig_height_px = fig_height_px + (fig_height_px % 2)
        figsize = (fig_width_px / self.dpi, fig_height_px / self.dpi)

        # Get colormap
        if self.cmap == "twilight_zero":
            try:
                actual_cmap = plt.get_cmap("twilight_zero")
            except ValueError:
                actual_cmap = get_twilight_zero_cmap()
        else:
            actual_cmap = self.cmap

        # Setup FFmpeg writer
        writer = FFMpegWriter(fps=self.fps, bitrate=5000)

        # Create figure
        fig = plt.figure(figsize=figsize)

        try:
            with writer.saving(fig, self.filename, dpi=self.dpi):
                for frame_idx, (frame_data, time_info) in enumerate(
                    zip(self.frames, self.times)
                ):
                    fig.clear()

                    if self.clean_visualization:
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.set_axis_off()
                    else:
                        ax = fig.add_subplot(111)

                    # Draw field
                    im = ax.imshow(
                        frame_data,
                        origin="lower",
                        cmap=actual_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        extent=self.extent,
                        aspect="equal",
                        interpolation=self.interpolation,
                    )

                    # Add colorbar and title if not clean
                    if not self.clean_visualization:
                        plt.colorbar(
                            im,
                            ax=ax,
                            orientation="vertical",
                            label=f"{self.field_name} ({self.units})",
                        )
                        t, step, num_steps = time_info
                        ax.set_title(
                            f"{self.field_name} at t = {t:.2e} s (step {step}/{num_steps})"
                        )

                    # Add structure overlays
                    if self.design is not None:
                        try:
                            tmp_design = self.design.copy()
                            tmp_design.unify_polygons()
                            overlay_structures = tmp_design.structures
                        except Exception:
                            overlay_structures = getattr(self.design, "structures", [])

                        for structure in overlay_structures or []:
                            if hasattr(structure, "is_pml") and structure.is_pml:
                                structure.add_to_plot(
                                    ax,
                                    edgecolor=self.line_color,
                                    linestyle="--",
                                    facecolor="none",
                                    alpha=self.line_opacity,
                                )
                            elif hasattr(structure, "vertices") and getattr(
                                structure, "vertices", None
                            ):
                                structure.add_to_plot(
                                    ax,
                                    facecolor="none",
                                    edgecolor=self.line_color,
                                    linestyle="-",
                                    alpha=self.line_opacity,
                                )

                        # Add sources
                        for source in getattr(self.design, "sources", []) or []:
                            if hasattr(source, "add_to_plot"):
                                source.add_to_plot(ax)

                        # Add monitors
                        for monitor in getattr(self.design, "monitors", []) or []:
                            if hasattr(monitor, "add_to_plot"):
                                monitor.add_to_plot(
                                    ax,
                                    edgecolor=self.line_color,
                                    alpha=self.line_opacity,
                                )

                    # Draw PML boundaries
                    if self.boundaries:
                        for boundary in self.boundaries:
                            draw_boundary(
                                ax,
                                boundary,
                                self.design,
                                edgecolor=self.line_color,
                                linestyle=":",
                                alpha=self.line_opacity,
                            )

                    # Add axis labels if not clean
                    if self.design is not None and not self.clean_visualization:
                        max_dim = max(self.design.width, self.design.height)
                        scale, unit = get_si_scale_and_label(max_dim)

                        xlabel, ylabel = "X", "Y"
                        if self.plane_2d == "yz":
                            xlabel, ylabel = "Y", "Z"
                        elif self.plane_2d == "xz":
                            xlabel, ylabel = "X", "Z"

                        ax.set_xlabel(f"{xlabel} ({unit})")
                        ax.set_ylabel(f"{ylabel} ({unit})")
                        ax.xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
                        ax.yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")

                    # Add scale bar for clean visualization
                    if self.clean_visualization and self.design is not None:
                        self._add_scale_bar(ax)

                    if not self.clean_visualization:
                        fig.tight_layout()

                    # Write frame
                    writer.grab_frame()

            print(
                f"Video saved to {self.filename} ({len(self.frames)} frames at {self.fps} fps)"
            )
        except Exception as e:
            print(f"Error saving video: {e}")
            print("Make sure FFmpeg is installed on your system.")
        finally:
            plt.close(fig)
            # Restore original backend
            try:
                plt.switch_backend(original_backend)
            except Exception:
                pass

    def _add_scale_bar(self, ax):
        """Add a scale bar to clean visualization."""
        import numpy as np

        if self.design is None:
            return

        max_dim = max(self.design.width, self.design.height)
        scale_factor, unit = get_si_scale_and_label(max_dim)

        if self.wavelength is not None:
            wavelength_um = self.wavelength * 1e6
            scale_bar_length_um = np.round(2 * wavelength_um)
            scale_bar_length = scale_bar_length_um * 1e-6
        else:
            min_dim = min(self.design.width, self.design.height)
            scale_bar_fraction = 0.18
            scale_bar_length_physical = min_dim * scale_bar_fraction

            if scale_bar_length_physical > 0:
                order = 10 ** np.floor(np.log10(scale_bar_length_physical))
                normalized = scale_bar_length_physical / order
                if normalized <= 1.25:
                    nice_value = 1 * order
                elif normalized <= 2.5:
                    nice_value = 2 * order
                elif normalized <= 6:
                    nice_value = 5 * order
                else:
                    nice_value = 10 * order
                scale_bar_length = nice_value
            else:
                scale_bar_length = min_dim * 0.15

        margin_x = self.design.width * 0.1
        margin_y = self.design.height * 0.1
        x_start = self.design.width - scale_bar_length - margin_x
        x_end = self.design.width - margin_x
        y_pos = margin_y

        ax.plot(
            [x_start, x_end], [y_pos, y_pos], "w", linewidth=3, solid_capstyle="butt"
        )

        label_y = y_pos - self.design.height * 0.02
        if self.wavelength is not None:
            scale_bar_length_display_um = scale_bar_length * 1e6
            label_text = f"{int(scale_bar_length_display_um)} µm"
        else:
            scale_bar_length_display = scale_bar_length * scale_factor
            if scale_bar_length_display >= 1:
                label_text = f"{scale_bar_length_display:.0f} {unit}"
            elif scale_bar_length_display >= 0.1:
                label_text = f"{scale_bar_length_display:.1f} {unit}"
            else:
                label_text = f"{scale_bar_length_display:.2f} {unit}"

        ax.text(
            (x_start + x_end) / 2,
            label_y,
            label_text,
            ha="center",
            va="top",
            color="white",
            fontsize=10,
        )


class JupyterAnimator:
    """Handles live animation and replay for Jupyter notebooks.

    Provides two modes:
    1. Live mode: Updates cell output during simulation using clear_output + display
    2. Replay mode: Returns an interactive animation widget after simulation

    Usage:
        animator = JupyterAnimator(...)
        # During simulation loop:
        animator.update(field_array, t, step, num_steps)
        # After simulation:
        animation = animator.get_animation()  # Returns playable HTML5 video
        widget = animator.get_widget()        # Returns interactive slider
    """

    def __init__(
        self,
        cmap="twilight_zero",
        axis_scale=None,
        clean_visualization=False,
        wavelength=None,
        line_color="gray",
        line_opacity=0.5,
        interpolation="bicubic",
        live_display=True,
        store_frames=True,
        display_interval=0.05,
    ):
        """Initialize the Jupyter animator.

        Args:
            cmap: Matplotlib colormap name
            axis_scale: Fixed (vmin, vmax) or None for auto-scaling
            clean_visualization: Hide axes/colorbar if True
            wavelength: For scale bar calculation
            line_color: Structure outline color
            line_opacity: Structure outline opacity
            interpolation: imshow interpolation method
            live_display: Show frames during simulation
            store_frames: Store frames for post-simulation replay
            display_interval: Minimum seconds between live display updates
        """
        self.cmap = cmap
        self.axis_scale = axis_scale
        self.clean_visualization = clean_visualization
        self.wavelength = wavelength
        self.line_color = line_color
        self.line_opacity = line_opacity
        self.interpolation = interpolation
        self.live_display = live_display
        self.store_frames = store_frames
        self.display_interval = display_interval

        # Frame storage
        self.frames = []
        self.times = []
        self.metadata = {}

        # Auto-scaling state
        self._global_vmax = 0.0

        # Timing for throttled display
        self._last_display_time = 0

        # Persistent figure elements for live display (reused across frames)
        self._fig = None
        self._ax = None
        self._im = None
        self._cbar = None
        self._title = None

    def update(
        self,
        field_array,
        t,
        step,
        num_steps,
        field_name="Ez",
        units="V/µm",
        extent=None,
        design=None,
        boundaries=None,
        plane_2d="xy",
    ):
        """Add a frame and optionally display it live.

        Args:
            field_array: 2D numpy array of field values
            t: Current simulation time
            step: Current step number
            num_steps: Total number of steps
            field_name: Name of field component ('Ez', 'Hx', etc.)
            units: Unit string for colorbar label
            extent: Matplotlib extent tuple (xmin, xmax, ymin, ymax)
            design: Design object for structure overlays
            boundaries: List of boundary objects for overlays
            plane_2d: Simulation plane ('xy', 'yz', 'xz')
        """
        import time

        frame_data = np.asarray(field_array, dtype=float).copy()

        # Store frame if enabled
        if self.store_frames:
            self.frames.append(frame_data)
            self.times.append((t, step, num_steps))

            # Store metadata on first frame
            if len(self.frames) == 1:
                self.metadata = {
                    "field_name": field_name,
                    "units": units,
                    "extent": extent,
                    "design": design,
                    "boundaries": boundaries,
                    "plane_2d": plane_2d,
                }

        # Track global max for auto-scaling
        if self.axis_scale is None:
            abs_data = np.abs(frame_data)
            if abs_data.size > 10:
                frame_max = np.percentile(abs_data, 99)
            else:
                frame_max = float(np.max(abs_data) or 1.0)
            if frame_max > self._global_vmax:
                self._global_vmax = frame_max

        # Live display with throttling
        if self.live_display:
            current_time = time.time()
            if current_time - self._last_display_time >= self.display_interval:
                self._display_frame(
                    frame_data,
                    t,
                    step,
                    num_steps,
                    field_name,
                    units,
                    extent,
                    design,
                    boundaries,
                    plane_2d,
                )
                self._last_display_time = current_time

    def _display_frame(
        self,
        frame_data,
        t,
        step,
        num_steps,
        field_name,
        units,
        extent,
        design,
        boundaries,
        plane_2d,
    ):
        """Display a single frame in Jupyter, reusing a persistent figure."""
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        # Determine color scale
        if self.axis_scale is not None:
            vmin, vmax = self.axis_scale
        else:
            vmax = self._global_vmax if self._global_vmax > 0 else 1.0
            vmin = -vmax

        # Get colormap
        if self.cmap == "twilight_zero":
            try:
                actual_cmap = plt.get_cmap("twilight_zero")
            except ValueError:
                actual_cmap = get_twilight_zero_cmap()
        else:
            actual_cmap = self.cmap

        # First frame: create the figure and all elements
        if self._fig is None:
            # Calculate figure size based on data aspect ratio for clean visualization
            if self.clean_visualization and extent:
                data_width = extent[1] - extent[0]
                data_height = extent[3] - extent[2]
                aspect_ratio = data_width / data_height
                fig_height = 8
                fig_width = fig_height * aspect_ratio
                self._fig = plt.figure(figsize=(fig_width, fig_height))
                self._fig.patch.set_facecolor("none")  # Transparent background
                # Create axes that fills the entire figure
                self._ax = self._fig.add_axes([0, 0, 1, 1])
            else:
                self._fig, self._ax = plt.subplots(figsize=(10, 8))

            self._im = self._ax.imshow(
                frame_data,
                origin="lower",
                cmap=actual_cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                interpolation=self.interpolation,
            )

            if self.clean_visualization:
                self._ax.set_axis_off()
                self._ax.set_frame_on(False)
                self._title = None
            else:
                self._cbar = plt.colorbar(
                    self._im, ax=self._ax, label=f"{field_name} ({units})"
                )
                self._title = self._ax.set_title(
                    f"{field_name} at t = {t:.2e} s (step {step}/{num_steps})"
                )
                plt.tight_layout()

            # Add structure overlays (static, only done once)
            self._add_overlays(self._ax, design, boundaries, plane_2d)

            # Add scale bar for clean visualization
            if self.clean_visualization:
                self._add_scale_bar(self._ax, design)

        else:
            # Subsequent frames: just update the data
            self._im.set_data(frame_data)
            self._im.set_clim(vmin, vmax)

            if self._title is not None:
                self._title.set_text(
                    f"{field_name} at t = {t:.2e} s (step {step}/{num_steps})"
                )

            if self._cbar is not None:
                self._cbar.mappable.set_clim(vmin, vmax)

        # Clear previous output and display updated figure
        clear_output(wait=True)
        display(self._fig)

    def finalize(self):
        """Close the live display figure after simulation completes."""
        import matplotlib.pyplot as plt

        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._im = None
            self._cbar = None
            self._title = None

    def _add_overlays(self, ax, design, boundaries, plane_2d):
        """Add structure, source, monitor, and boundary overlays."""
        if design is not None:
            try:
                tmp_design = design.copy()
                tmp_design.unify_polygons()
                overlay_structures = tmp_design.structures
            except Exception:
                overlay_structures = getattr(design, "structures", [])

            for structure in overlay_structures or []:
                # Skip the background structure (first structure that spans full design)
                if hasattr(structure, "vertices") and structure.vertices:
                    vertices = np.array(structure.vertices)
                    min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
                    min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
                    # Check if structure spans the full design dimensions
                    if (
                        abs(min_x) < 1e-10
                        and abs(min_y) < 1e-10
                        and abs(max_x - design.width) < 1e-10
                        and abs(max_y - design.height) < 1e-10
                    ):
                        continue  # Skip background structure

                if hasattr(structure, "is_pml") and structure.is_pml:
                    structure.add_to_plot(
                        ax,
                        edgecolor=self.line_color,
                        linestyle="--",
                        facecolor="none",
                        alpha=self.line_opacity,
                    )
                elif hasattr(structure, "vertices"):
                    structure.add_to_plot(
                        ax,
                        facecolor="none",
                        edgecolor=self.line_color,
                        linestyle="-",
                        alpha=self.line_opacity,
                    )

            for source in getattr(design, "sources", []) or []:
                if hasattr(source, "add_to_plot"):
                    source.add_to_plot(ax)

            for monitor in getattr(design, "monitors", []) or []:
                if hasattr(monitor, "add_to_plot"):
                    monitor.add_to_plot(
                        ax, edgecolor=self.line_color, alpha=self.line_opacity
                    )

        if boundaries:
            for boundary in boundaries:
                draw_boundary(
                    ax,
                    boundary,
                    design,
                    edgecolor=self.line_color,
                    linestyle=":",
                    alpha=self.line_opacity,
                )

    def _add_scale_bar(self, ax, design):
        """Add scale bar to the plot for clean visualization mode."""
        if design is None:
            return

        max_dim = max(design.width, design.height)
        scale_factor, unit = get_si_scale_and_label(max_dim)

        # Calculate scale bar length: 2 * wavelength rounded to nearest integer µm
        if self.wavelength is not None:
            wavelength_um = self.wavelength * 1e6
            scale_bar_length_um = np.round(2 * wavelength_um)
            scale_bar_length = scale_bar_length_um * 1e-6
        else:
            # Fallback: use design-based calculation
            min_dim = min(design.width, design.height)
            scale_bar_length_physical = min_dim * 0.18

            if scale_bar_length_physical > 0:
                order = 10 ** np.floor(np.log10(scale_bar_length_physical))
                normalized = scale_bar_length_physical / order
                if normalized <= 1.25:
                    nice_value = 1 * order
                elif normalized <= 2.5:
                    nice_value = 2 * order
                elif normalized <= 6:
                    nice_value = 5 * order
                else:
                    nice_value = 10 * order
                scale_bar_length = nice_value
            else:
                scale_bar_length = min_dim * 0.15

        # Position in bottom-right corner with some margin
        margin_x = design.width * 0.1
        margin_y = design.height * 0.1
        x_start = design.width - scale_bar_length - margin_x
        x_end = design.width - margin_x
        y_pos = margin_y

        # Draw scale bar line
        ax.plot(
            [x_start, x_end], [y_pos, y_pos], "w", linewidth=3, solid_capstyle="butt"
        )

        # Add text label below the bar
        label_y = y_pos - design.height * 0.02
        if self.wavelength is not None:
            scale_bar_length_display_um = scale_bar_length * 1e6
            label_text = f"{int(scale_bar_length_display_um)} µm"
        else:
            scale_bar_length_display = scale_bar_length * scale_factor
            if scale_bar_length_display >= 1:
                label_text = f"{scale_bar_length_display:.0f} {unit}"
            elif scale_bar_length_display >= 0.1:
                label_text = f"{scale_bar_length_display:.1f} {unit}"
            else:
                label_text = f"{scale_bar_length_display:.2f} {unit}"

        ax.text(
            (x_start + x_end) / 2,
            label_y,
            label_text,
            ha="center",
            va="top",
            color="white",
            fontsize=14,
        )

    def get_animation(self, fps=30):
        """Create an HTML5 video animation from stored frames.

        Args:
            fps: Frames per second for the animation

        Returns:
            IPython.display.HTML: Playable HTML5 video animation
        """
        import matplotlib.pyplot as plt
        from IPython.display import HTML
        from matplotlib.animation import FuncAnimation

        if not self.frames:
            print("No frames stored. Enable store_frames=True.")
            return None

        # Determine color scale from all frames
        if self.axis_scale is not None:
            vmin, vmax = self.axis_scale
        else:
            vmax = self._global_vmax if self._global_vmax > 0 else 1.0
            vmin = -vmax

        # Calculate figure size based on data aspect ratio for clean visualization
        extent = self.metadata.get("extent")
        if self.clean_visualization and extent:
            data_width = extent[1] - extent[0]
            data_height = extent[3] - extent[2]
            aspect_ratio = data_width / data_height
            fig_height = 8
            fig_width = fig_height * aspect_ratio
            fig = plt.figure(figsize=(fig_width, fig_height))
            fig.patch.set_facecolor("none")  # Transparent background
            ax = fig.add_axes([0, 0, 1, 1])
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Get colormap
        if self.cmap == "twilight_zero":
            try:
                actual_cmap = plt.get_cmap("twilight_zero")
            except ValueError:
                actual_cmap = get_twilight_zero_cmap()
        else:
            actual_cmap = self.cmap

        # Initial frame
        im = ax.imshow(
            self.frames[0],
            origin="lower",
            cmap=actual_cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation=self.interpolation,
        )

        title = None
        if self.clean_visualization:
            ax.set_axis_off()
            ax.set_frame_on(False)
        else:
            plt.colorbar(
                im,
                ax=ax,
                label=f"{self.metadata.get('field_name', 'Field')} ({self.metadata.get('units', '')})",
            )
            title = ax.set_title("")
            plt.tight_layout()

        # Add static overlays
        self._add_overlays(
            ax,
            self.metadata.get("design"),
            self.metadata.get("boundaries"),
            self.metadata.get("plane_2d", "xy"),
        )

        # Add scale bar for clean visualization
        if self.clean_visualization:
            self._add_scale_bar(ax, self.metadata.get("design"))

        def update(frame_idx):
            im.set_data(self.frames[frame_idx])
            if title is not None and self.times:
                t, step, num_steps = self.times[frame_idx]
                field_name = self.metadata.get("field_name", "Field")
                title.set_text(
                    f"{field_name} at t = {t:.2e} s (step {step}/{num_steps})"
                )
            return [im] if title is None else [im, title]

        anim = FuncAnimation(
            fig, update, frames=len(self.frames), interval=1000 / fps, blit=True
        )

        plt.close(fig)

        # Increase embed limit for larger animations (default is ~20MB)
        import matplotlib as mpl

        old_limit = mpl.rcParams.get("animation.embed_limit", 20)
        mpl.rcParams["animation.embed_limit"] = 200  # 200 MB limit

        try:
            # Convert to HTML5 video
            html_content = anim.to_jshtml()
            size_bytes = len(html_content.encode("utf-8"))
            size_mb = size_bytes / (1024 * 1024)
            print(f"Animation size: {size_mb:.1f} MB ({len(self.frames)} frames)")
            return HTML(html_content)
        finally:
            # Restore original limit
            mpl.rcParams["animation.embed_limit"] = old_limit

    def get_video(self, filename="animation.mp4", fps=30, dpi=150):
        """Create an MP4 video and display it in Jupyter notebook.

        Args:
            filename: Output filename for the MP4 video
            fps: Frames per second for the video
            dpi: Resolution (dots per inch) for video frames

        Returns:
            IPython.display.Video: Playable video widget
        """
        import os

        import matplotlib.pyplot as plt
        from IPython.display import Video
        from matplotlib.animation import FFMpegWriter, FuncAnimation

        if not self.frames:
            print("No frames stored. Enable store_frames=True.")
            return None

        # Determine color scale from all frames
        if self.axis_scale is not None:
            vmin, vmax = self.axis_scale
        else:
            vmax = self._global_vmax if self._global_vmax > 0 else 1.0
            vmin = -vmax

        # Calculate figure size based on data aspect ratio for clean visualization
        extent = self.metadata.get("extent")
        if self.clean_visualization and extent:
            data_width = extent[1] - extent[0]
            data_height = extent[3] - extent[2]
            aspect_ratio = data_width / data_height
            fig_height = 8
            fig_width = fig_height * aspect_ratio
            fig = plt.figure(figsize=(fig_width, fig_height))
            fig.patch.set_facecolor(
                "black"
            )  # Black background (MP4 doesn't support transparency)
            ax = fig.add_axes([0, 0, 1, 1])
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Get colormap
        if self.cmap == "twilight_zero":
            try:
                actual_cmap = plt.get_cmap("twilight_zero")
            except ValueError:
                actual_cmap = get_twilight_zero_cmap()
        else:
            actual_cmap = self.cmap

        # Initial frame
        im = ax.imshow(
            self.frames[0],
            origin="lower",
            cmap=actual_cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation=self.interpolation,
        )

        title = None
        if self.clean_visualization:
            ax.set_axis_off()
            ax.set_frame_on(False)
        else:
            plt.colorbar(
                im,
                ax=ax,
                label=f"{self.metadata.get('field_name', 'Field')} ({self.metadata.get('units', '')})",
            )
            title = ax.set_title("")
            plt.tight_layout()

        # Add static overlays
        self._add_overlays(
            ax,
            self.metadata.get("design"),
            self.metadata.get("boundaries"),
            self.metadata.get("plane_2d", "xy"),
        )

        # Add scale bar for clean visualization
        if self.clean_visualization:
            self._add_scale_bar(ax, self.metadata.get("design"))

        def update(frame_idx):
            im.set_data(self.frames[frame_idx])
            if title is not None and self.times:
                t, step, num_steps = self.times[frame_idx]
                field_name = self.metadata.get("field_name", "Field")
                title.set_text(
                    f"{field_name} at t = {t:.2e} s (step {step}/{num_steps})"
                )
            return [im] if title is None else [im, title]

        anim = FuncAnimation(
            fig, update, frames=len(self.frames), interval=1000 / fps, blit=True
        )

        # Save as MP4
        print(f"Rendering {len(self.frames)} frames to {filename}...")
        try:
            writer = FFMpegWriter(fps=fps, metadata={"title": "BEAMZ Simulation"})
            anim.save(
                filename, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "black"}
            )
            plt.close(fig)

            # Get file size
            size_bytes = os.path.getsize(filename)
            size_mb = size_bytes / (1024 * 1024)
            print(f"Video saved: {filename} ({size_mb:.1f} MB)")

            # Return video widget for Jupyter
            return Video(filename, embed=True)
        except Exception as e:
            plt.close(fig)
            print(f"Error creating video: {e}")
            print("Make sure ffmpeg is installed: conda install ffmpeg")
            return None

    def get_widget(self):
        """Create an interactive slider widget for frame-by-frame scrubbing.

        Returns:
            ipywidgets Output widget with interactive slider
        """
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display

        if not self.frames:
            print("No frames stored. Enable store_frames=True.")
            return None

        # Determine color scale
        if self.axis_scale is not None:
            vmin, vmax = self.axis_scale
        else:
            vmax = self._global_vmax if self._global_vmax > 0 else 1.0
            vmin = -vmax

        # Get colormap
        if self.cmap == "twilight_zero":
            try:
                actual_cmap = plt.get_cmap("twilight_zero")
            except ValueError:
                actual_cmap = get_twilight_zero_cmap()
        else:
            actual_cmap = self.cmap

        try:
            import ipywidgets as widgets
        except ImportError:
            print(
                "ipywidgets not installed. Use get_animation() instead or install with: pip install ipywidgets"
            )
            return None

        output = widgets.Output()

        def show_frame(frame=0):
            with output:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(10, 8))

                im = ax.imshow(
                    self.frames[frame],
                    origin="lower",
                    cmap=actual_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=self.metadata.get("extent"),
                    interpolation=self.interpolation,
                )

                if self.clean_visualization:
                    # Hide axes and remove all padding for clean visualization
                    ax.set_axis_off()
                    plt.subplots_adjust(
                        left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
                    )
                else:
                    plt.colorbar(
                        im,
                        ax=ax,
                        label=f"{self.metadata.get('field_name', 'Field')} ({self.metadata.get('units', '')})",
                    )
                    if self.times:
                        t, step, num_steps = self.times[frame]
                        field_name = self.metadata.get("field_name", "Field")
                        ax.set_title(
                            f"{field_name} at t = {t:.2e} s (step {step}/{num_steps})"
                        )
                    plt.tight_layout()

                self._add_overlays(
                    ax,
                    self.metadata.get("design"),
                    self.metadata.get("boundaries"),
                    self.metadata.get("plane_2d", "xy"),
                )

                plt.show()

        # Create slider
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.frames) - 1,
            step=1,
            description="Frame:",
            continuous_update=False,
        )

        # Play button
        play = widgets.Play(
            value=0,
            min=0,
            max=len(self.frames) - 1,
            step=1,
            interval=100,
            description="Play",
        )

        widgets.jslink((play, "value"), (slider, "value"))

        # Connect slider to display function
        widgets.interactive_output(show_frame, {"frame": slider})

        # Show initial frame
        show_frame(0)

        return widgets.VBox([widgets.HBox([play, slider]), output])


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
