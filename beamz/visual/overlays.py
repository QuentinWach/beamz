"""Shared overlay helpers used across animation, video, and field plotting."""

import numpy as np

from beamz.visual.helpers import get_si_scale_and_label


def resolve_cmap(cmap):
    """Resolve a colormap name to a matplotlib colormap object.

    Handles the custom 'twilight_zero' colormap specially.
    """
    import matplotlib.pyplot as plt

    if cmap == "twilight_zero":
        try:
            return plt.get_cmap("twilight_zero")
        except ValueError:
            from beamz.visual.animation import get_twilight_zero_cmap

            return get_twilight_zero_cmap()
    return cmap


def add_design_overlays(
    ax,
    design,
    line_color="gray",
    line_opacity=0.5,
    sources=None,
    show_monitors=True,
    skip_background=False,
):
    """Draw structure outlines, sources, and monitors on an axis.

    Args:
        ax: Matplotlib axes.
        design: Design object whose structures/sources/monitors to overlay.
        line_color: Edge colour for structures and monitors.
        line_opacity: Alpha for structure and monitor outlines.
        sources: If given, overlay these; otherwise use ``design.sources``.
        show_monitors: Whether to draw monitors.
        skip_background: If True, skip the first structure that spans the full design.
    """
    if design is None:
        return

    try:
        tmp_design = design.copy()
        tmp_design.unify_polygons()
        overlay_structures = tmp_design.structures
    except Exception:
        overlay_structures = getattr(design, "structures", [])

    for structure in overlay_structures or []:
        if skip_background and hasattr(structure, "vertices") and structure.vertices:
            vertices = np.array(structure.vertices)
            min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
            min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
            if (
                abs(min_x) < 1e-10
                and abs(min_y) < 1e-10
                and abs(max_x - design.width) < 1e-10
                and abs(max_y - design.height) < 1e-10
            ):
                continue

        if hasattr(structure, "is_pml") and structure.is_pml:
            structure.add_to_plot(
                ax,
                edgecolor=line_color,
                linestyle="--",
                facecolor="none",
                alpha=line_opacity,
            )
        elif hasattr(structure, "vertices") and getattr(structure, "vertices", None):
            structure.add_to_plot(
                ax,
                facecolor="none",
                edgecolor=line_color,
                linestyle="-",
                alpha=line_opacity,
            )

    for source in (
        sources if sources is not None else getattr(design, "sources", []) or []
    ):
        if hasattr(source, "add_to_plot"):
            source.add_to_plot(ax)

    if show_monitors:
        for monitor in getattr(design, "monitors", []) or []:
            if hasattr(monitor, "add_to_plot"):
                monitor.add_to_plot(ax, edgecolor=line_color, alpha=line_opacity)


def draw_scale_bar(ax, design, wavelength=None, fontsize=10):
    """Draw a white scale bar in the bottom-right corner of *ax*.

    Args:
        ax: Matplotlib axes (must already have correct extent).
        design: Design object (used for dimensions).
        wavelength: If given, bar length = 2 * wavelength rounded to nearest um.
        fontsize: Font size for the label text.
    """
    if design is None:
        return

    max_dim = max(design.width, design.height)
    scale_factor, unit = get_si_scale_and_label(max_dim)

    if wavelength is not None:
        wavelength_um = wavelength * 1e6
        scale_bar_length_um = np.round(2 * wavelength_um)
        scale_bar_length = scale_bar_length_um * 1e-6
    else:
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

    margin_x = design.width * 0.1
    margin_y = design.height * 0.1
    x_start = design.width - scale_bar_length - margin_x
    x_end = design.width - margin_x
    y_pos = margin_y

    ax.plot([x_start, x_end], [y_pos, y_pos], "w", linewidth=3, solid_capstyle="butt")

    label_y = y_pos - design.height * 0.02
    if wavelength is not None:
        scale_bar_length_display_um = scale_bar_length * 1e6
        label_text = f"{int(scale_bar_length_display_um)} \u00b5m"
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
        fontsize=fontsize,
    )


def configure_axes(ax, design, plane_2d="xy"):
    """Apply SI-scaled tick formatters and axis labels to *ax*.

    Args:
        ax: Matplotlib axes.
        design: Design object (used for max dimension).
        plane_2d: Which plane is being shown ('xy', 'yz', 'xz').
    """
    max_dim = max(design.width, design.height)
    scale, unit = get_si_scale_and_label(max_dim)

    xlabel, ylabel = "X", "Y"
    if plane_2d == "yz":
        xlabel, ylabel = "Y", "Z"
    elif plane_2d == "xz":
        xlabel, ylabel = "X", "Z"

    ax.set_xlabel(f"{xlabel} ({unit})")
    ax.set_ylabel(f"{ylabel} ({unit})")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")


def show_mesh_3d(grid_3d, design, field="permittivity", slice_spacing=1, alpha=0.3):
    """Display a 3D surface-slice visualization of a volumetric mesh.

    Args:
        grid_3d: 3D numpy array (nz, ny, nx).
        design: Design object (used for width/height/depth and SI scaling).
        field: Name of the field.
        slice_spacing: Show every *n*-th z-layer.
        alpha: Transparency of each slice surface.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    nz, ny, nx = grid_3d.shape
    x = np.linspace(0, design.width, nx)
    y = np.linspace(0, design.height, ny)
    z = np.linspace(0, design.depth, nz)

    for k in range(0, nz, slice_spacing):
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z[k])
        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=plt.cm.viridis(grid_3d[k, :, :]),
            alpha=alpha,
            linewidth=0,
            antialiased=True,
        )

    scale, unit = get_si_scale_and_label(max(design.width, design.height, design.depth))
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_zlabel(f"Z ({unit})")
    ax.set_title(f"3D {field.capitalize()} Distribution")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    ax.zaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")

    plt.tight_layout()
    plt.show()


def add_mode_source_to_plot(
    mode_source, ax, facecolor="none", edgecolor="crimson", alpha=0.8, linestyle="-"
):
    """Draw a ModeSource overlay (line + direction arrow) on *ax*."""
    from matplotlib.patches import FancyArrowPatch

    center = (
        mode_source.center
        if isinstance(mode_source.center, (tuple, list))
        else (mode_source.center, 0)
    )
    if len(center) == 3:
        x_pos = center[0]
        y_pos = center[1]
    else:
        x_pos, y_pos = center[0], center[1]

    half_width = mode_source.width / 2 if mode_source.width else 0.5e-6

    if mode_source.direction in ["+x", "-x"]:
        y_start = y_pos - half_width
        y_end = y_pos + half_width
        line_x = [x_pos, x_pos]
        line_y = [y_start, y_end]
    else:
        x_start = x_pos - half_width
        x_end = x_pos + half_width
        line_x = [x_start, x_end]
        line_y = [y_pos, y_pos]

    ax.plot(
        line_x,
        line_y,
        color=edgecolor,
        linewidth=3,
        alpha=alpha,
        solid_capstyle="round",
        label="ModeSource",
    )

    arrow_length = (
        mode_source.wavelength * 0.5 if hasattr(mode_source, "wavelength") else 0.5e-6
    )
    if mode_source.direction in ["+x", "-x"]:
        arrow_dx = arrow_length if mode_source.direction == "+x" else -arrow_length
        arrow_dy = 0
    else:
        arrow_dx = 0
        arrow_dy = arrow_length if mode_source.direction == "+y" else -arrow_length

    arrow = FancyArrowPatch(
        (x_pos, y_pos),
        (x_pos + arrow_dx, y_pos + arrow_dy),
        arrowstyle="-|>",
        mutation_scale=10,
        color=edgecolor,
        linewidth=2,
        alpha=alpha,
    )
    ax.add_patch(arrow)


def add_gaussian_source_to_plot(
    gaussian_source, ax, facecolor="none", edgecolor="orange", alpha=0.8, linestyle="-"
):
    """Draw a GaussianSource overlay (circle + center dot) on *ax*."""
    from matplotlib.patches import Circle

    if len(gaussian_source.position) >= 2:
        x_pos, y_pos = gaussian_source.position[0], gaussian_source.position[1]
    else:
        x_pos, y_pos = gaussian_source.position[0], 0

    circle = Circle(
        (x_pos, y_pos),
        radius=gaussian_source.width,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=2,
        alpha=alpha,
        linestyle=linestyle,
        label="GaussianSource",
    )
    ax.add_patch(circle)

    center_dot = Circle(
        (x_pos, y_pos),
        radius=gaussian_source.width * 0.1,
        facecolor=edgecolor,
        edgecolor="none",
        alpha=alpha,
    )
    ax.add_patch(center_dot)


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
