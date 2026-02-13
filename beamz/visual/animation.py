"""FDTD live animation and frame-by-frame replay."""

import numpy as np

from beamz.visual.design_viz import draw_boundary
from beamz.visual.helpers import display_status, get_si_scale_and_label


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

