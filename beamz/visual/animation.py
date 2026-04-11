"""FDTD live animation and frame-by-frame replay."""

import numpy as np

from beamz.visual.design_viz import draw_boundary
from beamz.visual.helpers import get_si_scale_and_label
from beamz.visual.overlays import (
    add_design_overlays,
    configure_axes,
    draw_scale_bar,
    resolve_cmap,
)


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
    sources=None,
    monitors=None,
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
        design: Optional FDTD design object to overlay structures.
        boundaries: Optional list of boundary objects (PML, ABC, etc.) to visualize.
        sources: Optional explicit source list for overlays.
        monitors: Optional explicit monitor list for overlays.
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
        actual_cmap = resolve_cmap(cmap)

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
            add_design_overlays(
                ax,
                design,
                line_color=line_color,
                line_opacity=line_opacity,
                sources=sources if show_sources else [],
                monitors=monitors if show_monitors else [],
                show_monitors=show_monitors,
            )

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
            configure_axes(ax, design, plane_2d=plane_2d)

        if clean_visualization and design is not None:
            draw_scale_bar(ax, design, wavelength=wavelength)

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
        sources=None,
        monitors=None,
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
            sources: Explicit source list for overlays
            monitors: Explicit monitor list for overlays
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
                    "sources": sources,
                    "monitors": monitors,
                    "plane_2d": plane_2d,
                }
            else:
                if sources is not None:
                    self.metadata["sources"] = sources
                if monitors is not None:
                    self.metadata["monitors"] = monitors
                if boundaries is not None:
                    self.metadata["boundaries"] = boundaries
                if design is not None:
                    self.metadata["design"] = design

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
                    sources,
                    monitors,
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
        sources,
        monitors,
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

        actual_cmap = resolve_cmap(self.cmap)

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
            self._add_overlays(self._ax, design, boundaries, sources, monitors)

            # Add scale bar for clean visualization
            if self.clean_visualization:
                draw_scale_bar(
                    self._ax, design, wavelength=self.wavelength, fontsize=14
                )

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

    def _add_overlays(self, ax, design, boundaries, sources, monitors):
        """Add structure, source, monitor, and boundary overlays."""
        add_design_overlays(
            ax,
            design,
            line_color=self.line_color,
            line_opacity=self.line_opacity,
            sources=sources,
            monitors=monitors,
            skip_background=True,
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

    def _overlay_metadata(self):
        """Return stored overlay metadata for replayed frames."""
        return (
            self.metadata.get("design"),
            self.metadata.get("boundaries"),
            self.metadata.get("sources"),
            self.metadata.get("monitors"),
        )

    def _build_replay(self, fps, facecolor="none"):
        """Build a FuncAnimation from stored frames.

        Args:
            fps: Frames per second for the animation.
            facecolor: Figure background color ("none" for HTML5, "black" for MP4).

        Returns:
            Tuple of (fig, anim) or (None, None) if no frames stored.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if not self.frames:
            print("No frames stored. Enable store_frames=True.")
            return None, None

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
            fig.patch.set_facecolor(facecolor)
            ax = fig.add_axes([0, 0, 1, 1])
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

        actual_cmap = resolve_cmap(self.cmap)

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

        design, boundaries, sources, monitors = self._overlay_metadata()
        self._add_overlays(ax, design, boundaries, sources, monitors)

        if self.clean_visualization:
            draw_scale_bar(ax, design, wavelength=self.wavelength, fontsize=14)

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

        return fig, anim

    def get_animation(self, fps=30):
        """Create an HTML5 video animation from stored frames.

        Args:
            fps: Frames per second for the animation

        Returns:
            IPython.display.HTML: Playable HTML5 video animation
        """
        import matplotlib.pyplot as plt
        from IPython.display import HTML

        fig, anim = self._build_replay(fps, facecolor="none")
        if fig is None:
            return None

        plt.close(fig)

        # Increase embed limit for larger animations (default is ~20MB)
        import matplotlib as mpl

        old_limit = mpl.rcParams.get("animation.embed_limit", 20)
        mpl.rcParams["animation.embed_limit"] = 200  # 200 MB limit

        try:
            html_content = anim.to_jshtml()
            size_bytes = len(html_content.encode("utf-8"))
            size_mb = size_bytes / (1024 * 1024)
            print(f"Animation size: {size_mb:.1f} MB ({len(self.frames)} frames)")
            return HTML(html_content)
        finally:
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
        from matplotlib.animation import FFMpegWriter

        fig, anim = self._build_replay(fps, facecolor="black")
        if fig is None:
            return None

        print(f"Rendering {len(self.frames)} frames to {filename}...")
        try:
            writer = FFMpegWriter(fps=fps, metadata={"title": "BEAMZ Simulation"})
            anim.save(
                filename, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "black"}
            )
            plt.close(fig)

            size_bytes = os.path.getsize(filename)
            size_mb = size_bytes / (1024 * 1024)
            print(f"Video saved: {filename} ({size_mb:.1f} MB)")

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

        actual_cmap = resolve_cmap(self.cmap)

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

                design, boundaries, sources, monitors = self._overlay_metadata()
                self._add_overlays(ax, design, boundaries, sources, monitors)

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
