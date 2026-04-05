"""FDTD live animation and frame-by-frame replay."""

import numpy as np

from beamz.visual.design_viz import draw_boundary
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


def _format_frame_title(field_name, t, step, num_steps):
    return f"{field_name} at t = {t:.2e} s (step {step}/{num_steps})"


def _figure_is_open(fig):
    import matplotlib.pyplot as plt

    number = getattr(fig, "number", None)
    return number is not None and plt.fignum_exists(number)


def _resolve_limits(axis_scale, vmax):
    if axis_scale is not None:
        return axis_scale
    vmax = vmax if vmax > 0 else 1.0
    return -vmax, vmax


def _create_figure_axes(*, clean_visualization, extent=None, facecolor=None):
    import matplotlib.pyplot as plt

    if clean_visualization and extent:
        data_width = extent[1] - extent[0]
        data_height = extent[3] - extent[2]
        aspect_ratio = data_width / data_height
        fig_height = 8
        fig_width = fig_height * aspect_ratio
        fig = plt.figure(figsize=(fig_width, fig_height))
        if facecolor is not None:
            fig.patch.set_facecolor(facecolor)
        ax = fig.add_axes([0, 0, 1, 1])
        return fig, ax

    fig, ax = plt.subplots(figsize=(10, 8))
    if facecolor is not None:
        fig.patch.set_facecolor(facecolor)
    return fig, ax


def _apply_static_overlays(
    ax,
    *,
    design,
    boundaries,
    line_color,
    line_opacity,
    plane_2d,
    clean_visualization,
    wavelength,
    show_structures=True,
    show_sources=True,
    show_monitors=True,
    skip_background=False,
):
    if design is not None and show_structures:
        add_design_overlays(
            ax,
            design,
            line_color=line_color,
            line_opacity=line_opacity,
            sources=getattr(design, "sources", []) if show_sources else [],
            show_monitors=show_monitors,
            skip_background=skip_background,
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
        draw_scale_bar(ax, design, wavelength=wavelength, fontsize=14)


def _create_frame_artists(
    data,
    *,
    extent,
    cmap,
    vmin,
    vmax,
    interpolation,
    clean_visualization,
    title_text,
    field_name,
    units,
    design,
    boundaries,
    line_color,
    line_opacity,
    plane_2d,
    wavelength,
    show_structures=True,
    show_sources=True,
    show_monitors=True,
    skip_background=False,
    facecolor=None,
):
    import matplotlib.pyplot as plt

    fig, ax = _create_figure_axes(
        clean_visualization=clean_visualization,
        extent=extent,
        facecolor=facecolor,
    )
    im = ax.imshow(
        data,
        origin="lower",
        cmap=resolve_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation=interpolation,
    )
    if clean_visualization:
        ax.set_axis_off()
        ax.set_frame_on(False)
        cbar = None
        title = None
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    else:
        cbar = plt.colorbar(im, ax=ax, label=f"{field_name} ({units})")
        title = ax.set_title(title_text) if title_text else None
        plt.tight_layout()

    _apply_static_overlays(
        ax,
        design=design,
        boundaries=boundaries,
        line_color=line_color,
        line_opacity=line_opacity,
        plane_2d=plane_2d,
        clean_visualization=clean_visualization,
        wavelength=wavelength,
        show_structures=show_structures,
        show_sources=show_sources,
        show_monitors=show_monitors,
        skip_background=skip_background,
    )
    return fig, ax, im, cbar, title


def _update_frame_artists(im, cbar, title, data, *, vmin, vmax, title_text=None):
    im.set_data(data)
    im.set_clim(vmin, vmax)
    if title is not None and title_text:
        title.set_text(title_text)
    if cbar is not None:
        cbar.mappable.set_clim(vmin, vmax)


def _refresh_live_canvas(fig, pause=0.0, *, force_draw=False):
    canvas = fig.canvas
    if force_draw:
        canvas.draw()
    else:
        canvas.draw_idle()
    canvas.flush_events()
    if pause and pause > 0:
        try:
            canvas.start_event_loop(float(pause))
        except Exception:
            import matplotlib.pyplot as plt

            plt.pause(pause)


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
    elif context.get("closed"):
        return context
    elif context.get("fig") is not None and not _figure_is_open(context["fig"]):
        context["closed"] = True
        return context

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
        field_name = "Field"
        if title and " at t =" in title:
            field_name = title.split(" at t =")[0]
        fig, ax, im, cbar, title_obj = _create_frame_artists(
            data,
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            clean_visualization=clean_visualization,
            title_text=title,
            field_name=field_name,
            units=units,
            design=design,
            boundaries=boundaries,
            line_color=line_color,
            line_opacity=line_opacity,
            plane_2d=plane_2d,
            wavelength=wavelength,
            show_structures=show_structures,
            show_sources=show_sources,
            show_monitors=show_monitors,
        )
        context["closed"] = False
        fig.canvas.mpl_connect(
            "close_event", lambda event: context.__setitem__("closed", True)
        )
        plt.show(block=False)
        _refresh_live_canvas(fig, pause=pause, force_draw=True)
        context.update(
            {
                "fig": fig,
                "ax": ax,
                "im": im,
                "cbar": cbar,
                "title": title_obj,
                "frame": 1,
                "clean_visualization": clean_visualization,
                "wavelength": wavelength,
            }
        )
        context.setdefault("auto_scale", vmax if axis_scale is None else None)
        return context

    # Update existing plot
    clean_visualization = context.get("clean_visualization", False)
    if not _figure_is_open(context["fig"]):
        context["closed"] = True
        return context
    _update_frame_artists(
        context["im"],
        context.get("cbar"),
        None if clean_visualization else context.get("title"),
        data,
        vmin=vmin,
        vmax=vmax,
        title_text=title,
    )
    context["frame"] = context.get("frame", 0) + 1
    fig = context["fig"]
    _refresh_live_canvas(fig, pause=pause)
    return context


class JupyterAnimator:
    """Handles live animation and replay for Jupyter notebooks.

    Provides two modes:
    1. Live mode: Updates cell output during simulation using clear_output + display
    2. Replay mode: Returns an HTML animation or MP4 after simulation

    Usage:
        animator = JupyterAnimator(...)
        # During simulation loop:
        animator.update(field_array, t, step, num_steps)
        # After simulation:
        animation = animator.get_animation()  # Returns playable HTML5 video
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

        title_text = _format_frame_title(field_name, t, step, num_steps)
        vmin, vmax = _resolve_limits(self.axis_scale, self._global_vmax)

        # First frame: create the figure and all elements
        if self._fig is None:
            self._fig, self._ax, self._im, self._cbar, self._title = _create_frame_artists(
                frame_data,
                extent=extent,
                cmap=self.cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation=self.interpolation,
                clean_visualization=self.clean_visualization,
                title_text=title_text,
                field_name=field_name,
                units=units,
                design=design,
                boundaries=boundaries,
                line_color=self.line_color,
                line_opacity=self.line_opacity,
                plane_2d=plane_2d,
                wavelength=self.wavelength,
                skip_background=True,
                facecolor="none",
            )

        else:
            _update_frame_artists(
                self._im,
                self._cbar,
                self._title,
                frame_data,
                vmin=vmin,
                vmax=vmax,
                title_text=title_text,
            )

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

        extent = self.metadata.get("extent")
        vmin, vmax = _resolve_limits(self.axis_scale, self._global_vmax)
        fig, ax, im, _, title = _create_frame_artists(
            self.frames[0],
            extent=extent,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=self.interpolation,
            clean_visualization=self.clean_visualization,
            title_text="",
            field_name=self.metadata.get("field_name", "Field"),
            units=self.metadata.get("units", ""),
            design=self.metadata.get("design"),
            boundaries=self.metadata.get("boundaries"),
            line_color=self.line_color,
            line_opacity=self.line_opacity,
            plane_2d=self.metadata.get("plane_2d", "xy"),
            wavelength=self.wavelength,
            skip_background=True,
            facecolor=facecolor,
        )

        def update(frame_idx):
            title_text = None
            if title is not None and self.times:
                t, step, num_steps = self.times[frame_idx]
                title_text = _format_frame_title(
                    self.metadata.get("field_name", "Field"), t, step, num_steps
                )
            _update_frame_artists(
                im,
                None,
                title,
                self.frames[frame_idx],
                vmin=vmin,
                vmax=vmax,
                title_text=title_text,
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
        try:
            from IPython.display import Video
        except ImportError:
            Video = None

        path = save_animation_mp4(self, filename=filename, fps=fps, dpi=dpi)
        if path is None or Video is None:
            return None
        return Video(path, embed=True)


def save_animation_mp4(animator, *, filename="animation.mp4", fps=30, dpi=150):
    """Render stored frames to an MP4 file and return the output path."""
    import os

    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    fig, anim = animator._build_replay(fps, facecolor="black")
    if fig is None:
        return None

    print(f"Rendering {len(animator.frames)} frames to {filename}...")
    try:
        writer = FFMpegWriter(fps=fps, metadata={"title": "BEAMZ Simulation"})
        anim.save(filename, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "black"})
        size_bytes = os.path.getsize(filename)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Video saved: {filename} ({size_mb:.1f} MB)")
        return filename
    except Exception as exc:
        print(f"Error creating video: {exc}")
        print("Make sure ffmpeg is installed: conda install ffmpeg")
        return None
    finally:
        plt.close(fig)
