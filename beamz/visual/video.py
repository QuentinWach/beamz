"""Video recording for FDTD simulations."""

import numpy as np

from beamz.visual.design_viz import (
    add_design_overlays,
    configure_axes,
    draw_boundary,
    draw_scale_bar,
    resolve_cmap,
)


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

        actual_cmap = resolve_cmap(self.cmap)

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
                    add_design_overlays(
                        ax, self.design,
                        line_color=self.line_color,
                        line_opacity=self.line_opacity,
                    )

                    # Draw PML boundaries
                    if self.boundaries:
                        for boundary in self.boundaries:
                            draw_boundary(
                                ax, boundary, self.design,
                                edgecolor=self.line_color, linestyle=":", alpha=self.line_opacity,
                            )

                    # Add axis labels if not clean
                    if self.design is not None and not self.clean_visualization:
                        configure_axes(ax, self.design, plane_2d=self.plane_2d)

                    # Add scale bar for clean visualization
                    if self.clean_visualization and self.design is not None:
                        draw_scale_bar(ax, self.design, wavelength=self.wavelength)

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


