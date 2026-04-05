"""Compatibility wrapper for saving simulation frames to video."""

from __future__ import annotations

from beamz.visual.animation import JupyterAnimator, save_animation_mp4


class VideoRecorder:
    """Collect frames and render them to MP4 via the shared animation pipeline."""

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
        self.filename = filename
        self.fps = fps
        self.dpi = dpi
        self._animator = JupyterAnimator(
            cmap=cmap,
            axis_scale=axis_scale,
            clean_visualization=clean_visualization,
            wavelength=wavelength,
            line_color=line_color,
            line_opacity=line_opacity,
            interpolation=interpolation,
            live_display=False,
            store_frames=True,
        )

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
        self._animator.update(
            field_array,
            t,
            step,
            num_steps,
            field_name=field_name,
            units=units,
            extent=extent,
            design=design,
            boundaries=boundaries,
            plane_2d=plane_2d,
        )

    def save(self):
        return save_animation_mp4(
            self._animator,
            filename=self.filename,
            fps=self.fps,
            dpi=self.dpi,
        )
