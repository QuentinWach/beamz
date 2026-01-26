from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import *
from beamz.design.core import Design
from beamz.devices.core import Device
from beamz.devices.monitors.monitors import Monitor
from beamz.simulation.boundaries import PML, Boundary
from beamz.simulation.fields import Fields
from beamz.simulation.ops import advance_e_field, advance_h_field
from beamz.visual.viz import VideoRecorder, animate_manual_field, close_fdtd_figure


class Simulation:
    """FDTD simulation class supporting both 2D and 3D electromagnetic simulations."""

    def __init__(
        self,
        design: Design = None,
        devices: list[Device] = [],
        boundaries: list[Boundary] = [],
        resolution: float = 0.02 * µm,
        time: np.ndarray = None,
        plane_2d: str = "xy",
    ):
        self.design = design
        self.resolution = resolution
        self.is_3d = design.is_3d and design.depth > 0
        self.plane_2d = plane_2d.lower()
        if self.plane_2d not in ["xy", "yz", "xz"]:
            self.plane_2d = "xy"

        # Get material grids from design (design owns the material grids, we reference them)
        permittivity, conductivity, permeability = design.get_material_grids(resolution)

        # Initialize time stepping first
        if time is None or len(time) < 2:
            raise ValueError("FDTD requires a time array with at least two entries")
        self.time, self.dt, self.num_steps = time, float(time[1] - time[0]), len(time)
        self.t, self.current_step = 0, 0

        # Create field storage (fields owns the E/H field arrays, references material grids)
        self.fields = Fields(
            permittivity, conductivity, permeability, resolution, plane_2d=self.plane_2d
        )

        # Initialize PML regions if present
        pml_boundaries = [b for b in boundaries if isinstance(b, PML)]
        if pml_boundaries:
            # Create PML regions (do this once, not every timestep)
            pml_data = {}
            for pml in pml_boundaries:
                pml_data.update(
                    pml.create_pml_regions(
                        self.fields, design, resolution, self.dt, plane_2d=self.plane_2d
                    )
                )
            self.pml_data = pml_data

            # Initialize split fields in Fields object - DEPRECATED/REMOVED in favor of effective conductivity
            # self.fields._init_upml_fields(pml_data)

            # Set effective conductivity for PML
            self.fields.set_pml_conductivity(pml_data)
        else:
            self.pml_data = None

        # Store device references (no duplication)
        self.devices = devices

        # Store boundary references (no duplication)
        self.boundaries = boundaries

    def step(self):
        """Perform one FDTD time step."""
        if self.current_step >= self.num_steps:
            return False

        # Inject source fields (if any) directly into the grid before update
        self._inject_sources()

        # Collect source terms from legacy devices (if any)
        source_j, source_m = self._collect_source_terms()

        # Update fields (legacy sources passed, new sources already injected)
        self.fields.update(self.dt, source_j=source_j, source_m=source_m)

        # Record monitor data (if monitors are in devices)
        self._record_monitors()

        # Update time and step counter
        self.t += self.dt
        self.current_step += 1
        return True

    def _record_monitors(self):
        """Record data from Monitor devices during simulation."""
        for device in self.devices:
            if hasattr(device, "should_record") and hasattr(device, "record_fields"):
                if device.should_record(self.current_step):
                    if not self.is_3d:
                        device.record_fields(
                            self.fields.Ez,
                            self.fields.Hx,
                            self.fields.Hy,
                            self.t,
                            self.resolution,
                            self.resolution,
                            self.current_step,
                        )
                    else:
                        device.record_fields(
                            self.fields.Ex,
                            self.fields.Ey,
                            self.fields.Ez,
                            self.fields.Hx,
                            self.fields.Hy,
                            self.fields.Hz,
                            self.t,
                            self.resolution,
                            self.resolution,
                            self.resolution,
                            self.current_step,
                        )

    def _inject_sources(self):
        """Inject source fields directly into the simulation grid."""
        for device in self.devices:
            if hasattr(device, "inject"):
                device.inject(
                    self.fields,
                    self.t,
                    self.dt,
                    self.current_step,
                    self.resolution,
                    self.design,
                )

    def _collect_source_terms(self):
        """Collect electric and magnetic current sources from all devices."""
        source_j = {}  # Electric currents for E-field update
        source_m = {}  # Magnetic currents for H-field update

        for device in self.devices:
            if hasattr(device, "get_source_terms"):
                j, m = device.get_source_terms(
                    self.fields,
                    self.t,
                    self.dt,
                    self.current_step,
                    self.resolution,
                    self.design,
                )
                if j:
                    source_j.update(j)
                if m:
                    source_m.update(m)

        return source_j, source_m

    def _create_jit_step_2d(self):
        """Create a JIT-compiled 2D FDTD step function for maximum performance.

        Returns a pure function that takes field arrays and returns updated field arrays.
        """
        # Extract static parameters for JIT compilation
        resolution = self.resolution
        dt = self.dt
        plane_2d = self.plane_2d

        # Material parameters (static for the simulation)
        eps_x, sig_x, region_x = (
            self.fields.eps_x,
            self.fields.sig_x,
            self.fields.region_x,
        )
        eps_y, sig_y, region_y = (
            self.fields.eps_y,
            self.fields.sig_y,
            self.fields.region_y,
        )
        eps_z, sig_z, region_z = (
            self.fields.eps_z,
            self.fields.sig_z,
            self.fields.region_z,
        )
        sigma_m_hx = self.fields.sigma_m_hx
        sigma_m_hy = self.fields.sigma_m_hy
        sigma_m_hz = self.fields.sigma_m_hz

        # Import curl operations
        from beamz.simulation.ops import curl_e_to_h_2d, curl_h_to_e_2d

        @jax.jit
        def step_2d(Ex, Ey, Ez, Hx, Hy, Hz):
            """Pure JIT-compiled FDTD step (no sources)."""
            # 1. Update H fields from E fields
            curlE_x, curlE_y, curlE_z = curl_e_to_h_2d(
                (Ex, Ey, Ez), resolution, plane=plane_2d
            )

            Hx_new = advance_h_field(Hx, curlE_x, sigma_m_hx, dt)
            Hy_new = advance_h_field(Hy, curlE_y, sigma_m_hy, dt)
            Hz_new = advance_h_field(Hz, curlE_z, sigma_m_hz, dt)

            # 2. Update E fields from H fields
            curlH_x, curlH_y, curlH_z = curl_h_to_e_2d(
                (Hx_new, Hy_new, Hz_new),
                resolution,
                (Ex.shape, Ey.shape, Ez.shape),
                plane=plane_2d,
            )

            Ex_new = advance_e_field(Ex, curlH_x, sig_x, eps_x, dt, region_x)
            Ey_new = advance_e_field(Ey, curlH_y, sig_y, eps_y, dt, region_y)
            Ez_new = advance_e_field(Ez, curlH_z, sig_z, eps_z, dt, region_z)

            return Ex_new, Ey_new, Ez_new, Hx_new, Hy_new, Hz_new

        return step_2d

    def _create_jit_step_3d(self):
        """Create a JIT-compiled 3D FDTD step function for maximum performance."""
        resolution = self.resolution
        dt = self.dt

        eps_x, sig_x, region_x = (
            self.fields.eps_x,
            self.fields.sig_x,
            self.fields.region_x,
        )
        eps_y, sig_y, region_y = (
            self.fields.eps_y,
            self.fields.sig_y,
            self.fields.region_y,
        )
        eps_z, sig_z, region_z = (
            self.fields.eps_z,
            self.fields.sig_z,
            self.fields.region_z,
        )
        sigma_m_hx = self.fields.sigma_m_hx
        sigma_m_hy = self.fields.sigma_m_hy
        sigma_m_hz = self.fields.sigma_m_hz

        from beamz.simulation.ops import curl_e_to_h_3d, curl_h_to_e_3d

        @jax.jit
        def step_3d(Ex, Ey, Ez, Hx, Hy, Hz):
            """Pure JIT-compiled 3D FDTD step (no sources)."""
            # 1. Update H fields from E fields
            curlE_x, curlE_y, curlE_z = curl_e_to_h_3d(Ex, Ey, Ez, resolution)

            Hx_new = advance_h_field(Hx, curlE_x, sigma_m_hx, dt)
            Hy_new = advance_h_field(Hy, curlE_y, sigma_m_hy, dt)
            Hz_new = advance_h_field(Hz, curlE_z, sigma_m_hz, dt)

            # 2. Update E fields from H fields
            curlH_x, curlH_y, curlH_z = curl_h_to_e_3d(
                Hx_new,
                Hy_new,
                Hz_new,
                resolution,
                ex_shape=Ex.shape,
                ey_shape=Ey.shape,
                ez_shape=Ez.shape,
            )

            Ex_new = advance_e_field(Ex, curlH_x, sig_x, eps_x, dt, region_x)
            Ey_new = advance_e_field(Ey, curlH_y, sig_y, eps_y, dt, region_y)
            Ez_new = advance_e_field(Ez, curlH_z, sig_z, eps_z, dt, region_z)

            return Ex_new, Ey_new, Ez_new, Hx_new, Hy_new, Hz_new

        return step_3d

    def run_fast(
        self, num_steps=None, record_interval=None, record_fields=None, progress=True
    ):
        """Run FDTD simulation with JIT-compiled loop for maximum performance.

        This method uses JAX's jax.lax.fori_loop for efficient time-stepping with full JIT compilation.
        Sources are injected at each step (not JIT-compiled), but the field update is fully optimized.

        Args:
            num_steps: Number of steps to run (default: remaining steps)
            record_interval: Record fields every N steps (default: None, don't record)
            record_fields: List of field names to record (default: ['Ez'])
            progress: Show progress bar (default: True)

        Returns:
            dict with:
                - 'fields': dict of recorded field arrays if record_interval was set
                - 'monitors': list of Monitor objects with recorded data
        """
        if num_steps is None:
            num_steps = self.num_steps - self.current_step

        if record_fields is None:
            record_fields = ["Ez"]

        # Create JIT-compiled step function
        jit_step = (
            self._create_jit_step_2d() if not self.is_3d else self._create_jit_step_3d()
        )

        # Warm up JIT (compile on first call)
        if progress:
            print("● JIT compiling FDTD kernel...", end=" ", flush=True)

        # Run one step to trigger compilation
        Ex, Ey, Ez, Hx, Hy, Hz = jit_step(
            self.fields.Ex,
            self.fields.Ey,
            self.fields.Ez,
            self.fields.Hx,
            self.fields.Hy,
            self.fields.Hz,
        )
        # Block until compilation is done
        Ex.block_until_ready()

        if progress:
            print("done!")

        # Initialize field history storage
        field_history = {name: [] for name in record_fields}

        # Main simulation loop with JIT-compiled steps
        try:
            for step_idx in range(num_steps):
                # Inject source fields (Python, not JIT-compiled)
                self._inject_sources()

                # Execute JIT-compiled field update
                (
                    self.fields.Ex,
                    self.fields.Ey,
                    self.fields.Ez,
                    self.fields.Hx,
                    self.fields.Hy,
                    self.fields.Hz,
                ) = jit_step(
                    self.fields.Ex,
                    self.fields.Ey,
                    self.fields.Ez,
                    self.fields.Hx,
                    self.fields.Hy,
                    self.fields.Hz,
                )

                # Record monitor data
                self._record_monitors()

                # Update time and step counter
                self.t += self.dt
                self.current_step += 1

                # Record fields if requested
                if record_interval and self.current_step % record_interval == 0:
                    for field_name in record_fields:
                        if hasattr(self.fields, field_name):
                            field_history[field_name].append(
                                np.array(getattr(self.fields, field_name))
                            )

                # Show progress
                if progress and (step_idx + 1) % max(1, num_steps // 20) == 0:
                    pct = 100 * (step_idx + 1) / num_steps
                    print(
                        f"\r● Progress: {pct:.0f}% ({step_idx + 1}/{num_steps} steps)",
                        end="",
                        flush=True,
                    )

            if progress:
                print()  # Newline after progress

        except KeyboardInterrupt:
            if progress:
                print(f"\n● Simulation interrupted at step {self.current_step}")

        # Collect monitor data
        monitors = [
            device for device in self.devices if hasattr(device, "power_history")
        ]

        # Convert field history to numpy arrays
        for name in field_history:
            if field_history[name]:
                field_history[name] = np.stack(field_history[name])

        result = {}
        if record_interval:
            result["fields"] = field_history
        if monitors:
            result["monitors"] = monitors

        return result if result else None

    def run_jit_scan(self, num_steps=None, progress=True):
        """Run FDTD simulation using jax.lax.scan for maximum performance.

        This method is optimized for simulations WITHOUT sources or with sources
        that can be pre-computed. It JIT-compiles the entire time loop.

        For simulations WITH time-dependent sources, use run_fast() instead.

        Args:
            num_steps: Number of steps to run (default: remaining steps)
            progress: Show compilation status (default: True)

        Returns:
            dict with final field state
        """
        if num_steps is None:
            num_steps = self.num_steps - self.current_step

        # Check if sources are present
        has_sources = any(
            hasattr(d, "inject") or hasattr(d, "get_source_terms") for d in self.devices
        )
        if has_sources:
            print(
                "● Warning: Sources detected. Using run_fast() instead for source injection support."
            )
            return self.run_fast(num_steps=num_steps, progress=progress)

        # Create pure FDTD step function for scan
        jit_step = (
            self._create_jit_step_2d() if not self.is_3d else self._create_jit_step_3d()
        )

        @jax.jit
        def scan_body(carry, _):
            Ex, Ey, Ez, Hx, Hy, Hz = carry
            Ex, Ey, Ez, Hx, Hy, Hz = jit_step(Ex, Ey, Ez, Hx, Hy, Hz)
            return (Ex, Ey, Ez, Hx, Hy, Hz), None

        if progress:
            print(
                f"● JIT compiling {num_steps}-step FDTD loop with jax.lax.scan...",
                end=" ",
                flush=True,
            )

        # Pack initial state
        init_state = (
            self.fields.Ex,
            self.fields.Ey,
            self.fields.Ez,
            self.fields.Hx,
            self.fields.Hy,
            self.fields.Hz,
        )

        # Run scan
        final_state, _ = jax.lax.scan(scan_body, init_state, None, length=num_steps)

        # Unpack final state
        (
            self.fields.Ex,
            self.fields.Ey,
            self.fields.Ez,
            self.fields.Hx,
            self.fields.Hy,
            self.fields.Hz,
        ) = final_state

        # Block until done
        self.fields.Ez.block_until_ready()

        if progress:
            print("done!")

        # Update time tracking
        self.t += num_steps * self.dt
        self.current_step += num_steps

        return {
            "Ex": np.array(self.fields.Ex),
            "Ey": np.array(self.fields.Ey),
            "Ez": np.array(self.fields.Ez),
            "Hx": np.array(self.fields.Hx),
            "Hy": np.array(self.fields.Hy),
            "Hz": np.array(self.fields.Hz),
        }

    def run(
        self,
        animate_live=None,
        animation_interval=10,
        axis_scale=None,
        cmap="twilight_zero",
        clean_visualization=False,
        wavelength=None,
        line_color="gray",
        line_opacity=0.5,
        save_fields=None,
        field_subsample=1,
        save_video=None,
        video_fps=30,
        video_dpi=150,
        video_field=None,
        interpolation="bicubic",
        jupyter_live=None,
        store_animation=True,
    ):
        """Run complete FDTD simulation with optional live field visualization.

        Args:
            animate_live: Field component to animate ('Ez', 'Hx', 'Hy', 'Ex', 'Ey', etc.) or None to disable
            animation_interval: Update visualization every N steps (higher = faster but less smooth)
            axis_scale: Tuple (min, max) for fixed color scale during animation, or None for auto-scaling
            cmap: Matplotlib colormap name (default: 'twilight_zero')
            clean_visualization: If True, hide axes, title, and colorbar (only show field and structures)
            wavelength: Wavelength for scale bar calculation (if None, tries to extract from devices)
            line_color: Color for structure and PML boundary outlines (default: 'gray')
            line_opacity: Opacity/transparency of structure and PML boundary outlines (0.0 to 1.0, default: 0.5)
            save_fields: List of field components to save ('Ez', 'Hx', etc.) or None to disable
            field_subsample: Save fields every N steps (default: 1, save all steps)
            save_video: Path to save MP4 video (e.g., 'simulation.mp4') or None to disable
            video_fps: Frames per second for the video (default: 30)
            video_dpi: Resolution (dots per inch) for video frames (default: 150)
            video_field: Field component to record for video ('Ez', 'Hx', etc.), defaults to animate_live if set
            interpolation: Interpolation method for field display ('nearest', 'bilinear', 'bicubic', etc.)
            jupyter_live: Override Jupyter environment detection (None=auto, True=Jupyter, False=script)
            store_animation: Store animation frames for replay in Jupyter (default: True)

        Returns:
            dict with keys:
                - 'fields': dict of field histories if save_fields was provided
                - 'monitors': list of Monitor objects with recorded data
                - 'animation': JupyterAnimator object if running in Jupyter with animate_live
        """
        # Handle 3D simulations - use monitor slice if available
        active_monitor = None
        if animate_live and self.is_3d:
            active_monitor = next(
                (d for d in self.devices if isinstance(d, Monitor) and d.is_3d), None
            )
            if not active_monitor:
                # print("● Live animation for 3D simulations requires a Monitor (add one to devices)")
                animate_live = None

        # Initialize animation context if requested
        viz_context = None
        if animate_live:
            # Validate field component exists
            available = self.fields.available_components()
            if animate_live not in available:
                # print(f"● Warning: Field '{animate_live}' not found. Available: {available}")
                animate_live = None

        # Extract wavelength from devices if not provided
        if wavelength is None:
            for device in self.devices:
                if hasattr(device, "wavelength"):
                    wavelength = device.wavelength
                    break

        # Detect Jupyter environment and initialize animator if needed
        from beamz.visual.viz import JupyterAnimator, is_jupyter_environment

        use_jupyter = (
            jupyter_live if jupyter_live is not None else is_jupyter_environment()
        )

        jupyter_animator = None
        if animate_live and use_jupyter:
            jupyter_animator = JupyterAnimator(
                cmap=cmap,
                axis_scale=axis_scale,
                clean_visualization=clean_visualization,
                wavelength=wavelength,
                line_color=line_color,
                line_opacity=line_opacity,
                interpolation=interpolation,
                live_display=True,
                store_frames=store_animation,
            )

        # Initialize video recorder if requested
        video_recorder = None
        if save_video:
            # Determine which field to record for video
            record_field = (
                video_field if video_field else (animate_live if animate_live else "Ez")
            )
            # Validate field component exists
            available = self.fields.available_components()
            if record_field not in available:
                print(
                    f"Warning: Field '{record_field}' not found for video. Available: {available}"
                )
                record_field = available[0] if available else None
            if record_field:
                video_recorder = VideoRecorder(
                    filename=save_video,
                    fps=video_fps,
                    dpi=video_dpi,
                    cmap=cmap,
                    axis_scale=axis_scale,
                    clean_visualization=clean_visualization,
                    wavelength=wavelength,
                    line_color=line_color,
                    line_opacity=line_opacity,
                    interpolation=interpolation,
                )

        # Initialize field storage if requested
        field_history = {}
        if save_fields:
            for field_name in save_fields:
                field_history[field_name] = []

        try:
            # Main simulation loop
            while self.step():
                # Save field history if requested
                # current_step is incremented in step(), so we check after increment
                if save_fields and (self.current_step % field_subsample == 0):
                    for field_name in save_fields:
                        if hasattr(self.fields, field_name):
                            field_history[field_name].append(
                                getattr(self.fields, field_name).copy()
                            )

                # Record video frame if enabled
                if video_recorder and self.current_step % animation_interval == 0:
                    record_field = (
                        video_field
                        if video_field
                        else (animate_live if animate_live else "Ez")
                    )
                    if hasattr(self.fields, record_field):
                        field_display = getattr(self.fields, record_field)
                        # Convert to V/µm for E-fields
                        field_display = (
                            field_display * 1e-6
                            if "E" in record_field
                            else field_display
                        )
                        extent = (0, self.design.width, 0, self.design.height)
                        video_recorder.add_frame(
                            field_display,
                            t=self.t,
                            step=self.current_step,
                            num_steps=self.num_steps,
                            field_name=record_field,
                            units="V/µm" if "E" in record_field else "A/m",
                            extent=extent,
                            design=self.design,
                            boundaries=self.boundaries,
                            plane_2d=self.plane_2d,
                        )

                # Update live animation if enabled
                if animate_live and self.current_step % animation_interval == 0:
                    if self.is_3d and active_monitor:
                        # Use monitor fields for 3D animation
                        if (
                            animate_live in active_monitor.fields
                            and active_monitor.fields[animate_live]
                        ):
                            field_display = active_monitor.fields[animate_live][-1]
                            # print(f"● 3D Animation slice shape: {field_display.shape}")
                            # Use monitor's physical extent
                            extent = (
                                active_monitor.start[0],
                                active_monitor.start[0] + active_monitor.size[0],
                                active_monitor.start[1],
                                active_monitor.start[1] + active_monitor.size[1],
                            )
                        else:
                            continue
                    else:
                        # Standard 2D animation
                        field_display = getattr(self.fields, animate_live)
                        extent = (0, self.design.width, 0, self.design.height)

                    # Convert to V/µm for display
                    field_display = (
                        field_display * 1e-6 if "E" in animate_live else field_display
                    )

                    if use_jupyter and jupyter_animator:
                        # Use Jupyter animator for notebook display
                        jupyter_animator.update(
                            field_display,
                            t=self.t,
                            step=self.current_step,
                            num_steps=self.num_steps,
                            field_name=animate_live,
                            units="V/µm" if "E" in animate_live else "A/m",
                            extent=extent,
                            design=self.design,
                            boundaries=self.boundaries,
                            plane_2d=self.plane_2d,
                        )
                    else:
                        # Use script-based matplotlib animation
                        title = f"{animate_live} at t = {self.t:.2e} s (step {self.current_step}/{self.num_steps})"
                        viz_context = animate_manual_field(
                            field_display,
                            context=viz_context,
                            extent=extent,
                            title=title,
                            units="V/µm" if "E" in animate_live else "A/m",
                            design=self.design,
                            boundaries=self.boundaries,
                            pause=0.001,
                            axis_scale=axis_scale,
                            cmap=cmap,
                            clean_visualization=clean_visualization,
                            wavelength=wavelength,
                            line_color=line_color,
                            line_opacity=line_opacity,
                            plane_2d=self.plane_2d,
                            interpolation=interpolation,
                        )
        finally:
            # Save video if recorder was used
            if video_recorder:
                video_recorder.save()

            # Cleanup Jupyter animator figure
            if jupyter_animator:
                jupyter_animator.finalize()

            # Cleanup: keep the final frame visible (script mode only)
            if not use_jupyter and viz_context and viz_context.get("fig"):
                import matplotlib.pyplot as plt

                plt.show(block=False)
                print("Simulation complete. Close the plot window to continue.")

        # Collect monitor data
        monitors = [
            device for device in self.devices if hasattr(device, "power_history")
        ]

        # Return results
        result = {}
        if save_fields:
            result["fields"] = field_history
        if monitors:
            result["monitors"] = monitors
        if jupyter_animator and jupyter_animator.frames:
            result["animation"] = jupyter_animator

        return result if result else None
