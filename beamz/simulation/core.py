import numpy as np
from beamz.const import *
from beamz.design.core import Design
from beamz.devices.core import Device
from beamz.devices.monitors.monitors import Monitor
from beamz.simulation.fields import Fields
from beamz.simulation.boundaries import Boundary, PML
from beamz.visual.viz import animate_manual_field, close_fdtd_figure, VideoRecorder

class Simulation:
    """FDTD simulation class supporting both 2D and 3D electromagnetic simulations."""
    def __init__(self,
        design:Design=None,
        devices:list[Device]=[],
        boundaries:list[Boundary]=[],
        resolution:float=0.02*µm,
        time:np.ndarray=None,
        plane_2d:str='xy'):
        self.design = design
        self.resolution = resolution
        self.is_3d = design.is_3d and design.depth > 0
        self.plane_2d = plane_2d.lower()
        if self.plane_2d not in ['xy', 'yz', 'xz']: self.plane_2d = 'xy'
        
        # Get material grids from design (design owns the material grids, we reference them)
        permittivity, conductivity, permeability = design.get_material_grids(resolution)
        
        # Initialize time stepping first
        if time is None or len(time) < 2: raise ValueError("FDTD requires a time array with at least two entries")
        self.time, self.dt, self.num_steps = time, float(time[1] - time[0]), len(time)
        self.t, self.current_step = 0, 0
        
        # Create field storage (fields owns the E/H field arrays, references material grids)
        self.fields = Fields(permittivity, conductivity, permeability, resolution, plane_2d=self.plane_2d)
        
        # Initialize PML regions if present
        pml_boundaries = [b for b in boundaries if isinstance(b, PML)]
        if pml_boundaries:
            # Create PML regions (do this once, not every timestep)
            pml_data = {}
            for pml in pml_boundaries:
                pml_data.update(pml.create_pml_regions(self.fields, design, resolution, self.dt, plane_2d=self.plane_2d))
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
        if self.current_step >= self.num_steps: return False
        
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
            if hasattr(device, 'should_record') and hasattr(device, 'record_fields'):
                if device.should_record(self.current_step):
                    if not self.is_3d:
                        device.record_fields(self.fields.Ez, self.fields.Hx, self.fields.Hy,
                                           self.t, self.resolution, self.resolution, self.current_step)
                    else:
                        device.record_fields(self.fields.Ex, self.fields.Ey, self.fields.Ez,
                                           self.fields.Hx, self.fields.Hy, self.fields.Hz,
                                           self.t, self.resolution, self.resolution, self.resolution, self.current_step)

    def _inject_sources(self):
        """Inject source fields directly into the simulation grid."""
        for device in self.devices:
            if hasattr(device, 'inject'):
                device.inject(self.fields, self.t, self.dt, self.current_step, self.resolution, self.design)

    def _collect_source_terms(self):
        """Collect electric and magnetic current sources from all devices."""
        source_j = {}  # Electric currents for E-field update
        source_m = {}  # Magnetic currents for H-field update
        
        for device in self.devices:
            if hasattr(device, 'get_source_terms'):
                j, m = device.get_source_terms(self.fields, self.t, self.dt, self.current_step, self.resolution, self.design)
                if j: source_j.update(j)
                if m: source_m.update(m)
        
        return source_j, source_m
    

    def run(self, animate_live=None, animation_interval=10, axis_scale=None, cmap='twilight_zero', clean_visualization=False, wavelength=None, line_color='gray', line_opacity=0.5, save_fields=None, field_subsample=1, save_video=None, video_fps=30, video_dpi=150, video_field=None, interpolation='bicubic', jupyter_live=None, store_animation=True):
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
            active_monitor = next((d for d in self.devices if isinstance(d, Monitor) and d.is_3d), None)
            if not active_monitor:
                #print("● Live animation for 3D simulations requires a Monitor (add one to devices)")
                animate_live = None
        
        # Initialize animation context if requested
        viz_context = None
        if animate_live:
            # Validate field component exists
            available = self.fields.available_components()
            if animate_live not in available:
                #print(f"● Warning: Field '{animate_live}' not found. Available: {available}")
                animate_live = None
        
        # Extract wavelength from devices if not provided
        if wavelength is None:
            for device in self.devices:
                if hasattr(device, 'wavelength'):
                    wavelength = device.wavelength
                    break

        # Detect Jupyter environment and initialize animator if needed
        from beamz.visual.viz import is_jupyter_environment, JupyterAnimator
        use_jupyter = jupyter_live if jupyter_live is not None else is_jupyter_environment()

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
                store_frames=store_animation
            )

        # Initialize video recorder if requested
        video_recorder = None
        if save_video:
            # Determine which field to record for video
            record_field = video_field if video_field else (animate_live if animate_live else 'Ez')
            # Validate field component exists
            available = self.fields.available_components()
            if record_field not in available:
                print(f"Warning: Field '{record_field}' not found for video. Available: {available}")
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
                    interpolation=interpolation
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
                            field_history[field_name].append(getattr(self.fields, field_name).copy())

                # Record video frame if enabled
                if video_recorder and self.current_step % animation_interval == 0:
                    record_field = video_field if video_field else (animate_live if animate_live else 'Ez')
                    if hasattr(self.fields, record_field):
                        field_display = getattr(self.fields, record_field)
                        # Convert to V/µm for E-fields
                        field_display = field_display * 1e-6 if 'E' in record_field else field_display
                        extent = (0, self.design.width, 0, self.design.height)
                        video_recorder.add_frame(
                            field_display,
                            t=self.t,
                            step=self.current_step,
                            num_steps=self.num_steps,
                            field_name=record_field,
                            units='V/µm' if 'E' in record_field else 'A/m',
                            extent=extent,
                            design=self.design,
                            boundaries=self.boundaries,
                            plane_2d=self.plane_2d
                        )

                # Update live animation if enabled
                if animate_live and self.current_step % animation_interval == 0:
                    if self.is_3d and active_monitor:
                        # Use monitor fields for 3D animation
                        if animate_live in active_monitor.fields and active_monitor.fields[animate_live]:
                            field_display = active_monitor.fields[animate_live][-1]
                            #print(f"● 3D Animation slice shape: {field_display.shape}")
                            # Use monitor's physical extent
                            extent = (active_monitor.start[0], active_monitor.start[0] + active_monitor.size[0],
                                      active_monitor.start[1], active_monitor.start[1] + active_monitor.size[1])
                        else:
                            continue
                    else:
                        # Standard 2D animation
                        field_display = getattr(self.fields, animate_live)
                        extent = (0, self.design.width, 0, self.design.height)

                    # Convert to V/µm for display
                    field_display = field_display * 1e-6 if 'E' in animate_live else field_display

                    if use_jupyter and jupyter_animator:
                        # Use Jupyter animator for notebook display
                        jupyter_animator.update(
                            field_display,
                            t=self.t,
                            step=self.current_step,
                            num_steps=self.num_steps,
                            field_name=animate_live,
                            units='V/µm' if 'E' in animate_live else 'A/m',
                            extent=extent,
                            design=self.design,
                            boundaries=self.boundaries,
                            plane_2d=self.plane_2d
                        )
                    else:
                        # Use script-based matplotlib animation
                        title = f'{animate_live} at t = {self.t:.2e} s (step {self.current_step}/{self.num_steps})'
                        viz_context = animate_manual_field(field_display, context=viz_context, extent=extent,
                                                          title=title, units='V/µm' if 'E' in animate_live else 'A/m',
                                                          design=self.design, boundaries=self.boundaries, pause=0.001,
                                                          axis_scale=axis_scale, cmap=cmap, clean_visualization=clean_visualization,
                                                          wavelength=wavelength, line_color=line_color, line_opacity=line_opacity,
                                                          plane_2d=self.plane_2d, interpolation=interpolation)
        finally:
            # Save video if recorder was used
            if video_recorder:
                video_recorder.save()

            # Cleanup Jupyter animator figure
            if jupyter_animator:
                jupyter_animator.finalize()

            # Cleanup: keep the final frame visible (script mode only)
            if not use_jupyter and viz_context and viz_context.get('fig'):
                import matplotlib.pyplot as plt
                plt.show(block=False)
                print("Simulation complete. Close the plot window to continue.")

        # Collect monitor data
        monitors = [device for device in self.devices if hasattr(device, 'power_history')]

        # Return results
        result = {}
        if save_fields:
            result['fields'] = field_history
        if monitors:
            result['monitors'] = monitors
        if jupyter_animator and jupyter_animator.frames:
            result['animation'] = jupyter_animator

        return result if result else None