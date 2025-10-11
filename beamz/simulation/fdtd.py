from typing import Dict, Optional, Sequence
import datetime
import numpy as np

from beamz.const import *
from beamz.simulation.meshing import BaseMeshGrid, RegularGrid, RegularGrid3D
from beamz.design.core import *
from beamz.devices.sources import ModeSource, GaussianSource
from beamz.devices.monitors import Monitor
from beamz.simulation.backends import get_backend
from beamz.helpers import display_status, create_rich_progress, display_parameters, display_time_elapsed
from beamz import viz as viz
from beamz.simulation import helper as sim_helper


class FDTD:
    """FDTD simulation class supporting both 2D and 3D electromagnetic simulations.
    
    Automatically detects whether the design is 2D or 3D and uses appropriate Maxwell equations:
    - 2D: TE-polarized (Ez, Hx, Hy fields)
    - 3D: Full Maxwell equations (Ex, Ey, Ez, Hx, Hy, Hz fields)
    """
    def __init__(self, design, time, mesh: str = "regular", resolution: float = 0.02*µm, backend="numpy", 
                        backend_options=None, devices=None):
        provided_mesh = None
        using_provided_mesh = False

        if isinstance(design, BaseMeshGrid):
            provided_mesh = design
            using_provided_mesh = True
            mesh_design = getattr(provided_mesh, "design", None)
            if mesh_design is None:
                raise ValueError("Provided mesh must expose its originating design via the `design` attribute.")
            design = mesh_design

        # Initialize the design and detect dimensionality
        self.design = design
        self.is_3d = design.is_3d and design.depth > 0

        mesh_obj = None
        if provided_mesh is not None:
            mesh_obj = provided_mesh
        elif isinstance(mesh, BaseMeshGrid):
            mesh_obj = mesh
            using_provided_mesh = True
        elif mesh == "regular":
            if self.is_3d:
                mesh_obj = RegularGrid3D(design=self.design, resolution_xy=resolution)
            else:
                mesh_obj = RegularGrid(design=self.design, resolution=resolution)
        else:
            raise ValueError(f"Unsupported mesh specification: {mesh!r}")

        if mesh_obj is None:
            raise ValueError("Failed to initialize FDTD mesh.")

        if hasattr(mesh_obj, "design") and mesh_obj.design is not None and mesh_obj.design is not self.design:
            raise ValueError("Provided mesh was generated from a different design instance.")

        if self.is_3d and not isinstance(mesh_obj, RegularGrid3D):
            raise ValueError("3D designs require a RegularGrid3D mesh.")
        if not self.is_3d and not isinstance(mesh_obj, RegularGrid):
            raise ValueError("2D designs require a RegularGrid mesh.")

        self.mesh = mesh_obj

        if hasattr(self.mesh, "resolution"):
            self.resolution = self.mesh.resolution
        elif hasattr(self.mesh, "resolution_xy"):
            self.resolution = self.mesh.resolution_xy
        else:
            self.resolution = resolution

        if using_provided_mesh:
            display_status("Using provided rasterized mesh", "info")

        if self.is_3d:
            display_status("Using 3D FDTD with full Maxwell equations", "info")
        else:
            display_status("Using 2D FDTD with TE-polarized Maxwell equations", "info")
        
        # Set grid resolutions
        self.dx = self.mesh.dx if hasattr(self.mesh, 'dx') else self.mesh.resolution_xy
        self.dy = self.mesh.dy if hasattr(self.mesh, 'dy') else self.mesh.resolution_xy
        if self.is_3d: self.dz = self.mesh.dz if hasattr(self.mesh, 'dz') else self.mesh.resolution_z
        
        # Set grid dimensions from mesh
        if self.is_3d: self.nz, self.ny, self.nx = self.mesh.permittivity.shape
        else: self.ny, self.nx = self.mesh.permittivity.shape

        # Get material properties
        self.epsilon_r = self.mesh.permittivity
        self.mu_r = self.mesh.permeability
        self.sigma = self.mesh.conductivity
        
        # Initialize the backend
        backend_options = backend_options or {}
        self.backend = get_backend(name=backend, **backend_options)
        
        # Initialize fields based on dimensionality
        self._init_fields()
        
        # Convert material properties to backend arrays
        self.epsilon_r = self.backend.from_numpy(self.epsilon_r)
        self.mu_r = self.backend.from_numpy(self.mu_r)
        self.sigma = self.backend.from_numpy(self.sigma)
        
        # Initialize the time
        self.time = time
        self.dt = self.time[1] - self.time[0]
        self.num_steps = len(self.time)
        
        # Initialize the sources
        if devices is None:
            devices = []
        self.sources = list(self.design.sources)
        self.monitors = list(self.design.monitors)

        for device in devices:
            if isinstance(device, (ModeSource, GaussianSource)):
                if device.design is None:
                    device.design = self.design
                self.sources.append(device)
            elif isinstance(device, Monitor):
                if device.design is None:
                    device.design = self.design
                self.monitors.append(device)
            else:
                raise TypeError(f"Unsupported device type: {type(device)}")
        
        # Grid-aligned mode solving: Recompute mode profiles on FDTD grid for perfect injection
        for source in self.sources:
            if isinstance(source, ModeSource):
                if self.is_3d:
                    source.compute_modes_on_fdtd_grid(self.dx, self.dy, self.dz)
                else:
                    source.compute_modes_on_fdtd_grid(self.dx, self.dy)
        
        # Initialize the results based on dimensionality
        if self.is_3d: self.results = {"Ex": [], "Ey": [], "Ez": [], "Hx": [], "Hy": [], "Hz": [], "t": []}
        else: self.results = {"Ez": [], "Hx": [], "Hy": [], "t": []}
            
        # Initialize animation attributes
        self.fig = None
        self.ax = None
        self.anim = None
        self.im = None
        
        # Initialize monitor data storage
        self.monitor_data = {}
        
        # Initialize power accumulation
        self.power_accumulated = None
        self.power_accumulation_count = 0
        
        # Initialize simulation start time
        self.start_time = None

    def _init_fields(self):
        """Initialize field arrays based on dimensionality."""
        if self.is_3d: 
            try:
                # Yee grid staggering in 3D (arrays ordered as [z, y, x])
                # Electric fields live on edges
                self.Ex = self.backend.zeros((self.nz,     self.ny,     self.nx-1), dtype=np.complex128)
                self.Ey = self.backend.zeros((self.nz,     self.ny-1,   self.nx    ), dtype=np.complex128)
                self.Ez = self.backend.zeros((self.nz-1,   self.ny,     self.nx    ), dtype=np.complex128)
                # Magnetic fields live on faces (Yee grid)
                # Hx: (nz-1, ny-1, nx), Hy: (nz-1, ny, nx-1), Hz: (nz, ny-1, nx-1)
                self.Hx = self.backend.zeros((self.nz-1,   self.ny-1,   self.nx    ), dtype=np.complex128)
                self.Hy = self.backend.zeros((self.nz-1,   self.ny,     self.nx-1  ), dtype=np.complex128)
                self.Hz = self.backend.zeros((self.nz,     self.ny-1,   self.nx-1  ), dtype=np.complex128)
            except TypeError:
                # Fallback if dtype not supported
                self.Ex = self.backend.zeros((self.nz,     self.ny,     self.nx-1))
                self.Ey = self.backend.zeros((self.nz,     self.ny-1,   self.nx    ))
                self.Ez = self.backend.zeros((self.nz-1,   self.ny,     self.nx    ))
                self.Hx = self.backend.zeros((self.nz-1,   self.ny-1,   self.nx    ))
                self.Hy = self.backend.zeros((self.nz-1,   self.ny,     self.nx-1  ))
                self.Hz = self.backend.zeros((self.nz,     self.ny-1,   self.nx-1  ))
                self.is_complex_backend = False
            else: self.is_complex_backend = True
        else: 
            try:
                # Try with dtype parameter if supported - use (ny, nx) format to match backend
                self.Ez = self.backend.zeros((self.ny, self.nx), dtype=np.complex128)
            except TypeError:
                # Fallback if dtype not supported - create real array and handle complex values in code
                self.Ez = self.backend.zeros((self.ny, self.nx))
                self.is_complex_backend = False
            else: self.is_complex_backend = True
            # Staggered magnetic fields: Hx has same rows as Ez and one fewer column; Hy has one fewer row and same columns
            self.Hx = self.backend.zeros((self.ny, self.nx-1))
            self.Hy = self.backend.zeros((self.ny-1, self.nx))

    def simulate_step(self):
        """Perform one FDTD step with stability checks."""
        if self.is_3d: self._update_3d_fields()
        else:
            # 2D TE-polarized Maxwell equations update
            self.Hx, self.Hy = self.backend.update_h_fields(
                self.Hx, self.Hy, self.Ez, self.sigma, 
                self.dx, self.dy, self.dt, MU_0, EPS_0)
            self.Ez = self.backend.update_e_field(
                self.Ez, self.Hx, self.Hy, self.sigma, self.epsilon_r,
                self.dx, self.dy, self.dt, EPS_0)

    def _update_3d_fields(self):
        """Full 3D Yee update for all 6 field components with PML via sigma arrays."""
        dt = self.dt; dx = self.dx; dy = self.dy; dz = self.dz
        mu0 = MU_0; eps0 = EPS_0

        # Convenience references
        Ex = self.Ex; Ey = self.Ey; Ez = self.Ez
        Hx = self.Hx; Hy = self.Hy; Hz = self.Hz
        eps_r = self.epsilon_r; sigma = self.sigma

        # --- Update H fields (n -> n+1/2) ---
        # Magnetic conductivities at staggered positions
        sigma_m_hx = (sigma[:-1, :-1, :] * mu0 / eps0) if sigma.ndim == 3 else sigma * 0
        sigma_m_hy = (sigma[:-1, :, :-1] * mu0 / eps0) if sigma.ndim == 3 else sigma * 0
        sigma_m_hz = (sigma[:, :-1, :-1] * mu0 / eps0) if sigma.ndim == 3 else sigma * 0

        # Curls of E at H locations
        # Hx shape: (nz-1, ny-1, nx)
        dEz_dy = (Ez[:, 1:, :] - Ez[:, :-1, :]) / dy          # (nz-1, ny-1, nx)
        dEy_dz = (Ey[1:, :, :] - Ey[:-1, :, :]) / dz          # (nz-1, ny-1, nx)
        curlE_x = dEz_dy - dEy_dz

        # Hy shape: (nz-1, ny, nx-1)
        dEx_dz = (Ex[1:, :, :] - Ex[:-1, :, :]) / dz          # (nz-1, ny, nx-1)
        dEz_dx = (Ez[:, :, 1:] - Ez[:, :, :-1]) / dx          # (nz-1, ny, nx-1)
        curlE_y = dEx_dz - dEz_dx

        # Hz shape: (nz, ny-1, nx-1)
        dEy_dx = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dx          # (nz, ny-1, nx-1)
        dEx_dy = (Ex[:, 1:, :] - Ex[:, :-1, :]) / dy          # (nz, ny-1, nx-1)
        curlE_z = dEy_dx - dEx_dy

        # Semi-implicit PML factors for H
        denom_hx = 1.0 + sigma_m_hx * dt / (2.0 * mu0)
        factor_hx = (1.0 - sigma_m_hx * dt / (2.0 * mu0)) / denom_hx
        source_hx = (dt / mu0) / denom_hx

        denom_hy = 1.0 + sigma_m_hy * dt / (2.0 * mu0)
        factor_hy = (1.0 - sigma_m_hy * dt / (2.0 * mu0)) / denom_hy
        source_hy = (dt / mu0) / denom_hy

        denom_hz = 1.0 + sigma_m_hz * dt / (2.0 * mu0)
        factor_hz = (1.0 - sigma_m_hz * dt / (2.0 * mu0)) / denom_hz
        source_hz = (dt / mu0) / denom_hz

        Hx[:] = factor_hx * Hx - source_hx * curlE_x
        Hy[:] = factor_hy * Hy - source_hy * curlE_y
        Hz[:] = factor_hz * Hz - source_hz * curlE_z

        # --- Update E fields (n+1/2 -> n+1) ---
        # Electric conductivities and permittivities at E locations (use interior slices)
        # Ex interior: [1:-1, 1:-1, :]
        if eps_r.ndim == 3:
            eps_ex = eps_r[1:-1, 1:-1, :-1]
            sig_ex = sigma[1:-1, 1:-1, :-1]
        else:
            eps_ex = eps_r
            sig_ex = sigma

        # Ey interior: [1:-1, :, 1:-1]
        if eps_r.ndim == 3:
            eps_ey = eps_r[1:-1, :-1, 1:-1]
            sig_ey = sigma[1:-1, :-1, 1:-1]
        else:
            eps_ey = eps_r
            sig_ey = sigma

        # Ez interior: [:, 1:-1, 1:-1]
        if eps_r.ndim == 3:
            eps_ez = eps_r[:-1, 1:-1, 1:-1]
            sig_ez = sigma[:-1, 1:-1, 1:-1]
        else:
            eps_ez = eps_r
            sig_ez = sigma

        # Curls of H at E locations (interior updates)
        # Ex interior indices
        dHz_dy_ex = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy              # (nz, ny-2, nx-1)
        dHy_dz_ex = (Hy[1:, :, :] - Hy[:-1, :, :]) / dz              # (nz-2, ny, nx-1)
        # Align to (nz-2, ny-2, nx-1)
        curlH_x = dHz_dy_ex[1:-1, :, :] - dHy_dz_ex[:, 1:-1, :]

        # Ey interior indices
        dHx_dz_ey = (Hx[1:, :, :] - Hx[:-1, :, :]) / dz              # (nz-2, ny-1, nx)
        dHz_dx_ey = (Hz[:, :, 1:] - Hz[:, :, :-1]) / dx              # (nz, ny-1, nx-1)
        # Align to (nz-2, ny-1, nx-2)
        curlH_y = dHx_dz_ey[:, :, 1:-1] - dHz_dx_ey[1:-1, :, :]

        # Ez interior indices
        dHy_dx_ez = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dx              # (nz-1, ny, nx-2)
        dHx_dy_ez = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy              # (nz-1, ny-2, nx)
        # Align to (nz-1, ny-2, nx-2)
        curlH_z = dHy_dx_ez[:, 1:-1, :] - dHx_dy_ez[:, :, 1:-1]

        # Semi-implicit PML factors for E
        denom_ex = 1.0 + sig_ex * dt / (2.0 * eps0 * eps_ex)
        factor_ex = (1.0 - sig_ex * dt / (2.0 * eps0 * eps_ex)) / denom_ex
        source_ex = (dt / (eps0 * eps_ex)) / denom_ex

        denom_ey = 1.0 + sig_ey * dt / (2.0 * eps0 * eps_ey)
        factor_ey = (1.0 - sig_ey * dt / (2.0 * eps0 * eps_ey)) / denom_ey
        source_ey = (dt / (eps0 * eps_ey)) / denom_ey

        denom_ez = 1.0 + sig_ez * dt / (2.0 * eps0 * eps_ez)
        factor_ez = (1.0 - sig_ez * dt / (2.0 * eps0 * eps_ez)) / denom_ez
        source_ez = (dt / (eps0 * eps_ez)) / denom_ez

        # Apply updates to interior regions
        Ex[1:-1, 1:-1, :] = factor_ex * Ex[1:-1, 1:-1, :] + source_ex * (curlH_x)
        Ey[1:-1, :, 1:-1] = factor_ey * Ey[1:-1, :, 1:-1] + source_ey * (curlH_y)
        Ez[:, 1:-1, 1:-1] = factor_ez * Ez[:, 1:-1, 1:-1] + source_ez * (curlH_z)

    def initialize_simulation(self, save=True, live=True, axis_scale=None, save_animation=False,
                             animation_filename='fdtd_animation.mp4', clean_visualization=True,
                             save_fields=None, decimate_save=1, accumulate_power=False,
                             save_memory_mode=False, fields_to_cache: Optional[Sequence[str]] = None):
        """Initialize the simulation before running steps.

        Args:
            save: Whether to save field data at each step.
            live: Whether to show live animation of the simulation.
            axis_scale: Color scale limits for the field visualization.
            save_animation: Whether to save an animation of the simulation as an mp4 file.
            animation_filename: Filename for the saved animation (must end in .mp4).
            save_fields: List of fields to save (None = auto-select based on dimensionality)
            decimate_save: Save only every nth time step (1 = save all, 10 = save every 10th step)
            accumulate_power: Instead of saving all fields, accumulate power and save that
            save_memory_mode: If True, avoid storing all field data and only keep monitors/power
            fields_to_cache: Fields that should always be cached (complex values preserved)
        """
        # Set default save_fields based on dimensionality
        if save_fields is None:
            if self.is_3d:
                save_fields = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
            else:
                save_fields = ['Ez', 'Hx', 'Hy']
        self._cache_fields = list(fields_to_cache) if fields_to_cache else []
        self._cache_frequency = 1 if self._cache_fields else None


        # Record start time
        self.start_time = datetime.datetime.now()
        # Initialize simulation state
        self.t = 0
        self.current_step = 0
        self._total_steps = self.num_steps
        self._save_results = save
        self._save_fields = save_fields
        self._decimate_save = decimate_save
        self._live = live
        # Determine default axis scale: power if accumulating, else field amplitude
        if axis_scale is None:
            if accumulate_power:
                power_scale = 0.0  # in W/m^2
                for src in self.design.sources:
                    if isinstance(src, ModeSource):
                        max_pd = float(getattr(src, "max_power_density", 0.0) or 0.0)
                        sig = float(getattr(src, "max_signal_magnitude", 1.0) or 1.0)
                        candidate = max_pd * (sig ** 2)
                        if np.isfinite(candidate): power_scale = max(power_scale, candidate)
                if power_scale <= 0.0: power_scale = 1.0
                # Store SI W/m^2 range; viz converts to W/µm²
                self._axis_scale = [0.0, power_scale]
                self._live_quantity = "power"
            else:
                scale = 0.0
                for src in self.design.sources:
                    if isinstance(src, ModeSource):
                        field_amp = getattr(src, "max_field_amplitude", None)
                        signal_amp = getattr(src, "max_signal_magnitude", None)
                        if field_amp is not None and signal_amp is not None:
                            try:
                                value = float(field_amp) * float(signal_amp)
                            except (TypeError, ValueError):
                                continue
                            if np.isfinite(value):
                                scale = max(scale, abs(value))
                if scale <= 0.0: scale = 1.0
                self._axis_scale = [-scale, scale]
                self._live_quantity = "field"
        else:
            self._axis_scale = axis_scale
            self._live_quantity = "field"

        # Reset stored results for a new run
        for key in list(self.results.keys()):
            self.results[key] = []

        # Save mode flags as class attributes for monitor access
        self.save_memory_mode = save_memory_mode
        self.accumulate_power = accumulate_power

        # Display simulation header and parameters
        sim_params = {
            "Domain size": f"{self.design.width:.2e} x {self.design.height:.2e} m",
            "Resolution": f"{self.resolution:.2e} m",
            "Time steps": self.num_steps,
            "Time delta": f"{self.dt:.2e} s",
            "Total time": f"{self.time[-1]:.2e} s",
            "Backend": self.backend.__class__.__name__,
            "Save fields": ", ".join(save_fields),
            "Memory-saving mode": "Enabled" if save_memory_mode else "Disabled",
            "Accumulate power": "Enabled" if accumulate_power else "Disabled",
            "Live animation": "Enabled" if live else "Disabled"
        }
        display_parameters(sim_params, "Simulation Parameters")
        
        # Check stability using the helper function
        from beamz.helpers import check_fdtd_stability
        n_max = np.sqrt(np.max(self.epsilon_r))
        is_stable, courant, safe_limit = check_fdtd_stability(dt=self.dt, dx=self.dx, dy=self.dy, n_max=n_max, safety_factor=1.0)
        if not is_stable:
            display_status(f"Simulation may be unstable! Courant number = {courant:.3f} > {safe_limit:.3f}", "warning")
            display_status("Consider reducing dt or increasing dx/dy", "warning")
        else: 
            display_status(f"Stability check passed (Courant number = {courant:.3f} / {safe_limit:.3f})", "success")
        
        # Set up power accumulation if requested
        if accumulate_power:
            self.power_accumulated = np.zeros((self.ny, self.nx))
            self.power_accumulation_count = 0
            
        # Determine optimal save frequency based on backend type
        is_gpu_backend = hasattr(self.backend, 'device') and getattr(self.backend, 'device', '') in ['cuda', 'mps', 'gpu']
        # For GPU backends, avoid excessive CPU-GPU transfers by batching the result saves
        if is_gpu_backend and save:
            save_freq = max(10, self.num_steps // 100)  # Save approximately every 1% of steps or min of 10 steps
            display_status(f"GPU backend detected: Optimizing result storage (saving every {save_freq} steps)", "info")
        else: 
            save_freq = 1  # Save every step for CPU backends
            
        # Apply additional decimation based on user setting
        self._effective_save_freq = save_freq * decimate_save
        if self._cache_frequency is None:
            self._cache_frequency = self._effective_save_freq
        
        # If in save_memory_mode, clear any existing results to start fresh
        if save_memory_mode and not self._cache_fields:
            for field in self.results:
                if field != 't':
                    self.results[field] = []
            display_status("Memory-saving mode active: Only storing monitor data and/or power accumulation", "info")

    def step(self) -> bool:
        """Perform one simulation step. Returns True if simulation should continue, False if complete."""
        if self.current_step >= self.num_steps: return False
        # Update fields
        self.simulate_step()
        # Apply sources
        sim_helper.apply_sources(self)
        # Record monitor data
        sim_helper.record_monitor_data(self, self.current_step)
        # Accumulate power if requested
        sim_helper.accumulate_power(self)
        # Save results if requested and at the right frequency
        sim_helper.save_step_results(self)
        # Show live animation if requested
        self._update_live_animation()
        # Update time & step counter
        self.t += self.dt
        self.current_step += 1
        return True

    def finalize_simulation(self):
        """Clean up and finalize the simulation."""
        # Clean up animation
        if self._live and self.fig is not None: viz.close_fdtd_figure(self)
        # Display completion information
        display_status("Simulation complete!", "success")
        display_time_elapsed(self.start_time)
        # Calculate final power average if accumulating
        if self.accumulate_power and self.power_accumulation_count > 0: self.power_accumulated /= self.power_accumulation_count
        # Display memory usage estimate
        memory_usage = self.estimate_memory_usage(time_steps=self.num_steps, save_fields=self._save_fields)
        display_status(f"Estimated memory usage: {memory_usage['Full simulation']['Total memory (MB)']:.2f} MB", "info")
        objective_results: Dict[str, float] = {}
        for idx, monitor in enumerate(self.monitors):
            if hasattr(monitor, 'evaluate_objective'):
                value = monitor.evaluate_objective()
            else:
                value = None
            if value is None:
                continue
            key = getattr(monitor, 'name', None) or f"monitor_{idx}"
            objective_results[key] = value
        if objective_results:
            self.results['objectives'] = objective_results
        else:
            self.results.pop('objectives', None)
        self.last_objectives = objective_results
        return self.results

    def run(self, steps: Optional[int] = None, save=True, live=True, axis_scale=None, save_animation=False,
            animation_filename='fdtd_animation.mp4', clean_visualization=True,
            save_fields=None, decimate_save=1, accumulate_power=False,
            save_memory_mode=False, fields_to_cache: Optional[Sequence[str]] = None) -> Dict:
        """Run the complete simulation using the new step-by-step approach.

        Args:
            steps: Number of steps to run. If None, run until the end of the time array.
            save: Whether to save field data at each step.
            live: Whether to show live animation of the simulation.
            axis_scale: Color scale limits for the field visualization.
            save_animation: Whether to save an animation of the simulation as an mp4 file.
            animation_filename: Filename for the saved animation (must end in .mp4).
            save_fields: List of fields to save (None = auto-select based on dimensionality)
            decimate_save: Save only every nth time step (1 = save all, 10 = save every 10th step)
            accumulate_power: Instead of saving all fields, accumulate power and save that
            save_memory_mode: If True, avoid storing all field data and only keep monitors/power
            fields_to_cache: Fields that should always be cached even when memory saving

        Returns:
            Dictionary containing the simulation results.
        """
        # Initialize the simulation
        self.initialize_simulation(save=save, live=live, axis_scale=axis_scale,
                                  save_animation=save_animation,
                                  animation_filename=animation_filename,
                                  clean_visualization=clean_visualization,
                                  save_fields=save_fields, decimate_save=decimate_save,
                                  accumulate_power=accumulate_power,
                                  save_memory_mode=save_memory_mode,
                                  fields_to_cache=fields_to_cache)
        
        # Run the simulation with progress tracking
        with create_rich_progress() as progress:
            task = progress.add_task("Running simulation...", total=self.num_steps)
            while self.step(): progress.update(task, advance=1)
        return self.finalize_simulation()

    def _apply_sources(self):
        """Apply all sources for the current time step (delegated to helper)."""
        from beamz.simulation import helper as sim_helper  # local import to avoid cycles
        return sim_helper.apply_sources(self)

    def _accumulate_power(self):
        """Accumulate power (delegated to helper)."""
        from beamz.simulation import helper as sim_helper  # local import to avoid cycles
        return sim_helper.accumulate_power(self)

    def _save_step_results(self):
        """Save step results (delegated to helper)."""
        from beamz.simulation import helper as sim_helper  # local import to avoid cycles
        return sim_helper.save_step_results(self)

    def plot_field(self, field: str = "Ez", t: float = None, z_slice: int = None) -> None:
        """Delegate to viz.plot_fdtd_field."""
        return viz.plot_fdtd_field(self, field=field, t=t, z_slice=z_slice)

    def animate_live(self, field_data=None, field="Ez", axis_scale=[-1,1], z_slice=None):
        """Delegate to viz.animate_fdtd_live."""
        return viz.animate_fdtd_live(self, field_data=field_data, field=field, axis_scale=axis_scale, z_slice=z_slice)

    def _update_live_animation(self):
        """Update live animation if requested."""
        if self._live and (self.current_step % 2 == 0 or self.current_step == self.num_steps - 1):
            field = "Ez"
            data = getattr(self, field)
            Ez_np = self.backend.to_numpy(data)
            # Enforce fixed axis scale if provided via sim.run(axis_scale=...)
            axis_scale = getattr(self, "_axis_scale", None)
            viz.animate_fdtd_live(self, field_data=Ez_np, field=field, axis_scale=axis_scale)

    def _record_monitor_data(self, step):
        """Record field data at monitor locations (delegated to helper)."""
        from beamz.simulation import helper as sim_helper  # local import to avoid cycles
        return sim_helper.record_monitor_data(self, step)

    def save_animation(self, field: str = "Ez", axis_scale=[-1, 1], filename='fdtd_animation.mp4', 
                       fps=60, frame_skip=4, clean_visualization=False):
        """Delegate to viz.save_fdtd_animation."""
        return viz.save_fdtd_animation(self, field=field, axis_scale=axis_scale, filename=filename, fps=fps, 
                                        frame_skip=frame_skip, clean_visualization=clean_visualization)
        
    def plot_power(self, cmap: str = "hot", vmin: float = None, vmax: float = None, db_colorbar: bool = False):
        """Delegate to viz.plot_fdtd_power."""
        return viz.plot_fdtd_power(self, cmap=cmap, vmin=vmin, vmax=vmax, db_colorbar=db_colorbar)

    def estimate_memory_usage(self, time_steps=None, save_fields=None):
        """Delegate to helper.estimate_memory_usage and display result."""
        from beamz.simulation import helper as sim_helper  # local import to avoid cycles
        result = sim_helper.estimate_memory_usage(self, time_steps=time_steps, save_fields=save_fields)
        display_status(f"Estimated memory usage: {result['Full simulation']['Total memory (MB)']:.2f} MB", "info")
        return result