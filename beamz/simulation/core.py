import numpy as np
from beamz.const import *
from beamz.design.core import Design
from beamz.devices.core import Device
from beamz.simulation.fields import Fields
from beamz.simulation.boundaries import Boundary, PML
from beamz.visual.viz import animate_manual_field, close_fdtd_figure

class Simulation:
    """FDTD simulation class supporting both 2D and 3D electromagnetic simulations."""
    def __init__(self, design:Design=None, devices:list[Device]=[], boundaries:list[Boundary]=[], resolution:float=0.02*µm, time:np.ndarray=None):
        """Initialize FDTD simulation with design object and extract material properties at specified resolution."""
        self.design = design
        self.resolution = resolution
        self.is_3d = design.is_3d and design.depth > 0
        
        # Get material grids from design (design owns the material grids, we reference them)
        permittivity, conductivity, permeability = design.get_material_grids(resolution)
        
        # Initialize time stepping first
        if time is None or len(time) < 2: raise ValueError("FDTD requires a time array with at least two entries")
        self.time, self.dt, self.num_steps = time, float(time[1] - time[0]), len(time)
        self.t, self.current_step = 0, 0
        
        # Create field storage (fields owns the E/H field arrays, references material grids)
        self.fields = Fields(permittivity, conductivity, permeability, resolution)
        
        # Initialize PML regions if present
        pml_boundaries = [b for b in boundaries if isinstance(b, PML)]
        if pml_boundaries:
            # Create PML regions (do this once, not every timestep)
            pml_data = {}
            for pml in pml_boundaries:
                pml_data.update(pml.create_pml_regions(self.fields, design, resolution, self.dt))
            self.pml_data = pml_data
            
            # Initialize split fields in Fields object
            self.fields._init_upml_fields(pml_data)
        else:
            self.pml_data = None
        
        # Store device references (no duplication)
        self.devices = devices
        
        # Store boundary references (no duplication)
        self.boundaries = boundaries

    def step(self):
        """Perform one FDTD time step."""
        if self.current_step >= self.num_steps: return False
        
        # Apply sources
        self._apply_sources()
        
        # Update fields (UPML is handled inside fields.update())
        self.fields.update(self.dt)
        
        # Update time and step counter
        self.t += self.dt
        self.current_step += 1
        return True
    
    def _apply_sources(self):
        """Apply all sources to the fields at the current time step."""
        for device in self.devices:
            if hasattr(device, 'signal') and hasattr(device, 'position'):
                # Get signal amplitude at current time step
                if hasattr(device.signal, '__len__') and len(device.signal) > self.current_step:
                    amplitude = device.signal[self.current_step]
                elif hasattr(device.signal, '__call__'):
                    amplitude = device.signal(self.t)
                else:
                    amplitude = 1.0  # Default amplitude
                
                # Apply to field based on source type (soft source)
                if hasattr(device, 'width'):
                    # Gaussian source - apply spatially distributed source
                    self._apply_gaussian_source(device, amplitude)
                else:
                    # Point source fallback
                    self._apply_point_source(device, amplitude)
    
    def _apply_gaussian_source(self, source, amplitude):
        """Apply a Gaussian-distributed source to the Ez field."""
        pos_x, pos_y = source.position[0], source.position[1]
        width = source.width if hasattr(source, 'width') else self.resolution * 3
        
        # Convert position to grid indices
        grid_shape = self.fields.Ez.shape
        nx, ny = grid_shape[1], grid_shape[0]
        
        # Create coordinate grids
        x = np.linspace(0, self.design.width, nx)
        y = np.linspace(0, self.design.height, ny)
        X, Y = np.meshgrid(x, y)
        
        # Gaussian distribution
        gaussian = amplitude * np.exp(-((X - pos_x)**2 + (Y - pos_y)**2) / (2 * width**2))
        
        # Soft source: add to existing field
        self.fields.Ez += gaussian
    
    def _apply_point_source(self, source, amplitude):
        """Apply a point source to the Ez field."""
        pos_x, pos_y = source.position[0], source.position[1]
        
        # Convert to grid index
        grid_shape = self.fields.Ez.shape
        i = int(pos_y / self.resolution)
        j = int(pos_x / self.resolution)
        
        # Bounds check
        if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1]:
            self.fields.Ez[i, j] += amplitude
    

    def run(self, animate_live=None, animation_interval=10):
        """Run complete FDTD simulation with optional live field visualization.
        
        Args:
            animate_live: Field component to animate ('Ez', 'Hx', 'Hy', 'Ex', 'Ey', etc.) or None to disable
            animation_interval: Update visualization every N steps (higher = faster but less smooth)
        """
        # Handle 3D simulations - require monitor for now (not implemented yet)
        if animate_live and self.is_3d:
            print("Live animation for 3D simulations requires a monitor (not yet implemented)")
            animate_live = None
        
        # Initialize animation context if requested
        viz_context = None
        if animate_live:
            # Validate field component exists
            if not hasattr(self.fields, animate_live):
                print(f"Warning: Field '{animate_live}' not found. Available: Ez, Hx, Hy (2D) or Ex,Ey,Ez,Hx,Hy,Hz (3D)")
                animate_live = None
        
        try:
            # Main simulation loop
            while self.step():
                # Update live animation if enabled
                if animate_live and self.current_step % animation_interval == 0:
                    field_data = getattr(self.fields, animate_live)
                    # Convert to V/µm for display
                    field_display = field_data * 1e-6 if 'E' in animate_live else field_data
                    extent = (0, self.design.width, 0, self.design.height)
                    title = f'{animate_live} at t = {self.t:.2e} s (step {self.current_step}/{self.num_steps})'
                    viz_context = animate_manual_field(field_display, context=viz_context, extent=extent, 
                                                      title=title, units='V/µm' if 'E' in animate_live else 'A/m',
                                                      design=self.design, pause=0.001)
        finally:
            # Cleanup: keep the final frame visible
            if viz_context and viz_context.get('fig'):
                import matplotlib.pyplot as plt
                plt.show(block=False)
                print("Simulation complete. Close the plot window to continue.")