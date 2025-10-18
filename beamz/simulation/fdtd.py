import numpy as np
from beamz.const import *
from beamz.design.core import Design
from beamz.devices.core import Device
from beamz.simulation.fields import Fields

class FDTD:
    """FDTD simulation class supporting both 2D and 3D electromagnetic simulations."""
    def __init__(self, design:Design=None, devices:list[Device]=[], resolution:float=0.02*µm, time:np.ndarray=None):
        """Initialize FDTD simulation with design object and extract material properties at specified resolution."""
        self.design = design
        self.resolution = resolution
        self.is_3d = design.is_3d and design.depth > 0
        
        # Set grid spacing from resolution
        self.dx = self.dy = resolution
        self.dz = resolution if self.is_3d else None
        
        # Get material properties from design (design handles internal meshing at resolution)
        self.epsilon_r = design.get_permittivity(resolution)
        self.sigma = design.get_conductivity(resolution)
        
        # Set grid dimensions from material arrays
        if self.is_3d: self.nz, self.ny, self.nx = self.epsilon_r.shape
        else: self.ny, self.nx = self.epsilon_r.shape
        
        # Create field storage and initialize fields
        grid_shape = (self.nz, self.ny, self.nx) if self.is_3d else (self.ny, self.nx)
        self.fields = Fields(epsilon_r=self.epsilon_r, sigma=self.sigma, grid_shape=grid_shape, 
                                resolution=self.resolution, is_3d=self.is_3d)
        
        # Initialize time stepping
        if time is None or len(time) < 2: raise ValueError("FDTD requires a time array with at least two entries")
        self.time, self.dt, self.num_steps = time, float(time[1] - time[0]), len(time)
        self.t, self.current_step = 0, 0

    def step(self):
        """Perform one FDTD time step by updating electromagnetic fields via Maxwell's equations."""
        if self.current_step >= self.num_steps: return False # Check if we've reached the end of the time array
        self.fields.update(self.dt) # Update fields via Maxwell's equations
        self.t += self.dt # Update time
        self.current_step += 1 # Update step counter
        return True

    def run(self):
        """Run complete FDTD simulation stepping through all time steps."""
        while self.step(): pass
