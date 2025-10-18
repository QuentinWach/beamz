import numpy as np
from beamz.const import *
from beamz.design.core import Design
from beamz.devices.core import Device
from beamz.simulation.fields import Fields

class Simulation:
    """FDTD simulation class supporting both 2D and 3D electromagnetic simulations."""
    def __init__(self, design:Design=None, devices:list[Device]=[], resolution:float=0.02*µm, time:np.ndarray=None):
        """Initialize FDTD simulation with design object and extract material properties at specified resolution."""
        self.design = design
        self.resolution = resolution
        self.is_3d = design.is_3d and design.depth > 0
        
        # Get material grids from design (design owns the material grids, we reference them)
        permittivity, conductivity, permeability = design.get_material_grids(resolution)
        
        # Create field storage (fields owns the E/H field arrays, references material grids)
        self.fields = Fields(permittivity, conductivity, permeability, resolution)
        
        # Store device references (no duplication)
        self.devices = devices
        
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