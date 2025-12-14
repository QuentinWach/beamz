import numpy as np
from beamz.const import µm, EPS_0, MU_0

class Boundary:
    """Abstract base class for all boundary conditions."""
    def __init__(self, edges, thickness):
        """
        Args:
            edges: list of edge names or 'all'
                   2D: ['left', 'right', 'top', 'bottom']
                   3D: ['left', 'right', 'top', 'bottom', 'front', 'back']
            thickness: physical thickness of boundary region
        """
        if edges == 'all':
            # Will be determined based on dimensionality in apply()
            self.edges = 'all'
        else:
            self.edges = edges if isinstance(edges, list) else [edges]
        self.thickness = thickness
    
    def apply(self, fields, design, resolution, dt):
        """Apply boundary condition to fields. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def _get_edges_for_dimensionality(self, is_3d):
        """Resolve 'all' edges based on dimensionality."""
        if self.edges == 'all':
            return ['left', 'right', 'top', 'bottom', 'front', 'back'] if is_3d else ['left', 'right', 'top', 'bottom']
        return self.edges

class PML(Boundary):
    """Perfectly Matched Layer boundary condition for FDTD simulations."""
    
    def __init__(self, edges='all', thickness=1*µm, sigma_max=None, m=3, kappa_max=1, alpha_max=0):
        """
        Initialize UPML with stretched-coordinate parameters.
        
        Args:
            edges: edges to apply PML
            thickness: PML thickness
            sigma_max: maximum conductivity (auto-calculated if None)
            m: conductivity grading order
            kappa_max: maximum real coordinate stretching
            alpha_max: maximum CFS alpha parameter (for better absorption at low frequencies)
        """
        super().__init__(edges, thickness)
        self.sigma_max = sigma_max
        self.m = m
        self.kappa_max = kappa_max
        self.alpha_max = alpha_max
    
    def apply(self, fields, design, resolution, dt):
        """Apply PML by modifying field update equations with PML conductivity."""
        # This method is now deprecated - use modify_conductivity instead
        pass
    
    def create_pml_regions(self, fields, design, resolution, dt):
        """Create permanent PML region masks and stretched-coordinate parameters.
        
        Returns dict with:
            - mask: boolean arrays indicating PML cells
            - sigma_x, sigma_y, sigma_z: conductivity profiles
            - kappa_x, kappa_y, kappa_z: real stretching factors
            - alpha_x, alpha_y, alpha_z: CFS alpha parameters
        """
        # Calculate optimal sigma_max if not provided
        if self.sigma_max is None:
            eta = np.sqrt(MU_0 / (EPS_0 * 1.0))
            self.sigma_max = 0.8 * (self.m + 1) / (eta * resolution)
        
        # Create graded profiles for each direction
        pml_data = self._create_upml_profiles_2d(fields, design, resolution, dt)
        return pml_data
    
    def _create_upml_profiles_2d(self, fields, design, resolution, dt):
        """Create UPML stretched-coordinate profiles for 2D."""
        ez_shape = fields.Ez.shape
        ny, nx = ez_shape
        
        # Initialize profile arrays
        sigma_x = np.zeros(ez_shape)
        sigma_y = np.zeros(ez_shape)
        kappa_x = np.ones(ez_shape)
        kappa_y = np.ones(ez_shape)
        alpha_x = np.zeros(ez_shape)
        alpha_y = np.zeros(ez_shape)
        
        # Create coordinate arrays
        x_coords = np.linspace(0, design.width, nx)
        y_coords = np.linspace(0, design.height, ny)
        
        edges = self._get_edges_for_dimensionality(False)
        
        # Apply graded profiles for each edge
        for edge in edges:
            if edge == 'left':
                for j in range(nx):
                    if x_coords[j] < self.thickness:
                        dist = self.thickness - x_coords[j]
                        sigma_x[:, j] = self._sigma_profile(dist, self.thickness)
                        kappa_x[:, j] = self._kappa_profile(dist, self.thickness)
                        alpha_x[:, j] = self._alpha_profile(dist, self.thickness)
            
            elif edge == 'right':
                for j in range(nx):
                    if x_coords[j] > (design.width - self.thickness):
                        dist = x_coords[j] - (design.width - self.thickness)
                        sigma_x[:, j] = self._sigma_profile(dist, self.thickness)
                        kappa_x[:, j] = self._kappa_profile(dist, self.thickness)
                        alpha_x[:, j] = self._alpha_profile(dist, self.thickness)
            
            elif edge == 'bottom':
                for i in range(ny):
                    if y_coords[i] < self.thickness:
                        dist = self.thickness - y_coords[i]
                        sigma_y[i, :] = self._sigma_profile(dist, self.thickness)
                        kappa_y[i, :] = self._kappa_profile(dist, self.thickness)
                        alpha_y[i, :] = self._alpha_profile(dist, self.thickness)
            
            elif edge == 'top':
                for i in range(ny):
                    if y_coords[i] > (design.height - self.thickness):
                        dist = y_coords[i] - (design.height - self.thickness)
                        sigma_y[i, :] = self._sigma_profile(dist, self.thickness)
                        kappa_y[i, :] = self._kappa_profile(dist, self.thickness)
                        alpha_y[i, :] = self._alpha_profile(dist, self.thickness)
        
        # Create PML mask (True where PML is active)
        pml_mask = (sigma_x > 0) | (sigma_y > 0)
        
        return {
            'mask': pml_mask,
            'sigma_x': sigma_x, 'sigma_y': sigma_y,
            'kappa_x': kappa_x, 'kappa_y': kappa_y,
            'alpha_x': alpha_x, 'alpha_y': alpha_y
        }
    
    def _modify_conductivity_3d(self, fields, design, resolution, dt, edges):
        """Modify conductivity arrays to include PML absorption in 3D."""
        # TODO: Implement 3D PML conductivity modification
        # For now, just pass to avoid breaking 3D simulations
        pass
    
    def _sigma_profile(self, dist, thickness):
        """Graded conductivity profile."""
        return self.sigma_max * (dist / thickness) ** self.m
    
    def _kappa_profile(self, dist, thickness):
        """Real coordinate stretching profile."""
        return 1 + (self.kappa_max - 1) * (dist / thickness) ** self.m
    
    def _alpha_profile(self, dist, thickness):
        """CFS alpha profile for low-frequency absorption."""
        return self.alpha_max * ((thickness - dist) / thickness) ** self.m

class ABC(Boundary):
    """Absorbing Boundary Condition (Mur, Liao, etc.) - placeholder."""
    def apply(self, fields, design, resolution, dt):
        pass  # TODO: implement

class PeriodicBoundary(Boundary):
    """Periodic boundary condition - placeholder."""
    def apply(self, fields, design, resolution, dt):
        pass  # TODO: implement