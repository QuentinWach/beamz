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
    
    def create_pml_regions(self, fields, design, resolution, dt, plane_2d='xy'):
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
        
        # Create graded profiles for each direction based on plane
        pml_data = self._create_pml_profiles_2d(fields, design, resolution, dt, plane_2d)
        return pml_data
    
    def _create_pml_profiles_2d(self, fields, design, resolution, dt, plane_2d):
        """Create UPML stretched-coordinate profiles for 2D plane."""
        # Grid shape from material grid (same as field shape for collocated/main grid)
        # Assuming design.width/height/depth match the grid dimensions
        
        if plane_2d == 'xy':
            shape = fields.permittivity.shape # (ny, nx)
            dim1, dim2 = shape
            len1, len2 = design.height, design.width
            labels = ['y', 'x']
        elif plane_2d == 'yz':
            shape = fields.permittivity.shape # (nz, ny)
            dim1, dim2 = shape
            len1, len2 = (design.depth if design.depth else 0), design.height # Assuming depth is defined for yz slice?
            # If 2D sim in yz, design.depth might be relevant or height/width mapping changes.
            # Assuming standard: design.width (x), design.height (y), design.depth (z)
            labels = ['z', 'y']
        elif plane_2d == 'xz':
            shape = fields.permittivity.shape # (nz, nx)
            dim1, dim2 = shape
            len1, len2 = (design.depth if design.depth else 0), design.width
            labels = ['z', 'x']
            
        # Initialize profile arrays
        profiles = {}
        for axis in ['x', 'y', 'z']:
            profiles[f'sigma_{axis}'] = np.zeros(shape)
            profiles[f'kappa_{axis}'] = np.ones(shape)
            profiles[f'alpha_{axis}'] = np.zeros(shape)
            
        # Create coordinate arrays
        coords1 = np.linspace(0, len1, dim1) # axis 0 (y or z)
        coords2 = np.linspace(0, len2, dim2) # axis 1 (x or y)
        
        edges = self._get_edges_for_dimensionality(False)
        
        # Map edges to axes based on plane
        # xy: Left/Right -> x (axis 1). Bottom/Top -> y (axis 0).
        # yz: Left/Right -> y (axis 1)? Or z? Usually Left/Right is horizontal on screen.
        # yz plane: horizontal=y, vertical=z? Or horizontal=y, vertical=z.
        # Let's assume consistent mapping: 
        # dim2 is "horizontal" (second index), dim1 is "vertical" (first index).
        # xy: x is horizontal (dim2), y is vertical (dim1).
        # yz: y is horizontal (dim2), z is vertical (dim1).
        # xz: x is horizontal (dim2), z is vertical (dim1).
        
        # Edges mapping:
        # Left/Right -> dim2 (coords2)
        # Bottom/Top -> dim1 (coords1)
        
        # Determine which sigma/kappa/alpha component to set
        # xy: dim2->x, dim1->y
        # yz: dim2->y, dim1->z
        # xz: dim2->x, dim1->z
        
        axis1 = labels[0] # vertical axis name
        axis2 = labels[1] # horizontal axis name
        
        for edge in edges:
            if edge == 'left': # Start of horizontal axis (dim2)
                for i in range(dim2):
                    if coords2[i] < self.thickness:
                        dist = self.thickness - coords2[i]
                        profiles[f'sigma_{axis2}'][:, i] = self._sigma_profile(dist, self.thickness)
                        profiles[f'kappa_{axis2}'][:, i] = self._kappa_profile(dist, self.thickness)
                        profiles[f'alpha_{axis2}'][:, i] = self._alpha_profile(dist, self.thickness)
                        
            elif edge == 'right': # End of horizontal axis (dim2)
                for i in range(dim2):
                    if coords2[i] > (len2 - self.thickness):
                        dist = coords2[i] - (len2 - self.thickness)
                        profiles[f'sigma_{axis2}'][:, i] = self._sigma_profile(dist, self.thickness)
                        profiles[f'kappa_{axis2}'][:, i] = self._kappa_profile(dist, self.thickness)
                        profiles[f'alpha_{axis2}'][:, i] = self._alpha_profile(dist, self.thickness)
                        
            elif edge == 'bottom': # Start of vertical axis (dim1)
                for j in range(dim1):
                    if coords1[j] < self.thickness:
                        dist = self.thickness - coords1[j]
                        profiles[f'sigma_{axis1}'][j, :] = self._sigma_profile(dist, self.thickness)
                        profiles[f'kappa_{axis1}'][j, :] = self._kappa_profile(dist, self.thickness)
                        profiles[f'alpha_{axis1}'][j, :] = self._alpha_profile(dist, self.thickness)
                        
            elif edge == 'top': # End of vertical axis (dim1)
                for j in range(dim1):
                    if coords1[j] > (len1 - self.thickness):
                        dist = coords1[j] - (len1 - self.thickness)
                        profiles[f'sigma_{axis1}'][j, :] = self._sigma_profile(dist, self.thickness)
                        profiles[f'kappa_{axis1}'][j, :] = self._kappa_profile(dist, self.thickness)
                        profiles[f'alpha_{axis1}'][j, :] = self._alpha_profile(dist, self.thickness)
        
        # Create PML mask (True where any PML sigma is active)
        pml_mask = (profiles[f'sigma_{axis1}'] > 0) | (profiles[f'sigma_{axis2}'] > 0)
        
        result = {'mask': pml_mask}
        result.update(profiles)
        return result
    
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