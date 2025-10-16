import numpy as np
from beamz.const import EPS_0, MU_0
from beamz import viz as viz

class PML:
    """Unified PML (Perfectly Matched Layer) class for absorbing boundary conditions."""
    def __init__(self, region_type, position, size, orientation, polynomial_order=2.0, sigma_factor=1.0, alpha_max=0.1):
        self.region_type = region_type  # "rect" or "corner"
        self.position = position
        self.orientation = orientation
        self.polynomial_order = polynomial_order  # Reduced to allow smoother transition
        self.sigma_factor = sigma_factor  # Reduced to allow waves to enter
        self.alpha_max = alpha_max  # Reduced frequency-shifting for smoother transition
        if region_type == "rect": self.width, self.height = size
        else: self.radius = size

    def add_to_plot(self, ax, facecolor='none', edgecolor="black", alpha=0.5, linestyle='--'):
        """Delegate PML drawing to beamz.viz.draw_pml."""
        return viz.draw_pml(ax, self, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle)

    def get_profile(self, normalized_distance):
        """Calculate PML absorption profile using gradual grading."""
        d = min(max(normalized_distance, 0.0), 1.0)
        if d < 0.05: sigma = 0.01 * (d/0.05)**2
        else: sigma = ((d - 0.05) / 0.95)**self.polynomial_order
        alpha = self.alpha_max * d**2
        return sigma, alpha
    
    def get_conductivity(self, x, y, dx=None, dt=None, eps_avg=None):
        """Calculate PML conductivity at a point using smooth-transition PML."""
        if dx is not None and eps_avg is not None:
            eta = np.sqrt(MU_0 / (EPS_0 * eps_avg))
            sigma_max = 1.2 / (eta * dx)
            sigma_max *= self.sigma_factor
        else: sigma_max = 1.0
        
        if self.region_type == "rect":
            if not (self.position[0] <= x <= self.position[0] + self.width and
                    self.position[1] <= y <= self.position[1] + self.height):
                return 0.0
            if self.orientation == "left": distance = 1.0 - (x - self.position[0]) / self.width
            elif self.orientation == "right": distance = (x - self.position[0]) / self.width
            elif self.orientation == "top": distance = (y - self.position[1]) / self.height
            elif self.orientation == "bottom": distance = 1.0 - (y - self.position[1]) / self.height
            else: return 0.0
        else:
            distance_from_corner = np.hypot(x - self.position[0], y - self.position[1])
            if distance_from_corner > self.radius: return 0.0
            dx_from_corner = x - self.position[0]
            dy_from_corner = y - self.position[1]
            if self.orientation == "top-left" and (dx_from_corner > 0 or dy_from_corner < 0): return 0.0
            elif self.orientation == "top-right" and (dx_from_corner < 0 or dy_from_corner < 0): return 0.0
            elif self.orientation == "bottom-left" and (dx_from_corner > 0 or dy_from_corner > 0): return 0.0
            elif self.orientation == "bottom-right" and (dx_from_corner < 0 or dy_from_corner > 0): return 0.0
            distance = distance_from_corner / self.radius
        
        sigma_profile, alpha_profile = self.get_profile(distance)
        conductivity = sigma_max * sigma_profile
        if dt is not None:
            frequency_factor = 1.0 / (1.0 + alpha_profile)
            conductivity *= frequency_factor
        return conductivity
    
    def copy(self):
        """Create a deep copy of the PML boundary."""
        return PML(
            region_type=self.region_type,
            position=self.position,
            size=(self.width, self.height) if self.region_type == "rect" else self.radius,
            orientation=self.orientation,
            polynomial_order=self.polynomial_order,
            sigma_factor=self.sigma_factor,
            alpha_max=self.alpha_max
        )


