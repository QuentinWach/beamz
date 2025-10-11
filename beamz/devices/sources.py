import numpy as np
from beamz.const import LIGHT_SPEED, µm, EPS_0, MU_0
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from beamz.devices.mode import solve_modes, tidy3d_mode_computation_wrapper

def _direction_to_axis(direction: str) -> int:
    """Convert direction string to axis index.
    
    Args:
        direction: Direction string ("+x", "-x", "+y", "-y", "+z", "-z")
        
    Returns:
        Axis index (0 for x, 1 for y, 2 for z)
    """
    if direction in ["+x", "-x"]:
        return 0
    elif direction in ["+y", "-y"]:
        return 1
    elif direction in ["+z", "-z"]:
        return 2
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be one of '+x', '-x', '+y', '-y', '+z', '-z'")

class GaussianSource():
    """A Gaussian current distribution in space.
    
    Args:
        position: Center of the Gaussian source (x, y) or (x, y, z).
        width: Standard deviation of the Gaussian distribution (spatial width).
        signal: Time-dependent signal.
    """
    def __init__(self, position=(0,0), width=1.0*µm, signal=0):
        self.position = self._ensure_3d_position(position)
        self.width = width
        self.signal = signal
        try:
            self.max_signal_magnitude = float(np.max(np.abs(signal)))
        except TypeError:
            self.max_signal_magnitude = float(abs(signal))
        if self.max_signal_magnitude == 0:
            self.max_signal_magnitude = 1.0
    
    def _ensure_3d_position(self, position):
        """Convert 2D position to 3D with z=0 if needed."""
        if len(position) == 2: return (position[0], position[1], 0)
        elif len(position) == 3: return position
        else: raise ValueError(f"Position must be 2D (x,y) or 3D (x,y,z), got {len(position)} dimensions")
    
    @property
    def position_2d(self):
        """Return 2D projection (x, y) for backwards compatibility."""
        return (self.position[0], self.position[1])
    
    def add_to_plot(self, ax, facecolor="crimson", edgecolor="crimson", alpha=1, linestyle="-"):
        # Use 2D projection for plotting
        ax.plot(self.position[0], self.position[1], 'o', color=facecolor, label='Gaussian Source')
    
    def copy(self):
        """Create a deep copy of the GaussianSource."""
        return GaussianSource(
            position=self.position,
            width=self.width,
            signal=self.signal
        )

# TODO: Add mode solver options to integrate the analytical mode solver in mode.py. Future: Add FDFD mode solver and Tidy3D mode solver.
# Make a comparison study!
class ModeSource():
    """Calculates and injects the mode profiles for a cross section.
    
    Args:
        design: Design object containing the structures
        position: Center position of the source (x,y) or (x,y,z)
        width: Width of the source cross-section (perpendicular to propagation)
        height: Height of the source cross-section (perpendicular to propagation, for 3D)
        wavelength: Source wavelength
        signal: Time-dependent signal
        direction: Direction of propagation ("+x", "-x", "+y", "-y", "+z", "-z")
        orientation: Orientation of the cross-section ("xy", "xz", "yz") - auto-determined from direction if not specified
        npml: Number of PML layers to use at boundaries
        num_modes: Number of modes to calculate
        mode_index: Index of the mode to inject (0 = fundamental mode, 1 = first higher-order mode, etc.)
        grid_resolution: Points per wavelength for grid resolution (higher = finer)
        mode_solver: Mode solver to use ("num_eigen" or "analytical")
        
        # Legacy support (deprecated):
        start: Starting point of the source line (x,y) or (x,y,z) - use position instead
        end: End point of the source line (x,y) or (x,y,z) - use position + width/height instead
    """
    def __init__(self, design, position=None, width=None, height=None, wavelength=1.55*µm, signal=0, direction="+x", 
                 orientation=None, npml=20, num_modes=2, mode_index=0, filter_pol=None, grid_resolution=2000, mode_solver="num_eigen",
                 start=None, end=None):
        # Handle legacy start/end parameters vs new position/width/height approach
        if start is not None and end is not None:
            # Legacy mode: use start and end points
            self.start = self._ensure_3d_position(start)
            self.end = self._ensure_3d_position(end)
            # Calculate position, width, height from start/end for consistency
            self._position = ((self.start[0] + self.end[0]) / 2, 
                            (self.start[1] + self.end[1]) / 2, 
                            (self.start[2] + self.end[2]) / 2)
            # Calculate width and height from the line
            line_vec = np.array([self.end[0] - self.start[0], 
                               self.end[1] - self.start[1], 
                               self.end[2] - self.start[2]])
            self.width = np.linalg.norm(line_vec)
            self.height = 0  # Line source has no height
            print("Warning: Using deprecated start/end parameters. Use position, width, height instead.")
        else:
            # New mode: use position, width, height
            if position is None:
                raise ValueError("Either (start, end) or position must be specified")
            
            self._position = self._ensure_3d_position(position)
            self.width = width if width is not None else wavelength  # Default width to wavelength
            self.height = height if height is not None else 0  # Default to 2D (line source)
            
            # Determine orientation from direction if not specified
            if orientation is None:
                if direction in ["+x", "-x"]:
                    orientation = "yz"  # Cross-section perpendicular to x
                elif direction in ["+y", "-y"]:
                    orientation = "xz"  # Cross-section perpendicular to y
                elif direction in ["+z", "-z"]:
                    orientation = "xy"  # Cross-section perpendicular to z
                else:
                    orientation = "yz"  # Default
            
            self.orientation = orientation
            
            # Calculate start and end points from position and dimensions
            self.start, self.end = self._calculate_start_end_from_position()
        
        self.wavelength = wavelength
        self.design = design
        self.signal = signal
        try:
            self.max_signal_magnitude = float(np.max(np.abs(signal)))
        except TypeError:
            if np.isscalar(signal):
                self.max_signal_magnitude = float(abs(signal))
            else:
                self.max_signal_magnitude = 1.0
        if not np.isfinite(self.max_signal_magnitude) or self.max_signal_magnitude == 0:
            self.max_signal_magnitude = 1.0
        self.direction = direction
        self.npml = npml
        self.num_modes = num_modes
        self.mode_index = mode_index
        self.grid_resolution = grid_resolution
        self.mode_solver = mode_solver
        
        # Auto-set filter_pol to "te" for 2D simulations if not specified
        # Robustly detect if this is a 2D or 3D simulation
        design_is_3d = False
        if hasattr(design, 'is_3d'):
            design_is_3d = design.is_3d
        elif hasattr(design, 'design') and hasattr(design.design, 'is_3d'):
            design_is_3d = design.design.is_3d
        # Also check depth
        design_depth = getattr(design, 'depth', None) or getattr(getattr(design, 'design', None), 'depth', None)
        if design_depth is not None and design_depth > 0:
            design_is_3d = True
        else:
            design_is_3d = False  # If depth is 0 or None, it's 2D
        
        is_2d = not design_is_3d
        if filter_pol is None and is_2d:
            filter_pol = "te"  # Force TE modes (Ez dominant) for 2D
        self.filter_pol = filter_pol
        # print(f"[DEBUG] ModeSource init: is_2d={is_2d}, filter_pol={filter_pol}, design_depth={design_depth}")
        
        # Validate mode_index
        if mode_index < 0:
            raise ValueError(f"mode_index must be non-negative, got {mode_index}")
        if mode_index >= num_modes:
            raise ValueError(f"mode_index ({mode_index}) must be less than num_modes ({num_modes})")
        
        # Validate source line orientation matches propagation direction
        self._validate_source_orientation()
        
        # Calculate and store mode profiles
        self.dL = self.wavelength / grid_resolution  # Sampling resolution
        self.omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        self.max_field_amplitude = 0.0
        self.max_power_density = 0.0
        
        # Solve modes using default resolution
        self._solve_modes_internal(resolution=self.dL)

    def _solve_modes_internal(self, resolution: float):
        """Internal method to solve modes with given resolution.
        
        Args:
            resolution: Grid spacing for mode solving (in meters)
        """
        # Sample permittivity along source line with given resolution
        x0, y0, z0 = self.start[0], self.start[1], self.start[2]
        x1, y1, z1 = self.end[0], self.end[1], self.end[2]
        line_length = np.hypot(x1 - x0, y1 - y0)
        num_points = max(int(line_length / resolution), 10)  # At least 10 points
        
        x = np.linspace(x0, x1, num_points)
        y = np.linspace(y0, y1, num_points)
        z = np.linspace(z0, z1, num_points)
        eps_1d = np.zeros(num_points)
        for i, (x_i, y_i, z_i) in enumerate(zip(x, y, z)):
            eps_1d[i], _, _ = self.design.get_material_value(x_i, y_i, z_i)
        
        # Solve modes
        (
            self.effective_indices,
            self.mode_e_fields,
            self.mode_h_fields,
            self.propagation_axis,
        ) = solve_modes(
            eps_1d,
            self.omega,
            resolution,
            npml=self.npml,
            m=self.num_modes,
            direction=self.direction,
            filter_pol=self.filter_pol,
            return_fields=True,
        )

        # Convert mode fields to profiles along source line
        mode_e_fields = self.mode_e_fields
        mode_h_fields = self.mode_h_fields
        self.mode_vectors = np.zeros((eps_1d.size, mode_e_fields.shape[0]), dtype=complex)
        self.mode_profiles = []
        
        spacing = line_length / max(num_points - 1, 1)

        for mode_idx in range(mode_e_fields.shape[0]):
            e_field = mode_e_fields[mode_idx]
            h_field = mode_h_fields[mode_idx]

            # Auto-detect dominant field components
            e_norms = [np.linalg.norm(e_field[i]) for i in range(3)]
            h_norms = [np.linalg.norm(h_field[i]) for i in range(3)]
            e_idx_use = int(np.argmax(e_norms))
            h_sorted = np.argsort(h_norms)[::-1]
            h1_idx_use = int(h_sorted[0])
            h2_idx_use = int(h_sorted[1])
            
            # Extract field components
            E_main_line = self._collapse_field_to_line(e_field, e_idx_use, num_points)
            H_trans1_line = self._collapse_field_to_line(h_field, h1_idx_use, num_points)
            H_trans2_line = self._collapse_field_to_line(h_field, h2_idx_use, num_points)
            
            # For 2D FDTD compatibility: Always map dominant E to "Ez"
            design_is_3d = False
            if hasattr(self.design, 'is_3d'):
                design_is_3d = self.design.is_3d
            elif hasattr(self.design, 'design') and hasattr(self.design.design, 'is_3d'):
                design_is_3d = self.design.design.is_3d
            design_depth = getattr(self.design, 'depth', None) or getattr(getattr(self.design, 'design', None), 'depth', None)
            if design_depth is not None and design_depth > 0:
                design_is_3d = True
            else:
                design_is_3d = False
            is_2d = not design_is_3d
            axis = _direction_to_axis(self.direction)
            
            if is_2d:
                # Map dominant E-field to "Ez" for 2D compatibility
                field_map = {
                    'E_main': 'Ez',
                    'H_trans1': 'Hx',
                    'H_trans2': 'Hy'
                }
            else:
                # 3D: Use actual component names
                if axis == 0:
                    comp_names = [['Ez', 'Ex', 'Ey'], ['Hz', 'Hx', 'Hy']]
                elif axis == 1:
                    comp_names = [['Ex', 'Ez', 'Ey'], ['Hx', 'Hz', 'Hy']]
                else:
                    comp_names = [['Ex', 'Ey', 'Ez'], ['Hx', 'Hy', 'Hz']]
                
                e_main_name = comp_names[0][e_idx_use]
                h1_name = comp_names[1][h1_idx_use]
                h2_name = comp_names[1][h2_idx_use]
                
                field_map = {
                    'E_main': e_main_name,
                    'H_trans1': h1_name,
                    'H_trans2': h2_name
                }

            # For backward propagation, flip H-field signs
            is_backward = not self.direction.startswith("+")
            if is_backward:
                H_trans1_line = -H_trans1_line
                H_trans2_line = -H_trans2_line

            # Calculate Poynting vector and normalize
            S_complex = E_main_line * np.conj(H_trans1_line)
            power_total = np.real(np.sum(S_complex) * spacing)
            if power_total == 0.0 or not np.isfinite(power_total):
                power_total = 1e-12
            scale = 1.0 / np.sqrt(abs(power_total))
            E_main_line *= scale
            H_trans1_line *= scale
            H_trans2_line *= scale
            
            # Additional normalization
            e_max = np.max(np.abs(E_main_line))
            if e_max > 1.0 and np.isfinite(e_max):
                renorm = 1.0 / e_max
                E_main_line *= renorm
                H_trans1_line *= renorm
                H_trans2_line *= renorm

            power_density_mag = np.abs(np.real(S_complex))
            if power_density_mag.size:
                self.max_power_density = max(self.max_power_density, float(np.max(power_density_mag)))

            self.mode_vectors[:, mode_idx] = E_main_line
            self.max_field_amplitude = max(self.max_field_amplitude, float(np.max(np.abs(E_main_line))))
            
            # Build profile with exact coordinates along source line
            profile = []
            for idx, (E_amp, H1_amp, H2_amp) in enumerate(zip(E_main_line, H_trans1_line, H_trans2_line)):
                x_pos = x[idx]
                y_pos = y[idx]
                z_pos = z[idx]
                
                point = {"x": x_pos, "y": y_pos, "z": z_pos}
                point[field_map['E_main']] = E_amp
                point[field_map['H_trans1']] = H1_amp
                point[field_map['H_trans2']] = H2_amp
                profile.append(point)
            
            self.mode_profiles.append(profile)

        if self.max_power_density <= 0.0:
            eta0 = np.sqrt(MU_0 / EPS_0)
            if self.max_field_amplitude <= 0.0:
                self.max_field_amplitude = 1.0
            self.max_power_density = (self.max_field_amplitude ** 2) / max(eta0, 1e-12)
    
    def compute_modes_on_fdtd_grid(self, dx: float, dy: float, dz: float = None):
        """Recompute modes using FDTD grid spacing for perfect alignment.
        
        This method resolves modes on the exact FDTD grid spacing, eliminating 
        interpolation errors and ensuring 1:1 mapping between mode profiles 
        and FDTD cells.
        
        Args:
            dx: FDTD grid spacing in x direction (meters)
            dy: FDTD grid spacing in y direction (meters)
            dz: FDTD grid spacing in z direction (meters), optional for 2D
        
        The mode solver resolution is set to the finest grid spacing to ensure
        the mode captures all grid-scale features of the permittivity distribution.
        """
        # Use the finest grid spacing as mode solver resolution
        # This ensures modes capture all grid-scale features
        if dz is not None:
            resolution = min(dx, dy, dz)
        else:
            resolution = min(dx, dy)
        
        # Recompute modes with FDTD grid resolution
        print(f"[ModeSource] Recomputing modes with FDTD grid resolution: {resolution*1e9:.3f} nm")
        self._solve_modes_internal(resolution=resolution)
        print(f"[ModeSource] Grid-aligned mode profiles generated: {len(self.mode_profiles)} modes")
    
    def _collapse_field_to_line(self, field: np.ndarray, component_idx: int, target_len: int) -> np.ndarray:
        component = np.squeeze(field[component_idx])
        if component.ndim == 2:
            # Average over axis perpendicular to the source span
            component = component.mean(axis=-1)
        src = np.linspace(0.0, 1.0, component.size)
        dst = np.linspace(0.0, 1.0, target_len)
        real_interp = np.interp(dst, src, component.real)
        imag_interp = np.interp(dst, src, component.imag)
        return real_interp + 1j * imag_interp
    
    def copy(self):
        """Create a deep copy of the ModeSource."""
        if hasattr(self, 'start') and hasattr(self, 'end'):
            # Legacy mode
            return ModeSource(
                design=self.design,  # Reference to same design is okay
                start=self.start,
                end=self.end,
                wavelength=self.wavelength,
                signal=self.signal,
                direction=self.direction,
                npml=self.npml,
                num_modes=self.num_modes,
                mode_index=self.mode_index,
                grid_resolution=self.grid_resolution,
                mode_solver=self.mode_solver
            )
        else:
            # New mode
            return ModeSource(
                design=self.design,  # Reference to same design is okay
                position=self._position,
                width=self.width,
                height=self.height,
                wavelength=self.wavelength,
                signal=self.signal,
                direction=self.direction,
                orientation=self.orientation,
                npml=self.npml,
                num_modes=self.num_modes,
                mode_index=self.mode_index,
                grid_resolution=self.grid_resolution,
                mode_solver=self.mode_solver
            )
            
    def _get_field_components_for_direction(self) -> dict:
        """Return the field component names for current propagation direction.
        
        For 2D TE-like propagation:
        - x-propagation: Main E-field is Ez (perpendicular to xy plane)
        - y-propagation: Main E-field is Ez (perpendicular to xy plane)
        
        Returns dict with keys: 'E_main', 'H_trans1', 'H_trans2'
        """
        axis = _direction_to_axis(self.direction)
        
        if axis == 0:  # x-propagation
            # TE mode: Ez is main, Hx and Hy are transverse
            return {
                'E_main': 'Ez',
                'H_trans1': 'Hx', 
                'H_trans2': 'Hy',
                'S_components': ('Ez', 'Hy')  # For Poynting: Sx = -Ez * Hy*
            }
        elif axis == 1:  # y-propagation
            # TE mode: Ez is main, Hz and Hx are transverse
            return {
                'E_main': 'Ez',
                'H_trans1': 'Hz',
                'H_trans2': 'Hx',
                'S_components': ('Ez', 'Hx')  # For Poynting: Sy = Ez * Hx*
            }
        else:  # z-propagation (3D only)
            # Use Ex as main field
            return {
                'E_main': 'Ex',
                'H_trans1': 'Hy',
                'H_trans2': 'Hz',
                'S_components': ('Ex', 'Hz')  # For Poynting: Sz = Ex * Hy* - Ey * Hx*
            }
    
    def _validate_source_orientation(self):
        """Validate that the source line is perpendicular to the propagation direction."""
        if not hasattr(self, 'start') or not hasattr(self, 'end'):
            return  # Skip validation for position/width/height mode
        
        dx = abs(self.end[0] - self.start[0])
        dy = abs(self.end[1] - self.start[1])
        dz = abs(self.end[2] - self.start[2])
        
        # Determine which axis the source line is along
        threshold = 1e-9
        line_is_along_x = dx > max(dy, dz) + threshold
        line_is_along_y = dy > max(dx, dz) + threshold
        line_is_along_z = dz > max(dx, dy) + threshold
        
        # Check if line orientation is perpendicular to propagation
        prop_axis = _direction_to_axis(self.direction)
        
        if prop_axis == 0:  # Propagation in x
            if line_is_along_x:
                raise ValueError(
                    f"ModeSource line must be perpendicular to propagation direction '{self.direction}'.\n"
                    f"For propagation in ±x, the source line should span in y or z direction.\n"
                    f"Current line: start={self.start}, end={self.end}"
                )
        elif prop_axis == 1:  # Propagation in y
            if line_is_along_y:
                raise ValueError(
                    f"ModeSource line must be perpendicular to propagation direction '{self.direction}'.\n"
                    f"For propagation in ±y, the source line should span in x or z direction.\n"
                    f"Current line: start={self.start}, end={self.end}"
                )
        elif prop_axis == 2:  # Propagation in z
            if line_is_along_z:
                raise ValueError(
                    f"ModeSource line must be perpendicular to propagation direction '{self.direction}'.\n"
                    f"For propagation in ±z, the source line should span in x or y direction.\n"
                    f"Current line: start={self.start}, end={self.end}"
                )
    
    def _ensure_3d_position(self, position):
        """Convert 2D position to 3D with z=0 if needed."""
        if position is None:
            return None
        if len(position) == 2: return (position[0], position[1], 0)
        elif len(position) == 3: return position
        else: raise ValueError(f"Position must be 2D (x,y) or 3D (x,y,z), got {len(position)} dimensions")
    
    def _calculate_start_end_from_position(self):
        """Calculate start and end points from position, width, height, and orientation."""
        x, y, z = self._position
        
        if self.orientation == "yz":
            # Cross-section in yz plane (propagation in x direction)
            if self.height == 0:
                # 2D line source in y direction
                start = (x, y - self.width/2, z)
                end = (x, y + self.width/2, z)
            else:
                # 3D rectangular source in yz plane
                # For mode calculation, we still need a line - use the center line in y direction
                start = (x, y - self.width/2, z)
                end = (x, y + self.width/2, z)
        elif self.orientation == "xz":
            # Cross-section in xz plane (propagation in y direction)
            if self.height == 0:
                # 2D line source in x direction
                start = (x - self.width/2, y, z)
                end = (x + self.width/2, y, z)
            else:
                # 3D rectangular source in xz plane
                start = (x - self.width/2, y, z)
                end = (x + self.width/2, y, z)
        elif self.orientation == "xy":
            # Cross-section in xy plane (propagation in z direction)
            if self.height == 0:
                # 2D line source in x direction (default)
                start = (x - self.width/2, y, z)
                end = (x + self.width/2, y, z)
            else:
                # 3D rectangular source in xy plane
                start = (x - self.width/2, y, z)
                end = (x + self.width/2, y, z)
        else:
            raise ValueError(f"Invalid orientation: {self.orientation}. Must be 'xy', 'xz', or 'yz'")
        
        return start, end
    
    @property
    def position(self):
        """Return the center position of the source."""
        if hasattr(self, '_position'):
            return self._position
        else:
            # Legacy mode: calculate from start and end points
            return ((self.start[0] + self.end[0]) / 2, 
                    (self.start[1] + self.end[1]) / 2, 
                    (self.start[2] + self.end[2]) / 2)
    
    @property
    def position_2d(self):
        """Return 2D projection of the midpoint for backwards compatibility."""
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)

    def get_eps_1d(self):
        """Calculate the 1D permittivity profile by stepping along the line from start to end point."""
        x0, y0, z0 = self.start[0], self.start[1], self.start[2]
        x1, y1, z1 = self.end[0], self.end[1], self.end[2]
        num_points = int(np.hypot(x1 - x0, y1 - y0) / self.dL)  # Use the class dL value
        x = np.linspace(x0, x1, num_points)
        y = np.linspace(y0, y1, num_points)
        z = np.linspace(z0, z1, num_points)
        eps_1d = np.zeros(num_points)
        for i, (x_i, y_i, z_i) in enumerate(zip(x, y, z)):
            eps_1d[i], _, _ = self.design.get_material_value(x_i, y_i, z_i)
        return eps_1d
    
    def get_xy_mode_line(self, vecs, mode_number):
        """Get the mode profile for a specific mode along the line."""
        x0, y0, z0 = self.start[0], self.start[1], self.start[2]
        x1, y1, z1 = self.end[0], self.end[1], self.end[2]
        num_points = vecs.shape[0]  # Number of points along the line
        x = np.linspace(x0, x1, num_points)
        y = np.linspace(y0, y1, num_points)
        z = np.linspace(z0, z1, num_points)
        # Create mode profile for the specified mode
        mode_profile = []
        for j in range(num_points):  # For each point
            # Use the complex field value to preserve phase information
            amplitude = vecs[j, mode_number]  # Keep complex value with phase
            mode_profile.append([amplitude, x[j], y[j], z[j]])
        return mode_profile

    def _build_3d_rect_mode_profiles(self):
        """Construct a 2D separable mode profile on the cross-section plane for 3D injections.
        For +x/-x propagation, use yz plane; for +y/-y use xz; for +z/-z use xy.
        Uses two 1D slab mode solves along the two cross-section axes and creates an outer-product field.
        """
        # Determine cross-section axes
        x0, y0, z0 = self.position
        # Sampling resolution on the cross-section
        dL = self.dL
        num_y = max(8, int(round(self.width / dL)))
        num_z = max(8, int(round(self.height / dL)))
        # Define sampling ranges centered at position
        y_min, y_max = y0 - self.width/2, y0 + self.width/2
        z_min, z_max = z0 - self.height/2, z0 + self.height/2
        ys = np.linspace(y_min, y_max, num_y)
        zs = np.linspace(z_min, z_max, num_z)
        # Infer core/cladding along y and z from design materials
        eps_line_y = np.array([self.design.get_material_value(x0, y, z0)[0] for y in ys])
        eps_line_z = np.array([self.design.get_material_value(x0, y0, z)[0] for z in zs])
        n_core_y = np.sqrt(np.max(eps_line_y)); n_clad_y = np.sqrt(np.min(eps_line_y))
        n_core_z = np.sqrt(np.max(eps_line_z)); n_clad_z = np.sqrt(np.min(eps_line_z))
        # Estimate effective core widths by thresholding at 90% of peak eps
        def estimate_width(coords, eps_line):
            thr = 0.9 * np.max(eps_line)
            idx = np.where(eps_line >= thr)[0]
            if idx.size > 0:
                return coords[idx[-1]] - coords[idx[0]]
            return max(coords[-1]-coords[0], dL)
        wy_eff = estimate_width(ys, eps_line_y)
        wz_eff = estimate_width(zs, eps_line_z)
        # Build 1D slab modes along y and z (fundamental)
        try:
            Ey_1d, _ = slab_mode_source(x=ys, w=wy_eff, n_WG=n_core_y, n0=n_clad_y, wavelength=self.wavelength, ind_m=0, x0=y0)
        except Exception:
            Ey_1d = np.ones_like(ys)
        try:
            Ez_1d, _ = slab_mode_source(x=zs, w=wz_eff, n_WG=n_core_z, n0=n_clad_z, wavelength=self.wavelength, ind_m=0, x0=z0)
        except Exception:
            Ez_1d = np.ones_like(zs)
        # Normalize
        Ey_1d = Ey_1d / np.max(np.abs(Ey_1d) + 1e-12)
        Ez_1d = Ez_1d / np.max(np.abs(Ez_1d) + 1e-12)
        # Outer product field on cross-section (choose TE-like Ez component)
        field_yz = np.outer(np.abs(Ey_1d), np.abs(Ez_1d))
        # Normalize amplitude
        field_yz /= (np.max(field_yz) + 1e-12)
        # Create list of [amplitude, x, y, z] samples across plane
        mode_profile = []
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                amp = field_yz[iy, iz]
                mode_profile.append([amp, x0, y, z])
        # Only one mode profile used for injection
        return [mode_profile]

    def _sample_mode_field(self, mode_number: int):
        ez_idx, hx_idx, hy_idx = self._field_component_indices()
        e_field = self.mode_e_fields[mode_number]
        h_field = self.mode_h_fields[mode_number]
        ny, nx = e_field.shape[1:]
        xs = np.linspace(self.start[0], self.end[0], nx)
        ys = np.linspace(self.start[1], self.end[1], ny)
        dx = (xs[1] - xs[0]) if nx > 1 else self.dL
        dy = (ys[1] - ys[0]) if ny > 1 else self.dL

        Ez_grid = e_field[ez_idx]
        Hx_grid = h_field[hx_idx]
        Hy_grid = h_field[hy_idx]
        power_density = -np.real(Ez_grid * np.conj(Hy_grid))
        power_total = np.sum(power_density) * dx * dy
        power_total = max(power_total, 1e-12)
        scale = 1.0 / np.sqrt(power_total)
        Ez_grid *= scale
        Hx_grid *= scale
        Hy_grid *= scale

        profile = []
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                profile.append({
                    "Ez": Ez_grid[j, i],
                    "Hx": Hx_grid[j, i],
                    "Hy": Hy_grid[j, i],
                    "x": x,
                    "y": y,
                    "z": 0.0,
                })
        return profile

    def _field_component_indices(self):
        """Return indices to extract (E_main, H_trans1, H_trans2) from mode solver output.
        
        Mode solver returns stacked arrays based on propagation axis:
        - axis=0 (x): E=[Ez, Ex, Ey], H=[Hz, Hx, Hy]
        - axis=1 (y): E=[Ex, Ez, Ey], H=[-Hx, -Hz, -Hy]
        - axis=2 (z): E=[Ex, Ey, Ez], H=[Hx, Hy, Hz]
        
        For 2D TE modes we want:
        - x-prop: Ez, Hx, Hy
        - y-prop: Ez, Hz, Hx
        - z-prop: Ex, Hy, Hz
        """
        axis = getattr(self, "propagation_axis", _direction_to_axis(self.direction))
        if axis == 0:  # x-propagation
            return 0, 1, 2  # Ez, Hx, Hy
        if axis == 1:  # y-propagation
            return 1, 1, 0  # Ez, Hz, Hx (note: H[1] is -Hz but we flip sign for backward)
        return 2, 0, 1  # Ex, Hy, Hz (for z-propagation)

    def show(self):
        """Visualize the mode profiles for this source."""
        if getattr(self.design, "is_3d", False) and self.height and self.height > 0:
            return self._show_3d_slice()
        return self._show_1d_profile()

    def _show_1d_profile(self):
        eps_1d = self.get_eps_1d()
        N = eps_1d.size
        # Recalculate physical coordinates for plotting (assuming linear path)
        # Use total length and N to get coordinates corresponding to eps_1d indices
        line_length = np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])
        # Create coordinate array from 0 to line_length
        coords = np.linspace(0, line_length, N) / µm # Plot in microns
        plot_unit = 'µm'
        vals, vecs = solve_modes(eps_1d, self.omega, self.dL, npml=self.npml, m=self.num_modes)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        # Plot permittivity profile vs physical coordinates
        ax1.plot(coords, eps_1d, color='black', label='1D Permittivity Profile')
        ax1.set_xlabel(f'Position along the line ({plot_unit})')
        ax1.set_ylabel('Relative Permittivity', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_xlim(coords[0], coords[-1]) # Set limits based on coordinate range
        # Create a second y-axis for the mode profiles
        ax2 = ax1.twinx()
        colors = ['crimson', 'blue', 'green', 'orange', 'purple']
        for i in range(vecs.shape[1]):
            ax2.plot(coords, np.abs(vecs[:, i])**2, color=colors[i % len(colors)], 
                     label=f'Mode {i+1} Effective index: {vals[i].real:.3f}')
        ax2.set_ylabel('Mode Intensity (|E|²)') # Changed label for clarity
        ax2.tick_params(axis='y')
        # Ensure y-axis starts at 0 for intensity
        ax2.set_ylim(bottom=0)

        # Add shaded regions for PML
        if self.npml > 0 and N > self.npml:
            pml_width_left = coords[self.npml-1] - coords[0]
            pml_width_right = coords[-1] - coords[N-self.npml]
            # Left PML region & right PML region
            ax1.add_patch(patches.Rectangle((coords[0], ax1.get_ylim()[0]), pml_width_left, 
                ax1.get_ylim()[1]-ax1.get_ylim()[0], facecolor='gray', alpha=0.2, label='PML Region'))
            ax1.add_patch(patches.Rectangle((coords[N-self.npml], ax1.get_ylim()[0]), pml_width_right, 
                ax1.get_ylim()[1]-ax1.get_ylim()[0], facecolor='gray', alpha=0.2))
            # Adjust xlim slightly to make patches fully visible if needed
            ax1.set_xlim(coords[0] - 0.01*line_length/µm, coords[-1] + 0.01*line_length/µm)

        plt.title('Mode Profiles')
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Avoid duplicate PML label if patch was added
        unique_labels = {} 
        for line, label in zip(lines1 + lines2, labels1 + labels2):
            if label not in unique_labels: unique_labels[label] = line
        ax2.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
        plt.grid(True)
        fig.tight_layout()
        plt.show()

    def _show_3d_slice(self):
        try:
            import tidy3d  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("ModeSource.show requires tidy3d for 3D designs.") from exc

        pos = self.position
        direction_axis = _direction_to_axis(self.direction)
        if direction_axis == 0:  # propagation ±x, slice in y-z plane
            axis1, axis2 = 1, 2
        elif direction_axis == 1:  # ±y
            axis1, axis2 = 0, 2
        else:  # ±z
            axis1, axis2 = 0, 1

        center1 = pos[axis1]
        center2 = pos[axis2]
        half1 = (self.width or (self.dL * 20)) / 2
        half2 = (self.height or self.width or (self.dL * 20)) / 2

        resolution = max(self.dL, 1e-8)
        n1 = max(64, int(np.round((2 * half1) / resolution)))
        n2 = max(64, int(np.round((2 * half2) / resolution)))

        coords1 = np.linspace(center1 - half1, center1 + half1, n1)
        coords2 = np.linspace(center2 - half2, center2 + half2, n2)

        eps = np.empty((n2, n1), dtype=float)
        for i, c2 in enumerate(coords2):
            for j, c1 in enumerate(coords1):
                sample = [pos[0], pos[1], pos[2]]
                sample[axis1] = c1
                sample[axis2] = c2
                eps_val, _, _ = self.design.get_material_value(*sample)
                eps[i, j] = float(eps_val)

        edges1 = np.linspace(coords1[0] - resolution / 2, coords1[-1] + resolution / 2, n1 + 1)
        edges2 = np.linspace(coords2[0] - resolution / 2, coords2[-1] + resolution / 2, n2 + 1)

        frequency = self.omega / (2 * np.pi)
        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency,
            permittivity_cross_section=eps,
            coords=[edges1 / µm, edges2 / µm],
            direction="+" if self.direction.startswith("+") else "-",
            num_modes=min(6, self.num_modes + 2),
            precision="double",
        )

        modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)
        window_um = (max(half1, half2) * 1.1) / µm

        extent = (edges1[0] / µm, edges1[-1] / µm, edges2[0] / µm, edges2[-1] / µm)

        fig, axes = plt.subplots(2, min(3, len(modes)), figsize=(12, 7), constrained_layout=True)
        axes = np.atleast_2d(axes)

        labels_axis = ['x', 'y', 'z']
        axis_labels = [labels_axis[axis1], labels_axis[axis2]]

        for col, mode in enumerate(modes[:3]):
            Ez = np.real(np.array(mode.Ez))
            Hy = np.real(np.array(mode.Hy))
            vmax_e = np.max(np.abs(Ez)) or 1.0
            vmax_h = np.max(np.abs(Hy)) or 1.0

            ax_top = axes[0, col]
            ax_top.imshow(Ez.T / vmax_e, origin="lower", extent=extent, cmap="viridis", vmin=-1, vmax=1)
            ax_top.set_aspect('equal')
            ax_top.set_xlim(-window_um, window_um)
            ax_top.set_ylim(-window_um, window_um)
            ax_top.set_title(f"Mode {col}: Re(Ez)\nneff = {float(np.real(mode.neff)):.3f}")
            ax_top.set_xlabel(f"{axis_labels[0]} (µm)")
            ax_top.set_ylabel(f"{axis_labels[1]} (µm)")

            ax_bottom = axes[1, col]
            im = ax_bottom.imshow(Hy.T / vmax_h, origin="lower", extent=extent, cmap="viridis", vmin=-1, vmax=1)
            ax_bottom.set_aspect('equal')
            ax_bottom.set_xlim(-window_um, window_um)
            ax_bottom.set_ylim(-window_um, window_um)
            ax_bottom.set_title(f"Mode {col}: Re(Hy)")
            ax_bottom.set_xlabel(f"{axis_labels[0]} (µm)")
            ax_bottom.set_ylabel(f"{axis_labels[1]} (µm)")
            plt.colorbar(im, ax=ax_bottom, fraction=0.046, pad=0.04)

        fig.suptitle("Mode profiles on source cross-section", y=1.02)
        plt.show()

    def add_to_plot(self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None):
        """Add the mode source to the plot."""
        if facecolor is None: facecolor = "crimson"
        if alpha is None: alpha = 1
        if linestyle is None: linestyle = '-'
        # Draw the source line using 2D projection
        start_2d = (self.start[0], self.start[1])
        end_2d = (self.end[0], self.end[1])
        ax.plot((start_2d[0], end_2d[0]), (start_2d[1], end_2d[1]), '-', lw=4, color=facecolor, label='Mode Source', zorder=10)
        ax.plot((start_2d[0], end_2d[0]), (start_2d[1], end_2d[1]), '-', lw=1, color=edgecolor, zorder=10)
        # Calculate arrow position and direction
        mid_x = (start_2d[0] + end_2d[0]) / 2
        mid_y = (start_2d[1] + end_2d[1]) / 2
        # Get the line length for scaling (2D projection)
        line_length = np.hypot(end_2d[0] - start_2d[0], end_2d[1] - start_2d[1])
        # Determine arrow direction based on self.direction parameter
        dx, dy = 0, 0
        if self.direction == "+x": dx, dy = 1, 0
        elif self.direction == "-x": dx, dy = -1, 0
        elif self.direction == "+y": dx, dy = 0, 1
        elif self.direction == "-y": dx, dy = 0, -1
        elif self.direction == "+z" or self.direction == "-z":
            # For z-direction, show an arrow along the 2D projection if available,
            # otherwise show a small perpendicular arrow
            if line_length > 0:
                # Use line direction for z-propagation indication
                line_dx = (end_2d[0] - start_2d[0]) / line_length
                line_dy = (end_2d[1] - start_2d[1]) / line_length
                dx, dy = line_dx, line_dy
            else:
                # Default to x-direction if no 2D projection
                dx, dy = 1, 0
        # Scale the arrow - adaptive sizing based on line length
        # Use minimum size for very short lines
        min_arrow_length = 0.8 * self.wavelength  # Increased minimum size
        arrow_length = max(line_length * 0.2, min_arrow_length) if line_length > 0 else min_arrow_length
        # Use normalized direction vector
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:  # Avoid division by zero
            dx = dx / magnitude * arrow_length
            dy = dy / magnitude * arrow_length
        # Calculate appropriate head width and length
        head_width = arrow_length * 0.7
        head_length = arrow_length * 0.5
        # Draw the arrow with higher zorder to ensure visibility
        ax.arrow(mid_x, mid_y, dx, dy, 
                head_width=head_width,
                head_length=head_length, 
                fc=facecolor, ec="black",  # Use black for better visibility
                alpha=alpha, linewidth=1,  # Thicker line
                width=head_width*0.5,  # Narrower arrow body
                length_includes_head=True,
                zorder=11)  # Higher zorder to ensure it's drawn on top