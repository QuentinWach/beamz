import numpy as np
from typing import Literal
from beamz.devices.sources.solve import solve_modes
from beamz.const import µm, LIGHT_SPEED

class ModeSource:
    """Unidirectional mode source using Huygens surface equivalent currents on the Yee grid.
    
    For TEz 2D with +x propagation:
    - J_z = H_y^mode injected at Ez Yee positions
    - M_y = E_z^mode injected at Hy Yee positions
    """
    
    def __init__(self, grid, center, width, wavelength, pol: Literal["te", "tm"], signal, direction: Literal["+x", "-x"] = "+x"):
        """Initialize the mode source.
        
        Args:
            grid: The simulation grid
            center: (x, y) center position of the source plane
            width: Width of the source in the y-direction
            wavelength: Operating wavelength
            pol: Polarization ("te" or "tm")
            signal: Time-dependent signal function s(t)
            direction: Propagation direction ("+x" or "-x")
        """
        self.grid = grid
        self.center = center if isinstance(center, (tuple, list)) else (center, grid.height / 2)
        self.width = width
        self.wavelength = wavelength
        self.pol = pol
        self.signal = signal
        self.direction = direction
        
        self._jz_profile = None
        self._my_profile = None
        self._ez_indices = None
        self._hy_indices = None
        self._neff = None
        
    def initialize(self, permittivity, resolution):
        """Compute the mode and set up the source currents on the Yee grid."""
        dx = dy = resolution
        ny, nx = permittivity.shape
        
        # Determine the x-column for the source plane
        x_ez_idx = int(np.clip(np.round(self.center[0] / dx - 0.5), 0, nx - 1))
        x_ez_coord = (x_ez_idx + 0.5) * dx
        
        # For unidirectional +x propagation: M_y at Hy column to the LEFT of Ez column
        # For -x propagation: M_y at Hy column to the RIGHT of Ez column
        if self.direction == "+x":
            x_hy_idx = max(0, x_ez_idx - 1)  # One column to the left
        else:  # "-x"
            x_hy_idx = min(nx - 1, x_ez_idx + 1)  # One column to the right
        
        # Determine the y-extent of the source
        y_min = max(0.0, self.center[1] - abs(self.width) / 2)
        y_max = min(self.grid.height, self.center[1] + abs(self.width) / 2)
        
        # Ez positions (center of cells): (i+1/2, j+1/2)
        y_ez_start = int(np.clip(np.floor(y_min / dy), 0, ny - 1))
        y_ez_end = int(np.clip(np.ceil(y_max / dy), y_ez_start + 1, ny))
        y_ez_coords = (np.arange(y_ez_start, y_ez_end) + 0.5) * dy
        
        # Hy positions (edge centers between Ez rows): one fewer sample in y
        # Ez rows: indices [y_ez_start .. y_ez_end-1] at (j+0.5)dy
        # Hy rows live between Ez rows -> y positions at (j+1.0)dy for j in [y_ez_start .. y_ez_end-2]
        y_hy_start = y_ez_start
        y_hy_end = max(y_ez_start, y_ez_end - 1)
        y_hy_coords = (np.arange(y_hy_start, y_hy_end) + 1.0) * dy
        
        # Sample permittivity at Ez positions for mode solving
        eps_profile = permittivity[y_ez_start:y_ez_end, x_ez_idx]
        
        # Solve for the mode
        # The solve_modes function treats the 1D eps profile as varying along axis 1 (y in our case)
        # and assumes propagation along axis 0 (which becomes x for us)
        # It returns fields with propagation_axis indicating which axis is propagation
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        neff, e_fields, h_fields, prop_axis = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=dy,
            m=1,
            direction=self.direction,
            filter_pol=self.pol,
            return_fields=True
        )
        
        self._neff = neff[0]
        
        # Extract mode fields
        E_mode = e_fields[0]
        H_mode = h_fields[0]
        
        # The mode solver returns fields where prop_axis indicates propagation direction
        # For 2D with propagation in x and variation in y, the solver reorders components
        # Based on polarization, extract the appropriate field components for Huygens currents
        if self.pol == "te":
            # TE: E transverse → Ez (out of plane), Hx, Hy (in plane)
            Ez_mode = np.squeeze(E_mode[2])  # Ez
            Hy_mode = np.squeeze(H_mode[1])  # Hy
            # Fallback if Ez is near zero (solver returned TM-like)
            if np.max(np.abs(Ez_mode)) < 1e-9 and np.max(np.abs(np.squeeze(E_mode[1]))) > 1e-9:
                if not hasattr(self, "_warned_tm_fallback"):
                    print("[ModeSource] Warning: TE Ez component near zero; falling back to TM mapping (Ey→Ez, Hz→Hy)")
                    self._warned_tm_fallback = True
                Ez_mode = np.squeeze(E_mode[1])
                Hy_mode = np.squeeze(H_mode[2])
        elif self.pol == "tm":
            # TM: H transverse → Hz (out of plane), Ex, Ey (in plane)
            # Map to equivalent TEz-like fields: Ey→Ez, Hz→Hy
            Ez_mode = np.squeeze(E_mode[1])  # Ey as Ez equivalent
            Hy_mode = np.squeeze(H_mode[2])  # Hz as Hy equivalent
        else:
            raise ValueError(f"Unknown polarization: {self.pol}")
        
        # Ensure proper propagation direction by checking Poynting vector
        S_x = np.real(Ez_mode * np.conj(Hy_mode))
        power_x = np.sum(S_x)
        direction_sign = 1.0 if self.direction.startswith("+") else -1.0
        if power_x * direction_sign < 0:
            Hy_mode = -Hy_mode
        
        # Phase align TOGETHER to preserve impedance relationship
        # Find the peak of E_z and align both fields to that phase
        idx_max = np.argmax(np.abs(Ez_mode))
        phase_ref = np.angle(Ez_mode[idx_max])
        Ez_mode = Ez_mode * np.exp(-1j * phase_ref)
        Hy_mode = Hy_mode * np.exp(-1j * phase_ref)
        
        # Debug: Check mode profile
        print(f"[DEBUG] Mode profile at peak:")
        print(f"  Ez: real={np.real(Ez_mode[idx_max]):.6f}, imag={np.imag(Ez_mode[idx_max]):.6f}")
        print(f"  Hy: real={np.real(Hy_mode[idx_max]):.6f}, imag={np.imag(Hy_mode[idx_max]):.6f}")
        print(f"  Impedance ratio Hy/Ez: {np.abs(Hy_mode[idx_max]/Ez_mode[idx_max]):.6f}")
        print(f"  Phase diff: {np.angle(Hy_mode[idx_max]/Ez_mode[idx_max])*180/np.pi:.2f} degrees")

        # Simple multi-lobe guard: warn if more than one local maximum in |Ez|
        mag = np.abs(Ez_mode)
        if mag.size >= 3:
            peak_mask = (mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])
            num_peaks = int(np.count_nonzero(peak_mask))
            if num_peaks > 1:
                print(f"[ModeSource] Warning: Detected {num_peaks} transverse peaks in |Ez|; consider narrowing source width to avoid exciting higher modes.")
        
        # Huygens surface currents for +x propagation:
        # J_z at Ez positions needs H_y sampled at Ez y-coords (already aligned)
        # M_y at Hy positions needs E_z sampled at Hy y-coords (interpolate)
        jz_profile = Hy_mode.copy()  # same y grid as Ez_mode/y_ez_coords
        # Interpolate Ez onto Hy y-positions to respect Yee staggering
        if y_hy_end > y_hy_start:
            my_profile = np.interp(y_hy_coords, y_ez_coords, np.real(Ez_mode))
        else:
            my_profile = np.array([], dtype=float)

        # Apply Hann window across transverse extent to minimize truncation ripples
        if jz_profile.size > 1:
            w_ez = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(jz_profile.size) / (jz_profile.size - 1)))
            jz_profile = np.real(jz_profile) * w_ez
        if my_profile.size > 1:
            w_hy = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(my_profile.size) / (my_profile.size - 1)))
            my_profile = np.real(my_profile) * w_hy
        
        # For -x propagation, flip signs
        if self.direction == "-x":
            jz_profile = -jz_profile
            my_profile = -my_profile
        
        # Store the profiles (as real-valued after phase alignment/interp)
        self._jz_profile = np.asarray(np.real(jz_profile), dtype=np.float64)
        self._my_profile = np.asarray(np.real(my_profile), dtype=np.float64)
        
        # Store grid indices for injection
        self._ez_indices = (slice(y_ez_start, y_ez_end), x_ez_idx)
        self._hy_indices = (slice(y_hy_start, y_hy_end), x_hy_idx)
        
        x_hy_coord = x_hy_idx * dx
        print(f"[ModeSource] Initialized at x={x_ez_coord/µm:.3f}µm, neff={self._neff:.4f}")
        print(f"[ModeSource] Direction: {self.direction}")
        print(f"[ModeSource] J_z at Ez[{y_ez_start}:{y_ez_end}, {x_ez_idx}] (x={x_ez_coord/µm:.3f}µm)")
        print(f"[ModeSource] M_y at Hy[{y_hy_start}:{y_hy_end}, {x_hy_idx}] (x={x_hy_coord/µm:.3f}µm)")
        print(f"[ModeSource] Spatial offset: {(x_hy_coord - x_ez_coord)/µm:.3f}µm = {(x_hy_idx - x_ez_idx)} cells")
        
        # Plot mode profile for debugging
        self._plot_mode_profile(y_ez_coords, self._jz_profile, self._my_profile)
        
    def _enforce_propagation_direction(self, E, H, axis):
        """Ensure the mode propagates in the correct direction by checking Poynting vector."""
        S = np.cross(E, np.conjugate(H), axis=0)
        power = float(np.real(np.sum(S[axis])))
        
        direction_sign = 1.0 if self.direction.startswith("+") else -1.0
        
        # If power has wrong sign, flip H to reverse propagation
        if power * direction_sign < 0:
            H = -H
            
        return E, H
    
    def _phase_align(self, field):
        """Align phase so field is mostly real at the peak amplitude."""
        idx_max = np.argmax(np.abs(field))
        phase = np.angle(field[idx_max])
        return field * np.exp(-1j * phase)
    
    def _plot_mode_profile(self, y_coords, jz_profile, my_profile):
        """Plot the mode profile for debugging."""
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot J_z (H_y)
            ax1.plot(y_coords/µm, np.real(jz_profile), 'b-', label='Real(J_z) = Real(H_y)')
            ax1.plot(y_coords/µm, np.imag(jz_profile), 'b--', label='Imag(J_z) = Imag(H_y)')
            ax1.set_xlabel('y (µm)')
            ax1.set_ylabel('J_z amplitude')
            ax1.set_title(f'Electric Current J_z = H_y (neff={self._neff:.4f})')
            ax1.legend()
            ax1.grid(True)
            
            # Plot M_y (E_z)
            ax2.plot(y_coords/µm, np.real(my_profile), 'r-', label='Real(M_y) = Real(E_z)')
            ax2.plot(y_coords/µm, np.imag(my_profile), 'r--', label='Imag(M_y) = Imag(E_z)')
            ax2.set_xlabel('y (µm)')
            ax2.set_ylabel('M_y amplitude')
            ax2.set_title(f'Magnetic Current M_y = E_z (dir={self.direction})')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('/tmp/mode_profile.png', dpi=150, bbox_inches='tight')
            print(f"[ModeSource] Mode profile saved to /tmp/mode_profile.png")
            plt.close()
        except Exception as e:
            print(f"[ModeSource] Could not plot mode profile: {e}")
    
    def get_source_terms(self, fields, t, dt, current_step, resolution, design):
        """Return the source terms to be added to the FDTD update equations.
        
        Returns:
            source_j: dict with electric current terms {component: (array, indices)}
            source_m: dict with magnetic current terms {component: (array, indices)}
        """
        if self._jz_profile is None:
            permittivity = design.rasterize(resolution=resolution).permittivity
            self.initialize(permittivity, resolution)
        
        source_j = {}
        source_m = {}
        
        # Get the temporal modulation (already includes any carrier; don't add extra exp(iωt))
        signal_value_e = self.signal[current_step] if current_step < len(self.signal) else 0.0
        # Leapfrog staggering: H is at half-steps; approximate s(t+dt/2) by linear interpolation
        if current_step + 1 < len(self.signal):
            signal_value_h = 0.5 * (self.signal[current_step] + self.signal[current_step + 1])
        else:
            signal_value_h = self.signal[current_step] if current_step < len(self.signal) else 0.0
        
        # Add J_z source (subtract in Ez update: ∂_t E_z = (1/ε)[curl H - J_z])
        jz_current = self._jz_profile * signal_value_e
        jz_current = -jz_current / resolution  # Negative sign for subtraction, normalize by cell size
        source_j["Ez"] = (jz_current.astype(np.float64), self._ez_indices)
        
        # Add M_y source (affects Hy via: ∂_t H_y = (1/μ)[∂_x E_z - M_y])
        # curl_ey = -∂E_z/∂x, so adding +M_y gives: H_y = H_y - dt/μ*(-∂E_z/∂x + M_y) = H_y + dt/μ*∂E_z/∂x - dt/μ*M_y ✓
        my_current = self._my_profile * signal_value_h
        my_current = my_current / resolution  # Positive M_y, will be subtracted via advance_h_field
        source_m["Hy"] = (my_current.astype(np.float64), self._hy_indices)
        
        # Diagnostics: estimate Poynting ratio (first 50 steps)
        try:
            if current_step < 50:
                # Determine adjacent Ez columns for left/right slabs
                y_ez_slice, x_ez = self._ez_indices
                y_hy_slice, x_hy = self._hy_indices
                ny_ez = y_ez_slice.stop - y_ez_slice.start
                ny_hy = y_hy_slice.stop - y_hy_slice.start
                n = max(0, min(ny_ez - 1, ny_hy))
                if n > 0:
                    # Right slab uses Ez at x_ez+1 (if exists) and Hy at x_hy
                    if x_ez + 1 < fields.Ez.shape[1]:
                        Ez_r = fields.Ez[y_ez_slice.start:y_ez_slice.start + n, x_ez + 1]
                        Hy_r = fields.Hy[y_hy_slice.start:y_hy_slice.start + n, x_hy]
                        P_r = float(np.sum(np.real(Ez_r * Hy_r)))
                    else:
                        P_r = 0.0
                    # Left slab uses Ez at x_ez-1 (if exists) and Hy at x_hy (same Hy column approximates)
                    if x_ez - 1 >= 0:
                        Ez_l = fields.Ez[y_ez_slice.start:y_ez_slice.start + n, x_ez - 1]
                        Hy_l = fields.Hy[y_hy_slice.start:y_hy_slice.start + n, x_hy]
                        P_l = float(np.sum(np.real(Ez_l * Hy_l)))
                    else:
                        P_l = 0.0
                    ratio = (abs(P_r) / (abs(P_l) + 1e-18)) if (abs(P_r) + abs(P_l)) > 0 else 0.0
                    print(f"[ModeSource] Poynting ratio (right/left) ≈ {ratio:.2e} at step {current_step}")
        except Exception as _:
            pass

        return source_j, source_m

