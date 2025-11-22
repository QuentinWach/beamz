import numpy as np
from typing import Literal
from beamz.devices.sources.solve import solve_modes
from beamz.const import µm, LIGHT_SPEED, EPS_0, MU_0

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
        
        # For unidirectional +x propagation: M_y at Hx column to the LEFT of Ez column
        # Note: fields.Hx[i,j] is at x=(j+0.5)dx, y=idy. 
        # fields.Ez[i,j] is at x=(j+0.5)dx, y=(i+0.5)dy? No, usually Ez at integer or half?
        # In beamz ops.py: Hx from diff(Ez, axis=1). Hx is staggered in X relative to Ez?
        # Actually, Hx and Ez share the same Y-coordinate index logic in beamz (based on curl_ex shape).
        # But they are spatially staggered by 0.5 dx.
        # If we want M_y at x_ez - 0.5 dx, we use index x_ez - 1 for Hx?
        # Ez[j] is at x[j]. Hx[j-1] is at x[j]-0.5dx. 
        if self.direction == "+x":
            x_hx_idx = max(0, x_ez_idx - 1)  # One column to the left
        else:  # "-x"
            x_hx_idx = min(nx - 2, x_ez_idx)  # One column to the right? 
            # If Ez at j. Hx at j is at x[j]+0.5dx.
            # So for -x (source right of Ez), we use Hx at j.
            
        # Determine the y-extent of the source
        # Update: We now use the FULL grid height to ensure the mode solver sees the open boundary (cladding/PML)
        # and produces a correct eigenmode ("soft source").
        # The 'width' parameter is now only used for logical centering or visualization if needed.
        
        y_ez_start = 0
        y_ez_end = ny
        y_ez_coords = (np.arange(y_ez_start, y_ez_end) + 0.5) * dy
        
        # Sample permittivity at Ez positions (cell centers: i+1/2, j+1/2) for the FULL column
        eps_profile = permittivity[:, x_ez_idx]
        
        # Solve for the mode once
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        neff_val, e_fields, h_fields, prop_axis = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=dy,
            m=1,
            direction=self.direction,
            filter_pol=self.pol,
            return_fields=True
        )
        
        self._neff = neff_val[0]
        self._k = self._neff * 2 * np.pi / self.wavelength
        
        # Extract H_y (for J_z) and E_z (for M_y)
        # H_y is physical Hy component. In solve_modes output (Ex, Ez, Ey) -> (Hx, Hz, Hy) for axis 1?
        # solve_modes doc says: if prop_axis=1 (y): E=[Ex, Ez, Ey], H=[-Hx, -Hz, -Hy].
        # But we are solving 1D mode with propagation in x (axis 0 of 3D or just 1D solver).
        # solve_modes uses 1D epsilon. It returns H_y in h_fields[0][2] for TM?
        H_mode = h_fields[0]
        E_mode = e_fields[0]
        
        if self.pol == "te":
            Hy_mode = np.squeeze(H_mode[1])  # Hy
            Ez_mode = np.squeeze(E_mode[2])  # Ez
            # Fallback if needed
            if np.max(np.abs(Hy_mode)) < 1e-9: Hy_mode = np.squeeze(H_mode[2])
            if np.max(np.abs(Ez_mode)) < 1e-9: Ez_mode = np.squeeze(E_mode[1])
        elif self.pol == "tm":
            Hy_mode = np.squeeze(H_mode[2])  # Hz as Hy equivalent
            Ez_mode = np.squeeze(E_mode[1])  # Ey as Ez equivalent
        else:
            raise ValueError(f"Unknown polarization: {self.pol}")
            
        # Ensure consistent phase alignment
        # Find peak of Hy and align phase to 0
        idx_max = np.argmax(np.abs(Hy_mode))
        phase_ref = np.angle(Hy_mode[idx_max])
        Hy_mode = Hy_mode * np.exp(-1j * phase_ref)
        Ez_mode = Ez_mode * np.exp(-1j * phase_ref)
        
        # No interpolation needed for Ez_mode since Hx grid shares Y-coordinates with Ez grid!
        Ez_mode_at_hx = Ez_mode

        # Impedance matching correction
        # Calculate physical wave impedance for the mode: Z = eta0 / neff
        ETA_0 = np.sqrt(MU_0 / EPS_0)
        Z_phys = ETA_0 / np.real(self._neff)
        
        # Check current ratio
        norm_hy = np.max(np.abs(Hy_mode))
        norm_ez = np.max(np.abs(Ez_mode_at_hx))
            
        if norm_hy > 1e-12 and norm_ez > 1e-12:
            current_Z = norm_ez / norm_hy
            correction_factor = Z_phys / current_Z
            print(f"[ModeSource] Correcting impedance: Z_sim={current_Z:.2f}, Z_phys={Z_phys:.2f}. Scaling M_y by {correction_factor:.4f}")
            Ez_mode_at_hx *= correction_factor
        
        # Store profiles
        # Note: We do NOT apply Hann windowing here because we want the exact eigenmode 
        # which naturally decays to zero at the boundaries (if domain is large enough).
        jz_profile = np.real(Hy_mode).copy()
        my_profile = np.real(Ez_mode_at_hx).copy()
        
        # For +x propagation, Jz (Hy) and My (Ez) must have same sign to launch forward wave.
        # But mode solver returns them with opposite signs for +x flux.
        # So we flip Jz to align them for Huygens source.
        if self.direction == "+x":
            jz_profile = -jz_profile
            
        # For -x propagation, flip both signs to match direction
        # (Note: For -x, Hy and Ez are naturally aligned in sign, so they work as Huygens pair)
        if self.direction == "-x":
            jz_profile = -jz_profile
            my_profile = -my_profile
        
        # Store the profiles (as real-valued after phase alignment)
        self._jz_profile = np.asarray(np.real(jz_profile), dtype=np.float64)
        self._my_profile = np.asarray(np.real(my_profile), dtype=np.float64)
        
        # Store grid indices for injection
        self._ez_indices = (slice(y_ez_start, y_ez_end), x_ez_idx)
        # Hx indices (replaces Hy indices)
        self._hx_indices = (slice(y_ez_start, y_ez_end), x_hx_idx)
        
        x_hx_coord = x_hx_idx * dx # Approx
        print(f"[ModeSource] Initialized at x={x_ez_coord/µm:.3f}µm, neff={self._neff:.4f}")
        
        # Check for potential radiation issues if source width is much larger than typical single mode
        # Update: We now inject on the full grid, so width is only used for "center".
        # The user might be confused if they set a tiny width but we inject everywhere.
        # But this is the "Soft Source" fix.
        if self.width < 2.0 * µm: # If user tried to restrict it
             print(f"[ModeSource] Note: Source injection extended to full grid height to prevent mode truncation radiation.")
             
        print(f"[ModeSource] Direction: {self.direction}")
        print(f"[ModeSource] J_z at Ez[{y_ez_start}:{y_ez_end}, {x_ez_idx}] (x={x_ez_coord/µm:.3f}µm)")
        print(f"[ModeSource] M_y at Hx[{y_ez_start}:{y_ez_end}, {x_hx_idx}] (x~{x_hx_coord/µm:.3f}µm)")
        
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
            # Use correct y-coords for M_y (Hy positions) if different length
            if my_profile.size == y_coords.size:
                y_my_coords = y_coords
            else:
                # Reconstruct Hy coords
                start, end = self._hy_indices[0].start, self._hy_indices[0].stop
                dL = y_coords[1] - y_coords[0]
                y_my_coords = (np.arange(start, end) + 1.0) * dL
                
            ax2.plot(y_my_coords/µm, np.real(my_profile), 'r-', label='Real(M_y) = Real(E_z)')
            ax2.plot(y_my_coords/µm, np.imag(my_profile), 'r--', label='Imag(M_y) = Imag(E_z)')
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
    
    def _get_signal_value(self, time, dt):
        """Interpolate signal value at arbitrary time."""
        idx_float = float(time / dt)
        idx_low = int(np.floor(idx_float))
        idx_high = idx_low + 1
        frac = idx_float - idx_low
        
        if 0 <= idx_low < len(self.signal) - 1:
            return (1.0 - frac) * self.signal[idx_low] + frac * self.signal[idx_high]
        elif idx_low == len(self.signal) - 1:
            return self.signal[idx_low]
        else:
            return 0.0

    def inject(self, fields, t, dt, current_step, resolution, design):
        """Inject source fields directly into the simulation grid before the FDTD update step.
        
        Args:
            fields: The Fields object containing E and H arrays
            t: Current simulation time
            dt: Time step size
            current_step: Current time step index
            resolution: Spatial resolution (dx, dy)
            design: The simulation Design object
        """
        from beamz.const import EPS_0, MU_0
        
        if self._jz_profile is None:
            permittivity = design.rasterize(resolution=resolution).permittivity
            self.initialize(permittivity, resolution)
        
        # Simplified timing for TF/SF boundary condition
        # Both J and M act at the same effective boundary location.
        # J (driving E) is evaluated at t + 0.5*dt (half-step).
        # M (driving H) is evaluated at t (integer step).
        
        # J_z (Electric current) centered at t + 0.5*dt
        signal_value_e = self._get_signal_value(t + 0.5 * dt, dt)
        
        # M_y (Magnetic current) centered at t
        signal_value_h = self._get_signal_value(t, dt)
        
        # Inject J_z source into Ez field
        # Update equation: ∂_t E_z = (1/ε)[curl H - J_z]
        # Discrete update: E_z^(n+1) = E_z^n + (dt/ε) * [curl H - J_z]
        # We inject -J_z contribution directly: E_z_new = E_z_old - (dt/ε) * J_z
        
        # Get permittivity at injection location
        y_ez_slice, x_ez_idx = self._ez_indices
        eps_at_source = fields.permittivity[y_ez_slice, x_ez_idx]
        
        jz_term = self._jz_profile * signal_value_e / resolution
        # Note: epsilon is relative permittivity, so multiply by EPS_0
        ez_injection = -jz_term * dt / (EPS_0 * eps_at_source)
        
        fields.Ez[self._ez_indices] += ez_injection
        
        # Inject M_y source into Hx field (which corresponds to physical -Hy)
        # Update equation for physical H_y: ∂_t H_y = (1/μ)[∂_x E_z - M_y]
        # Beamz Hx is actually -H_y.
        # So ∂_t (-H_y) = - (1/μ)[∂_x E_z - M_y] = (1/μ)[-∂_x E_z + M_y]
        # Discrete update: Hx_new = Hx_old - coeff * curl_ex + coeff * M_y
        # We inject +M_y contribution directly: Hx_new = Hx_old + (dt/μ) * M_y
        
        # Get permeability at injection location (if available, otherwise assume 1.0)
        # Use Hx indices
        y_hx_slice, x_hx_idx = self._hx_indices
        if hasattr(fields, 'permeability'):
            mu_at_source = fields.permeability[y_hx_slice, x_hx_idx]
        else:
            mu_at_source = 1.0
            
        my_term = self._my_profile * signal_value_h / resolution
        # Note: mu is relative permeability, so multiply by MU_0
        # Sign is POSITIVE because we are updating -H_y with +M_y term
        hx_injection = +my_term * dt / (MU_0 * mu_at_source)
        
        fields.Hx[self._hx_indices] += hx_injection
        
        # Diagnostics: estimate Poynting ratio (first 50 steps)
        # try:
        #     if current_step < 50:
        #         # Determine adjacent Ez columns for left/right slabs
        #         y_ez_slice, x_ez = self._ez_indices
        #         # Hx indices are same y-slice as Ez
        #         y_hx_slice, x_hx = self._hx_indices
        #         
        #         ny = y_ez_slice.stop - y_ez_slice.start
        #         
        #         # Right probe (forward) - assume x_ez < x_ez+3
        #         xr = x_ez + 3
        #         if xr < fields.Ez.shape[1]:
        #             Ez_r = fields.Ez[y_ez_slice, xr][:ny]
        #             # Hx is at same y as Ez.
        #             # Sx = Ez * fields.Hx.
        #             # To get Hx at xr, average Hx[xr] and Hx[xr-1].
        #             if xr > 0:
        #                  Hx_r_avg = 0.5 * (fields.Hx[y_hx_slice, xr][:ny] + fields.Hx[y_hx_slice, xr-1][:ny])
        #             else:
        #                  Hx_r_avg = fields.Hx[y_hx_slice, xr][:ny]
        #             
        #             P_r = float(np.sum(np.real(Ez_r * Hx_r_avg)))
        #         else:
        #             P_r = 0.0
        #
        #         # Left probe (backward)
        #         xl = x_ez - 3
        #         if xl >= 0:
        #             Ez_l = fields.Ez[y_ez_slice, xl][:ny]
        #             if xl > 0:
        #                  Hx_l_avg = 0.5 * (fields.Hx[y_hx_slice, xl][:ny] + fields.Hx[y_hx_slice, xl-1][:ny])
        #             else:
        #                  Hx_l_avg = fields.Hx[y_hx_slice, xl][:ny]
        #             P_l = float(np.sum(np.real(Ez_l * Hx_l_avg)))
        #         else:
        #             P_l = 0.0
        #             
        #         # Ratio should be P_r / P_l. If P_l is small, ratio is huge.
        #         # But initially both are 0. Then P_r grows. P_l should stay small.
        #         # If P_l is negative (backward flux), we take abs?
        #         # Ratio of forward power to backward power.
        #         ratio = (abs(P_r) / (abs(P_l) + 1e-18)) if (abs(P_r) + abs(P_l)) > 1e-12 else 0.0
        #         print(f"[ModeSource] Poynting ratio (right/left) ≈ {ratio:.2e} at step {current_step}")
        # except Exception as _:
        #     pass

