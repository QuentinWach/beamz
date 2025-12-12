import numpy as np
from beamz.devices.sources.solve import solve_modes
from beamz.const import µm, LIGHT_SPEED, EPS_0, MU_0

class ModeSource:
    """Huygens mode source on Yee grid supporting ±x/±y propagation."""
    
    def __init__(self, grid, center, width, wavelength, pol, signal, direction="+x"):
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
        self._h_indices = None
        self._h_component = None
        self._neff = None
        
    def initialize(self, permittivity, resolution):
        """Compute the mode and set up the source currents."""
        dx = dy = resolution
        ny, nx = permittivity.shape
        axis = "x" if self.direction in ("+x", "-x") else "y"
        # For y propagation only: Forward (+) directions use j = -H_t, m = +E_t; reverse flips both
        sign_map = {"+y": (-1.0, 1.0), "-y": (1.0, -1.0)}
        h_sign, m_sign = sign_map.get(self.direction, (-1.0, 1.0))
        
        if axis == "x":
            x_ez_idx = int(np.clip(np.round(self.center[0] / dx - 0.5), 0, nx - 1))
            x_ez_coord = (x_ez_idx + 0.5) * dx
            y_ez_slice = slice(0, ny)
            y_coords = (np.arange(ny) + 0.5) * dy
            eps_profile = permittivity[:, x_ez_idx]
            if self.direction == "+x": x_h_idx = max(0, x_ez_idx - 1)
            else: x_h_idx = min(nx - 2, x_ez_idx)
            self._ez_indices = (y_ez_slice, x_ez_idx)
            self._h_indices = (y_ez_slice, x_h_idx)
            self._h_component = "Hx"
            h_coord = x_h_idx * dx
            print(f"[ModeSource] Direction: {self.direction}, Ez column {x_ez_idx}, Hx column {x_h_idx}")
        else:
            y_ez_idx = int(np.clip(np.round(self.center[1] / dy - 0.5), 0, ny - 1))
            y_ez_coord = (y_ez_idx + 0.5) * dy
            x_ez_slice = slice(0, nx)
            x_coords = (np.arange(nx) + 0.5) * dx
            eps_profile = permittivity[y_ez_idx, :]
            if self.direction == "+y": y_h_idx = max(0, y_ez_idx - 1)
            else: y_h_idx = min(ny - 2, y_ez_idx)
            self._ez_indices = (y_ez_idx, x_ez_slice)
            # For y propagation: Following TFSF boundary pattern, need row offset like x propagation has column offset
            # For +y: use row above (y_ez_idx - 1), for -y: use same row but bound to ny-2
            # Use Hy (staggered in y) to create proper TFSF boundary, similar to how x propagation uses Hx offset
            # #region agent log
            import json
            with open('/Users/quentinwach/Code/beamz/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"fix7","hypothesisId":"F","location":"mode.py:54","message":"y propagation: using Hy with row offset for TFSF boundary","data":{"direction":self.direction,"y_ez_idx":int(y_ez_idx),"y_h_idx":int(y_h_idx),"ny":int(ny),"nx":int(nx)},"timestamp":int(__import__('time').time()*1000)})+"\n")
            # #endregion
            self._h_indices = (y_h_idx, x_ez_slice)
            self._h_component = "Hy"
            h_coord = (y_h_idx + 0.5) * dy
            print(f"[ModeSource] Direction: {self.direction}, Ez row {y_ez_idx}, Hy row {y_h_idx}")
        
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        dL = dy if axis == "x" else dx
        neff_val, e_fields, h_fields, prop_axis = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=dL,
            m=1,
            direction=self.direction,
            filter_pol=self.pol,
            return_fields=True
        )
        
        self._neff = neff_val[0]
        self._k = self._neff * 2 * np.pi / self.wavelength
        H_mode = h_fields[0]
        E_mode = e_fields[0]
        
        if axis == "x":
            # Original x propagation logic - extract fields exactly as original code
            if self.pol == "te":
                Hy_mode = np.squeeze(H_mode[1])  # Hy
                Ez_mode = np.squeeze(E_mode[2])  # Ez
                if np.max(np.abs(Hy_mode)) < 1e-9: Hy_mode = np.squeeze(H_mode[2])
                if np.max(np.abs(Ez_mode)) < 1e-9: Ez_mode = np.squeeze(E_mode[1])
            elif self.pol == "tm":
                Hy_mode = np.squeeze(H_mode[2])  # Hz as Hy equivalent
                Ez_mode = np.squeeze(E_mode[1])  # Ey as Ez equivalent
            else:
                raise ValueError(f"Unknown polarization: {self.pol}")
            
            # Phase align as original
            idx_max = np.argmax(np.abs(Hy_mode))
            phase_ref = np.angle(Hy_mode[idx_max])
            Hy_mode = Hy_mode * np.exp(-1j * phase_ref)
            Ez_mode = Ez_mode * np.exp(-1j * phase_ref)
            
            Ez_mode_at_h = Ez_mode
            h_t = Hy_mode
            e_t = Ez_mode
        else:
            # Y propagation logic - use generalized extraction
            hx_mode = np.squeeze(H_mode[1])
            hy_mode = np.squeeze(H_mode[2]) if H_mode.shape[0] > 2 else hx_mode * 0.0
            ez_mode = np.squeeze(E_mode[0])
            ex_mode = np.squeeze(E_mode[1]) if E_mode.shape[0] > 1 else ez_mode * 0.0
            ey_mode = np.squeeze(E_mode[2]) if E_mode.shape[0] > 2 else ez_mode * 0.0
            
            h_t = hx_mode if np.max(np.abs(hx_mode)) > 1e-12 else hy_mode
            
            if self.pol == "te":
                e_t = ez_mode if np.max(np.abs(ez_mode)) > 1e-12 else ey_mode
            elif self.pol == "tm":
                e_t = ex_mode
                if np.max(np.abs(e_t)) < 1e-12: e_t = ez_mode
            else:
                raise ValueError(f"Unknown polarization: {self.pol}")
            
            idx_max = np.argmax(np.abs(h_t))
            phase_ref = np.angle(h_t[idx_max])
            h_t = h_t * np.exp(-1j * phase_ref)
            e_t = e_t * np.exp(-1j * phase_ref)
            
            Ez_mode_at_h = e_t
        
        ETA_0 = np.sqrt(MU_0 / EPS_0)
        Z_phys = ETA_0 / np.real(self._neff)
        
        if axis == "x":
            # Original x propagation impedance correction
            norm_hy = np.max(np.abs(h_t))  # This is Hy_mode after phase alignment
            norm_ez = np.max(np.abs(Ez_mode_at_h))
            if norm_hy > 1e-12 and norm_ez > 1e-12:
                current_Z = norm_ez / norm_hy
                correction_factor = Z_phys / current_Z
                print(f"[ModeSource] Correcting impedance: Z_sim={current_Z:.2f}, Z_phys={Z_phys:.2f}. Scaling M_y by {correction_factor:.4f}")
                Ez_mode_at_h *= correction_factor
        else:
            # Y propagation impedance correction
            norm_h = np.max(np.abs(h_t))
            norm_e = np.max(np.abs(Ez_mode_at_h))
            if norm_h > 1e-12 and norm_e > 1e-12:
                current_Z = norm_e / norm_h
                correction_factor = Z_phys / current_Z
                print(f"[ModeSource] Correcting impedance: Z_sim={current_Z:.2f}, Z_phys={Z_phys:.2f}. Scaling M by {correction_factor:.4f}")
                Ez_mode_at_h *= correction_factor
        
        # Apply signs - for x propagation, use original explicit flip logic
        # For y propagation, use sign_map
        if axis == "x":
            # Original code: start with real parts, then flip
            jz_profile = np.real(h_t).copy()
            my_profile = np.real(Ez_mode_at_h).copy()
            # #region agent log
            import json
            with open('/Users/quentinwach/Code/beamz/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"direction_fix","hypothesisId":"A","location":"mode.py:160","message":"before sign flip","data":{"direction":self.direction,"jz_initial_sign":float(np.sign(jz_profile[0])) if len(jz_profile) > 0 else 0,"my_initial_sign":float(np.sign(my_profile[0])) if len(my_profile) > 0 else 0,"jz_initial_val":float(jz_profile[0]) if len(jz_profile) > 0 else 0,"my_initial_val":float(my_profile[0]) if len(my_profile) > 0 else 0},"timestamp":int(__import__('time').time()*1000)})+"\n")
            # #endregion
            # For +x: flip jz to align Jz and My (mode solver returns them opposite)
            # For -x: flip only My to reverse direction (keep Jz same as initial to get opposite propagation)
            if self.direction == "+x":
                jz_profile = -jz_profile
            elif self.direction == "-x":
                my_profile = -my_profile
            # #region agent log
            import json
            with open('/Users/quentinwach/Code/beamz/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"direction_fix","hypothesisId":"A","location":"mode.py:169","message":"after sign flip","data":{"direction":self.direction,"jz_final_sign":float(np.sign(jz_profile[0])) if len(jz_profile) > 0 else 0,"my_final_sign":float(np.sign(my_profile[0])) if len(my_profile) > 0 else 0,"jz_final_val":float(jz_profile[0]) if len(jz_profile) > 0 else 0,"my_final_val":float(my_profile[0]) if len(my_profile) > 0 else 0},"timestamp":int(__import__('time').time()*1000)})+"\n")
            # #endregion
        else:
            jz_profile = h_sign * np.real(h_t)
            my_profile = m_sign * np.real(Ez_mode_at_h)
        
        if axis == "y":
            # #region agent log
            import json
            with open('/Users/quentinwach/Code/beamz/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"fix7","hypothesisId":"B","location":"mode.py:130","message":"y propagation: profile lengths for Hy grid","data":{"jz_profile_len":len(jz_profile),"my_profile_len":len(my_profile),"nx":int(nx)},"timestamp":int(__import__('time').time()*1000)})+"\n")
            # #endregion
            # Align magnetic current to Hy grid (full x row, ny-1 rows)
            jz_profile = jz_profile[:nx]  # Ez uses full row (nx elements)
            my_profile = my_profile[:nx]  # Hy uses full x row (nx elements)
        
        self._jz_profile = np.asarray(np.real(jz_profile), dtype=np.float64)
        self._my_profile = np.asarray(np.real(my_profile), dtype=np.float64)
        
        if axis == "x":
            print(f"[ModeSource] x={x_ez_coord/µm:.3f}µm, neff={self._neff:.4f}")
            print(f"[ModeSource] J_z at Ez[:, {x_ez_idx}]")
            print(f"[ModeSource] M at Hx[:, {self._h_indices[1]}] (x~{h_coord/µm:.3f}µm)")
            plot_coords = y_coords
        else:
            print(f"[ModeSource] y={y_ez_coord/µm:.3f}µm, neff={self._neff:.4f}")
            print(f"[ModeSource] J_z at Ez[{self._ez_indices[0]}, :]")
            print(f"[ModeSource] M at Hy[{self._h_indices[0]}, :] (y~{h_coord/µm:.3f}µm)")
            plot_coords = x_coords
        
        if self.width < 2.0 * µm:
            print("[ModeSource] Note: Source injection extended to full transverse span.")
        
        self._plot_mode_profile(plot_coords, self._jz_profile, self._my_profile)
        
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
    
    def _plot_mode_profile(self, coords, jz_profile, my_profile):
        """Plot the mode profile for debugging."""
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot J_z (H_y)
            ax1.plot(coords/µm, np.real(jz_profile), 'b-', label='Real(J_z)')
            ax1.plot(coords/µm, np.imag(jz_profile), 'b--', label='Imag(J_z)')
            ax1.set_xlabel('coord (µm)')
            ax1.set_ylabel('J_z amplitude')
            ax1.set_title(f'Electric Current J_z = H_y (neff={self._neff:.4f})')
            ax1.legend()
            ax1.grid(True)
            
            # Plot M_y (E_z) - handle different profile lengths
            if len(my_profile) == len(coords):
                ax2.plot(coords/µm, np.real(my_profile), 'r-', label='Real(M)')
                ax2.plot(coords/µm, np.imag(my_profile), 'r--', label='Imag(M)')
            else:
                # If my_profile is shorter (e.g., for Hx), use subset of coords
                plot_coords = coords[:len(my_profile)]
                ax2.plot(plot_coords/µm, np.real(my_profile), 'r-', label='Real(M)')
                ax2.plot(plot_coords/µm, np.imag(my_profile), 'r--', label='Imag(M)')
            ax2.set_xlabel('coord (µm)')
            ax2.set_ylabel('M amplitude')
            ax2.set_title(f'Magnetic Current M (dir={self.direction})')
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
        """Inject source fields directly into the grid before the update step."""
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
        eps_at_source = fields.permittivity[self._ez_indices]
        
        jz_term = self._jz_profile * signal_value_e / resolution
        # Note: epsilon is relative permittivity, so multiply by EPS_0
        ez_injection = -jz_term * dt / (EPS_0 * eps_at_source)
        
        fields.Ez[self._ez_indices] += ez_injection
        
        if hasattr(fields, 'permeability'):
            mu_at_source = fields.permeability[self._h_indices]
        else:
            mu_at_source = 1.0
            
        my_term = self._my_profile * signal_value_h / resolution
        h_injection = +my_term * dt / (MU_0 * mu_at_source)
        
        # #region agent log
        import json
        with open('/Users/quentinwach/Code/beamz/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"fix7","hypothesisId":"F","location":"mode.py:259","message":"inject: injecting into H component","data":{"h_component":self._h_component,"h_indices":str(self._h_indices),"direction":self.direction,"jz_sign":float(np.sign(self._jz_profile[0])) if len(self._jz_profile) > 0 else 0,"my_sign":float(np.sign(self._my_profile[0])) if len(self._my_profile) > 0 else 0},"timestamp":int(__import__('time').time()*1000)})+"\n")
        # #endregion
        if self._h_component == "Hx":
            fields.Hx[self._h_indices] += h_injection
        else:
            fields.Hy[self._h_indices] += h_injection
        
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

