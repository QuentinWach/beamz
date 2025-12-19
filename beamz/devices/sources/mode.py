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
        
        # TE profiles
        self._mz_profile = None
        self._jy_profile = None
        self._jx_profile = None
        
        self._ez_indices = None
        self._h_indices = None
        
        # TE indices
        self._hz_indices = None
        self._e_indices = None # For Ex/Ey
        
        self._h_component = None
        self._e_component = None # For TE (Ex/Ey)
        self._neff = None
        self._dt_physical = 0.0
        
    def initialize(self, permittivity, resolution):
        """Compute the mode and set up the source currents."""
        dx = dy = resolution
        is_3d = (permittivity.ndim == 3)
        
        if is_3d:
            nz, ny, nx = permittivity.shape
            dz = resolution
        else:
            ny, nx = permittivity.shape
            nz = 1
            
        axis = "x" if self.direction in ("+x", "-x") else "y"
        self._dt_physical = 0.0
        
        # 1. Setup indices and profiles
        if axis == "x":
            center_idx = int(np.clip(np.round(self.center[0] / dx - 0.5), 0, nx - 1))
            
            # TFSF Offset
            if self.direction == "+x": offset_idx = max(0, center_idx - 1)
            else: offset_idx = min(nx - 2, center_idx)
            
            # For 3D, take a 2D slice for the mode solver
            if is_3d:
                eps_profile = permittivity[:, :, center_idx]
                z_slice_ez = slice(0, nz - 1)
                z_slice_h = slice(0, nz - 1)
                y_slice = slice(0, ny)
            else:
                eps_profile = permittivity[:, center_idx]
                z_slice_ez = z_slice_h = None # Not used in 2D indexing
                y_slice = slice(0, ny)

            if self.pol == "tm":
                self._ez_indices = (z_slice_ez, y_slice, center_idx) if is_3d else (y_slice, center_idx)
                self._h_indices = (z_slice_h, y_slice if is_3d else slice(0, ny), offset_idx)
                if is_3d:
                    # In 3D, for TM x-prop, we inject into Hy (transverse H). Hx is normal.
                    # Wait, for x-prop, Hx is longitudinal. Hy and Hz are transverse.
                    # TM mode has Ez and Hy (transverse).
                    self._h_indices = (z_slice_h, y_slice, offset_idx)
                    self._h_component = "Hy"
                else:
                    self._h_component = "Hx" # In 2D xy, Hx is the transverse H for x-prop TM.
                
                plot_len = len(eps_profile) - 1 if is_3d else len(eps_profile)
                plot_coords = (np.arange(plot_len) + 0.5) * (dz if is_3d else dy)
            else: # TE
                # Hz (scalar) at center_idx, Ey (transverse) at offset_idx
                if self.direction == "+x": hz_col = max(0, offset_idx - 1)
                else: hz_col = min(nx - 2, offset_idx)
                
                # Adjust shapes for staggered grid
                ny_eff = ny - 1
                nz_eff = nz - 1 if is_3d else 1
                
                self._hz_indices = (slice(0, nz) if is_3d else None, slice(0, ny_eff), hz_col)
                if not is_3d: self._hz_indices = self._hz_indices[1:]
                
                self._e_indices = (slice(0, nz) if is_3d else None, slice(0, ny_eff), offset_idx)
                if not is_3d: self._e_indices = self._e_indices[1:]
                
                self._e_component = "Ey"
                plot_len = len(eps_profile) if is_3d else len(eps_profile) - 1
                plot_coords = (np.arange(plot_len) + (0.5 if is_3d else 1.0)) * (dz if is_3d else dy)
                
        else: # axis == "y"
            center_idx = int(np.clip(np.round(self.center[1] / dy - 0.5), 0, ny - 1))
            
            # TFSF Offset
            if self.direction == "+y": offset_idx = max(0, center_idx - 1)
            else: offset_idx = min(ny - 2, center_idx)
            
            # For 3D, take a 2D slice
            if is_3d:
                eps_profile = permittivity[:, center_idx, :]
                z_slice_ez = slice(0, nz - 1)
                z_slice_h = slice(0, nz - 1)
                x_slice = slice(0, nx)
            else:
                eps_profile = permittivity[center_idx, :]
                z_slice_ez = z_slice_h = None
                x_slice = slice(0, nx)

            if self.pol == "tm":
                self._ez_indices = (z_slice_ez, center_idx, x_slice) if is_3d else (center_idx, x_slice)
                self._h_indices = (z_slice_h, offset_idx, x_slice) if is_3d else (offset_idx, x_slice)
                if is_3d:
                    # In 3D, for TM y-prop, we inject into Hx (transverse H). Hy is longitudinal.
                    self._h_component = "Hx"
                else:
                    self._h_component = "Hy" # In 2D xy, Hy is the transverse H for y-prop TM.
                
                plot_len = len(eps_profile) - 1 if is_3d else len(eps_profile)
                plot_coords = (np.arange(plot_len) + 0.5) * (dz if is_3d else dx)
            else: # TE
                # Hz (scalar) at center_idx, Ex (transverse) at offset_idx
                if self.direction == "+y": hz_row = max(0, offset_idx - 1)
                else: hz_row = min(ny - 2, offset_idx)
                
                nx_eff = nx - 1
                nz_eff = nz - 1 if is_3d else 1
                
                self._hz_indices = (slice(0, nz) if is_3d else None, hz_row, slice(0, nx_eff))
                if not is_3d: self._hz_indices = self._hz_indices[1:]
                
                self._e_indices = (slice(0, nz) if is_3d else None, offset_idx, slice(0, nx_eff))
                if not is_3d: self._e_indices = self._e_indices[1:]
                
                self._e_component = "Ex"
                plot_len = len(eps_profile) if is_3d else len(eps_profile) - 1
                plot_coords = (np.arange(plot_len) + (0.5 if is_3d else 1.0)) * (dz if is_3d else dx)

        # 2. Solve Modes
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        dL = dz if is_3d else (dy if axis == "x" else dx)
        neff_val, e_fields, h_fields, _ = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=dL,
            m=1,
            direction=self.direction,
            filter_pol=self.pol,
            return_fields=True
        )
        self._neff = neff_val[0]
        H_mode = h_fields[0]
        E_mode = e_fields[0]
        
        # 2.5. Compute physical time shift between E and H injection planes based on Yee-grid locations.
        if self._neff is not None:
            coord_e = 0.0
            coord_h = 0.0
            if axis == "x":
                if self.pol == "tm":
                    # Ez at x = (i + 0.5)dx, Hx at x = (i + 1.0)dx
                    idx_e = self._ez_indices[2] if is_3d else self._ez_indices[1]
                    idx_h = self._h_indices[2] if is_3d else self._h_indices[1]
                    coord_e = (idx_e + 0.5) * dx
                    coord_h = (idx_h + 1.0) * dx
                else:
                    # Ey at x = (i + 0.5)dx, Hz at x = (i + 1.0)dx
                    idx_e = self._e_indices[2] if is_3d else self._e_indices[1]
                    idx_h = self._hz_indices[2] if is_3d else self._hz_indices[1]
                    coord_e = (idx_e + 0.5) * dx
                    coord_h = (idx_h + 1.0) * dx
            else: # axis == "y"
                if self.pol == "tm":
                    # Ez at y = (j + 0.5)dy, Hy at y = (j + 1.0)dy
                    idx_e = self._ez_indices[1] if is_3d else self._ez_indices[0]
                    idx_h = self._h_indices[1] if is_3d else self._h_indices[0]
                    coord_e = (idx_e + 0.5) * dy
                    coord_h = (idx_h + 1.0) * dy
                else:
                    # Ex at y = (j + 0.5)dy, Hz at y = (j + 1.0)dy
                    idx_e = self._e_indices[1] if is_3d else self._e_indices[0]
                    idx_h = self._hz_indices[1] if is_3d else self._hz_indices[0]
                    coord_e = (idx_e + 0.5) * dy
                    coord_h = (idx_h + 1.0) * dy
            self._dt_physical = (coord_e - coord_h) * float(np.real(self._neff)) / LIGHT_SPEED
        
        # 3. Extract Fields & Profiles
        if axis == "x":
            if self.pol == "tm":
                Hy_raw = np.squeeze(H_mode[1])
                Ez_raw = np.squeeze(E_mode[2])
                if np.max(np.abs(Hy_raw)) < 1e-9: Hy_raw = np.squeeze(H_mode[2])
                if np.max(np.abs(Ez_raw)) < 1e-9: Ez_raw = np.squeeze(E_mode[1])
                
                # Phase align
                idx_max = np.argmax(np.abs(Hy_raw))
                phase_ref = np.angle(Hy_raw.flatten()[idx_max])
                Hy_profile = Hy_raw * np.exp(-1j * phase_ref)
                Ez_profile = Ez_raw * np.exp(-1j * phase_ref)
                
                # Stagger if needed for 3D
                if is_3d:
                    # In 3D, for x-prop TM, Ez and Hy are both at z-1/2
                    # Mode solver along Z gives profiles at integer z
                    # Hy_profile is (nz, ny). We stagger along Z (axis 0).
                    if Hy_profile.shape[0] > 1:
                        Hy_profile = 0.5 * (Hy_profile[:-1, :] + Hy_profile[1:, :])
                        Ez_profile = 0.5 * (Ez_profile[:-1, :] + Ez_profile[1:, :])
                
                # Impedance correction
                ETA_0 = np.sqrt(MU_0 / EPS_0)
                Z_phys = ETA_0 / np.real(self._neff)
                norm_h, norm_e = np.max(np.abs(Hy_profile)), np.max(np.abs(Ez_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ez_profile *= corr
                
                # Physical currents
                dir_sign = 1.0 if self.direction == "+x" else -1.0
                self._jz_profile = dir_sign * np.real(Hy_profile)
                self._my_profile = dir_sign * np.real(Ez_profile)
                
                plot_vals = (self._jz_profile, self._my_profile)
                
            else: # TE
                h_candidates = [np.squeeze(H_mode[i]) for i in range(3)]
                e_candidates = [np.squeeze(E_mode[i]) for i in range(3)]
                h_scores = [float(np.max(np.abs(hc))) for hc in h_candidates]
                e_scores = [float(np.max(np.abs(ec))) for ec in e_candidates]
                Hz_raw = h_candidates[int(np.argmax(h_scores))]
                Ey_raw = e_candidates[int(np.argmax(e_scores))]
                
                # Interpolate to staggered grid if needed
                if is_3d:
                    # In 3D, for x-prop TE, Ey and Hz are both at integer z
                    # but they are at y-1/2.
                    # Hz_raw is (nz, ny). We stagger along Y (axis 1).
                    Hz_profile_raw = 0.5 * (Hz_raw[:, :-1] + Hz_raw[:, 1:])
                    Ey_profile_raw = 0.5 * (Ey_raw[:, :-1] + Ey_raw[:, 1:])
                else:
                    # In 2D, Ey and Hz are at y-1/2, so we stagger from integer y
                    Hz_profile_raw = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
                    Ey_profile_raw = 0.5 * (Ey_raw[:-1] + Ey_raw[1:])
                
                # Phase align
                idx_max = np.argmax(np.abs(Hz_profile_raw))
                phase_ref = np.angle(Hz_profile_raw.flatten()[idx_max])
                Hz_profile = Hz_profile_raw * np.exp(-1j * phase_ref)
                Ey_profile = Ey_profile_raw * np.exp(-1j * phase_ref)
                
                ETA_0 = np.sqrt(MU_0 / EPS_0)
                Z_phys = ETA_0 / np.real(self._neff)
                norm_h, norm_e = np.max(np.abs(Hz_profile)), np.max(np.abs(Ey_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ey_profile *= corr
                
                if self.direction == "+x":
                    self._jy_profile = np.real(Hz_profile)
                    self._mz_profile = np.real(Ey_profile)
                else:
                    self._jy_profile = -np.real(Hz_profile)
                    self._mz_profile = -np.real(Ey_profile)
                
                plot_vals = (self._jy_profile, self._mz_profile)
                
        else: # axis == "y"
            if self.pol == "tm":
                Hx_raw = np.squeeze(H_mode[1])
                Ez_raw = np.squeeze(E_mode[2])
                if np.max(np.abs(Hx_raw)) < 1e-9: Hx_raw = np.squeeze(H_mode[2])
                if np.max(np.abs(Ez_raw)) < 1e-9: Ez_raw = np.squeeze(E_mode[1])
                
                idx_max = np.argmax(np.abs(Hx_raw))
                phase_ref = np.angle(Hx_raw.flatten()[idx_max])
                Hx_profile = Hx_raw * np.exp(-1j * phase_ref)
                Ez_profile = Ez_raw * np.exp(-1j * phase_ref)
                
                # Stagger if needed for 3D
                if is_3d:
                    # In 3D, for y-prop TM, Ez and Hx are both at z-1/2
                    # Hx_profile is (nz, nx). We stagger along Z (axis 0).
                    if Hx_profile.shape[0] > 1:
                        Hx_profile = 0.5 * (Hx_profile[:-1, :] + Hx_profile[1:, :])
                        Ez_profile = 0.5 * (Ez_profile[:-1, :] + Ez_profile[1:, :])
                
                ETA_0 = np.sqrt(MU_0 / EPS_0)
                Z_phys = ETA_0 / np.real(self._neff)
                norm_h, norm_e = np.max(np.abs(Hx_profile)), np.max(np.abs(Ez_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ez_profile *= corr
                
                if self.direction == "+y":
                    self._jz_profile = -np.real(Hx_profile)
                    self._my_profile = np.real(Ez_profile)
                else:
                    self._jz_profile = np.real(Hx_profile)
                    self._my_profile = -np.real(Ez_profile)
                
                plot_vals = (self._jz_profile, self._my_profile)
                
            else: # TE y-prop
                h_candidates = [np.squeeze(H_mode[i]) for i in range(3)]
                e_candidates = [np.squeeze(E_mode[i]) for i in range(3)]
                h_scores = [float(np.max(np.abs(hc))) for hc in h_candidates]
                e_scores = [float(np.max(np.abs(ec))) for ec in e_candidates]
                Hz_raw = h_candidates[int(np.argmax(h_scores))]
                Ex_raw = e_candidates[int(np.argmax(e_scores))]
                
                # Interpolate to staggered grid if needed
                if is_3d:
                    # In 3D, for y-prop TE, Ex and Hz are both at integer z
                    # but they are at x-1/2.
                    # Hz_raw is (nz, nx). We stagger along X (axis 1).
                    Hz_profile_raw = 0.5 * (Hz_raw[:, :-1] + Hz_raw[:, 1:])
                    Ex_profile_raw = 0.5 * (Ex_raw[:, :-1] + Ex_raw[:, 1:])
                else:
                    # In 2D, Ex and Hz are at x-1/2 (Wait, x-prop? No, y-prop. So staggered in X).
                    # But solver was along X. So stagger X.
                    Hz_profile_raw = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
                    Ex_profile_raw = 0.5 * (Ex_raw[:-1] + Ex_raw[1:])
                
                idx_max = np.argmax(np.abs(Hz_profile_raw))
                phase_ref = np.angle(Hz_profile_raw.flatten()[idx_max])
                Hz_profile = Hz_profile_raw * np.exp(-1j * phase_ref)
                Ex_profile = Ex_profile_raw * np.exp(-1j * phase_ref)
                
                ETA_0 = np.sqrt(MU_0 / EPS_0)
                Z_phys = ETA_0 / np.real(self._neff)
                norm_h, norm_e = np.max(np.abs(Hz_profile)), np.max(np.abs(Ex_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ex_profile *= corr
                
                dir_sign = 1.0 if self.direction == "+y" else -1.0
                self._jx_profile = dir_sign * np.real(Hz_profile)
                self._mz_profile = dir_sign * np.real(Ex_profile)
                
                plot_vals = (self._jx_profile, self._mz_profile)

        if self.width < 2.0 * µm:
            print("[ModeSource] Note: Source injection extended to full transverse span.")
        
        self._plot_mode_profile(plot_coords, *plot_vals)


        
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
    
    def show(self, field=None):
        """Visualize the 2D mode profile (for 3D simulations) or 1D profile (for 2D).
        
        Args:
            field (str, optional): Which field to show. 
                For TM: 'Ez' (default) or 'Hy'/'Hx'
                For TE: 'Ey'/'Ex' (default) or 'Hz'
        """
        import matplotlib.pyplot as plt
        from beamz.const import µm
        
        if self._jz_profile is None and self._mz_profile is None:
            if self.grid is not None and hasattr(self.grid, 'permittivity'):
                # Try to auto-initialize if grid is available
                # We need a resolution. Try to guess from grid or use a default.
                res = getattr(self.grid, 'resolution', 0.05e-6)
                self.initialize(self.grid.permittivity, res)
            else:
                print("[ModeSource] Source not initialized. Call Simulation or initialize manually.")
                return

        is_3d = self._jz_profile.ndim > 1 if self._jz_profile is not None else self._mz_profile.ndim > 1
        
        if self.pol == "tm":
            main_field = "Ez" if field is None else field
            if main_field in ["Ez", "Ez_profile"]:
                profile = self._my_profile # My corresponds to Ez in TM injection
                title = "Ez (transverse E)"
            else:
                profile = self._jz_profile # Jz corresponds to Hy/Hx
                title = f"{self._h_component} (transverse H)"
        else: # TE
            main_field = self._e_component if field is None else field
            if main_field in ["Ex", "Ey", "Ex_profile", "Ey_profile"]:
                profile = self._mz_profile # Mz corresponds to Ey/Ex in TE injection
                title = f"{self._e_component} (transverse E)"
            else:
                profile = self._jy_profile if hasattr(self, '_jy_profile') else self._jx_profile
                title = "Hz (transverse H)"

        # Strip any extra dimensions used for broadcasting
        if profile.ndim > 2:
            profile = np.squeeze(profile)
        elif profile.ndim == 2 and (profile.shape[0] == 1 or profile.shape[1] == 1):
            # Still 1D essentially
            profile = np.squeeze(profile)

        plt.figure(figsize=(8, 6))
        if profile.ndim == 2:
            # 2D visualization for 3D simulation
            im = plt.imshow(np.abs(profile), origin='lower', cmap='magma', aspect='auto')
            plt.colorbar(im, label='Absolute Amplitude')
            plt.title(f"Mode Source 2D Profile: {title} (neff={self._neff:.4f})")
            
            # Label axes based on propagation direction
            if self.direction in ["+x", "-x"]:
                plt.xlabel("Y-axis")
                plt.ylabel("Z-axis")
            else:
                plt.xlabel("X-axis")
                plt.ylabel("Z-axis")
        else:
            # 1D visualization for 2D simulation
            plt.plot(np.abs(profile), 'k-')
            plt.title(f"Mode Source 1D Profile: {title} (neff={self._neff:.4f})")
            plt.xlabel("Transverse Coordinate (cells)")
            plt.ylabel("Absolute Amplitude")
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()

    def _plot_mode_profile(self, coords, jz_profile, my_profile):
        """Plot the mode profile for debugging."""
        try:
            import matplotlib.pyplot as plt
            from beamz.const import µm
            
            # If 2D (3D simulation), we handle it differently
            if jz_profile.ndim > 1 or my_profile.ndim > 1:
                # Just save a summary image for debugging
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # J profile
                im1 = ax1.imshow(np.abs(jz_profile), origin='lower', cmap='magma', aspect='auto')
                plt.colorbar(im1, ax=ax1)
                ax1.set_title(f'Transverse H (J-current)')
                
                # M profile
                im2 = ax2.imshow(np.abs(my_profile), origin='lower', cmap='magma', aspect='auto')
                plt.colorbar(im2, ax=ax2)
                ax2.set_title(f'Transverse E (M-current)')
                
                plt.tight_layout()
                plt.savefig("mode_profile.png", dpi=150, bbox_inches='tight')
                print(f"[ModeSource] 2D Mode profiles saved to mode_profile.png")
                plt.close()
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot J_z (H_y)
            ax1.plot(coords/µm, np.real(jz_profile), 'b-', label='Real(J)')
            ax1.plot(coords/µm, np.imag(jz_profile), 'b--', label='Imag(J)')
            ax1.set_xlabel('coord (µm)')
            ax1.set_ylabel('J amplitude')
            ax1.set_title(f'J current (neff={self._neff:.4f})')
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
            ax2.set_title(f'M current (dir={self.direction})')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig("mode_profile.png", dpi=150, bbox_inches='tight')
            print(f"[ModeSource] Mode profile saved to mode_profile.png")
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
        
        if self._jz_profile is None and self._mz_profile is None:
            permittivity = design.rasterize(resolution=resolution).permittivity
            self.initialize(permittivity, resolution)
        
        # Timing:
        # E source (J) is evaluated at t + 0.5*dt. H source (M) is evaluated at a shifted time to match plane offset.
        signal_value_e = self._get_signal_value(t + 0.5 * dt, dt)
        signal_value_h = self._get_signal_value(t + 0.5 * dt + self._dt_physical, dt)
        
        if self.pol == "tm":
            # TM Injection: Jz -> Ez, M -> Hx/Hy
            # Inject J_z source into Ez field
            eps_at_source = fields.permittivity[self._ez_indices]
            jz_term = self._jz_profile * signal_value_e / resolution
            ez_injection = -jz_term * dt / (EPS_0 * eps_at_source)
            fields.Ez[self._ez_indices] += ez_injection
            
            # Inject M source into H field
            if hasattr(fields, 'permeability'):
                mu_at_source = fields.permeability[self._h_indices]
            else:
                mu_at_source = 1.0
                
            my_term = self._my_profile * signal_value_h / resolution
            # Magnetic current enters H update with opposite sign to curl(E): ∂H/∂t = -(curlE + M)/μ.
            h_injection = -my_term * dt / (MU_0 * mu_at_source)
            
            if self._h_component == "Hx":
                fields.Hx[self._h_indices] += h_injection
            else:
                fields.Hy[self._h_indices] += h_injection
                
        else: # TE
            # TE Injection: J -> Ex/Ey, Mz -> Hz
            # Inject J source into Ex/Ey field
            eps_at_source = fields.permittivity[self._e_indices]
            
            if self._e_component == "Ex": j_profile = self._jx_profile
            else: j_profile = self._jy_profile
            
            j_term = j_profile * signal_value_e / resolution
            e_injection = -j_term * dt / (EPS_0 * eps_at_source)
            
            if self._e_component == "Ex": fields.Ex[self._e_indices] += e_injection
            else: fields.Ey[self._e_indices] += e_injection
            
            # Inject Mz source into Hz field
            if hasattr(fields, 'permeability'):
                mu_at_source = fields.permeability[self._hz_indices]
            else:
                mu_at_source = 1.0
                
            mz_term = self._mz_profile * signal_value_h / resolution
            hz_injection = -mz_term * dt / (MU_0 * mu_at_source)
            fields.Hz[self._hz_indices] += hz_injection
        
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

