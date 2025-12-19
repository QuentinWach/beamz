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
        ny, nx = permittivity.shape
        axis = "x" if self.direction in ("+x", "-x") else "y"
        self._dt_physical = 0.0
        
        # 1. Setup indices and profiles
        if axis == "x":
            center_idx = int(np.clip(np.round(self.center[0] / dx - 0.5), 0, nx - 1))
            coord = (center_idx + 0.5) * dx
            eps_profile = permittivity[:, center_idx]
            
            # TFSF Offset
            if self.direction == "+x": offset_idx = max(0, center_idx - 1)
            else: offset_idx = min(nx - 2, center_idx)
            
            if self.pol == "tm":
                self._ez_indices = (slice(0, ny), center_idx)
                self._h_indices = (slice(0, ny), offset_idx)
                self._h_component = "Hx"
                print(f"[ModeSource] TM x-prop: Ez col {center_idx}, Hx col {offset_idx}")
                plot_coords = (np.arange(ny) + 0.5) * dy
            else: # TE
                # Hz (scalar) at center_idx, Ey (transverse) at offset_idx
                # Hz grid is (ny-1, nx-1). Ey grid is (ny-1, nx).
                # Choose Hz column so that Hz plane is one half-cell "upstream" of Ey plane.
                if self.direction == "+x": hz_col = max(0, offset_idx - 1)
                else: hz_col = min(nx - 2, offset_idx)
                
                self._hz_indices = (slice(0, ny-1), hz_col)
                self._e_indices = (slice(0, ny-1), offset_idx)
                self._e_component = "Ey"
                print(f"[ModeSource] TE x-prop: Hz col {hz_col}, Ey col {offset_idx}")
                plot_coords = (np.arange(ny-1) + 1.0) * dy # staggered y-coords
                
        else: # axis == "y"
            center_idx = int(np.clip(np.round(self.center[1] / dy - 0.5), 0, ny - 1))
            coord = (center_idx + 0.5) * dy
            eps_profile = permittivity[center_idx, :]
            
            # TFSF Offset
            if self.direction == "+y": offset_idx = max(0, center_idx - 1)
            else: offset_idx = min(ny - 2, center_idx)
            
            if self.pol == "tm":
                self._ez_indices = (center_idx, slice(0, nx))
                self._h_indices = (offset_idx, slice(0, nx))
                self._h_component = "Hy"
                print(f"[ModeSource] TM y-prop: Ez row {center_idx}, Hy row {offset_idx}")
                plot_coords = (np.arange(nx) + 0.5) * dx
            else: # TE
                # Hz (scalar) at center_idx, Ex (transverse) at offset_idx
                # Hz grid is (ny-1, nx-1). Ex grid is (ny, nx-1).
                # Choose Hz row so that Hz plane is one half-cell "upstream" of Ex plane.
                if self.direction == "+y": hz_row = max(0, offset_idx - 1)
                else: hz_row = min(ny - 2, offset_idx)
                
                self._hz_indices = (hz_row, slice(0, nx-1))
                self._e_indices = (offset_idx, slice(0, nx-1))
                self._e_component = "Ex"
                print(f"[ModeSource] TE y-prop: Hz row {hz_row}, Ex row {offset_idx}")
                plot_coords = (np.arange(nx-1) + 1.0) * dx # staggered x-coords

        # 2. Solve Modes
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        dL = dy if axis == "x" else dx
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
        # Use dt_physical = (coord_E - coord_H) / v_g  (approx), with v ≈ c / neff.
        # Positive dt_physical means H signal is sampled at a later time than E (H is upstream).
        if self._neff is not None:
            coord_e = 0.0
            coord_h = 0.0
            if axis == "x":
                if self.pol == "tm":
                    # Ez at x = (i + 0.5)dx, Hx at x = (i + 1.0)dx
                    coord_e = (self._ez_indices[1] + 0.5) * dx
                    coord_h = (self._h_indices[1] + 1.0) * dx
                else:
                    # Ey at x = (i + 0.5)dx, Hz at x = (i + 1.0)dx
                    coord_e = (self._e_indices[1] + 0.5) * dx
                    coord_h = (self._hz_indices[1] + 1.0) * dx
            else:
                if self.pol == "tm":
                    # Ez at y = (j + 0.5)dy, Hy at y = (j + 1.0)dy
                    coord_e = (self._ez_indices[0] + 0.5) * dy
                    coord_h = (self._h_indices[0] + 1.0) * dy
                else:
                    # Ex at y = (j + 0.5)dy, Hz at y = (j + 1.0)dy
                    coord_e = (self._e_indices[0] + 0.5) * dy
                    coord_h = (self._hz_indices[0] + 1.0) * dy
            self._dt_physical = (coord_e - coord_h) * float(np.real(self._neff)) / LIGHT_SPEED
        
        # 3. Extract Fields & Profiles
        if axis == "x":
            if self.pol == "tm":
                # TM: Main components Ez, Hy.
                # Mode solver returns (Ex, Ey, Ez, Hx, Hy, Hz).
                # For 2D TM x-prop: Ez is E_mode[2] or E_mode[1] (depending on solver output mapping).
                # Previous code: Ez_mode = E_mode[2] (if defined) or E_mode[1].
                # Hy_mode = H_mode[1] or H_mode[2].
                Hy_raw = np.squeeze(H_mode[1])
                Ez_raw = np.squeeze(E_mode[2])
                if np.max(np.abs(Hy_raw)) < 1e-9: Hy_raw = np.squeeze(H_mode[2])
                if np.max(np.abs(Ez_raw)) < 1e-9: Ez_raw = np.squeeze(E_mode[1])
                
                # Phase align
                idx_max = np.argmax(np.abs(Hy_raw))
                phase_ref = np.angle(Hy_raw[idx_max])
                Hy_profile = Hy_raw * np.exp(-1j * phase_ref)
                Ez_profile = Ez_raw * np.exp(-1j * phase_ref)
                
                # Impedance correction
                ETA_0 = np.sqrt(MU_0 / EPS_0)
                Z_phys = ETA_0 / np.real(self._neff)
                norm_h, norm_e = np.max(np.abs(Hy_profile)), np.max(np.abs(Ez_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ez_profile *= corr
                
                # Physical Huygens currents for TM x-prop:
                # Jz = ±Hy, My = ±Ez with sign given by direction (n = ±x).
                dir_sign = 1.0 if self.direction == "+x" else -1.0
                self._jz_profile = dir_sign * np.real(Hy_profile)
                self._my_profile = dir_sign * np.real(Ez_profile)
                
                plot_vals = (self._jz_profile, self._my_profile)
                
            else: # TE
                # TE: Main components Hz, Ey.
                # The mode solver field ordering depends on its internal propagation axis mapping.
                # For robust injection, pick the dominant components on this 1D cross-section.
                h_candidates = [np.squeeze(H_mode[i]) for i in range(3)]
                e_candidates = [np.squeeze(E_mode[i]) for i in range(3)]
                h_scores = [float(np.max(np.abs(hc))) for hc in h_candidates]
                e_scores = [float(np.max(np.abs(ec))) for ec in e_candidates]
                Hz_raw = h_candidates[int(np.argmax(h_scores))]
                Ey_raw = e_candidates[int(np.argmax(e_scores))]
                
                # Interpolate to staggered grid (ny-1)
                # Raw profiles are length ny. We need length ny-1.
                Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
                Ey_staggered = 0.5 * (Ey_raw[:-1] + Ey_raw[1:])
                
                # Phase align
                idx_max = np.argmax(np.abs(Hz_staggered))
                phase_ref = np.angle(Hz_staggered[idx_max])
                Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
                Ey_profile = Ey_staggered * np.exp(-1j * phase_ref)
                
                # Impedance correction (Z = Ey/Hz ?)
                # TE Impedance Z = E/H = Ey/Hz.
                # Z_phys = ETA_0 / neff (approx).
                ETA_0 = np.sqrt(MU_0 / EPS_0)
                Z_phys = ETA_0 / np.real(self._neff)
                norm_h, norm_e = np.max(np.abs(Hz_profile)), np.max(np.abs(Ey_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ey_profile *= corr

                # Ensure propagation direction (use Poynting sign on the 1D cross-section).
                # For TE x-prop, Sx ~ Re(Ey * conj(Hz)).
                desired_sign = 1.0 if self.direction == "+x" else -1.0
                power = float(np.sum(np.real(Ey_profile * np.conjugate(Hz_profile))))
                if power * desired_sign < 0.0: Hz_profile = -Hz_profile
                
                # Physical Huygens currents for TE x-prop in Beamz sign convention.
                # With Beamz's curl sign and our magnetic injection (-Mz), the sign pairing that launches +x is:
                # +x: Jy = +Hz, Mz = +Ey
                # -x: Jy = -Hz, Mz = -Ey
                if self.direction == "+x":
                    self._jy_profile = np.real(Hz_profile)
                    self._mz_profile = np.real(Ey_profile)
                else:
                    self._jy_profile = -np.real(Hz_profile)
                    self._mz_profile = -np.real(Ey_profile)
                
                plot_vals = (self._jy_profile, self._mz_profile) # J, M
                
        else: # axis == "y"
            # Y-propagation
            if self.pol == "tm":
                # TM: Main components Ez, Hx.
                # Mode solver uses propagation_axis=0 with ordering:
                # E=(Ez, Ex, Ey), H=(Hz, Hx, Hy). For "tm", Ey dominates and pairs with Hx for power flux.
                # We map solver Ey -> FDTD Ez and solver Hx -> transverse H for Jz.
                Hx_raw = np.squeeze(H_mode[1])
                Ez_raw = np.squeeze(E_mode[2])
                if np.max(np.abs(Hx_raw)) < 1e-9: Hx_raw = np.squeeze(H_mode[2])
                if np.max(np.abs(Ez_raw)) < 1e-9: Ez_raw = np.squeeze(E_mode[1])
                
                # TM y-prop: Jz = Hx (if n=y, J=y x Hx x = -Hx z? No y x x = -z. Jz = -Hx)
                # M = -y x Ez z = -Ez x. Mx = -Ez.
                
                # Original code used generic extraction.
                # h_t = hx_mode. e_t = ez_mode.
                
                # Phase align
                idx_max = np.argmax(np.abs(Hx_raw))
                phase_ref = np.angle(Hx_raw[idx_max])
                Hx_profile = Hx_raw * np.exp(-1j * phase_ref)
                Ez_profile = Ez_raw * np.exp(-1j * phase_ref)
                
                # Impedance
                ETA_0 = np.sqrt(MU_0 / EPS_0)
                Z_phys = ETA_0 / np.real(self._neff)
                norm_h, norm_e = np.max(np.abs(Hx_profile)), np.max(np.abs(Ez_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ez_profile *= corr
                
                # Signs (using sign_map)
                # For +y: j = -H, m = +E?
                # Original: jz = h_sign * real(h). my = m_sign * real(e).
                # +y: h_sign=-1.0 (so Jz = -Hx). m_sign=1.0 (so Mx = Ez).
                # Wait, original injected into "My" profile but mapped to "Hx" component?
                # Ah, original code injects `_my_profile` into `_h_component`.
                # If `_h_component` is "Hy" (transverse H), then `M` should be `My`?
                # But for TM y-prop, H is Hx (transverse). E is Ez.
                # `M` should be `Mx`. `J` should be `Jz`.
                # Original code set `_h_component="Hy"` for y-prop TM?
                # Line 64: `self._h_component = "Hy"`.
                # But for TM y-prop, the transverse field is Hx.
                # Why inject into Hy?
                # Maybe because the legacy code swapped Hx/Hy?
                # In legacy `fields.py`: `Hy` is `(ny-1, nx)`. `Hx` is `(ny, nx-1)`.
                # For y-prop (constant y slice):
                # We are at row `y_ez_idx`.
                # `Hy` has rows `0..ny-2`.
                # `Hx` has rows `0..ny-1`.
                # So `Hx` fits the slice. `Hy` does not.
                # So physically we want to inject into the component that fits the slice length `nx`.
                # `Ez` is `nx`. `Hx` (legacy) is `nx-1`. `Hy` (legacy) is `nx`.
                # So `Hy` (legacy) fits `Ez` length `nx`.
                # So we inject into `Hy` (legacy).
                # But physically `Hy` (legacy) is `Hx` (transverse).
                # So it matches! We inject `Mx` into `Hy` (legacy/code).
                
                # Physical Huygens currents for TM y-prop (Beamz curl convention):
                # Empirically, unidirectionality requires Mx to flip relative to Jz for ±y.
                # +y: Jz = -Hx, Mx = +Ez
                # -y: Jz = +Hx, Mx = -Ez
                if self.direction == "+y":
                    self._jz_profile = -np.real(Hx_profile)
                    self._my_profile = np.real(Ez_profile)
                else:
                    self._jz_profile = np.real(Hx_profile)
                    self._my_profile = -np.real(Ez_profile)
                plot_vals = (self._jz_profile, self._my_profile)
                
            else: # TE y-prop
                # TE: Main components Hz, Ex.
                # Robustly pick dominant components on this 1D cross-section.
                h_candidates = [np.squeeze(H_mode[i]) for i in range(3)]
                e_candidates = [np.squeeze(E_mode[i]) for i in range(3)]
                h_scores = [float(np.max(np.abs(hc))) for hc in h_candidates]
                e_scores = [float(np.max(np.abs(ec))) for ec in e_candidates]
                Hz_raw = h_candidates[int(np.argmax(h_scores))]
                Ex_raw = e_candidates[int(np.argmax(e_scores))]
                
                # Interpolate to staggered (nx-1)
                Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
                Ex_staggered = 0.5 * (Ex_raw[:-1] + Ex_raw[1:])
                
                # Phase align
                idx_max = np.argmax(np.abs(Hz_staggered))
                phase_ref = np.angle(Hz_staggered[idx_max])
                Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
                Ex_profile = Ex_staggered * np.exp(-1j * phase_ref)
                
                # Impedance
                ETA_0 = np.sqrt(MU_0 / EPS_0)
                Z_phys = ETA_0 / np.real(self._neff)
                norm_h, norm_e = np.max(np.abs(Hz_profile)), np.max(np.abs(Ex_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ex_profile *= corr
                
                # TE y-prop:
                # J = n x H = y x (0,0,Hz) = (Hz, 0, 0). So Jx = Hz.
                # M = -n x E = -y x (Ex,0,0) = -(-Ex z) = Ex z. So Mz = Ex.
                
                # Physical Huygens currents for TE y-prop:
                # Jx = +Hz for +y, -Hz for -y. Mz = +Ex for +y, -Ex for -y.
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

