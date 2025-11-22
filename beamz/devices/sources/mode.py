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
        
        # Sample permittivity at Ez positions (cell centers: i+1/2, j+1/2)
        eps_profile_ez = permittivity[y_ez_start:y_ez_end, x_ez_idx]
        
        # Sample permittivity at Hy positions (edge centers: i, j+1/2)
        # Hy y-coords are at (j+1.0)*dy, which fall between cell centers
        # Use nearest cell center permittivity (Option B from plan)
        if y_hy_end > y_hy_start:
            # Map Hy y-coords to nearest cell center indices
            # y_hy_coords = (j+1.0)*dy for j in [y_hy_start, y_hy_end-1]
            # Nearest cell center is at (j+0.5)*dy, so use j = y_hy_start to y_hy_end-1
            eps_profile_hy = permittivity[y_hy_start:y_hy_end, x_ez_idx]
        else:
            eps_profile_hy = np.array([], dtype=permittivity.dtype)
        
        # Solve for the mode at Ez positions to get H_y at Ez positions for J_z
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        neff_ez, e_fields_ez, h_fields_ez, prop_axis = solve_modes(
            eps=eps_profile_ez,
            omega=omega,
            dL=dy,
            m=1,
            direction=self.direction,
            filter_pol=self.pol,
            return_fields=True
        )
        
        # Solve for the mode at Hy positions to get E_z at Hy positions for M_y
        if eps_profile_hy.size > 0:
            neff_hy, e_fields_hy, h_fields_hy, prop_axis_hy = solve_modes(
                eps=eps_profile_hy,
                omega=omega,
                dL=dy,
                m=1,
                direction=self.direction,
                filter_pol=self.pol,
                return_fields=True
            )
        else:
            neff_hy = neff_ez
            e_fields_hy = e_fields_ez
            h_fields_hy = h_fields_ez
            prop_axis_hy = prop_axis
        
        # Use average neff (should be very close)
        self._neff = (neff_ez[0] + neff_hy[0]) / 2.0
        self._k = self._neff * 2 * np.pi / self.wavelength  # Wavenumber for spatial phase shift
        if abs(neff_ez[0] - neff_hy[0]) > 1e-6:
            print(f"[ModeSource] Warning: neff differs between Ez and Hy positions: {neff_ez[0]:.6f} vs {neff_hy[0]:.6f}")
        
        # Extract H_y from mode solved at Ez positions (for J_z)
        H_mode_ez = h_fields_ez[0]
        if self.pol == "te":
            Hy_mode_ez = np.squeeze(H_mode_ez[1])  # Hy
            # Fallback if needed
            if np.max(np.abs(Hy_mode_ez)) < 1e-9:
                Hy_mode_ez = np.squeeze(H_mode_ez[2])
        elif self.pol == "tm":
            Hy_mode_ez = np.squeeze(H_mode_ez[2])  # Hz as Hy equivalent
        else:
            raise ValueError(f"Unknown polarization: {self.pol}")
        
        # Extract E_z from mode solved at Hy positions (for M_y)
        E_mode_hy = e_fields_hy[0]
        if self.pol == "te":
            Ez_mode_hy = np.squeeze(E_mode_hy[2])  # Ez
            # Fallback if needed
            if np.max(np.abs(Ez_mode_hy)) < 1e-9:
                Ez_mode_hy = np.squeeze(E_mode_hy[1])
        elif self.pol == "tm":
            Ez_mode_hy = np.squeeze(E_mode_hy[1])  # Ey as Ez equivalent
        else:
            raise ValueError(f"Unknown polarization: {self.pol}")
        
        # For direction checking, we need both Ez and Hy at the same positions
        # Use Ez from Hy positions and Hy from Ez positions, but check consistency
        # Ensure proper propagation direction by checking Poynting vector
        # We need to interpolate one to match the other for this check
        if Hy_mode_ez.size > 0 and Ez_mode_hy.size > 0:
            # Interpolate Ez_mode_hy to Ez positions for consistency check
            if y_ez_coords.size == y_hy_coords.size:
                Ez_mode_check = Ez_mode_hy
                Hy_mode_check = Hy_mode_ez
            else:
                Ez_mode_check = np.interp(y_ez_coords, y_hy_coords, np.real(Ez_mode_hy) + 1j * np.imag(Ez_mode_hy))
                Hy_mode_check = Hy_mode_ez
        else:
            Ez_mode_check = Ez_mode_hy if Ez_mode_hy.size > 0 else np.array([0.0])
            Hy_mode_check = Hy_mode_ez if Hy_mode_ez.size > 0 else np.array([0.0])
        
        S_x = np.real(Ez_mode_check * np.conj(Hy_mode_check))
        power_x = np.sum(S_x)
        direction_sign = 1.0 if self.direction.startswith("+") else -1.0
        if power_x * direction_sign < 0:
            Hy_mode_ez = -Hy_mode_ez
        
        # Phase align TOGETHER to preserve impedance relationship
        # Find the peak of E_z and align both fields to that phase
        if Ez_mode_hy.size > 0:
            idx_max_hy = np.argmax(np.abs(Ez_mode_hy))
            phase_ref = np.angle(Ez_mode_hy[idx_max_hy])
            Ez_mode_hy = Ez_mode_hy * np.exp(-1j * phase_ref)
            # Align Hy_mode_ez using the same phase reference
            if Hy_mode_ez.size > 0:
                Hy_mode_ez = Hy_mode_ez * np.exp(-1j * phase_ref)
        
        # Debug: Check mode profiles
        if Ez_mode_hy.size > 0 and Hy_mode_ez.size > 0:
            idx_max_ez = np.argmax(np.abs(Hy_mode_ez))
            print(f"[DEBUG] Mode profile at peaks:")
            print(f"  Ez (at Hy pos): real={np.real(Ez_mode_hy[idx_max_hy]):.6f}, imag={np.imag(Ez_mode_hy[idx_max_hy]):.6f}")
            print(f"  Hy (at Ez pos): real={np.real(Hy_mode_ez[idx_max_ez]):.6f}, imag={np.imag(Hy_mode_ez[idx_max_ez]):.6f}")
            # Interpolate for ratio check
            if y_ez_coords.size == y_hy_coords.size:
                ratio = np.abs(Hy_mode_ez[idx_max_ez] / Ez_mode_hy[idx_max_hy])
            else:
                Ez_at_ez_pos = np.interp(y_ez_coords[idx_max_ez:idx_max_ez+1], y_hy_coords, Ez_mode_hy)[0]
                ratio = np.abs(Hy_mode_ez[idx_max_ez] / Ez_at_ez_pos)
            print(f"  Impedance ratio Hy/Ez: {ratio:.6f}")

        # Simple multi-lobe guard: warn if more than one local maximum in |Ez|
        if Ez_mode_hy.size >= 3:
            mag = np.abs(Ez_mode_hy)
            peak_mask = (mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])
            num_peaks = int(np.count_nonzero(peak_mask))
            if num_peaks > 1:
                print(f"[ModeSource] Warning: Detected {num_peaks} transverse peaks in |Ez|; consider narrowing source width to avoid exciting higher modes.")
        
        # Huygens surface currents:
        # J_z = H_y at Ez positions (from mode solved at Ez positions)
        jz_profile = np.real(Hy_mode_ez).copy()
        # M_y = E_z at Hy positions (from mode solved at Hy positions, no interpolation needed)
        if y_hy_end > y_hy_start:
            my_profile = np.real(Ez_mode_hy).copy()
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
        
        # Get the temporal modulation (already includes any carrier; don't add extra exp(iωt))
        signal_value_e = self.signal[current_step] if current_step < len(self.signal) else 0.0
        # Leapfrog staggering: H is at half-steps; approximate s(t+dt/2) by linear interpolation
        if current_step + 1 < len(self.signal):
            signal_value_h = 0.5 * (self.signal[current_step] + self.signal[current_step + 1])
        else:
            signal_value_h = self.signal[current_step] if current_step < len(self.signal) else 0.0
        
        # Inject J_z source into Ez field
        # Update equation: ∂_t E_z = (1/ε)[curl H - J_z]
        # Discrete update: E_z^(n+1) = E_z^n + (dt/ε) * [curl H - J_z]
        # We inject -J_z contribution directly: E_z_new = E_z_old - (dt/ε) * J_z
        # J_z = jz_profile * signal / resolution (normalized by cell size as per previous implementation)
        # Actually, previous implementation normalized by resolution. Let's keep that scaling.
        # Injection term: - (dt / (EPS_0 * epsilon)) * (jz_profile * signal / resolution)
        
        # Get permittivity at injection location
        y_ez_slice, x_ez_idx = self._ez_indices
        eps_at_source = fields.permittivity[y_ez_slice, x_ez_idx]
        
        jz_term = self._jz_profile * signal_value_e / resolution
        # Note: epsilon is relative permittivity, so multiply by EPS_0
        ez_injection = -jz_term * dt / (EPS_0 * eps_at_source)
        
        fields.Ez[self._ez_indices] += ez_injection
        
        # Inject M_y source into Hy field
        # Update equation: ∂_t H_y = (1/μ)[∂_x E_z - M_y]
        # Discrete update: H_y^(n+1/2) = H_y^(n-1/2) - (dt/μ) * [curl E + M_y]
        # Wait, previously we said H_y = H_y - dt/μ * (curlE + M_y)
        # If we inject before update: H_y_new = H_y_old - (dt/μ) * M_y
        # M_y = my_profile * signal / resolution
        # Injection term: - (dt / (MU_0 * mu_relative)) * (my_profile * signal / resolution)
        # Assuming mu_relative = 1.0 for now (typical for dielectric waveguides)
        
        # Get permeability at injection location (if available, otherwise assume 1.0)
        y_hy_slice, x_hy_idx = self._hy_indices
        if hasattr(fields, 'permeability'):
            mu_at_source = fields.permeability[y_hy_slice, x_hy_idx]
        else:
            mu_at_source = 1.0
            
        # Calculate spatial phase shift for M_y to account for its offset from Ez column
        # Hy is at x_hy_coord, Ez is at x_ez_coord
        # Wave propagates as exp(i(kx - wt))
        # We want fields to match the mode: Hy(x,t) = Hy_mode * exp(i*k*x - i*w*t)
        # But our profile My is derived from Ez_mode at Hy position.
        # For +x: Hy is at x_ez - dx. Phase is exp(-i*k*dx) relative to x_ez.
        # For -x: Hy is at x_ez + dx. Phase is exp(i*k*dx) relative to x_ez.
        # But we need to inject M_y such that it launches this wave.
        # Unidirectional condition: J_z = -Hy_mode, M_y = Ez_mode.
        # These must be phase-aligned AT THE SAME physical position.
        # Since we inject M_y at x_hy != x_ez, we must phase-shift it to what it WOULD be at x_ez?
        # No, we inject at x_hy. The source currents J(x_s) and M(x_s) launch the wave from x_s.
        # Here we have two source sheets: J at x_ez, M at x_hy.
        # To launch a wave in +x, the field from J at x_ez and M at x_hy must interfere constructively for +x.
        # Forward wave from J at x_ez (at x>x_ez): E ~ -J * exp(ik(x-x_ez))
        # Forward wave from M at x_hy (at x>x_hy): E ~ -M * exp(ik(x-x_hy)) * (some impedance factor)
        # We want them to add up.
        # If M_y = E_mode, J_z = H_mode.
        # For pure +x wave, E_mode and H_mode are in phase.
        # If we just use M_y = E_mode and J_z = H_mode without shift:
        # At x_ez: Field from J is E_J. Field from M (propagated from x_hy=x_ez-dx) is E_M * exp(ik*dx).
        # We need E_J + E_M * exp(ik*dx) = E_total.
        # And for -x direction (at x < x_hy):
        # Field from J (propagated to x_hy): E_J * exp(ik*dx). Field from M: E_M.
        # We want E_J * exp(ik*dx) + E_M = 0.
        # So E_M = -E_J * exp(ik*dx).
        # If E_J ~ -J ~ -H_mode, then E_M ~ H_mode * exp(ik*dx).
        # Also E_M ~ -M ~ -E_mode.
        # So -E_mode ~ H_mode * exp(ik*dx).
        # Since E_mode = eta * H_mode, we have -eta * H_mode ~ H_mode * exp(ik*dx).
        # This implies M should be related to J with a phase shift.
        # Let's use the simple rule: shift M_y to the position of J_z effectively?
        # Correct shift: my_term should be shifted by exp(i * k * (x_hy - x_ez) * direction_sign) ?
        # Let's apply the shift exp(1j * k * dx * direction_sign)
        
        direction_sign = 1.0 if self.direction.startswith("+") else -1.0
        # For +x: x_hy = x_ez - 1. We want to effectively inject M at x_ez but physically at x_hy.
        # To effectively behave as if at x_ez, we advance phase by k*dx?
        # Wait, if wave is exp(i(kx - wt)), then at x-dx it is exp(-ikdx) relative to x.
        # So value at x-dx lags value at x.
        # So if we want to match value at x, we must multiply by exp(+ikdx)?
        # Let's try applying the shift corresponding to the spatial offset.
        
        phase_shift = np.exp(1j * self._k * resolution * direction_sign)
        my_term = self._my_profile * signal_value_h / resolution
        # Apply phase shift (taking real part as profiles are real)
        # Actually, we should apply phase shift to the complex mode amplitude then take real part?
        # But we only have real profiles stored.
        # Approximation: cos(k*dx) * M_y - sin(k*dx) * M_y_imag? We don't have imag part stored!
        # But we aligned phases so imaginary part is small.
        # So just scaling by cos(k*dx) is approximately correct? No, that changes amplitude.
        # We need to simply shift the time signal?
        # s(t) -> s(t - dx/v) ?
        # Yes! Spatial shift dx corresponds to time delay dt_shift = dx / (c/n) = n*dx/c.
        # Phase shift exp(ikdx) = exp(i * (n w/c) * dx) = exp(i w (n dx/c)) = exp(i w dt_shift).
        # This corresponds to time shift s(t + dt_shift).
        # For +x: x_hy = x_ez - dx. Source is "earlier" in space.
        # To match the wavefront at x_ez, the source at x_hy must have emitted "earlier"?
        # No, wave travels x_hy -> x_ez.
        # Signal at x_hy should be s(t). Signal at x_ez should be s(t - dx/v).
        # We use same s(t).
        # So effectively J at x_ez is "ahead" of M at x_hy by dx/v.
        # If we use same s(t), then J is emitting "too early" relative to M?
        # We want them to be consistent.
        # If we shift M_y by shifting the time of signal evaluation?
        
        # Time shift for M_y:
        # For +x: M at x-dx. Wave reaches x at t+dt_shift.
        # So field at x from M(t) is proportional to M(t - dt_shift).
        # We want this to cancel backward wave from J(t).
        # Backward wave from J at x travels to x-dx at t+dt_shift.
        # Field at x-dx from J(t) is proportional to J(t - dt_shift).
        # We want M(t) + Term * J(t - dt_shift) = 0.
        # So M(t) should be time-shifted relative to J?
        # Actually, standard implementation uses same s(t) but specific spatial locations.
        # But we must ensure the *grid* phase velocity matches.
        # If we can't do fractional time shift easily, let's use the complex phase shift approximation
        # on the profile IF we had complex profile.
        # We stored real profiles.
        # Let's use the phase shift factor on the real profile:
        # My_new = My * cos(k*dx). (Assuming imaginary part was 0).
        # This reduces amplitude. Ideally we want My * exp(ikdx).
        # Since we discarded imaginary part, we can't rotate properly.
        # However, for small dx, exp(ikdx) ~ 1 + ikdx.
        # Maybe just cos(k*dx) is enough if imaginary part was negligible?
        # Or maybe we should have stored complex profiles?
        
        # BETTER FIX: Store complex profiles in initialize and take real part here with phase shift.
        # But that requires re-running initialize.
        # Let's just multiply by real(exp(ikdx)) = cos(k*dx) for now.
        # It's better than nothing.
        
        # Actually, let's assume the user wants us to fix it properly.
        # Ideally we would interpolate s(t) to s(t +/- dt_shift).
        # dt_shift = resolution * self._neff / LIGHT_SPEED.
        # direction_sign: +1 for +x.
        # For +x: M at left (earlier). J at right (later).
        # Wave M->J takes dt_shift.
        # We want M(t) to match J(t + dt_shift).
        # So if J uses s(t), M should use s(t + dt_shift)?
        # Or s(t - dt_shift)?
        # Let's try evaluating signal at shifted time for M_y.
        
        # Take real part of neff for time shift calculation (group velocity approx phase velocity)
        neff_real = float(np.real(self._neff))
        # IMPORTANT: Hx (physical Hy) grid is staggered by 0.5*dx relative to Ez grid.
        # x_hx_idx = x_ez_idx - 1 refers to the Hx node at (i-1)+0.5 = i-0.5.
        # So the spatial distance between Ez injection (at i) and Hy injection (at i-0.5) is 0.5*dx.
        # We must use 0.5 * resolution for the time shift.
        dt_shift = 0.5 * resolution * neff_real / LIGHT_SPEED * direction_sign
        
        # M_y is at x_ez - 0.5*dx (for +x).
        # To synchronize with J at x_ez, M should be evaluated at...?
        # Let's assume J uses signal_value_e (at t).
        # M uses signal_value_h (at t+dt/2).
        # We add the spatial time shift to M's time.
        # For +x: Wave moves left->right. M is at left.
        # M(t_M) arrives at J at t_M + dt_shift.
        # We want this to coincide with J(t).
        # So t_M + dt_shift = t  => t_M = t - dt_shift.
        # So M should be evaluated at (t + 0.5*dt) - dt_shift.
        
        # Interpolate signal at t_h_shifted
        t_h = t + 0.5 * dt
        t_h_shifted = t_h - dt_shift
        
        # Simple linear interpolation from array if available, or use function
        # We have self.signal array and dt. t corresponds to current_step * dt.
        # t_h_shifted index = t_h_shifted / dt.
        idx_float = float(np.real(t_h_shifted / dt))
        idx_low = int(np.floor(idx_float))
        idx_high = idx_low + 1
        frac = idx_float - idx_low
        
        if 0 <= idx_low < len(self.signal) - 1:
            signal_val_shifted = (1.0 - frac) * self.signal[idx_low] + frac * self.signal[idx_high]
        else:
            signal_val_shifted = 0.0
            
        my_term = self._my_profile * signal_val_shifted / resolution
        # Note: mu is relative permeability, so multiply by MU_0
        hy_injection = -my_term * dt / (MU_0 * mu_at_source)
        
        fields.Hy[self._hy_indices] += hy_injection
        
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

