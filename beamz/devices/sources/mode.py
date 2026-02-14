import logging

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0, µm
from beamz.devices.sources.solve import solve_modes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _jax_tukey_window(M: int, alpha: float = 0.5) -> jnp.ndarray:
    """JAX-compatible Tukey (tapered cosine) window.

    Replaces scipy.signal.windows.tukey for differentiable source initialization.

    Args:
        M: Number of points in the window
        alpha: Shape parameter (0 = rectangular, 1 = Hann)

    Returns:
        The Tukey window as a JAX array
    """
    if M <= 0:
        return jnp.array([])
    if M == 1:
        return jnp.ones(1)

    n = jnp.arange(M)
    width = alpha * (M - 1) / 2.0

    # Avoid division by zero when alpha=0
    width = jnp.maximum(width, 1e-10)

    # Three regions: taper up, flat, taper down
    left_taper = 0.5 * (1 + jnp.cos(jnp.pi * (n / width - 1)))
    right_taper = 0.5 * (1 + jnp.cos(jnp.pi * ((n - (M - 1 - width)) / width)))

    window = jnp.where(
        n < width, left_taper, jnp.where(n > (M - 1) - width, right_taper, 1.0)
    )
    return window


def _scipy_tukey(n, alpha=0.3):
    from scipy.signal.windows import tukey
    return tukey(n, alpha=alpha)


def _crop_and_window_2d(profile, z_s, z_e, t_s, t_e, window):
    """Crop a 2D profile to [z_s:z_e, t_s:t_e] and multiply by window.

    Handles shape mismatches by taking the minimum overlap region.
    """
    cropped = profile[z_s:z_e, t_s:t_e]
    if cropped.size == 0:
        return cropped
    if cropped.shape == window.shape:
        return cropped * window
    z_min = min(cropped.shape[0], window.shape[0])
    t_min = min(cropped.shape[1], window.shape[1])
    return cropped[:z_min, :t_min] * window[:z_min, :t_min]


def _make_tukey_window_2d(height_cells, width_cells, alpha=0.3, use_jax=True):
    """Create a 2D Tukey window via outer product of 1D windows."""
    _make = _jax_tukey_window if use_jax else _scipy_tukey
    _ones = jnp.ones if use_jax else np.ones

    wz = _make(height_cells, alpha=alpha) if height_cells > 2 else _ones(max(1, height_cells))
    wt = _make(width_cells, alpha=alpha) if width_cells > 2 else _ones(max(1, width_cells))

    if use_jax:
        return wz[:, jnp.newaxis] * wt[jnp.newaxis, :]
    return wz[:, np.newaxis] * wt[np.newaxis, :]


def _stagger_half(field, axis):
    """Average adjacent cells along *axis* (0 or 1) for Yee half-grid staggering."""
    if field.shape[axis] <= 1:
        return field
    if axis == 0:
        return 0.5 * (field[:-1, :] + field[1:, :])
    return 0.5 * (field[:, :-1] + field[:, 1:])


def _stagger_both(field):
    """Stagger along both axes (for longitudinal H that sits at half-grid in both)."""
    out = field
    if out.shape[1] > 1:
        out = 0.5 * (out[:, :-1] + out[:, 1:])
    if out.shape[0] > 1:
        out = 0.5 * (out[:-1, :] + out[1:, :])
    return out


def _compute_transverse_bounds(center_val, extent, resolution, grid_max):
    """Return (start_idx, end_idx) for the injection window along a transverse axis."""
    center_idx = int(round(center_val / resolution))
    half_idx = int(round((extent / 2) / resolution))
    return max(0, center_idx - half_idx), min(grid_max, center_idx + half_idx)


def _impedance_correct_e_fields(E_components, H_components, Z_phys, use_jax=True):
    """Scale E-field profiles so E/H ratio matches physical impedance Z_phys.

    Parameters
    ----------
    E_components : list of arrays
        E-field profiles to scale.
    H_components : list of arrays
        H-field profiles used to measure current impedance.
    Z_phys : float
        Target impedance.
    use_jax : bool
        Whether to use JAX operations (True for 3D path) or numpy (False for 2D y-axis path).

    Returns
    -------
    list of arrays
        Corrected E-field profiles.
    """
    _mod = jnp if use_jax else np
    eps = 1e-12

    norm_E = eps
    for e in E_components:
        norm_E = max(norm_E, float(_mod.max(_mod.abs(e))))

    norm_H = eps
    for h in H_components:
        norm_H = max(norm_H, float(_mod.max(_mod.abs(h))))

    if norm_E > eps and norm_H > eps:
        corr = Z_phys / (norm_E / norm_H)
        return [e * corr for e in E_components]
    return list(E_components)


def _impedance_match_e_profile(e_profile, h_profile, Z_phys, eps=1e-12):
    """Match one E-profile to one H-profile with a shared 2D rule.

    Uses mean-absolute amplitudes so TE/TM use the same normalization
    and the result is less sensitive to single-cell peaks.
    """
    norm_h = np.mean(np.abs(h_profile))
    norm_e = np.mean(np.abs(e_profile))
    if norm_h > eps and norm_e > eps:
        return e_profile * (Z_phys / (norm_e / norm_h))
    return e_profile


def _build_3d_profiles(
    Ex, Ey, Ez, Hx, Hy, Hz,
    axis, direction,
    center, width, height,
    center_idx, offset_idx,
    grid_shape, resolution, neff,
):
    """Build staggered, windowed, impedance-corrected injection profiles for 3D.

    Returns
    -------
    profiles : dict
        Mapping component name -> numpy real profile array.
    indices : dict
        Mapping component name -> index tuple for field injection.
    """
    nz, ny, nx = grid_shape
    dir_sign = 1.0 if direction.startswith("+") else -1.0
    ETA_0 = np.sqrt(MU_0 / EPS_0)
    Z_phys = ETA_0 / max(np.real(neff), 1e-6)

    if axis == "x":
        return _build_3d_x(
            Ex, Ey, Ez, Hx, Hy, Hz,
            dir_sign, Z_phys,
            center, width, height,
            center_idx, offset_idx,
            nz, ny, nx, resolution,
        )
    else:
        return _build_3d_y(
            Ex, Ey, Ez, Hx, Hy, Hz,
            dir_sign, Z_phys,
            center, width, height,
            center_idx, offset_idx,
            nz, ny, nx, resolution,
        )


def _build_3d_x(
    Ex, Ey, Ez, Hx, Hy, Hz,
    dir_sign, Z_phys,
    center, width, height,
    center_idx, offset_idx,
    nz, ny, nx, resolution,
):
    # --- stagger ---
    Ey_s = _stagger_half(Ey, axis=1)
    Ez_s = _stagger_half(Ez, axis=0)
    Ex_s = Ex
    Hx_s = _stagger_both(Hx)
    Hy_s = _stagger_half(Hy, axis=0)
    Hz_s = _stagger_half(Hz, axis=1)

    # --- transverse bounds ---
    y_start, y_end = _compute_transverse_bounds(center[1], width, resolution, ny)
    z_center = center[2] if len(center) > 2 else (nz // 2) * resolution
    z_start, z_end = _compute_transverse_bounds(z_center, height, resolution, nz)

    # --- indices  (z_slice, y_slice, x_index) ---
    indices = {
        "Ex": (slice(z_start, min(z_end, Ex_s.shape[0], nz)),
               slice(y_start, min(y_end, Ex_s.shape[1], ny)),
               offset_idx),
        "Ey": (slice(z_start, min(z_end, Ey_s.shape[0], nz)),
               slice(y_start, min(y_end, Ey_s.shape[1], ny - 1)),
               center_idx),
        "Ez": (slice(z_start, min(z_end, Ez_s.shape[0], nz - 1)),
               slice(y_start, min(y_end, Ez_s.shape[1], ny)),
               center_idx),
        "Hx": (slice(z_start, min(z_end, Hx_s.shape[0], nz - 1)),
               slice(y_start, min(y_end, Hx_s.shape[1], ny - 1)),
               center_idx),
        "Hy": (slice(z_start, min(z_end, Hy_s.shape[0], nz - 1)),
               slice(y_start, min(y_end, Hy_s.shape[1], ny)),
               offset_idx),
        "Hz": (slice(z_start, min(z_end, Hz_s.shape[0], nz)),
               slice(y_start, min(y_end, Hz_s.shape[1], ny - 1)),
               offset_idx),
    }

    # --- impedance correction (JAX) ---
    Ex_s, Ey_s, Ez_s = _impedance_correct_e_fields(
        [Ex_s, Ey_s, Ez_s], [Hy_s, Hz_s], Z_phys, use_jax=True,
    )

    # --- crop & window ---
    staggered = {"Ex": Ex_s, "Ey": Ey_s, "Ez": Ez_s, "Hx": Hx_s, "Hy": Hy_s, "Hz": Hz_s}
    profiles = _crop_and_window_all(
        staggered, z_start, z_end, y_start, y_end, dir_sign, use_jax=True,
    )

    extra = {"_y_start": y_start, "_y_end": y_end, "_z_start": z_start, "_z_end": z_end,
             "_h_component": "Hy", "_e_component": "Ey"}
    return profiles, indices, extra


def _build_3d_y(
    Ex, Ey, Ez, Hx, Hy, Hz,
    dir_sign, Z_phys,
    center, width, height,
    center_idx, offset_idx,
    nz, ny, nx, resolution,
):
    # --- stagger (y-propagation) ---
    Ex_s = _stagger_half(Ex, axis=1)
    Ey_s = Ey
    Ez_s = _stagger_half(Ez, axis=0)
    Hx_s = _stagger_half(Hx, axis=0)
    Hy_s = _stagger_both(Hy)
    Hz_s = _stagger_half(Hz, axis=1)

    # --- transverse bounds ---
    x_start, x_end = _compute_transverse_bounds(center[0], width, resolution, nx)
    z_center = center[2] if len(center) > 2 else (nz // 2) * resolution
    z_start, z_end = _compute_transverse_bounds(z_center, height, resolution, nz)

    # --- indices  (z_slice, y_index, x_slice) ---
    indices = {
        "Ex": (slice(z_start, min(z_end, Ex_s.shape[0], nz)),
               center_idx,
               slice(x_start, min(x_end, Ex_s.shape[1], nx - 1))),
        "Ey": (slice(z_start, min(z_end, Ey_s.shape[0], nz)),
               offset_idx,
               slice(x_start, min(x_end, Ey_s.shape[1], nx))),
        "Ez": (slice(z_start, min(z_end, Ez_s.shape[0], nz - 1)),
               center_idx,
               slice(x_start, min(x_end, Ez_s.shape[1], nx))),
        "Hx": (slice(z_start, min(z_end, Hx_s.shape[0], nz - 1)),
               center_idx,
               slice(x_start, min(x_end, Hx_s.shape[1], nx))),
        "Hy": (slice(z_start, min(z_end, Hy_s.shape[0], nz - 1)),
               center_idx,
               slice(x_start, min(x_end, Hy_s.shape[1], nx - 1))),
        "Hz": (slice(z_start, min(z_end, Hz_s.shape[0], nz)),
               offset_idx,
               slice(x_start, min(x_end, Hz_s.shape[1], nx - 1))),
    }

    # --- impedance correction (numpy path — matches original y-axis code) ---
    Ex_s, Ey_s, Ez_s = _impedance_correct_e_fields(
        [Ex_s, Ey_s, Ez_s], [Hx_s, Hz_s], Z_phys, use_jax=False,
    )

    # --- crop & window ---
    staggered = {"Ex": Ex_s, "Ey": Ey_s, "Ez": Ez_s, "Hx": Hx_s, "Hy": Hy_s, "Hz": Hz_s}
    profiles = _crop_and_window_all(
        staggered, z_start, z_end, x_start, x_end, dir_sign, use_jax=False,
    )

    extra = {"_x_start": x_start, "_x_end": x_end, "_z_start": z_start, "_z_end": z_end,
             "_h_component": "Hx", "_e_component": "Ex"}
    return profiles, indices, extra


def _crop_and_window_all(staggered, z_start, z_end, t_start, t_end, dir_sign, use_jax):
    """Crop all six staggered profiles and multiply by a 2D Tukey window."""
    ref = next(iter(staggered.values()))
    pz_end = min(z_end, ref.shape[0])
    pt_end = min(t_end, ref.shape[1])
    h_cells = pz_end - z_start
    w_cells = pt_end - t_start

    window = _make_tukey_window_2d(h_cells, w_cells, alpha=0.3, use_jax=use_jax)

    profiles = {}
    for name, field in staggered.items():
        fe = min(z_end, field.shape[0])
        te = min(t_end, field.shape[1])
        profiles[name] = dir_sign * np.real(
            _crop_and_window_2d(field, z_start, fe, t_start, te, window)
        )
    return profiles


# ---------------------------------------------------------------------------
# Huygens cross-product sign tables
# sign = multiplier in: target += sign * source_profile * sig * dt / (material * dx)
# Only transverse components are injected; longitudinal components are skipped.
# ---------------------------------------------------------------------------
_HUYGENS_SIGNS = {
    "x": {
        "e": [("Ey", "Hz", -1), ("Ez", "Hy", +1)],
        "h": [("Hy", "Ez", -1), ("Hz", "Ey", +1)],
    },
    "y": {
        "e": [("Ex", "Hz", +1), ("Ez", "Hx", -1)],
        "h": [("Hx", "Ez", +1), ("Hz", "Ex", -1)],
    },
}


def _inject_3d_e_fields(fields, profiles, indices, signal_e, dt, resolution, axis):
    """Inject E-field components for 3D Huygens source (J = n x H), after E update."""
    for e_comp, h_source, sign in _HUYGENS_SIGNS[axis]["e"]:
        _inject_e_component(fields, e_comp, profiles, indices, h_source,
                            signal_e, dt, resolution, sign=sign)


def _inject_3d_h_fields(fields, profiles, indices, signal_h, dt, resolution, axis):
    """Inject H-field components for 3D Huygens source (M = -n x E), after H update."""
    for h_comp, e_source, sign in _HUYGENS_SIGNS[axis]["h"]:
        _inject_h_component(fields, h_comp, profiles, indices, e_source,
                            signal_h, dt, resolution, sign=sign)


def _inject_3d_fields(fields, profiles, indices, signal_e, signal_h, dt, resolution, axis="x"):
    """Inject all field components into a 3D field object (backward compat wrapper)."""
    _inject_3d_h_fields(fields, profiles, indices, signal_h, dt, resolution, axis)
    _inject_3d_e_fields(fields, profiles, indices, signal_e, dt, resolution, axis)


def _inject_e_component(fields, comp, profiles, indices, j_source, sig, dt, res, sign=-1):
    """Inject one E-field component via J = cross(n, H)."""
    profile = profiles.get(comp)
    idx = indices.get(comp)
    j_term = profiles.get(j_source)
    if profile is None or idx is None or j_term is None:
        return
    target = getattr(fields, comp)[idx]
    j_term = _match_shape(j_term, target.shape)
    if j_term is None:
        logger.debug("Shape mismatch injecting %s, skipping", comp)
        return
    eps = fields.permittivity[idx]
    setattr(fields, comp,
            getattr(fields, comp).at[idx].add(sign * j_term * sig * dt / (EPS_0 * eps * res)))


def _inject_h_component(fields, comp, profiles, indices, m_source, sig, dt, res, sign=-1):
    """Inject one H-field component via M = -cross(n, E)."""
    profile = profiles.get(comp)
    idx = indices.get(comp)
    m_term = profiles.get(m_source)
    if profile is None or idx is None or m_term is None:
        return
    target = getattr(fields, comp)[idx]
    m_term = _match_shape(m_term, target.shape)
    if m_term is None:
        logger.debug("Shape mismatch injecting %s, skipping", comp)
        return
    mu = getattr(fields, "permeability", None)
    mu_val = mu[idx] if mu is not None else 1.0
    setattr(fields, comp,
            getattr(fields, comp).at[idx].add(sign * m_term * sig * dt / (MU_0 * mu_val * res)))


def _match_shape(profile, target_shape):
    """Match profile shape to target field shape, trimming or padding as needed."""
    if profile is None:
        return None
    profile = np.squeeze(profile)
    if profile.shape == target_shape:
        return profile
    if profile.ndim == len(target_shape):
        slices = tuple(slice(0, min(profile.shape[i], target_shape[i]))
                       for i in range(profile.ndim))
        trimmed = profile[slices]
        if trimmed.shape == target_shape:
            return trimmed
        result = np.zeros(target_shape, dtype=profile.dtype)
        result[tuple(slice(0, trimmed.shape[i]) for i in range(trimmed.ndim))] = trimmed
        return result
    return None


# ---------------------------------------------------------------------------
# ModeSource class
# ---------------------------------------------------------------------------

class ModeSource:
    """Huygens mode source on Yee grid supporting ±x/±y propagation.

    In 3D, injects all 6 field components (Ex, Ey, Ez, Hx, Hy, Hz) for accurate
    mode injection, accounting for proper Yee grid staggering.
    """

    def __init__(
        self, grid, center, width, wavelength, pol, signal, direction="+x", height=None
    ):
        self.grid = grid
        self.center = (
            center if isinstance(center, (tuple, list)) else (center, grid.height / 2)
        )
        self.width = width
        self.height = height
        self.wavelength = wavelength
        self.pol = str(pol).lower()
        if self.pol not in {"te", "tm"}:
            raise ValueError(f"pol must be 'te' or 'tm', got {pol!r}")
        self.signal = signal
        self.direction = direction

        # Storage for all 6 field component profiles (for 3D injection)
        self._Ex_profile = None
        self._Ey_profile = None
        self._Ez_profile = None
        self._Hx_profile = None
        self._Hy_profile = None
        self._Hz_profile = None

        # Indices for each component's injection position
        self._Ex_indices = None
        self._Ey_indices = None
        self._Ez_indices = None
        self._Hx_indices = None
        self._Hy_indices = None
        self._Hz_indices = None

        # Legacy attributes for compatibility and 2D
        self._jz_profile = None
        self._my_profile = None
        self._mz_profile = None
        self._jy_profile = None
        self._jx_profile = None
        self._ez_indices = None
        self._h_indices = None
        self._hz_indices = None
        self._e_indices = None

        self._h_component = None
        self._e_component = None
        self._neff = None
        self._dt_physical = 0.0
        self._initialized = False

    def initialize(self, permittivity, resolution):
        """Compute the mode and set up the source currents for all 6 components in 3D."""
        dx = dy = resolution
        is_3d = permittivity.ndim == 3
        self._resolution = resolution
        self._is_3d = is_3d

        if is_3d:
            nz, ny, nx = permittivity.shape
            dz = resolution
            self._grid_shape = (nz, ny, nx)
            if self.height is None:
                self.height = self.width
        else:
            ny, nx = permittivity.shape
            nz = 1
            self._grid_shape = (ny, nx)
            self.height = None

        axis = "x" if self.direction in ("+x", "-x") else "y"
        self._axis = axis
        self._dt_physical = 0.0

        # 1. Get center index for injection plane
        if axis == "x":
            center_idx = int(np.clip(np.round(self.center[0] / dx - 0.5), 0, nx - 1))
            if self.direction == "+x":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(nx - 2, center_idx)

            if is_3d:
                eps_profile = permittivity[:, :, center_idx]
                self._eps_profile_2d = eps_profile
            else:
                eps_profile = permittivity[:, center_idx]
                self._eps_profile_2d = None

        else:  # axis == "y"
            center_idx = int(np.clip(np.round(self.center[1] / dy - 0.5), 0, ny - 1))
            if self.direction == "+y":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(ny - 2, center_idx)

            if is_3d:
                eps_profile = permittivity[:, center_idx, :]
                self._eps_profile_2d = eps_profile
            else:
                eps_profile = permittivity[center_idx, :]
                self._eps_profile_2d = None

        # 2. Solve for mode fields
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        dL = dz if is_3d else (dy if axis == "x" else dx)
        neff_val, e_fields, h_fields, _ = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=dL,
            m=1,
            direction=self.direction,
            filter_pol=self.pol,
            return_fields=True,
        )
        self._neff = neff_val[0]
        E_mode = e_fields[0]
        H_mode = h_fields[0]

        # 3. Extract all 6 components and convert to JAX arrays
        Ex_raw = jnp.asarray(jnp.squeeze(E_mode[0]))
        Ey_raw = jnp.asarray(jnp.squeeze(E_mode[1]))
        Ez_raw = jnp.asarray(jnp.squeeze(E_mode[2]))
        Hx_raw = jnp.asarray(jnp.squeeze(H_mode[0]))
        Hy_raw = jnp.asarray(jnp.squeeze(H_mode[1]))
        Hz_raw = jnp.asarray(jnp.squeeze(H_mode[2]))

        # 4. Phase align all components to dominant field (JAX-compatible)
        if self.pol == "tm":
            ref_field = jnp.where(
                jnp.max(jnp.abs(Ez_raw)) > jnp.max(jnp.abs(Ey_raw)), Ez_raw, Ey_raw
            )
        else:
            ref_field = Ey_raw if axis == "x" else Ex_raw
            ref_field = jnp.where(jnp.max(jnp.abs(ref_field)) < 1e-9, Ez_raw, ref_field)

        idx_max = jnp.argmax(jnp.abs(ref_field))
        phase_ref = jnp.angle(ref_field.flatten()[idx_max])

        Ex_aligned = Ex_raw * jnp.exp(-1j * phase_ref)
        Ey_aligned = Ey_raw * jnp.exp(-1j * phase_ref)
        Ez_aligned = Ez_raw * jnp.exp(-1j * phase_ref)
        Hx_aligned = Hx_raw * jnp.exp(-1j * phase_ref)
        Hy_aligned = Hy_raw * jnp.exp(-1j * phase_ref)
        Hz_aligned = Hz_raw * jnp.exp(-1j * phase_ref)

        # 5. Apply Yee grid staggering and set up indices
        if is_3d:
            self._setup_3d_injection(
                Ex_aligned, Ey_aligned, Ez_aligned,
                Hx_aligned, Hy_aligned, Hz_aligned,
                center_idx, offset_idx, axis,
                nz, ny, nx, resolution,
            )
        else:
            self._setup_2d_injection(
                E_mode, H_mode, center_idx, offset_idx, axis, ny, nx, resolution
            )

        self._compute_dt_physical(axis, is_3d, dx, dy)
        self._initialized = True

    def _setup_3d_injection(
        self, Ex, Ey, Ez, Hx, Hy, Hz,
        center_idx, offset_idx, axis,
        nz, ny, nx, resolution,
    ):
        """Set up full 6-component injection for 3D simulations."""
        profiles, indices, extra = _build_3d_profiles(
            Ex, Ey, Ez, Hx, Hy, Hz,
            axis=axis,
            direction=self.direction,
            center=self.center,
            width=self.width,
            height=self.height,
            center_idx=center_idx,
            offset_idx=offset_idx,
            grid_shape=(nz, ny, nx),
            resolution=resolution,
            neff=self._neff,
        )

        # Store profiles on self
        self._Ex_profile = profiles.get("Ex")
        self._Ey_profile = profiles.get("Ey")
        self._Ez_profile = profiles.get("Ez")
        self._Hx_profile = profiles.get("Hx")
        self._Hy_profile = profiles.get("Hy")
        self._Hz_profile = profiles.get("Hz")

        # Store indices on self
        self._Ex_indices = indices.get("Ex")
        self._Ey_indices = indices.get("Ey")
        self._Ez_indices = indices.get("Ez")
        self._Hx_indices = indices.get("Hx")
        self._Hy_indices = indices.get("Hy")
        self._Hz_indices = indices.get("Hz")

        # Store extra metadata
        for key, val in extra.items():
            setattr(self, key, val)

        # Legacy compatibility
        self._jz_profile = self._Hz_profile
        self._my_profile = self._Ez_profile

    def _setup_2d_injection(
        self, E_mode, H_mode, center_idx, offset_idx, axis, ny, nx, resolution
    ):
        """2D injection setup using explicit global component mapping.

        `solve_modes(..., return_fields=True)` returns fields ordered as:
        E_mode = [Ex, Ey, Ez], H_mode = [Hx, Hy, Hz] in global components.
        We pick the physically matching TE/TM pair for the chosen propagation axis.
        """
        dir_sign = 1.0 if self.direction.startswith("+") else -1.0
        ETA_0 = np.sqrt(MU_0 / EPS_0)
        Z_phys = ETA_0 / max(np.real(self._neff), 1e-6)

        if axis == "x":
            self._setup_2d_x(
                E_mode, H_mode, center_idx, offset_idx, ny, nx, resolution,
                dir_sign, Z_phys,
            )
        else:
            self._setup_2d_y(
                E_mode, H_mode, center_idx, offset_idx, ny, nx, resolution,
                dir_sign, Z_phys,
            )

    def _setup_2d_x(
        self, E_mode, H_mode, center_idx, offset_idx, ny, nx, resolution,
        dir_sign, Z_phys,
    ):
        """2D injection setup for x-propagation."""
        center_y_idx = int(round(self.center[1] / resolution))
        half_width_idx = int(round((self.width / 2) / resolution))
        y_start = max(0, center_y_idx - half_width_idx)
        y_end = min(ny, center_y_idx + half_width_idx)
        y_slice = slice(y_start, y_end)
        self._y_start = y_start
        self._y_end = y_end

        if self.pol == "tm":
            self._ez_indices = (y_slice, center_idx)
            self._h_indices = (y_slice, offset_idx)
            self._h_component = "Hx"

            # +x TM: (Ez, Hy)
            Hy_raw = np.squeeze(H_mode[1])
            Ez_raw = np.squeeze(E_mode[2])

            idx_max = np.argmax(np.abs(Hy_raw))
            phase_ref = np.angle(Hy_raw.flatten()[idx_max])
            Hy_profile = Hy_raw * np.exp(-1j * phase_ref)
            Ez_profile = Ez_raw * np.exp(-1j * phase_ref)

            Ez_profile = _impedance_match_e_profile(Ez_profile, Hy_profile, Z_phys)

            width_cells = y_end - y_start
            window = self._make_1d_window(width_cells)

            Hy_cropped = np.real(Hy_profile)[y_start:y_end]
            Ez_cropped = np.real(Ez_profile)[y_start:y_end]
            if len(Hy_cropped) == len(window):
                Hy_cropped = Hy_cropped * window
                Ez_cropped = Ez_cropped * window

            self._jz_profile = dir_sign * Hy_cropped
            self._my_profile = dir_sign * Ez_cropped

        else:  # TE
            hz_col = (
                max(0, offset_idx - 1) if self.direction == "+x"
                else min(nx - 2, offset_idx)
            )

            self._hz_indices = (slice(y_start, min(y_end, ny - 1)), hz_col)
            self._e_indices = (slice(y_start, min(y_end, ny - 1)), offset_idx)
            self._e_component = "Ey"

            # +x TE: (Ey, Hz)
            Hz_raw = np.squeeze(H_mode[2])
            Ey_raw = np.squeeze(E_mode[1])

            Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
            Ey_staggered = 0.5 * (Ey_raw[:-1] + Ey_raw[1:])

            idx_max = np.argmax(np.abs(Hz_staggered))
            phase_ref = np.angle(Hz_staggered.flatten()[idx_max])
            Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
            Ey_profile = Ey_staggered * np.exp(-1j * phase_ref)

            Ey_profile = _impedance_match_e_profile(Ey_profile, Hz_profile, Z_phys)

            width_cells = min(y_end, len(Hz_profile)) - y_start
            window = self._make_1d_window(width_cells)

            Hz_cropped = np.real(Hz_profile)[y_start : min(y_end, len(Hz_profile))]
            Ey_cropped = np.real(Ey_profile)[y_start : min(y_end, len(Ey_profile))]
            if len(Hz_cropped) == len(window):
                Hz_cropped = Hz_cropped * window
                Ey_cropped = Ey_cropped * window

            # Relative J/M sign controls propagation handedness for TE in x-propagation.
            self._jy_profile = dir_sign * Hz_cropped
            self._mz_profile = -dir_sign * Ey_cropped

    def _setup_2d_y(
        self, E_mode, H_mode, center_idx, offset_idx, ny, nx, resolution,
        dir_sign, Z_phys,
    ):
        """2D injection setup for y-propagation."""
        center_x_idx = int(round(self.center[0] / resolution))
        half_width_idx = int(round((self.width / 2) / resolution))
        x_start = max(0, center_x_idx - half_width_idx)
        x_end = min(nx, center_x_idx + half_width_idx)
        x_slice = slice(x_start, x_end)
        self._x_start = x_start
        self._x_end = x_end

        if self.pol == "tm":
            self._ez_indices = (center_idx, x_slice)
            self._h_indices = (offset_idx, x_slice)
            self._h_component = "Hy"

            # +y TM uses the rotated x-solver basis: (Ez, Hx) <- (Ez, Hy_xbasis)
            Hx_raw = np.squeeze(H_mode[1])
            Ez_raw = np.squeeze(E_mode[2])

            idx_max = np.argmax(np.abs(Hx_raw))
            phase_ref = np.angle(Hx_raw.flatten()[idx_max])
            Hx_profile = Hx_raw * np.exp(-1j * phase_ref)
            Ez_profile = Ez_raw * np.exp(-1j * phase_ref)

            Ez_profile = _impedance_match_e_profile(Ez_profile, Hx_profile, Z_phys)

            width_cells = x_end - x_start
            window = self._make_1d_window(width_cells)

            Hx_cropped = np.real(Hx_profile)[x_start:x_end]
            Ez_cropped = np.real(Ez_profile)[x_start:x_end]
            if len(Hx_cropped) == len(window):
                Hx_cropped = Hx_cropped * window
                Ez_cropped = Ez_cropped * window

            # Relative J/M sign controls propagation handedness for TM in y-propagation.
            self._jz_profile = dir_sign * Hx_cropped
            self._my_profile = -dir_sign * Ez_cropped

        else:  # TE y-prop
            hz_row = (
                max(0, offset_idx - 1) if self.direction == "+y"
                else min(ny - 2, offset_idx)
            )

            self._hz_indices = (hz_row, slice(x_start, min(x_end, nx - 1)))
            self._e_indices = (offset_idx, slice(x_start, min(x_end, nx - 1)))
            self._e_component = "Ex"

            # +y TE uses the rotated x-solver basis: (Ex, Hz) <- (Ey_xbasis, Hz_xbasis)
            Hz_raw = np.squeeze(H_mode[2])
            Ex_raw = np.squeeze(E_mode[1])

            Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
            Ex_staggered = 0.5 * (Ex_raw[:-1] + Ex_raw[1:])

            idx_max = np.argmax(np.abs(Hz_staggered))
            phase_ref = np.angle(Hz_staggered.flatten()[idx_max])
            Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
            Ex_profile = Ex_staggered * np.exp(-1j * phase_ref)

            Ex_profile = _impedance_match_e_profile(Ex_profile, Hz_profile, Z_phys)

            width_cells = min(x_end, len(Hz_profile)) - x_start
            window = self._make_1d_window(width_cells)

            Hz_cropped = np.real(Hz_profile)[x_start : min(x_end, len(Hz_profile))]
            Ex_cropped = np.real(Ex_profile)[x_start : min(x_end, len(Ex_profile))]
            if len(Hz_cropped) == len(window):
                Hz_cropped = Hz_cropped * window
                Ex_cropped = Ex_cropped * window

            self._jx_profile = -dir_sign * Hz_cropped
            self._mz_profile = -dir_sign * Ex_cropped

    @staticmethod
    def _make_1d_window(width_cells, alpha=0.3):
        """Create a 1D Tukey window for smooth edges."""
        if width_cells > 2:
            from scipy.signal.windows import tukey
            return tukey(width_cells, alpha=alpha)
        return np.ones(max(1, width_cells))

    def _compute_dt_physical(self, axis, is_3d, dx, dy):
        """Compute physical time shift between E and H injection planes."""
        if self._neff is None:
            return

        coord_e = 0.0
        coord_h = 0.0

        if is_3d:
            if axis == "x":
                if self._Ez_indices is not None:
                    coord_e = (self._Ez_indices[2] + 0.5) * dx
                if self._Hy_indices is not None:
                    coord_h = (self._Hy_indices[2] + 1.0) * dx
            else:
                if self._Ez_indices is not None:
                    coord_e = (self._Ez_indices[1] + 0.5) * dy
                if self._Hx_indices is not None:
                    coord_h = (self._Hx_indices[1] + 1.0) * dy
        else:
            if axis == "x":
                if self.pol == "tm":
                    idx_e = self._ez_indices[1] if self._ez_indices else 0
                    idx_h = self._h_indices[1] if self._h_indices else 0
                else:
                    idx_e = self._e_indices[1] if self._e_indices else 0
                    idx_h = self._hz_indices[1] if self._hz_indices else 0
                coord_e = (idx_e + 0.5) * dx
                coord_h = (idx_h + 1.0) * dx
            else:
                if self.pol == "tm":
                    idx_e = self._ez_indices[0] if self._ez_indices else 0
                    idx_h = self._h_indices[0] if self._h_indices else 0
                else:
                    idx_e = self._e_indices[0] if self._e_indices else 0
                    idx_h = self._hz_indices[0] if self._hz_indices else 0
                coord_e = (idx_e + 0.5) * dy
                coord_h = (idx_h + 1.0) * dy

        # Use the physical E/H plane separation magnitude for phase delay.
        # Directionality is set by J/M cross-product signs, not by changing this delay sign.
        self._dt_physical = (
            abs(coord_e - coord_h) * float(np.real(self._neff)) / LIGHT_SPEED
        )

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

    def inject_h(self, fields, t, dt, current_step, resolution, design):
        """Inject magnetic current (M) into H-fields after the H update."""
        if (
            (not self._initialized)
            or (self._grid_shape != fields.permittivity.shape)
            or (self._resolution is None)
            or (not np.isclose(self._resolution, resolution))
        ):
            self.initialize(fields.permittivity, resolution)

        # M=-n×E is injected on the H update at the standard half-step time.
        signal_value_h = self._get_signal_value(t + 0.5 * dt, dt)

        if self._Ex_profile is not None and self._is_3d:
            self._inject_3d_h(fields, signal_value_h, dt, resolution)
        else:
            self._inject_2d_h(fields, signal_value_h, dt, resolution)

    def inject_e(self, fields, t, dt, current_step, resolution, design):
        """Inject electric current (J) into E-fields after the E update."""
        if (
            (not self._initialized)
            or (self._grid_shape != fields.permittivity.shape)
            or (self._resolution is None)
            or (not np.isclose(self._resolution, resolution))
        ):
            self.initialize(fields.permittivity, resolution)

        # J=n×H is evaluated on the E update and needs the physical E/H plane offset.
        signal_value_e = self._get_signal_value(t + 0.5 * dt + self._dt_physical, dt)

        if self._Ex_profile is not None and self._is_3d:
            self._inject_3d_e(fields, signal_value_e, dt, resolution)
        else:
            self._inject_2d_e(fields, signal_value_e, dt, resolution)

    def inject(self, fields, t, dt, current_step, resolution, design):
        """Inject source fields (calls inject_h + inject_e for backward compatibility)."""
        self.inject_h(fields, t, dt, current_step, resolution, design)
        self.inject_e(fields, t, dt, current_step, resolution, design)

    # -- 3D injection (split) ------------------------------------------

    def _get_3d_profiles_and_indices(self):
        profiles = {
            "Ex": self._Ex_profile, "Ey": self._Ey_profile, "Ez": self._Ez_profile,
            "Hx": self._Hx_profile, "Hy": self._Hy_profile, "Hz": self._Hz_profile,
        }
        indices = {
            "Ex": self._Ex_indices, "Ey": self._Ey_indices, "Ez": self._Ez_indices,
            "Hx": self._Hx_indices, "Hy": self._Hy_indices, "Hz": self._Hz_indices,
        }
        return profiles, indices

    def _inject_3d_h(self, fields, signal_h, dt, resolution):
        """Inject H-field components for 3D Huygens source."""
        profiles, indices = self._get_3d_profiles_and_indices()
        _inject_3d_h_fields(fields, profiles, indices, signal_h, dt, resolution, self._axis)

    def _inject_3d_e(self, fields, signal_e, dt, resolution):
        """Inject E-field components for 3D Huygens source."""
        profiles, indices = self._get_3d_profiles_and_indices()
        _inject_3d_e_fields(fields, profiles, indices, signal_e, dt, resolution, self._axis)

    # -- 2D injection (split, with corrected signs) ---------------------

    def _inject_2d_h(self, fields, signal_h, dt, resolution):
        """Inject magnetic current into H-fields for 2D (after H update)."""
        if self.pol == "tm":
            if self._h_indices is not None and self._my_profile is not None:
                mu_val = getattr(fields, "permeability", None)
                mu_at_source = mu_val[self._h_indices] if mu_val is not None else 1.0
                my_term = self._my_profile * signal_h / resolution
                h_injection = -my_term * dt / (MU_0 * mu_at_source)

                if self._h_component == "Hx":
                    fields.Hx = fields.Hx.at[self._h_indices].add(h_injection)
                else:
                    fields.Hy = fields.Hy.at[self._h_indices].add(h_injection)
        else:  # TE
            if self._hz_indices is not None and self._mz_profile is not None:
                mu_val = getattr(fields, "permeability", None)
                mu_at_source = mu_val[self._hz_indices] if mu_val is not None else 1.0
                mz_term = self._mz_profile * signal_h / resolution
                hz_injection = +mz_term * dt / (MU_0 * mu_at_source)
                fields.Hz = fields.Hz.at[self._hz_indices].add(hz_injection)

    def _inject_2d_e(self, fields, signal_e, dt, resolution):
        """Inject electric current into E-fields for 2D (after E update)."""
        if self.pol == "tm":
            if self._ez_indices is not None and self._jz_profile is not None:
                eps_at_source = fields.permittivity[self._ez_indices]
                jz_term = self._jz_profile * signal_e / resolution
                ez_injection = +jz_term * dt / (EPS_0 * eps_at_source)
                fields.Ez = fields.Ez.at[self._ez_indices].add(ez_injection)
        else:  # TE
            if self._e_indices is not None:
                j_profile = (
                    self._jx_profile if self._e_component == "Ex" else self._jy_profile
                )
                if j_profile is not None:
                    eps_at_source = fields.permittivity[self._e_indices]
                    j_term = j_profile * signal_e / resolution
                    e_injection = -j_term * dt / (EPS_0 * eps_at_source)

                    if self._e_component == "Ex":
                        fields.Ex = fields.Ex.at[self._e_indices].add(e_injection)
                    else:
                        fields.Ey = fields.Ey.at[self._e_indices].add(e_injection)

    def show(self, field=None):
        """Visualize the 2D mode profile (for 3D simulations) or 1D profile (for 2D)."""
        from beamz.visual.source_plots import show_mode_profile

        show_mode_profile(self, field=field)

    def add_to_plot(
        self, ax, facecolor="none", edgecolor="crimson", alpha=0.8, linestyle="-"
    ):
        """Add source visualization to 2D matplotlib plot."""
        from beamz.visual.overlays import add_mode_source_to_plot

        add_mode_source_to_plot(
            self, ax, facecolor=facecolor, edgecolor=edgecolor,
            alpha=alpha, linestyle=linestyle,
        )
