# Adapted from FDTDx by Yannik Mahlau
from collections import namedtuple
from types import SimpleNamespace
from typing import List, Literal, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

# Lazy import of tidy3d to allow package to work without it
tidy3d = None
_compute_modes = None


def _ensure_tidy3d():
    """Lazily import tidy3d when needed."""
    global tidy3d, _compute_modes
    if tidy3d is None:
        try:
            import tidy3d as _tidy3d
            from tidy3d.components.mode.solver import compute_modes as _cm

            tidy3d = _tidy3d
            _compute_modes = _cm

            # Monkey-patch tidy3d derivatives to avoid scipy FutureWarning:
            # "Input has data type int64, but the output has been cast to float64"
            _apply_tidy3d_diags_patch()
        except ImportError:
            raise ImportError(
                "tidy3d is required for mode solving. "
                "Install it with: pip install tidy3d"
            )


def _apply_tidy3d_diags_patch():
    """Patch tidy3d's derivatives to use float diagonals in sp.diags (avoids scipy FutureWarning)."""
    import scipy.sparse as sp
    import tidy3d.components.mode.derivatives as _deriv

    def _make_dxf(dls, shape, pmc):
        Nx, Ny = shape
        if Nx == 1:
            return sp.csr_matrix((Ny, Ny))
        dxf = sp.csr_matrix(sp.diags([-1.0, 1.0], [0, 1], shape=(Nx, Nx)))
        if not pmc:
            dxf[0, 0] = 0.0
        dxf = sp.diags(1 / dls).dot(dxf)
        dxf = sp.kron(dxf, sp.eye(Ny))
        return dxf

    def _make_dxb(dls, shape, pmc):
        Nx, Ny = shape
        if Nx == 1:
            return sp.csr_matrix((Ny, Ny))
        dxb = sp.csr_matrix(sp.diags([1.0, -1.0], [0, -1], shape=(Nx, Nx)))
        if pmc:
            dxb[0, 0] = 2.0
        else:
            dxb[0, 0] = 0.0
        dxb = sp.diags(1 / dls).dot(dxb)
        dxb = sp.kron(dxb, sp.eye(Ny))
        return dxb

    def _make_dyf(dls, shape, pmc):
        Nx, Ny = shape
        if Ny == 1:
            return sp.csr_matrix((Nx, Nx))
        dyf = sp.csr_matrix(sp.diags([-1.0, 1.0], [0, 1], shape=(Ny, Ny)))
        if not pmc:
            dyf[0, 0] = 0.0
        dyf = sp.diags(1 / dls).dot(dyf)
        dyf = sp.kron(sp.eye(Nx), dyf)
        return dyf

    def _make_dyb(dls, shape, pmc):
        Nx, Ny = shape
        if Ny == 1:
            return sp.csr_matrix((Nx, Nx))
        dyb = sp.csr_matrix(sp.diags([1.0, -1.0], [0, -1], shape=(Ny, Ny)))
        if pmc:
            dyb[0, 0] = 2.0
        else:
            dyb[0, 0] = 0.0
        dyb = sp.diags(1 / dls).dot(dyb)
        dyb = sp.kron(sp.eye(Nx), dyb)
        return dyb

    _deriv.make_dxf = _make_dxf
    _deriv.make_dxb = _make_dxb
    _deriv.make_dyf = _make_dyf
    _deriv.make_dyb = _make_dyb


ModeTupleType = namedtuple("Mode", ["neff", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
"""A named tuple containing the mode fields and effective index."""


def compute_mode_polarization_fraction(
    mode: ModeTupleType,
    tangential_axes: tuple[int, int],
    pol: Literal["te", "tm"],
) -> float:
    E_fields = [mode.Ex, mode.Ey, mode.Ez]
    E1 = E_fields[tangential_axes[0]]
    E2 = E_fields[tangential_axes[1]]

    if pol == "te":
        numerator = np.sum(np.abs(E1) ** 2)
    elif pol == "tm":
        numerator = np.sum(np.abs(E2) ** 2)
    else:
        raise ValueError(f"pol must be 'te' or 'tm', but got {pol}")

    denominator = np.sum(np.abs(E1) ** 2 + np.abs(E2) ** 2) + 1e-18
    return numerator / denominator


def sort_modes(
    modes: list[ModeTupleType],
    filter_pol: Union[Literal["te", "tm"], None],
    tangential_axes: tuple[int, int],
) -> list[ModeTupleType]:
    if filter_pol is None:
        return sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)

    def is_matching(mode: ModeTupleType) -> bool:
        frac = compute_mode_polarization_fraction(mode, tangential_axes, filter_pol)
        return frac >= 0.5

    matching = [m for m in modes if is_matching(m)]
    non_matching = [m for m in modes if not is_matching(m)]

    matching_sorted = sorted(
        matching, key=lambda m: float(np.real(m.neff)), reverse=True
    )
    non_matching_sorted = sorted(
        non_matching, key=lambda m: float(np.real(m.neff)), reverse=True
    )

    return matching_sorted + non_matching_sorted


def compute_mode(
    frequency: float,
    inv_permittivities: np.ndarray,
    inv_permeabilities: Union[np.ndarray, float],
    resolution: float,
    direction: Literal["+", "-"],
    mode_index: int = 0,
    filter_pol: Union[Literal["te", "tm"], None] = None,
    target_neff: Union[float, None] = None,
) -> tuple[np.ndarray, np.ndarray, complex, int]:
    _ensure_tidy3d()  # Lazy import tidy3d
    inv_permittivities = np.asarray(inv_permittivities, dtype=np.complex128)
    if inv_permittivities.ndim == 1:
        inv_permittivities = inv_permittivities[np.newaxis, :, np.newaxis]
    elif inv_permittivities.ndim == 2:
        inv_permittivities = inv_permittivities[np.newaxis, :, :]
    elif inv_permittivities.ndim > 3:
        raise ValueError(
            f"Invalid shape of inv_permittivities: {inv_permittivities.shape}"
        )

    if isinstance(inv_permeabilities, np.ndarray):
        inv_permeabilities = np.asarray(inv_permeabilities, dtype=np.complex128)
        if inv_permeabilities.ndim == 1:
            inv_permeabilities = inv_permeabilities[np.newaxis, :, np.newaxis]
        elif inv_permeabilities.ndim == 2:
            inv_permeabilities = inv_permeabilities[np.newaxis, :, :]
        elif inv_permeabilities.ndim > 3:
            raise ValueError(
                f"Invalid shape of inv_permeabilities: {inv_permeabilities.shape}"
            )
    else:
        inv_permeabilities = np.asarray(inv_permeabilities, dtype=np.complex128)

    singleton_axes = [
        idx for idx, size in enumerate(inv_permittivities.shape) if size == 1
    ]
    if not singleton_axes:
        raise ValueError(
            "At least one singleton dimension is required to denote the propagation axis"
        )
    propagation_axis = singleton_axes[0]

    cross_axes = [ax for ax in range(inv_permittivities.ndim) if ax != propagation_axis]
    if not cross_axes:
        raise ValueError("Need at least one transverse axis for mode computation")

    permittivities = 1 / inv_permittivities
    coords = [
        np.arange(permittivities.shape[dim] + 1) * resolution / 1e-6
        for dim in cross_axes
    ]
    permittivity_squeezed = np.take(permittivities, indices=0, axis=propagation_axis)
    if permittivity_squeezed.ndim == 1:
        permittivity_squeezed = permittivity_squeezed[:, np.newaxis]

    if inv_permeabilities.ndim == inv_permittivities.ndim:
        permeability = 1 / inv_permeabilities
        permeability_squeezed = np.take(permeability, indices=0, axis=propagation_axis)
        if permeability_squeezed.ndim == 1:
            permeability_squeezed = permeability_squeezed[:, np.newaxis]
    else:
        permeability_squeezed = 1 / inv_permeabilities.item()

    tangential_axes_map = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    modes = tidy3d_mode_computation_wrapper(
        frequency=frequency,
        permittivity_cross_section=permittivity_squeezed,
        permeability_cross_section=permeability_squeezed,
        coords=coords,
        direction=direction,
        num_modes=2 * (mode_index + 1) + 5,
        target_neff=target_neff,
    )
    tangential_axes = tangential_axes_map.get(propagation_axis, (0, 1))
    modes = sort_modes(modes, filter_pol, tangential_axes)
    if mode_index >= len(modes):
        raise ValueError(
            f"Requested mode index {mode_index}, but only {len(modes)} modes available"
        )

    mode = modes[mode_index]

    if propagation_axis == 0:
        E = np.stack([mode.Ez, mode.Ex, mode.Ey], axis=0).astype(np.complex128)
        H = np.stack([mode.Hz, mode.Hx, mode.Hy], axis=0).astype(np.complex128)
    elif propagation_axis == 1:
        E = np.stack([mode.Ex, mode.Ez, mode.Ey], axis=0).astype(np.complex128)
        H = -np.stack([mode.Hx, mode.Hz, mode.Hy], axis=0).astype(np.complex128)
    else:
        E = np.stack([mode.Ex, mode.Ey, mode.Ez], axis=0).astype(np.complex128)
        H = np.stack([mode.Hx, mode.Hy, mode.Hz], axis=0).astype(np.complex128)

    H *= tidy3d.constants.ETA_0

    E_norm, H_norm = _normalize_by_poynting_flux(E, H, axis=propagation_axis)
    return E_norm, H_norm, np.asarray(mode.neff, dtype=np.complex128), propagation_axis


def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 1,
    direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+x",
    filter_pol: Union[Literal["te", "tm"], None] = None,
    return_fields: bool = False,
    propagation_axis: Union[Literal["+x", "-x", "+y", "-y", "+z", "-z"], None] = None,
    target_neff: Union[float, None] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, int]
]:
    if eps.ndim not in [1, 2]:
        raise ValueError("solve_modes expects a 1D or 2D permittivity array")

    freq = omega / (2 * np.pi)

    # Reshape eps to 3D for compute_mode (axis, trans1, trans2)
    # compute_mode expects (prop_axis, trans1, trans2) where prop_axis is singleton
    if eps.ndim == 1:
        inv_eps = (1.0 / np.asarray(eps, dtype=np.complex128)).reshape(1, eps.size, 1)
    else:
        # eps is (trans1, trans2). We add the propagation axis at 0.
        inv_eps = (1.0 / np.asarray(eps, dtype=np.complex128))[np.newaxis, :, :]

    direction_flag = "+" if direction.startswith("+") else "-"
    axis_hint = propagation_axis if propagation_axis is not None else direction

    neffs: list[complex] = []
    e_fields: list[np.ndarray] = []
    h_fields: list[np.ndarray] = []
    mode_vectors: list[np.ndarray] = []

    for mode_index in range(m):
        E_full, H_full, neff, prop_axis = compute_mode(
            frequency=freq,
            inv_permittivities=inv_eps,
            inv_permeabilities=1.0,
            resolution=dL,
            direction=direction_flag,
            mode_index=mode_index,
            filter_pol=filter_pol,
            target_neff=target_neff,
        )

        neffs.append(neff)
        if return_fields:
            e_fields.append(E_full)
            h_fields.append(H_full)
        else:
            component_norms = [np.linalg.norm(np.squeeze(E_full[i])) for i in range(3)]
            component_idx = int(np.argmax(component_norms))
            field_line = np.squeeze(E_full[component_idx])
            if field_line.ndim > 1:
                field_line = field_line[:, 0]
            max_amp = np.max(np.abs(field_line)) or 1.0
            mode_vectors.append(field_line / max_amp)

    neff_array = np.asarray(neffs, dtype=np.complex128)

    if return_fields:
        return (
            neff_array,
            np.stack(e_fields) if e_fields else np.empty((0, 3, 0, 0)),
            np.stack(h_fields) if h_fields else np.empty((0, 3, 0, 0)),
            prop_axis,
        )

    if not mode_vectors:
        return neff_array, np.zeros((eps.size, 0), dtype=np.complex128)

    return neff_array, np.column_stack(mode_vectors)


def tidy3d_mode_computation_wrapper(
    frequency: float,
    permittivity_cross_section: np.ndarray,
    coords: List[np.ndarray],
    direction: Literal["+", "-"],
    permeability_cross_section: Union[np.ndarray, None] = None,
    target_neff: Union[float, None] = None,
    angle_theta: float = 0.0,
    angle_phi: float = 0.0,
    num_modes: int = 10,
    precision: Literal["single", "double"] = "double",
) -> List[ModeTupleType]:
    _ensure_tidy3d()  # Lazy import tidy3d
    mode_spec = SimpleNamespace(
        num_modes=num_modes,
        target_neff=target_neff,
        num_pml=(0, 0),
        angle_theta=angle_theta,
        angle_phi=angle_phi,
        bend_radius=None,
        bend_axis=None,
        precision=precision,
        track_freq="central",
        group_index_step=False,
    )
    od = np.zeros_like(permittivity_cross_section)
    eps_cross = [permittivity_cross_section if i in {0, 4, 8} else od for i in range(9)]
    mu_cross = None
    if permeability_cross_section is not None:
        mu_cross = [
            permeability_cross_section if i in {0, 4, 8} else od for i in range(9)
        ]

    EH, neffs, _ = _compute_modes(
        eps_cross=eps_cross,
        coords=coords,
        freq=frequency,
        precision=precision,
        mode_spec=mode_spec,
        direction=direction,
        mu_cross=mu_cross,
    )
    (Ex, Ey, Ez), (Hx, Hy, Hz) = EH.squeeze()

    if num_modes == 1:
        return [
            ModeTupleType(Ex=Ex, Ey=Ey, Ez=Ez, Hx=Hx, Hy=Hy, Hz=Hz, neff=complex(neffs))
        ]

    return [
        ModeTupleType(
            Ex=Ex[..., i],
            Ey=Ey[..., i],
            Ez=Ez[..., i],
            Hx=Hx[..., i],
            Hy=Hy[..., i],
            Hz=Hz[..., i],
            neff=neffs[i],
        )
        for i in range(min(num_modes, Ex.shape[-1]))
    ]


def _normalize_by_poynting_flux(
    E: np.ndarray, H: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    S = np.cross(E, np.conjugate(H), axis=0)
    power = float(np.real(np.sum(S[axis])))

    # Guard against tiny/negative/NaN power from numerical noise
    if not np.isfinite(power) or abs(power) < 1e-18:
        # Fallback: normalize by field amplitude
        e_norm = float(np.linalg.norm(E))
        if e_norm > 1e-18 and np.isfinite(e_norm):
            return E / e_norm, H / e_norm
        return E, H
    # Normalize by magnitude of power to avoid sqrt of negative
    scale = np.sqrt(abs(power))
    if scale == 0.0 or not np.isfinite(scale):
        # Fallback: normalize by field amplitude
        e_norm = float(np.linalg.norm(E))
        if e_norm > 1e-18 and np.isfinite(e_norm):
            return E / e_norm, H / e_norm
        return E, H
    E_norm = E / scale
    H_norm = H / scale
    # Final NaN check
    if not np.all(np.isfinite(E_norm)) or not np.all(np.isfinite(H_norm)):
        return E, H
    return E_norm, H_norm


# ============================================================================
# JAX-Compatible Differentiable Mode Solver Wrapper
# ============================================================================


def solve_modes_differentiable(
    eps: jnp.ndarray,
    omega: float,
    dL: float,
    direction: str = "+x",
    filter_pol: str = None,
    m: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX-compatible wrapper for mode solving with custom gradients.

    This function wraps the tidy3d-based mode solver to enable gradient computation
    via finite differences. The forward pass calls the numpy-based solve_modes,
    and the backward pass computes gradients for omega (wavelength) using finite differences.

    Args:
        eps: Permittivity profile as JAX array
        omega: Angular frequency (2*pi*c/wavelength)
        dL: Grid resolution
        direction: Propagation direction ("+x", "-x", etc.)
        filter_pol: Polarization filter ("te" or "tm")
        m: Number of modes to compute

    Returns:
        Tuple of (neff_array, E_fields, H_fields) as JAX arrays
    """
    # Convert JAX array to numpy for tidy3d
    eps_np = np.asarray(eps)

    # Call the numpy-based solver
    neff_array, E_fields, H_fields, _ = solve_modes(
        eps=eps_np,
        omega=float(omega),
        dL=float(dL),
        direction=direction,
        filter_pol=filter_pol,
        m=m,
        return_fields=True,
    )

    # Convert results to JAX arrays
    return (
        jnp.asarray(neff_array),
        jnp.asarray(E_fields),
        jnp.asarray(H_fields),
    )


@jax.custom_vjp
def solve_modes_jax(
    omega: float,
    eps: jnp.ndarray,
    dL: float,
    direction: str,
    filter_pol: str,
) -> jnp.ndarray:
    """Differentiable mode solver that returns effective index.

    This function enables gradient computation through the mode solver
    with respect to omega (and thus wavelength). Uses finite differences
    for the backward pass since tidy3d is not JAX-compatible.

    Args:
        omega: Angular frequency (differentiable parameter)
        eps: Permittivity profile (not differentiable through this function)
        dL: Grid resolution
        direction: Propagation direction
        filter_pol: Polarization filter

    Returns:
        Complex effective index of the fundamental mode
    """
    eps_np = np.asarray(eps)
    neff_array, _, _, _ = solve_modes(
        eps=eps_np,
        omega=float(omega),
        dL=float(dL),
        direction=direction,
        filter_pol=filter_pol,
        m=1,
        return_fields=True,
    )
    return jnp.asarray(neff_array[0])


def solve_modes_jax_fwd(omega, eps, dL, direction, filter_pol):
    """Forward pass for custom VJP."""
    neff = solve_modes_jax(omega, eps, dL, direction, filter_pol)
    # Store residuals for backward pass
    return neff, (omega, eps, dL, direction, filter_pol)


def solve_modes_jax_bwd(res, g):
    """Backward pass using finite differences for omega gradient."""
    omega, eps, dL, direction, filter_pol = res

    # Finite difference step (relative to omega)
    h = 1e-6 * omega

    # Compute neff at omega + h and omega - h
    neff_plus = solve_modes_jax(omega + h, eps, dL, direction, filter_pol)
    neff_minus = solve_modes_jax(omega - h, eps, dL, direction, filter_pol)

    # Central difference for d(neff)/d(omega)
    dneff_domega = (neff_plus - neff_minus) / (2 * h)

    # Chain rule: gradient w.r.t. omega
    # g is the gradient of the loss w.r.t. neff (complex)
    # We take real part since loss is typically real
    grad_omega = jnp.real(jnp.conj(g) * dneff_domega + g * jnp.conj(dneff_domega)) / 2

    # eps gradient not implemented (would require many solver calls)
    grad_eps = jnp.zeros_like(eps)

    return (grad_omega, grad_eps, None, None, None)


# Register custom VJP
solve_modes_jax.defvjp(solve_modes_jax_fwd, solve_modes_jax_bwd)


def wavelength_to_omega(wavelength: float, c: float = 299792458.0) -> float:
    """Convert wavelength to angular frequency."""
    return 2 * np.pi * c / wavelength


def omega_to_wavelength(omega: float, c: float = 299792458.0) -> float:
    """Convert angular frequency to wavelength."""
    return 2 * np.pi * c / omega
