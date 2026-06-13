from collections import namedtuple
from typing import Literal, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

# Lazy import of micromode to allow package to work without it
micromode = None


def _ensure_micromode():
    """Lazily import micromode when needed."""
    global micromode
    if micromode is None:
        try:
            import micromode as _micromode

            micromode = _micromode
        except ImportError:
            raise ImportError(
                "micromode is required for mode solving. "
                "Install it with: pip install micromode"
            )


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
        # Common convention:
        # - TE: dominant in-plane transverse E component for the selected propagation axis.
        # - TM: dominant orthogonal transverse E component.
        #
        # For +x propagation this maps to TE~Ey and TM~Ez.
        numerator = np.sum(np.abs(E1) ** 2)
    elif pol == "tm":
        numerator = np.sum(np.abs(E2) ** 2)
    else:
        raise ValueError(f"pol must be 'te' or 'tm', but got {pol}")

    denominator = np.sum(np.abs(E1) ** 2 + np.abs(E2) ** 2) + 1e-18
    return numerator / denominator


def _field_plane(data_array, normal_axis: int, mode_index: int) -> np.ndarray:
    """Extract a BeamZ transverse field plane from a micromode field component."""
    normal_dim = ("x", "y", "z")[normal_axis]
    selected = data_array.isel(f=0, mode_index=mode_index)
    normal_position = selected.dims.index(normal_dim)
    return np.take(np.asarray(selected.values), indices=0, axis=normal_position)


def _remap_mode_tuple_to_global(
    mode: ModeTupleType,
    local_axis_to_global: tuple[int, int, int],
) -> ModeTupleType:
    """Map micromode local-axis component labels to BEAMZ global Cartesian labels."""

    def _remap_components(components):
        out = [None, None, None]
        for local_axis, global_axis in enumerate(local_axis_to_global):
            out[int(global_axis)] = components[int(local_axis)]
        if any(component is None for component in out):
            raise ValueError(
                f"Invalid local-to-global axis mapping: {local_axis_to_global!r}"
            )
        return out

    Ex, Ey, Ez = _remap_components((mode.Ex, mode.Ey, mode.Ez))
    Hx, Hy, Hz = _remap_components((mode.Hx, mode.Hy, mode.Hz))
    transform = np.zeros((3, 3), dtype=float)
    for local_axis, global_axis in enumerate(local_axis_to_global):
        transform[int(global_axis), int(local_axis)] = 1.0
    axial_sign = float(round(np.linalg.det(transform)))
    return ModeTupleType(
        neff=mode.neff,
        Ex=Ex,
        Ey=Ey,
        Ez=Ez,
        Hx=axial_sign * Hx,
        Hy=axial_sign * Hy,
        Hz=axial_sign * Hz,
    )


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
    local_axis_to_global: tuple[int, int, int] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray, complex, int]:
    _ensure_micromode()  # Lazy import micromode
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
    is_1d_profile = len(singleton_axes) >= 2

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

    if np.ndim(permeability_squeezed) == 0:
        mu_xx = np.full_like(permittivity_squeezed, permeability_squeezed)
    else:
        mu_xx = np.asarray(permeability_squeezed, dtype=np.complex128)

    result = micromode.solve_grid(
        eps_xx=permittivity_squeezed,
        mu_xx=mu_xx,
        x_edges=coords[0],
        y_edges=coords[1],
        freqs=[frequency],
        direction=direction,
        num_modes=2 * (mode_index + 1) + 5,
        target_neff=target_neff,
        normal_axis=propagation_axis,
    )
    modes = []
    for idx in range(result.n_complex.shape[1]):
        local_mode = ModeTupleType(
            neff=result.n_complex.values[0, idx],
            Ex=_field_plane(result.field_components["Ex"], propagation_axis, idx),
            Ey=_field_plane(result.field_components["Ey"], propagation_axis, idx),
            Ez=_field_plane(result.field_components["Ez"], propagation_axis, idx),
            Hx=_field_plane(result.field_components["Hx"], propagation_axis, idx),
            Hy=_field_plane(result.field_components["Hy"], propagation_axis, idx),
            Hz=_field_plane(result.field_components["Hz"], propagation_axis, idx),
        )
        modes.append(_remap_mode_tuple_to_global(local_mode, local_axis_to_global))
    tangential_axes = tuple(ax for ax in range(3) if ax != propagation_axis)
    modes = sort_modes(modes, filter_pol, tangential_axes)
    if mode_index >= len(modes):
        raise ValueError(
            f"Requested mode index {mode_index}, but only {len(modes)} modes available"
        )

    mode = modes[mode_index]

    E = np.stack([mode.Ex, mode.Ey, mode.Ez], axis=0).astype(np.complex128)
    H = np.stack([mode.Hx, mode.Hy, mode.Hz], axis=0).astype(np.complex128)
    if propagation_axis == 1:
        H = -H
    if is_1d_profile:
        E = E[..., 0]
        H = H[..., 0]

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
    if npml < 0:
        raise ValueError("npml must be non-negative")

    freq = omega / (2 * np.pi)
    axis_hint = propagation_axis if propagation_axis is not None else direction
    axis_char = (
        str(axis_hint)[1] if str(axis_hint).startswith(("+", "-")) else str(axis_hint)
    )
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis_char)
    if axis_index is None:
        raise ValueError(
            f"Unsupported propagation axis '{axis_hint}'. Use one of ±x/±y/±z."
        )

    # Reshape eps to 3D for compute_mode (axis, trans1, trans2)
    # compute_mode expects (prop_axis, trans1, trans2) where prop_axis is singleton
    local_axis_to_global = (0, 1, 2)
    if eps.ndim == 1:
        line = np.asarray(eps, dtype=np.complex128)
        if axis_index == 0:
            inv_eps = (1.0 / line).reshape(1, line.size, 1)
            local_axis_to_global = (0, 1, 2)
        elif axis_index == 1:
            inv_eps = (1.0 / line).reshape(line.size, 1, 1)
            local_axis_to_global = (0, 1, 2)
        else:
            inv_eps = (1.0 / line).reshape(line.size, 1, 1)
            local_axis_to_global = (0, 1, 2)
    else:
        eps_arr = 1.0 / np.asarray(eps, dtype=np.complex128)
        if axis_index == 0:
            inv_eps = eps_arr[np.newaxis, :, :]
            local_axis_to_global = (0, 2, 1)
        elif axis_index == 1:
            inv_eps = eps_arr[:, np.newaxis, :]
            local_axis_to_global = (2, 1, 0)
        else:
            inv_eps = eps_arr[:, :, np.newaxis]
            local_axis_to_global = (1, 0, 2)

    direction_flag = "+" if direction.startswith("+") else "-"

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
            local_axis_to_global=local_axis_to_global,
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

    This function wraps the micromode-based mode solver to enable gradient computation
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
    # Convert JAX array to numpy for micromode
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
    for the backward pass since micromode is not JAX-compatible.

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
