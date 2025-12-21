# Adapted from FDTDx by Yannik Mahlau
from collections import namedtuple
from types import SimpleNamespace
from typing import List, Literal, Tuple, Union

import numpy as np
import tidy3d
from tidy3d.components.mode.solver import compute_modes as _compute_modes

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

    if pol == "te": numerator = np.sum(np.abs(E1) ** 2)
    elif pol == "tm": numerator = np.sum(np.abs(E2) ** 2)
    else: raise ValueError(f"pol must be 'te' or 'tm', but got {pol}")

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

    matching_sorted = sorted(matching, key=lambda m: float(np.real(m.neff)), reverse=True)
    non_matching_sorted = sorted(non_matching, key=lambda m: float(np.real(m.neff)), reverse=True)

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
    inv_permittivities = np.asarray(inv_permittivities, dtype=np.complex128)
    if inv_permittivities.ndim == 1: inv_permittivities = inv_permittivities[np.newaxis, :, np.newaxis]
    elif inv_permittivities.ndim == 2: inv_permittivities = inv_permittivities[np.newaxis, :, :]
    elif inv_permittivities.ndim > 3: raise ValueError(f"Invalid shape of inv_permittivities: {inv_permittivities.shape}")

    if isinstance(inv_permeabilities, np.ndarray):
        inv_permeabilities = np.asarray(inv_permeabilities, dtype=np.complex128)
        if inv_permeabilities.ndim == 1: inv_permeabilities = inv_permeabilities[np.newaxis, :, np.newaxis]
        elif inv_permeabilities.ndim == 2: inv_permeabilities = inv_permeabilities[np.newaxis, :, :]
        elif inv_permeabilities.ndim > 3: raise ValueError(f"Invalid shape of inv_permeabilities: {inv_permeabilities.shape}")
    else:
        inv_permeabilities = np.asarray(inv_permeabilities, dtype=np.complex128)

    singleton_axes = [idx for idx, size in enumerate(inv_permittivities.shape) if size == 1]
    if not singleton_axes: raise ValueError("At least one singleton dimension is required to denote the propagation axis")
    propagation_axis = singleton_axes[0]

    cross_axes = [ax for ax in range(inv_permittivities.ndim) if ax != propagation_axis]
    if not cross_axes: raise ValueError("Need at least one transverse axis for mode computation")

    permittivities = 1 / inv_permittivities
    coords = [np.arange(permittivities.shape[dim] + 1) * resolution / 1e-6 for dim in cross_axes]
    permittivity_squeezed = np.take(permittivities, indices=0, axis=propagation_axis)
    if permittivity_squeezed.ndim == 1: permittivity_squeezed = permittivity_squeezed[:, np.newaxis]

    if inv_permeabilities.ndim == inv_permittivities.ndim:
        permeability = 1 / inv_permeabilities
        permeability_squeezed = np.take(permeability, indices=0, axis=propagation_axis)
        if permeability_squeezed.ndim == 1: permeability_squeezed = permeability_squeezed[:, np.newaxis]
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
    if mode_index >= len(modes): raise ValueError(f"Requested mode index {mode_index}, but only {len(modes)} modes available")

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
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    if eps.ndim not in [1, 2]: raise ValueError("solve_modes expects a 1D or 2D permittivity array")

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
            if field_line.ndim > 1: field_line = field_line[:, 0]
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

    if not mode_vectors: return neff_array, np.zeros((eps.size, 0), dtype=np.complex128)

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
        mu_cross = [permeability_cross_section if i in {0, 4, 8} else od for i in range(9)]

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

    if num_modes == 1: return [ModeTupleType(Ex=Ex, Ey=Ey, Ez=Ez, Hx=Hx, Hy=Hy, Hz=Hz, neff=complex(neffs))]

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

def _normalize_by_poynting_flux(E: np.ndarray, H: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    S = np.cross(E, np.conjugate(H), axis=0)
    power = float(np.real(np.sum(S[axis])))
    
    # Debug: check which normalization path is taken
    # print(f"[DEBUG normalize] axis={axis}, power={power:.3e}, E_norm={np.linalg.norm(E):.3e}, H_norm={np.linalg.norm(H):.3e}")
    
    # Guard against tiny/negative/NaN power from numerical noise
    if not np.isfinite(power) or abs(power) < 1e-18:
        # Fallback: normalize by field amplitude
        e_norm = float(np.linalg.norm(E))
        if e_norm > 1e-18 and np.isfinite(e_norm):
            # print(f"[DEBUG normalize] Using E-norm fallback: {e_norm:.3e}")
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
