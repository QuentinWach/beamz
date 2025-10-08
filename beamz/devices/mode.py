# Adapted from FDTDx by Yannik Mahlau
from collections import namedtuple
from types import SimpleNamespace
from typing import List, Literal
import numpy as np
from tidy3d.components.mode.solver import compute_modes as _compute_modes


ModeTupleType = namedtuple("Mode", ["neff", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
def compute_mode_polarization_fraction(mode: ModeTupleType, tangential_axes: tuple[int, int], pol: Literal["te", "tm"]) -> float:
    """Mode polarization fraction.

    Args:
        mode (ModeTupleType): a ModeTupleType instance
        tangential_axes (tuple[int, int]): indices of transverse E-field component axes.
        pol (Literal["te", "tm"]): "te" or "tm" determines which axis is 'E1'

    Returns:
        float: Polarization fraction between 0 and 1.
    """
    E_fields = [mode.Ex, mode.Ey, mode.Ez]
    E1 = E_fields[tangential_axes[0]]
    E2 = E_fields[tangential_axes[1]]

    if pol == "te": numerator = np.sum(np.abs(E1) ** 2)
    elif pol == "tm": numerator = np.sum(np.abs(E2) ** 2)
    else: raise ValueError(f"pol must be 'te' or 'tm', but got {pol}")
    
    denominator = np.sum(np.abs(E1) ** 2 + np.abs(E2) ** 2) + 1e-18
    return numerator / denominator


def sort_modes(modes: list[ModeTupleType], filter_pol: Literal["te", "tm"] | None, 
    tangential_axes: tuple[int, int]) -> list[ModeTupleType]:
    """
    Sort modes by polarization.

    Args:
        modes (list[ModeTupleType]): list of modes.
        filter_pol (Literal["te", "tm"] | None): If not none, sort by polarization specificaton.
        tangential_axes (tuple[int, int]): indices of transverse E-field component axes.

    Returns:
        list[ModeTupleType]: sorted list of modes.
    """
    if filter_pol is None: return sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)

    def is_matching(mode):
        frac = compute_mode_polarization_fraction(mode, tangential_axes, filter_pol)
        return frac >= 0.5

    matching = [m for m in modes if is_matching(m)]
    non_matching = [m for m in modes if not is_matching(m)]

    matching_sorted = sorted(matching, key=lambda m: float(np.real(m.neff)), reverse=True)
    non_matching_sorted = sorted(non_matching, key=lambda m: float(np.real(m.neff)), reverse=True)

    return matching_sorted + non_matching_sorted


def compute_mode(frequency: float, inv_permittivities: np.ndarray,
    inv_permeabilities: np.ndarray | float, resolution: float, direction: Literal["+", "-"], mode_index: int = 0,
    filter_pol: Literal["te", "tm"] | None = None) -> tuple[np.ndarray, np.ndarray, complex]:
    """Compute optical modes of a waveguide cross-section.

    This function uses the Tidy3D mode solver to compute the optical modes of a given waveguide cross-section defined
    by its permittivity distribution.

    By default modes are sorted by their effective index. The mode_index argument indexes this sorted list of modes and
    returns the desired mode. With filter_pol, it is also possible to only index a specific polarization.

    Args:
        frequency (float): Operating frequency in Hz
        inv_permittivities (jax.Array): 3D array of inverse relative permittivity values
        inv_permeabilities (jax.Array | float): 3D array of inverse relative permittivity values or single float for
            uniform permeability distribution.
        resolution (float): resolution of the simulation grid in meter. For example a grid spacing of 10nm should be
            given as 10e-9.
        direction (Literal["+", "-"]): Propagation direction, either "+" or "-".
        mode_index (int, optional): Index of the mode to compute. Defaults to 0.
        filter_pol (Literal["te", "tm"] | None, optional). If not None, modes are filtered by polarization.

    Returns:
        Tuple[jax.Array, jax.Array, jax.Array]:
            Tuple of E, H field and the effective index as complex-valued jax arrays.
    """
    # Input validation
    inv_permittivities = np.asarray(inv_permittivities)
    if inv_permittivities.squeeze().ndim != 2: raise Exception(f"Invalid shape of inv_permittivities: {inv_permittivities.shape}")
    if isinstance(inv_permeabilities, np.ndarray) and inv_permeabilities.squeeze().ndim != 2:
        raise Exception(f"Invalid shape of inv_permeabilities: {inv_permeabilities.shape}")

    def mode_helper(permittivity, permeability):
        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency,
            permittivity_cross_section=permittivity,
            permeability_cross_section=permeability,
            coords=coords,
            direction=direction,
            num_modes=2 * (mode_index + 1) + 10,
        )

        # sort modes by polarization
        # tidy3d assumes propagation in the z-direction. The tangential axes are therefore x and y.
        modes = sort_modes(modes, filter_pol, (0, 1))
        mode = modes[mode_index]

        if propagation_axis == 0:
            mode_E, mode_H = (
                np.stack([mode.Ez, mode.Ex, mode.Ey], axis=0).astype(np.complex64),
                np.stack([mode.Hz, mode.Hx, mode.Hy], axis=0).astype(np.complex64),
            )
        elif propagation_axis == 1:
            mode_E, mode_H = (
                np.stack([mode.Ex, mode.Ez, mode.Ey], axis=0).astype(np.complex64),
                -np.stack([mode.Hx, mode.Hz, mode.Hy], axis=0).astype(np.complex64),
            )
        elif propagation_axis == 2:
            mode_E, mode_H = (
                np.stack([mode.Ex, mode.Ey, mode.Ez], axis=0).astype(np.complex64),
                np.stack([mode.Hx, mode.Hy, mode.Hz], axis=0).astype(np.complex64),
            )
        else:
            raise Exception("This should never happen")

        neff = np.asarray(mode.neff, dtype=np.complex128)
        return mode_E, mode_H, neff

    # compute input to tidy3d Mode solver
    permittivities = 1 / inv_permittivities
    permittivities = np.asarray(permittivities, dtype=np.complex128)
    dims = permittivities.shape
    try:
        propagation_axis = next(idx for idx, size in enumerate(dims) if size == 1)
    except StopIteration:
        raise ValueError("Expected a singleton propagation axis in permittivity array")
    other_axes = [ax for ax in range(3) if ax != propagation_axis]
    coords = [np.arange(dims[dim] + 1) * resolution / 1e-6 for dim in other_axes]
    permittivity_squeezed = np.take(permittivities, indices=0, axis=propagation_axis)

    if isinstance(inv_permeabilities, np.ndarray):
        permeability_squeezed = np.take(1 / inv_permeabilities, indices=0, axis=propagation_axis)
    else:
        permeability_squeezed = 1 / inv_permeabilities

    mode_E_raw, mode_H_raw, eff_idx = mode_helper(permittivity_squeezed, permeability_squeezed)
    mode_E = np.expand_dims(mode_E_raw, axis=propagation_axis + 1)
    mode_H = np.expand_dims(mode_H_raw, axis=propagation_axis + 1)

    mode_E_norm, mode_H_norm = _normalize_by_poynting_flux(mode_E, mode_H, axis=propagation_axis)

    return mode_E_norm, mode_H_norm, eff_idx


def tidy3d_mode_computation_wrapper(
    frequency: float,
    permittivity_cross_section: np.ndarray,
    coords: List[np.ndarray],
    direction: Literal["+", "-"],
    permeability_cross_section: np.ndarray | None = None,
    target_neff: float | None = None,
    angle_theta: float = 0.0,
    angle_phi: float = 0.0,
    num_modes: int = 10,
    precision: Literal["single", "double"] = "double",
) -> List[ModeTupleType]:
    """Compute optical modes of a waveguide cross-section.

    This function uses the Tidy3D mode solver to compute the optical modes of a given
    waveguide cross-section defined by its permittivity distribution.

    Args:
        frequency (float): Operating frequency in Hz
        permittivity_cross_section (np.ndarray): 2D array of relative permittivity values
        coords (List[np.ndarray]): List of coordinate arrays [x, y] defining the grid
        direction (Literal["+", "-"], optional): Propagation direction, either "+" or "-"
        permeability_cross_section (np.ndarray | None, optional): 2D array of relative permeability values.
            Defauts to None.
        target_neff (float | None, optional): Target effective index to search around. Defaults to None.
        angle_theta (float, optional): Polar angle in radians. Defaults to 0.0.
        angle_phi (float, optional): Azimuthal angle in radians. Defaults to 0.0.
        num_modes (int, optional): Number of modes to compute. Defaults to 10.
        precision (Literal["single", "double"], optional): Numerical precision. Defaults to "double".

    Notes:
        tidy3d assumes propagation in z-direction. The output fields should be handled accordingly.

    Returns:
        List[ModeTupleType]: List of computed modes sorted by decreasing real part of
            effective index. Each mode contains the field components and effective index.
    """
    # see https://docs.flexcompute.com/projects/tidy3d/en/latest/_autosummary/tidy3d.ModeSpec.html#tidy3d.ModeSpec
    mode_spec = SimpleNamespace(
        # Note that the filter_pol argument is not used here since it does not work from tidy3d
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
    eps_cross = [
        permittivity_cross_section,
        od,
        od,
        od,
        permittivity_cross_section,
        od,
        od,
        od,
        permittivity_cross_section,
    ]
    mu_cross = None
    if permeability_cross_section is not None:
        mu_cross = [
            permeability_cross_section,
            od,
            od,
            od,
            permeability_cross_section,
            od,
            od,
            od,
            permeability_cross_section,
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
    ((Ex, Ey, Ez), (Hx, Hy, Hz)) = EH.squeeze()

    if num_modes == 1:
        modes = [
            ModeTupleType(
                Ex=Ex,
                Ey=Ey,
                Ez=Ez,
                Hx=Hx,
                Hy=Hy,
                Hz=Hz,
                neff=float(neffs.real) + 1j * float(neffs.imag),
            )
            for _ in range(num_modes)
        ]
    else:
        modes = [
            ModeTupleType(
                Ex=Ex[..., i],
                Ey=Ey[..., i],
                Ez=Ez[..., i],
                Hx=Hx[..., i],
                Hy=Hy[..., i],
                Hz=Hz[..., i],
                neff=neffs[i],
            )
            for i in range(num_modes)
        ]
    return modes


def _normalize_by_poynting_flux(E: np.ndarray, H: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalize fields so the power flow along the propagation axis equals 1."""

    # Poynting vector components (assuming time-harmonic convention exp(-iωt))
    S = np.cross(E, np.conjugate(H), axis=0)
    power = np.real(np.sum(S[axis]))
    if power == 0.0:
        return E, H

    scale = np.sqrt(power)
    return E / scale, H / scale