# Adapted from FDTDx by Yannik Mahlau
from collections import namedtuple
from types import SimpleNamespace
from typing import List, Literal
import numpy as np
from tidy3d.components.mode.solver import compute_modes as _compute_modes

ModeTupleType = namedtuple("Mode", ["neff", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
"""A named tuple containing the mode fields and effective index."""


def compute_mode_polarization_fraction(mode: ModeTupleType, tangential_axes: tuple[int, int], pol: Literal["te", "tm"]) -> float:
    numerator = np.sum(np.abs(mode.Ex if tangential_axes[0] == 0 else mode.Ey) ** 2) if pol == "te" else np.sum(np.abs(mode.Ex if tangential_axes[1] == 0 else mode.Ey) ** 2)
    denominator = np.sum(np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2) + 1e-18
    return numerator / denominator


def sort_modes(modes: list[ModeTupleType], filter_pol: Literal["te", "tm"] | None, tangential_axes: tuple[int, int]) -> list[ModeTupleType]:
    if filter_pol is None:
        return sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)

    def is_matching(mode: ModeTupleType) -> bool:
        return compute_mode_polarization_fraction(mode, tangential_axes, filter_pol) >= 0.5

    matching = [m for m in modes if is_matching(m)]
    non_matching = [m for m in modes if not is_matching(m)]
    return (
        sorted(matching, key=lambda m: float(np.real(m.neff)), reverse=True)
        + sorted(non_matching, key=lambda m: float(np.real(m.neff)), reverse=True)
    )


def compute_mode(
    frequency: float,
    inv_permittivities: np.ndarray,
    inv_permeabilities: np.ndarray | float,
    resolution: float,
    direction: Literal["+", "-"],
    mode_index: int = 0,
    filter_pol: Literal["te", "tm"] | None = None,
) -> tuple[np.ndarray, np.ndarray, complex]:
    inv_permittivities = np.asarray(inv_permittivities)
    if inv_permittivities.squeeze().ndim != 2:
        raise ValueError(f"Invalid shape of inv_permittivities: {inv_permittivities.shape}")
    if isinstance(inv_permeabilities, np.ndarray) and inv_permeabilities.squeeze().ndim != 2:
        raise ValueError(f"Invalid shape of inv_permeabilities: {inv_permeabilities.shape}")

    permittivities = 1 / inv_permittivities.astype(np.complex128)
    try:
        propagation_axis = next(idx for idx, size in enumerate(permittivities.shape) if size == 1)
    except StopIteration as exc:
        raise ValueError("Expected a singleton propagation axis in permittivity array") from exc

    other_axes = [ax for ax in range(3) if ax != propagation_axis]
    coords = [np.linspace(0.0, (permittivities.shape[dim]) * resolution / 1e-6, permittivities.shape[dim] + 1) for dim in other_axes]
    permittivity_squeezed = np.take(permittivities, indices=0, axis=propagation_axis)
    permeability_squeezed = 1 / np.asarray(inv_permeabilities, dtype=np.complex128)
    if isinstance(inv_permeabilities, np.ndarray):
        permeability_squeezed = np.take(permeability_squeezed, indices=0, axis=propagation_axis)

    modes = tidy3d_mode_computation_wrapper(
        frequency=frequency,
        permittivity_cross_section=permittivity_squeezed,
        permeability_cross_section=permeability_squeezed,
        coords=coords,
        direction=direction,
        num_modes=2 * (mode_index + 1) + 5,
    )

    modes = sort_modes(modes, filter_pol, (0, 1))
    if mode_index >= len(modes):
        raise ValueError(f"Requested mode index {mode_index} but only {len(modes)} modes were computed")

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

    E_norm, H_norm = _normalize_by_poynting_flux(E, H, axis=propagation_axis)
    return E_norm, H_norm, np.asarray(mode.neff, dtype=np.complex128)


def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 2,
    direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+x",
    filter_pol: Literal["te", "tm", None] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if eps.ndim != 1:
        raise ValueError("solve_modes expects a 1D permittivity array")

    freq = omega / (2 * np.pi)
    inv_eps = (1.0 / np.asarray(eps, dtype=np.complex128)).reshape(1, eps.size, 1)
    direction_flag = "+" if direction.startswith("+") else "-"

    neffs: list[complex] = []
    mode_vectors: list[np.ndarray] = []

    for mode_index in range(m):
        E_full, _H_full, neff = compute_mode(
            frequency=freq,
            inv_permittivities=inv_eps,
            inv_permeabilities=1.0,
            resolution=dL,
            direction=direction_flag,
            mode_index=mode_index,
            filter_pol=filter_pol,
        )

        Ez_line = np.squeeze(E_full[direction_flag == "+"])
        if Ez_line.ndim > 1:
            Ez_line = Ez_line[:, 0]

        neffs.append(neff)
        mode_vectors.append(Ez_line)

    return np.asarray(neffs, dtype=np.complex128), np.column_stack(mode_vectors)


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

    if num_modes == 1:
        return [ModeTupleType(Ex=Ex, Ey=Ey, Ez=Ez, Hx=Hx, Hy=Hy, Hz=Hz, neff=complex(neffs))]

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
    power = np.real(np.sum(S[axis]))
    if power == 0.0:
        return E, H
    scale = np.sqrt(power)
    return E / scale, H / scale