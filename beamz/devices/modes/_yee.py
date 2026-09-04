"""Fixed-beta refinement on BeamZ's x-normal Yee cross section."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

_C0 = 299_792_458.0
_EPS0 = 8.854_187_812_8e-12
_MU0 = 1.256_637_062_12e-6

_E_COMPONENTS = ("Ex", "Ey", "Ez")
_H_COMPONENTS = ("Hx", "Hy", "Hz")


def refine_x_mode_at_fixed_beta(
    profiles,
    indices,
    *,
    component_permittivity,
    component_permeability,
    omega,
    dt,
    resolution,
    k_num,
    direction_sign,
    transverse_cell_widths=None,
    _correct_beta=True,
):
    """Refine E at the selected beta, then reconstruct H from Faraday's law."""
    from scipy import sparse
    from scipy.sparse import linalg as spla

    out = {
        name: np.asarray(value, dtype=np.complex128) for name, value in profiles.items()
    }
    nz, ny = out["Ex"].shape
    d = float(resolution)
    sign = 1.0 if float(direction_sign) >= 0.0 else -1.0
    delta = -2j * sign * np.sin(0.5 * float(k_num) * d) / d
    curl_e, curl_h, curl_e_beta, curl_h_beta = _yee_curl_operators(
        sparse,
        out,
        resolution=d,
        delta=delta,
        transverse_cell_widths=transverse_cell_widths,
    )
    eps, mu = _component_material_vectors(
        out,
        indices,
        component_permittivity=component_permittivity,
        component_permeability=component_permeability,
    )
    ex_n, ey_n, _ez_n = (out[name].size for name in _E_COMPONENTS)
    hx_n, hy_n, _hz_n = (out[name].size for name in _H_COMPONENTS)
    operator = curl_h @ sparse.diags(1.0 / mu) @ curl_e
    mass = sparse.diags(eps)
    omega_d = (
        float(omega)
        if dt is None
        else 2.0 * np.sin(0.5 * float(omega) * float(dt)) / float(dt)
    )
    target = (omega_d / _C0) ** 2
    seed = np.concatenate([out[name].reshape(-1) for name in ("Ex", "Ey", "Ez")])
    count = min(4, operator.shape[0] - 2)
    values, vectors = _converged_shift_invert_eigenpairs(
        spla,
        operator,
        mass,
        target=target,
        seed=seed,
        count=count,
    )
    overlaps = np.abs(np.conjugate(seed) @ vectors)
    selected = int(np.argmax(overlaps))
    electric = np.asarray(vectors[:, selected], dtype=np.complex128)
    phase = np.vdot(seed, electric)
    if abs(phase) > 1e-30:
        electric *= np.exp(-1j * np.angle(phase))
    ex, ey, ez = np.split(electric, (ex_n, ex_n + ey_n))
    out["Ex"], out["Ey"], out["Ez"] = (
        ex.reshape(nz, ny),
        ey.reshape(nz, ny - 1),
        ez.reshape(nz - 1, ny),
    )

    magnetic = 1j * (curl_e @ electric) / (_MU0 * omega_d * mu)
    hx, hy, hz = np.split(magnetic, (hx_n, hx_n + hy_n))
    out["Hx"], out["Hy"], out["Hz"] = (
        hx.reshape(nz - 1, ny - 1),
        hy.reshape(nz - 1, ny),
        hz.reshape(nz, ny - 1),
    )
    residual = np.linalg.norm(
        operator @ electric - values[selected] * (mass @ electric)
    )
    residual /= max(
        np.linalg.norm(values[selected] * (mass @ electric)), np.finfo(float).eps
    )
    eigenvalue = float(np.real(values[selected]))
    frequency_ratio = float(np.sqrt(max(eigenvalue / target, 0.0)))
    frequency_ratio_initial = frequency_ratio
    if _correct_beta:
        q_axis = 2.0 * np.sin(0.5 * float(k_num) * d) / d
        q_probe = 0.98 * q_axis
        delta_probe = -1j * sign * q_probe
        delta_change = delta_probe - delta
        operator_probe = (
            (curl_h + delta_change * curl_h_beta)
            @ sparse.diags(1.0 / mu)
            @ (curl_e + delta_change * curl_e_beta)
        )
        denominator = np.vdot(electric, mass @ electric)
        eigenvalue_probe = float(
            np.real(np.vdot(electric, operator_probe @ electric) / denominator)
        )
        slope = (eigenvalue_probe - eigenvalue) / (q_probe**2 - q_axis**2)
        if not np.isfinite(slope) or abs(slope) <= np.finfo(float).eps:
            slope = 0.5
        q_corrected = np.sqrt(max(q_axis**2 - (eigenvalue - target) / slope, 0.0))
        k_corrected = 2.0 * np.arcsin(np.clip(0.5 * q_corrected * d, -1.0, 1.0)) / d
        if np.isfinite(k_corrected) and k_corrected > 0.0:
            refined, residual, frequency_ratio, _, _ = refine_x_mode_at_fixed_beta(
                out,
                indices,
                component_permittivity=component_permittivity,
                component_permeability=component_permeability,
                omega=omega,
                dt=dt,
                resolution=resolution,
                k_num=k_corrected,
                direction_sign=direction_sign,
                transverse_cell_widths=transverse_cell_widths,
                _correct_beta=False,
            )
            return (
                refined,
                residual,
                frequency_ratio,
                float(k_corrected),
                frequency_ratio_initial,
            )
    return out, float(residual), frequency_ratio, float(k_num), frequency_ratio


def _converged_shift_invert_eigenpairs(
    spla,
    operator,
    mass,
    *,
    target,
    seed,
    count,
):
    """Return numerically valid eigenpairs near ``target``.

    Some ARPACK/SciPy combinations can report converged null-space vectors for
    this singular curl-curl operator even though their generalized eigenpair
    residuals are large. Retrying with a wider search reliably exposes the
    physical modes; independently checking the residual prevents a false mode
    from reaching the refinement validator.
    """
    attempts = (count, min(8, operator.shape[0] - 2))
    for candidate_count in dict.fromkeys(attempts):
        eigenpairs = spla.eigs(
            operator,
            M=mass,
            k=candidate_count,
            sigma=target,
            v0=seed,
            tol=1e-9,  # pyright: ignore[reportArgumentType] -- SciPy infers int from its untyped default.
        )
        values, vectors = cast(tuple[np.ndarray, np.ndarray], eigenpairs)
        valid = []
        for index, value in enumerate(values):
            vector = vectors[:, index]
            right = value * (mass @ vector)
            relative_residual = np.linalg.norm(operator @ vector - right) / max(
                np.linalg.norm(right), np.finfo(float).eps
            )
            if np.isfinite(relative_residual) and relative_residual <= 1e-6:
                valid.append(index)
        if valid:
            return values[valid], vectors[:, valid]
    raise RuntimeError(
        "fixed-beta mode refinement did not produce a converged eigenpair"
    )


def validate_x_mode_refinement(
    seed_profiles,
    candidate_profiles,
    indices,
    *,
    component_permittivity,
    component_permeability,
    omega,
    dt,
    resolution,
    k_num,
    direction_sign,
    transverse_cell_widths=None,
    integration_weights=None,
    minimum_electric_overlap=None,
    minimum_magnetic_overlap=0.5,
    maximum_impedance_change=4.0,
    maximum_energy_change=4.0,
    maximum_energy_imbalance=4.0,
    maximum_maxwell_residual=5e-3,
):
    """Validate that a refined x-normal profile remains the same physical mode.

    The refinement eigen-residual alone only establishes that the selected
    electric vector solves the assembled curl-curl problem.  These checks also
    require mode identity, a consistent E/H balance, forward signed power, and
    both first-order discrete Maxwell equations. Electric overlap remains a
    diagnostic by default because Yee refinement can legitimately redistribute
    longitudinal electric components; callers may opt into a hard threshold.
    """
    from scipy import sparse

    seed = {
        name: np.asarray(value, dtype=np.complex128)
        for name, value in seed_profiles.items()
    }
    candidate = {
        name: np.asarray(value, dtype=np.complex128)
        for name, value in candidate_profiles.items()
    }
    diagnostics: dict[str, Any] = {
        "electric_overlap": _normalized_component_overlap(
            seed, candidate, _E_COMPONENTS
        ),
        "magnetic_overlap": _normalized_component_overlap(
            seed, candidate, _H_COMPONENTS
        ),
    }
    seed_impedance = _rms_impedance(seed)
    candidate_impedance = _rms_impedance(candidate)
    diagnostics["seed_rms_impedance"] = seed_impedance
    diagnostics["candidate_rms_impedance"] = candidate_impedance
    diagnostics["impedance_change"] = _ratio_change(candidate_impedance, seed_impedance)

    eps, mu = _component_material_vectors(
        candidate,
        indices,
        component_permittivity=component_permittivity,
        component_permeability=component_permeability,
    )
    seed_energy_ratio = _energy_ratio(
        seed, eps, mu, integration_weights=integration_weights
    )
    candidate_energy_ratio = _energy_ratio(
        candidate, eps, mu, integration_weights=integration_weights
    )
    diagnostics["seed_energy_ratio"] = seed_energy_ratio
    diagnostics["candidate_energy_ratio"] = candidate_energy_ratio
    diagnostics["energy_ratio_change"] = _ratio_change(
        candidate_energy_ratio, seed_energy_ratio
    )
    diagnostics["candidate_energy_imbalance"] = _ratio_change(
        candidate_energy_ratio, 1.0
    )

    seed_power = _signed_x_power(
        seed,
        resolution=resolution,
        direction_sign=direction_sign,
        integration_weights=integration_weights,
    )
    candidate_power = _signed_x_power(
        candidate,
        resolution=resolution,
        direction_sign=direction_sign,
        integration_weights=integration_weights,
    )
    diagnostics["seed_signed_power"] = seed_power
    diagnostics["candidate_signed_power"] = candidate_power

    sign = 1.0 if float(direction_sign) >= 0.0 else -1.0
    delta = (
        -2j * sign * np.sin(0.5 * float(k_num) * float(resolution)) / float(resolution)
    )
    curl_e, curl_h, _curl_e_beta, _curl_h_beta = _yee_curl_operators(
        sparse,
        candidate,
        resolution=float(resolution),
        delta=delta,
        transverse_cell_widths=transverse_cell_widths,
    )
    omega_d = (
        float(omega)
        if dt is None
        else 2.0 * np.sin(0.5 * float(omega) * float(dt)) / float(dt)
    )
    electric = _component_vector(candidate, _E_COMPONENTS)
    magnetic = _component_vector(candidate, _H_COMPONENTS)
    faraday_left = curl_e @ electric
    faraday_right = -1j * _MU0 * omega_d * mu * magnetic
    ampere_left = curl_h @ magnetic
    ampere_right = 1j * _EPS0 * omega_d * eps * electric
    diagnostics["faraday_residual"] = _relative_pair_residual(
        faraday_left, faraday_right
    )
    diagnostics["ampere_residual"] = _relative_pair_residual(ampere_left, ampere_right)

    rejection_reasons = []
    if minimum_electric_overlap is not None and diagnostics["electric_overlap"] < float(
        minimum_electric_overlap
    ):
        rejection_reasons.append("electric overlap")
    if diagnostics["magnetic_overlap"] < float(minimum_magnetic_overlap):
        rejection_reasons.append("magnetic overlap")
    if diagnostics["impedance_change"] > float(maximum_impedance_change):
        rejection_reasons.append("E/H impedance change")
    if diagnostics["energy_ratio_change"] > float(maximum_energy_change):
        rejection_reasons.append("electric/magnetic energy change")
    if diagnostics["candidate_energy_imbalance"] > float(maximum_energy_imbalance):
        rejection_reasons.append("electric/magnetic energy imbalance")
    if not np.isfinite(candidate_power) or candidate_power <= np.finfo(float).tiny:
        rejection_reasons.append("non-forward signed power")
    if diagnostics["faraday_residual"] > float(maximum_maxwell_residual):
        rejection_reasons.append("Faraday residual")
    if diagnostics["ampere_residual"] > float(maximum_maxwell_residual):
        rejection_reasons.append("Ampere residual")
    diagnostics["accepted"] = not rejection_reasons
    diagnostics["rejection_reason"] = ", ".join(rejection_reasons)
    return not rejection_reasons, diagnostics


def _yee_curl_operators(
    sparse, profiles, *, resolution, delta, transverse_cell_widths=None
):
    nz, ny = profiles["Ex"].shape
    if transverse_cell_widths is None:
        dz = _forward_difference(sparse, nz, resolution)
        dy = _forward_difference(sparse, ny, resolution)
        bz, by = -dz.T, -dy.T
    else:
        widths_z, widths_y = (
            np.asarray(values, dtype=float).reshape(-1)
            for values in transverse_cell_widths
        )
        if widths_z.size != nz or widths_y.size != ny:
            raise ValueError(
                "Rectilinear transverse widths must match the x-normal mode plane"
            )
        dz = _center_to_edge_difference(sparse, widths_z)
        dy = _center_to_edge_difference(sparse, widths_y)
        bz = _edge_to_center_difference(sparse, widths_z)
        by = _edge_to_center_difference(sparse, widths_y)
    iz = sparse.eye(nz, format="csc")
    iy = sparse.eye(ny, format="csc")
    izm = sparse.eye(nz - 1, format="csc")
    iym = sparse.eye(ny - 1, format="csc")
    dy_ez = sparse.kron(izm, dy, format="csc")
    dz_ey = sparse.kron(dz, iym, format="csc")
    dz_ex = sparse.kron(dz, iy, format="csc")
    dy_ex = sparse.kron(iz, dy, format="csc")
    by_hz = sparse.kron(iz, by, format="csc")
    bz_hy = sparse.kron(bz, iy, format="csc")
    bz_hx = sparse.kron(bz, iym, format="csc")
    by_hx = sparse.kron(izm, by, format="csc")
    ex_n, ey_n, ez_n = (profiles[name].size for name in _E_COMPONENTS)
    hx_n, hy_n, hz_n = (profiles[name].size for name in _H_COMPONENTS)
    z = sparse.csc_matrix
    curl_e = sparse.bmat(
        [
            [z((hx_n, ex_n)), -dz_ey, dy_ez],
            [dz_ex, z((hy_n, ey_n)), -delta * sparse.eye(ez_n)],
            [-dy_ex, delta * sparse.eye(ey_n), z((hz_n, ez_n))],
        ],
        format="csc",
    )
    curl_h = sparse.bmat(
        [
            [z((ex_n, hx_n)), -bz_hy, by_hz],
            [bz_hx, z((ey_n, hy_n)), -delta * sparse.eye(hz_n)],
            [-by_hx, delta * sparse.eye(hy_n), z((ez_n, hz_n))],
        ],
        format="csc",
    )
    curl_e_beta = sparse.bmat(
        [
            [z((hx_n, ex_n)), z((hx_n, ey_n)), z((hx_n, ez_n))],
            [z((hy_n, ex_n)), z((hy_n, ey_n)), -sparse.eye(ez_n)],
            [z((hz_n, ex_n)), sparse.eye(ey_n), z((hz_n, ez_n))],
        ],
        format="csc",
    )
    curl_h_beta = sparse.bmat(
        [
            [z((ex_n, hx_n)), z((ex_n, hy_n)), z((ex_n, hz_n))],
            [z((ey_n, hx_n)), z((ey_n, hy_n)), -sparse.eye(hz_n)],
            [z((ez_n, hx_n)), sparse.eye(hy_n), z((ez_n, hz_n))],
        ],
        format="csc",
    )
    return curl_e, curl_h, curl_e_beta, curl_h_beta


def _component_material_vectors(
    profiles,
    indices,
    *,
    component_permittivity,
    component_permeability,
):
    eps = np.concatenate(
        [
            _material(component_permittivity[name], indices[name], profiles[name].shape)
            for name in _E_COMPONENTS
        ]
    )
    mu = np.concatenate(
        [
            _material(component_permeability[name], indices[name], profiles[name].shape)
            for name in _H_COMPONENTS
        ]
    )
    return eps, mu


def _component_vector(profiles, names):
    return np.concatenate(
        [np.asarray(profiles[name], dtype=np.complex128).reshape(-1) for name in names]
    )


def _normalized_component_overlap(seed, candidate, names):
    left = _component_vector(seed, names)
    right = _component_vector(candidate, names)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        return 0.0
    return float(abs(np.vdot(left, right)) / denominator)


def _rms_impedance(profiles):
    electric_norm = np.linalg.norm(_component_vector(profiles, _E_COMPONENTS))
    magnetic_norm = np.linalg.norm(_component_vector(profiles, _H_COMPONENTS))
    if (
        not np.isfinite(electric_norm)
        or not np.isfinite(magnetic_norm)
        or magnetic_norm <= np.finfo(float).tiny
    ):
        return np.inf
    return float(electric_norm / magnetic_norm)


def _energy_ratio(profiles, eps, mu, *, integration_weights=None):
    electric = _component_vector(profiles, _E_COMPONENTS)
    magnetic = _component_vector(profiles, _H_COMPONENTS)
    electric_weights = _component_weight_vector(
        profiles, _E_COMPONENTS, integration_weights
    )
    magnetic_weights = _component_weight_vector(
        profiles, _H_COMPONENTS, integration_weights
    )
    electric_energy = _EPS0 * float(
        np.real(np.vdot(electric, electric_weights * eps * electric))
    )
    magnetic_energy = _MU0 * float(
        np.real(np.vdot(magnetic, magnetic_weights * mu * magnetic))
    )
    if (
        not np.isfinite(electric_energy)
        or not np.isfinite(magnetic_energy)
        or magnetic_energy <= np.finfo(float).tiny
    ):
        return np.inf
    return float(electric_energy / magnetic_energy)


def _ratio_change(candidate, seed):
    if (
        not np.isfinite(candidate)
        or not np.isfinite(seed)
        or min(candidate, seed) <= np.finfo(float).tiny
    ):
        return np.inf
    return float(max(candidate / seed, seed / candidate))


def _signed_x_power(profiles, *, resolution, direction_sign, integration_weights=None):
    if integration_weights is None:
        ey_weights = ez_weights = float(resolution) ** 2
    else:
        ey_weights = np.asarray(integration_weights["Ey"], dtype=float).reshape(-1)
        ez_weights = np.asarray(integration_weights["Ez"], dtype=float).reshape(-1)
    flux = np.vdot(
        profiles["Hz"].reshape(-1),
        ey_weights * profiles["Ey"].reshape(-1),
    )
    flux -= np.vdot(
        profiles["Hy"].reshape(-1),
        ez_weights * profiles["Ez"].reshape(-1),
    )
    return float(0.5 * float(direction_sign) * np.real(flux))


def _component_weight_vector(profiles, names, integration_weights):
    if integration_weights is None:
        return np.ones(sum(np.asarray(profiles[name]).size for name in names))
    return np.concatenate(
        [
            np.asarray(integration_weights[name], dtype=float).reshape(-1)
            for name in names
        ]
    )


def _relative_pair_residual(left, right):
    denominator = max(np.linalg.norm(left), np.linalg.norm(right), np.finfo(float).eps)
    return float(np.linalg.norm(left - right) / denominator)


def _material(values, index, shape):
    arr = np.asarray(values, dtype=np.complex128)
    profile = arr[index] if arr.ndim == 3 else arr
    profile = np.asarray(profile, dtype=np.complex128).squeeze()
    if profile.shape != shape:
        raise ValueError(f"Material shape {profile.shape} does not match {shape}")
    return profile.reshape(-1)


def _forward_difference(sparse, count, spacing):
    rows = np.repeat(np.arange(count - 1), 2)
    cols = np.column_stack((np.arange(count - 1), np.arange(1, count))).reshape(-1)
    data = np.tile(np.asarray([-1.0, 1.0]) / spacing, count - 1)
    return sparse.csc_matrix((data, (rows, cols)), shape=(count - 1, count))


def _center_to_edge_difference(sparse, cell_widths):
    widths = np.asarray(cell_widths, dtype=float).reshape(-1)
    spacing = 0.5 * (widths[:-1] + widths[1:])
    rows = np.repeat(np.arange(widths.size - 1), 2)
    cols = np.column_stack(
        (np.arange(widths.size - 1), np.arange(1, widths.size))
    ).reshape(-1)
    data = np.column_stack((-1.0 / spacing, 1.0 / spacing)).reshape(-1)
    return sparse.csc_matrix((data, (rows, cols)), shape=(widths.size - 1, widths.size))


def _edge_to_center_difference(sparse, cell_widths):
    widths = np.asarray(cell_widths, dtype=float).reshape(-1)
    rows = np.concatenate(
        (
            np.asarray([0]),
            np.repeat(np.arange(1, widths.size - 1), 2),
            np.asarray([widths.size - 1]),
        )
    )
    cols = np.concatenate(
        (
            np.asarray([0]),
            np.column_stack(
                (np.arange(widths.size - 2), np.arange(1, widths.size - 1))
            ).reshape(-1),
            np.asarray([widths.size - 2]),
        )
    )
    data = np.concatenate(
        (
            np.asarray([1.0 / widths[0]]),
            np.column_stack(
                (
                    -1.0 / widths[1:-1],
                    1.0 / widths[1:-1],
                )
            ).reshape(-1),
            np.asarray([-1.0 / widths[-1]]),
        )
    )
    return sparse.csc_matrix((data, (rows, cols)), shape=(widths.size, widths.size - 1))
