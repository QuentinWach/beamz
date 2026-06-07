"""Shared solver helper functions used by step and compiled engines."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0


@dataclass(frozen=True)
class CpmlTm2DxyTerms:
    sigma_h_terms: jnp.ndarray
    kappa_h_aux_terms: jnp.ndarray
    alpha_h_terms: jnp.ndarray
    kappa_h_direct_terms: jnp.ndarray
    sigma_e_terms: jnp.ndarray
    kappa_e_terms: jnp.ndarray
    alpha_e_terms: jnp.ndarray


@dataclass(frozen=True)
class Cpml3DTerms:
    a_h_terms: tuple[jnp.ndarray, ...]
    b_h_terms: tuple[jnp.ndarray, ...]
    inv_kappa_h_terms: tuple[jnp.ndarray, ...]
    a_e_terms: tuple[jnp.ndarray, ...]
    b_e_terms: tuple[jnp.ndarray, ...]
    inv_kappa_e_terms: tuple[jnp.ndarray, ...]


@dataclass(frozen=True)
class Cpml3DPrimitiveTerms:
    sigma_h_terms: tuple[jnp.ndarray, ...]
    kappa_h_terms: tuple[jnp.ndarray, ...]
    alpha_h_terms: tuple[jnp.ndarray, ...]
    sigma_e_terms: tuple[jnp.ndarray, ...]
    kappa_e_terms: tuple[jnp.ndarray, ...]
    alpha_e_terms: tuple[jnp.ndarray, ...]


@dataclass(frozen=True)
class CpmlDerivative3DSpec:
    """A single split-field CPML derivative term on the Yee lattice."""

    name: str
    target_component: str
    source_component: str
    derivative_axis: str


CPML_3D_H_DERIVATIVES: tuple[CpmlDerivative3DSpec, ...] = (
    CpmlDerivative3DSpec("Hxy", "Hx", "Ez", "y"),
    CpmlDerivative3DSpec("Hxz", "Hx", "Ey", "z"),
    CpmlDerivative3DSpec("Hyz", "Hy", "Ex", "z"),
    CpmlDerivative3DSpec("Hyx", "Hy", "Ez", "x"),
    CpmlDerivative3DSpec("Hzx", "Hz", "Ey", "x"),
    CpmlDerivative3DSpec("Hzy", "Hz", "Ex", "y"),
)

CPML_3D_E_DERIVATIVES: tuple[CpmlDerivative3DSpec, ...] = (
    CpmlDerivative3DSpec("Exy", "Ex", "Hz", "y"),
    CpmlDerivative3DSpec("Exz", "Ex", "Hy", "z"),
    CpmlDerivative3DSpec("Eyz", "Ey", "Hx", "z"),
    CpmlDerivative3DSpec("Eyx", "Ey", "Hz", "x"),
    CpmlDerivative3DSpec("Ezx", "Ez", "Hy", "x"),
    CpmlDerivative3DSpec("Ezy", "Ez", "Hx", "y"),
)


def full_tm_xy_component_to_centered_grid(component: str, values):
    """Project full-lattice TM samples onto centered monitor/sample points."""
    field = values
    if component == "Ez":
        if field.ndim != 2 or field.shape[0] < 2 or field.shape[1] < 2:
            raise ValueError(
                f"Ez full-TM field must be at least 2x2, got {field.shape}"
            )
        return 0.25 * (
            field[:-1, :-1] + field[:-1, 1:] + field[1:, :-1] + field[1:, 1:]
        )
    if component == "Hx":
        if field.ndim != 2 or field.shape[1] < 2:
            raise ValueError(
                f"Hx full-TM field must have width >= 2, got {field.shape}"
            )
        return 0.5 * (field[:, :-1] + field[:, 1:])
    if component == "Hy":
        if field.ndim != 2 or field.shape[0] < 2:
            raise ValueError(
                f"Hy full-TM field must have height >= 2, got {field.shape}"
            )
        return 0.5 * (field[:-1, :] + field[1:, :])
    raise ValueError(f"Unsupported full-TM centered-grid component {component!r}")


def is_full_tm_xy_lattice(ez, hx, hy) -> bool:
    """Return True when Ez/Hx/Hy follow BeamZ's physical xy-TM staggering."""
    return (
        getattr(ez, "ndim", None) == 2
        and getattr(hx, "ndim", None) == 2
        and getattr(hy, "ndim", None) == 2
        and hx.shape[0] == ez.shape[0] - 1
        and hx.shape[1] == ez.shape[1]
        and hy.shape[0] == ez.shape[0]
        and hy.shape[1] == ez.shape[1] - 1
    )


def embed_tm_xy_h_terms(
    term0: jnp.ndarray, term1: jnp.ndarray, ez_shape: tuple[int, int]
) -> jnp.ndarray:
    out = jnp.zeros((2, *ez_shape), dtype=term0.dtype)
    out = out.at[0, :-1, :].set(term0)
    out = out.at[1, :, :-1].set(term1)
    return out


def cpml_precompute_native_terms(
    sigma_terms: tuple[jnp.ndarray, ...],
    kappa_terms: tuple[jnp.ndarray, ...],
    alpha_terms: tuple[jnp.ndarray, ...],
    dt: float,
) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
    a_terms = []
    b_terms = []
    inv_kappa_terms = []
    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    eps0 = jnp.asarray(EPS_0, dtype=jnp.float32)
    for sigma, kappa, alpha in zip(sigma_terms, kappa_terms, alpha_terms, strict=True):
        sigma = jnp.asarray(sigma, dtype=jnp.float32)
        kappa = jnp.maximum(jnp.asarray(kappa, dtype=jnp.float32), 1.0)
        alpha = jnp.asarray(alpha, dtype=jnp.float32)
        decay = (sigma / kappa + alpha) * (dt_arr / eps0)
        b = jnp.expm1(-decay) + 1.0
        denom = sigma + kappa * alpha
        a = jnp.nan_to_num(
            ((b - 1.0) * sigma) / jnp.maximum(denom * kappa, 1e-30),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        a_terms.append(a)
        b_terms.append(b)
        inv_kappa_terms.append(1.0 / kappa)
    return tuple(a_terms), tuple(b_terms), tuple(inv_kappa_terms)


def build_tm_xy_cpml_terms(
    tm_xy: dict[str, jnp.ndarray] | None,
    *,
    ez_shape: tuple[int, int],
) -> CpmlTm2DxyTerms | None:
    if tm_xy is None:
        return None
    sigma_hx = jnp.asarray(tm_xy["Hx_y_sigma"], dtype=jnp.float32)
    kappa_hx = jnp.asarray(tm_xy["Hx_y_kappa"], dtype=jnp.float32)
    alpha_hx = jnp.asarray(tm_xy["Hx_y_alpha"], dtype=jnp.float32)
    sigma_hy = jnp.asarray(tm_xy["Hy_x_sigma"], dtype=jnp.float32)
    kappa_hy = jnp.asarray(tm_xy["Hy_x_kappa"], dtype=jnp.float32)
    alpha_hy = jnp.asarray(tm_xy["Hy_x_alpha"], dtype=jnp.float32)
    sigma_ez_x = jnp.asarray(tm_xy["Ez_x_sigma"], dtype=jnp.float32)
    kappa_ez_x = jnp.asarray(tm_xy["Ez_x_kappa"], dtype=jnp.float32)
    alpha_ez_x = jnp.asarray(tm_xy["Ez_x_alpha"], dtype=jnp.float32)
    sigma_ez_y = jnp.asarray(tm_xy["Ez_y_sigma"], dtype=jnp.float32)
    kappa_ez_y = jnp.asarray(tm_xy["Ez_y_kappa"], dtype=jnp.float32)
    alpha_ez_y = jnp.asarray(tm_xy["Ez_y_alpha"], dtype=jnp.float32)
    return CpmlTm2DxyTerms(
        sigma_h_terms=embed_tm_xy_h_terms(sigma_hx, sigma_hy, ez_shape),
        kappa_h_aux_terms=embed_tm_xy_h_terms(kappa_hx, kappa_hy, ez_shape),
        alpha_h_terms=embed_tm_xy_h_terms(alpha_hx, alpha_hy, ez_shape),
        kappa_h_direct_terms=embed_tm_xy_h_terms(kappa_hx, kappa_hy, ez_shape),
        sigma_e_terms=jnp.stack((sigma_ez_x, sigma_ez_y), axis=0),
        kappa_e_terms=jnp.stack((kappa_ez_x, kappa_ez_y), axis=0),
        alpha_e_terms=jnp.stack((alpha_ez_x, alpha_ez_y), axis=0),
    )


def build_cpml_3d_terms(
    pml_data: dict[str, jnp.ndarray] | None,
    *,
    dt: float,
) -> Cpml3DTerms | None:
    if pml_data is None:
        return None

    def read_terms(specs, suffix):
        return tuple(
            jnp.asarray(pml_data[f"cpml3d_{spec.name}_{suffix}"], dtype=jnp.float32)
            for spec in specs
        )

    sigma_h_terms = read_terms(CPML_3D_H_DERIVATIVES, "sigma")
    kappa_h_terms = read_terms(CPML_3D_H_DERIVATIVES, "kappa")
    alpha_h_terms = read_terms(CPML_3D_H_DERIVATIVES, "alpha")
    sigma_e_terms = read_terms(CPML_3D_E_DERIVATIVES, "sigma")
    kappa_e_terms = read_terms(CPML_3D_E_DERIVATIVES, "kappa")
    alpha_e_terms = read_terms(CPML_3D_E_DERIVATIVES, "alpha")
    a_h_terms, b_h_terms, inv_kappa_h_terms = cpml_precompute_native_terms(
        sigma_h_terms, kappa_h_terms, alpha_h_terms, dt
    )
    a_e_terms, b_e_terms, inv_kappa_e_terms = cpml_precompute_native_terms(
        sigma_e_terms, kappa_e_terms, alpha_e_terms, dt
    )
    return Cpml3DTerms(
        a_h_terms=a_h_terms,
        b_h_terms=b_h_terms,
        inv_kappa_h_terms=inv_kappa_h_terms,
        a_e_terms=a_e_terms,
        b_e_terms=b_e_terms,
        inv_kappa_e_terms=inv_kappa_e_terms,
    )


def build_cpml_3d_primitive_terms(
    pml_data: dict[str, jnp.ndarray] | None,
) -> Cpml3DPrimitiveTerms | None:
    if pml_data is None:
        return None

    axis_index = {"z": 0, "y": 1, "x": 2}

    def compact_axis_profile(arr, axis_name):
        arr = jnp.asarray(arr, dtype=jnp.float32)
        if arr.ndim != 3:
            return arr
        idx = [0, 0, 0]
        idx[axis_index[axis_name]] = slice(None)
        profile = arr[tuple(idx)]
        shape = [1, 1, 1]
        shape[axis_index[axis_name]] = profile.shape[0]
        compact = jnp.reshape(profile, tuple(shape))
        if not np.allclose(
            np.asarray(arr),
            np.asarray(jnp.broadcast_to(compact, arr.shape)),
            rtol=1e-6,
            atol=1e-7,
        ):
            return None
        return compact

    def read_terms(specs, suffix):
        terms = []
        for spec in specs:
            term = compact_axis_profile(
                pml_data[f"cpml3d_{spec.name}_{suffix}"],
                spec.derivative_axis,
            )
            if term is None:
                return None
            terms.append(term)
        return tuple(terms)

    sigma_h_terms = read_terms(CPML_3D_H_DERIVATIVES, "sigma")
    kappa_h_terms = read_terms(CPML_3D_H_DERIVATIVES, "kappa")
    alpha_h_terms = read_terms(CPML_3D_H_DERIVATIVES, "alpha")
    sigma_e_terms = read_terms(CPML_3D_E_DERIVATIVES, "sigma")
    kappa_e_terms = read_terms(CPML_3D_E_DERIVATIVES, "kappa")
    alpha_e_terms = read_terms(CPML_3D_E_DERIVATIVES, "alpha")
    if any(
        terms is None
        for terms in (
            sigma_h_terms,
            kappa_h_terms,
            alpha_h_terms,
            sigma_e_terms,
            kappa_e_terms,
            alpha_e_terms,
        )
    ):
        return None

    return Cpml3DPrimitiveTerms(
        sigma_h_terms=sigma_h_terms,
        kappa_h_terms=kappa_h_terms,
        alpha_h_terms=alpha_h_terms,
        sigma_e_terms=sigma_e_terms,
        kappa_e_terms=kappa_e_terms,
        alpha_e_terms=alpha_e_terms,
    )


def poynting_magnitude_2d(ez, hx, hy):
    """Return |E x H| for 2D TM monitor samples."""
    sx = -ez * hy
    sy = ez * hx
    return (sx * sx + sy * sy) ** 0.5


def poynting_flux_2d(ez, hx, hy, normal_axis, normal_sign=1.0):
    """Return n . (E x H) for 2D TM monitor samples."""
    sx = -ez * hy
    sy = ez * hx
    axis = str(normal_axis).lower()
    component = sx if axis == "x" else sy
    return component * normal_sign


def poynting_magnitude_3d(ex, ey, ez, hx, hy, hz):
    """Return |E x H| for 3D monitor samples."""
    sx = ey * hz - ez * hy
    sy = ez * hx - ex * hz
    sz = ex * hy - ey * hx
    return (sx * sx + sy * sy + sz * sz) ** 0.5


def poynting_flux_3d(ex, ey, ez, hx, hy, hz, normal_axis, normal_sign=1.0):
    """Return n . (E x H) for 3D monitor samples."""
    sx = ey * hz - ez * hy
    sy = ez * hx - ex * hz
    sz = ex * hy - ey * hx
    axis = str(normal_axis).lower()
    if axis == "x":
        component = sx
    elif axis == "y":
        component = sy
    else:
        component = sz
    return component * normal_sign


def physical_dft_sample_scale(weight, base_dt, record_interval, length_unit):
    """Return the physical-unit DFT accumulation scale for one sample."""
    return weight * (
        base_dt * record_interval * LIGHT_SPEED / length_unit / np.sqrt(2.0 * np.pi)
    )


def advance_h_from_curl(field, curl, sigma_m, dt):
    """Advance one H component from its curl and magnetic conductivity."""
    denom = 1.0 + sigma_m * (dt / (2.0 * MU_0))
    factor = (1.0 - sigma_m * (dt / (2.0 * MU_0))) / denom
    source_coeff = (dt / MU_0) / denom
    return field * factor - source_coeff * curl


def advance_e_from_curl(field, curl, conductivity, permittivity, dt, region):
    """Advance one E component from its curl and material slices."""
    denom = 1.0 + conductivity * (dt / (2.0 * EPS_0 * permittivity))
    factor = (1.0 - conductivity * (dt / (2.0 * EPS_0 * permittivity))) / denom
    source = (dt / (EPS_0 * permittivity)) / denom
    new_values = field[region] * factor + source * curl[region]
    return field.at[region].set(new_values.astype(field.dtype))


def advance_h_from_coefficients(field, curl, decay, source):
    """Advance one H component using precomputed dense coefficients."""
    return decay * field - source * curl


def advance_e_from_coefficients(field, curl, decay, source):
    """Advance one E component using precomputed dense coefficients."""
    return decay * field + source * curl


def apply_zero_mask(field, mask):
    """Zero out constrained samples when a mask is present."""
    if mask is None:
        return field
    return jnp.where(mask, jnp.asarray(0.0, dtype=field.dtype), field)


def step_hits_interval(step, interval):
    """Return True when a zero-based step lands on a direct modulo interval."""
    step_idx = jnp.asarray(step, dtype=jnp.int32)
    interval_idx = jnp.maximum(jnp.asarray(interval, dtype=jnp.int32), 1)
    return (step_idx % interval_idx) == 0


def monitor_records_on_step(step, record_interval):
    """Return True when a monitor should emit a power/history sample on this step."""
    step_idx = jnp.asarray(step, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    return step_hits_interval(step_idx, record_interval)


def monitor_dft_should_accumulate(enabled, step, t, t_start, t_end, record_interval):
    """Return True when a DFT monitor should accumulate on this sample."""
    t_now = jnp.asarray(t, dtype=jnp.float32)
    return (
        jnp.asarray(enabled)
        & (t_now >= jnp.asarray(t_start, dtype=jnp.float32))
        & (t_now <= jnp.asarray(t_end, dtype=jnp.float32))
        & step_hits_interval(step, record_interval)
    )


def monitor_dft_window_weight(t, t_start, t_end, use_hann):
    """Return the DFT window weight for one sample."""
    t_now = jnp.asarray(t, dtype=jnp.float32)
    t0 = jnp.asarray(t_start, dtype=jnp.float32)
    t1 = jnp.asarray(t_end, dtype=jnp.float32)
    use_hann = jnp.asarray(use_hann)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    one = jnp.asarray(1.0, dtype=jnp.float32)
    span = jnp.maximum(t1 - t0, jnp.asarray(1e-30, dtype=jnp.float32))
    tau = jnp.clip((t_now - t0) / span, zero, one)
    hann = jnp.asarray(0.5, dtype=jnp.float32) * (
        one - jnp.cos(jnp.asarray(2.0 * np.pi, dtype=jnp.float32) * tau)
    )
    finite_span = jnp.isfinite(t1) & (t1 > t0)
    return jnp.where(use_hann & finite_span, hann, one)


def monitor_dft_sample_scale(
    weight,
    *,
    normalization_code,
    base_dt,
    record_interval,
    length_unit,
):
    """Return the scaled DFT sample multiplier for native or physical mode."""
    if isinstance(weight, (jax.Array, jax.core.Tracer)):
        zero = jnp.asarray(0.0, dtype=jnp.result_type(weight, base_dt, record_interval))
        native = jnp.where(weight > zero, weight, zero)
        physical = jnp.where(
            weight > zero,
            physical_dft_sample_scale(
                native,
                base_dt,
                record_interval,
                length_unit,
            ),
            zero,
        )
        return jnp.where(jnp.asarray(normalization_code) == 1, physical, native)
    native = weight if float(weight) > 0.0 else 0.0
    if int(normalization_code) == 1:
        return physical_dft_sample_scale(
            native,
            base_dt,
            record_interval,
            length_unit,
        )
    return native
