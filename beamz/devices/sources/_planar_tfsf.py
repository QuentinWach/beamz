"""Planar launched-side TF/SF residual machinery for 3D sources."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from beamz._yee import component_axis_offsets_3d
from beamz.devices.sources._profiles import FieldProfile3D
from beamz.devices.sources._time import _real_phasor_sample

_FIELD_COMPONENTS_3D = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_E_COMPONENTS_3D = ("Ex", "Ey", "Ez")
_H_COMPONENTS_3D = ("Hx", "Hy", "Hz")
_AXIS_POS_3D = {"z": 0, "y": 1, "x": 2}
_STAGGERED_ALONG_AXIS = {
    "x": {"Ex", "Hy", "Hz"},
    "y": {"Ey", "Hx", "Hz"},
    "z": {"Ez", "Hx", "Hy"},
}


@dataclass(frozen=True)
class _ModeSource3DResidual:
    """Compact local 3D source residual emitted by ModeSource compilation."""

    component: str
    timing: str
    index: tuple[slice, slice, slice]
    residual: np.ndarray


def _numeric_phase_delay(omega, k_num, delta_s, eps=1e-30):
    """Convert numerical phase advance into a time delay."""
    omega_r = max(abs(float(omega)), eps)
    return float((float(k_num) * float(delta_s)) / omega_r)


def _axis_index_from_component_indices(indices, axis):
    """Extract scalar axis index from a 3D component index tuple."""
    if indices is None:
        return None
    axis_pos = {"x": 2, "y": 1, "z": 0}[axis]
    val = indices[axis_pos]
    if isinstance(val, slice):
        return None
    return int(val)


def _component_axis_coord(component_name, axis_index, axis, dx, dy, dz):
    """Yee-location coordinate along propagation axis for one component plane index."""
    if axis_index is None:
        return 0.0

    d_axis = {"x": dx, "y": dy, "z": dz}[axis]
    offset = 1.0 if component_name in _STAGGERED_ALONG_AXIS[axis] else 0.5
    return (axis_index + offset) * d_axis


def _shift_component_indices_along_axis(indices, axis, shift, field_shape):
    """Shift a component support tuple by integer cells along the propagation axis."""
    if indices is None:
        return None
    axis_pos = _AXIS_POS_3D[axis]
    out = list(indices)
    plane_idx = out[axis_pos]
    if isinstance(plane_idx, slice):
        return None
    plane_new = int(plane_idx) + int(shift)
    if plane_new < 0 or plane_new >= int(field_shape[axis_pos]):
        return None
    out[axis_pos] = plane_new
    return tuple(out)


def _require_k_axis(field_profile: FieldProfile3D) -> float:
    if field_profile.k_axis is None:
        raise RuntimeError(
            "3D incident state requested without discrete launch metadata"
        )
    return float(field_profile.k_axis)


def _field_arrays_like(fields, *, dtype) -> dict[str, np.ndarray]:
    return {
        component: np.zeros_like(np.asarray(getattr(fields, component)), dtype=dtype)
        for component in _FIELD_COMPONENTS_3D
    }


def build_incident_3d_state(
    field_profile: FieldProfile3D,
    fields,
    *,
    resolution: float,
    t_e,
    t_h,
    dt,
    masked: bool,
    get_signal_value,
    get_signal_quadrature_value,
    max_shift: int,
) -> dict[str, np.ndarray]:
    """Construct the local 3D incident field state used by a discrete source."""
    dx = dy = dz = float(resolution)
    axis = field_profile.axis
    k_num = _require_k_axis(field_profile)
    omega = float(field_profile.omega)
    plane_coord = float(field_profile.phase_plane_coord)
    ref_coord = float(field_profile.phase_ref_coord)
    d_axis = {"x": dx, "y": dy, "z": dz}[axis]
    direction_sign = float(field_profile.direction_sign)
    max_shift = int(max(1, max_shift))

    field_arrays = _field_arrays_like(fields, dtype=np.float64)
    field_shapes = {name: arr.shape for name, arr in field_arrays.items()}

    for comp_name, profile in field_profile.components.items():
        idx = field_profile.indices.get(comp_name)
        if idx is None:
            continue

        base_axis_idx = _axis_index_from_component_indices(idx, axis)
        base_coord = _component_axis_coord(comp_name, base_axis_idx, axis, dx, dy, dz)
        profile_arr = np.asarray(profile, dtype=np.complex128)
        base_time = float(t_e if comp_name.startswith("E") else t_h)

        for shift in range(-max_shift, max_shift + 1):
            shifted_idx = _shift_component_indices_along_axis(
                idx, axis, shift, field_shapes[comp_name]
            )
            if shifted_idx is None:
                continue

            coord = float(base_coord + shift * d_axis)
            if masked:
                mask_coord = (
                    ref_coord
                    if comp_name in _STAGGERED_ALONG_AXIS[axis]
                    else plane_coord
                )
                if direction_sign * (coord - mask_coord) < -1e-12:
                    continue

            delay = _numeric_phase_delay(omega, k_num, coord - ref_coord)
            signal_time = base_time - delay
            amp_re = float(get_signal_value(signal_time, dt))
            amp_im = float(get_signal_quadrature_value(signal_time, dt))
            if amp_re == 0.0 and amp_im == 0.0:
                continue

            field_arrays[comp_name][shifted_idx] = field_arrays[comp_name][
                shifted_idx
            ] + _real_phasor_sample(profile_arr, amp_re, amp_im)

    return field_arrays


def build_incident_3d_phasor_state(
    field_profile: FieldProfile3D,
    fields,
    *,
    resolution: float,
    t_e,
    t_h,
    masked: bool,
    max_shift: int,
) -> dict[str, np.ndarray]:
    """Construct a complex carrier phasor for a local 3D incident field."""
    dx = dy = dz = float(resolution)
    axis = field_profile.axis
    k_num = _require_k_axis(field_profile)
    omega = float(field_profile.omega)
    plane_coord = float(field_profile.phase_plane_coord)
    ref_coord = float(field_profile.phase_ref_coord)
    d_axis = {"x": dx, "y": dy, "z": dz}[axis]
    direction_sign = float(field_profile.direction_sign)
    max_shift = int(max(1, max_shift))

    field_arrays = _field_arrays_like(fields, dtype=np.complex128)
    field_shapes = {name: arr.shape for name, arr in field_arrays.items()}

    for comp_name, profile in field_profile.components.items():
        idx = field_profile.indices.get(comp_name)
        if idx is None:
            continue

        base_axis_idx = _axis_index_from_component_indices(idx, axis)
        base_coord = _component_axis_coord(comp_name, base_axis_idx, axis, dx, dy, dz)
        profile_arr = np.asarray(profile, dtype=np.complex128)
        base_time = float(t_e if comp_name.startswith("E") else t_h)

        for shift in range(-max_shift, max_shift + 1):
            shifted_idx = _shift_component_indices_along_axis(
                idx, axis, shift, field_shapes[comp_name]
            )
            if shifted_idx is None:
                continue

            coord = float(base_coord + shift * d_axis)
            if masked:
                mask_coord = (
                    ref_coord
                    if comp_name in _STAGGERED_ALONG_AXIS[axis]
                    else plane_coord
                )
                if direction_sign * (coord - mask_coord) < -1e-12:
                    continue

            delay = _numeric_phase_delay(omega, k_num, coord - ref_coord)
            phase = omega * (base_time - delay)
            field_arrays[comp_name][shifted_idx] = field_arrays[comp_name][
                shifted_idx
            ] + profile_arr * np.exp(1j * phase)

    return field_arrays


def deembed_3d_phasor_profiles(
    field_profile: FieldProfile3D,
    state: dict[str, np.ndarray],
    *,
    resolution: float,
    t_e,
    t_h,
) -> dict[str, np.ndarray]:
    """Return local source-plane phasors in the source profile gauge."""
    dx = dy = dz = float(resolution)
    axis = field_profile.axis
    k_num = _require_k_axis(field_profile)
    omega = float(field_profile.omega)
    ref_coord = float(field_profile.phase_ref_coord)
    out: dict[str, np.ndarray] = {}
    for component in _FIELD_COMPONENTS_3D:
        idx = field_profile.indices.get(component)
        if idx is None or component not in state:
            continue
        values = np.asarray(state[component], dtype=np.complex128)
        axis_idx = _axis_index_from_component_indices(idx, axis)
        coord = _component_axis_coord(component, axis_idx, axis, dx, dy, dz)
        base_time = float(t_e if component.startswith("E") else t_h)
        delay = _numeric_phase_delay(omega, k_num, coord - ref_coord)
        phase = omega * (base_time - delay)
        out[component] = values[idx] * np.exp(-1j * phase)
    return out


def launched_side_component_mask_3d(
    field_profile: FieldProfile3D,
    component: str,
    shape: tuple[int, int, int],
    *,
    resolution: float,
) -> np.ndarray:
    """Return a broadcast mask for the component-local launched side."""
    axis = field_profile.axis
    d_axis = float(resolution)
    axis_pos = _AXIS_POS_3D[axis]
    offset = 1.0 if component in _STAGGERED_ALONG_AXIS[axis] else 0.5
    coord = (np.arange(int(shape[axis_pos]), dtype=np.float64) + offset) * d_axis
    mask_coord = (
        float(field_profile.phase_ref_coord)
        if component in _STAGGERED_ALONG_AXIS[axis]
        else float(field_profile.phase_plane_coord)
    )
    launched = float(field_profile.direction_sign) * (coord - mask_coord) >= -1e-12
    reshape = [1, 1, 1]
    reshape[axis_pos] = int(launched.size)
    return launched.reshape(tuple(reshape))


def mask_incident_3d_state_to_launched_side(
    field_profile: FieldProfile3D,
    state: dict[str, np.ndarray],
    *,
    resolution: float,
) -> dict[str, np.ndarray]:
    """Keep only the side of an incident state that should enter the grid."""
    out: dict[str, np.ndarray] = {}
    for component, values in state.items():
        arr = np.asarray(values)
        mask = launched_side_component_mask_3d(
            field_profile,
            component,
            arr.shape,
            resolution=resolution,
        )
        out[component] = np.where(mask, arr, np.zeros((), dtype=arr.dtype))
    return out


def component_slices_to_cell_bbox(
    component: str,
    index: tuple[slice, slice, slice],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    offsets = component_axis_offsets_3d(component)
    bounds: list[tuple[int, int]] = []
    for axis_name, item in zip(("z", "y", "x"), index, strict=True):
        start = int(item.start or 0)
        stop = int(item.stop or start)
        if float(offsets[axis_name]) == 0.5:
            stop += 1
        bounds.append((start, stop))
    return tuple(bounds)  # type: ignore[return-value]


def crop_local_residual(
    component: str,
    timing: str,
    local_index: tuple[slice, slice, slice],
    residual: np.ndarray,
    *,
    atol: float = 1e-30,
) -> _ModeSource3DResidual | None:
    values = np.asarray(residual, dtype=np.complex128)
    if values.size == 0:
        return None
    mask = np.abs(values) > float(atol)
    if not np.any(mask):
        return None
    coords = np.argwhere(mask)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    local_crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    global_crop = tuple(
        slice(
            int(parent.start or 0) + int(child.start or 0),
            int(parent.start or 0) + int(child.stop or 0),
        )
        for parent, child in zip(local_index, local_crop, strict=True)
    )
    return _ModeSource3DResidual(
        component=component,
        timing=timing,
        index=global_crop,  # type: ignore[arg-type]
        residual=values[local_crop].copy(),
    )


def normalize_3d_component_index(
    index: tuple,
    shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    out: list[slice] = []
    for item, dim in zip(index, shape, strict=True):
        if isinstance(item, slice):
            start, stop, step = item.indices(int(dim))
            if step != 1:
                raise ValueError("3D source component slices must be contiguous")
            out.append(slice(int(start), int(stop)))
        else:
            idx = int(item)
            if idx < 0:
                idx += int(dim)
            out.append(slice(idx, idx + 1))
    return tuple(out)  # type: ignore[return-value]


def component_slices_from_cell_bounds(
    component: str,
    cell_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    field_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    offsets = component_axis_offsets_3d(component)
    out: list[slice] = []
    for axis, (lo, hi), dim in zip(("z", "y", "x"), cell_bounds, field_shape):
        stop = int(hi) - (1 if float(offsets[axis]) == 0.5 else 0)
        start = max(0, min(int(lo), int(dim)))
        stop = max(start, min(stop, int(dim)))
        out.append(slice(start, stop))
    return tuple(out)  # type: ignore[return-value]


def shift_3d_component_index_to_local(
    index: tuple,
    component_slice: tuple[slice, slice, slice],
    field_shape: tuple[int, int, int],
) -> tuple:
    out: list[int | slice] = []
    for item, parent, dim in zip(index, component_slice, field_shape, strict=True):
        base = int(parent.start or 0)
        if isinstance(item, slice):
            start, stop, step = item.indices(int(dim))
            if step != 1:
                raise ValueError("3D source component slices must be contiguous")
            out.append(slice(int(start) - base, int(stop) - base))
        else:
            idx = int(item)
            if idx < 0:
                idx += int(dim)
            out.append(idx - base)
    return tuple(out)


def translate_region_to_local(
    region: tuple[slice, slice, slice],
    component_slice: tuple[slice, slice, slice],
    field_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    out: list[slice] = []
    for item, parent, dim in zip(region, component_slice, field_shape, strict=True):
        r_start, r_stop, r_step = item.indices(int(dim))
        if r_step != 1:
            raise ValueError("3D source update regions must be contiguous")
        base = int(parent.start or 0)
        parent_stop = int(parent.stop or base)
        start = max(int(r_start), base)
        stop = min(int(r_stop), parent_stop)
        stop = max(start, stop)
        out.append(slice(start - base, stop - base))
    return tuple(out)  # type: ignore[return-value]


def local_3d_phasor_context(
    field_profile: FieldProfile3D,
    fields,
    *,
    resolution: float,
    max_shift: int,
):
    from beamz.simulation.boundaries import has_full_pec_3d

    if has_full_pec_3d(getattr(fields, "boundaries", None)):
        return None

    axis = field_profile.axis
    axis_pos = _AXIS_POS_3D[axis]
    field_shapes = {
        component: tuple(int(v) for v in getattr(fields, component).shape)
        for component in _FIELD_COMPONENTS_3D
    }
    grid_shape = tuple(int(v) for v in fields.permittivity.shape)
    lows = [int(v) for v in grid_shape]
    highs = [0, 0, 0]
    found = False

    max_shift = int(max(1, max_shift))
    for component, profile in field_profile.components.items():
        index = field_profile.indices.get(component)
        if profile is None or index is None:
            continue
        for shift in range(-max_shift, max_shift + 1):
            shifted = _shift_component_indices_along_axis(
                index,
                axis,
                shift,
                field_shapes[component],
            )
            if shifted is None:
                continue
            shifted_slices = normalize_3d_component_index(
                shifted,
                field_shapes[component],
            )
            bounds = component_slices_to_cell_bbox(component, shifted_slices)
            for dim, (lo, hi) in enumerate(bounds):
                lows[dim] = min(lows[dim], int(lo))
                highs[dim] = max(highs[dim], int(hi))
            found = True

    if not found:
        return None

    halo = 4
    cell_bounds = tuple(
        (
            max(0, int(lo) - halo),
            min(int(size), int(hi) + halo),
        )
        for lo, hi, size in zip(lows, highs, grid_shape, strict=True)
    )
    component_slices = {
        component: component_slices_from_cell_bounds(
            component,
            cell_bounds,  # type: ignore[arg-type]
            field_shapes[component],
        )
        for component in _FIELD_COMPONENTS_3D
    }
    if any(
        any(int(s.stop or 0) <= int(s.start or 0) for s in slices)
        for slices in component_slices.values()
    ):
        return None

    cell_slice = tuple(slice(int(lo), int(hi)) for lo, hi in cell_bounds)
    cell_shape = tuple(int(s.stop or 0) - int(s.start or 0) for s in cell_slice)
    local_fields = SimpleNamespace(
        boundaries=getattr(fields, "boundaries", None),
        permittivity=np.empty(cell_shape, dtype=fields.permittivity.dtype),
    )

    def local_material_attr(
        attr: str,
        slices: tuple[slice, slice, slice],
    ) -> np.ndarray:
        value = getattr(fields, attr)
        arr = np.asarray(value)
        local_shape = tuple(int(s.stop or 0) - int(s.start or 0) for s in slices)
        if arr.ndim == 0:
            return np.full(local_shape, arr.item(), dtype=arr.dtype)
        return np.asarray(value[slices])

    for component, slices in component_slices.items():
        field = getattr(fields, component)
        shape = tuple(int(s.stop or 0) - int(s.start or 0) for s in slices)
        setattr(local_fields, component, np.zeros(shape, dtype=field.dtype))

    for component, attr_prefix in (
        ("Ex", "x"),
        ("Ey", "y"),
        ("Ez", "z"),
    ):
        slices = component_slices[component]
        full_shape = field_shapes[component]
        eps = local_material_attr(f"eps_{attr_prefix}", slices)
        sig = local_material_attr(f"sig_{attr_prefix}", slices)
        region = getattr(
            fields,
            f"region_{attr_prefix}",
            (slice(None), slice(None), slice(None)),
        )
        setattr(local_fields, f"eps_{attr_prefix}", eps)
        setattr(local_fields, f"sig_{attr_prefix}", sig)
        setattr(
            local_fields,
            f"region_{attr_prefix}",
            translate_region_to_local(region, slices, full_shape),
        )

    for component, attr in (
        ("Hx", "sigma_m_hx"),
        ("Hy", "sigma_m_hy"),
        ("Hz", "sigma_m_hz"),
    ):
        setattr(
            local_fields,
            attr,
            local_material_attr(attr, component_slices[component]),
        )

    offset = float(cell_bounds[axis_pos][0]) * float(resolution)
    local_indices = {
        component: shift_3d_component_index_to_local(
            index,
            component_slices[component],
            field_shapes[component],
        )
        for component, index in field_profile.indices.items()
        if index is not None
    }
    local_profile = FieldProfile3D(
        components=field_profile.components,
        indices=local_indices,  # type: ignore[arg-type]
        axis=field_profile.axis,
        direction_sign=field_profile.direction_sign,
        omega=field_profile.omega,
        k_axis=field_profile.k_axis,
        phase_ref_coord=float(field_profile.phase_ref_coord) - offset,
        phase_plane_coord=float(field_profile.phase_plane_coord) - offset,
    )
    return local_profile, local_fields, component_slices


def dense_3d_delta_residuals(
    delta: dict[str, np.ndarray],
    *,
    timing: str,
    component_indices: dict[str, tuple[slice, slice, slice]] | None = None,
) -> tuple[_ModeSource3DResidual, ...]:
    out: list[_ModeSource3DResidual] = []
    for component, values in delta.items():
        arr = np.asarray(values, dtype=np.complex128)
        full_index = (
            component_indices[component]
            if component_indices is not None
            else tuple(slice(0, int(size)) for size in arr.shape)
        )
        residual = crop_local_residual(
            component,
            timing,
            full_index,  # type: ignore[arg-type]
            arr,
        )
        if residual is not None:
            out.append(residual)
    return tuple(out)


def advance_incident_h_3d(fields, state, dt, *, resolution: float):
    """Advance an incident 3D state through the source-free H half-step."""
    from beamz.simulation import ops
    from beamz.simulation.boundaries import (
        has_full_pec_3d,
        initialize_full_pec_3d_state,
        pec_curl_e_to_h_3d,
    )

    ex = jnp.asarray(state["Ex"])
    ey = jnp.asarray(state["Ey"])
    ez = jnp.asarray(state["Ez"])
    hx = jnp.asarray(state["Hx"])
    hy = jnp.asarray(state["Hy"])
    hz = jnp.asarray(state["Hz"])
    boundaries = getattr(fields, "boundaries", None)

    if has_full_pec_3d(boundaries):
        fp_state = initialize_full_pec_3d_state(fields)
        for comp in _FIELD_COMPONENTS_3D:
            compact = jnp.asarray(state[comp])
            full = jnp.asarray(getattr(fp_state, comp), dtype=compact.dtype)
            full = full.at[:-1, :-1, :-1].set(compact)
            zero = jnp.asarray(0.0, dtype=full.dtype)
            setattr(fp_state, comp, jnp.where(fp_state.masks[comp], zero, full))
        curl_hx, curl_hy, curl_hz = pec_curl_e_to_h_3d(
            fp_state.Ex,
            fp_state.Ey,
            fp_state.Ez,
            resolution,
            fp_state.Hx.shape,
            fp_state.Hy.shape,
            fp_state.Hz.shape,
        )
        hx_next = ops.advance_h_field(fp_state.Hx, curl_hx, fp_state.sigma_m_hx, dt)
        hy_next = ops.advance_h_field(fp_state.Hy, curl_hy, fp_state.sigma_m_hy, dt)
        hz_next = ops.advance_h_field(fp_state.Hz, curl_hz, fp_state.sigma_m_hz, dt)
        hx_next = jnp.where(
            fp_state.masks["Hx"], jnp.asarray(0.0, dtype=hx_next.dtype), hx_next
        )
        hy_next = jnp.where(
            fp_state.masks["Hy"], jnp.asarray(0.0, dtype=hy_next.dtype), hy_next
        )
        hz_next = jnp.where(
            fp_state.masks["Hz"], jnp.asarray(0.0, dtype=hz_next.dtype), hz_next
        )
        return {
            "Hx": np.asarray(hx_next[:-1, :-1, :-1]),
            "Hy": np.asarray(hy_next[:-1, :-1, :-1]),
            "Hz": np.asarray(hz_next[:-1, :-1, :-1]),
        }

    curl_hx, curl_hy, curl_hz = ops.curl_e_to_h_3d(ex, ey, ez, resolution)
    return {
        "Hx": np.asarray(
            ops.advance_h_field(hx, curl_hx, fields.sigma_m_hx, dt),
        ),
        "Hy": np.asarray(
            ops.advance_h_field(hy, curl_hy, fields.sigma_m_hy, dt),
        ),
        "Hz": np.asarray(
            ops.advance_h_field(hz, curl_hz, fields.sigma_m_hz, dt),
        ),
    }


def advance_incident_e_3d(fields, state, h_next, dt, *, resolution: float):
    """Advance an incident 3D state through the source-free E half-step."""
    from beamz.simulation import ops
    from beamz.simulation.boundaries import (
        build_h_boundary_views_for_e_3d,
        full_pec_e_update_coefficients_3d,
        full_pec_update_e_from_h_3d,
        has_full_pec_3d,
        initialize_full_pec_3d_state,
    )

    ex = jnp.asarray(state["Ex"])
    ey = jnp.asarray(state["Ey"])
    ez = jnp.asarray(state["Ez"])
    hx = jnp.asarray(h_next["Hx"])
    hy = jnp.asarray(h_next["Hy"])
    hz = jnp.asarray(h_next["Hz"])
    boundaries = getattr(fields, "boundaries", None)

    if has_full_pec_3d(boundaries):
        fp_state = initialize_full_pec_3d_state(fields)
        for comp in _E_COMPONENTS_3D:
            compact = jnp.asarray(state[comp])
            full = jnp.asarray(getattr(fp_state, comp), dtype=compact.dtype)
            full = full.at[:-1, :-1, :-1].set(compact)
            zero = jnp.asarray(0.0, dtype=full.dtype)
            setattr(fp_state, comp, jnp.where(fp_state.masks[comp], zero, full))
        for comp, arr in (("Hx", hx), ("Hy", hy), ("Hz", hz)):
            compact = jnp.asarray(arr)
            full = jnp.asarray(getattr(fp_state, comp), dtype=compact.dtype)
            full = full.at[:-1, :-1, :-1].set(compact)
            zero = jnp.asarray(0.0, dtype=full.dtype)
            setattr(fp_state, comp, jnp.where(fp_state.masks[comp], zero, full))
        e_decay, e_source = full_pec_e_update_coefficients_3d(fp_state, dt)
        ex_next, ey_next, ez_next = full_pec_update_e_from_h_3d(
            fp_state.Hx,
            fp_state.Hy,
            fp_state.Hz,
            fp_state.Ex,
            fp_state.Ey,
            fp_state.Ez,
            resolution,
            e_decay=e_decay,
            e_source=e_source,
            e_mask=(
                fp_state.masks["Ex"],
                fp_state.masks["Ey"],
                fp_state.masks["Ez"],
            ),
        )
        return {
            "Ex": np.asarray(ex_next[:-1, :-1, :-1]),
            "Ey": np.asarray(ey_next[:-1, :-1, :-1]),
            "Ez": np.asarray(ez_next[:-1, :-1, :-1]),
        }

    boundaries = getattr(fields, "boundaries", None)
    boundary_views = build_h_boundary_views_for_e_3d(hx, hy, hz, boundaries)
    curl_hx, curl_hy, curl_hz = ops.curl_h_to_e_3d(
        hx,
        hy,
        hz,
        resolution,
        ex_shape=ex.shape,
        ey_shape=ey.shape,
        ez_shape=ez.shape,
        boundary_views=boundary_views,
    )
    return {
        "Ex": np.asarray(
            ops.advance_e_field(
                ex, curl_hx, fields.sig_x, fields.eps_x, dt, fields.region_x
            ),
        ),
        "Ey": np.asarray(
            ops.advance_e_field(
                ey, curl_hy, fields.sig_y, fields.eps_y, dt, fields.region_y
            ),
        ),
        "Ez": np.asarray(
            ops.advance_e_field(
                ez, curl_hz, fields.sig_z, fields.eps_z, dt, fields.region_z
            ),
        ),
    }


def _compute_3d_h_phasor_residuals_dense(
    field_profile: FieldProfile3D,
    fields,
    *,
    resolution: float,
    max_shift: int,
    dt: float,
    component_indices: dict[str, tuple[slice, slice, slice]] | None = None,
) -> tuple[_ModeSource3DResidual, ...]:
    full_prev = build_incident_3d_phasor_state(
        field_profile,
        fields,
        resolution=resolution,
        t_e=0.0,
        t_h=-0.5 * float(dt),
        masked=False,
        max_shift=max_shift,
    )
    masked_prev = build_incident_3d_phasor_state(
        field_profile,
        fields,
        resolution=resolution,
        t_e=0.0,
        t_h=-0.5 * float(dt),
        masked=True,
        max_shift=max_shift,
    )
    h_full_next = advance_incident_h_3d(fields, full_prev, dt, resolution=resolution)
    h_target_next = mask_incident_3d_state_to_launched_side(
        field_profile,
        h_full_next,
        resolution=resolution,
    )
    h_mask_next = advance_incident_h_3d(fields, masked_prev, dt, resolution=resolution)
    delta = {
        component: h_target_next[component] - h_mask_next[component]
        for component in _H_COMPONENTS_3D
    }
    return dense_3d_delta_residuals(
        delta,
        timing="h",
        component_indices=component_indices,
    )


def compute_discrete_3d_h_phasor_residuals(
    field_profile: FieldProfile3D,
    fields,
    *,
    resolution: float,
    max_shift: int,
    dt: float,
) -> tuple[_ModeSource3DResidual, ...]:
    """Complex carrier H residuals for the launched-side TF/SF update."""
    context = local_3d_phasor_context(
        field_profile,
        fields,
        resolution=resolution,
        max_shift=max_shift,
    )
    if context is None:
        return _compute_3d_h_phasor_residuals_dense(
            field_profile,
            fields,
            resolution=resolution,
            max_shift=max_shift,
            dt=dt,
        )
    local_profile, local_fields, component_slices = context
    return _compute_3d_h_phasor_residuals_dense(
        local_profile,
        local_fields,
        resolution=resolution,
        max_shift=max_shift,
        dt=dt,
        component_indices=component_slices,
    )


def _compute_3d_e_phasor_residuals_dense(
    field_profile: FieldProfile3D,
    fields,
    *,
    resolution: float,
    max_shift: int,
    dt: float,
    component_indices: dict[str, tuple[slice, slice, slice]] | None = None,
) -> tuple[_ModeSource3DResidual, ...]:
    full_prev = build_incident_3d_phasor_state(
        field_profile,
        fields,
        resolution=resolution,
        t_e=0.0,
        t_h=-0.5 * float(dt),
        masked=False,
        max_shift=max_shift,
    )
    masked_prev = build_incident_3d_phasor_state(
        field_profile,
        fields,
        resolution=resolution,
        t_e=0.0,
        t_h=-0.5 * float(dt),
        masked=True,
        max_shift=max_shift,
    )
    h_full_next = advance_incident_h_3d(fields, full_prev, dt, resolution=resolution)
    h_target_next = mask_incident_3d_state_to_launched_side(
        field_profile,
        h_full_next,
        resolution=resolution,
    )
    e_full_next = advance_incident_e_3d(
        fields,
        full_prev,
        h_full_next,
        dt,
        resolution=resolution,
    )
    e_target_next = mask_incident_3d_state_to_launched_side(
        field_profile,
        e_full_next,
        resolution=resolution,
    )
    e_mask_next = advance_incident_e_3d(
        fields,
        masked_prev,
        h_target_next,
        dt,
        resolution=resolution,
    )
    delta = {
        component: e_target_next[component] - e_mask_next[component]
        for component in _E_COMPONENTS_3D
    }
    return dense_3d_delta_residuals(
        delta,
        timing="e",
        component_indices=component_indices,
    )


def compute_discrete_3d_e_phasor_residuals(
    field_profile: FieldProfile3D,
    fields,
    *,
    resolution: float,
    max_shift: int,
    dt: float,
) -> tuple[_ModeSource3DResidual, ...]:
    """Complex carrier E residuals for the launched-side TF/SF update."""
    context = local_3d_phasor_context(
        field_profile,
        fields,
        resolution=resolution,
        max_shift=max_shift,
    )
    if context is None:
        return _compute_3d_e_phasor_residuals_dense(
            field_profile,
            fields,
            resolution=resolution,
            max_shift=max_shift,
            dt=dt,
        )
    local_profile, local_fields, component_slices = context
    return _compute_3d_e_phasor_residuals_dense(
        local_profile,
        local_fields,
        resolution=resolution,
        max_shift=max_shift,
        dt=dt,
        component_indices=component_slices,
    )


def compute_discrete_3d_h_delta(
    field_profile: FieldProfile3D,
    fields,
    *,
    resolution: float,
    max_shift: int,
    get_signal_value,
    get_signal_quadrature_value,
    t,
    dt,
) -> dict[str, np.ndarray]:
    """Exact discrete H-source residual for the current split launch step."""
    full_prev = build_incident_3d_state(
        field_profile,
        fields,
        resolution=resolution,
        t_e=float(t),
        t_h=float(t - 0.5 * dt),
        dt=dt,
        masked=False,
        get_signal_value=get_signal_value,
        get_signal_quadrature_value=get_signal_quadrature_value,
        max_shift=max_shift,
    )
    masked_prev = build_incident_3d_state(
        field_profile,
        fields,
        resolution=resolution,
        t_e=float(t),
        t_h=float(t - 0.5 * dt),
        dt=dt,
        masked=True,
        get_signal_value=get_signal_value,
        get_signal_quadrature_value=get_signal_quadrature_value,
        max_shift=max_shift,
    )
    h_full_next = advance_incident_h_3d(fields, full_prev, dt, resolution=resolution)
    h_target_next = mask_incident_3d_state_to_launched_side(
        field_profile,
        h_full_next,
        resolution=resolution,
    )
    h_mask_next = advance_incident_h_3d(fields, masked_prev, dt, resolution=resolution)
    return {
        "Hx": h_target_next["Hx"] - h_mask_next["Hx"],
        "Hy": h_target_next["Hy"] - h_mask_next["Hy"],
        "Hz": h_target_next["Hz"] - h_mask_next["Hz"],
    }


def compute_discrete_3d_e_delta(
    field_profile: FieldProfile3D,
    fields,
    *,
    resolution: float,
    max_shift: int,
    get_signal_value,
    get_signal_quadrature_value,
    t,
    dt,
) -> dict[str, np.ndarray]:
    """Exact discrete E-source residual for the current split launch step."""
    full_prev = build_incident_3d_state(
        field_profile,
        fields,
        resolution=resolution,
        t_e=float(t),
        t_h=float(t - 0.5 * dt),
        dt=dt,
        masked=False,
        get_signal_value=get_signal_value,
        get_signal_quadrature_value=get_signal_quadrature_value,
        max_shift=max_shift,
    )
    masked_prev = build_incident_3d_state(
        field_profile,
        fields,
        resolution=resolution,
        t_e=float(t),
        t_h=float(t - 0.5 * dt),
        dt=dt,
        masked=True,
        get_signal_value=get_signal_value,
        get_signal_quadrature_value=get_signal_quadrature_value,
        max_shift=max_shift,
    )
    h_full_next = advance_incident_h_3d(fields, full_prev, dt, resolution=resolution)
    h_target_next = mask_incident_3d_state_to_launched_side(
        field_profile,
        h_full_next,
        resolution=resolution,
    )
    e_full_next = advance_incident_e_3d(
        fields,
        full_prev,
        h_full_next,
        dt,
        resolution=resolution,
    )
    e_target_next = mask_incident_3d_state_to_launched_side(
        field_profile,
        e_full_next,
        resolution=resolution,
    )
    e_mask_next = advance_incident_e_3d(
        fields,
        masked_prev,
        h_target_next,
        dt,
        resolution=resolution,
    )
    return {
        "Ex": e_target_next["Ex"] - e_mask_next["Ex"],
        "Ey": e_target_next["Ey"] - e_mask_next["Ey"],
        "Ez": e_target_next["Ez"] - e_mask_next["Ez"],
    }


def expand_3d_residuals(
    residuals: tuple[_ModeSource3DResidual, ...],
    fields,
    components: tuple[str, ...],
) -> dict[str, np.ndarray]:
    expanded = {
        component: np.zeros(
            tuple(int(v) for v in getattr(fields, component).shape),
            dtype=np.complex128,
        )
        for component in components
    }
    for residual in residuals:
        if residual.component in expanded:
            expanded[residual.component][residual.index] += np.asarray(
                residual.residual,
                dtype=np.complex128,
            )
    return expanded
