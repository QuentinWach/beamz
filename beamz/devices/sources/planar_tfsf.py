"""Planar launched-side TF/SF residual machinery for 3D sources."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import jax.numpy as jnp
import numpy as np

from beamz.devices._immutable import readonly_array
from beamz.devices.modes.discrete import ComponentIndex
from beamz.devices.modes.fields import _axis_coordinate, _axis_index, _phase_delay
from beamz.lattice import (
    advance_e_field,
    advance_h_field,
    build_h_boundary_views_for_e_3d,
    component_axis_offsets_3d,
    curl_e_to_h_3d,
    curl_e_to_h_3d_metric,
    curl_h_to_e_3d,
    curl_h_to_e_3d_metric,
)

from .specs import FieldProfile3D

_FIELD_COMPONENTS_3D = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_E_COMPONENTS_3D = ("Ex", "Ey", "Ez")
_H_COMPONENTS_3D = ("Hx", "Hy", "Hz")
_AXIS_POS_3D = {"z": 0, "y": 1, "x": 2}
_STAGGERED_ALONG_AXIS = {
    "x": {"Ex", "Hy", "Hz"},
    "y": {"Ey", "Hx", "Hz"},
    "z": {"Ez", "Hx", "Hy"},
}


def _shift_component_indices_along_axis(
    indices: ComponentIndex | None, axis, shift, field_shape
) -> ComponentIndex | None:
    """Shift component support by integer cells along the propagation axis."""
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
    return cast(ComponentIndex, tuple(out))


def _shape3(shape) -> tuple[int, int, int]:
    values = tuple(int(v) for v in shape)
    if len(values) != 3:
        raise ValueError(f"Expected a 3D shape, got {values!r}.")
    return (values[0], values[1], values[2])


@dataclass(frozen=True)
class ModeSource3DResidual:
    """Compact local 3D source residual emitted by ModeSource compilation."""

    component: str
    timing: str
    index: tuple[slice, slice, slice]
    residual: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "residual", readonly_array(self.residual))


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


def _component_axis_coordinate(fields, component, axis, index, resolution) -> float:
    grid = getattr(fields, "geometry", None)
    if grid is None or grid.metric_kind == "isotropic_uniform":
        return _axis_coordinate(component, index, axis, resolution)
    return _axis_coordinate(component, index, axis, resolution, grid)


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
    axis = field_profile.axis
    k_num = _require_k_axis(field_profile)
    omega = float(field_profile.omega)
    plane_coord = float(field_profile.phase_plane_coord)
    ref_coord = float(field_profile.phase_ref_coord)
    direction_sign = float(field_profile.direction_sign)
    max_shift = int(max(1, max_shift))

    field_arrays = _field_arrays_like(fields, dtype=np.complex128)
    field_shapes = {name: arr.shape for name, arr in field_arrays.items()}

    for comp_name, profile in field_profile.components.items():
        idx = field_profile.indices.get(comp_name)
        if idx is None:
            continue

        profile_arr = np.asarray(profile, dtype=np.complex128)
        base_time = float(t_e if comp_name.startswith("E") else t_h)

        for shift in range(-max_shift, max_shift + 1):
            shifted_idx = _shift_component_indices_along_axis(
                idx, axis, shift, field_shapes[comp_name]
            )
            if shifted_idx is None:
                continue

            shifted_axis_idx = _axis_index(shifted_idx, axis)
            coord = _component_axis_coordinate(
                fields, comp_name, axis, shifted_axis_idx, resolution
            )
            if masked:
                mask_coord = (
                    ref_coord
                    if comp_name in _STAGGERED_ALONG_AXIS[axis]
                    else plane_coord
                )
                if direction_sign * (coord - mask_coord) < -1e-12:
                    continue

            delay = _phase_delay(omega, k_num, coord - ref_coord)
            phase = omega * (base_time - delay)
            field_arrays[comp_name][shifted_idx] = field_arrays[comp_name][
                shifted_idx
            ] + profile_arr * np.exp(1j * phase)

    return field_arrays


def deembed_3d_phasor_profiles(
    field_profile: FieldProfile3D,
    state: dict[str, np.ndarray],
    fields,
    *,
    resolution: float,
    t_e,
    t_h,
) -> dict[str, np.ndarray]:
    """Return local source-plane phasors in the source profile gauge."""
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
        axis_idx = _axis_index(idx, axis)
        coord = _component_axis_coordinate(
            fields, component, axis, axis_idx, resolution
        )
        base_time = float(t_e if component.startswith("E") else t_h)
        delay = _phase_delay(omega, k_num, coord - ref_coord)
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
    axis_pos = _AXIS_POS_3D[axis]
    grid = getattr(field_profile, "grid", None)
    if grid is None or grid.metric_kind == "isotropic_uniform":
        offset = 1.0 if component in _STAGGERED_ALONG_AXIS[axis] else 0.5
        coord = (np.arange(int(shape[axis_pos]), dtype=np.float64) + offset) * float(
            resolution
        )
    else:
        base = (
            np.asarray(grid.axis_edges(axis))[1:]
            if component in _STAGGERED_ALONG_AXIS[axis]
            else np.asarray(grid.centers(axis))
        )
        count = int(shape[axis_pos])
        if base.size < count:
            widths = np.asarray(grid.cell_widths(axis))
            extra = base[-1] + widths[-1] * np.arange(1, count - base.size + 1)
            base = np.concatenate((base, extra))
        coord = base[:count]
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
) -> ModeSource3DResidual | None:
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
    return ModeSource3DResidual(
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
    for axis, (lo, hi), dim in zip(
        ("z", "y", "x"), cell_bounds, field_shape, strict=True
    ):
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
    geometry = getattr(fields, "geometry", None)
    if geometry is not None and geometry.metric_kind != "isotropic_uniform":
        # A cropped component support may omit one outer edge, while derivative
        # metrics are cell based. Keep the exact full-grid metric alignment for
        # rectilinear launches; compact uniform launches retain the fast path.
        return None
    axis = field_profile.axis
    axis_pos = _AXIS_POS_3D[axis]
    field_shapes = {
        component: _shape3(getattr(fields, component).shape)
        for component in _FIELD_COMPONENTS_3D
    }
    grid_shape = _shape3(fields.permittivity.shape)
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
    cell_bounds = cast(
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        tuple(
            (
                max(0, int(lo) - halo),
                min(int(size), int(hi) + halo),
            )
            for lo, hi, size in zip(lows, highs, grid_shape, strict=True)
        ),
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
        grid=getattr(local_fields, "geometry", None),
        power_weights=field_profile.power_weights,
    )
    return local_profile, local_fields, component_slices


def dense_3d_delta_residuals(
    delta: dict[str, np.ndarray],
    *,
    timing: str,
    component_indices: dict[str, tuple[slice, slice, slice]] | None = None,
) -> tuple[ModeSource3DResidual, ...]:
    out: list[ModeSource3DResidual] = []
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
    ex = jnp.asarray(state["Ex"])
    ey = jnp.asarray(state["Ey"])
    ez = jnp.asarray(state["Ez"])
    hx = jnp.asarray(state["Hx"])
    hy = jnp.asarray(state["Hy"])
    hz = jnp.asarray(state["Hz"])
    grid = getattr(fields, "geometry", None)
    curl_hx, curl_hy, curl_hz = (
        curl_e_to_h_3d(ex, ey, ez, resolution)
        if grid is None or grid.metric_kind == "isotropic_uniform"
        else curl_e_to_h_3d_metric(ex, ey, ez, grid)
    )
    return {
        "Hx": np.asarray(
            advance_h_field(hx, curl_hx, fields.sigma_m_hx, dt),
        ),
        "Hy": np.asarray(
            advance_h_field(hy, curl_hy, fields.sigma_m_hy, dt),
        ),
        "Hz": np.asarray(
            advance_h_field(hz, curl_hz, fields.sigma_m_hz, dt),
        ),
    }


def advance_incident_e_3d(fields, state, h_next, dt, *, resolution: float):
    """Advance an incident 3D state through the source-free E half-step."""
    from beamz.devices._boundary_compile import resolve_metallic_edges

    ex = jnp.asarray(state["Ex"])
    ey = jnp.asarray(state["Ey"])
    ez = jnp.asarray(state["Ez"])
    hx = jnp.asarray(h_next["Hx"])
    hy = jnp.asarray(h_next["Hy"])
    hz = jnp.asarray(h_next["Hz"])
    boundaries = getattr(fields, "boundaries", None)
    boundary_views = build_h_boundary_views_for_e_3d(
        hx, hy, hz, frozenset(resolve_metallic_edges(boundaries, is_3d=True))
    )
    grid = getattr(fields, "geometry", None)
    curl_kwargs = dict(
        ex_shape=ex.shape,
        ey_shape=ey.shape,
        ez_shape=ez.shape,
        boundary_views=boundary_views,
    )
    curl_hx, curl_hy, curl_hz = (
        curl_h_to_e_3d(hx, hy, hz, resolution, **curl_kwargs)
        if grid is None or grid.metric_kind == "isotropic_uniform"
        else curl_h_to_e_3d_metric(hx, hy, hz, grid, **curl_kwargs)
    )
    return {
        "Ex": np.asarray(
            advance_e_field(
                ex, curl_hx, fields.sig_x, fields.eps_x, dt, fields.region_x
            ),
        ),
        "Ey": np.asarray(
            advance_e_field(
                ey, curl_hy, fields.sig_y, fields.eps_y, dt, fields.region_y
            ),
        ),
        "Ez": np.asarray(
            advance_e_field(
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
) -> tuple[ModeSource3DResidual, ...]:
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
) -> tuple[ModeSource3DResidual, ...]:
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
) -> tuple[ModeSource3DResidual, ...]:
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
) -> tuple[ModeSource3DResidual, ...]:
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
