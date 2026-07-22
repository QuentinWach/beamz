"""Optional multi-device lowering, padding, placement, and result cropping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation.model import (
    CpmlPackedSlabSpec,
    ShardingConfig,
    ShardingLayout,
    ShardingPlan,
)

# Sharding may pad the high side for equal device partitions. Padding is storage-only;
# logical component shapes remain authoritative for curls, monitors, and result crops.
_COMPONENT_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_AXIS_TO_INDEX = {"z": 0, "y": 1, "x": 2}
_INDEX_TO_AXIS = ("z", "y", "x")
_MESH_AXIS = "fdtd"


def _disabled_sharding_config() -> ShardingConfig:
    # Use one canonical disabled value so cache tokens do not depend on input spelling.
    return ShardingConfig(enabled=False)


def normalize_sharding_config(value) -> ShardingConfig:
    """Normalize public sharding input into a stable configuration."""

    # Resolve device layout once on the host so execution receives concrete placement
    # objects.
    if value is None or value is False:
        return _disabled_sharding_config()
    if isinstance(value, ShardingConfig):
        return value
    if value is True:
        return ShardingConfig(enabled=True)
    if isinstance(value, Mapping):
        raw = dict(value)
        enabled = bool(raw.pop("enabled", True))
        return ShardingConfig(enabled=enabled, **raw)
    raise TypeError(
        "sharding must be None, bool, dict, or beamz.simulation.ShardingConfig"
    )


def sharding_cache_token(value) -> tuple[bool, str, int | None, str | None]:
    """Return a stable scalar cache token for a sharding selection."""
    # Key cached work by semantic execution inputs so equivalent states reuse
    # compilation safely.
    cfg = normalize_sharding_config(value)
    return (bool(cfg.enabled), cfg.axis, cfg.num_devices, cfg.backend)


def _select_sharding_axis(axis: str, grid_shape: tuple[int, int, int]) -> str:
    # Resolve device layout once on the host so execution receives concrete placement
    # objects.
    axis = str(axis).lower()
    if axis != "auto":
        if axis not in _AXIS_TO_INDEX:
            raise ValueError("sharding axis must be one of: 'auto', 'z', 'y', 'x'")
        return axis
    lengths = tuple(int(v) for v in grid_shape)
    best = max(range(3), key=lambda idx: (lengths[idx], -idx))
    return _INDEX_TO_AXIS[best]


def _jax_devices_for_config(cfg: ShardingConfig) -> tuple[object, ...]:
    # Choose devices deterministically to keep cache keys and array placement
    # reproducible.
    if not cfg.enabled:
        return ()
    try:
        devices = (
            tuple(jax.devices(cfg.backend)) if cfg.backend else tuple(jax.devices())
        )
    except Exception as exc:
        backend = cfg.backend or "default"
        raise ValueError(
            f"No JAX devices are available for backend {backend!r}"
        ) from exc
    if not devices:
        backend = cfg.backend or "default"
        raise ValueError(f"No JAX devices are available for backend {backend!r}")
    num_devices = len(devices) if cfg.num_devices is None else int(cfg.num_devices)
    if num_devices <= 1:
        return ()
    if num_devices > len(devices):
        raise ValueError(
            f"Requested {num_devices} sharding devices, but only {len(devices)} "
            f"are available for backend {cfg.backend or 'default'}"
        )
    return devices[:num_devices]


def _pad_shape_for_devices(
    shape: tuple[int, ...], axis: int, num_devices: int
) -> tuple[int, ...]:
    # Read dimensions from the authoritative layout so all allocations share one
    # static contract.
    out = list(int(v) for v in shape)
    size = out[int(axis)]
    remainder = size % int(num_devices)
    if remainder:
        out[int(axis)] = size + (int(num_devices) - remainder)
    return tuple(out)


def build_sharding_plan(
    fields,
    cfg: ShardingConfig,
    *,
    is_3d: bool,
) -> ShardingPlan:
    # 1. Normalize the material-grid rank to canonical z-y-x order so every later shape
    # calculation starts from the same physical domain contract.
    logical_base_values = tuple(int(v) for v in fields.permittivity.shape)
    if len(logical_base_values) == 2:
        logical_base_shape = (
            1,
            logical_base_values[0],
            logical_base_values[1],
        )
    elif len(logical_base_values) == 3:
        logical_base_shape = (
            logical_base_values[0],
            logical_base_values[1],
            logical_base_values[2],
        )
    else:
        raise ValueError(
            f"Compiled storage requires a rank-2 or rank-3 material grid, got "
            f"{logical_base_values!r}."
        )
    # 2. Build the unsharded layout directly from the canonical physical supports.
    if not cfg.enabled:
        logical_shapes = {
            name: tuple(int(v) for v in getattr(fields, name).shape)
            for name in _COMPONENT_NAMES
        }
        return ShardingPlan(
            ShardingLayout(
                enabled=False,
                axis_name="z",
                axis=0,
                num_devices=1,
                backend=cfg.backend,
                logical_shapes=logical_shapes,
                padded_shapes=dict(logical_shapes),
            ),
            None,
        )
    # 3. Validate the sharded path early because its partitioning assumptions only hold
    # for three-dimensional component grids.
    if not is_3d:
        raise NotImplementedError("compiled sharding currently supports 3D runs only")
    if len(logical_base_shape) != 3:
        raise ValueError(
            f"3D sharding requires a 3-axis grid, got {logical_base_shape}"
        )

    # 4. Resolve eligible devices, falling back through this same planner when the
    # requested backend supplies none.
    devices = _jax_devices_for_config(cfg)
    if not devices:
        return build_sharding_plan(
            fields,
            ShardingConfig(enabled=False, backend=cfg.backend),
            is_3d=is_3d,
        )
    # 5. Select the partition axis and pad every staggered component independently so
    # each device receives an equal static extent.
    axis_name = _select_sharding_axis(cfg.axis, logical_base_shape)
    axis = _AXIS_TO_INDEX[axis_name]
    num_devices = len(devices)
    logical_shapes = {
        name: tuple(int(v) for v in getattr(fields, name).shape)
        for name in _COMPONENT_NAMES
    }
    padded_shapes = {
        name: _pad_shape_for_devices(shape, axis, num_devices)
        for name, shape in logical_shapes.items()
    }
    # 6. Resolve the mesh once; every later placement reuses this exact object.
    return ShardingPlan(
        ShardingLayout(
            enabled=True,
            axis_name=axis_name,
            axis=axis,
            num_devices=num_devices,
            backend=cfg.backend,
            logical_shapes=logical_shapes,
            padded_shapes=padded_shapes,
        ),
        jax.sharding.Mesh(np.asarray(devices, dtype=object), (_MESH_AXIS,)),
    )


def _array_shape(arr) -> tuple[int, ...]:
    # Read dimensions from the authoritative layout so all allocations share one
    # static contract.
    return tuple(int(v) for v in getattr(arr, "shape", np.shape(arr)))


def _array_ndim(arr) -> int:
    # Make conversion explicit to keep dtype and device placement from changing
    # tracing signatures.
    ndim = getattr(arr, "ndim", None)
    if ndim is not None:
        return int(ndim)
    return len(_array_shape(arr))


def _pad_high_to_shape(arr, shape: tuple[int, ...], *, pad_value=0.0):
    # Preserve NumPy setup arrays on the host and use the same crop/pad recipe for JAX.
    target_shape = tuple(int(v) for v in shape)
    xp = np if isinstance(arr, np.ndarray) else jnp
    arr = xp.asarray(arr)
    if tuple(arr.shape) == target_shape:
        return arr
    if arr.ndim != len(target_shape):
        raise ValueError(
            f"Cannot fit rank-{arr.ndim} array to rank-{len(target_shape)} shape"
        )
    out = arr[
        tuple(
            slice(0, min(int(size), target))
            for size, target in zip(arr.shape, target_shape, strict=True)
        )
    ]
    padding = tuple(
        (0, max(0, target - int(size)))
        for size, target in zip(out.shape, target_shape, strict=True)
    )
    return (
        xp.pad(out, padding, mode="constant", constant_values=pad_value)
        if any(high for _, high in padding)
        else out
    )


def _crop_high_to_shape(arr, shape: tuple[int, ...]) -> jnp.ndarray:
    # Read dimensions from the authoritative layout so all allocations share one
    # static contract.
    slices = tuple(slice(0, int(v)) for v in shape)
    return jnp.asarray(arr)[slices]


_COEFFICIENT_COMPONENTS = {
    **{
        f"h_{kind}_{axis}": f"H{axis}"
        for axis in "xyz"
        for kind in ("decay", "source", "sigma_m")
    },
    **{
        f"e_{kind}_{axis}": f"E{axis}"
        for axis in "xyz"
        for kind in (
            "decay",
            "source",
            "conductivity",
            "permittivity",
        )
    },
}


def _lower_coefficients(coefficients, layout: ShardingLayout):
    """Pad compiled material coefficients without recreating a fields object."""
    if not layout.enabled:
        return coefficients
    updates = {}
    for name, component in _COEFFICIENT_COMPONENTS.items():
        value = getattr(coefficients, name)
        if _array_ndim(value) == 0 or not int(np.prod(_array_shape(value))):
            continue
        neutral = 1.0 if "_decay_" in name or "_permittivity_" in name else 0.0
        updates[name] = _pad_high_to_shape(
            value, layout.padded_shapes[component], pad_value=neutral
        )
    return coefficients._replace(**updates)


def _lower_cpml_term(term, layout: ShardingLayout):
    """Extend one packed CPML slab across padded transverse dimensions."""
    target_shape = list(layout.padded_shapes[term.component])
    target_shape[term.axis] = term.slab.low + term.slab.high
    target_shape = tuple(target_shape)
    slab = CpmlPackedSlabSpec(term.axis, term.slab.low, term.slab.high, target_shape)
    return replace(
        term,
        a=_pad_high_to_shape(term.a, target_shape, pad_value=0.0),
        b=_pad_high_to_shape(term.b, target_shape, pad_value=1.0),
        inv_kappa=_pad_high_to_shape(term.inv_kappa, target_shape, pad_value=1.0),
        slab=slab,
    )


def lower_compiled_arrays(coefficients, boundary, layout: ShardingLayout):
    """Lower logical coefficients and boundary arrays to a sharded storage layout."""
    if not layout.enabled:
        return coefficients, boundary
    coefficients = _lower_coefficients(coefficients, layout)
    cpml = replace(
        boundary.cpml,
        h_terms=tuple(_lower_cpml_term(term, layout) for term in boundary.cpml.h_terms),
        e_terms=tuple(_lower_cpml_term(term, layout) for term in boundary.cpml.e_terms),
    )
    masks = {
        f"{component.lower()}_mask": _pad_high_to_shape(
            getattr(boundary.metallic, f"{component.lower()}_mask"),
            layout.padded_shapes[component],
            # Storage-only cells behave like PEC cells and remain identically zero.
            pad_value=True,
        )
        for component in _COMPONENT_NAMES
    }
    return coefficients, replace(
        boundary,
        cpml=cpml,
        metallic=replace(boundary.metallic, **masks),
    )


def _replicated_sharding(mesh):
    return (
        None
        if mesh is None
        else jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    )


def _array_sharding(program, arr, mesh):
    shape, axis = _array_shape(arr), int(program.sharding.layout.axis)
    if (
        len(shape) > axis
        and int(shape[axis]) > 0
        and int(shape[axis]) % int(program.sharding.layout.num_devices) == 0
    ):
        spec: list[str | None] = [None] * len(shape)
        spec[axis] = _MESH_AXIS
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
    return _replicated_sharding(mesh)


def place_tree(program, tree, *, shard_arrays: bool = True):
    """Place one complete pytree according to the program's backend plan."""
    if not program.sharding.layout.enabled:
        try:
            devices = jax.devices(jax.default_backend())
        except Exception:
            devices = jax.devices()
        return jax.tree_util.tree_map(
            lambda value: (
                jax.device_put(jnp.asarray(value), devices[0])
                if devices
                else jnp.asarray(value)
            ),
            tree,
        )
    mesh = program.sharding.mesh
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(
            value,
            _array_sharding(program, value, mesh)
            if shard_arrays
            else _replicated_sharding(mesh),
        ),
        tree,
    )


def pad_component(program, component: str, value):
    """Pad one logical component to its optional device-partition shape."""
    return _pad_high_to_shape(
        value,
        program.sharding.layout.padded_shapes[component],
        pad_value=0.0,
    )


def crop_component(program, component: str, value):
    """Crop one backend component back to its logical Yee support."""
    return _crop_high_to_shape(value, program.sharding.layout.logical_shapes[component])


def prepare_state(program, state, *, replicated_fields):
    """Pad and place a logical runtime state for execution."""
    state = state._replace(
        **{
            name.lower(): pad_component(program, name, getattr(state, name.lower()))
            for name in _COMPONENT_NAMES
        }
    )
    placed = place_tree(program, state)
    return placed._replace(
        **{
            name: place_tree(program, getattr(state, name), shard_arrays=False)
            for name in replicated_fields
        }
    )


def crop_state(program, state):
    """Remove backend padding before publishing a continuation state."""
    if not program.sharding.layout.enabled:
        return state
    return state._replace(
        **{
            name.lower(): crop_component(program, name, getattr(state, name.lower()))
            for name in _COMPONENT_NAMES
        }
    )
