"""Lower discretized simulation requests into immutable execution programs."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

import beamz.simulation.kernels as ops
from beamz._cache_tokens import HashToken, cache_token
from beamz.devices._boundary_compile import (
    CPML_3D_E_DERIVATIVES,
    CPML_3D_H_DERIVATIVES,
    BoundaryData,
    lower_boundaries,
)
from beamz.devices.monitors.compiler import compile_monitor_specs
from beamz.devices.sources.compiler import compile_source_specs
from beamz.lattice import (
    build_material_coefficients,
    component_shapes,
    sample_voxel_grid_at_component_2d,
    sample_voxel_grid_at_e_component_3d_centered,
)
from beamz.simulation.model import (
    BoundaryPlan,
    CompiledGrid,
    CompiledProgram,
    CpmlPlan,
    DerivativeMetricPlan,
    MetallicPlan,
    RunConfig,
    ShardingConfig,
    ShardingPlan,
    ShardingToken,
    SimulationRequest,
    UpdateCoefficients,
)

from .observe import MONITOR_FIELDS, empty_monitor_values
from .sharding import (
    build_sharding_plan,
    lower_compiled_arrays,
    normalize_sharding_config,
    sharding_cache_token,
)

# Cache identity contains only values that affect generated code or static storage;
# runtime field values stay excluded so one executable can serve many states.


@dataclass(frozen=True, slots=True)
class CompiledProgramKey:
    # Group these values because their positional and shape relationships form one
    # invariant.
    num_steps: int
    total_steps: int
    t0: float
    dt: float
    is_3d: bool
    plane_2d: str
    polarization_2d: str
    loop_kind: str
    source_single_slab_dense: bool
    sharding: ShardingToken
    materials: HashToken
    sources: tuple[HashToken, ...]
    monitors: tuple[HashToken, ...]
    boundaries: tuple[HashToken, ...]

    @classmethod
    def from_request(cls, request: SimulationRequest) -> CompiledProgramKey:
        # Reduce rich objects to semantic tokens; large arrays must not become key members.
        return cls(
            request.run.num_steps,
            request.run.total_steps,
            request.run.t0,
            request.run.dt,
            request.domain.is_3d,
            request.domain.plane_2d,
            request.domain.polarization_2d,
            request.run.loop_kind,
            request.run.source_single_slab_dense,
            request.run.sharding,
            cache_token(request.materials),
            tuple(cache_token(source) for source in request.sources),
            tuple(cache_token(monitor) for monitor in request.monitors),
            tuple(cache_token(boundary) for boundary in request.boundaries),
        )


_MAX_COMPILED_PROGRAMS = 4
_PROGRAM_CACHE: OrderedDict[CompiledProgramKey, CompiledProgram] = OrderedDict()


def _resolved_setup_device_context(resolved_device):
    """Place setup work on CPU only when explicitly requested."""
    if str(resolved_device).strip().lower() != "cpu":
        return nullcontext(None)
    try:
        devices = jax.devices("cpu")
        return jax.default_device(devices[0]) if devices else nullcontext(None)
    except Exception:
        return nullcontext(None)


def clear_program_cache() -> None:
    """Drop cached immutable compiled plans."""
    _PROGRAM_CACHE.clear()


def _elide_zero_conductivity_grid(value):
    """Return a scalar zero when a conductivity-like grid is identically zero."""

    # Zero padding/disabled state explicitly so non-physical cells cannot inject
    # energy.
    arr_np = np.asarray(value)
    if arr_np.size and not bool(np.any(arr_np != 0.0)):
        return jnp.asarray(0.0, dtype=getattr(value, "dtype", jnp.float32))
    return value


def _compile_derivative_metrics(material_grid) -> DerivativeMetricPlan:
    """Precompute O(nx + ny + nz) staggered inverse-distance metrics."""
    kind = material_grid.metric_kind
    empty = jnp.zeros((0,), dtype=jnp.float32)
    if kind == "isotropic_uniform":
        return DerivativeMetricPlan(*(empty for _ in range(6)))

    assert material_grid.grid is not None
    active_axes = ("x", "y", "z") if len(material_grid.shape) == 3 else ("x", "y")
    forward = {}
    backward = {}
    for axis in active_axes:
        widths = material_grid.grid.cell_widths(axis)
        if kind == "axis_uniform":
            forward[axis] = backward[axis] = jnp.asarray(
                1.0 / float(widths[0]), dtype=jnp.float32
            )
            continue
        inverse_forward = 1.0 / widths
        inverse_backward = np.empty(widths.size + 1, dtype=np.float64)
        inverse_backward[0] = 1.0 / widths[0]
        inverse_backward[-1] = 1.0 / widths[-1]
        if widths.size > 1:
            inverse_backward[1:-1] = 2.0 / (widths[:-1] + widths[1:])
        forward[axis] = jnp.asarray(inverse_forward, dtype=jnp.float32)
        backward[axis] = jnp.asarray(inverse_backward, dtype=jnp.float32)
    return DerivativeMetricPlan(
        *(forward.get(axis, empty) for axis in ("x", "y", "z")),
        *(backward.get(axis, empty) for axis in ("x", "y", "z")),
    )


def _compile_cpml_plan(
    fields, *, dt, is_3d, metallic_edges, polarization_2d: str = "tm"
) -> CpmlPlan:
    """Compile every active derivative into the same packed coefficient record."""
    data = getattr(fields, "pml_data", None)
    enabled = bool(getattr(fields, "has_cpml", False) and data)
    if not enabled:
        return CpmlPlan(False, frozenset(metallic_edges), (), ())
    assert data is not None

    def term(prefix, component, axis, sign, shape):
        if data is None:
            raise RuntimeError("CPML profiles disappeared during compilation.")
        return ops.compile_cpml_term(
            component=component,
            axis=axis,
            sign=sign,
            sigma=data[f"{prefix}_sigma"],
            kappa=data[f"{prefix}_kappa"],
            alpha=data[f"{prefix}_alpha"],
            dt=dt,
            full_shape=tuple(int(value) for value in shape),
        )

    if is_3d:
        axis_index = {"z": 0, "y": 1, "x": 2}

        def terms_for(specs):
            return tuple(
                term(
                    f"cpml3d_{spec.name}",
                    spec.target_component,
                    axis_index[spec.derivative_axis],
                    1.0 if index % 2 == 0 else -1.0,
                    getattr(fields, spec.target_component).shape,
                )
                for index, spec in enumerate(specs)
            )

        h_terms = terms_for(CPML_3D_H_DERIVATIVES)
        e_terms = terms_for(CPML_3D_E_DERIVATIVES)
    else:
        profiles = data.get(f"{polarization_2d}_xy_cpml")
        if profiles is None:
            return CpmlPlan(False, frozenset(metallic_edges), (), ())
        data = profiles
        if polarization_2d == "tm":
            h_terms = (
                term("Hx_y", "Hx", 0, 1.0, fields.Hx.shape),
                term("Hy_x", "Hy", 1, -1.0, fields.Hy.shape),
            )
            e_terms = (
                term("Ez_x", "Ez", 1, 1.0, fields.Ez.shape),
                term("Ez_y", "Ez", 0, -1.0, fields.Ez.shape),
            )
        else:
            h_terms = (
                term("Hz_x", "Hz", 1, 1.0, fields.Hz.shape),
                term("Hz_y", "Hz", 0, -1.0, fields.Hz.shape),
            )
            e_terms = (
                term("Ex_y", "Ex", 0, 1.0, fields.Ex.shape),
                term("Ey_x", "Ey", 1, -1.0, fields.Ey.shape),
            )
    return CpmlPlan(True, frozenset(metallic_edges), h_terms, e_terms)


@dataclass(frozen=True, slots=True)
class _CompileSetup:
    """Shape-defining values shared by all compilation phases."""

    dt: float
    sharding: ShardingPlan
    source_specs: tuple[Any, ...]
    monitor_specs: tuple[Any, ...]
    config: RunConfig


def _prepare_compilation(
    request: SimulationRequest, logical_fields: CompiledGrid
) -> _CompileSetup:
    """Resolve shapes, devices, sources, monitors, and scalar run configuration."""
    resolution = float(request.materials.resolution)
    dt = float(request.run.dt)
    num_steps = int(request.run.num_steps)
    sharding_cfg = normalize_sharding_config(request.compiler_sharding)
    sharding = build_sharding_plan(
        logical_fields,
        sharding_cfg,
        is_3d=bool(request.domain.is_3d),
    )
    sharding_layout = sharding.layout
    effective_sharding = (
        ShardingConfig(
            enabled=True,
            axis=cast(Literal["auto", "z", "y", "x"], sharding_layout.axis_name),
            num_devices=sharding_layout.num_devices,
            backend=sharding_cfg.backend,
        )
        if sharding_layout.enabled
        else ShardingConfig(
            enabled=False,
            axis=sharding_cfg.axis,
            num_devices=sharding_cfg.num_devices,
            backend=sharding_cfg.backend,
        )
    )
    source_specs = compile_source_specs(
        request.sources,
        logical_fields,
        dt,
        resolution,
        num_steps,
        request.run.t0,
        request.run.total_steps,
        request.domain,
        grid=logical_fields.geometry,
    )
    monitor_fields = SimpleNamespace(
        **{
            name: SimpleNamespace(shape=shape)
            for name, shape in sharding_layout.padded_shapes.items()
        },
        permittivity=logical_fields.permittivity,
        plane_2d=logical_fields.plane_2d,
        polarization_2d=logical_fields.polarization_2d,
        _logical_component_shapes=sharding_layout.logical_shapes,
    )
    monitor_specs, _ = compile_monitor_specs(
        request.monitors,
        monitor_fields,
        resolution,
        num_steps,
        dt,
        request.domain.plane_2d,
        request.domain.polarization_2d,
        grid=logical_fields.geometry,
    )
    loop_aliases = {
        "fori": "fori_loop",
        "fori_loop": "fori_loop",
        "fori-loop": "fori_loop",
        "scan": "scan",
    }
    try:
        loop_kind = loop_aliases[str(request.run.loop_kind).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "Invalid compiled loop kind. Use one of: scan, fori_loop."
        ) from exc
    config = RunConfig(
        resolution=resolution,
        dt=dt,
        num_steps=num_steps,
        plane_2d=request.domain.plane_2d,
        is_3d=bool(request.domain.is_3d),
        metric_kind=request.materials.metric_kind,
        polarization_2d=request.domain.polarization_2d,
        loop_kind=loop_kind,
        source_single_slab_dense=bool(request.run.source_single_slab_dense),
        sharding=effective_sharding,
    )
    return _CompileSetup(
        dt,
        sharding,
        source_specs,
        monitor_specs,
        config,
    )


def _compile_grid(
    request: SimulationRequest, boundary_data: BoundaryData
) -> CompiledGrid:
    """Lower cell materials and boundary data into one frozen logical Yee lattice."""
    material_grid = request.materials
    assert material_grid.grid is not None
    local_geometry = material_grid.grid.translated(request.domain.coordinate_offset)
    material_source = boundary_data
    profiles = boundary_data.profiles
    shapes = MappingProxyType(
        dict(component_shapes(material_grid.shape, request.domain.polarization_2d))
    )
    components = {
        component: jnp.zeros(shape, dtype=jnp.float32)
        for component, shape in shapes.items()
    }
    pml_data = (
        None
        if profiles is None
        else MappingProxyType(
            {
                key: jnp.asarray(value) if hasattr(value, "__array__") else value
                for key, value in profiles.items()
            }
        )
    )
    values = {
        "material_grid": material_grid,
        "geometry": local_geometry,
        "component_shapes": shapes,
        "resolution": material_grid.resolution,
        "plane_2d": "xy" if not request.domain.is_3d else request.domain.plane_2d,
        "polarization_2d": request.domain.polarization_2d,
        "permittivity": jnp.asarray(material_source.permittivity),
        "conductivity": jnp.asarray(material_source.conductivity),
        "permeability": jnp.asarray(material_source.permeability),
        "metallic_masks": MappingProxyType(dict(boundary_data.masks)),
        "boundaries": tuple(request.boundaries),
        "has_pml": profiles is not None,
        "has_cpml": bool(
            profiles is not None
            and str(profiles.get("formulation", "sponge")).lower() == "cpml"
        ),
        "pml_data": pml_data,
        **components,
    }
    assembly = SimpleNamespace(**values)
    materials = build_material_coefficients(assembly)
    direct = (
        dict(boundary_data.yee_materials)
        if material_grid.uses_direct_yee_materials
        else {}
    )
    if direct:
        values_by_name = dict(materials.items())
        for source, target in (
            ("eps_x", "eps_x"),
            ("eps_y", "eps_y"),
            ("eps_z", "eps_z"),
            ("sig_x", "sig_x"),
            ("sig_y", "sig_y"),
            ("sig_z", "sig_z"),
            ("mu_hx", "mu_hx"),
            ("mu_hy", "mu_hy"),
            ("mu_hz", "mu_hz"),
        ):
            if source in direct:
                value = jnp.asarray(direct[source])
                if (
                    source.startswith("sig_")
                    and assembly.has_pml
                    and not assembly.has_cpml
                ):
                    component = {"sig_x": "Ex", "sig_y": "Ey", "sig_z": "Ez"}[source]
                    centered = (
                        sample_voxel_grid_at_e_component_3d_centered(
                            assembly.conductivity,
                            component,
                        )
                        if assembly.permittivity.ndim == 3
                        else sample_voxel_grid_at_component_2d(
                            assembly.conductivity,
                            component,
                            "xy",
                            assembly.polarization_2d,
                        )
                    )
                    value = value + values_by_name[target] - centered
                values_by_name[target] = value
        values_by_name.update(
            eps_ex=values_by_name["eps_x"],
            eps_ey=values_by_name["eps_y"],
            eps_ez=values_by_name["eps_z"],
        )
        materials = type(materials)(**values_by_name)
    return CompiledGrid(
        **values,
        materials=materials,
        **dict(materials.items()),
    )


def _compile_boundary(fields, cpml, boundary_data, *, is_3d: bool) -> BoundaryPlan:
    """Assemble canonical CPML and metallic values on the logical lattice."""
    masks = fields.metallic_masks
    return BoundaryPlan(
        metallic_edges_2d=(frozenset() if is_3d else boundary_data.metallic_edges),
        cpml=cpml,
        metallic=MetallicPlan(
            masks["Ex"],
            masks["Ey"],
            masks["Ez"],
            masks["Hx"],
            masks["Hy"],
            masks["Hz"],
        ),
    )


def compile_simulation(request: SimulationRequest) -> CompiledProgram:
    """Build an immutable executable plan from a simulation request."""
    # 1. Lower all boundaries once, then collocate materials on the resulting lattice.
    boundary_data = lower_boundaries(
        request.materials,
        component_shapes(request.materials.shape, request.domain.polarization_2d),
        request.boundaries,
        request.domain.size,
        request.run.dt,
        polarization_2d=request.domain.polarization_2d,
    )
    logical_grid = _compile_grid(request, boundary_data)
    setup = _prepare_compilation(request, logical_grid)
    fields = logical_grid
    dt = setup.dt
    sharding = setup.sharding
    sharding_layout = sharding.layout
    source_specs, monitor_specs, config = (
        setup.source_specs,
        setup.monitor_specs,
        setup.config,
    )

    # 4. Precompute ordinary Yee update coefficients. Native 3D material kernels keep
    # material arrays instead because they form coefficients at their exact stagger.
    use_3d_material_coefficients = bool(request.domain.is_3d)

    empty3 = jnp.zeros((0, 0, 0), dtype=jnp.float32)
    if use_3d_material_coefficients:
        h_decay_x = h_source_x = h_decay_y = h_source_y = empty3
        h_decay_z = h_source_z = empty3
        h_sigma_m_x = _elide_zero_conductivity_grid(fields.sigma_m_hx)
        h_sigma_m_y = _elide_zero_conductivity_grid(fields.sigma_m_hy)
        h_sigma_m_z = _elide_zero_conductivity_grid(fields.sigma_m_hz)
    else:
        (
            (h_decay_x, h_source_x),
            (h_decay_y, h_source_y),
            (h_decay_z, h_source_z),
        ) = (
            ops.precompute_h_update_coefficients(fields.sigma_m_hx, dt),
            ops.precompute_h_update_coefficients(fields.sigma_m_hy, dt),
            ops.precompute_h_update_coefficients(fields.sigma_m_hz, dt),
        )
        h_sigma_m_x = h_sigma_m_y = h_sigma_m_z = empty3

    if use_3d_material_coefficients:
        e_decay_x = e_source_x = e_decay_y = e_source_y = empty3
        e_decay_z = e_source_z = empty3
        e_conductivity_x = _elide_zero_conductivity_grid(fields.sig_x)
        e_conductivity_y = _elide_zero_conductivity_grid(fields.sig_y)
        e_conductivity_z = _elide_zero_conductivity_grid(fields.sig_z)
        e_permittivity_x = fields.eps_x
        e_permittivity_y = fields.eps_y
        e_permittivity_z = fields.eps_z
    else:
        (
            (e_decay_x, e_source_x),
            (e_decay_y, e_source_y),
            (e_decay_z, e_source_z),
        ) = (
            ops.precompute_e_update_coefficients(
                shape=fields.Ex.shape,
                conductivity=fields.sig_x,
                permittivity=fields.eps_x,
                dt=dt,
                region=fields.region_x,
            ),
            ops.precompute_e_update_coefficients(
                shape=fields.Ey.shape,
                conductivity=fields.sig_y,
                permittivity=fields.eps_y,
                dt=dt,
                region=fields.region_y,
            ),
            ops.precompute_e_update_coefficients(
                shape=fields.Ez.shape,
                conductivity=fields.sig_z,
                permittivity=fields.eps_z,
                dt=dt,
                region=fields.region_z,
            ),
        )
        e_conductivity_x = e_conductivity_y = e_conductivity_z = empty3
        e_permittivity_x = e_permittivity_y = e_permittivity_z = empty3
    # 5. Precompute the same packed recurrence record for every 2D or 3D derivative.
    cpml = _compile_cpml_plan(
        fields,
        dt=dt,
        is_3d=bool(request.domain.is_3d),
        metallic_edges=boundary_data.metallic_edges,
        polarization_2d=request.domain.polarization_2d,
    )

    # Assemble coefficients explicitly so renaming a planning local cannot silently
    # alter the compiled program through reflection.
    update_coefficients = UpdateCoefficients(
        h_decay_x=h_decay_x,
        h_source_x=h_source_x,
        h_sigma_m_x=h_sigma_m_x,
        h_decay_y=h_decay_y,
        h_source_y=h_source_y,
        h_sigma_m_y=h_sigma_m_y,
        h_decay_z=h_decay_z,
        h_source_z=h_source_z,
        h_sigma_m_z=h_sigma_m_z,
        e_decay_x=e_decay_x,
        e_source_x=e_source_x,
        e_conductivity_x=e_conductivity_x,
        e_permittivity_x=e_permittivity_x,
        e_decay_y=e_decay_y,
        e_source_y=e_source_y,
        e_conductivity_y=e_conductivity_y,
        e_permittivity_y=e_permittivity_y,
        e_decay_z=e_decay_z,
        e_source_z=e_source_z,
        e_conductivity_z=e_conductivity_z,
        e_permittivity_z=e_permittivity_z,
    )
    boundary = _compile_boundary(
        fields,
        cpml,
        boundary_data,
        is_3d=bool(request.domain.is_3d),
    )
    update_coefficients, boundary = lower_compiled_arrays(
        update_coefficients, boundary, sharding_layout
    )
    metrics = _compile_derivative_metrics(request.materials)
    return CompiledProgram(
        grid=logical_grid,
        config=config,
        coefficients=update_coefficients,
        metrics=metrics,
        boundary=boundary,
        sources=source_specs,
        monitors=monitor_specs,
        sharding=sharding,
    )


def compile_program(
    simulation,
    *,
    num_steps: int | None = None,
    sharding=None,
    progress: bool = False,
    setup_context_factory=None,
    compile_factory=None,
) -> CompiledProgram:
    """Compile or reuse the immutable program for one execution horizon."""
    steps = int(simulation.num_steps if num_steps is None else num_steps)
    if steps <= 0:
        raise ValueError("num_steps must be > 0")

    # Environment switches are compilation inputs, so normalize them before building
    # the request and its semantic cache key.
    loop_env = os.getenv("BEAMZ_COMPILED_LOOP_KIND", "scan").strip().lower()
    if loop_env in {"fori", "fori_loop", "fori-loop"}:
        loop_kind = "fori_loop"
    elif loop_env == "scan":
        loop_kind = "scan"
    else:
        raise ValueError("Invalid BEAMZ_COMPILED_LOOP_KIND (use: scan, fori_loop).")
    dense = os.getenv("BEAMZ_SOURCE_SINGLE_SLAB_DENSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    request = simulation.to_request(
        num_steps=steps,
        loop_kind=loop_kind,
        source_single_slab_dense=dense,
        sharding=sharding_cache_token(sharding),
        compiler_sharding=sharding,
        progress=progress,
    )
    signature = CompiledProgramKey.from_request(request)
    if cached := _PROGRAM_CACHE.get(signature):
        _PROGRAM_CACHE.move_to_end(signature)
        return cached

    setup_context_factory = setup_context_factory or _resolved_setup_device_context
    compile_factory = compile_factory or compile_simulation
    with setup_context_factory(simulation.setup_device_resolved):
        program = compile_factory(request)
    _PROGRAM_CACHE[signature] = program
    if len(_PROGRAM_CACHE) > _MAX_COMPILED_PROGRAMS:
        _PROGRAM_CACHE.popitem(last=False)
    return program


_FIELD_GROUPS = {
    "yee_fields": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    "material_center_grids": (
        "permittivity",
        "conductivity",
        "permeability",
        "total_conductivity",
    ),
    "component_material_grids": (
        "eps_x",
        "eps_y",
        "eps_z",
        "sig_x",
        "sig_y",
        "sig_z",
        "sigma_m_hx",
        "sigma_m_hy",
        "sigma_m_hz",
    ),
    "field_masks": (
        "tm_ez_mask",
        "tm_hx_mask",
        "tm_hy_mask",
        "ex_metal_mask",
        "ey_metal_mask",
        "ez_metal_mask",
        "hx_metal_mask",
        "hy_metal_mask",
        "hz_metal_mask",
    ),
}

_REFERENCED_COEFFICIENTS = {
    "h_sigma_m_x",
    "h_sigma_m_y",
    "h_sigma_m_z",
    "e_conductivity_x",
    "e_conductivity_y",
    "e_conductivity_z",
    "e_permittivity_x",
    "e_permittivity_y",
    "e_permittivity_z",
}


def _add_memory_arrays(entries, name, value, category, residency="persistent"):
    """Flatten array tuples into stable allocation-report entries."""
    try:
        shape, dtype = tuple(int(v) for v in value.shape), np.dtype(value.dtype)
    except Exception:
        if isinstance(value, tuple):
            for index, item in enumerate(value):
                _add_memory_arrays(
                    entries, f"{name}[{index}]", item, category, residency
                )
        return
    entries.append(
        {
            "name": str(name),
            "category": str(category),
            "residency": str(residency),
            "shape": list(shape),
            "dtype": str(dtype),
            "bytes": int(np.prod(shape, dtype=np.int64) * dtype.itemsize),
        }
    )


def _memory_summary(entries):
    categories, residencies = {}, {}
    for entry in entries:
        size = int(entry["bytes"])
        category, residency = entry["category"], entry["residency"]
        categories[category] = categories.get(category, 0) + size
        residencies[residency] = residencies.get(residency, 0) + size
    total = sum(int(entry["bytes"]) for entry in entries)
    return {
        "total_bytes": total,
        "total_gib": total / 1024**3,
        "totals_by_category": categories,
        "totals_by_residency": residencies,
        "entries": entries,
    }


def _compiled_memory_entries(program):
    owned, referenced = [], []
    for name, value in program.coefficients._asdict().items():
        target = referenced if name in _REFERENCED_COEFFICIENTS else owned
        _add_memory_arrays(
            target,
            name,
            value,
            "compiled_referenced_inputs"
            if target is referenced
            else "compiled_update_coefficients",
            "reference" if target is referenced else "persistent",
        )
    for phase, terms in (
        ("h", program.boundary.cpml.h_terms),
        ("e", program.boundary.cpml.e_terms),
    ):
        for index, term in enumerate(terms):
            for coefficient in ("a", "b", "inv_kappa"):
                _add_memory_arrays(
                    owned,
                    f"cpml_{phase}[{index}].{coefficient}",
                    getattr(term, coefficient),
                    "compiled_static_terms",
                )
    for field in dataclass_fields(program.boundary.metallic):
        _add_memory_arrays(
            owned,
            field.name.replace("_mask", "_metal_mask"),
            getattr(program.boundary.metallic, field.name),
            "compiled_static_terms",
        )
    for index, source in enumerate(program.sources):
        for name in ("coeff", "waveform"):
            _add_memory_arrays(
                owned, f"sources[{index}].{name}", getattr(source, name), "sources"
            )
    for index, monitor in enumerate(program.monitors):
        for field in dataclass_fields(monitor):
            _add_memory_arrays(
                owned,
                f"monitors[{index}].{field.name}",
                getattr(monitor, field.name),
                "monitors",
            )
    return owned, referenced


def _compiled_memory_report(program):
    owned, referenced = _compiled_memory_entries(program)
    runtime = empty_monitor_values(program)
    for name in MONITOR_FIELDS:
        _add_memory_arrays(
            owned, f"monitor_state.{name}", runtime[name], "monitor_state", "runtime"
        )

    layout, config = program.sharding.layout, program.config
    report = _memory_summary(owned)
    report["referenced_inputs"] = _memory_summary(referenced)
    report["config"] = {
        "num_steps": int(config.num_steps),
        "is_3d": bool(config.is_3d),
        "loop_kind": config.loop_kind,
        "use_cpml_3d": bool(config.is_3d and program.boundary.cpml.enabled),
        "sharding": {
            "enabled": bool(layout.enabled),
            "axis": layout.axis_name,
            "num_devices": int(layout.num_devices),
            "backend": layout.backend,
            "logical_shapes": {k: list(v) for k, v in layout.logical_shapes.items()},
            "padded_shapes": {k: list(v) for k, v in layout.padded_shapes.items()},
        },
    }
    if layout.enabled:
        report["per_device_total_bytes"] = int(
            np.ceil(report["total_bytes"] / layout.num_devices)
        )
        report["per_device_total_gib"] = report["per_device_total_bytes"] / 1024**3
    return report


def simulation_memory_estimate(
    simulation,
    *,
    include_compiled: bool = True,
    num_steps: int | None = None,
    sharding=None,
) -> dict:
    """Return deterministic array storage derived from a compiled program."""
    program = simulation.compile(num_steps=num_steps, sharding=sharding)
    fields, entries = program.grid, []
    for category, names in _FIELD_GROUPS.items():
        for name in names:
            if hasattr(fields, name):
                _add_memory_arrays(
                    entries, f"fields.{name}", getattr(fields, name), category
                )
    pml_data = getattr(fields, "pml_data", None)
    if isinstance(pml_data, Mapping):
        for name, value in pml_data.items():
            _add_memory_arrays(entries, f"pml_data.{name}", value, "pml_data")

    report = _memory_summary(entries)
    report.update(
        grid_shape_zyx=[int(v) for v in fields.permittivity.shape],
        is_3d=bool(simulation.is_3d),
    )
    if include_compiled:
        compiled = _compiled_memory_report(program)
        report["compiled"] = compiled
        report["total_with_compiled_bytes"] = (
            report["total_bytes"] + compiled["total_bytes"]
        )
        report["total_with_compiled_gib"] = (
            report["total_with_compiled_bytes"] / 1024**3
        )
    return report
