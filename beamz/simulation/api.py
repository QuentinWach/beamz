"""Public immutable simulation specification and convenience API."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

import numpy as np

from beamz._cache_tokens import cache_token
from beamz._helpers import env_bool, positive_float

# ``Simulation`` is an immutable specification; explicit ``SimulationState`` values
# carry all mutable runtime data.
from beamz.const import µm
from beamz.design.core import Design
from beamz.design.discretization import MaterialGrid, build_material_grid
from beamz.design.grid import RectilinearGrid
from beamz.design.grid_spec import (
    GridSpec,
    _realize_uniform_grid,
    _validate_grid_budget,
)
from beamz.design.materials import Material, MaterialProtocol
from beamz.design.structures import Box
from beamz.devices.boundaries import normalize_boundaries
from beamz.devices.monitors.monitors import (
    FieldMonitor,
    FieldRecorder,
    FluxMonitor,
    ModeMonitor,
    _Monitor,
)
from beamz.devices.sources import (
    CANONICAL_SOURCE_TYPES,
    GaussianBeamSource,
)
from beamz.lattice import (
    grid_vector_to_physical_2d,
    in_plane_vector_2d,
    normalize_polarization_2d,
    physical_vector_to_grid_2d,
)
from beamz.simulation.compile import CompiledProgramKey as CompiledProgramKey
from beamz.simulation.compile import (
    _resolved_setup_device_context as _resolved_setup_device_context,
)
from beamz.simulation.compile import clear_program_cache, compile_program
from beamz.simulation.compile import compile_simulation as compile_simulation
from beamz.simulation.compile import simulation_memory_estimate as memory_estimate
from beamz.simulation.execute import (
    clear_execution_cache,
    execute_step,
    run_simulation_program,
)
from beamz.simulation.execute import (
    compiled_xla_memory_analysis as analyze_compiled_xla_memory,
)
from beamz.simulation.model import (
    CompiledProgram,
    DomainSpec,
    RunSpec,
    SimulationRequest,
    SimulationState,
)
from beamz.simulation.results import MonitorResults as MonitorResults
from beamz.simulation.results import SimulationResults as SimulationResults
from beamz.simulation.results import SimulationRun as SimulationRun

_MONITOR_TYPES = (FieldMonitor, FieldRecorder, FluxMonitor, ModeMonitor)
_MAX_MATERIAL_GRIDS = 4
_MATERIAL_GRID_CACHE = OrderedDict()


def _analysis_function(module: str, name: str):
    """Resolve analysis lazily so solver imports remain dependency-safe."""
    return getattr(import_module(f"beamz.analysis.{module}"), name)


def _design_depth(design) -> float:
    return float(design.depth)


def _design_domain(design):
    # Assemble extents in public x/y/z order rather than the z/y/x array-storage order.
    return (float(design.width), float(design.height), _design_depth(design))


def _design_is_3d(design) -> bool:
    return bool(design.is_3d)


def _normalize_domain(domain, size):
    # Collapse flexible public inputs here so internal code handles one validated representation.
    if domain is not None and size is not None and tuple(domain) != tuple(size):
        raise ValueError("Pass only one of domain=... or size=....")
    sim_size = size if domain is None else domain
    if sim_size is None:
        return None
    sim_size = tuple(float(v) for v in sim_size)
    if len(sim_size) == 2:
        sim_size = (sim_size[0], sim_size[1], 0.0)
    if len(sim_size) != 3:
        raise ValueError("Simulation size must be a 2D or 3D tuple.")
    return sim_size


def _structure_to_domain(structure, offset, domain_size):
    # Clip infinite boxes to the finite simulation domain before shifting centered geometry.
    if isinstance(structure, Box):
        if any(not np.isfinite(v) for v in structure.size):
            structure = Box(
                center=structure.center,
                size=tuple(
                    float(d) if not np.isfinite(s) else min(float(s), float(d))
                    for s, d in zip(structure.size, domain_size, strict=True)
                ),
                material=structure.material,
            )
        return structure.to_rectangle(offset=offset, material=structure.material)
    return structure.shift(*offset)


def _shift_device_to_domain(device, offset, *, plane_2d):
    # Simulation domains always use public xyz offsets. A 2D Gaussian source
    # intentionally stores only its in-plane position, so lower the offset at
    # this domain-normalization boundary rather than widening its standalone API.
    position = getattr(device, "position", None)
    if position is not None and len(position) == 2 and len(offset) == 3:
        offset = in_plane_vector_2d(offset, plane_2d)
    return device.shifted(offset)


def _material_index(material) -> float:
    # Use the largest real permittivity to obtain a conservative index for grid resolution.
    if not isinstance(material, MaterialProtocol):
        raise TypeError("Simulation materials must satisfy MaterialProtocol.")
    eps = material.max_permittivity
    if eps is None:
        raise ValueError(
            "GridSpec.auto requires materials with finite max_permittivity."
        )
    return float(np.sqrt(max(float(np.max(np.real(np.asarray(eps)))), 1.0)))


def _max_index_for_specs(background, structures) -> float:
    # Include background and every structure so automatic meshing resolves the worst case.
    materials = [s.material for s in structures or ()]
    return max(_material_index(m) for m in (background, *materials) if m is not None)


def _resolve_grid_resolution(grid_spec, background, structures) -> float:
    if grid_spec.resolution is not None:
        resolution = grid_spec._target_spacing()
    else:
        resolution = grid_spec._target_spacing(
            max_index=_max_index_for_specs(background, structures)
        )
    return positive_float(resolution, name="Simulation resolution")


def _devices_require_uniform_grid(sources, monitors) -> bool:
    """Return whether current device operators require isotropic spacing."""
    del monitors
    return any(isinstance(source, GaussianBeamSource) for source in sources or ())


def _normalize_plane_2d(plane) -> str:
    if not isinstance(plane, str):
        raise TypeError("plane_2d must be 'xy', 'yz', or 'xz'.")
    plane = plane.lower()
    if plane not in {"xy", "yz", "xz"}:
        raise ValueError("plane_2d must be 'xy', 'yz', or 'xz'.")
    return plane


def _time_from_run_time(time, run_time, grid_spec, resolution, dims):
    # Preserve an explicit time grid; otherwise derive a stable step and include the endpoint.
    if time is not None or run_time is None:
        return time
    spec = grid_spec
    if spec is None:
        representative = (
            resolution.minimum_spacing
            if hasattr(resolution, "minimum_spacing")
            else resolution
        )
        spec = GridSpec.uniform(representative)
    dt = spec.resolve_time_step(resolution, dims=dims)
    return np.arange(0.0, float(run_time) + 0.5 * dt, dt)


def _resolve_design_time_and_grid(
    design,
    *,
    grid_spec,
    resolution,
    time,
    run_time,
    require_uniform_grid=False,
):
    # Compute and cache the authoritative setup value so later stages do not repeat expensive work.
    use_realized_grid = bool(
        grid_spec is not None and grid_spec.is_automatic and not require_uniform_grid
    )
    if use_realized_grid:
        assert grid_spec is not None
        grid = grid_spec.realize(design)
        resolution = grid.minimum_spacing
    else:
        if grid_spec is not None:
            resolution = _resolve_grid_resolution(
                grid_spec, design.background, list(design.structures)
            )
        grid = (
            grid_spec.realize(design)
            if grid_spec is not None and not grid_spec.is_automatic
            else _realize_uniform_grid(design, resolution, spec=grid_spec)
        )
    dims = 3 if _design_is_3d(design) else 2
    if grid_spec is not None:
        grid = _validate_grid_budget(grid, grid_spec, dimensions=dims)
    time = _time_from_run_time(time, run_time, grid_spec, grid, dims)
    return resolution, grid, time, use_realized_grid


def _prepare_design(design, *, domain, size, background, grid_spec, resolution, time, run_time, require_uniform_grid, plane_2d):  # fmt: skip
    # 1. Resolve the optional domain/size alias. An already domain-based design can keep its coordinates and only needs grid/time normalization.
    sim_size = _normalize_domain(domain, size)
    if design is not None and not isinstance(design, Design):
        raise TypeError("Simulation design must be a canonical Design.")
    if design is not None and sim_size is None:
        resolution, grid, time, use_realized_grid = _resolve_design_time_and_grid(
            design,
            grid_spec=grid_spec,
            resolution=resolution,
            time=time,
            run_time=run_time,
            require_uniform_grid=require_uniform_grid,
        )
        return design, resolution, grid, time, (0.0, 0.0, 0.0), use_realized_grid
    if design is not None and not design._centered_coordinates:
        resolution, grid, time, use_realized_grid = _resolve_design_time_and_grid(
            design,
            grid_spec=grid_spec,
            resolution=resolution,
            time=time,
            run_time=run_time,
            require_uniform_grid=require_uniform_grid,
        )
        return design, resolution, grid, time, (0.0, 0.0, 0.0), use_realized_grid
    # 2. A centered-coordinate design or explicit size must be converted into Beamz's
    # positive-domain coordinate system; reject the request when neither form is present.
    if sim_size is None:
        raise ValueError("Simulation requires either design=... or domain=....")
    design_structures = list(design.structures) if design is not None else []
    source_structures = design_structures
    background = background or (
        design.background if design is not None else Material(1.0)
    )
    offset = (0.5 * sim_size[0], 0.5 * sim_size[1], 0.5 * sim_size[2])
    # 3. Rebuild the design at the requested domain size and translate every structure by the half-domain offset so its physical placement is preserved.
    new_design = Design(
        width=sim_size[0], height=sim_size[1], depth=sim_size[2], background=background
    )
    for structure in source_structures:
        new_design += _structure_to_domain(structure, offset, sim_size)
    # 4. Resolve adaptive spacing from the rebuilt material set, derive the final time
    # grid, and return the offset needed to shift sources and monitors consistently.
    resolution, grid, time, use_realized_grid = _resolve_design_time_and_grid(
        new_design,
        grid_spec=grid_spec,
        resolution=resolution,
        time=time,
        run_time=run_time,
        require_uniform_grid=require_uniform_grid,
    )
    public_offset = (
        offset if new_design.is_3d else grid_vector_to_physical_2d(offset, plane_2d)
    )
    return new_design, resolution, grid, time, public_offset, use_realized_grid


def _prepare_material_grid(
    material_grid,
    *,
    design,
    domain,
    size,
    background,
    grid_spec,
    raster_options,
    time,
    run_time,
    plane_2d,
):
    """Create domain metadata for an already rasterized solver grid."""

    if not isinstance(material_grid, MaterialGrid):
        raise TypeError("Simulation material_grid must be a MaterialGrid.")
    if design is not None:
        raise ValueError("Pass either design=... or material_grid=..., not both.")
    if background is not None:
        raise ValueError("background cannot be combined with material_grid.")
    if grid_spec is not None:
        raise ValueError(
            "grid_spec cannot be combined with material_grid; the grid has "
            "already been rasterized."
        )
    if raster_options is not None:
        raise ValueError(
            "raster_options cannot be combined with material_grid; smoothing has "
            "already been applied."
        )

    assert material_grid.grid is not None
    inferred_size = (
        material_grid.grid.extent
        if len(material_grid.shape) == 3
        else (*material_grid.grid.extent[:2], 0.0)
    )
    requested_size = _normalize_domain(domain, size)
    if requested_size is not None and not np.allclose(
        requested_size,
        inferred_size,
        rtol=32.0 * np.finfo(float).eps,
        atol=0.0,
    ):
        raise ValueError(
            "Simulation domain/size must match the material_grid grid extents."
        )

    metadata_design = Design(
        width=inferred_size[0],
        height=inferred_size[1],
        depth=inferred_size[2],
        background=Material(1.0),
    )
    dimensions = 3 if len(material_grid.shape) == 3 else 2
    resolved_time = _time_from_run_time(
        time,
        run_time,
        None,
        material_grid.grid,
        dimensions,
    )
    origin = material_grid.grid.origin
    grid_offset = (-float(origin[0]), -float(origin[1]), -float(origin[2]))
    offset = (
        grid_offset
        if dimensions == 3
        else grid_vector_to_physical_2d(grid_offset, plane_2d)
    )
    return (
        metadata_design,
        material_grid.resolution,
        material_grid.grid,
        resolved_time,
        offset,
        False,
    )


def _rasterize_scene_for_simulation(
    scene, raster_grid, raster_options, polarization: Literal["tm", "te"]
):
    """Convert an imported scene into the same solver grid used by Design."""

    from beamz.design.raster import Grid, RasterOptions, Scene

    if not isinstance(scene, Scene):
        raise TypeError("Simulation scene must be a raster Scene.")
    if not isinstance(raster_grid, Grid):
        raise TypeError("Simulation raster_grid must be a raster Grid.")
    options = (
        RasterOptions(smoothing="farjadpour_diagonal")
        if raster_options is None
        else raster_options
    )
    if not isinstance(options, RasterOptions):
        raise TypeError("raster_options must be RasterOptions or None.")
    dimensions = 2 if raster_grid.shape[2] == 1 else 3
    if dimensions == 3 and polarization != "tm":
        raise ValueError("polarization applies only to 2D simulations.")
    options = RasterOptions(
        quality=options.quality,
        smoothing=options.smoothing,
        components=(f"two_dimensional_{polarization}" if dimensions == 2 else "all"),
    )
    return MaterialGrid.from_raster_result(
        scene.rasterize(raster_grid, options=options),
        dimensions=dimensions,
        polarization=polarization,
    )


def _setup_device_policy_label(policy) -> str:
    # An explicit argument overrides the environment; both become one normalized label later.
    return os.getenv("BEAMZ_SETUP_DEVICE", "auto") if policy is None else str(policy)


def _setup_device_context(policy, *, design, resolution):
    # Device selection affects setup allocation only; numerical compilation chooses separately.
    del design, resolution
    resolved = _setup_device_policy_label(policy)
    normalized = resolved.strip().lower()
    if normalized not in {
        "auto",
        "cpu",
        "host",
        "default",
        "device",
        "gpu",
        "accelerator",
    }:
        raise ValueError(f"setup_device must be one of 'auto', 'cpu', or 'default', got {policy!r}.")  # fmt: skip
    use_cpu = normalized in {"cpu", "host"} or (
        env_bool("BEAMZ_SETUP_CPU", False) and normalized == "auto"
    )
    if not use_cpu:
        return nullcontext(None), "default"
    try:
        import jax

        devices = jax.devices("cpu")
        return (
            (jax.default_device(devices[0]), "cpu")
            if devices
            else (nullcontext(None), "default")
        )
    except Exception:
        return nullcontext(None), "default"


def _normalize_devices(*, sources, monitors):
    def dedupe(devices, *, expected_types, kind):
        # Deduplicate by identity, not equality, because equal devices may be intentional repeats.
        seen, out = set(), []
        for device in devices:
            if type(device) not in expected_types:
                expected = ", ".join(cls.__name__ for cls in expected_types)
                raise TypeError(
                    f"Simulation {kind} must use canonical immutable device types "
                    f"({expected}); got {type(device).__name__}."
                )
            if id(device) not in seen:
                seen.add(id(device))
                out.append(device)
        return out

    # Normalize both collections through the same identity/type policy before name checks.
    sources = dedupe(sources, expected_types=CANONICAL_SOURCE_TYPES, kind="sources")
    monitors = dedupe(monitors, expected_types=_MONITOR_TYPES, kind="monitors")
    duplicates = sorted(
        {
            monitor.name
            for monitor in monitors
            if sum(m.name == monitor.name for m in monitors) > 1
        }
    )
    if duplicates:
        raise ValueError(f"Duplicate monitor names: {', '.join(duplicates)}.")
    return tuple(sources), tuple(monitors)


def _normalize_time(time):
    # Collapse flexible public inputs here so internal code handles one validated representation.
    if time is None:
        raise ValueError("Simulation requires time=... or run_time=....")
    values = np.array(time, dtype=float, copy=True)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("FDTD requires a 1D time array with at least two entries")
    steps = np.diff(values)
    if not np.all(np.isfinite(values)) or np.any(steps <= 0.0):
        raise ValueError(
            "Simulation time values must be finite and strictly increasing"
        )
    if not np.allclose(steps, steps[0], rtol=1e-7, atol=0.0):
        raise ValueError("Simulation time values must be uniformly spaced")
    values.setflags(write=False)
    return values


@dataclass(frozen=True, init=False, eq=False)
class Simulation:
    """Define an immutable FDTD simulation specification.

    ``Simulation`` contains configuration only: geometry, devices, boundaries,
    discretization, and the time grid. It never stores evolving electromagnetic
    fields. Use :meth:`run` for the normal complete-simulation workflow and
    :meth:`advance` only when an application needs explicit continuation state.

    Parameters
    ----------
    design : Design, optional
        Physical geometry and material distribution. Supply exactly one of
        ``design``, ``material_grid``, or ``scene``.
    material_grid : MaterialGrid, optional
        A compatible uniform or rectilinear solver grid, such as one produced by
        ``MaterialGrid.from_raster_result()``. This is the simulation bridge for
        imported GDS, STL, Gmsh, and raw-mesh scenes.
    scene : beamz.design.raster.Scene, optional
        Reusable scene produced by an importer. Supply it with ``raster_grid``;
        BeamZ rasterizes and validates it during simulation setup.
    raster_grid : beamz.design.raster.Grid, optional
        Explicit uniform or nonuniform rectilinear solver grid for ``scene``.
    sources : sequence, optional
        Immutable source specifications to inject during execution.
    monitors : sequence of monitor specifications, optional
        Quantities to record. Results are keyed by monitor name.
    boundaries : sequence of PEC, PML, or Absorber, optional
        Domain boundary conditions. An all-edge PEC boundary is used when omitted.
    resolution : float, default=0.02 * um
        Uniform cell spacing in metres when ``grid_spec`` does not override it.
        Geometry-aware specifications expose their exact mesh through ``grid``;
        this attribute remains a representative scalar for compatibility.
    time : numpy.ndarray, optional
        One-dimensional, uniformly spaced time samples in seconds. Exactly one of
        ``time`` and ``run_time`` may be supplied.
    plane_2d : {"xy", "xz", "yz"}, default="xy"
        Physical plane represented by a two-dimensional design.
    polarization : {"tm", "te"}, default="tm"
        Electromagnetic polarization for a 2D simulation. ``"tm"`` advances the
        out-of-plane electric field; ``"te"`` advances the out-of-plane magnetic
        field. Three-dimensional simulations require the default.
    domain, size : sequence of float, optional
        Alternative domain extents in public ``(x, y, z)`` order. Passing both is
        invalid. Two values create a 2D domain.
    background : Material, optional
        Background material used when constructing a design from ``domain`` or
        ``size``.
    grid_spec : GridSpec, optional
        Automatic or explicit spatial-discretization policy.
    run_time : float, optional
        Total physical duration in seconds. BeamZ derives a stable time grid when
        this is supplied instead of ``time``.
    setup_device : {"auto", "cpu", "default"}, optional
        Device policy for setup-time rasterization and lowering. Runtime placement
        is controlled independently by ``sharding`` on execution methods.
    normalize_source : int or None, default=0
        Source index used to normalize frequency-domain monitor data. Use ``None``
        to retain raw acquisitions.
    raster_options : RasterOptions, optional
        Native raster quality and smoothing policy. Component selection remains
        automatic for the simulation dimensionality. Simulations default to
        ``farjadpour_diagonal``; select ``farjadpour_full`` to retain lossless
        off-diagonal electric coupling.

    Notes
    -----
    The three execution methods have deliberately different contracts:

    - :meth:`run` executes the complete time grid and returns only durable,
      immutable :class:`SimulationResults`.
    - :meth:`advance` executes a fresh or continued segment and returns a
      :class:`SimulationRun` containing both results and the next
      :class:`SimulationState`.
    - :meth:`step` advances state by exactly one timestep without constructing
      analysis results. It is intended for debugging and numerical verification.

    Compilation is lazy and cached. Calling :meth:`run` does not require a prior
    call to :meth:`compile`.

    Examples
    --------
    Run a complete simulation and inspect a named monitor:

    >>> import numpy as np
    >>> import beamz as bz
    >>> sim = bz.Simulation(
    ...     domain=(4 * bz.um, 3 * bz.um),
    ...     time=np.arange(100) * 1e-16,
    ...     monitors=[bz.FieldRecorder(("Ez",), name="fields")],
    ... )
    >>> results = sim.run()
    >>> frames = results["fields"].fields["Ez"]

    Execute two continuation segments while preserving the branch point:

    >>> first = sim.advance(num_steps=40)
    >>> second = sim.advance(state=first.state, num_steps=30)
    >>> alternative = sim.advance(state=first.state, num_steps=20)
    """

    # The class stores semantic configuration only; mutable compilation and execution resources are intentionally external.
    design: Design
    material_grid: MaterialGrid | None
    sources: tuple[object, ...]
    monitors: tuple[_Monitor, ...]
    boundaries: tuple[object, ...]
    resolution: float
    grid: RectilinearGrid
    time: np.ndarray
    plane_2d: str
    polarization: str
    grid_spec: GridSpec | None
    run_time: float | None
    setup_device_policy: str
    setup_device_resolved: str
    normalize_source: int | None
    raster_options: Any
    coordinate_offset: tuple[float, float, float]
    _uses_realized_grid: bool

    def __init__(
        self,
        design: Design | None = None,
        sources: Sequence[Any] | None = None,
        monitors: Sequence[_Monitor] | None = None,
        boundaries: Sequence[Any] | None = None,
        resolution: float = 0.02 * µm,
        time: np.ndarray | None = None,
        plane_2d: str = "xy",
        polarization: Literal["tm", "te"] = "tm",
        *,
        domain=None,
        size=None,
        background=None,
        material_grid: MaterialGrid | None = None,
        scene=None,
        raster_grid=None,
        grid_spec=None,
        run_time: float | None = None,
        setup_device: Literal["auto", "cpu", "default"] | None = None,
        normalize_source: int | None = 0,
        raster_options=None,
    ):
        polarization = normalize_polarization_2d(polarization)
        plane_2d = _normalize_plane_2d(plane_2d)
        if time is not None and run_time is not None:
            raise ValueError("Pass only one of time=... or run_time=....")
        resolution = positive_float(resolution, name="Simulation resolution")
        if run_time is not None:
            run_time = positive_float(run_time, name="run_time")
        if scene is not None or raster_grid is not None:
            if scene is None or raster_grid is None:
                raise ValueError("Pass scene and raster_grid together.")
            if design is not None or material_grid is not None:
                raise ValueError(
                    "Pass exactly one material source: design, material_grid, or scene."
                )
            if background is not None:
                raise ValueError("background cannot be combined with scene.")
            if grid_spec is not None:
                raise ValueError(
                    "grid_spec cannot be combined with scene; use raster_grid."
                )
            material_grid = _rasterize_scene_for_simulation(
                scene,
                raster_grid,
                raster_options,
                polarization,
            )
            raster_options = None
        require_uniform_grid = _devices_require_uniform_grid(sources, monitors)
        # 1. Normalize alternative domain, grid, and time forms into one concrete design;
        # this also returns any coordinate translation introduced by size-based domains.
        if material_grid is None:
            design, resolution, grid, time, offset, uses_realized_grid = (
                _prepare_design(
                    design=design,
                    domain=domain,
                    size=size,
                    background=background,
                    grid_spec=grid_spec,
                    resolution=resolution,
                    time=time,
                    run_time=run_time,
                    require_uniform_grid=require_uniform_grid,
                    plane_2d=plane_2d,
                )
            )
        else:
            design, resolution, grid, time, offset, uses_realized_grid = (
                _prepare_material_grid(
                    material_grid,
                    design=design,
                    domain=domain,
                    size=size,
                    background=background,
                    grid_spec=grid_spec,
                    raster_options=raster_options,
                    time=time,
                    run_time=run_time,
                    plane_2d=plane_2d,
                )
            )
        # 2. Shift sources and monitors into the normalized domain coordinate system so their public positions remain physically unchanged.
        if offset != (0.0, 0.0, 0.0):
            sources = tuple(
                _shift_device_to_domain(s, offset, plane_2d=plane_2d)
                for s in (sources or ())
            )
            monitors = tuple(
                _shift_device_to_domain(m, offset, plane_2d=plane_2d)
                for m in (monitors or ())
            )

        # 3. Canonicalize devices, time, plane, and boundaries before resolving the setup
        # device; subsequent cache identity depends on these normalized values.
        is_3d = _design_is_3d(design)
        if is_3d and polarization != "tm":
            raise ValueError("polarization applies only to 2D simulations.")
        if (
            material_grid is not None
            and not is_3d
            and material_grid.polarization != polarization
        ):
            raise ValueError(
                "MaterialGrid polarization does not match Simulation polarization."
            )
        sources, monitors = _normalize_devices(
            sources=sources or (), monitors=monitors or ()
        )
        if normalize_source is not None:
            normalize_source = int(normalize_source)
            if normalize_source < 0:
                raise ValueError(
                    "normalize_source must be a non-negative index or None."
                )
            if sources and normalize_source >= len(sources):
                raise ValueError(
                    f"normalize_source={normalize_source} exceeds {len(sources)} sources."
                )
        time = _normalize_time(time)
        boundaries = normalize_boundaries(boundaries)
        if raster_options is not None:
            from beamz.design.raster import RasterOptions

            if not isinstance(raster_options, RasterOptions):
                raise TypeError("raster_options must be RasterOptions or None.")
        setup_device_policy = _setup_device_policy_label(setup_device)
        _, setup_device_resolved = _setup_device_context(
            setup_device, design=design, resolution=resolution
        )
        # 4. Assign the fully normalized specification atomically through the frozen
        # dataclass escape hatch so no partially initialized simulation is observable.
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "material_grid", material_grid)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "monitors", monitors)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "plane_2d", plane_2d)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "grid_spec", grid_spec)
        object.__setattr__(self, "run_time", run_time)
        object.__setattr__(self, "setup_device_policy", setup_device_policy)
        object.__setattr__(self, "setup_device_resolved", setup_device_resolved)
        object.__setattr__(self, "normalize_source", normalize_source)
        object.__setattr__(self, "raster_options", raster_options)
        object.__setattr__(self, "coordinate_offset", tuple(float(v) for v in offset))
        object.__setattr__(self, "_uses_realized_grid", uses_realized_grid)

    @property
    def size(self):
        """Return the domain extents in public ``(x, y, z)`` order.

        Returns
        -------
        tuple of float
            Width, height, and depth in metres. A 2D simulation has zero depth.
        """
        return _design_domain(self.design)

    @property
    def domain(self):
        """Return the domain extents; an alias of :attr:`size`.

        Returns
        -------
        tuple of float
            Domain extents in public ``(x, y, z)`` order.
        """
        return self.size

    @property
    def is_3d(self):
        """Return whether the simulation has a positive z extent.

        Returns
        -------
        bool
            ``True`` for a three-dimensional lattice and ``False`` for a 2D plane.
        """
        return _design_is_3d(self.design)

    @property
    def dt(self):
        """Return the uniform simulation timestep.

        Returns
        -------
        float
            Difference between consecutive entries of :attr:`time`, in seconds.
        """
        return float(self.time[1] - self.time[0])

    @property
    def num_steps(self):
        """Return the number of timesteps executed by :meth:`run`.

        Returns
        -------
        int
            Length of the simulation time grid.
        """
        return int(self.time.size)

    def canonical_spec(self):
        """Return the immutable values defining simulation identity.

        Returns
        -------
        tuple
            Canonical design, device, boundary, grid, time, and normalization
            values used by equality, hashing, and compilation caches.

        Notes
        -----
        This is an advanced introspection method. Use :meth:`updated_copy` to
        construct a modified simulation instead of editing the returned values.
        """
        return (
            self.design,
            self.material_grid,
            self.sources,
            self.monitors,
            self.boundaries,
            self.resolution,
            self.grid,
            self.time,
            self.plane_2d,
            self.polarization,
            self.grid_spec,
            self.run_time,
            self.setup_device_policy,
            self.normalize_source,
            self.raster_options,
        )

    def __eq__(self, other):
        # Compare canonical cache tokens so array content, not object identity, defines equality.
        if not isinstance(other, Simulation):
            return NotImplemented
        return cache_token(self.canonical_spec()) == cache_token(other.canonical_spec())

    def __hash__(self):
        # Hash the same canonical specification used by equality to preserve the hash contract.
        return hash(cache_token(self.canonical_spec()))

    def updated_copy(self, **changes):
        """Return a validated simulation with selected fields replaced.

        Parameters
        ----------
        **changes : object
            Any supported constructor field, including ``design``, ``sources``,
            ``monitors``, ``boundaries``, ``resolution``, ``time``, ``plane_2d``,
            ``grid_spec``, ``run_time``, ``setup_device``, and
            ``normalize_source``, ``material_grid``, and ``raster_options``.

        Returns
        -------
        Simulation
            A new immutable specification. The original simulation is unchanged.

        Raises
        ------
        TypeError
            If a replacement key is not a supported simulation field.
        ValueError
            If the reconstructed configuration violates a simulation invariant.

        Examples
        --------
        >>> longer = sim.updated_copy(time=np.arange(200) * sim.dt)
        >>> raw = sim.updated_copy(normalize_source=None)

        Notes
        -----
        Replacement goes through the normal constructor validation and therefore
        produces independent compilation-cache identity when numerical inputs change.
        """
        # Reject unknown replacement keys before reconstructing through the canonical constructor.
        allowed = {
            "design",
            "material_grid",
            "sources",
            "monitors",
            "boundaries",
            "resolution",
            "time",
            "plane_2d",
            "polarization",
            "grid_spec",
            "run_time",
            "setup_device",
            "normalize_source",
            "raster_options",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise TypeError(
                f"Unknown Simulation field(s): {', '.join(sorted(unknown))}."
            )
        # 2. Preserve the existing coordinate translation unless the design itself changes; replacement devices must be shifted into that normalized domain.
        material_source_changed = "design" in changes or "material_grid" in changes
        offset = (0.0, 0.0, 0.0) if material_source_changed else self.coordinate_offset
        if offset != (0.0, 0.0, 0.0):
            for name in ("sources", "monitors"):
                if name in changes:
                    changes[name] = tuple(
                        _shift_device_to_domain(value, offset, plane_2d=self.plane_2d)
                        for value in changes[name] or ()
                    )
        # 3. Start from the current public specification and overlay validated changes so omitted values retain their normalized forms.
        values = {
            "design": None if self.material_grid is not None else self.design,
            "material_grid": self.material_grid,
            "sources": self.sources,
            "monitors": self.monitors,
            "boundaries": self.boundaries,
            "resolution": self.resolution,
            "time": self.time,
            "plane_2d": self.plane_2d,
            "polarization": self.polarization,
            "grid_spec": self.grid_spec,
            "run_time": self.run_time,
            "setup_device": self.setup_device_policy,
            "normalize_source": self.normalize_source,
            "raster_options": self.raster_options,
        }
        values.update(changes)
        if "design" in changes and "material_grid" not in changes:
            values["material_grid"] = None
        if "material_grid" in changes and "design" not in changes:
            values["design"] = None
            if changes["material_grid"] is not None and "raster_options" not in changes:
                values["raster_options"] = None
        if self.material_grid is not None:
            inverse_offset = tuple(-value for value in self.coordinate_offset)
            for name in ("sources", "monitors"):
                if material_source_changed and name in changes:
                    continue
                values[name] = tuple(
                    _shift_device_to_domain(
                        value, inverse_offset, plane_2d=self.plane_2d
                    )
                    for value in values[name] or ()
                )
        # 4. Keep the two mutually exclusive time specifications mutually
        # exclusive during reconstruction. A simulation originally created from
        # run_time must regenerate its time grid when other fields change (for
        # example, resolution can change dt). An explicit replacement time grid
        # takes precedence over the stored run_time convenience specification.
        if "time" in changes and "run_time" not in changes:
            values["run_time"] = None
        elif "time" not in changes and (
            "run_time" in changes or self.run_time is not None
        ):
            values["time"] = None
        result = type(self)(**values)
        if self.material_grid is None and not material_source_changed:
            preserved_offset = offset
            target_plane = str(values["plane_2d"])
            if not self.is_3d and target_plane != self.plane_2d:
                preserved_offset = grid_vector_to_physical_2d(
                    physical_vector_to_grid_2d(offset, self.plane_2d), target_plane
                )
            object.__setattr__(result, "coordinate_offset", preserved_offset)
        return result

    def _material_grid(self, *, progress: bool = False):
        """Return Design's immutable cell-centered material raster."""
        if self.material_grid is not None:
            return self.material_grid
        token = self._material_grid_token()
        cached = _MATERIAL_GRID_CACHE.get(token)
        if cached is None:
            raster_kwargs = {}
            if self.raster_options is not None:
                raster_kwargs = {
                    "quality": self.raster_options.quality,
                    "smoothing": self.raster_options.smoothing,
                }
            with _resolved_setup_device_context(self.setup_device_resolved):
                cached = build_material_grid(
                    self.design,
                    self.grid if self._uses_realized_grid else self.resolution,
                    progress=progress,
                    polarization=self.polarization,
                    **raster_kwargs,
                )
            _MATERIAL_GRID_CACHE[token] = cached
            if len(_MATERIAL_GRID_CACHE) > _MAX_MATERIAL_GRIDS:
                _MATERIAL_GRID_CACHE.popitem(last=False)
        else:
            _MATERIAL_GRID_CACHE.move_to_end(token)
        return cached

    def _material_grid_token(self):
        """Hash only values that change Design's cell-centered raster."""
        if self.material_grid is not None:
            return cache_token(self.material_grid)
        design = self.design
        return cache_token(
            (
                self.resolution,
                self.grid,
                design.width,
                design.height,
                design.depth,
                design.background,
                design.structures,
                self.raster_options,
                self.polarization,
            )
        )

    @property
    def pml_data(self):
        """Return compiled absorber profiles for the current lattice.

        Returns
        -------
        mapping or None
            Grid-aligned PML/absorber coefficient arrays, or ``None`` when no
            absorber profile is present.

        Notes
        -----
        Accessing this property may rasterize and compile the simulation. It is
        intended for diagnostics; normal execution does not require it.
        """
        return self.compile().grid.pml_data

    def initial_state(self) -> SimulationState:
        """Create a fresh runtime state at the beginning of the time grid.

        Returns
        -------
        SimulationState
            Zero-initialized fields, boundary memory, monitor accumulators, time,
            and timestep index compatible with this simulation.

        Examples
        --------
        >>> state = sim.initial_state()
        >>> int(state.current_step)
        0

        Notes
        -----
        Normal applications should call :meth:`run`. Create state explicitly only
        for :meth:`step`, branching, debugging, or custom continuation workflows.
        """
        # Allocate disabled features as empty fixed-rank arrays to keep runtime state structurally stable.
        return SimulationState.initial(self.compile().grid, t=float(self.time[0]))

    def step(self, state=None, *, donate_state=False) -> SimulationState:
        """Advance runtime state by exactly one timestep.

        Parameters
        ----------
        state : SimulationState, optional
            State to advance. A fresh state is created when omitted.
        donate_state : bool, default=False
            Allow JAX to recycle buffers owned by ``state``. After a donating call,
            the input state must not be read or reused.

        Returns
        -------
        SimulationState
            State after one timestep, or the unchanged input when it is already at
            the end of the time grid.

        Examples
        --------
        >>> state = sim.initial_state()
        >>> state = sim.step(state)
        >>> int(state.current_step)
        1

        Notes
        -----
        ``step`` returns state only and does not detach :class:`SimulationResults`.
        Use :meth:`run` for normal simulations and :meth:`advance` for multi-step
        continuation with monitor results.
        """
        # Advance explicit state without mutating the immutable Simulation specification.
        current_step = 0 if state is None else int(state.current_step)
        if current_step >= self.num_steps:
            assert state is not None
            return state
        program = self.compile(num_steps=1)
        if state is None:
            state = SimulationState.initial(program.grid, t=float(self.time[0]))
        return execute_step(
            program,
            state,
            monitor_steps=max(1, self.num_steps - current_step),
            donate_state=bool(donate_state),
        )

    def to_request(
        self,
        *,
        num_steps: int | None = None,
        loop_kind: str = "scan",
        source_single_slab_dense: bool = False,
        sharding=(False, "auto", None, None),
        compiler_sharding=None,
        progress: bool = False,
    ) -> SimulationRequest:
        """Build the immutable compiler request for this simulation.

        Parameters
        ----------
        num_steps : int, optional
            Number of timesteps represented by the request. The full time grid is
            used when omitted.
        loop_kind : {"scan", "fori_loop"}, default="scan"
            JAX control-flow lowering selected for the timestep loop.
        source_single_slab_dense : bool, default=False
            Use dense lowering for a single slab source. This is an advanced
            compilation experiment and does not change source semantics.
        sharding : tuple, optional
            Stable sharding token used in request and compilation cache identity.
        compiler_sharding : object, optional
            Runtime sharding configuration consumed by the compiler.
        progress : bool, default=False
            Display setup progress while rasterizing the material grid.

        Returns
        -------
        SimulationRequest
            Immutable, data-only input to the BeamZ compiler.

        Notes
        -----
        Most users should call :meth:`run` or :meth:`advance`; both construct and
        cache requests automatically. This method is exposed for diagnostics,
        compiler development, and reproducibility tooling.
        """
        # 1. Resolve Design's material raster; numerical boundary lowering belongs to compile.py.
        material_grid = self._material_grid(progress=progress)
        requested_steps = int(self.num_steps if num_steps is None else num_steps)
        # 2. Snapshot grid, run, and domain controls into immutable value objects, then
        # attach normalized sources, monitors, boundaries, and sharding configuration.
        return SimulationRequest(
            RunSpec(
                float(self.dt),
                requested_steps,
                int(self.num_steps),
                float(np.ravel(np.asarray(self.time, dtype=float))[0]),
                str(loop_kind),
                bool(source_single_slab_dense),
                sharding,
            ),
            DomainSpec(
                tuple(float(v) for v in self.size),  # type: ignore[arg-type]
                bool(self.is_3d),
                str(self.plane_2d),
                self.coordinate_offset,
                self.polarization,
            ),
            material_grid,
            tuple(self.sources),
            tuple(self.monitors),
            tuple(self.boundaries),
            compiler_sharding,
        )

    def compile(
        self,
        num_steps=None,
        sharding=None,
        progress: bool = False,
    ) -> CompiledProgram:
        """Prepare or retrieve a reusable compiled execution plan.

        Parameters
        ----------
        num_steps : int, optional
            Static segment length compiled into the JAX loop. The complete time-grid
            length is used when omitted.
        sharding : ShardingConfig or compatible value, optional
            Runtime device-sharding policy. ``None`` selects the default placement.
        progress : bool, default=False
            Display setup progress while rasterizing and lowering the simulation.

        Returns
        -------
        CompiledProgram
            Immutable numerical plan. The backend executable itself is JIT-compiled
            lazily on first execution and cached outside the simulation value.

        Examples
        --------
        >>> program = sim.compile(num_steps=50)
        >>> program.config.num_steps
        50

        Notes
        -----
        Calling :meth:`compile` is optional. :meth:`run`, :meth:`advance`, and
        :meth:`step` compile and reuse the appropriate plan automatically.
        """
        # Lower an immutable request and cache by every value that changes generated code or storage.
        return compile_program(
            self,
            num_steps=num_steps,
            sharding=sharding,
            progress=progress,
            setup_context_factory=_resolved_setup_device_context,
            compile_factory=compile_simulation,
        )

    def clear_compiled_cache(self) -> None:
        """Clear BeamZ's in-process simulation compilation caches.

        Returns
        -------
        None

        Notes
        -----
        This clears cached material grids, immutable compiled plans, and JIT function
        references for all simulations in the current process, not only this object.
        It does not modify the simulation or any existing results and does not remove
        JAX's optional persistent on-disk compilation cache.
        """
        _MATERIAL_GRID_CACHE.clear()
        clear_program_cache()
        clear_execution_cache()

    def memory_estimate(
        self,
        *,
        include_compiled: bool = True,
        num_steps: int | None = None,
        sharding=None,
    ) -> dict:
        """Estimate deterministic array storage for a simulation plan.

        Parameters
        ----------
        include_compiled : bool, default=True
            Include update coefficients, boundary terms, sources, monitors, and
            initialized monitor-state buffers in addition to the discretized grid.
        num_steps : int, optional
            Segment length used when sizing step-dependent monitor buffers.
        sharding : ShardingConfig or compatible value, optional
            Device-sharding policy used to report padded and per-device storage.

        Returns
        -------
        dict
            Allocation report containing total bytes, GiB totals, category and
            residency summaries, individual entries, grid shape, and—when requested—
            a nested ``compiled`` report.

        Examples
        --------
        >>> report = sim.memory_estimate(num_steps=100)
        >>> report["total_bytes"] > 0
        True

        Notes
        -----
        The report counts known array payloads. Backend temporary allocations and
        compiler-dependent workspace are available, when supported, through
        :meth:`compiled_xla_memory_analysis`.
        """
        return memory_estimate(
            self,
            include_compiled=include_compiled,
            num_steps=num_steps,
            sharding=sharding,
        )

    def compiled_xla_memory_analysis(
        self, *, num_steps: int | None = None, sharding=None
    ) -> dict:
        """Return backend memory analysis for the lowered JAX executable.

        Parameters
        ----------
        num_steps : int, optional
            Static segment length to lower. The full time grid is used when omitted.
        sharding : ShardingConfig or compatible value, optional
            Runtime placement used during lowering.

        Returns
        -------
        dict
            Scalar fields reported by the backend's compiled memory analysis. The
            mapping contains ``available=False`` when the backend provides no report.

        Notes
        -----
        Unlike :meth:`memory_estimate`, this method lowers and compiles an executable
        and may therefore be expensive. It does not advance simulation state.
        """
        # Lower an immutable request and cache by every value that changes generated code or storage.
        program = self.compile(num_steps=num_steps, sharding=sharding)
        state = SimulationState.initial(program.grid, t=float(self.time[0]))
        return analyze_compiled_xla_memory(
            program,
            state,
            monitor_steps=max(1, int(program.config.num_steps)),
        )

    def advance(
        self,
        *,
        state=None,
        num_steps=None,
        progress=False,
        store_full_materials=False,
        sharding=None,
        donate_state=False,
    ) -> SimulationRun:
        """Execute a fresh or continued segment and return results plus next state.

        Parameters
        ----------
        state : SimulationState, optional
            Continuation state. Omitting it starts from zero fields at the beginning
            of the simulation time grid.
        num_steps : int, optional
            Number of timesteps to execute. The remaining time grid is used when
            omitted.
        progress : bool, default=False
            Display JIT and execution progress in an interactive terminal.
        store_full_materials : bool, default=False
            Retain the full material grid in result metadata. By default BeamZ stores
            only material regions required by configured analysis monitors.
        sharding : ShardingConfig or compatible value, optional
            Runtime device-sharding policy.
        donate_state : bool, default=False
            Transfer ownership of the input state's device buffers to JAX. This can
            reduce peak memory, but the input state must never be used afterward.

        Returns
        -------
        SimulationRun
            A small owner containing durable ``results`` and the separate
            continuation ``state`` produced at the end of the segment.

        Raises
        ------
        ValueError
            If ``state.current_step`` is outside the time grid or ``num_steps`` is
            not positive and within the remaining number of timesteps.

        Examples
        --------
        Start a segmented run, then continue it:

        >>> first = sim.advance(num_steps=40)
        >>> second = sim.advance(state=first.state, num_steps=20)
        >>> int(second.state.current_step)
        60

        Branch safely from the same state; preservation is the default:

        >>> branch_a = sim.advance(state=first.state, num_steps=10)
        >>> branch_b = sim.advance(state=first.state, num_steps=10)

        Opt into buffer donation only for a state that will not be reused:

        >>> last = sim.advance(
        ...     state=second.state,
        ...     num_steps=10,
        ...     donate_state=True,
        ... )

        Notes
        -----
        Use :meth:`run` when final analysis data is all that is needed. ``run``
        executes the complete grid, returns :class:`SimulationResults` directly,
        and safely donates its private initial state. ``advance`` exists specifically
        for checkpointing, branching, streaming, and other continuation workflows.
        ``SimulationResults`` never contains continuation state.
        """
        current_step = 0 if state is None else int(state.current_step)
        if not 0 <= current_step <= self.num_steps:
            raise ValueError("state.current_step is outside the simulation time grid.")
        remaining = self.num_steps - current_step
        steps = remaining if num_steps is None else int(num_steps)
        if not 0 < steps <= remaining:
            raise ValueError(f"num_steps must be in [1, {remaining}], got {steps}.")
        program = self.compile(num_steps=steps, sharding=sharding, progress=progress)
        if state is None:
            state = SimulationState.initial(program.grid, t=float(self.time[0]))
        return run_simulation_program(
            self,
            program,
            state,
            progress=bool(progress),
            store_full_materials=bool(store_full_materials),
            monitor_steps=remaining,
            donate_state=bool(donate_state),
        )

    def run(
        self,
        *,
        progress=False,
        store_full_materials=False,
        sharding=None,
    ) -> SimulationResults:
        """Execute the complete simulation and return immutable analysis results.

        Parameters
        ----------
        progress : bool, default=False
            Display JIT and execution progress in an interactive terminal.
        store_full_materials : bool, default=False
            Retain the full material grid in result metadata. The default stores only
            material regions needed by configured analysis monitors.
        sharding : ShardingConfig or compatible value, optional
            Runtime device-sharding policy.

        Returns
        -------
        SimulationResults
            Detached monitor acquisitions, immutable simulation metadata, and source
            normalization information. No continuation state is attached.

        Examples
        --------
        >>> results = sim.run(progress=True)
        >>> field_data = results["fields"]

        Notes
        -----
        This is the normal user-facing execution method. Compilation is automatic.
        Since the initial runtime state belongs only to this call and is not returned,
        BeamZ donates its buffers internally to reduce peak device memory.

        Use :meth:`advance` instead when execution must be split into segments or the
        final :class:`SimulationState` is required. Use :meth:`step` only for
        state-only single-timestep debugging.
        """
        # The initial state is private to this call, so donating it is always safe and
        # avoids retaining a second full set of device buffers during execution.
        return self.advance(
            progress=bool(progress),
            store_full_materials=bool(store_full_materials),
            sharding=sharding,
            donate_state=True,
        ).results

    def plot(self, **kwargs):
        """Create a Matplotlib view of the simulation layout.

        Parameters
        ----------
        **kwargs : object
            Plot options forwarded to the analysis plotting backend, including
            ``ax``, ``figsize``, ``z``, ``y``, axis limits, and marker controls.
            ``show`` defaults to ``False`` for this method.

        Returns
        -------
        tuple
            Matplotlib ``(figure, axes)`` objects. A 3D cross-section layout may
            return an array of axes.

        Examples
        --------
        >>> fig, ax = sim.plot()
        """
        kwargs.setdefault("show", False)
        return _analysis_function("plotting", "plot_simulation")(self, **kwargs)

    def show(self, *, mode="auto", open_browser=True, **kwargs):
        """Display the simulation layout using Matplotlib.

        Parameters
        ----------
        mode : str, default="auto"
            Compatibility parameter retained for older viewer calls. The static
            Matplotlib backend ignores it.
        open_browser : bool, default=True
            Compatibility parameter retained for older viewer calls. No browser is
            opened by the current backend.
        **kwargs : object
            Plot options accepted by :meth:`plot`. ``show`` defaults to ``True``.

        Returns
        -------
        tuple
            Matplotlib ``(figure, axes)`` objects.
        """
        # Route display through the plotting API while changing only the presentation flag.
        del mode, open_browser
        kwargs.setdefault("show", True)
        return self.plot(**kwargs)

    def view3d(self, **kwargs):
        """Create a static, 3D-oriented cross-section view.

        Parameters
        ----------
        **kwargs : object
            Cross-section and Matplotlib options forwarded to the plotting backend.
            ``show`` defaults to ``False``.

        Returns
        -------
        tuple
            Matplotlib figure and axes for the selected cross sections.

        Notes
        -----
        This method creates static notebook-friendly cross sections; it does not
        launch an interactive browser viewer.
        """
        return _analysis_function("plotting", "view_simulation_3d")(self, **kwargs)

    def show3d(self, **kwargs):
        """Return the same static 3D-oriented view as :meth:`view3d`.

        Parameters
        ----------
        **kwargs : object
            Cross-section and Matplotlib options forwarded to :meth:`view3d`.

        Returns
        -------
        tuple
            Matplotlib figure and axes for the selected cross sections.

        Notes
        -----
        ``show3d`` is currently an alias and does not force interactive display.
        """
        # Route display through the plotting API while changing only the presentation flag.
        return self.view3d(**kwargs)
