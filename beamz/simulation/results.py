"""Detached raw outputs produced by simulation execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib import import_module
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from beamz._cache_tokens import cache_token
from beamz.design.grid import RectilinearGrid
from beamz.devices._immutable import immutable_snapshot, readonly_array
from beamz.devices._placement import SnappedRegion
from beamz.devices.monitors.compiler import CompiledMonitorSpec
from beamz.devices.monitors.monitors import _Monitor
from beamz.lattice import canonical_component_2d

from .model import CompiledGrid, SimulationState
from .observe import (
    monitor_dft_component,
    monitor_dft_flux,
    monitor_flux,
    monitor_frequencies,
    renormalize_result,
)


class _MonitorBuffers(Protocol):
    """Packed monitor arrays consumed when detaching a run."""

    powers: Any
    timestamps: Any
    counts: Any
    freq_flux_re: Any
    freq_flux_im: Any
    dft_vec_re: Any
    dft_vec_im: Any
    dft_weight_sum: Any
    recorded_fields: Any
    recorded_steps: Any
    recorded_times: Any
    recorded_counts: Any


class _RunConfigLike(Protocol):
    """Execution values needed to decode packed monitor arrays."""

    dt: float
    resolution: float
    is_3d: bool
    plane_2d: str
    polarization_2d: str


def _array_snapshot(value: Any, dtype=None) -> np.ndarray:
    # Copy only analysis-required data so results remain reproducible without live
    # simulation ownership.
    return readonly_array(value, dtype=dtype)


def _analysis_function(module: str, name: str):
    """Resolve analysis lazily so result values do not import it eagerly."""
    return getattr(import_module(f"beamz.analysis.{module}"), name)


@dataclass(frozen=True, slots=True)
class MaterialRegion:
    """Detached material neighborhood retained for deferred analysis."""

    permittivity: np.ndarray
    permeability: np.ndarray
    origin: tuple[int, ...]
    full_shape: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "permittivity", readonly_array(self.permittivity))
        object.__setattr__(self, "permeability", readonly_array(self.permeability))
        object.__setattr__(self, "origin", tuple(int(value) for value in self.origin))
        object.__setattr__(
            self, "full_shape", tuple(int(value) for value in self.full_shape)
        )


def material_region_for_monitor(simulation, monitor: _Monitor, *, runtime_fields):
    """Return a compact material neighborhood around a DFT monitor plane."""
    if monitor.freqs.size == 0:
        return None
    axis = monitor.plane_normal
    permittivity = np.asarray(runtime_fields.permittivity)
    permeability = np.asarray(runtime_fields.permeability)
    full_shape = tuple(int(value) for value in permittivity.shape)

    grid = runtime_fields.geometry
    if permittivity.ndim == 3:
        from beamz.devices._placement import snap_plane_region_grid

        snapped_region = snap_plane_region_grid(
            center=monitor.center,
            size=monitor.size,
            plane_normal=monitor.plane_normal,
            grid=grid,
        )
        if snapped_region is None:
            return None

        def axis_index(axis):
            if axis == snapped_region.normal_axis:
                return snapped_region.plane_index
            interval = snapped_region.axis_interval(axis)
            if interval is None:
                raise RuntimeError(f"Monitor region omits transverse axis {axis!r}.")
            return interval.as_slice()

        indices = tuple(axis_index(axis) for axis in ("z", "y", "x"))
        dim = {"z": 0, "y": 1, "x": 2}[axis]
        normal = indices[dim]
        if isinstance(normal, slice):
            start = 0 if normal.start is None else int(normal.start)
            stop = full_shape[dim] if normal.stop is None else int(normal.stop)
            plane = (start + max(start + 1, stop) - 1) // 2
        else:
            plane = int(normal)
    elif permittivity.ndim == 2:
        from beamz.devices._placement import (
            line_region_points,
            snap_axis_aligned_line_region_grid,
        )

        snapped_region = snap_axis_aligned_line_region_grid(
            monitor.start, monitor.end, grid
        )
        if snapped_region is None:
            return None
        points = line_region_points(snapped_region)
        if not points:
            return None
        if axis not in {"x", "y"}:
            start, end = np.asarray(monitor.start), np.asarray(monitor.end)
            axis = (
                "x"
                if abs(float(end[0] - start[0])) <= abs(float(end[1] - start[1]))
                else "y"
            )
        dim = 1 if axis == "x" else 0
        point_axis = 0 if axis == "x" else 1
        plane = int(round(float(np.mean([point[point_axis] for point in points]))))
    else:
        return None

    plane = int(np.clip(plane, 0, full_shape[dim] - 1))
    start, stop = max(0, plane - 2), min(full_shape[dim], plane + 3)
    crop = [slice(None)] * permittivity.ndim
    crop[dim] = slice(start, stop)
    origin = [0] * permittivity.ndim
    origin[dim] = start
    permeability_region = (
        permeability if permeability.ndim == 0 else permeability[tuple(crop)]
    )
    return MaterialRegion(
        permittivity[tuple(crop)],
        permeability_region,
        tuple(origin),
        full_shape,
    )


@dataclass(frozen=True, slots=True)
class FieldMetadata:
    """Describe stored field-array shapes and optional material data.

    Attributes
    ----------
    grid_shape : tuple of int
        Shape of the cell-centered material grid in canonical storage order.
    component_shapes : mapping of str to tuple of int
        Public ``Ex`` through ``Hz`` component shapes.
    materials : MaterialRegion or None
        Optional immutable material snapshot retained for analysis.
    """

    grid_shape: tuple[int, ...]
    component_shapes: Mapping[str, tuple[int, ...]]
    materials: MaterialRegion | None = None

    def __post_init__(self):
        if self.materials is not None and not isinstance(
            self.materials, MaterialRegion
        ):
            raise TypeError("FieldMetadata.materials must be a MaterialRegion or None.")
        if not isinstance(self.component_shapes, Mapping):
            raise TypeError("FieldMetadata.component_shapes must be a mapping.")
        object.__setattr__(self, "grid_shape", tuple(int(v) for v in self.grid_shape))
        object.__setattr__(
            self,
            "component_shapes",
            MappingProxyType(
                {
                    str(name): tuple(int(v) for v in shape)
                    for name, shape in self.component_shapes.items()
                }
            ),
        )

    @property
    def permittivity(self):
        """Return stored relative permittivity, if available.

        Returns
        -------
        numpy.ndarray or None
            Read-only material-region permittivity, or ``None`` when material data
            was not retained.
        """
        return None if self.materials is None else self.materials.permittivity

    @property
    def permeability(self):
        """Return stored relative permeability, if available.

        Returns
        -------
        numpy.ndarray or None
            Read-only material-region permeability, or ``None`` when material data
            was not retained.
        """
        return None if self.materials is None else self.materials.permeability


@dataclass(frozen=True, slots=True, eq=False)
class SimulationMetadata:
    """Store compact immutable context required to analyze results.

    Attributes
    ----------
    dt, resolution : float
        Simulation timestep in seconds and spatial cell size in metres.
    is_3d : bool
        Whether the source simulation used a three-dimensional lattice.
    plane_2d : {"xy", "xz", "yz"}
        Physical plane represented by two-dimensional field arrays.
    polarization_2d : {"tm", "te"}
        Polarization used to interpret two-dimensional field arrays.
    coordinate_offset : tuple of float
        Translation added to public coordinates to obtain solver-local coordinates.
    time : numpy.ndarray
        Complete immutable simulation time grid in seconds.
    width, height, depth : float
        Domain extents in metres.
    fields : FieldMetadata
        Field shapes and optional material information.
    grid : RectilinearGrid, optional
        Exact physical grid used by the simulation. Older manually constructed
        metadata may omit it and fall back to ``resolution``.

    Notes
    -----
    Metadata intentionally contains values needed for interpretation rather than a
    reference to the originating ``Simulation``. Results therefore remain usable
    after the simulation object is deleted.
    """

    dt: float
    resolution: float
    is_3d: bool
    plane_2d: str
    coordinate_offset: tuple[float, float, float]
    time: np.ndarray
    width: float
    height: float
    depth: float
    fields: FieldMetadata
    polarization_2d: str = "tm"
    grid: RectilinearGrid | None = None

    def __post_init__(self):
        if not isinstance(self.fields, FieldMetadata):
            raise TypeError("SimulationMetadata.fields must be FieldMetadata.")
        if self.grid is not None and not isinstance(self.grid, RectilinearGrid):
            raise TypeError("SimulationMetadata.grid must be RectilinearGrid or None.")
        offset = tuple(float(value) for value in self.coordinate_offset)
        if len(offset) != 3:
            raise ValueError(
                "SimulationMetadata.coordinate_offset must have three values."
            )
        time = _array_snapshot(self.time, dtype=float)
        if time.ndim != 1 or not np.all(np.isfinite(time)):
            raise ValueError("SimulationMetadata.time must be a finite 1D array.")
        plane_2d = str(self.plane_2d).lower()
        if plane_2d not in {"xy", "xz", "yz"}:
            raise ValueError("SimulationMetadata.plane_2d must be 'xy', 'xz', or 'yz'.")
        polarization_2d = str(self.polarization_2d).lower()
        if polarization_2d not in {"tm", "te"}:
            raise ValueError("SimulationMetadata.polarization_2d must be 'tm' or 'te'.")
        for name in ("dt", "resolution", "width", "height", "depth"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"SimulationMetadata.{name} must be finite.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "is_3d", bool(self.is_3d))
        object.__setattr__(self, "plane_2d", plane_2d)
        object.__setattr__(self, "polarization_2d", polarization_2d)
        object.__setattr__(self, "coordinate_offset", offset)
        object.__setattr__(self, "time", time)

    def canonical_spec(self):
        """Return immutable values defining metadata equality and hashing.

        Returns
        -------
        tuple
            Canonical scalar, coordinate, time-grid, and field metadata values.
        """
        return (
            self.dt,
            self.resolution,
            self.is_3d,
            self.plane_2d,
            self.polarization_2d,
            self.coordinate_offset,
            self.time,
            self.width,
            self.height,
            self.depth,
            self.fields,
            self.grid,
        )

    def __eq__(self, other):
        if not isinstance(other, SimulationMetadata):
            return NotImplemented
        return cache_token(self.canonical_spec()) == cache_token(other.canonical_spec())

    def __hash__(self):
        return hash(cache_token(self.canonical_spec()))

    @classmethod
    def from_simulation(
        cls,
        simulation,
        *,
        runtime_fields: CompiledGrid,
        store_full_materials: bool = False,
    ) -> "SimulationMetadata":
        """Detach analysis metadata from a simulation and compiled lattice.

        Parameters
        ----------
        simulation : Simulation
            Immutable source specification.
        runtime_fields : CompiledGrid
            Compiled lattice providing material and Yee-component shapes.
        store_full_materials : bool, default=False
            Retain full-domain permittivity and permeability arrays.

        Returns
        -------
        SimulationMetadata
            Independent, immutable analysis context.

        Raises
        ------
        TypeError
            If either input is not a canonical BeamZ simulation or compiled grid.

        Notes
        -----
        Execution uses this constructor internally. Normal applications read the
        ``metadata`` attribute of ``SimulationResults``.
        """
        from .api import Simulation

        if not isinstance(simulation, Simulation):
            raise TypeError("Result metadata requires a canonical Simulation.")
        if not isinstance(runtime_fields, CompiledGrid):
            raise TypeError("Result metadata requires a canonical CompiledGrid.")
        fields = runtime_fields
        design = simulation.design
        offset = simulation.coordinate_offset
        grid_shape = tuple(int(v) for v in np.asarray(fields.permittivity).shape)
        components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        component_shapes = {}
        for name in components:
            canonical = (
                name
                if simulation.is_3d
                else canonical_component_2d(
                    name, simulation.plane_2d, simulation.polarization
                )
            )
            component = None if canonical is None else getattr(fields, canonical, None)
            component_shapes[name] = (
                (1, 1) if component is None else tuple(int(v) for v in component.shape)
            )
        materials = (
            MaterialRegion(
                np.asarray(fields.permittivity),
                np.asarray(fields.permeability),
                (0,) * len(grid_shape),
                grid_shape,
            )
            if store_full_materials
            else None
        )
        return cls(
            dt=float(simulation.dt),
            resolution=float(simulation.resolution),
            is_3d=bool(simulation.is_3d),
            plane_2d=str(simulation.plane_2d),
            coordinate_offset=(offset[0], offset[1], offset[2]),
            time=_array_snapshot(simulation.time, dtype=float),
            width=float(design.width),
            height=float(design.height),
            depth=float(design.depth),
            fields=FieldMetadata(
                grid_shape=grid_shape,
                component_shapes=component_shapes,
                materials=materials,
            ),
            polarization_2d=simulation.polarization,
            # Field arrays are indexed on the compiler's normalized geometry. Imported
            # material grids may retain a nonzero public origin on ``simulation.grid``.
            grid=runtime_fields.geometry,
        )


@dataclass(frozen=True)
class MonitorResults:
    """Store the immutable acquisitions produced by one monitor.

    Instances are created during execution and owned by :class:`SimulationResults`.
    Arrays and mappings are detached from JAX runtime buffers and made read-only, so
    retaining or analyzing monitor data never affects continuation state.

    Attributes
    ----------
    monitor : monitor specification
        Immutable monitor geometry and acquisition settings used for this result.
    fields : mapping of str to numpy.ndarray
        Recorded time-domain field frames keyed by component name.
    power_history, power_timestamps : numpy.ndarray
        Time-domain power samples and their physical times.
    power_spectrum : numpy.ndarray
        Complex frequency-domain flux samples after the selected source
        normalization is applied.
    dft_fields : mapping of str to numpy.ndarray
        Complex frequency-domain field samples keyed by component name.
    dft_frequencies : numpy.ndarray
        Frequencies in hertz corresponding to the first DFT axis.
    dft_weight_sum : numpy.ndarray
        Temporal-window normalization accumulated at each DFT frequency.
    dft_base_dt : float
        Base timestep in seconds used for DFT accumulation.
    resolution : float
        Spatial cell size in metres used for flux integration.
    normal_axis, normal_sign, power_scale
        Orientation and scaling metadata for flux monitors.
    field_times, field_steps : numpy.ndarray
        Physical times and completed-step indices for recorded field frames.
    objective_value : float or None
        Optional scalar objective associated with this acquisition.
    material_region : MaterialRegion or None
        Material snapshot colocated with the monitor for downstream modal analysis.
    sample_region : SnappedRegion or None
        Exact grid-aware spatial region retained from monitor compilation.

    Examples
    --------
    >>> results = sim.run()
    >>> monitor = results["fields"]
    >>> frames = monitor.fields["Ez"]
    >>> steps = monitor.field_steps

    Notes
    -----
    Different monitor types populate different fields. For example, a
    ``FieldRecorder`` fills ``fields`` and frame coordinates, while a frequency
    ``FieldMonitor`` fills ``dft_fields`` and ``dft_frequencies``. Empty arrays or
    mappings indicate that an acquisition was not configured.
    """

    monitor: _Monitor
    fields: Mapping[str, np.ndarray]
    power_history: np.ndarray
    power_timestamps: np.ndarray
    power_spectrum: np.ndarray
    dft_fields: Mapping[str, np.ndarray] = field(default_factory=dict)
    dft_frequencies: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    dft_weight_sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    dft_base_dt: float = 0.0
    resolution: float = 0.0
    normal_axis: int = -1
    normal_sign: float = 1.0
    power_scale: float = 0.0
    integration_weights: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    _raw_dft_fields: Mapping[str, np.ndarray] | None = field(
        default=None, compare=False, repr=False
    )
    _raw_power_spectrum: np.ndarray | None = field(
        default=None, compare=False, repr=False
    )
    field_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    field_steps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    objective_value: float | None = None
    material_region: MaterialRegion | None = field(
        default=None, compare=False, repr=False
    )
    sample_region: SnappedRegion | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.monitor, _Monitor):
            raise TypeError("MonitorResults.monitor must be a canonical monitor.")
        if self.material_region is not None and not isinstance(
            self.material_region, MaterialRegion
        ):
            raise TypeError(
                "MonitorResults.material_region must be a MaterialRegion or None."
            )
        if self.sample_region is not None and not isinstance(
            self.sample_region, SnappedRegion
        ):
            raise TypeError(
                "MonitorResults.sample_region must be a SnappedRegion or None."
            )
        if not isinstance(self.fields, Mapping) or not isinstance(
            self.dft_fields, Mapping
        ):
            raise TypeError("MonitorResults field collections must be mappings.")
        object.__setattr__(self, "fields", immutable_snapshot(self.fields))
        object.__setattr__(self, "dft_fields", immutable_snapshot(self.dft_fields))
        for name in (
            "power_history",
            "power_timestamps",
            "power_spectrum",
            "field_times",
            "field_steps",
            "dft_frequencies",
            "dft_weight_sum",
            "integration_weights",
        ):
            object.__setattr__(self, name, _array_snapshot(getattr(self, name)))
        if self.field_times.shape != self.field_steps.shape:
            raise ValueError(
                "Recorded field times and steps must have matching shapes."
            )
        if self.objective_value is not None:
            object.__setattr__(self, "objective_value", float(self.objective_value))
        for name in ("dft_base_dt", "resolution", "normal_sign", "power_scale"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "normal_axis", int(self.normal_axis))
        raw_fields = (
            self.dft_fields if self._raw_dft_fields is None else self._raw_dft_fields
        )
        raw_power = (
            self.power_spectrum
            if self._raw_power_spectrum is None
            else self._raw_power_spectrum
        )
        object.__setattr__(self, "_raw_dft_fields", immutable_snapshot(raw_fields))
        object.__setattr__(self, "_raw_power_spectrum", _array_snapshot(raw_power))

    def get_dft_frequencies(self):
        """Return frequencies associated with frequency-domain acquisitions.

        Returns
        -------
        numpy.ndarray
            One-dimensional frequency coordinates in hertz.

        Notes
        -----
        This compatibility accessor applies the same frequency interpretation used
        by BeamZ analysis adapters. The returned data is derived from this immutable
        result and does not access simulation state.
        """
        return monitor_frequencies(self)

    def get_dft_component(self, component: str):
        """Return one complex frequency-domain field component.

        Parameters
        ----------
        component : str
            Public field component name, such as ``"Ez"`` or ``"Hy"``.

        Returns
        -------
        numpy.ndarray
            Complex DFT samples with frequency as the leading axis.

        Raises
        ------
        ValueError
            If the requested component was not acquired by this monitor.

        Examples
        --------
        >>> ez = results["frequency_fields"].get_dft_component("Ez")
        """
        return monitor_dft_component(self, component)

    def get_dft_flux(self):
        """Return the monitor's complex frequency-domain flux.

        Returns
        -------
        numpy.ndarray
            Complex flux indexed by :meth:`get_dft_frequencies`.

        Notes
        -----
        Orientation, integration scale, and source normalization have already been
        applied to this result view.
        """
        return monitor_dft_flux(self)

    @property
    def flux(self):
        """Return the most appropriate flux acquisition for this monitor.

        Returns
        -------
        numpy.ndarray
            Frequency-domain flux when configured, otherwise the recorded
            time-domain power history.

        Examples
        --------
        >>> transmitted = results["output"].flux
        """
        return monitor_flux(self)

    def snapshot(self, component: str, index: int = -1) -> Mapping[str, Any]:
        """Return one immutable view of a recorded field frame.

        Parameters
        ----------
        component : str
            Recorded field component to select.
        index : int, default=-1
            Frame index. Standard Python negative indexing is supported.

        Returns
        -------
        mapping
            Read-only mapping with ``field``, ``field_name``, physical time ``t``,
            and completed ``step``.

        Raises
        ------
        ValueError
            If the monitor did not record ``component``.
        IndexError
            If ``index`` is outside the recorded frame range.

        Examples
        --------
        >>> final = results["fields"].snapshot("Ez")
        >>> final["step"] == results["fields"].field_steps[-1]
        True
        """
        try:
            frames = self.fields[str(component)]
        except KeyError as exc:
            raise ValueError(
                f"Monitor did not record {component!r}; available: {tuple(self.fields)}"
            ) from exc
        return MappingProxyType(
            {
                "field": frames[index],
                "field_name": str(component),
                "t": float(self.field_times[index]),
                "step": int(self.field_steps[index]),
            }
        )

    @classmethod
    def from_compiled_state(
        cls,
        monitor: _Monitor,
        spec: CompiledMonitorSpec,
        state: _MonitorBuffers,
        config: _RunConfigLike,
    ) -> "MonitorResults":
        """Detach one monitor's immutable data from packed runtime state.

        Parameters
        ----------
        monitor : monitor specification
            Public monitor associated with the compiled entry.
        spec : CompiledMonitorSpec
            Grid-aware packed acquisition plan.
        state : monitor-buffer object
            Completed runtime carry containing packed monitor arrays.
        config : run-config object
            Time-step, resolution, dimensionality, and plane metadata.

        Returns
        -------
        MonitorResults
            NumPy-backed, read-only monitor acquisitions independent of runtime state.

        Notes
        -----
        This constructor is part of BeamZ's execution-to-results boundary. Typical
        application code receives instances from ``Simulation.run()`` or
        ``Simulation.advance()`` rather than calling it directly.
        """
        i = int(spec.monitor_index)
        fc = int(spec.freq_count)
        pc = int(spec.dft_point_count)
        count = int(np.asarray(state.counts[i]))
        if spec.recorder_index >= 0:
            recorder_count = int(np.asarray(state.recorded_counts[spec.recorder_index]))
            fields = {
                component: np.asarray(
                    state.recorded_fields[buffer_index][:recorder_count]
                ).copy()
                for component, buffer_index in zip(
                    spec.components, spec.field_buffer_indices, strict=True
                )
            }
            return cls(
                monitor=monitor,
                fields=fields,
                power_history=np.empty(0, dtype=float),
                power_timestamps=np.empty(0, dtype=float),
                power_spectrum=np.empty(0, dtype=np.complex64),
                field_times=np.asarray(
                    state.recorded_times[spec.recorder_index][:recorder_count],
                    dtype=float,
                ).copy(),
                field_steps=np.asarray(
                    state.recorded_steps[spec.recorder_index][:recorder_count],
                    dtype=int,
                ).copy(),
                sample_region=spec.sample_region,
            )
        dft_fields: dict[str, np.ndarray] = {}
        if spec.dft_enabled and fc > 0 and pc > 0:
            from beamz.lattice import public_component_2d

            comp_mask_arr = np.asarray(spec.dft_component_mask, dtype=np.float32)
            for comp_i, comp_name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
                if comp_mask_arr[comp_i] <= 0.0:
                    continue
                re = np.asarray(state.dft_vec_re[i, comp_i, :fc, :pc], dtype=np.float64)
                im = np.asarray(state.dft_vec_im[i, comp_i, :fc, :pc], dtype=np.float64)
                if not bool(config.is_3d) and comp_name in (
                    {"Ez", "Hx", "Hy"}
                    if config.polarization_2d == "tm"
                    else {"Ex", "Ey", "Hz"}
                ):
                    public_name, sign = public_component_2d(
                        comp_name, config.plane_2d, config.polarization_2d
                    )
                    dft_fields[public_name] = sign * (re + 1j * im)
                else:
                    dft_fields[comp_name] = re + 1j * im
        power_spectrum = (
            np.asarray(state.freq_flux_re[i, :fc], dtype=np.float32)
            + 1j * np.asarray(state.freq_flux_im[i, :fc], dtype=np.float32)
            if spec.accumulate_frequency and fc > 0
            else np.zeros((0,), dtype=np.complex64)
        )
        configured_frequencies = np.asarray(monitor.freqs, dtype=float).reshape(-1)
        dft_frequencies = (
            configured_frequencies[:fc]
            if configured_frequencies.size >= fc
            else np.asarray(spec.freq_hz[:fc], dtype=float)
        )
        return cls(
            monitor=monitor,
            fields={},
            power_history=np.asarray(state.powers[i, :count], dtype=float).copy(),
            power_timestamps=np.asarray(
                state.timestamps[i, :count], dtype=float
            ).copy(),
            power_spectrum=np.asarray(power_spectrum, dtype=np.complex64).copy(),
            dft_fields=dft_fields,
            dft_frequencies=dft_frequencies.copy(),
            dft_weight_sum=np.asarray(
                state.dft_weight_sum[i, :fc], dtype=np.float64
            ).copy(),
            dft_base_dt=float(config.dt),
            resolution=float(config.resolution),
            normal_axis=int(spec.normal_axis),
            normal_sign=float(spec.normal_sign),
            power_scale=float(spec.power_scale),
            integration_weights=np.asarray(spec.integration_weights),
            objective_value=None,
            sample_region=spec.sample_region,
        )


@dataclass(frozen=True)
class SimulationResults:
    """Store durable, immutable analysis outputs from simulation execution.

    ``SimulationResults`` contains monitor acquisitions and the metadata required
    to interpret them after the solver has finished. It deliberately does not own
    electromagnetic continuation state. A segmented :meth:`Simulation.advance`
    call returns a separate :class:`SimulationRun` whose ``state`` may be consumed
    independently while these results remain valid.

    Attributes
    ----------
    metadata : SimulationMetadata
        Frozen grid, time, dimensionality, coordinate, and optional material data.
    monitors : mapping of str to MonitorResults
        Immutable monitor outputs keyed by canonical monitor name.
    sources : tuple
        Immutable source specifications needed for normalization and analysis.
    normalization_source : int or None
        Source index currently used to normalize frequency-domain acquisitions.
        ``None`` means raw monitor data.
    source_launch_powers : tuple of float or None
        Internally calibrated launched powers corresponding to ``sources``.

    Examples
    --------
    Normal execution returns this type directly:

    >>> results = sim.run()
    >>> output = results["output"]
    >>> flux = output.flux

    Segmented execution keeps results and state separate:

    >>> segment = sim.advance(num_steps=50)
    >>> results = segment.results
    >>> state = segment.state
    >>> not hasattr(results, "state")
    True

    Notes
    -----
    NumPy arrays are copied from runtime buffers and made read-only. Result mappings
    are also read-only, making a retained result safe across later continuations,
    buffer donation, cache clearing, and deletion of the originating simulation.
    """

    metadata: SimulationMetadata
    monitors: Mapping[str, MonitorResults] = field(default_factory=dict)
    sources: tuple[object, ...] = field(default_factory=tuple, repr=False)
    normalization_source: int | None = None
    source_launch_powers: tuple[float | None, ...] = field(
        default_factory=tuple, repr=False
    )

    def __post_init__(self):
        if not isinstance(self.metadata, SimulationMetadata):
            raise TypeError("SimulationResults.metadata must be SimulationMetadata.")
        if not isinstance(self.monitors, Mapping):
            raise TypeError("SimulationResults.monitors must be a mapping.")
        for name, result in self.monitors.items():
            if not isinstance(name, str):
                raise TypeError("SimulationResults.monitors keys must be strings.")
            if not isinstance(result, MonitorResults):
                raise TypeError(
                    "SimulationResults.monitors values must be "
                    f"MonitorResults; got {type(result).__name__}."
                )
        object.__setattr__(self, "monitors", MappingProxyType(dict(self.monitors)))
        object.__setattr__(self, "sources", immutable_snapshot(tuple(self.sources)))
        launch_powers = tuple(self.source_launch_powers)
        if not launch_powers:
            launch_powers = (None,) * len(self.sources)
        if len(launch_powers) != len(self.sources):
            raise ValueError(
                "SimulationResults.source_launch_powers must match the number of sources."
            )
        object.__setattr__(
            self,
            "source_launch_powers",
            tuple(None if value is None else float(value) for value in launch_powers),
        )
        source = self.normalization_source
        if source is not None:
            source = int(source)
            if source < 0 or source >= len(self.sources):
                raise ValueError(
                    f"normalization source {source} is invalid for {len(self.sources)} sources."
                )
        object.__setattr__(self, "normalization_source", source)

    def __getitem__(self, name):
        """Return a named monitor result.

        Parameters
        ----------
        name : str
            Canonical monitor name.

        Returns
        -------
        MonitorResults
            Immutable acquisition associated with ``name``.

        Raises
        ------
        KeyError
            If no monitor with that name exists.

        Examples
        --------
        >>> fields = results["fields"]
        """
        return self.monitor(name)

    def monitor(self, name: str) -> MonitorResults:
        """Return the result-owned acquisition for a named monitor.

        Parameters
        ----------
        name : str
            Canonical monitor name.

        Returns
        -------
        MonitorResults
            Immutable monitor acquisition.

        Raises
        ------
        KeyError
            If ``name`` is absent from :attr:`monitors`.
        """
        return self.monitors[str(name)]

    def launched_power(self, source: int = 0) -> float:
        """Return the internally calibrated net power leaving a mode source.

        The solved source mode is integrated with the mode solver's exact
        component-staggered transverse metric after applying the compiler's
        launch-amplitude correction. It does not require an additional runtime
        monitor or reference simulation.

        Parameters
        ----------
        source : int, default=0
            Index into the simulation's source sequence.

        Returns
        -------
        float
            Positive net launched power in watts.

        Raises
        ------
        ValueError
            If ``source`` is outside the configured source range.
        RuntimeError
            If BeamZ could not calibrate a finite positive launch power for the
            selected source type.

        Examples
        --------
        >>> input_power = results.launched_power(0)
        >>> transmission = results["output"].flux / input_power
        """
        source = int(source)
        if source < 0 or source >= len(self.sources):
            raise ValueError(
                f"source={source} is invalid for {len(self.sources)} sources."
            )
        power = self.source_launch_powers[source]
        if power is None or not np.isfinite(power) or power <= 0.0:
            raise RuntimeError(
                f"An internally calibrated launched power is unavailable for source {source}."
            )
        return float(power)

    def renormalize(self, source: int | None):
        """Return an immutable raw or source-normalized result view.

        Parameters
        ----------
        source : int or None
            Source index used for field-amplitude and power normalization. Pass
            ``None`` to recover raw frequency-domain acquisitions.

        Returns
        -------
        SimulationResults
            New immutable result sharing semantic metadata while owning the selected
            normalized monitor views.

        Raises
        ------
        ValueError
            If ``source`` is outside the configured source range.

        Examples
        --------
        >>> raw = results.renormalize(None)
        >>> source_1 = results.renormalize(1)

        Notes
        -----
        Renormalization always derives from retained raw acquisitions, so repeatedly
        switching source indices does not compound scaling or lose information.
        """
        return renormalize_result(self, source)

    def to_xarray(self):
        """Convert stored acquisitions to labeled xarray data.

        Returns
        -------
        xarray.Dataset
            Dataset containing recorded fields and coordinates derived from result
            metadata.

        Notes
        -----
        xarray integration is imported lazily. Conversion uses detached result data
        and does not require the original simulation or runtime state.
        """
        return _analysis_function("adapters", "to_xarray")(self)

    def mode(self, name: str):
        """Return labeled modal data for a named mode monitor.

        Parameters
        ----------
        name : str
            Name of a ``ModeMonitor`` acquisition.

        Returns
        -------
        ModeMonitorData
            Modal amplitudes, frequencies, mode metadata, and coordinates.

        Raises
        ------
        KeyError
            If no monitor named ``name`` exists.
        TypeError or ValueError
            If the named acquisition cannot be interpreted as modal data.
        """
        return _analysis_function("adapters", "mode_data")(self, name)

    def plot_field(self, *args, **kwargs):
        """Plot a recorded field frame or frequency-domain monitor field.

        Parameters
        ----------
        *args : object
            Optional positional monitor and component selectors. Prefer explicit
            keywords in reusable code.
        **kwargs : object
            Field-selection and Matplotlib options such as ``monitor_name``,
            ``field_name``, ``frequency``, ``frame``, ``val``, ``ax``, and
            ``figsize``. ``show`` defaults to ``False``.

        Returns
        -------
        tuple
            Matplotlib ``(figure, axes)`` objects.

        Examples
        --------
        >>> fig, ax = results.plot_field(
        ...     monitor_name="frequency_fields",
        ...     field_name="Ez",
        ...     val="abs",
        ... )
        """
        kwargs.setdefault("show", False)
        return _analysis_function("plotting", "plot_result_field")(
            self, *args, **kwargs
        )

    def plot(self, **kwargs):
        """Plot stored simulation data without forcing interactive display.

        Parameters
        ----------
        **kwargs : object
            Options accepted by :meth:`plot_field`. ``show`` defaults to ``False``.

        Returns
        -------
        tuple
            Matplotlib figure and axes.
        """
        # Import plotting lazily so numerical simulation does not require
        # visualization setup.
        kwargs.setdefault("show", False)
        return self.plot_field(**kwargs)

    def show(self, **kwargs):
        """Plot stored simulation data and request interactive display.

        Parameters
        ----------
        **kwargs : object
            Options accepted by :meth:`plot_field`. ``show`` defaults to ``True``.

        Returns
        -------
        tuple
            Matplotlib figure and axes.
        """
        # Route display through the plotting API while changing only the presentation
        # flag.
        kwargs.setdefault("show", True)
        return self.plot_field(**kwargs)

    @classmethod
    def from_run(
        cls,
        simulation,
        *,
        runtime_fields: CompiledGrid,
        monitor_results: Mapping[str, MonitorResults],
        store_full_materials: bool = False,
        source_launch_powers: tuple[float | None, ...] = (),
    ) -> "SimulationResults":
        """Detach canonical analysis outputs from completed runtime buffers.

        Parameters
        ----------
        simulation : Simulation
            Immutable specification that produced the run.
        runtime_fields : CompiledGrid
            Compiled lattice used for field shapes and material metadata.
        monitor_results : mapping of str to MonitorResults
            Already-decoded monitor acquisitions.
        store_full_materials : bool, default=False
            Retain the complete material grid instead of monitor-local regions only.
        source_launch_powers : tuple of float or None, optional
            Calibrated launched powers corresponding to simulation sources.

        Returns
        -------
        SimulationResults
            Detached, normalized, immutable analysis result.

        Raises
        ------
        TypeError
            If ``simulation`` or ``runtime_fields`` has the wrong type.

        Notes
        -----
        This is the internal construction boundary used by execution. Application
        code should obtain results from ``Simulation.run()`` or from the ``results``
        member of ``Simulation.advance()``.
        """
        from .api import Simulation

        if not isinstance(simulation, Simulation):
            raise TypeError("SimulationResults.from_run requires Simulation.")
        if not isinstance(runtime_fields, CompiledGrid):
            raise TypeError("SimulationResults.from_run requires CompiledGrid.")
        monitor_results_dict = dict(monitor_results)
        metadata = SimulationMetadata.from_simulation(
            simulation,
            runtime_fields=runtime_fields,
            store_full_materials=store_full_materials,
        )
        monitor_results_dict = {
            name: replace(
                result,
                material_region=material_region_for_monitor(
                    simulation, result.monitor, runtime_fields=runtime_fields
                ),
            )
            for name, result in monitor_results_dict.items()
        }
        result = cls(
            metadata=metadata,
            monitors=monitor_results_dict,
            sources=simulation.sources,
            source_launch_powers=source_launch_powers,
        )
        source = simulation.normalize_source
        return result.renormalize(None if not result.sources else source)


@dataclass(frozen=True, slots=True)
class SimulationRun:
    """Own the two outputs of a segmented simulation execution.

    ``SimulationRun`` is returned by :meth:`Simulation.advance`. It groups values
    for convenience without merging their lifecycles: ``results`` is immutable,
    detached analysis data, while ``state`` is an explicitly owned continuation
    value that may later be preserved, branched, or donated.

    Attributes
    ----------
    results : SimulationResults
        Durable monitor acquisitions and metadata for all samples accumulated up to
        the end of this segment.
    state : SimulationState
        Complete runtime carry positioned at the next timestep after this segment.

    Examples
    --------
    Continue a segmented execution:

    >>> first = sim.advance(num_steps=50)
    >>> results = first.results
    >>> second = sim.advance(state=first.state, num_steps=25)

    Branch from the same preserved state:

    >>> branch_a = sim.advance(state=first.state, num_steps=10)
    >>> branch_b = sim.advance(state=first.state, num_steps=10)

    Transfer state ownership for the lower-memory path:

    >>> final = sim.advance(
    ...     state=second.state,
    ...     num_steps=25,
    ...     donate_state=True,
    ... )
    >>> # Do not access second.state after the donating call.

    Notes
    -----
    ``Simulation.run()`` intentionally returns ``SimulationResults`` directly, not
    ``SimulationRun``, because a complete normal run has no public continuation
    requirement. Use this container only when working with ``advance``.

    Donation affects only the state passed into the next execution. It never
    invalidates ``results`` from this or any earlier segment.
    """

    results: SimulationResults
    state: SimulationState = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.results, SimulationResults):
            raise TypeError("SimulationRun.results must be SimulationResults.")
        if not isinstance(self.state, SimulationState):
            raise TypeError("SimulationRun.state must be SimulationState.")
