"""Immutable simulation request, compiled-program, and runtime-state values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, TypeAlias

import jax.numpy as jnp

from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.devices._immutable import immutable_snapshot
from beamz.devices.monitors.compiler import CompiledMonitorSpec
from beamz.devices.monitors.monitors import _Monitor
from beamz.devices.sources.compiler import CompiledSourceSpec

# Requests are detached, hashable compiler inputs. They prevent later mutation of a
# public Simulation or device object from changing a cached program's meaning.
ShardingToken: TypeAlias = tuple[bool, str, int | None, str | None]


@dataclass(frozen=True, slots=True)
class RunSpec:
    # This record is an immutable ownership boundary between flexible public input and
    # deterministic compilation.
    dt: float
    num_steps: int
    total_steps: int
    t0: float
    loop_kind: str
    source_single_slab_dense: bool
    sharding: ShardingToken


@dataclass(frozen=True, slots=True)
class DomainSpec:
    # This record is an immutable ownership boundary between flexible public input and
    # deterministic compilation.
    size: tuple[float, float, float]
    is_3d: bool
    plane_2d: str
    coordinate_offset: tuple[float, float, float]
    polarization_2d: str = "tm"


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    # This record is an immutable ownership boundary between flexible public input and
    # deterministic compilation.
    run: RunSpec
    domain: DomainSpec
    materials: MaterialGrid
    sources: tuple[object, ...]
    monitors: tuple[_Monitor, ...]
    boundaries: tuple[object, ...]
    compiler_sharding: Any = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        # Snapshot the only request member that may arrive as a mutable backend object.
        object.__setattr__(
            self, "compiler_sharding", immutable_snapshot(self.compiler_sharding)
        )


# ``CompiledGrid`` owns setup-time layout and material metadata only. The runtime has
# one explicit state value below.


@dataclass(frozen=True, slots=True, eq=False)
class CompiledGrid:
    """One lowered Yee lattice with materials, masks, and absorber profiles."""

    # This object centralizes Yee arrays and their static material metadata; runtime
    # evolution lives in explicit state values.
    material_grid: MaterialGrid
    geometry: RectilinearGrid
    component_shapes: Mapping[str, tuple[int, ...]]
    resolution: float
    plane_2d: str
    polarization_2d: str
    permittivity: jnp.ndarray
    conductivity: jnp.ndarray
    permeability: jnp.ndarray
    metallic_masks: Any
    boundaries: tuple[Any, ...]
    has_pml: bool
    has_cpml: bool
    pml_data: Mapping[str, Any] | None
    materials: Any
    Ex: jnp.ndarray
    Ey: jnp.ndarray
    Ez: jnp.ndarray
    Hx: jnp.ndarray
    Hy: jnp.ndarray
    Hz: jnp.ndarray
    total_conductivity: jnp.ndarray
    eps_x: jnp.ndarray
    eps_y: jnp.ndarray
    eps_z: jnp.ndarray
    sig_x: jnp.ndarray
    sig_y: jnp.ndarray
    sig_z: jnp.ndarray
    region_x: jnp.ndarray
    region_y: jnp.ndarray
    region_z: jnp.ndarray
    eps_ex: jnp.ndarray
    eps_ey: jnp.ndarray
    eps_ez: jnp.ndarray
    sigma_m_hx: jnp.ndarray
    sigma_m_hy: jnp.ndarray
    sigma_m_hz: jnp.ndarray
    mu_hx: jnp.ndarray
    mu_hy: jnp.ndarray
    mu_hz: jnp.ndarray


class SimulationState(NamedTuple):
    """Store every evolving value required to continue a simulation.

    ``SimulationState`` is an immutable JAX pytree: replacing the Python tuple is
    cheap, while its array leaves may reside on CPU, GPU, or sharded devices. Users
    normally receive state from :meth:`Simulation.advance` and pass it to a later
    ``advance`` call unchanged.

    Attributes
    ----------
    ex, ey, ez : array-like
        Electric Yee-field components in BeamZ's canonical storage layout.
    hx, hy, hz : array-like
        Magnetic Yee-field components in canonical storage layout.
    cpml_psi_h_terms, cpml_psi_e_terms : tuple of array-like
        Packed convolutional-PML recurrence memory for magnetic and electric
        updates. Empty when CPML is disabled.
    powers, timestamps, counts : array-like
        Time-domain monitor accumulators and valid-sample counts.
    freq_flux_re, freq_flux_im : array-like
        Real and imaginary frequency-domain flux accumulators.
    freq_phase_re, freq_phase_im : array-like
        Recurrence phases used by frequency-domain monitor accumulation.
    dft_vec_re, dft_vec_im, dft_weight_sum : array-like
        Packed field-DFT accumulators and normalization weights.
    recorded_fields : tuple of array-like
        Preallocated field-recorder buffers.
    recorded_steps, recorded_times, recorded_counts : tuple of array-like
        Recorder indices, physical sample times, and valid-frame counts.
    t : array-like
        Scalar physical time in seconds.
    current_step : array-like
        Scalar zero-based index of the next timestep to execute.

    Notes
    -----
    State and analysis results have separate lifecycles. ``SimulationResults`` is
    durable and never contains a ``SimulationState``; ``SimulationRun`` owns the two
    values side by side for continuation workflows.

    Passing state to ``Simulation.advance`` or ``Simulation.step`` preserves it by
    default. If ``donate_state=True`` is used, JAX may invalidate any of its device
    buffers and the input state must be treated as consumed.

    Examples
    --------
    >>> segment = sim.advance(num_steps=25)
    >>> state = segment.state
    >>> int(state.current_step)
    25
    >>> continued = sim.advance(state=state, num_steps=25)
    """

    # Field arrays, absorber memory, monitor accumulation, and clocks advance atomically. A
    # continuation therefore cannot accidentally combine values from different steps.
    ex: Any
    ey: Any
    ez: Any
    hx: Any
    hy: Any
    hz: Any
    cpml_psi_h_terms: tuple[Any, ...]
    cpml_psi_e_terms: tuple[Any, ...]
    powers: Any
    timestamps: Any
    counts: Any
    freq_flux_re: Any
    freq_flux_im: Any
    freq_phase_re: Any
    freq_phase_im: Any
    dft_vec_re: Any
    dft_vec_im: Any
    dft_weight_sum: Any
    recorded_fields: tuple[Any, ...]
    recorded_steps: tuple[Any, ...]
    recorded_times: tuple[Any, ...]
    recorded_counts: tuple[Any, ...]
    t: Any
    current_step: Any

    @classmethod
    def initial(cls, fields, *, t: float, current_step: int = 0):
        """Allocate fresh runtime state from a compiled Yee lattice.

        Parameters
        ----------
        fields : CompiledGrid or compatible object
            Compiled lattice providing initial ``Ex`` through ``Hz`` arrays.
        t : float
            Initial physical time in seconds.
        current_step : int, default=0
            Index of the next timestep to execute.

        Returns
        -------
        SimulationState
            Independent field arrays with empty boundary and monitor accumulators.

        Notes
        -----
        Application code should normally use ``Simulation.initial_state()`` so the
        state is guaranteed to match the simulation's compiled lattice.
        """
        empty2 = jnp.zeros((0, 0), dtype=jnp.float32)
        return cls(
            # Copies keep the compiled lattice reusable when JAX donates runtime buffers.
            ex=jnp.array(fields.Ex),
            ey=jnp.array(fields.Ey),
            ez=jnp.array(fields.Ez),
            hx=jnp.array(fields.Hx),
            hy=jnp.array(fields.Hy),
            hz=jnp.array(fields.Hz),
            cpml_psi_h_terms=(),
            cpml_psi_e_terms=(),
            powers=empty2,
            timestamps=empty2,
            counts=jnp.zeros((0,), dtype=jnp.int32),
            freq_flux_re=empty2,
            freq_flux_im=empty2,
            freq_phase_re=empty2,
            freq_phase_im=empty2,
            dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
            dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
            dft_weight_sum=empty2,
            recorded_fields=(),
            recorded_steps=(),
            recorded_times=(),
            recorded_counts=(),
            t=jnp.asarray(t, dtype=jnp.float32),
            current_step=jnp.asarray(current_step, dtype=jnp.int32),
        )


# The compiler plans below are values in the same lifecycle as SimulationRequest and
# SimulationState. Keeping them here removes the former second, compiled-only type model.
class UpdateCoefficients(NamedTuple):
    h_decay_x: jnp.ndarray
    h_source_x: jnp.ndarray
    h_sigma_m_x: jnp.ndarray
    h_decay_y: jnp.ndarray
    h_source_y: jnp.ndarray
    h_sigma_m_y: jnp.ndarray
    h_decay_z: jnp.ndarray
    h_source_z: jnp.ndarray
    h_sigma_m_z: jnp.ndarray
    e_decay_x: jnp.ndarray
    e_source_x: jnp.ndarray
    e_conductivity_x: jnp.ndarray
    e_permittivity_x: jnp.ndarray
    e_decay_y: jnp.ndarray
    e_source_y: jnp.ndarray
    e_conductivity_y: jnp.ndarray
    e_permittivity_y: jnp.ndarray
    e_decay_z: jnp.ndarray
    e_source_z: jnp.ndarray
    e_conductivity_z: jnp.ndarray
    e_permittivity_z: jnp.ndarray


class DerivativeMetricPlan(NamedTuple):
    """Separable inverse distances for staggered curl derivatives.

    Isotropic grids retain the scalar update path and use empty leaves. Axis-uniform
    grids store one scalar per direction; fully rectilinear grids store one vector
    per direction and stagger.
    """

    e_to_h_x: jnp.ndarray
    e_to_h_y: jnp.ndarray
    e_to_h_z: jnp.ndarray
    h_to_e_x: jnp.ndarray
    h_to_e_y: jnp.ndarray
    h_to_e_z: jnp.ndarray


class CpmlPackedSlabSpec(NamedTuple):
    axis: int
    low: int
    high: int
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class CpmlTerm:
    """One precomputed CPML recurrence on a packed derivative slab."""

    component: str
    axis: int
    sign: float
    a: jnp.ndarray
    b: jnp.ndarray
    inv_kappa: jnp.ndarray
    slab: CpmlPackedSlabSpec


@dataclass(frozen=True, slots=True)
class ShardingConfig:
    enabled: bool = False
    axis: Literal["auto", "z", "y", "x"] = "auto"
    num_devices: int | None = None
    backend: Literal["cpu", "gpu"] | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ShardingLayout:
    enabled: bool
    axis_name: str
    axis: int
    num_devices: int
    backend: str | None
    logical_shapes: Mapping[str, tuple[int, ...]]
    padded_shapes: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        # Freeze nested mappings too; a frozen dataclass alone does not protect them.
        for name in (
            "logical_shapes",
            "padded_shapes",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True, slots=True)
class RunConfig:
    resolution: float
    dt: float
    num_steps: int
    plane_2d: str
    is_3d: bool
    metric_kind: str = "isotropic_uniform"
    polarization_2d: str = "tm"
    loop_kind: str = "scan"
    source_single_slab_dense: bool = False
    sharding: ShardingConfig = ShardingConfig()


@dataclass(frozen=True, slots=True, eq=False)
class MetallicPlan:
    ex_mask: Any
    ey_mask: Any
    ez_mask: Any
    hx_mask: Any
    hy_mask: Any
    hz_mask: Any


@dataclass(frozen=True, slots=True, eq=False)
class CpmlPlan:
    enabled: bool
    metallic_edges: frozenset[str]
    h_terms: tuple[CpmlTerm, ...]
    e_terms: tuple[CpmlTerm, ...]


@dataclass(frozen=True, slots=True, eq=False)
class BoundaryPlan:
    metallic_edges_2d: frozenset[str]
    cpml: CpmlPlan
    metallic: MetallicPlan


@dataclass(frozen=True, slots=True, eq=False)
class ShardingPlan:
    layout: ShardingLayout
    mesh: Any


@dataclass(frozen=True, slots=True, eq=False)
class CompiledProgram:
    """Immutable numerical plan consumed by the execution runtime."""

    grid: CompiledGrid
    config: RunConfig
    coefficients: UpdateCoefficients
    metrics: DerivativeMetricPlan
    boundary: BoundaryPlan
    sources: tuple[CompiledSourceSpec, ...]
    monitors: tuple[CompiledMonitorSpec, ...]
    sharding: ShardingPlan
