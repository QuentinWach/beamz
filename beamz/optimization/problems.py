"""Reusable inverse-design problem and multi-port sweep orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from beamz.analysis import s_parameters as extract_s_parameters
from beamz.analysis.data import analysis_inputs
from beamz.analysis.mode_projection import build_port_projection
from beamz.analysis.sparameters import wave_selectors
from beamz.devices.ports import Port
from beamz.simulation.api import Simulation
from beamz.simulation.results import SimulationResults

from .trainable import DesignRegion, DifferentiableResult, DifferentiableSimulation


@dataclass(frozen=True, slots=True)
class _PortProjection:
    monitor_name: str
    components: tuple[str, ...]
    pinv: jax.Array
    component_phases: jax.Array


class DifferentiablePortProjector:
    """Apply fixed port modes to differentiable frequency-domain fields.

    Port eigensolutions are calibrated once from detached simulation metadata.
    They remain constant during optimization, matching the Ceviche convention
    that ports lie outside the trainable region and mode-solver derivatives are
    excluded from the computational graph.
    """

    def __init__(self, results: SimulationResults, ports: Sequence[Port]) -> None:
        ports = tuple(ports)
        inputs = analysis_inputs(results)
        context = next(iter(inputs.values()))
        if context.is_3d:
            raise NotImplementedError(
                "Differentiable modal projection currently supports 2D simulations."
            )
        self.ports = MappingProxyType({port.name: port for port in ports})
        if len(self.ports) != len(ports):
            raise ValueError("Differentiable projector port names must be unique.")
        projection_cache = {}
        projections = {}
        frequency_grid = None
        for port in ports:
            try:
                data = inputs[port.monitor_name]
            except KeyError as exc:
                raise ValueError(
                    f"Missing mode monitor {port.monitor_name!r} for port "
                    f"{port.name!r}."
                ) from exc
            monitor = data.monitor_geometry
            frequencies = np.asarray(data.frequencies, dtype=float)
            if frequency_grid is None:
                frequency_grid = frequencies
            elif not np.allclose(frequency_grid, frequencies):
                raise ValueError("All differentiable ports must share frequencies.")
            pinv = []
            phases = []
            components = None
            for frequency in frequencies:
                projection = build_port_projection(
                    data,
                    port,
                    monitor,
                    float(frequency),
                    projection_cache,
                )
                current_components = tuple(projection["components"])
                if components is None:
                    components = current_components
                elif current_components != components:
                    raise ValueError("Port projection components changed by frequency.")
                pinv.append(np.asarray(projection["pinv"], dtype=np.complex64))
                delay = float(projection.get("modal_plane_delay_s", 0.0))
                phases.append(
                    [
                        np.exp(
                            (-1j * np.pi * float(frequency) * float(data.dt))
                            if component.startswith("H")
                            else (1j * 2.0 * np.pi * float(frequency) * delay)
                        )
                        for component in current_components
                    ]
                )
            projections[port.name] = _PortProjection(
                monitor_name=str(monitor.name),
                components=tuple(components or ()),
                pinv=jnp.asarray(np.stack(pinv)),
                component_phases=jnp.asarray(np.asarray(phases, dtype=np.complex64)),
            )
        self.frequencies = np.asarray(frequency_grid, dtype=float)
        self._projections = MappingProxyType(projections)

    def amplitudes(
        self, result: DifferentiableResult, port: str | Port
    ) -> tuple[jax.Array, jax.Array]:
        """Return positive- and negative-basis modal amplitudes."""

        name = port.name if isinstance(port, Port) else str(port)
        try:
            projection = self._projections[name]
        except KeyError as exc:
            raise KeyError(f"Unknown differentiable port {name!r}.") from exc
        fields = [
            result.field(projection.monitor_name, component)
            * projection.component_phases[:, index, None]
            for index, component in enumerate(projection.components)
        ]
        vector = jnp.concatenate(fields, axis=1)
        amplitudes = jnp.einsum("fij,fj->fi", projection.pinv, vector)
        return amplitudes[:, 0], amplitudes[:, 1]

    def s_parameter(
        self,
        result: DifferentiableResult,
        *,
        source_port: str | Port,
        output_port: str | Port,
    ) -> jax.Array:
        """Return a differentiable complex modal S-parameter vector."""

        source_name = (
            source_port.name if isinstance(source_port, Port) else str(source_port)
        )
        output_name = (
            output_port.name if isinstance(output_port, Port) else str(output_port)
        )
        source = self.ports[source_name]
        output = self.ports[output_name]
        source_branches = self.amplitudes(result, source_name)
        output_branches = self.amplitudes(result, output_name)
        incident, _ = wave_selectors(source, is_3d=False)
        _, scattered = wave_selectors(output, is_3d=False)
        denominator = source_branches[0 if incident == "plus" else 1]
        numerator = output_branches[0 if scattered == "plus" else 1]
        return jnp.where(
            jnp.abs(denominator) > 1e-18,
            numerator / denominator,
            jnp.zeros_like(numerator),
        )


@dataclass(frozen=True, slots=True)
class PortSweepResult:
    """Dense multi-frequency scattering data and steady-state fields.

    ``s_parameters`` uses axes ``(frequency, excited_port, output_port)``.
    ``fields`` uses ``(field_frequency, excited_port, *material_grid_shape)``;
    ``field_frequencies`` is intentionally independent of the port-monitor
    ``frequencies`` grid.
    """

    frequencies: np.ndarray
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    s_parameters: np.ndarray
    fields: np.ndarray | None
    runs: Mapping[str, SimulationResults]
    field_frequencies: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("frequencies", "s_parameters"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.field_frequencies is not None:
            field_frequencies = np.array(self.field_frequencies, copy=True)
            field_frequencies.setflags(write=False)
            object.__setattr__(self, "field_frequencies", field_frequencies)
        if self.fields is not None:
            fields = np.array(self.fields, copy=True)
            fields.setflags(write=False)
            object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "input_ports", tuple(self.input_ports))
        object.__setattr__(self, "output_ports", tuple(self.output_ports))
        object.__setattr__(self, "runs", MappingProxyType(dict(self.runs)))


class InverseDesignProblem:
    """Bind fixed per-port simulations to one trainable material region.

    A separate simulation specification is supplied for each excitable port because
    sources are static compiled geometry. All simulations share the same density
    variable and can acquire many wavelengths in a single broadband FDTD run.
    """

    def __init__(
        self,
        simulations: Mapping[str, Simulation],
        ports: Sequence[Port],
        design_region: DesignRegion,
        *,
        field_monitor: str | None = None,
        field_component: str = "Ez",
        rematerialize: bool = True,
    ) -> None:
        ports = tuple(ports)
        if not ports:
            raise ValueError("InverseDesignProblem requires at least one Port.")
        if len({port.name for port in ports}) != len(ports):
            raise ValueError("InverseDesignProblem port names must be unique.")
        simulations = dict(simulations)
        unknown = set(simulations) - {port.name for port in ports}
        if unknown:
            raise ValueError(
                f"Simulations contain unknown source ports: {sorted(unknown)}"
            )
        if not simulations:
            raise ValueError("InverseDesignProblem requires at least one simulation.")
        self.ports = ports
        self.design_region = design_region
        self.field_monitor = None if field_monitor is None else str(field_monitor)
        self.field_component = str(field_component)
        self.simulations = MappingProxyType(simulations)
        self._trainable = MappingProxyType(
            {
                name: DifferentiableSimulation(
                    simulation,
                    design_region,
                    rematerialize=rematerialize,
                )
                for name, simulation in simulations.items()
            }
        )
        shapes = {trainable.variable_shape for trainable in self._trainable.values()}
        if len(shapes) != 1:
            raise ValueError("All source simulations must share one density shape.")

    @property
    def variable_shape(self) -> tuple[int, ...]:
        return next(iter(self._trainable.values())).variable_shape

    def differentiable(self, source_port: str | Port) -> DifferentiableSimulation:
        name = source_port.name if isinstance(source_port, Port) else str(source_port)
        try:
            return self._trainable[name]
        except KeyError as exc:
            raise KeyError(
                f"Port {name!r} has no configured source simulation."
            ) from exc

    def port_projector(
        self,
        density,
        *,
        source_port: str | Port | None = None,
    ) -> DifferentiablePortProjector:
        """Calibrate fixed port modes for differentiable S-parameter objectives."""

        name = (
            next(iter(self.simulations))
            if source_port is None
            else (
                source_port.name if isinstance(source_port, Port) else str(source_port)
            )
        )
        results = self.differentiable(name).calibration_results(density)
        return DifferentiablePortProjector(results, self.ports)

    def run(
        self,
        density,
        *,
        excite_ports: Sequence[str | Port] | None = None,
    ) -> PortSweepResult:
        names = (
            tuple(self.simulations)
            if excite_ports is None
            else tuple(
                port.name if isinstance(port, Port) else str(port)
                for port in excite_ports
            )
        )
        if not names:
            raise ValueError("excite_ports must contain at least one configured port.")
        output_names = tuple(port.name for port in self.ports)
        runs: dict[str, SimulationResults] = {}
        columns = []
        fields = []
        frequencies = None
        field_frequencies = None
        for name in names:
            result = self.differentiable(name).run_results(density)
            runs[name] = result
            extracted = extract_s_parameters(
                result,
                source_port=name,
                ports=self.ports,
                output_ports=output_names,
            )
            current_frequencies = np.asarray(extracted.frequencies, dtype=float)
            if frequencies is None:
                frequencies = current_frequencies
            elif not np.allclose(frequencies, current_frequencies):
                raise ValueError(
                    "Source simulations returned different frequency grids."
                )
            columns.append(
                np.stack(
                    [extracted.s_matrix[(output, name)] for output in output_names],
                    axis=-1,
                )
            )
            if self.field_monitor is not None:
                field_result = result[self.field_monitor]
                current_field_frequencies = np.asarray(
                    field_result.get_dft_frequencies(),
                    dtype=float,
                )
                if field_frequencies is None:
                    field_frequencies = current_field_frequencies
                elif not np.allclose(
                    field_frequencies,
                    current_field_frequencies,
                ):
                    raise ValueError(
                        "Source simulations returned different field-monitor "
                        "frequency grids."
                    )
                values = field_result.get_dft_component(self.field_component)
                grid_shape = tuple(result.metadata.fields.grid_shape)
                fields.append(
                    np.asarray(values).reshape(
                        (len(current_field_frequencies), *grid_shape)
                    )
                )
        dense_s = np.stack(columns, axis=1)
        dense_fields = None if not fields else np.stack(fields, axis=1)
        return PortSweepResult(
            frequencies=np.asarray(frequencies),
            input_ports=names,
            output_ports=output_names,
            s_parameters=dense_s,
            field_frequencies=(
                None if field_frequencies is None else np.asarray(field_frequencies)
            ),
            fields=dense_fields,
            runs=runs,
        )


__all__ = [
    "DifferentiablePortProjector",
    "InverseDesignProblem",
    "PortSweepResult",
]
