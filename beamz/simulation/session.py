from __future__ import annotations

from beamz.simulation import build
from beamz.simulation.state import SimulationRuntime


class SimulationSession:
    """Runtime owner for a Simulation model."""

    _RUNTIME_INIT_ATTRS = frozenset(
        {
            "fields",
            "dt",
            "num_steps",
            "t",
            "current_step",
            "pml_data",
        }
    )
    _RUNTIME_DIRECT_ATTRS = frozenset(
        {
            "compiled_program",
            "compiled_program_signature",
            "compiled_program_cache",
            "compiled_monitor_state",
        }
    )

    def __init__(self, simulation, runtime_state: SimulationRuntime | None = None):
        object.__setattr__(self, "simulation", simulation)
        object.__setattr__(
            self,
            "runtime",
            SimulationRuntime() if runtime_state is None else runtime_state,
        )

    @property
    def _design(self):
        return getattr(self.simulation, "_design", None)

    @property
    def spec(self):
        return self.simulation.spec

    @property
    def design(self):
        return self.simulation.design

    @property
    def devices(self):
        return self.simulation.devices

    @property
    def boundaries(self):
        return self.simulation.boundaries

    def _ensure_runtime_initialized(self):
        build.ensure_runtime_initialized(self)

    def invalidate_runtime(self):
        build.invalidate_runtime(self)

    def reset_compiled_state(self):
        runtime = self.runtime
        runtime.compiled_program = None
        runtime.compiled_program_signature = None
        runtime.compiled_program_cache.clear()
        runtime.compiled_monitor_state = None

    def reset(self, *, invalidate_runtime: bool):
        if invalidate_runtime:
            self.invalidate_runtime()
        self.reset_compiled_state()

    def __getattr__(self, name):
        if name in self._RUNTIME_INIT_ATTRS:
            self._ensure_runtime_initialized()
            return getattr(self.runtime, name)
        if name in self._RUNTIME_DIRECT_ATTRS:
            return getattr(self.runtime, name)
        return getattr(self.simulation, name)

    def __setattr__(self, name, value):
        if name in {"simulation", "runtime"}:
            object.__setattr__(self, name, value)
            return
        if name in self._RUNTIME_INIT_ATTRS | self._RUNTIME_DIRECT_ATTRS:
            setattr(self.runtime, name, value)
            return
        object.__setattr__(self, name, value)
