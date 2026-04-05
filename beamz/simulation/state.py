from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SimulationRuntime:
    initialized: bool = False
    fields: object = None
    dt: float = 0.0
    num_steps: int = 0
    t: float = 0.0
    current_step: int = 0
    pml_data: object = None
    compiled_program: object = None
    compiled_program_signature: object = None
    compiled_program_cache: dict = field(default_factory=dict)
    compiled_monitor_state: object = None
