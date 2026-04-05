import json

import numpy as np
import pytest

from beamz import Design, GaussianSource, Material, Monitor, PML, Simulation
from beamz.simulation.spec import SimulationSpec


def _build_spec_backed_simulation():
    resolution = 0.1
    time = np.array([0.0, 0.1, 0.2, 0.3], dtype=float)
    design = Design(width=1.0, height=1.0, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(0.5, 0.5),
        width=0.12,
        signal=np.array([0.0, 1.0, 0.5, 0.0], dtype=float),
    )
    monitor = Monitor(
        start=(0.25, 0.25),
        end=(0.25, 0.75),
        record_interval=1,
        name="m0",
    )
    return Simulation(
        design=design,
        devices=[source, monitor],
        boundaries=[PML(thickness=0.2)],
        resolution=resolution,
        time=time,
    )


def test_compiled_engine_runs_from_deserialized_spec_graph():
    sim = _build_spec_backed_simulation()
    payload = json.loads(json.dumps(sim.spec.to_dict()))
    spec = SimulationSpec.from_dict(payload)
    restored = Simulation.from_spec(spec)

    object.__setattr__(restored, "_design", None)
    object.__setattr__(restored, "_boundaries", ())

    result = restored.session.run_compiled(num_steps=2, record_interval=1, progress=False)

    assert restored.session.spec == spec
    assert restored.runtime.initialized is True
    assert restored.runtime.fields is not None
    assert restored.current_step == 2
    assert restored.pml_data is not None
    assert "fields" in result
    assert result["fields"]["Ez"].shape[0] == 2


def test_deprecated_s_matrix_entrypoints_raise_clear_errors():
    sim = _build_spec_backed_simulation()

    with pytest.raises(RuntimeError, match="deprecated and removed"):
        sim.get_S_matrix()

    with pytest.raises(RuntimeError, match="deprecated and removed"):
        sim.get_s_matrix()
