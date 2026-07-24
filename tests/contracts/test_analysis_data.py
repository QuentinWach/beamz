from dataclasses import fields

import numpy as np
import pytest

from beamz import FieldMonitor
from beamz.analysis.data import AnalysisData, analysis_data
from beamz.simulation.results import FieldMetadata, SimulationMetadata


def test_analysis_data_is_the_five_part_immutable_contract():
    monitor = FieldMonitor(
        center=(0.0, 0.0, 0.0),
        size=(1.0, 0.0, 0.0),
        freqs=[2.0],
        fields=("Ez",),
        name="field",
    )
    coordinates = SimulationMetadata(
        dt=0.25,
        resolution=0.5,
        is_3d=False,
        plane_2d="xy",
        coordinate_offset=(0.0, 0.0, 0.0),
        time=np.empty(0),
        width=1.0,
        height=1.0,
        depth=0.0,
        fields=FieldMetadata((1, 1), {"Ez": (1, 1)}),
    )
    contract = AnalysisData(
        coordinates=coordinates,
        fields={"Ez": np.asarray([[3.0 + 4.0j]])},
        materials=None,
        frequencies=np.asarray([2.0]),
        monitor_geometry=monitor,
    )

    data = analysis_data(contract)

    assert tuple(item.name for item in fields(AnalysisData)) == (
        "coordinates",
        "fields",
        "materials",
        "frequencies",
        "monitor_geometry",
    )
    assert data.name == "field"
    assert data.dt == 0.25
    assert data.resolution == 0.5
    np.testing.assert_allclose(data.field("Ez"), [[3.0 + 4.0j]])
    with pytest.raises(ValueError):
        data.frequencies[0] = 3.0
    with pytest.raises(TypeError):
        data.fields["Ez"] = np.zeros((1, 1))
