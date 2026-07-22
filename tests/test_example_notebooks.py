from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "notebooks" / "cosine_waveguide_crossing.ipynb"
NOTEBOOKS = tuple(sorted((ROOT / "examples" / "notebooks").glob("*.ipynb")))


def _notebook_source(path: Path = NOTEBOOK) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", ())) for cell in notebook.get("cells", ())
    )


def test_cosine_crossing_notebook_uses_canonical_monitor_geometry():
    source = _notebook_source()

    assert "Lxy_target = 2 * l_t + w_out + 2 * l_wg" in source
    assert "source_x = -Lx / 2 + l_wg / 2" in source
    assert "through_x = Lx / 2 - l_wg / 2" in source
    assert "cross_y = Ly / 2 - l_wg / 2" in source
    assert "field_monitor = bz.FieldMonitor(" in source
    assert "center=(0.0, 0.0, 0.0)" in source
    assert "size=(Lx, Ly, 0.0)" in source
    assert "freqs=[freq0]" in source
    assert "grid_shape = sim0.to_request(num_steps=1).materials.shape" in source
    assert ".to_request(num_steps=1).grid" not in source
    assert "source/monitor clearance to x/y CPML" in source
    assert "port_edge_clearance" not in source


def test_cosine_crossing_notebook_uses_internal_launch_power_normalization():
    source = _notebook_source()

    assert "min_steps_per_wvl = 10" in source
    assert 'flux_through = sim_data["flux_through"].flux' in source
    assert 'flux_cross = sim_data["flux_cross"].flux' in source
    assert "source_power = sim_data.launched_power(source=0)" in source
    assert "T_through = flux_through / source_power" in source
    assert "T_cross = flux_cross / source_power" in source
    assert "same Yee-grid Poynting convention" in source
    assert "flux_input" not in source
    assert "sim_reference" not in source
    assert "flux_reference" not in source
    assert "np.clip" not in source


def test_cosine_crossing_notebook_has_no_stale_error_outputs():
    raw = NOTEBOOK.read_text(encoding="utf-8")
    notebook = json.loads(raw)

    assert "BEAMZ 3D ModeSource requires" not in raw
    assert all(
        not cell.get("outputs")
        for cell in notebook.get("cells", ())
        if cell.get("cell_type") == "code"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_example_notebook_code_cells_compile(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook.get("cells", ())):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ()))
        compile(source, f"{path.name}:cell-{index}", "exec")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_example_notebooks_use_detached_results_workflow(path):
    source = _notebook_source(path)
    forbidden = (
        "record_fields=",
        "sim0.fields",
        "sim.fields.permittivity",
        "sim.current_step",
        "sim.engine",
        "sim.s_parameters(",
        "beamz.simulation.compiled",
        ".copy(update=",
        "bz.Monitor(",
        "bz.PortSpec(",
        "reference_monitor=",
        "power_spectrum_frequencies=",
        "dft_frequencies=",
        "dft_enabled=",
    )

    assert ".run(" in source or ".advance(" in source
    assert "run_compiled" not in source
    for token in forbidden:
        assert token not in source


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_example_notebooks_have_no_cached_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert all(
        not cell.get("outputs")
        for cell in notebook.get("cells", ())
        if cell.get("cell_type") == "code"
    )


def test_waveguide_notebook_uses_field_recorder_and_result_analysis():
    source = _notebook_source(ROOT / "examples" / "notebooks" / "waveguide_demo.ipynb")

    assert "bz.FieldRecorder(" in source
    assert 'sim_data.monitor("ey_slice")' in source
    assert "s_parameters(\n    sim_data," in source


def test_modal_notebook_projects_modes_from_results():
    source = _notebook_source(
        ROOT / "examples" / "notebooks" / "modal_sources_monitors.ipynb"
    )

    assert 'sim_data_single.mode("mode")' in source
    assert 'sim_data_bb.mode("mode")' in source
    assert 'sim_data_jct_bb.mode("mode")' in source
    assert "mode_source_request.mode_spec, num_freqs=1" in source
    assert "np.testing.assert_allclose(single_profile_freqs, [freq0])" in source
    assert "sim_jct_bb.num_steps" in source
    assert ".state.current_step" not in source


def test_tiny_crossing_example_uses_current_simulation_modules():
    path = ROOT / "examples" / "compact_models" / "tiny_beamz_crossing.py"
    source = path.read_text(encoding="utf-8")

    compile(source, str(path), "exec")
    assert "beamz.simulation.core" not in source
    assert "use_fixed_micromode_y_projection_convention" not in source
