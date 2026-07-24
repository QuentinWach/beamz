from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "optimization"
ADDITIONAL_EXAMPLES = {
    "ceviche_beam_splitter_o_band.py",
    "ceviche_mode_converter_o_band.py",
    "ceviche_wdm_o_band.py",
}
BEND_EXAMPLES = {
    "ceviche_bend_1550nm.py",
    "ceviche_bend_o_band.py",
}


def test_optimization_directory_has_three_small_additional_examples():
    files = {path.name for path in EXAMPLE_DIR.glob("*.py")}

    assert files == BEND_EXAMPLES | ADDITIONAL_EXAMPLES
    for name in ADDITIONAL_EXAMPLES:
        assert len((EXAMPLE_DIR / name).read_text().splitlines()) <= 20
