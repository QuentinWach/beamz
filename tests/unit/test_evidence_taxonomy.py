import pytest

from tests.evidence import evidence_markers_for_path


@pytest.mark.parametrize(
    "relative,expected",
    [
        ("unit/test_helper.py", ("contract",)),
        ("contracts/test_api.py", ("contract",)),
        ("kernels/test_curl.py", ("invariant",)),
        ("integration/test_run.py", ("contract",)),
        ("characterization/test_wave_smoke.py", ("characterization", "smoke")),
        ("validation/analytical/test_fresnel.py", ("validation",)),
        ("validation/invariants/test_energy.py", ("invariant",)),
        ("validation/convergence/test_order.py", ("validation",)),
        ("validation/regression/test_bug_123.py", ("validation",)),
        ("differential/test_meep.py", ("validation", "differential")),
        ("hardware/test_cuda.py", ("contract", "hardware")),
        ("performance/test_runtime.py", ("characterization", "performance")),
        ("docs/test_quickstart.py", ("contract", "docs")),
        ("pdk/test_external.py", ("contract", "pdk")),
    ],
)
def test_evidence_marker_follows_directory_contract(relative, expected, tmp_path):
    tests_root = tmp_path / "tests"
    path = tests_root / relative

    assert evidence_markers_for_path(path, tests_root=tests_root) == expected


def test_unknown_evidence_directory_is_rejected(tmp_path):
    tests_root = tmp_path / "tests"

    with pytest.raises(ValueError, match="outside the evidence-oriented"):
        evidence_markers_for_path(
            tests_root / "misc" / "test_something.py", tests_root=tests_root
        )
