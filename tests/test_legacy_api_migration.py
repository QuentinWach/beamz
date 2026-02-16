import warnings

import pytest

from beamz.design import library as legacy_library
from beamz.design.materials import Material


def test_design_library_shim_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mats = legacy_library.list_materials()
    assert mats
    assert any(issubclass(w.category, DeprecationWarning) for w in rec)


def test_legacy_get_material_returns_material_for_migration():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mat = legacy_library.get_material("Gold")

    assert isinstance(mat, Material)
    eps, mu, sigma = mat.get_sample()
    assert eps is not None
    assert mu is not None
    assert sigma is not None


def test_legacy_get_material_handles_uniaxial_entry():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mat = legacy_library.get_material("LiNbO3")

    assert isinstance(mat, Material)


def test_symbolic_entries_require_allow_symbolic():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError):
            legacy_library.get_material("PEC")
        payload = legacy_library.get_material("PEC", allow_symbolic=True)

    assert payload["symbolic"] is True
