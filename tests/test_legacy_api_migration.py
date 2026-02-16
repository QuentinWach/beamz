import warnings

import pytest

from beamz.design import library as legacy_library
from beamz.design.materials import DrudeMaterial


def test_design_library_shim_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mats = legacy_library.list_materials()
    assert mats
    assert any(issubclass(w.category, DeprecationWarning) for w in rec)


def test_legacy_get_material_still_returns_material_for_migration():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mat = legacy_library.get_material("Ag")
    eps, mu, sigma = mat.get_sample()
    assert eps is not None
    assert mu is not None
    assert sigma is not None


def test_symbolic_entries_require_allow_symbolic():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError):
            legacy_library.get_material("PEC")
        payload = legacy_library.get_material("PEC", allow_symbolic=True)
    assert payload["symbolic"] is True


def test_legacy_dispersive_classes_warn_on_init():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        DrudeMaterial(name="legacy", eps_inf=1.0, plasma_frequency=1e16, damping=1e14)
    assert any(issubclass(w.category, DeprecationWarning) for w in rec)
