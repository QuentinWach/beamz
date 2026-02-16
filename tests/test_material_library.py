import pytest

from beamz.design.library import get_material, list_materials, material_info


def test_catalog_has_at_least_55_entries():
    assert len(list_materials()) >= 55


def test_required_named_materials_exist():
    required = {
        "Vacuum",
        "Air",
        "SiO2",
        "Si3N4",
        "BK7",
        "Sapphire",
        "Diamond",
        "Silicon",
        "Germanium",
        "GaAs",
        "InP",
        "LiNbO3",
        "Gold",
        "Silver",
        "Copper",
        "Aluminum",
        "Chromium",
        "Titanium",
        "PMMA",
        "SU8",
        "Polystyrene",
        "PDMS",
        "HSQ",
        "Water",
        "Ethanol",
        "IPA",
        "Glycerol",
        "ITO",
        "TiO2",
        "HfO2",
        "ZnO",
        "AlN",
        "MgF2",
        "CaF2",
        "ZnSe",
        "ZnS",
        "PEC",
        "PMC",
    }
    names = set(list_materials())
    assert required.issubset(names)


def test_case_insensitive_alias_lookup():
    sio2_a = get_material("sio2")
    sio2_b = get_material("SiO2")
    sio2_c = get_material("silica")

    assert sio2_a.permittivity == pytest.approx(sio2_b.permittivity)
    assert sio2_c.permittivity == pytest.approx(sio2_b.permittivity)


def test_list_materials_is_deterministic_and_sorted():
    first = list_materials()
    second = list_materials()
    assert first == second
    assert first == sorted(first, key=str.lower)


def test_category_filtering_and_symbolic_toggle():
    metals = list_materials(category="metals")
    assert "Gold" in metals
    assert "Air" not in metals

    no_symbolic = list_materials(include_symbolic=False)
    assert "PEC" not in no_symbolic
    assert "PMC" not in no_symbolic


def test_material_info_contains_metadata_fields():
    info = material_info("SiO2")
    assert info["canonical_name"] == "SiO2"
    assert info["source"]
    assert info["reference_wavelength_m"] == pytest.approx(1.55e-6)
    assert info["symbolic"] is False


def test_symbolic_pec_pmc_behavior():
    with pytest.raises(ValueError):
        get_material("PEC")

    pec_info = get_material("PEC", allow_symbolic=True)
    assert pec_info["symbolic"] is True
    assert pec_info["canonical_name"] == "PEC"

    pmc_info = material_info("pmc")
    assert pmc_info["symbolic"] is True
