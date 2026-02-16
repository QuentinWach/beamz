from beamz import material_library
from beamz.design.materials import (
    DebyeMaterial,
    DrudeMaterial,
    LorentzMaterial,
    PECMaterial,
    PMCMaterial,
    SellmeierMaterial,
)
from beamz.design.library import (
    MaterialItem,
    MaterialItemUniaxial,
    VariantItem,
)


EXPECTED_KEYS = {
    "Vacuum",
    "Air",
    "SiO2",
    "Si3N4",
    "Silicon",
    "LiNbO3",
    "Gold",
    "Aluminum",
    "Copper",
    "Water",
    "ITO",
    "PEC",
    "PMC",
}


def test_curated_keys_exact_match():
    assert set(material_library.keys()) == EXPECTED_KEYS


def test_bulk_items_have_single_default_variant():
    for key in EXPECTED_KEYS - {"LiNbO3"}:
        item = material_library[key]
        assert isinstance(item, MaterialItem)
        assert item.default == "Default"
        assert set(item.variants.keys()) == {"Default"}


def test_default_medium_and_variant_access_contract():
    gold = material_library["Gold"]
    assert isinstance(gold["Default"], DrudeMaterial)
    assert gold["Default"] is gold.medium


def test_variant_item_required_fields_present():
    sio2_variant = material_library["SiO2"].variants["Default"]
    assert isinstance(sio2_variant, VariantItem)
    assert isinstance(sio2_variant.medium, SellmeierMaterial)
    assert hasattr(sio2_variant, "reference")
    assert hasattr(sio2_variant, "data_url")
    assert hasattr(sio2_variant, "notes")
    assert hasattr(sio2_variant, "frequency_range")


def test_uniaxial_axis_aware_contract_for_lino3():
    lno = material_library["LiNbO3"]
    assert isinstance(lno, MaterialItemUniaxial)

    anis_x = lno.medium("x")
    anis_z = lno.medium("z")

    assert isinstance(anis_x.xx, SellmeierMaterial)
    assert isinstance(anis_z.zz, SellmeierMaterial)
    eps_xx = anis_x.xx.to_material(wavelength=1.55e-6).permittivity
    eps_yy = anis_x.yy.to_material(wavelength=1.55e-6).permittivity
    eps_zz = anis_z.zz.to_material(wavelength=1.55e-6).permittivity
    eps_zx = anis_z.xx.to_material(wavelength=1.55e-6).permittivity
    assert eps_xx != eps_yy
    assert eps_zz != eps_zx


def test_symbolic_materials_are_present_and_typed():
    pec = material_library["PEC"].medium
    pmc = material_library["PMC"].medium

    assert isinstance(pec, PECMaterial)
    assert isinstance(pmc, PMCMaterial)


def test_expected_model_classes_by_key():
    assert isinstance(material_library["Si3N4"].medium, SellmeierMaterial)
    assert isinstance(material_library["Silicon"].medium, LorentzMaterial)
    assert isinstance(material_library["Water"].medium, DebyeMaterial)
    assert isinstance(material_library["ITO"].medium, DrudeMaterial)
