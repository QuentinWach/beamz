from beamz import material_library
from beamz.design.materials import Material, PECMaterial, PMCMaterial
from beamz.design.material_library import (
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
    assert isinstance(gold["Default"], Material)
    assert gold["Default"] is gold.medium


def test_variant_item_required_fields_present():
    sio2_variant = material_library["SiO2"].variants["Default"]
    assert isinstance(sio2_variant, VariantItem)
    assert sio2_variant.medium is not None
    assert hasattr(sio2_variant, "reference")
    assert hasattr(sio2_variant, "data_url")
    assert hasattr(sio2_variant, "notes")
    assert hasattr(sio2_variant, "frequency_range")


def test_uniaxial_axis_aware_contract_for_lino3():
    lno = material_library["LiNbO3"]
    assert isinstance(lno, MaterialItemUniaxial)

    anis_x = lno.medium("x")
    anis_z = lno.medium("z")

    assert anis_x.xx.permittivity != anis_x.yy.permittivity
    assert anis_z.zz.permittivity != anis_z.xx.permittivity


def test_symbolic_materials_are_present_and_typed():
    pec = material_library["PEC"].medium
    pmc = material_library["PMC"].medium

    assert isinstance(pec, PECMaterial)
    assert isinstance(pmc, PMCMaterial)
