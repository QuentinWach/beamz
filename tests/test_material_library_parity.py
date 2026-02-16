import pytest

from beamz import Graphene, material_library
from beamz.material_library.material_library import (
    MaterialItem,
    MaterialItem2D,
    MaterialItemUniaxial,
    VariantItem,
)


def test_dict_style_access_contract_ag():
    ag = material_library["Ag"]
    assert isinstance(ag, MaterialItem)
    assert hasattr(ag, "variants")
    assert ag.default in ag.variants


def test_default_medium_and_variant_access_ag():
    ag = material_library["Ag"]
    default_medium = ag.medium
    variant_medium = ag["Rakic1998BB"]
    assert variant_medium is default_medium


def test_variant_item_required_fields_present():
    ag_variant = material_library["Ag"].variants["Rakic1998BB"]
    assert isinstance(ag_variant, VariantItem)
    assert ag_variant.medium is not None
    # Some variants have data urls / references, should be at least defined as attributes.
    assert hasattr(ag_variant, "reference")
    assert hasattr(ag_variant, "data_url")
    assert hasattr(ag_variant, "notes")
    assert hasattr(ag_variant, "frequency_range")


def test_uniaxial_axis_aware_contract():
    lno = material_library["LiNbO3"]
    assert isinstance(lno, MaterialItemUniaxial)

    anis_x = lno.medium("x")
    anis_z = lno.medium("z")

    assert anis_x.xx is not None
    assert anis_z.zz is not None


def test_representative_keys_and_variants():
    assert "Ag" in material_library
    assert "Au" in material_library
    assert "SiO2" in material_library
    assert "graphene" in material_library

    assert "Rakic1998BB" in material_library["Ag"].variants
    assert "Palik_Lossless" in material_library["SiO2"].variants


def test_graphene_entry_is_parametric_class():
    graphene_entry = material_library["graphene"]
    assert graphene_entry is Graphene


def test_2d_material_items_present_and_typed():
    mos2 = material_library["MoS2"]
    assert isinstance(mos2, MaterialItem2D)
    assert mos2.medium is not None
