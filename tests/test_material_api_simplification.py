import beamz


def test_material_first_public_symbols_exist():
    expected = {
        "Material",
        "CustomMaterial",
        "PoleResidueMaterial",
        "SellmeierMaterial",
        "DrudeMaterial",
        "LorentzMaterial",
        "DebyeMaterial",
        "Material2D",
        "AnisotropicMaterial",
    }
    for symbol in expected:
        assert hasattr(beamz, symbol)


def test_medium_named_public_symbols_are_removed():
    removed = {
        "Medium",
        "PECMedium",
        "PMCMedium",
        "PECMaterial",
        "PMCMaterial",
        "PEC",
        "PMC",
        "PoleResidue",
        "Sellmeier",
        "Drude",
        "Lorentz",
        "Debye",
    }
    for symbol in removed:
        assert not hasattr(beamz, symbol)


def test_design_method_surface_is_stable():
    design = beamz.Design()
    methods = {
        "add",
        "rasterize",
        "get_material_grids",
        "get_thermal_grids",
        "solve_thermal",
        "solve_static_thermal",
        "sweep_mzi_heater",
        "copy",
        "show",
        "unify_polygons",
        "get_material_value",
    }
    for method_name in methods:
        assert hasattr(design, method_name)
