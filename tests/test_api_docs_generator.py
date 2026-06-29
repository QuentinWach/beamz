import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_generator():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "update_api_docs.py"
    spec = importlib.util.spec_from_file_location("update_api_docs", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_api_docs_generator_writes_stable_public_reference(tmp_path, monkeypatch):
    generator = _load_generator()
    sentinel = object()

    def exported(value=sentinel):
        """Example public function."""

    fake_module = types.ModuleType("fake_beamz_api_docs")
    fake_module.__all__ = ["VALUE", "exported"]
    fake_module.VALUE = 1.25
    fake_module.exported = exported
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    generator.generate(tmp_path, module_names=(fake_module.__name__,))
    reference = (tmp_path / f"{fake_module.__name__}.md").read_text(encoding="utf-8")
    reference_json = json.loads((tmp_path / "api-reference.json").read_text(encoding="utf-8"))
    entries = reference_json["modules"][0]["entries"]

    assert f"::: {fake_module.__name__}" in reference
    assert [entry["name"] for entry in entries] == ["VALUE", "exported"]
    assert entries[1]["signature"] == "(value=<object>)"
    assert "0x" not in json.dumps(reference_json)
    assert generator.check(tmp_path, module_names=(fake_module.__name__,)) == 0


def test_api_docs_generator_includes_public_class_methods(tmp_path, monkeypatch):
    generator = _load_generator()

    class BaseClass:
        def inherited(self):
            """Inherited method."""

    class ExportedClass(BaseClass):
        """Example public class."""

        def run(self, value: int) -> str:
            """Run the example.

            Parameters
            ----------
            value
                Value to process.

            Returns
            -------
            str
                Processed value.
            """
            return str(value)

        def _private(self):
            """Do not include private methods."""

    fake_module = types.ModuleType("fake_beamz_class_docs")
    BaseClass.__module__ = fake_module.__name__
    BaseClass.__qualname__ = "BaseClass"
    ExportedClass.__module__ = fake_module.__name__
    ExportedClass.__qualname__ = "ExportedClass"
    fake_module.__all__ = ["BaseClass", "ExportedClass"]
    fake_module.BaseClass = BaseClass
    fake_module.ExportedClass = ExportedClass
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    generator.generate(tmp_path, module_names=(fake_module.__name__,))
    reference_json = json.loads((tmp_path / "api-reference.json").read_text(encoding="utf-8"))
    base_entry = reference_json["modules"][0]["entries"][0]
    class_entry = reference_json["modules"][0]["entries"][1]

    assert class_entry["name"] == "ExportedClass"
    assert class_entry["bases"] == [
        {
            "external": True,
            "import_path": f"from {fake_module.__name__} import BaseClass",
            "module": fake_module.__name__,
            "name": "BaseClass",
            "qualified_name": f"{fake_module.__name__}.BaseClass",
            "reference": {
                "anchor": base_entry["anchor"],
                "module": fake_module.__name__,
                "name": "BaseClass",
                "qualified_name": f"{fake_module.__name__}.BaseClass",
            },
        }
    ]
    assert [member["name"] for member in class_entry["members"]] == ["run"]
    assert class_entry["members"][0]["kind"] == "method"
    assert class_entry["members"][0]["signature"] == "(value: int) -> str"
    assert class_entry["members"][0]["defined_in"] == f"{fake_module.__name__}.ExportedClass"
    assert class_entry["members"][0]["inherited"] is False


def test_api_docs_generator_prefers_specific_base_reference():
    generator = _load_generator()

    def class_entry(module_name, name, canonical_path, anchor, bases=None):
        return generator.ApiEntry(
            name=name,
            qualified_name=f"{module_name}.{name}",
            canonical_path=canonical_path,
            anchor=anchor,
            kind="class",
            type_name="type",
            signature=None,
            import_path=f"from {module_name} import {name}",
            source=generator.ApiSource(
                module=canonical_path.rsplit(".", 1)[0],
                file=None,
                line=None,
            ),
            value_repr=None,
            summary=None,
            description=None,
            docstring=None,
            bases=bases or [],
        )

    polygon_root = class_entry(
        "beamz",
        "Polygon",
        "beamz.design.structures.Polygon",
        "api-root-polygon",
    )
    rectangle_root = class_entry(
        "beamz",
        "Rectangle",
        "beamz.design.structures.Rectangle",
        "api-root-rectangle",
        bases=[
            generator.ApiBaseClass(
                name="Polygon",
                qualified_name="beamz.design.structures.Polygon",
                module="beamz.design.structures",
                import_path="from beamz.design.structures import Polygon",
                external=False,
            )
        ],
    )
    polygon_design = class_entry(
        "beamz.design",
        "Polygon",
        "beamz.design.structures.Polygon",
        "api-design-polygon",
    )
    reference = generator.ApiReference(
        schema_version=4,
        generated_by="test",
        modules=[
            generator.ApiModuleReference(
                name="beamz",
                title="beamz API Reference",
                summary="",
                entries=[polygon_root, rectangle_root],
            ),
            generator.ApiModuleReference(
                name="beamz.design",
                title="beamz.design API Reference",
                summary="",
                entries=[polygon_design],
            ),
        ],
    )

    resolved = generator._resolve_reference_targets(reference)
    resolved_rectangle = resolved.modules[0].entries[1]

    assert resolved_rectangle.bases[0].reference == generator.ApiReferenceTarget(
        module="beamz.design",
        anchor="api-design-polygon",
        name="Polygon",
        qualified_name="beamz.design.structures.Polygon",
    )


def test_api_docs_check_detects_stale_reference(tmp_path, monkeypatch):
    generator = _load_generator()

    fake_module = types.ModuleType("fake_beamz_stale_docs")
    fake_module.__all__ = ["VALUE"]
    fake_module.VALUE = 1.25
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    generator.generate(tmp_path, module_names=(fake_module.__name__,))
    (tmp_path / f"{fake_module.__name__}.md").write_text("# stale\n", encoding="utf-8")

    assert generator.check(tmp_path, module_names=(fake_module.__name__,)) == 1


def test_docs_check_ignores_untracked_reference_json(tmp_path, monkeypatch):
    generator = _load_generator()

    fake_module = types.ModuleType("fake_beamz_docs_json_artifact")
    fake_module.__all__ = ["VALUE"]
    fake_module.VALUE = 1.25
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    generator.generate_docs(tmp_path, module_names=(fake_module.__name__,))
    (tmp_path / "api" / "reference" / "api-reference.json").unlink()

    assert generator.check_docs(tmp_path, module_names=(fake_module.__name__,)) == 0
