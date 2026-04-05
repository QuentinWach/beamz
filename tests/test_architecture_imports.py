import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SPEC_MODULES = [
    "beamz/design/materials.py",
    "beamz/design/spec.py",
    "beamz/design/structure_specs.py",
    "beamz/devices/sources/spec.py",
    "beamz/devices/monitors/spec.py",
    "beamz/simulation/boundary_specs.py",
    "beamz/simulation/spec.py",
]

FORBIDDEN_PREFIXES = (
    "beamz.visual",
    "beamz.simulation.build",
    "beamz.simulation.compiled",
    "beamz.simulation.fields",
    "beamz.simulation.jit",
    "beamz.simulation.loop",
    "beamz.simulation.ops",
    "beamz.simulation.runtime",
    "beamz.simulation.session",
    "beamz.simulation.step",
    "beamz.simulation.view",
    "beamz.devices.sources.apply",
    "beamz.devices.sources.compiler",
    "beamz.devices.sources.inject",
    "beamz.devices.sources.setup",
    "beamz.devices.sources.state",
    "beamz.devices.monitors.compiler",
    "beamz.devices.monitors.live",
    "beamz.devices.monitors.record",
    "beamz.devices.monitors.state",
    "beamz.devices.monitors.store",
)


def _imported_modules(rel_path: str) -> set[str]:
    tree = ast.parse((ROOT / rel_path).read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_spec_modules_do_not_import_visual_or_runtime_helpers():
    violations = {}
    for rel_path in SPEC_MODULES:
        imported = _imported_modules(rel_path)
        bad = sorted(
            module
            for module in imported
            if module.startswith(FORBIDDEN_PREFIXES)
        )
        if bad:
            violations[rel_path] = bad

    assert violations == {}
