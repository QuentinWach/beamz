from types import SimpleNamespace

from beamz.simulation.boundaries import PML
from beamz.simulation.core import Simulation
from beamz.visual.scene import (
    CameraSpec,
    ClipPlaneSpec,
    MaterialSpec,
    Object3D,
    SceneSpec,
    demo_scene,
    inline_iframe_html,
    inline_iframe_src,
    scene_from_dict,
    simulation_to_scene,
    view3d,
)
from beamz.visual.scene._browser import open_in_browser


class FakeMaterial:
    def __init__(self, permittivity=3.47, permeability=1.0, conductivity=0.0):
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity


class FakeStructure:
    def __init__(self):
        self.vertices = [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        self.interiors = []
        self.depth = 0.22
        self.z = 0.11
        self.color = "#2563eb"
        self.material = FakeMaterial()
        self.is_pml = False


class FakeStructureTwin(FakeStructure):
    def __init__(self, x_offset=0.0, color="#16a34a"):
        super().__init__()
        self.vertices = [
            (0.0 + x_offset, 0.0, 0.0),
            (2.0 + x_offset, 0.0, 0.0),
            (2.0 + x_offset, 1.0, 0.0),
            (0.0 + x_offset, 1.0, 0.0),
        ]
        self.color = color


class FakeGaussianSource:
    def __init__(self):
        self.position = (0.5, 0.5, 0.11)
        self.width = 0.3


class FakeModeSource:
    def __init__(self):
        self.center = (1.0, 0.5, 0.11)
        self.width = 0.8
        self.height = 0.22
        self.direction = "+x"
        self.wavelength = 1.55e-6
        self.pol = "te"


class FakeMonitor:
    def __init__(self):
        self.name = "flux"
        self.is_3d = True
        self.start = (0.2, 0.1, 0.11)
        self.end = (1.8, 0.9, 0.11)
        self.monitor_type = "plane"
        self.plane_normal = "z"


def _make_design():
    source = FakeModeSource()
    monitor = FakeMonitor()
    return SimpleNamespace(
        structures=[FakeStructure()],
        sources=[source],
        monitors=[monitor],
        width=2.0,
        height=1.0,
        depth=0.22,
        is_3d=True,
    )


def _make_design_with_repeated_material():
    source = FakeModeSource()
    monitor = FakeMonitor()
    return SimpleNamespace(
        structures=[
            FakeStructureTwin(x_offset=0.0, color="#2563eb"),
            FakeStructureTwin(x_offset=2.5, color="#f97316"),
        ],
        sources=[source],
        monitors=[monitor],
        width=5.0,
        height=1.0,
        depth=0.22,
        is_3d=True,
    )


def _make_design_with_overlapping_material():
    source = FakeModeSource()
    monitor = FakeMonitor()
    return SimpleNamespace(
        structures=[
            FakeStructureTwin(x_offset=0.0, color="#2563eb"),
            FakeStructureTwin(x_offset=1.0, color="#f97316"),
        ],
        sources=[source],
        monitors=[monitor],
        width=4.0,
        height=1.0,
        depth=0.22,
        is_3d=True,
    )


def _make_simulation():
    design = _make_design()
    extra_source = FakeGaussianSource()
    sim = Simulation.__new__(Simulation)
    sim.design = design
    sim.devices = [design.sources[0], design.monitors[0], extra_source]
    sim.boundaries = [PML(edges=["left", "right"], thickness=0.15)]
    sim.resolution = 2.5e-8
    sim.is_3d = True
    sim.plane_2d = "xy"
    sim.dt = 1e-16
    sim.num_steps = 64
    return sim


def test_scene_round_trip_preserves_structure():
    scene = SceneSpec(
        title="Example",
        objects=[
            Object3D(
                kind="box",
                label="Domain",
                geometry={"center": [0.0, 0.0, 0.0], "size": [1.0, 2.0, 3.0]},
                material=MaterialSpec(color="#123456", opacity=0.2, wireframe=True),
            )
        ],
        clip_planes=[
            ClipPlaneSpec(normal=(0.0, 0.0, 1.0), constant=0.0, enabled=False)
        ],
        camera=CameraSpec(position=(3.0, 2.0, 1.0), target=(0.0, 0.0, 0.0)),
    )

    payload = scene.to_dict()
    restored = scene_from_dict(payload)

    assert restored.title == "Example"
    assert restored.objects[0].kind == "box"
    assert restored.objects[0].material.wireframe is True
    assert restored.camera.position == (3.0, 2.0, 1.0)
    assert restored.clip_planes[0].enabled is False


def test_open_in_browser_writes_html_without_launching():
    url = open_in_browser(demo_scene(), open_browser=False)
    assert url.startswith("file://")


def test_inline_iframe_html_contains_iframe():
    html = inline_iframe_html(demo_scene())
    assert "<iframe" in html
    assert 'src="data:text/html;base64,' in html
    assert "Standalone browser rendering" not in html


def test_inline_iframe_src_uses_data_url():
    src = inline_iframe_src(demo_scene())
    assert src.startswith("data:text/html;base64,")


def test_simulation_to_scene_includes_devices_boundaries_and_metadata():
    scene = simulation_to_scene(_make_simulation())

    boundary_objects = [
        obj for obj in scene.objects if obj.metadata.get("kind") == "boundary"
    ]
    source_objects = [
        obj for obj in scene.objects if obj.metadata.get("kind") == "source"
    ]

    assert scene.title == "BEAMZ Simulation Setup"
    assert scene.metadata["resolution"] == 2.5e-8
    assert scene.metadata["num_devices"] == 3
    assert len(boundary_objects) == 2
    assert any(
        obj.metadata.get("type") == "FakeGaussianSource" for obj in source_objects
    )


def test_simulation_show_delegates_to_view3d(monkeypatch):
    sim = _make_simulation()
    captured = {}

    def fake_view3d(value, **kwargs):
        captured["value"] = value
        captured["kwargs"] = kwargs
        return "scene-view"

    monkeypatch.setattr("beamz.visual.scene.view3d", fake_view3d)

    result = sim.show(mode="browser", open_browser=False)

    assert result == "scene-view"
    assert isinstance(captured["value"], SceneSpec)
    assert captured["kwargs"] == {"mode": "browser", "open_browser": False}


def test_design_to_scene_merges_adjacent_structures_with_same_material():
    scene = simulation_to_scene(
        SimpleNamespace(
            design=_make_design_with_repeated_material(),
            devices=[],
            boundaries=[],
            resolution=2.5e-8,
            is_3d=True,
            plane_2d="xy",
            dt=1e-16,
            num_steps=8,
        )
    )

    structure_objects = [
        obj for obj in scene.objects if obj.metadata.get("kind") == "structure"
    ]

    assert len(structure_objects) == 1
    assert structure_objects[0].geometry["items"]
    assert len(structure_objects[0].geometry["items"]) == 2
    assert structure_objects[0].material.color == "#2563eb"
    assert structure_objects[0].metadata["structure_count"] == 2


def test_scene_objects_get_stable_display_order_metadata():
    scene = simulation_to_scene(_make_simulation())
    display_orders = [obj.metadata["display_order"] for obj in scene.objects]
    assert display_orders == list(range(len(scene.objects)))


def test_design_to_scene_unions_overlapping_same_material_structures():
    scene = simulation_to_scene(
        SimpleNamespace(
            design=_make_design_with_overlapping_material(),
            devices=[],
            boundaries=[],
            resolution=2.5e-8,
            is_3d=True,
            plane_2d="xy",
            dt=1e-16,
            num_steps=8,
        )
    )

    structure_objects = [
        obj for obj in scene.objects if obj.metadata.get("kind") == "structure"
    ]

    assert len(structure_objects) == 1
    assert "items" not in structure_objects[0].geometry
    assert structure_objects[0].kind == "poly_extrusion"
    assert structure_objects[0].material.color == "#2563eb"


def test_view3d_inline_returns_ipython_html():
    result = view3d(demo_scene(), mode="inline")
    assert result.__class__.__name__ == "IFrame"
    assert str(result.src).startswith("data:text/html;base64,")
