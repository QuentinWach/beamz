from types import SimpleNamespace

from beamz.const import BLUE, RED
from beamz.design.structures import Sphere
from beamz.devices.monitors.monitors import Monitor
from beamz.simulation.boundaries import PML
from beamz.simulation.core import Simulation
from beamz.visual.scene import (
    CameraSpec,
    ClipPlaneSpec,
    MaterialSpec,
    Object3D,
    SceneSpec,
    _frontend,
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


class FakeAirStructure(FakeStructureTwin):
    def __init__(self, x_offset=0.0):
        super().__init__(x_offset=x_offset, color="#ffffff")
        self.material = FakeMaterial(permittivity=1.0)


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


class FakeMonitorX:
    def __init__(self):
        self.name = "flux_yz"
        self.is_3d = True
        self.start = (0.2, 0.1, 0.05)
        self.end = (0.2, 0.9, 0.45)
        self.monitor_type = "plane"
        self.plane_normal = "x"


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


def _make_design_with_two_materials_and_air():
    source = FakeModeSource()
    monitor = FakeMonitor()
    air = FakeAirStructure(x_offset=4.5)
    return SimpleNamespace(
        structures=[
            FakeStructureTwin(x_offset=0.0),
            FakeStructureTwin(x_offset=2.5),
            air,
        ],
        sources=[source],
        monitors=[monitor],
        width=7.0,
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


def test_simulation_to_scene_preserves_monitor_extents_for_x_normal_planes():
    design = _make_design()
    design.monitors = [FakeMonitorX()]
    sim = Simulation.__new__(Simulation)
    sim.design = design
    sim.devices = [design.monitors[0]]
    sim.boundaries = []
    sim.resolution = 2.5e-8
    sim.is_3d = True
    sim.plane_2d = "xy"
    sim.dt = 1e-16
    sim.num_steps = 8

    scene = simulation_to_scene(sim)

    monitor_object = next(
        obj for obj in scene.objects if obj.metadata.get("kind") == "monitor"
    )

    assert monitor_object.geometry["size"] == [0.8, 0.4]
    assert monitor_object.geometry["normal"] == [1.0, 0.0, 0.0]


def test_simulation_to_scene_preserves_monitor_center_for_legacy_plane_monitors():
    design = _make_design()
    legacy_monitor = Monitor(
        design=design,
        start=(0.4, 0.2, 0.11),
        end=None,
        plane_normal="z",
        size=(0.8, 0.4),
        name="legacy_flux",
    )
    design.monitors = [legacy_monitor]
    sim = Simulation.__new__(Simulation)
    sim.design = design
    sim.devices = [legacy_monitor]
    sim.boundaries = []
    sim.resolution = 2.5e-8
    sim.is_3d = True
    sim.plane_2d = "xy"
    sim.dt = 1e-16
    sim.num_steps = 8

    scene = simulation_to_scene(sim)

    monitor_object = next(
        obj for obj in scene.objects if obj.metadata.get("kind") == "monitor"
    )

    assert monitor_object.geometry["center"] == [0.8, 0.4, 0.11]
    assert monitor_object.geometry["size"] == [0.8, 0.4]
    assert monitor_object.geometry["normal"] == [0.0, 0.0, 1.0]


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


def test_mode_source_visualization_does_not_emit_direction_arrow():
    design = _make_design()
    design.sources[0].wavelength = 99.0
    sim = Simulation.__new__(Simulation)
    sim.design = design
    sim.devices = [design.sources[0]]
    sim.boundaries = []
    sim.resolution = 2.5e-8
    sim.is_3d = True
    sim.plane_2d = "xy"
    sim.dt = 1e-16
    sim.num_steps = 8

    scene = simulation_to_scene(sim)

    assert not any(
        obj.metadata.get("kind") == "source_direction" for obj in scene.objects
    )


def test_design_to_scene_keeps_adjacent_same_material_structures_separate():
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

    assert len(structure_objects) == 2
    assert [obj.material.color for obj in structure_objects] == [BLUE, BLUE]
    assert all("items" not in obj.geometry for obj in structure_objects)
    assert all("structure_count" not in obj.metadata for obj in structure_objects)


def test_design_to_scene_includes_sphere_structures():
    scene = simulation_to_scene(
        SimpleNamespace(
            design=SimpleNamespace(
                structures=[Sphere(position=(1.0, 0.5, 0.3), radius=0.2)],
                sources=[],
                monitors=[],
                width=2.0,
                height=1.0,
                depth=0.6,
                is_3d=True,
            ),
            devices=[],
            boundaries=[],
            resolution=2.5e-8,
            is_3d=True,
            plane_2d="xy",
            dt=1e-16,
            num_steps=8,
        )
    )

    sphere_object = next(
        obj for obj in scene.objects if obj.metadata.get("type") == "Sphere"
    )

    assert sphere_object.kind == "sphere"
    assert sphere_object.geometry["center"] == [1.0, 0.5, 0.3]
    assert sphere_object.geometry["radius"] == 0.2


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
    assert structure_objects[0].material.color == BLUE


def test_structure_colors_are_deterministic_and_only_air_is_transparent():
    scene = simulation_to_scene(
        SimpleNamespace(
            design=_make_design_with_two_materials_and_air(),
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

    assert [obj.material.color for obj in structure_objects] == [BLUE, BLUE, RED]
    assert [obj.material.opacity for obj in structure_objects] == [1.0, 1.0, 0.0]


def test_frontend_html_reads_static_assets_lazily(monkeypatch):
    assets = {
        "viewer_core.js": "console.log('fresh-viewer-core');",
        "browser_wrapper.js": "console.log('fresh-browser-wrapper');",
        "widget_wrapper.js": "console.log('fresh-widget-wrapper');",
        "widget.css": ".zview-root { outline: 1px solid red; }",
        "viewer.html": (
            "<html><head><title>__ZVIEW_TITLE__</title><style>__ZVIEW_CSS__</style></head>"
            "<body><div id='scene'>__ZVIEW_SCENE_JSON__</div><script>__ZVIEW_MODULE_SOURCE__</script></body></html>"
        ),
    }

    monkeypatch.setattr(_frontend, "_read_static_text", lambda name: assets[name])

    html = _frontend.browser_html(demo_scene())

    assert "fresh-viewer-core" in html
    assert "fresh-browser-wrapper" in html
    assert "outline: 1px solid red" in html


def test_view3d_inline_returns_ipython_html():
    result = view3d(demo_scene(), mode="inline")
    assert result.__class__.__name__ == "IFrame"
    assert str(result.src).startswith("data:text/html;base64,")
