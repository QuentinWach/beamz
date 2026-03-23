from types import SimpleNamespace

from beamz.visual.scene import beamz_to_scene


class FakeMaterial:
    def __init__(self, permittivity=3.47, permeability=1.0, conductivity=0.0):
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity


class FakeStructure:
    def __init__(self):
        self.vertices = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        self.interiors = []
        self.depth = 0.22
        self.z = 0.11
        self.color = "#2563eb"
        self.material = FakeMaterial()
        self.is_pml = False


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


def test_beamz_design_adapts_to_scene():
    design = SimpleNamespace(
        structures=[FakeStructure()],
        sources=[FakeGaussianSource(), FakeModeSource()],
        monitors=[FakeMonitor()],
        width=2.0,
        height=1.0,
        depth=0.22,
        is_3d=True,
    )

    scene = beamz_to_scene(design)

    assert scene.title == "BEAMZ Design"
    assert any(obj.metadata.get("kind") == "domain" for obj in scene.objects)
    assert any(obj.kind == "poly_extrusion" for obj in scene.objects)
    assert any(obj.metadata.get("kind") == "source" for obj in scene.objects)
    assert any(obj.metadata.get("kind") == "monitor" for obj in scene.objects)


def test_beamz_simulation_adapts_to_scene():
    design = SimpleNamespace(
        structures=[],
        sources=[],
        monitors=[],
        width=2.0,
        height=1.0,
        depth=0.22,
        is_3d=True,
    )
    simulation = SimpleNamespace(design=design, resolution=2.5e-8, is_3d=True)

    scene = beamz_to_scene(simulation)

    assert scene.title == "BEAMZ Simulation Setup"
    assert scene.metadata["resolution"] == 2.5e-8
    assert any(obj.metadata.get("kind") == "simulation" for obj in scene.objects)
