from __future__ import annotations

from ._scene import CameraSpec, ClipPlaneSpec, MaterialSpec, Object3D, SceneSpec
from ._widget import view3d


def demo_scene() -> SceneSpec:
    objects = [
        Object3D(
            kind="box",
            label="Simulation Domain",
            geometry={"center": [5.0, 3.0, 0.45], "size": [10.0, 6.0, 0.9]},
            material=MaterialSpec(color="#0f172a", opacity=0.06, wireframe=True),
            metadata={"kind": "domain"},
        ),
        Object3D(
            kind="poly_extrusion",
            label="Silicon Bus",
            geometry={
                "vertices": [[0.8, 2.6], [9.2, 2.6], [9.2, 3.4], [0.8, 3.4]],
                "holes": [],
                "depth": 0.22,
                "z0": 0.22,
            },
            material=MaterialSpec(color="#2563eb", opacity=0.72),
            metadata={"kind": "structure", "material": {"permittivity": 12.1}},
        ),
        Object3D(
            kind="poly_extrusion",
            label="Ring Resonator",
            geometry={
                "vertices": [[5.2, 1.6], [6.8, 1.6], [6.8, 4.4], [5.2, 4.4]],
                "holes": [[[5.55, 1.95], [6.45, 1.95], [6.45, 4.05], [5.55, 4.05]]],
                "depth": 0.22,
                "z0": 0.22,
            },
            material=MaterialSpec(color="#0f766e", opacity=0.66),
            metadata={"kind": "structure", "material": {"permittivity": 4.0}},
        ),
        Object3D(
            kind="plane",
            label="Mode Source",
            geometry={
                "center": [1.2, 3.0, 0.33],
                "size": [1.2, 0.8],
                "normal": [1.0, 0.0, 0.0],
            },
            material=MaterialSpec(color="#f59e0b", opacity=0.32),
            metadata={
                "kind": "source",
                "type": "ModeSource",
                "direction": "+x",
                "wavelength": 1.55e-6,
                "polarization": "te",
            },
        ),
        Object3D(
            kind="arrow",
            label="Launch Direction",
            geometry={
                "origin": [1.2, 3.0, 0.33],
                "direction": [1.0, 0.0, 0.0],
                "length": 1.3,
            },
            material=MaterialSpec(color="#d97706", opacity=1.0),
            metadata={"kind": "source_direction"},
        ),
        Object3D(
            kind="plane",
            label="Flux Monitor",
            geometry={
                "center": [8.4, 3.0, 0.33],
                "size": [1.4, 1.2],
                "normal": [1.0, 0.0, 0.0],
            },
            material=MaterialSpec(color="#dc2626", opacity=0.24),
            metadata={"kind": "monitor", "type": "plane"},
        ),
        Object3D(
            kind="sphere",
            label="Gaussian Probe",
            geometry={"center": [6.0, 3.9, 0.33], "radius": 0.18},
            material=MaterialSpec(color="#ea580c", opacity=0.88, emissive="#7c2d12"),
            metadata={"kind": "source", "type": "GaussianSource", "width": 0.36},
        ),
    ]
    return SceneSpec(
        title="BEAMZ Scene Demo",
        units="um",
        background="#ffffff",
        camera=CameraSpec(
            position=(11.0, -3.5, 5.5), target=(5.0, 3.0, 0.33), fov=36.0
        ),
        clip_planes=[
            ClipPlaneSpec(normal=(1.0, 0.0, 0.0), constant=-5.0, enabled=False),
            ClipPlaneSpec(normal=(0.0, 1.0, 0.0), constant=-3.0, enabled=False),
            ClipPlaneSpec(normal=(0.0, 0.0, 1.0), constant=-0.33, enabled=False),
        ],
        objects=objects,
        metadata={"is_3d": True, "demo": True},
    )


def demo_widget():
    return view3d(demo_scene(), mode="widget")
