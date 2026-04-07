from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from beamz.visual.scene import view3d


class Material:
    def __init__(self, permittivity: float) -> None:
        self.permittivity = permittivity
        self.permeability = 1.0
        self.conductivity = 0.0


class Structure:
    def __init__(self) -> None:
        self.vertices = [
            (0.0, 1.2, 0.22),
            (8.0, 1.2, 0.22),
            (8.0, 1.8, 0.22),
            (0.0, 1.8, 0.22),
        ]
        self.interiors = []
        self.depth = 0.22
        self.z = 0.22
        self.color = "#2563eb"
        self.material = Material(12.1)
        self.is_pml = False


class ModeSource:
    def __init__(self) -> None:
        self.center = (0.9, 1.5, 0.33)
        self.width = 0.8
        self.height = 0.6
        self.direction = "+x"
        self.wavelength = 1.55e-6
        self.pol = "te"


class Monitor:
    def __init__(self) -> None:
        self.name = "output_flux"
        self.is_3d = True
        self.start = (7.2, 1.1, 0.33)
        self.end = (7.2, 1.9, 0.33)
        self.monitor_type = "plane"
        self.plane_normal = "x"


def make_design() -> SimpleNamespace:
    return SimpleNamespace(
        structures=[Structure()],
        sources=[ModeSource()],
        monitors=[Monitor()],
        width=8.0,
        height=3.0,
        depth=0.8,
        is_3d=True,
    )


def main() -> None:
    result = view3d(make_design())
    if isinstance(result, str):
        print("BeamZ-like demo opened in browser.")
        print(f"url: {result}")
        return
    print("BeamZ-like demo widget created.")
    print(f"scene title: {result.scene_json.get('title')}")
    print(f"object count: {len(result.scene_json.get('objects', []))}")
    try:
        from IPython.display import display
    except ImportError:
        return
    display(result)


if __name__ == "__main__":
    main()
