"""Interactive 3D scene viewing for BEAMZ designs and simulations."""

from ._beamz import (
    beamz_to_scene,
    design_to_scene,
    looks_like_beamz_design,
    looks_like_beamz_simulation,
    simulation_to_scene,
)
from ._browser import inline_iframe_html, inline_iframe_src, open_in_browser
from ._demo import demo_scene, demo_widget
from ._scene import (
    CameraSpec,
    ClipPlaneSpec,
    MaterialSpec,
    Object3D,
    SceneSpec,
    scene_from_dict,
)
from ._widget import SceneWidget, ZViewWidget, view3d

__all__ = [
    "CameraSpec",
    "ClipPlaneSpec",
    "MaterialSpec",
    "Object3D",
    "SceneSpec",
    "SceneWidget",
    "ZViewWidget",
    "beamz_to_scene",
    "demo_scene",
    "demo_widget",
    "design_to_scene",
    "inline_iframe_html",
    "inline_iframe_src",
    "looks_like_beamz_design",
    "looks_like_beamz_simulation",
    "open_in_browser",
    "scene_from_dict",
    "simulation_to_scene",
    "view3d",
]
