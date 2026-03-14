from __future__ import annotations

from typing import Any

from traitlets import Dict, List, Unicode

from ._beamz import beamz_to_scene, looks_like_beamz_design, looks_like_beamz_simulation
from ._browser import inline_iframe_html, open_in_browser
from ._frontend import widget_css, widget_esm
from ._scene import SceneSpec, scene_from_dict

try:
    import anywidget
except (
    ImportError
):  # pragma: no cover - exercised indirectly when widget extra is missing
    anywidget = None

_ESM = widget_esm()
_CSS = widget_css()


if anywidget is not None:
    _WidgetBase = anywidget.AnyWidget
else:  # pragma: no cover - import fallback only
    from traitlets import HasTraits

    class _WidgetBase(HasTraits):
        pass


class SceneWidget(_WidgetBase):
    _esm = _ESM
    _css = _CSS

    scene_json = Dict(default_value={}).tag(sync=True)
    clip_planes = List(default_value=[]).tag(sync=True)
    selected_object_id = Unicode(allow_none=True, default_value=None).tag(sync=True)
    hovered_object_id = Unicode(allow_none=True, default_value=None).tag(sync=True)

    def __init__(self, scene: SceneSpec | dict[str, Any], **kwargs: Any) -> None:
        if anywidget is None:
            raise RuntimeError(
                "Rendering the BEAMZ scene widget requires `anywidget`. "
                "Reinstall BeamZ or install `anywidget` in this environment."
            )
        scene_spec = scene if isinstance(scene, SceneSpec) else scene_from_dict(scene)
        scene_json = scene_spec.to_dict()
        super().__init__(
            scene_json=scene_json,
            clip_planes=scene_json.get("clip_planes", []),
            **kwargs,
        )


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def _coerce_scene(value: SceneSpec | dict[str, Any] | Any) -> SceneSpec:
    if isinstance(value, SceneSpec):
        return value
    if isinstance(value, dict):
        return scene_from_dict(value)
    if looks_like_beamz_design(value) or looks_like_beamz_simulation(value):
        return beamz_to_scene(value)
    raise TypeError(
        "view3d() expects a SceneSpec, scene dictionary, or BeamZ-like design/simulation object."
    )


def view3d(
    value: SceneSpec | dict[str, Any] | Any,
    *,
    mode: str = "auto",
    open_browser: bool = True,
    **kwargs: Any,
) -> Any:
    scene = _coerce_scene(value)
    chosen_mode = mode
    if chosen_mode == "auto":
        chosen_mode = "inline" if _in_notebook() else "browser"
    if chosen_mode == "inline":
        try:
            from IPython.display import HTML
        except ImportError as exc:
            raise RuntimeError("mode='inline' requires IPython.") from exc
        return HTML(inline_iframe_html(scene))
    if chosen_mode == "browser":
        return open_in_browser(scene, open_browser=open_browser)
    if chosen_mode != "widget":
        raise ValueError("mode must be one of: 'auto', 'inline', 'widget', 'browser'")
    return SceneWidget(scene=scene, **kwargs)


ZViewWidget = SceneWidget
