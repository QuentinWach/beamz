from __future__ import annotations

import html
import json
from pathlib import Path

from ._scene import SceneSpec


_STATIC_DIR = Path(__file__).parent / "static"


def _read_static_text(name: str) -> str:
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


def widget_esm() -> str:
    return f"{_read_static_text('viewer_core.js')}\n{_read_static_text('widget_wrapper.js')}"


def widget_css() -> str:
    return _read_static_text("widget.css")


def browser_html(scene: SceneSpec) -> str:
    scene_json = json.dumps(scene.to_dict(), ensure_ascii=False)
    title = html.escape(scene.title or "ZView")
    css = _read_static_text("widget.css")
    module_source = f"{_read_static_text('viewer_core.js')}\n{_read_static_text('browser_wrapper.js')}"
    return (
        _read_static_text("viewer.html").replace("__ZVIEW_TITLE__", title)
        .replace("__ZVIEW_CSS__", css)
        .replace("__ZVIEW_SCENE_JSON__", scene_json)
        .replace("__ZVIEW_MODULE_SOURCE__", module_source)
    )
