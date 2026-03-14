from __future__ import annotations

import html
import json
from pathlib import Path

from ._scene import SceneSpec


_STATIC_DIR = Path(__file__).parent / "static"
_VIEWER_CORE = (_STATIC_DIR / "viewer_core.js").read_text(encoding="utf-8")
_WIDGET_WRAPPER = (_STATIC_DIR / "widget_wrapper.js").read_text(encoding="utf-8")
_BROWSER_WRAPPER = (_STATIC_DIR / "browser_wrapper.js").read_text(encoding="utf-8")
_HTML_TEMPLATE = (_STATIC_DIR / "viewer.html").read_text(encoding="utf-8")
_CSS = (_STATIC_DIR / "widget.css").read_text(encoding="utf-8")


def widget_esm() -> str:
    return f"{_VIEWER_CORE}\n{_WIDGET_WRAPPER}"


def widget_css() -> str:
    return _CSS


def browser_html(scene: SceneSpec) -> str:
    scene_json = json.dumps(scene.to_dict(), ensure_ascii=False)
    title = html.escape(scene.title or "ZView")
    module_source = f"{_VIEWER_CORE}\n{_BROWSER_WRAPPER}"
    return (
        _HTML_TEMPLATE.replace("__ZVIEW_TITLE__", title)
        .replace("__ZVIEW_CSS__", _CSS)
        .replace("__ZVIEW_SCENE_JSON__", scene_json)
        .replace("__ZVIEW_MODULE_SOURCE__", module_source)
    )
