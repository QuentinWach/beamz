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
_INLINE_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>__ZVIEW_TITLE__</title>
    <style>
      html, body {
        margin: 0;
        min-height: 100%;
        background: transparent;
      }
__ZVIEW_CSS__
      #zview-root {
        width: 100%;
        min-height: 480px;
      }
    </style>
  </head>
  <body>
    <div id="zview-root"></div>
    <script>
      window.__ZVIEW_SCENE__ = __ZVIEW_SCENE_JSON__;
    </script>
    <script type="module">
__ZVIEW_MODULE_SOURCE__
    </script>
  </body>
</html>
"""


def widget_esm() -> str:
    return f"{_VIEWER_CORE}\n{_WIDGET_WRAPPER}"


def widget_css() -> str:
    return _CSS


def _viewer_html(scene: SceneSpec, template: str) -> str:
    scene_json = json.dumps(scene.to_dict(), ensure_ascii=False)
    title = html.escape(scene.title or "BEAMZ Scene")
    module_source = f"{_VIEWER_CORE}\n{_BROWSER_WRAPPER}"
    return (
        template.replace("__ZVIEW_TITLE__", title)
        .replace("__ZVIEW_CSS__", _CSS)
        .replace("__ZVIEW_SCENE_JSON__", scene_json)
        .replace("__ZVIEW_MODULE_SOURCE__", module_source)
    )


def browser_html(scene: SceneSpec) -> str:
    return _viewer_html(scene, _HTML_TEMPLATE)


def inline_html(scene: SceneSpec) -> str:
    return _viewer_html(scene, _INLINE_HTML_TEMPLATE)
