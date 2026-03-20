from __future__ import annotations

import html
import json
from pathlib import Path

from ._scene import SceneSpec


_STATIC_DIR = Path(__file__).parent / "static"
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


def _read_static_text(name: str) -> str:
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


def widget_esm() -> str:
    return f"{_read_static_text('viewer_core.js')}\n{_read_static_text('widget_wrapper.js')}"


def widget_css() -> str:
    return _read_static_text("widget.css")


def _viewer_html(scene: SceneSpec, template: str) -> str:
    scene_json = json.dumps(scene.to_dict(), ensure_ascii=False)
    title = html.escape(scene.title or "BEAMZ Scene")
    css = _read_static_text("widget.css")
    module_source = f"{_read_static_text('viewer_core.js')}\n{_read_static_text('browser_wrapper.js')}"
    return (
        template.replace("__ZVIEW_TITLE__", title)
        .replace("__ZVIEW_CSS__", css)
        .replace("__ZVIEW_SCENE_JSON__", scene_json)
        .replace("__ZVIEW_MODULE_SOURCE__", module_source)
    )


def browser_html(scene: SceneSpec) -> str:
    return _viewer_html(scene, _read_static_text("viewer.html"))


def inline_html(scene: SceneSpec) -> str:
    return _viewer_html(scene, _INLINE_HTML_TEMPLATE)
