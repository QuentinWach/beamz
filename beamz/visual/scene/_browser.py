from __future__ import annotations

import html
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

from ._frontend import browser_html
from ._scene import SceneSpec, scene_from_dict


def open_in_browser(
    scene: SceneSpec | dict[str, Any], *, open_browser: bool = True
) -> str:
    scene_spec = scene if isinstance(scene, SceneSpec) else scene_from_dict(scene)
    html_text = browser_html(scene_spec)
    tmp_dir = Path(tempfile.mkdtemp(prefix="zview-"))
    html_path = tmp_dir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")
    url = html_path.resolve().as_uri()
    if open_browser:
        webbrowser.open(url)
    return url


def inline_iframe_html(
    scene: SceneSpec | dict[str, Any],
    *,
    height: int = 640,
) -> str:
    scene_spec = scene if isinstance(scene, SceneSpec) else scene_from_dict(scene)
    srcdoc = html.escape(browser_html(scene_spec), quote=True)
    return (
        f'<iframe srcdoc="{srcdoc}" '
        'style="width: 100%; min-height: 480px; '
        f'height: {int(height)}px; border: 0; border-radius: 16px;" '
        'sandbox="allow-scripts allow-same-origin"></iframe>'
    )
