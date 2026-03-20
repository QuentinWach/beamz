from beamz.visual.scene import demo_scene
from beamz.visual.scene._browser import open_in_browser


def test_open_in_browser_writes_html_without_launching():
    url = open_in_browser(demo_scene(), open_browser=False)
    assert url.startswith("file://")
