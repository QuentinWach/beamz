from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from beamz.visual.scene import demo_scene, view3d


def main() -> None:
    result = view3d(demo_scene())
    if isinstance(result, str):
        print("ZView demo opened in browser.")
        print(f"url: {result}")
        return
    print("ZView demo widget created.")
    print(f"scene title: {result.scene_json.get('title')}")
    print(f"object count: {len(result.scene_json.get('objects', []))}")
    try:
        from IPython.display import display
    except ImportError:
        return
    display(result)


if __name__ == "__main__":
    main()
