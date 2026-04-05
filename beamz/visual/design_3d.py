"""Compatibility wrapper for 3D design viewing."""

from __future__ import annotations


def show_design_3d(
    design,
    unify_structures: bool = True,
    max_vertices_for_unification: int = 50,
    *,
    mode: str = "browser",
    open_browser: bool = True,
    **kwargs,
):
    """Render a 3D design through the scene viewer.

    The old Plotly-specific implementation duplicated the newer scene pipeline.
    Keep the public entry point, but route it through the single maintained viewer.
    """
    del max_vertices_for_unification
    if unify_structures and hasattr(design, "copy") and hasattr(design, "unify_polygons"):
        design = design.copy()
        design.unify_polygons()

    from beamz.visual.scene import view3d

    return view3d(design, mode=mode, open_browser=open_browser, **kwargs)
