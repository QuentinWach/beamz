"""Backward-compatible data helpers for design visualization."""

from beamz.visual.data import boundary_plot_data, design_plot_data, structure_plot_data


def determine_if_3d(design):
    """Determine if the design should be visualized in 3D based on structure properties."""
    if design.depth and design.depth > 0:
        for structure in design.structures:
            if hasattr(structure, "is_pml") and structure.is_pml:
                continue
            if hasattr(structure, "depth") and structure.depth and structure.depth > 0:
                return True
            if hasattr(structure, "z") and structure.z and structure.z != 0:
                return True
            if (
                hasattr(structure, "position")
                and len(structure.position) > 2
                and structure.position[2] != 0
            ):
                return True
            if hasattr(structure, "vertices") and structure.vertices:
                for vertex in structure.vertices:
                    if len(vertex) > 2 and vertex[2] != 0:
                        return True
    return False


def show_design(*args, **kwargs):
    raise RuntimeError(
        "Design.show()/show_design() were removed from beamz. "
        "Use Design.to_plot_data() and render in examples."
    )


def show_design_2d(*args, **kwargs):
    raise RuntimeError(
        "show_design_2d() was removed from beamz. "
        "Use Design.to_plot_data() and render in examples."
    )


def draw_polygon(*args, **kwargs):
    raise RuntimeError(
        "draw_polygon() was removed from beamz. "
        "Use structure_plot_data() and draw in examples."
    )


def draw_boundary(boundary, design, **kwargs):
    return boundary_plot_data(boundary, design, **kwargs)


__all__ = [
    "boundary_plot_data",
    "design_plot_data",
    "determine_if_3d",
    "draw_boundary",
    "show_design",
    "show_design_2d",
    "structure_plot_data",
]
