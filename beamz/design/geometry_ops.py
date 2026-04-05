from __future__ import annotations

import numpy as np


def rotate_vertices(vertices, angle_rad, axis, center):
    """Rotate a list of 3D vertices around center by angle_rad on the given axis."""
    cx, cy, cz = center
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "z":
        return [
            (
                cx + (v[0] - cx) * cos_a - (v[1] - cy) * sin_a,
                cy + (v[0] - cx) * sin_a + (v[1] - cy) * cos_a,
                v[2],
            )
            for v in vertices
        ]
    if axis == "x":
        return [
            (
                v[0],
                cy + (v[1] - cy) * cos_a - (v[2] - cz) * sin_a,
                cz + (v[1] - cy) * sin_a + (v[2] - cz) * cos_a,
            )
            for v in vertices
        ]
    if axis == "y":
        return [
            (
                cx + (v[0] - cx) * cos_a + (v[2] - cz) * sin_a,
                v[1],
                cz - (v[0] - cx) * sin_a + (v[2] - cz) * cos_a,
            )
            for v in vertices
        ]
    raise ValueError(f"Invalid rotation axis '{axis}'. Must be 'x', 'y', or 'z'.")


def normalize_position(position, z=None):
    """Ensure position is a 3-tuple, optionally overriding z."""
    if len(position) == 2:
        position = (position[0], position[1], 0.0)
    elif len(position) != 3:
        raise ValueError("Position must be (x,y) or (x,y,z)")
    if z is not None:
        position = (position[0], position[1], z)
    return tuple(float(v) for v in position)


def require_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_nonnegative(name, value):
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def vertices_center(vertices):
    return tuple(sum(v[i] for v in vertices) / len(vertices) for i in range(3))


def vertices_bbox(vertices):
    x_vals = [v[0] for v in vertices]
    y_vals = [v[1] for v in vertices]
    z_vals = [v[2] for v in vertices]
    return (
        min(x_vals),
        min(y_vals),
        min(z_vals),
        max(x_vals),
        max(y_vals),
        max(z_vals),
    )


def circle_vertices(position, radius, points, *, theta=None, reverse=False):
    theta = (
        np.linspace(0, 2 * np.pi, points, endpoint=False)
        if theta is None
        else np.asarray(theta, dtype=float)
    )
    angles = reversed(theta) if reverse else theta
    return [
        (
            position[0] + radius * np.cos(t),
            position[1] + radius * np.sin(t),
            position[2],
        )
        for t in angles
    ]


def ring_vertices(position, inner_radius, outer_radius, points):
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    outer = circle_vertices(position, outer_radius, points, theta=theta)
    inner = circle_vertices(position, inner_radius, points, theta=theta, reverse=True)
    return outer, inner


def bend_vertices(position, inner_radius, outer_radius, angle, rotation, points):
    theta = np.linspace(0, np.radians(angle), points)
    rotation_rad = np.radians(rotation)
    angles = theta + rotation_rad
    outer = circle_vertices(position, outer_radius, points, theta=angles)
    inner = circle_vertices(position, inner_radius, points, theta=angles, reverse=True)
    return outer + inner


def freeze_vertices(vertices):
    return tuple(tuple(float(c) for c in vertex) for vertex in vertices)


def freeze_interiors(interiors):
    return tuple(freeze_vertices(path) for path in interiors if path)


def transform_geometry(vertices, interiors, transform):
    return (
        [transform(v) for v in vertices],
        [[transform(v) for v in path] for path in interiors if path],
    )
