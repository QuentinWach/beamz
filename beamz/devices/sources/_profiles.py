"""Internal field-profile containers shared by source implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FieldAxis3D = Literal["x", "y", "z"]


@dataclass(frozen=True)
class FieldProfile3D:
    """Discrete 3D E/H profile on BeamZ's Yee component grids."""

    components: dict[str, np.ndarray]
    indices: dict[str, tuple[slice, slice, slice]]
    axis: FieldAxis3D
    direction_sign: float
    omega: float
    k_axis: float | None
    phase_ref_coord: float
    phase_plane_coord: float
