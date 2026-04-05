import numpy as np


class MaterialGrids:
    """Bundles the core EM material property arrays with bulk operations."""

    NAMES = ("permittivity", "permeability", "conductivity")
    DEFAULTS = (1.0, 1.0, 0.0)

    def __init__(self, shape):
        for name, default in zip(self.NAMES, self.DEFAULTS):
            setattr(self, name, np.full(shape, default))

    def fill_all(self, props):
        """Fill all grids with material property tuple."""
        for name, val in zip(self.NAMES, props):
            getattr(self, name).fill(val)

    def set_at(self, idx, props):
        """Set all properties at index (i,j) or (k,i,j)."""
        for name, val in zip(self.NAMES, props):
            getattr(self, name)[idx] = val

    def blend_at(self, idx, props, factor):
        """Blend properties at index with given factor."""
        for name, val in zip(self.NAMES, props):
            arr = getattr(self, name)
            arr[idx] = arr[idx] * (1 - factor) + val * factor

    def set_region(self, slices, props):
        """Set all properties for a slice/index-array region."""
        for name, val in zip(self.NAMES, props):
            getattr(self, name)[slices] = val

    def assign_to(self, target):
        """Copy all grids as attributes onto target object."""
        for name in self.NAMES:
            setattr(target, name, getattr(self, name))
