# Medium: Dispersionless medium.
class Material:
    def __init__(self, permittivity=1.0, permeability=1.0, conductivity=0.0):
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity

    def get_sample(self):
        return self.permittivity, self.permeability, self.conductivity


# CustomMaterial: Function-based material for inverse design
class CustomMaterial:
    def __init__(
        self,
        permittivity_func=None,
        permeability_func=None,
        conductivity_func=None,
        permittivity_grid=None,
        permeability_grid=None,
        conductivity_grid=None,
        bounds=None,
        interpolation="linear",
    ):
        """
        Custom material with spatially-varying properties for inverse design.

        Args:
            permittivity_func: Function that takes (x, y) or (x, y, z) and returns permittivity
            permeability_func: Function that takes (x, y) or (x, y, z) and returns permeability
            conductivity_func: Function that takes (x, y) or (x, y, z) and returns conductivity
            permittivity_grid: 2D numpy array of permittivity values for grid-based interpolation
            permeability_grid: 2D numpy array of permeability values for grid-based interpolation
            conductivity_grid: 2D numpy array of conductivity values for grid-based interpolation
            bounds: Tuple ((x_min, x_max), (y_min, y_max)) defining the spatial bounds for grid interpolation
            interpolation: 'linear', 'cubic', or 'nearest' for grid interpolation

        Examples:
            # Function-based material
            def perm_func(x, y):
                return 2.0 + 0.5 * np.sin(x) * np.cos(y)
            material = CustomMaterial(permittivity_func=perm_func)

            # Grid-based material for inverse design
            perm_grid = np.ones((50, 50)) * 2.0
            perm_grid[20:30, 20:30] = 4.0  # High index region
            material = CustomMaterial(
                permittivity_grid=perm_grid,
                bounds=((0, 10e-6), (0, 10e-6))  # 10 micron x 10 micron
            )
        """
        import numpy as np

        # Store function-based definitions
        self.permittivity_func = permittivity_func
        self.permeability_func = permeability_func
        self.conductivity_func = conductivity_func

        # Store grid-based definitions
        self.permittivity_grid = permittivity_grid
        self.permeability_grid = permeability_grid
        self.conductivity_grid = conductivity_grid

        # Spatial bounds for grid interpolation
        self.bounds = bounds
        self.interpolation = interpolation

        # Default values
        self.default_permittivity = 1.0
        self.default_permeability = 1.0
        self.default_conductivity = 0.0

        # Create interpolation functions for grids
        if permittivity_grid is not None and bounds is not None:
            self._create_grid_interpolator("permittivity")
        if permeability_grid is not None and bounds is not None:
            self._create_grid_interpolator("permeability")
        if conductivity_grid is not None and bounds is not None:
            self._create_grid_interpolator("conductivity")

    @property
    def permittivity(self):
        """Return representative permittivity for display purposes."""
        if self.permittivity_grid is not None:
            import numpy as np

            return f"grid({np.min(self.permittivity_grid):.3f}-{np.max(self.permittivity_grid):.3f})"
        elif self.permittivity_func is not None:
            return "function"
        else:
            return self.default_permittivity

    @property
    def permeability(self):
        """Return representative permeability for display purposes."""
        if self.permeability_grid is not None:
            import numpy as np

            return f"grid({np.min(self.permeability_grid):.3f}-{np.max(self.permeability_grid):.3f})"
        elif self.permeability_func is not None:
            return "function"
        else:
            return self.default_permeability

    @property
    def conductivity(self):
        """Return representative conductivity for display purposes."""
        if self.conductivity_grid is not None:
            import numpy as np

            return f"grid({np.min(self.conductivity_grid):.3f}-{np.max(self.conductivity_grid):.3f})"
        elif self.conductivity_func is not None:
            return "function"
        else:
            return self.default_conductivity

    def _create_grid_interpolator(self, property_name):
        """Create scipy interpolator for grid-based material property."""
        try:
            import numpy as np
            from scipy.interpolate import RegularGridInterpolator

            grid = getattr(self, f"{property_name}_grid")
            if grid is None:
                return

            # Create coordinate arrays
            x_coords = np.linspace(self.bounds[0][0], self.bounds[0][1], grid.shape[1])
            y_coords = np.linspace(self.bounds[1][0], self.bounds[1][1], grid.shape[0])

            # Create interpolator
            interpolator = RegularGridInterpolator(
                (y_coords, x_coords),
                grid,
                method=self.interpolation,
                bounds_error=False,
                fill_value=getattr(self, f"default_{property_name}"),
            )

            # Store interpolator
            setattr(self, f"_{property_name}_interpolator", interpolator)

        except ImportError:
            print("Warning: scipy not available, using nearest neighbor interpolation")
            setattr(self, f"_{property_name}_interpolator", None)

    def get_permittivity(self, x, y, z=None):
        """Get permittivity at spatial coordinates (x, y, z)."""
        if self.permittivity_func is not None:
            if z is not None:
                return self.permittivity_func(x, y, z)
            else:
                return self.permittivity_func(x, y)
        elif (
            hasattr(self, "_permittivity_interpolator")
            and self._permittivity_interpolator is not None
        ):
            import numpy as np

            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._permittivity_interpolator(points)
        else:
            return self.default_permittivity

    def get_permeability(self, x, y, z=None):
        """Get permeability at spatial coordinates (x, y, z)."""
        if self.permeability_func is not None:
            if z is not None:
                return self.permeability_func(x, y, z)
            else:
                return self.permeability_func(x, y)
        elif (
            hasattr(self, "_permeability_interpolator")
            and self._permeability_interpolator is not None
        ):
            import numpy as np

            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._permeability_interpolator(points)
        else:
            return self.default_permeability

    def get_conductivity(self, x, y, z=None):
        """Get conductivity at spatial coordinates (x, y, z)."""
        if self.conductivity_func is not None:
            if z is not None:
                return self.conductivity_func(x, y, z)
            else:
                return self.conductivity_func(x, y)
        elif (
            hasattr(self, "_conductivity_interpolator")
            and self._conductivity_interpolator is not None
        ):
            import numpy as np

            points = np.column_stack([np.atleast_1d(y), np.atleast_1d(x)])
            return self._conductivity_interpolator(points)
        else:
            return self.default_conductivity

    def get_sample(self, x=0, y=0, z=None):
        """Get material properties at spatial coordinates for backward compatibility."""
        return (
            self.get_permittivity(x, y, z),
            self.get_permeability(x, y, z),
            self.get_conductivity(x, y, z),
        )

    def update_grid(self, property_name, new_grid):
        """Update material property grid (for optimization)."""
        if property_name == "permittivity":
            self.permittivity_grid = new_grid
            self._create_grid_interpolator("permittivity")
        elif property_name == "permeability":
            self.permeability_grid = new_grid
            self._create_grid_interpolator("permeability")
        elif property_name == "conductivity":
            self.conductivity_grid = new_grid
            self._create_grid_interpolator("conductivity")
        else:
            raise ValueError(f"Unknown property: {property_name}")

    def copy(self):
        """Create a deep copy of the CustomMaterial."""
        import numpy as np

        # Deep copy grids if they exist
        perm_grid = (
            self.permittivity_grid.copy()
            if self.permittivity_grid is not None
            else None
        )
        permeability_grid = (
            self.permeability_grid.copy()
            if self.permeability_grid is not None
            else None
        )
        cond_grid = (
            self.conductivity_grid.copy()
            if self.conductivity_grid is not None
            else None
        )

        # Create new CustomMaterial with copied data
        return CustomMaterial(
            permittivity_func=self.permittivity_func,  # Functions can be shared
            permeability_func=self.permeability_func,
            conductivity_func=self.conductivity_func,
            permittivity_grid=perm_grid,  # Deep copied grids
            permeability_grid=permeability_grid,
            conductivity_grid=cond_grid,
            bounds=self.bounds,  # Bounds can be shared (tuples are immutable)
            interpolation=self.interpolation,
        )


# ================================

# PoleResidue: A dispersive medium described by the pole-residue pair model.

# Lorentz: A dispersive medium described by the Lorentz model.

# Sellmeier: A dispersive medium described by the Sellmeier model.

# Drude: A dispersive medium described by the Drude model.

# Debye: A dispersive medium described by the Debye model.
