class Device:
    """Base class for all simulation devices (sources, monitors, etc.)."""

    def inject(self, fields, t, dt, current_step, resolution, design):
        """Inject source fields directly into the simulation grid.

        This method is called before the field update step to add source contributions directly
        to the field arrays.
        """
        pass

    def get_source_terms(self, fields, t, dt, current_step, resolution, design):
        """Return source current terms for FDTD update.

        Returns:
            source_j: dict mapping field components to (current_array, indices) tuples
            source_m: dict mapping field components to (current_array, indices) tuples
        """
        return {}, {}  # Override in subclasses
