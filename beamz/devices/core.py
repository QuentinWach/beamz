class Device:
    """Base class for all simulation devices (sources, monitors, etc.)."""
    
    def get_source_terms(self, fields, t, dt, current_step, resolution, design):
        """Return source current terms for FDTD update.
        
        Returns:
            source_j: dict mapping field components to (current_array, indices) tuples
            source_m: dict mapping field components to (current_array, indices) tuples
        """
        return {}, {}  # Override in subclasses