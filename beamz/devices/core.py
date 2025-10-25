class Device:
    """Base class for all simulation devices (sources, monitors, etc.)."""
    
    def inject(self, fields, t, dt, current_step, resolution, design):
        """Inject device contribution into fields (soft injection).
        
        Args:
            fields: Fields object with E/H field arrays
            t: Current simulation time
            dt: Time step
            current_step: Current step index
            resolution: Grid resolution
            design: Design object for spatial information
        """
        pass  # Override in subclasses