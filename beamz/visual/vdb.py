"""VDB export functionality for 3D FDTD simulation data.

This module provides the VDBExporter class for exporting 3D volumetric field data
to OpenVDB format (.vdb files), which can be imported into visualization software
like Blender for rendering and animation.

OpenVDB is a hierarchical data structure optimized for sparse volumetric data,
commonly used in visual effects for representing volumes like smoke, fire, and clouds.
"""

import os
import numpy as np


class VDBExporter:
    """Export 3D simulation fields to OpenVDB format for visualization in Blender.

    This class collects 3D field snapshots during simulation and exports them as
    numbered VDB files that can be imported into Blender or other VFX software
    for volumetric rendering and animation.

    Usage:
        # Create exporter for specific fields
        exporter = VDBExporter(
            output_dir='vdb_output',
            fields=['Ez', 'Hy'],
            interval=10  # Export every 10 steps
        )

        # During simulation loop:
        if exporter.should_export(step):
            exporter.add_frame(simulation.fields, step, t, resolution)

        # After simulation:
        exporter.save()  # Writes all VDB files

    In Blender:
        1. Import VDB files via File > Import > Volume
        2. Use sequence import for animation
        3. Apply volume shader for visualization
    """

    def __init__(self, output_dir='vdb_output', fields=None, interval=1,
                 normalize=True, grid_name_prefix='', clamp_percentile=99.0):
        """Initialize the VDB exporter.

        Args:
            output_dir: Directory to save VDB files (created if doesn't exist)
            fields: List of field names to export (default: ['Ez'])
                    Available: 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'
            interval: Export every N simulation steps (default: 1)
            normalize: If True, normalize field values to [-1, 1] range for
                      better compatibility with Blender shaders (default: True)
            grid_name_prefix: Prefix for grid names in VDB file (default: '')
            clamp_percentile: Percentile for clamping extreme values before
                             normalization (default: 99.0). Set to 100 to disable.
        """
        self.output_dir = output_dir
        self.fields = fields if fields is not None else ['Ez']
        self.interval = interval
        self.normalize = normalize
        self.grid_name_prefix = grid_name_prefix
        self.clamp_percentile = clamp_percentile

        # Storage for frames
        self.frames = []  # List of dicts: {'step': int, 't': float, 'fields': {name: array}}

        # Metadata
        self.resolution = None
        self.voxel_size = None
        self._global_max = {}  # Track global max per field for consistent normalization

        # Check if pyopenvdb is available
        self._vdb_available = None

    def _check_vdb_available(self):
        """Check if pyopenvdb/openvdb is installed and available."""
        if self._vdb_available is None:
            try:
                import pyopenvdb
                self._vdb_available = True
            except ImportError:
                try:
                    import openvdb
                    self._vdb_available = True
                except ImportError:
                    self._vdb_available = False
        return self._vdb_available

    def should_export(self, step):
        """Check if current step should trigger an export.

        Args:
            step: Current simulation step number

        Returns:
            bool: True if this step should be exported
        """
        return step % self.interval == 0

    def add_frame(self, fields_obj, step, t, resolution):
        """Add a frame of field data for export.

        Args:
            fields_obj: Fields object from simulation (has Ex, Ey, Ez, Hx, Hy, Hz)
            step: Current simulation step number
            t: Current simulation time
            resolution: Grid resolution (voxel size in physical units)
        """
        # Store resolution on first frame
        if self.resolution is None:
            self.resolution = resolution
            self.voxel_size = float(resolution)

        # Collect requested field data
        frame_data = {
            'step': step,
            't': t,
            'fields': {}
        }

        for field_name in self.fields:
            if hasattr(fields_obj, field_name):
                field_array = getattr(fields_obj, field_name)
                # Convert to numpy and copy
                arr = np.asarray(field_array, dtype=np.float32).copy()

                # Only store if 3D
                if arr.ndim == 3:
                    frame_data['fields'][field_name] = arr

                    # Track global max for normalization
                    if self.normalize:
                        if self.clamp_percentile < 100:
                            frame_max = np.percentile(np.abs(arr), self.clamp_percentile)
                        else:
                            frame_max = np.max(np.abs(arr))

                        if field_name not in self._global_max:
                            self._global_max[field_name] = frame_max
                        else:
                            self._global_max[field_name] = max(self._global_max[field_name], frame_max)
                else:
                    print(f"Warning: Field '{field_name}' is not 3D (shape: {arr.shape}), skipping")
            else:
                print(f"Warning: Field '{field_name}' not found in simulation")

        if frame_data['fields']:
            self.frames.append(frame_data)

    def _normalize_field(self, arr, field_name):
        """Normalize field values to [-1, 1] range.

        Args:
            arr: Field array to normalize
            field_name: Name of the field (for global max lookup)

        Returns:
            Normalized array
        """
        if not self.normalize:
            return arr

        max_val = self._global_max.get(field_name, 1.0)
        if max_val > 0:
            # Clamp and normalize
            if self.clamp_percentile < 100:
                arr = np.clip(arr, -max_val, max_val)
            return arr / max_val
        return arr

    def _create_vdb_grid(self, arr, grid_name, voxel_size):
        """Create an OpenVDB FloatGrid from a numpy array.

        Args:
            arr: 3D numpy array (z, y, x) ordering
            grid_name: Name for the grid
            voxel_size: Size of each voxel

        Returns:
            pyopenvdb.FloatGrid or openvdb.FloatGrid
        """
        try:
            import pyopenvdb as vdb
        except ImportError:
            import openvdb as vdb

        # Create grid with proper voxel size transform
        grid = vdb.FloatGrid()
        grid.name = grid_name

        # Set the voxel size transform
        # OpenVDB uses a 4x4 transformation matrix
        # The conda-forge openvdb expects a scalar float, not a tuple
        voxel_size_float = float(voxel_size) if isinstance(voxel_size, (int, float)) else float(voxel_size[0])

        # Scale voxel size to micrometers for better Blender visualization
        # Physical simulations use meters (e.g., 1e-7 m), but openvdb requires voxel_size > ~1e-6
        # Converting to µm makes the values reasonable (e.g., 0.1 µm) for Blender
        voxel_size_um = voxel_size_float * 1e6  # Convert meters to micrometers

        grid.transform = vdb.createLinearTransform(voxelSize=voxel_size_um)

        # Copy data to grid
        # OpenVDB expects (x, y, z) indexing, numpy uses (z, y, x)
        # We need to transpose for correct spatial orientation
        arr_xyz = np.ascontiguousarray(arr.transpose(2, 1, 0))
        grid.copyFromArray(arr_xyz)

        return grid

    def save(self, verbose=True):
        """Save all collected frames as VDB files.

        Creates numbered VDB files in the output directory, one per frame.
        Each file contains grids for all requested fields.

        Args:
            verbose: Print progress information (default: True)

        Returns:
            List of saved file paths
        """
        if not self._check_vdb_available():
            print("Error: openvdb is not installed. Install it with:")
            print("  pip install pyopenvdb")
            print("Or on conda: conda install -c conda-forge openvdb")
            return []

        if not self.frames:
            print("No frames to save.")
            return []

        try:
            import pyopenvdb as vdb
        except ImportError:
            import openvdb as vdb

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        saved_files = []
        num_frames = len(self.frames)

        if verbose:
            print(f"Saving {num_frames} VDB frames to '{self.output_dir}/'...")

        # Determine padding for frame numbers
        padding = len(str(num_frames))

        for i, frame in enumerate(self.frames):
            step = frame['step']
            t = frame['t']

            # Create grids for each field
            grids = []
            for field_name, arr in frame['fields'].items():
                # Normalize if requested
                arr_norm = self._normalize_field(arr, field_name)

                # Create grid
                grid_name = f"{self.grid_name_prefix}{field_name}" if self.grid_name_prefix else field_name
                grid = self._create_vdb_grid(arr_norm, grid_name, self.voxel_size)

                # Add metadata
                grid.metadata = {
                    'step': step,
                    'time': t,
                    'field': field_name,
                    'resolution': self.resolution,
                    'normalized': self.normalize
                }

                grids.append(grid)

            # Write VDB file
            frame_num = str(i).zfill(padding)
            filename = os.path.join(self.output_dir, f"frame_{frame_num}.vdb")
            vdb.write(filename, grids=grids)
            saved_files.append(filename)

            if verbose and (i + 1) % max(1, num_frames // 10) == 0:
                pct = 100 * (i + 1) / num_frames
                print(f"  Progress: {pct:.0f}% ({i + 1}/{num_frames} frames)")

        if verbose:
            print(f"Saved {len(saved_files)} VDB files to '{self.output_dir}/'")
            print(f"Fields exported: {list(frame['fields'].keys())}")
            if self.normalize:
                print(f"Normalization: enabled (values scaled to [-1, 1])")
                for field_name, max_val in self._global_max.items():
                    print(f"  {field_name} max: {max_val:.2e}")

        return saved_files

    def save_single_frame(self, fields_obj, filename, step=0, t=0.0, resolution=None):
        """Save a single frame directly without accumulating.

        Useful for exporting a snapshot at a specific time.

        Args:
            fields_obj: Fields object from simulation
            filename: Output filename (should end in .vdb)
            step: Step number for metadata (default: 0)
            t: Time value for metadata (default: 0.0)
            resolution: Grid resolution (default: use stored or 1.0)

        Returns:
            Path to saved file or None if failed
        """
        if not self._check_vdb_available():
            print("Error: openvdb is not installed. Install it with:")
            print("  pip install pyopenvdb")
            print("Or on conda: conda install -c conda-forge openvdb")
            return None

        try:
            import pyopenvdb as vdb
        except ImportError:
            import openvdb as vdb

        voxel_size = float(resolution) if resolution else (self.voxel_size or 1.0)

        grids = []
        for field_name in self.fields:
            if hasattr(fields_obj, field_name):
                field_array = getattr(fields_obj, field_name)
                arr = np.asarray(field_array, dtype=np.float32)

                if arr.ndim != 3:
                    print(f"Warning: Field '{field_name}' is not 3D, skipping")
                    continue

                # Normalize if requested
                if self.normalize:
                    max_val = np.percentile(np.abs(arr), self.clamp_percentile)
                    if max_val > 0:
                        arr = np.clip(arr, -max_val, max_val) / max_val

                grid_name = f"{self.grid_name_prefix}{field_name}" if self.grid_name_prefix else field_name
                grid = self._create_vdb_grid(arr, grid_name, voxel_size)
                grid.metadata = {
                    'step': step,
                    'time': t,
                    'field': field_name,
                    'resolution': voxel_size
                }
                grids.append(grid)

        if grids:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            vdb.write(filename, grids=grids)
            return filename

        return None

    def get_frame_count(self):
        """Return the number of frames collected."""
        return len(self.frames)

    def clear(self):
        """Clear all stored frames."""
        self.frames = []
        self._global_max = {}
