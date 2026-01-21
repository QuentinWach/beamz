"""VDB export functionality for 3D FDTD simulation data.

This module provides the VDBExporter class for exporting 3D volumetric field data
to OpenVDB format (.vdb files), which can be imported into visualization software
like Blender for rendering and animation.

OpenVDB is a hierarchical data structure optimized for sparse volumetric data,
commonly used in visual effects for representing volumes like smoke, fire, and clouds.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class FieldSpec:
    """Specification for a field to export to VDB.

    Attributes:
        field_type: Either 'component' (Ex, Ey, etc.) or 'magnitude' (|E|, |H|)
        base_field: Base field name ('E' or 'H') for magnitude fields
        component: Component name ('Ex', 'Ey', etc.) for component fields
        grid_name: Name to use in VDB file
    """
    field_type: Literal['component', 'magnitude']
    base_field: str = None
    component: str = None
    grid_name: str = None


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
                 normalize=True, vdb_log_scale=None, vdb_log_epsilon=None,
                 grid_name_prefix='', clamp_percentile=99.0):
        """Initialize the VDB exporter.

        Args:
            output_dir: Directory to save VDB files (created if doesn't exist)
            fields: List of field names to export (default: ['|E|'])
                    Available: 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', '|E|', '|H|'
                    Magnitude fields like '|E|' automatically enable log scaling
            interval: Export every N simulation steps (default: 1)
            normalize: If True, normalize field values for Blender compatibility (default: True)
                      Magnitude fields: [0, 1], Component fields: [-1, 1]
            vdb_log_scale: Apply logarithmic transform for dynamic range compression (default: None)
                          None = auto-enable for magnitude fields, True/False = force on/off
            vdb_log_epsilon: Floor value for log transform (default: None = auto-compute)
                            Auto-computed as max_value × 1e-6
            grid_name_prefix: Prefix for grid names in VDB file (default: '')
            clamp_percentile: Percentile for clamping extreme values before
                             normalization (default: 99.0). Set to 100 to disable.
        """
        self.output_dir = output_dir
        self.raw_fields = fields if fields is not None else ['|E|']
        self.interval = interval
        self.normalize = normalize
        self.grid_name_prefix = grid_name_prefix
        self.clamp_percentile = clamp_percentile
        self.log_epsilon = vdb_log_epsilon

        # Parse field specifications
        self.field_specs = [self._parse_field_spec(f) for f in self.raw_fields]

        # Auto-enable log scaling for magnitude fields if not specified
        if vdb_log_scale is None:
            self.log_scale = self._has_magnitude_fields()
        else:
            self.log_scale = vdb_log_scale

        # Storage for frames
        self.frames = []  # List of dicts: {'step': int, 't': float, 'fields': {name: array}}

        # Metadata
        self.resolution = None
        self.voxel_size = None
        self._global_max = {}  # Track global max per field for consistent normalization
        self._ez_reference_shape = None  # Store Ez grid shape for interpolation

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

    def _parse_field_spec(self, spec):
        """Parse a field specification string into a FieldSpec object.

        Args:
            spec: Field specification string (e.g., 'Ez', '|E|', '|H|')

        Returns:
            FieldSpec object

        Raises:
            ValueError: If the specification is invalid
        """
        spec = spec.strip()

        # Check for magnitude field syntax: |E| or |H|
        if spec.startswith('|') and spec.endswith('|'):
            base = spec[1:-1].upper()
            if base not in ['E', 'H']:
                raise ValueError(
                    f"Invalid magnitude field '{spec}'. "
                    f"Valid options: '|E|', '|H|'"
                )
            return FieldSpec(
                field_type='magnitude',
                base_field=base,
                grid_name=f'{base}_mag'
            )

        # Component field: Ex, Ey, Ez, Hx, Hy, Hz
        valid_components = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        if spec in valid_components:
            return FieldSpec(
                field_type='component',
                component=spec,
                grid_name=spec
            )

        # Invalid specification
        raise ValueError(
            f"Invalid field specification '{spec}'. "
            f"Valid options: {', '.join(valid_components)}, '|E|', '|H|'"
        )

    def _has_magnitude_fields(self):
        """Check if any magnitude fields are in the export list.

        Returns:
            bool: True if any magnitude fields are present
        """
        return any(spec.field_type == 'magnitude' for spec in self.field_specs)

    def _interpolate_to_ez_grid(self, arr):
        """Interpolate field from Yee grid to Ez grid reference.

        The Yee grid staggers field components at different locations:
        - Ez grid: (nz-1, ny, nx)
        - Ex grid: (nz, ny, nx-1) → average in z, pad in x
        - Ey grid: (nz, ny-1, nx) → average in z, pad in y

        Args:
            arr: Field array to interpolate

        Returns:
            Array interpolated to Ez grid shape
        """
        if self._ez_reference_shape is None:
            raise RuntimeError("Ez reference shape not set. Call add_frame first.")

        target = self._ez_reference_shape

        # Already at target shape
        if arr.shape == target:
            return arr.astype(np.float32)

        # Ex: (nz, ny, nx-1) → (nz-1, ny, nx)
        if arr.shape[2] == target[2] - 1:  # Missing x dimension
            arr_z = 0.5 * (arr[:-1] + arr[1:])  # Average in z
            return np.pad(arr_z, ((0, 0), (0, 0), (0, 1)), mode='edge').astype(np.float32)

        # Ey: (nz, ny-1, nx) → (nz-1, ny, nx)
        elif arr.shape[1] == target[1] - 1:  # Missing y dimension
            arr_z = 0.5 * (arr[:-1] + arr[1:])  # Average in z
            return np.pad(arr_z, ((0, 0), (1, 0), (0, 0)), mode='edge').astype(np.float32)

        else:
            raise ValueError(
                f"Cannot interpolate array with shape {arr.shape} to Ez grid {target}. "
                f"Expected shape close to {target}."
            )

    def _compute_magnitude_field(self, fields_obj, base_field):
        """Compute |E| or |H| from components.

        Components live on different Yee grid locations, so we interpolate
        all components to the Ez (or Hz) grid reference and compute magnitude.

        Args:
            fields_obj: Fields object from simulation
            base_field: 'E' or 'H'

        Returns:
            Magnitude field as 3D array on Ez/Hz grid
        """
        if base_field == 'E':
            components = ['Ex', 'Ey', 'Ez']
        elif base_field == 'H':
            components = ['Hx', 'Hy', 'Hz']
        else:
            raise ValueError(f"Invalid base field '{base_field}'. Must be 'E' or 'H'.")

        # Check that all components are available
        for comp in components:
            if not hasattr(fields_obj, comp):
                raise RuntimeError(
                    f"Magnitude field '|{base_field}|' requires {', '.join(components)} components. "
                    f"Component '{comp}' not found in simulation."
                )

        # Get component arrays
        comp_arrays = []
        for comp in components:
            arr = np.asarray(getattr(fields_obj, comp), dtype=np.float32)
            if arr.ndim != 3:
                raise RuntimeError(f"Component '{comp}' is not 3D (shape: {arr.shape})")
            comp_arrays.append(arr)

        # Set Ez reference shape on first call
        if self._ez_reference_shape is None:
            # Use the z-component grid as reference (Ez or Hz)
            self._ez_reference_shape = comp_arrays[2].shape

        # Interpolate all components to Ez grid
        interp_arrays = []
        for i, arr in enumerate(comp_arrays):
            if i == 2:  # z-component is already on target grid
                interp_arrays.append(arr)
            else:
                interp_arrays.append(self._interpolate_to_ez_grid(arr))

        # Compute magnitude: sqrt(Ex² + Ey² + Ez²)
        mag_sq = sum(arr**2 for arr in interp_arrays)
        return np.sqrt(mag_sq)

    def _apply_log_transform(self, arr, is_magnitude=False):
        """Apply log10 transform for dynamic range compression.

        Auto-computes epsilon as max_value × 1e-6 to capture 6 orders of
        magnitude below peak. For signed fields (components), preserves sign.

        Args:
            arr: Field array to transform
            is_magnitude: If True, treat as unsigned magnitude field

        Returns:
            Log-transformed array
        """
        # Auto-compute epsilon if not provided
        if self.log_epsilon is None:
            max_val = np.max(np.abs(arr))
            epsilon = max(max_val * 1e-6, 1e-12)  # Floor at 1e-12
        else:
            epsilon = self.log_epsilon

        # Handle signed vs unsigned fields
        if is_magnitude or not np.any(arr < 0):
            # Magnitude fields: log10(x + ε)
            return np.log10(arr + epsilon)
        else:
            # Component fields: sign(x) * log10(|x| + ε) - preserves wave phase
            sign = np.sign(arr)
            log_abs = np.log10(np.abs(arr) + epsilon)
            return sign * log_abs

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
            'fields': {},
            'field_specs': {}  # Store field spec for each grid
        }

        for field_spec in self.field_specs:
            try:
                # Handle magnitude fields
                if field_spec.field_type == 'magnitude':
                    arr = self._compute_magnitude_field(fields_obj, field_spec.base_field)
                    grid_name = field_spec.grid_name

                # Handle component fields
                else:
                    if hasattr(fields_obj, field_spec.component):
                        field_array = getattr(fields_obj, field_spec.component)
                        arr = np.asarray(field_array, dtype=np.float32).copy()
                        grid_name = field_spec.grid_name

                        if arr.ndim != 3:
                            print(f"Warning: Field '{field_spec.component}' is not 3D (shape: {arr.shape}), skipping")
                            continue
                    else:
                        print(f"Warning: Field '{field_spec.component}' not found in simulation")
                        continue

                # Store field data
                frame_data['fields'][grid_name] = arr
                frame_data['field_specs'][grid_name] = field_spec

                # Track global max for normalization
                if self.normalize:
                    if self.clamp_percentile < 100:
                        frame_max = np.percentile(np.abs(arr), self.clamp_percentile)
                    else:
                        frame_max = np.max(np.abs(arr))

                    if grid_name not in self._global_max:
                        self._global_max[grid_name] = frame_max
                    else:
                        self._global_max[grid_name] = max(self._global_max[grid_name], frame_max)

            except Exception as e:
                print(f"Warning: Failed to compute field '{field_spec.grid_name}': {e}")
                continue

        if frame_data['fields']:
            self.frames.append(frame_data)

    def _normalize_field(self, arr, field_spec, grid_name):
        """Normalize field values for Blender compatibility.

        Magnitude fields: [0, 1] - suitable for volume density
        Component fields: [-1, 1] - preserve wave phase

        Args:
            arr: Field array to normalize
            field_spec: FieldSpec object for this field
            grid_name: Grid name for global max lookup

        Returns:
            Normalized array
        """
        if not self.normalize:
            return arr

        if field_spec.field_type == 'magnitude':
            # [0, 1] normalization for magnitude fields
            min_val = np.min(arr)
            max_val = self._global_max.get(grid_name, 1.0)
            if max_val > min_val:
                if self.clamp_percentile < 100:
                    arr = np.clip(arr, min_val, max_val)
                return (arr - min_val) / (max_val - min_val)
            return arr * 0

        else:
            # [-1, 1] normalization for component fields
            max_val = self._global_max.get(grid_name, 1.0)
            if max_val > 0:
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
            for grid_name, arr in frame['fields'].items():
                field_spec = frame['field_specs'][grid_name]

                # Apply log transform if enabled (before normalization)
                if self.log_scale:
                    arr = self._apply_log_transform(arr, is_magnitude=(field_spec.field_type == 'magnitude'))

                # Normalize if requested (after log transform)
                arr_norm = self._normalize_field(arr, field_spec, grid_name)

                # Create grid
                vdb_grid_name = f"{self.grid_name_prefix}{grid_name}" if self.grid_name_prefix else grid_name
                grid = self._create_vdb_grid(arr_norm, vdb_grid_name, self.voxel_size)

                # Add metadata
                grid.metadata = {
                    'step': step,
                    'time': t,
                    'field': grid_name,
                    'field_type': field_spec.field_type,
                    'resolution': self.resolution,
                    'normalized': self.normalize,
                    'log_scaled': self.log_scale
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
            print(f"Fields exported: {list(self.frames[0]['fields'].keys())}")
            if self.log_scale:
                print(f"Log transform: enabled (dynamic range compression)")
            if self.normalize:
                print(f"Normalization: enabled")
                for grid_name, max_val in self._global_max.items():
                    field_spec = self.frames[0]['field_specs'][grid_name]
                    range_str = "[0, 1]" if field_spec.field_type == 'magnitude' else "[-1, 1]"
                    print(f"  {grid_name}: max {max_val:.2e} → {range_str}")

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
        for field_spec in self.field_specs:
            try:
                # Handle magnitude fields
                if field_spec.field_type == 'magnitude':
                    arr = self._compute_magnitude_field(fields_obj, field_spec.base_field)
                # Handle component fields
                else:
                    if not hasattr(fields_obj, field_spec.component):
                        print(f"Warning: Field '{field_spec.component}' not found, skipping")
                        continue
                    field_array = getattr(fields_obj, field_spec.component)
                    arr = np.asarray(field_array, dtype=np.float32)

                    if arr.ndim != 3:
                        print(f"Warning: Field '{field_spec.component}' is not 3D, skipping")
                        continue

                # Apply log transform if enabled
                if self.log_scale:
                    arr = self._apply_log_transform(arr, is_magnitude=(field_spec.field_type == 'magnitude'))

                # Normalize if requested
                if self.normalize:
                    if field_spec.field_type == 'magnitude':
                        # [0, 1] normalization
                        min_val, max_val = np.min(arr), np.max(arr)
                        if max_val > min_val:
                            if self.clamp_percentile < 100:
                                arr = np.clip(arr, min_val, max_val)
                            arr = (arr - min_val) / (max_val - min_val)
                    else:
                        # [-1, 1] normalization
                        max_val = np.percentile(np.abs(arr), self.clamp_percentile)
                        if max_val > 0:
                            arr = np.clip(arr, -max_val, max_val) / max_val

                grid_name = f"{self.grid_name_prefix}{field_spec.grid_name}" if self.grid_name_prefix else field_spec.grid_name
                grid = self._create_vdb_grid(arr, grid_name, voxel_size)
                grid.metadata = {
                    'step': step,
                    'time': t,
                    'field': field_spec.grid_name,
                    'field_type': field_spec.field_type,
                    'resolution': voxel_size,
                    'log_scaled': self.log_scale
                }
                grids.append(grid)

            except Exception as e:
                print(f"Warning: Failed to export field '{field_spec.grid_name}': {e}")
                continue

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
