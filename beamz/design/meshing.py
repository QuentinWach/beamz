from beamz.design import raster2d, raster3d
from beamz.design.grids import BaseMeshGrid
from beamz.visual.helpers import display_status


class RegularGrid(BaseMeshGrid):
    """2D regular-grid meshing for 2D designs."""

    def __init__(
        self,
        design,
        resolution,
        aa_mode="legacy_grid",
        aa_samples=64,
        aa_seed=0,
    ):
        super().__init__(design, resolution)
        self._configure_antialiasing(
            aa_mode=aa_mode,
            aa_samples=aa_samples,
            aa_seed=aa_seed,
        )

        if design.is_3d and design.depth > 0:
            display_status(
                "Warning: Using 2D RegularGrid for a 3D design. Use RegularGrid3D for proper 3D meshing.",
                "warning",
            )

        self.is_3d = design.is_3d and design.depth > 0
        self.__rasterize__()
        self.shape = self.permittivity.shape
        self.dx = self.resolution
        self.dy = self.resolution
        self.width = self.design.width
        self.height = self.design.height

    def __rasterize__(self):
        raster2d.rasterize(self)

    def slice2d(self, *, field="permittivity", title=None):
        """Return the 2D grid field as plotting-ready Slice2D data."""
        from beamz.visual.data import Slice2D

        values = getattr(self, field, None)
        if values is None:
            raise ValueError(f"Unknown field: {field!r}")
        return Slice2D(
            values=values,
            extent=(0.0, float(self.width), 0.0, float(self.height)),
            value_label=field,
            plane="xy",
            title=title or f"{field} slice (xy)",
        )

    def show(self, field="permittivity"):
        """Display the rasterized grid with properly scaled SI units."""
        self.slice2d(field=field).plot()


class RegularGrid3D(BaseMeshGrid):
    """3D regular-grid meshing for 3D designs."""

    def __init__(
        self,
        design,
        resolution_xy=None,
        resolution_z=None,
        aa_mode="legacy_grid",
        aa_samples=64,
        aa_seed=0,
    ):
        if isinstance(design, (int, float)) and resolution_xy is None:
            resolution = design
            design = resolution_xy
            resolution_xy = resolution
            resolution_z = resolution
        elif resolution_xy is None:
            resolution_xy = getattr(design, "resolution", resolution_xy)
            resolution_z = resolution_xy
        elif resolution_z is None:
            resolution_z = resolution_xy

        super().__init__(design, resolution_xy)
        self._configure_antialiasing(
            aa_mode=aa_mode,
            aa_samples=aa_samples,
            aa_seed=aa_seed,
        )

        self.resolution_xy = resolution_xy
        self.resolution_z = resolution_z
        self.__rasterize_3d__()

        width, height, depth = self.design.width, self.design.height, self.design.depth
        grid_width = int(width / self.resolution_xy)
        grid_height = int(height / self.resolution_xy)
        grid_depth = int(depth / self.resolution_z) if depth > 0 else 1

        self.shape = self.permittivity.shape
        self.dx = self.resolution_xy
        self.dy = self.resolution_xy
        self.dz = self.resolution_z
        self.width = width
        self.height = height
        self.depth = depth
        display_status(
            f"Created 3D mesh: {grid_width} × {grid_height} × {grid_depth} cells",
            "success",
        )

    def __rasterize_3d__(self):
        raster3d.rasterize(self)

    def get_2d_slice(self, z_index=None, z_position=None):
        """Extract a 2D slice from the 3D grid."""
        if z_index is None and z_position is None:
            z_index = self.shape[0] // 2
        elif z_position is not None:
            z_index = int(z_position / self.resolution_z)
            z_index = max(0, min(self.shape[0] - 1, z_index))

        return {
            "permittivity": self.permittivity[z_index, :, :],
            "permeability": self.permeability[z_index, :, :],
            "conductivity": self.conductivity[z_index, :, :],
        }

    def slice2d(self, *, field="permittivity", z_index=None, z_position=None, title=None):
        """Return a 2D material slice as plotting-ready Slice2D data."""
        from beamz.visual.data import Slice2D

        slice_data = self.get_2d_slice(z_index=z_index, z_position=z_position)
        if field not in slice_data:
            raise ValueError(f"Unknown field: {field!r}")

        if z_index is None and z_position is not None:
            z_index = int(float(z_position) / float(self.resolution_z))
        if z_index is None:
            z_index = self.shape[0] // 2
        z_index = max(0, min(int(z_index), self.shape[0] - 1))
        position = float(z_index) * float(self.resolution_z)
        return Slice2D(
            values=slice_data[field],
            extent=(0.0, float(self.width), 0.0, float(self.height)),
            value_label=field,
            plane="xy",
            position=position,
            title=title or f"{field} slice (xy, z={position:.3e} m)",
        )

    def show_3d(self, field="permittivity", slice_spacing=1, alpha=0.3):
        """Display a 3D visualization of the mesh."""
        from beamz.visual.overlays import show_mesh_3d

        grid = getattr(self, field, None)
        if grid is None:
            raise ValueError(f"Unknown field: {field}")
        show_mesh_3d(grid, self.design, field, slice_spacing, alpha)

    def show(self, field="permittivity", z_index=None, z_position=None):
        """Display a 2D slice of the 3D mesh."""
        self.slice2d(field=field, z_index=z_index, z_position=z_position).plot()


def create_mesh(design, resolution, auto_select=True, force_3d=False, **kwargs):
    """Create a mesh by selecting 2D or 3D based on design properties."""
    resolution_z = kwargs.pop("resolution_z", None)
    if force_3d or (auto_select and design.is_3d and design.depth > 0):
        display_status("Auto-selecting 3D meshing for 3D design", "info")
        return RegularGrid3D(
            design,
            resolution_xy=resolution,
            resolution_z=resolution_z,
            **kwargs,
        )

    if auto_select and design.is_3d:
        display_status(
            "Auto-selecting 2D meshing for effectively 2D design (depth=0)", "info"
        )
    return RegularGrid(design, resolution, **kwargs)
