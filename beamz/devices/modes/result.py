"""Mode-solver result container and user-facing post-processing helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .constants import C_0

_SPATIAL_DIMS = ("x", "y", "z")
_E_COMPONENTS = ("Ex", "Ey", "Ez")
_H_COMPONENTS = ("Hx", "Hy", "Hz")
_OVERLAP_KINDS = {"electric", "power", "lorentz"}


# Result is the post-processing boundary: solver outputs are immutable xarray
# arrays, and convenience methods derive metrics without mutating fields.
@dataclass(frozen=True)
class Result:
    """Mode solver result data.

    The object intentionally stays small: it stores complex effective indices
    and xarray-backed field components, then exposes the result-level helpers
    users tend to reach for first.
    """

    n_complex: xr.DataArray
    field_components: dict[str, xr.DataArray]
    n_group: xr.DataArray | None = None
    dispersion: xr.DataArray | None = None
    solver_info: dict[str, Any] | None = None

    @property
    def n_eff(self) -> xr.DataArray:
        """Real effective index."""

        # Keep n_complex as the source of truth; derived views preserve xarray
        # coordinates for frequency and mode index.
        return self.n_complex.real

    @property
    def k_eff(self) -> xr.DataArray:
        """Imaginary part of the complex effective index."""

        # The imaginary part is used by loss calculations and release summaries.
        return self.n_complex.imag

    @cached_property
    def pol_fraction(self) -> xr.Dataset:
        """TE/TM fraction from the two electric components in the mode plane."""

        tangential_dims = self._tangential_dims()
        if len(tangential_dims) != 2:
            raise ValueError("exactly two tangential field dimensions are required")

        # The two transverse electric components define the TE/TM split in the
        # local mode plane.
        first, second = (f"E{dim}" for dim in tangential_dims)
        self._require_components((first, second))

        # Integrate component intensities over the spatial grid, then normalize
        # to avoid amplitude-dependent fractions.
        first_power = self._integrated_power(first)
        second_power = self._integrated_power(second)
        total = np.maximum(first_power + second_power, np.finfo(float).eps)
        te = first_power / total
        return self._fraction_dataset(te, 1.0 - te)

    @cached_property
    def pol_fraction_waveguide(self) -> xr.Dataset:
        """Waveguide TE/TM fractions using normal E and H field components.

        The mode plane is inferred from the single spatial dimension with size
        one. For a z-normal plane, this computes ``1 - |Ez|^2 / |E|^2`` and
        ``1 - |Hz|^2 / |H|^2`` after integrating over the spatial grid.
        """

        normal_dim = self._normal_dim()
        normal_e = f"E{normal_dim}"
        normal_h = f"H{normal_dim}"
        self._require_components((*_E_COMPONENTS, *_H_COMPONENTS))

        # Waveguide fractions measure how much total E/H energy is transverse to
        # the propagation normal.
        total_e = sum(self._integrated_power(component) for component in _E_COMPONENTS)
        total_h = sum(self._integrated_power(component) for component in _H_COMPONENTS)
        te = 1.0 - self._integrated_power(normal_e) / np.maximum(
            total_e, np.finfo(float).eps
        )
        tm = 1.0 - self._integrated_power(normal_h) / np.maximum(
            total_h, np.finfo(float).eps
        )
        return self._fraction_dataset(te, tm)

    @cached_property
    def mode_area(self) -> xr.DataArray:
        """Effective mode area from electric-field intensity."""

        self._require_components(_E_COMPONENTS)

        # Compute |E|^2 on the full grid before applying area weights.
        intensity = sum(
            np.abs(self.field_components[name].values) ** 2 for name in _E_COMPONENTS
        )
        axes = self._spatial_axes(self.field_components["Ex"])
        weights = self._spatial_weights(self.field_components["Ex"])

        # Standard effective-area definition: (integral I dA)^2 /
        # integral(I^2 dA), guarded against empty or zero fields.
        numerator = np.sum(intensity * weights, axis=axes) ** 2
        denominator = np.maximum(
            np.sum(intensity**2 * weights, axis=axes), np.finfo(float).eps
        )
        return self._mode_data_array(numerator / denominator)

    @cached_property
    def modes_info(self) -> xr.Dataset:
        """Tabular mode metrics as an xarray dataset."""

        freq = self.n_complex.coords["f"]
        wavelength = xr.DataArray(C_0 / freq.values, dims=("f",), coords={"f": freq})
        wavelength_cm = wavelength / 1e4

        # Start with metrics that require no optional field components.
        metrics: dict[str, xr.DataArray] = {
            "wavelength": wavelength,
            "n eff": self.n_eff,
            "k eff": self.k_eff,
            "loss (dB/cm)": 20
            * 2
            * np.pi
            * np.log10(np.e)
            * self.k_eff
            / wavelength_cm,
        }

        # Optional arrays are preserved when callers provide group index or
        # dispersion calculations externally.
        if self.n_group is not None:
            metrics["group index"] = self.n_group
        if self.dispersion is not None:
            metrics["dispersion"] = self.dispersion

        # Field-derived metrics are best-effort because users may request only a
        # subset of components from solve_modes.
        self._add_optional_metric(
            metrics, "TE fraction", lambda: self.pol_fraction["te"]
        )
        self._add_optional_metric(
            metrics, "TM fraction", lambda: self.pol_fraction["tm"]
        )
        self._add_optional_metric(
            metrics, "wg TE fraction", lambda: self.pol_fraction_waveguide["te"]
        )
        self._add_optional_metric(
            metrics, "wg TM fraction", lambda: self.pol_fraction_waveguide["tm"]
        )
        self._add_optional_metric(metrics, "mode area", lambda: self.mode_area)
        return xr.Dataset(metrics)

    def to_dataframe(self):
        """Return mode metrics as a pandas DataFrame indexed by frequency and mode."""

        return self.modes_info.to_dataframe()

    def plot_field(
        self,
        component: str = "Ex",
        *,
        f: int | float = 0,
        mode_index: int = 0,
        ax: Any | None = None,
        val: str = "real",
        cmap: str | None = None,
        colorbar: bool = True,
        **imshow_kwargs: Any,
    ) -> Any:
        """Plot one field component on the two-dimensional mode plane."""

        if component not in self.field_components:
            raise ValueError(f"field component {component!r} is not available")
        import matplotlib.pyplot as plt

        # Select a single frequency/mode slice before deciding how many spatial
        # dimensions need to be shown.
        data_array = self._select_field(component, f=f, mode_index=mode_index)
        plane_dims = [
            dim
            for dim in _SPATIAL_DIMS
            if dim in data_array.dims and data_array.sizes[dim] > 1
        ]
        if len(plane_dims) not in {1, 2}:
            raise ValueError(
                "field plotting requires one or two non-singleton spatial dimensions"
            )

        # Put spatial axes in canonical x/y/z order so 1D and 2D plotting code
        # does not depend on solver normal direction.
        data_array = data_array.transpose(
            *[dim for dim in _SPATIAL_DIMS if dim in data_array.dims]
        )
        values = np.asarray(data_array.values).squeeze()

        # Convert complex fields to the requested real-valued plotting channel.
        if val == "real":
            values = np.real(values)
            default_cmap = "RdBu_r"
        elif val in {"abs", "magnitude"}:
            values = np.abs(values)
            default_cmap = "magma"
        elif val == "imag":
            values = np.imag(values)
            default_cmap = "RdBu_r"
        else:
            raise ValueError("val must be one of 'real', 'imag', or 'abs'")

        if ax is None:
            _, ax = plt.subplots()

        # Thin slice results become a line plot along the one non-singleton axis.
        if len(plane_dims) == 1:
            coord = np.asarray(data_array.coords[plane_dims[0]].values)
            ax.plot(coord, values)
            ax.set_xlabel(f"{plane_dims[0]} (um)")
            ax.set_ylabel(component)
            ax.set_title(
                f"{component}, f={self._selected_frequency(f):.6g}, mode={mode_index}"
            )
            return ax

        # Full 2D mode planes render with physical coordinate extents.
        x_coord = np.asarray(data_array.coords[plane_dims[0]].values)
        y_coord = np.asarray(data_array.coords[plane_dims[1]].values)
        extent = [x_coord.min(), x_coord.max(), y_coord.min(), y_coord.max()]
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap or default_cmap,
            **imshow_kwargs,
        )
        ax.set_xlabel(f"{plane_dims[0]} (um)")
        ax.set_ylabel(f"{plane_dims[1]} (um)")
        ax.set_title(
            f"{component}, f={self._selected_frequency(f):.6g}, mode={mode_index}"
        )
        if colorbar:
            ax.figure.colorbar(image, ax=ax)
        return ax

    def plot_field_components(
        self,
        *,
        f: int | float = 0,
        mode_index: int = 0,
        components: Iterable[str] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        val: str = "real",
        **imshow_kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot a grid of field components and return ``(fig, axes)``."""

        import matplotlib.pyplot as plt

        # Plot only requested components that are present in this Result.
        available = [
            component for component in components if component in self.field_components
        ]
        if not available:
            raise ValueError("none of the requested field components are available")

        # Use a compact grid with up to three columns and hide unused axes.
        ncols = min(3, len(available))
        nrows = int(np.ceil(len(available) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, squeeze=False, figsize=(4.0 * ncols, 3.2 * nrows)
        )
        flat_axes = axes.ravel()
        for ax, component in zip(flat_axes, available, strict=False):
            self.plot_field(
                component, f=f, mode_index=mode_index, ax=ax, val=val, **imshow_kwargs
            )
        for ax in flat_axes[len(available) :]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig, axes

    def plot(self, **kwargs: Any) -> tuple[Any, Any]:
        """Alias for :meth:`plot_field_components`."""

        return self.plot_field_components(**kwargs)

    def overlap(
        self,
        other: Result | None = None,
        *,
        mode_index: int = 0,
        other_mode_index: int | None = None,
        f: int | float = 0,
        other_f: int | float | None = None,
        kind: str = "power",
        normalize: bool = True,
    ) -> complex:
        """Return the overlap between two modes.

        ``kind="power"`` computes the transverse power-product
        ``integral((E_a x H_b*) . n) dA``. ``kind="lorentz"`` computes the
        unconjugated reciprocal product used by the solver orthogonalization
        pass. ``kind="electric"`` computes a simpler electric-field inner
        product for mode tracking. All overlap kinds use the mode-plane cell
        widths as integration weights and require matching result grids.
        """

        other = self if other is None else other
        other_mode_index = mode_index if other_mode_index is None else other_mode_index
        other_f = f if other_f is None else other_f

        # Extract exactly one mode from each result before computing the selected
        # overlap product.
        selected_self = self._selected_mode_fields(f=f, mode_index=mode_index)
        selected_other = other._selected_mode_fields(
            f=other_f, mode_index=other_mode_index
        )
        value = _overlap_value(self, selected_self, other, selected_other, kind=kind)
        if not normalize:
            return value
        # Normalize by the two self-overlaps so mode tracking compares shape and
        # phase alignment, not arbitrary field amplitude.
        self_norm = _overlap_value(self, selected_self, self, selected_self, kind=kind)
        other_norm = _overlap_value(
            other, selected_other, other, selected_other, kind=kind
        )
        denom = np.sqrt(abs(self_norm) * abs(other_norm))
        if denom <= np.finfo(float).eps:
            return 0.0 + 0.0j
        return complex(value / denom)

    def overlap_matrix(
        self,
        other: Result | None = None,
        *,
        f: int | float = 0,
        other_f: int | float | None = None,
        kind: str = "power",
        normalize: bool = True,
    ) -> xr.DataArray:
        """Return pairwise overlaps between all modes at one frequency."""

        other = self if other is None else other
        other_f = f if other_f is None else other_f

        # Fill the dense pairwise matrix by delegating to the single-overlap
        # method so normalization and validation behavior stays identical.
        values = np.empty(
            (self.n_complex.sizes["mode_index"], other.n_complex.sizes["mode_index"]),
            dtype=np.complex128,
        )
        for left_index in range(values.shape[0]):
            for right_index in range(values.shape[1]):
                values[left_index, right_index] = self.overlap(
                    other,
                    mode_index=left_index,
                    other_mode_index=right_index,
                    f=f,
                    other_f=other_f,
                    kind=kind,
                    normalize=normalize,
                )
        return xr.DataArray(
            values,
            dims=("mode_index", "other_mode_index"),
            coords={
                "mode_index": self.n_complex.coords["mode_index"].values,
                "other_mode_index": other.n_complex.coords["mode_index"].values,
            },
        )

    def to_hdf5(self, path: str | Path) -> Path:
        """Save the result to a compact HDF5 file."""

        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency path.
            raise ImportError(
                "Result.to_hdf5() requires the `beamz[mode-io]` extra"
            ) from exc

        destination = Path(path)
        with h5py.File(destination, "w") as handle:
            # Store a tiny file-level marker before writing xarray-shaped groups.
            handle.attrs["format"] = "beamz.devices.modes.Result"
            handle.attrs["version"] = 1
            self._write_data_array(handle, "n_complex", self.n_complex)

            # Optional scalar arrays are omitted when absent rather than storing
            # empty sentinel datasets.
            if self.n_group is not None:
                self._write_data_array(handle, "n_group", self.n_group)
            if self.dispersion is not None:
                self._write_data_array(handle, "dispersion", self.dispersion)
            if self.solver_info is not None:
                handle.attrs["solver_info"] = json.dumps(_json_safe(self.solver_info))

            # Field components live under one group so readers can discover all
            # available components without knowing what the solve requested.
            fields_group = handle.create_group("field_components")
            for component, data_array in self.field_components.items():
                self._write_data_array(fields_group, component, data_array)
        return destination

    @classmethod
    def from_hdf5(cls, path: str | Path) -> Result:
        """Load a :class:`Result` saved with :meth:`to_hdf5`."""

        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency path.
            raise ImportError(
                "Result.from_hdf5() requires the `beamz[mode-io]` extra"
            ) from exc

        with h5py.File(path, "r") as handle:
            # Rehydrate required and optional DataArrays from the compact group
            # layout written by to_hdf5.
            n_complex = cls._read_data_array(handle["n_complex"])
            n_group = (
                cls._read_data_array(handle["n_group"]) if "n_group" in handle else None
            )
            dispersion = (
                cls._read_data_array(handle["dispersion"])
                if "dispersion" in handle
                else None
            )
            solver_info = (
                _loads_hdf5_json_attr(handle.attrs["solver_info"])
                if "solver_info" in handle.attrs
                else None
            )
            field_group: Any = handle["field_components"]

            # Preserve whatever subset of fields was saved.
            field_components = {
                component: cls._read_data_array(group)
                for component, group in field_group.items()
            }
        return cls(
            n_complex=n_complex,
            field_components=field_components,
            n_group=n_group,
            dispersion=dispersion,
            solver_info=solver_info,
        )

    def _require_components(self, components: Iterable[str]) -> None:
        """Raise if required field components are absent."""
        # Centralize missing-component errors so derived metrics and overlaps use
        # consistent messages.
        missing = [
            component
            for component in components
            if component not in self.field_components
        ]
        if missing:
            raise ValueError(
                f"field component(s) required but missing: {', '.join(missing)}"
            )

    def _normal_dim(self) -> str:
        """Infer the singleton propagation-normal dimension."""
        reference = self._reference_field()

        # Solver-created results annotate the normal dimension explicitly.
        attr_normal = reference.attrs.get("normal_dim")
        if attr_normal in _SPATIAL_DIMS and attr_normal in reference.dims:
            return str(attr_normal)

        # External Results may not carry attrs, so fall back to singleton spatial
        # dimensions and then the smallest spatial dimension.
        spatial_dims = [dim for dim in _SPATIAL_DIMS if dim in reference.dims]
        singleton_dims = [dim for dim in spatial_dims if reference.sizes[dim] == 1]
        if len(singleton_dims) == 1:
            return singleton_dims[0]
        if spatial_dims:
            return min(spatial_dims, key=lambda dim: reference.sizes[dim])
        raise ValueError("fields must contain at least one spatial dimension")

    def _tangential_dims(self) -> tuple[str, ...]:
        """Return the spatial dimensions transverse to propagation."""
        normal = self._normal_dim()
        reference = self._reference_field()
        return tuple(
            dim for dim in _SPATIAL_DIMS if dim in reference.dims and dim != normal
        )

    def _reference_field(self) -> xr.DataArray:
        """Return one field array used for dimension inference."""
        if not self.field_components:
            raise ValueError("at least one field component is required")
        return next(iter(self.field_components.values()))

    def _spatial_axes(self, data_array: xr.DataArray) -> tuple[int, ...]:
        """Return axis indices corresponding to spatial dimensions."""
        return tuple(
            axis for axis, dim in enumerate(data_array.dims) if dim in _SPATIAL_DIMS
        )

    def _integrated_power(self, component: str) -> np.ndarray:
        """Integrate squared field magnitude over the spatial grid."""
        # Used by mode metrics such as polarization fraction and mode area. The
        # integration weights come from the mode-plane cell widths stored on each
        # field DataArray.
        data_array = self.field_components[component]
        return np.sum(
            np.abs(data_array.values) ** 2 * self._spatial_weights(data_array),
            axis=self._spatial_axes(data_array),
        )

    def _spatial_weights(self, data_array: xr.DataArray) -> np.ndarray:
        """Build broadcastable cell-area weights from width coordinates."""
        # Prefer explicit width coordinates written by the solver. Fall back to
        # midpoint-derived widths so externally constructed Results still work.
        weights: np.ndarray | float = 1.0
        for axis, dim in enumerate(data_array.dims):
            if dim not in _SPATIAL_DIMS:
                continue
            width_coord = f"{dim}_width"
            if width_coord in data_array.coords:
                # Width coordinates are written by solve_modes and preserve the
                # original nonuniform grid exactly.
                widths = np.asarray(data_array.coords[width_coord].values, dtype=float)
            else:
                widths = self._cell_widths(
                    np.asarray(data_array.coords[dim].values, dtype=float)
                )

            # Reshape each 1D width vector so NumPy broadcasting multiplies it
            # along the matching spatial axis only.
            shape = [1] * data_array.ndim
            shape[axis] = len(widths)
            weights = weights * widths.reshape(shape)
        return np.asarray(weights)

    @staticmethod
    def _cell_widths(values: np.ndarray) -> np.ndarray:
        """Infer cell widths from coordinate midpoints."""
        # A singleton dimension represents the propagation normal and contributes
        # unit width to area/volume weights.
        if values.size <= 1:
            return np.ones(values.shape, dtype=float)

        # Reconstruct edges halfway between adjacent centers, with endpoint
        # extrapolation matching the nearest cell spacing.
        midpoints = 0.5 * (values[1:] + values[:-1])
        edges = np.concatenate(
            (
                [values[0] - 0.5 * (values[1] - values[0])],
                midpoints,
                [values[-1] + 0.5 * (values[-1] - values[-2])],
            )
        )
        return np.abs(np.diff(edges))

    def _fraction_dataset(self, te: np.ndarray, tm: np.ndarray) -> xr.Dataset:
        """Package TE/TM fractions as a dataset aligned to modes."""
        return xr.Dataset(
            {"te": self._mode_data_array(te), "tm": self._mode_data_array(tm)}
        )

    def _mode_data_array(self, values: np.ndarray) -> xr.DataArray:
        """Wrap per-mode values with the n_complex coordinates."""
        return xr.DataArray(
            values, dims=self.n_complex.dims, coords=self.n_complex.coords
        )

    def _add_optional_metric(
        self,
        metrics: dict[str, xr.DataArray],
        name: str,
        getter: Any,
    ) -> None:
        """Add a metric if its required field components are available."""
        # Missing field components raise ValueError in the metric getter; skip
        # those metrics while letting unexpected exceptions surface.
        try:
            metrics[name] = getter()
        except ValueError:
            return

    def _select_field(
        self, component: str, *, f: int | float, mode_index: int
    ) -> xr.DataArray:
        """Select one component at a frequency and mode index."""
        # Integer f means positional selection; float f means nearest frequency
        # coordinate, which is friendlier for wavelength-derived frequencies.
        data_array = self.field_components[component]
        data_array = (
            data_array.isel(f=f)
            if isinstance(f, int)
            else data_array.sel(f=f, method="nearest")
        )
        return data_array.isel(mode_index=mode_index)

    def _selected_mode_fields(
        self, *, f: int | float, mode_index: int
    ) -> dict[str, xr.DataArray]:
        """Select all available fields for one mode."""
        return {
            component: self._select_field(component, f=f, mode_index=mode_index)
            for component in self.field_components
        }

    def _selected_frequency(self, f: int | float) -> float:
        """Return the numeric frequency selected by index or nearest value."""
        freq = self.n_complex.coords["f"]
        if isinstance(f, int):
            return float(freq.values[f])
        return float(freq.sel(f=f, method="nearest").values)

    @staticmethod
    def _write_data_array(parent: Any, name: str, data_array: xr.DataArray) -> None:
        """Write an xarray DataArray to the compact HDF5 layout."""
        group = parent.create_group(name)

        # Store dimensions as metadata and raw values as a single dataset.
        group.attrs["dims"] = json.dumps(list(data_array.dims))
        for attr_name, attr_value in data_array.attrs.items():
            if isinstance(
                attr_value, (str, int, float, bool, np.integer, np.floating, np.bool_)
            ):
                group.attrs[f"attr:{attr_name}"] = attr_value
        group.create_dataset("values", data=data_array.values)

        # Coordinates keep their own dimensions because xarray coordinates can be
        # scalar, 1D, or attached to nonmatching axes.
        coords = group.create_group("coords")
        for coord_name, coord in data_array.coords.items():
            coord_values = np.asarray(coord.values)
            dataset = coords.create_dataset(coord_name, data=coord_values)
            dataset.attrs["dims"] = json.dumps(list(coord.dims))

    @staticmethod
    def _read_data_array(group: Any) -> xr.DataArray:
        """Read an xarray DataArray from the compact HDF5 layout."""
        # Rebuild attrs and coordinates first, then hand them to xarray with the
        # stored value dataset.
        dims = tuple(json.loads(group.attrs["dims"]))
        attrs = {
            key.removeprefix("attr:"): value
            for key, value in group.attrs.items()
            if isinstance(key, str) and key.startswith("attr:")
        }
        coords = {}
        for coord_name, dataset in group["coords"].items():
            coord_dims = tuple(json.loads(dataset.attrs["dims"]))
            values = dataset[()]
            coords[coord_name] = (coord_dims, values)
        return xr.DataArray(group["values"][()], dims=dims, coords=coords, attrs=attrs)


def overlap(
    left: Result,
    right: Result | None = None,
    **kwargs: Any,
) -> complex:
    """Convenience wrapper for :meth:`Result.overlap`."""

    return left.overlap(right, **kwargs)


def _json_safe(value: Any) -> Any:
    """Convert NumPy and complex values into JSON-safe objects."""
    # solver_info may contain NumPy arrays/scalars, complex values, and nested
    # containers; JSON attributes require plain Python data.
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        complex_value = complex(value)
        return {"real": complex_value.real, "imag": complex_value.imag}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    # Lists and tuples are both serialized as JSON arrays.
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _loads_hdf5_json_attr(value: Any) -> Any:
    """Decode a JSON HDF5 attribute stored as bytes or text."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def _overlap_value(
    left_result: Result,
    left: dict[str, xr.DataArray],
    right_result: Result,
    right: dict[str, xr.DataArray],
    *,
    kind: str,
) -> complex:
    """Compute one unnormalized field overlap integral."""
    # Validate the requested overlap family before checking required components.
    if kind not in _OVERLAP_KINDS:
        raise ValueError("kind must be 'electric', 'power', or 'lorentz'")
    if kind == "electric":
        # Electric overlap is a simpler field-shape metric. It is useful for
        # mode tracking when magnetic components are unavailable or not needed.
        missing = [
            component
            for component in _E_COMPONENTS
            if component not in left or component not in right
        ]
        if missing:
            raise ValueError(
                f"electric overlap requires field component(s): {', '.join(missing)}"
            )
        _validate_overlap_grid(left_result, left, right_result, right)

        # Electric overlap is a weighted dot product across all electric
        # components.
        weights = left_result._spatial_weights(left["Ex"])
        integrand = sum(
            left[component].values * np.conj(right[component].values)
            for component in _E_COMPONENTS
        )
        return complex(np.sum(integrand * weights))

    # Power and Lorentz overlaps require the complete six-component mode. Power
    # uses H* and measures physical flux. Lorentz deliberately does not
    # conjugate either mode; it is the reciprocal product used to orthogonalize
    # the mode set.
    missing = [
        component
        for component in (*_E_COMPONENTS, *_H_COMPONENTS)
        if component not in left or component not in right
    ]
    if missing:
        raise ValueError(
            f"{kind} overlap requires field component(s): {', '.join(missing)}"
        )
    _validate_overlap_grid(left_result, left, right_result, right)
    weights = left_result._spatial_weights(left["Ex"])
    normal = left_result._normal_dim()

    # Dispatch to the cross-product component aligned with the mode normal.
    if kind == "lorentz":
        integrand = _normal_lorentz_integrand(left, right, normal)
    else:
        integrand = _normal_power_integrand(left, right, normal)
    return complex(np.sum(integrand * weights))


def _normal_power_integrand(
    left: dict[str, xr.DataArray],
    right: dict[str, xr.DataArray],
    normal: str,
) -> np.ndarray:
    """Return the normal Poynting-flux integrand."""
    # Compute (E_left x H_right*) . n for whichever global axis is normal to the
    # mode plane.
    if normal == "x":
        return left["Ey"].values * np.conj(right["Hz"].values) - left[
            "Ez"
        ].values * np.conj(right["Hy"].values)
    if normal == "y":
        return left["Ez"].values * np.conj(right["Hx"].values) - left[
            "Ex"
        ].values * np.conj(right["Hz"].values)
    if normal == "z":
        return left["Ex"].values * np.conj(right["Hy"].values) - left[
            "Ey"
        ].values * np.conj(right["Hx"].values)
    raise ValueError("fields must contain a valid singleton normal dimension")


def _normal_lorentz_integrand(
    left: dict[str, xr.DataArray],
    right: dict[str, xr.DataArray],
    normal: str,
) -> np.ndarray:
    """Return the symmetrized unconjugated Lorentz integrand."""
    # Lorentz reciprocity uses both E_left x H_right and E_right x H_left.
    left_cross_right = _normal_unconjugated_cross_integrand(left, right, normal)
    right_cross_left = _normal_unconjugated_cross_integrand(right, left, normal)
    return 0.5 * (left_cross_right + right_cross_left)


def _normal_unconjugated_cross_integrand(
    left: dict[str, xr.DataArray],
    right: dict[str, xr.DataArray],
    normal: str,
) -> np.ndarray:
    """Return one unconjugated cross-product component."""
    if normal == "x":
        return (
            left["Ey"].values * right["Hz"].values
            - left["Ez"].values * right["Hy"].values
        )
    if normal == "y":
        return (
            left["Ez"].values * right["Hx"].values
            - left["Ex"].values * right["Hz"].values
        )
    if normal == "z":
        return (
            left["Ex"].values * right["Hy"].values
            - left["Ey"].values * right["Hx"].values
        )
    raise ValueError("fields must contain a valid singleton normal dimension")


def _validate_overlap_grid(
    left_result: Result,
    left: dict[str, xr.DataArray],
    right_result: Result,
    right: dict[str, xr.DataArray],
) -> None:
    """Ensure two modes live on compatible spatial grids."""
    # Overlap values are meaningful only when both fields use the same normal and
    # spatial coordinate system.
    if left_result._normal_dim() != right_result._normal_dim():
        raise ValueError("mode overlap requires matching mode-plane normal dimensions")
    reference_left = left["Ex"]
    reference_right = right["Ex"]

    # Dimension order must match before comparing coordinate values.
    if reference_left.dims != reference_right.dims:
        raise ValueError("mode overlap requires matching field dimensions")
    for dim in reference_left.dims:
        if dim not in _SPATIAL_DIMS:
            continue

        # Compare each spatial coordinate axis with tolerance to allow harmless
        # floating-point roundoff in externally constructed Results.
        left_coord = np.asarray(reference_left.coords[dim].values, dtype=float)
        right_coord = np.asarray(reference_right.coords[dim].values, dtype=float)
        if left_coord.shape != right_coord.shape or not np.allclose(
            left_coord, right_coord
        ):
            raise ValueError("mode overlap requires matching spatial grids")
