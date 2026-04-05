from __future__ import annotations

import warnings

import numpy as np


def profile_data(source, field=None):
    """Return the current source profile as plotting-ready data."""
    from beamz.visual.data import Slice2D, Trace1D

    if source._Ez_profile is None and source._jz_profile is None:
        if source.grid is not None and hasattr(source.grid, "permittivity"):
            resolution = getattr(source.grid, "resolution", 0.05e-6)
            source.initialize(source.grid.permittivity, resolution)
        else:
            raise ValueError(
                "Mode source is not initialized and no grid permittivity is available."
            )

    choices = {
        "ez": ("Ez", source._Ez_profile),
        "hz": ("Hz", source._jz_profile),
        "jz": ("Hz", source._jz_profile),
    }
    key = None if field is None else str(field).strip().lower()
    if key is None:
        label, profile = ("Ez", source._Ez_profile)
        if profile is None:
            label, profile = ("Hz", source._jz_profile)
    elif key in choices:
        label, profile = choices[key]
    else:
        raise ValueError("field must be one of None, 'Ez', 'Hz', or 'Jz'.")

    if profile is None:
        raise ValueError(f"No profile data available for field '{field}'.")

    profile = np.squeeze(np.asarray(profile))
    title = f"{label} mode profile"
    if source._neff is not None:
        title = f"{title} (neff={source._neff:.4f})"

    if profile.ndim == 2:
        if source.direction in {"+x", "-x"}:
            plane, x_label, y_label = "yz", "Y index", "Z index"
        else:
            plane, x_label, y_label = "xz", "X index", "Z index"
        height, width = profile.shape
        return Slice2D(
            values=profile,
            extent=(0.0, float(max(width - 1, 1)), 0.0, float(max(height - 1, 1))),
            value_label="Amplitude",
            plane=plane,
            title=title,
            x_label=x_label,
            y_label=y_label,
            style={"cmap": "magma", "origin": "lower", "aspect": "auto"},
        )

    return Trace1D(
        values=profile.reshape(-1),
        coords=np.arange(profile.size, dtype=float),
        coord_label="index",
        value_label="Amplitude",
        title=title,
        style={"color": "black", "linewidth": 2},
    )


def show(source, field=None):
    """Visualize the 2D mode profile (for 3D simulations) or 1D profile (for 2D)."""
    import matplotlib.pyplot as plt
    from beamz.visual.data import Slice2D

    try:
        plot_data = profile_data(source, field=field)
    except ValueError as exc:
        warnings.warn(str(exc), stacklevel=2)
        return None
    if isinstance(plot_data, Slice2D):
        plot_data.plot(cmap="magma", abs_value=True, aspect="auto")
    else:
        ax = plot_data.plot(color="k", abs_value=True)
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    return plot_data


def add_to_plot(source, ax, facecolor="none", edgecolor="crimson", alpha=0.8, linestyle="-"):
    """Add source visualization to a 2D matplotlib plot."""
    from beamz.visual.overlays import add_mode_source_to_plot

    add_mode_source_to_plot(
        source,
        ax,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linestyle=linestyle,
    )
