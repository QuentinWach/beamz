"""FDTD field plotting and power visualization."""

import numpy as np

from beamz.visual.helpers import get_si_scale_and_label


def plot_fdtd_field(
    fdtd, field: str = "Ez", t: float = None, z_slice: int = None
) -> None:
    """Plot an FDTD field at a given time with proper scaling and units."""
    import matplotlib.pyplot as plt

    if len(fdtd.results["t"]) == 0:
        current_field = getattr(fdtd, field)
        current_t = fdtd.t
    else:
        t_idx = int(np.argmin(np.abs(np.array(fdtd.results["t"]) - t)))
        current_field = fdtd.results[field][t_idx]
        current_t = fdtd.results["t"][t_idx]

    if hasattr(current_field, "device"):
        current_field = fdtd.backend.to_numpy(current_field)

    if fdtd.is_3d and len(current_field.shape) == 3:
        if z_slice is None:
            z_slice = current_field.shape[0] // 2
        current_field = current_field[z_slice, :, :]
        slice_info = f" (z-slice {z_slice})"
    else:
        slice_info = ""

    if np.iscomplexobj(current_field):
        current_field = np.real(current_field)
        field_label = f"Re({field}){slice_info}"
    else:
        field_label = field + slice_info

    scale, unit = get_si_scale_and_label(max(fdtd.design.width, fdtd.design.height))
    grid_height, grid_width = current_field.shape
    aspect_ratio = grid_width / grid_height
    base_size = 6
    figsize = (
        (base_size * aspect_ratio, base_size)
        if aspect_ratio > 1
        else (base_size, base_size / aspect_ratio)
    )

    plt.figure(figsize=figsize)
    plt.imshow(
        current_field,
        origin="lower",
        extent=(0, fdtd.design.width, 0, fdtd.design.height),
        cmap="RdBu",
        aspect="equal",
        interpolation="bicubic",
    )
    plt.colorbar(label=f"{field_label}")
    plt.title(f"{field_label} Field at t = {current_t:.2e} s")
    plt.xlabel(f"X ({unit})")
    plt.ylabel(f"Y ({unit})")
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")

    try:
        tmp_design = fdtd.design.copy()
        tmp_design.unify_polygons()
        overlay_structures = tmp_design.structures
    except Exception:
        overlay_structures = fdtd.design.structures
    for structure in overlay_structures:
        if hasattr(structure, "is_pml") and structure.is_pml:
            structure.add_to_plot(
                plt.gca(),
                edgecolor="black",
                linestyle="--",
                facecolor="none",
                alpha=0.5,
            )
        elif hasattr(structure, "vertices") and getattr(structure, "vertices", None):
            structure.add_to_plot(
                plt.gca(), facecolor="none", edgecolor="black", linestyle="-"
            )
    for source in fdtd.design.sources:
        if hasattr(source, "add_to_plot"):
            source.add_to_plot(plt.gca())
    for monitor in fdtd.design.monitors:
        if hasattr(monitor, "add_to_plot"):
            monitor.add_to_plot(plt.gca(), edgecolor="black")
    plt.tight_layout()
    plt.show()

def plot_fdtd_power(
    fdtd,
    cmap: str = "hot",
    vmin: float = None,
    vmax: float = None,
    db_colorbar: bool = False,
):
    """Plot time-integrated power distribution from FDTD fields."""
    import matplotlib.pyplot as plt

    if fdtd.power_accumulated is not None:
        power = fdtd.power_accumulated
        print("Using accumulated power data")
    elif (
        len(fdtd.results["Ez"]) > 0
        and len(fdtd.results["Hx"]) > 0
        and len(fdtd.results["Hy"]) > 0
    ):
        print("Calculating power from saved field data")
        power = np.zeros((fdtd.nx, fdtd.ny))
        for t_idx in range(len(fdtd.results["t"])):
            Ez = fdtd.results["Ez"][t_idx]
            Hx_raw = fdtd.results["Hx"][t_idx]
            Hy_raw = fdtd.results["Hy"][t_idx]
            is_complex = (
                np.iscomplexobj(Ez)
                or np.iscomplexobj(Hx_raw)
                or np.iscomplexobj(Hy_raw)
            )
            if np.iscomplexobj(Ez):
                Ez_real = np.real(Ez)
                Ez_imag = np.imag(Ez)
            else:
                Ez_real = Ez
                Ez_imag = np.zeros_like(Ez)
            if is_complex:
                Hx = np.zeros_like(Ez, dtype=np.complex128)
                Hy = np.zeros_like(Ez, dtype=np.complex128)
            else:
                Hx = np.zeros_like(Ez_real)
                Hy = np.zeros_like(Ez_real)
            Hx[:, :-1] = Hx_raw
            Hy[:-1, :] = Hy_raw
            if is_complex:
                Hx_real = np.real(Hx)
                Hx_imag = np.imag(Hx)
                Hy_real = np.real(Hy)
                Hy_imag = np.imag(Hy)
                Sx = -Ez_real * Hy_real - Ez_imag * Hy_imag
                Sy = Ez_real * Hx_real + Ez_imag * Hx_imag
            else:
                Sx = -Ez_real * Hy
                Sy = Ez_real * Hx
            power_mag = Sx**2 + Sy**2
            power += power_mag
        power /= len(fdtd.results["t"])
    else:
        print(
            "No field data to calculate power. Make sure to run the simulation with save=True or accumulate_power=True."
        )
        return

    # Normalize power using 99th percentile to avoid source-dominated colormaps
    # This makes propagated power visible by clipping source peaks
    power_sorted = np.sort(power.flatten())
    nonzero_power = power_sorted[power_sorted > 0]
    if len(nonzero_power) > 100:  # Need sufficient data points
        p99 = np.percentile(nonzero_power, 99)
        power_clipped = np.clip(power, 0, p99)
        if p99 > 0:
            power_normalized = power_clipped / p99
        else:
            power_normalized = power
    else:
        # Fallback to max normalization for small datasets
        power_max = np.max(power)
        if power_max > 0 and np.isfinite(power_max):
            power_normalized = power / power_max
        else:
            power_normalized = power

    scale, unit = get_si_scale_and_label(max(fdtd.design.width, fdtd.design.height))
    aspect_ratio = power.shape[1] / power.shape[0]
    base_size = 8
    figsize = (
        (base_size * aspect_ratio, base_size)
        if aspect_ratio > 1
        else (base_size, base_size / aspect_ratio)
    )

    fdtd.fig, fdtd.ax = plt.subplots(figsize=figsize)
    # Use normalized power for display to avoid numerical precision issues with tiny values
    display_power = power_normalized if vmin is None and vmax is None else power
    fdtd.im = fdtd.ax.imshow(
        display_power,
        origin="lower",
        extent=(0, fdtd.design.width, 0, fdtd.design.height),
        cmap=cmap,
        aspect="equal",
        interpolation="bicubic",
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = plt.colorbar(fdtd.im, orientation="vertical", aspect=30, extend="both")
    if db_colorbar:
        # dB scale now works on normalized power (0 to 1)
        def db_formatter(x, pos):
            if x <= 0:
                return "-∞ dB"
            ratio = max(x, 1e-10)  # x is already normalized to max=1
            db_val = 10 * np.log10(ratio)
            return f"{db_val:.1f} dB"

        colorbar.formatter = plt.FuncFormatter(db_formatter)
        colorbar.update_ticks()
        colorbar.set_label("Relative Power (dB)")
    else:
        colorbar.set_label("Normalized Power")

    try:
        tmp_design = fdtd.design.copy()
        tmp_design.unify_polygons()
        overlay_structures = tmp_design.structures
    except Exception:
        overlay_structures = fdtd.design.structures
    for structure in overlay_structures:
        if hasattr(structure, "is_pml") and structure.is_pml:
            structure.add_to_plot(
                fdtd.ax, edgecolor="white", linestyle="--", facecolor="none", alpha=0.5
            )
        elif hasattr(structure, "vertices") and getattr(structure, "vertices", None):
            structure.add_to_plot(
                fdtd.ax, facecolor="none", edgecolor="white", linestyle="-"
            )
    # Sources are not shown in power plot to avoid visual clutter
    for monitor in fdtd.design.monitors:
        if hasattr(monitor, "add_to_plot"):
            monitor.add_to_plot(fdtd.ax, edgecolor="white")

    plt.xlabel(f"X ({unit})")
    plt.ylabel(f"Y ({unit})")
    fdtd.ax.xaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    fdtd.ax.yaxis.set_major_formatter(lambda x, pos: f"{x*scale:.1f}")
    plt.title("Time-Averaged Power Distribution")
    plt.tight_layout()
    plt.show()

