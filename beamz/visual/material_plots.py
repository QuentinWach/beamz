"""Material plotting utilities used by material model `.show()` methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from beamz.const import LIGHT_SPEED

if TYPE_CHECKING:
    from beamz.design.materials import (
        AnisotropicMaterial,
        CustomMaterial,
        Material,
        Material2D,
        _DispersiveBase,
    )


UM = 1e-6


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Material visualization requires matplotlib. Install with `pip install matplotlib`."
        ) from exc
    return plt


def _sample_wavelengths(
    wavelength_range_um: tuple[float, float],
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(wavelength_range_um) != 2:
        raise ValueError("wavelength_range_um must be a tuple of (min_um, max_um).")
    wl_min_um, wl_max_um = wavelength_range_um
    wl_min_um = float(wl_min_um)
    wl_max_um = float(wl_max_um)
    if wl_min_um <= 0 or wl_max_um <= 0:
        raise ValueError("wavelength_range_um values must be > 0.")
    if wl_max_um <= wl_min_um:
        raise ValueError("wavelength_range_um max must be greater than min.")
    if int(num_points) < 2:
        raise ValueError("num_points must be >= 2.")
    wavelengths_um = np.linspace(wl_min_um, wl_max_um, int(num_points))
    return wavelengths_um, wavelengths_um * UM


def _plot_dispersion_panels(
    *,
    wavelengths_um: np.ndarray,
    epsilon: np.ndarray,
    n_complex: np.ndarray,
    title: str,
) -> None:
    plt = _require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = np.asarray(axes)

    n_real = np.real(n_complex)
    n_imag = np.imag(n_complex)
    eps_real = np.real(epsilon)
    eps_imag = np.imag(epsilon)

    axes[0, 0].plot(wavelengths_um, n_real, color="#1f77b4")
    axes[0, 0].set_ylabel("n")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(wavelengths_um, n_imag, color="#d62728")
    axes[0, 1].set_ylabel("k")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(wavelengths_um, eps_real, color="#2ca02c")
    axes[1, 0].set_ylabel("Re(eps_r)")
    axes[1, 0].set_xlabel("Wavelength (um)")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(wavelengths_um, eps_imag, color="#9467bd")
    axes[1, 1].set_ylabel("Im(eps_r)")
    axes[1, 1].set_xlabel("Wavelength (um)")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def show_material(
    material: "Material",
    *,
    wavelength_range_um: tuple[float, float] = (0.4, 2.0),
    num_points: int = 300,
    title: str | None = None,
) -> None:
    plt = _require_matplotlib()
    wavelengths_um, wavelengths_m = _sample_wavelengths(wavelength_range_um, num_points)
    eps = material.eps_model(LIGHT_SPEED / wavelengths_m)
    n_complex = np.lib.scimath.sqrt(eps)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(wavelengths_um, np.real(n_complex), label="n")
    axes[0].plot(wavelengths_um, np.imag(n_complex), label="k")
    axes[0].set_xlabel("Wavelength (um)")
    axes[0].set_ylabel("Index")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].axis("off")
    summary = [
        f"permittivity: {material.permittivity:.6g}",
        f"permeability: {material.permeability:.6g}",
        f"conductivity: {material.conductivity:.6g} S/m",
        "",
        f"k: {material.k:.6g} W/(m*K)",
        f"rho: {material.rho:.6g} kg/m^3",
        f"cp: {material.cp:.6g} J/(kg*K)",
        f"dn_dT: {material.dn_dT:.6g} 1/K",
        f"T0: {material.T0:.6g} K",
    ]
    axes[1].text(0.0, 1.0, "\n".join(summary), va="top", ha="left", family="monospace")

    fig.suptitle(title or (material.name or "Material"))
    fig.tight_layout()
    plt.show()


def show_dispersive_material(
    material: "_DispersiveBase",
    *,
    wavelength_range_um: tuple[float, float] = (0.4, 2.0),
    num_points: int = 300,
    title: str | None = None,
) -> None:
    wavelengths_um, wavelengths_m = _sample_wavelengths(wavelength_range_um, num_points)
    eps = material.epsilon(wavelength=wavelengths_m)
    n_cmp = material.n_complex(wavelength=wavelengths_m)
    _plot_dispersion_panels(
        wavelengths_um=wavelengths_um,
        epsilon=eps,
        n_complex=n_cmp,
        title=title or (material.name or material.__class__.__name__),
    )


def show_custom_material(
    material: "CustomMaterial",
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    grid_shape: tuple[int, int] = (150, 150),
    title: str | None = None,
) -> None:
    plt = _require_matplotlib()
    plot_bounds = bounds or material.bounds
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    if nx < 2 or ny < 2:
        raise ValueError("grid_shape values must be >= 2.")

    entries: list[tuple[str, np.ndarray, tuple[float, float, float, float] | None]] = []
    for prop in ("permittivity", "permeability", "conductivity"):
        grid = getattr(material, f"{prop}_grid")
        func = getattr(material, f"{prop}_func")
        if grid is not None:
            extent = None
            if material.bounds is not None:
                extent = (
                    float(material.bounds[0][0]),
                    float(material.bounds[0][1]),
                    float(material.bounds[1][0]),
                    float(material.bounds[1][1]),
                )
            entries.append((prop, np.asarray(grid, dtype=float), extent))
            continue
        if func is None or plot_bounds is None:
            continue

        x_min, x_max = float(plot_bounds[0][0]), float(plot_bounds[0][1])
        y_min, y_max = float(plot_bounds[1][0]), float(plot_bounds[1][1])
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("bounds must define increasing x/y ranges.")
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        xg, yg = np.meshgrid(x, y)
        getter = getattr(material, f"get_{prop}")

        def _sample_one(xv: float, yv: float) -> float:
            val = getter(xv, yv)
            return float(np.real(np.asarray(val).reshape(-1)[0]))

        vals = np.vectorize(_sample_one)(xg, yg)
        entries.append((prop, np.asarray(vals, dtype=float), (x_min, x_max, y_min, y_max)))

    if not entries:
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        ax.axis("off")
        display_title = title or "CustomMaterial"
        summary = [
            f"permittivity: {material.permittivity}",
            f"permeability: {material.permeability}",
            f"conductivity: {material.conductivity}",
            f"k={material.k}, rho={material.rho}, cp={material.cp}",
            f"dn_dT={material.dn_dT}, T0={material.T0}",
            "No spatial grids/functions with plottable bounds.",
        ]
        ax.text(0.02, 0.98, "\n".join(summary), va="top", ha="left", family="monospace")
        ax.set_title(display_title)
        fig.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(1, len(entries), figsize=(5.2 * len(entries), 4.0))
    axes_arr = np.atleast_1d(axes)
    display_title = title or "CustomMaterial Spatial Properties"
    for ax, (prop, values, extent) in zip(axes_arr, entries):
        if extent is None:
            image = ax.imshow(values, origin="lower", aspect="auto")
            ax.set_xlabel("x index")
            ax.set_ylabel("y index")
        else:
            image = ax.imshow(values, origin="lower", extent=extent, aspect="auto")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        ax.set_title(prop)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(display_title)
    fig.tight_layout()
    plt.show()


def show_material2d(
    material: "Material2D",
    *,
    wavelength_range_um: tuple[float, float] = (0.4, 2.0),
    num_points: int = 300,
    title: str | None = None,
) -> None:
    plt = _require_matplotlib()
    wavelengths_um, wavelengths_m = _sample_wavelengths(wavelength_range_um, num_points)
    frequency = LIGHT_SPEED / wavelengths_m
    eps_ss, eps_tt = material.eps_model(frequency)
    n_ss = np.lib.scimath.sqrt(eps_ss)
    n_tt = np.lib.scimath.sqrt(eps_tt)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = np.asarray(axes)

    axes[0, 0].plot(wavelengths_um, np.real(n_ss), label="ss")
    axes[0, 0].plot(wavelengths_um, np.real(n_tt), label="tt", linestyle="--")
    axes[0, 0].set_ylabel("n")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(wavelengths_um, np.imag(n_ss), label="ss")
    axes[0, 1].plot(wavelengths_um, np.imag(n_tt), label="tt", linestyle="--")
    axes[0, 1].set_ylabel("k")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(wavelengths_um, np.real(eps_ss), label="ss")
    axes[1, 0].plot(wavelengths_um, np.real(eps_tt), label="tt", linestyle="--")
    axes[1, 0].set_ylabel("Re(eps_r)")
    axes[1, 0].set_xlabel("Wavelength (um)")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(wavelengths_um, np.imag(eps_ss), label="ss")
    axes[1, 1].plot(wavelengths_um, np.imag(eps_tt), label="tt", linestyle="--")
    axes[1, 1].set_ylabel("Im(eps_r)")
    axes[1, 1].set_xlabel("Wavelength (um)")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(title or (material.name or "Material2D"))
    fig.tight_layout()
    plt.show()


def show_anisotropic_material(
    material: "AnisotropicMaterial",
    *,
    wavelength_range_um: tuple[float, float] = (0.4, 2.0),
    num_points: int = 300,
    title: str | None = None,
) -> None:
    plt = _require_matplotlib()
    wavelengths_um, wavelengths_m = _sample_wavelengths(wavelength_range_um, num_points)
    frequency = LIGHT_SPEED / wavelengths_m
    eps_xx = material.xx.eps_model(frequency)
    eps_yy = material.yy.eps_model(frequency)
    eps_zz = material.zz.eps_model(frequency)

    n_xx = np.lib.scimath.sqrt(eps_xx)
    n_yy = np.lib.scimath.sqrt(eps_yy)
    n_zz = np.lib.scimath.sqrt(eps_zz)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = np.asarray(axes)

    axes[0, 0].plot(wavelengths_um, np.real(n_xx), label="xx")
    axes[0, 0].plot(wavelengths_um, np.real(n_yy), label="yy")
    axes[0, 0].plot(wavelengths_um, np.real(n_zz), label="zz")
    axes[0, 0].set_ylabel("n")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(wavelengths_um, np.imag(n_xx), label="xx")
    axes[0, 1].plot(wavelengths_um, np.imag(n_yy), label="yy")
    axes[0, 1].plot(wavelengths_um, np.imag(n_zz), label="zz")
    axes[0, 1].set_ylabel("k")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(wavelengths_um, np.real(eps_xx), label="xx")
    axes[1, 0].plot(wavelengths_um, np.real(eps_yy), label="yy")
    axes[1, 0].plot(wavelengths_um, np.real(eps_zz), label="zz")
    axes[1, 0].set_ylabel("Re(eps_r)")
    axes[1, 0].set_xlabel("Wavelength (um)")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(wavelengths_um, np.imag(eps_xx), label="xx")
    axes[1, 1].plot(wavelengths_um, np.imag(eps_yy), label="yy")
    axes[1, 1].plot(wavelengths_um, np.imag(eps_zz), label="zz")
    axes[1, 1].set_ylabel("Im(eps_r)")
    axes[1, 1].set_xlabel("Wavelength (um)")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(title or "AnisotropicMaterial")
    fig.tight_layout()
    plt.show()

