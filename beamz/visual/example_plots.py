from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamz.const import µm


def plot_simulation_overview(path: Path, eps: np.ndarray, *, width, height, depth, z_focus, source_plane, monitor_planes):
    z_idx = int(np.clip(round(float(z_focus) / max(float(depth), 1e-30) * (eps.shape[0] - 1)), 0, eps.shape[0] - 1))
    y_idx = eps.shape[1] // 2
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=260)
    axes[0].imshow(eps[z_idx], origin="lower", extent=[0, width / µm, 0, height / µm], cmap="viridis", aspect="equal")
    axes[1].imshow(eps[:, y_idx, :], origin="lower", extent=[0, width / µm, 0, depth / µm], cmap="viridis", aspect="auto")
    for ax, title, ylabel in ((axes[0], "XY overview", "y (um)"), (axes[1], "XZ overview", "z (um)")):
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel(ylabel)
    for name, plane in {"source": source_plane, **monitor_planes}.items():
        (x0, y0, z0), (x1, y1, z1) = plane
        color = "red" if name == "source" else "white"
        axes[0].plot([x0 / µm, x1 / µm], [y0 / µm, y1 / µm], color=color, lw=1.5)
        axes[1].plot([0.5 * (x0 + x1) / µm, 0.5 * (x0 + x1) / µm], [z0 / µm, z1 / µm], color=color, lw=1.5)
        axes[0].text(0.5 * (x0 + x1) / µm, 0.5 * (y0 + y1) / µm + 0.08, name, color=color, fontsize=7, ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=320)
    plt.close(fig)


def plot_sparameters_db(path: Path, wavelengths_um: np.ndarray, s_matrix: dict[tuple[str, str], np.ndarray], *, source_port="o1", ports=("o1", "o2", "o3", "o4")):
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.5), dpi=320)
    colors = {"o1": "black", "o2": "tab:blue", "o3": "tab:orange", "o4": "tab:green"}
    for port in ports:
        y_db = 20.0 * np.log10(np.maximum(np.abs(np.asarray(s_matrix[(port, source_port)], dtype=np.complex128)), 1e-12))
        ax.plot(wavelengths_um, y_db, "o-", lw=2.0, ms=4.0, color=colors.get(port), label=rf"$|S_{{{port[1:]}{source_port[1:]}}}|$")
    ax.set_xlim(float(np.min(wavelengths_um)), float(np.max(wavelengths_um)))
    ax.set_ylim(-55.0, 0.0)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Crossing S-Parameters")
    ax.grid(which="major", alpha=0.25, lw=0.6)
    ax.minorticks_on()
    ax.grid(which="minor", alpha=0.12, lw=0.4)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=320)
    plt.close(fig)
