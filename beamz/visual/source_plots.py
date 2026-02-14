"""Source-related plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np


def plot_signal(signals, t, save_path=None):
    """Create a single signal or a list of signals on the same plot."""
    # Convert time to seconds
    t_seconds = t
    # Determine appropriate time unit and scaling factor
    if t_seconds[-1] < 1e-12:
        t_scaled = t_seconds * 1e15  # Convert to fs
        unit = "fs"
    elif t_seconds[-1] < 1e-9:  # Less than 1 ns
        t_scaled = t_seconds * 1e12  # Convert to ps
        unit = "ps"
    elif t_seconds[-1] < 1e-6:  # Less than 1 µs
        t_scaled = t_seconds * 1e9  # Convert to ns
        unit = "ns"
    elif t_seconds[-1] < 1e-3:  # Less than 1 ms
        t_scaled = t_seconds * 1e6  # Convert to µs
        unit = "µs"
    elif t_seconds[-1] < 1:  # Less than 1 s
        t_scaled = t_seconds * 1e3  # Convert to ms
        unit = "ms"
    else:
        t_scaled = t_seconds
        unit = "s"
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(9, 4))
    if isinstance(signals, list):
        i = 0
        for signal in signals:
            ax.plot(t_scaled, signal, label=f"Signal {i}")
            i += 1
        ax.legend()
    else:
        ax.plot(t_scaled, signals, color="black")
    ax.set_xlim(t_scaled[0], t_scaled[-1])
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def show_mode_profile(mode_source, field=None):
    """Visualize the 2D mode profile (for 3D simulations) or 1D profile (for 2D).

    Args:
        mode_source: A ModeSource instance.
        field: Unused, kept for backward compatibility.
    """
    if mode_source._Ez_profile is None and mode_source._jz_profile is None:
        if mode_source.grid is not None and hasattr(mode_source.grid, "permittivity"):
            res = getattr(mode_source.grid, "resolution", 0.05e-6)
            mode_source.initialize(mode_source.grid.permittivity, res)
        else:
            print(
                "[ModeSource] Source not initialized. Call Simulation or initialize manually."
            )
            return

    if mode_source._Ez_profile is not None:
        profile = mode_source._Ez_profile
        title = "Ez (mode profile)"
    elif mode_source._jz_profile is not None:
        profile = mode_source._jz_profile
        title = "Hz (mode profile)"
    else:
        print("[ModeSource] No profiles available.")
        return

    profile = np.squeeze(profile)

    plt.figure(figsize=(8, 6))
    if profile.ndim == 2:
        im = plt.imshow(np.abs(profile), origin="lower", cmap="magma", aspect="auto")
        plt.colorbar(im, label="Absolute Amplitude")
        plt.title(f"Mode Source 2D Profile: {title} (neff={mode_source._neff:.4f})")
        if mode_source.direction in ["+x", "-x"]:
            plt.xlabel("Y-axis")
            plt.ylabel("Z-axis")
        else:
            plt.xlabel("X-axis")
            plt.ylabel("Z-axis")
    else:
        plt.plot(np.abs(profile), "k-")
        plt.title(f"Mode Source 1D Profile: {title} (neff={mode_source._neff:.4f})")
        plt.xlabel("Transverse Coordinate (cells)")
        plt.ylabel("Absolute Amplitude")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
