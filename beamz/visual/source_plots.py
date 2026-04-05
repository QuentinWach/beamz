"""Source-related plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np

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
