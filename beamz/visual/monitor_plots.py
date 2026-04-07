"""Standalone visualization functions for Monitor data.

These were extracted from the Monitor class to separate data collection
from visualization concerns.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_monitor_fields(monitor, field="Ez", figsize=(10, 6), time_index=-1):
    """Plot field data from a monitor.

    Args:
        monitor: Monitor instance with recorded data.
        field: Field component to plot ('Ez', 'Ex', 'Ey', 'Hx', 'Hy', 'Hz').
        figsize: Figure size tuple.
        time_index: Time index to plot (-1 for latest).

    Returns:
        (fig, ax) tuple.
    """
    if not monitor.fields["t"]:
        print("No field data recorded.")
        return None, None

    if field not in monitor.fields:
        print(
            f"Field '{field}' not available. Available fields: {list(monitor.fields.keys())}"
        )
        return None, None

    if not monitor.fields[field]:
        print(f"No data for field '{field}'.")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    if monitor.monitor_type == "line":
        field_data = monitor.fields[field][time_index]
        x_pos = range(len(field_data))
        ax.plot(x_pos, field_data, "b-", linewidth=2)
        ax.set_xlabel("Position along monitor line")
        ax.set_ylabel(f"{field} amplitude")
        ax.set_title(f'{field} at t = {monitor.fields["t"][time_index]:.2e} s')
        ax.grid(True, alpha=0.3)
    else:
        field_data = monitor.fields[field][time_index]
        im = ax.imshow(field_data, cmap="RdBu", origin="lower", aspect="auto")
        plt.colorbar(im, ax=ax, label=f"{field} amplitude")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
        ax.set_title(f'{field} at t = {monitor.fields["t"][time_index]:.2e} s')

    plt.tight_layout()
    return fig, ax


def plot_monitor_power(monitor, figsize=(10, 6), log_scale=False, db_scale=False):
    """Plot power history from a monitor.

    Args:
        monitor: Monitor instance with recorded power data.
        figsize: Figure size tuple.
        log_scale: Use logarithmic scale.
        db_scale: Use dB scale (10*log10).

    Returns:
        (fig, ax) tuple.
    """
    if not monitor.power_history:
        print("No power data recorded.")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    time_steps = range(len(monitor.power_history))
    power_data = np.array(monitor.power_history)

    if db_scale:
        power_data = 10 * np.log10(np.maximum(power_data, 1e-12))
        ax.plot(time_steps, power_data, "r-", linewidth=2)
        ax.set_ylabel("Power (dB)")
    elif log_scale:
        ax.semilogy(time_steps, power_data, "r-", linewidth=2)
        ax.set_ylabel("Power (log scale)")
    else:
        ax.plot(time_steps, power_data, "r-", linewidth=2)
        ax.set_ylabel("Power")

    ax.set_xlabel("Time step")
    ax.set_title("Power vs Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def animate_monitor_fields(
    monitor, field="Ez", figsize=(8, 6), interval=100, save_filename=None
):
    """Create an animation of field evolution from a monitor.

    Args:
        monitor: Monitor instance with recorded data.
        field: Field component to animate.
        figsize: Figure size tuple.
        interval: Animation interval in milliseconds.
        save_filename: Optional filename to save animation.

    Returns:
        Animation object.
    """
    if not monitor.fields["t"] or field not in monitor.fields:
        print(f"No data available for field '{field}'.")
        return None

    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=figsize)

    if monitor.monitor_type == "line":
        (line,) = ax.plot([], [], "b-", linewidth=2)
        ax.set_xlabel("Position along monitor line")
        ax.set_ylabel(f"{field} amplitude")

        all_data = np.concatenate(monitor.fields[field])
        ax.set_xlim(0, len(monitor.fields[field][0]))
        ax.set_ylim(np.min(all_data), np.max(all_data))

        def animate(frame):
            field_data = monitor.fields[field][frame]
            x_pos = range(len(field_data))
            line.set_data(x_pos, field_data)
            ax.set_title(f'{field} at t = {monitor.fields["t"][frame]:.2e} s')
            return (line,)

    else:
        field_data = monitor.fields[field][0]
        im = ax.imshow(
            field_data, cmap="RdBu", origin="lower", aspect="auto", animated=True
        )
        plt.colorbar(im, ax=ax, label=f"{field} amplitude")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")

        all_data = np.array(monitor.fields[field])
        vmin, vmax = np.min(all_data), np.max(all_data)
        im.set_clim(vmin, vmax)

        def animate(frame):
            field_data = monitor.fields[field][frame]
            im.set_array(field_data)
            ax.set_title(f'{field} at t = {monitor.fields["t"][frame]:.2e} s')
            return [im]

    anim = FuncAnimation(
        fig,
        animate,
        frames=len(monitor.fields["t"]),
        interval=interval,
        blit=True,
        repeat=True,
    )

    if save_filename:
        anim.save(save_filename, writer="pillow", fps=1000 // interval)
        print(f"Animation saved to {save_filename}")

    plt.tight_layout()
    return anim
