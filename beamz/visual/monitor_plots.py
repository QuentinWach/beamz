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
    fig, ax = plt.subplots(figsize=figsize)
    snapshot = monitor.field_snapshot(field=field, time_index=time_index)
    snapshot.plot(ax=ax, cmap="RdBu", colorbar=not monitor.monitor_type == "line")
    if monitor.monitor_type == "line":
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
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
    fig, ax = plt.subplots(figsize=figsize)
    trace = monitor.power_trace(db_scale=db_scale)
    if log_scale and not db_scale:
        ax.semilogy(trace.coords, trace.values, "r-", linewidth=2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Power")
        ax.set_title(trace.title)
    else:
        trace.plot(ax=ax, color="r", linewidth=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
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
