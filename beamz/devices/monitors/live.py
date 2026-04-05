import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def start_visualization(monitor, field_component="Ez"):
    """Start live field visualization."""
    if not monitor.live_update:
        monitor.live_update = True
    if monitor.is_3d:
        setup_plot_3d(monitor, field_component)
    else:
        setup_plot_2d(monitor, field_component)


def setup_plot_2d(monitor, field_component):
    """Set up live plotting for a 2D monitor."""
    monitor.live_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title(f"{field_component} along monitor line")
    ax1.set_xlabel("Position along line")
    ax1.set_ylabel(f"{field_component} amplitude")
    monitor.live_plots["field_line"] = ax1.plot([], [], "b-")[0]

    ax2.set_title("Power vs Time")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Total power")
    monitor.live_plots["power_time"] = ax2.plot([], [], "r-")[0]
    plt.tight_layout()
    plt.ion()
    plt.show()


def setup_plot_3d(monitor, field_component):
    """Set up live plotting for a 3D monitor."""
    monitor.live_fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(12, 10)
    )
    ax1.set_title(f"{field_component} magnitude on plane")
    monitor.live_plots["field_2d"] = ax1.imshow(
        np.zeros((10, 10)), cmap="RdBu", animated=True
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2.set_title("Power vs Time")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Total power")
    monitor.live_plots["power_time"] = ax2.plot([], [], "r-")[0]

    ax3.set_title(f"{field_component} along X (center)")
    ax3.set_xlabel("X position")
    ax3.set_ylabel(f"{field_component} amplitude")
    monitor.live_plots["field_x"] = ax3.plot([], [], "b-")[0]

    ax4.set_title(f"{field_component} along Y (center)")
    ax4.set_xlabel("Y position")
    ax4.set_ylabel(f"{field_component} amplitude")
    monitor.live_plots["field_y"] = ax4.plot([], [], "g-")[0]
    plt.tight_layout()
    plt.ion()
    plt.show()


def update_plot_2d(monitor):
    """Update a 2D live monitor plot."""
    if monitor.live_fig is None or not monitor.fields["t"]:
        return
    try:
        latest_field = monitor.fields["Ez"][-1]
        x_pos = range(len(latest_field))
        monitor.live_plots["field_line"].set_data(x_pos, latest_field)
        monitor.live_plots["power_time"].set_data(
            range(len(monitor.power_history)), monitor.power_history
        )
        for ax in monitor.live_fig.axes:
            ax.relim()
            ax.autoscale_view()
        monitor.live_fig.canvas.draw()
        monitor.live_fig.canvas.flush_events()
    except Exception:
        logger.debug("Failed to update 2D monitor live plot.", exc_info=True)


def update_plot_3d(monitor):
    """Update a 3D live monitor plot."""
    if monitor.live_fig is None or not monitor.fields["t"]:
        return
    try:
        latest_field = monitor.fields["Ez"][-1]
        monitor.live_plots["field_2d"].set_array(latest_field)
        monitor.live_plots["field_2d"].set_clim(
            vmin=np.min(latest_field), vmax=np.max(latest_field)
        )
        monitor.live_plots["power_time"].set_data(
            range(len(monitor.power_history)), monitor.power_history
        )

        center_y = latest_field.shape[0] // 2
        center_x = latest_field.shape[1] // 2
        monitor.live_plots["field_x"].set_data(
            range(latest_field.shape[1]), latest_field[center_y, :]
        )
        monitor.live_plots["field_y"].set_data(
            range(latest_field.shape[0]), latest_field[:, center_x]
        )

        for ax in monitor.live_fig.axes[1:]:
            ax.relim()
            ax.autoscale_view()
        monitor.live_fig.canvas.draw()
        monitor.live_fig.canvas.flush_events()
    except Exception:
        logger.debug("Failed to update 3D monitor live plot.", exc_info=True)
