import logging
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MonitorLiveState:
    fig: object | None = None
    axes: object | None = None
    plots: dict = field(default_factory=dict)


def create_state():
    return MonitorLiveState()


def _state(monitor):
    state = getattr(monitor, "_live_state", None)
    if state is None:
        state = create_state()
        object.__setattr__(monitor, "_live_state", state)
    return state


def should_refresh(monitor):
    interval = max(1, int(getattr(monitor, "update_interval", 10)))
    return len(monitor.state.fields["t"]) % interval == 0


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
    state = _state(monitor)
    state.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    state.axes = (ax1, ax2)
    ax1.set_title(f"{field_component} along monitor line")
    ax1.set_xlabel("Position along line")
    ax1.set_ylabel(f"{field_component} amplitude")
    state.plots["field_line"] = ax1.plot([], [], "b-")[0]

    ax2.set_title("Power vs Time")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Total power")
    state.plots["power_time"] = ax2.plot([], [], "r-")[0]
    plt.tight_layout()
    plt.ion()
    plt.show()


def setup_plot_3d(monitor, field_component):
    """Set up live plotting for a 3D monitor."""
    state = _state(monitor)
    state.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(12, 10)
    )
    state.axes = ((ax1, ax2), (ax3, ax4))
    ax1.set_title(f"{field_component} magnitude on plane")
    state.plots["field_2d"] = ax1.imshow(
        np.zeros((10, 10)), cmap="RdBu", animated=True
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2.set_title("Power vs Time")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Total power")
    state.plots["power_time"] = ax2.plot([], [], "r-")[0]

    ax3.set_title(f"{field_component} along X (center)")
    ax3.set_xlabel("X position")
    ax3.set_ylabel(f"{field_component} amplitude")
    state.plots["field_x"] = ax3.plot([], [], "b-")[0]

    ax4.set_title(f"{field_component} along Y (center)")
    ax4.set_xlabel("Y position")
    ax4.set_ylabel(f"{field_component} amplitude")
    state.plots["field_y"] = ax4.plot([], [], "g-")[0]
    plt.tight_layout()
    plt.ion()
    plt.show()


def update_plot_2d(monitor):
    """Update a 2D live monitor plot."""
    live_state = _state(monitor)
    if live_state.fig is None or not monitor.state.fields["t"]:
        return
    try:
        state = monitor.state
        latest_field = state.fields["Ez"][-1]
        x_pos = range(len(latest_field))
        live_state.plots["field_line"].set_data(x_pos, latest_field)
        live_state.plots["power_time"].set_data(
            range(len(state.power_history)), state.power_history
        )
        for ax in live_state.fig.axes:
            ax.relim()
            ax.autoscale_view()
        live_state.fig.canvas.draw()
        live_state.fig.canvas.flush_events()
    except Exception:
        logger.debug("Failed to update 2D monitor live plot.", exc_info=True)


def update_plot_3d(monitor):
    """Update a 3D live monitor plot."""
    live_state = _state(monitor)
    if live_state.fig is None or not monitor.state.fields["t"]:
        return
    try:
        state = monitor.state
        latest_field = state.fields["Ez"][-1]
        live_state.plots["field_2d"].set_array(latest_field)
        live_state.plots["field_2d"].set_clim(
            vmin=np.min(latest_field), vmax=np.max(latest_field)
        )
        live_state.plots["power_time"].set_data(
            range(len(state.power_history)), state.power_history
        )

        center_y = latest_field.shape[0] // 2
        center_x = latest_field.shape[1] // 2
        live_state.plots["field_x"].set_data(
            range(latest_field.shape[1]), latest_field[center_y, :]
        )
        live_state.plots["field_y"].set_data(
            range(latest_field.shape[0]), latest_field[:, center_x]
        )

        for ax in live_state.fig.axes[1:]:
            ax.relim()
            ax.autoscale_view()
        live_state.fig.canvas.draw()
        live_state.fig.canvas.flush_events()
    except Exception:
        logger.debug("Failed to update 3D monitor live plot.", exc_info=True)
