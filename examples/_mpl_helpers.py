from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from beamz.visual.helpers import get_si_scale_and_label
from beamz.visual.data import mode_profile_data, signal_plot_data


def get_twilight_zero_cmap():
    colors = [
        (1.0, 1.0, 1.0),
        (0.2, 0.3, 0.8),
        (0.1, 0.1, 0.5),
        (0.1, 0.1, 0.1),
        (0.5, 0.1, 0.1),
        (0.8, 0.3, 0.2),
        (1.0, 1.0, 1.0),
    ]
    return LinearSegmentedColormap.from_list("twilight_zero", colors, N=256)


def resolve_cmap(cmap):
    if cmap == "twilight_zero":
        return get_twilight_zero_cmap()
    return cmap


def _draw_polygon(ax, payload):
    vertices = payload["vertices"]
    if not vertices:
        return

    coords = []
    codes = []
    coords.extend(vertices)
    coords.append(vertices[0])
    codes.append(MplPath.MOVETO)
    if len(vertices) > 1:
        codes.extend([MplPath.LINETO] * (len(vertices) - 1))
    codes.append(MplPath.CLOSEPOLY)

    for hole in payload.get("interiors", []):
        if not hole:
            continue
        coords.extend(hole)
        coords.append(hole[0])
        codes.append(MplPath.MOVETO)
        if len(hole) > 1:
            codes.extend([MplPath.LINETO] * (len(hole) - 1))
        codes.append(MplPath.CLOSEPOLY)

    path = MplPath(np.asarray(coords), np.asarray(codes))
    style = payload["style"]
    patch = PathPatch(
        path,
        facecolor=style.get("facecolor", "none"),
        edgecolor=style.get("edgecolor", "black"),
        alpha=style.get("alpha", 1.0),
        linestyle=style.get("linestyle", "-"),
    )
    ax.add_patch(patch)


def _draw_source(ax, payload):
    style = payload["style"]
    if payload["shape"] == "gaussian":
        circle = Circle(
            tuple(payload["position"][:2]),
            radius=payload["radius"],
            facecolor=style.get("facecolor", "none"),
            edgecolor=style.get("edgecolor", "orange"),
            linewidth=2,
            alpha=style.get("alpha", 0.8),
            linestyle=style.get("linestyle", "-"),
        )
        ax.add_patch(circle)
        ax.add_patch(
            Circle(
                tuple(payload["position"][:2]),
                radius=max(payload["radius"] * 0.1, 1e-9),
                facecolor=style.get("edgecolor", "orange"),
                edgecolor="none",
                alpha=style.get("alpha", 0.8),
            )
        )
        return

    if payload["shape"] != "mode":
        return

    center = payload["center"]
    half_width = (payload.get("width") or 0.5e-6) / 2.0
    if payload["direction"] in {"+x", "-x"}:
        x = [center[0], center[0]]
        y = [center[1] - half_width, center[1] + half_width]
    else:
        x = [center[0] - half_width, center[0] + half_width]
        y = [center[1], center[1]]

    ax.plot(
        x,
        y,
        color=style.get("edgecolor", "crimson"),
        linewidth=3,
        alpha=style.get("alpha", 0.8),
        solid_capstyle="round",
    )

    arrow_length = (payload.get("wavelength") or 0.5e-6) * 0.5
    dx, dy = 0.0, 0.0
    if payload["direction"] == "+x":
        dx = arrow_length
    elif payload["direction"] == "-x":
        dx = -arrow_length
    elif payload["direction"] == "+y":
        dy = arrow_length
    elif payload["direction"] == "-y":
        dy = -arrow_length

    end_x = center[0] + dx
    end_y = center[1] + dy
    ax.plot(
        [center[0], end_x],
        [center[1], end_y],
        color=style.get("edgecolor", "crimson"),
        linewidth=2,
        alpha=style.get("alpha", 0.8),
    )
    marker = {
        "+x": ">",
        "-x": "<",
        "+y": "^",
        "-y": "v",
    }.get(payload["direction"], "o")
    ax.plot(
        [end_x],
        [end_y],
        marker=marker,
        markersize=7,
        color=style.get("edgecolor", "crimson"),
        alpha=style.get("alpha", 0.8),
        linestyle="none",
    )


def _draw_monitor(ax, payload):
    style = payload["style"]
    if payload["shape"] == "line":
        x0, y0 = payload["start"][:2]
        x1, y1 = payload["end"][:2]
        color = style.get("edgecolor", "navy")
        ax.plot([x0, x1], [y0, y1], lw=4, color=color, alpha=style.get("alpha", 1.0))
        ax.plot(
            [x0, x1],
            [y0, y1],
            lw=1,
            color=color,
            linestyle=style.get("linestyle", "-"),
        )
        return

    x0, y0 = payload["start"][:2]
    width, height = payload["size"][:2]
    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fill=style.get("facecolor", "none") != "none",
        facecolor=style.get("facecolor", "none"),
        edgecolor=style.get("edgecolor", "navy"),
        alpha=style.get("alpha", 1.0) * 0.3,
        linestyle=style.get("linestyle", "-"),
        linewidth=2,
    )
    ax.add_patch(rect)
    if payload.get("position") is not None:
        ax.text(
            payload["position"][0],
            payload["position"][1],
            payload["name"],
            ha="center",
            va="center",
            fontsize=8,
            color=style.get("edgecolor", "navy"),
        )


def _draw_boundaries(ax, layout, line_color="gray", line_opacity=0.5):
    for boundary in layout.get("boundaries", []):
        for rect in boundary["rectangles"]:
            ax.add_patch(
                Rectangle(
                    rect["origin"],
                    rect["width"],
                    rect["height"],
                    facecolor="none",
                    edgecolor=line_color,
                    linestyle=":",
                    alpha=line_opacity,
                )
            )


def _configure_axes(ax, design_payload):
    unit = design_payload["scale_unit"]
    scale = design_payload["scale_factor"]
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x * scale:.1f}")
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y * scale:.1f}")


def _draw_scale_bar(ax, design_payload, wavelength=None, fontsize=10):
    width = design_payload["width"]
    height = design_payload["height"]
    scale_factor = design_payload["scale_factor"]
    unit = design_payload["scale_unit"]

    if wavelength is not None:
        scale_bar_length_um = np.round(2 * wavelength * 1e6)
        scale_bar_length = scale_bar_length_um * 1e-6
        label_text = f"{int(scale_bar_length_um)} µm"
    else:
        min_dim = min(width, height)
        scale_bar_length_physical = min_dim * 0.18
        if scale_bar_length_physical > 0:
            order = 10 ** np.floor(np.log10(scale_bar_length_physical))
            normalized = scale_bar_length_physical / order
            if normalized <= 1.25:
                nice_value = 1 * order
            elif normalized <= 2.5:
                nice_value = 2 * order
            elif normalized <= 6:
                nice_value = 5 * order
            else:
                nice_value = 10 * order
            scale_bar_length = nice_value
        else:
            scale_bar_length = min_dim * 0.15

        display_value = scale_bar_length * scale_factor
        if display_value >= 1:
            label_text = f"{display_value:.0f} {unit}"
        elif display_value >= 0.1:
            label_text = f"{display_value:.1f} {unit}"
        else:
            label_text = f"{display_value:.2f} {unit}"

    margin_x = width * 0.1
    margin_y = height * 0.1
    x_start = width - scale_bar_length - margin_x
    x_end = width - margin_x
    y_pos = margin_y
    ax.plot([x_start, x_end], [y_pos, y_pos], "w", linewidth=3, solid_capstyle="butt")
    ax.text(
        (x_start + x_end) / 2,
        y_pos - height * 0.02,
        label_text,
        ha="center",
        va="top",
        color="white",
        fontsize=fontsize,
    )


def plot_design(design, *, sources=None, monitors=None):
    payload = design.to_plot_data(sources=sources, monitors=monitors)
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * design.height / design.width))
    for structure in payload["structures"]:
        _draw_polygon(ax, structure)
    for source in payload["sources"]:
        _draw_source(ax, source)
    for monitor in payload["monitors"]:
        _draw_monitor(ax, monitor)
    ax.set_title("Design Layout")
    ax.set_xlim(*payload["xlim"])
    ax.set_ylim(*payload["ylim"])
    ax.set_aspect("equal")
    _configure_axes(ax, payload)
    plt.tight_layout()
    plt.show()


def plot_grid(grid, *, field="permittivity", z_index=None, z_position=None):
    payload = grid.to_plot_data(field=field, z_index=z_index, z_position=z_position)
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * payload["design"]["height"] / payload["design"]["width"]))
    im = ax.imshow(
        payload["array"],
        origin="lower",
        cmap="Grays",
        extent=payload["extent"],
    )
    fig.colorbar(im, ax=ax, label=field)
    ax.set_title("Rasterized Design Grid")
    scale_factor, scale_unit = get_si_scale_and_label(
        max(payload["design"]["width"], payload["design"]["height"])
    )
    design_payload = {
        "width": payload["design"]["width"],
        "height": payload["design"]["height"],
        "scale_factor": scale_factor,
        "scale_unit": scale_unit,
    }
    _configure_axes(ax, design_payload)
    plt.tight_layout()
    plt.show()


def plot_signal(signals, t, *, save_path=None):
    payload = signal_plot_data(signals, t)
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, values in enumerate(payload["signals"]):
        kwargs = {"label": f"Signal {idx}"} if len(payload["signals"]) > 1 else {}
        ax.plot(payload["t_scaled"], values, **kwargs)
    ax.set_xlim(*payload["xlim"])
    ax.set_xlabel(f"Time ({payload['time_unit']})")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal")
    if len(payload["signals"]) > 1:
        ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_mode_profile(source, *, save_path=None):
    payload = mode_profile_data(source)
    fig = plt.figure(figsize=(8, 6))
    if payload["is_2d"]:
        im = plt.imshow(
            payload["amplitude"],
            origin="lower",
            cmap="magma",
            aspect="auto",
        )
        plt.colorbar(im, label="Absolute Amplitude")
        plt.title(f"Mode Source 2D Profile: {payload['title']} (neff={payload['neff']:.4f})")
        if payload["direction"] in ["+x", "-x"]:
            plt.xlabel("Y-axis")
            plt.ylabel("Z-axis")
        else:
            plt.xlabel("X-axis")
            plt.ylabel("Z-axis")
    else:
        plt.plot(payload["amplitude"], "k-")
        plt.title(f"Mode Source 1D Profile: {payload['title']} (neff={payload['neff']:.4f})")
        plt.xlabel("Transverse Coordinate (cells)")
        plt.ylabel("Absolute Amplitude")
        plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def _snapshot_figure(snapshot, *, cmap, clean_visualization, interpolation, figure=None, axes=None):
    layout = snapshot["layout"]
    design_payload = layout["design"]
    field = snapshot["field"]
    extent = snapshot["extent"]
    actual_cmap = resolve_cmap(cmap)

    if figure is None or axes is None:
        if clean_visualization:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_axes([0, 0, 1, 1])
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = figure
        fig.clear()
        if clean_visualization:
            ax = fig.add_axes([0, 0, 1, 1])
        else:
            ax = fig.add_subplot(111)

    im = ax.imshow(
        field,
        origin="lower",
        cmap=actual_cmap,
        extent=extent,
        interpolation=interpolation,
    )

    for structure in design_payload["structures"]:
        style = dict(structure["style"])
        if not structure.get("is_pml"):
            style["facecolor"] = "none"
            style["edgecolor"] = "gray"
            style["alpha"] = 0.5
        overlay = dict(structure)
        overlay["style"] = style
        _draw_polygon(ax, overlay)
    for source in design_payload["sources"]:
        _draw_source(ax, source)
    for monitor in design_payload["monitors"]:
        overlay = dict(monitor)
        style = dict(overlay["style"])
        style["edgecolor"] = "gray"
        style["alpha"] = 0.5
        overlay["style"] = style
        _draw_monitor(ax, overlay)
    _draw_boundaries(ax, layout)

    if clean_visualization:
        ax.set_axis_off()
        _draw_scale_bar(ax, design_payload)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    else:
        fig.colorbar(im, ax=ax, orientation="vertical", label=f"{snapshot['field_name']} ({snapshot['units']})")
        ax.set_title(
            f"{snapshot['field_name']} at t = {snapshot['time']:.2e} s "
            f"(step {snapshot['step']}/{snapshot['num_steps']})"
        )
        _configure_axes(ax, design_payload)
        plt.tight_layout()

    return fig, ax


def run_with_snapshots(
    sim,
    *,
    snapshot_field,
    snapshot_interval=10,
    cmap="twilight_zero",
    clean_visualization=False,
    interpolation="bicubic",
    live_display=True,
    save_video=None,
    video_fps=30,
):
    context = {"fig": None, "ax": None}

    def callback(snapshot):
        if not live_display:
            return
        fig, ax = _snapshot_figure(
            snapshot,
            cmap=cmap,
            clean_visualization=clean_visualization,
            interpolation=interpolation,
            figure=context["fig"],
            axes=context["ax"],
        )
        context["fig"], context["ax"] = fig, ax
        plt.show(block=False)
        plt.pause(0.001)

    results = sim.run(
        snapshot_field=snapshot_field,
        snapshot_interval=snapshot_interval,
        snapshot_callback=callback if live_display else None,
        store_snapshots=save_video is not None,
    )

    if save_video is not None and results is not None and results.snapshots:
        save_snapshot_video(
            results.snapshots,
            filename=save_video,
            fps=video_fps,
            cmap=cmap,
            clean_visualization=clean_visualization,
            interpolation=interpolation,
        )

    return results


def save_snapshot_video(
    snapshots,
    *,
    filename,
    fps=30,
    dpi=150,
    cmap="twilight_zero",
    clean_visualization=False,
    interpolation="bicubic",
):
    if not snapshots:
        return

    output = Path(filename)
    fig, ax = plt.subplots(figsize=(10, 8))
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    with writer.saving(fig, str(output), dpi=dpi):
        for snapshot in snapshots:
            _snapshot_figure(
                snapshot,
                cmap=cmap,
                clean_visualization=clean_visualization,
                interpolation=interpolation,
                figure=fig,
                axes=ax,
            )
            writer.grab_frame()
    plt.close(fig)
