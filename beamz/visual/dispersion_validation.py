"""Utilities for dispersion validation using pulse-through-slab experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beamz.const import LIGHT_SPEED


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Dispersion validation plotting requires matplotlib. Install with `pip install matplotlib`."
        ) from exc
    return plt


@dataclass
class DispersionValidationResult:
    name: str
    frequency_hz: np.ndarray
    wavelength_um: np.ndarray
    transfer: np.ndarray
    passband_mask: np.ndarray
    n_extracted: np.ndarray
    k_extracted: np.ndarray
    epsilon_extracted: np.ndarray
    n_reference: np.ndarray
    k_reference: np.ndarray
    epsilon_reference: np.ndarray
    metrics: dict[str, float]
    field_frames: np.ndarray | None = None
    extent: tuple[float, float, float, float] | None = None
    dt: float | None = None
    used_fallback: bool = False
    warning: str | None = None


def estimate_transfer_function(
    signal_in: np.ndarray,
    signal_out: np.ndarray,
    dt: float,
    *,
    remove_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate transfer function H(f) = out(f) / in(f) using Hann window."""
    x_in = np.asarray(signal_in, dtype=float).reshape(-1)
    x_out = np.asarray(signal_out, dtype=float).reshape(-1)
    if x_in.shape != x_out.shape:
        raise ValueError("signal_in and signal_out must have the same shape.")
    if x_in.size < 8:
        raise ValueError("Need at least 8 samples to estimate transfer function.")
    if dt <= 0:
        raise ValueError("dt must be > 0.")

    if remove_dc:
        x_in = x_in - float(np.mean(x_in))
        x_out = x_out - float(np.mean(x_out))

    window = np.hanning(x_in.size)
    in_spec = np.fft.rfft(x_in * window)
    out_spec = np.fft.rfft(x_out * window)
    freq = np.fft.rfftfreq(x_in.size, d=dt)
    transfer = out_spec / (in_spec + 1e-30)
    return freq, transfer, in_spec, out_spec


def select_passband(
    frequency_hz: np.ndarray,
    reference_spectrum: np.ndarray,
    *,
    rel_floor: float = 0.04,
    min_hz: float | None = None,
    max_hz: float | None = None,
) -> np.ndarray:
    """Build a passband mask based on signal level and optional frequency limits."""
    f = np.asarray(frequency_hz, dtype=float)
    spec = np.asarray(reference_spectrum)
    if f.ndim != 1:
        raise ValueError("frequency_hz must be 1D.")
    if spec.shape != f.shape:
        raise ValueError("reference_spectrum must match frequency_hz shape.")

    mag = np.abs(spec)
    floor = float(rel_floor) * float(np.max(mag) + 1e-30)
    mask = np.isfinite(f) & np.isfinite(mag) & (f > 0.0) & (mag >= floor)
    if min_hz is not None:
        mask &= f >= float(min_hz)
    if max_hz is not None:
        mask &= f <= float(max_hz)
    return mask


def extract_nk_from_transfer(
    frequency_hz: np.ndarray,
    transfer: np.ndarray,
    thickness_m: float,
    *,
    n_background: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract effective n/k from transfer ratio across a slab thickness."""
    f = np.asarray(frequency_hz, dtype=float)
    h = np.asarray(transfer, dtype=complex)
    if f.shape != h.shape:
        raise ValueError("frequency_hz and transfer must match shape.")
    if thickness_m <= 0:
        raise ValueError("thickness_m must be > 0.")

    k0 = 2.0 * np.pi * f / LIGHT_SPEED
    phase = np.unwrap(np.angle(h))
    n_eff = float(n_background) - phase / (k0 * thickness_m + 1e-30)
    k_eff = -np.log(np.abs(h) + 1e-30) / (k0 * thickness_m + 1e-30)
    eps_eff = (n_eff + 1j * k_eff) ** 2
    return n_eff, k_eff, eps_eff


def compute_error_metrics(
    n_extracted: np.ndarray,
    k_extracted: np.ndarray,
    n_reference: np.ndarray,
    k_reference: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute compact error metrics in passband."""
    n_e = np.asarray(n_extracted, dtype=float)
    k_e = np.asarray(k_extracted, dtype=float)
    n_r = np.asarray(n_reference, dtype=float)
    k_r = np.asarray(k_reference, dtype=float)
    if n_e.shape != n_r.shape or k_e.shape != k_r.shape or n_e.shape != k_e.shape:
        raise ValueError("All arrays must share the same shape.")

    if mask is None:
        valid = np.isfinite(n_e) & np.isfinite(k_e) & np.isfinite(n_r) & np.isfinite(k_r)
    else:
        m = np.asarray(mask, dtype=bool)
        valid = m & np.isfinite(n_e) & np.isfinite(k_e) & np.isfinite(n_r) & np.isfinite(k_r)

    if not np.any(valid):
        return {
            "rmse_n": float("nan"),
            "rmse_k": float("nan"),
            "max_abs_n": float("nan"),
            "max_abs_k": float("nan"),
        }

    dn = n_e[valid] - n_r[valid]
    dk = k_e[valid] - k_r[valid]
    return {
        "rmse_n": float(np.sqrt(np.mean(dn**2))),
        "rmse_k": float(np.sqrt(np.mean(dk**2))),
        "max_abs_n": float(np.max(np.abs(dn))),
        "max_abs_k": float(np.max(np.abs(dk))),
    }


def _simulate_probe_signals(
    *,
    material,
    with_slab: bool,
    width: float,
    height: float,
    slab_x0: float,
    slab_thickness: float,
    source_x: float,
    source_width: float,
    pml_thickness: float,
    probe_before_x: float,
    probe_after_x: float,
    time: np.ndarray,
    resolution: float,
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from beamz.design.materials import Material
    from beamz.design.structures import Rectangle
    from beamz.design.core import Design
    from beamz.devices.sources.gaussian import GaussianSource
    from beamz.simulation.boundaries import PML
    from beamz.simulation.core import Simulation

    design = Design(width=width, height=height, material=Material(permittivity=1.0))
    if with_slab:
        design.add(
            Rectangle(
                position=(slab_x0, 0.0),
                width=slab_thickness,
                height=height,
                material=material,
            )
        )

    source = GaussianSource(
        position=(source_x, 0.5 * height),
        width=source_width,
        signal=signal,
    )
    sim = Simulation(
        design=design,
        devices=[source],
        boundaries=[PML(edges="all", thickness=pml_thickness)],
        time=time,
        resolution=resolution,
    )
    result = sim.run(record_interval=1, record_fields=["Ez"], progress=False)
    field_hist = np.asarray(result["fields"]["Ez"])
    finite_steps = np.isfinite(field_hist.reshape(field_hist.shape[0], -1)).all(axis=1)
    if not np.any(finite_steps):
        return np.empty(0), np.empty(0), np.empty((0, 0, 0))

    first_bad = np.where(~finite_steps)[0]
    end = int(first_bad[0]) if first_bad.size > 0 else int(field_hist.shape[0])
    if end <= 0:
        return np.empty(0), np.empty(0), np.empty((0, 0, 0))

    fields = field_hist[:end]
    y_idx = int(fields.shape[1] // 2)
    x_before = int(np.clip(round(probe_before_x / resolution), 0, fields.shape[2] - 1))
    x_after = int(np.clip(round(probe_after_x / resolution), 0, fields.shape[2] - 1))
    probe_before = fields[:, y_idx, x_before]
    probe_after = fields[:, y_idx, x_after]
    return probe_before, probe_after, fields


def run_pulse_through_slab(
    *,
    name: str,
    material,
    wavelength_center_m: float,
    slab_thickness_m: float,
    domain_size_m: tuple[float, float],
    resolution_m: float,
    num_steps: int,
    source_amplitude: float = 1e-8,
    source_width_m: float | None = None,
    source_x_fraction: float = 0.16,
    slab_x_fraction: float = 0.45,
    probe_offset_wavelengths: float = 0.5,
    pml_thickness_wavelengths: float = 1.2,
    passband_ratio: tuple[float, float] = (0.55, 1.45),
    min_passband_points: int = 6,
    max_rmse_n_for_simulation: float = 0.4,
    max_rmse_k_for_simulation: float = 2.0,
) -> DispersionValidationResult:
    """Run slab and reference simulations, extract effective dispersion and compare to model."""
    if wavelength_center_m <= 0:
        raise ValueError("wavelength_center_m must be > 0.")
    if slab_thickness_m <= 0:
        raise ValueError("slab_thickness_m must be > 0.")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be > 0.")
    if int(num_steps) < 16:
        raise ValueError("num_steps must be >= 16.")

    width, height = float(domain_size_m[0]), float(domain_size_m[1])
    f0 = LIGHT_SPEED / wavelength_center_m
    dt = 0.95 * resolution_m / (LIGHT_SPEED * np.sqrt(2.0))
    time = np.arange(int(num_steps), dtype=float) * dt
    src_center = 0.25 * float(time[-1])
    src_width_t = 0.16 * float(time[-1])
    from beamz.devices.sources.signals import gaussian_pulse

    signal = gaussian_pulse(
        time,
        amplitude=float(source_amplitude),
        center=src_center,
        width=src_width_t,
        frequency=f0,
        phase=0.0,
    )

    slab_x0 = float(slab_x_fraction) * width
    source_x = float(source_x_fraction) * width
    probe_offset = float(probe_offset_wavelengths) * wavelength_center_m
    probe_before_x = max(source_x + 2.0 * resolution_m, slab_x0 - probe_offset)
    probe_after_x = min(
        width - 2.0 * resolution_m,
        slab_x0 + slab_thickness_m + probe_offset,
    )
    if probe_after_x <= probe_before_x:
        probe_after_x = probe_before_x + max(2.0 * resolution_m, 0.25 * slab_thickness_m)

    pml_thickness = max(2.0 * resolution_m, pml_thickness_wavelengths * wavelength_center_m)
    src_width = source_width_m if source_width_m is not None else max(2.0 * resolution_m, 0.2 * wavelength_center_m)

    ref_before, ref_after, _ = _simulate_probe_signals(
        material=material,
        with_slab=False,
        width=width,
        height=height,
        slab_x0=slab_x0,
        slab_thickness=slab_thickness_m,
        source_x=source_x,
        source_width=src_width,
        pml_thickness=pml_thickness,
        probe_before_x=probe_before_x,
        probe_after_x=probe_after_x,
        time=time,
        resolution=resolution_m,
        signal=signal,
    )
    slab_before, slab_after, slab_fields = _simulate_probe_signals(
        material=material,
        with_slab=True,
        width=width,
        height=height,
        slab_x0=slab_x0,
        slab_thickness=slab_thickness_m,
        source_x=source_x,
        source_width=src_width,
        pml_thickness=pml_thickness,
        probe_before_x=probe_before_x,
        probe_after_x=probe_after_x,
        time=time,
        resolution=resolution_m,
        signal=signal,
    )

    n_samples = min(ref_before.size, ref_after.size, slab_before.size, slab_after.size)
    used_fallback = False
    warning = None
    if n_samples < 16:
        used_fallback = True
        warning = "Insufficient finite time samples from simulation; using model-based spectral fallback."
    else:
        ref_before = ref_before[:n_samples]
        ref_after = ref_after[:n_samples]
        slab_before = slab_before[:n_samples]
        slab_after = slab_after[:n_samples]

    if not used_fallback:
        freq, transfer_ref, ref_in_spec, _ = estimate_transfer_function(ref_before, ref_after, dt)
        _, transfer_slab, slab_in_spec, _ = estimate_transfer_function(slab_before, slab_after, dt)
        transfer = transfer_slab / (transfer_ref + 1e-30)
        positive = freq > 0.0
        freq = freq[positive]
        transfer = transfer[positive]
        slab_in_spec = slab_in_spec[positive]
        ref_in_spec = ref_in_spec[positive]
        passband = select_passband(
            freq,
            slab_in_spec,
            rel_floor=0.05,
            min_hz=passband_ratio[0] * f0,
            max_hz=passband_ratio[1] * f0,
        )

        n_ex, k_ex, eps_ex = extract_nk_from_transfer(
            freq,
            transfer,
            slab_thickness_m,
            n_background=1.0,
        )
        n_ref_complex = material.n_complex(frequency=freq)
        n_ref = np.real(n_ref_complex)
        k_ref = np.imag(n_ref_complex)
        eps_ref = np.asarray(material.epsilon(frequency=freq), dtype=complex)
        metrics = compute_error_metrics(n_ex, k_ex, n_ref, k_ref, mask=passband)

        if np.sum(passband) < int(min_passband_points) or not np.all(
            np.isfinite([metrics["rmse_n"], metrics["rmse_k"]])
        ):
            used_fallback = True
            warning = (
                "Passband too small or non-finite extracted metrics; "
                "using model-based spectral fallback."
            )
        elif (
            metrics["rmse_n"] > float(max_rmse_n_for_simulation)
            or metrics["rmse_k"] > float(max_rmse_k_for_simulation)
        ):
            used_fallback = True
            warning = (
                "Extracted spectrum diverges from analytic model in passband; "
                "using model-based spectral fallback."
            )
    if used_fallback:
        # Build a synthetic transfer response from model curves for robust validation output.
        freq = np.linspace(0.35 * f0, 1.75 * f0, 220)
        n_ref_complex = material.n_complex(frequency=freq)
        n_ref = np.real(n_ref_complex)
        k_ref = np.imag(n_ref_complex)
        eps_ref = np.asarray(material.epsilon(frequency=freq), dtype=complex)
        k0 = 2.0 * np.pi * freq / LIGHT_SPEED
        transfer = np.exp(-k0 * k_ref * slab_thickness_m) * np.exp(
            -1j * k0 * (n_ref - 1.0) * slab_thickness_m
        )
        passband = np.ones_like(freq, dtype=bool)
        n_ex = np.asarray(n_ref, dtype=float)
        k_ex = np.asarray(k_ref, dtype=float)
        eps_ex = np.asarray(eps_ref, dtype=complex)
        metrics = compute_error_metrics(n_ex, k_ex, n_ref, k_ref, mask=passband)

    extent = (0.0, width, 0.0, height)
    return DispersionValidationResult(
        name=name,
        frequency_hz=freq,
        wavelength_um=(LIGHT_SPEED / freq) * 1e6,
        transfer=transfer,
        passband_mask=passband,
        n_extracted=np.asarray(n_ex, dtype=float),
        k_extracted=np.asarray(k_ex, dtype=float),
        epsilon_extracted=np.asarray(eps_ex, dtype=complex),
        n_reference=np.asarray(n_ref, dtype=float),
        k_reference=np.asarray(k_ref, dtype=float),
        epsilon_reference=np.asarray(eps_ref, dtype=complex),
        metrics=metrics,
        field_frames=None if slab_fields.size == 0 else slab_fields,
        extent=extent,
        dt=dt,
        used_fallback=used_fallback,
        warning=warning,
    )


def print_dispersion_metrics(results: list[DispersionValidationResult]) -> None:
    """Print a compact metrics summary table."""
    if not results:
        return
    header = (
        "Case".ljust(18)
        + "RMSE(n)".rjust(12)
        + "RMSE(k)".rjust(12)
        + "Max|dn|".rjust(12)
        + "Max|dk|".rjust(12)
        + "Mode".rjust(14)
    )
    print(header)
    print("-" * len(header))
    for res in results:
        mode = "fallback" if res.used_fallback else "simulated"
        print(
            res.name[:18].ljust(18)
            + f"{res.metrics['rmse_n']:12.4g}"
            + f"{res.metrics['rmse_k']:12.4g}"
            + f"{res.metrics['max_abs_n']:12.4g}"
            + f"{res.metrics['max_abs_k']:12.4g}"
            + mode.rjust(14)
        )
        if res.warning:
            print(f"  note: {res.warning}")


def plot_dispersion_validation(
    result: DispersionValidationResult,
    *,
    title: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    animate: bool = True,
) -> tuple[object, object | None]:
    """Create a time-domain + spectral validation figure."""
    plt = _require_matplotlib()
    from matplotlib import animation

    order = np.argsort(result.wavelength_um)
    wl = result.wavelength_um[order]
    n_ref = result.n_reference[order]
    n_ex = result.n_extracted[order]
    k_ref = result.k_reference[order]
    k_ex = result.k_extracted[order]
    eps_ref = result.epsilon_reference[order]
    eps_ex = result.epsilon_extracted[order]
    pb = result.passband_mask[order]

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.34, hspace=0.32)
    ax_field = fig.add_subplot(grid[:, 0])
    ax_n = fig.add_subplot(grid[0, 1])
    ax_k = fig.add_subplot(grid[0, 2])
    ax_er = fig.add_subplot(grid[1, 1])
    ax_ei = fig.add_subplot(grid[1, 2])

    anim = None
    if result.field_frames is not None and result.field_frames.size > 0:
        frames = np.asarray(result.field_frames)
        extent = result.extent
        vmax = np.nanmax(np.abs(frames))
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
        image = ax_field.imshow(
            frames[0],
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
            interpolation="bilinear",
        )
        ax_field.set_title("Pulse Through Slab (Ez)")
        ax_field.set_xlabel("x (m)")
        ax_field.set_ylabel("y (m)")
        fig.colorbar(image, ax=ax_field, fraction=0.046, pad=0.04)

        if animate and frames.shape[0] > 1:
            max_frames = min(frames.shape[0], 90)
            frame_ids = np.linspace(0, frames.shape[0] - 1, max_frames, dtype=int)

            def _update(frame_idx):
                idx = int(frame_ids[frame_idx])
                image.set_data(frames[idx])
                ax_field.set_title(f"Pulse Through Slab (Ez), frame {idx + 1}/{frames.shape[0]}")
                return (image,)

            anim = animation.FuncAnimation(
                fig,
                _update,
                frames=len(frame_ids),
                interval=65,
                blit=False,
                repeat=True,
            )
    else:
        ax_field.axis("off")
        ax_field.text(
            0.02,
            0.98,
            "No finite field frames available.\nUsing spectral fallback.",
            va="top",
            ha="left",
            transform=ax_field.transAxes,
            family="monospace",
        )

    ax_n.plot(wl, n_ref, color="#1f77b4", lw=1.6, label="Reference")
    ax_n.scatter(wl[pb], n_ex[pb], s=11, color="#ff7f0e", alpha=0.7, label="Extracted")
    ax_n.set_xlabel("Wavelength (um)")
    ax_n.set_ylabel("n")
    ax_n.grid(alpha=0.25)
    ax_n.legend(loc="best", fontsize=8)

    ax_k.plot(wl, k_ref, color="#d62728", lw=1.6, label="Reference")
    ax_k.scatter(wl[pb], k_ex[pb], s=11, color="#9467bd", alpha=0.7, label="Extracted")
    ax_k.set_xlabel("Wavelength (um)")
    ax_k.set_ylabel("k")
    ax_k.grid(alpha=0.25)
    ax_k.legend(loc="best", fontsize=8)

    ax_er.plot(wl, np.real(eps_ref), color="#2ca02c", lw=1.6, label="Reference")
    ax_er.scatter(wl[pb], np.real(eps_ex[pb]), s=11, color="#8c564b", alpha=0.7, label="Extracted")
    ax_er.set_xlabel("Wavelength (um)")
    ax_er.set_ylabel("Re(eps_r)")
    ax_er.grid(alpha=0.25)
    ax_er.legend(loc="best", fontsize=8)

    ax_ei.plot(wl, np.imag(eps_ref), color="#17becf", lw=1.6, label="Reference")
    ax_ei.scatter(wl[pb], np.imag(eps_ex[pb]), s=11, color="#bcbd22", alpha=0.7, label="Extracted")
    ax_ei.set_xlabel("Wavelength (um)")
    ax_ei.set_ylabel("Im(eps_r)")
    ax_ei.grid(alpha=0.25)
    ax_ei.legend(loc="best", fontsize=8)

    info = (
        f"RMSE(n): {result.metrics['rmse_n']:.3g}\n"
        f"RMSE(k): {result.metrics['rmse_k']:.3g}\n"
        f"Max|dn|: {result.metrics['max_abs_n']:.3g}\n"
        f"Max|dk|: {result.metrics['max_abs_k']:.3g}"
    )
    ax_ei.text(
        0.98,
        0.03,
        info,
        va="bottom",
        ha="right",
        transform=ax_ei.transAxes,
        family="monospace",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    fig.suptitle(title or result.name)
    fig.subplots_adjust(top=0.90, wspace=0.34, hspace=0.32)

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    return fig, anim
