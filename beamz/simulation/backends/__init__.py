"""
Backend implementations for FDTD simulations.
"""
import logging

logger = logging.getLogger(__name__)


def _load_numpy_backend(**kwargs):
    from beamz.simulation.backends.numpy_backend import NumPyBackend

    return NumPyBackend(**kwargs)


def _load_jax_backend(**kwargs):
    try:
        import jax  # noqa: F401  # Lazy import to detect availability
    except ImportError as exc:  # pragma: no cover - handled by caller
        raise ImportError("JAX backend requested but JAX is not installed") from exc

    from beamz.simulation.backends.jax_backend import JAXBackend

    return JAXBackend(**kwargs)


def get_backend(name="auto", **kwargs):
    """Select the backend used for FDTD simulations.

    Parameters
    ----------
    name:
        Name of the backend to load. Supported values are ``"jax"``, ``"numpy"`` and
        ``"torch"``.  When ``"auto"`` (the default) is provided the function attempts to
        initialise the high-performance JAX backend and falls back to NumPy if JAX is not
        available.
    **kwargs:
        Keyword arguments forwarded to the backend initialiser.
    """

    if name is None:
        name = "auto"

    name = name.lower()

    if name == "auto":
        try:
            return _load_jax_backend(**kwargs)
        except ImportError as exc:
            logger.warning("JAX backend unavailable (%s); falling back to NumPy", exc)
            return _load_numpy_backend(**kwargs)

    if name == "numpy":
        return _load_numpy_backend(**kwargs)

    if name == "jax":
        try:
            return _load_jax_backend(**kwargs)
        except ImportError as exc:
            logger.warning("JAX backend unavailable (%s); falling back to NumPy", exc)
            return _load_numpy_backend(**kwargs)

    if name == "torch":
        try:
            import torch  # noqa: F401
            from beamz.simulation.backends.torch_backend import TorchBackend

            return TorchBackend(**kwargs)
        except ImportError:
            logger.warning("PyTorch not available, falling back to NumPy backend")
            return _load_numpy_backend(**kwargs)

    raise ValueError(f"Unknown backend: {name}")


# Export available backends
__all__ = ["get_backend"]
