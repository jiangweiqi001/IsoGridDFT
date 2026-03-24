"""Minimal JAX runtime helpers for the scientific path."""

from __future__ import annotations

from importlib.util import find_spec

JAX_X64_DEFAULT = True
SCIENTIFIC_DTYPE_NAME = "float64"


def is_jax_available() -> bool:
    """Return whether JAX can be imported in the current environment."""

    return find_spec("jax") is not None


def require_jax():
    """Import JAX or raise a clear runtime error."""

    try:
        import jax
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "JAX is required for the IsoGridDFT scientific runtime. "
            "Install a suitable JAX build before calling runtime helpers."
        ) from exc
    return jax


def configure_jax_runtime(enable_x64: bool = JAX_X64_DEFAULT):
    """Apply the minimum JAX runtime settings used by the scientific path."""

    jax = require_jax()
    jax.config.update("jax_enable_x64", enable_x64)
    return jax


def get_jax_scientific_dtype():
    """Return the scientific-path JAX dtype after importing JAX."""

    jax = require_jax()
    return jax.numpy.float64
