"""Central JAX runtime helpers for the scientific path.

This module is the single place where IsoGridDFT configures JAX runtime
behavior for the scientific path. The current first-stage defaults are:

- float64 enabled by default
- JIT enabled by default
- platform selection optional and environment-driven

The goal here is not to design a full runtime framework yet. It is simply to
keep the JAX configuration explicit and out of the numerical kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import os

JAX_X64_DEFAULT = True
JAX_DISABLE_JIT_DEFAULT = False
JAX_PLATFORM_DEFAULT: str | None = None
SCIENTIFIC_DTYPE_NAME = "float64"
ISOGRID_JAX_DISABLE_JIT_ENVVAR = "ISOGRID_JAX_DISABLE_JIT"
ISOGRID_JAX_PLATFORM_ENVVAR = "ISOGRID_JAX_PLATFORM"


@dataclass(frozen=True)
class JaxRuntimeConfiguration:
    """Resolved JAX runtime settings for the scientific path."""

    enable_x64: bool
    disable_jit: bool
    platform_name: str | None


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
            "Install a suitable JAX build before calling JAX kernels."
        ) from exc
    return jax


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable {name} must encode a boolean value; received `{value}`."
    )


def get_jax_runtime_configuration(
    *,
    enable_x64: bool = JAX_X64_DEFAULT,
    disable_jit: bool | None = None,
    platform_name: str | None = None,
) -> JaxRuntimeConfiguration:
    """Resolve the JAX runtime configuration for the scientific path."""

    resolved_disable_jit = (
        _env_bool(ISOGRID_JAX_DISABLE_JIT_ENVVAR, JAX_DISABLE_JIT_DEFAULT)
        if disable_jit is None
        else bool(disable_jit)
    )
    resolved_platform_name = (
        os.environ.get(ISOGRID_JAX_PLATFORM_ENVVAR, JAX_PLATFORM_DEFAULT)
        if platform_name is None
        else platform_name
    )
    if resolved_platform_name is not None:
        resolved_platform_name = resolved_platform_name.strip() or None
    return JaxRuntimeConfiguration(
        enable_x64=bool(enable_x64),
        disable_jit=resolved_disable_jit,
        platform_name=resolved_platform_name,
    )


def configure_jax_runtime(
    *,
    enable_x64: bool = JAX_X64_DEFAULT,
    disable_jit: bool | None = None,
    platform_name: str | None = None,
):
    """Apply the minimum JAX runtime settings used by the scientific path."""

    configuration = get_jax_runtime_configuration(
        enable_x64=enable_x64,
        disable_jit=disable_jit,
        platform_name=platform_name,
    )
    jax = require_jax()
    jax.config.update("jax_enable_x64", configuration.enable_x64)
    jax.config.update("jax_disable_jit", configuration.disable_jit)
    if configuration.platform_name is not None:
        jax.config.update("jax_platform_name", configuration.platform_name)
    return jax


def get_configured_jax(
    *,
    enable_x64: bool = JAX_X64_DEFAULT,
    disable_jit: bool | None = None,
    platform_name: str | None = None,
):
    """Return JAX after applying the scientific-path runtime settings."""

    return configure_jax_runtime(
        enable_x64=enable_x64,
        disable_jit=disable_jit,
        platform_name=platform_name,
    )


def get_configured_jax_numpy(
    *,
    enable_x64: bool = JAX_X64_DEFAULT,
    disable_jit: bool | None = None,
    platform_name: str | None = None,
):
    """Return `jax.numpy` after applying the scientific-path runtime settings."""

    jax = get_configured_jax(
        enable_x64=enable_x64,
        disable_jit=disable_jit,
        platform_name=platform_name,
    )
    return jax.numpy


def get_jax_scientific_dtype():
    """Return the JAX float64 dtype used by the scientific path."""

    jnp = get_configured_jax_numpy()
    return jnp.float64


__all__ = [
    "ISOGRID_JAX_DISABLE_JIT_ENVVAR",
    "ISOGRID_JAX_PLATFORM_ENVVAR",
    "JAX_DISABLE_JIT_DEFAULT",
    "JAX_PLATFORM_DEFAULT",
    "JAX_X64_DEFAULT",
    "JaxRuntimeConfiguration",
    "SCIENTIFIC_DTYPE_NAME",
    "configure_jax_runtime",
    "get_configured_jax",
    "get_configured_jax_numpy",
    "get_jax_runtime_configuration",
    "get_jax_scientific_dtype",
    "is_jax_available",
    "require_jax",
]
