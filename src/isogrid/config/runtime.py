"""Backward-compatible runtime imports for the scientific JAX path."""

from .runtime_jax import ISOGRID_JAX_DISABLE_JIT_ENVVAR
from .runtime_jax import ISOGRID_JAX_PLATFORM_ENVVAR
from .runtime_jax import JAX_DISABLE_JIT_DEFAULT
from .runtime_jax import JAX_PLATFORM_DEFAULT
from .runtime_jax import JAX_X64_DEFAULT
from .runtime_jax import JaxRuntimeConfiguration
from .runtime_jax import SCIENTIFIC_DTYPE_NAME
from .runtime_jax import configure_jax_runtime
from .runtime_jax import get_configured_jax
from .runtime_jax import get_configured_jax_numpy
from .runtime_jax import get_jax_runtime_configuration
from .runtime_jax import get_jax_scientific_dtype
from .runtime_jax import is_jax_available
from .runtime_jax import require_jax

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
