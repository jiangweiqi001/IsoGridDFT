"""Configuration helpers for benchmark defaults and runtime setup."""

from .defaults import H2_BASIS_CONVERGENCE_BASES
from .defaults import H2_BASIS_CONVERGENCE_PROTOCOL_DESCRIPTION
from .defaults import H2_BENCHMARK_CASE
from .defaults import H2_BOND_LENGTH_BOHR
from .defaults import PYSCF_AUDIT_DEFAULT_CASE
from .model import AtomSpec
from .model import BenchmarkCase
from .model import MoleculeGeometry
from .model import ReferenceModelSettings
from .model import ScfSettings
from .model import SpinStateSpec
from .runtime import JAX_X64_DEFAULT
from .runtime import SCIENTIFIC_DTYPE_NAME
from .runtime import configure_jax_runtime
from .runtime import get_jax_scientific_dtype
from .runtime import is_jax_available
from .runtime import require_jax

__all__ = [
    "AtomSpec",
    "BenchmarkCase",
    "MoleculeGeometry",
    "ReferenceModelSettings",
    "ScfSettings",
    "SpinStateSpec",
    "H2_BASIS_CONVERGENCE_BASES",
    "H2_BASIS_CONVERGENCE_PROTOCOL_DESCRIPTION",
    "H2_BENCHMARK_CASE",
    "H2_BOND_LENGTH_BOHR",
    "PYSCF_AUDIT_DEFAULT_CASE",
    "JAX_X64_DEFAULT",
    "SCIENTIFIC_DTYPE_NAME",
    "configure_jax_runtime",
    "get_jax_scientific_dtype",
    "is_jax_available",
    "require_jax",
]
