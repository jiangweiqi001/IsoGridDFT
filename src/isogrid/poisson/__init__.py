"""Open-boundary Poisson and Hartree helpers for the stage-1 prototype."""

from .hartree import HartreeEvaluation
from .hartree import build_hartree_action
from .hartree import evaluate_hartree_energy
from .hartree import evaluate_hartree_terms
from .hartree import solve_hartree_potential
from .hartree import validate_density_field
from .open_boundary import OpenBoundaryMultipoleBoundary
from .open_boundary import OpenBoundaryPoissonResult
from .open_boundary import solve_open_boundary_poisson
from .poisson_jax import MonitorPoissonJaxSolveDiagnostics
from .poisson_jax import apply_monitor_open_boundary_poisson_operator_jax
from .poisson_jax import build_monitor_open_boundary_poisson_matvec_jax
from .poisson_jax import solve_open_boundary_poisson_monitor_jax

__all__ = [
    "HartreeEvaluation",
    "MonitorPoissonJaxSolveDiagnostics",
    "OpenBoundaryMultipoleBoundary",
    "OpenBoundaryPoissonResult",
    "apply_monitor_open_boundary_poisson_operator_jax",
    "build_monitor_open_boundary_poisson_matvec_jax",
    "build_hartree_action",
    "evaluate_hartree_energy",
    "evaluate_hartree_terms",
    "solve_open_boundary_poisson_monitor_jax",
    "solve_hartree_potential",
    "solve_open_boundary_poisson",
    "validate_density_field",
]
