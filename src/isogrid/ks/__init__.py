"""Kohn-Sham Hamiltonian helpers for the structured-grid prototype."""

from .eigensolver import FixedPotentialEigensolverResult
from .eigensolver import FixedPotentialOperatorContext
from .eigensolver import FixedPotentialStaticLocalOperatorContext
from .eigensolver import FixedPotentialStaticLocalPreparationProfile
from .eigensolver import apply_fixed_potential_static_ks_block
from .eigensolver import apply_fixed_potential_static_ks_operator
from .eigensolver import apply_fixed_potential_static_local_block
from .eigensolver import apply_fixed_potential_static_local_operator
from .eigensolver import build_fixed_potential_static_ks_operator
from .eigensolver import build_fixed_potential_static_local_operator
from .eigensolver import flatten_orbital_block
from .eigensolver import prepare_fixed_potential_static_ks_operator
from .eigensolver import prepare_fixed_potential_static_local_operator
from .eigensolver import prepare_fixed_potential_static_local_operator_profiled
from .eigensolver import reshape_orbital_columns
from .eigensolver import solve_fixed_potential_eigenproblem
from .eigensolver import solve_fixed_potential_static_local_eigenproblem
from .eigensolver import validate_orbital_block
from .eigensolver import weighted_orbital_norms
from .eigensolver import weighted_orthonormalize_orbitals
from .eigensolver import weighted_overlap_matrix
from .hamiltonian_local_jax import apply_fixed_potential_static_local_block_jax
from .hamiltonian_local_jax import apply_fixed_potential_static_local_operator_jax
from .hamiltonian_local_jax import build_fixed_potential_static_local_operator_jax
from .local_hamiltonian import LocalHamiltonianTerms
from .local_hamiltonian import apply_local_hamiltonian
from .local_hamiltonian import build_default_h2_local_hamiltonian_action
from .local_hamiltonian import evaluate_local_hamiltonian_terms
from .static_hamiltonian import StaticKSHamiltonianTerms
from .static_hamiltonian import apply_static_ks_hamiltonian
from .static_hamiltonian import build_default_h2_static_ks_action
from .static_hamiltonian import build_orbital_density
from .static_hamiltonian import build_singlet_like_spin_densities
from .static_hamiltonian import build_total_density
from .static_hamiltonian import evaluate_static_ks_terms

__all__ = [
    "FixedPotentialEigensolverResult",
    "FixedPotentialOperatorContext",
    "FixedPotentialStaticLocalOperatorContext",
    "FixedPotentialStaticLocalPreparationProfile",
    "LocalHamiltonianTerms",
    "StaticKSHamiltonianTerms",
    "apply_fixed_potential_static_ks_block",
    "apply_fixed_potential_static_ks_operator",
    "apply_fixed_potential_static_local_block",
    "apply_fixed_potential_static_local_block_jax",
    "apply_fixed_potential_static_local_operator",
    "apply_fixed_potential_static_local_operator_jax",
    "apply_local_hamiltonian",
    "apply_static_ks_hamiltonian",
    "build_default_h2_local_hamiltonian_action",
    "build_default_h2_static_ks_action",
    "build_fixed_potential_static_ks_operator",
    "build_fixed_potential_static_local_operator",
    "build_fixed_potential_static_local_operator_jax",
    "build_orbital_density",
    "build_singlet_like_spin_densities",
    "build_total_density",
    "evaluate_local_hamiltonian_terms",
    "evaluate_static_ks_terms",
    "flatten_orbital_block",
    "prepare_fixed_potential_static_ks_operator",
    "prepare_fixed_potential_static_local_operator",
    "prepare_fixed_potential_static_local_operator_profiled",
    "reshape_orbital_columns",
    "solve_fixed_potential_eigenproblem",
    "solve_fixed_potential_static_local_eigenproblem",
    "validate_orbital_block",
    "weighted_orbital_norms",
    "weighted_orthonormalize_orbitals",
    "weighted_overlap_matrix",
]
