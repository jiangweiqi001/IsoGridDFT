"""First local-Hamiltonian slice for the structured-grid prototype.

The current local Hamiltonian includes only

    H_local psi = T psi + V_local psi

with T given by the first-stage structured-grid kinetic operator and V_local
currently restricted to the local ionic GTH pseudopotential plus an optional
extra local potential term reserved for future density-derived contributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.ops import apply_kinetic_operator
from isogrid.ops import validate_orbital_field
from isogrid.pseudo import LocalIonicPotentialEvaluation
from isogrid.pseudo import evaluate_local_ionic_potential


@dataclass(frozen=True)
class LocalHamiltonianTerms:
    """Resolved local-Hamiltonian pieces for one orbital field."""

    psi: np.ndarray
    local_potential: np.ndarray
    kinetic_action: np.ndarray
    local_potential_action: np.ndarray
    total_action: np.ndarray


def _resolve_local_potential_array(
    grid_geometry: StructuredGridGeometry,
    case: BenchmarkCase,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None,
    extra_local_potential: np.ndarray | None,
) -> np.ndarray:
    if local_ionic_potential is None:
        base_potential = evaluate_local_ionic_potential(
            case=case,
            grid_geometry=grid_geometry,
        ).total_local_potential
    elif isinstance(local_ionic_potential, LocalIonicPotentialEvaluation):
        base_potential = local_ionic_potential.total_local_potential
    else:
        base_potential = validate_orbital_field(
            local_ionic_potential,
            grid_geometry=grid_geometry,
            name="local_ionic_potential",
        )

    total_potential = np.asarray(base_potential, dtype=np.float64)
    if extra_local_potential is not None:
        total_potential = total_potential + validate_orbital_field(
            extra_local_potential,
            grid_geometry=grid_geometry,
            name="extra_local_potential",
        )
    return total_potential


def evaluate_local_hamiltonian_terms(
    psi: np.ndarray,
    grid_geometry: StructuredGridGeometry,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
    extra_local_potential: np.ndarray | None = None,
) -> LocalHamiltonianTerms:
    """Resolve T psi, V_local psi, and their sum for one orbital field."""

    field = validate_orbital_field(psi, grid_geometry=grid_geometry)
    local_potential = _resolve_local_potential_array(
        grid_geometry=grid_geometry,
        case=case,
        local_ionic_potential=local_ionic_potential,
        extra_local_potential=extra_local_potential,
    )
    kinetic_action = apply_kinetic_operator(psi=field, grid_geometry=grid_geometry)
    local_potential_action = local_potential * field
    total_action = kinetic_action + local_potential_action
    return LocalHamiltonianTerms(
        psi=field,
        local_potential=local_potential,
        kinetic_action=kinetic_action,
        local_potential_action=local_potential_action,
        total_action=total_action,
    )


def apply_local_hamiltonian(
    psi: np.ndarray,
    grid_geometry: StructuredGridGeometry,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
    extra_local_potential: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the current first-stage local Hamiltonian to one orbital field."""

    return evaluate_local_hamiltonian_terms(
        psi=psi,
        grid_geometry=grid_geometry,
        case=case,
        local_ionic_potential=local_ionic_potential,
        extra_local_potential=extra_local_potential,
    ).total_action


def build_default_h2_local_hamiltonian_action(
    psi: np.ndarray,
    grid_geometry: StructuredGridGeometry | None = None,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
) -> LocalHamiltonianTerms:
    """Convenience wrapper for the default H2 benchmark configuration."""

    if grid_geometry is None:
        grid_geometry = build_default_h2_grid_geometry(case=H2_BENCHMARK_CASE)
    return evaluate_local_hamiltonian_terms(
        psi=psi,
        grid_geometry=grid_geometry,
        case=H2_BENCHMARK_CASE,
        local_ionic_potential=local_ionic_potential,
    )
