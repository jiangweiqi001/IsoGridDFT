"""First static KS Hamiltonian backbone for the structured-grid prototype.

The current static KS slice assembles

    H_ks_static psi = T psi + V_loc,ion psi + V_nl,ion psi + V_xc psi

with the following scope restrictions:

- no Hartree term yet
- no SCF loop yet
- rho_up and rho_down are external inputs
- only the current PySCF-aligned `lda,vwn` LSDA local term is supported

This module is meant to be the first formal static KS backbone, not the final
production Hamiltonian path.
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
from isogrid.pseudo import NonlocalIonicActionEvaluation
from isogrid.pseudo import evaluate_local_ionic_potential
from isogrid.pseudo import evaluate_nonlocal_ionic_action
from isogrid.xc import LSDAEvaluation
from isogrid.xc import evaluate_lsda_terms

_VALID_SPIN_CHANNELS = {"up", "down"}


@dataclass(frozen=True)
class StaticKSHamiltonianTerms:
    """Resolved static KS action pieces for one orbital and one spin channel."""

    psi: np.ndarray
    rho_up: np.ndarray
    rho_down: np.ndarray
    spin_channel: str
    kinetic_action: np.ndarray
    local_ionic_potential: np.ndarray
    local_ionic_action: np.ndarray
    nonlocal_ionic_action: np.ndarray
    xc_potential: np.ndarray
    xc_action: np.ndarray
    total_action: np.ndarray
    local_ionic_evaluation: LocalIonicPotentialEvaluation | None
    nonlocal_ionic_evaluation: NonlocalIonicActionEvaluation | None
    lsda_evaluation: LSDAEvaluation


def _normalize_spin_channel(spin_channel: str) -> str:
    normalized = spin_channel.strip().lower()
    if normalized not in _VALID_SPIN_CHANNELS:
        raise ValueError(
            "spin_channel must be `up` or `down`; "
            f"received `{spin_channel}`."
        )
    return normalized


def _resolve_local_ionic_potential(
    grid_geometry: StructuredGridGeometry,
    case: BenchmarkCase,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None,
) -> tuple[np.ndarray, LocalIonicPotentialEvaluation | None]:
    if local_ionic_potential is None:
        evaluation = evaluate_local_ionic_potential(case=case, grid_geometry=grid_geometry)
        return evaluation.total_local_potential, evaluation
    if isinstance(local_ionic_potential, LocalIonicPotentialEvaluation):
        return local_ionic_potential.total_local_potential, local_ionic_potential
    return (
        validate_orbital_field(
            local_ionic_potential,
            grid_geometry=grid_geometry,
            name="local_ionic_potential",
        ),
        None,
    )


def _resolve_nonlocal_ionic_action(
    psi: np.ndarray,
    grid_geometry: StructuredGridGeometry,
    case: BenchmarkCase,
    nonlocal_ionic_action: NonlocalIonicActionEvaluation | np.ndarray | None,
) -> tuple[np.ndarray, NonlocalIonicActionEvaluation | None]:
    if nonlocal_ionic_action is None:
        evaluation = evaluate_nonlocal_ionic_action(
            case=case,
            grid_geometry=grid_geometry,
            psi=psi,
        )
        return evaluation.total_nonlocal_action, evaluation
    if isinstance(nonlocal_ionic_action, NonlocalIonicActionEvaluation):
        return nonlocal_ionic_action.total_nonlocal_action, nonlocal_ionic_action
    return (
        validate_orbital_field(
            nonlocal_ionic_action,
            grid_geometry=grid_geometry,
            name="nonlocal_ionic_action",
        ),
        None,
    )


def build_orbital_density(
    psi: np.ndarray,
    grid_geometry: StructuredGridGeometry,
    occupation: float = 1.0,
) -> np.ndarray:
    """Build a simple orbital density rho = occupation * |psi|^2."""

    if occupation < 0.0:
        raise ValueError(f"occupation must be non-negative; received {occupation}.")
    field = validate_orbital_field(psi, grid_geometry=grid_geometry)
    return occupation * np.abs(field) ** 2


def build_singlet_like_spin_densities(
    psi: np.ndarray,
    grid_geometry: StructuredGridGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a minimal closed-shell-like spin density pair from one orbital."""

    orbital_density = build_orbital_density(
        psi=psi,
        grid_geometry=grid_geometry,
        occupation=1.0,
    )
    return orbital_density, orbital_density.copy()


def evaluate_static_ks_terms(
    psi: np.ndarray,
    grid_geometry: StructuredGridGeometry,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    spin_channel: str,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
    nonlocal_ionic_action: NonlocalIonicActionEvaluation | np.ndarray | None = None,
    xc_functional: str | None = None,
) -> StaticKSHamiltonianTerms:
    """Resolve the current static KS action and all of its explicit pieces."""

    field = validate_orbital_field(psi, grid_geometry=grid_geometry)
    rho_up_field = validate_orbital_field(rho_up, grid_geometry=grid_geometry, name="rho_up")
    rho_down_field = validate_orbital_field(rho_down, grid_geometry=grid_geometry, name="rho_down")
    normalized_spin = _normalize_spin_channel(spin_channel)
    local_potential, local_evaluation = _resolve_local_ionic_potential(
        grid_geometry=grid_geometry,
        case=case,
        local_ionic_potential=local_ionic_potential,
    )
    nonlocal_action, nonlocal_evaluation = _resolve_nonlocal_ionic_action(
        psi=field,
        grid_geometry=grid_geometry,
        case=case,
        nonlocal_ionic_action=nonlocal_ionic_action,
    )
    lsda_evaluation = evaluate_lsda_terms(
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        functional=case.reference_model.xc if xc_functional is None else xc_functional,
    )
    xc_potential = (
        lsda_evaluation.v_xc_up if normalized_spin == "up" else lsda_evaluation.v_xc_down
    )

    kinetic_action = apply_kinetic_operator(psi=field, grid_geometry=grid_geometry)
    local_ionic_action = local_potential * field
    xc_action = xc_potential * field
    total_action = kinetic_action + local_ionic_action + nonlocal_action + xc_action
    return StaticKSHamiltonianTerms(
        psi=field,
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        spin_channel=normalized_spin,
        kinetic_action=kinetic_action,
        local_ionic_potential=local_potential,
        local_ionic_action=local_ionic_action,
        nonlocal_ionic_action=nonlocal_action,
        xc_potential=xc_potential,
        xc_action=xc_action,
        total_action=total_action,
        local_ionic_evaluation=local_evaluation,
        nonlocal_ionic_evaluation=nonlocal_evaluation,
        lsda_evaluation=lsda_evaluation,
    )


def apply_static_ks_hamiltonian(
    psi: np.ndarray,
    grid_geometry: StructuredGridGeometry,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    spin_channel: str,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
    nonlocal_ionic_action: NonlocalIonicActionEvaluation | np.ndarray | None = None,
    xc_functional: str | None = None,
) -> np.ndarray:
    """Apply the first-stage static KS Hamiltonian to one orbital field."""

    return evaluate_static_ks_terms(
        psi=psi,
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel=spin_channel,
        case=case,
        local_ionic_potential=local_ionic_potential,
        nonlocal_ionic_action=nonlocal_ionic_action,
        xc_functional=xc_functional,
    ).total_action


def build_default_h2_static_ks_action(
    psi: np.ndarray,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    spin_channel: str = "up",
    grid_geometry: StructuredGridGeometry | None = None,
    local_ionic_potential: LocalIonicPotentialEvaluation | np.ndarray | None = None,
    nonlocal_ionic_action: NonlocalIonicActionEvaluation | np.ndarray | None = None,
) -> StaticKSHamiltonianTerms:
    """Convenience wrapper for the default H2 benchmark configuration."""

    if grid_geometry is None:
        grid_geometry = build_default_h2_grid_geometry(case=H2_BENCHMARK_CASE)
    return evaluate_static_ks_terms(
        psi=psi,
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel=spin_channel,
        case=H2_BENCHMARK_CASE,
        local_ionic_potential=local_ionic_potential,
        nonlocal_ionic_action=nonlocal_ionic_action,
    )
