"""First minimal H2 SCF driver for the structured-grid prototype.

This module intentionally implements only the smallest formal SCF loop needed
to close the H2 single-point path. The current stage-1 flow is:

1. choose one of the configured H2 candidate spin states
2. build a deterministic initial orbital / density guess
3. freeze the current density and solve the static KS problem for each spin
   channel with the existing fixed-potential eigensolver
4. rebuild rho_up and rho_down from the occupied orbitals
5. apply simple linear density mixing
6. monitor density residual and total-energy change
7. report a single-point total energy together with explicit component terms

This is the first formal SCF driver slice. It is restricted on purpose:

- only the current neutral H2 benchmark is supported
- only the configured singlet and triplet candidates are supported
- only linear density mixing is implemented
- the total energy is evaluated as

      E = T_s + E_loc,ion + E_nl,ion + E_H + E_xc + E_II

  on the current discrete adaptive grid

It is not yet a general SCF framework and it is not the final production path.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_RADIUS_SCALE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_WEIGHT_SCALE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.ks import FixedPotentialEigensolverResult
from isogrid.ks import FixedPotentialStaticLocalOperatorContext
from isogrid.ks import FixedPotentialStaticLocalPreparationProfile
from isogrid.ks import build_total_density
from isogrid.ks import prepare_fixed_potential_static_local_operator_profiled
from isogrid.ks import solve_fixed_potential_eigenproblem
from isogrid.ks import solve_fixed_potential_static_local_eigenproblem
from isogrid.ks import validate_orbital_block
from isogrid.ks import weighted_overlap_matrix
from isogrid.ks import weighted_orthonormalize_orbitals
from isogrid.ops import apply_kinetic_operator
from isogrid.ops import integrate_field
from isogrid.ops import validate_orbital_field
from isogrid.ops import weighted_l2_norm
from isogrid.ops.kinetic import apply_monitor_grid_kinetic_operator_trial_boundary_fix
from isogrid.poisson import evaluate_hartree_energy
from isogrid.poisson import solve_hartree_potential
from isogrid.pseudo import evaluate_local_ionic_potential
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential_with_patch
from isogrid.pseudo import evaluate_nonlocal_ionic_action
from isogrid.pseudo import LocalPotentialPatchParameters
from isogrid.pseudo import load_case_gth_pseudo_data
from isogrid.xc import evaluate_lsda_terms
from isogrid.poisson.poisson_jax import clear_monitor_poisson_jax_kernel_cache
from isogrid.poisson.poisson_jax import get_last_monitor_poisson_jax_solve_diagnostics

_SUPPORTED_CASE_NAME = H2_BENCHMARK_CASE.name
_ZERO_BLOCK_DTYPE = np.float64
GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry


@dataclass(frozen=True)
class SpinOccupations:
    """Minimal spin occupation data for the current H2 SCF driver."""

    label: str
    spin: int
    total_electrons: int
    n_alpha: int
    n_beta: int
    occupations_up: tuple[float, ...]
    occupations_down: tuple[float, ...]


@dataclass(frozen=True)
class FixedPotentialSolveSummary:
    """Compact audit-facing summary of one frozen-potential solve."""

    spin_channel: str
    target_orbitals: int
    solver_method: str
    solver_note: str
    converged: bool
    eigenvalues: np.ndarray
    residual_norms: np.ndarray


@dataclass(frozen=True)
class SinglePointEnergyComponents:
    """Single-point energy components for the current SCF density/orbitals."""

    kinetic: float
    local_ionic: float
    nonlocal_ionic: float
    hartree: float
    xc: float
    ion_ion_repulsion: float
    total: float


@dataclass(frozen=True)
class ScfIterationRecord:
    """Per-iteration audit record for the minimal H2 SCF driver."""

    iteration: int
    input_rho_up: np.ndarray
    input_rho_down: np.ndarray
    output_rho_up: np.ndarray
    output_rho_down: np.ndarray
    mixed_rho_up: np.ndarray
    mixed_rho_down: np.ndarray
    density_residual: float
    energy: SinglePointEnergyComponents
    energy_change: float | None
    solve_up: FixedPotentialSolveSummary | None
    solve_down: FixedPotentialSolveSummary | None


@dataclass(frozen=True)
class H2ScfResult:
    """Final result of the first minimal H2 single-point SCF loop."""

    converged: bool
    spin_state_label: str
    spin: int
    occupations: SpinOccupations
    iteration_count: int
    history: tuple[ScfIterationRecord, ...]
    eigenvalues_up: np.ndarray
    eigenvalues_down: np.ndarray
    orbitals_up: np.ndarray
    orbitals_down: np.ndarray
    rho_up: np.ndarray
    rho_down: np.ndarray
    energy: SinglePointEnergyComponents
    solve_up: FixedPotentialEigensolverResult | None
    solve_down: FixedPotentialEigensolverResult | None


@dataclass(frozen=True)
class H2ScfDryRunParameterSummary:
    """Fixed parameter summary for the monitor-grid SCF dry-run."""

    grid_shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    weight_scale: float
    radius_scale: float
    patch_radius_scale: float
    patch_grid_shape: tuple[int, int, int]
    correction_strength: float
    interpolation_neighbors: int
    kinetic_version: str
    hartree_backend: str
    use_jax_hartree_cached_operator: bool
    jax_hartree_cg_impl: str
    jax_hartree_cg_preconditioner: str
    jax_hartree_line_preconditioner_impl: str
    use_jax_block_kernels: bool
    use_step_local_static_local_reuse: bool
    includes_nonlocal: bool
    cycle_breaker_enabled: bool
    cycle_breaker_weight: float
    diis_enabled: bool
    diis_warmup_iterations: int
    diis_history_length: int
    diis_residual_definition: str


@dataclass(frozen=True)
class MonitorGridDiisHistoryEntry:
    """Small DIIS history item for the monitor-grid singlet dry-run."""

    mixed_rho_up: np.ndarray
    mixed_rho_down: np.ndarray
    residual_up: np.ndarray
    residual_down: np.ndarray


@dataclass(frozen=True)
class StaticLocalEnergyEvaluationProfile:
    """Very small timing summary for local-only single-point energy evaluation."""

    kinetic_wall_time_seconds: float
    local_ionic_wall_time_seconds: float
    hartree_energy_wall_time_seconds: float
    xc_energy_wall_time_seconds: float
    ion_ion_wall_time_seconds: float


@dataclass(frozen=True)
class H2StaticLocalScfDryRunResult:
    """Result of the first H2 A-grid static-local SCF dry-run."""

    path_type: str
    kinetic_version: str
    includes_nonlocal: bool
    spin_state_label: str
    spin: int
    occupations: SpinOccupations
    parameter_summary: H2ScfDryRunParameterSummary
    hartree_backend: str
    use_jax_hartree_cached_operator: bool
    jax_hartree_cg_impl: str
    jax_hartree_cg_preconditioner: str
    jax_hartree_line_preconditioner_impl: str
    use_jax_block_kernels: bool
    use_step_local_static_local_reuse: bool
    cycle_breaker_enabled: bool
    cycle_breaker_weight: float
    cycle_breaker_triggered_iterations: tuple[int, ...]
    diis_enabled: bool
    diis_warmup_iterations: int
    diis_history_length: int
    diis_residual_definition: str
    diis_used_iterations: tuple[int, ...]
    diis_history_sizes: tuple[int, ...]
    diis_fallback_iterations: tuple[int, ...]
    converged: bool
    iteration_count: int
    history: tuple[ScfIterationRecord, ...]
    energy_history: tuple[float, ...]
    density_residual_history: tuple[float, ...]
    eigenvalues_up: np.ndarray
    eigenvalues_down: np.ndarray
    orbitals_up: np.ndarray
    orbitals_down: np.ndarray
    rho_up: np.ndarray
    rho_down: np.ndarray
    energy: SinglePointEnergyComponents
    lowest_eigenvalue: float | None
    solve_up: FixedPotentialEigensolverResult | None
    solve_down: FixedPotentialEigensolverResult | None
    total_wall_time_seconds: float
    average_iteration_wall_time_seconds: float | None
    eigensolver_wall_time_seconds: float
    static_local_prepare_wall_time_seconds: float
    hartree_solve_wall_time_seconds: float
    local_ionic_resolve_wall_time_seconds: float
    xc_resolve_wall_time_seconds: float
    energy_evaluation_wall_time_seconds: float
    kinetic_energy_wall_time_seconds: float
    local_ionic_energy_wall_time_seconds: float
    hartree_energy_wall_time_seconds: float
    xc_energy_wall_time_seconds: float
    ion_ion_energy_wall_time_seconds: float
    density_update_wall_time_seconds: float
    bookkeeping_wall_time_seconds: float
    hartree_solve_call_count: int
    average_hartree_solve_wall_time_seconds: float | None
    first_hartree_solve_wall_time_seconds: float | None
    repeated_hartree_solve_average_wall_time_seconds: float | None
    repeated_hartree_solve_min_wall_time_seconds: float | None
    repeated_hartree_solve_max_wall_time_seconds: float | None
    average_hartree_cg_iterations: float | None
    first_hartree_cg_iterations: int | None
    repeated_hartree_cg_iteration_average: float | None
    average_hartree_boundary_condition_wall_time_seconds: float | None
    average_hartree_build_wall_time_seconds: float | None
    average_hartree_rhs_assembly_wall_time_seconds: float | None
    average_hartree_cg_wall_time_seconds: float | None
    average_hartree_matvec_call_count: float | None
    average_hartree_matvec_wall_time_seconds: float | None
    average_hartree_matvec_wall_time_per_call_seconds: float | None
    average_hartree_preconditioner_apply_count: float | None
    average_hartree_preconditioner_apply_wall_time_seconds: float | None
    average_hartree_preconditioner_apply_wall_time_per_call_seconds: float | None
    average_hartree_preconditioner_setup_wall_time_seconds: float | None
    average_hartree_preconditioner_axis_reorder_wall_time_seconds: float | None
    average_hartree_preconditioner_tridiagonal_solve_wall_time_seconds: float | None
    average_hartree_preconditioner_other_overhead_wall_time_seconds: float | None
    hartree_cached_operator_usage_count: int
    hartree_cached_operator_first_solve_count: int
    hartree_solve_wall_time_seconds_history: tuple[float, ...]
    hartree_cg_iterations_history: tuple[int, ...]
    hartree_boundary_condition_wall_time_seconds_history: tuple[float, ...]
    hartree_build_wall_time_seconds_history: tuple[float, ...]
    hartree_rhs_assembly_wall_time_seconds_history: tuple[float, ...]
    hartree_cg_wall_time_seconds_history: tuple[float, ...]
    hartree_matvec_call_count_history: tuple[int, ...]
    hartree_matvec_wall_time_seconds_history: tuple[float, ...]
    hartree_preconditioner_apply_count_history: tuple[int, ...]
    hartree_preconditioner_apply_wall_time_seconds_history: tuple[float, ...]
    hartree_preconditioner_setup_wall_time_seconds_history: tuple[float, ...]
    hartree_preconditioner_axis_reorder_wall_time_seconds_history: tuple[float, ...]
    hartree_preconditioner_tridiagonal_solve_wall_time_seconds_history: tuple[float, ...]
    hartree_preconditioner_other_overhead_wall_time_seconds_history: tuple[float, ...]
    eigensolver_iteration_wall_time_seconds: tuple[float, ...]
    energy_evaluation_iteration_wall_time_seconds: tuple[float, ...]
    density_update_iteration_wall_time_seconds: tuple[float, ...]
    bookkeeping_iteration_wall_time_seconds: tuple[float, ...]


def _empty_orbital_block(grid_geometry: GridGeometryLike) -> np.ndarray:
    return np.zeros((0,) + grid_geometry.spec.shape, dtype=_ZERO_BLOCK_DTYPE)


def _find_spin_state(case: BenchmarkCase, spin_label: str):
    normalized = spin_label.strip().lower()
    for spin_state in case.spin_states:
        if spin_state.label.lower() == normalized:
            return spin_state
    available = ", ".join(state.label for state in case.spin_states)
    raise ValueError(
        f"Spin state `{spin_label}` is not configured for `{case.name}`. "
        f"Available states: {available}."
    )


def _validate_h2_case(case: BenchmarkCase) -> None:
    if case.name != _SUPPORTED_CASE_NAME:
        raise ValueError(
            "The current minimal SCF driver supports only the default H2 benchmark; "
            f"received `{case.name}`."
        )
    if case.charge != 0:
        raise ValueError(
            "The current minimal SCF driver supports only neutral H2; "
            f"received charge={case.charge}."
        )
    if case.geometry.name.upper() != "H2":
        raise ValueError(
            "The current minimal SCF driver supports only H2 geometries; "
            f"received `{case.geometry.name}`."
        )
    if case.geometry.unit.lower() != "bohr":
        raise ValueError(
            "The current minimal SCF driver expects Bohr geometries; "
            f"received `{case.geometry.unit}`."
        )


def resolve_h2_spin_occupations(
    spin_label: str,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> SpinOccupations:
    """Resolve the explicit alpha/beta occupations for H2 singlet/triplet."""

    _validate_h2_case(case)
    spin_state = _find_spin_state(case, spin_label)
    pseudo_by_element = load_case_gth_pseudo_data(case)
    total_electrons = sum(
        pseudo_by_element[atom.element].ionic_charge
        for atom in case.geometry.atoms
    ) - case.charge
    if total_electrons <= 0:
        raise ValueError("The current SCF driver expects a positive H2 valence-electron count.")
    if (total_electrons + spin_state.spin) % 2 != 0:
        raise ValueError(
            "The requested spin state is inconsistent with the total electron count."
        )
    n_alpha = (total_electrons + spin_state.spin) // 2
    n_beta = total_electrons - n_alpha
    if n_alpha < 0 or n_beta < 0:
        raise ValueError("Resolved alpha/beta occupations must be non-negative.")

    occupations_up = tuple(1.0 for _ in range(n_alpha))
    occupations_down = tuple(1.0 for _ in range(n_beta))
    return SpinOccupations(
        label=spin_state.label,
        spin=spin_state.spin,
        total_electrons=total_electrons,
        n_alpha=n_alpha,
        n_beta=n_beta,
        occupations_up=occupations_up,
        occupations_down=occupations_down,
    )


def _expectation_value(
    orbital: np.ndarray,
    action: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> float:
    overlap = weighted_overlap_matrix(
        orbital,
        grid_geometry=grid_geometry,
        other=action,
    )
    return float(np.real_if_close(overlap[0, 0]))


def _sum_weighted_expectations(
    orbitals: np.ndarray,
    actions: np.ndarray,
    occupations: tuple[float, ...],
    grid_geometry: GridGeometryLike,
) -> float:
    orbital_block = validate_orbital_block(orbitals, grid_geometry=grid_geometry, name="orbitals")
    action_block = validate_orbital_block(actions, grid_geometry=grid_geometry, name="actions")
    if orbital_block.shape[0] != len(occupations) or action_block.shape[0] != len(occupations):
        raise ValueError("Orbitals/actions must match the occupation count.")
    return sum(
        occupation * _expectation_value(
            orbital_block[index : index + 1],
            action_block[index : index + 1],
            grid_geometry=grid_geometry,
        )
        for index, occupation in enumerate(occupations)
    )


def _build_density_from_occupied_orbitals(
    orbitals: np.ndarray,
    occupations: tuple[float, ...],
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    if len(occupations) == 0:
        return np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    orbital_block = validate_orbital_block(orbitals, grid_geometry=grid_geometry, name="orbitals")
    if orbital_block.shape[0] < len(occupations):
        raise ValueError("Not enough orbitals were provided for the requested occupations.")
    density = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    for index, occupation in enumerate(occupations):
        density += occupation * np.abs(orbital_block[index]) ** 2
    return density


def _renormalize_density(
    rho: np.ndarray,
    target_electrons: float,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    if target_electrons == 0.0:
        return np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    density = validate_orbital_field(rho, grid_geometry=grid_geometry, name="rho").astype(np.float64)
    integral = float(integrate_field(density, grid_geometry=grid_geometry))
    if integral <= 0.0:
        raise ValueError("A positive density integral is required for renormalization.")
    return density * (target_electrons / integral)


def _density_residual(
    rho_up_in: np.ndarray,
    rho_down_in: np.ndarray,
    rho_up_out: np.ndarray,
    rho_down_out: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> float:
    up_norm = weighted_l2_norm(rho_up_out - rho_up_in, grid_geometry=grid_geometry)
    down_norm = weighted_l2_norm(rho_down_out - rho_down_in, grid_geometry=grid_geometry)
    return float(np.sqrt(up_norm * up_norm + down_norm * down_norm))


def _build_h2_trial_orbitals(
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    atom_fields = []
    for atom in case.geometry.atoms:
        dx = grid_geometry.x_points - atom.position[0]
        dy = grid_geometry.y_points - atom.position[1]
        dz = grid_geometry.z_points - atom.position[2]
        atom_fields.append(np.exp(-0.8 * (dx * dx + dy * dy + dz * dz)))

    bonding = atom_fields[0] + atom_fields[1]
    antibonding = atom_fields[0] - atom_fields[1]
    if weighted_l2_norm(antibonding, grid_geometry=grid_geometry) < 1.0e-12:
        center = np.asarray(grid_geometry.spec.reference_center, dtype=np.float64)
        z_shift = grid_geometry.z_points - center[2]
        antibonding = z_shift * np.exp(
            -0.6
            * (
                (grid_geometry.x_points - center[0]) ** 2
                + (grid_geometry.y_points - center[1]) ** 2
                + z_shift**2
            )
        )
    orbital_block = np.asarray([bonding, antibonding], dtype=np.float64)
    return weighted_orthonormalize_orbitals(
        orbital_block,
        grid_geometry=grid_geometry,
        require_full_rank=True,
    )


def build_h2_initial_density_guess(
    occupations: SpinOccupations,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: GridGeometryLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the current minimal H2 initial density and orbital guesses."""

    _validate_h2_case(case)
    if grid_geometry is None:
        grid_geometry = build_default_h2_grid_geometry(case=case)
    trial_orbitals = _build_h2_trial_orbitals(case=case, grid_geometry=grid_geometry)

    if occupations.n_alpha > trial_orbitals.shape[0] or occupations.n_beta > trial_orbitals.shape[0]:
        raise ValueError("The current H2 initial guess builder provides only two trial orbitals.")

    orbitals_up = trial_orbitals[: occupations.n_alpha] if occupations.n_alpha else _empty_orbital_block(grid_geometry)
    orbitals_down = trial_orbitals[: occupations.n_beta] if occupations.n_beta else _empty_orbital_block(grid_geometry)
    rho_up = _build_density_from_occupied_orbitals(
        orbitals_up,
        occupations.occupations_up,
        grid_geometry=grid_geometry,
    )
    rho_down = _build_density_from_occupied_orbitals(
        orbitals_down,
        occupations.occupations_down,
        grid_geometry=grid_geometry,
    )
    return (
        _renormalize_density(rho_up, occupations.n_alpha, grid_geometry=grid_geometry),
        _renormalize_density(rho_down, occupations.n_beta, grid_geometry=grid_geometry),
        orbitals_up,
        orbitals_down,
    )


def _build_solve_summary(
    result: FixedPotentialEigensolverResult | None,
    spin_channel: str,
    target_orbitals: int,
) -> FixedPotentialSolveSummary | None:
    if result is None:
        return None
    return FixedPotentialSolveSummary(
        spin_channel=spin_channel,
        target_orbitals=target_orbitals,
        solver_method=result.solver_method,
        solver_note=result.solver_note,
        converged=result.converged,
        eigenvalues=np.asarray(result.eigenvalues, dtype=np.float64),
        residual_norms=np.asarray(result.residual_norms, dtype=np.float64),
    )


def _is_h2_closed_shell_singlet(occupations: SpinOccupations) -> bool:
    return (
        occupations.spin == 0
        and occupations.n_alpha == 1
        and occupations.n_beta == 1
    )


def evaluate_ion_ion_repulsion(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> float:
    """Evaluate the valence-ion Coulomb repulsion using the current GTH charges."""

    _validate_h2_case(case)
    pseudo_by_element = load_case_gth_pseudo_data(case)
    repulsion = 0.0
    atoms = case.geometry.atoms
    for first_index, first_atom in enumerate(atoms):
        charge_i = pseudo_by_element[first_atom.element].ionic_charge
        for second_atom in atoms[first_index + 1 :]:
            charge_j = pseudo_by_element[second_atom.element].ionic_charge
            displacement = np.subtract(first_atom.position, second_atom.position)
            distance = float(np.linalg.norm(displacement))
            if distance <= 0.0:
                raise ValueError("Distinct ions must not occupy the same position.")
            repulsion += charge_i * charge_j / distance
    return repulsion


def evaluate_single_point_energy(
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    orbitals_up: np.ndarray,
    orbitals_down: np.ndarray,
    occupations: SpinOccupations,
    grid_geometry: StructuredGridGeometry,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> SinglePointEnergyComponents:
    """Evaluate the first-stage single-point total energy on the current grid."""

    _validate_h2_case(case)
    rho_up_field = _renormalize_density(
        rho_up,
        occupations.n_alpha,
        grid_geometry=grid_geometry,
    )
    rho_down_field = _renormalize_density(
        rho_down,
        occupations.n_beta,
        grid_geometry=grid_geometry,
    )
    rho_total = build_total_density(
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        grid_geometry=grid_geometry,
    )

    kinetic_up = np.asarray(
        [apply_kinetic_operator(orbital, grid_geometry=grid_geometry) for orbital in orbitals_up],
        dtype=np.float64,
    ) if occupations.n_alpha else _empty_orbital_block(grid_geometry)
    kinetic_down = np.asarray(
        [apply_kinetic_operator(orbital, grid_geometry=grid_geometry) for orbital in orbitals_down],
        dtype=np.float64,
    ) if occupations.n_beta else _empty_orbital_block(grid_geometry)

    local_ionic_evaluation = evaluate_local_ionic_potential(
        case=case,
        grid_geometry=grid_geometry,
    )
    hartree_result = solve_hartree_potential(
        grid_geometry=grid_geometry,
        rho=rho_total,
    )
    lsda_evaluation = evaluate_lsda_terms(
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        functional=case.reference_model.xc,
    )

    kinetic = _sum_weighted_expectations(
        orbitals_up,
        kinetic_up,
        occupations.occupations_up,
        grid_geometry=grid_geometry,
    ) + _sum_weighted_expectations(
        orbitals_down,
        kinetic_down,
        occupations.occupations_down,
        grid_geometry=grid_geometry,
    )

    local_ionic = float(
        integrate_field(
            rho_total * local_ionic_evaluation.total_local_potential,
            grid_geometry=grid_geometry,
        )
    )

    nonlocal_ionic = 0.0
    for index, occupation in enumerate(occupations.occupations_up):
        nonlocal_action = evaluate_nonlocal_ionic_action(
            case=case,
            grid_geometry=grid_geometry,
            psi=orbitals_up[index],
        ).total_nonlocal_action
        nonlocal_ionic += occupation * _expectation_value(
            orbitals_up[index : index + 1],
            nonlocal_action[np.newaxis, ...],
            grid_geometry=grid_geometry,
        )
    for index, occupation in enumerate(occupations.occupations_down):
        nonlocal_action = evaluate_nonlocal_ionic_action(
            case=case,
            grid_geometry=grid_geometry,
            psi=orbitals_down[index],
        ).total_nonlocal_action
        nonlocal_ionic += occupation * _expectation_value(
            orbitals_down[index : index + 1],
            nonlocal_action[np.newaxis, ...],
            grid_geometry=grid_geometry,
        )

    hartree = evaluate_hartree_energy(
        rho=rho_total,
        grid_geometry=grid_geometry,
        hartree_potential=hartree_result,
    )
    xc = float(integrate_field(lsda_evaluation.energy_density, grid_geometry=grid_geometry))
    ion_ion_repulsion = evaluate_ion_ion_repulsion(case=case)
    total = kinetic + local_ionic + nonlocal_ionic + hartree + xc + ion_ion_repulsion
    return SinglePointEnergyComponents(
        kinetic=float(kinetic),
        local_ionic=float(local_ionic),
        nonlocal_ionic=float(nonlocal_ionic),
        hartree=float(hartree),
        xc=float(xc),
        ion_ion_repulsion=float(ion_ion_repulsion),
        total=float(total),
    )


def _default_monitor_patch_parameters() -> LocalPotentialPatchParameters:
    return LocalPotentialPatchParameters(
        patch_radius_scale=0.75,
        patch_grid_shape=(25, 25, 25),
        correction_strength=1.30,
        interpolation_neighbors=8,
    )


def _monitor_grid_scf_parameter_summary(
    patch_parameters: LocalPotentialPatchParameters,
    *,
    kinetic_version: str,
    hartree_backend: str,
    use_jax_hartree_cached_operator: bool,
    jax_hartree_cg_impl: str,
    jax_hartree_cg_preconditioner: str,
    jax_hartree_line_preconditioner_impl: str,
    use_jax_block_kernels: bool,
    use_step_local_static_local_reuse: bool,
    cycle_breaker_enabled: bool,
    cycle_breaker_weight: float,
    diis_enabled: bool,
    diis_warmup_iterations: int,
    diis_history_length: int,
) -> H2ScfDryRunParameterSummary:
    return H2ScfDryRunParameterSummary(
        grid_shape=H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE,
        box_half_extents_bohr=H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR,
        weight_scale=H2_MONITOR_LOCAL_PATCH_BASELINE_WEIGHT_SCALE,
        radius_scale=H2_MONITOR_LOCAL_PATCH_BASELINE_RADIUS_SCALE,
        patch_radius_scale=float(patch_parameters.patch_radius_scale),
        patch_grid_shape=patch_parameters.patch_grid_shape,
        correction_strength=float(patch_parameters.correction_strength),
        interpolation_neighbors=int(patch_parameters.interpolation_neighbors),
        kinetic_version=kinetic_version,
        hartree_backend=hartree_backend,
        use_jax_hartree_cached_operator=bool(use_jax_hartree_cached_operator),
        jax_hartree_cg_impl=jax_hartree_cg_impl,
        jax_hartree_cg_preconditioner=jax_hartree_cg_preconditioner,
        jax_hartree_line_preconditioner_impl=jax_hartree_line_preconditioner_impl,
        use_jax_block_kernels=bool(use_jax_block_kernels),
        use_step_local_static_local_reuse=bool(use_step_local_static_local_reuse),
        includes_nonlocal=False,
        cycle_breaker_enabled=bool(cycle_breaker_enabled),
        cycle_breaker_weight=float(cycle_breaker_weight),
        diis_enabled=bool(diis_enabled),
        diis_warmup_iterations=int(diis_warmup_iterations),
        diis_history_length=int(diis_history_length),
        diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
    )


def _detect_monitor_grid_singlet_alternation(
    history: list[ScfIterationRecord],
    *,
    rho_up_out: np.ndarray,
    rho_down_out: np.ndarray,
    energy_change: float | None,
    grid_geometry: MonitorGridGeometry,
) -> bool:
    """Detect a weak even/odd alternation in the monitor-grid singlet dry-run."""

    if len(history) < 2 or energy_change is None:
        return False
    previous_energy_change = history[-1].energy_change
    if previous_energy_change is None or energy_change * previous_energy_change >= 0.0:
        return False
    distance_to_previous = _density_residual(
        rho_up_in=history[-1].output_rho_up,
        rho_down_in=history[-1].output_rho_down,
        rho_up_out=rho_up_out,
        rho_down_out=rho_down_out,
        grid_geometry=grid_geometry,
    )
    distance_to_two_steps_ago = _density_residual(
        rho_up_in=history[-2].output_rho_up,
        rho_down_in=history[-2].output_rho_down,
        rho_up_out=rho_up_out,
        rho_down_out=rho_down_out,
        grid_geometry=grid_geometry,
    )
    return bool(distance_to_two_steps_ago < 0.95 * distance_to_previous)


def _record_last_jax_hartree_solve_diagnostics(
    *,
    enabled: bool,
    solve_times: list[float],
    cg_iterations: list[int],
    cached_use_flags: list[bool],
    cached_first_flags: list[bool],
    boundary_condition_times: list[float] | None = None,
    build_times: list[float] | None = None,
    rhs_assembly_times: list[float] | None = None,
    cg_times: list[float] | None = None,
    matvec_call_counts: list[int] | None = None,
    matvec_times: list[float] | None = None,
    preconditioner_apply_counts: list[int] | None = None,
    preconditioner_apply_times: list[float] | None = None,
    preconditioner_setup_times: list[float] | None = None,
    preconditioner_axis_reorder_times: list[float] | None = None,
    preconditioner_tridiagonal_solve_times: list[float] | None = None,
    preconditioner_other_overhead_times: list[float] | None = None,
) -> None:
    if not enabled:
        return
    diagnostics = get_last_monitor_poisson_jax_solve_diagnostics()
    if diagnostics is None:
        return
    solve_times.append(float(diagnostics.total_wall_time_seconds))
    cg_iterations.append(int(diagnostics.iteration_count))
    cached_use_flags.append(bool(diagnostics.used_cached_operator))
    cached_first_flags.append(bool(diagnostics.first_solve_for_cached_operator))
    if boundary_condition_times is not None:
        boundary_condition_times.append(float(diagnostics.boundary_condition_wall_time_seconds))
    if build_times is not None:
        build_times.append(float(diagnostics.build_wall_time_seconds))
    if rhs_assembly_times is not None:
        rhs_assembly_times.append(float(diagnostics.rhs_assembly_wall_time_seconds))
    if cg_times is not None:
        cg_times.append(float(diagnostics.cg_wall_time_seconds))
    if matvec_call_counts is not None:
        matvec_call_counts.append(int(diagnostics.matvec_call_count))
    if matvec_times is not None:
        matvec_times.append(float(diagnostics.matvec_wall_time_seconds))
    if preconditioner_apply_counts is not None:
        preconditioner_apply_counts.append(int(diagnostics.preconditioner_apply_count))
    if preconditioner_apply_times is not None:
        preconditioner_apply_times.append(float(diagnostics.preconditioner_apply_wall_time_seconds))
    if preconditioner_setup_times is not None:
        preconditioner_setup_times.append(float(diagnostics.preconditioner_setup_wall_time_seconds))
    if preconditioner_axis_reorder_times is not None:
        preconditioner_axis_reorder_times.append(
            float(diagnostics.preconditioner_axis_reorder_wall_time_seconds)
        )
    if preconditioner_tridiagonal_solve_times is not None:
        preconditioner_tridiagonal_solve_times.append(
            float(diagnostics.preconditioner_tridiagonal_solve_wall_time_seconds)
        )
    if preconditioner_other_overhead_times is not None:
        preconditioner_other_overhead_times.append(
            float(diagnostics.preconditioner_other_overhead_wall_time_seconds)
        )


def _density_residual_fields(
    rho_up_in: np.ndarray,
    rho_down_in: np.ndarray,
    rho_up_out: np.ndarray,
    rho_down_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(rho_up_out - rho_up_in, dtype=np.float64),
        np.asarray(rho_down_out - rho_down_in, dtype=np.float64),
    )


def _weighted_spin_density_dot(
    left_up: np.ndarray,
    left_down: np.ndarray,
    right_up: np.ndarray,
    right_down: np.ndarray,
    *,
    grid_geometry: GridGeometryLike,
) -> float:
    return float(
        integrate_field(
            left_up * right_up + left_down * right_down,
            grid_geometry=grid_geometry,
        )
    )


def _apply_monitor_grid_density_diis(
    history: tuple[MonitorGridDiisHistoryEntry, ...],
    *,
    grid_geometry: MonitorGridGeometry,
    n_alpha: int,
    n_beta: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    history_length = len(history)
    if history_length < 2:
        return None

    b_matrix = np.empty((history_length + 1, history_length + 1), dtype=np.float64)
    b_matrix.fill(-1.0)
    b_matrix[-1, -1] = 0.0
    for row_index, row in enumerate(history):
        for column_index, column in enumerate(history):
            b_matrix[row_index, column_index] = _weighted_spin_density_dot(
                row.residual_up,
                row.residual_down,
                column.residual_up,
                column.residual_down,
                grid_geometry=grid_geometry,
            )

    regularization_scale = max(float(np.max(np.abs(b_matrix[:-1, :-1]))), 1.0)
    b_matrix[:-1, :-1] += 1.0e-12 * regularization_scale * np.eye(history_length, dtype=np.float64)
    rhs = np.zeros(history_length + 1, dtype=np.float64)
    rhs[-1] = -1.0
    try:
        coefficients = np.linalg.solve(b_matrix, rhs)[:-1]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coefficients)):
        return None
    if float(np.max(np.abs(coefficients))) > 20.0:
        return None

    rho_up_diis = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    rho_down_diis = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    for coefficient, entry in zip(coefficients, history, strict=True):
        rho_up_diis += float(coefficient) * entry.mixed_rho_up
        rho_down_diis += float(coefficient) * entry.mixed_rho_down

    if n_alpha > 0:
        rho_up_diis = np.maximum(rho_up_diis, 0.0)
        if float(integrate_field(rho_up_diis, grid_geometry=grid_geometry)) <= 0.0:
            return None
    else:
        rho_up_diis = np.zeros(grid_geometry.spec.shape, dtype=np.float64)

    if n_beta > 0:
        rho_down_diis = np.maximum(rho_down_diis, 0.0)
        if float(integrate_field(rho_down_diis, grid_geometry=grid_geometry)) <= 0.0:
            return None
    else:
        rho_down_diis = np.zeros(grid_geometry.spec.shape, dtype=np.float64)

    return (
        _renormalize_density(rho_up_diis, n_alpha, grid_geometry=grid_geometry),
        _renormalize_density(rho_down_diis, n_beta, grid_geometry=grid_geometry),
    )


def _apply_kinetic_for_static_local_energy(
    orbital: np.ndarray,
    grid_geometry: GridGeometryLike,
    *,
    kinetic_version: str,
) -> np.ndarray:
    if kinetic_version == "trial_fix" and isinstance(grid_geometry, MonitorGridGeometry):
        return apply_monitor_grid_kinetic_operator_trial_boundary_fix(
            orbital,
            grid_geometry=grid_geometry,
        )
    return apply_kinetic_operator(orbital, grid_geometry=grid_geometry)


def evaluate_static_local_single_point_energy(
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    orbitals_up: np.ndarray,
    orbitals_down: np.ndarray,
    occupations: SpinOccupations,
    grid_geometry: GridGeometryLike,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    *,
    use_monitor_patch: bool = False,
    patch_parameters: LocalPotentialPatchParameters | None = None,
    kinetic_version: str = "production",
    hartree_backend: str = "python",
    use_jax_hartree_cached_operator: bool = False,
    jax_hartree_cg_impl: str = "baseline",
    jax_hartree_cg_preconditioner: str = "none",
    jax_hartree_line_preconditioner_impl: str = "baseline",
) -> SinglePointEnergyComponents:
    """Evaluate the local-only single-point energy on either supported grid.

    This helper intentionally contains only

        T_s + E_loc,ion + E_H + E_xc + E_II

    and sets the nonlocal contribution to zero. It is meant for the current
    A-grid SCF dry-run and related local-only audits; it is not a replacement
    for the full legacy total-energy path.
    """

    _validate_h2_case(case)
    rho_up_field = _renormalize_density(
        rho_up,
        occupations.n_alpha,
        grid_geometry=grid_geometry,
    )
    rho_down_field = _renormalize_density(
        rho_down,
        occupations.n_beta,
        grid_geometry=grid_geometry,
    )
    rho_total = build_total_density(
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        grid_geometry=grid_geometry,
    )

    kinetic_up = (
        np.asarray(
            [
                _apply_kinetic_for_static_local_energy(
                    orbital,
                    grid_geometry=grid_geometry,
                    kinetic_version=kinetic_version,
                )
                for orbital in orbitals_up
            ],
            dtype=np.float64,
        )
        if occupations.n_alpha
        else _empty_orbital_block(grid_geometry)
    )
    kinetic_down = (
        np.asarray(
            [
                _apply_kinetic_for_static_local_energy(
                    orbital,
                    grid_geometry=grid_geometry,
                    kinetic_version=kinetic_version,
                )
                for orbital in orbitals_down
            ],
            dtype=np.float64,
        )
        if occupations.n_beta
        else _empty_orbital_block(grid_geometry)
    )

    kinetic = _sum_weighted_expectations(
        orbitals_up,
        kinetic_up,
        occupations.occupations_up,
        grid_geometry=grid_geometry,
    ) + _sum_weighted_expectations(
        orbitals_down,
        kinetic_down,
        occupations.occupations_down,
        grid_geometry=grid_geometry,
    )

    if isinstance(grid_geometry, MonitorGridGeometry) and use_monitor_patch:
        if patch_parameters is None:
            patch_parameters = _default_monitor_patch_parameters()
        patch_evaluation = evaluate_monitor_grid_local_ionic_potential_with_patch(
            case=case,
            grid_geometry=grid_geometry,
            density_field=rho_total,
            patch_parameters=patch_parameters,
        )
        local_ionic = float(patch_evaluation.corrected_local_energy)
    else:
        local_ionic_evaluation = evaluate_local_ionic_potential(
            case=case,
            grid_geometry=grid_geometry,
        )
        local_ionic = float(
            integrate_field(
                rho_total * local_ionic_evaluation.total_local_potential,
                grid_geometry=grid_geometry,
            )
        )

    hartree_result = solve_hartree_potential(
        grid_geometry=grid_geometry,
        rho=rho_total,
        backend=hartree_backend,
        use_jax_cached_operator=use_jax_hartree_cached_operator,
        cg_impl=jax_hartree_cg_impl,
        cg_preconditioner=jax_hartree_cg_preconditioner,
        line_preconditioner_impl=jax_hartree_line_preconditioner_impl,
    )
    lsda_evaluation = evaluate_lsda_terms(
        rho_up=rho_up_field,
        rho_down=rho_down_field,
        functional=case.reference_model.xc,
    )

    hartree = evaluate_hartree_energy(
        rho=rho_total,
        grid_geometry=grid_geometry,
        hartree_potential=hartree_result,
    )
    xc = float(integrate_field(lsda_evaluation.energy_density, grid_geometry=grid_geometry))
    ion_ion_repulsion = evaluate_ion_ion_repulsion(case=case)
    total = kinetic + local_ionic + hartree + xc + ion_ion_repulsion
    return SinglePointEnergyComponents(
        kinetic=float(kinetic),
        local_ionic=float(local_ionic),
        nonlocal_ionic=0.0,
        hartree=float(hartree),
        xc=float(xc),
        ion_ion_repulsion=float(ion_ion_repulsion),
        total=float(total),
    )


def evaluate_static_local_single_point_energy_from_context(
    operator_context: FixedPotentialStaticLocalOperatorContext,
    orbitals_up: np.ndarray,
    orbitals_down: np.ndarray,
    occupations: SpinOccupations,
) -> tuple[SinglePointEnergyComponents, StaticLocalEnergyEvaluationProfile]:
    """Evaluate the local-only single-point energy from one frozen static-local context.

    This reuses the step-local `rho_total`, local ionic slice, Hartree potential,
    and LSDA evaluation that were already assembled for the fixed-potential solve.
    """

    grid_geometry = operator_context.grid_geometry
    kinetic_start = perf_counter()
    kinetic_up = (
        np.asarray(
            [
                _apply_kinetic_for_static_local_energy(
                    orbital,
                    grid_geometry=grid_geometry,
                    kinetic_version=operator_context.kinetic_version,
                )
                for orbital in orbitals_up
            ],
            dtype=np.float64,
        )
        if occupations.n_alpha
        else _empty_orbital_block(grid_geometry)
    )
    kinetic_down = (
        np.asarray(
            [
                _apply_kinetic_for_static_local_energy(
                    orbital,
                    grid_geometry=grid_geometry,
                    kinetic_version=operator_context.kinetic_version,
                )
                for orbital in orbitals_down
            ],
            dtype=np.float64,
        )
        if occupations.n_beta
        else _empty_orbital_block(grid_geometry)
    )
    kinetic = _sum_weighted_expectations(
        orbitals_up,
        kinetic_up,
        occupations.occupations_up,
        grid_geometry=grid_geometry,
    ) + _sum_weighted_expectations(
        orbitals_down,
        kinetic_down,
        occupations.occupations_down,
        grid_geometry=grid_geometry,
    )
    kinetic_elapsed = perf_counter() - kinetic_start

    local_start = perf_counter()
    if operator_context.frozen_patch_local_embedding is not None:
        local_ionic = float(
            operator_context.frozen_patch_local_embedding.patch_evaluation.corrected_local_energy
        )
    elif operator_context.local_ionic_evaluation is not None:
        local_ionic = float(
            integrate_field(
                operator_context.rho_total
                * operator_context.local_ionic_evaluation.total_local_potential,
                grid_geometry=grid_geometry,
            )
        )
    else:
        local_ionic = float(
            integrate_field(
                operator_context.rho_total * operator_context.local_ionic_potential,
                grid_geometry=grid_geometry,
            )
        )
    local_elapsed = perf_counter() - local_start

    hartree_start = perf_counter()
    hartree = evaluate_hartree_energy(
        rho=operator_context.rho_total,
        grid_geometry=grid_geometry,
        hartree_potential=(
            operator_context.hartree_poisson_result
            if operator_context.hartree_poisson_result is not None
            else operator_context.hartree_potential
        ),
    )
    hartree_elapsed = perf_counter() - hartree_start

    xc_start = perf_counter()
    lsda_evaluation = operator_context.lsda_evaluation
    if lsda_evaluation is None:
        lsda_evaluation = evaluate_lsda_terms(
            rho_up=operator_context.rho_up,
            rho_down=operator_context.rho_down,
            functional=operator_context.case.reference_model.xc,
        )
    xc = float(integrate_field(lsda_evaluation.energy_density, grid_geometry=grid_geometry))
    xc_elapsed = perf_counter() - xc_start

    ion_ion_start = perf_counter()
    ion_ion_repulsion = evaluate_ion_ion_repulsion(case=operator_context.case)
    ion_ion_elapsed = perf_counter() - ion_ion_start

    total = kinetic + local_ionic + hartree + xc + ion_ion_repulsion
    return (
        SinglePointEnergyComponents(
            kinetic=float(kinetic),
            local_ionic=float(local_ionic),
            nonlocal_ionic=0.0,
            hartree=float(hartree),
            xc=float(xc),
            ion_ion_repulsion=float(ion_ion_repulsion),
            total=float(total),
        ),
        StaticLocalEnergyEvaluationProfile(
            kinetic_wall_time_seconds=float(kinetic_elapsed),
            local_ionic_wall_time_seconds=float(local_elapsed),
            hartree_energy_wall_time_seconds=float(hartree_elapsed),
            xc_energy_wall_time_seconds=float(xc_elapsed),
            ion_ion_wall_time_seconds=float(ion_ion_elapsed),
        ),
    )


def _check_density_electron_count(
    density: np.ndarray,
    target_electrons: float,
    grid_geometry: GridGeometryLike,
    name: str,
) -> None:
    actual = float(integrate_field(density, grid_geometry=grid_geometry))
    if abs(actual - target_electrons) > 1.0e-6:
        raise ValueError(
            f"{name} integrates to {actual}, expected {target_electrons} electrons."
        )


def run_h2_minimal_scf(
    spin_label: str,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: StructuredGridGeometry | None = None,
    max_iterations: int = 8,
    mixing: float = 0.6,
    density_tolerance: float = 2.5e-3,
    energy_tolerance: float = 1.0e-5,
    eigensolver_tolerance: float = 5.0e-3,
    eigensolver_ncv: int = 20,
) -> H2ScfResult:
    """Run the first minimal H2 single-point SCF loop for one spin state."""

    _validate_h2_case(case)
    if grid_geometry is None:
        grid_geometry = build_default_h2_grid_geometry(case=case)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if not (0.0 < mixing <= 1.0):
        raise ValueError("mixing must satisfy 0 < mixing <= 1.")
    if density_tolerance <= 0.0 or energy_tolerance <= 0.0 or eigensolver_tolerance <= 0.0:
        raise ValueError("SCF tolerances must be positive.")

    occupations = resolve_h2_spin_occupations(spin_label=spin_label, case=case)
    rho_up, rho_down, guess_up, guess_down = build_h2_initial_density_guess(
        occupations=occupations,
        case=case,
        grid_geometry=grid_geometry,
    )

    history: list[ScfIterationRecord] = []
    previous_energy_total: float | None = None
    final_orbitals_up = guess_up
    final_orbitals_down = guess_down
    final_eigenvalues_up = np.zeros(occupations.n_alpha, dtype=np.float64)
    final_eigenvalues_down = np.zeros(occupations.n_beta, dtype=np.float64)
    final_solve_up: FixedPotentialEigensolverResult | None = None
    final_solve_down: FixedPotentialEigensolverResult | None = None
    final_energy = evaluate_single_point_energy(
        rho_up=rho_up,
        rho_down=rho_down,
        orbitals_up=guess_up,
        orbitals_down=guess_down,
        occupations=occupations,
        grid_geometry=grid_geometry,
        case=case,
    )
    converged = False

    for iteration in range(1, max_iterations + 1):
        solve_up = None
        solve_down = None

        if occupations.n_alpha > 0:
            solve_up = solve_fixed_potential_eigenproblem(
                grid_geometry=grid_geometry,
                rho_up=rho_up,
                rho_down=rho_down,
                spin_channel="up",
                k=occupations.n_alpha,
                case=case,
                initial_guess_orbitals=guess_up,
                tolerance=eigensolver_tolerance,
                ncv=eigensolver_ncv,
            )
            orbitals_up = solve_up.orbitals
        else:
            orbitals_up = _empty_orbital_block(grid_geometry)

        if occupations.n_beta > 0 and _is_h2_closed_shell_singlet(occupations):
            orbitals_down = np.asarray(orbitals_up, dtype=np.float64)
        elif occupations.n_beta > 0:
            solve_down = solve_fixed_potential_eigenproblem(
                grid_geometry=grid_geometry,
                rho_up=rho_up,
                rho_down=rho_down,
                spin_channel="down",
                k=occupations.n_beta,
                case=case,
                initial_guess_orbitals=guess_down,
                tolerance=eigensolver_tolerance,
                ncv=eigensolver_ncv,
            )
            orbitals_down = solve_down.orbitals
        else:
            orbitals_down = _empty_orbital_block(grid_geometry)

        rho_up_out = _renormalize_density(
            _build_density_from_occupied_orbitals(
                orbitals_up,
                occupations.occupations_up,
                grid_geometry=grid_geometry,
            ),
            occupations.n_alpha,
            grid_geometry=grid_geometry,
        )
        rho_down_out = _renormalize_density(
            _build_density_from_occupied_orbitals(
                orbitals_down,
                occupations.occupations_down,
                grid_geometry=grid_geometry,
            ),
            occupations.n_beta,
            grid_geometry=grid_geometry,
        )

        _check_density_electron_count(rho_up_out, occupations.n_alpha, grid_geometry, "rho_up_out")
        _check_density_electron_count(rho_down_out, occupations.n_beta, grid_geometry, "rho_down_out")

        density_residual = _density_residual(
            rho_up_in=rho_up,
            rho_down_in=rho_down,
            rho_up_out=rho_up_out,
            rho_down_out=rho_down_out,
            grid_geometry=grid_geometry,
        )
        energy = evaluate_single_point_energy(
            rho_up=rho_up_out,
            rho_down=rho_down_out,
            orbitals_up=orbitals_up,
            orbitals_down=orbitals_down,
            occupations=occupations,
            grid_geometry=grid_geometry,
            case=case,
        )
        energy_change = None if previous_energy_total is None else energy.total - previous_energy_total

        rho_up_mixed = _renormalize_density(
            (1.0 - mixing) * rho_up + mixing * rho_up_out,
            occupations.n_alpha,
            grid_geometry=grid_geometry,
        )
        rho_down_mixed = _renormalize_density(
            (1.0 - mixing) * rho_down + mixing * rho_down_out,
            occupations.n_beta,
            grid_geometry=grid_geometry,
        )

        history.append(
            ScfIterationRecord(
                iteration=iteration,
                input_rho_up=np.asarray(rho_up, dtype=np.float64),
                input_rho_down=np.asarray(rho_down, dtype=np.float64),
                output_rho_up=np.asarray(rho_up_out, dtype=np.float64),
                output_rho_down=np.asarray(rho_down_out, dtype=np.float64),
                mixed_rho_up=np.asarray(rho_up_mixed, dtype=np.float64),
                mixed_rho_down=np.asarray(rho_down_mixed, dtype=np.float64),
                density_residual=float(density_residual),
                energy=energy,
                energy_change=None if energy_change is None else float(energy_change),
                solve_up=_build_solve_summary(solve_up, "up", occupations.n_alpha),
                solve_down=(
                    _build_solve_summary(solve_up, "down", occupations.n_beta)
                    if occupations.n_beta > 0 and _is_h2_closed_shell_singlet(occupations)
                    else _build_solve_summary(solve_down, "down", occupations.n_beta)
                ),
            )
        )

        final_orbitals_up = orbitals_up
        final_orbitals_down = orbitals_down
        final_eigenvalues_up = (
            np.asarray(solve_up.eigenvalues, dtype=np.float64)
            if solve_up is not None
            else np.zeros(0, dtype=np.float64)
        )
        final_eigenvalues_down = (
            np.asarray(final_eigenvalues_up, dtype=np.float64)
            if occupations.n_beta > 0 and _is_h2_closed_shell_singlet(occupations)
            else np.asarray(solve_down.eigenvalues, dtype=np.float64)
            if solve_down is not None
            else np.zeros(0, dtype=np.float64)
        )
        final_solve_up = solve_up
        final_solve_down = solve_up if occupations.n_beta > 0 and _is_h2_closed_shell_singlet(occupations) else solve_down
        final_energy = energy

        if density_residual < density_tolerance and (
            energy_change is None or abs(energy_change) < energy_tolerance
        ):
            rho_up = rho_up_out
            rho_down = rho_down_out
            converged = True
            break

        rho_up = rho_up_mixed
        rho_down = rho_down_mixed
        guess_up = orbitals_up
        guess_down = orbitals_down
        previous_energy_total = energy.total

    return H2ScfResult(
        converged=converged,
        spin_state_label=occupations.label,
        spin=occupations.spin,
        occupations=occupations,
        iteration_count=len(history),
        history=tuple(history),
        eigenvalues_up=final_eigenvalues_up,
        eigenvalues_down=final_eigenvalues_down,
        orbitals_up=np.asarray(final_orbitals_up, dtype=np.float64),
        orbitals_down=np.asarray(final_orbitals_down, dtype=np.float64),
        rho_up=np.asarray(rho_up, dtype=np.float64),
        rho_down=np.asarray(rho_down, dtype=np.float64),
        energy=final_energy,
        solve_up=final_solve_up,
        solve_down=final_solve_down,
    )


def run_h2_monitor_grid_scf_dry_run(
    spin_label: str,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    *,
    patch_parameters: LocalPotentialPatchParameters | None = None,
    max_iterations: int = 10,
    mixing: float = 0.35,
    density_tolerance: float = 5.0e-3,
    energy_tolerance: float = 5.0e-5,
    eigensolver_tolerance: float = 1.0e-3,
    eigensolver_ncv: int = 20,
    kinetic_version: str = "trial_fix",
    hartree_backend: str = "python",
    use_jax_hartree_cached_operator: bool = False,
    jax_hartree_cg_impl: str = "baseline",
    jax_hartree_cg_preconditioner: str = "none",
    jax_hartree_line_preconditioner_impl: str = "baseline",
    use_jax_block_kernels: bool = False,
    use_step_local_static_local_reuse: bool = False,
    enable_cycle_breaker: bool = False,
    cycle_breaker_weight: float = 0.5,
    enable_diis: bool = False,
    diis_warmup_iterations: int = 3,
    diis_history_length: int = 4,
) -> H2StaticLocalScfDryRunResult:
    """Run the first monitor-grid H2 SCF dry-run on the local static chain."""

    _validate_h2_case(case)
    if grid_geometry is None:
        grid_geometry = build_h2_local_patch_development_monitor_grid()
    if patch_parameters is None:
        patch_parameters = _default_monitor_patch_parameters()
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if not (0.0 < mixing <= 1.0):
        raise ValueError("mixing must satisfy 0 < mixing <= 1.")
    if not (0.0 <= cycle_breaker_weight <= 1.0):
        raise ValueError("cycle_breaker_weight must satisfy 0 <= cycle_breaker_weight <= 1.")
    if enable_cycle_breaker and enable_diis:
        raise ValueError("The minimal monitor-grid dry-run supports cycle-breaker or DIIS, not both.")
    if diis_warmup_iterations < 0:
        raise ValueError("diis_warmup_iterations must be non-negative.")
    if diis_history_length < 2:
        raise ValueError("diis_history_length must be at least 2.")
    if density_tolerance <= 0.0 or energy_tolerance <= 0.0 or eigensolver_tolerance <= 0.0:
        raise ValueError("SCF tolerances must be positive.")
    normalized_hartree_backend = hartree_backend.strip().lower()
    if normalized_hartree_backend not in {"python", "jax"}:
        raise ValueError(
            "hartree_backend must be `python` or `jax`; "
            f"received `{hartree_backend}`."
        )
    if use_jax_hartree_cached_operator and normalized_hartree_backend != "jax":
        raise ValueError(
            "use_jax_hartree_cached_operator requires hartree_backend='jax'."
        )
    normalized_jax_hartree_cg_impl = jax_hartree_cg_impl.strip().lower()
    if normalized_jax_hartree_cg_impl not in {"baseline", "jax_loop"}:
        raise ValueError(
            "jax_hartree_cg_impl must be `baseline` or `jax_loop`; "
            f"received `{jax_hartree_cg_impl}`."
        )
    if normalized_hartree_backend != "jax" and normalized_jax_hartree_cg_impl != "baseline":
        raise ValueError(
            "jax_hartree_cg_impl requires hartree_backend='jax'."
        )
    normalized_jax_hartree_cg_preconditioner = jax_hartree_cg_preconditioner.strip().lower()
    if normalized_jax_hartree_cg_preconditioner not in {"none", "diag", "jacobi", "separable", "line"}:
        raise ValueError(
            "jax_hartree_cg_preconditioner must be `none`, `diag`, `jacobi`, `separable`, or `line`; "
            f"received `{jax_hartree_cg_preconditioner}`."
        )
    if normalized_jax_hartree_cg_preconditioner == "jacobi":
        normalized_jax_hartree_cg_preconditioner = "diag"
    normalized_jax_hartree_line_preconditioner_impl = (
        jax_hartree_line_preconditioner_impl.strip().lower()
    )
    if normalized_jax_hartree_line_preconditioner_impl not in {"baseline", "optimized"}:
        raise ValueError(
            "jax_hartree_line_preconditioner_impl must be `baseline` or `optimized`; "
            f"received `{jax_hartree_line_preconditioner_impl}`."
        )
    if (
        normalized_hartree_backend != "jax"
        and normalized_jax_hartree_cg_preconditioner != "none"
    ):
        raise ValueError(
            "jax_hartree_cg_preconditioner requires hartree_backend='jax'."
        )
    if (
        normalized_jax_hartree_cg_preconditioner != "none"
        and normalized_jax_hartree_cg_impl != "jax_loop"
    ):
        raise ValueError(
            "jax_hartree_cg_preconditioner currently requires jax_hartree_cg_impl='jax_loop'."
        )

    occupations = resolve_h2_spin_occupations(spin_label=spin_label, case=case)
    rho_up, rho_down, guess_up, guess_down = build_h2_initial_density_guess(
        occupations=occupations,
        case=case,
        grid_geometry=grid_geometry,
    )

    history: list[ScfIterationRecord] = []
    previous_energy_total: float | None = None
    final_orbitals_up = guess_up
    final_orbitals_down = guess_down
    final_eigenvalues_up = np.zeros(occupations.n_alpha, dtype=np.float64)
    final_eigenvalues_down = np.zeros(occupations.n_beta, dtype=np.float64)
    final_solve_up: FixedPotentialEigensolverResult | None = None
    final_solve_down: FixedPotentialEigensolverResult | None = None
    cycle_breaker_triggered_iterations: list[int] = []
    diis_history: list[MonitorGridDiisHistoryEntry] = []
    diis_used_iterations: list[int] = []
    diis_history_sizes: list[int] = []
    diis_fallback_iterations: list[int] = []
    iteration_eigensolver_times: list[float] = []
    iteration_energy_evaluation_times: list[float] = []
    iteration_density_update_times: list[float] = []
    iteration_bookkeeping_times: list[float] = []
    hartree_solve_times: list[float] = []
    hartree_cg_iterations: list[int] = []
    hartree_cached_use_flags: list[bool] = []
    hartree_cached_first_flags: list[bool] = []
    hartree_boundary_condition_times: list[float] = []
    hartree_build_times: list[float] = []
    hartree_rhs_assembly_times: list[float] = []
    hartree_cg_times: list[float] = []
    hartree_matvec_call_counts: list[int] = []
    hartree_matvec_times: list[float] = []
    hartree_preconditioner_apply_counts: list[int] = []
    hartree_preconditioner_apply_times: list[float] = []
    hartree_preconditioner_setup_times: list[float] = []
    hartree_preconditioner_axis_reorder_times: list[float] = []
    hartree_preconditioner_tridiagonal_solve_times: list[float] = []
    hartree_preconditioner_other_overhead_times: list[float] = []
    eigensolver_wall_time = 0.0
    static_local_prepare_wall_time = 0.0
    hartree_solve_wall_time = 0.0
    local_ionic_resolve_wall_time = 0.0
    xc_resolve_wall_time = 0.0
    energy_evaluation_wall_time = 0.0
    kinetic_energy_wall_time = 0.0
    local_ionic_energy_wall_time = 0.0
    hartree_energy_wall_time = 0.0
    xc_energy_wall_time = 0.0
    ion_ion_energy_wall_time = 0.0
    density_update_wall_time = 0.0
    hartree_solve_call_count = 0
    cached_base_local_ionic_evaluation = (
        evaluate_local_ionic_potential(case=case, grid_geometry=grid_geometry)
        if use_step_local_static_local_reuse
        else None
    )
    if normalized_hartree_backend == "jax":
        clear_monitor_poisson_jax_kernel_cache()
    total_wall_start = perf_counter()
    if use_step_local_static_local_reuse:
        initial_context, initial_profile = prepare_fixed_potential_static_local_operator_profiled(
            grid_geometry=grid_geometry,
            rho_up=rho_up,
            rho_down=rho_down,
            spin_channel="up",
            case=case,
            use_monitor_patch=True,
            patch_parameters=patch_parameters,
            kinetic_version=kinetic_version,
            base_local_ionic_evaluation=cached_base_local_ionic_evaluation,
            hartree_backend=normalized_hartree_backend,
            use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
            jax_hartree_cg_impl=normalized_jax_hartree_cg_impl,
            jax_hartree_cg_preconditioner=normalized_jax_hartree_cg_preconditioner,
            jax_hartree_line_preconditioner_impl=normalized_jax_hartree_line_preconditioner_impl,
        )
        static_local_prepare_wall_time += initial_profile.total_wall_time_seconds
        hartree_solve_wall_time += initial_profile.hartree_resolve_wall_time_seconds
        local_ionic_resolve_wall_time += initial_profile.local_ionic_resolve_wall_time_seconds
        xc_resolve_wall_time += initial_profile.xc_resolve_wall_time_seconds
        hartree_solve_call_count += 1
        _record_last_jax_hartree_solve_diagnostics(
            enabled=(normalized_hartree_backend == "jax"),
            solve_times=hartree_solve_times,
            cg_iterations=hartree_cg_iterations,
            cached_use_flags=hartree_cached_use_flags,
            cached_first_flags=hartree_cached_first_flags,
            boundary_condition_times=hartree_boundary_condition_times,
            build_times=hartree_build_times,
            rhs_assembly_times=hartree_rhs_assembly_times,
            cg_times=hartree_cg_times,
            matvec_call_counts=hartree_matvec_call_counts,
            matvec_times=hartree_matvec_times,
            preconditioner_apply_counts=hartree_preconditioner_apply_counts,
            preconditioner_apply_times=hartree_preconditioner_apply_times,
            preconditioner_setup_times=hartree_preconditioner_setup_times,
            preconditioner_axis_reorder_times=hartree_preconditioner_axis_reorder_times,
            preconditioner_tridiagonal_solve_times=hartree_preconditioner_tridiagonal_solve_times,
            preconditioner_other_overhead_times=hartree_preconditioner_other_overhead_times,
        )
        initial_energy_start = perf_counter()
        final_energy, initial_energy_profile = evaluate_static_local_single_point_energy_from_context(
            initial_context,
            orbitals_up=guess_up,
            orbitals_down=guess_down,
            occupations=occupations,
        )
        energy_evaluation_wall_time += perf_counter() - initial_energy_start
        kinetic_energy_wall_time += initial_energy_profile.kinetic_wall_time_seconds
        local_ionic_energy_wall_time += initial_energy_profile.local_ionic_wall_time_seconds
        hartree_energy_wall_time += initial_energy_profile.hartree_energy_wall_time_seconds
        xc_energy_wall_time += initial_energy_profile.xc_energy_wall_time_seconds
        ion_ion_energy_wall_time += initial_energy_profile.ion_ion_wall_time_seconds
    else:
        initial_energy_start = perf_counter()
        final_energy = evaluate_static_local_single_point_energy(
            rho_up=rho_up,
            rho_down=rho_down,
            orbitals_up=guess_up,
            orbitals_down=guess_down,
            occupations=occupations,
            grid_geometry=grid_geometry,
            case=case,
            use_monitor_patch=True,
            patch_parameters=patch_parameters,
            kinetic_version=kinetic_version,
            hartree_backend=normalized_hartree_backend,
            use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
            jax_hartree_cg_impl=normalized_jax_hartree_cg_impl,
            jax_hartree_cg_preconditioner=normalized_jax_hartree_cg_preconditioner,
        )
        energy_evaluation_wall_time += perf_counter() - initial_energy_start
        hartree_solve_call_count += 1
        _record_last_jax_hartree_solve_diagnostics(
            enabled=(normalized_hartree_backend == "jax"),
            solve_times=hartree_solve_times,
            cg_iterations=hartree_cg_iterations,
            cached_use_flags=hartree_cached_use_flags,
            cached_first_flags=hartree_cached_first_flags,
            boundary_condition_times=hartree_boundary_condition_times,
            build_times=hartree_build_times,
            rhs_assembly_times=hartree_rhs_assembly_times,
            cg_times=hartree_cg_times,
            matvec_call_counts=hartree_matvec_call_counts,
            matvec_times=hartree_matvec_times,
            preconditioner_apply_counts=hartree_preconditioner_apply_counts,
            preconditioner_apply_times=hartree_preconditioner_apply_times,
            preconditioner_setup_times=hartree_preconditioner_setup_times,
            preconditioner_axis_reorder_times=hartree_preconditioner_axis_reorder_times,
            preconditioner_tridiagonal_solve_times=hartree_preconditioner_tridiagonal_solve_times,
            preconditioner_other_overhead_times=hartree_preconditioner_other_overhead_times,
        )
    converged = False

    for iteration in range(1, max_iterations + 1):
        iteration_start = perf_counter()
        solve_up = None
        solve_down = None
        up_operator_context: FixedPotentialStaticLocalOperatorContext | None = None
        down_operator_context: FixedPotentialStaticLocalOperatorContext | None = None
        up_preparation_profile: FixedPotentialStaticLocalPreparationProfile | None = None
        down_preparation_profile: FixedPotentialStaticLocalPreparationProfile | None = None
        iteration_static_local_prepare_elapsed = 0.0

        eigensolver_start = perf_counter()
        if occupations.n_alpha > 0:
            if use_step_local_static_local_reuse:
                up_operator_context, up_preparation_profile = (
                    prepare_fixed_potential_static_local_operator_profiled(
                        grid_geometry=grid_geometry,
                        rho_up=rho_up,
                        rho_down=rho_down,
                        spin_channel="up",
                        case=case,
                        use_monitor_patch=True,
                        patch_parameters=patch_parameters,
                        kinetic_version=kinetic_version,
                        base_local_ionic_evaluation=cached_base_local_ionic_evaluation,
                        hartree_backend=normalized_hartree_backend,
                        use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
                        jax_hartree_cg_impl=normalized_jax_hartree_cg_impl,
                        jax_hartree_cg_preconditioner=normalized_jax_hartree_cg_preconditioner,
                        jax_hartree_line_preconditioner_impl=normalized_jax_hartree_line_preconditioner_impl,
                    )
                )
                static_local_prepare_wall_time += up_preparation_profile.total_wall_time_seconds
                hartree_solve_wall_time += up_preparation_profile.hartree_resolve_wall_time_seconds
                local_ionic_resolve_wall_time += up_preparation_profile.local_ionic_resolve_wall_time_seconds
                xc_resolve_wall_time += up_preparation_profile.xc_resolve_wall_time_seconds
                iteration_static_local_prepare_elapsed += (
                    up_preparation_profile.total_wall_time_seconds
                )
                hartree_solve_call_count += 1
                _record_last_jax_hartree_solve_diagnostics(
                    enabled=(normalized_hartree_backend == "jax"),
                    solve_times=hartree_solve_times,
                    cg_iterations=hartree_cg_iterations,
                    cached_use_flags=hartree_cached_use_flags,
                    cached_first_flags=hartree_cached_first_flags,
                    boundary_condition_times=hartree_boundary_condition_times,
                    build_times=hartree_build_times,
                    rhs_assembly_times=hartree_rhs_assembly_times,
                    cg_times=hartree_cg_times,
                    matvec_call_counts=hartree_matvec_call_counts,
                    matvec_times=hartree_matvec_times,
                    preconditioner_apply_counts=hartree_preconditioner_apply_counts,
                    preconditioner_apply_times=hartree_preconditioner_apply_times,
                    preconditioner_setup_times=hartree_preconditioner_setup_times,
                    preconditioner_axis_reorder_times=hartree_preconditioner_axis_reorder_times,
                    preconditioner_tridiagonal_solve_times=hartree_preconditioner_tridiagonal_solve_times,
                    preconditioner_other_overhead_times=hartree_preconditioner_other_overhead_times,
                )
            solve_up = solve_fixed_potential_static_local_eigenproblem(
                grid_geometry=grid_geometry,
                rho_up=rho_up,
                rho_down=rho_down,
                spin_channel="up",
                k=occupations.n_alpha,
                case=case,
                initial_guess_orbitals=guess_up,
                tolerance=eigensolver_tolerance,
                ncv=eigensolver_ncv,
                use_monitor_patch=True,
                patch_parameters=patch_parameters,
                kinetic_version=kinetic_version,
                use_jax_block_kernels=use_jax_block_kernels,
                operator_context=up_operator_context,
                operator_preparation_profile=up_preparation_profile,
                base_local_ionic_evaluation=cached_base_local_ionic_evaluation,
                hartree_backend=normalized_hartree_backend,
                use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
                jax_hartree_cg_impl=normalized_jax_hartree_cg_impl,
                jax_hartree_cg_preconditioner=normalized_jax_hartree_cg_preconditioner,
                jax_hartree_line_preconditioner_impl=normalized_jax_hartree_line_preconditioner_impl,
            )
            orbitals_up = solve_up.orbitals
        else:
            orbitals_up = _empty_orbital_block(grid_geometry)

        if occupations.n_beta > 0 and _is_h2_closed_shell_singlet(occupations):
            orbitals_down = np.asarray(orbitals_up, dtype=np.float64)
        elif occupations.n_beta > 0:
            if use_step_local_static_local_reuse:
                down_operator_context, down_preparation_profile = (
                    prepare_fixed_potential_static_local_operator_profiled(
                        grid_geometry=grid_geometry,
                        rho_up=rho_up,
                        rho_down=rho_down,
                        spin_channel="down",
                        case=case,
                        use_monitor_patch=True,
                        patch_parameters=patch_parameters,
                        kinetic_version=kinetic_version,
                        base_local_ionic_evaluation=cached_base_local_ionic_evaluation,
                        hartree_backend=normalized_hartree_backend,
                        use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
                        jax_hartree_cg_impl=normalized_jax_hartree_cg_impl,
                        jax_hartree_cg_preconditioner=normalized_jax_hartree_cg_preconditioner,
                        jax_hartree_line_preconditioner_impl=normalized_jax_hartree_line_preconditioner_impl,
                    )
                )
                static_local_prepare_wall_time += down_preparation_profile.total_wall_time_seconds
                hartree_solve_wall_time += down_preparation_profile.hartree_resolve_wall_time_seconds
                local_ionic_resolve_wall_time += down_preparation_profile.local_ionic_resolve_wall_time_seconds
                xc_resolve_wall_time += down_preparation_profile.xc_resolve_wall_time_seconds
                iteration_static_local_prepare_elapsed += (
                    down_preparation_profile.total_wall_time_seconds
                )
                hartree_solve_call_count += 1
                _record_last_jax_hartree_solve_diagnostics(
                    enabled=(normalized_hartree_backend == "jax"),
                    solve_times=hartree_solve_times,
                    cg_iterations=hartree_cg_iterations,
                    cached_use_flags=hartree_cached_use_flags,
                    cached_first_flags=hartree_cached_first_flags,
                    boundary_condition_times=hartree_boundary_condition_times,
                    build_times=hartree_build_times,
                    rhs_assembly_times=hartree_rhs_assembly_times,
                    cg_times=hartree_cg_times,
                    matvec_call_counts=hartree_matvec_call_counts,
                    matvec_times=hartree_matvec_times,
                    preconditioner_apply_counts=hartree_preconditioner_apply_counts,
                    preconditioner_apply_times=hartree_preconditioner_apply_times,
                    preconditioner_setup_times=hartree_preconditioner_setup_times,
                    preconditioner_axis_reorder_times=hartree_preconditioner_axis_reorder_times,
                    preconditioner_tridiagonal_solve_times=hartree_preconditioner_tridiagonal_solve_times,
                    preconditioner_other_overhead_times=hartree_preconditioner_other_overhead_times,
                )
            solve_down = solve_fixed_potential_static_local_eigenproblem(
                grid_geometry=grid_geometry,
                rho_up=rho_up,
                rho_down=rho_down,
                spin_channel="down",
                k=occupations.n_beta,
                case=case,
                initial_guess_orbitals=guess_down,
                tolerance=eigensolver_tolerance,
                ncv=eigensolver_ncv,
                use_monitor_patch=True,
                patch_parameters=patch_parameters,
                kinetic_version=kinetic_version,
                use_jax_block_kernels=use_jax_block_kernels,
                operator_context=down_operator_context,
                operator_preparation_profile=down_preparation_profile,
                base_local_ionic_evaluation=cached_base_local_ionic_evaluation,
                hartree_backend=normalized_hartree_backend,
                use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
                jax_hartree_cg_impl=normalized_jax_hartree_cg_impl,
                jax_hartree_cg_preconditioner=normalized_jax_hartree_cg_preconditioner,
                jax_hartree_line_preconditioner_impl=normalized_jax_hartree_line_preconditioner_impl,
            )
            orbitals_down = solve_down.orbitals
        else:
            orbitals_down = _empty_orbital_block(grid_geometry)
        eigensolver_elapsed = perf_counter() - eigensolver_start
        if not use_step_local_static_local_reuse:
            for solve_result in (solve_up, solve_down):
                if solve_result is None or solve_result.static_local_preparation_profile is None:
                    continue
                static_local_prepare_wall_time += (
                    solve_result.static_local_preparation_profile.total_wall_time_seconds
                )
                hartree_solve_wall_time += (
                    solve_result.static_local_preparation_profile.hartree_resolve_wall_time_seconds
                )
                local_ionic_resolve_wall_time += (
                    solve_result.static_local_preparation_profile.local_ionic_resolve_wall_time_seconds
                )
                xc_resolve_wall_time += (
                    solve_result.static_local_preparation_profile.xc_resolve_wall_time_seconds
                )
                iteration_static_local_prepare_elapsed += (
                    solve_result.static_local_preparation_profile.total_wall_time_seconds
                )
                hartree_solve_call_count += 1
        eigensolver_core_elapsed = max(
            0.0,
            float(eigensolver_elapsed - iteration_static_local_prepare_elapsed),
        )
        eigensolver_wall_time += eigensolver_core_elapsed

        density_update_start = perf_counter()
        rho_up_out = _renormalize_density(
            _build_density_from_occupied_orbitals(
                orbitals_up,
                occupations.occupations_up,
                grid_geometry=grid_geometry,
            ),
            occupations.n_alpha,
            grid_geometry=grid_geometry,
        )
        rho_down_out = _renormalize_density(
            _build_density_from_occupied_orbitals(
                orbitals_down,
                occupations.occupations_down,
                grid_geometry=grid_geometry,
            ),
            occupations.n_beta,
            grid_geometry=grid_geometry,
        )

        _check_density_electron_count(rho_up_out, occupations.n_alpha, grid_geometry, "rho_up_out")
        _check_density_electron_count(rho_down_out, occupations.n_beta, grid_geometry, "rho_down_out")

        density_residual = _density_residual(
            rho_up_in=rho_up,
            rho_down_in=rho_down,
            rho_up_out=rho_up_out,
            rho_down_out=rho_down_out,
            grid_geometry=grid_geometry,
        )
        density_update_elapsed = perf_counter() - density_update_start
        density_update_wall_time += density_update_elapsed

        energy_evaluation_start = perf_counter()
        if use_step_local_static_local_reuse:
            energy_spin_channel = (
                "up"
                if occupations.n_alpha > 0
                else "down"
            )
            energy_context, energy_preparation_profile = (
                prepare_fixed_potential_static_local_operator_profiled(
                    grid_geometry=grid_geometry,
                    rho_up=rho_up_out,
                    rho_down=rho_down_out,
                    spin_channel=energy_spin_channel,
                    case=case,
                    use_monitor_patch=True,
                    patch_parameters=patch_parameters,
                    kinetic_version=kinetic_version,
                    base_local_ionic_evaluation=cached_base_local_ionic_evaluation,
                    hartree_backend=normalized_hartree_backend,
                    use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
                    jax_hartree_cg_impl=normalized_jax_hartree_cg_impl,
                    jax_hartree_cg_preconditioner=normalized_jax_hartree_cg_preconditioner,
                    jax_hartree_line_preconditioner_impl=normalized_jax_hartree_line_preconditioner_impl,
                )
            )
            static_local_prepare_wall_time += energy_preparation_profile.total_wall_time_seconds
            hartree_solve_wall_time += energy_preparation_profile.hartree_resolve_wall_time_seconds
            local_ionic_resolve_wall_time += (
                energy_preparation_profile.local_ionic_resolve_wall_time_seconds
            )
            xc_resolve_wall_time += energy_preparation_profile.xc_resolve_wall_time_seconds
            hartree_solve_call_count += 1
            _record_last_jax_hartree_solve_diagnostics(
                enabled=(normalized_hartree_backend == "jax"),
                solve_times=hartree_solve_times,
                cg_iterations=hartree_cg_iterations,
                cached_use_flags=hartree_cached_use_flags,
                cached_first_flags=hartree_cached_first_flags,
                boundary_condition_times=hartree_boundary_condition_times,
                build_times=hartree_build_times,
                rhs_assembly_times=hartree_rhs_assembly_times,
                cg_times=hartree_cg_times,
                matvec_call_counts=hartree_matvec_call_counts,
                matvec_times=hartree_matvec_times,
                preconditioner_apply_counts=hartree_preconditioner_apply_counts,
                preconditioner_apply_times=hartree_preconditioner_apply_times,
                preconditioner_setup_times=hartree_preconditioner_setup_times,
                preconditioner_axis_reorder_times=hartree_preconditioner_axis_reorder_times,
                preconditioner_tridiagonal_solve_times=hartree_preconditioner_tridiagonal_solve_times,
                preconditioner_other_overhead_times=hartree_preconditioner_other_overhead_times,
            )
            energy, energy_profile = evaluate_static_local_single_point_energy_from_context(
                energy_context,
                orbitals_up=orbitals_up,
                orbitals_down=orbitals_down,
                occupations=occupations,
            )
            kinetic_energy_wall_time += energy_profile.kinetic_wall_time_seconds
            local_ionic_energy_wall_time += energy_profile.local_ionic_wall_time_seconds
            hartree_energy_wall_time += energy_profile.hartree_energy_wall_time_seconds
            xc_energy_wall_time += energy_profile.xc_energy_wall_time_seconds
            ion_ion_energy_wall_time += energy_profile.ion_ion_wall_time_seconds
        else:
            energy = evaluate_static_local_single_point_energy(
                rho_up=rho_up_out,
                rho_down=rho_down_out,
                orbitals_up=orbitals_up,
                orbitals_down=orbitals_down,
                occupations=occupations,
                grid_geometry=grid_geometry,
                case=case,
                use_monitor_patch=True,
                patch_parameters=patch_parameters,
                kinetic_version=kinetic_version,
                hartree_backend=normalized_hartree_backend,
                use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
                jax_hartree_cg_impl=normalized_jax_hartree_cg_impl,
                jax_hartree_cg_preconditioner=normalized_jax_hartree_cg_preconditioner,
            )
            hartree_solve_call_count += 1
            _record_last_jax_hartree_solve_diagnostics(
                enabled=(normalized_hartree_backend == "jax"),
                solve_times=hartree_solve_times,
                cg_iterations=hartree_cg_iterations,
                cached_use_flags=hartree_cached_use_flags,
                cached_first_flags=hartree_cached_first_flags,
                boundary_condition_times=hartree_boundary_condition_times,
                build_times=hartree_build_times,
                rhs_assembly_times=hartree_rhs_assembly_times,
                cg_times=hartree_cg_times,
                matvec_call_counts=hartree_matvec_call_counts,
                matvec_times=hartree_matvec_times,
                preconditioner_apply_counts=hartree_preconditioner_apply_counts,
                preconditioner_apply_times=hartree_preconditioner_apply_times,
                preconditioner_setup_times=hartree_preconditioner_setup_times,
                preconditioner_axis_reorder_times=hartree_preconditioner_axis_reorder_times,
                preconditioner_tridiagonal_solve_times=hartree_preconditioner_tridiagonal_solve_times,
                preconditioner_other_overhead_times=hartree_preconditioner_other_overhead_times,
            )
        energy_evaluation_elapsed = perf_counter() - energy_evaluation_start
        energy_evaluation_wall_time += energy_evaluation_elapsed

        bookkeeping_start = perf_counter()
        energy_change = None if previous_energy_total is None else energy.total - previous_energy_total
        residual_up, residual_down = _density_residual_fields(
            rho_up_in=rho_up,
            rho_down_in=rho_down,
            rho_up_out=rho_up_out,
            rho_down_out=rho_down_out,
        )

        rho_up_mixed = _renormalize_density(
            (1.0 - mixing) * rho_up + mixing * rho_up_out,
            occupations.n_alpha,
            grid_geometry=grid_geometry,
        )
        rho_down_mixed = _renormalize_density(
            (1.0 - mixing) * rho_down + mixing * rho_down_out,
            occupations.n_beta,
            grid_geometry=grid_geometry,
        )
        if (
            enable_cycle_breaker
            and _is_h2_closed_shell_singlet(occupations)
            and _detect_monitor_grid_singlet_alternation(
                history,
                rho_up_out=rho_up_out,
                rho_down_out=rho_down_out,
                energy_change=energy_change,
                grid_geometry=grid_geometry,
            )
        ):
            rho_up_mixed = _renormalize_density(
                (1.0 - cycle_breaker_weight) * rho_up_mixed
                + cycle_breaker_weight * history[-1].mixed_rho_up,
                occupations.n_alpha,
                grid_geometry=grid_geometry,
            )
            rho_down_mixed = _renormalize_density(
                (1.0 - cycle_breaker_weight) * rho_down_mixed
                + cycle_breaker_weight * history[-1].mixed_rho_down,
                occupations.n_beta,
                grid_geometry=grid_geometry,
            )
            cycle_breaker_triggered_iterations.append(iteration)
        elif enable_diis:
            diis_history.append(
                MonitorGridDiisHistoryEntry(
                    mixed_rho_up=np.asarray(rho_up_mixed, dtype=np.float64),
                    mixed_rho_down=np.asarray(rho_down_mixed, dtype=np.float64),
                    residual_up=residual_up,
                    residual_down=residual_down,
                )
            )
            if len(diis_history) > diis_history_length:
                diis_history = diis_history[-diis_history_length:]
            if iteration >= diis_warmup_iterations and len(diis_history) >= 2:
                diis_candidate = _apply_monitor_grid_density_diis(
                    tuple(diis_history),
                    grid_geometry=grid_geometry,
                    n_alpha=occupations.n_alpha,
                    n_beta=occupations.n_beta,
                )
                if diis_candidate is not None:
                    rho_up_mixed, rho_down_mixed = diis_candidate
                    diis_used_iterations.append(iteration)
                else:
                    diis_fallback_iterations.append(iteration)
            diis_history_sizes.append(len(diis_history))
        else:
            diis_history_sizes.append(0)

        history.append(
            ScfIterationRecord(
                iteration=iteration,
                input_rho_up=np.asarray(rho_up, dtype=np.float64),
                input_rho_down=np.asarray(rho_down, dtype=np.float64),
                output_rho_up=np.asarray(rho_up_out, dtype=np.float64),
                output_rho_down=np.asarray(rho_down_out, dtype=np.float64),
                mixed_rho_up=np.asarray(rho_up_mixed, dtype=np.float64),
                mixed_rho_down=np.asarray(rho_down_mixed, dtype=np.float64),
                density_residual=float(density_residual),
                energy=energy,
                energy_change=None if energy_change is None else float(energy_change),
                solve_up=_build_solve_summary(solve_up, "up", occupations.n_alpha),
                solve_down=(
                    _build_solve_summary(solve_up, "down", occupations.n_beta)
                    if occupations.n_beta > 0 and _is_h2_closed_shell_singlet(occupations)
                    else _build_solve_summary(solve_down, "down", occupations.n_beta)
                ),
            )
        )

        final_orbitals_up = orbitals_up
        final_orbitals_down = orbitals_down
        final_eigenvalues_up = (
            np.asarray(solve_up.eigenvalues, dtype=np.float64)
            if solve_up is not None
            else np.zeros(0, dtype=np.float64)
        )
        final_eigenvalues_down = (
            np.asarray(final_eigenvalues_up, dtype=np.float64)
            if occupations.n_beta > 0 and _is_h2_closed_shell_singlet(occupations)
            else np.asarray(solve_down.eigenvalues, dtype=np.float64)
            if solve_down is not None
            else np.zeros(0, dtype=np.float64)
        )
        final_solve_up = solve_up
        final_solve_down = (
            solve_up
            if occupations.n_beta > 0 and _is_h2_closed_shell_singlet(occupations)
            else solve_down
        )
        final_energy = energy

        if density_residual < density_tolerance and (
            energy_change is None or abs(energy_change) < energy_tolerance
        ):
            rho_up = rho_up_out
            rho_down = rho_down_out
            converged = True
            break

        rho_up = rho_up_mixed
        rho_down = rho_down_mixed
        guess_up = orbitals_up
        guess_down = orbitals_down
        previous_energy_total = energy.total
        bookkeeping_elapsed = perf_counter() - bookkeeping_start

        iteration_eigensolver_times.append(float(eigensolver_core_elapsed))
        iteration_energy_evaluation_times.append(float(energy_evaluation_elapsed))
        iteration_density_update_times.append(float(density_update_elapsed))
        iteration_bookkeeping_times.append(float(bookkeeping_elapsed))

    lowest_eigenvalue = None
    if final_eigenvalues_up.size:
        lowest_eigenvalue = float(final_eigenvalues_up[0])
    if final_eigenvalues_down.size:
        candidate = float(final_eigenvalues_down[0])
        lowest_eigenvalue = candidate if lowest_eigenvalue is None else min(lowest_eigenvalue, candidate)

    total_wall_time_seconds = float(perf_counter() - total_wall_start)
    iteration_count = len(history)
    average_iteration_wall_time_seconds = (
        None if iteration_count == 0 else float(total_wall_time_seconds / iteration_count)
    )
    average_hartree_solve_wall_time_seconds = (
        None if not hartree_solve_times else float(np.mean(hartree_solve_times))
    )
    first_hartree_solve_wall_time_seconds = (
        None if not hartree_solve_times else float(hartree_solve_times[0])
    )
    repeated_hartree_solve_average_wall_time_seconds = (
        None
        if len(hartree_solve_times) <= 1
        else float(np.mean(hartree_solve_times[1:]))
    )
    repeated_hartree_solve_min_wall_time_seconds = (
        None
        if len(hartree_solve_times) <= 1
        else float(np.min(hartree_solve_times[1:]))
    )
    repeated_hartree_solve_max_wall_time_seconds = (
        None
        if len(hartree_solve_times) <= 1
        else float(np.max(hartree_solve_times[1:]))
    )
    average_hartree_cg_iterations = (
        None if not hartree_cg_iterations else float(np.mean(hartree_cg_iterations))
    )
    first_hartree_cg_iterations = (
        None if not hartree_cg_iterations else int(hartree_cg_iterations[0])
    )
    repeated_hartree_cg_iteration_average = (
        None
        if len(hartree_cg_iterations) <= 1
        else float(np.mean(hartree_cg_iterations[1:]))
    )
    average_hartree_boundary_condition_wall_time_seconds = (
        None
        if not hartree_boundary_condition_times
        else float(np.mean(hartree_boundary_condition_times))
    )
    average_hartree_build_wall_time_seconds = (
        None if not hartree_build_times else float(np.mean(hartree_build_times))
    )
    average_hartree_rhs_assembly_wall_time_seconds = (
        None if not hartree_rhs_assembly_times else float(np.mean(hartree_rhs_assembly_times))
    )
    average_hartree_cg_wall_time_seconds = (
        None if not hartree_cg_times else float(np.mean(hartree_cg_times))
    )
    average_hartree_matvec_call_count = (
        None if not hartree_matvec_call_counts else float(np.mean(hartree_matvec_call_counts))
    )
    average_hartree_matvec_wall_time_seconds = (
        None if not hartree_matvec_times else float(np.mean(hartree_matvec_times))
    )
    average_hartree_matvec_wall_time_per_call_seconds = (
        None
        if not hartree_matvec_times or not hartree_matvec_call_counts or sum(hartree_matvec_call_counts) == 0
        else float(sum(hartree_matvec_times) / sum(hartree_matvec_call_counts))
    )
    average_hartree_preconditioner_apply_count = (
        None
        if not hartree_preconditioner_apply_counts
        else float(np.mean(hartree_preconditioner_apply_counts))
    )
    average_hartree_preconditioner_apply_wall_time_seconds = (
        None
        if not hartree_preconditioner_apply_times
        else float(np.mean(hartree_preconditioner_apply_times))
    )
    average_hartree_preconditioner_apply_wall_time_per_call_seconds = (
        None
        if not hartree_preconditioner_apply_times
        or not hartree_preconditioner_apply_counts
        or sum(hartree_preconditioner_apply_counts) == 0
        else float(
            sum(hartree_preconditioner_apply_times) / sum(hartree_preconditioner_apply_counts)
        )
    )
    average_hartree_preconditioner_setup_wall_time_seconds = (
        None
        if not hartree_preconditioner_setup_times
        else float(np.mean(hartree_preconditioner_setup_times))
    )
    average_hartree_preconditioner_axis_reorder_wall_time_seconds = (
        None
        if not hartree_preconditioner_axis_reorder_times
        else float(np.mean(hartree_preconditioner_axis_reorder_times))
    )
    average_hartree_preconditioner_tridiagonal_solve_wall_time_seconds = (
        None
        if not hartree_preconditioner_tridiagonal_solve_times
        else float(np.mean(hartree_preconditioner_tridiagonal_solve_times))
    )
    average_hartree_preconditioner_other_overhead_wall_time_seconds = (
        None
        if not hartree_preconditioner_other_overhead_times
        else float(np.mean(hartree_preconditioner_other_overhead_times))
    )
    bookkeeping_wall_time_seconds = float(
        total_wall_time_seconds
        - eigensolver_wall_time
        - energy_evaluation_wall_time
        - density_update_wall_time
    )

    return H2StaticLocalScfDryRunResult(
        path_type="monitor_a_grid_plus_patch",
        kinetic_version=kinetic_version,
        includes_nonlocal=False,
        spin_state_label=occupations.label,
        spin=occupations.spin,
        occupations=occupations,
        parameter_summary=_monitor_grid_scf_parameter_summary(
            patch_parameters,
            kinetic_version=kinetic_version,
            hartree_backend=normalized_hartree_backend,
            use_jax_hartree_cached_operator=use_jax_hartree_cached_operator,
            jax_hartree_cg_impl=normalized_jax_hartree_cg_impl,
            jax_hartree_cg_preconditioner=normalized_jax_hartree_cg_preconditioner,
            jax_hartree_line_preconditioner_impl=normalized_jax_hartree_line_preconditioner_impl,
            use_jax_block_kernels=use_jax_block_kernels,
            use_step_local_static_local_reuse=use_step_local_static_local_reuse,
            cycle_breaker_enabled=enable_cycle_breaker,
            cycle_breaker_weight=cycle_breaker_weight,
            diis_enabled=enable_diis,
            diis_warmup_iterations=diis_warmup_iterations,
            diis_history_length=diis_history_length,
        ),
        hartree_backend=normalized_hartree_backend,
        use_jax_hartree_cached_operator=bool(use_jax_hartree_cached_operator),
        jax_hartree_cg_impl=normalized_jax_hartree_cg_impl,
        jax_hartree_cg_preconditioner=normalized_jax_hartree_cg_preconditioner,
        jax_hartree_line_preconditioner_impl=normalized_jax_hartree_line_preconditioner_impl,
        use_jax_block_kernels=bool(use_jax_block_kernels),
        use_step_local_static_local_reuse=bool(use_step_local_static_local_reuse),
        cycle_breaker_enabled=bool(enable_cycle_breaker),
        cycle_breaker_weight=float(cycle_breaker_weight),
        cycle_breaker_triggered_iterations=tuple(cycle_breaker_triggered_iterations),
        diis_enabled=bool(enable_diis),
        diis_warmup_iterations=int(diis_warmup_iterations),
        diis_history_length=int(diis_history_length),
        diis_residual_definition="density_fixed_point_residual=rho_out-rho_in",
        diis_used_iterations=tuple(diis_used_iterations),
        diis_history_sizes=tuple(diis_history_sizes),
        diis_fallback_iterations=tuple(diis_fallback_iterations),
        converged=converged,
        iteration_count=len(history),
        history=tuple(history),
        energy_history=tuple(record.energy.total for record in history),
        density_residual_history=tuple(record.density_residual for record in history),
        eigenvalues_up=final_eigenvalues_up,
        eigenvalues_down=final_eigenvalues_down,
        orbitals_up=np.asarray(final_orbitals_up, dtype=np.float64),
        orbitals_down=np.asarray(final_orbitals_down, dtype=np.float64),
        rho_up=np.asarray(rho_up, dtype=np.float64),
        rho_down=np.asarray(rho_down, dtype=np.float64),
        energy=final_energy,
        lowest_eigenvalue=lowest_eigenvalue,
        solve_up=final_solve_up,
        solve_down=final_solve_down,
        total_wall_time_seconds=total_wall_time_seconds,
        average_iteration_wall_time_seconds=average_iteration_wall_time_seconds,
        eigensolver_wall_time_seconds=float(eigensolver_wall_time),
        static_local_prepare_wall_time_seconds=float(static_local_prepare_wall_time),
        hartree_solve_wall_time_seconds=float(hartree_solve_wall_time),
        local_ionic_resolve_wall_time_seconds=float(local_ionic_resolve_wall_time),
        xc_resolve_wall_time_seconds=float(xc_resolve_wall_time),
        energy_evaluation_wall_time_seconds=float(energy_evaluation_wall_time),
        kinetic_energy_wall_time_seconds=float(kinetic_energy_wall_time),
        local_ionic_energy_wall_time_seconds=float(local_ionic_energy_wall_time),
        hartree_energy_wall_time_seconds=float(hartree_energy_wall_time),
        xc_energy_wall_time_seconds=float(xc_energy_wall_time),
        ion_ion_energy_wall_time_seconds=float(ion_ion_energy_wall_time),
        density_update_wall_time_seconds=float(density_update_wall_time),
        bookkeeping_wall_time_seconds=bookkeeping_wall_time_seconds,
        hartree_solve_call_count=int(hartree_solve_call_count),
        average_hartree_solve_wall_time_seconds=average_hartree_solve_wall_time_seconds,
        first_hartree_solve_wall_time_seconds=first_hartree_solve_wall_time_seconds,
        repeated_hartree_solve_average_wall_time_seconds=repeated_hartree_solve_average_wall_time_seconds,
        repeated_hartree_solve_min_wall_time_seconds=repeated_hartree_solve_min_wall_time_seconds,
        repeated_hartree_solve_max_wall_time_seconds=repeated_hartree_solve_max_wall_time_seconds,
        average_hartree_cg_iterations=average_hartree_cg_iterations,
        first_hartree_cg_iterations=first_hartree_cg_iterations,
        repeated_hartree_cg_iteration_average=repeated_hartree_cg_iteration_average,
        average_hartree_boundary_condition_wall_time_seconds=average_hartree_boundary_condition_wall_time_seconds,
        average_hartree_build_wall_time_seconds=average_hartree_build_wall_time_seconds,
        average_hartree_rhs_assembly_wall_time_seconds=average_hartree_rhs_assembly_wall_time_seconds,
        average_hartree_cg_wall_time_seconds=average_hartree_cg_wall_time_seconds,
        average_hartree_matvec_call_count=average_hartree_matvec_call_count,
        average_hartree_matvec_wall_time_seconds=average_hartree_matvec_wall_time_seconds,
        average_hartree_matvec_wall_time_per_call_seconds=average_hartree_matvec_wall_time_per_call_seconds,
        average_hartree_preconditioner_apply_count=average_hartree_preconditioner_apply_count,
        average_hartree_preconditioner_apply_wall_time_seconds=average_hartree_preconditioner_apply_wall_time_seconds,
        average_hartree_preconditioner_apply_wall_time_per_call_seconds=average_hartree_preconditioner_apply_wall_time_per_call_seconds,
        average_hartree_preconditioner_setup_wall_time_seconds=average_hartree_preconditioner_setup_wall_time_seconds,
        average_hartree_preconditioner_axis_reorder_wall_time_seconds=average_hartree_preconditioner_axis_reorder_wall_time_seconds,
        average_hartree_preconditioner_tridiagonal_solve_wall_time_seconds=average_hartree_preconditioner_tridiagonal_solve_wall_time_seconds,
        average_hartree_preconditioner_other_overhead_wall_time_seconds=average_hartree_preconditioner_other_overhead_wall_time_seconds,
        hartree_cached_operator_usage_count=int(sum(hartree_cached_use_flags)),
        hartree_cached_operator_first_solve_count=int(sum(hartree_cached_first_flags)),
        hartree_solve_wall_time_seconds_history=tuple(float(value) for value in hartree_solve_times),
        hartree_cg_iterations_history=tuple(int(value) for value in hartree_cg_iterations),
        hartree_boundary_condition_wall_time_seconds_history=tuple(
            float(value) for value in hartree_boundary_condition_times
        ),
        hartree_build_wall_time_seconds_history=tuple(float(value) for value in hartree_build_times),
        hartree_rhs_assembly_wall_time_seconds_history=tuple(
            float(value) for value in hartree_rhs_assembly_times
        ),
        hartree_cg_wall_time_seconds_history=tuple(float(value) for value in hartree_cg_times),
        hartree_matvec_call_count_history=tuple(int(value) for value in hartree_matvec_call_counts),
        hartree_matvec_wall_time_seconds_history=tuple(float(value) for value in hartree_matvec_times),
        hartree_preconditioner_apply_count_history=tuple(
            int(value) for value in hartree_preconditioner_apply_counts
        ),
        hartree_preconditioner_apply_wall_time_seconds_history=tuple(
            float(value) for value in hartree_preconditioner_apply_times
        ),
        hartree_preconditioner_setup_wall_time_seconds_history=tuple(
            float(value) for value in hartree_preconditioner_setup_times
        ),
        hartree_preconditioner_axis_reorder_wall_time_seconds_history=tuple(
            float(value) for value in hartree_preconditioner_axis_reorder_times
        ),
        hartree_preconditioner_tridiagonal_solve_wall_time_seconds_history=tuple(
            float(value) for value in hartree_preconditioner_tridiagonal_solve_times
        ),
        hartree_preconditioner_other_overhead_wall_time_seconds_history=tuple(
            float(value) for value in hartree_preconditioner_other_overhead_times
        ),
        eigensolver_iteration_wall_time_seconds=tuple(iteration_eigensolver_times),
        energy_evaluation_iteration_wall_time_seconds=tuple(iteration_energy_evaluation_times),
        density_update_iteration_wall_time_seconds=tuple(iteration_density_update_times),
        bookkeeping_iteration_wall_time_seconds=tuple(iteration_bookkeeping_times),
    )
