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

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.ks import FixedPotentialEigensolverResult
from isogrid.ks import build_total_density
from isogrid.ks import solve_fixed_potential_eigenproblem
from isogrid.ks import validate_orbital_block
from isogrid.ks import weighted_overlap_matrix
from isogrid.ks import weighted_orthonormalize_orbitals
from isogrid.ops import apply_kinetic_operator
from isogrid.ops import integrate_field
from isogrid.ops import validate_orbital_field
from isogrid.ops import weighted_l2_norm
from isogrid.poisson import evaluate_hartree_energy
from isogrid.poisson import solve_hartree_potential
from isogrid.pseudo import evaluate_local_ionic_potential
from isogrid.pseudo import evaluate_nonlocal_ionic_action
from isogrid.pseudo import load_case_gth_pseudo_data
from isogrid.xc import evaluate_lsda_terms

_SUPPORTED_CASE_NAME = H2_BENCHMARK_CASE.name
_ZERO_BLOCK_DTYPE = np.float64


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


def _empty_orbital_block(grid_geometry: StructuredGridGeometry) -> np.ndarray:
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
    grid_geometry: StructuredGridGeometry,
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
    grid_geometry: StructuredGridGeometry,
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
    grid_geometry: StructuredGridGeometry,
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
    grid_geometry: StructuredGridGeometry,
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
    grid_geometry: StructuredGridGeometry,
) -> float:
    up_norm = weighted_l2_norm(rho_up_out - rho_up_in, grid_geometry=grid_geometry)
    down_norm = weighted_l2_norm(rho_down_out - rho_down_in, grid_geometry=grid_geometry)
    return float(np.sqrt(up_norm * up_norm + down_norm * down_norm))


def _build_h2_trial_orbitals(
    case: BenchmarkCase,
    grid_geometry: StructuredGridGeometry,
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
    grid_geometry: StructuredGridGeometry | None = None,
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


def _check_density_electron_count(
    density: np.ndarray,
    target_electrons: float,
    grid_geometry: StructuredGridGeometry,
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
