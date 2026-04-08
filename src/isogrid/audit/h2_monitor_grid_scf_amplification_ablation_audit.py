"""Lightweight early-step SCF amplification/channel-ablation audit on the H2 monitor grid."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ks import FixedPotentialEigensolverResult
from isogrid.ks import FixedPotentialStaticLocalOperatorContext
from isogrid.ks import prepare_fixed_potential_static_local_operator
from isogrid.ks import solve_fixed_potential_static_local_eigenproblem
from isogrid.scf.active_subspace import ActiveSubspaceConfig
from isogrid.scf.active_subspace import ActiveSubspaceState
from isogrid.scf.active_subspace import initialize_active_subspace
from isogrid.scf.active_subspace import update_active_subspace
from isogrid.ks import weighted_overlap_matrix
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import run_h2_monitor_grid_scf_dry_run
from isogrid.scf.driver import _build_density_from_occupied_orbitals
from isogrid.scf.driver import _build_h2_trial_orbitals
from isogrid.scf.driver import _closed_shell_singlet_tracking_solve_count
from isogrid.scf.driver import _default_monitor_patch_parameters
from isogrid.scf.driver import _density_residual
from isogrid.scf.driver import _is_h2_closed_shell_singlet
from isogrid.scf.driver import _renormalize_density
from isogrid.scf.driver import _select_closed_shell_singlet_tracked_occupied_orbital
from isogrid.scf.driver import resolve_h2_spin_occupations

_DEFAULT_SHAPE = (9, 9, 11)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (6.0, 6.0, 8.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 2
_DEFAULT_MIXING = 0.20
_DEFAULT_DENSITY_TOLERANCE = 1.0e-2
_DEFAULT_ENERGY_TOLERANCE = 1.0e-4
_DEFAULT_EIGENSOLVER_TOLERANCE = 1.0e-2
_DEFAULT_EIGENSOLVER_NCV = 8
_ROUTE_LABELS = ("baseline", "freeze_hartree", "freeze_xc")


@dataclass(frozen=True)
class H2MonitorGridScfAmplificationReplayRoute:
    """One replay route evaluated on one early-step secant pair."""

    route_label: str
    pair_iterations: tuple[int, int]
    density_residual: float
    density_residual_ratio: float | None
    delta_hartree_potential_rms: float
    delta_xc_potential_rms: float
    delta_local_potential_rms: float
    hartree_share: float | None
    xc_share: float | None
    local_share: float | None
    occupied_orbital_overlap_abs: float | None
    lowest2_subspace_overlap_min_singular_value: float | None
    lowest2_subspace_rotation_max_angle_deg: float | None
    lowest_gap_ha: float | None
    lowest_gap_delta_ha: float | None


@dataclass(frozen=True)
class H2MonitorGridScfAmplificationPairAudit:
    """Replay comparison for one consecutive input-density pair."""

    pair_iterations: tuple[int, int]
    baseline: H2MonitorGridScfAmplificationReplayRoute
    freeze_hartree: H2MonitorGridScfAmplificationReplayRoute
    freeze_xc: H2MonitorGridScfAmplificationReplayRoute


@dataclass(frozen=True)
class H2MonitorGridScfAmplificationSpinAudit:
    """All early-step replay audits for one spin state."""

    spin_state_label: str
    source_iteration_count: int
    density_residual_history: tuple[float, ...]
    pair_audits: tuple[H2MonitorGridScfAmplificationPairAudit, ...]


@dataclass(frozen=True)
class H2MonitorGridScfAmplificationAblationAuditResult:
    """Top-level lightweight amplification/channel-ablation result."""

    case_name: str
    grid_parameter_summary: str
    singlet: H2MonitorGridScfAmplificationSpinAudit
    triplet: H2MonitorGridScfAmplificationSpinAudit
    note: str


def _weighted_field_norm(field: np.ndarray, *, grid_geometry: MonitorGridGeometry) -> float:
    weights = np.asarray(grid_geometry.cell_volumes, dtype=np.float64)
    value = float(np.sum(np.asarray(field, dtype=np.float64) ** 2 * weights, dtype=np.float64))
    return float(np.sqrt(max(value, 0.0)))


def _channel_shares(
    *,
    previous_context: FixedPotentialStaticLocalOperatorContext,
    current_context: FixedPotentialStaticLocalOperatorContext,
) -> tuple[float, float, float, float | None, float | None, float | None]:
    delta_hartree = np.asarray(
        current_context.hartree_potential - previous_context.hartree_potential,
        dtype=np.float64,
    )
    delta_xc = np.asarray(
        current_context.xc_potential - previous_context.xc_potential,
        dtype=np.float64,
    )
    delta_total = np.asarray(
        current_context.effective_local_potential - previous_context.effective_local_potential,
        dtype=np.float64,
    )
    delta_local = np.asarray(delta_total - delta_hartree - delta_xc, dtype=np.float64)
    hartree_norm = _weighted_field_norm(delta_hartree, grid_geometry=current_context.grid_geometry)
    xc_norm = _weighted_field_norm(delta_xc, grid_geometry=current_context.grid_geometry)
    local_norm = _weighted_field_norm(delta_local, grid_geometry=current_context.grid_geometry)
    denominator = hartree_norm + xc_norm + local_norm
    if denominator <= 1.0e-16:
        return hartree_norm, xc_norm, local_norm, None, None, None
    return (
        hartree_norm,
        xc_norm,
        local_norm,
        float(hartree_norm / denominator),
        float(xc_norm / denominator),
        float(local_norm / denominator),
    )


def _route_context(
    *,
    route_label: str,
    previous_context: FixedPotentialStaticLocalOperatorContext,
    current_context: FixedPotentialStaticLocalOperatorContext,
) -> FixedPotentialStaticLocalOperatorContext:
    if route_label == "baseline":
        return current_context
    if route_label == "freeze_hartree":
        return replace(
            current_context,
            hartree_potential=np.asarray(previous_context.hartree_potential, dtype=np.float64),
            effective_local_potential=np.asarray(
                current_context.local_ionic_potential
                + previous_context.hartree_potential
                + current_context.xc_potential,
                dtype=np.float64,
            ),
            hartree_poisson_result=previous_context.hartree_poisson_result,
        )
    if route_label == "freeze_xc":
        return replace(
            current_context,
            xc_potential=np.asarray(previous_context.xc_potential, dtype=np.float64),
            effective_local_potential=np.asarray(
                current_context.local_ionic_potential
                + current_context.hartree_potential
                + previous_context.xc_potential,
                dtype=np.float64,
            ),
            lsda_evaluation=previous_context.lsda_evaluation,
        )
    raise ValueError(f"Unsupported route_label `{route_label}`.")


def _empty_orbital_block(grid_geometry: MonitorGridGeometry) -> np.ndarray:
    return np.zeros((0,) + grid_geometry.spec.shape, dtype=np.float64)


def _shared_source_result(
    *,
    spin_label: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    source_iteration_count: int,
    controller_name: str,
) -> H2StaticLocalScfDryRunResult:
    return run_h2_monitor_grid_scf_dry_run(
        spin_label,
        case=case,
        grid_geometry=grid_geometry,
        max_iterations=source_iteration_count,
        mixing=_DEFAULT_MIXING,
        density_tolerance=_DEFAULT_DENSITY_TOLERANCE,
        energy_tolerance=_DEFAULT_ENERGY_TOLERANCE,
        eigensolver_tolerance=_DEFAULT_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_DEFAULT_EIGENSOLVER_NCV,
        kinetic_version="trial_fix",
        hartree_backend="python",
        monitor_boundary_construction_mode="corrected_moments",
        controller_name=controller_name,
    )


def _baseline_contexts(
    *,
    source_result: H2StaticLocalScfDryRunResult,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
) -> tuple[FixedPotentialStaticLocalOperatorContext, ...]:
    return tuple(
        prepare_fixed_potential_static_local_operator(
            grid_geometry=grid_geometry,
            rho_up=record.input_rho_up,
            rho_down=record.input_rho_down,
            spin_channel="up",
            case=case,
            use_monitor_patch=True,
            patch_parameters=_default_monitor_patch_parameters(),
            kinetic_version="trial_fix",
            hartree_backend="python",
            monitor_boundary_construction_mode="corrected_moments",
        )
        for record in source_result.history
    )


def _k_track(occupations, *, track_lowest_two_states: bool) -> int:
    occupied_count = max(
        int(occupations.n_alpha),
        int(occupations.n_beta),
        _closed_shell_singlet_tracking_solve_count(occupations),
        1,
    )
    if track_lowest_two_states:
        return max(2, occupied_count)
    return occupied_count


def _build_initial_track_guess(
    *,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    count: int,
) -> np.ndarray:
    trial_orbitals = _build_h2_trial_orbitals(case=case, grid_geometry=grid_geometry)
    return np.asarray(trial_orbitals[:count], dtype=np.float64)


def _solve_track_block(
    *,
    operator_context: FixedPotentialStaticLocalOperatorContext,
    k: int,
    case: BenchmarkCase,
    initial_guess_orbitals: np.ndarray,
) -> FixedPotentialEigensolverResult:
    return solve_fixed_potential_static_local_eigenproblem(
        grid_geometry=operator_context.grid_geometry,
        rho_up=operator_context.rho_up,
        rho_down=operator_context.rho_down,
        spin_channel=operator_context.spin_channel,
        k=k,
        case=case,
        initial_guess_orbitals=initial_guess_orbitals,
        operator_context=operator_context,
        tolerance=_DEFAULT_EIGENSOLVER_TOLERANCE,
        ncv=_DEFAULT_EIGENSOLVER_NCV,
        use_monitor_patch=True,
        patch_parameters=_default_monitor_patch_parameters(),
        kinetic_version="trial_fix",
        solver_backend="scipy_fallback",
    )


def _baseline_track_solves(
    *,
    contexts: tuple[FixedPotentialStaticLocalOperatorContext, ...],
    case: BenchmarkCase,
    count: int,
    occupations,
    grid_geometry: MonitorGridGeometry,
) -> tuple[tuple[FixedPotentialEigensolverResult, ...], tuple[np.ndarray, ...]]:
    solves: list[FixedPotentialEigensolverResult] = []
    tracked_occupied_blocks: list[np.ndarray] = []
    initial_guess = _build_initial_track_guess(
        case=case,
        grid_geometry=contexts[0].grid_geometry,
        count=count,
    )
    tracked_reference_orbital = np.asarray(initial_guess[:1], dtype=np.float64)
    active_subspace_config = (
        ActiveSubspaceConfig(enabled=True, subspace_size=2, target_occupied_count=1)
        if _is_h2_closed_shell_singlet(occupations) and count >= 2
        else None
    )
    active_subspace_state: ActiveSubspaceState | None = None
    for index, context in enumerate(contexts):
        if index > 0:
            initial_guess = np.asarray(solves[-1].orbitals[:count], dtype=np.float64)
        solve = _solve_track_block(
            operator_context=context,
            k=count,
            case=case,
            initial_guess_orbitals=initial_guess,
        )
        solves.append(solve)
        if _is_h2_closed_shell_singlet(occupations):
            if active_subspace_config is not None:
                selection = (
                    initialize_active_subspace(
                        raw_subspace_orbitals=np.asarray(
                            solve.orbitals[: active_subspace_config.subspace_size],
                            dtype=np.float64,
                        ),
                        grid_geometry=grid_geometry,
                        config=active_subspace_config,
                    )
                    if active_subspace_state is None
                    else update_active_subspace(
                        raw_subspace_orbitals=np.asarray(
                            solve.orbitals[: active_subspace_config.subspace_size],
                            dtype=np.float64,
                        ),
                        state=active_subspace_state,
                        grid_geometry=grid_geometry,
                    )
                )
                active_subspace_state = selection.state
                tracked_occupied = np.asarray(selection.occupied_orbitals, dtype=np.float64)
            else:
                tracked_occupied = _select_closed_shell_singlet_tracked_occupied_orbital(
                    solve.orbitals,
                    reference_orbital=tracked_reference_orbital,
                    grid_geometry=grid_geometry,
                )
            tracked_reference_orbital = np.asarray(tracked_occupied, dtype=np.float64)
        else:
            tracked_occupied = np.asarray(
                solve.orbitals[: occupations.n_alpha],
                dtype=np.float64,
            )
        tracked_occupied_blocks.append(tracked_occupied)
    return tuple(solves), tuple(tracked_occupied_blocks)


def _density_from_route_solve(
    *,
    solve_up: FixedPotentialEigensolverResult,
    occupations,
    grid_geometry: MonitorGridGeometry,
    occupied_up_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if _is_h2_closed_shell_singlet(occupations):
        occupied = (
            np.asarray(occupied_up_override, dtype=np.float64)
            if occupied_up_override is not None
            else np.asarray(solve_up.orbitals[:1], dtype=np.float64)
        )
        rho_up = _build_density_from_occupied_orbitals(
            occupied,
            occupations.occupations_up,
            grid_geometry=grid_geometry,
        )
        rho_down = _build_density_from_occupied_orbitals(
            occupied,
            occupations.occupations_down,
            grid_geometry=grid_geometry,
        )
        return (
            _renormalize_density(rho_up, occupations.n_alpha, grid_geometry=grid_geometry),
            _renormalize_density(rho_down, occupations.n_beta, grid_geometry=grid_geometry),
        )

    rho_up = _build_density_from_occupied_orbitals(
        solve_up.orbitals[: occupations.n_alpha],
        occupations.occupations_up,
        grid_geometry=grid_geometry,
    )
    rho_down = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    return (
        _renormalize_density(rho_up, occupations.n_alpha, grid_geometry=grid_geometry),
        _renormalize_density(rho_down, occupations.n_beta, grid_geometry=grid_geometry),
    )


def _overlap_tracking(
    *,
    previous_solve: FixedPotentialEigensolverResult,
    current_solve: FixedPotentialEigensolverResult,
    previous_occupied_orbitals: np.ndarray,
    current_occupied_orbitals: np.ndarray,
    grid_geometry: MonitorGridGeometry,
    track_lowest_two_states: bool,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    occupied_overlap_abs = float(
        abs(
            weighted_overlap_matrix(
                current_occupied_orbitals[:1],
                grid_geometry=grid_geometry,
                other=previous_occupied_orbitals[:1],
            )[0, 0]
        )
    )
    if not track_lowest_two_states:
        return occupied_overlap_abs, None, None, None, None

    current_block = np.asarray(current_solve.orbitals[:2], dtype=np.float64)
    previous_block = np.asarray(previous_solve.orbitals[:2], dtype=np.float64)
    overlap = weighted_overlap_matrix(
        current_block,
        grid_geometry=grid_geometry,
        other=previous_block,
    )
    singular_values = np.linalg.svd(np.asarray(overlap, dtype=np.float64), compute_uv=False)
    singular_values = np.clip(singular_values.astype(np.float64), 0.0, 1.0)
    min_singular = float(np.min(singular_values))
    max_angle_deg = float(np.degrees(np.arccos(min_singular)))
    current_gap = float(current_solve.eigenvalues[1] - current_solve.eigenvalues[0])
    previous_gap = float(previous_solve.eigenvalues[1] - previous_solve.eigenvalues[0])
    return (
        occupied_overlap_abs,
        min_singular,
        max_angle_deg,
        current_gap,
        float(current_gap - previous_gap),
    )


def _build_route_result(
    *,
    route_label: str,
    pair_iterations: tuple[int, int],
    previous_context: FixedPotentialStaticLocalOperatorContext,
    current_context: FixedPotentialStaticLocalOperatorContext,
    previous_baseline_solve: FixedPotentialEigensolverResult,
    baseline_current_solve: FixedPotentialEigensolverResult,
    previous_baseline_tracked_occupied: np.ndarray,
    baseline_current_tracked_occupied: np.ndarray,
    current_record,
    previous_density_residual: float | None,
    occupations,
    case: BenchmarkCase,
    track_count: int,
    track_lowest_two_states: bool,
) -> H2MonitorGridScfAmplificationReplayRoute:
    route_context = _route_context(
        route_label=route_label,
        previous_context=previous_context,
        current_context=current_context,
    )
    if route_label == "baseline":
        route_solve = baseline_current_solve
        route_tracked_occupied = np.asarray(
            baseline_current_tracked_occupied,
            dtype=np.float64,
        )
    else:
        route_solve = _solve_track_block(
            operator_context=route_context,
            k=track_count,
            case=case,
            initial_guess_orbitals=np.asarray(previous_baseline_solve.orbitals[:track_count], dtype=np.float64),
        )
        if _is_h2_closed_shell_singlet(occupations):
            active_subspace_size = min(2, track_count, int(route_solve.orbitals.shape[0]))
            if active_subspace_size >= 2:
                reference_state = ActiveSubspaceState(
                    config=ActiveSubspaceConfig(
                        enabled=True,
                        subspace_size=active_subspace_size,
                        target_occupied_count=1,
                    ),
                    reference_subspace_orbitals=np.asarray(
                        previous_baseline_solve.orbitals[:active_subspace_size],
                        dtype=np.float64,
                    ),
                    reference_occupied_orbitals=np.asarray(
                        previous_baseline_tracked_occupied[:1],
                        dtype=np.float64,
                    ),
                )
                selection = update_active_subspace(
                    raw_subspace_orbitals=np.asarray(
                        route_solve.orbitals[:active_subspace_size],
                        dtype=np.float64,
                    ),
                    state=reference_state,
                    grid_geometry=current_context.grid_geometry,
                )
                route_tracked_occupied = np.asarray(selection.occupied_orbitals, dtype=np.float64)
            else:
                route_tracked_occupied = _select_closed_shell_singlet_tracked_occupied_orbital(
                    route_solve.orbitals,
                    reference_orbital=previous_baseline_tracked_occupied,
                    grid_geometry=current_context.grid_geometry,
                )
        else:
            route_tracked_occupied = np.asarray(
                route_solve.orbitals[: occupations.n_alpha],
                dtype=np.float64,
            )
    rho_up_out, rho_down_out = _density_from_route_solve(
        solve_up=route_solve,
        occupations=occupations,
        grid_geometry=current_context.grid_geometry,
        occupied_up_override=route_tracked_occupied,
    )
    density_residual = _density_residual(
        rho_up_in=current_record.input_rho_up,
        rho_down_in=current_record.input_rho_down,
        rho_up_out=rho_up_out,
        rho_down_out=rho_down_out,
        grid_geometry=current_context.grid_geometry,
    )
    density_residual_ratio = None
    if previous_density_residual is not None and previous_density_residual > 1.0e-16:
        density_residual_ratio = float(density_residual / previous_density_residual)
    (
        hartree_norm,
        xc_norm,
        local_norm,
        hartree_share,
        xc_share,
        local_share,
    ) = _channel_shares(
        previous_context=previous_context,
        current_context=route_context,
    )
    (
        occupied_overlap_abs,
        min_singular,
        max_angle_deg,
        current_gap,
        gap_delta,
    ) = _overlap_tracking(
        previous_solve=previous_baseline_solve,
        current_solve=route_solve,
        previous_occupied_orbitals=previous_baseline_tracked_occupied,
        current_occupied_orbitals=route_tracked_occupied,
        grid_geometry=current_context.grid_geometry,
        track_lowest_two_states=track_lowest_two_states,
    )
    return H2MonitorGridScfAmplificationReplayRoute(
        route_label=route_label,
        pair_iterations=pair_iterations,
        density_residual=float(density_residual),
        density_residual_ratio=density_residual_ratio,
        delta_hartree_potential_rms=float(hartree_norm),
        delta_xc_potential_rms=float(xc_norm),
        delta_local_potential_rms=float(local_norm),
        hartree_share=hartree_share,
        xc_share=xc_share,
        local_share=local_share,
        occupied_orbital_overlap_abs=occupied_overlap_abs,
        lowest2_subspace_overlap_min_singular_value=min_singular,
        lowest2_subspace_rotation_max_angle_deg=max_angle_deg,
        lowest_gap_ha=current_gap,
        lowest_gap_delta_ha=gap_delta,
    )


def _build_spin_audit(
    *,
    spin_label: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    source_iteration_count: int,
    track_lowest_two_states: bool,
    controller_name: str,
) -> H2MonitorGridScfAmplificationSpinAudit:
    occupations = resolve_h2_spin_occupations(spin_label=spin_label, case=case)
    source_result = _shared_source_result(
        spin_label=spin_label,
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        controller_name=controller_name,
    )
    contexts = _baseline_contexts(
        source_result=source_result,
        case=case,
        grid_geometry=grid_geometry,
    )
    track_count = _k_track(occupations, track_lowest_two_states=track_lowest_two_states)
    baseline_solves, baseline_tracked_occupied = _baseline_track_solves(
        contexts=contexts,
        case=case,
        count=track_count,
        occupations=occupations,
        grid_geometry=grid_geometry,
    )
    pair_audits: list[H2MonitorGridScfAmplificationPairAudit] = []
    for index in range(1, len(source_result.history)):
        previous_record = source_result.history[index - 1]
        current_record = source_result.history[index]
        pair_iterations = (int(previous_record.iteration), int(current_record.iteration))
        previous_context = contexts[index - 1]
        current_context = contexts[index]
        previous_baseline_solve = baseline_solves[index - 1]
        baseline_current_solve = baseline_solves[index]
        previous_density_residual = float(previous_record.density_residual)
        routes = {
            route_label: _build_route_result(
                route_label=route_label,
                pair_iterations=pair_iterations,
                previous_context=previous_context,
                current_context=current_context,
                previous_baseline_solve=previous_baseline_solve,
                baseline_current_solve=baseline_current_solve,
                previous_baseline_tracked_occupied=baseline_tracked_occupied[index - 1],
                baseline_current_tracked_occupied=baseline_tracked_occupied[index],
                current_record=current_record,
                previous_density_residual=previous_density_residual,
                occupations=occupations,
                case=case,
                track_count=track_count,
                track_lowest_two_states=track_lowest_two_states,
            )
            for route_label in _ROUTE_LABELS
        }
        pair_audits.append(
            H2MonitorGridScfAmplificationPairAudit(
                pair_iterations=pair_iterations,
                baseline=routes["baseline"],
                freeze_hartree=routes["freeze_hartree"],
                freeze_xc=routes["freeze_xc"],
            )
        )
    return H2MonitorGridScfAmplificationSpinAudit(
        spin_state_label=spin_label,
        source_iteration_count=int(source_iteration_count),
        density_residual_history=tuple(float(value) for value in source_result.density_residual_history),
        pair_audits=tuple(pair_audits),
    )


def run_h2_monitor_grid_scf_amplification_ablation_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    track_lowest_two_states: bool = True,
    controller_name: str = "baseline_linear",
) -> H2MonitorGridScfAmplificationAblationAuditResult:
    """Run a lightweight early-step baseline/freeze-channel replay audit."""

    if grid_geometry is None:
        grid_geometry = build_monitor_grid_for_case(
            case,
            shape=_DEFAULT_SHAPE,
            box_half_extents=_DEFAULT_BOX_HALF_EXTENTS_BOHR,
            element_parameters=build_h2_local_patch_development_element_parameters(),
        )
    bounds = grid_geometry.spec.box_bounds
    box_half_extents_bohr = (
        0.5 * float(bounds[0][1] - bounds[0][0]),
        0.5 * float(bounds[1][1] - bounds[1][0]),
        0.5 * float(bounds[2][1] - bounds[2][0]),
    )
    singlet = _build_spin_audit(
        spin_label="singlet",
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        track_lowest_two_states=track_lowest_two_states,
        controller_name=controller_name,
    )
    triplet = _build_spin_audit(
        spin_label="triplet",
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        track_lowest_two_states=track_lowest_two_states,
        controller_name=controller_name,
    )
    return H2MonitorGridScfAmplificationAblationAuditResult(
        case_name=case.name,
        grid_parameter_summary=(
            f"shape={grid_geometry.spec.shape}, "
            f"box_half_extents_bohr={box_half_extents_bohr}"
        ),
        singlet=singlet,
        triplet=triplet,
        note=(
            "This audit stays on the same small H2 monitor-grid local-only case and replays only "
            "the first 2-3 source input-density snapshots. It compares the baseline fixed-point map "
            "to freeze-Hartree and freeze-XC replays, and optionally tracks the lowest-two-state "
            "subspace to separate channel feedback from low-state orbital/subspace rotation. "
            f"controller={controller_name}"
        ),
    )


def _print_route(route: H2MonitorGridScfAmplificationReplayRoute) -> None:
    print(
        f"    {route.route_label}: "
        f"density_residual={route.density_residual:.12e}, "
        f"residual_ratio={route.density_residual_ratio}, "
        f"delta_vH={route.delta_hartree_potential_rms:.12e}, "
        f"delta_vXC={route.delta_xc_potential_rms:.12e}, "
        f"delta_vLocal={route.delta_local_potential_rms:.12e}, "
        f"shares=({route.hartree_share}, {route.xc_share}, {route.local_share}), "
        f"overlap={route.occupied_orbital_overlap_abs}, "
        f"subspace_rot_deg={route.lowest2_subspace_rotation_max_angle_deg}, "
        f"gap={route.lowest_gap_ha}, gap_delta={route.lowest_gap_delta_ha}"
    )


def _print_spin_audit(audit: H2MonitorGridScfAmplificationSpinAudit) -> None:
    print(f"spin: {audit.spin_state_label}")
    print(f"  source density residual history: {audit.density_residual_history}")
    for pair_audit in audit.pair_audits:
        print(f"  pair {pair_audit.pair_iterations[0]}->{pair_audit.pair_iterations[1]}")
        _print_route(pair_audit.baseline)
        _print_route(pair_audit.freeze_hartree)
        _print_route(pair_audit.freeze_xc)


def print_h2_monitor_grid_scf_amplification_ablation_summary(
    result: H2MonitorGridScfAmplificationAblationAuditResult,
) -> None:
    """Print the compact early-step SCF amplification/channel-ablation summary."""

    print("IsoGridDFT H2 monitor-grid SCF amplification/channel-ablation audit")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"note: {result.note}")
    print()
    _print_spin_audit(result.singlet)
    print()
    _print_spin_audit(result.triplet)


def main() -> int:
    result = run_h2_monitor_grid_scf_amplification_ablation_audit()
    print_h2_monitor_grid_scf_amplification_ablation_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
