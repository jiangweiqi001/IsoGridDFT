"""Audit the effective-potential -> occupied-density response on the plateau mode."""

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
from isogrid.ops import weighted_l2_norm
from isogrid.scf import resolve_h2_spin_occupations
from isogrid.scf.projector_route import ProjectorRouteConfig
from isogrid.scf.projector_route import ProjectorRouteSelectionResult
from isogrid.scf.projector_route import ProjectorRouteState
from isogrid.scf.projector_route import initialize_projector_route
from isogrid.scf.projector_route import rebuild_density_from_projector_route
from isogrid.scf.projector_route import update_projector_route

from .h2_monitor_grid_local_linear_response_audit import _build_context
from .h2_monitor_grid_local_linear_response_audit import _density_from_tracked_solve
from .h2_monitor_grid_plateau_mode_effective_potential_orbital_response_audit import (
    _grid_parameter_summary,
)
from .h2_monitor_grid_plateau_mode_effective_potential_orbital_response_audit import (
    _principal_mode,
)
from .h2_monitor_grid_plateau_mode_effective_potential_orbital_response_audit import (
    _weighted_inner,
)
from .h2_monitor_grid_scf_amplification_ablation_audit import _build_initial_track_guess
from .h2_monitor_grid_scf_amplification_ablation_audit import _baseline_track_solves
from .h2_monitor_grid_scf_amplification_ablation_audit import _k_track
from .h2_monitor_grid_scf_amplification_ablation_audit import _overlap_tracking
from .h2_monitor_grid_scf_amplification_ablation_audit import _solve_track_block
from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result

_DEFAULT_SHAPE = (15, 15, 17)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (9.0, 9.0, 11.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 12
_DEFAULT_PROBE_ITERATION = 12
_DEFAULT_LATE_WINDOW_SIZE = 5
_DEFAULT_EFFECTIVE_MODE_AMPLITUDE = 1.0e-3
_DEFAULT_PROJECTOR_ROUTE_MIXING = 0.20


@dataclass(frozen=True)
class H2MonitorGridPlateauModeEffectivePotentialToOccupiedDensityResponseAuditResult:
    """Signed response estimate for effective-potential perturbations on the plateau mode."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    probe_iteration: int
    principal_mode_explained_fraction: float
    effective_potential_mode_norm: float
    effective_potential_to_density_mode_alignment: float | None
    occupied_density_signed_gain: float
    occupied_density_sign_regime: str
    occupied_overlap_abs_min: float | None
    lowest2_subspace_rotation_max_angle_deg: float | None
    lowest_gap_delta_abs_max_ha: float | None
    verdict: str


def _charge_residual_field(record) -> np.ndarray:
    return np.asarray(
        (record.output_rho_up + record.output_rho_down)
        - (record.input_rho_up + record.input_rho_down),
        dtype=np.float64,
    )


def _track_solves_with_optional_projector_route(
    *,
    contexts,
    case: BenchmarkCase,
    count: int,
    occupations,
    grid_geometry: MonitorGridGeometry,
    singlet_experimental_route_name: str,
) -> tuple[
    tuple[FixedPotentialEigensolverResult, ...],
    tuple[np.ndarray, ...],
    tuple[ProjectorRouteSelectionResult | None, ...],
]:
    route_name = singlet_experimental_route_name.strip().lower()
    if route_name != "projector_mixing":
        solves, tracked = _baseline_track_solves(
            contexts=contexts,
            case=case,
            count=count,
            occupations=occupations,
            grid_geometry=grid_geometry,
        )
        return solves, tracked, tuple(None for _ in solves)

    solves: list[FixedPotentialEigensolverResult] = []
    tracked_occupied_blocks: list[np.ndarray] = []
    selections: list[ProjectorRouteSelectionResult | None] = []
    initial_guess = _build_initial_track_guess(
        case=case,
        grid_geometry=contexts[0].grid_geometry,
        count=count,
    )
    projector_route_state: ProjectorRouteState | None = None
    projector_route_config = ProjectorRouteConfig.local_only_h2_singlet_default(
        projector_mixing=_DEFAULT_PROJECTOR_ROUTE_MIXING
    )
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
        if projector_route_state is None:
            selection = initialize_projector_route(
                raw_subspace_orbitals=np.asarray(
                    solve.orbitals[: projector_route_config.active_subspace_size],
                    dtype=np.float64,
                ),
                grid_geometry=grid_geometry,
                config=projector_route_config,
            )
        else:
            selection = update_projector_route(
                raw_subspace_orbitals=np.asarray(
                    solve.orbitals[: projector_route_state.config.active_subspace_size],
                    dtype=np.float64,
                ),
                state=projector_route_state,
                grid_geometry=grid_geometry,
            )
        projector_route_state = selection.state
        selections.append(selection)
        tracked_occupied_blocks.append(np.asarray(selection.occupied_orbitals, dtype=np.float64))
    return tuple(solves), tuple(tracked_occupied_blocks), tuple(selections)


def _density_from_selection_or_tracked_solve(
    *,
    solve_up,
    tracked_occupied_orbitals: np.ndarray,
    projector_selection: ProjectorRouteSelectionResult | None,
    occupations,
    grid_geometry: MonitorGridGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    if projector_selection is not None:
        return rebuild_density_from_projector_route(
            selection=projector_selection,
            occupations=occupations,
            grid_geometry=grid_geometry,
        )
    return _density_from_tracked_solve(
        solve_up=solve_up,
        tracked_occupied_orbitals=tracked_occupied_orbitals,
        occupations=occupations,
        grid_geometry=grid_geometry,
    )


def run_h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    probe_iteration: int = _DEFAULT_PROBE_ITERATION,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
    singlet_experimental_route_name: str = "none",
) -> H2MonitorGridPlateauModeEffectivePotentialToOccupiedDensityResponseAuditResult:
    """Estimate the occupied-density response to a plateau-aligned effective-potential mode."""

    if grid_geometry is None:
        grid_geometry = build_monitor_grid_for_case(
            case,
            shape=_DEFAULT_SHAPE,
            box_half_extents=_DEFAULT_BOX_HALF_EXTENTS_BOHR,
            element_parameters=build_h2_local_patch_development_element_parameters(),
        )
    source_result = _shared_source_result(
        spin_label=spin_label,
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        controller_name=controller_name,
        singlet_experimental_route_name=singlet_experimental_route_name,
    )
    if len(source_result.history) < 2:
        raise ValueError(
            "Effective-potential -> occupied-density response audit requires at least two SCF iterations."
        )

    occupations = resolve_h2_spin_occupations(spin_label=spin_label, case=case)
    window_size = max(2, min(int(late_window_size), len(source_result.history)))
    residual_fields = tuple(_charge_residual_field(record) for record in source_result.history[-window_size:])
    mode, explained_fraction = _principal_mode(
        residual_fields=residual_fields,
        grid_geometry=grid_geometry,
    )
    residual_coefficients = [
        _weighted_inner(field, mode, grid_geometry=grid_geometry) for field in residual_fields
    ]
    if residual_coefficients[-1] < 0.0:
        mode = -mode

    probe_index = max(0, min(int(probe_iteration) - 1, len(source_result.history) - 1))
    probe_record = source_result.history[probe_index]
    baseline_context = _build_context(
        case=case,
        grid_geometry=grid_geometry,
        rho_up=probe_record.input_rho_up,
        rho_down=probe_record.input_rho_down,
    )
    mixed_context = _build_context(
        case=case,
        grid_geometry=grid_geometry,
        rho_up=probe_record.mixed_rho_up,
        rho_down=probe_record.mixed_rho_down,
    )
    effective_delta = np.asarray(
        mixed_context.effective_local_potential - baseline_context.effective_local_potential,
        dtype=np.float64,
    )
    effective_mode_norm = weighted_l2_norm(effective_delta, grid_geometry=grid_geometry)
    if effective_mode_norm <= 1.0e-16:
        raise ValueError("Plateau-aligned effective-potential mode has near-zero weighted norm.")
    effective_mode = np.asarray(effective_delta / effective_mode_norm, dtype=np.float64)
    effective_mode_alignment = float(
        _weighted_inner(effective_mode, mode, grid_geometry=grid_geometry)
    )

    amplitude = float(_DEFAULT_EFFECTIVE_MODE_AMPLITUDE)
    pos_context = replace(
        baseline_context,
        effective_local_potential=np.asarray(
            baseline_context.effective_local_potential + amplitude * effective_mode,
            dtype=np.float64,
        ),
    )
    neg_context = replace(
        baseline_context,
        effective_local_potential=np.asarray(
            baseline_context.effective_local_potential - amplitude * effective_mode,
            dtype=np.float64,
        ),
    )

    track_count = _k_track(occupations, track_lowest_two_states=True)
    baseline_neg_solves, baseline_neg_tracked, baseline_neg_selections = _track_solves_with_optional_projector_route(
        contexts=(baseline_context, neg_context),
        case=case,
        count=track_count,
        occupations=occupations,
        grid_geometry=grid_geometry,
        singlet_experimental_route_name=singlet_experimental_route_name,
    )
    baseline_pos_solves, baseline_pos_tracked, baseline_pos_selections = _track_solves_with_optional_projector_route(
        contexts=(baseline_context, pos_context),
        case=case,
        count=track_count,
        occupations=occupations,
        grid_geometry=grid_geometry,
        singlet_experimental_route_name=singlet_experimental_route_name,
    )

    baseline_solve = baseline_pos_solves[0]
    baseline_tracked = baseline_pos_tracked[0]
    pos_solve = baseline_pos_solves[1]
    pos_tracked = baseline_pos_tracked[1]
    neg_solve = baseline_neg_solves[1]
    neg_tracked = baseline_neg_tracked[1]

    pos_rho_up_out, pos_rho_down_out = _density_from_selection_or_tracked_solve(
        solve_up=pos_solve,
        tracked_occupied_orbitals=pos_tracked,
        projector_selection=baseline_pos_selections[1],
        occupations=occupations,
        grid_geometry=grid_geometry,
    )
    neg_rho_up_out, neg_rho_down_out = _density_from_selection_or_tracked_solve(
        solve_up=neg_solve,
        tracked_occupied_orbitals=neg_tracked,
        projector_selection=baseline_neg_selections[1],
        occupations=occupations,
        grid_geometry=grid_geometry,
    )
    output_delta = 0.5 * (
        np.asarray(pos_rho_up_out + pos_rho_down_out, dtype=np.float64)
        - np.asarray(neg_rho_up_out + neg_rho_down_out, dtype=np.float64)
    )
    output_mode_coefficient = _weighted_inner(output_delta, mode, grid_geometry=grid_geometry)
    occupied_density_signed_gain = float(output_mode_coefficient / amplitude)

    if occupied_density_signed_gain > 0.05:
        occupied_density_sign_regime = "positive"
        verdict = "The occupied-density response follows the plateau mode with positive signed gain."
    elif occupied_density_signed_gain < -0.05:
        occupied_density_sign_regime = "negative"
        verdict = "The occupied-density response flips sign along the plateau mode."
    else:
        occupied_density_sign_regime = "near_zero"
        verdict = "The occupied-density response is nearly neutral along the plateau mode."

    pos_overlap_abs, _, pos_angle_deg, _, pos_gap_delta = _overlap_tracking(
        previous_solve=baseline_solve,
        current_solve=pos_solve,
        previous_occupied_orbitals=baseline_tracked,
        current_occupied_orbitals=pos_tracked,
        grid_geometry=grid_geometry,
        track_lowest_two_states=True,
    )
    neg_overlap_abs, _, neg_angle_deg, _, neg_gap_delta = _overlap_tracking(
        previous_solve=baseline_solve,
        current_solve=neg_solve,
        previous_occupied_orbitals=baseline_tracked,
        current_occupied_orbitals=neg_tracked,
        grid_geometry=grid_geometry,
        track_lowest_two_states=True,
    )
    overlap_candidates = [value for value in (pos_overlap_abs, neg_overlap_abs) if value is not None]
    angle_candidates = [value for value in (pos_angle_deg, neg_angle_deg) if value is not None]
    gap_delta_candidates = [abs(value) for value in (pos_gap_delta, neg_gap_delta) if value is not None]

    return H2MonitorGridPlateauModeEffectivePotentialToOccupiedDensityResponseAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=int(source_iteration_count),
        probe_iteration=int(probe_iteration),
        principal_mode_explained_fraction=explained_fraction,
        effective_potential_mode_norm=float(effective_mode_norm),
        effective_potential_to_density_mode_alignment=effective_mode_alignment,
        occupied_density_signed_gain=occupied_density_signed_gain,
        occupied_density_sign_regime=occupied_density_sign_regime,
        occupied_overlap_abs_min=(
            float(min(overlap_candidates)) if overlap_candidates else None
        ),
        lowest2_subspace_rotation_max_angle_deg=(
            float(max(angle_candidates)) if angle_candidates else None
        ),
        lowest_gap_delta_abs_max_ha=(
            float(max(gap_delta_candidates)) if gap_delta_candidates else None
        ),
        verdict=verdict,
    )


def print_h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_summary(
    result: H2MonitorGridPlateauModeEffectivePotentialToOccupiedDensityResponseAuditResult,
) -> None:
    """Print a compact summary for the effective-potential -> occupied-density audit."""

    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin_state: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    print(f"source_iteration_count: {result.source_iteration_count}")
    print(f"probe_iteration: {result.probe_iteration}")
    print(f"principal_mode_explained_fraction: {result.principal_mode_explained_fraction:.6e}")
    print(f"effective_potential_mode_norm: {result.effective_potential_mode_norm:.6e}")
    print(
        "effective_potential_to_density_mode_alignment: "
        f"{result.effective_potential_to_density_mode_alignment:.6e}"
    )
    print(f"occupied_density_signed_gain: {result.occupied_density_signed_gain:.6e}")
    print(f"occupied_density_sign_regime: {result.occupied_density_sign_regime}")
    print(f"occupied_overlap_abs_min: {result.occupied_overlap_abs_min}")
    print(f"lowest2_subspace_rotation_max_angle_deg: {result.lowest2_subspace_rotation_max_angle_deg}")
    print(f"lowest_gap_delta_abs_max_ha: {result.lowest_gap_delta_abs_max_ha}")
    print(f"verdict: {result.verdict}")


if __name__ == "__main__":
    grid = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=_DEFAULT_SHAPE,
        box_half_extents=_DEFAULT_BOX_HALF_EXTENTS_BOHR,
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    print_h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_summary(
        run_h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit(
            case=H2_BENCHMARK_CASE,
            grid_geometry=grid,
        )
    )
