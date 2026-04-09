"""Audit lowest-state eigenvalue / occupation sensitivity on the plateau mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case

from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _DEFAULT_BOX_HALF_EXTENTS_BOHR,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _DEFAULT_LATE_WINDOW_SIZE,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _DEFAULT_PROBE_ITERATION,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _DEFAULT_SHAPE,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _DEFAULT_SOURCE_ITERATION_COUNT,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    H2MonitorGridPlateauModeEffectivePotentialToOccupiedDensityResponseAuditResult,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    run_h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit,
)


@dataclass(frozen=True)
class H2MonitorGridPlateauModeLowestStateEigenvalueOccupationSensitivityAuditResult:
    """Signed lowest-state spectral sensitivity estimate along the plateau mode."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    probe_iteration: int
    principal_mode_explained_fraction: float
    effective_potential_mode_norm: float
    lowest_eigenvalue_signed_gain: float
    second_eigenvalue_signed_gain: float
    lowest_gap_signed_gain: float
    occupied_density_signed_gain: float
    occupied_overlap_abs_min: float | None
    lowest2_subspace_rotation_max_angle_deg: float | None
    lowest_gap_delta_abs_max_ha: float | None
    spectral_sensitivity_regime: str
    verdict: str


def _spectral_regime(*, gap_gain: float, occupied_gain: float, rotation_deg: float | None) -> tuple[str, str]:
    if abs(gap_gain) <= 1.0e-3 and abs(occupied_gain) <= 1.0e-3 and (rotation_deg is None or rotation_deg <= 1.0):
        return (
            "weak",
            "The plateau mode is spectrally weak at the lowest two states; occupied-density insensitivity is consistent with weak eigenvalue/gap response.",
        )
    if abs(gap_gain) >= 1.0e-2 and abs(occupied_gain) <= 1.0e-3:
        return (
            "mixed",
            "The lowest-state spectrum responds more strongly than the occupied density, so the plateau mode is not explained by a purely weak spectral response.",
        )
    return (
        "strong",
        "The plateau mode shows non-negligible lowest-state spectral sensitivity and should not be treated as purely occupation-insensitive.",
    )


def run_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    probe_iteration: int = _DEFAULT_PROBE_ITERATION,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
    singlet_experimental_route_name: str = "none",
) -> H2MonitorGridPlateauModeLowestStateEigenvalueOccupationSensitivityAuditResult:
    """Estimate whether the plateau mode is spectrally weak at the lowest two states."""

    if grid_geometry is None:
        grid_geometry = build_monitor_grid_for_case(
            case,
            shape=_DEFAULT_SHAPE,
            box_half_extents=_DEFAULT_BOX_HALF_EXTENTS_BOHR,
            element_parameters=build_h2_local_patch_development_element_parameters(),
        )

    from .h2_monitor_grid_local_linear_response_audit import _build_context
    from .h2_monitor_grid_plateau_mode_effective_potential_orbital_response_audit import _principal_mode
    from .h2_monitor_grid_plateau_mode_effective_potential_orbital_response_audit import _weighted_inner
    from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
        _charge_residual_field,
    )
    from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
        _grid_parameter_summary,
    )
    from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
        _track_solves_with_optional_projector_route,
    )
    from .h2_monitor_grid_scf_amplification_ablation_audit import _k_track
    from .h2_monitor_grid_scf_amplification_ablation_audit import _overlap_tracking
    from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result
    from isogrid.ops import weighted_l2_norm
    from isogrid.scf import resolve_h2_spin_occupations

    source_result = _shared_source_result(
        spin_label=spin_label,
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        controller_name=controller_name,
        singlet_experimental_route_name=singlet_experimental_route_name,
    )
    if len(source_result.history) < 2:
        raise ValueError("Lowest-state eigenvalue sensitivity audit requires at least two SCF iterations.")

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
    effective_potential_mode_norm = weighted_l2_norm(effective_delta, grid_geometry=grid_geometry)
    if effective_potential_mode_norm <= 1.0e-16:
        raise ValueError("Plateau-aligned effective-potential mode has near-zero weighted norm.")
    effective_mode = np.asarray(effective_delta / effective_potential_mode_norm, dtype=np.float64)

    # Reuse the established occupied-density response audit to keep the perturbation semantics aligned.
    occupied_response: H2MonitorGridPlateauModeEffectivePotentialToOccupiedDensityResponseAuditResult = (
        run_h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit(
            case=case,
            grid_geometry=grid_geometry,
            spin_label=spin_label,
            source_iteration_count=source_iteration_count,
            probe_iteration=probe_iteration,
            late_window_size=late_window_size,
            controller_name=controller_name,
            singlet_experimental_route_name=singlet_experimental_route_name,
        )
    )

    amplitude = 1.0e-3
    from dataclasses import replace

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
    baseline_pos_solves, baseline_pos_tracked, _ = _track_solves_with_optional_projector_route(
        contexts=(baseline_context, pos_context),
        case=case,
        count=track_count,
        occupations=occupations,
        grid_geometry=grid_geometry,
        singlet_experimental_route_name=singlet_experimental_route_name,
    )
    baseline_neg_solves, baseline_neg_tracked, _ = _track_solves_with_optional_projector_route(
        contexts=(baseline_context, neg_context),
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

    lowest_eigenvalue_signed_gain = float((pos_solve.eigenvalues[0] - neg_solve.eigenvalues[0]) / (2.0 * amplitude))
    second_eigenvalue_signed_gain = float((pos_solve.eigenvalues[1] - neg_solve.eigenvalues[1]) / (2.0 * amplitude))
    pos_gap = float(pos_solve.eigenvalues[1] - pos_solve.eigenvalues[0])
    neg_gap = float(neg_solve.eigenvalues[1] - neg_solve.eigenvalues[0])
    lowest_gap_signed_gain = float((pos_gap - neg_gap) / (2.0 * amplitude))

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

    spectral_sensitivity_regime, verdict = _spectral_regime(
        gap_gain=lowest_gap_signed_gain,
        occupied_gain=occupied_response.occupied_density_signed_gain,
        rotation_deg=float(max(angle_candidates)) if angle_candidates else None,
    )

    return H2MonitorGridPlateauModeLowestStateEigenvalueOccupationSensitivityAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=int(source_iteration_count),
        probe_iteration=int(probe_iteration),
        principal_mode_explained_fraction=explained_fraction,
        effective_potential_mode_norm=float(effective_potential_mode_norm),
        lowest_eigenvalue_signed_gain=lowest_eigenvalue_signed_gain,
        second_eigenvalue_signed_gain=second_eigenvalue_signed_gain,
        lowest_gap_signed_gain=lowest_gap_signed_gain,
        occupied_density_signed_gain=occupied_response.occupied_density_signed_gain,
        occupied_overlap_abs_min=float(min(overlap_candidates)) if overlap_candidates else None,
        lowest2_subspace_rotation_max_angle_deg=float(max(angle_candidates)) if angle_candidates else None,
        lowest_gap_delta_abs_max_ha=float(max(gap_delta_candidates)) if gap_delta_candidates else None,
        spectral_sensitivity_regime=spectral_sensitivity_regime,
        verdict=verdict,
    )


def print_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_summary(
    result: H2MonitorGridPlateauModeLowestStateEigenvalueOccupationSensitivityAuditResult,
) -> None:
    """Print a compact summary for the plateau-mode lowest-state spectral sensitivity audit."""

    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin_state: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    print(f"source_iteration_count: {result.source_iteration_count}")
    print(f"probe_iteration: {result.probe_iteration}")
    print(f"principal_mode_explained_fraction: {result.principal_mode_explained_fraction:.6e}")
    print(f"effective_potential_mode_norm: {result.effective_potential_mode_norm:.6e}")
    print(f"lowest_eigenvalue_signed_gain: {result.lowest_eigenvalue_signed_gain:.6e}")
    print(f"second_eigenvalue_signed_gain: {result.second_eigenvalue_signed_gain:.6e}")
    print(f"lowest_gap_signed_gain: {result.lowest_gap_signed_gain:.6e}")
    print(f"occupied_density_signed_gain: {result.occupied_density_signed_gain:.6e}")
    print(f"occupied_overlap_abs_min: {result.occupied_overlap_abs_min}")
    print(f"lowest2_subspace_rotation_max_angle_deg: {result.lowest2_subspace_rotation_max_angle_deg}")
    print(f"lowest_gap_delta_abs_max_ha: {result.lowest_gap_delta_abs_max_ha}")
    print(f"spectral_sensitivity_regime: {result.spectral_sensitivity_regime}")
    print(f"verdict: {result.verdict}")


if __name__ == "__main__":
    grid = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=_DEFAULT_SHAPE,
        box_half_extents=_DEFAULT_BOX_HALF_EXTENTS_BOHR,
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    print_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_summary(
        run_h2_monitor_grid_plateau_mode_lowest_state_eigenvalue_occupation_sensitivity_audit(
            case=H2_BENCHMARK_CASE,
            grid_geometry=grid,
        )
    )
