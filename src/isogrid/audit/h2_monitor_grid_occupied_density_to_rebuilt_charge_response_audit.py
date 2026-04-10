"""Audit occupied-density to rebuilt-charge response on the plateau mode."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ops import weighted_l2_norm
from isogrid.scf import resolve_h2_spin_occupations
from isogrid.scf.driver import _build_density_from_occupied_orbitals

from .h2_monitor_grid_plateau_mode_effective_potential_orbital_response_audit import (
    _grid_parameter_summary,
)
from .h2_monitor_grid_plateau_mode_effective_potential_orbital_response_audit import (
    _principal_mode,
)
from .h2_monitor_grid_plateau_mode_effective_potential_orbital_response_audit import (
    _weighted_inner,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _charge_residual_field,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _DEFAULT_BOX_HALF_EXTENTS_BOHR,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _DEFAULT_EFFECTIVE_MODE_AMPLITUDE,
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
    _density_from_selection_or_tracked_solve,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _track_solves_with_optional_projector_route,
)
from .h2_monitor_grid_scf_amplification_ablation_audit import _k_track
from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result

_STAGE_LOSS_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class H2MonitorGridOccupiedDensityToRebuiltChargeResponseAuditResult:
    """Signed response diagnostics from occupied densities to rebuilt charge density."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    probe_iteration: int
    principal_mode_explained_fraction: float
    effective_potential_mode_norm: float
    occupied_density_signed_gain: float
    rebuilt_charge_density_signed_gain: float
    renormalized_residual_signed_gain: float
    occupied_to_rebuilt_stage_loss: float
    rebuilt_to_residual_stage_loss: float
    rebuilt_charge_sign_regime: str
    verdict: str


def _baseline_contexts_for_probe(
    *,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    probe_record,
) -> tuple:
    from .h2_monitor_grid_local_linear_response_audit import _build_context

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
    return baseline_context, mixed_context


def _regime(value: float) -> str:
    if value > 0.05:
        return "positive"
    if value < -0.05:
        return "negative"
    return "near_zero"


def run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    probe_iteration: int = _DEFAULT_PROBE_ITERATION,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
    singlet_experimental_route_name: str = "none",
) -> H2MonitorGridOccupiedDensityToRebuiltChargeResponseAuditResult:
    """Audit whether density reconstruction suppresses the plateau-mode response."""

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
            "Occupied-density -> rebuilt-charge audit requires at least two SCF iterations."
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
    baseline_context, mixed_context = _baseline_contexts_for_probe(
        case=case,
        grid_geometry=grid_geometry,
        probe_record=probe_record,
    )
    effective_delta = np.asarray(
        mixed_context.effective_local_potential - baseline_context.effective_local_potential,
        dtype=np.float64,
    )
    effective_potential_mode_norm = weighted_l2_norm(effective_delta, grid_geometry=grid_geometry)
    if effective_potential_mode_norm <= 1.0e-16:
        raise ValueError("Plateau-aligned effective-potential mode has near-zero weighted norm.")
    effective_mode = np.asarray(effective_delta / effective_potential_mode_norm, dtype=np.float64)
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
    pos_solves, pos_tracked_blocks, pos_selections = _track_solves_with_optional_projector_route(
        contexts=(baseline_context, pos_context),
        case=case,
        count=track_count,
        occupations=occupations,
        grid_geometry=grid_geometry,
        singlet_experimental_route_name=singlet_experimental_route_name,
    )
    neg_solves, neg_tracked_blocks, neg_selections = _track_solves_with_optional_projector_route(
        contexts=(baseline_context, neg_context),
        case=case,
        count=track_count,
        occupations=occupations,
        grid_geometry=grid_geometry,
        singlet_experimental_route_name=singlet_experimental_route_name,
    )
    pos_tracked = pos_tracked_blocks[1]
    neg_tracked = neg_tracked_blocks[1]

    pos_raw_up = _build_density_from_occupied_orbitals(
        pos_tracked,
        occupations.occupations_up,
        grid_geometry=grid_geometry,
    )
    pos_raw_down = _build_density_from_occupied_orbitals(
        pos_tracked,
        occupations.occupations_down,
        grid_geometry=grid_geometry,
    )
    neg_raw_up = _build_density_from_occupied_orbitals(
        neg_tracked,
        occupations.occupations_up,
        grid_geometry=grid_geometry,
    )
    neg_raw_down = _build_density_from_occupied_orbitals(
        neg_tracked,
        occupations.occupations_down,
        grid_geometry=grid_geometry,
    )
    occupied_density_delta = 0.5 * (
        np.asarray(pos_raw_up + pos_raw_down, dtype=np.float64)
        - np.asarray(neg_raw_up + neg_raw_down, dtype=np.float64)
    )

    pos_rebuilt_up, pos_rebuilt_down = _density_from_selection_or_tracked_solve(
        solve_up=pos_solves[1],
        tracked_occupied_orbitals=pos_tracked,
        projector_selection=pos_selections[1],
        occupations=occupations,
        grid_geometry=grid_geometry,
    )
    neg_rebuilt_up, neg_rebuilt_down = _density_from_selection_or_tracked_solve(
        solve_up=neg_solves[1],
        tracked_occupied_orbitals=neg_tracked,
        projector_selection=neg_selections[1],
        occupations=occupations,
        grid_geometry=grid_geometry,
    )
    rebuilt_charge_delta = 0.5 * (
        np.asarray(pos_rebuilt_up + pos_rebuilt_down, dtype=np.float64)
        - np.asarray(neg_rebuilt_up + neg_rebuilt_down, dtype=np.float64)
    )
    residual_delta = np.asarray(rebuilt_charge_delta, dtype=np.float64)

    occupied_density_signed_gain = float(
        _weighted_inner(occupied_density_delta, mode, grid_geometry=grid_geometry) / amplitude
    )
    rebuilt_charge_density_signed_gain = float(
        _weighted_inner(rebuilt_charge_delta, mode, grid_geometry=grid_geometry) / amplitude
    )
    renormalized_residual_signed_gain = float(
        _weighted_inner(residual_delta, mode, grid_geometry=grid_geometry) / amplitude
    )
    occupied_to_rebuilt_stage_loss = float(
        rebuilt_charge_density_signed_gain - occupied_density_signed_gain
    )
    rebuilt_to_residual_stage_loss = float(
        renormalized_residual_signed_gain - rebuilt_charge_density_signed_gain
    )
    rebuilt_charge_sign_regime = _regime(rebuilt_charge_density_signed_gain)

    max_stage_loss = max(abs(occupied_to_rebuilt_stage_loss), abs(rebuilt_to_residual_stage_loss))
    if max_stage_loss <= _STAGE_LOSS_TOLERANCE:
        verdict = "The rebuilt charge density remains closely aligned with the occupied-density response."
    elif abs(occupied_to_rebuilt_stage_loss) >= abs(rebuilt_to_residual_stage_loss):
        verdict = "The occupied-density response is materially altered by charge-density rebuilding."
    else:
        verdict = "The renormalized residual response materially departs from the rebuilt charge density."

    return H2MonitorGridOccupiedDensityToRebuiltChargeResponseAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=int(source_iteration_count),
        probe_iteration=int(probe_iteration),
        principal_mode_explained_fraction=explained_fraction,
        effective_potential_mode_norm=float(effective_potential_mode_norm),
        occupied_density_signed_gain=occupied_density_signed_gain,
        rebuilt_charge_density_signed_gain=rebuilt_charge_density_signed_gain,
        renormalized_residual_signed_gain=renormalized_residual_signed_gain,
        occupied_to_rebuilt_stage_loss=occupied_to_rebuilt_stage_loss,
        rebuilt_to_residual_stage_loss=rebuilt_to_residual_stage_loss,
        rebuilt_charge_sign_regime=rebuilt_charge_sign_regime,
        verdict=verdict,
    )


def print_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_summary(
    result: H2MonitorGridOccupiedDensityToRebuiltChargeResponseAuditResult,
) -> None:
    """Print a compact summary for the occupied-density -> rebuilt-charge audit."""

    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin_state: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    print(f"source_iteration_count: {result.source_iteration_count}")
    print(f"probe_iteration: {result.probe_iteration}")
    print(f"principal_mode_explained_fraction: {result.principal_mode_explained_fraction:.6e}")
    print(f"effective_potential_mode_norm: {result.effective_potential_mode_norm:.6e}")
    print(f"occupied_density_signed_gain: {result.occupied_density_signed_gain:.6e}")
    print(f"rebuilt_charge_density_signed_gain: {result.rebuilt_charge_density_signed_gain:.6e}")
    print(f"renormalized_residual_signed_gain: {result.renormalized_residual_signed_gain:.6e}")
    print(f"occupied_to_rebuilt_stage_loss: {result.occupied_to_rebuilt_stage_loss:.6e}")
    print(f"rebuilt_to_residual_stage_loss: {result.rebuilt_to_residual_stage_loss:.6e}")
    print(f"rebuilt_charge_sign_regime: {result.rebuilt_charge_sign_regime}")
    print(f"verdict: {result.verdict}")


if __name__ == "__main__":
    grid = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=_DEFAULT_SHAPE,
        box_half_extents=_DEFAULT_BOX_HALF_EXTENTS_BOHR,
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    print_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_summary(
        run_h2_monitor_grid_occupied_density_to_rebuilt_charge_response_audit(
            case=H2_BENCHMARK_CASE,
            grid_geometry=grid,
        )
    )
