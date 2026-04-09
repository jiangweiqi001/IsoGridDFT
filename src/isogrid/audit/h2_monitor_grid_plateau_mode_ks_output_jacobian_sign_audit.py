"""Signed Jacobian audit of the KS output map along the plateau mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ops.kinetic import weighted_l2_norm
from isogrid.scf import resolve_h2_spin_occupations

from .h2_monitor_grid_local_linear_response_audit import _build_context
from .h2_monitor_grid_local_linear_response_audit import _density_from_tracked_solve
from .h2_monitor_grid_scf_amplification_ablation_audit import _baseline_track_solves
from .h2_monitor_grid_scf_amplification_ablation_audit import _k_track
from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result
from .h2_monitor_grid_scf_amplification_ablation_audit import _weighted_field_norm

_DEFAULT_SHAPE = (15, 15, 17)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (9.0, 9.0, 11.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 12
_DEFAULT_PROBE_ITERATION = 12
_DEFAULT_MODE_AMPLITUDE = 0.01


@dataclass(frozen=True)
class H2MonitorGridPlateauModeKsOutputJacobianSignAuditResult:
    """Signed one-step Jacobian estimate along the dominant plateau mode."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    probe_iteration: int
    principal_mode_explained_fraction: float
    input_mode_norm: float
    effective_potential_signed_gain: float
    output_density_signed_gain: float
    output_density_sign_regime: str
    verdict: str


def _charge_density(record) -> np.ndarray:
    return np.asarray(record.input_rho_up + record.input_rho_down, dtype=np.float64)


def _charge_residual_field(record) -> np.ndarray:
    return np.asarray(
        (record.output_rho_up + record.output_rho_down)
        - (record.input_rho_up + record.input_rho_down),
        dtype=np.float64,
    )


def _weighted_inner(
    field_a: np.ndarray,
    field_b: np.ndarray,
    *,
    grid_geometry: MonitorGridGeometry,
) -> float:
    weights = np.asarray(grid_geometry.cell_volumes, dtype=np.float64)
    return float(
        np.sum(
            np.asarray(field_a, dtype=np.float64) * np.asarray(field_b, dtype=np.float64) * weights,
            dtype=np.float64,
        )
    )


def _principal_mode(
    *,
    residual_fields: tuple[np.ndarray, ...],
    grid_geometry: MonitorGridGeometry,
) -> tuple[np.ndarray, float]:
    sqrt_weights = np.sqrt(np.asarray(grid_geometry.cell_volumes, dtype=np.float64))
    matrix = np.stack(
        [np.asarray(field, dtype=np.float64).reshape(-1) * sqrt_weights.reshape(-1) for field in residual_fields],
        axis=0,
    )
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    weighted_mode = np.asarray(right_vectors[0], dtype=np.float64).reshape(grid_geometry.spec.shape)
    mode = np.asarray(weighted_mode / np.maximum(sqrt_weights, 1.0e-30), dtype=np.float64)
    norm = weighted_l2_norm(mode, grid_geometry=grid_geometry)
    if norm <= 1.0e-16:
        raise ValueError("Principal plateau mode has near-zero weighted norm.")
    explained_fraction = float((singular_values[0] ** 2) / np.sum(singular_values**2, dtype=np.float64))
    return mode / norm, explained_fraction


def _grid_parameter_summary(grid_geometry: MonitorGridGeometry) -> str:
    return (
        f"shape={grid_geometry.spec.shape}, "
        f"box_half_extents_bohr=("
        f"{float(np.max(np.abs(grid_geometry.x_points))):.3f}, "
        f"{float(np.max(np.abs(grid_geometry.y_points))):.3f}, "
        f"{float(np.max(np.abs(grid_geometry.z_points))):.3f})"
    )


def _renormalized_charge_perturbation(
    *,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    occupations,
    grid_geometry: MonitorGridGeometry,
    mode: np.ndarray,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    scale = float(amplitude)
    positive_factor = np.maximum(1.0 + scale * np.asarray(mode, dtype=np.float64), 0.0)
    rho_up_perturbed = np.asarray(rho_up, dtype=np.float64) * positive_factor
    rho_down_perturbed = np.asarray(rho_down, dtype=np.float64) * positive_factor
    from isogrid.scf.driver import _renormalize_density

    return (
        _renormalize_density(rho_up_perturbed, occupations.n_alpha, grid_geometry=grid_geometry),
        _renormalize_density(rho_down_perturbed, occupations.n_beta, grid_geometry=grid_geometry),
    )


def run_h2_monitor_grid_plateau_mode_ks_output_jacobian_sign_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    probe_iteration: int = _DEFAULT_PROBE_ITERATION,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridPlateauModeKsOutputJacobianSignAuditResult:
    """Estimate the sign of the one-step KS output Jacobian along the plateau mode."""

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
    )
    if len(source_result.history) < 2:
        raise ValueError("Plateau-mode Jacobian sign audit requires at least two SCF iterations.")
    occupations = resolve_h2_spin_occupations(spin_label=spin_label, case=case)
    residual_fields = tuple(_charge_residual_field(record) for record in source_result.history)
    mode, explained_fraction = _principal_mode(
        residual_fields=residual_fields[-5:],
        grid_geometry=grid_geometry,
    )
    probe_index = max(0, min(int(probe_iteration) - 1, len(source_result.history) - 1))
    probe_record = source_result.history[probe_index]

    baseline_context = _build_context(
        case=case,
        grid_geometry=grid_geometry,
        rho_up=probe_record.input_rho_up,
        rho_down=probe_record.input_rho_down,
    )
    track_count = _k_track(occupations, track_lowest_two_states=True)
    baseline_solve, baseline_tracked = _baseline_track_solves(
        contexts=(baseline_context,),
        case=case,
        count=track_count,
        occupations=occupations,
        grid_geometry=grid_geometry,
    )
    baseline_rho_up_out, baseline_rho_down_out = _density_from_tracked_solve(
        solve_up=baseline_solve[0],
        tracked_occupied_orbitals=baseline_tracked[0],
        occupations=occupations,
        grid_geometry=grid_geometry,
    )

    rho_up_pos, rho_down_pos = _renormalized_charge_perturbation(
        rho_up=probe_record.input_rho_up,
        rho_down=probe_record.input_rho_down,
        occupations=occupations,
        grid_geometry=grid_geometry,
        mode=mode,
        amplitude=_DEFAULT_MODE_AMPLITUDE,
    )
    rho_up_neg, rho_down_neg = _renormalized_charge_perturbation(
        rho_up=probe_record.input_rho_up,
        rho_down=probe_record.input_rho_down,
        occupations=occupations,
        grid_geometry=grid_geometry,
        mode=mode,
        amplitude=-_DEFAULT_MODE_AMPLITUDE,
    )

    pos_context = _build_context(
        case=case,
        grid_geometry=grid_geometry,
        rho_up=rho_up_pos,
        rho_down=rho_down_pos,
    )
    neg_context = _build_context(
        case=case,
        grid_geometry=grid_geometry,
        rho_up=rho_up_neg,
        rho_down=rho_down_neg,
    )
    pos_solve, pos_tracked = _baseline_track_solves(
        contexts=(pos_context,),
        case=case,
        count=track_count,
        occupations=occupations,
        grid_geometry=grid_geometry,
    )
    neg_solve, neg_tracked = _baseline_track_solves(
        contexts=(neg_context,),
        case=case,
        count=track_count,
        occupations=occupations,
        grid_geometry=grid_geometry,
    )
    pos_rho_up_out, pos_rho_down_out = _density_from_tracked_solve(
        solve_up=pos_solve[0],
        tracked_occupied_orbitals=pos_tracked[0],
        occupations=occupations,
        grid_geometry=grid_geometry,
    )
    neg_rho_up_out, neg_rho_down_out = _density_from_tracked_solve(
        solve_up=neg_solve[0],
        tracked_occupied_orbitals=neg_tracked[0],
        occupations=occupations,
        grid_geometry=grid_geometry,
    )

    input_charge_pos = np.asarray(rho_up_pos + rho_down_pos, dtype=np.float64)
    input_charge_neg = np.asarray(rho_up_neg + rho_down_neg, dtype=np.float64)
    baseline_charge = np.asarray(probe_record.input_rho_up + probe_record.input_rho_down, dtype=np.float64)
    input_delta = 0.5 * (input_charge_pos - input_charge_neg)
    effective_delta = 0.5 * (
        np.asarray(pos_context.effective_local_potential, dtype=np.float64)
        - np.asarray(neg_context.effective_local_potential, dtype=np.float64)
    )
    output_delta = 0.5 * (
        np.asarray(pos_rho_up_out + pos_rho_down_out, dtype=np.float64)
        - np.asarray(neg_rho_up_out + neg_rho_down_out, dtype=np.float64)
    )

    input_mode_coefficient = _weighted_inner(input_delta, mode, grid_geometry=grid_geometry)
    effective_mode_coefficient = _weighted_inner(effective_delta, mode, grid_geometry=grid_geometry)
    output_mode_coefficient = _weighted_inner(output_delta, mode, grid_geometry=grid_geometry)
    if abs(input_mode_coefficient) <= 1.0e-16:
        raise ValueError("Plateau-mode input perturbation projected to near zero.")

    effective_potential_signed_gain = float(effective_mode_coefficient / input_mode_coefficient)
    output_density_signed_gain = float(output_mode_coefficient / input_mode_coefficient)

    if output_density_signed_gain > 0.05:
        output_density_sign_regime = "positive"
        verdict = "The KS output map responds with a positive signed gain along the plateau mode."
    elif output_density_signed_gain < -0.05:
        output_density_sign_regime = "negative"
        verdict = "The KS output map responds with a negative signed gain along the plateau mode."
    else:
        output_density_sign_regime = "near_zero"
        verdict = "The KS output map is nearly neutral along the plateau mode."

    return H2MonitorGridPlateauModeKsOutputJacobianSignAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=source_iteration_count,
        probe_iteration=int(probe_record.iteration),
        principal_mode_explained_fraction=float(explained_fraction),
        input_mode_norm=float(_weighted_field_norm(input_delta, grid_geometry=grid_geometry)),
        effective_potential_signed_gain=effective_potential_signed_gain,
        output_density_signed_gain=output_density_signed_gain,
        output_density_sign_regime=output_density_sign_regime,
        verdict=verdict,
    )


def print_h2_monitor_grid_plateau_mode_ks_output_jacobian_sign_summary(
    result: H2MonitorGridPlateauModeKsOutputJacobianSignAuditResult,
) -> None:
    """Print a compact summary of the plateau-mode KS output Jacobian sign audit."""

    print("=== H2 monitor-grid plateau-mode KS output Jacobian sign audit ===")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    print(f"source iterations: {result.source_iteration_count}")
    print(f"probe iteration: {result.probe_iteration}")
    print(f"principal_mode_explained_fraction: {result.principal_mode_explained_fraction:.6f}")
    print(f"effective_potential_signed_gain: {result.effective_potential_signed_gain:.6e}")
    print(f"output_density_signed_gain: {result.output_density_signed_gain:.6e}")
    print(f"output_density_sign_regime: {result.output_density_sign_regime}")
    print(f"verdict: {result.verdict}")
