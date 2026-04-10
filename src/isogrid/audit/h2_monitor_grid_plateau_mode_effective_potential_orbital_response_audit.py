"""Audit plateau-mode remapping through effective-potential and orbital response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ops.kinetic import weighted_l2_norm

from .h2_monitor_grid_local_linear_response_audit import _build_context
from .h2_monitor_grid_local_linear_response_audit import _overlap_tracking
from .h2_monitor_grid_scf_amplification_ablation_audit import _k_track
from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _density_from_selection_or_tracked_solve,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    _track_solves_with_optional_projector_route,
)

_DEFAULT_SHAPE = (15, 15, 17)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (9.0, 9.0, 11.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 12
_DEFAULT_LATE_WINDOW_SIZE = 5


@dataclass(frozen=True)
class H2MonitorGridPlateauModeEffectivePotentialOrbitalResponseSample:
    """One late-step plateau-mode remapping sample."""

    iteration: int
    mode_residual_coefficient: float
    mixed_update_to_mode_ratio: float | None
    hartree_mode_coefficient: float
    xc_mode_coefficient: float
    effective_mode_coefficient: float
    hartree_mode_alignment: float | None
    xc_mode_alignment: float | None
    effective_mode_alignment: float | None
    output_response_to_mixed_ratio: float | None
    occupied_orbital_overlap_abs: float | None
    lowest2_subspace_overlap_min_singular_value: float | None
    lowest2_subspace_rotation_max_angle_deg: float | None
    lowest_gap_ha: float | None
    lowest_gap_delta_ha: float | None
    dominant_channel: str


@dataclass(frozen=True)
class H2MonitorGridPlateauModeEffectivePotentialOrbitalResponseAuditResult:
    """Top-level plateau-mode effective-potential/orbital-response audit result."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    late_window_size: int
    principal_mode_explained_fraction: float
    last_hartree_mode_alignment: float
    last_xc_mode_alignment: float
    dominant_remapping_channel: str
    samples: tuple[H2MonitorGridPlateauModeEffectivePotentialOrbitalResponseSample, ...]
    verdict: str


def _charge_density(record, kind: str) -> np.ndarray:
    if kind == "input":
        return np.asarray(record.input_rho_up + record.input_rho_down, dtype=np.float64)
    if kind == "mixed":
        return np.asarray(record.mixed_rho_up + record.mixed_rho_down, dtype=np.float64)
    if kind == "output":
        return np.asarray(record.output_rho_up + record.output_rho_down, dtype=np.float64)
    raise ValueError(f"Unsupported charge density kind: {kind}")


def _charge_residual_field(record) -> np.ndarray:
    return np.asarray(_charge_density(record, "output") - _charge_density(record, "input"), dtype=np.float64)


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


def _weighted_field_norm(field: np.ndarray, *, grid_geometry: MonitorGridGeometry) -> float:
    weights = np.asarray(grid_geometry.cell_volumes, dtype=np.float64)
    value = float(np.sum(np.asarray(field, dtype=np.float64) ** 2 * weights, dtype=np.float64))
    return float(np.sqrt(max(value, 0.0)))


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


def _field_mode_alignment(
    field: np.ndarray,
    *,
    mode: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> tuple[float, float | None]:
    coefficient = _weighted_inner(field, mode, grid_geometry=grid_geometry)
    field_norm = _weighted_field_norm(field, grid_geometry=grid_geometry)
    alignment = None if field_norm <= 1.0e-16 else float(coefficient / field_norm)
    return float(coefficient), alignment


def _sample_dominant_channel(
    *,
    hartree_mode_coefficient: float,
    xc_mode_coefficient: float,
    occupied_overlap_abs: float | None,
    subspace_rotation_deg: float | None,
    output_response_to_mixed_ratio: float | None,
) -> str:
    if (
        output_response_to_mixed_ratio is not None
        and output_response_to_mixed_ratio < -0.2
        and (
            (occupied_overlap_abs is not None and occupied_overlap_abs < 0.5)
            or (subspace_rotation_deg is not None and subspace_rotation_deg > 45.0)
        )
    ):
        return "orbital"
    return "hartree" if abs(hartree_mode_coefficient) >= abs(xc_mode_coefficient) else "xc"


def run_h2_monitor_grid_plateau_mode_effective_potential_orbital_response_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
    singlet_experimental_route_name: str = "none",
) -> H2MonitorGridPlateauModeEffectivePotentialOrbitalResponseAuditResult:
    """Audit which channel remaps the plateau mode on the local-only SCF trace."""

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
        raise ValueError("Plateau-mode response audit requires at least two SCF iterations.")

    occupations = source_result.occupations
    window_size = max(2, min(int(late_window_size), len(source_result.history)))
    records = tuple(source_result.history[-window_size:])
    residual_fields = tuple(_charge_residual_field(record) for record in records)
    mode, explained_fraction = _principal_mode(
        residual_fields=residual_fields,
        grid_geometry=grid_geometry,
    )
    residual_coefficients = [
        _weighted_inner(field, mode, grid_geometry=grid_geometry) for field in residual_fields
    ]
    if abs(residual_coefficients[-1]) < abs(residual_coefficients[0]):
        mode = -mode
        residual_coefficients = [-value for value in residual_coefficients]

    samples: list[H2MonitorGridPlateauModeEffectivePotentialOrbitalResponseSample] = []
    for index, record in enumerate(records[:-1]):
        residual_coefficient = float(residual_coefficients[index])
        input_context = _build_context(
            case=case,
            grid_geometry=grid_geometry,
            rho_up=record.input_rho_up,
            rho_down=record.input_rho_down,
        )
        mixed_context = _build_context(
            case=case,
            grid_geometry=grid_geometry,
            rho_up=record.mixed_rho_up,
            rho_down=record.mixed_rho_down,
        )
        hartree_delta = np.asarray(
            mixed_context.hartree_potential - input_context.hartree_potential,
            dtype=np.float64,
        )
        xc_delta = np.asarray(
            mixed_context.xc_potential - input_context.xc_potential,
            dtype=np.float64,
        )
        effective_delta = np.asarray(
            mixed_context.effective_local_potential - input_context.effective_local_potential,
            dtype=np.float64,
        )
        hartree_coeff, hartree_alignment = _field_mode_alignment(
            hartree_delta, mode=mode, grid_geometry=grid_geometry
        )
        xc_coeff, xc_alignment = _field_mode_alignment(
            xc_delta, mode=mode, grid_geometry=grid_geometry
        )
        effective_coeff, effective_alignment = _field_mode_alignment(
            effective_delta, mode=mode, grid_geometry=grid_geometry
        )

        track_count = _k_track(occupations, track_lowest_two_states=True)
        solves, tracked, selections = _track_solves_with_optional_projector_route(
            contexts=(input_context, mixed_context),
            case=case,
            count=track_count,
            occupations=occupations,
            grid_geometry=grid_geometry,
            singlet_experimental_route_name=singlet_experimental_route_name,
        )
        baseline_solve, mixed_solve = solves
        baseline_tracked, mixed_tracked = tracked
        baseline_selection, mixed_selection = selections
        baseline_rho_up_out, baseline_rho_down_out = _density_from_selection_or_tracked_solve(
            solve_up=baseline_solve,
            tracked_occupied_orbitals=baseline_tracked,
            projector_selection=baseline_selection,
            occupations=occupations,
            grid_geometry=grid_geometry,
        )
        mixed_rho_up_out, mixed_rho_down_out = _density_from_selection_or_tracked_solve(
            solve_up=mixed_solve,
            tracked_occupied_orbitals=mixed_tracked,
            projector_selection=mixed_selection,
            occupations=occupations,
            grid_geometry=grid_geometry,
        )
        output_response = np.asarray(
            (mixed_rho_up_out + mixed_rho_down_out) - (baseline_rho_up_out + baseline_rho_down_out),
            dtype=np.float64,
        )
        input_charge = _charge_density(record, "input")
        mixed_charge = _charge_density(record, "mixed")
        mixed_update = np.asarray(mixed_charge - input_charge, dtype=np.float64)
        mixed_update_coefficient = _weighted_inner(mixed_update, mode, grid_geometry=grid_geometry)
        mixed_update_to_mode_ratio = None
        output_response_to_mixed_ratio = None
        if abs(residual_coefficient) > 1.0e-16:
            mixed_update_to_mode_ratio = float(mixed_update_coefficient / residual_coefficient)
        if abs(mixed_update_coefficient) > 1.0e-16:
            output_response_to_mixed_ratio = float(
                _weighted_inner(output_response, mode, grid_geometry=grid_geometry)
                / mixed_update_coefficient
            )
        (
            occupied_overlap_abs,
            min_singular,
            max_angle_deg,
            current_gap,
            gap_delta,
        ) = _overlap_tracking(
            previous_solve=baseline_solve,
            current_solve=mixed_solve,
            previous_occupied_orbitals=baseline_tracked,
            current_occupied_orbitals=mixed_tracked,
            grid_geometry=grid_geometry,
            track_lowest_two_states=True,
        )
        dominant_channel = _sample_dominant_channel(
            hartree_mode_coefficient=hartree_coeff,
            xc_mode_coefficient=xc_coeff,
            occupied_overlap_abs=occupied_overlap_abs,
            subspace_rotation_deg=max_angle_deg,
            output_response_to_mixed_ratio=output_response_to_mixed_ratio,
        )
        samples.append(
            H2MonitorGridPlateauModeEffectivePotentialOrbitalResponseSample(
                iteration=int(record.iteration),
                mode_residual_coefficient=residual_coefficient,
                mixed_update_to_mode_ratio=mixed_update_to_mode_ratio,
                hartree_mode_coefficient=hartree_coeff,
                xc_mode_coefficient=xc_coeff,
                effective_mode_coefficient=effective_coeff,
                hartree_mode_alignment=hartree_alignment,
                xc_mode_alignment=xc_alignment,
                effective_mode_alignment=effective_alignment,
                output_response_to_mixed_ratio=output_response_to_mixed_ratio,
                occupied_orbital_overlap_abs=occupied_overlap_abs,
                lowest2_subspace_overlap_min_singular_value=min_singular,
                lowest2_subspace_rotation_max_angle_deg=max_angle_deg,
                lowest_gap_ha=current_gap,
                lowest_gap_delta_ha=gap_delta,
                dominant_channel=dominant_channel,
            )
        )

    if not samples:
        raise ValueError("Plateau-mode response audit requires at least one late-step sample.")
    last_sample = samples[-1]
    dominant_remapping_channel = last_sample.dominant_channel
    if dominant_remapping_channel == "orbital":
        verdict = (
            "The plateau mode is remapped primarily through lowest-state orbital/subspace response, "
            "not a direct Hartree/XC potential alignment alone."
        )
    elif dominant_remapping_channel == "hartree":
        verdict = (
            "The plateau mode remains most strongly aligned with the Hartree response among the "
            "effective-potential channels."
        )
    else:
        verdict = (
            "The plateau mode remains most strongly aligned with the XC response among the "
            "effective-potential channels."
        )

    return H2MonitorGridPlateauModeEffectivePotentialOrbitalResponseAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=source_iteration_count,
        late_window_size=window_size,
        principal_mode_explained_fraction=float(explained_fraction),
        last_hartree_mode_alignment=(
            0.0 if last_sample.hartree_mode_alignment is None else float(last_sample.hartree_mode_alignment)
        ),
        last_xc_mode_alignment=(
            0.0 if last_sample.xc_mode_alignment is None else float(last_sample.xc_mode_alignment)
        ),
        dominant_remapping_channel=dominant_remapping_channel,
        samples=tuple(samples),
        verdict=verdict,
    )


def print_h2_monitor_grid_plateau_mode_effective_potential_orbital_response_summary(
    result: H2MonitorGridPlateauModeEffectivePotentialOrbitalResponseAuditResult,
) -> None:
    """Print a compact summary of the plateau-mode remapping audit."""

    print("=== H2 monitor-grid plateau-mode effective-potential/orbital-response audit ===")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    print(f"source iterations: {result.source_iteration_count}")
    print(f"late window size: {result.late_window_size}")
    print(f"principal_mode_explained_fraction: {result.principal_mode_explained_fraction:.6f}")
    print(f"last_hartree_mode_alignment: {result.last_hartree_mode_alignment:.6e}")
    print(f"last_xc_mode_alignment: {result.last_xc_mode_alignment:.6e}")
    print(f"dominant_remapping_channel: {result.dominant_remapping_channel}")
    print(f"verdict: {result.verdict}")
