"""Audit the projected response from mixed density to the next output density."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ops.kinetic import weighted_l2_norm

from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result

_DEFAULT_SHAPE = (15, 15, 17)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (9.0, 9.0, 11.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 12
_DEFAULT_LATE_WINDOW_SIZE = 5


@dataclass(frozen=True)
class H2MonitorGridMixedDensityToOutputResponseSample:
    """One late-step sample projected onto the dominant charge mode."""

    iteration: int
    mode_residual_coefficient: float
    mixed_update_coefficient: float
    mixed_update_to_mode_ratio: float | None
    next_output_response_coefficient: float | None
    next_output_response_to_mixed_ratio: float | None


@dataclass(frozen=True)
class H2MonitorGridMixedDensityToOutputResponseAuditResult:
    """Projected mixed-density to next-output response on the dominant charge mode."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    late_window_size: int
    principal_mode_explained_fraction: float
    current_mixed_update_to_mode_ratio: float
    last_available_next_output_response_to_mixed_ratio: float | None
    response_regime: str
    samples: tuple[H2MonitorGridMixedDensityToOutputResponseSample, ...]
    verdict: str


def _charge_density_from_pair(rho_up: np.ndarray, rho_down: np.ndarray) -> np.ndarray:
    return np.asarray(rho_up + rho_down, dtype=np.float64)


def _charge_density(record, kind: str) -> np.ndarray:
    if kind == "input":
        return _charge_density_from_pair(record.input_rho_up, record.input_rho_down)
    if kind == "output":
        return _charge_density_from_pair(record.output_rho_up, record.output_rho_down)
    if kind == "mixed":
        return _charge_density_from_pair(record.mixed_rho_up, record.mixed_rho_down)
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
        raise ValueError("Principal mixed-to-output response mode has near-zero weighted norm.")
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


def run_h2_monitor_grid_mixed_density_to_output_response_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridMixedDensityToOutputResponseAuditResult:
    """Project the mixed-density to next-output response onto the dominant charge mode."""

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
        raise ValueError("Mixed-density response audit requires at least two SCF iterations.")
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

    samples: list[H2MonitorGridMixedDensityToOutputResponseSample] = []
    response_ratios: list[float] = []
    for index, record in enumerate(records):
        residual_coefficient = float(residual_coefficients[index])
        input_charge = _charge_density(record, "input")
        mixed_charge = _charge_density(record, "mixed")
        mixed_update_coefficient = _weighted_inner(
            mixed_charge - input_charge,
            mode,
            grid_geometry=grid_geometry,
        )
        mixed_update_ratio = None
        if abs(residual_coefficient) > 1.0e-16:
            mixed_update_ratio = float(mixed_update_coefficient / residual_coefficient)

        next_output_response_coefficient = None
        next_output_response_ratio = None
        if index + 1 < len(records) and abs(mixed_update_coefficient) > 1.0e-16:
            next_output_charge = _charge_density(records[index + 1], "output")
            next_output_response_coefficient = _weighted_inner(
                next_output_charge - mixed_charge,
                mode,
                grid_geometry=grid_geometry,
            )
            next_output_response_ratio = float(
                next_output_response_coefficient / mixed_update_coefficient
            )
            response_ratios.append(next_output_response_ratio)

        samples.append(
            H2MonitorGridMixedDensityToOutputResponseSample(
                iteration=int(record.iteration),
                mode_residual_coefficient=residual_coefficient,
                mixed_update_coefficient=float(mixed_update_coefficient),
                mixed_update_to_mode_ratio=mixed_update_ratio,
                next_output_response_coefficient=next_output_response_coefficient,
                next_output_response_to_mixed_ratio=next_output_response_ratio,
            )
        )

    current_mixed_update_to_mode_ratio = (
        0.0 if samples[-1].mixed_update_to_mode_ratio is None else float(samples[-1].mixed_update_to_mode_ratio)
    )
    last_available_next_output_response_to_mixed_ratio = (
        None if not response_ratios else float(response_ratios[-1])
    )
    regime_value = 0.0 if last_available_next_output_response_to_mixed_ratio is None else last_available_next_output_response_to_mixed_ratio
    if regime_value <= -0.2:
        response_regime = "counteract"
        verdict = (
            "The next output density counteracts the mixed-density update along the dominant plateau mode."
        )
    elif regime_value >= 0.2:
        response_regime = "follow"
        verdict = (
            "The next output density tends to follow the mixed-density update along the dominant plateau mode."
        )
    else:
        response_regime = "neutral"
        verdict = (
            "The next output density shows only a weak net response to the mixed-density update on the dominant plateau mode."
        )

    return H2MonitorGridMixedDensityToOutputResponseAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=source_iteration_count,
        late_window_size=window_size,
        principal_mode_explained_fraction=float(explained_fraction),
        current_mixed_update_to_mode_ratio=current_mixed_update_to_mode_ratio,
        last_available_next_output_response_to_mixed_ratio=last_available_next_output_response_to_mixed_ratio,
        response_regime=response_regime,
        samples=tuple(samples),
        verdict=verdict,
    )


def print_h2_monitor_grid_mixed_density_to_output_response_summary(
    result: H2MonitorGridMixedDensityToOutputResponseAuditResult,
) -> None:
    """Print a compact summary of the mixed-density response audit."""

    print("=== H2 monitor-grid mixed-density to output-density response audit ===")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    print(f"source iterations: {result.source_iteration_count}")
    print(f"late window size: {result.late_window_size}")
    print(f"principal_mode_explained_fraction: {result.principal_mode_explained_fraction:.6f}")
    print(f"current_mixed_update_to_mode_ratio: {result.current_mixed_update_to_mode_ratio:.6e}")
    print(
        "last_available_next_output_response_to_mixed_ratio: "
        + (
            "None"
            if result.last_available_next_output_response_to_mixed_ratio is None
            else f"{result.last_available_next_output_response_to_mixed_ratio:.6e}"
        )
    )
    print(f"response_regime: {result.response_regime}")
    print(f"verdict: {result.verdict}")
