"""Late-step base charge-operator gain audit on the H2 monitor grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ops.kinetic import weighted_l2_norm
from isogrid.scf.controller import ScfControllerConfig
from isogrid.scf.controller import _build_preconditioned_base_charge_trial

from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result

_DEFAULT_SHAPE = (15, 15, 17)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (9.0, 9.0, 11.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 12
_DEFAULT_LATE_WINDOW_SIZE = 5


@dataclass(frozen=True)
class H2MonitorGridBaseChargeOperatorGainSample:
    """One late-step sample projected onto the dominant charge-residual mode."""

    iteration: int
    density_residual: float
    charge_mixing: float
    mode_residual_coefficient: float
    base_update_coefficient: float
    base_update_to_mode_ratio: float | None
    realized_next_mode_ratio: float | None


@dataclass(frozen=True)
class H2MonitorGridBaseChargeOperatorGainAuditResult:
    """Late-step comparison between base charge update and realized residual reduction."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    late_window_size: int
    principal_mode_explained_fraction: float
    current_base_update_to_mode_ratio: float
    realized_residual_reduction_ratio: float | None
    base_gain_shortfall: float | None
    samples: tuple[H2MonitorGridBaseChargeOperatorGainSample, ...]
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
        raise ValueError("Principal base-operator mode has near-zero weighted norm.")
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


def _base_charge_update_field(
    *,
    record,
    charge_mixing: float,
    controller_name: str,
    grid_geometry: MonitorGridGeometry,
) -> np.ndarray:
    rho_charge_current = _charge_density(record)
    charge_residual = _charge_residual_field(record)
    if controller_name == "generic_charge_spin_preconditioned":
        components = _build_preconditioned_base_charge_trial(
            rho_charge_current=rho_charge_current,
            charge_residual=charge_residual,
            next_charge_mixing=float(charge_mixing),
            config=ScfControllerConfig.generic_charge_spin_preconditioned(),
            grid_geometry=grid_geometry,
        )
        return np.asarray(components.rho_charge_trial - rho_charge_current, dtype=np.float64)
    return np.asarray(float(charge_mixing) * charge_residual, dtype=np.float64)


def run_h2_monitor_grid_base_charge_operator_gain_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridBaseChargeOperatorGainAuditResult:
    """Compare the base charge update to the realized residual reduction on the plateau."""

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
        raise ValueError("Base charge-operator gain audit requires at least two SCF iterations.")
    window_size = max(2, min(int(late_window_size), len(source_result.history)))
    records = tuple(source_result.history[-window_size:])
    charge_mixing_history = tuple(source_result.controller_charge_mixing_history[-window_size:])

    residual_fields = tuple(_charge_residual_field(record) for record in records)
    base_update_fields = tuple(
        _base_charge_update_field(
            record=record,
            charge_mixing=charge_mixing_history[index],
            controller_name=controller_name,
            grid_geometry=grid_geometry,
        )
        for index, record in enumerate(records)
    )
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
    base_update_coefficients = [
        _weighted_inner(field, mode, grid_geometry=grid_geometry) for field in base_update_fields
    ]

    samples: list[H2MonitorGridBaseChargeOperatorGainSample] = []
    for index, record in enumerate(records):
        residual_coefficient = float(residual_coefficients[index])
        base_update_coefficient = float(base_update_coefficients[index])
        base_update_to_mode_ratio = None
        if abs(residual_coefficient) > 1.0e-16:
            base_update_to_mode_ratio = float(base_update_coefficient / residual_coefficient)
        realized_next_mode_ratio = None
        if index + 1 < len(records) and abs(residual_coefficient) > 1.0e-16:
            realized_next_mode_ratio = float(residual_coefficients[index + 1] / residual_coefficient)
        samples.append(
            H2MonitorGridBaseChargeOperatorGainSample(
                iteration=int(record.iteration),
                density_residual=float(record.density_residual),
                charge_mixing=float(charge_mixing_history[index]),
                mode_residual_coefficient=residual_coefficient,
                base_update_coefficient=base_update_coefficient,
                base_update_to_mode_ratio=base_update_to_mode_ratio,
                realized_next_mode_ratio=realized_next_mode_ratio,
            )
        )

    current_sample = samples[-1]
    previous_sample = samples[-2] if len(samples) >= 2 else None
    current_base_update_to_mode_ratio = (
        0.0
        if current_sample.base_update_to_mode_ratio is None
        else float(current_sample.base_update_to_mode_ratio)
    )
    realized_residual_reduction_ratio = None
    base_gain_shortfall = None
    if previous_sample is not None and previous_sample.realized_next_mode_ratio is not None:
        realized_residual_reduction_ratio = float(1.0 - previous_sample.realized_next_mode_ratio)
        base_gain_shortfall = float(
            realized_residual_reduction_ratio - current_base_update_to_mode_ratio
        )

    if base_gain_shortfall is None:
        verdict = "The late-step window is too short to compare base gain against realized reduction."
    elif abs(base_gain_shortfall) < 1.0e-6:
        verdict = (
            "The base charge operator and realized residual reduction are closely matched on the "
            "dominant plateau mode."
        )
    elif base_gain_shortfall > 0.0:
        verdict = (
            "The base charge operator under-predicts the dominant plateau-mode reduction; the "
            "base gain is too weak."
        )
    else:
        verdict = (
            "The base charge operator over-predicts the dominant plateau-mode reduction; the "
            "base gain is too strong relative to the realized map."
        )

    return H2MonitorGridBaseChargeOperatorGainAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=source_iteration_count,
        late_window_size=window_size,
        principal_mode_explained_fraction=float(explained_fraction),
        current_base_update_to_mode_ratio=current_base_update_to_mode_ratio,
        realized_residual_reduction_ratio=realized_residual_reduction_ratio,
        base_gain_shortfall=base_gain_shortfall,
        samples=tuple(samples),
        verdict=verdict,
    )


def print_h2_monitor_grid_base_charge_operator_gain_summary(
    result: H2MonitorGridBaseChargeOperatorGainAuditResult,
) -> None:
    """Print a compact summary of the base charge-operator gain audit."""

    print("=== H2 monitor-grid base charge-operator gain audit ===")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    print(f"source iterations: {result.source_iteration_count}")
    print(f"late window size: {result.late_window_size}")
    print(f"principal_mode_explained_fraction: {result.principal_mode_explained_fraction:.6f}")
    print(f"current_base_update_to_mode_ratio: {result.current_base_update_to_mode_ratio:.6e}")
    print(
        "realized_residual_reduction_ratio: "
        + (
            "None"
            if result.realized_residual_reduction_ratio is None
            else f"{result.realized_residual_reduction_ratio:.6e}"
        )
    )
    print(
        "base_gain_shortfall: "
        + ("None" if result.base_gain_shortfall is None else f"{result.base_gain_shortfall:.6e}")
    )
    print(f"verdict: {result.verdict}")
