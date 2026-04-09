"""Compare candidate charge preconditioning variables on the local-only plateau."""

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
from isogrid.scf.controller import _density_support_weight
from isogrid.scf.controller import _grid_preconditioner_scale
from isogrid.scf.controller import _smooth_charge_residual

from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result

_DEFAULT_SHAPE = (15, 15, 17)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (9.0, 9.0, 11.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 12
_DEFAULT_LATE_WINDOW_SIZE = 5


@dataclass(frozen=True)
class H2MonitorGridPreconditionerVariableSample:
    """Principal-mode summary for one candidate preconditioning variable."""

    variable_name: str
    principal_mode_explained_fraction: float
    current_mode_coefficient: float
    current_update_projection: float
    current_update_to_mode_ratio: float | None
    last_realized_mode_ratio: float | None


@dataclass(frozen=True)
class H2MonitorGridPreconditionerVariableAuditResult:
    """Comparison of several candidate charge preconditioning variables."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    late_window_size: int
    variables: tuple[H2MonitorGridPreconditionerVariableSample, ...]
    best_alignment_variable_name: str
    verdict: str


def _charge_density(record) -> np.ndarray:
    return np.asarray(record.input_rho_up + record.input_rho_down, dtype=np.float64)


def _charge_output_density(record) -> np.ndarray:
    return np.asarray(record.output_rho_up + record.output_rho_down, dtype=np.float64)


def _charge_mixed_density(record) -> np.ndarray:
    return np.asarray(record.mixed_rho_up + record.mixed_rho_down, dtype=np.float64)


def _charge_residual_field(record) -> np.ndarray:
    return np.asarray(_charge_output_density(record) - _charge_density(record), dtype=np.float64)


def _charge_update_field(record) -> np.ndarray:
    return np.asarray(_charge_mixed_density(record) - _charge_density(record), dtype=np.float64)


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
    fields: tuple[np.ndarray, ...],
    grid_geometry: MonitorGridGeometry,
) -> tuple[np.ndarray, float]:
    sqrt_weights = np.sqrt(np.asarray(grid_geometry.cell_volumes, dtype=np.float64))
    matrix = np.stack(
        [np.asarray(field, dtype=np.float64).reshape(-1) * sqrt_weights.reshape(-1) for field in fields],
        axis=0,
    )
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    weighted_mode = np.asarray(right_vectors[0], dtype=np.float64).reshape(grid_geometry.spec.shape)
    mode = np.asarray(
        weighted_mode / np.maximum(sqrt_weights, 1.0e-30),
        dtype=np.float64,
    )
    norm = weighted_l2_norm(mode, grid_geometry=grid_geometry)
    if norm <= 1.0e-16:
        raise ValueError("Candidate variable principal mode has near-zero weighted norm.")
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


def _candidate_variable_fields(
    *,
    records: tuple,
    grid_geometry: MonitorGridGeometry,
    controller_name: str,
) -> tuple[tuple[str, tuple[np.ndarray, ...]], ...]:
    charge_residual_fields = tuple(_charge_residual_field(record) for record in records)
    smoothed_charge_residual_fields = tuple(
        _smooth_charge_residual(
            field,
            smoothing_passes=ScfControllerConfig.generic_charge_spin_preconditioned().preconditioned_smoothing_passes,
        )
        for field in charge_residual_fields
    )
    if controller_name == "generic_charge_spin_preconditioned":
        grid_scale = _grid_preconditioner_scale(
            grid_point_count=int(np.prod(grid_geometry.spec.shape)),
            reference_grid_point_count=int(
                ScfControllerConfig.generic_charge_spin_preconditioned().preconditioned_grid_point_boost_threshold
            ),
        )
        support_weighted_fields = []
        for record, raw_field, smooth_field in zip(
            records, charge_residual_fields, smoothed_charge_residual_fields, strict=True
        ):
            support_weight = _density_support_weight(_charge_density(record))
            support_weighted_fields.append(
                np.asarray(
                    (1.0 - grid_scale * support_weight) * smooth_field
                    + (grid_scale * support_weight) * raw_field,
                    dtype=np.float64,
                )
            )
        support_weighted_charge_residual_fields = tuple(support_weighted_fields)
    else:
        support_weighted_charge_residual_fields = charge_residual_fields
    realized_charge_update_fields = tuple(_charge_update_field(record) for record in records)
    return (
        ("charge_residual", charge_residual_fields),
        ("smoothed_charge_residual", smoothed_charge_residual_fields),
        ("support_weighted_charge_residual", support_weighted_charge_residual_fields),
        ("realized_charge_update", realized_charge_update_fields),
    )


def run_h2_monitor_grid_preconditioner_variable_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridPreconditionerVariableAuditResult:
    """Compare several candidate charge variables against the realized update."""

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
        raise ValueError("Preconditioner-variable audit requires at least two SCF iterations.")
    window_size = max(2, min(int(late_window_size), len(source_result.history)))
    records = tuple(source_result.history[-window_size:])
    realized_update_fields = tuple(_charge_update_field(record) for record in records)
    candidate_variables = _candidate_variable_fields(
        records=records,
        grid_geometry=grid_geometry,
        controller_name=controller_name,
    )

    samples: list[H2MonitorGridPreconditionerVariableSample] = []
    best_name = ""
    best_score = -np.inf
    for variable_name, fields in candidate_variables:
        mode, explained_fraction = _principal_mode(fields=fields, grid_geometry=grid_geometry)
        coefficients = [_weighted_inner(field, mode, grid_geometry=grid_geometry) for field in fields]
        if abs(coefficients[-1]) < abs(coefficients[0]):
            mode = -mode
            coefficients = [-value for value in coefficients]
        update_coefficients = [
            _weighted_inner(field, mode, grid_geometry=grid_geometry) for field in realized_update_fields
        ]
        current_mode_coefficient = float(coefficients[-1])
        current_update_projection = float(update_coefficients[-1])
        current_update_to_mode_ratio = None
        if abs(current_mode_coefficient) > 1.0e-16:
            current_update_to_mode_ratio = float(current_update_projection / current_mode_coefficient)
        last_realized_mode_ratio = None
        if len(coefficients) >= 2 and abs(coefficients[-2]) > 1.0e-16:
            last_realized_mode_ratio = float(coefficients[-1] / coefficients[-2])
        samples.append(
            H2MonitorGridPreconditionerVariableSample(
                variable_name=variable_name,
                principal_mode_explained_fraction=float(explained_fraction),
                current_mode_coefficient=current_mode_coefficient,
                current_update_projection=current_update_projection,
                current_update_to_mode_ratio=current_update_to_mode_ratio,
                last_realized_mode_ratio=last_realized_mode_ratio,
            )
        )
        if variable_name != "realized_charge_update" and current_update_to_mode_ratio is not None:
            score = abs(current_update_to_mode_ratio) * float(explained_fraction)
            if score > best_score:
                best_score = score
                best_name = variable_name

    if not best_name:
        best_name = samples[0].variable_name
    if best_name == "charge_residual":
        verdict = (
            "The current raw charge residual remains the strongest-aligned candidate under the "
            "late-step variable comparison metric."
        )
    else:
        verdict = (
            "A transformed charge variable aligns better with the realized plateau update than "
            "the raw charge residual."
        )
    return H2MonitorGridPreconditionerVariableAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=source_iteration_count,
        late_window_size=window_size,
        variables=tuple(samples),
        best_alignment_variable_name=best_name,
        verdict=verdict,
    )


def print_h2_monitor_grid_preconditioner_variable_summary(
    result: H2MonitorGridPreconditionerVariableAuditResult,
) -> None:
    """Print the compact preconditioner-variable comparison summary."""

    print("IsoGridDFT H2 monitor-grid preconditioner variable audit")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin state: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    for sample in result.variables:
        print()
        print(f"variable: {sample.variable_name}")
        print(f"  principal explained fraction: {sample.principal_mode_explained_fraction}")
        print(f"  current mode coefficient: {sample.current_mode_coefficient}")
        print(f"  current update projection: {sample.current_update_projection}")
        print(f"  update/mode ratio: {sample.current_update_to_mode_ratio}")
        print(f"  last realized mode ratio: {sample.last_realized_mode_ratio}")
    print()
    print(f"best alignment variable: {result.best_alignment_variable_name}")
    print(f"verdict: {result.verdict}")


def main() -> int:
    result = run_h2_monitor_grid_preconditioner_variable_audit()
    print_h2_monitor_grid_preconditioner_variable_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
