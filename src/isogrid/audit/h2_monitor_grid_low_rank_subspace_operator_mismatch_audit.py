"""Audit mismatch between projected low-rank updates and realized residual maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.scf.controller import ScfControllerConfig
from isogrid.scf.controller import _accepted_low_rank_modes
from isogrid.scf.controller import _dominant_residual_modes
from isogrid.scf.controller import _prepare_low_rank_charge_variable
from isogrid.scf.controller import _weighted_inner

from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result

_DEFAULT_SHAPE = (15, 15, 17)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (9.0, 9.0, 11.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 12
_DEFAULT_LATE_WINDOW_SIZE = 5


@dataclass(frozen=True)
class H2MonitorGridLowRankSubspaceOperatorMismatchSample:
    """One late-step sample in the accepted low-rank subspace."""

    iteration: int
    current_coefficients: tuple[float, ...]
    projected_update_coefficients: tuple[float, ...]
    next_residual_coefficients: tuple[float, ...] | None


@dataclass(frozen=True)
class H2MonitorGridLowRankSubspaceOperatorMismatchAuditResult:
    """Mismatch summary between the projected update and realized residual map."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    late_window_size: int
    subspace_rank: int
    mode_explained_fractions: tuple[float, ...]
    projected_update_operator: tuple[tuple[float, ...], ...]
    realized_next_residual_operator: tuple[tuple[float, ...], ...]
    residual_reduction_operator: tuple[tuple[float, ...], ...]
    operator_mismatch_frobenius_norm: float
    diagonal_gain_shortfall_norm: float
    off_diagonal_coupling_norm: float
    samples: tuple[H2MonitorGridLowRankSubspaceOperatorMismatchSample, ...]
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


def _grid_parameter_summary(grid_geometry: MonitorGridGeometry) -> str:
    return (
        f"shape={grid_geometry.spec.shape}, "
        f"box_half_extents_bohr=("
        f"{float(np.max(np.abs(grid_geometry.x_points))):.3f}, "
        f"{float(np.max(np.abs(grid_geometry.y_points))):.3f}, "
        f"{float(np.max(np.abs(grid_geometry.z_points))):.3f})"
    )


def _least_squares_operator(
    *,
    inputs: tuple[np.ndarray, ...],
    outputs: tuple[np.ndarray, ...],
    rank: int,
) -> np.ndarray:
    if not inputs or not outputs:
        return np.zeros((rank, rank), dtype=np.float64)
    input_matrix = np.stack(inputs, axis=0)
    output_matrix = np.stack(outputs, axis=0)
    operator, *_ = np.linalg.lstsq(input_matrix, output_matrix, rcond=None)
    return np.asarray(operator.T, dtype=np.float64)


def _tuple_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(matrix, dtype=np.float64)
    )


def run_h2_monitor_grid_low_rank_subspace_operator_mismatch_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridLowRankSubspaceOperatorMismatchAuditResult:
    """Compare the low-rank projected update with the realized residual mapping."""

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
    if len(source_result.history) < 3:
        raise ValueError("Operator mismatch audit requires at least three SCF iterations.")
    window_size = max(3, min(int(late_window_size), len(source_result.history)))
    records = tuple(source_result.history[-window_size:])

    config = ScfControllerConfig.generic_charge_spin_preconditioned()
    residual_history = tuple(_charge_residual_field(record) for record in records)
    prepared_history, _ = _prepare_low_rank_charge_variable(
        history=residual_history,
        current_residual=residual_history[-1],
        smoothing_passes=config.preconditioned_smoothing_passes,
    )
    dominant_modes = _dominant_residual_modes(
        history=prepared_history,
        grid_geometry=grid_geometry,
        max_modes=int(config.low_rank_mode_count),
    )
    accepted_modes = _accepted_low_rank_modes(
        dominant_modes=dominant_modes,
        config=config,
    )
    if not accepted_modes:
        raise ValueError("No accepted low-rank modes were available for the mismatch audit.")
    basis = tuple(mode for mode, _ in accepted_modes)
    explained_fractions = tuple(float(explained_fraction) for _, explained_fraction in accepted_modes)
    rank = len(basis)

    low_rank_variable_fields = prepared_history
    update_fields = tuple(_charge_update_field(record) for record in records)

    current_coefficients_series = tuple(
        np.asarray(
            [_weighted_inner(field, mode, grid_geometry=grid_geometry) for mode in basis],
            dtype=np.float64,
        )
        for field in low_rank_variable_fields
    )
    projected_update_series = tuple(
        np.asarray(
            [_weighted_inner(field, mode, grid_geometry=grid_geometry) for mode in basis],
            dtype=np.float64,
        )
        for field in update_fields
    )
    next_residual_series = current_coefficients_series[1:]
    current_input_series = current_coefficients_series[:-1]
    projected_update_inputs = projected_update_series[:-1]
    residual_reduction_series = tuple(
        np.asarray(current - nxt, dtype=np.float64)
        for current, nxt in zip(current_input_series, next_residual_series, strict=True)
    )

    projected_update_operator = _least_squares_operator(
        inputs=current_input_series,
        outputs=projected_update_inputs,
        rank=rank,
    )
    realized_next_residual_operator = _least_squares_operator(
        inputs=current_input_series,
        outputs=next_residual_series,
        rank=rank,
    )
    residual_reduction_operator = np.eye(rank, dtype=np.float64) - realized_next_residual_operator
    operator_mismatch = projected_update_operator - residual_reduction_operator
    operator_mismatch_frobenius_norm = float(np.linalg.norm(operator_mismatch, ord="fro"))
    diagonal_gain_shortfall_norm = float(
        np.linalg.norm(
            np.diag(np.diag(projected_update_operator) - np.diag(residual_reduction_operator)),
            ord="fro",
        )
    )
    off_diagonal_coupling_norm = float(
        np.linalg.norm(
            residual_reduction_operator - np.diag(np.diag(residual_reduction_operator)),
            ord="fro",
        )
    )

    samples = tuple(
        H2MonitorGridLowRankSubspaceOperatorMismatchSample(
            iteration=int(record.iteration),
            current_coefficients=tuple(float(value) for value in current_coefficients_series[index]),
            projected_update_coefficients=tuple(
                float(value) for value in projected_update_series[index]
            ),
            next_residual_coefficients=(
                None
                if index + 1 >= len(current_coefficients_series)
                else tuple(float(value) for value in current_coefficients_series[index + 1])
            ),
        )
        for index, record in enumerate(records)
    )

    if diagonal_gain_shortfall_norm >= max(off_diagonal_coupling_norm, 1.0e-12):
        verdict = (
            "The dominant mismatch is diagonal gain shortfall: the projected low-rank update is "
            "too weak relative to the realized residual-reduction operator."
        )
    else:
        verdict = (
            "The dominant mismatch comes from subspace coupling: the realized next-residual map "
            "contains off-diagonal mixing that the projected low-rank update does not match."
        )

    return H2MonitorGridLowRankSubspaceOperatorMismatchAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=source_iteration_count,
        late_window_size=window_size,
        subspace_rank=rank,
        mode_explained_fractions=explained_fractions,
        projected_update_operator=_tuple_matrix(projected_update_operator),
        realized_next_residual_operator=_tuple_matrix(realized_next_residual_operator),
        residual_reduction_operator=_tuple_matrix(residual_reduction_operator),
        operator_mismatch_frobenius_norm=operator_mismatch_frobenius_norm,
        diagonal_gain_shortfall_norm=diagonal_gain_shortfall_norm,
        off_diagonal_coupling_norm=off_diagonal_coupling_norm,
        samples=samples,
        verdict=verdict,
    )


def print_h2_monitor_grid_low_rank_subspace_operator_mismatch_summary(
    result: H2MonitorGridLowRankSubspaceOperatorMismatchAuditResult,
) -> None:
    """Print the compact low-rank subspace operator mismatch summary."""

    print("IsoGridDFT H2 monitor-grid low-rank subspace operator mismatch audit")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin state: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    print(f"subspace rank: {result.subspace_rank}")
    print(f"mode explained fractions: {result.mode_explained_fractions}")
    print(f"projected update operator: {result.projected_update_operator}")
    print(f"realized next residual operator: {result.realized_next_residual_operator}")
    print(f"residual reduction operator: {result.residual_reduction_operator}")
    print(f"operator mismatch frobenius norm: {result.operator_mismatch_frobenius_norm}")
    print(f"diagonal gain shortfall norm: {result.diagonal_gain_shortfall_norm}")
    print(f"off-diagonal coupling norm: {result.off_diagonal_coupling_norm}")
    print(f"verdict: {result.verdict}")


def main() -> int:
    result = run_h2_monitor_grid_low_rank_subspace_operator_mismatch_audit()
    print_h2_monitor_grid_low_rank_subspace_operator_mismatch_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
