"""Late-step charge-residual mode / compression audit on the H2 monitor grid."""

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
class H2MonitorGridChargeModeCompressionSample:
    """One late-step residual/update sample projected onto the principal charge mode."""

    iteration: int
    density_residual: float
    residual_ratio: float | None
    hartree_share: float | None
    charge_mixing: float
    mode_residual_coefficient: float
    mode_update_coefficient: float
    update_to_residual_ratio: float | None
    realized_next_mode_ratio: float | None


@dataclass(frozen=True)
class H2MonitorGridChargeModeCompressionAuditResult:
    """Principal late-step charge-mode audit result for one local-only SCF trace."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    late_window_size: int
    principal_mode_explained_fraction: float
    principal_mode_residual_norm: float
    principal_mode_peak_abs: float
    current_update_mode_coefficient: float
    current_update_to_residual_ratio: float
    last_realized_mode_ratio: float | None
    mode_energy_fraction_captured: float
    samples: tuple[H2MonitorGridChargeModeCompressionSample, ...]
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
    residual_fields: tuple[np.ndarray, ...],
    grid_geometry: MonitorGridGeometry,
) -> tuple[np.ndarray, float]:
    sqrt_weights = np.sqrt(np.asarray(grid_geometry.cell_volumes, dtype=np.float64))
    matrix = np.stack(
        [np.asarray(field, dtype=np.float64).reshape(-1) * sqrt_weights.reshape(-1) for field in residual_fields],
        axis=0,
    )
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    leading_weighted_vector = np.asarray(right_vectors[0], dtype=np.float64).reshape(grid_geometry.spec.shape)
    mode = np.asarray(
        leading_weighted_vector / np.maximum(sqrt_weights, 1.0e-30),
        dtype=np.float64,
    )
    norm = weighted_l2_norm(mode, grid_geometry=grid_geometry)
    if norm <= 1.0e-16:
        raise ValueError("Principal charge mode has near-zero weighted norm.")
    mode = mode / norm
    explained_fraction = float(
        (singular_values[0] ** 2) / np.sum(singular_values**2, dtype=np.float64)
    )
    return mode, explained_fraction


def _grid_parameter_summary(grid_geometry: MonitorGridGeometry) -> str:
    return (
        f"shape={grid_geometry.spec.shape}, "
        f"box_half_extents_bohr=("
        f"{float(np.max(np.abs(grid_geometry.x_points))):.3f}, "
        f"{float(np.max(np.abs(grid_geometry.y_points))):.3f}, "
        f"{float(np.max(np.abs(grid_geometry.z_points))):.3f})"
    )


def run_h2_monitor_grid_charge_mode_compression_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridChargeModeCompressionAuditResult:
    """Measure the dominant late-step charge residual mode and its one-step compression."""

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
        raise ValueError("Charge-mode compression audit requires at least two SCF iterations.")
    window_size = max(2, min(int(late_window_size), len(source_result.history)))
    records = tuple(source_result.history[-window_size:])
    signals = tuple(source_result.controller_signals_history[-window_size:])
    charge_mixing_history = tuple(source_result.controller_charge_mixing_history[-window_size:])

    residual_fields = tuple(_charge_residual_field(record) for record in records)
    update_fields = tuple(_charge_update_field(record) for record in records)
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
    update_coefficients = [
        _weighted_inner(field, mode, grid_geometry=grid_geometry) for field in update_fields
    ]
    total_residual_energy = float(
        sum(
            weighted_l2_norm(field, grid_geometry=grid_geometry) ** 2
            for field in residual_fields
        )
    )
    captured_energy = float(sum(value * value for value in residual_coefficients))
    mode_energy_fraction_captured = (
        float(captured_energy / total_residual_energy)
        if total_residual_energy > 1.0e-16
        else 0.0
    )

    samples: list[H2MonitorGridChargeModeCompressionSample] = []
    for index, record in enumerate(records):
        residual_coefficient = float(residual_coefficients[index])
        update_coefficient = float(update_coefficients[index])
        next_ratio = None
        if index + 1 < len(records) and abs(residual_coefficient) > 1.0e-16:
            next_ratio = float(residual_coefficients[index + 1] / residual_coefficient)
        update_to_residual_ratio = None
        if abs(residual_coefficient) > 1.0e-16:
            update_to_residual_ratio = float(update_coefficient / residual_coefficient)
        signal = signals[index] if index < len(signals) else None
        charge_mixing = (
            float(charge_mixing_history[index])
            if index < len(charge_mixing_history)
            else 0.0
        )
        samples.append(
            H2MonitorGridChargeModeCompressionSample(
                iteration=int(record.iteration),
                density_residual=float(record.density_residual),
                residual_ratio=(
                    None if signal is None or signal.density_residual_ratio is None else float(signal.density_residual_ratio)
                ),
                hartree_share=(
                    None if signal is None or signal.hartree_share is None else float(signal.hartree_share)
                ),
                charge_mixing=charge_mixing,
                mode_residual_coefficient=residual_coefficient,
                mode_update_coefficient=update_coefficient,
                update_to_residual_ratio=update_to_residual_ratio,
                realized_next_mode_ratio=next_ratio,
            )
        )

    current_sample = samples[-1]
    previous_sample = samples[-2] if len(samples) >= 2 else None
    last_realized_mode_ratio = (
        None if previous_sample is None else previous_sample.realized_next_mode_ratio
    )
    current_update_to_residual_ratio = (
        0.0
        if current_sample.update_to_residual_ratio is None
        else float(current_sample.update_to_residual_ratio)
    )
    principal_mode_residual_norm = weighted_l2_norm(mode, grid_geometry=grid_geometry)
    principal_mode_peak_abs = float(np.max(np.abs(mode)))

    if last_realized_mode_ratio is None:
        verdict = "The late-step window is too short to measure a realized one-step mode ratio."
    elif abs(last_realized_mode_ratio) < 0.9:
        verdict = (
            "The dominant late-step charge mode is contracting materially under the current "
            "preconditioned update."
        )
    elif abs(last_realized_mode_ratio) < 1.05:
        verdict = (
            "The dominant late-step charge mode is near-neutral: the update stabilizes it, "
            "but only weakly compresses the plateau mode."
        )
    else:
        verdict = (
            "The dominant late-step charge mode is still weakly amplified; the current "
            "preconditioned update does not compress the platform mode enough."
        )

    return H2MonitorGridChargeModeCompressionAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=int(source_iteration_count),
        late_window_size=window_size,
        principal_mode_explained_fraction=float(explained_fraction),
        principal_mode_residual_norm=float(principal_mode_residual_norm),
        principal_mode_peak_abs=principal_mode_peak_abs,
        current_update_mode_coefficient=float(current_sample.mode_update_coefficient),
        current_update_to_residual_ratio=float(current_update_to_residual_ratio),
        last_realized_mode_ratio=(
            None if last_realized_mode_ratio is None else float(last_realized_mode_ratio)
        ),
        mode_energy_fraction_captured=float(mode_energy_fraction_captured),
        samples=tuple(samples),
        verdict=verdict,
    )


def print_h2_monitor_grid_charge_mode_compression_summary(
    result: H2MonitorGridChargeModeCompressionAuditResult,
) -> None:
    """Print a compact text summary for the charge-mode compression audit."""

    print("=== H2 Monitor-Grid Charge-Mode Compression Audit ===")
    print(f"Case: {result.case_name}")
    print(f"Spin state: {result.spin_state_label}")
    print(f"Controller: {result.controller_name}")
    print(f"Grid: {result.grid_parameter_summary}")
    print(f"Source iterations: {result.source_iteration_count}")
    print(f"Late window size: {result.late_window_size}")
    print(f"Principal-mode explained fraction: {result.principal_mode_explained_fraction:.6f}")
    print(f"Mode-energy fraction captured: {result.mode_energy_fraction_captured:.6f}")
    print(f"Current update / residual ratio on mode: {result.current_update_to_residual_ratio:.6f}")
    if result.last_realized_mode_ratio is not None:
        print(f"Last realized one-step mode ratio: {result.last_realized_mode_ratio:.6f}")
    print("Late-step samples:")
    for sample in result.samples:
        ratio_text = (
            "None"
            if sample.realized_next_mode_ratio is None
            else f"{sample.realized_next_mode_ratio:.6f}"
        )
        print(
            f"  iter {sample.iteration}: residual={sample.density_residual:.6f}, "
            f"mode_residual={sample.mode_residual_coefficient:.6f}, "
            f"mode_update={sample.mode_update_coefficient:.6f}, "
            f"next_ratio={ratio_text}, charge_mixing={sample.charge_mixing:.6f}, "
            f"hartree_share={sample.hartree_share}"
        )
    print(f"Verdict: {result.verdict}")


if __name__ == "__main__":
    summary = run_h2_monitor_grid_charge_mode_compression_audit()
    print_h2_monitor_grid_charge_mode_compression_summary(summary)
