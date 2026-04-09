"""Audit how the base charge trial maps into the realized next residual."""

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
class H2MonitorGridBaseTrialToRealizedResidualMapSample:
    """One late-step sample projected onto the dominant charge mode."""

    iteration: int
    mode_residual_coefficient: float
    trial_update_ratio: float | None
    charge_next_update_ratio: float | None
    postclip_update_ratio: float | None
    realized_reduction_ratio: float | None


@dataclass(frozen=True)
class H2MonitorGridBaseTrialToRealizedResidualMapAuditResult:
    """Projected map from charge trial stages to the realized next residual."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    late_window_size: int
    principal_mode_explained_fraction: float
    trial_to_charge_next_loss: float | None
    charge_next_to_postclip_loss: float | None
    postclip_to_realized_loss: float | None
    dominant_loss_stage: str
    samples: tuple[H2MonitorGridBaseTrialToRealizedResidualMapSample, ...]
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
        raise ValueError("Principal base-trial map mode has near-zero weighted norm.")
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


def run_h2_monitor_grid_base_trial_to_realized_residual_map_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridBaseTrialToRealizedResidualMapAuditResult:
    """Project the base-trial map and realized residual onto the dominant slow mode."""

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
        raise ValueError("Base-trial map audit requires at least two SCF iterations.")
    if not source_result.controller_charge_trial_history:
        raise ValueError("Controller charge-trial history is unavailable for this source result.")
    window_size = max(2, min(int(late_window_size), len(source_result.history)))
    records = tuple(source_result.history[-window_size:])
    charge_current_fields = tuple(_charge_density(record) for record in records)
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

    trial_history = tuple(source_result.controller_charge_trial_history[-window_size:])
    charge_next_history = tuple(source_result.controller_charge_preclip_next_history[-window_size:])
    postclip_history = tuple(source_result.controller_charge_postclip_next_history[-window_size:])

    trial_ratios: list[float] = []
    charge_next_ratios: list[float] = []
    postclip_ratios: list[float] = []
    realized_reduction_ratios: list[float] = []
    samples: list[H2MonitorGridBaseTrialToRealizedResidualMapSample] = []
    for index, record in enumerate(records):
        residual_coefficient = float(residual_coefficients[index])
        charge_current = np.asarray(charge_current_fields[index], dtype=np.float64)
        trial_ratio = None
        charge_next_ratio = None
        postclip_ratio = None
        realized_reduction_ratio = None
        if abs(residual_coefficient) > 1.0e-16:
            trial_field = None if trial_history[index] is None else np.asarray(trial_history[index], dtype=np.float64)
            charge_next_field = None if charge_next_history[index] is None else np.asarray(charge_next_history[index], dtype=np.float64)
            postclip_field = None if postclip_history[index] is None else np.asarray(postclip_history[index], dtype=np.float64)
            if trial_field is not None:
                trial_ratio = float(
                    _weighted_inner(trial_field - charge_current, mode, grid_geometry=grid_geometry)
                    / residual_coefficient
                )
                trial_ratios.append(trial_ratio)
            if charge_next_field is not None:
                charge_next_ratio = float(
                    _weighted_inner(charge_next_field - charge_current, mode, grid_geometry=grid_geometry)
                    / residual_coefficient
                )
                charge_next_ratios.append(charge_next_ratio)
            if postclip_field is not None:
                postclip_ratio = float(
                    _weighted_inner(postclip_field - charge_current, mode, grid_geometry=grid_geometry)
                    / residual_coefficient
                )
                postclip_ratios.append(postclip_ratio)
            if index + 1 < len(records):
                realized_reduction_ratio = float(1.0 - (residual_coefficients[index + 1] / residual_coefficient))
                realized_reduction_ratios.append(realized_reduction_ratio)
        samples.append(
            H2MonitorGridBaseTrialToRealizedResidualMapSample(
                iteration=int(record.iteration),
                mode_residual_coefficient=residual_coefficient,
                trial_update_ratio=trial_ratio,
                charge_next_update_ratio=charge_next_ratio,
                postclip_update_ratio=postclip_ratio,
                realized_reduction_ratio=realized_reduction_ratio,
            )
        )

    def _latest_abs_diff(lhs: list[float], rhs: list[float]) -> float | None:
        if not lhs or not rhs:
            return None
        return float(abs(lhs[-1] - rhs[-1]))

    trial_to_charge_next_loss = _latest_abs_diff(trial_ratios, charge_next_ratios)
    charge_next_to_postclip_loss = _latest_abs_diff(charge_next_ratios, postclip_ratios)
    postclip_to_realized_loss = _latest_abs_diff(postclip_ratios[:-1], realized_reduction_ratios)

    stage_losses = {
        "trial_to_charge_next": -1.0 if trial_to_charge_next_loss is None else trial_to_charge_next_loss,
        "charge_next_to_postclip": -1.0 if charge_next_to_postclip_loss is None else charge_next_to_postclip_loss,
        "postclip_to_realized": -1.0 if postclip_to_realized_loss is None else postclip_to_realized_loss,
    }
    dominant_loss_stage = max(stage_losses, key=stage_losses.get)

    if stage_losses[dominant_loss_stage] <= 0.0:
        verdict = "The late-step window is too short to identify a dominant base-trial map mismatch stage."
    elif dominant_loss_stage == "trial_to_charge_next":
        verdict = (
            "The largest projected loss occurs before charge renormalization: the raw charge trial is "
            "already diverging from the renormalized charge update."
        )
    elif dominant_loss_stage == "charge_next_to_postclip":
        verdict = (
            "The largest projected loss occurs during charge-to-spin recombination / postclip density assembly."
        )
    else:
        verdict = (
            "The largest projected loss occurs after postclip density assembly: the realized next residual map "
            "does not follow the controller-side charge update."
        )

    return H2MonitorGridBaseTrialToRealizedResidualMapAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=source_iteration_count,
        late_window_size=window_size,
        principal_mode_explained_fraction=float(explained_fraction),
        trial_to_charge_next_loss=trial_to_charge_next_loss,
        charge_next_to_postclip_loss=charge_next_to_postclip_loss,
        postclip_to_realized_loss=postclip_to_realized_loss,
        dominant_loss_stage=dominant_loss_stage,
        samples=tuple(samples),
        verdict=verdict,
    )


def print_h2_monitor_grid_base_trial_to_realized_residual_map_summary(
    result: H2MonitorGridBaseTrialToRealizedResidualMapAuditResult,
) -> None:
    """Print a compact summary of the base-trial map audit."""

    print("=== H2 monitor-grid base-trial to realized-residual map audit ===")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    print(f"source iterations: {result.source_iteration_count}")
    print(f"late window size: {result.late_window_size}")
    print(f"principal_mode_explained_fraction: {result.principal_mode_explained_fraction:.6f}")
    print(
        "trial_to_charge_next_loss: "
        + ("None" if result.trial_to_charge_next_loss is None else f"{result.trial_to_charge_next_loss:.6e}")
    )
    print(
        "charge_next_to_postclip_loss: "
        + (
            "None"
            if result.charge_next_to_postclip_loss is None
            else f"{result.charge_next_to_postclip_loss:.6e}"
        )
    )
    print(
        "postclip_to_realized_loss: "
        + (
            "None"
            if result.postclip_to_realized_loss is None
            else f"{result.postclip_to_realized_loss:.6e}"
        )
    )
    print(f"dominant_loss_stage: {result.dominant_loss_stage}")
    print(f"verdict: {result.verdict}")
