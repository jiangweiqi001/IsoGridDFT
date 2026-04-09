"""Audit whether charge clipping / renormalization distorts the platform slow mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ops import integrate_field
from isogrid.scf.controller import ScfControllerConfig
from isogrid.scf.controller import ScfControllerState
from isogrid.scf.controller import propose_next_density

from .h2_monitor_grid_charge_mode_compression_audit import _charge_residual_field
from .h2_monitor_grid_charge_mode_compression_audit import _grid_parameter_summary
from .h2_monitor_grid_charge_mode_compression_audit import _principal_mode
from .h2_monitor_grid_charge_mode_compression_audit import _weighted_inner
from .h2_monitor_grid_scf_amplification_ablation_audit import _shared_source_result

_DEFAULT_SHAPE = (15, 15, 17)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (9.0, 9.0, 11.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 12
_DEFAULT_LATE_WINDOW_SIZE = 5


@dataclass(frozen=True)
class H2MonitorGridChargePostprocessSample:
    """One late-step sample comparing pre-clip and post-clip charge updates."""

    iteration: int
    density_residual: float
    residual_ratio: float | None
    hartree_share: float | None
    rho_charge_unbounded_trial_min_value: float
    rho_charge_trial_min_value: float
    clipped_negative_charge_integral: float
    preclip_mode_projection: float
    postclip_mode_projection: float
    mode_projection_delta: float


@dataclass(frozen=True)
class H2MonitorGridChargePostprocessAuditResult:
    """Platform-tail audit for charge trial clipping / renormalization distortion."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    late_window_size: int
    principal_mode_explained_fraction: float
    samples: tuple[H2MonitorGridChargePostprocessSample, ...]
    verdict: str


def _controller_config(controller_name: str, *, baseline_mixing: float = 0.2) -> ScfControllerConfig:
    if controller_name == "generic_charge_spin_preconditioned":
        return ScfControllerConfig.generic_charge_spin_preconditioned(
            baseline_mixing=baseline_mixing
        )
    if controller_name == "generic_charge_spin":
        return ScfControllerConfig.generic_charge_spin(baseline_mixing=baseline_mixing)
    raise ValueError(
        "Charge postprocess audit only supports generic controller routes; "
        f"received `{controller_name}`."
    )


def run_h2_monitor_grid_charge_postprocess_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    spin_label: str = "singlet",
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridChargePostprocessAuditResult:
    """Measure how much charge clipping / renormalization distorts the slow mode."""

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
        raise ValueError("Charge postprocess audit requires at least two SCF iterations.")

    controller_config = _controller_config(controller_name)
    controller_state = ScfControllerState.initial(charge_mixing=0.2, spin_mixing=0.2)
    replay_steps = []
    for record, signals in zip(source_result.history, source_result.controller_signals_history):
        step = propose_next_density(
            occupations=source_result.occupations,
            rho_up_current=record.input_rho_up,
            rho_down_current=record.input_rho_down,
            rho_up_output=record.output_rho_up,
            rho_down_output=record.output_rho_down,
            grid_geometry=grid_geometry,
            config=controller_config,
            state=controller_state,
            signals=signals,
        )
        replay_steps.append(step)
        controller_state = step.state

    window_size = max(2, min(int(late_window_size), len(source_result.history)))
    records = tuple(source_result.history[-window_size:])
    signals = tuple(source_result.controller_signals_history[-window_size:])
    steps = tuple(replay_steps[-window_size:])

    residual_fields = tuple(_charge_residual_field(record) for record in records)
    mode, explained_fraction = _principal_mode(
        residual_fields=residual_fields,
        grid_geometry=grid_geometry,
    )

    samples: list[H2MonitorGridChargePostprocessSample] = []
    for record, signal, step in zip(records, signals, steps):
        charge_current = np.asarray(record.input_rho_up + record.input_rho_down, dtype=np.float64)
        preclip_update = np.asarray(step.rho_charge_preclip_next - charge_current, dtype=np.float64)
        postclip_update = np.asarray(step.rho_charge_postclip_next - charge_current, dtype=np.float64)
        clipped_negative_charge_integral = float(
            integrate_field(
                np.maximum(-np.asarray(step.rho_charge_preclip_next, dtype=np.float64), 0.0),
                grid_geometry=grid_geometry,
            )
        )
        preclip_projection = float(
            _weighted_inner(preclip_update, mode, grid_geometry=grid_geometry)
        )
        postclip_projection = float(
            _weighted_inner(postclip_update, mode, grid_geometry=grid_geometry)
        )
        samples.append(
            H2MonitorGridChargePostprocessSample(
                iteration=int(record.iteration),
                density_residual=float(record.density_residual),
                residual_ratio=(
                    None
                    if signal.density_residual_ratio is None
                    else float(signal.density_residual_ratio)
                ),
                hartree_share=(
                    None if signal.hartree_share is None else float(signal.hartree_share)
                ),
                rho_charge_unbounded_trial_min_value=float(np.min(step.rho_charge_unbounded_trial)),
                rho_charge_trial_min_value=float(np.min(step.rho_charge_trial)),
                clipped_negative_charge_integral=clipped_negative_charge_integral,
                preclip_mode_projection=preclip_projection,
                postclip_mode_projection=postclip_projection,
                mode_projection_delta=float(postclip_projection - preclip_projection),
            )
        )

    max_clipped = max(sample.clipped_negative_charge_integral for sample in samples)
    max_projection_delta = max(abs(sample.mode_projection_delta) for sample in samples)
    if max_clipped <= 1.0e-12 and max_projection_delta <= 1.0e-12:
        verdict = (
            "Charge clipping / renormalization is numerically inactive on the platform tail; "
            "it does not materially distort the dominant slow mode."
        )
    elif max_projection_delta > max_clipped:
        verdict = (
            "The dominant slow mode is being reshaped more by clipping/postprocessing "
            "than by pure negative-density removal alone."
        )
    else:
        verdict = (
            "Charge clipping removes a measurable negative-density mass on the platform tail "
            "and correspondingly distorts the dominant slow-mode projection."
        )

    return H2MonitorGridChargePostprocessAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label=spin_label,
        controller_name=controller_name,
        source_iteration_count=int(source_iteration_count),
        late_window_size=window_size,
        principal_mode_explained_fraction=float(explained_fraction),
        samples=tuple(samples),
        verdict=verdict,
    )


def print_h2_monitor_grid_charge_postprocess_summary(
    result: H2MonitorGridChargePostprocessAuditResult,
) -> None:
    """Print a compact summary for the charge postprocess audit."""

    print("=== H2 Monitor-Grid Charge Postprocess Audit ===")
    print(f"Case: {result.case_name}")
    print(f"Spin state: {result.spin_state_label}")
    print(f"Controller: {result.controller_name}")
    print(f"Grid: {result.grid_parameter_summary}")
    print(f"Principal-mode explained fraction: {result.principal_mode_explained_fraction:.6f}")
    for sample in result.samples:
        print(
            f"  iter {sample.iteration}: raw_trial_min={sample.rho_charge_unbounded_trial_min_value:.6e}, "
            f"trial_min={sample.rho_charge_trial_min_value:.6e}, "
            f"clipped_negative_integral={sample.clipped_negative_charge_integral:.6e}, "
            f"preclip_mode_proj={sample.preclip_mode_projection:.6e}, "
            f"postclip_mode_proj={sample.postclip_mode_projection:.6e}, "
            f"delta={sample.mode_projection_delta:.6e}"
        )
    print(f"Verdict: {result.verdict}")


if __name__ == "__main__":
    summary = run_h2_monitor_grid_charge_postprocess_audit()
    print_h2_monitor_grid_charge_postprocess_summary(summary)
