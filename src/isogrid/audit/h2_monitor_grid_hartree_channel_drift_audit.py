"""Lightweight pre/post boundary-fix comparison for early H2 Hartree-channel drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ks import FixedPotentialStaticLocalOperatorContext
from isogrid.ks import prepare_fixed_potential_static_local_operator
from isogrid.ops import integrate_field
from isogrid.scf import H2StaticLocalScfDryRunResult
from isogrid.scf import run_h2_monitor_grid_scf_dry_run
from isogrid.scf.driver import _default_monitor_patch_parameters
from isogrid.scf.driver import _estimate_singlet_hartree_tail_channel_shares

_DEFAULT_SHAPE = (9, 9, 11)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (6.0, 6.0, 8.0)
_DEFAULT_ITERATION_LIMIT = 3
_DEFAULT_MIXING = 0.20
_DEFAULT_DENSITY_TOLERANCE = 1.0e-2
_DEFAULT_ENERGY_TOLERANCE = 1.0e-4
_DEFAULT_EIGENSOLVER_TOLERANCE = 1.0e-2
_DEFAULT_EIGENSOLVER_NCV = 8
_DEFAULT_FAR_FIELD_FRACTION = 0.60


@dataclass(frozen=True)
class H2MonitorGridHartreeChannelDriftStep:
    """One SCF-step Hartree drift snapshot for the fixed lightweight case."""

    iteration: int
    density_residual: float
    boundary_value_correction_rms: float | None
    boundary_value_correction_max_abs: float | None
    hartree_tail_far_field_mean_abs: float
    hartree_tail_far_field_signed_mean: float
    effective_hartree_potential_rms: float
    effective_total_potential_rms: float
    effective_hartree_component_share: float | None


@dataclass(frozen=True)
class H2MonitorGridHartreeChannelDriftSecantPair:
    """Consecutive-iteration Hartree share summary."""

    pair_iterations: tuple[int, int]
    density_secant_norm: float
    density_residual_ratio: float | None
    hartree_contribution_share: float | None
    xc_contribution_share: float | None
    local_orbital_contribution_share: float | None


@dataclass(frozen=True)
class H2MonitorGridHartreeChannelDriftRoute:
    """One spin/mode route for the early-step Hartree drift audit."""

    spin_state_label: str
    boundary_construction_mode: str
    iteration_limit: int
    steps: tuple[H2MonitorGridHartreeChannelDriftStep, ...]
    secant_pairs: tuple[H2MonitorGridHartreeChannelDriftSecantPair, ...]


@dataclass(frozen=True)
class H2MonitorGridHartreeChannelDriftComparison:
    """Before/after comparison for one spin state."""

    spin_state_label: str
    legacy_split: H2MonitorGridHartreeChannelDriftRoute
    corrected_moments: H2MonitorGridHartreeChannelDriftRoute
    boundary_value_correction_rms_reduction: float | None
    final_density_residual_change: float
    last_secant_hartree_share_change: float | None
    verdict: str


@dataclass(frozen=True)
class H2MonitorGridHartreeChannelDriftAuditResult:
    """Top-level lightweight audit result."""

    case_name: str
    grid_parameter_summary: str
    singlet: H2MonitorGridHartreeChannelDriftComparison
    triplet: H2MonitorGridHartreeChannelDriftComparison
    note: str


def _weighted_field_norm(field: np.ndarray, *, grid_geometry: MonitorGridGeometry) -> float:
    value = float(integrate_field(field * field, grid_geometry=grid_geometry))
    return float(np.sqrt(max(value, 0.0)))


def _far_field_mask(grid_geometry: MonitorGridGeometry) -> np.ndarray:
    bounds = grid_geometry.spec.box_bounds
    center_x = 0.5 * (bounds[0][0] + bounds[0][1])
    center_y = 0.5 * (bounds[1][0] + bounds[1][1])
    center_z = 0.5 * (bounds[2][0] + bounds[2][1])
    half_extent_x = 0.5 * (bounds[0][1] - bounds[0][0])
    half_extent_y = 0.5 * (bounds[1][1] - bounds[1][0])
    half_extent_z = 0.5 * (bounds[2][1] - bounds[2][0])
    return (
        (np.abs(grid_geometry.x_points - center_x) >= _DEFAULT_FAR_FIELD_FRACTION * half_extent_x)
        | (np.abs(grid_geometry.y_points - center_y) >= _DEFAULT_FAR_FIELD_FRACTION * half_extent_y)
        | (np.abs(grid_geometry.z_points - center_z) >= _DEFAULT_FAR_FIELD_FRACTION * half_extent_z)
    )


def _build_iteration_context(
    *,
    record,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    monitor_boundary_construction_mode: str,
) -> FixedPotentialStaticLocalOperatorContext:
    return prepare_fixed_potential_static_local_operator(
        grid_geometry=grid_geometry,
        rho_up=record.input_rho_up,
        rho_down=record.input_rho_down,
        spin_channel="up",
        case=case,
        use_monitor_patch=True,
        patch_parameters=_default_monitor_patch_parameters(),
        kinetic_version="trial_fix",
        hartree_backend="python",
        monitor_boundary_construction_mode=monitor_boundary_construction_mode,
    )


def _effective_hartree_component_share(
    context: FixedPotentialStaticLocalOperatorContext,
) -> float | None:
    hartree_norm = _weighted_field_norm(
        context.hartree_potential,
        grid_geometry=context.grid_geometry,
    )
    xc_norm = _weighted_field_norm(
        context.xc_potential,
        grid_geometry=context.grid_geometry,
    )
    local_norm = _weighted_field_norm(
        context.local_ionic_potential,
        grid_geometry=context.grid_geometry,
    )
    denominator = hartree_norm + xc_norm + local_norm
    if denominator <= 1.0e-16:
        return None
    return float(hartree_norm / denominator)


def _build_step(
    *,
    record,
    snapshot,
    context: FixedPotentialStaticLocalOperatorContext,
) -> H2MonitorGridHartreeChannelDriftStep:
    far_field_mask = _far_field_mask(context.grid_geometry)
    hartree_tail_values = np.asarray(context.hartree_potential[far_field_mask], dtype=np.float64)
    return H2MonitorGridHartreeChannelDriftStep(
        iteration=int(record.iteration),
        density_residual=float(record.density_residual),
        boundary_value_correction_rms=(
            None
            if snapshot.boundary_value_correction_rms is None
            else float(snapshot.boundary_value_correction_rms)
        ),
        boundary_value_correction_max_abs=(
            None
            if snapshot.boundary_value_correction_max_abs is None
            else float(snapshot.boundary_value_correction_max_abs)
        ),
        hartree_tail_far_field_mean_abs=float(np.mean(np.abs(hartree_tail_values))),
        hartree_tail_far_field_signed_mean=float(np.mean(hartree_tail_values)),
        effective_hartree_potential_rms=_weighted_field_norm(
            context.hartree_potential,
            grid_geometry=context.grid_geometry,
        ),
        effective_total_potential_rms=_weighted_field_norm(
            context.effective_local_potential,
            grid_geometry=context.grid_geometry,
        ),
        effective_hartree_component_share=_effective_hartree_component_share(context),
    )


def _build_secant_pair(
    *,
    previous_record,
    current_record,
    previous_context: FixedPotentialStaticLocalOperatorContext,
    current_context: FixedPotentialStaticLocalOperatorContext,
) -> H2MonitorGridHartreeChannelDriftSecantPair:
    delta_up = np.asarray(
        current_record.input_rho_up - previous_record.input_rho_up,
        dtype=np.float64,
    )
    delta_down = np.asarray(
        current_record.input_rho_down - previous_record.input_rho_down,
        dtype=np.float64,
    )
    density_secant_norm = float(
        np.sqrt(
            max(
                float(
                    integrate_field(
                        delta_up * delta_up + delta_down * delta_down,
                        grid_geometry=current_context.grid_geometry,
                    )
                ),
                0.0,
            )
        )
    )
    previous_residual = float(previous_record.density_residual)
    current_residual = float(current_record.density_residual)
    residual_ratio = None
    if previous_residual > 1.0e-16:
        residual_ratio = float(current_residual / previous_residual)
    hartree_share, xc_share, local_share = _estimate_singlet_hartree_tail_channel_shares(
        input_context=previous_context,
        output_context=current_context,
    )
    return H2MonitorGridHartreeChannelDriftSecantPair(
        pair_iterations=(int(previous_record.iteration), int(current_record.iteration)),
        density_secant_norm=float(density_secant_norm),
        density_residual_ratio=residual_ratio,
        hartree_contribution_share=hartree_share,
        xc_contribution_share=xc_share,
        local_orbital_contribution_share=local_share,
    )


def _build_route(
    *,
    spin_label: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    iteration_limit: int,
    monitor_boundary_construction_mode: str,
) -> H2MonitorGridHartreeChannelDriftRoute:
    result: H2StaticLocalScfDryRunResult = run_h2_monitor_grid_scf_dry_run(
        spin_label,
        case=case,
        grid_geometry=grid_geometry,
        max_iterations=iteration_limit,
        mixing=_DEFAULT_MIXING,
        density_tolerance=_DEFAULT_DENSITY_TOLERANCE,
        energy_tolerance=_DEFAULT_ENERGY_TOLERANCE,
        eigensolver_tolerance=_DEFAULT_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_DEFAULT_EIGENSOLVER_NCV,
        kinetic_version="trial_fix",
        hartree_backend="python",
        monitor_boundary_construction_mode=monitor_boundary_construction_mode,
    )
    contexts = tuple(
        _build_iteration_context(
            record=record,
            case=case,
            grid_geometry=grid_geometry,
            monitor_boundary_construction_mode=monitor_boundary_construction_mode,
        )
        for record in result.history
    )
    steps = tuple(
        _build_step(record=record, snapshot=snapshot, context=context)
        for record, snapshot, context in zip(
            result.history,
            result.hartree_response_diagnostics_history,
            contexts,
            strict=True,
        )
    )
    secant_pairs = tuple(
        _build_secant_pair(
            previous_record=result.history[index - 1],
            current_record=result.history[index],
            previous_context=contexts[index - 1],
            current_context=contexts[index],
        )
        for index in range(1, len(result.history))
    )
    return H2MonitorGridHartreeChannelDriftRoute(
        spin_state_label=spin_label,
        boundary_construction_mode=monitor_boundary_construction_mode,
        iteration_limit=int(iteration_limit),
        steps=steps,
        secant_pairs=secant_pairs,
    )


def _comparison_verdict(
    *,
    spin_label: str,
    legacy_route: H2MonitorGridHartreeChannelDriftRoute,
    corrected_route: H2MonitorGridHartreeChannelDriftRoute,
) -> tuple[float | None, float, float | None, str]:
    legacy_boundary = legacy_route.steps[-1].boundary_value_correction_rms
    corrected_boundary = corrected_route.steps[-1].boundary_value_correction_rms
    boundary_reduction = None
    if legacy_boundary is not None and corrected_boundary is not None:
        boundary_reduction = float(legacy_boundary - corrected_boundary)

    final_density_residual_change = float(
        corrected_route.steps[-1].density_residual - legacy_route.steps[-1].density_residual
    )
    legacy_hartree_share = (
        None
        if not legacy_route.secant_pairs
        else legacy_route.secant_pairs[-1].hartree_contribution_share
    )
    corrected_hartree_share = (
        None
        if not corrected_route.secant_pairs
        else corrected_route.secant_pairs[-1].hartree_contribution_share
    )
    hartree_share_change = None
    if legacy_hartree_share is not None and corrected_hartree_share is not None:
        hartree_share_change = float(corrected_hartree_share - legacy_hartree_share)

    if spin_label == "singlet":
        if (
            hartree_share_change is not None
            and hartree_share_change < -1.0e-3
            and final_density_residual_change <= 0.0
        ):
            verdict = (
                "corrected boundary reduces the singlet Hartree share on the last early secant pair "
                "and does not worsen the early density residual"
            )
        elif hartree_share_change is not None and hartree_share_change < -1.0e-3:
            verdict = (
                "corrected boundary lowers the singlet Hartree share, but the early density residual "
                "has not improved yet"
            )
        else:
            verdict = (
                "no clear early singlet Hartree-share reduction is visible in this lightweight audit"
            )
    else:
        if final_density_residual_change <= 0.05 * max(legacy_route.steps[-1].density_residual, 1.0e-12):
            verdict = "triplet early residual remains stable under the corrected boundary"
        else:
            verdict = "triplet early residual worsens materially under the corrected boundary"
    return boundary_reduction, final_density_residual_change, hartree_share_change, verdict


def _build_comparison(
    *,
    spin_label: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    iteration_limit: int,
) -> H2MonitorGridHartreeChannelDriftComparison:
    legacy_route = _build_route(
        spin_label=spin_label,
        case=case,
        grid_geometry=grid_geometry,
        iteration_limit=iteration_limit,
        monitor_boundary_construction_mode="legacy_split",
    )
    corrected_route = _build_route(
        spin_label=spin_label,
        case=case,
        grid_geometry=grid_geometry,
        iteration_limit=iteration_limit,
        monitor_boundary_construction_mode="corrected_moments",
    )
    (
        boundary_reduction,
        final_density_residual_change,
        hartree_share_change,
        verdict,
    ) = _comparison_verdict(
        spin_label=spin_label,
        legacy_route=legacy_route,
        corrected_route=corrected_route,
    )
    return H2MonitorGridHartreeChannelDriftComparison(
        spin_state_label=spin_label,
        legacy_split=legacy_route,
        corrected_moments=corrected_route,
        boundary_value_correction_rms_reduction=boundary_reduction,
        final_density_residual_change=final_density_residual_change,
        last_secant_hartree_share_change=hartree_share_change,
        verdict=verdict,
    )


def run_h2_monitor_grid_hartree_channel_drift_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    iteration_limit: int = _DEFAULT_ITERATION_LIMIT,
) -> H2MonitorGridHartreeChannelDriftAuditResult:
    """Compare legacy and corrected monitor-boundary modes on early H2 SCF steps."""

    if grid_geometry is None:
        grid_geometry = build_monitor_grid_for_case(
            case,
            shape=_DEFAULT_SHAPE,
            box_half_extents=_DEFAULT_BOX_HALF_EXTENTS_BOHR,
            element_parameters=build_h2_local_patch_development_element_parameters(),
        )
    bounds = grid_geometry.spec.box_bounds
    box_half_extents_bohr = (
        0.5 * float(bounds[0][1] - bounds[0][0]),
        0.5 * float(bounds[1][1] - bounds[1][0]),
        0.5 * float(bounds[2][1] - bounds[2][0]),
    )
    singlet = _build_comparison(
        spin_label="singlet",
        case=case,
        grid_geometry=grid_geometry,
        iteration_limit=iteration_limit,
    )
    triplet = _build_comparison(
        spin_label="triplet",
        case=case,
        grid_geometry=grid_geometry,
        iteration_limit=iteration_limit,
    )
    return H2MonitorGridHartreeChannelDriftAuditResult(
        case_name=case.name,
        grid_parameter_summary=(
            f"shape={grid_geometry.spec.shape}, "
            f"box_half_extents_bohr={box_half_extents_bohr}"
        ),
        singlet=singlet,
        triplet=triplet,
        note=(
            "This audit intentionally stays lightweight: local-only H2, fixed small monitor grid, "
            "python Hartree backend, first 2-3 SCF steps only, no extra stabilizers."
        ),
    )


def _print_route(route: H2MonitorGridHartreeChannelDriftRoute) -> None:
    print(
        f"route: spin={route.spin_state_label}, "
        f"boundary_mode={route.boundary_construction_mode}, "
        f"steps={route.iteration_limit}"
    )
    for step in route.steps:
        print(
            f"  step {step.iteration}: "
            f"density_residual={step.density_residual:.12e}, "
            f"boundary_correction_rms={step.boundary_value_correction_rms}, "
            f"hartree_tail_mean_abs={step.hartree_tail_far_field_mean_abs:.12e}, "
            f"hartree_tail_signed_mean={step.hartree_tail_far_field_signed_mean:.12e}, "
            f"v_h_rms={step.effective_hartree_potential_rms:.12e}, "
            f"v_eff_rms={step.effective_total_potential_rms:.12e}, "
            f"hartree_component_share={step.effective_hartree_component_share}"
        )
    for secant in route.secant_pairs:
        print(
            f"  secant {secant.pair_iterations[0]}->{secant.pair_iterations[1]}: "
            f"density_secant_norm={secant.density_secant_norm:.12e}, "
            f"density_residual_ratio={secant.density_residual_ratio}, "
            f"hartree_share={secant.hartree_contribution_share}, "
            f"xc_share={secant.xc_contribution_share}, "
            f"local_share={secant.local_orbital_contribution_share}"
        )


def _print_comparison(comparison: H2MonitorGridHartreeChannelDriftComparison) -> None:
    print(f"spin: {comparison.spin_state_label}")
    _print_route(comparison.legacy_split)
    _print_route(comparison.corrected_moments)
    print(
        "  comparison: "
        f"boundary_correction_rms_reduction={comparison.boundary_value_correction_rms_reduction}, "
        f"final_density_residual_change={comparison.final_density_residual_change:+.12e}, "
        f"last_secant_hartree_share_change={comparison.last_secant_hartree_share_change}"
    )
    print(f"  verdict: {comparison.verdict}")


def print_h2_monitor_grid_hartree_channel_drift_summary(
    result: H2MonitorGridHartreeChannelDriftAuditResult,
) -> None:
    """Print a compact early-step Hartree-channel drift comparison."""

    print("IsoGridDFT H2 monitor-grid Hartree-channel drift audit")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"note: {result.note}")
    print()
    _print_comparison(result.singlet)
    print()
    _print_comparison(result.triplet)


def main() -> int:
    result = run_h2_monitor_grid_hartree_channel_drift_audit()
    print_h2_monitor_grid_hartree_channel_drift_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
