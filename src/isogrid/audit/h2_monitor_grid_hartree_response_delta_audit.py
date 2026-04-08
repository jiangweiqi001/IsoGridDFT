"""Fixed-density legacy-vs-corrected Hartree response delta audit on the monitor grid."""

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

_DEFAULT_SHAPE = (9, 9, 11)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (6.0, 6.0, 8.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 2
_DEFAULT_MIXING = 0.20
_DEFAULT_DENSITY_TOLERANCE = 1.0e-2
_DEFAULT_ENERGY_TOLERANCE = 1.0e-4
_DEFAULT_EIGENSOLVER_TOLERANCE = 1.0e-2
_DEFAULT_EIGENSOLVER_NCV = 8
_DEFAULT_FAR_FIELD_FRACTION = 0.60
_ROUTE_MODES = ("legacy_split", "corrected_moments")


@dataclass(frozen=True)
class H2MonitorGridHartreeResponseDeltaSnapshot:
    """Hartree/open-boundary summary for one fixed-density snapshot and one boundary mode."""

    source_snapshot_label: str
    density_electrons: float
    boundary_value_correction_rms: float | None
    corrected_moment_boundary_rms_mismatch: float | None
    boundary_value_rms: float | None
    boundary_source_laplacian_rms: float | None
    rhs_l2_norm: float | None
    interior_potential_rms: float | None
    interior_poisson_residual_l2_norm: float | None
    interior_poisson_residual_max_abs: float | None
    hartree_tail_far_field_mean_abs: float
    hartree_tail_far_field_signed_mean: float
    hartree_potential_rms: float
    effective_hartree_component_share: float | None


@dataclass(frozen=True)
class H2MonitorGridHartreeResponseDeltaRoute:
    """All fixed-density snapshots for one spin state and one boundary mode."""

    spin_state_label: str
    boundary_construction_mode: str
    source_snapshot_count: int
    snapshots: tuple[H2MonitorGridHartreeResponseDeltaSnapshot, ...]


@dataclass(frozen=True)
class H2MonitorGridHartreeResponseDeltaComparison:
    """Legacy-vs-corrected delta for one shared input-density snapshot."""

    spin_state_label: str
    source_snapshot_label: str
    legacy_split: H2MonitorGridHartreeResponseDeltaSnapshot
    corrected_moments: H2MonitorGridHartreeResponseDeltaSnapshot
    hartree_potential_delta_rms: float
    hartree_potential_relative_delta: float | None
    effective_potential_delta_rms: float
    boundary_mismatch_delta: float | None
    boundary_source_laplacian_rms_delta: float | None
    interior_potential_rms_delta: float | None
    hartree_tail_mean_abs_delta: float
    verdict: str


@dataclass(frozen=True)
class H2MonitorGridHartreeResponseDeltaAuditResult:
    """Top-level fixed-density Hartree response delta audit result."""

    case_name: str
    grid_parameter_summary: str
    singlet_routes: tuple[
        H2MonitorGridHartreeResponseDeltaRoute,
        H2MonitorGridHartreeResponseDeltaRoute,
    ]
    triplet_routes: tuple[
        H2MonitorGridHartreeResponseDeltaRoute,
        H2MonitorGridHartreeResponseDeltaRoute,
    ]
    singlet_comparisons: tuple[H2MonitorGridHartreeResponseDeltaComparison, ...]
    triplet_comparisons: tuple[H2MonitorGridHartreeResponseDeltaComparison, ...]
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


def _shared_source_result(
    *,
    spin_label: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    source_iteration_count: int,
) -> H2StaticLocalScfDryRunResult:
    return run_h2_monitor_grid_scf_dry_run(
        spin_label,
        case=case,
        grid_geometry=grid_geometry,
        max_iterations=source_iteration_count,
        mixing=_DEFAULT_MIXING,
        density_tolerance=_DEFAULT_DENSITY_TOLERANCE,
        energy_tolerance=_DEFAULT_ENERGY_TOLERANCE,
        eigensolver_tolerance=_DEFAULT_EIGENSOLVER_TOLERANCE,
        eigensolver_ncv=_DEFAULT_EIGENSOLVER_NCV,
        kinetic_version="trial_fix",
        hartree_backend="python",
        monitor_boundary_construction_mode="corrected_moments",
    )


def _source_spin_channel(spin_label: str) -> str:
    # Hartree itself is spin-independent; use the up-channel effective potential
    # consistently so the Hartree share is comparable across snapshots.
    del spin_label
    return "up"


def _build_context(
    *,
    spin_label: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    monitor_boundary_construction_mode: str,
) -> FixedPotentialStaticLocalOperatorContext:
    return prepare_fixed_potential_static_local_operator(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel=_source_spin_channel(spin_label),
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


def _build_snapshot(
    *,
    source_snapshot_label: str,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
    context: FixedPotentialStaticLocalOperatorContext,
) -> H2MonitorGridHartreeResponseDeltaSnapshot:
    poisson_result = context.hartree_poisson_result
    if poisson_result is None:
        raise ValueError("Expected Hartree Poisson diagnostics for the fixed-density monitor-grid audit.")
    boundary_diagnostics = poisson_result.boundary_condition.diagnostics
    response_diagnostics = poisson_result.response_diagnostics
    far_field_mask = _far_field_mask(context.grid_geometry)
    hartree_tail_values = np.asarray(context.hartree_potential[far_field_mask], dtype=np.float64)
    density_electrons = float(
        integrate_field(rho_up + rho_down, grid_geometry=context.grid_geometry)
    )
    return H2MonitorGridHartreeResponseDeltaSnapshot(
        source_snapshot_label=source_snapshot_label,
        density_electrons=density_electrons,
        boundary_value_correction_rms=(
            None
            if boundary_diagnostics is None
            else float(boundary_diagnostics.boundary_value_correction_rms)
        ),
        corrected_moment_boundary_rms_mismatch=(
            None
            if boundary_diagnostics is None
            else float(boundary_diagnostics.corrected_moment_boundary_rms_mismatch)
        ),
        boundary_value_rms=(
            None
            if response_diagnostics is None
            else float(response_diagnostics.boundary_value_rms)
        ),
        boundary_source_laplacian_rms=(
            None
            if response_diagnostics is None
            else float(response_diagnostics.boundary_source_laplacian_rms)
        ),
        rhs_l2_norm=(
            None
            if response_diagnostics is None
            else float(response_diagnostics.rhs_l2_norm)
        ),
        interior_potential_rms=(
            None
            if response_diagnostics is None
            else float(response_diagnostics.interior_potential_rms)
        ),
        interior_poisson_residual_l2_norm=(
            None
            if response_diagnostics is None
            else float(response_diagnostics.interior_poisson_residual_l2_norm)
        ),
        interior_poisson_residual_max_abs=(
            None
            if response_diagnostics is None
            else float(response_diagnostics.interior_poisson_residual_max_abs)
        ),
        hartree_tail_far_field_mean_abs=float(np.mean(np.abs(hartree_tail_values))),
        hartree_tail_far_field_signed_mean=float(np.mean(hartree_tail_values)),
        hartree_potential_rms=_weighted_field_norm(
            context.hartree_potential,
            grid_geometry=context.grid_geometry,
        ),
        effective_hartree_component_share=_effective_hartree_component_share(context),
    )


def _build_route(
    *,
    spin_label: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    source_result: H2StaticLocalScfDryRunResult,
    monitor_boundary_construction_mode: str,
) -> tuple[H2MonitorGridHartreeResponseDeltaRoute, tuple[FixedPotentialStaticLocalOperatorContext, ...]]:
    contexts: list[FixedPotentialStaticLocalOperatorContext] = []
    snapshots: list[H2MonitorGridHartreeResponseDeltaSnapshot] = []
    for record in source_result.history:
        context = _build_context(
            spin_label=spin_label,
            case=case,
            grid_geometry=grid_geometry,
            rho_up=record.input_rho_up,
            rho_down=record.input_rho_down,
            monitor_boundary_construction_mode=monitor_boundary_construction_mode,
        )
        contexts.append(context)
        snapshots.append(
            _build_snapshot(
                source_snapshot_label=f"iter{int(record.iteration)}_input",
                rho_up=record.input_rho_up,
                rho_down=record.input_rho_down,
                context=context,
            )
        )
    return (
        H2MonitorGridHartreeResponseDeltaRoute(
            spin_state_label=spin_label,
            boundary_construction_mode=monitor_boundary_construction_mode,
            source_snapshot_count=len(snapshots),
            snapshots=tuple(snapshots),
        ),
        tuple(contexts),
    )


def _safe_delta(right: float | None, left: float | None) -> float | None:
    if right is None or left is None:
        return None
    return float(right - left)


def _comparison_verdict(
    *,
    comparison: H2MonitorGridHartreeResponseDeltaComparison,
) -> str:
    if (
        comparison.boundary_mismatch_delta is not None
        and comparison.boundary_mismatch_delta < -1.0e-8
        and comparison.hartree_potential_relative_delta is not None
        and comparison.hartree_potential_relative_delta <= 1.0e-3
    ):
        return (
            "boundary self-consistency improves strongly, but the same-density Hartree response "
            "moves only weakly"
        )
    if (
        comparison.hartree_potential_relative_delta is not None
        and comparison.hartree_potential_relative_delta > 1.0e-3
    ):
        return "same-density Hartree response moves materially under the corrected boundary"
    return "legacy and corrected routes remain close at the Poisson/Hartree response layer"


def _build_comparisons(
    *,
    spin_label: str,
    legacy_route: H2MonitorGridHartreeResponseDeltaRoute,
    corrected_route: H2MonitorGridHartreeResponseDeltaRoute,
    legacy_contexts: tuple[FixedPotentialStaticLocalOperatorContext, ...],
    corrected_contexts: tuple[FixedPotentialStaticLocalOperatorContext, ...],
) -> tuple[H2MonitorGridHartreeResponseDeltaComparison, ...]:
    comparisons: list[H2MonitorGridHartreeResponseDeltaComparison] = []
    for legacy_snapshot, corrected_snapshot, legacy_context, corrected_context in zip(
        legacy_route.snapshots,
        corrected_route.snapshots,
        legacy_contexts,
        corrected_contexts,
        strict=True,
    ):
        comparison = H2MonitorGridHartreeResponseDeltaComparison(
            spin_state_label=spin_label,
            source_snapshot_label=legacy_snapshot.source_snapshot_label,
            legacy_split=legacy_snapshot,
            corrected_moments=corrected_snapshot,
            hartree_potential_delta_rms=_weighted_field_norm(
                corrected_context.hartree_potential - legacy_context.hartree_potential,
                grid_geometry=legacy_context.grid_geometry,
            ),
            hartree_potential_relative_delta=None,
            effective_potential_delta_rms=_weighted_field_norm(
                corrected_context.effective_local_potential
                - legacy_context.effective_local_potential,
                grid_geometry=legacy_context.grid_geometry,
            ),
            boundary_mismatch_delta=_safe_delta(
                corrected_snapshot.corrected_moment_boundary_rms_mismatch,
                legacy_snapshot.corrected_moment_boundary_rms_mismatch,
            ),
            boundary_source_laplacian_rms_delta=_safe_delta(
                corrected_snapshot.boundary_source_laplacian_rms,
                legacy_snapshot.boundary_source_laplacian_rms,
            ),
            interior_potential_rms_delta=_safe_delta(
                corrected_snapshot.interior_potential_rms,
                legacy_snapshot.interior_potential_rms,
            ),
            hartree_tail_mean_abs_delta=float(
                corrected_snapshot.hartree_tail_far_field_mean_abs
                - legacy_snapshot.hartree_tail_far_field_mean_abs
            ),
            verdict="",
        )
        comparisons.append(
            H2MonitorGridHartreeResponseDeltaComparison(
                spin_state_label=comparison.spin_state_label,
                source_snapshot_label=comparison.source_snapshot_label,
                legacy_split=comparison.legacy_split,
                corrected_moments=comparison.corrected_moments,
                hartree_potential_delta_rms=comparison.hartree_potential_delta_rms,
                hartree_potential_relative_delta=(
                    None
                    if corrected_snapshot.hartree_potential_rms <= 1.0e-16
                    else float(
                        comparison.hartree_potential_delta_rms
                        / corrected_snapshot.hartree_potential_rms
                    )
                ),
                effective_potential_delta_rms=comparison.effective_potential_delta_rms,
                boundary_mismatch_delta=comparison.boundary_mismatch_delta,
                boundary_source_laplacian_rms_delta=comparison.boundary_source_laplacian_rms_delta,
                interior_potential_rms_delta=comparison.interior_potential_rms_delta,
                hartree_tail_mean_abs_delta=comparison.hartree_tail_mean_abs_delta,
                verdict=_comparison_verdict(comparison=comparison),
            )
        )
    return tuple(comparisons)


def _run_spin_audit(
    *,
    spin_label: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    source_iteration_count: int,
) -> tuple[
    tuple[H2MonitorGridHartreeResponseDeltaRoute, H2MonitorGridHartreeResponseDeltaRoute],
    tuple[H2MonitorGridHartreeResponseDeltaComparison, ...],
]:
    source_result = _shared_source_result(
        spin_label=spin_label,
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
    )
    (legacy_route, legacy_contexts) = _build_route(
        spin_label=spin_label,
        case=case,
        grid_geometry=grid_geometry,
        source_result=source_result,
        monitor_boundary_construction_mode="legacy_split",
    )
    (corrected_route, corrected_contexts) = _build_route(
        spin_label=spin_label,
        case=case,
        grid_geometry=grid_geometry,
        source_result=source_result,
        monitor_boundary_construction_mode="corrected_moments",
    )
    comparisons = _build_comparisons(
        spin_label=spin_label,
        legacy_route=legacy_route,
        corrected_route=corrected_route,
        legacy_contexts=legacy_contexts,
        corrected_contexts=corrected_contexts,
    )
    return (legacy_route, corrected_route), comparisons


def run_h2_monitor_grid_hartree_response_delta_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
) -> H2MonitorGridHartreeResponseDeltaAuditResult:
    """Compare legacy and corrected Poisson/Hartree response on shared density snapshots."""

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
    singlet_routes, singlet_comparisons = _run_spin_audit(
        spin_label="singlet",
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
    )
    triplet_routes, triplet_comparisons = _run_spin_audit(
        spin_label="triplet",
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
    )
    return H2MonitorGridHartreeResponseDeltaAuditResult(
        case_name=case.name,
        grid_parameter_summary=(
            f"shape={grid_geometry.spec.shape}, "
            f"box_half_extents_bohr={box_half_extents_bohr}"
        ),
        singlet_routes=singlet_routes,
        triplet_routes=triplet_routes,
        singlet_comparisons=singlet_comparisons,
        triplet_comparisons=triplet_comparisons,
        note=(
            "This audit compares legacy vs corrected monitor-grid boundary construction on the same "
            "shared early input-density snapshots. The source snapshots come from a short corrected "
            "local-only dry-run; the comparison itself is fixed-density Hartree/open-boundary only."
        ),
    )


def _print_route(route: H2MonitorGridHartreeResponseDeltaRoute) -> None:
    print(
        f"route: spin={route.spin_state_label}, "
        f"boundary_mode={route.boundary_construction_mode}, "
        f"snapshots={route.source_snapshot_count}"
    )
    for snapshot in route.snapshots:
        print(
            f"  {snapshot.source_snapshot_label}: "
            f"electrons={snapshot.density_electrons:.12f}, "
            f"boundary_correction_rms={snapshot.boundary_value_correction_rms}, "
            f"boundary_mismatch={snapshot.corrected_moment_boundary_rms_mismatch}, "
            f"boundary_source_rms={snapshot.boundary_source_laplacian_rms}, "
            f"interior_v_rms={snapshot.interior_potential_rms}, "
            f"poisson_residual_l2={snapshot.interior_poisson_residual_l2_norm}, "
            f"hartree_tail_mean_abs={snapshot.hartree_tail_far_field_mean_abs:.12e}, "
            f"hartree_rms={snapshot.hartree_potential_rms:.12e}, "
            f"hartree_component_share={snapshot.effective_hartree_component_share}"
        )


def _print_comparisons(
    comparisons: tuple[H2MonitorGridHartreeResponseDeltaComparison, ...],
) -> None:
    for comparison in comparisons:
        print(
            f"  compare {comparison.source_snapshot_label}: "
            f"boundary_mismatch_delta={comparison.boundary_mismatch_delta}, "
            f"boundary_source_delta={comparison.boundary_source_laplacian_rms_delta}, "
            f"interior_v_delta={comparison.interior_potential_rms_delta}, "
            f"hartree_delta_rms={comparison.hartree_potential_delta_rms:.12e}, "
            f"hartree_relative_delta={comparison.hartree_potential_relative_delta}, "
            f"effective_delta_rms={comparison.effective_potential_delta_rms:.12e}, "
            f"hartree_tail_delta={comparison.hartree_tail_mean_abs_delta:+.12e}"
        )
        print(f"    verdict: {comparison.verdict}")


def print_h2_monitor_grid_hartree_response_delta_summary(
    result: H2MonitorGridHartreeResponseDeltaAuditResult,
) -> None:
    """Print the fixed-density Hartree/open-boundary response delta summary."""

    print("IsoGridDFT H2 monitor-grid Hartree response delta audit")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"note: {result.note}")
    print()
    _print_route(result.singlet_routes[0])
    _print_route(result.singlet_routes[1])
    _print_comparisons(result.singlet_comparisons)
    print()
    _print_route(result.triplet_routes[0])
    _print_route(result.triplet_routes[1])
    _print_comparisons(result.triplet_comparisons)


def main() -> int:
    result = run_h2_monitor_grid_hartree_response_delta_audit()
    print_h2_monitor_grid_hartree_response_delta_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
