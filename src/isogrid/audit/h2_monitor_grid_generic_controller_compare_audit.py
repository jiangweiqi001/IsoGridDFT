"""Compare the generic charge/spin controller against the current baseline route."""

from __future__ import annotations

from dataclasses import dataclass

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.scf import run_h2_monitor_grid_scf_dry_run

from .h2_monitor_grid_targeted_bad_pair_audit import (
    H2MonitorGridTargetedBadPairAuditResult,
)
from .h2_monitor_grid_targeted_bad_pair_audit import run_h2_monitor_grid_targeted_bad_pair_audit


@dataclass(frozen=True)
class H2MonitorGridGenericControllerRouteComparison:
    """Compact comparison summary for one controller route."""

    controller_name: str
    singlet_density_residual_history: tuple[float, ...]
    triplet_density_residual_history: tuple[float, ...]
    singlet_max_targeted_density_gap: float
    singlet_has_late_targeted_bad_pair: bool
    singlet_max_hartree_share: float | None
    singlet_final_density_residual: float
    triplet_final_density_residual: float
    singlet_targeted_bad_pair_count: int
    triplet_targeted_bad_pair_count: int
    verdict: str


@dataclass(frozen=True)
class H2MonitorGridGenericControllerCompareAuditResult:
    """Top-level baseline vs generic-controller comparison result."""

    case_name: str
    grid_parameter_summary: str
    baseline: H2MonitorGridGenericControllerRouteComparison
    generic_charge_spin: H2MonitorGridGenericControllerRouteComparison
    note: str


def _small_grid_geometry(case: BenchmarkCase) -> MonitorGridGeometry:
    return build_monitor_grid_for_case(
        case,
        shape=(9, 9, 11),
        box_half_extents=(6.0, 6.0, 8.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )


def _max_or_none(values: tuple[float | None, ...]) -> float | None:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return None
    return max(finite_values)


def _route_summary(
    *,
    controller_name: str,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
    source_iteration_count: int,
    short_run_iterations: int,
) -> H2MonitorGridGenericControllerRouteComparison:
    singlet = run_h2_monitor_grid_scf_dry_run(
        "singlet",
        case=case,
        grid_geometry=grid_geometry,
        max_iterations=short_run_iterations,
        mixing=0.2,
        density_tolerance=1.0e-2,
        energy_tolerance=1.0e-4,
        eigensolver_tolerance=1.0e-2,
        eigensolver_ncv=8,
        controller_name=controller_name,
    )
    triplet = run_h2_monitor_grid_scf_dry_run(
        "triplet",
        case=case,
        grid_geometry=grid_geometry,
        max_iterations=short_run_iterations,
        mixing=0.2,
        density_tolerance=1.0e-2,
        energy_tolerance=1.0e-4,
        eigensolver_tolerance=1.0e-2,
        eigensolver_ncv=8,
        controller_name=controller_name,
    )
    targeted = run_h2_monitor_grid_targeted_bad_pair_audit(
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        controller_name=controller_name,
    )
    singlet_targeted = targeted.singlet.targeted_pairs
    singlet_max_targeted_density_gap = (
        0.0
        if not singlet_targeted
        else max(pair.baseline_minus_freeze_hartree_density_residual for pair in singlet_targeted)
    )
    singlet_has_late_targeted_bad_pair = any(
        pair.pair_iterations[0] >= 4 for pair in singlet_targeted
    )
    singlet_max_hartree_share = _max_or_none(
        tuple(signal.hartree_share for signal in singlet.controller_signals_history)
    )
    if singlet_max_targeted_density_gap < 0.10 and not singlet_has_late_targeted_bad_pair:
        verdict = "No strong late singlet bad pair remains under the generic comparison thresholds."
    else:
        verdict = "A material singlet Hartree-dominated bad pair still remains in the short-run window."
    return H2MonitorGridGenericControllerRouteComparison(
        controller_name=controller_name,
        singlet_density_residual_history=tuple(
            float(value) for value in singlet.density_residual_history
        ),
        triplet_density_residual_history=tuple(
            float(value) for value in triplet.density_residual_history
        ),
        singlet_max_targeted_density_gap=float(singlet_max_targeted_density_gap),
        singlet_has_late_targeted_bad_pair=bool(singlet_has_late_targeted_bad_pair),
        singlet_max_hartree_share=singlet_max_hartree_share,
        singlet_final_density_residual=float(singlet.density_residual_history[-1]),
        triplet_final_density_residual=float(triplet.density_residual_history[-1]),
        singlet_targeted_bad_pair_count=len(singlet_targeted),
        triplet_targeted_bad_pair_count=len(targeted.triplet.targeted_pairs),
        verdict=verdict,
    )


def run_h2_monitor_grid_generic_controller_compare_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    source_iteration_count: int = 6,
    short_run_iterations: int = 6,
) -> H2MonitorGridGenericControllerCompareAuditResult:
    """Run the first lightweight baseline vs generic-controller comparison."""

    if grid_geometry is None:
        grid_geometry = _small_grid_geometry(case)
    bounds = grid_geometry.spec.box_bounds
    box_half_extents_bohr = (
        0.5 * float(bounds[0][1] - bounds[0][0]),
        0.5 * float(bounds[1][1] - bounds[1][0]),
        0.5 * float(bounds[2][1] - bounds[2][0]),
    )
    baseline = _route_summary(
        controller_name="baseline_linear",
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        short_run_iterations=short_run_iterations,
    )
    generic = _route_summary(
        controller_name="generic_charge_spin",
        case=case,
        grid_geometry=grid_geometry,
        source_iteration_count=source_iteration_count,
        short_run_iterations=short_run_iterations,
    )
    return H2MonitorGridGenericControllerCompareAuditResult(
        case_name=case.name,
        grid_parameter_summary=(
            f"shape={grid_geometry.spec.shape}, "
            f"box_half_extents_bohr={box_half_extents_bohr}"
        ),
        baseline=baseline,
        generic_charge_spin=generic,
        note=(
            "This lightweight compare audit runs the same small H2 A-grid local-only case under "
            "the current baseline route and the new generic charge/spin controller, then compares "
            "targeted singlet bad-pair gaps and short-run residual histories."
        ),
    )


def print_h2_monitor_grid_generic_controller_compare_summary(
    result: H2MonitorGridGenericControllerCompareAuditResult,
) -> None:
    """Print the compact baseline vs generic-controller comparison summary."""

    print("IsoGridDFT H2 monitor-grid generic controller compare audit")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"note: {result.note}")
    for route in (result.baseline, result.generic_charge_spin):
        print()
        print(f"controller: {route.controller_name}")
        print(f"  singlet residual history: {route.singlet_density_residual_history}")
        print(f"  triplet residual history: {route.triplet_density_residual_history}")
        print(f"  singlet max targeted density gap: {route.singlet_max_targeted_density_gap}")
        print(f"  singlet late bad pair: {route.singlet_has_late_targeted_bad_pair}")
        print(f"  singlet max hartree share: {route.singlet_max_hartree_share}")
        print(f"  singlet final residual: {route.singlet_final_density_residual}")
        print(f"  triplet final residual: {route.triplet_final_density_residual}")
        print(f"  verdict: {route.verdict}")


def main() -> int:
    result = run_h2_monitor_grid_generic_controller_compare_audit()
    print_h2_monitor_grid_generic_controller_compare_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
