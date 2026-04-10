"""Compare plateau-mode response under the default singlet route vs projector-mixing."""

from __future__ import annotations

from dataclasses import dataclass

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case

from .h2_monitor_grid_mixed_density_to_output_response_audit import (
    H2MonitorGridMixedDensityToOutputResponseAuditResult,
)
from .h2_monitor_grid_mixed_density_to_output_response_audit import (
    run_h2_monitor_grid_mixed_density_to_output_response_audit,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    H2MonitorGridPlateauModeEffectivePotentialToOccupiedDensityResponseAuditResult,
)
from .h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit import (
    run_h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit,
)

_DEFAULT_SHAPE = (15, 15, 17)
_DEFAULT_BOX_HALF_EXTENTS_BOHR = (9.0, 9.0, 11.0)
_DEFAULT_SOURCE_ITERATION_COUNT = 12
_DEFAULT_LATE_WINDOW_SIZE = 5


@dataclass(frozen=True)
class H2MonitorGridProjectorRoutePlateauCompareAuditResult:
    """Compare plateau-mode response under the default and projector routes."""

    case_name: str
    grid_parameter_summary: str
    spin_state_label: str
    controller_name: str
    source_iteration_count: int
    late_window_size: int
    baseline_route_name: str
    projector_route_name: str
    guarded_route_name: str
    baseline_occupied_density_signed_gain: float
    projector_occupied_density_signed_gain: float
    guarded_occupied_density_signed_gain: float
    baseline_current_mixed_update_to_mode_ratio: float
    projector_current_mixed_update_to_mode_ratio: float
    guarded_current_mixed_update_to_mode_ratio: float
    baseline_next_output_response_regime: str
    projector_next_output_response_regime: str
    guarded_next_output_response_regime: str
    baseline_next_output_response_ratio: float | None
    projector_next_output_response_ratio: float | None
    guarded_next_output_response_ratio: float | None
    comparison_regime: str
    verdict: str


def _grid_parameter_summary(grid_geometry: MonitorGridGeometry) -> str:
    return (
        f"shape={grid_geometry.spec.shape}, "
        f"box_half_extents_bohr=("
        f"{float(max(abs(grid_geometry.spec.box_bounds[0][0]), abs(grid_geometry.spec.box_bounds[0][1]))):.3f}, "
        f"{float(max(abs(grid_geometry.spec.box_bounds[1][0]), abs(grid_geometry.spec.box_bounds[1][1]))):.3f}, "
        f"{float(max(abs(grid_geometry.spec.box_bounds[2][0]), abs(grid_geometry.spec.box_bounds[2][1]))):.3f})"
    )


def _classify(
    *,
    baseline_occ: H2MonitorGridPlateauModeEffectivePotentialToOccupiedDensityResponseAuditResult,
    projector_occ: H2MonitorGridPlateauModeEffectivePotentialToOccupiedDensityResponseAuditResult,
    baseline_mixed: H2MonitorGridMixedDensityToOutputResponseAuditResult,
    projector_mixed: H2MonitorGridMixedDensityToOutputResponseAuditResult,
) -> tuple[str, str]:
    baseline_occ_mag = abs(float(baseline_occ.occupied_density_signed_gain))
    projector_occ_mag = abs(float(projector_occ.occupied_density_signed_gain))
    baseline_update_mag = abs(float(baseline_mixed.current_mixed_update_to_mode_ratio))
    projector_update_mag = abs(float(projector_mixed.current_mixed_update_to_mode_ratio))
    baseline_response = baseline_mixed.response_regime
    projector_response = projector_mixed.response_regime

    if (
        projector_occ_mag > max(1.0e-12, 5.0 * baseline_occ_mag)
        and projector_update_mag <= max(1.0e-12, 1.1 * baseline_update_mag)
        and projector_response in {"neutral", "counteract"}
    ):
        return (
            "response_amplified_only",
            "The projector route materially amplifies occupied-density response on the plateau mode, but the mixed-density to next-output response does not yet show a corresponding contraction improvement.",
        )
    if (
        projector_update_mag > max(1.0e-12, 1.5 * baseline_update_mag)
        and baseline_response == "counteract"
        and projector_response in {"neutral", "follow"}
    ):
        return (
            "plateau_contraction_improved",
            "The projector route strengthens the plateau-mode mixed update and improves the next-output response regime toward contraction.",
        )
    return (
        "no_material_change",
        "The projector route does not yet produce a clear plateau-mode contraction improvement relative to the default route under the current comparison metrics.",
    )


def run_h2_monitor_grid_projector_route_plateau_compare_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    grid_geometry: MonitorGridGeometry | None = None,
    source_iteration_count: int = _DEFAULT_SOURCE_ITERATION_COUNT,
    late_window_size: int = _DEFAULT_LATE_WINDOW_SIZE,
    controller_name: str = "generic_charge_spin_preconditioned",
) -> H2MonitorGridProjectorRoutePlateauCompareAuditResult:
    """Compare the plateau-mode response of the default singlet route and projector-mixing."""

    if grid_geometry is None:
        grid_geometry = build_monitor_grid_for_case(
            case,
            shape=_DEFAULT_SHAPE,
            box_half_extents=_DEFAULT_BOX_HALF_EXTENTS_BOHR,
            element_parameters=build_h2_local_patch_development_element_parameters(),
        )

    baseline_occ = run_h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit(
        case=case,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=source_iteration_count,
        probe_iteration=source_iteration_count,
        late_window_size=late_window_size,
        controller_name=controller_name,
        singlet_experimental_route_name="none",
    )
    projector_occ = run_h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit(
        case=case,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=source_iteration_count,
        probe_iteration=source_iteration_count,
        late_window_size=late_window_size,
        controller_name=controller_name,
        singlet_experimental_route_name="projector_mixing",
    )
    guarded_occ = run_h2_monitor_grid_plateau_mode_effective_potential_to_occupied_density_response_audit(
        case=case,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=source_iteration_count,
        probe_iteration=source_iteration_count,
        late_window_size=late_window_size,
        controller_name=controller_name,
        singlet_experimental_route_name="guarded_projector_mixing",
    )
    baseline_mixed = run_h2_monitor_grid_mixed_density_to_output_response_audit(
        case=case,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=source_iteration_count,
        late_window_size=late_window_size,
        controller_name=controller_name,
        singlet_experimental_route_name="none",
    )
    projector_mixed = run_h2_monitor_grid_mixed_density_to_output_response_audit(
        case=case,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=source_iteration_count,
        late_window_size=late_window_size,
        controller_name=controller_name,
        singlet_experimental_route_name="projector_mixing",
    )
    guarded_mixed = run_h2_monitor_grid_mixed_density_to_output_response_audit(
        case=case,
        grid_geometry=grid_geometry,
        spin_label="singlet",
        source_iteration_count=source_iteration_count,
        late_window_size=late_window_size,
        controller_name=controller_name,
        singlet_experimental_route_name="guarded_projector_mixing",
    )

    comparison_regime, verdict = _classify(
        baseline_occ=baseline_occ,
        projector_occ=projector_occ,
        baseline_mixed=baseline_mixed,
        projector_mixed=projector_mixed,
    )
    return H2MonitorGridProjectorRoutePlateauCompareAuditResult(
        case_name=case.name,
        grid_parameter_summary=_grid_parameter_summary(grid_geometry),
        spin_state_label="singlet",
        controller_name=controller_name,
        source_iteration_count=int(source_iteration_count),
        late_window_size=int(late_window_size),
        baseline_route_name="none",
        projector_route_name="projector_mixing",
        guarded_route_name="guarded_projector_mixing",
        baseline_occupied_density_signed_gain=float(baseline_occ.occupied_density_signed_gain),
        projector_occupied_density_signed_gain=float(projector_occ.occupied_density_signed_gain),
        guarded_occupied_density_signed_gain=float(guarded_occ.occupied_density_signed_gain),
        baseline_current_mixed_update_to_mode_ratio=float(baseline_mixed.current_mixed_update_to_mode_ratio),
        projector_current_mixed_update_to_mode_ratio=float(projector_mixed.current_mixed_update_to_mode_ratio),
        guarded_current_mixed_update_to_mode_ratio=float(guarded_mixed.current_mixed_update_to_mode_ratio),
        baseline_next_output_response_regime=baseline_mixed.response_regime,
        projector_next_output_response_regime=projector_mixed.response_regime,
        guarded_next_output_response_regime=guarded_mixed.response_regime,
        baseline_next_output_response_ratio=baseline_mixed.last_available_next_output_response_to_mixed_ratio,
        projector_next_output_response_ratio=projector_mixed.last_available_next_output_response_to_mixed_ratio,
        guarded_next_output_response_ratio=guarded_mixed.last_available_next_output_response_to_mixed_ratio,
        comparison_regime=comparison_regime,
        verdict=verdict,
    )


def print_h2_monitor_grid_projector_route_plateau_compare_summary(
    result: H2MonitorGridProjectorRoutePlateauCompareAuditResult,
) -> None:
    """Print a compact summary of the projector-route plateau comparison."""

    print("IsoGridDFT H2 projector-route plateau compare audit")
    print(f"case: {result.case_name}")
    print(f"grid: {result.grid_parameter_summary}")
    print(f"spin: {result.spin_state_label}")
    print(f"controller: {result.controller_name}")
    print(f"baseline route: {result.baseline_route_name}")
    print(f"projector route: {result.projector_route_name}")
    print(f"guarded route: {result.guarded_route_name}")
    print(f"baseline occupied-density gain: {result.baseline_occupied_density_signed_gain:.6e}")
    print(f"projector occupied-density gain: {result.projector_occupied_density_signed_gain:.6e}")
    print(f"guarded occupied-density gain: {result.guarded_occupied_density_signed_gain:.6e}")
    print(f"baseline mixed-update ratio: {result.baseline_current_mixed_update_to_mode_ratio:.6e}")
    print(f"projector mixed-update ratio: {result.projector_current_mixed_update_to_mode_ratio:.6e}")
    print(f"guarded mixed-update ratio: {result.guarded_current_mixed_update_to_mode_ratio:.6e}")
    print(f"baseline next-output regime: {result.baseline_next_output_response_regime}")
    print(f"projector next-output regime: {result.projector_next_output_response_regime}")
    print(f"guarded next-output regime: {result.guarded_next_output_response_regime}")
    print(f"comparison_regime: {result.comparison_regime}")
    print(f"verdict: {result.verdict}")


def main() -> int:
    result = run_h2_monitor_grid_projector_route_plateau_compare_audit()
    print_h2_monitor_grid_projector_route_plateau_compare_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
