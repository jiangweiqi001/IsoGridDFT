"""H2 singlet fair-calibration audit for the A-grid `T_s + E_loc,ion` reconnect."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import NearCoreElementParameters
from isogrid.grid import build_default_near_core_element_parameters
from isogrid.grid import build_monitor_grid_for_case

from .h2_monitor_grid_ts_eloc_audit import H2GridEnergyGeometrySummary
from .h2_monitor_grid_ts_eloc_audit import H2TsElocGridResult
from .h2_monitor_grid_ts_eloc_audit import evaluate_h2_singlet_ts_eloc_on_monitor_grid
from .h2_monitor_grid_ts_eloc_audit import run_h2_monitor_grid_ts_eloc_audit


@dataclass(frozen=True)
class H2MonitorFairCalibrationParameters:
    """One A-grid calibration point for the H2 singlet fairness scan."""

    label: str
    shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    weight_scale: float
    radius_scale: float = 1.0


@dataclass(frozen=True)
class H2MonitorFairCalibrationPoint:
    """Resolved H2 singlet `T_s + E_loc,ion` data for one A-grid calibration point."""

    parameters: H2MonitorFairCalibrationParameters
    geometry_summary: H2GridEnergyGeometrySummary
    box_not_smaller_than_legacy: bool
    near_core_not_coarser_than_legacy: bool
    positive_jacobian: bool
    is_fair_point: bool
    kinetic_energy: float
    local_ionic_energy: float
    ts_plus_eloc_energy: float
    delta_ts_vs_legacy_ha: float
    delta_eloc_vs_legacy_ha: float
    delta_ts_plus_eloc_vs_legacy_ha: float
    delta_ts_vs_legacy_mha: float
    delta_eloc_vs_legacy_mha: float
    delta_ts_plus_eloc_vs_legacy_mha: float
    reference_offset_ha: float
    reference_offset_mha: float
    improvement_vs_legacy_ha: float
    improvement_vs_legacy_mha: float


@dataclass(frozen=True)
class H2MonitorFairCalibrationAuditResult:
    """Top-level H2 singlet fairness audit for the A-grid calibration scan."""

    legacy_baseline: H2TsElocGridResult
    default_monitor_point: H2TsElocGridResult
    fair_scan_points: tuple[H2MonitorFairCalibrationPoint, ...]
    best_fair_point: H2MonitorFairCalibrationPoint | None
    fairness_definition: str
    note: str


def _default_scan_parameters() -> tuple[H2MonitorFairCalibrationParameters, ...]:
    return (
        H2MonitorFairCalibrationParameters(
            label="legacy_box_shape_41x41x49_w1p50_r1p00",
            shape=(41, 41, 49),
            box_half_extents_bohr=(8.0, 8.0, 10.0),
            weight_scale=1.50,
            radius_scale=1.00,
        ),
        H2MonitorFairCalibrationParameters(
            label="legacy_box_shape_45x45x55_w2p00_r0p90",
            shape=(45, 45, 55),
            box_half_extents_bohr=(8.0, 8.0, 10.0),
            weight_scale=2.00,
            radius_scale=0.90,
        ),
        H2MonitorFairCalibrationParameters(
            label="legacy_box_shape_59x59x71_w4p00_r0p70",
            shape=(59, 59, 71),
            box_half_extents_bohr=(8.0, 8.0, 10.0),
            weight_scale=4.00,
            radius_scale=0.70,
        ),
        H2MonitorFairCalibrationParameters(
            label="legacy_box_shape_61x61x75_w3p50_r0p70",
            shape=(61, 61, 75),
            box_half_extents_bohr=(8.0, 8.0, 10.0),
            weight_scale=3.50,
            radius_scale=0.70,
        ),
        H2MonitorFairCalibrationParameters(
            label="legacy_box_shape_63x63x77_w4p00_r0p70",
            shape=(63, 63, 77),
            box_half_extents_bohr=(8.0, 8.0, 10.0),
            weight_scale=4.00,
            radius_scale=0.70,
        ),
        H2MonitorFairCalibrationParameters(
            label="legacy_box_shape_67x67x81_w4p00_r0p70",
            shape=(67, 67, 81),
            box_half_extents_bohr=(8.0, 8.0, 10.0),
            weight_scale=4.00,
            radius_scale=0.70,
        ),
    )


def _scaled_element_parameters(
    case: BenchmarkCase,
    *,
    weight_scale: float,
    radius_scale: float,
) -> dict[str, NearCoreElementParameters]:
    base_parameters = build_default_near_core_element_parameters(case)
    return {
        element: replace(
            parameters,
            near_core_radius=parameters.near_core_radius * radius_scale,
            local_radius=parameters.local_radius * radius_scale,
            kinetic_weight=parameters.kinetic_weight * weight_scale,
            local_weight=parameters.local_weight * weight_scale,
        )
        for element, parameters in base_parameters.items()
    }


def _build_scan_point(
    case: BenchmarkCase,
    legacy_baseline: H2TsElocGridResult,
    parameters: H2MonitorFairCalibrationParameters,
) -> H2MonitorFairCalibrationPoint:
    element_parameters = _scaled_element_parameters(
        case,
        weight_scale=parameters.weight_scale,
        radius_scale=parameters.radius_scale,
    )
    grid_geometry = build_monitor_grid_for_case(
        case,
        shape=parameters.shape,
        box_half_extents=parameters.box_half_extents_bohr,
        element_parameters=element_parameters,
    )
    kinetic_energy, local_ionic_energy, ts_plus_eloc_energy, geometry_summary = (
        evaluate_h2_singlet_ts_eloc_on_monitor_grid(
            case=case,
            grid_geometry=grid_geometry,
        )
    )

    legacy_summary = legacy_baseline.geometry_summary
    box_condition = all(
        current >= baseline
        for current, baseline in zip(
            geometry_summary.box_half_extents_bohr,
            legacy_summary.box_half_extents_bohr,
        )
    )
    spacing_condition = (
        geometry_summary.near_core_min_spacing_bohr
        <= legacy_summary.near_core_min_spacing_bohr
    )
    jacobian_condition = geometry_summary.min_jacobian > 0.0
    reference_offset_ha = ts_plus_eloc_energy - legacy_baseline.reference_pyscf_total_energy
    delta_ts = kinetic_energy - legacy_baseline.kinetic_energy
    delta_eloc = local_ionic_energy - legacy_baseline.local_ionic_energy
    delta_total = ts_plus_eloc_energy - legacy_baseline.ts_plus_eloc_energy
    improvement = abs(legacy_baseline.reference_offset_ha) - abs(reference_offset_ha)
    return H2MonitorFairCalibrationPoint(
        parameters=parameters,
        geometry_summary=geometry_summary,
        box_not_smaller_than_legacy=box_condition,
        near_core_not_coarser_than_legacy=spacing_condition,
        positive_jacobian=jacobian_condition,
        is_fair_point=box_condition and spacing_condition and jacobian_condition,
        kinetic_energy=float(kinetic_energy),
        local_ionic_energy=float(local_ionic_energy),
        ts_plus_eloc_energy=float(ts_plus_eloc_energy),
        delta_ts_vs_legacy_ha=float(delta_ts),
        delta_eloc_vs_legacy_ha=float(delta_eloc),
        delta_ts_plus_eloc_vs_legacy_ha=float(delta_total),
        delta_ts_vs_legacy_mha=float(delta_ts * 1000.0),
        delta_eloc_vs_legacy_mha=float(delta_eloc * 1000.0),
        delta_ts_plus_eloc_vs_legacy_mha=float(delta_total * 1000.0),
        reference_offset_ha=float(reference_offset_ha),
        reference_offset_mha=float(reference_offset_ha * 1000.0),
        improvement_vs_legacy_ha=float(improvement),
        improvement_vs_legacy_mha=float(improvement * 1000.0),
    )


def _pick_best_fair_point(
    points: tuple[H2MonitorFairCalibrationPoint, ...],
) -> H2MonitorFairCalibrationPoint | None:
    fair_points = [point for point in points if point.is_fair_point]
    if not fair_points:
        return None
    return max(
        fair_points,
        key=lambda point: (
            point.improvement_vs_legacy_ha,
            -point.geometry_summary.near_core_min_spacing_bohr,
        ),
    )


def run_h2_monitor_grid_fair_calibration_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    scan_parameters: tuple[H2MonitorFairCalibrationParameters, ...] | None = None,
) -> H2MonitorFairCalibrationAuditResult:
    """Run the H2 singlet A-grid fairness calibration scan."""

    baseline_audit = run_h2_monitor_grid_ts_eloc_audit(case=case)
    if scan_parameters is None:
        scan_parameters = _default_scan_parameters()
    fair_scan_points = tuple(
        _build_scan_point(case, baseline_audit.legacy, parameters)
        for parameters in scan_parameters
    )
    return H2MonitorFairCalibrationAuditResult(
        legacy_baseline=baseline_audit.legacy,
        default_monitor_point=baseline_audit.monitor_grid,
        fair_scan_points=fair_scan_points,
        best_fair_point=_pick_best_fair_point(fair_scan_points),
        fairness_definition=(
            "Fair means: (1) box half-extents are not smaller than the legacy baseline "
            "along all three axes, (2) the near-core minimum spacing around the H nuclei "
            "is not larger than the legacy baseline, (3) the Jacobian remains strictly "
            "positive, and (4) the same H2 singlet bonding trial orbital and frozen "
            "density rho = 2|psi|^2 are used on both grids."
        ),
        note=(
            "This scan still compares only T_s and E_loc,ion. "
            "Nonlocal, Hartree, XC, eigensolver, and SCF have not been migrated "
            "to the A-grid yet."
        ),
    )


def _print_geometry_fairness_header(result: H2MonitorFairCalibrationAuditResult) -> None:
    legacy_summary = result.legacy_baseline.geometry_summary
    print("IsoGridDFT H2 singlet A-grid fair calibration audit")
    print(f"fairness definition: {result.fairness_definition}")
    print(f"note: {result.note}")
    print("legacy fairness baseline:")
    print(
        "  box half-extents [Bohr]: "
        f"({legacy_summary.box_half_extents_bohr[0]:.3f}, "
        f"{legacy_summary.box_half_extents_bohr[1]:.3f}, "
        f"{legacy_summary.box_half_extents_bohr[2]:.3f})"
    )
    print(f"  near-core min spacing [Bohr]: {legacy_summary.near_core_min_spacing_bohr:.6f}")
    print(f"  T_s [Ha]: {result.legacy_baseline.kinetic_energy:.12f}")
    print(f"  E_loc,ion [Ha]: {result.legacy_baseline.local_ionic_energy:.12f}")
    print(f"  T_s + E_loc,ion [Ha]: {result.legacy_baseline.ts_plus_eloc_energy:.12f}")
    print(f"  offset vs PySCF singlet total [mHa]: {result.legacy_baseline.reference_offset_mha:+.3f}")


def _print_original_default(result: H2MonitorFairCalibrationAuditResult) -> None:
    default_summary = result.default_monitor_point.geometry_summary
    print()
    print("current raw A-grid default point:")
    print(f"  grid shape: {default_summary.grid_shape}")
    print(
        "  box half-extents [Bohr]: "
        f"({default_summary.box_half_extents_bohr[0]:.3f}, "
        f"{default_summary.box_half_extents_bohr[1]:.3f}, "
        f"{default_summary.box_half_extents_bohr[2]:.3f})"
    )
    print(f"  near-core min spacing [Bohr]: {default_summary.near_core_min_spacing_bohr:.6f}")
    print(f"  T_s + E_loc,ion [Ha]: {result.default_monitor_point.ts_plus_eloc_energy:.12f}")
    print(f"  improvement vs legacy [mHa]: {result.default_monitor_point.improvement_vs_legacy_mha:+.3f}")


def _print_scan_point(point: H2MonitorFairCalibrationPoint) -> None:
    summary = point.geometry_summary
    parameters = point.parameters
    print()
    print(f"scan point: {parameters.label}")
    print(
        "  params: "
        f"shape={parameters.shape}, "
        f"box={parameters.box_half_extents_bohr}, "
        f"weight_scale={parameters.weight_scale:.2f}, "
        f"radius_scale={parameters.radius_scale:.2f}"
    )
    print(f"  min Jacobian: {summary.min_jacobian:.6e}")
    print(f"  near-core min spacing [Bohr]: {summary.near_core_min_spacing_bohr:.6f}")
    print(
        "  fairness checks: "
        f"box_ok={point.box_not_smaller_than_legacy}, "
        f"spacing_ok={point.near_core_not_coarser_than_legacy}, "
        f"jacobian_ok={point.positive_jacobian}, "
        f"fair_point={point.is_fair_point}"
    )
    print(f"  T_s [Ha]: {point.kinetic_energy:.12f} ({point.delta_ts_vs_legacy_mha:+.3f} mHa vs legacy)")
    print(
        f"  E_loc,ion [Ha]: {point.local_ionic_energy:.12f} "
        f"({point.delta_eloc_vs_legacy_mha:+.3f} mHa vs legacy)"
    )
    print(
        f"  T_s + E_loc,ion [Ha]: {point.ts_plus_eloc_energy:.12f} "
        f"({point.delta_ts_plus_eloc_vs_legacy_mha:+.3f} mHa vs legacy)"
    )
    print(f"  offset vs PySCF singlet total [mHa]: {point.reference_offset_mha:+.3f}")
    print(f"  improvement vs legacy [mHa]: {point.improvement_vs_legacy_mha:+.3f}")


def print_h2_monitor_grid_fair_calibration_summary(
    result: H2MonitorFairCalibrationAuditResult,
) -> None:
    """Print the compact fairness calibration summary."""

    _print_geometry_fairness_header(result)
    _print_original_default(result)
    print()
    print("fair calibration scan points:")
    for point in result.fair_scan_points:
        _print_scan_point(point)
    print()
    if result.best_fair_point is None:
        print("best fair point: none found in the current scan")
    else:
        print(f"best fair point: {result.best_fair_point.parameters.label}")
        print(
            "  status vs legacy: "
            f"{result.best_fair_point.improvement_vs_legacy_mha:+.3f} mHa "
            "on the T_s + E_loc,ion trend reference"
        )


def main() -> int:
    result = run_h2_monitor_grid_fair_calibration_audit()
    print_h2_monitor_grid_fair_calibration_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
