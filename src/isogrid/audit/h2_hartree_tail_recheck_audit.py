"""Very small H2 Hartree tail recheck on the repaired monitor Poisson path."""

from __future__ import annotations

from dataclasses import dataclass

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case

from .h2_monitor_grid_poisson_operator_audit import PoissonOperatorRouteResult
from .h2_monitor_grid_poisson_operator_audit import ScalarFieldSummary
from .h2_monitor_grid_poisson_operator_audit import evaluate_poisson_operator_route
from .h2_monitor_grid_ts_eloc_audit import _build_h2_bonding_trial_orbital

_FROZEN_PATCH_PARAMETER_SUMMARY = (
    "patch fixed: radius_scale=0.75, grid_shape=(25, 25, 25), "
    "correction_strength=1.30, interpolation_neighbors=8"
)


@dataclass(frozen=True)
class H2HartreeTailRecheckPoint:
    """One very small A-grid Hartree tail-recheck point."""

    geometry_point_label: str
    grid_parameter_summary: str
    shape: tuple[int, int, int]
    box_half_extents_bohr: tuple[float, float, float]
    hartree_energy: float
    potential_summary: ScalarFieldSummary
    residual_summary: ScalarFieldSummary
    negative_interior_fraction: float
    far_field_potential_mean: float
    far_field_residual_rms: float
    far_field_negative_potential_fraction: float
    centerline_far_field_potential_mean: float
    hartree_delta_vs_legacy_mha: float
    hartree_delta_vs_baseline_mha: float
    solver_method: str
    solver_reported_residual_max: float


@dataclass(frozen=True)
class H2HartreeTailRecheckAuditResult:
    """Top-level H2 Hartree tail-recheck result on the repaired monitor path."""

    density_label: str
    legacy_hartree_energy_ha: float
    legacy_centerline_far_field_potential_mean: float
    frozen_patch_parameter_summary: str
    baseline_point: H2HartreeTailRecheckPoint
    finer_shape_point: H2HartreeTailRecheckPoint
    larger_box_point: H2HartreeTailRecheckPoint
    diagnosis: str
    note: str


def _build_h2_frozen_density(case: BenchmarkCase, grid_geometry) -> object:
    orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=grid_geometry)
    return 2.0 * abs(orbital) ** 2


def _centerline_far_field_potential_mean(route_result: PoissonOperatorRouteResult) -> float:
    outer_values = [
        sample.potential_value
        for sample in route_result.centerline_samples
        if abs(sample.z_coordinate_bohr) >= 6.0
    ]
    return float(sum(outer_values) / len(outer_values))


def _far_field_region(route_result: PoissonOperatorRouteResult):
    return next(region for region in route_result.region_diagnostics if region.region_name == "far_field")


def _build_point(
    *,
    point_label: str,
    shape: tuple[int, int, int],
    box_half_extents_bohr: tuple[float, float, float],
    legacy_hartree_energy_ha: float,
    baseline_hartree_energy_ha: float,
    route_result: PoissonOperatorRouteResult,
) -> H2HartreeTailRecheckPoint:
    far_field = _far_field_region(route_result)
    return H2HartreeTailRecheckPoint(
        geometry_point_label=point_label,
        grid_parameter_summary=(
            f"{point_label}: shape={shape}, box={box_half_extents_bohr}, "
            "weight_scale=4.00, radius_scale=0.70"
        ),
        shape=shape,
        box_half_extents_bohr=box_half_extents_bohr,
        hartree_energy=float(route_result.hartree_energy),
        potential_summary=route_result.potential_summary,
        residual_summary=route_result.residual_summary,
        negative_interior_fraction=float(route_result.negative_interior_fraction),
        far_field_potential_mean=float(far_field.potential_mean),
        far_field_residual_rms=float(far_field.residual_rms),
        far_field_negative_potential_fraction=float(far_field.negative_potential_fraction),
        centerline_far_field_potential_mean=_centerline_far_field_potential_mean(route_result),
        hartree_delta_vs_legacy_mha=float(
            (route_result.hartree_energy - legacy_hartree_energy_ha) * 1000.0
        ),
        hartree_delta_vs_baseline_mha=float(
            (route_result.hartree_energy - baseline_hartree_energy_ha) * 1000.0
        ),
        solver_method=route_result.solver_method,
        solver_reported_residual_max=float(route_result.solver_reported_residual_max),
    )


def _diagnosis(
    baseline_point: H2HartreeTailRecheckPoint,
    finer_shape_point: H2HartreeTailRecheckPoint,
    larger_box_point: H2HartreeTailRecheckPoint,
) -> str:
    if finer_shape_point.hartree_delta_vs_legacy_mha < baseline_point.hartree_delta_vs_legacy_mha and (
        larger_box_point.hartree_delta_vs_legacy_mha >= baseline_point.hartree_delta_vs_legacy_mha
    ):
        return (
            "The residual Hartree offset now looks more like an A-grid geometry / resolution "
            "tail than a surviving monitor-Poisson system bias: the slightly finer shape moves "
            "E_H toward legacy, while the slightly larger box does not."
        )
    if abs(larger_box_point.hartree_delta_vs_legacy_mha) < abs(baseline_point.hartree_delta_vs_legacy_mha):
        return (
            "The larger-box point reduces the Hartree offset more than the finer-shape point, "
            "so the remaining tail still looks boundary-sensitive."
        )
    return (
        "The remaining Hartree tail is small enough that this very limited recheck is not fully "
        "decisive, but it no longer resembles the earlier monitor-Poisson split bug."
    )


def run_h2_hartree_tail_recheck_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2HartreeTailRecheckAuditResult:
    """Run a very small H2 singlet Hartree tail recheck on the repaired A-grid path."""

    element_parameters = build_h2_local_patch_development_element_parameters()

    legacy_grid = build_default_h2_grid_geometry(case=case)
    legacy_density = _build_h2_frozen_density(case, legacy_grid)
    legacy_route = evaluate_poisson_operator_route(
        case=case,
        density_field=legacy_density,
        density_label="h2_singlet_frozen_density",
        grid_geometry=legacy_grid,
        grid_type="legacy",
    )

    point_specs = (
        ("baseline", (67, 67, 81), (8.0, 8.0, 10.0)),
        ("finer-shape", (75, 75, 91), (8.0, 8.0, 10.0)),
        ("larger-box", (67, 67, 81), (9.0, 9.0, 11.0)),
    )
    route_by_label: dict[str, tuple[tuple[int, int, int], tuple[float, float, float], PoissonOperatorRouteResult]] = {}
    for label, shape, box_half_extents in point_specs:
        grid_geometry = build_monitor_grid_for_case(
            case,
            shape=shape,
            box_half_extents=box_half_extents,
            element_parameters=element_parameters,
        )
        density_field = _build_h2_frozen_density(case, grid_geometry)
        route_by_label[label] = (
            shape,
            box_half_extents,
            evaluate_poisson_operator_route(
                case=case,
                density_field=density_field,
                density_label="h2_singlet_frozen_density",
                grid_geometry=grid_geometry,
                grid_type="monitor_a_grid",
            ),
        )

    baseline_shape, baseline_box, baseline_route = route_by_label["baseline"]
    finer_shape, finer_box, finer_route = route_by_label["finer-shape"]
    larger_shape, larger_box, larger_route = route_by_label["larger-box"]

    baseline_point = _build_point(
        point_label="baseline",
        shape=baseline_shape,
        box_half_extents_bohr=baseline_box,
        legacy_hartree_energy_ha=legacy_route.hartree_energy,
        baseline_hartree_energy_ha=baseline_route.hartree_energy,
        route_result=baseline_route,
    )
    finer_shape_point = _build_point(
        point_label="finer-shape",
        shape=finer_shape,
        box_half_extents_bohr=finer_box,
        legacy_hartree_energy_ha=legacy_route.hartree_energy,
        baseline_hartree_energy_ha=baseline_route.hartree_energy,
        route_result=finer_route,
    )
    larger_box_point = _build_point(
        point_label="larger-box",
        shape=larger_shape,
        box_half_extents_bohr=larger_box,
        legacy_hartree_energy_ha=legacy_route.hartree_energy,
        baseline_hartree_energy_ha=baseline_route.hartree_energy,
        route_result=larger_route,
    )

    return H2HartreeTailRecheckAuditResult(
        density_label="h2_singlet_frozen_density",
        legacy_hartree_energy_ha=float(legacy_route.hartree_energy),
        legacy_centerline_far_field_potential_mean=_centerline_far_field_potential_mean(legacy_route),
        frozen_patch_parameter_summary=_FROZEN_PATCH_PARAMETER_SUMMARY,
        baseline_point=baseline_point,
        finer_shape_point=finer_shape_point,
        larger_box_point=larger_box_point,
        diagnosis=_diagnosis(baseline_point, finer_shape_point, larger_box_point),
        note=(
            "This is a very small Hartree tail recheck on the repaired monitor Poisson path. "
            "Patch parameters remain frozen but patch does not directly modify Hartree."
        ),
    )


def _print_point(point: H2HartreeTailRecheckPoint) -> None:
    print(f"point: {point.geometry_point_label}")
    print(f"  grid summary: {point.grid_parameter_summary}")
    print(f"  E_H [Ha]: {point.hartree_energy:.12f}")
    print(f"  delta vs legacy [mHa]: {point.hartree_delta_vs_legacy_mha:+.3f}")
    print(f"  delta vs baseline [mHa]: {point.hartree_delta_vs_baseline_mha:+.3f}")
    print(
        "  v_H summary [Ha]: "
        f"min={point.potential_summary.minimum:.12f}, "
        f"max={point.potential_summary.maximum:.12f}, "
        f"mean={point.potential_summary.mean:.12f}"
    )
    print(
        "  full residual rms [Ha/Bohr^2]: "
        f"{point.residual_summary.rms:.12e}"
    )
    print(f"  negative interior fraction: {point.negative_interior_fraction:.6f}")
    print(
        "  far-field: "
        f"v_mean={point.far_field_potential_mean:.12f}, "
        f"res_rms={point.far_field_residual_rms:.12e}, "
        f"neg_frac={point.far_field_negative_potential_fraction:.6f}"
    )
    print(
        "  center-line far-field v_H mean [Ha]: "
        f"{point.centerline_far_field_potential_mean:.12f}"
    )
    print(
        "  solver: "
        f"{point.solver_method}, reported residual={point.solver_reported_residual_max:.3e}"
    )


def print_h2_hartree_tail_recheck_summary(result: H2HartreeTailRecheckAuditResult) -> None:
    """Print the compact H2 Hartree tail-recheck summary."""

    print("IsoGridDFT H2 singlet Hartree tail recheck")
    print(f"note: {result.note}")
    print(f"patch status: {result.frozen_patch_parameter_summary}")
    print(f"legacy E_H [Ha]: {result.legacy_hartree_energy_ha:.12f}")
    print(
        "legacy center-line far-field v_H mean [Ha]: "
        f"{result.legacy_centerline_far_field_potential_mean:.12f}"
    )
    print()
    _print_point(result.baseline_point)
    print()
    _print_point(result.finer_shape_point)
    print()
    _print_point(result.larger_box_point)
    print()
    print(f"diagnosis: {result.diagnosis}")


def main() -> int:
    result = run_h2_hartree_tail_recheck_audit()
    print_h2_hartree_tail_recheck_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
