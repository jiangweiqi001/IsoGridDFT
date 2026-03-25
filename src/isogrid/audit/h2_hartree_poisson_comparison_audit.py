"""H2 singlet frozen-density Hartree / Poisson comparison audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.ops import integrate_field
from isogrid.poisson import evaluate_hartree_energy
from isogrid.poisson import solve_hartree_potential

from .h2_monitor_grid_ts_eloc_audit import _build_h2_bonding_trial_orbital

GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry
_DEFAULT_MULTIPOLE_ORDER = 2
_DEFAULT_TOLERANCE = 1.0e-8
_DEFAULT_MAX_ITERATIONS = 400
_DEFAULT_SAMPLE_Z_POSITIONS = (-9.0, -6.0, -3.0, -1.0, -0.7, 0.0, 0.7, 1.0, 3.0, 6.0, 9.0)


@dataclass(frozen=True)
class HartreeBoundarySummary:
    """Compact open-boundary summary for one Hartree solve."""

    multipole_order: int
    total_charge: float
    dipole_norm: float
    quadrupole_norm: float
    boundary_min: float
    boundary_max: float
    boundary_mean: float
    description: str


@dataclass(frozen=True)
class HartreeCenterLineSample:
    """One center-line Hartree potential sample on the molecular axis."""

    z_coordinate_bohr: float
    potential: float


@dataclass(frozen=True)
class H2HartreeRouteResult:
    """Resolved Hartree / Poisson result for one grid family."""

    grid_type: str
    grid_parameter_summary: str
    density_integral: float
    density_integral_error: float
    potential_min: float
    potential_max: float
    hartree_energy: float
    solver_method: str
    solver_iterations: int
    residual_max: float
    boundary_summary: HartreeBoundarySummary
    center_potential: float
    near_atom_potential: float
    far_boundary_mean: float
    centerline_samples: tuple[HartreeCenterLineSample, ...]
    mirror_symmetric: bool


@dataclass(frozen=True)
class H2HartreeDifferenceSummary:
    """Difference summary between the legacy and A-grid Hartree routes."""

    density_integral_difference: float
    hartree_energy_difference_ha: float
    hartree_energy_difference_mha: float
    center_potential_difference: float
    near_atom_potential_difference: float
    far_boundary_mean_difference: float
    centerline_max_abs_difference: float
    centerline_inner_mean_abs_difference: float
    centerline_middle_mean_abs_difference: float
    centerline_outer_mean_abs_difference: float
    likely_difference_pattern: str


@dataclass(frozen=True)
class HartreeBoundaryScanPoint:
    """One multipole-order scan point for one grid family."""

    grid_type: str
    multipole_order: int
    hartree_energy: float
    residual_max: float
    boundary_mean: float
    center_potential: float
    delta_vs_default_ha: float
    delta_vs_default_mha: float


@dataclass(frozen=True)
class HartreeToleranceScanPoint:
    """One Poisson tolerance scan point on the A-grid."""

    tolerance: float
    hartree_energy: float
    residual_max: float
    solver_method: str
    solver_iterations: int
    delta_vs_default_ha: float
    delta_vs_default_mha: float


@dataclass(frozen=True)
class H2HartreePoissonComparisonAuditResult:
    """Top-level H2 singlet Hartree / Poisson comparison audit result."""

    legacy_result: H2HartreeRouteResult
    monitor_result: H2HartreeRouteResult
    difference_summary: H2HartreeDifferenceSummary
    boundary_scan_results: tuple[HartreeBoundaryScanPoint, ...]
    tolerance_scan_results: tuple[HartreeToleranceScanPoint, ...]
    diagnosis: str
    note: str


def _grid_parameter_summary(grid_type: str) -> str:
    if grid_type == "legacy":
        return "legacy structured sinh baseline"
    return "A-grid baseline: shape=(67, 67, 81), box=(8.0, 8.0, 10.0), weight_scale=4.00, radius_scale=0.70"


def _build_frozen_density(
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=grid_geometry)
    rho_up = np.abs(orbital) ** 2
    rho_down = np.abs(orbital) ** 2
    rho_total = rho_up + rho_down
    return rho_up, rho_down, rho_total


def _boundary_mask(shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    return mask


def _sample_at_point(
    field: np.ndarray,
    grid_geometry: GridGeometryLike,
    point: tuple[float, float, float],
) -> float:
    radius_squared = (
        (grid_geometry.x_points - point[0]) ** 2
        + (grid_geometry.y_points - point[1]) ** 2
        + (grid_geometry.z_points - point[2]) ** 2
    )
    index = np.unravel_index(np.argmin(radius_squared), grid_geometry.spec.shape)
    return float(field[index])


def _build_boundary_summary(poisson_result) -> HartreeBoundarySummary:
    boundary = poisson_result.boundary_condition
    boundary_mask = _boundary_mask(poisson_result.potential.shape)
    boundary_values = boundary.boundary_values[boundary_mask]
    return HartreeBoundarySummary(
        multipole_order=boundary.multipole_order,
        total_charge=float(boundary.total_charge),
        dipole_norm=float(np.linalg.norm(boundary.dipole_moment)),
        quadrupole_norm=float(np.linalg.norm(boundary.quadrupole_tensor)),
        boundary_min=float(np.min(boundary_values)),
        boundary_max=float(np.max(boundary_values)),
        boundary_mean=float(np.mean(boundary_values)),
        description=boundary.description,
    )


def _build_centerline_samples(
    potential: np.ndarray,
    grid_geometry: GridGeometryLike,
    *,
    z_positions: tuple[float, ...] = _DEFAULT_SAMPLE_Z_POSITIONS,
) -> tuple[HartreeCenterLineSample, ...]:
    samples = []
    for z_coordinate in z_positions:
        samples.append(
            HartreeCenterLineSample(
                z_coordinate_bohr=float(z_coordinate),
                potential=_sample_at_point(
                    potential,
                    grid_geometry=grid_geometry,
                    point=(0.0, 0.0, z_coordinate),
                ),
            )
        )
    return tuple(samples)


def evaluate_h2_singlet_hartree_route(
    *,
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
    grid_type: str,
    multipole_order: int = _DEFAULT_MULTIPOLE_ORDER,
    tolerance: float = _DEFAULT_TOLERANCE,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    solver: str = "auto",
) -> H2HartreeRouteResult:
    """Evaluate one Hartree / Poisson route for the fixed H2 singlet density."""

    _, _, rho_total = _build_frozen_density(case=case, grid_geometry=grid_geometry)
    poisson_result = solve_hartree_potential(
        grid_geometry=grid_geometry,
        rho=rho_total,
        multipole_order=multipole_order,
        tolerance=tolerance,
        max_iterations=max_iterations,
        solver=solver,
    )
    hartree_energy = evaluate_hartree_energy(
        rho=rho_total,
        grid_geometry=grid_geometry,
        hartree_potential=poisson_result,
    )
    boundary_summary = _build_boundary_summary(poisson_result)
    boundary_mask = _boundary_mask(grid_geometry.spec.shape)
    centerline_samples = _build_centerline_samples(
        potential=poisson_result.potential,
        grid_geometry=grid_geometry,
    )
    return H2HartreeRouteResult(
        grid_type=grid_type,
        grid_parameter_summary=_grid_parameter_summary(grid_type),
        density_integral=float(integrate_field(rho_total, grid_geometry=grid_geometry)),
        density_integral_error=float(integrate_field(rho_total, grid_geometry=grid_geometry) - 2.0),
        potential_min=float(np.min(poisson_result.potential)),
        potential_max=float(np.max(poisson_result.potential)),
        hartree_energy=float(hartree_energy),
        solver_method=poisson_result.solver_method,
        solver_iterations=int(poisson_result.solver_iterations),
        residual_max=float(poisson_result.residual_max),
        boundary_summary=boundary_summary,
        center_potential=_sample_at_point(
            poisson_result.potential,
            grid_geometry=grid_geometry,
            point=(0.0, 0.0, 0.0),
        ),
        near_atom_potential=_sample_at_point(
            poisson_result.potential,
            grid_geometry=grid_geometry,
            point=case.geometry.atoms[0].position,
        ),
        far_boundary_mean=float(np.mean(poisson_result.potential[boundary_mask])),
        centerline_samples=centerline_samples,
        mirror_symmetric=bool(np.allclose(poisson_result.potential, poisson_result.potential[:, :, ::-1])),
    )


def _classify_difference_pattern(
    *,
    inner_mean_abs: float,
    middle_mean_abs: float,
    outer_mean_abs: float,
    boundary_mean_difference: float,
) -> str:
    if outer_mean_abs > 1.5 * max(inner_mean_abs, middle_mean_abs):
        return "far_field_boundary_dominated"
    if inner_mean_abs > 1.5 * max(middle_mean_abs, outer_mean_abs):
        return "near_core_or_geometry_dominated"
    if abs(boundary_mean_difference) > max(inner_mean_abs, middle_mean_abs):
        return "global_offset_like"
    return "broad_discretization_shift"


def compare_h2_hartree_routes(
    legacy_result: H2HartreeRouteResult,
    monitor_result: H2HartreeRouteResult,
) -> H2HartreeDifferenceSummary:
    """Compare the resolved legacy and A-grid Hartree routes."""

    legacy_samples = legacy_result.centerline_samples
    monitor_samples = monitor_result.centerline_samples
    differences = np.array(
        [
            monitor_sample.potential - legacy_sample.potential
            for legacy_sample, monitor_sample in zip(legacy_samples, monitor_samples, strict=True)
        ],
        dtype=np.float64,
    )
    z_values = np.array([sample.z_coordinate_bohr for sample in legacy_samples], dtype=np.float64)
    inner_mask = np.abs(z_values) <= 1.0
    middle_mask = (np.abs(z_values) > 1.0) & (np.abs(z_values) <= 4.0)
    outer_mask = np.abs(z_values) > 4.0
    inner_mean_abs = float(np.mean(np.abs(differences[inner_mask])))
    middle_mean_abs = float(np.mean(np.abs(differences[middle_mask])))
    outer_mean_abs = float(np.mean(np.abs(differences[outer_mask])))
    return H2HartreeDifferenceSummary(
        density_integral_difference=float(
            monitor_result.density_integral - legacy_result.density_integral
        ),
        hartree_energy_difference_ha=float(
            monitor_result.hartree_energy - legacy_result.hartree_energy
        ),
        hartree_energy_difference_mha=float(
            (monitor_result.hartree_energy - legacy_result.hartree_energy) * 1000.0
        ),
        center_potential_difference=float(
            monitor_result.center_potential - legacy_result.center_potential
        ),
        near_atom_potential_difference=float(
            monitor_result.near_atom_potential - legacy_result.near_atom_potential
        ),
        far_boundary_mean_difference=float(
            monitor_result.far_boundary_mean - legacy_result.far_boundary_mean
        ),
        centerline_max_abs_difference=float(np.max(np.abs(differences))),
        centerline_inner_mean_abs_difference=inner_mean_abs,
        centerline_middle_mean_abs_difference=middle_mean_abs,
        centerline_outer_mean_abs_difference=outer_mean_abs,
        likely_difference_pattern=_classify_difference_pattern(
            inner_mean_abs=inner_mean_abs,
            middle_mean_abs=middle_mean_abs,
            outer_mean_abs=outer_mean_abs,
            boundary_mean_difference=float(
                monitor_result.far_boundary_mean - legacy_result.far_boundary_mean
            ),
        ),
    )


def _run_boundary_scan(
    *,
    case: BenchmarkCase,
    legacy_geometry: StructuredGridGeometry,
    monitor_geometry: MonitorGridGeometry,
    default_legacy: H2HartreeRouteResult,
    default_monitor: H2HartreeRouteResult,
    multipole_orders: tuple[int, ...],
) -> tuple[HartreeBoundaryScanPoint, ...]:
    results: list[HartreeBoundaryScanPoint] = []
    for grid_type, grid_geometry, default_result in (
        ("legacy", legacy_geometry, default_legacy),
        ("monitor_a_grid", monitor_geometry, default_monitor),
    ):
        for multipole_order in multipole_orders:
            route_result = evaluate_h2_singlet_hartree_route(
                case=case,
                grid_geometry=grid_geometry,
                grid_type=grid_type,
                multipole_order=multipole_order,
            )
            delta = route_result.hartree_energy - default_result.hartree_energy
            results.append(
                HartreeBoundaryScanPoint(
                    grid_type=grid_type,
                    multipole_order=multipole_order,
                    hartree_energy=route_result.hartree_energy,
                    residual_max=route_result.residual_max,
                    boundary_mean=route_result.boundary_summary.boundary_mean,
                    center_potential=route_result.center_potential,
                    delta_vs_default_ha=float(delta),
                    delta_vs_default_mha=float(delta * 1000.0),
                )
            )
    return tuple(results)


def _run_tolerance_scan(
    *,
    case: BenchmarkCase,
    monitor_geometry: MonitorGridGeometry,
    default_monitor: H2HartreeRouteResult,
    tolerances: tuple[float, ...],
) -> tuple[HartreeToleranceScanPoint, ...]:
    results: list[HartreeToleranceScanPoint] = []
    for tolerance in tolerances:
        route_result = evaluate_h2_singlet_hartree_route(
            case=case,
            grid_geometry=monitor_geometry,
            grid_type="monitor_a_grid",
            tolerance=tolerance,
            max_iterations=600,
        )
        delta = route_result.hartree_energy - default_monitor.hartree_energy
        results.append(
            HartreeToleranceScanPoint(
                tolerance=float(tolerance),
                hartree_energy=route_result.hartree_energy,
                residual_max=route_result.residual_max,
                solver_method=route_result.solver_method,
                solver_iterations=route_result.solver_iterations,
                delta_vs_default_ha=float(delta),
                delta_vs_default_mha=float(delta * 1000.0),
            )
        )
    return tuple(results)


def _build_diagnosis(
    difference_summary: H2HartreeDifferenceSummary,
    boundary_scan_results: tuple[HartreeBoundaryScanPoint, ...],
    tolerance_scan_results: tuple[HartreeToleranceScanPoint, ...],
) -> str:
    hartree_gap_mha = abs(difference_summary.hartree_energy_difference_mha)
    max_boundary_effect_mha = max(abs(result.delta_vs_default_mha) for result in boundary_scan_results)
    max_tolerance_effect_mha = max(abs(result.delta_vs_default_mha) for result in tolerance_scan_results)
    density_integral_mismatch = abs(difference_summary.density_integral_difference)

    if density_integral_mismatch > 1.0e-6:
        return (
            "The Hartree gap still tracks a noticeable density-integral mismatch, so the "
            "first issue to fix is density / quadrature consistency across the two grids."
        )
    if max_tolerance_effect_mha > 0.1 * hartree_gap_mha:
        return (
            "The Hartree gap is too sensitive to Poisson tolerance to rule residual control out; "
            "the current monitor-grid solve needs tighter convergence study before blaming geometry."
        )
    if max_boundary_effect_mha > 0.1 * hartree_gap_mha:
        return (
            "The Hartree gap shows a sizable dependence on multipole boundary order, so the "
            "current open-boundary truncation is a leading suspect."
        )
    return (
        "The Hartree gap is much larger than both the multipole-order sensitivity and the "
        "Poisson-tolerance sensitivity, while the density integral is matched. The current "
        "main suspect is therefore the A-grid geometry / Laplacian discretization in the "
        "monitor-grid Poisson path, not the energy definition or residual control."
    )


def run_h2_hartree_poisson_comparison_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    multipole_orders: tuple[int, ...] = (0, 1, 2),
    tolerance_scan: tuple[float, ...] = (1.0e-6, 1.0e-8),
) -> H2HartreePoissonComparisonAuditResult:
    """Run the H2 singlet frozen-density Hartree / Poisson comparison audit."""

    legacy_geometry = build_default_h2_grid_geometry(case=case)
    monitor_geometry = build_h2_local_patch_development_monitor_grid()
    legacy_result = evaluate_h2_singlet_hartree_route(
        case=case,
        grid_geometry=legacy_geometry,
        grid_type="legacy",
    )
    monitor_result = evaluate_h2_singlet_hartree_route(
        case=case,
        grid_geometry=monitor_geometry,
        grid_type="monitor_a_grid",
    )
    difference_summary = compare_h2_hartree_routes(legacy_result, monitor_result)
    boundary_scan_results = _run_boundary_scan(
        case=case,
        legacy_geometry=legacy_geometry,
        monitor_geometry=monitor_geometry,
        default_legacy=legacy_result,
        default_monitor=monitor_result,
        multipole_orders=multipole_orders,
    )
    tolerance_scan_results = _run_tolerance_scan(
        case=case,
        monitor_geometry=monitor_geometry,
        default_monitor=monitor_result,
        tolerances=tolerance_scan,
    )
    diagnosis = _build_diagnosis(
        difference_summary=difference_summary,
        boundary_scan_results=boundary_scan_results,
        tolerance_scan_results=tolerance_scan_results,
    )
    return H2HartreePoissonComparisonAuditResult(
        legacy_result=legacy_result,
        monitor_result=monitor_result,
        difference_summary=difference_summary,
        boundary_scan_results=boundary_scan_results,
        tolerance_scan_results=tolerance_scan_results,
        diagnosis=diagnosis,
        note=(
            "This is a frozen-density Hartree / Poisson audit only. It does not touch SCF, "
            "eigensolver, nonlocal migration, or patch-assisted Hartree corrections."
        ),
    )


def _print_route_result(result: H2HartreeRouteResult) -> None:
    print(f"path: {result.grid_type}")
    print(f"  grid summary: {result.grid_parameter_summary}")
    print(f"  rho_total integral: {result.density_integral:.12f}")
    print(f"  rho_total integral error: {result.density_integral_error:+.3e}")
    print(f"  v_H min/max [Ha]: {result.potential_min:.12f} / {result.potential_max:.12f}")
    print(f"  E_H [Ha]: {result.hartree_energy:.12f}")
    print(
        "  solver: "
        f"{result.solver_method}, iterations={result.solver_iterations}, residual={result.residual_max:.3e}"
    )
    print(
        "  boundary summary: "
        f"order={result.boundary_summary.multipole_order}, "
        f"min={result.boundary_summary.boundary_min:.12f}, "
        f"max={result.boundary_summary.boundary_max:.12f}, "
        f"mean={result.boundary_summary.boundary_mean:.12f}"
    )
    print(f"  boundary note: {result.boundary_summary.description}")
    print(
        "  samples [Ha]: "
        f"center={result.center_potential:.12f}, "
        f"near_atom={result.near_atom_potential:.12f}, "
        f"far_boundary_mean={result.far_boundary_mean:.12f}"
    )
    print(f"  mirror symmetry: {result.mirror_symmetric}")
    print("  center-line samples:")
    for sample in result.centerline_samples:
        print(f"    z={sample.z_coordinate_bohr:+5.1f} -> {sample.potential:.12f}")


def print_h2_hartree_poisson_comparison_summary(
    result: H2HartreePoissonComparisonAuditResult,
) -> None:
    """Print the compact Hartree / Poisson comparison summary."""

    print("IsoGridDFT H2 singlet Hartree / Poisson comparison audit")
    print(f"note: {result.note}")
    print()
    _print_route_result(result.legacy_result)
    print()
    _print_route_result(result.monitor_result)
    print()
    diff = result.difference_summary
    print("legacy vs A-grid differences:")
    print(f"  density integral difference: {diff.density_integral_difference:+.3e}")
    print(f"  E_H difference [mHa]: {diff.hartree_energy_difference_mha:+.3f}")
    print(f"  center potential difference [Ha]: {diff.center_potential_difference:+.12f}")
    print(f"  near-atom potential difference [Ha]: {diff.near_atom_potential_difference:+.12f}")
    print(f"  far-boundary mean difference [Ha]: {diff.far_boundary_mean_difference:+.12f}")
    print(f"  center-line max abs difference [Ha]: {diff.centerline_max_abs_difference:.12f}")
    print(
        "  center-line mean abs difference [Ha]: "
        f"inner={diff.centerline_inner_mean_abs_difference:.12f}, "
        f"middle={diff.centerline_middle_mean_abs_difference:.12f}, "
        f"outer={diff.centerline_outer_mean_abs_difference:.12f}"
    )
    print(f"  pattern guess: {diff.likely_difference_pattern}")
    print()
    print("multipole-order scan:")
    for scan in result.boundary_scan_results:
        print(
            "  "
            f"{scan.grid_type}, order={scan.multipole_order}: "
            f"E_H={scan.hartree_energy:.12f} Ha, "
            f"delta_vs_order2={scan.delta_vs_default_mha:+.3f} mHa, "
            f"boundary_mean={scan.boundary_mean:.12f}, "
            f"residual={scan.residual_max:.3e}"
        )
    print()
    print("A-grid tolerance scan:")
    for scan in result.tolerance_scan_results:
        print(
            "  "
            f"tol={scan.tolerance:.1e}: "
            f"E_H={scan.hartree_energy:.12f} Ha, "
            f"delta_vs_default={scan.delta_vs_default_mha:+.3f} mHa, "
            f"solver={scan.solver_method}, "
            f"iterations={scan.solver_iterations}, "
            f"residual={scan.residual_max:.3e}"
        )
    print()
    print(f"diagnosis: {result.diagnosis}")


def main() -> int:
    result = run_h2_hartree_poisson_comparison_audit()
    print_h2_hartree_poisson_comparison_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
