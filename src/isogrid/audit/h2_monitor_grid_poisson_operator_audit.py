"""A-grid Poisson-operator audit for the H2 singlet frozen density."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ops import apply_legacy_laplacian_operator
from isogrid.ops import apply_monitor_grid_laplacian_operator
from isogrid.ops import integrate_field
from isogrid.poisson import solve_hartree_potential

from .h2_monitor_grid_ts_eloc_audit import _build_h2_bonding_trial_orbital

_FOUR_PI = 4.0 * np.pi
_DEFAULT_SAMPLE_Z_POSITIONS = (-9.0, -6.0, -3.0, -1.0, -0.7, 0.0, 0.7, 1.0, 3.0, 6.0, 9.0)


@dataclass(frozen=True)
class ScalarFieldSummary:
    """Compact scalar-field statistics on one selected mask."""

    minimum: float
    maximum: float
    mean: float
    rms: float


@dataclass(frozen=True)
class RegionDiagnostic:
    """Regional statistics for v_H, L(v_H), and the Poisson residual."""

    region_name: str
    point_count: int
    potential_mean: float
    potential_min: float
    potential_max: float
    laplacian_mean: float
    residual_mean: float
    residual_rms: float
    negative_potential_fraction: float


@dataclass(frozen=True)
class CenterLineSample:
    """One center-line sample on the H2 molecular axis."""

    z_coordinate_bohr: float
    rho_value: float
    potential_value: float
    laplacian_value: float
    residual_value: float


@dataclass(frozen=True)
class BoundaryConditionSummary:
    """Compact open-boundary multipole data."""

    multipole_order: int
    total_charge: float
    dipole_norm: float
    quadrupole_norm: float
    boundary_min: float
    boundary_max: float
    boundary_mean: float
    description: str


@dataclass(frozen=True)
class SelfAdjointnessProbe:
    """Lightweight numerical clue for discrete self-adjointness."""

    lhs: float
    rhs: float
    absolute_defect: float
    relative_defect: float


@dataclass(frozen=True)
class BoundarySplitDiagnostic:
    """Diagnostic for consistency between full and split Poisson residuals."""

    boundary_laplacian_summary: ScalarFieldSummary
    full_vs_twice_boundary_max_abs: float
    full_vs_twice_boundary_rms: float
    correlation_with_boundary_laplacian: float


@dataclass(frozen=True)
class PoissonOperatorRouteResult:
    """Resolved Poisson-operator audit result on one grid path."""

    density_label: str
    grid_type: str
    grid_parameter_summary: str
    density_integral: float
    density_integral_error: float
    potential_summary: ScalarFieldSummary
    laplacian_summary: ScalarFieldSummary
    residual_summary: ScalarFieldSummary
    hartree_energy: float
    solver_method: str
    solver_iterations: int
    solver_reported_residual_max: float
    boundary_summary: BoundaryConditionSummary
    centerline_samples: tuple[CenterLineSample, ...]
    region_diagnostics: tuple[RegionDiagnostic, ...]
    mirror_symmetric: bool
    negative_interior_fraction: float
    self_adjointness_probe: SelfAdjointnessProbe
    boundary_split_diagnostic: BoundarySplitDiagnostic | None


@dataclass(frozen=True)
class PoissonOperatorDifferenceSummary:
    """Comparison summary between legacy and A-grid routes."""

    hartree_energy_difference_ha: float
    hartree_energy_difference_mha: float
    potential_min_difference: float
    potential_max_difference: float
    negative_interior_fraction_difference: float
    residual_rms_ratio: float
    centerline_max_abs_difference: float
    centerline_inner_mean_abs_difference: float
    centerline_middle_mean_abs_difference: float
    centerline_outer_mean_abs_difference: float
    likely_difference_pattern: str


@dataclass(frozen=True)
class MonitorShapeScanPoint:
    """Very small A-grid shape scan point for operator-diagnostic trend checks."""

    shape: tuple[int, int, int]
    hartree_energy: float
    residual_rms: float
    negative_interior_fraction: float
    center_potential: float
    delta_vs_baseline_mha: float


@dataclass(frozen=True)
class H2MonitorGridPoissonOperatorAuditResult:
    """Top-level A-grid Poisson-operator audit result."""

    legacy_h2_result: PoissonOperatorRouteResult
    monitor_h2_result: PoissonOperatorRouteResult
    legacy_gaussian_result: PoissonOperatorRouteResult
    monitor_gaussian_result: PoissonOperatorRouteResult
    difference_summary: PoissonOperatorDifferenceSummary
    shape_scan_results: tuple[MonitorShapeScanPoint, ...]
    diagnosis: str
    note: str


def _boundary_mask(shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    return mask


def _interior_mask(shape: tuple[int, int, int]) -> np.ndarray:
    return ~_boundary_mask(shape)


def _grid_parameter_summary(grid_type: str) -> str:
    if grid_type == "legacy":
        return "legacy structured sinh baseline"
    return "A-grid baseline: shape=(67, 67, 81), box=(8.0, 8.0, 10.0), weight_scale=4.00, radius_scale=0.70"


def _build_h2_frozen_density(
    case: BenchmarkCase,
    grid_geometry: StructuredGridGeometry | MonitorGridGeometry,
) -> np.ndarray:
    orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=grid_geometry)
    return 2.0 * np.abs(orbital) ** 2


def _build_gaussian_density(
    grid_geometry: StructuredGridGeometry | MonitorGridGeometry,
    *,
    alpha: float = 0.5,
    target_charge: float = 2.0,
) -> np.ndarray:
    rho = np.exp(
        -alpha
        * (
            grid_geometry.x_points**2
            + grid_geometry.y_points**2
            + grid_geometry.z_points**2
        )
    )
    normalization = float(integrate_field(rho, grid_geometry=grid_geometry))
    return target_charge * rho / normalization


def _sample_at_point(
    field: np.ndarray,
    grid_geometry: StructuredGridGeometry | MonitorGridGeometry,
    point: tuple[float, float, float],
) -> float:
    radius_squared = (
        (grid_geometry.x_points - point[0]) ** 2
        + (grid_geometry.y_points - point[1]) ** 2
        + (grid_geometry.z_points - point[2]) ** 2
    )
    index = np.unravel_index(np.argmin(radius_squared), grid_geometry.spec.shape)
    return float(field[index])


def _scalar_summary(values: np.ndarray) -> ScalarFieldSummary:
    return ScalarFieldSummary(
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        rms=float(np.sqrt(np.mean(values * values))),
    )


def _boundary_summary(poisson_result) -> BoundaryConditionSummary:
    boundary = poisson_result.boundary_condition
    boundary_mask = _boundary_mask(poisson_result.potential.shape)
    boundary_values = boundary.boundary_values[boundary_mask]
    return BoundaryConditionSummary(
        multipole_order=boundary.multipole_order,
        total_charge=float(boundary.total_charge),
        dipole_norm=float(np.linalg.norm(boundary.dipole_moment)),
        quadrupole_norm=float(np.linalg.norm(boundary.quadrupole_tensor)),
        boundary_min=float(np.min(boundary_values)),
        boundary_max=float(np.max(boundary_values)),
        boundary_mean=float(np.mean(boundary_values)),
        description=boundary.description,
    )


def _region_masks(
    case: BenchmarkCase,
    grid_geometry: StructuredGridGeometry | MonitorGridGeometry,
) -> tuple[tuple[str, np.ndarray], ...]:
    interior = _interior_mask(grid_geometry.spec.shape)
    center = (
        sum(atom.position[0] for atom in case.geometry.atoms) / len(case.geometry.atoms),
        sum(atom.position[1] for atom in case.geometry.atoms) / len(case.geometry.atoms),
        sum(atom.position[2] for atom in case.geometry.atoms) / len(case.geometry.atoms),
    )
    nearest_atom_distance = np.full(grid_geometry.spec.shape, np.inf, dtype=np.float64)
    near_core = np.zeros(grid_geometry.spec.shape, dtype=bool)

    for atom in case.geometry.atoms:
        radius = np.sqrt(
            (grid_geometry.x_points - atom.position[0]) ** 2
            + (grid_geometry.y_points - atom.position[1]) ** 2
            + (grid_geometry.z_points - atom.position[2]) ** 2,
            dtype=np.float64,
        )
        nearest_atom_distance = np.minimum(nearest_atom_distance, radius)
        near_core |= radius < 0.9

    center_radius = np.sqrt(
        (grid_geometry.x_points - center[0]) ** 2
        + (grid_geometry.y_points - center[1]) ** 2
        + (grid_geometry.z_points - center[2]) ** 2,
        dtype=np.float64,
    )
    center_mask = (center_radius < 0.8) & (nearest_atom_distance > 0.35)

    if isinstance(grid_geometry, MonitorGridGeometry):
        bounds = grid_geometry.spec.box_bounds
    else:
        bounds = (
            (
                grid_geometry.spec.reference_center[0] + grid_geometry.spec.x_axis.lower_offset,
                grid_geometry.spec.reference_center[0] + grid_geometry.spec.x_axis.upper_offset,
            ),
            (
                grid_geometry.spec.reference_center[1] + grid_geometry.spec.y_axis.lower_offset,
                grid_geometry.spec.reference_center[1] + grid_geometry.spec.y_axis.upper_offset,
            ),
            (
                grid_geometry.spec.reference_center[2] + grid_geometry.spec.z_axis.lower_offset,
                grid_geometry.spec.reference_center[2] + grid_geometry.spec.z_axis.upper_offset,
            ),
        )
    distance_to_boundary = np.minimum.reduce(
        [
            grid_geometry.x_points - bounds[0][0],
            bounds[0][1] - grid_geometry.x_points,
            grid_geometry.y_points - bounds[1][0],
            bounds[1][1] - grid_geometry.y_points,
            grid_geometry.z_points - bounds[2][0],
            bounds[2][1] - grid_geometry.z_points,
        ]
    )
    far_field = (distance_to_boundary < 1.5) & (nearest_atom_distance > 1.5)

    return (
        ("near_core", interior & near_core),
        ("center", interior & center_mask),
        ("far_field", interior & far_field),
    )


def _region_diagnostics(
    case: BenchmarkCase,
    grid_geometry: StructuredGridGeometry | MonitorGridGeometry,
    potential: np.ndarray,
    laplacian: np.ndarray,
    residual: np.ndarray,
) -> tuple[RegionDiagnostic, ...]:
    diagnostics = []
    for region_name, mask in _region_masks(case, grid_geometry):
        values_v = potential[mask]
        values_l = laplacian[mask]
        values_r = residual[mask]
        diagnostics.append(
            RegionDiagnostic(
                region_name=region_name,
                point_count=int(np.count_nonzero(mask)),
                potential_mean=float(np.mean(values_v)),
                potential_min=float(np.min(values_v)),
                potential_max=float(np.max(values_v)),
                laplacian_mean=float(np.mean(values_l)),
                residual_mean=float(np.mean(values_r)),
                residual_rms=float(np.sqrt(np.mean(values_r * values_r))),
                negative_potential_fraction=float(np.mean(values_v < 0.0)),
            )
        )
    return tuple(diagnostics)


def _centerline_samples(
    field_rho: np.ndarray,
    field_v: np.ndarray,
    field_laplacian: np.ndarray,
    field_residual: np.ndarray,
    grid_geometry: StructuredGridGeometry | MonitorGridGeometry,
) -> tuple[CenterLineSample, ...]:
    return tuple(
        CenterLineSample(
            z_coordinate_bohr=float(z_coordinate),
            rho_value=_sample_at_point(field_rho, grid_geometry, (0.0, 0.0, z_coordinate)),
            potential_value=_sample_at_point(field_v, grid_geometry, (0.0, 0.0, z_coordinate)),
            laplacian_value=_sample_at_point(field_laplacian, grid_geometry, (0.0, 0.0, z_coordinate)),
            residual_value=_sample_at_point(field_residual, grid_geometry, (0.0, 0.0, z_coordinate)),
        )
        for z_coordinate in _DEFAULT_SAMPLE_Z_POSITIONS
    )


def _self_adjointness_probe(
    grid_geometry: StructuredGridGeometry | MonitorGridGeometry,
) -> SelfAdjointnessProbe:
    u = np.exp(
        -0.35
        * (
            grid_geometry.x_points**2
            + grid_geometry.y_points**2
            + grid_geometry.z_points**2
        )
    )
    w = np.exp(
        -0.25
        * (
            grid_geometry.x_points**2
            + grid_geometry.y_points**2
            + (grid_geometry.z_points - 1.0) ** 2
        )
    )
    if isinstance(grid_geometry, MonitorGridGeometry):
        lu = apply_monitor_grid_laplacian_operator(u, grid_geometry=grid_geometry)
        lw = apply_monitor_grid_laplacian_operator(w, grid_geometry=grid_geometry)
    else:
        lu = apply_legacy_laplacian_operator(u, grid_geometry=grid_geometry)
        lw = apply_legacy_laplacian_operator(w, grid_geometry=grid_geometry)
    lhs = float(integrate_field(u * lw, grid_geometry=grid_geometry))
    rhs = float(integrate_field(lu * w, grid_geometry=grid_geometry))
    absolute_defect = abs(lhs - rhs)
    relative_defect = absolute_defect / max(abs(lhs), abs(rhs), 1.0e-14)
    return SelfAdjointnessProbe(
        lhs=lhs,
        rhs=rhs,
        absolute_defect=float(absolute_defect),
        relative_defect=float(relative_defect),
    )


def _boundary_split_diagnostic(
    grid_geometry: MonitorGridGeometry,
    poisson_result,
    full_residual: np.ndarray,
) -> BoundarySplitDiagnostic:
    interior = _interior_mask(grid_geometry.spec.shape)
    boundary_field = np.array(poisson_result.boundary_condition.boundary_values, copy=True)
    boundary_field[interior] = 0.0
    boundary_laplacian = apply_monitor_grid_laplacian_operator(
        boundary_field,
        grid_geometry=grid_geometry,
    )
    consistency = full_residual[interior] - 2.0 * boundary_laplacian[interior]
    correlation = np.corrcoef(
        full_residual[interior].reshape(-1),
        boundary_laplacian[interior].reshape(-1),
    )[0, 1]
    return BoundarySplitDiagnostic(
        boundary_laplacian_summary=_scalar_summary(boundary_laplacian[interior]),
        full_vs_twice_boundary_max_abs=float(np.max(np.abs(consistency))),
        full_vs_twice_boundary_rms=float(np.sqrt(np.mean(consistency * consistency))),
        correlation_with_boundary_laplacian=float(correlation),
    )


def evaluate_poisson_operator_route(
    *,
    case: BenchmarkCase,
    density_field: np.ndarray,
    density_label: str,
    grid_geometry: StructuredGridGeometry | MonitorGridGeometry,
    grid_type: str,
    multipole_order: int = 2,
    tolerance: float = 1.0e-8,
    max_iterations: int = 400,
) -> PoissonOperatorRouteResult:
    """Evaluate one legacy or A-grid Poisson-operator route for a fixed density."""

    poisson_result = solve_hartree_potential(
        grid_geometry=grid_geometry,
        rho=density_field,
        multipole_order=multipole_order,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if isinstance(grid_geometry, MonitorGridGeometry):
        laplacian = apply_monitor_grid_laplacian_operator(
            poisson_result.potential,
            grid_geometry=grid_geometry,
        )
    else:
        laplacian = apply_legacy_laplacian_operator(
            poisson_result.potential,
            grid_geometry=grid_geometry,
        )
    residual = laplacian + _FOUR_PI * density_field
    interior = _interior_mask(grid_geometry.spec.shape)
    hartree_energy = 0.5 * float(
        integrate_field(density_field * poisson_result.potential, grid_geometry=grid_geometry)
    )
    boundary_split = (
        _boundary_split_diagnostic(grid_geometry, poisson_result, residual)
        if isinstance(grid_geometry, MonitorGridGeometry)
        else None
    )
    return PoissonOperatorRouteResult(
        density_label=density_label,
        grid_type=grid_type,
        grid_parameter_summary=_grid_parameter_summary(grid_type),
        density_integral=float(integrate_field(density_field, grid_geometry=grid_geometry)),
        density_integral_error=float(
            integrate_field(density_field, grid_geometry=grid_geometry) - 2.0
        ),
        potential_summary=_scalar_summary(poisson_result.potential[interior]),
        laplacian_summary=_scalar_summary(laplacian[interior]),
        residual_summary=_scalar_summary(residual[interior]),
        hartree_energy=hartree_energy,
        solver_method=poisson_result.solver_method,
        solver_iterations=int(poisson_result.solver_iterations),
        solver_reported_residual_max=float(poisson_result.residual_max),
        boundary_summary=_boundary_summary(poisson_result),
        centerline_samples=_centerline_samples(
            density_field,
            poisson_result.potential,
            laplacian,
            residual,
            grid_geometry,
        ),
        region_diagnostics=_region_diagnostics(
            case,
            grid_geometry,
            poisson_result.potential,
            laplacian,
            residual,
        ),
        mirror_symmetric=bool(np.allclose(poisson_result.potential, poisson_result.potential[:, :, ::-1])),
        negative_interior_fraction=float(np.mean(poisson_result.potential[interior] < 0.0)),
        self_adjointness_probe=_self_adjointness_probe(grid_geometry),
        boundary_split_diagnostic=boundary_split,
    )


def _difference_summary(
    legacy_result: PoissonOperatorRouteResult,
    monitor_result: PoissonOperatorRouteResult,
) -> PoissonOperatorDifferenceSummary:
    legacy_samples = legacy_result.centerline_samples
    monitor_samples = monitor_result.centerline_samples
    differences = np.array(
        [
            monitor_sample.potential_value - legacy_sample.potential_value
            for legacy_sample, monitor_sample in zip(legacy_samples, monitor_samples, strict=True)
        ],
        dtype=np.float64,
    )
    z_values = np.array([sample.z_coordinate_bohr for sample in legacy_samples], dtype=np.float64)
    inner = np.abs(z_values) <= 1.0
    middle = (np.abs(z_values) > 1.0) & (np.abs(z_values) <= 4.0)
    outer = np.abs(z_values) > 4.0
    inner_mean = float(np.mean(np.abs(differences[inner])))
    middle_mean = float(np.mean(np.abs(differences[middle])))
    outer_mean = float(np.mean(np.abs(differences[outer])))
    if outer_mean > 1.5 * max(inner_mean, middle_mean):
        pattern = "far_field_dominated"
    elif inner_mean > 1.5 * max(middle_mean, outer_mean):
        pattern = "near_core_dominated"
    else:
        pattern = "broad_offset_like"
    return PoissonOperatorDifferenceSummary(
        hartree_energy_difference_ha=float(
            monitor_result.hartree_energy - legacy_result.hartree_energy
        ),
        hartree_energy_difference_mha=float(
            (monitor_result.hartree_energy - legacy_result.hartree_energy) * 1000.0
        ),
        potential_min_difference=float(
            monitor_result.potential_summary.minimum - legacy_result.potential_summary.minimum
        ),
        potential_max_difference=float(
            monitor_result.potential_summary.maximum - legacy_result.potential_summary.maximum
        ),
        negative_interior_fraction_difference=float(
            monitor_result.negative_interior_fraction - legacy_result.negative_interior_fraction
        ),
        residual_rms_ratio=float(
            monitor_result.residual_summary.rms / max(legacy_result.residual_summary.rms, 1.0e-16)
        ),
        centerline_max_abs_difference=float(np.max(np.abs(differences))),
        centerline_inner_mean_abs_difference=inner_mean,
        centerline_middle_mean_abs_difference=middle_mean,
        centerline_outer_mean_abs_difference=outer_mean,
        likely_difference_pattern=pattern,
    )


def _shape_scan(
    case: BenchmarkCase,
    shapes: tuple[tuple[int, int, int], ...] = ((59, 59, 71), (67, 67, 81), (75, 75, 91)),
) -> tuple[MonitorShapeScanPoint, ...]:
    base_parameters = build_h2_local_patch_development_element_parameters()
    results: list[MonitorShapeScanPoint] = []
    baseline_energy = None
    for shape in shapes:
        grid_geometry = build_monitor_grid_for_case(
            case,
            shape=shape,
            box_half_extents=(8.0, 8.0, 10.0),
            element_parameters=base_parameters,
        )
        density_field = _build_h2_frozen_density(case, grid_geometry)
        route_result = evaluate_poisson_operator_route(
            case=case,
            density_field=density_field,
            density_label="h2_singlet_frozen_density",
            grid_geometry=grid_geometry,
            grid_type="monitor_a_grid",
        )
        if baseline_energy is None and shape == (67, 67, 81):
            baseline_energy = route_result.hartree_energy
        center_sample = next(
            sample.potential_value
            for sample in route_result.centerline_samples
            if abs(sample.z_coordinate_bohr) < 1.0e-12
        )
        results.append(
            MonitorShapeScanPoint(
                shape=shape,
                hartree_energy=route_result.hartree_energy,
                residual_rms=route_result.residual_summary.rms,
                negative_interior_fraction=route_result.negative_interior_fraction,
                center_potential=float(center_sample),
                delta_vs_baseline_mha=0.0,
            )
        )
    if baseline_energy is None:
        baseline_energy = results[0].hartree_energy
    return tuple(
        MonitorShapeScanPoint(
            shape=result.shape,
            hartree_energy=result.hartree_energy,
            residual_rms=result.residual_rms,
            negative_interior_fraction=result.negative_interior_fraction,
            center_potential=result.center_potential,
            delta_vs_baseline_mha=float((result.hartree_energy - baseline_energy) * 1000.0),
        )
        for result in results
    )


def _diagnosis(
    h2_monitor: PoissonOperatorRouteResult,
    gaussian_monitor: PoissonOperatorRouteResult,
    shape_scan_results: tuple[MonitorShapeScanPoint, ...],
) -> str:
    if h2_monitor.boundary_split_diagnostic is not None:
        split = h2_monitor.boundary_split_diagnostic
        if (
            split.full_vs_twice_boundary_max_abs < 1.0e-6
            and abs(split.correlation_with_boundary_laplacian) > 0.999
            and h2_monitor.residual_summary.rms > 1.0
            and h2_monitor.solver_reported_residual_max < 1.0e-6
        ):
            return (
                "The dominant A-grid defect now looks like a monitor-path boundary-splitting / "
                "sign-consistency problem, not a solver-tolerance problem. For the H2 frozen "
                "density, the full operator residual L(v_H)+4*pi*rho matches 2*L(v_boundary) to "
                "roundoff, which strongly suggests that the monitor Poisson RHS assembly is "
                "inconsistent with the full operator identity."
            )

    shape_span = max(abs(point.delta_vs_baseline_mha) for point in shape_scan_results)
    if (
        gaussian_monitor.negative_interior_fraction > 0.5
        and h2_monitor.negative_interior_fraction > 0.5
        and h2_monitor.self_adjointness_probe.relative_defect < 1.0e-10
        and shape_span < 10.0
    ):
        return (
            "The same negative-far-field pathology appears for both the H2 frozen density and a "
            "simple Gaussian density, while the self-adjointness probe stays near machine "
            "precision and a small shape scan does not remove the problem. That points to a "
            "general monitor-grid Poisson assembly inconsistency rather than H2-specific physics "
            "or ordinary resolution error."
        )

    return (
        "The current A-grid Hartree mismatch is not explained by density normalization or by "
        "ordinary solver convergence. The remaining main suspect is the monitor-grid Poisson "
        "operator assembly itself, especially the consistency between boundary splitting and the "
        "full curvilinear Laplacian."
    )


def run_h2_monitor_grid_poisson_operator_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridPoissonOperatorAuditResult:
    """Run the H2 singlet frozen-density A-grid Poisson-operator audit."""

    legacy_grid = build_default_h2_grid_geometry(case=case)
    monitor_grid = build_h2_local_patch_development_monitor_grid()

    legacy_h2_density = _build_h2_frozen_density(case, legacy_grid)
    monitor_h2_density = _build_h2_frozen_density(case, monitor_grid)
    legacy_h2_result = evaluate_poisson_operator_route(
        case=case,
        density_field=legacy_h2_density,
        density_label="h2_singlet_frozen_density",
        grid_geometry=legacy_grid,
        grid_type="legacy",
    )
    monitor_h2_result = evaluate_poisson_operator_route(
        case=case,
        density_field=monitor_h2_density,
        density_label="h2_singlet_frozen_density",
        grid_geometry=monitor_grid,
        grid_type="monitor_a_grid",
    )

    legacy_gaussian_density = _build_gaussian_density(legacy_grid)
    monitor_gaussian_density = _build_gaussian_density(monitor_grid)
    legacy_gaussian_result = evaluate_poisson_operator_route(
        case=case,
        density_field=legacy_gaussian_density,
        density_label="single_center_gaussian_density",
        grid_geometry=legacy_grid,
        grid_type="legacy",
    )
    monitor_gaussian_result = evaluate_poisson_operator_route(
        case=case,
        density_field=monitor_gaussian_density,
        density_label="single_center_gaussian_density",
        grid_geometry=monitor_grid,
        grid_type="monitor_a_grid",
    )

    difference_summary = _difference_summary(legacy_h2_result, monitor_h2_result)
    shape_scan_results = _shape_scan(case)
    diagnosis = _diagnosis(
        h2_monitor=monitor_h2_result,
        gaussian_monitor=monitor_gaussian_result,
        shape_scan_results=shape_scan_results,
    )
    return H2MonitorGridPoissonOperatorAuditResult(
        legacy_h2_result=legacy_h2_result,
        monitor_h2_result=monitor_h2_result,
        legacy_gaussian_result=legacy_gaussian_result,
        monitor_gaussian_result=monitor_gaussian_result,
        difference_summary=difference_summary,
        shape_scan_results=shape_scan_results,
        diagnosis=diagnosis,
        note=(
            "This is a Poisson-operator localization audit only. It does not repair the "
            "implementation and does not migrate SCF, eigensolver, nonlocal, or patch-assisted "
            "Hartree logic."
        ),
    )


def _print_route_result(result: PoissonOperatorRouteResult) -> None:
    print(f"path: {result.grid_type} [{result.density_label}]")
    print(f"  grid summary: {result.grid_parameter_summary}")
    print(f"  rho_total integral: {result.density_integral:.12f}")
    print(f"  rho_total integral error: {result.density_integral_error:+.3e}")
    print(
        "  v_H summary [Ha]: "
        f"min={result.potential_summary.minimum:.12f}, "
        f"max={result.potential_summary.maximum:.12f}, "
        f"mean={result.potential_summary.mean:.12f}"
    )
    print(
        "  laplacian(v_H) summary [Ha/Bohr^2]: "
        f"min={result.laplacian_summary.minimum:.12f}, "
        f"max={result.laplacian_summary.maximum:.12f}, "
        f"mean={result.laplacian_summary.mean:.12f}, "
        f"rms={result.laplacian_summary.rms:.12f}"
    )
    print(
        "  residual summary [Ha/Bohr^2]: "
        f"min={result.residual_summary.minimum:.12f}, "
        f"max={result.residual_summary.maximum:.12f}, "
        f"mean={result.residual_summary.mean:.12f}, "
        f"rms={result.residual_summary.rms:.12f}"
    )
    print(f"  E_H [Ha]: {result.hartree_energy:.12f}")
    print(
        "  solver: "
        f"{result.solver_method}, iterations={result.solver_iterations}, "
        f"reported residual={result.solver_reported_residual_max:.3e}"
    )
    print(
        "  boundary summary: "
        f"order={result.boundary_summary.multipole_order}, "
        f"min={result.boundary_summary.boundary_min:.12f}, "
        f"max={result.boundary_summary.boundary_max:.12f}, "
        f"mean={result.boundary_summary.boundary_mean:.12f}"
    )
    print(f"  mirror symmetry: {result.mirror_symmetric}")
    print(f"  negative interior fraction: {result.negative_interior_fraction:.6f}")
    print(
        "  self-adjointness probe: "
        f"abs={result.self_adjointness_probe.absolute_defect:.3e}, "
        f"rel={result.self_adjointness_probe.relative_defect:.3e}"
    )
    if result.boundary_split_diagnostic is not None:
        split = result.boundary_split_diagnostic
        print(
            "  boundary-split diagnostic: "
            f"boundary_L_rms={split.boundary_laplacian_summary.rms:.12f}, "
            f"max|full_res-2Lb|={split.full_vs_twice_boundary_max_abs:.3e}, "
            f"rms(full_res-2Lb)={split.full_vs_twice_boundary_rms:.3e}, "
            f"corr={split.correlation_with_boundary_laplacian:.12f}"
        )
    print("  regional diagnostics:")
    for region in result.region_diagnostics:
        print(
            "    "
            f"{region.region_name}: count={region.point_count}, "
            f"v_mean={region.potential_mean:.12f}, "
            f"lap_mean={region.laplacian_mean:.12f}, "
            f"res_mean={region.residual_mean:.12f}, "
            f"res_rms={region.residual_rms:.12f}, "
            f"neg_frac={region.negative_potential_fraction:.6f}"
        )
    print("  center-line samples:")
    for sample in result.centerline_samples:
        print(
            "    "
            f"z={sample.z_coordinate_bohr:+5.1f} -> "
            f"rho={sample.rho_value:.12f}, "
            f"v={sample.potential_value:.12f}, "
            f"L(v)={sample.laplacian_value:.12f}, "
            f"res={sample.residual_value:.12f}"
        )


def print_h2_monitor_grid_poisson_operator_summary(
    result: H2MonitorGridPoissonOperatorAuditResult,
) -> None:
    """Print the compact A-grid Poisson-operator audit summary."""

    print("IsoGridDFT H2 singlet A-grid Poisson-operator audit")
    print(f"note: {result.note}")
    print()
    _print_route_result(result.legacy_h2_result)
    print()
    _print_route_result(result.monitor_h2_result)
    print()
    print("legacy vs A-grid [H2 frozen density]:")
    diff = result.difference_summary
    print(f"  E_H difference [mHa]: {diff.hartree_energy_difference_mha:+.3f}")
    print(f"  residual rms ratio: {diff.residual_rms_ratio:.6e}")
    print(
        "  center-line mean abs diff [Ha]: "
        f"inner={diff.centerline_inner_mean_abs_difference:.12f}, "
        f"middle={diff.centerline_middle_mean_abs_difference:.12f}, "
        f"outer={diff.centerline_outer_mean_abs_difference:.12f}"
    )
    print(f"  pattern guess: {diff.likely_difference_pattern}")
    print()
    print("Gaussian sanity density comparison:")
    _print_route_result(result.legacy_gaussian_result)
    print()
    _print_route_result(result.monitor_gaussian_result)
    print()
    print("A-grid shape scan:")
    for scan in result.shape_scan_results:
        print(
            "  "
            f"shape={scan.shape}: E_H={scan.hartree_energy:.12f} Ha, "
            f"delta_vs_baseline={scan.delta_vs_baseline_mha:+.3f} mHa, "
            f"residual_rms={scan.residual_rms:.12f}, "
            f"negative_frac={scan.negative_interior_fraction:.6f}, "
            f"center_v={scan.center_potential:.12f}"
        )
    print()
    print(f"diagnosis: {result.diagnosis}")


def main() -> int:
    result = run_h2_monitor_grid_poisson_operator_audit()
    print_h2_monitor_grid_poisson_operator_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
