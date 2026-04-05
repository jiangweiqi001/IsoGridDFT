"""Fixed-density Hartree/open-boundary diagnosis audit for the H2 A-grid path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import (
    H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR,
)
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import StructuredGridGeometry
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.ops import integrate_field

from .h2_monitor_grid_poisson_operator_audit import evaluate_poisson_operator_route
from .h2_monitor_grid_poisson_operator_audit import PoissonOperatorRouteResult
from .h2_monitor_grid_ts_eloc_audit import _build_h2_bonding_trial_orbital

GridGeometryLike = StructuredGridGeometry | MonitorGridGeometry
_DEFAULT_EXPANDED_MONITOR_BOX_HALF_EXTENTS_BOHR = (10.0, 10.0, 12.0)
_DEFAULT_TOLERANCE = 1.0e-8
_DEFAULT_MAX_ITERATIONS = 400
_DEFAULT_GAUSSIAN_ALPHA = 0.5
_DEFAULT_GAUSSIAN_SHIFT_BOHR = 1.5


@dataclass(frozen=True)
class FixedDensityHartreeRouteSummary:
    """Compact Hartree summary for one fixed-density route."""

    density_label: str
    grid_type: str
    box_half_extents_bohr: tuple[float, float, float]
    density_integral: float
    hartree_energy: float
    residual_rms: float
    center_potential: float
    far_field_mean_potential: float
    far_field_min_potential: float
    far_field_max_potential: float
    far_field_negative_potential_fraction: float
    boundary_mean: float
    outer_centerline_mean_abs_potential: float
    solver_iterations: int


@dataclass(frozen=True)
class FixedDensityDifferenceSummary:
    """Difference summary between legacy and A-grid routes for one density."""

    density_label: str
    monitor_minus_legacy_hartree_energy_mha: float
    monitor_minus_legacy_center_potential: float
    monitor_minus_legacy_far_field_mean_potential: float
    monitor_minus_legacy_boundary_mean: float
    centerline_inner_mean_abs_difference: float
    centerline_middle_mean_abs_difference: float
    centerline_outer_mean_abs_difference: float
    monitor_minus_legacy_far_field_negative_fraction: float
    likely_difference_pattern: str


@dataclass(frozen=True)
class MonitorBoxSensitivitySummary:
    """Sensitivity of the A-grid Hartree route to monitor-box enlargement."""

    density_label: str
    expanded_minus_baseline_hartree_energy_mha: float
    expanded_minus_baseline_center_potential: float
    expanded_minus_baseline_far_field_mean_potential: float
    expanded_minus_baseline_boundary_mean: float
    expanded_minus_baseline_outer_centerline_mean_abs_difference: float
    expanded_minus_baseline_far_field_negative_fraction: float
    likely_sensitivity_pattern: str


@dataclass(frozen=True)
class GaussianShiftSensitivitySummary:
    """Translation-reasonableness smoke check on the same A-grid box."""

    shifted_minus_centered_hartree_energy_mha: float
    shifted_minus_centered_center_potential: float
    shifted_minus_centered_far_field_mean_potential: float
    shifted_minus_centered_boundary_mean: float
    shifted_minus_centered_outer_centerline_mean_abs_difference: float
    shifted_minus_centered_far_field_negative_fraction: float
    likely_sensitivity_pattern: str


@dataclass(frozen=True)
class H2HartreeBoundaryDiagnosisAuditResult:
    """Top-level diagnosis result for fixed-density Hartree/open-boundary behavior."""

    gaussian_centered_legacy: FixedDensityHartreeRouteSummary
    gaussian_centered_monitor: FixedDensityHartreeRouteSummary
    gaussian_centered_monitor_expanded: FixedDensityHartreeRouteSummary
    gaussian_shifted_monitor: FixedDensityHartreeRouteSummary
    h2_frozen_legacy: FixedDensityHartreeRouteSummary
    h2_frozen_monitor: FixedDensityHartreeRouteSummary
    h2_frozen_monitor_expanded: FixedDensityHartreeRouteSummary
    gaussian_centered_difference: FixedDensityDifferenceSummary
    h2_frozen_difference: FixedDensityDifferenceSummary
    gaussian_centered_box_sensitivity: MonitorBoxSensitivitySummary
    h2_frozen_box_sensitivity: MonitorBoxSensitivitySummary
    gaussian_shift_sensitivity: GaussianShiftSensitivitySummary
    primary_verdict: str
    diagnosis: str
    note: str


def _box_half_extents(
    grid_geometry: GridGeometryLike,
) -> tuple[float, float, float]:
    if isinstance(grid_geometry, MonitorGridGeometry):
        bounds = grid_geometry.spec.box_bounds
        return tuple(
            0.5 * float(upper - lower)
            for lower, upper in bounds
        )
    return (
        float(max(abs(grid_geometry.spec.x_axis.lower_offset), abs(grid_geometry.spec.x_axis.upper_offset))),
        float(max(abs(grid_geometry.spec.y_axis.lower_offset), abs(grid_geometry.spec.y_axis.upper_offset))),
        float(max(abs(grid_geometry.spec.z_axis.lower_offset), abs(grid_geometry.spec.z_axis.upper_offset))),
    )


def _build_h2_frozen_density(
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    orbital = _build_h2_bonding_trial_orbital(case=case, grid_geometry=grid_geometry)
    return np.asarray(2.0 * np.abs(orbital) ** 2, dtype=np.float64)


def _build_gaussian_density(
    grid_geometry: GridGeometryLike,
    *,
    alpha: float = _DEFAULT_GAUSSIAN_ALPHA,
    target_charge: float = 2.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    rho = np.exp(
        -alpha
        * (
            (grid_geometry.x_points - center[0]) ** 2
            + (grid_geometry.y_points - center[1]) ** 2
            + (grid_geometry.z_points - center[2]) ** 2
        )
    )
    normalization = float(integrate_field(rho, grid_geometry=grid_geometry))
    return np.asarray(target_charge * rho / normalization, dtype=np.float64)


def _far_field_diagnostic(route_result: PoissonOperatorRouteResult):
    for diagnostic in route_result.region_diagnostics:
        if diagnostic.region_name == "far_field":
            return diagnostic
    raise ValueError("Expected far_field region diagnostic to be present.")


def _centerline_band_means(
    route_result: PoissonOperatorRouteResult,
) -> tuple[float, float, float]:
    z_values = np.array(
        [sample.z_coordinate_bohr for sample in route_result.centerline_samples],
        dtype=np.float64,
    )
    potentials = np.array(
        [sample.potential_value for sample in route_result.centerline_samples],
        dtype=np.float64,
    )
    inner = np.abs(z_values) <= 1.0
    middle = (np.abs(z_values) > 1.0) & (np.abs(z_values) <= 4.0)
    outer = np.abs(z_values) > 4.0
    return (
        float(np.mean(np.abs(potentials[inner]))),
        float(np.mean(np.abs(potentials[middle]))),
        float(np.mean(np.abs(potentials[outer]))),
    )


def evaluate_fixed_density_hartree_route(
    *,
    case: BenchmarkCase,
    grid_geometry: GridGeometryLike,
    grid_type: str,
    density_field: np.ndarray,
    density_label: str,
    tolerance: float = _DEFAULT_TOLERANCE,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> FixedDensityHartreeRouteSummary:
    """Evaluate one fixed-density Hartree route on a selected grid."""

    route_result = evaluate_poisson_operator_route(
        case=case,
        density_field=density_field,
        density_label=density_label,
        grid_geometry=grid_geometry,
        grid_type=grid_type,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    far_field = _far_field_diagnostic(route_result)
    _, _, outer_mean_abs_potential = _centerline_band_means(route_result)
    return FixedDensityHartreeRouteSummary(
        density_label=density_label,
        grid_type=grid_type,
        box_half_extents_bohr=_box_half_extents(grid_geometry),
        density_integral=float(route_result.density_integral),
        hartree_energy=float(route_result.hartree_energy),
        residual_rms=float(route_result.residual_summary.rms),
        center_potential=float(
            next(
                sample.potential_value
                for sample in route_result.centerline_samples
                if abs(sample.z_coordinate_bohr) < 1.0e-12
            )
        ),
        far_field_mean_potential=float(far_field.potential_mean),
        far_field_min_potential=float(far_field.potential_min),
        far_field_max_potential=float(far_field.potential_max),
        far_field_negative_potential_fraction=float(far_field.negative_potential_fraction),
        boundary_mean=float(route_result.boundary_summary.boundary_mean),
        outer_centerline_mean_abs_potential=float(outer_mean_abs_potential),
        solver_iterations=int(route_result.solver_iterations),
    )


def _classify_difference_pattern(
    *,
    inner_mean_abs: float,
    middle_mean_abs: float,
    outer_mean_abs: float,
    far_field_negative_fraction_difference: float,
) -> str:
    if outer_mean_abs > 1.5 * max(inner_mean_abs, middle_mean_abs):
        return "far_field_tail_dominated"
    if abs(far_field_negative_fraction_difference) > 0.05:
        return "far_field_sign_sensitive"
    if inner_mean_abs > 1.5 * max(middle_mean_abs, outer_mean_abs):
        return "near_core_or_geometry_dominated"
    return "broad_or_mixed"


def _difference_summary(
    *,
    density_label: str,
    legacy_route: FixedDensityHartreeRouteSummary,
    monitor_route: FixedDensityHartreeRouteSummary,
    legacy_operator_route: PoissonOperatorRouteResult,
    monitor_operator_route: PoissonOperatorRouteResult,
) -> FixedDensityDifferenceSummary:
    legacy_samples = np.array(
        [sample.potential_value for sample in legacy_operator_route.centerline_samples],
        dtype=np.float64,
    )
    monitor_samples = np.array(
        [sample.potential_value for sample in monitor_operator_route.centerline_samples],
        dtype=np.float64,
    )
    z_values = np.array(
        [sample.z_coordinate_bohr for sample in legacy_operator_route.centerline_samples],
        dtype=np.float64,
    )
    differences = monitor_samples - legacy_samples
    inner = np.abs(z_values) <= 1.0
    middle = (np.abs(z_values) > 1.0) & (np.abs(z_values) <= 4.0)
    outer = np.abs(z_values) > 4.0
    inner_mean_abs = float(np.mean(np.abs(differences[inner])))
    middle_mean_abs = float(np.mean(np.abs(differences[middle])))
    outer_mean_abs = float(np.mean(np.abs(differences[outer])))
    negative_fraction_difference = float(
        monitor_route.far_field_negative_potential_fraction
        - legacy_route.far_field_negative_potential_fraction
    )
    return FixedDensityDifferenceSummary(
        density_label=density_label,
        monitor_minus_legacy_hartree_energy_mha=float(
            (monitor_route.hartree_energy - legacy_route.hartree_energy) * 1000.0
        ),
        monitor_minus_legacy_center_potential=float(
            monitor_route.center_potential - legacy_route.center_potential
        ),
        monitor_minus_legacy_far_field_mean_potential=float(
            monitor_route.far_field_mean_potential - legacy_route.far_field_mean_potential
        ),
        monitor_minus_legacy_boundary_mean=float(
            monitor_route.boundary_mean - legacy_route.boundary_mean
        ),
        centerline_inner_mean_abs_difference=inner_mean_abs,
        centerline_middle_mean_abs_difference=middle_mean_abs,
        centerline_outer_mean_abs_difference=outer_mean_abs,
        monitor_minus_legacy_far_field_negative_fraction=negative_fraction_difference,
        likely_difference_pattern=_classify_difference_pattern(
            inner_mean_abs=inner_mean_abs,
            middle_mean_abs=middle_mean_abs,
            outer_mean_abs=outer_mean_abs,
            far_field_negative_fraction_difference=negative_fraction_difference,
        ),
    )


def _box_sensitivity_summary(
    *,
    density_label: str,
    baseline_route: FixedDensityHartreeRouteSummary,
    expanded_route: FixedDensityHartreeRouteSummary,
    baseline_operator_route: PoissonOperatorRouteResult,
    expanded_operator_route: PoissonOperatorRouteResult,
) -> MonitorBoxSensitivitySummary:
    baseline_samples = np.array(
        [sample.potential_value for sample in baseline_operator_route.centerline_samples],
        dtype=np.float64,
    )
    expanded_samples = np.array(
        [sample.potential_value for sample in expanded_operator_route.centerline_samples],
        dtype=np.float64,
    )
    z_values = np.array(
        [sample.z_coordinate_bohr for sample in baseline_operator_route.centerline_samples],
        dtype=np.float64,
    )
    outer = np.abs(z_values) > 4.0
    outer_mean_abs_difference = float(
        np.mean(np.abs(expanded_samples[outer] - baseline_samples[outer]))
    )
    negative_fraction_difference = float(
        expanded_route.far_field_negative_potential_fraction
        - baseline_route.far_field_negative_potential_fraction
    )
    return MonitorBoxSensitivitySummary(
        density_label=density_label,
        expanded_minus_baseline_hartree_energy_mha=float(
            (expanded_route.hartree_energy - baseline_route.hartree_energy) * 1000.0
        ),
        expanded_minus_baseline_center_potential=float(
            expanded_route.center_potential - baseline_route.center_potential
        ),
        expanded_minus_baseline_far_field_mean_potential=float(
            expanded_route.far_field_mean_potential - baseline_route.far_field_mean_potential
        ),
        expanded_minus_baseline_boundary_mean=float(
            expanded_route.boundary_mean - baseline_route.boundary_mean
        ),
        expanded_minus_baseline_outer_centerline_mean_abs_difference=outer_mean_abs_difference,
        expanded_minus_baseline_far_field_negative_fraction=negative_fraction_difference,
        likely_sensitivity_pattern=_classify_difference_pattern(
            inner_mean_abs=abs(expanded_route.center_potential - baseline_route.center_potential),
            middle_mean_abs=abs(
                expanded_route.far_field_mean_potential - baseline_route.far_field_mean_potential
            ),
            outer_mean_abs=outer_mean_abs_difference,
            far_field_negative_fraction_difference=negative_fraction_difference,
        ),
    )


def _gaussian_shift_sensitivity_summary(
    *,
    centered_route: FixedDensityHartreeRouteSummary,
    shifted_route: FixedDensityHartreeRouteSummary,
    centered_operator_route: PoissonOperatorRouteResult,
    shifted_operator_route: PoissonOperatorRouteResult,
) -> GaussianShiftSensitivitySummary:
    centered_samples = np.array(
        [sample.potential_value for sample in centered_operator_route.centerline_samples],
        dtype=np.float64,
    )
    shifted_samples = np.array(
        [sample.potential_value for sample in shifted_operator_route.centerline_samples],
        dtype=np.float64,
    )
    z_values = np.array(
        [sample.z_coordinate_bohr for sample in centered_operator_route.centerline_samples],
        dtype=np.float64,
    )
    outer = np.abs(z_values) > 4.0
    outer_mean_abs_difference = float(
        np.mean(np.abs(shifted_samples[outer] - centered_samples[outer]))
    )
    negative_fraction_difference = float(
        shifted_route.far_field_negative_potential_fraction
        - centered_route.far_field_negative_potential_fraction
    )
    return GaussianShiftSensitivitySummary(
        shifted_minus_centered_hartree_energy_mha=float(
            (shifted_route.hartree_energy - centered_route.hartree_energy) * 1000.0
        ),
        shifted_minus_centered_center_potential=float(
            shifted_route.center_potential - centered_route.center_potential
        ),
        shifted_minus_centered_far_field_mean_potential=float(
            shifted_route.far_field_mean_potential - centered_route.far_field_mean_potential
        ),
        shifted_minus_centered_boundary_mean=float(
            shifted_route.boundary_mean - centered_route.boundary_mean
        ),
        shifted_minus_centered_outer_centerline_mean_abs_difference=outer_mean_abs_difference,
        shifted_minus_centered_far_field_negative_fraction=negative_fraction_difference,
        likely_sensitivity_pattern=_classify_difference_pattern(
            inner_mean_abs=abs(shifted_route.center_potential - centered_route.center_potential),
            middle_mean_abs=abs(
                shifted_route.far_field_mean_potential - centered_route.far_field_mean_potential
            ),
            outer_mean_abs=outer_mean_abs_difference,
            far_field_negative_fraction_difference=negative_fraction_difference,
        ),
    )


def _build_monitor_geometry(
    case: BenchmarkCase,
    *,
    shape: tuple[int, int, int],
    box_half_extents: tuple[float, float, float],
) -> MonitorGridGeometry:
    return build_monitor_grid_for_case(
        case,
        shape=shape,
        box_half_extents=box_half_extents,
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )


def _run_operator_route(
    *,
    case: BenchmarkCase,
    density_field: np.ndarray,
    density_label: str,
    grid_geometry: GridGeometryLike,
    grid_type: str,
    tolerance: float,
    max_iterations: int,
) -> PoissonOperatorRouteResult:
    return evaluate_poisson_operator_route(
        case=case,
        density_field=density_field,
        density_label=density_label,
        grid_geometry=grid_geometry,
        grid_type=grid_type,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _diagnose(
    *,
    gaussian_centered_difference: FixedDensityDifferenceSummary,
    h2_frozen_difference: FixedDensityDifferenceSummary,
    gaussian_centered_box_sensitivity: MonitorBoxSensitivitySummary,
    h2_frozen_box_sensitivity: MonitorBoxSensitivitySummary,
    gaussian_shift_sensitivity: GaussianShiftSensitivitySummary,
    gaussian_centered_monitor: FixedDensityHartreeRouteSummary,
) -> tuple[str, str]:
    gaussian_gap_mha = abs(
        gaussian_centered_difference.monitor_minus_legacy_hartree_energy_mha
    )
    gaussian_box_mha = abs(
        gaussian_centered_box_sensitivity.expanded_minus_baseline_hartree_energy_mha
    )
    gaussian_shift_mha = abs(
        gaussian_shift_sensitivity.shifted_minus_centered_hartree_energy_mha
    )
    h2_gap_mha = abs(h2_frozen_difference.monitor_minus_legacy_hartree_energy_mha)
    gaussian_tail_dominated = (
        gaussian_centered_difference.likely_difference_pattern == "far_field_tail_dominated"
        or gaussian_centered_box_sensitivity.likely_sensitivity_pattern == "far_field_tail_dominated"
        or gaussian_shift_sensitivity.likely_sensitivity_pattern == "far_field_tail_dominated"
    )
    gaussian_tail_sign_issue = (
        gaussian_centered_difference.likely_difference_pattern == "far_field_sign_sensitive"
        or gaussian_centered_monitor.far_field_negative_potential_fraction > 0.05
    )

    if (
        gaussian_gap_mha > 1.0
        and (
            gaussian_box_mha > 0.25 * gaussian_gap_mha
            or gaussian_shift_mha > 0.25 * gaussian_gap_mha
            or gaussian_tail_dominated
            or gaussian_tail_sign_issue
        )
    ):
        return (
            "likely_electrostatics_or_boundary",
            "The SCF-free Gaussian density already shows a material A-grid-vs-legacy Hartree gap, "
            "and that gap couples to far-field / box / translation sensitivity. That points more to "
            "open-boundary electrostatics or tail treatment than to the SCF fixed-point map itself.",
        )
    if (
        gaussian_gap_mha < 0.5
        and gaussian_box_mha < 0.25
        and gaussian_shift_mha < 0.25
        and h2_gap_mha > 1.0
    ):
        return (
            "likely_scf_or_preconditioning",
            "The SCF-free Gaussian route looks reasonably stable across legacy/A-grid and box/shift checks, "
            "while the H2 frozen-density route still shows the larger mismatch. That points more to a "
            "density-response / fixed-point / preconditioning issue than to a primary Hartree boundary defect.",
        )
    if (
        abs(h2_frozen_box_sensitivity.expanded_minus_baseline_hartree_energy_mha) > 0.5
        or gaussian_tail_dominated
        or gaussian_tail_sign_issue
    ):
        return (
            "likely_electrostatics_or_boundary",
            "The strongest sensitivity remains tied to far-field behavior and monitor-box changes, "
            "so the A-grid Hartree/open-boundary path still looks like the leading suspect.",
        )
    return (
        "mixed_or_inconclusive",
        "The fixed-density audit does not isolate one dominant failure mode cleanly. There is some "
        "A-grid Hartree sensitivity, but not enough yet to rule out remaining SCF/preconditioning effects.",
    )


def run_h2_hartree_boundary_diagnosis_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    monitor_shape: tuple[int, int, int] = H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE,
    baseline_monitor_box_half_extents: tuple[float, float, float] = (
        H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
    ),
    expanded_monitor_box_half_extents: tuple[float, float, float] = (
        _DEFAULT_EXPANDED_MONITOR_BOX_HALF_EXTENTS_BOHR
    ),
    gaussian_shift_bohr: float = _DEFAULT_GAUSSIAN_SHIFT_BOHR,
    tolerance: float = _DEFAULT_TOLERANCE,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> H2HartreeBoundaryDiagnosisAuditResult:
    """Run a fixed-density Hartree/open-boundary diagnosis audit for H2."""

    legacy_geometry = build_default_h2_grid_geometry(case=case)
    monitor_geometry = _build_monitor_geometry(
        case,
        shape=monitor_shape,
        box_half_extents=baseline_monitor_box_half_extents,
    )
    monitor_geometry_expanded = _build_monitor_geometry(
        case,
        shape=monitor_shape,
        box_half_extents=expanded_monitor_box_half_extents,
    )

    gaussian_centered_legacy_density = _build_gaussian_density(legacy_geometry)
    gaussian_centered_monitor_density = _build_gaussian_density(monitor_geometry)
    gaussian_centered_monitor_expanded_density = _build_gaussian_density(monitor_geometry_expanded)
    gaussian_shifted_monitor_density = _build_gaussian_density(
        monitor_geometry,
        center=(0.0, 0.0, gaussian_shift_bohr),
    )
    h2_frozen_legacy_density = _build_h2_frozen_density(case, legacy_geometry)
    h2_frozen_monitor_density = _build_h2_frozen_density(case, monitor_geometry)
    h2_frozen_monitor_expanded_density = _build_h2_frozen_density(case, monitor_geometry_expanded)

    gaussian_centered_legacy_operator = _run_operator_route(
        case=case,
        density_field=gaussian_centered_legacy_density,
        density_label="gaussian_centered",
        grid_geometry=legacy_geometry,
        grid_type="legacy",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    gaussian_centered_monitor_operator = _run_operator_route(
        case=case,
        density_field=gaussian_centered_monitor_density,
        density_label="gaussian_centered",
        grid_geometry=monitor_geometry,
        grid_type="monitor_a_grid",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    gaussian_centered_monitor_expanded_operator = _run_operator_route(
        case=case,
        density_field=gaussian_centered_monitor_expanded_density,
        density_label="gaussian_centered",
        grid_geometry=monitor_geometry_expanded,
        grid_type="monitor_a_grid_expanded_box",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    gaussian_shifted_monitor_operator = _run_operator_route(
        case=case,
        density_field=gaussian_shifted_monitor_density,
        density_label="gaussian_shifted",
        grid_geometry=monitor_geometry,
        grid_type="monitor_a_grid",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    h2_frozen_legacy_operator = _run_operator_route(
        case=case,
        density_field=h2_frozen_legacy_density,
        density_label="h2_frozen_bonding_density",
        grid_geometry=legacy_geometry,
        grid_type="legacy",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    h2_frozen_monitor_operator = _run_operator_route(
        case=case,
        density_field=h2_frozen_monitor_density,
        density_label="h2_frozen_bonding_density",
        grid_geometry=monitor_geometry,
        grid_type="monitor_a_grid",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    h2_frozen_monitor_expanded_operator = _run_operator_route(
        case=case,
        density_field=h2_frozen_monitor_expanded_density,
        density_label="h2_frozen_bonding_density",
        grid_geometry=monitor_geometry_expanded,
        grid_type="monitor_a_grid_expanded_box",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    gaussian_centered_legacy = evaluate_fixed_density_hartree_route(
        case=case,
        grid_geometry=legacy_geometry,
        grid_type="legacy",
        density_field=gaussian_centered_legacy_density,
        density_label="gaussian_centered",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    gaussian_centered_monitor = evaluate_fixed_density_hartree_route(
        case=case,
        grid_geometry=monitor_geometry,
        grid_type="monitor_a_grid",
        density_field=gaussian_centered_monitor_density,
        density_label="gaussian_centered",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    gaussian_centered_monitor_expanded = evaluate_fixed_density_hartree_route(
        case=case,
        grid_geometry=monitor_geometry_expanded,
        grid_type="monitor_a_grid_expanded_box",
        density_field=gaussian_centered_monitor_expanded_density,
        density_label="gaussian_centered",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    gaussian_shifted_monitor = evaluate_fixed_density_hartree_route(
        case=case,
        grid_geometry=monitor_geometry,
        grid_type="monitor_a_grid",
        density_field=gaussian_shifted_monitor_density,
        density_label="gaussian_shifted",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    h2_frozen_legacy = evaluate_fixed_density_hartree_route(
        case=case,
        grid_geometry=legacy_geometry,
        grid_type="legacy",
        density_field=h2_frozen_legacy_density,
        density_label="h2_frozen_bonding_density",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    h2_frozen_monitor = evaluate_fixed_density_hartree_route(
        case=case,
        grid_geometry=monitor_geometry,
        grid_type="monitor_a_grid",
        density_field=h2_frozen_monitor_density,
        density_label="h2_frozen_bonding_density",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    h2_frozen_monitor_expanded = evaluate_fixed_density_hartree_route(
        case=case,
        grid_geometry=monitor_geometry_expanded,
        grid_type="monitor_a_grid_expanded_box",
        density_field=h2_frozen_monitor_expanded_density,
        density_label="h2_frozen_bonding_density",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    gaussian_centered_difference = _difference_summary(
        density_label="gaussian_centered",
        legacy_route=gaussian_centered_legacy,
        monitor_route=gaussian_centered_monitor,
        legacy_operator_route=gaussian_centered_legacy_operator,
        monitor_operator_route=gaussian_centered_monitor_operator,
    )
    h2_frozen_difference = _difference_summary(
        density_label="h2_frozen_bonding_density",
        legacy_route=h2_frozen_legacy,
        monitor_route=h2_frozen_monitor,
        legacy_operator_route=h2_frozen_legacy_operator,
        monitor_operator_route=h2_frozen_monitor_operator,
    )
    gaussian_centered_box_sensitivity = _box_sensitivity_summary(
        density_label="gaussian_centered",
        baseline_route=gaussian_centered_monitor,
        expanded_route=gaussian_centered_monitor_expanded,
        baseline_operator_route=gaussian_centered_monitor_operator,
        expanded_operator_route=gaussian_centered_monitor_expanded_operator,
    )
    h2_frozen_box_sensitivity = _box_sensitivity_summary(
        density_label="h2_frozen_bonding_density",
        baseline_route=h2_frozen_monitor,
        expanded_route=h2_frozen_monitor_expanded,
        baseline_operator_route=h2_frozen_monitor_operator,
        expanded_operator_route=h2_frozen_monitor_expanded_operator,
    )
    gaussian_shift_sensitivity = _gaussian_shift_sensitivity_summary(
        centered_route=gaussian_centered_monitor,
        shifted_route=gaussian_shifted_monitor,
        centered_operator_route=gaussian_centered_monitor_operator,
        shifted_operator_route=gaussian_shifted_monitor_operator,
    )
    primary_verdict, diagnosis = _diagnose(
        gaussian_centered_difference=gaussian_centered_difference,
        h2_frozen_difference=h2_frozen_difference,
        gaussian_centered_box_sensitivity=gaussian_centered_box_sensitivity,
        h2_frozen_box_sensitivity=h2_frozen_box_sensitivity,
        gaussian_shift_sensitivity=gaussian_shift_sensitivity,
        gaussian_centered_monitor=gaussian_centered_monitor,
    )
    return H2HartreeBoundaryDiagnosisAuditResult(
        gaussian_centered_legacy=gaussian_centered_legacy,
        gaussian_centered_monitor=gaussian_centered_monitor,
        gaussian_centered_monitor_expanded=gaussian_centered_monitor_expanded,
        gaussian_shifted_monitor=gaussian_shifted_monitor,
        h2_frozen_legacy=h2_frozen_legacy,
        h2_frozen_monitor=h2_frozen_monitor,
        h2_frozen_monitor_expanded=h2_frozen_monitor_expanded,
        gaussian_centered_difference=gaussian_centered_difference,
        h2_frozen_difference=h2_frozen_difference,
        gaussian_centered_box_sensitivity=gaussian_centered_box_sensitivity,
        h2_frozen_box_sensitivity=h2_frozen_box_sensitivity,
        gaussian_shift_sensitivity=gaussian_shift_sensitivity,
        primary_verdict=primary_verdict,
        diagnosis=diagnosis,
        note=(
            "This audit fixes the input density and diagnoses Hartree/open-boundary behavior without "
            "running SCF. It is intended to separate electrostatics/boundary sensitivity from outer "
            "fixed-point/preconditioning difficulty."
        ),
    )


def main() -> None:
    result = run_h2_hartree_boundary_diagnosis_audit()
    print(f"primary verdict: {result.primary_verdict}")
    print(f"diagnosis: {result.diagnosis}")
    print(
        "gaussian centered: "
        f"monitor-legacy dE_H={result.gaussian_centered_difference.monitor_minus_legacy_hartree_energy_mha:+.3f} mHa, "
        f"box sensitivity={result.gaussian_centered_box_sensitivity.expanded_minus_baseline_hartree_energy_mha:+.3f} mHa, "
        f"shift sensitivity={result.gaussian_shift_sensitivity.shifted_minus_centered_hartree_energy_mha:+.3f} mHa"
    )
    print(
        "h2 frozen: "
        f"monitor-legacy dE_H={result.h2_frozen_difference.monitor_minus_legacy_hartree_energy_mha:+.3f} mHa, "
        f"box sensitivity={result.h2_frozen_box_sensitivity.expanded_minus_baseline_hartree_energy_mha:+.3f} mHa"
    )


if __name__ == "__main__":
    main()
