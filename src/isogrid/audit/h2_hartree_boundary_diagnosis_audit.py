"""Fixed-density Hartree/open-boundary diagnosis audit for the H2 A-grid path."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from math import erf
from math import sqrt

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
from isogrid.grid.monitor_builder import build_monitor_grid_spec_for_case
from isogrid.grid.monitor_geometry import _backtracking_update
from isogrid.grid.monitor_geometry import _basic_geometry_from_coordinates
from isogrid.grid.monitor_geometry import _smooth_monitor_field
from isogrid.grid.monitor_geometry import _solve_weighted_harmonic_coordinates
from isogrid.grid.monitor_geometry import build_reference_box_coordinates
from isogrid.grid.monitor_geometry import evaluate_global_monitor_field

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
    total_charge: float
    dipole_norm: float
    quadrupole_norm: float
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
class MonitorVolumeConsistencySummary:
    """Consistency of the monitor integration measure against the physical box volume."""

    physical_box_volume: float
    cell_volume_sum: float
    trapezoidal_cell_volume_sum: float
    point_volume_relative_error: float
    trapezoidal_relative_error: float


@dataclass(frozen=True)
class GaussianRepresentationConsistencySummary:
    """Centered-Gaussian moments under uniform-box and mapped-monitor measures."""

    uniform_box_total_charge: float
    uniform_box_dipole_norm: float
    uniform_box_quadrupole_norm: float
    uniform_box_with_monitor_weights_total_charge: float
    uniform_box_with_monitor_weights_dipole_norm: float
    uniform_box_with_monitor_weights_quadrupole_norm: float
    mapped_monitor_with_uniform_weights_total_charge: float
    mapped_monitor_with_uniform_weights_dipole_norm: float
    mapped_monitor_with_uniform_weights_quadrupole_norm: float
    mapped_monitor_total_charge: float
    mapped_monitor_dipole_norm: float
    mapped_monitor_quadrupole_norm: float


@dataclass(frozen=True)
class MonitorInversionSymmetrySummary:
    """Very local inversion-pairing diagnostics for the centered-Gaussian monitor route."""

    coordinate_pairing_max_abs: float
    cell_volume_pairing_max_abs: float
    gaussian_density_pairing_rms: float
    gaussian_dipole_integrand_pairing_rms: float


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
    monitor_volume_consistency: MonitorVolumeConsistencySummary
    gaussian_representation_consistency: GaussianRepresentationConsistencySummary
    monitor_inversion_symmetry: MonitorInversionSymmetrySummary
    primary_verdict: str
    diagnosis: str
    note: str


@dataclass(frozen=True)
class HartreeBoundaryShapeSweepPoint:
    """One fixed-density Hartree diagnosis sample at a selected monitor shape."""

    monitor_shape: tuple[int, int, int]
    gaussian_centered_monitor_quadrupole_norm: float
    gaussian_monitor_minus_legacy_hartree_energy_mha: float
    h2_frozen_monitor_minus_legacy_hartree_energy_mha: float
    gaussian_box_expand_sensitivity_mha: float
    h2_frozen_box_expand_sensitivity_mha: float


@dataclass(frozen=True)
class HartreeBoundaryShapeSweepAuditResult:
    """Resolution sweep summary for fixed-density Hartree/open-boundary behavior."""

    points: tuple[HartreeBoundaryShapeSweepPoint, ...]
    trend_verdict: str
    diagnosis: str
    note: str


@dataclass(frozen=True)
class MeasureLedgerIntegralSummary:
    """One test-function integral under one discrete measure."""

    function_label: str
    value: float
    reference_value: float
    bias: float


@dataclass(frozen=True)
class MeasureLedgerPathSummary:
    """One audit ledger row for a concrete path/measure pairing."""

    path_name: str
    role: str
    measure_name: str
    measure_description: str
    integrals: tuple[MeasureLedgerIntegralSummary, ...]


@dataclass(frozen=True)
class HartreeMeasureLedgerAuditResult:
    """Ledger audit for monitor-grid measures used by Hartree-related paths."""

    path_summaries: tuple[MeasureLedgerPathSummary, ...]
    note: str


@dataclass(frozen=True)
class CellVolumeConstructionSummary:
    """Trace how monitor cell volumes are constructed from the mapping Jacobian."""

    logical_cell_volume: float
    physical_box_volume: float
    stored_cell_volume_sum: float
    recomputed_cell_volume_sum: float
    max_abs_cell_volume_difference: float
    rms_cell_volume_difference: float


@dataclass(frozen=True)
class PolynomialExactnessRow:
    """Integral biases for one polynomial under several discrete measures."""

    function_label: str
    reference_value: float
    uniform_weight_value: float
    uniform_weight_bias: float
    current_cell_volume_value: float
    current_cell_volume_bias: float
    trapezoidal_adjusted_value: float
    trapezoidal_adjusted_bias: float
    uniform_coordinates_monitor_weights_value: float
    uniform_coordinates_monitor_weights_bias: float
    mapped_coordinates_uniform_weights_value: float
    mapped_coordinates_uniform_weights_bias: float


@dataclass(frozen=True)
class SecondOrderRegionRow:
    """Regional mapping-distortion summary for one second-order polynomial."""

    function_label: str
    boundary_mean_abs_mapping_distortion: float
    interior_mean_abs_mapping_distortion: float
    high_jacobian_mean_abs_mapping_distortion: float
    low_jacobian_mean_abs_mapping_distortion: float
    z_dominant_mean_abs_mapping_distortion: float
    xy_comparable_mean_abs_mapping_distortion: float


@dataclass(frozen=True)
class GeometryRepresentationErrorRegionSummary:
    """Where mapping and weight distortions are largest on the monitor grid."""

    boundary_mean_abs_r2_mapping_distortion: float
    interior_mean_abs_r2_mapping_distortion: float
    high_jacobian_mean_abs_r2_mapping_distortion: float
    low_jacobian_mean_abs_r2_mapping_distortion: float
    boundary_mean_abs_weight_delta: float
    interior_mean_abs_weight_delta: float


@dataclass(frozen=True)
class MappingZStretchSummary:
    """Local z-direction stretch diagnostics for the monitor mapping."""

    uniform_physical_dz: float
    mean_spacing_z: float
    high_jacobian_mean_spacing_z: float
    low_jacobian_mean_spacing_z: float
    mean_abs_dz_dzeta: float
    high_jacobian_mean_abs_dz_dzeta: float
    low_jacobian_mean_abs_dz_dzeta: float
    mean_abs_second_z_variation: float
    high_jacobian_mean_abs_second_z_variation: float
    low_jacobian_mean_abs_second_z_variation: float


@dataclass(frozen=True)
class AxisMappingRow:
    """Per-axis first/second-derivative summary for the monitor mapping."""

    axis_label: str
    uniform_physical_spacing: float
    mean_spacing: float
    high_jacobian_mean_spacing: float
    low_jacobian_mean_spacing: float
    mean_abs_first_derivative: float
    high_jacobian_mean_abs_first_derivative: float
    low_jacobian_mean_abs_first_derivative: float
    mean_abs_second_derivative: float
    high_jacobian_mean_abs_second_derivative: float
    low_jacobian_mean_abs_second_derivative: float


@dataclass(frozen=True)
class MonitorStrengthProxySummary:
    """Very-light proxy summary relating monitor strength to second-order distortion."""

    mean_monitor_value: float
    high_jacobian_mean_monitor_value: float
    low_jacobian_mean_monitor_value: float
    monitor_vs_z2_distortion_correlation: float
    monitor_vs_r2_distortion_correlation: float


@dataclass(frozen=True)
class HartreeGeometryRepresentationAuditResult:
    """Geometry/measure representation audit for the monitor grid."""

    cell_volume_construction: CellVolumeConstructionSummary
    polynomial_exactness_rows: tuple[PolynomialExactnessRow, ...]
    second_order_region_rows: tuple[SecondOrderRegionRow, ...]
    error_region_summary: GeometryRepresentationErrorRegionSummary
    mapping_z_stretch_summary: MappingZStretchSummary
    axis_mapping_rows: tuple[AxisMappingRow, ...]
    monitor_strength_proxy_summary: MonitorStrengthProxySummary
    note: str


@dataclass(frozen=True)
class MappingStageAttributionRow:
    """One stage row in the monitor-mapping attribution ledger."""

    stage_name: str
    stage_metric_basis: str
    x2_related_metric: float
    y2_related_metric: float
    z2_related_metric: float
    r2_related_metric: float
    quadrupole_related_metric: float
    high_jacobian_distortion_metric: float
    worsened_vs_previous_stage: bool
    optional_hartree_observable_mha: float | None = None


@dataclass(frozen=True)
class H2HartreeMappingStageAttributionAuditResult:
    """Stage-by-stage attribution audit for monitor mapping error amplification."""

    stage_rows: tuple[MappingStageAttributionRow, ...]
    first_clearly_worse_stage: str
    diagnosis: str
    note: str


@dataclass(frozen=True)
class MappingSolveStageAttributionRow:
    """Deeper mapping-solve stage row with derivative-based geometry diagnostics."""

    stage_name: str
    stage_metric_basis: str
    x_related_first_metric: float
    y_related_first_metric: float
    z_related_first_metric: float
    x_related_second_metric: float
    y_related_second_metric: float
    z_related_second_metric: float
    z_related_displacement_metric: float
    high_jacobian_distortion_metric: float
    low_jacobian_distortion_metric: float
    z_dominant_distortion_metric: float
    xy_comparable_distortion_metric: float
    worsened_vs_previous_stage: bool


@dataclass(frozen=True)
class H2HartreeMappingSolveStageAttributionAuditResult:
    """Deeper attribution audit focused on the mapping-solve chain itself."""

    stage_rows: tuple[MappingSolveStageAttributionRow, ...]
    first_clearly_worse_stage: str
    diagnosis: str
    note: str


@dataclass(frozen=True)
class ReferenceQuadratureIntegralRow:
    """Production nodal measure versus audit-only reference quadrature for one function."""

    function_label: str
    reference_value: float
    production_value: float
    production_bias: float
    reference_quadrature_value: float
    reference_quadrature_bias: float


@dataclass(frozen=True)
class H2HartreeReferenceQuadratureAuditResult:
    """Audit whether a higher-quality local reference quadrature improves mapped-grid exactness."""

    subcell_divisions: tuple[int, int, int]
    integral_rows: tuple[ReferenceQuadratureIntegralRow, ...]
    gaussian_production_total_charge: float
    gaussian_reference_quadrature_total_charge: float
    gaussian_production_quadrupole_norm: float
    gaussian_reference_quadrature_quadrupole_norm: float
    note: str


@dataclass(frozen=True)
class InsideCellRepresentationCellSummary:
    """One representative mapped cell compared under analytic vs trilinear Gaussian representation."""

    cell_label: str
    cell_index: tuple[int, int, int]
    region_label: str
    mean_jacobian: float
    profile_rms_error_x: float
    profile_rms_error_y: float
    profile_rms_error_z: float
    profile_max_abs_error_x: float
    profile_max_abs_error_y: float
    profile_max_abs_error_z: float
    local_x2_contribution_error: float
    local_y2_contribution_error: float
    local_z2_contribution_error: float
    local_r2_contribution_error: float
    local_quadrupole_component_error: float


@dataclass(frozen=True)
class H2HartreeInsideCellRepresentationAuditResult:
    """Audit whether fake quadrupole is already formed inside mapped-cell Gaussian representation."""

    cell_summaries: tuple[InsideCellRepresentationCellSummary, ...]
    diagnosis: str
    note: str


@dataclass(frozen=True)
class InsideCellReconstructionSummary:
    """One local field reconstruction compared against analytic Gaussian on a mapped cell."""

    reconstruction_label: str
    profile_rms_error_x: float
    profile_rms_error_y: float
    profile_rms_error_z: float
    profile_max_abs_error_x: float
    profile_max_abs_error_y: float
    profile_max_abs_error_z: float
    local_x2_contribution_error: float
    local_y2_contribution_error: float
    local_z2_contribution_error: float
    local_r2_contribution_error: float
    local_quadrupole_component_error: float


@dataclass(frozen=True)
class InsideCellReconstructionComparisonCellSummary:
    """One representative cell with multiple reconstruction-only comparisons."""

    cell_label: str
    cell_index: tuple[int, int, int]
    region_label: str
    mean_jacobian: float
    reconstruction_summaries: tuple[InsideCellReconstructionSummary, ...]


@dataclass(frozen=True)
class H2HartreeInsideCellReconstructionComparisonAuditResult:
    """Audit-only comparison of local Gaussian field reconstructions on mapped cells."""

    cell_summaries: tuple[InsideCellReconstructionComparisonCellSummary, ...]
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


def _trapezoidal_logical_weights(shape: tuple[int, int, int]) -> np.ndarray:
    wx = np.ones(shape[0], dtype=np.float64)
    wy = np.ones(shape[1], dtype=np.float64)
    wz = np.ones(shape[2], dtype=np.float64)
    wx[[0, -1]] = 0.5
    wy[[0, -1]] = 0.5
    wz[[0, -1]] = 0.5
    return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]


def _analytic_box_integrals(
    box_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    *,
    gaussian_alpha: float = _DEFAULT_GAUSSIAN_ALPHA,
) -> dict[str, float]:
    x_lower, x_upper = box_bounds[0]
    y_lower, y_upper = box_bounds[1]
    z_lower, z_upper = box_bounds[2]
    half_x = 0.5 * (x_upper - x_lower)
    half_y = 0.5 * (y_upper - y_lower)
    half_z = 0.5 * (z_upper - z_lower)
    volume = (x_upper - x_lower) * (y_upper - y_lower) * (z_upper - z_lower)
    gaussian_axis_prefactor = sqrt(np.pi / gaussian_alpha)
    gaussian_integral = (
        gaussian_axis_prefactor * erf(sqrt(gaussian_alpha) * half_x)
        * gaussian_axis_prefactor * erf(sqrt(gaussian_alpha) * half_y)
        * gaussian_axis_prefactor * erf(sqrt(gaussian_alpha) * half_z)
    )
    return {
        "1": float(volume),
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "r2": float(volume * (half_x * half_x + half_y * half_y + half_z * half_z) / 3.0),
        "centered_gaussian": float(gaussian_integral),
    }


def _measure_ledger_integrals(
    *,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
    weights: np.ndarray,
    box_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[MeasureLedgerIntegralSummary, ...]:
    fields = {
        "1": np.ones_like(x_points, dtype=np.float64),
        "x": np.asarray(x_points, dtype=np.float64),
        "y": np.asarray(y_points, dtype=np.float64),
        "z": np.asarray(z_points, dtype=np.float64),
        "r2": np.asarray(x_points * x_points + y_points * y_points + z_points * z_points, dtype=np.float64),
        "centered_gaussian": np.asarray(
            np.exp(-_DEFAULT_GAUSSIAN_ALPHA * (x_points * x_points + y_points * y_points + z_points * z_points)),
            dtype=np.float64,
        ),
    }
    references = _analytic_box_integrals(box_bounds)
    return tuple(
        MeasureLedgerIntegralSummary(
            function_label=label,
            value=float(np.sum(field * weights, dtype=np.float64)),
            reference_value=float(references[label]),
            bias=float(np.sum(field * weights, dtype=np.float64) - references[label]),
        )
        for label, field in fields.items()
    )


def _logical_cell_volume(
    monitor_geometry: MonitorGridGeometry,
) -> float:
    dx = float(np.diff(np.asarray(monitor_geometry.logical_x, dtype=np.float64))[0])
    dy = float(np.diff(np.asarray(monitor_geometry.logical_y, dtype=np.float64))[0])
    dz = float(np.diff(np.asarray(monitor_geometry.logical_z, dtype=np.float64))[0])
    return dx * dy * dz


def _polynomial_reference_values(
    box_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> dict[str, float]:
    x_lower, x_upper = box_bounds[0]
    y_lower, y_upper = box_bounds[1]
    z_lower, z_upper = box_bounds[2]
    half_x = 0.5 * (x_upper - x_lower)
    half_y = 0.5 * (y_upper - y_lower)
    half_z = 0.5 * (z_upper - z_lower)
    volume = (x_upper - x_lower) * (y_upper - y_lower) * (z_upper - z_lower)
    return {
        "1": float(volume),
        "x2": float(volume * half_x * half_x / 3.0),
        "y2": float(volume * half_y * half_y / 3.0),
        "z2": float(volume * half_z * half_z / 3.0),
        "r2": float(volume * (half_x * half_x + half_y * half_y + half_z * half_z) / 3.0),
        "xy": 0.0,
        "xz": 0.0,
        "yz": 0.0,
    }


def _polynomial_fields(
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "1": np.ones_like(x_points, dtype=np.float64),
        "x2": np.asarray(x_points * x_points, dtype=np.float64),
        "y2": np.asarray(y_points * y_points, dtype=np.float64),
        "z2": np.asarray(z_points * z_points, dtype=np.float64),
        "r2": np.asarray(x_points * x_points + y_points * y_points + z_points * z_points, dtype=np.float64),
        "xy": np.asarray(x_points * y_points, dtype=np.float64),
        "xz": np.asarray(x_points * z_points, dtype=np.float64),
        "yz": np.asarray(y_points * z_points, dtype=np.float64),
    }


def _polynomial_exactness_rows(
    monitor_geometry: MonitorGridGeometry,
) -> tuple[PolynomialExactnessRow, ...]:
    box_bounds = monitor_geometry.spec.box_bounds
    references = _polynomial_reference_values(box_bounds)
    mapped_fields = _polynomial_fields(
        monitor_geometry.x_points,
        monitor_geometry.y_points,
        monitor_geometry.z_points,
    )
    uniform_box_geometry = _build_uniform_box_measure_geometry(
        monitor_geometry,
        use_monitor_cell_volumes=False,
    )
    uniform_fields = _polynomial_fields(
        uniform_box_geometry.x_points,
        uniform_box_geometry.y_points,
        uniform_box_geometry.z_points,
    )
    physical_box_volume = references["1"]
    uniform_weight = np.full(
        monitor_geometry.spec.shape,
        physical_box_volume / np.prod(monitor_geometry.spec.shape),
        dtype=np.float64,
    )
    current_weight = np.asarray(monitor_geometry.cell_volumes, dtype=np.float64)
    trapezoidal_weight = current_weight * _trapezoidal_logical_weights(monitor_geometry.spec.shape)
    rows: list[PolynomialExactnessRow] = []
    for label, mapped_field in mapped_fields.items():
        uniform_field = uniform_fields[label]
        uniform_value = float(np.sum(mapped_field * uniform_weight, dtype=np.float64))
        current_value = float(np.sum(mapped_field * current_weight, dtype=np.float64))
        trapezoidal_value = float(np.sum(mapped_field * trapezoidal_weight, dtype=np.float64))
        uniform_coordinates_monitor_weights_value = float(
            np.sum(uniform_field * current_weight, dtype=np.float64)
        )
        mapped_coordinates_uniform_weights_value = float(
            np.sum(mapped_field * uniform_weight, dtype=np.float64)
        )
        reference_value = float(references[label])
        rows.append(
            PolynomialExactnessRow(
                function_label=label,
                reference_value=reference_value,
                uniform_weight_value=uniform_value,
                uniform_weight_bias=float(uniform_value - reference_value),
                current_cell_volume_value=current_value,
                current_cell_volume_bias=float(current_value - reference_value),
                trapezoidal_adjusted_value=trapezoidal_value,
                trapezoidal_adjusted_bias=float(trapezoidal_value - reference_value),
                uniform_coordinates_monitor_weights_value=uniform_coordinates_monitor_weights_value,
                uniform_coordinates_monitor_weights_bias=float(
                    uniform_coordinates_monitor_weights_value - reference_value
                ),
                mapped_coordinates_uniform_weights_value=mapped_coordinates_uniform_weights_value,
                mapped_coordinates_uniform_weights_bias=float(
                    mapped_coordinates_uniform_weights_value - reference_value
                ),
            )
        )
    return tuple(rows)


def _second_order_region_rows(
    monitor_geometry: MonitorGridGeometry,
) -> tuple[SecondOrderRegionRow, ...]:
    uniform_box_geometry = _build_uniform_box_measure_geometry(
        monitor_geometry,
        use_monitor_cell_volumes=False,
    )
    mapped_fields = _polynomial_fields(
        monitor_geometry.x_points,
        monitor_geometry.y_points,
        monitor_geometry.z_points,
    )
    uniform_fields = _polynomial_fields(
        uniform_box_geometry.x_points,
        uniform_box_geometry.y_points,
        uniform_box_geometry.z_points,
    )
    boundary_mask = np.zeros(monitor_geometry.spec.shape, dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    interior_mask = ~boundary_mask
    jacobian = np.asarray(monitor_geometry.jacobian, dtype=np.float64)
    high_jacobian_threshold = float(np.quantile(jacobian, 0.9))
    high_jacobian_mask = jacobian >= high_jacobian_threshold
    low_jacobian_mask = jacobian < high_jacobian_threshold
    x2_distortion = np.abs(mapped_fields["x2"] - uniform_fields["x2"])
    y2_distortion = np.abs(mapped_fields["y2"] - uniform_fields["y2"])
    z2_distortion = np.abs(mapped_fields["z2"] - uniform_fields["z2"])
    xy_scale = 0.5 * (x2_distortion + y2_distortion)
    z_dominant_mask = z2_distortion > 1.25 * xy_scale
    xy_comparable_mask = ~z_dominant_mask
    rows: list[SecondOrderRegionRow] = []
    for label in ("x2", "y2", "z2", "r2"):
        distortion = np.abs(mapped_fields[label] - uniform_fields[label])
        rows.append(
            SecondOrderRegionRow(
                function_label=label,
                boundary_mean_abs_mapping_distortion=float(np.mean(distortion[boundary_mask])),
                interior_mean_abs_mapping_distortion=float(np.mean(distortion[interior_mask])),
                high_jacobian_mean_abs_mapping_distortion=float(
                    np.mean(distortion[high_jacobian_mask])
                ),
                low_jacobian_mean_abs_mapping_distortion=float(
                    np.mean(distortion[low_jacobian_mask])
                ),
                z_dominant_mean_abs_mapping_distortion=float(
                    np.mean(distortion[z_dominant_mask])
                ),
                xy_comparable_mean_abs_mapping_distortion=float(
                    np.mean(distortion[xy_comparable_mask])
                ),
            )
        )
    return tuple(rows)


def _mapping_z_stretch_summary(
    monitor_geometry: MonitorGridGeometry,
) -> MappingZStretchSummary:
    jacobian = np.asarray(monitor_geometry.jacobian, dtype=np.float64)
    high_jacobian_threshold = float(np.quantile(jacobian, 0.9))
    high_jacobian_mask = jacobian >= high_jacobian_threshold
    low_jacobian_mask = jacobian < high_jacobian_threshold
    box_bounds = monitor_geometry.spec.box_bounds
    uniform_physical_dz = float(
        (box_bounds[2][1] - box_bounds[2][0]) / (monitor_geometry.spec.shape[2] - 1)
    )
    spacing_z = np.asarray(monitor_geometry.spacing_z, dtype=np.float64)
    dz_dzeta = np.asarray(monitor_geometry.covariant_basis[..., 2, 2], dtype=np.float64)
    second_z_variation = np.gradient(
        dz_dzeta,
        np.asarray(monitor_geometry.logical_z, dtype=np.float64),
        axis=2,
        edge_order=2,
    )
    return MappingZStretchSummary(
        uniform_physical_dz=uniform_physical_dz,
        mean_spacing_z=float(np.mean(spacing_z)),
        high_jacobian_mean_spacing_z=float(np.mean(spacing_z[high_jacobian_mask])),
        low_jacobian_mean_spacing_z=float(np.mean(spacing_z[low_jacobian_mask])),
        mean_abs_dz_dzeta=float(np.mean(np.abs(dz_dzeta))),
        high_jacobian_mean_abs_dz_dzeta=float(np.mean(np.abs(dz_dzeta[high_jacobian_mask]))),
        low_jacobian_mean_abs_dz_dzeta=float(np.mean(np.abs(dz_dzeta[low_jacobian_mask]))),
        mean_abs_second_z_variation=float(np.mean(np.abs(second_z_variation))),
        high_jacobian_mean_abs_second_z_variation=float(
            np.mean(np.abs(second_z_variation[high_jacobian_mask]))
        ),
        low_jacobian_mean_abs_second_z_variation=float(
            np.mean(np.abs(second_z_variation[low_jacobian_mask]))
        ),
    )


def _axis_mapping_rows(
    monitor_geometry: MonitorGridGeometry,
) -> tuple[AxisMappingRow, ...]:
    jacobian = np.asarray(monitor_geometry.jacobian, dtype=np.float64)
    high_jacobian_threshold = float(np.quantile(jacobian, 0.9))
    high_jacobian_mask = jacobian >= high_jacobian_threshold
    low_jacobian_mask = jacobian < high_jacobian_threshold
    axis_specs = (
        ("x", monitor_geometry.spacing_x, monitor_geometry.covariant_basis[..., 0, 0], monitor_geometry.logical_x, 0),
        ("y", monitor_geometry.spacing_y, monitor_geometry.covariant_basis[..., 1, 1], monitor_geometry.logical_y, 1),
        ("z", monitor_geometry.spacing_z, monitor_geometry.covariant_basis[..., 2, 2], monitor_geometry.logical_z, 2),
    )
    box_bounds = monitor_geometry.spec.box_bounds
    box_widths = (
        box_bounds[0][1] - box_bounds[0][0],
        box_bounds[1][1] - box_bounds[1][0],
        box_bounds[2][1] - box_bounds[2][0],
    )
    rows: list[AxisMappingRow] = []
    for axis_label, spacing, first_derivative, logical_axis, axis_index in axis_specs:
        uniform_spacing = float(box_widths[axis_index] / (monitor_geometry.spec.shape[axis_index] - 1))
        second_derivative = np.gradient(
            np.asarray(first_derivative, dtype=np.float64),
            np.asarray(logical_axis, dtype=np.float64),
            axis=axis_index,
            edge_order=2,
        )
        spacing = np.asarray(spacing, dtype=np.float64)
        first_derivative = np.asarray(first_derivative, dtype=np.float64)
        rows.append(
            AxisMappingRow(
                axis_label=axis_label,
                uniform_physical_spacing=uniform_spacing,
                mean_spacing=float(np.mean(spacing)),
                high_jacobian_mean_spacing=float(np.mean(spacing[high_jacobian_mask])),
                low_jacobian_mean_spacing=float(np.mean(spacing[low_jacobian_mask])),
                mean_abs_first_derivative=float(np.mean(np.abs(first_derivative))),
                high_jacobian_mean_abs_first_derivative=float(
                    np.mean(np.abs(first_derivative[high_jacobian_mask]))
                ),
                low_jacobian_mean_abs_first_derivative=float(
                    np.mean(np.abs(first_derivative[low_jacobian_mask]))
                ),
                mean_abs_second_derivative=float(np.mean(np.abs(second_derivative))),
                high_jacobian_mean_abs_second_derivative=float(
                    np.mean(np.abs(second_derivative[high_jacobian_mask]))
                ),
                low_jacobian_mean_abs_second_derivative=float(
                    np.mean(np.abs(second_derivative[low_jacobian_mask]))
                ),
            )
        )
    return tuple(rows)


def _monitor_strength_proxy_summary(
    monitor_geometry: MonitorGridGeometry,
) -> MonitorStrengthProxySummary:
    jacobian = np.asarray(monitor_geometry.jacobian, dtype=np.float64)
    high_jacobian_threshold = float(np.quantile(jacobian, 0.9))
    high_jacobian_mask = jacobian >= high_jacobian_threshold
    low_jacobian_mask = jacobian < high_jacobian_threshold
    uniform_box_geometry = _build_uniform_box_measure_geometry(
        monitor_geometry,
        use_monitor_cell_volumes=False,
    )
    mapped_fields = _polynomial_fields(
        monitor_geometry.x_points,
        monitor_geometry.y_points,
        monitor_geometry.z_points,
    )
    uniform_fields = _polynomial_fields(
        uniform_box_geometry.x_points,
        uniform_box_geometry.y_points,
        uniform_box_geometry.z_points,
    )
    z2_distortion = np.abs(mapped_fields["z2"] - uniform_fields["z2"]).reshape(-1)
    r2_distortion = np.abs(mapped_fields["r2"] - uniform_fields["r2"]).reshape(-1)
    monitor_values = np.asarray(monitor_geometry.monitor_field.values, dtype=np.float64).reshape(-1)
    z2_corr = float(np.corrcoef(monitor_values, z2_distortion)[0, 1])
    r2_corr = float(np.corrcoef(monitor_values, r2_distortion)[0, 1])
    return MonitorStrengthProxySummary(
        mean_monitor_value=float(np.mean(monitor_geometry.monitor_field.values)),
        high_jacobian_mean_monitor_value=float(
            np.mean(monitor_geometry.monitor_field.values[high_jacobian_mask])
        ),
        low_jacobian_mean_monitor_value=float(
            np.mean(monitor_geometry.monitor_field.values[low_jacobian_mask])
        ),
        monitor_vs_z2_distortion_correlation=z2_corr,
        monitor_vs_r2_distortion_correlation=r2_corr,
    )


def _geometry_representation_error_region_summary(
    monitor_geometry: MonitorGridGeometry,
) -> GeometryRepresentationErrorRegionSummary:
    uniform_box_geometry = _build_uniform_box_measure_geometry(
        monitor_geometry,
        use_monitor_cell_volumes=False,
    )
    mapped_r2 = (
        monitor_geometry.x_points * monitor_geometry.x_points
        + monitor_geometry.y_points * monitor_geometry.y_points
        + monitor_geometry.z_points * monitor_geometry.z_points
    )
    uniform_r2 = (
        uniform_box_geometry.x_points * uniform_box_geometry.x_points
        + uniform_box_geometry.y_points * uniform_box_geometry.y_points
        + uniform_box_geometry.z_points * uniform_box_geometry.z_points
    )
    r2_mapping_distortion = np.abs(mapped_r2 - uniform_r2)
    boundary_mask = np.zeros(monitor_geometry.spec.shape, dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    interior_mask = ~boundary_mask
    jacobian = np.asarray(monitor_geometry.jacobian, dtype=np.float64)
    high_jacobian_threshold = float(np.quantile(jacobian, 0.9))
    high_jacobian_mask = jacobian >= high_jacobian_threshold
    low_jacobian_mask = jacobian < high_jacobian_threshold
    current_weight = np.asarray(monitor_geometry.cell_volumes, dtype=np.float64)
    trapezoidal_weight = current_weight * _trapezoidal_logical_weights(monitor_geometry.spec.shape)
    weight_delta = np.abs(current_weight - trapezoidal_weight)
    return GeometryRepresentationErrorRegionSummary(
        boundary_mean_abs_r2_mapping_distortion=float(np.mean(r2_mapping_distortion[boundary_mask])),
        interior_mean_abs_r2_mapping_distortion=float(np.mean(r2_mapping_distortion[interior_mask])),
        high_jacobian_mean_abs_r2_mapping_distortion=float(
            np.mean(r2_mapping_distortion[high_jacobian_mask])
        ),
        low_jacobian_mean_abs_r2_mapping_distortion=float(
            np.mean(r2_mapping_distortion[low_jacobian_mask])
        ),
        boundary_mean_abs_weight_delta=float(np.mean(weight_delta[boundary_mask])),
        interior_mean_abs_weight_delta=float(np.mean(weight_delta[interior_mask])),
    )


def _quadrupole_tensor(
    density_field: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    dx = grid_geometry.x_points
    dy = grid_geometry.y_points
    dz = grid_geometry.z_points
    radius_squared = dx * dx + dy * dy + dz * dz
    return np.array(
        [
            [
                integrate_field(density_field * (3.0 * dx * dx - radius_squared), grid_geometry=grid_geometry),
                integrate_field(density_field * (3.0 * dx * dy), grid_geometry=grid_geometry),
                integrate_field(density_field * (3.0 * dx * dz), grid_geometry=grid_geometry),
            ],
            [
                integrate_field(density_field * (3.0 * dy * dx), grid_geometry=grid_geometry),
                integrate_field(density_field * (3.0 * dy * dy - radius_squared), grid_geometry=grid_geometry),
                integrate_field(density_field * (3.0 * dy * dz), grid_geometry=grid_geometry),
            ],
            [
                integrate_field(density_field * (3.0 * dz * dx), grid_geometry=grid_geometry),
                integrate_field(density_field * (3.0 * dz * dy), grid_geometry=grid_geometry),
                integrate_field(density_field * (3.0 * dz * dz - radius_squared), grid_geometry=grid_geometry),
            ],
        ],
        dtype=np.float64,
    )


def _dipole_vector(
    density_field: np.ndarray,
    grid_geometry: GridGeometryLike,
) -> np.ndarray:
    return np.array(
        [
            integrate_field(density_field * grid_geometry.x_points, grid_geometry=grid_geometry),
            integrate_field(density_field * grid_geometry.y_points, grid_geometry=grid_geometry),
            integrate_field(density_field * grid_geometry.z_points, grid_geometry=grid_geometry),
        ],
        dtype=np.float64,
    )


def _build_uniform_box_measure_geometry(
    monitor_geometry: MonitorGridGeometry,
    *,
    use_monitor_cell_volumes: bool,
) -> GridGeometryLike:
    bounds = monitor_geometry.spec.box_bounds
    x_axis = np.linspace(bounds[0][0], bounds[0][1], monitor_geometry.spec.shape[0], dtype=np.float64)
    y_axis = np.linspace(bounds[1][0], bounds[1][1], monitor_geometry.spec.shape[1], dtype=np.float64)
    z_axis = np.linspace(bounds[2][0], bounds[2][1], monitor_geometry.spec.shape[2], dtype=np.float64)
    x_points, y_points, z_points = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    if use_monitor_cell_volumes:
        cell_volumes = np.asarray(monitor_geometry.cell_volumes, dtype=np.float64)
    else:
        dx = float((bounds[0][1] - bounds[0][0]) / (monitor_geometry.spec.shape[0] - 1))
        dy = float((bounds[1][1] - bounds[1][0]) / (monitor_geometry.spec.shape[1] - 1))
        dz = float((bounds[2][1] - bounds[2][0]) / (monitor_geometry.spec.shape[2] - 1))
        cell_volumes = np.full(monitor_geometry.spec.shape, dx * dy * dz, dtype=np.float64)
    return SimpleNamespace(
        spec=SimpleNamespace(shape=monitor_geometry.spec.shape),
        x_points=x_points,
        y_points=y_points,
        z_points=z_points,
        cell_volumes=cell_volumes,
    )


def _build_mapped_coordinate_measure_geometry(
    monitor_geometry: MonitorGridGeometry,
    *,
    use_monitor_cell_volumes: bool,
) -> GridGeometryLike:
    if use_monitor_cell_volumes:
        cell_volumes = np.asarray(monitor_geometry.cell_volumes, dtype=np.float64)
    else:
        bounds = monitor_geometry.spec.box_bounds
        dx = float((bounds[0][1] - bounds[0][0]) / (monitor_geometry.spec.shape[0] - 1))
        dy = float((bounds[1][1] - bounds[1][0]) / (monitor_geometry.spec.shape[1] - 1))
        dz = float((bounds[2][1] - bounds[2][0]) / (monitor_geometry.spec.shape[2] - 1))
        cell_volumes = np.full(monitor_geometry.spec.shape, dx * dy * dz, dtype=np.float64)
    return SimpleNamespace(
        spec=SimpleNamespace(shape=monitor_geometry.spec.shape),
        x_points=np.asarray(monitor_geometry.x_points, dtype=np.float64),
        y_points=np.asarray(monitor_geometry.y_points, dtype=np.float64),
        z_points=np.asarray(monitor_geometry.z_points, dtype=np.float64),
        cell_volumes=cell_volumes,
    )


def _monitor_volume_consistency_summary(
    monitor_geometry: MonitorGridGeometry,
) -> MonitorVolumeConsistencySummary:
    box_bounds = monitor_geometry.spec.box_bounds
    physical_box_volume = float(
        (box_bounds[0][1] - box_bounds[0][0])
        * (box_bounds[1][1] - box_bounds[1][0])
        * (box_bounds[2][1] - box_bounds[2][0])
    )
    cell_volume_sum = float(np.sum(monitor_geometry.cell_volumes, dtype=np.float64))
    trapezoidal_cell_volume_sum = float(
        np.sum(
            np.asarray(monitor_geometry.cell_volumes, dtype=np.float64)
            * _trapezoidal_logical_weights(monitor_geometry.spec.shape),
            dtype=np.float64,
        )
    )
    return MonitorVolumeConsistencySummary(
        physical_box_volume=physical_box_volume,
        cell_volume_sum=cell_volume_sum,
        trapezoidal_cell_volume_sum=trapezoidal_cell_volume_sum,
        point_volume_relative_error=float(cell_volume_sum / physical_box_volume - 1.0),
        trapezoidal_relative_error=float(trapezoidal_cell_volume_sum / physical_box_volume - 1.0),
    )


def _gaussian_representation_consistency_summary(
    monitor_geometry: MonitorGridGeometry,
) -> GaussianRepresentationConsistencySummary:
    uniform_box_geometry = _build_uniform_box_measure_geometry(
        monitor_geometry,
        use_monitor_cell_volumes=False,
    )
    uniform_box_monitor_weight_geometry = _build_uniform_box_measure_geometry(
        monitor_geometry,
        use_monitor_cell_volumes=True,
    )
    mapped_monitor_uniform_weight_geometry = _build_mapped_coordinate_measure_geometry(
        monitor_geometry,
        use_monitor_cell_volumes=False,
    )
    uniform_box_density = _build_gaussian_density(uniform_box_geometry)
    uniform_box_monitor_weight_density = _build_gaussian_density(uniform_box_monitor_weight_geometry)
    mapped_monitor_uniform_weight_density = _build_gaussian_density(
        mapped_monitor_uniform_weight_geometry
    )
    mapped_monitor_density = _build_gaussian_density(monitor_geometry)
    uniform_box_dipole = _dipole_vector(uniform_box_density, uniform_box_geometry)
    uniform_box_monitor_weight_dipole = _dipole_vector(
        uniform_box_monitor_weight_density,
        uniform_box_monitor_weight_geometry,
    )
    mapped_monitor_uniform_weight_dipole = _dipole_vector(
        mapped_monitor_uniform_weight_density,
        mapped_monitor_uniform_weight_geometry,
    )
    mapped_monitor_dipole = _dipole_vector(mapped_monitor_density, monitor_geometry)
    uniform_box_quadrupole = _quadrupole_tensor(uniform_box_density, uniform_box_geometry)
    uniform_box_monitor_weight_quadrupole = _quadrupole_tensor(
        uniform_box_monitor_weight_density,
        uniform_box_monitor_weight_geometry,
    )
    mapped_monitor_uniform_weight_quadrupole = _quadrupole_tensor(
        mapped_monitor_uniform_weight_density,
        mapped_monitor_uniform_weight_geometry,
    )
    mapped_monitor_quadrupole = _quadrupole_tensor(mapped_monitor_density, monitor_geometry)
    return GaussianRepresentationConsistencySummary(
        uniform_box_total_charge=float(integrate_field(uniform_box_density, grid_geometry=uniform_box_geometry)),
        uniform_box_dipole_norm=float(np.linalg.norm(uniform_box_dipole)),
        uniform_box_quadrupole_norm=float(np.linalg.norm(uniform_box_quadrupole)),
        uniform_box_with_monitor_weights_total_charge=float(
            integrate_field(
                uniform_box_monitor_weight_density,
                grid_geometry=uniform_box_monitor_weight_geometry,
            )
        ),
        uniform_box_with_monitor_weights_dipole_norm=float(
            np.linalg.norm(uniform_box_monitor_weight_dipole)
        ),
        uniform_box_with_monitor_weights_quadrupole_norm=float(
            np.linalg.norm(uniform_box_monitor_weight_quadrupole)
        ),
        mapped_monitor_with_uniform_weights_total_charge=float(
            integrate_field(
                mapped_monitor_uniform_weight_density,
                grid_geometry=mapped_monitor_uniform_weight_geometry,
            )
        ),
        mapped_monitor_with_uniform_weights_dipole_norm=float(
            np.linalg.norm(mapped_monitor_uniform_weight_dipole)
        ),
        mapped_monitor_with_uniform_weights_quadrupole_norm=float(
            np.linalg.norm(mapped_monitor_uniform_weight_quadrupole)
        ),
        mapped_monitor_total_charge=float(integrate_field(mapped_monitor_density, grid_geometry=monitor_geometry)),
        mapped_monitor_dipole_norm=float(np.linalg.norm(mapped_monitor_dipole)),
        mapped_monitor_quadrupole_norm=float(np.linalg.norm(mapped_monitor_quadrupole)),
    )


def _monitor_inversion_symmetry_summary(
    monitor_geometry: MonitorGridGeometry,
) -> MonitorInversionSymmetrySummary:
    mirrored_x = np.flip(monitor_geometry.x_points)
    mirrored_y = np.flip(monitor_geometry.y_points)
    mirrored_z = np.flip(monitor_geometry.z_points)
    mirrored_cell_volumes = np.flip(monitor_geometry.cell_volumes)
    gaussian_density = _build_gaussian_density(monitor_geometry)
    dipole_integrand = gaussian_density * monitor_geometry.z_points
    mirrored_dipole_integrand = np.flip(dipole_integrand)
    return MonitorInversionSymmetrySummary(
        coordinate_pairing_max_abs=float(
            max(
                np.max(np.abs(monitor_geometry.x_points + mirrored_x)),
                np.max(np.abs(monitor_geometry.y_points + mirrored_y)),
                np.max(np.abs(monitor_geometry.z_points + mirrored_z)),
            )
        ),
        cell_volume_pairing_max_abs=float(
            np.max(np.abs(monitor_geometry.cell_volumes - mirrored_cell_volumes))
        ),
        gaussian_density_pairing_rms=float(
            np.sqrt(np.mean((gaussian_density - np.flip(gaussian_density)) ** 2))
        ),
        gaussian_dipole_integrand_pairing_rms=float(
            np.sqrt(np.mean((dipole_integrand + mirrored_dipole_integrand) ** 2))
        ),
    )


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
        total_charge=float(route_result.boundary_summary.total_charge),
        dipole_norm=float(route_result.boundary_summary.dipole_norm),
        quadrupole_norm=float(route_result.boundary_summary.quadrupole_norm),
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


def _is_nearly_nonincreasing(values: tuple[float, ...]) -> bool:
    if len(values) < 2:
        return True
    total_drop = max(0.0, values[0] - values[-1])
    local_scale = max(abs(value) for value in values)
    tolerance = max(1.0e-12, 0.05 * local_scale, 0.20 * total_drop)
    return all(next_value <= current_value + tolerance for current_value, next_value in zip(values, values[1:]))


def _shape_sweep_diagnosis(
    points: tuple[HartreeBoundaryShapeSweepPoint, ...],
) -> tuple[str, str]:
    gaussian_quadrupoles = tuple(point.gaussian_centered_monitor_quadrupole_norm for point in points)
    gaussian_gaps = tuple(point.gaussian_monitor_minus_legacy_hartree_energy_mha for point in points)
    h2_gaps = tuple(point.h2_frozen_monitor_minus_legacy_hartree_energy_mha for point in points)
    gaussian_box = tuple(point.gaussian_box_expand_sensitivity_mha for point in points)
    h2_box = tuple(point.h2_frozen_box_expand_sensitivity_mha for point in points)
    sequences = (
        gaussian_quadrupoles,
        gaussian_gaps,
        h2_gaps,
        gaussian_box,
        h2_box,
    )
    near_nonincreasing = all(_is_nearly_nonincreasing(sequence) for sequence in sequences)
    strong_improvements = sum(
        sequence[-1] <= 0.85 * sequence[0]
        for sequence in sequences
        if abs(sequence[0]) > 1.0e-12
    )
    if near_nonincreasing and strong_improvements >= 4:
        return (
            "resolution_improving",
            "The fake Gaussian quadrupole, Hartree gaps, and box sensitivities all decrease nearly "
            "monotonically with shape. The dominant issue still looks like monitor representation "
            "resolution / box-domain coupling rather than a separate SCF defect.",
        )
    if near_nonincreasing:
        return (
            "resolution_plateau",
            "The monitored quantities do not systematically worsen with shape, but the improvement is "
            "already flattening. Representation resolution helps, yet the remaining error may not be "
            "removed quickly by further baseline enlargement alone.",
        )
    return (
        "resolution_mixed",
        "The shape sweep is not cleanly monotone. Representation resolution still matters, but the "
        "remaining behavior likely includes a more systematic mapping / metric / weighting bias.",
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
    monitor_volume_consistency = _monitor_volume_consistency_summary(monitor_geometry)
    gaussian_representation_consistency = _gaussian_representation_consistency_summary(
        monitor_geometry
    )
    monitor_inversion_symmetry = _monitor_inversion_symmetry_summary(monitor_geometry)
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
        monitor_volume_consistency=monitor_volume_consistency,
        gaussian_representation_consistency=gaussian_representation_consistency,
        monitor_inversion_symmetry=monitor_inversion_symmetry,
        primary_verdict=primary_verdict,
        diagnosis=diagnosis,
        note=(
            "This audit fixes the input density and diagnoses Hartree/open-boundary behavior without "
            "running SCF. It is intended to separate electrostatics/boundary sensitivity from outer "
            "fixed-point/preconditioning difficulty."
        ),
    )


def run_h2_hartree_boundary_shape_sweep_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    shapes: tuple[tuple[int, int, int], ...] = (
        (67, 67, 81),
        (75, 75, 91),
        (83, 83, 101),
        (91, 91, 111),
    ),
    baseline_monitor_box_half_extents: tuple[float, float, float] = (
        H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
    ),
    expanded_monitor_box_half_extents: tuple[float, float, float] = (
        _DEFAULT_EXPANDED_MONITOR_BOX_HALF_EXTENTS_BOHR
    ),
    gaussian_shift_bohr: float = _DEFAULT_GAUSSIAN_SHIFT_BOHR,
    tolerance: float = _DEFAULT_TOLERANCE,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> HartreeBoundaryShapeSweepAuditResult:
    """Run a small monitor-shape sweep for fixed-density Hartree/open-boundary diagnosis."""

    points = []
    for monitor_shape in shapes:
        diagnosis_result = run_h2_hartree_boundary_diagnosis_audit(
            case=case,
            monitor_shape=monitor_shape,
            baseline_monitor_box_half_extents=baseline_monitor_box_half_extents,
            expanded_monitor_box_half_extents=expanded_monitor_box_half_extents,
            gaussian_shift_bohr=gaussian_shift_bohr,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        points.append(
            HartreeBoundaryShapeSweepPoint(
                monitor_shape=monitor_shape,
                gaussian_centered_monitor_quadrupole_norm=float(
                    diagnosis_result.gaussian_centered_monitor.quadrupole_norm
                ),
                gaussian_monitor_minus_legacy_hartree_energy_mha=float(
                    abs(
                        diagnosis_result.gaussian_centered_difference.monitor_minus_legacy_hartree_energy_mha
                    )
                ),
                h2_frozen_monitor_minus_legacy_hartree_energy_mha=float(
                    abs(diagnosis_result.h2_frozen_difference.monitor_minus_legacy_hartree_energy_mha)
                ),
                gaussian_box_expand_sensitivity_mha=float(
                    abs(
                        diagnosis_result.gaussian_centered_box_sensitivity.expanded_minus_baseline_hartree_energy_mha
                    )
                ),
                h2_frozen_box_expand_sensitivity_mha=float(
                    abs(
                        diagnosis_result.h2_frozen_box_sensitivity.expanded_minus_baseline_hartree_energy_mha
                    )
                ),
            )
        )
    point_tuple = tuple(points)
    trend_verdict, diagnosis = _shape_sweep_diagnosis(point_tuple)
    return HartreeBoundaryShapeSweepAuditResult(
        points=point_tuple,
        trend_verdict=trend_verdict,
        diagnosis=diagnosis,
        note=(
            "This sweep keeps the density and monitor-box protocol fixed while varying only the A-grid "
            "shape, so it diagnoses whether fake moments and fixed-density Hartree gaps are actually "
            "converging with representation resolution."
        ),
    )


def run_h2_hartree_measure_ledger_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    monitor_shape: tuple[int, int, int] = H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE,
    baseline_monitor_box_half_extents: tuple[float, float, float] = (
        H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
    ),
) -> HartreeMeasureLedgerAuditResult:
    """Audit the discrete measures used by monitor-grid Hartree-related paths."""

    monitor_geometry = _build_monitor_geometry(
        case,
        shape=monitor_shape,
        box_half_extents=baseline_monitor_box_half_extents,
    )
    box_bounds = monitor_geometry.spec.box_bounds
    cell_volumes = np.asarray(monitor_geometry.cell_volumes, dtype=np.float64)
    trapezoidal_weights = cell_volumes * _trapezoidal_logical_weights(monitor_geometry.spec.shape)
    identity_weights = np.ones(monitor_geometry.spec.shape, dtype=np.float64)

    path_summaries = (
        MeasureLedgerPathSummary(
            path_name="moments_path",
            role="total charge / dipole / quadrupole via integrate_field",
            measure_name="cell_volumes",
            measure_description="Mapped coordinates with monitor point-centered cell volumes.",
            integrals=_measure_ledger_integrals(
                x_points=monitor_geometry.x_points,
                y_points=monitor_geometry.y_points,
                z_points=monitor_geometry.z_points,
                weights=cell_volumes,
                box_bounds=box_bounds,
            ),
        ),
        MeasureLedgerPathSummary(
            path_name="hartree_energy_path",
            role="0.5 * integrate_field(rho * v_H)",
            measure_name="cell_volumes",
            measure_description="Mapped coordinates with the same monitor point-centered cell volumes.",
            integrals=_measure_ledger_integrals(
                x_points=monitor_geometry.x_points,
                y_points=monitor_geometry.y_points,
                z_points=monitor_geometry.z_points,
                weights=cell_volumes,
                box_bounds=box_bounds,
            ),
        ),
        MeasureLedgerPathSummary(
            path_name="poisson_rhs_path",
            role="pointwise RHS collocation (-4 pi rho plus boundary split term)",
            measure_name="identity_collocation",
            measure_description="No quadrature weight; nodal field samples enter the discrete operator directly.",
            integrals=_measure_ledger_integrals(
                x_points=monitor_geometry.x_points,
                y_points=monitor_geometry.y_points,
                z_points=monitor_geometry.z_points,
                weights=identity_weights,
                box_bounds=box_bounds,
            ),
        ),
        MeasureLedgerPathSummary(
            path_name="trapezoidal_reference",
            role="audit-only logical half-weight reference on the mapped monitor box",
            measure_name="cell_volumes_times_logical_half_weight",
            measure_description="Mapped coordinates with monitor cell volumes times tensor-product logical trapezoidal factors.",
            integrals=_measure_ledger_integrals(
                x_points=monitor_geometry.x_points,
                y_points=monitor_geometry.y_points,
                z_points=monitor_geometry.z_points,
                weights=trapezoidal_weights,
                box_bounds=box_bounds,
            ),
        ),
    )
    return HartreeMeasureLedgerAuditResult(
        path_summaries=path_summaries,
        note=(
            "This ledger compares the actual discrete measures used by monitor-grid moments, Hartree "
            "energy evaluation, and Poisson RHS assembly against a common fixed-box reference. It is "
            "an audit only and does not modify the production Poisson/Hartree path."
        ),
    )


def run_h2_hartree_geometry_representation_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    monitor_shape: tuple[int, int, int] = H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE,
    baseline_monitor_box_half_extents: tuple[float, float, float] = (
        H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
    ),
) -> HartreeGeometryRepresentationAuditResult:
    """Audit the monitor-grid cell-volume construction and polynomial exactness."""

    monitor_geometry = _build_monitor_geometry(
        case,
        shape=monitor_shape,
        box_half_extents=baseline_monitor_box_half_extents,
    )
    logical_cell_volume = _logical_cell_volume(monitor_geometry)
    recomputed_cell_volumes = np.asarray(monitor_geometry.jacobian, dtype=np.float64) * logical_cell_volume
    box_bounds = monitor_geometry.spec.box_bounds
    physical_box_volume = float(
        (box_bounds[0][1] - box_bounds[0][0])
        * (box_bounds[1][1] - box_bounds[1][0])
        * (box_bounds[2][1] - box_bounds[2][0])
    )
    cell_volume_construction = CellVolumeConstructionSummary(
        logical_cell_volume=float(logical_cell_volume),
        physical_box_volume=physical_box_volume,
        stored_cell_volume_sum=float(np.sum(monitor_geometry.cell_volumes, dtype=np.float64)),
        recomputed_cell_volume_sum=float(np.sum(recomputed_cell_volumes, dtype=np.float64)),
        max_abs_cell_volume_difference=float(
            np.max(np.abs(np.asarray(monitor_geometry.cell_volumes, dtype=np.float64) - recomputed_cell_volumes))
        ),
        rms_cell_volume_difference=float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(monitor_geometry.cell_volumes, dtype=np.float64)
                        - recomputed_cell_volumes
                    )
                    ** 2
                )
            )
        ),
    )
    return HartreeGeometryRepresentationAuditResult(
        cell_volume_construction=cell_volume_construction,
        polynomial_exactness_rows=_polynomial_exactness_rows(monitor_geometry),
        second_order_region_rows=_second_order_region_rows(monitor_geometry),
        error_region_summary=_geometry_representation_error_region_summary(monitor_geometry),
        mapping_z_stretch_summary=_mapping_z_stretch_summary(monitor_geometry),
        axis_mapping_rows=_axis_mapping_rows(monitor_geometry),
        monitor_strength_proxy_summary=_monitor_strength_proxy_summary(monitor_geometry),
        note=(
            "This audit traces monitor cell_volumes back to J * dxi * deta * dzeta, then checks how "
            "well equal weights, current cell_volumes, and trapezoidal-adjusted weights recover low-order "
            "polynomials on the mapped box."
        ),
    )


def _geometry_namespace(
    *,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
    cell_volumes: np.ndarray,
    shape: tuple[int, int, int],
) -> GridGeometryLike:
    return SimpleNamespace(
        spec=SimpleNamespace(shape=shape),
        x_points=np.asarray(x_points, dtype=np.float64),
        y_points=np.asarray(y_points, dtype=np.float64),
        z_points=np.asarray(z_points, dtype=np.float64),
        cell_volumes=np.asarray(cell_volumes, dtype=np.float64),
    )


def _stage_polynomial_biases(
    *,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
    weights: np.ndarray,
    box_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> dict[str, float]:
    references = _polynomial_reference_values(box_bounds)
    fields = _polynomial_fields(x_points, y_points, z_points)
    return {
        label: float(np.sum(field * weights, dtype=np.float64) - references[label])
        for label, field in fields.items()
    }


def _stage_high_jacobian_contribution_distortion(
    *,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
    weights: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    reference_z: np.ndarray,
    reference_weights: np.ndarray,
    high_jacobian_interior_mask: np.ndarray,
) -> float:
    stage_r2_contribution = weights * (
        x_points * x_points + y_points * y_points + z_points * z_points
    )
    reference_r2_contribution = reference_weights * (
        reference_x * reference_x + reference_y * reference_y + reference_z * reference_z
    )
    distortion = np.abs(stage_r2_contribution - reference_r2_contribution)
    return float(np.mean(distortion[high_jacobian_interior_mask]))


def _stage_row_from_geometry(
    *,
    stage_name: str,
    stage_metric_basis: str,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
    weights: np.ndarray,
    shape: tuple[int, int, int],
    box_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    reference_z: np.ndarray,
    reference_weights: np.ndarray,
    high_jacobian_interior_mask: np.ndarray,
    optional_hartree_observable_mha: float | None = None,
) -> MappingStageAttributionRow:
    geometry = _geometry_namespace(
        x_points=x_points,
        y_points=y_points,
        z_points=z_points,
        cell_volumes=weights,
        shape=shape,
    )
    gaussian_density = _build_gaussian_density(geometry)
    quadrupole_norm = float(np.linalg.norm(_quadrupole_tensor(gaussian_density, geometry)))
    biases = _stage_polynomial_biases(
        x_points=x_points,
        y_points=y_points,
        z_points=z_points,
        weights=weights,
        box_bounds=box_bounds,
    )
    high_j_distortion = _stage_high_jacobian_contribution_distortion(
        x_points=x_points,
        y_points=y_points,
        z_points=z_points,
        weights=weights,
        reference_x=reference_x,
        reference_y=reference_y,
        reference_z=reference_z,
        reference_weights=reference_weights,
        high_jacobian_interior_mask=high_jacobian_interior_mask,
    )
    return MappingStageAttributionRow(
        stage_name=stage_name,
        stage_metric_basis=stage_metric_basis,
        x2_related_metric=float(abs(biases["x2"])),
        y2_related_metric=float(abs(biases["y2"])),
        z2_related_metric=float(abs(biases["z2"])),
        r2_related_metric=float(abs(biases["r2"])),
        quadrupole_related_metric=quadrupole_norm,
        high_jacobian_distortion_metric=high_j_distortion,
        worsened_vs_previous_stage=False,
        optional_hartree_observable_mha=optional_hartree_observable_mha,
    )


def _mapping_stage_worsened(
    previous: MappingStageAttributionRow,
    current: MappingStageAttributionRow,
) -> bool:
    def _material_increase(new_value: float, old_value: float) -> bool:
        scale = max(1.0e-12, abs(old_value))
        return new_value > old_value + max(1.0e-12, 0.10 * scale)

    increases = (
        _material_increase(current.z2_related_metric, previous.z2_related_metric),
        _material_increase(current.quadrupole_related_metric, previous.quadrupole_related_metric),
        _material_increase(
            current.high_jacobian_distortion_metric,
            previous.high_jacobian_distortion_metric,
        ),
    )
    return sum(increases) >= 2


def _stage_rows_with_worsening_flags(
    rows: list[MappingStageAttributionRow],
) -> tuple[MappingStageAttributionRow, ...]:
    if not rows:
        return ()
    flagged_rows = [rows[0]]
    for row in rows[1:]:
        previous = flagged_rows[-1]
        flagged_rows.append(
            replace(
                row,
                worsened_vs_previous_stage=_mapping_stage_worsened(previous, row),
            )
        )
    return tuple(flagged_rows)


_MAPPING_SOLVE_JACOBIAN_FLOOR = 1.0e-12


def _solve_weighted_harmonic_coordinates_trace(
    *,
    coefficient: np.ndarray,
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    logical_z: np.ndarray,
    boundary_coordinates: np.ndarray,
    initial_coordinates: np.ndarray,
    inner_iterations: int,
    tolerance: float,
    relaxation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Audit-only trace of the production harmonic solve."""

    monitor = np.asarray(coefficient, dtype=np.float64)
    coordinates = np.asarray(initial_coordinates, dtype=np.float64).copy()
    boundary_mask = np.zeros(monitor.shape, dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    coordinates[boundary_mask] = boundary_coordinates[boundary_mask]

    dx = float(np.diff(np.asarray(logical_x, dtype=np.float64))[0])
    dy = float(np.diff(np.asarray(logical_y, dtype=np.float64))[0])
    dz = float(np.diff(np.asarray(logical_z, dtype=np.float64))[0])

    ax_minus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[:-2, 1:-1, 1:-1]) / (dx * dx)
    ax_plus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[2:, 1:-1, 1:-1]) / (dx * dx)
    ay_minus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[1:-1, :-2, 1:-1]) / (dy * dy)
    ay_plus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[1:-1, 2:, 1:-1]) / (dy * dy)
    az_minus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[1:-1, 1:-1, :-2]) / (dz * dz)
    az_plus = 0.5 * (monitor[1:-1, 1:-1, 1:-1] + monitor[1:-1, 1:-1, 2:]) / (dz * dz)
    diagonal = ax_minus + ax_plus + ay_minus + ay_plus + az_minus + az_plus

    first_updated_coordinates: np.ndarray | None = None
    for _ in range(inner_iterations):
        candidate = (
            ax_minus[..., None] * coordinates[:-2, 1:-1, 1:-1, :]
            + ax_plus[..., None] * coordinates[2:, 1:-1, 1:-1, :]
            + ay_minus[..., None] * coordinates[1:-1, :-2, 1:-1, :]
            + ay_plus[..., None] * coordinates[1:-1, 2:, 1:-1, :]
            + az_minus[..., None] * coordinates[1:-1, 1:-1, :-2, :]
            + az_plus[..., None] * coordinates[1:-1, 1:-1, 2:, :]
        ) / diagonal[..., None]

        updated = np.array(coordinates, copy=True)
        updated[1:-1, 1:-1, 1:-1, :] = (
            (1.0 - relaxation) * coordinates[1:-1, 1:-1, 1:-1, :]
            + relaxation * candidate
        )
        updated[boundary_mask] = boundary_coordinates[boundary_mask]
        if first_updated_coordinates is None:
            first_updated_coordinates = np.asarray(updated, dtype=np.float64)
        max_change = float(
            np.max(np.abs(updated[1:-1, 1:-1, 1:-1, :] - coordinates[1:-1, 1:-1, 1:-1, :]))
        )
        coordinates = updated
        if max_change < tolerance:
            break

    if first_updated_coordinates is None:
        first_updated_coordinates = np.asarray(coordinates, dtype=np.float64)
    return first_updated_coordinates, np.asarray(coordinates, dtype=np.float64)


def _backtracking_update_trace(
    *,
    current_coordinates: np.ndarray,
    solved_coordinates: np.ndarray,
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    logical_z: np.ndarray,
    relaxation: float,
) -> tuple[tuple[tuple[float, np.ndarray, float, bool], ...], np.ndarray]:
    """Audit-only trace of production backtracking candidates."""

    candidate_rows: list[tuple[float, np.ndarray, float, bool]] = []
    alpha = relaxation
    while alpha >= 1.0e-3:
        trial_coordinates = current_coordinates + alpha * (solved_coordinates - current_coordinates)
        _, jacobian, _, _, _, _ = _basic_geometry_from_coordinates(
            logical_x,
            logical_y,
            logical_z,
            trial_coordinates,
        )
        min_jacobian = float(np.min(jacobian))
        accepted = min_jacobian > _MAPPING_SOLVE_JACOBIAN_FLOOR
        candidate_rows.append((float(alpha), np.asarray(trial_coordinates, dtype=np.float64), min_jacobian, accepted))
        if accepted:
            return tuple(candidate_rows), np.asarray(trial_coordinates, dtype=np.float64)
        alpha *= 0.5
    return tuple(candidate_rows), np.asarray(current_coordinates, dtype=np.float64)


def _second_derivative_fields_from_coordinates(
    *,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    logical_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_second = np.gradient(
        np.gradient(np.asarray(x_points, dtype=np.float64), logical_x, axis=0, edge_order=2),
        logical_x,
        axis=0,
        edge_order=2,
    )
    y_second = np.gradient(
        np.gradient(np.asarray(y_points, dtype=np.float64), logical_y, axis=1, edge_order=2),
        logical_y,
        axis=1,
        edge_order=2,
    )
    z_second = np.gradient(
        np.gradient(np.asarray(z_points, dtype=np.float64), logical_z, axis=2, edge_order=2),
        logical_z,
        axis=2,
        edge_order=2,
    )
    return x_second, y_second, z_second


def _stage_coordinate_masks(
    *,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    logical_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    basis, jacobian, _, _, _, _ = _basic_geometry_from_coordinates(
        np.asarray(logical_x, dtype=np.float64),
        np.asarray(logical_y, dtype=np.float64),
        np.asarray(logical_z, dtype=np.float64),
        np.stack([x_points, y_points, z_points], axis=-1),
    )
    del basis
    shape = x_points.shape
    boundary_mask = np.zeros(shape, dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    interior_mask = ~boundary_mask
    high_threshold = float(np.quantile(jacobian[interior_mask], 0.9))
    low_threshold = float(np.quantile(jacobian[interior_mask], 0.1))
    high_j_mask = interior_mask & (jacobian >= high_threshold)
    low_j_mask = interior_mask & (jacobian <= low_threshold)
    return interior_mask, high_j_mask, low_j_mask, jacobian


def _mapping_solve_monitor_stage_row(
    *,
    stage_name: str,
    stage_metric_basis: str,
    values: np.ndarray,
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    logical_z: np.ndarray,
    downstream_high_j_mask: np.ndarray,
    downstream_low_j_mask: np.ndarray,
) -> MappingSolveStageAttributionRow:
    def _masked_mean_abs(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs(values)[mask]))

    x_first = np.gradient(np.asarray(values, dtype=np.float64), logical_x, axis=0, edge_order=2)
    y_first = np.gradient(np.asarray(values, dtype=np.float64), logical_y, axis=1, edge_order=2)
    z_first = np.gradient(np.asarray(values, dtype=np.float64), logical_z, axis=2, edge_order=2)
    x_second = np.gradient(x_first, logical_x, axis=0, edge_order=2)
    y_second = np.gradient(y_first, logical_y, axis=1, edge_order=2)
    z_second = np.gradient(z_first, logical_z, axis=2, edge_order=2)
    xy_scale = 0.5 * (np.abs(x_second) + np.abs(y_second))
    z_dominant_mask = np.abs(z_second) > 1.25 * xy_scale
    xy_comparable_mask = ~z_dominant_mask
    return MappingSolveStageAttributionRow(
        stage_name=stage_name,
        stage_metric_basis=stage_metric_basis,
        x_related_first_metric=float(np.mean(np.abs(x_first))),
        y_related_first_metric=float(np.mean(np.abs(y_first))),
        z_related_first_metric=float(np.mean(np.abs(z_first))),
        x_related_second_metric=float(np.mean(np.abs(x_second))),
        y_related_second_metric=float(np.mean(np.abs(y_second))),
        z_related_second_metric=float(np.mean(np.abs(z_second))),
        z_related_displacement_metric=0.0,
        high_jacobian_distortion_metric=_masked_mean_abs(z_second, downstream_high_j_mask),
        low_jacobian_distortion_metric=_masked_mean_abs(z_second, downstream_low_j_mask),
        z_dominant_distortion_metric=_masked_mean_abs(z_second, z_dominant_mask),
        xy_comparable_distortion_metric=_masked_mean_abs(z_second, xy_comparable_mask),
        worsened_vs_previous_stage=False,
    )


def _mapping_solve_coordinate_stage_row(
    *,
    stage_name: str,
    stage_metric_basis: str,
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_points: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    reference_z: np.ndarray,
    previous_z_points: np.ndarray,
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    logical_z: np.ndarray,
) -> MappingSolveStageAttributionRow:
    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(np.mean(values[mask]))

    x_first = np.gradient(np.asarray(x_points, dtype=np.float64), logical_x, axis=0, edge_order=2)
    y_first = np.gradient(np.asarray(y_points, dtype=np.float64), logical_y, axis=1, edge_order=2)
    z_first = np.gradient(np.asarray(z_points, dtype=np.float64), logical_z, axis=2, edge_order=2)
    x_second, y_second, z_second = _second_derivative_fields_from_coordinates(
        x_points=x_points,
        y_points=y_points,
        z_points=z_points,
        logical_x=logical_x,
        logical_y=logical_y,
        logical_z=logical_z,
    )
    interior_mask, high_j_mask, low_j_mask, _ = _stage_coordinate_masks(
        x_points=x_points,
        y_points=y_points,
        z_points=z_points,
        logical_x=logical_x,
        logical_y=logical_y,
        logical_z=logical_z,
    )
    del interior_mask
    x2_distortion = np.abs(np.asarray(x_points, dtype=np.float64) ** 2 - np.asarray(reference_x, dtype=np.float64) ** 2)
    y2_distortion = np.abs(np.asarray(y_points, dtype=np.float64) ** 2 - np.asarray(reference_y, dtype=np.float64) ** 2)
    z2_distortion = np.abs(np.asarray(z_points, dtype=np.float64) ** 2 - np.asarray(reference_z, dtype=np.float64) ** 2)
    xy_scale = 0.5 * (x2_distortion + y2_distortion)
    z_dominant_mask = z2_distortion > 1.25 * xy_scale
    xy_comparable_mask = ~z_dominant_mask
    return MappingSolveStageAttributionRow(
        stage_name=stage_name,
        stage_metric_basis=stage_metric_basis,
        x_related_first_metric=float(np.mean(np.abs(x_first))),
        y_related_first_metric=float(np.mean(np.abs(y_first))),
        z_related_first_metric=float(np.mean(np.abs(z_first))),
        x_related_second_metric=float(np.mean(np.abs(x_second))),
        y_related_second_metric=float(np.mean(np.abs(y_second))),
        z_related_second_metric=float(np.mean(np.abs(z_second))),
        z_related_displacement_metric=float(
            np.mean(np.abs(np.asarray(z_points, dtype=np.float64) - np.asarray(previous_z_points, dtype=np.float64)))
        ),
        high_jacobian_distortion_metric=_masked_mean(z2_distortion, high_j_mask),
        low_jacobian_distortion_metric=_masked_mean(z2_distortion, low_j_mask),
        z_dominant_distortion_metric=_masked_mean(z2_distortion, z_dominant_mask),
        xy_comparable_distortion_metric=_masked_mean(z2_distortion, xy_comparable_mask),
        worsened_vs_previous_stage=False,
    )


def _mapping_solve_stage_worsened(
    previous: MappingSolveStageAttributionRow,
    current: MappingSolveStageAttributionRow,
) -> bool:
    def _material_increase(new_value: float, old_value: float) -> bool:
        scale = max(1.0e-12, abs(old_value))
        return new_value > old_value + max(1.0e-12, 0.10 * scale)

    increases = (
        _material_increase(current.z_related_second_metric, previous.z_related_second_metric),
        _material_increase(current.high_jacobian_distortion_metric, previous.high_jacobian_distortion_metric),
        _material_increase(current.z_dominant_distortion_metric, previous.z_dominant_distortion_metric),
    )
    return sum(increases) >= 2


def _mapping_solve_rows_with_worsening_flags(
    rows: list[MappingSolveStageAttributionRow],
) -> tuple[MappingSolveStageAttributionRow, ...]:
    if not rows:
        return ()
    flagged_rows = [rows[0]]
    for row in rows[1:]:
        previous = flagged_rows[-1]
        flagged_rows.append(
            replace(
                row,
                worsened_vs_previous_stage=_mapping_solve_stage_worsened(previous, row),
            )
        )
    return tuple(flagged_rows)


def run_h2_hartree_mapping_solve_stage_attribution_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    monitor_shape: tuple[int, int, int] = H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE,
    baseline_monitor_box_half_extents: tuple[float, float, float] = (
        H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
    ),
    max_inner_iterations_override: int | None = None,
) -> H2HartreeMappingSolveStageAttributionAuditResult:
    """Deeper stage-by-stage attribution focused on the first mapping outer update."""

    spec = build_monitor_grid_spec_for_case(
        case,
        shape=monitor_shape,
        box_half_extents=baseline_monitor_box_half_extents,
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    if max_inner_iterations_override is not None:
        spec = replace(
            spec,
            harmonic_inner_iterations=min(
                spec.harmonic_inner_iterations,
                int(max_inner_iterations_override),
            ),
        )

    logical_x, logical_y, logical_z, x_ref, y_ref, z_ref = build_reference_box_coordinates(spec)
    boundary_coordinates = np.stack([x_ref, y_ref, z_ref], axis=-1)
    pre_update_coordinates = np.array(boundary_coordinates, copy=True)

    monitor_field = evaluate_global_monitor_field(
        case=case,
        spec=spec,
        x_points=pre_update_coordinates[..., 0],
        y_points=pre_update_coordinates[..., 1],
        z_points=pre_update_coordinates[..., 2],
    )
    smoothed_monitor = _smooth_monitor_field(
        monitor_field.values,
        smoothing=spec.monitor_smoothing,
    )
    first_inner_coordinates, solved_coordinates = _solve_weighted_harmonic_coordinates_trace(
        coefficient=smoothed_monitor,
        logical_x=logical_x,
        logical_y=logical_y,
        logical_z=logical_z,
        boundary_coordinates=boundary_coordinates,
        initial_coordinates=pre_update_coordinates,
        inner_iterations=spec.harmonic_inner_iterations,
        tolerance=spec.harmonic_tolerance,
        relaxation=spec.inner_relaxation,
    )
    backtracking_candidates, accepted_coordinates = _backtracking_update_trace(
        current_coordinates=pre_update_coordinates,
        solved_coordinates=solved_coordinates,
        logical_x=logical_x,
        logical_y=logical_y,
        logical_z=logical_z,
        relaxation=spec.harmonic_relaxation,
    )

    coordinates = np.array(pre_update_coordinates, copy=True)
    coordinates = accepted_coordinates
    for _ in range(1, spec.harmonic_outer_iterations):
        later_monitor_field = evaluate_global_monitor_field(
            case=case,
            spec=spec,
            x_points=coordinates[..., 0],
            y_points=coordinates[..., 1],
            z_points=coordinates[..., 2],
        )
        later_smoothed = _smooth_monitor_field(
            later_monitor_field.values,
            smoothing=spec.monitor_smoothing,
        )
        later_solved = _solve_weighted_harmonic_coordinates(
            coefficient=later_smoothed,
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
            boundary_coordinates=boundary_coordinates,
            initial_coordinates=coordinates,
            inner_iterations=spec.harmonic_inner_iterations,
            tolerance=spec.harmonic_tolerance,
            relaxation=spec.inner_relaxation,
        )
        later_updated = _backtracking_update(
            current_coordinates=coordinates,
            solved_coordinates=later_solved,
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
            relaxation=spec.harmonic_relaxation,
        )
        max_displacement = float(np.max(np.abs(later_updated - coordinates)))
        coordinates = later_updated
        if max_displacement < spec.harmonic_tolerance:
            break

    _, final_jacobian, _, _, final_cell_volumes, _ = _basic_geometry_from_coordinates(
        logical_x,
        logical_y,
        logical_z,
        coordinates,
    )
    boundary_mask = np.zeros(spec.shape, dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    interior_mask = ~boundary_mask
    final_high_threshold = float(np.quantile(final_jacobian[interior_mask], 0.9))
    final_low_threshold = float(np.quantile(final_jacobian[interior_mask], 0.1))
    final_high_j_mask = interior_mask & (final_jacobian >= final_high_threshold)
    final_low_j_mask = interior_mask & (final_jacobian <= final_low_threshold)

    stage_rows: list[MappingSolveStageAttributionRow] = [
        _mapping_solve_monitor_stage_row(
            stage_name="raw_monitor_field",
            stage_metric_basis=(
                "logical-grid monitor derivative proxy on the first outer iteration"
            ),
            values=np.asarray(monitor_field.values, dtype=np.float64),
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
            downstream_high_j_mask=final_high_j_mask,
            downstream_low_j_mask=final_low_j_mask,
        ),
        _mapping_solve_monitor_stage_row(
            stage_name="smoothed_monitor_field",
            stage_metric_basis=(
                "logical-grid smoothed-monitor derivative proxy on the first outer iteration"
            ),
            values=np.asarray(smoothed_monitor, dtype=np.float64),
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
            downstream_high_j_mask=final_high_j_mask,
            downstream_low_j_mask=final_low_j_mask,
        ),
        _mapping_solve_coordinate_stage_row(
            stage_name="pre_update_coordinate_state",
            stage_metric_basis="reference-box coordinates before the first harmonic inner solve",
            x_points=pre_update_coordinates[..., 0],
            y_points=pre_update_coordinates[..., 1],
            z_points=pre_update_coordinates[..., 2],
            reference_x=x_ref,
            reference_y=y_ref,
            reference_z=z_ref,
            previous_z_points=pre_update_coordinates[..., 2],
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
        ),
        _mapping_solve_coordinate_stage_row(
            stage_name="first_coordinate_update_output",
            stage_metric_basis="first inner harmonic-solve Jacobi update before convergence of the inner solve",
            x_points=first_inner_coordinates[..., 0],
            y_points=first_inner_coordinates[..., 1],
            z_points=first_inner_coordinates[..., 2],
            reference_x=x_ref,
            reference_y=y_ref,
            reference_z=z_ref,
            previous_z_points=pre_update_coordinates[..., 2],
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
        ),
        _mapping_solve_coordinate_stage_row(
            stage_name="solved_coordinate_output",
            stage_metric_basis="full inner harmonic solve output before outer backtracking acceptance",
            x_points=solved_coordinates[..., 0],
            y_points=solved_coordinates[..., 1],
            z_points=solved_coordinates[..., 2],
            reference_x=x_ref,
            reference_y=y_ref,
            reference_z=z_ref,
            previous_z_points=pre_update_coordinates[..., 2],
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
        ),
    ]
    for alpha, trial_coordinates, min_jacobian, accepted in backtracking_candidates:
        stage_rows.append(
            _mapping_solve_coordinate_stage_row(
                stage_name=f"backtracking_candidate_alpha_{alpha:.3f}",
                stage_metric_basis=(
                    f"outer backtracking trial with alpha={alpha:.3f}, min_jacobian={min_jacobian:.3e}, "
                    f"accepted={accepted}"
                ),
                x_points=trial_coordinates[..., 0],
                y_points=trial_coordinates[..., 1],
                z_points=trial_coordinates[..., 2],
                reference_x=x_ref,
                reference_y=y_ref,
                reference_z=z_ref,
                previous_z_points=pre_update_coordinates[..., 2],
                logical_x=logical_x,
                logical_y=logical_y,
                logical_z=logical_z,
            )
        )
    stage_rows.extend(
        [
            _mapping_solve_coordinate_stage_row(
                stage_name="accepted_post_backtracking_update",
                stage_metric_basis="accepted first outer update after backtracking",
                x_points=accepted_coordinates[..., 0],
                y_points=accepted_coordinates[..., 1],
                z_points=accepted_coordinates[..., 2],
                reference_x=x_ref,
                reference_y=y_ref,
                reference_z=z_ref,
                previous_z_points=pre_update_coordinates[..., 2],
                logical_x=logical_x,
                logical_y=logical_y,
                logical_z=logical_z,
            ),
            _mapping_solve_coordinate_stage_row(
                stage_name="final_mapped_coordinates",
                stage_metric_basis="final converged mapped coordinates after the outer mapping loop",
                x_points=coordinates[..., 0],
                y_points=coordinates[..., 1],
                z_points=coordinates[..., 2],
                reference_x=x_ref,
                reference_y=y_ref,
                reference_z=z_ref,
                previous_z_points=accepted_coordinates[..., 2],
                logical_x=logical_x,
                logical_y=logical_y,
                logical_z=logical_z,
            ),
            _mapping_solve_coordinate_stage_row(
                stage_name="jacobian_metric_derived_quantities",
                stage_metric_basis=(
                    "final mapped coordinates observed together with final Jacobian-derived measure"
                ),
                x_points=coordinates[..., 0],
                y_points=coordinates[..., 1],
                z_points=coordinates[..., 2],
                reference_x=x_ref,
                reference_y=y_ref,
                reference_z=z_ref,
                previous_z_points=accepted_coordinates[..., 2],
                logical_x=logical_x,
                logical_y=logical_y,
                logical_z=logical_z,
            ),
        ]
    )
    stage_row_tuple = _mapping_solve_rows_with_worsening_flags(stage_rows)
    first_clearly_worse_stage = next(
        (
            row.stage_name
            for row in stage_row_tuple
            if row.worsened_vs_previous_stage
        ),
        "none",
    )
    diagnosis = (
        "The first clearly worse stage is where at least two of three geometry metrics "
        "(z second derivative, high-jacobian z^2 distortion, and z-dominant distortion) "
        "materially increase relative to the immediately preceding stage."
    )
    return H2HartreeMappingSolveStageAttributionAuditResult(
        stage_rows=stage_row_tuple,
        first_clearly_worse_stage=first_clearly_worse_stage,
        diagnosis=diagnosis,
        note=(
            "This audit keeps the production mapping unchanged and only decomposes the first harmonic "
            "outer update into monitor, inner-solve, backtracking, and accepted-update stages."
        ),
    )


def run_h2_hartree_mapping_stage_attribution_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    monitor_shape: tuple[int, int, int] = H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE,
    baseline_monitor_box_half_extents: tuple[float, float, float] = (
        H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
    ),
    max_inner_iterations_override: int | None = None,
) -> H2HartreeMappingStageAttributionAuditResult:
    """Attribute where mapping-chain stages first amplify z-directed second-moment error."""

    spec = build_monitor_grid_spec_for_case(
        case,
        shape=monitor_shape,
        box_half_extents=baseline_monitor_box_half_extents,
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    if max_inner_iterations_override is not None:
        spec = replace(
            spec,
            harmonic_inner_iterations=min(
                spec.harmonic_inner_iterations,
                int(max_inner_iterations_override),
            ),
        )

    logical_x, logical_y, logical_z, x_ref, y_ref, z_ref = build_reference_box_coordinates(spec)
    boundary_coordinates = np.stack([x_ref, y_ref, z_ref], axis=-1)
    coordinates = np.array(boundary_coordinates, copy=True)

    first_raw_monitor_values: np.ndarray | None = None
    first_smoothed_monitor_values: np.ndarray | None = None
    first_updated_coordinates: np.ndarray | None = None

    for outer_iteration in range(spec.harmonic_outer_iterations):
        monitor_field = evaluate_global_monitor_field(
            case=case,
            spec=spec,
            x_points=coordinates[..., 0],
            y_points=coordinates[..., 1],
            z_points=coordinates[..., 2],
        )
        smoothed_monitor = _smooth_monitor_field(
            monitor_field.values,
            smoothing=spec.monitor_smoothing,
        )
        solved_coordinates = _solve_weighted_harmonic_coordinates(
            coefficient=smoothed_monitor,
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
            boundary_coordinates=boundary_coordinates,
            initial_coordinates=coordinates,
            inner_iterations=spec.harmonic_inner_iterations,
            tolerance=spec.harmonic_tolerance,
            relaxation=spec.inner_relaxation,
        )
        updated_coordinates = _backtracking_update(
            current_coordinates=coordinates,
            solved_coordinates=solved_coordinates,
            logical_x=logical_x,
            logical_y=logical_y,
            logical_z=logical_z,
            relaxation=spec.harmonic_relaxation,
        )
        if outer_iteration == 0:
            first_raw_monitor_values = np.asarray(monitor_field.values, dtype=np.float64)
            first_smoothed_monitor_values = np.asarray(smoothed_monitor, dtype=np.float64)
            first_updated_coordinates = np.asarray(updated_coordinates, dtype=np.float64)
        max_displacement = float(np.max(np.abs(updated_coordinates - coordinates)))
        coordinates = updated_coordinates
        if max_displacement < spec.harmonic_tolerance:
            break

    final_monitor_field = evaluate_global_monitor_field(
        case=case,
        spec=spec,
        x_points=coordinates[..., 0],
        y_points=coordinates[..., 1],
        z_points=coordinates[..., 2],
    )
    basis, jacobian, metric_tensor, inverse_metric_tensor, cell_volumes, point_spacings = (
        _basic_geometry_from_coordinates(
            logical_x,
            logical_y,
            logical_z,
            coordinates,
        )
    )
    spacing_measure = np.mean(point_spacings, axis=-1)
    monitor_geometry = MonitorGridGeometry(
        spec=spec,
        logical_x=logical_x,
        logical_y=logical_y,
        logical_z=logical_z,
        x_points=np.asarray(coordinates[..., 0], dtype=np.float64),
        y_points=np.asarray(coordinates[..., 1], dtype=np.float64),
        z_points=np.asarray(coordinates[..., 2], dtype=np.float64),
        covariant_basis=basis,
        jacobian=jacobian,
        metric_tensor=metric_tensor,
        inverse_metric_tensor=inverse_metric_tensor,
        cell_volumes=cell_volumes,
        spacing_x=point_spacings[..., 0],
        spacing_y=point_spacings[..., 1],
        spacing_z=point_spacings[..., 2],
        spacing_measure=spacing_measure,
        monitor_field=final_monitor_field,
        patch_interfaces=(),
        quality_report=SimpleNamespace(),
    )
    diagnosis_result = run_h2_hartree_boundary_diagnosis_audit(
        case=case,
        monitor_shape=monitor_shape,
        baseline_monitor_box_half_extents=baseline_monitor_box_half_extents,
        tolerance=1.0e-6,
        max_iterations=200,
    )

    if first_raw_monitor_values is None or first_smoothed_monitor_values is None or first_updated_coordinates is None:
        raise RuntimeError("Expected the monitor mapping loop to capture at least one outer iteration.")

    shape = spec.shape
    box_bounds = spec.box_bounds
    physical_box_volume = float(
        (box_bounds[0][1] - box_bounds[0][0])
        * (box_bounds[1][1] - box_bounds[1][0])
        * (box_bounds[2][1] - box_bounds[2][0])
    )
    uniform_point_weight = np.full(shape, physical_box_volume / np.prod(shape), dtype=np.float64)
    raw_monitor_weights = uniform_point_weight * (
        first_raw_monitor_values / float(np.mean(first_raw_monitor_values))
    )
    smoothed_monitor_weights = uniform_point_weight * (
        first_smoothed_monitor_values / float(np.mean(first_smoothed_monitor_values))
    )
    boundary_mask = np.zeros(shape, dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    interior_mask = ~boundary_mask
    high_j_threshold = float(np.quantile(jacobian, 0.9))
    high_jacobian_interior_mask = (jacobian >= high_j_threshold) & interior_mask

    stage_rows = [
        _stage_row_from_geometry(
            stage_name="raw_monitor_field",
            stage_metric_basis=(
                "reference-box centered Gaussian under raw-monitor-induced pseudo-measure"
            ),
            x_points=x_ref,
            y_points=y_ref,
            z_points=z_ref,
            weights=raw_monitor_weights,
            shape=shape,
            box_bounds=box_bounds,
            reference_x=x_ref,
            reference_y=y_ref,
            reference_z=z_ref,
            reference_weights=uniform_point_weight,
            high_jacobian_interior_mask=high_jacobian_interior_mask,
        ),
        _stage_row_from_geometry(
            stage_name="smoothed_monitor_field",
            stage_metric_basis=(
                "reference-box centered Gaussian under smoothed-monitor-induced pseudo-measure"
            ),
            x_points=x_ref,
            y_points=y_ref,
            z_points=z_ref,
            weights=smoothed_monitor_weights,
            shape=shape,
            box_bounds=box_bounds,
            reference_x=x_ref,
            reference_y=y_ref,
            reference_z=z_ref,
            reference_weights=uniform_point_weight,
            high_jacobian_interior_mask=high_jacobian_interior_mask,
        ),
        _stage_row_from_geometry(
            stage_name="first_coordinate_update_output",
            stage_metric_basis="first outer-iteration updated coordinates with uniform point measure",
            x_points=first_updated_coordinates[..., 0],
            y_points=first_updated_coordinates[..., 1],
            z_points=first_updated_coordinates[..., 2],
            weights=uniform_point_weight,
            shape=shape,
            box_bounds=box_bounds,
            reference_x=x_ref,
            reference_y=y_ref,
            reference_z=z_ref,
            reference_weights=uniform_point_weight,
            high_jacobian_interior_mask=high_jacobian_interior_mask,
        ),
        _stage_row_from_geometry(
            stage_name="final_mapped_coordinates",
            stage_metric_basis="final mapped coordinates with uniform point measure",
            x_points=monitor_geometry.x_points,
            y_points=monitor_geometry.y_points,
            z_points=monitor_geometry.z_points,
            weights=uniform_point_weight,
            shape=shape,
            box_bounds=box_bounds,
            reference_x=x_ref,
            reference_y=y_ref,
            reference_z=z_ref,
            reference_weights=uniform_point_weight,
            high_jacobian_interior_mask=high_jacobian_interior_mask,
        ),
        _stage_row_from_geometry(
            stage_name="jacobian_metric_derived_measure",
            stage_metric_basis="reference-box coordinates with final Jacobian-derived cell-volume weights",
            x_points=x_ref,
            y_points=y_ref,
            z_points=z_ref,
            weights=np.asarray(monitor_geometry.cell_volumes, dtype=np.float64),
            shape=shape,
            box_bounds=box_bounds,
            reference_x=x_ref,
            reference_y=y_ref,
            reference_z=z_ref,
            reference_weights=uniform_point_weight,
            high_jacobian_interior_mask=high_jacobian_interior_mask,
        ),
        _stage_row_from_geometry(
            stage_name="cell_volumes_measure",
            stage_metric_basis="final mapped coordinates with final cell-volume measure",
            x_points=monitor_geometry.x_points,
            y_points=monitor_geometry.y_points,
            z_points=monitor_geometry.z_points,
            weights=np.asarray(monitor_geometry.cell_volumes, dtype=np.float64),
            shape=shape,
            box_bounds=box_bounds,
            reference_x=x_ref,
            reference_y=y_ref,
            reference_z=z_ref,
            reference_weights=uniform_point_weight,
            high_jacobian_interior_mask=high_jacobian_interior_mask,
        ),
        _stage_row_from_geometry(
            stage_name="moments_hartree_observable",
            stage_metric_basis="final mapped measure plus Gaussian Hartree-gap observable",
            x_points=monitor_geometry.x_points,
            y_points=monitor_geometry.y_points,
            z_points=monitor_geometry.z_points,
            weights=np.asarray(monitor_geometry.cell_volumes, dtype=np.float64),
            shape=shape,
            box_bounds=box_bounds,
            reference_x=x_ref,
            reference_y=y_ref,
            reference_z=z_ref,
            reference_weights=uniform_point_weight,
            high_jacobian_interior_mask=high_jacobian_interior_mask,
            optional_hartree_observable_mha=float(
                abs(
                    diagnosis_result.gaussian_centered_difference.monitor_minus_legacy_hartree_energy_mha
                )
            ),
        ),
    ]
    stage_row_tuple = _stage_rows_with_worsening_flags(stage_rows)
    first_clearly_worse_stage = next(
        (
            row.stage_name
            for row in stage_row_tuple
            if row.worsened_vs_previous_stage
        ),
        "none",
    )
    diagnosis = (
        "The first clearly worse stage is where at least two of three metrics (z^2 bias, centered-"
        "Gaussian quadrupole norm, and high-jacobian interior contribution distortion) materially "
        "increase relative to the immediately preceding stage."
    )
    return H2HartreeMappingStageAttributionAuditResult(
        stage_rows=stage_row_tuple,
        first_clearly_worse_stage=first_clearly_worse_stage,
        diagnosis=diagnosis,
        note=(
            "This audit keeps the production mapping unchanged and only attributes where the centered-"
            "Gaussian z-directed second-moment error first gets amplified along the monitor-to-"
            "mapping chain."
        ),
    )


def _trilinear_sample(
    nodal_values: np.ndarray,
    *,
    u: float,
    v: float,
    w: float,
) -> np.ndarray:
    c000 = nodal_values[:-1, :-1, :-1]
    c100 = nodal_values[1:, :-1, :-1]
    c010 = nodal_values[:-1, 1:, :-1]
    c110 = nodal_values[1:, 1:, :-1]
    c001 = nodal_values[:-1, :-1, 1:]
    c101 = nodal_values[1:, :-1, 1:]
    c011 = nodal_values[:-1, 1:, 1:]
    c111 = nodal_values[1:, 1:, 1:]
    um = 1.0 - u
    vm = 1.0 - v
    wm = 1.0 - w
    return (
        um * vm * wm * c000
        + u * vm * wm * c100
        + um * v * wm * c010
        + u * v * wm * c110
        + um * vm * w * c001
        + u * vm * w * c101
        + um * v * w * c011
        + u * v * w * c111
    )


def _trilinear_cell_value(
    cell_corners: np.ndarray,
    *,
    u: float,
    v: float,
    w: float,
) -> float:
    um = 1.0 - u
    vm = 1.0 - v
    wm = 1.0 - w
    return float(
        um * vm * wm * cell_corners[0, 0, 0]
        + u * vm * wm * cell_corners[1, 0, 0]
        + um * v * wm * cell_corners[0, 1, 0]
        + u * v * wm * cell_corners[1, 1, 0]
        + um * vm * w * cell_corners[0, 0, 1]
        + u * vm * w * cell_corners[1, 0, 1]
        + um * v * w * cell_corners[0, 1, 1]
        + u * v * w * cell_corners[1, 1, 1]
    )


def _cell_average_from_nodal_field(field: np.ndarray) -> np.ndarray:
    return (
        field[:-1, :-1, :-1]
        + field[1:, :-1, :-1]
        + field[:-1, 1:, :-1]
        + field[1:, 1:, :-1]
        + field[:-1, :-1, 1:]
        + field[1:, :-1, 1:]
        + field[:-1, 1:, 1:]
        + field[1:, 1:, 1:]
    ) / 8.0


def _representative_cell_indices(
    monitor_geometry: MonitorGridGeometry,
) -> tuple[tuple[str, tuple[int, int, int], str], ...]:
    cell_jacobian = _cell_average_from_nodal_field(np.asarray(monitor_geometry.jacobian, dtype=np.float64))
    cell_shape = cell_jacobian.shape
    boundary_mask = np.zeros(cell_shape, dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    interior_mask = ~boundary_mask
    interior_indices = np.argwhere(interior_mask)
    interior_values = cell_jacobian[interior_mask]
    high_flat_index = int(np.argmax(interior_values))
    low_flat_index = int(np.argmin(interior_values))
    median_value = float(np.median(interior_values))
    median_flat_index = int(np.argmin(np.abs(interior_values - median_value)))
    return (
        ("high_jacobian_interior", tuple(int(v) for v in interior_indices[high_flat_index]), "high_jacobian_interior"),
        ("median_jacobian_interior", tuple(int(v) for v in interior_indices[median_flat_index]), "ordinary_interior"),
        ("low_jacobian_interior", tuple(int(v) for v in interior_indices[low_flat_index]), "low_jacobian_interior"),
    )


def _inside_cell_profile_errors(
    *,
    x_cell: np.ndarray,
    y_cell: np.ndarray,
    z_cell: np.ndarray,
    rho_cell: np.ndarray,
    normalization_constant: float,
    cell_subsamples: int,
) -> dict[str, tuple[float, float]]:
    if cell_subsamples < 2:
        raise ValueError("cell_subsamples must be at least 2.")
    profile_errors: dict[str, tuple[float, float]] = {}
    axis_map = {
        "x": lambda t: (t, 0.5, 0.5),
        "y": lambda t: (0.5, t, 0.5),
        "z": lambda t: (0.5, 0.5, t),
    }
    for axis_label, point_builder in axis_map.items():
        errors = []
        for t in np.linspace(0.0, 1.0, cell_subsamples, dtype=np.float64):
            u, v, w = point_builder(float(t))
            x_value = _trilinear_cell_value(x_cell, u=u, v=v, w=w)
            y_value = _trilinear_cell_value(y_cell, u=u, v=v, w=w)
            z_value = _trilinear_cell_value(z_cell, u=u, v=v, w=w)
            analytic_density = normalization_constant * np.exp(
                -_DEFAULT_GAUSSIAN_ALPHA * (x_value * x_value + y_value * y_value + z_value * z_value)
            )
            reconstructed_density = _trilinear_cell_value(rho_cell, u=u, v=v, w=w)
            errors.append(reconstructed_density - analytic_density)
        error_array = np.asarray(errors, dtype=np.float64)
        profile_errors[axis_label] = (
            float(np.sqrt(np.mean(error_array * error_array))),
            float(np.max(np.abs(error_array))),
        )
    return profile_errors


def _inside_cell_moment_errors(
    *,
    x_cell: np.ndarray,
    y_cell: np.ndarray,
    z_cell: np.ndarray,
    jacobian_cell: np.ndarray,
    rho_cell: np.ndarray,
    normalization_constant: float,
    logical_cell_volume: float,
    moment_subcell_divisions: tuple[int, int, int],
) -> tuple[float, float, float, float, float]:
    sx, sy, sz = moment_subcell_divisions
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError("moment_subcell_divisions must be positive in every direction.")
    subcell_volume = logical_cell_volume / float(sx * sy * sz)
    analytic_x2 = 0.0
    analytic_y2 = 0.0
    analytic_z2 = 0.0
    analytic_r2 = 0.0
    analytic_qzz = 0.0
    reconstructed_x2 = 0.0
    reconstructed_y2 = 0.0
    reconstructed_z2 = 0.0
    reconstructed_r2 = 0.0
    reconstructed_qzz = 0.0
    for ix in range(sx):
        u = (ix + 0.5) / sx
        for iy in range(sy):
            v = (iy + 0.5) / sy
            for iz in range(sz):
                w = (iz + 0.5) / sz
                x_value = _trilinear_cell_value(x_cell, u=u, v=v, w=w)
                y_value = _trilinear_cell_value(y_cell, u=u, v=v, w=w)
                z_value = _trilinear_cell_value(z_cell, u=u, v=v, w=w)
                jacobian_value = _trilinear_cell_value(jacobian_cell, u=u, v=v, w=w)
                analytic_density = normalization_constant * np.exp(
                    -_DEFAULT_GAUSSIAN_ALPHA * (x_value * x_value + y_value * y_value + z_value * z_value)
                )
                reconstructed_density = _trilinear_cell_value(rho_cell, u=u, v=v, w=w)
                weight = jacobian_value * subcell_volume
                x2 = x_value * x_value
                y2 = y_value * y_value
                z2 = z_value * z_value
                r2 = x2 + y2 + z2
                qzz_factor = 3.0 * z2 - r2
                analytic_x2 += analytic_density * x2 * weight
                analytic_y2 += analytic_density * y2 * weight
                analytic_z2 += analytic_density * z2 * weight
                analytic_r2 += analytic_density * r2 * weight
                analytic_qzz += analytic_density * qzz_factor * weight
                reconstructed_x2 += reconstructed_density * x2 * weight
                reconstructed_y2 += reconstructed_density * y2 * weight
                reconstructed_z2 += reconstructed_density * z2 * weight
                reconstructed_r2 += reconstructed_density * r2 * weight
                reconstructed_qzz += reconstructed_density * qzz_factor * weight
    return (
        float(reconstructed_x2 - analytic_x2),
        float(reconstructed_y2 - analytic_y2),
        float(reconstructed_z2 - analytic_z2),
        float(reconstructed_r2 - analytic_r2),
        float(reconstructed_qzz - analytic_qzz),
    )


def _analytic_gaussian_value(
    x_value: float,
    y_value: float,
    z_value: float,
    *,
    normalization_constant: float,
) -> float:
    return float(
        normalization_constant
        * np.exp(-_DEFAULT_GAUSSIAN_ALPHA * (x_value * x_value + y_value * y_value + z_value * z_value))
    )


def _local_quadratic_fit_coefficients(
    *,
    x_stencil: np.ndarray,
    y_stencil: np.ndarray,
    z_stencil: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    x_flat = np.asarray(x_stencil, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y_stencil, dtype=np.float64).reshape(-1)
    z_flat = np.asarray(z_stencil, dtype=np.float64).reshape(-1)
    design = np.column_stack(
        [
            np.ones_like(x_flat),
            x_flat,
            y_flat,
            z_flat,
            x_flat * x_flat,
            y_flat * y_flat,
            z_flat * z_flat,
            x_flat * y_flat,
            x_flat * z_flat,
            y_flat * z_flat,
        ]
    )
    coefficients, *_ = np.linalg.lstsq(
        design,
        np.asarray(values, dtype=np.float64).reshape(-1),
        rcond=None,
    )
    return np.asarray(coefficients, dtype=np.float64)


def _evaluate_quadratic_fit(
    coefficients: np.ndarray,
    *,
    x_value: float,
    y_value: float,
    z_value: float,
) -> float:
    basis = np.array(
        [
            1.0,
            x_value,
            y_value,
            z_value,
            x_value * x_value,
            y_value * y_value,
            z_value * z_value,
            x_value * y_value,
            x_value * z_value,
            y_value * z_value,
        ],
        dtype=np.float64,
    )
    return float(np.dot(coefficients, basis))


def _cell_and_neighbor_slices(
    cell_index: tuple[int, int, int],
    *,
    shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    i, j, k = cell_index
    nx, ny, nz = shape
    return (
        slice(max(i - 1, 0), min(i + 3, nx)),
        slice(max(j - 1, 0), min(j + 3, ny)),
        slice(max(k - 1, 0), min(k + 3, nz)),
    )


def _inside_cell_profile_errors_from_reconstruction(
    *,
    x_cell: np.ndarray,
    y_cell: np.ndarray,
    z_cell: np.ndarray,
    reconstruction,
    normalization_constant: float,
    cell_subsamples: int,
) -> dict[str, tuple[float, float]]:
    if cell_subsamples < 2:
        raise ValueError("cell_subsamples must be at least 2.")
    profile_errors: dict[str, tuple[float, float]] = {}
    axis_map = {
        "x": lambda t: (t, 0.5, 0.5),
        "y": lambda t: (0.5, t, 0.5),
        "z": lambda t: (0.5, 0.5, t),
    }
    for axis_label, point_builder in axis_map.items():
        errors = []
        for t in np.linspace(0.0, 1.0, cell_subsamples, dtype=np.float64):
            u, v, w = point_builder(float(t))
            x_value = _trilinear_cell_value(x_cell, u=u, v=v, w=w)
            y_value = _trilinear_cell_value(y_cell, u=u, v=v, w=w)
            z_value = _trilinear_cell_value(z_cell, u=u, v=v, w=w)
            analytic_density = _analytic_gaussian_value(
                x_value,
                y_value,
                z_value,
                normalization_constant=normalization_constant,
            )
            reconstructed_density = float(reconstruction(u=u, v=v, w=w, x=x_value, y=y_value, z=z_value))
            errors.append(reconstructed_density - analytic_density)
        error_array = np.asarray(errors, dtype=np.float64)
        profile_errors[axis_label] = (
            float(np.sqrt(np.mean(error_array * error_array))),
            float(np.max(np.abs(error_array))),
        )
    return profile_errors


def _inside_cell_moment_errors_from_reconstruction(
    *,
    x_cell: np.ndarray,
    y_cell: np.ndarray,
    z_cell: np.ndarray,
    jacobian_cell: np.ndarray,
    reconstruction,
    normalization_constant: float,
    logical_cell_volume: float,
    moment_subcell_divisions: tuple[int, int, int],
) -> tuple[float, float, float, float, float]:
    sx, sy, sz = moment_subcell_divisions
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError("moment_subcell_divisions must be positive in every direction.")
    subcell_volume = logical_cell_volume / float(sx * sy * sz)
    analytic_x2 = 0.0
    analytic_y2 = 0.0
    analytic_z2 = 0.0
    analytic_r2 = 0.0
    analytic_qzz = 0.0
    reconstructed_x2 = 0.0
    reconstructed_y2 = 0.0
    reconstructed_z2 = 0.0
    reconstructed_r2 = 0.0
    reconstructed_qzz = 0.0
    for ix in range(sx):
        u = (ix + 0.5) / sx
        for iy in range(sy):
            v = (iy + 0.5) / sy
            for iz in range(sz):
                w = (iz + 0.5) / sz
                x_value = _trilinear_cell_value(x_cell, u=u, v=v, w=w)
                y_value = _trilinear_cell_value(y_cell, u=u, v=v, w=w)
                z_value = _trilinear_cell_value(z_cell, u=u, v=v, w=w)
                jacobian_value = _trilinear_cell_value(jacobian_cell, u=u, v=v, w=w)
                analytic_density = _analytic_gaussian_value(
                    x_value,
                    y_value,
                    z_value,
                    normalization_constant=normalization_constant,
                )
                reconstructed_density = float(reconstruction(u=u, v=v, w=w, x=x_value, y=y_value, z=z_value))
                weight = jacobian_value * subcell_volume
                x2 = x_value * x_value
                y2 = y_value * y_value
                z2 = z_value * z_value
                r2 = x2 + y2 + z2
                qzz_factor = 3.0 * z2 - r2
                analytic_x2 += analytic_density * x2 * weight
                analytic_y2 += analytic_density * y2 * weight
                analytic_z2 += analytic_density * z2 * weight
                analytic_r2 += analytic_density * r2 * weight
                analytic_qzz += analytic_density * qzz_factor * weight
                reconstructed_x2 += reconstructed_density * x2 * weight
                reconstructed_y2 += reconstructed_density * y2 * weight
                reconstructed_z2 += reconstructed_density * z2 * weight
                reconstructed_r2 += reconstructed_density * r2 * weight
                reconstructed_qzz += reconstructed_density * qzz_factor * weight
    return (
        float(reconstructed_x2 - analytic_x2),
        float(reconstructed_y2 - analytic_y2),
        float(reconstructed_z2 - analytic_z2),
        float(reconstructed_r2 - analytic_r2),
        float(reconstructed_qzz - analytic_qzz),
    )


def _inside_cell_reconstruction_summary(
    *,
    reconstruction_label: str,
    x_cell: np.ndarray,
    y_cell: np.ndarray,
    z_cell: np.ndarray,
    jacobian_cell: np.ndarray,
    reconstruction,
    normalization_constant: float,
    logical_cell_volume: float,
    cell_subsamples: int,
    moment_subcell_divisions: tuple[int, int, int],
) -> InsideCellReconstructionSummary:
    profile_errors = _inside_cell_profile_errors_from_reconstruction(
        x_cell=x_cell,
        y_cell=y_cell,
        z_cell=z_cell,
        reconstruction=reconstruction,
        normalization_constant=normalization_constant,
        cell_subsamples=cell_subsamples,
    )
    x2_error, y2_error, z2_error, r2_error, qzz_error = (
        _inside_cell_moment_errors_from_reconstruction(
            x_cell=x_cell,
            y_cell=y_cell,
            z_cell=z_cell,
            jacobian_cell=jacobian_cell,
            reconstruction=reconstruction,
            normalization_constant=normalization_constant,
            logical_cell_volume=logical_cell_volume,
            moment_subcell_divisions=moment_subcell_divisions,
        )
    )
    return InsideCellReconstructionSummary(
        reconstruction_label=reconstruction_label,
        profile_rms_error_x=profile_errors["x"][0],
        profile_rms_error_y=profile_errors["y"][0],
        profile_rms_error_z=profile_errors["z"][0],
        profile_max_abs_error_x=profile_errors["x"][1],
        profile_max_abs_error_y=profile_errors["y"][1],
        profile_max_abs_error_z=profile_errors["z"][1],
        local_x2_contribution_error=x2_error,
        local_y2_contribution_error=y2_error,
        local_z2_contribution_error=z2_error,
        local_r2_contribution_error=r2_error,
        local_quadrupole_component_error=qzz_error,
    )


def run_h2_hartree_inside_cell_representation_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    monitor_shape: tuple[int, int, int] = H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE,
    baseline_monitor_box_half_extents: tuple[float, float, float] = (
        H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
    ),
    cell_subsamples: int = 9,
    moment_subcell_divisions: tuple[int, int, int] = (4, 4, 4),
) -> H2HartreeInsideCellRepresentationAuditResult:
    """Audit Gaussian inside-cell representation error on representative mapped cells."""

    monitor_geometry = _build_monitor_geometry(
        case,
        shape=monitor_shape,
        box_half_extents=baseline_monitor_box_half_extents,
    )
    gaussian_density = _build_gaussian_density(monitor_geometry)
    raw_gaussian = np.exp(
        -_DEFAULT_GAUSSIAN_ALPHA
        * (
            monitor_geometry.x_points * monitor_geometry.x_points
            + monitor_geometry.y_points * monitor_geometry.y_points
            + monitor_geometry.z_points * monitor_geometry.z_points
        )
    )
    normalization_constant = float(
        np.sum(gaussian_density * np.asarray(monitor_geometry.cell_volumes, dtype=np.float64), dtype=np.float64)
        / np.sum(raw_gaussian * np.asarray(monitor_geometry.cell_volumes, dtype=np.float64), dtype=np.float64)
    )
    logical_cell_volume = _logical_cell_volume(monitor_geometry)
    cell_summaries = []
    for cell_label, cell_index, region_label in _representative_cell_indices(monitor_geometry):
        i, j, k = cell_index
        x_cell = np.asarray(monitor_geometry.x_points[i : i + 2, j : j + 2, k : k + 2], dtype=np.float64)
        y_cell = np.asarray(monitor_geometry.y_points[i : i + 2, j : j + 2, k : k + 2], dtype=np.float64)
        z_cell = np.asarray(monitor_geometry.z_points[i : i + 2, j : j + 2, k : k + 2], dtype=np.float64)
        jacobian_cell = np.asarray(monitor_geometry.jacobian[i : i + 2, j : j + 2, k : k + 2], dtype=np.float64)
        rho_cell = np.asarray(gaussian_density[i : i + 2, j : j + 2, k : k + 2], dtype=np.float64)
        profile_errors = _inside_cell_profile_errors(
            x_cell=x_cell,
            y_cell=y_cell,
            z_cell=z_cell,
            rho_cell=rho_cell,
            normalization_constant=normalization_constant,
            cell_subsamples=cell_subsamples,
        )
        x2_error, y2_error, z2_error, r2_error, qzz_error = _inside_cell_moment_errors(
            x_cell=x_cell,
            y_cell=y_cell,
            z_cell=z_cell,
            jacobian_cell=jacobian_cell,
            rho_cell=rho_cell,
            normalization_constant=normalization_constant,
            logical_cell_volume=logical_cell_volume,
            moment_subcell_divisions=moment_subcell_divisions,
        )
        cell_summaries.append(
            InsideCellRepresentationCellSummary(
                cell_label=cell_label,
                cell_index=cell_index,
                region_label=region_label,
                mean_jacobian=float(np.mean(jacobian_cell)),
                profile_rms_error_x=profile_errors["x"][0],
                profile_rms_error_y=profile_errors["y"][0],
                profile_rms_error_z=profile_errors["z"][0],
                profile_max_abs_error_x=profile_errors["x"][1],
                profile_max_abs_error_y=profile_errors["y"][1],
                profile_max_abs_error_z=profile_errors["z"][1],
                local_x2_contribution_error=x2_error,
                local_y2_contribution_error=y2_error,
                local_z2_contribution_error=z2_error,
                local_r2_contribution_error=r2_error,
                local_quadrupole_component_error=qzz_error,
            )
        )
    high_j_cell = next(
        cell for cell in cell_summaries if cell.cell_label == "high_jacobian_interior"
    )
    diagnosis = (
        "This audit compares analytic centered-Gaussian values on the trilinearly mapped cell against "
        "the trilinear reconstruction of nodal Gaussian values. If the high-jacobian interior cell "
        "already shows larger z-profile and z^2/qzz contribution errors than ordinary cells, the fake "
        "quadrupole is being formed inside the mapped-cell field representation itself."
    )
    return H2HartreeInsideCellRepresentationAuditResult(
        cell_summaries=tuple(cell_summaries),
        diagnosis=diagnosis,
        note=(
            "The production measure is kept fixed. Only the cell-local representation of the centered "
            "Gaussian is compared against a denser audit-only reference within representative mapped cells. "
            f"Current high-jacobian interior z-profile RMS error: {high_j_cell.profile_rms_error_z:.6e}."
        ),
    )


def run_h2_hartree_inside_cell_reconstruction_comparison_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    monitor_shape: tuple[int, int, int] = H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE,
    baseline_monitor_box_half_extents: tuple[float, float, float] = (
        H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
    ),
    cell_subsamples: int = 9,
    moment_subcell_divisions: tuple[int, int, int] = (4, 4, 4),
) -> H2HartreeInsideCellReconstructionComparisonAuditResult:
    """Compare audit-only local Gaussian reconstructions on representative mapped cells."""

    monitor_geometry = _build_monitor_geometry(
        case,
        shape=monitor_shape,
        box_half_extents=baseline_monitor_box_half_extents,
    )
    gaussian_density = _build_gaussian_density(monitor_geometry)
    raw_gaussian = np.exp(
        -_DEFAULT_GAUSSIAN_ALPHA
        * (
            monitor_geometry.x_points * monitor_geometry.x_points
            + monitor_geometry.y_points * monitor_geometry.y_points
            + monitor_geometry.z_points * monitor_geometry.z_points
        )
    )
    normalization_constant = float(
        np.sum(gaussian_density * np.asarray(monitor_geometry.cell_volumes, dtype=np.float64), dtype=np.float64)
        / np.sum(raw_gaussian * np.asarray(monitor_geometry.cell_volumes, dtype=np.float64), dtype=np.float64)
    )
    logical_cell_volume = _logical_cell_volume(monitor_geometry)
    cell_summaries = []
    shape = monitor_geometry.spec.shape
    for cell_label, cell_index, region_label in _representative_cell_indices(monitor_geometry):
        i, j, k = cell_index
        x_cell = np.asarray(monitor_geometry.x_points[i : i + 2, j : j + 2, k : k + 2], dtype=np.float64)
        y_cell = np.asarray(monitor_geometry.y_points[i : i + 2, j : j + 2, k : k + 2], dtype=np.float64)
        z_cell = np.asarray(monitor_geometry.z_points[i : i + 2, j : j + 2, k : k + 2], dtype=np.float64)
        jacobian_cell = np.asarray(monitor_geometry.jacobian[i : i + 2, j : j + 2, k : k + 2], dtype=np.float64)
        rho_cell = np.asarray(gaussian_density[i : i + 2, j : j + 2, k : k + 2], dtype=np.float64)

        def _trilinear_reconstruction(*, u: float, v: float, w: float, x: float, y: float, z: float) -> float:
            del x, y, z
            return _trilinear_cell_value(rho_cell, u=u, v=v, w=w)

        sx, sy, sz = _cell_and_neighbor_slices(cell_index, shape=shape)
        quadratic_coefficients = _local_quadratic_fit_coefficients(
            x_stencil=np.asarray(monitor_geometry.x_points[sx, sy, sz], dtype=np.float64),
            y_stencil=np.asarray(monitor_geometry.y_points[sx, sy, sz], dtype=np.float64),
            z_stencil=np.asarray(monitor_geometry.z_points[sx, sy, sz], dtype=np.float64),
            values=np.asarray(gaussian_density[sx, sy, sz], dtype=np.float64),
        )

        def _quadratic_reconstruction(*, u: float, v: float, w: float, x: float, y: float, z: float) -> float:
            del u, v, w
            return _evaluate_quadratic_fit(
                quadratic_coefficients,
                x_value=x,
                y_value=y,
                z_value=z,
            )

        reconstruction_summaries = (
            _inside_cell_reconstruction_summary(
                reconstruction_label="trilinear_nodal",
                x_cell=x_cell,
                y_cell=y_cell,
                z_cell=z_cell,
                jacobian_cell=jacobian_cell,
                reconstruction=_trilinear_reconstruction,
                normalization_constant=normalization_constant,
                logical_cell_volume=logical_cell_volume,
                cell_subsamples=cell_subsamples,
                moment_subcell_divisions=moment_subcell_divisions,
            ),
            _inside_cell_reconstruction_summary(
                reconstruction_label="local_quadratic_fit",
                x_cell=x_cell,
                y_cell=y_cell,
                z_cell=z_cell,
                jacobian_cell=jacobian_cell,
                reconstruction=_quadratic_reconstruction,
                normalization_constant=normalization_constant,
                logical_cell_volume=logical_cell_volume,
                cell_subsamples=cell_subsamples,
                moment_subcell_divisions=moment_subcell_divisions,
            ),
        )
        cell_summaries.append(
            InsideCellReconstructionComparisonCellSummary(
                cell_label=cell_label,
                cell_index=cell_index,
                region_label=region_label,
                mean_jacobian=float(np.mean(jacobian_cell)),
                reconstruction_summaries=reconstruction_summaries,
            )
        )

    diagnosis = (
        "This audit keeps mapping, Jacobian, and quadrature fixed, and only swaps the local field "
        "reconstruction inside representative mapped cells. If local z^2 / Qzz errors drop sharply "
        "when trilinear nodal reconstruction is replaced by a local quadratic fit, the low-level fake "
        "quadrupole is primarily a field-representation issue."
    )
    return H2HartreeInsideCellReconstructionComparisonAuditResult(
        cell_summaries=tuple(cell_summaries),
        diagnosis=diagnosis,
        note=(
            "The local quadratic fit is audit-only and uses a small neighboring nodal stencil. "
            "It does not modify the production mapped grid, Jacobian, or measure."
        ),
    )


def _reference_quadrature_summary(
    monitor_geometry: MonitorGridGeometry,
    *,
    subcell_divisions: tuple[int, int, int],
    gaussian_target_charge: float = 2.0,
) -> tuple[tuple[ReferenceQuadratureIntegralRow, ...], float, float]:
    sx, sy, sz = subcell_divisions
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError("subcell_divisions must be positive in every direction.")
    box_bounds = monitor_geometry.spec.box_bounds
    references = _analytic_box_integrals(box_bounds)
    polynomial_references = _polynomial_reference_values(box_bounds)
    logical_subcell_volume = _logical_cell_volume(monitor_geometry) / float(sx * sy * sz)
    x_nodal = np.asarray(monitor_geometry.x_points, dtype=np.float64)
    y_nodal = np.asarray(monitor_geometry.y_points, dtype=np.float64)
    z_nodal = np.asarray(monitor_geometry.z_points, dtype=np.float64)
    jacobian_nodal = np.asarray(monitor_geometry.jacobian, dtype=np.float64)

    accumulators = {
        "1": 0.0,
        "x2": 0.0,
        "y2": 0.0,
        "z2": 0.0,
        "r2": 0.0,
        "centered_gaussian": 0.0,
    }
    gaussian_tensor = np.zeros((3, 3), dtype=np.float64)
    for ix in range(sx):
        u = (ix + 0.5) / sx
        for iy in range(sy):
            v = (iy + 0.5) / sy
            for iz in range(sz):
                w = (iz + 0.5) / sz
                xs = _trilinear_sample(x_nodal, u=u, v=v, w=w)
                ys = _trilinear_sample(y_nodal, u=u, v=v, w=w)
                zs = _trilinear_sample(z_nodal, u=u, v=v, w=w)
                js = _trilinear_sample(jacobian_nodal, u=u, v=v, w=w)
                weight = js * logical_subcell_volume
                x2 = xs * xs
                y2 = ys * ys
                z2 = zs * zs
                r2 = x2 + y2 + z2
                gaussian = np.exp(-_DEFAULT_GAUSSIAN_ALPHA * r2)
                accumulators["1"] += float(np.sum(weight, dtype=np.float64))
                accumulators["x2"] += float(np.sum(x2 * weight, dtype=np.float64))
                accumulators["y2"] += float(np.sum(y2 * weight, dtype=np.float64))
                accumulators["z2"] += float(np.sum(z2 * weight, dtype=np.float64))
                accumulators["r2"] += float(np.sum(r2 * weight, dtype=np.float64))
                accumulators["centered_gaussian"] += float(np.sum(gaussian * weight, dtype=np.float64))
                gaussian_tensor[0, 0] += float(
                    np.sum(gaussian * (3.0 * x2 - r2) * weight, dtype=np.float64)
                )
                gaussian_tensor[1, 1] += float(
                    np.sum(gaussian * (3.0 * y2 - r2) * weight, dtype=np.float64)
                )
                gaussian_tensor[2, 2] += float(
                    np.sum(gaussian * (3.0 * z2 - r2) * weight, dtype=np.float64)
                )
                xy = xs * ys
                xz = xs * zs
                yz = ys * zs
                offdiag_xy = float(np.sum(gaussian * 3.0 * xy * weight, dtype=np.float64))
                offdiag_xz = float(np.sum(gaussian * 3.0 * xz * weight, dtype=np.float64))
                offdiag_yz = float(np.sum(gaussian * 3.0 * yz * weight, dtype=np.float64))
                gaussian_tensor[0, 1] += offdiag_xy
                gaussian_tensor[1, 0] += offdiag_xy
                gaussian_tensor[0, 2] += offdiag_xz
                gaussian_tensor[2, 0] += offdiag_xz
                gaussian_tensor[1, 2] += offdiag_yz
                gaussian_tensor[2, 1] += offdiag_yz

    production_fields = _polynomial_fields(
        monitor_geometry.x_points,
        monitor_geometry.y_points,
        monitor_geometry.z_points,
    )
    production_weights = np.asarray(monitor_geometry.cell_volumes, dtype=np.float64)
    integral_rows = []
    for label in ("1", "x2", "y2", "z2", "r2"):
        reference_value = polynomial_references[label]
        production_value = float(
            np.sum(production_fields[label] * production_weights, dtype=np.float64)
        )
        reference_quadrature_value = float(accumulators[label])
        integral_rows.append(
            ReferenceQuadratureIntegralRow(
                function_label=label,
                reference_value=float(reference_value),
                production_value=production_value,
                production_bias=float(production_value - reference_value),
                reference_quadrature_value=reference_quadrature_value,
                reference_quadrature_bias=float(reference_quadrature_value - reference_value),
            )
        )
    gaussian_production_value = float(
        np.sum(
            np.exp(
                -_DEFAULT_GAUSSIAN_ALPHA
                * (
                    monitor_geometry.x_points * monitor_geometry.x_points
                    + monitor_geometry.y_points * monitor_geometry.y_points
                    + monitor_geometry.z_points * monitor_geometry.z_points
                )
            )
            * production_weights,
            dtype=np.float64,
        )
    )
    integral_rows.append(
        ReferenceQuadratureIntegralRow(
            function_label="centered_gaussian",
            reference_value=float(references["centered_gaussian"]),
            production_value=gaussian_production_value,
            production_bias=float(gaussian_production_value - references["centered_gaussian"]),
            reference_quadrature_value=float(accumulators["centered_gaussian"]),
            reference_quadrature_bias=float(
                accumulators["centered_gaussian"] - references["centered_gaussian"]
            ),
        )
    )
    gaussian_production_density = _build_gaussian_density(
        monitor_geometry,
        target_charge=gaussian_target_charge,
    )
    gaussian_production_quadrupole = _quadrupole_tensor(
        gaussian_production_density,
        monitor_geometry,
    )
    gaussian_reference_quadrupole = (
        (gaussian_target_charge / accumulators["centered_gaussian"]) * gaussian_tensor
    )
    return (
        tuple(integral_rows),
        float(np.linalg.norm(gaussian_production_quadrupole)),
        float(np.linalg.norm(gaussian_reference_quadrupole)),
    )


def run_h2_hartree_reference_quadrature_audit(
    *,
    case: BenchmarkCase = H2_BENCHMARK_CASE,
    monitor_shape: tuple[int, int, int] = H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE,
    baseline_monitor_box_half_extents: tuple[float, float, float] = (
        H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
    ),
    subcell_divisions: tuple[int, int, int] = (2, 2, 2),
) -> H2HartreeReferenceQuadratureAuditResult:
    """Compare the production nodal monitor measure against an audit-only subcell quadrature."""

    monitor_geometry = _build_monitor_geometry(
        case,
        shape=monitor_shape,
        box_half_extents=baseline_monitor_box_half_extents,
    )
    integral_rows, production_quadrupole_norm, reference_quadrupole_norm = (
        _reference_quadrature_summary(
            monitor_geometry,
            subcell_divisions=subcell_divisions,
        )
    )
    gaussian_row = next(
        row for row in integral_rows if row.function_label == "centered_gaussian"
    )
    return H2HartreeReferenceQuadratureAuditResult(
        subcell_divisions=subcell_divisions,
        integral_rows=integral_rows,
        gaussian_production_total_charge=float(gaussian_row.production_value),
        gaussian_reference_quadrature_total_charge=float(gaussian_row.reference_quadrature_value),
        gaussian_production_quadrupole_norm=production_quadrupole_norm,
        gaussian_reference_quadrature_quadrupole_norm=reference_quadrupole_norm,
        note=(
            "This audit keeps the mapped grid fixed and compares the production nodal J*dξ*dη*dζ "
            "measure against an audit-only trilinear subcell midpoint quadrature on each logical cell."
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
