r"""Geometry-consistency audit for the H2 monitor-grid kinetic path.

This audit does not modify the monitor geometry or the kinetic implementation.
It checks whether the stored monitor-grid geometry quantities are internally
consistent and whether the A-grid kinetic energy identity

    <psi, T psi>  vs  1/2 \int g^{ab} (d_a psi) (d_b psi) J dxi

closes on the same geometry for the frozen H2 trial orbital, the current bad
fixed-potential eigensolver orbital, and a pair of smooth probe fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isogrid.config import BenchmarkCase
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR
from isogrid.grid import H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE
from isogrid.grid import MonitorGridGeometry
from isogrid.grid import build_h2_local_patch_development_monitor_grid
from isogrid.ks import solve_fixed_potential_static_local_eigenproblem
from isogrid.ops import apply_monitor_grid_kinetic_operator
from isogrid.ops import validate_orbital_field
from isogrid.pseudo import LocalPotentialPatchParameters

from .baselines import H2_KINETIC_FORM_AUDIT_BASELINE
from .baselines import H2_KINETIC_OPERATOR_AUDIT_BASELINE
from .h2_monitor_grid_kinetic_operator_audit import _build_smooth_field
from .h2_monitor_grid_operator_audit import ScalarFieldSummary
from .h2_monitor_grid_operator_audit import _build_frozen_density
from .h2_monitor_grid_operator_audit import _compute_region_masks
from .h2_monitor_grid_operator_audit import _field_summary
from .h2_monitor_grid_operator_audit import _weighted_inner_product


@dataclass(frozen=True)
class GeometryMismatchSummary:
    """One global geometry mismatch summary field."""

    label: str
    absolute_summary: ScalarFieldSummary
    relative_summary: ScalarFieldSummary


@dataclass(frozen=True)
class GeometryRegionSummary:
    """Regionwise geometry summary on the monitor grid."""

    region_name: str
    point_fraction: float
    jacobian_summary: ScalarFieldSummary
    cell_volume_summary: ScalarFieldSummary
    spacing_summary: ScalarFieldSummary
    inverse_metric_trace_summary: ScalarFieldSummary
    metric_condition_summary: ScalarFieldSummary
    cell_volume_jacobian_relative_rms: float
    inverse_metric_relative_rms: float


@dataclass(frozen=True)
class GeometryConsistencySummary:
    """Top-level stored/reconstructed geometry consistency summary."""

    jacobian_summary: ScalarFieldSummary
    cell_volume_summary: ScalarFieldSummary
    spacing_summary: ScalarFieldSummary
    inverse_metric_diagonal_summaries: tuple[ScalarFieldSummary, ScalarFieldSummary, ScalarFieldSummary]
    inverse_metric_offdiagonal_rms: tuple[float, float, float]
    total_physical_volume: float
    logical_cell_volume_element: float
    cell_volume_vs_jacobian: GeometryMismatchSummary
    reconstructed_jacobian: GeometryMismatchSummary
    reconstructed_inverse_metric: GeometryMismatchSummary
    sqrt_det_metric_vs_jacobian: GeometryMismatchSummary
    metric_inverse_identity_summary: ScalarFieldSummary
    region_summaries: tuple[GeometryRegionSummary, ...]


@dataclass(frozen=True)
class KineticIdentityCenterlineSample:
    """Center-line sample for operator/reference kinetic-density comparison."""

    sample_index: int
    z_coordinate_bohr: float
    orbital_value: float
    operator_indicator: float
    gradient_indicator: float
    delta_indicator: float


@dataclass(frozen=True)
class KineticIdentityRegionSummary:
    """Regionwise operator-vs-gradient kinetic-energy diagnostics."""

    region_name: str
    point_fraction: float
    operator_indicator_summary: ScalarFieldSummary
    gradient_indicator_summary: ScalarFieldSummary
    delta_indicator_summary: ScalarFieldSummary
    operator_weighted_contribution_ha: float
    gradient_weighted_contribution_ha: float
    delta_weighted_contribution_mha: float


@dataclass(frozen=True)
class KineticIdentityFieldResult:
    """Kinetic-energy identity diagnostics for one field on the A-grid."""

    shape_label: str
    field_label: str
    weighted_norm: float
    operator_kinetic_ha: float
    gradient_reference_ha: float
    delta_kinetic_mha: float
    operator_action_summary: ScalarFieldSummary
    operator_indicator_summary: ScalarFieldSummary
    gradient_indicator_summary: ScalarFieldSummary
    delta_indicator_summary: ScalarFieldSummary
    region_summaries: tuple[KineticIdentityRegionSummary, ...]
    centerline_samples: tuple[KineticIdentityCenterlineSample, ...]
    source_eigenvalue_ha: float | None
    source_residual_norm: float | None
    source_converged: bool | None


@dataclass(frozen=True)
class H2MonitorGridGeometryConsistencyAuditResult:
    """Top-level H2 monitor-grid geometry-consistency audit result."""

    geometry_summary: GeometryConsistencySummary
    frozen_trial_result: KineticIdentityFieldResult
    bad_eigen_result: KineticIdentityFieldResult
    smooth_field_results: tuple[KineticIdentityFieldResult, ...]
    legacy_frozen_kinetic_reference_ha: float
    legacy_bad_eigen_kinetic_reference_ha: float
    diagnosis: str
    note: str


def _default_patch_parameters() -> LocalPotentialPatchParameters:
    return LocalPotentialPatchParameters(
        patch_radius_scale=0.75,
        patch_grid_shape=(25, 25, 25),
        correction_strength=1.30,
        interpolation_neighbors=8,
    )


def _logical_spacing(logical_coordinates: np.ndarray) -> float:
    coordinates = np.asarray(logical_coordinates, dtype=np.float64)
    spacings = np.diff(coordinates)
    if not np.allclose(spacings, spacings[0]):
        raise ValueError("The geometry-consistency audit expects uniform logical coordinates.")
    return float(spacings[0])


def _relative_mismatch(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    reference = np.maximum(np.maximum(np.abs(numerator), np.abs(denominator)), 1.0e-30)
    return np.abs(numerator - denominator) / reference


def _reconstruct_geometry_from_coordinates(
    grid_geometry: MonitorGridGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.stack(
        [grid_geometry.x_points, grid_geometry.y_points, grid_geometry.z_points],
        axis=-1,
    )
    basis = np.zeros(coordinates.shape[:-1] + (3, 3), dtype=np.float64)
    for component in range(3):
        basis[..., 0, component] = np.gradient(
            coordinates[..., component],
            grid_geometry.logical_x,
            axis=0,
            edge_order=2,
        )
        basis[..., 1, component] = np.gradient(
            coordinates[..., component],
            grid_geometry.logical_y,
            axis=1,
            edge_order=2,
        )
        basis[..., 2, component] = np.gradient(
            coordinates[..., component],
            grid_geometry.logical_z,
            axis=2,
            edge_order=2,
        )
    metric = np.einsum("...ai,...bi->...ab", basis, basis)
    inverse_metric = np.linalg.inv(metric)
    jacobian = np.linalg.det(basis)
    dxi = _logical_spacing(grid_geometry.logical_x)
    deta = _logical_spacing(grid_geometry.logical_y)
    dzeta = _logical_spacing(grid_geometry.logical_z)
    cell_volumes = jacobian * dxi * deta * dzeta
    return basis, metric, inverse_metric, jacobian, cell_volumes


def _geometry_region_summaries(
    grid_geometry: MonitorGridGeometry,
    *,
    cell_volume_jacobian_relative: np.ndarray,
    inverse_metric_relative: np.ndarray,
    case: BenchmarkCase,
) -> tuple[GeometryRegionSummary, ...]:
    summaries: list[GeometryRegionSummary] = []
    total_points = float(np.prod(grid_geometry.spec.shape))
    inverse_metric_trace = np.trace(grid_geometry.inverse_metric_tensor, axis1=-2, axis2=-1)
    metric_eigenvalues = np.linalg.eigvalsh(grid_geometry.metric_tensor)
    metric_condition = metric_eigenvalues[..., -1] / np.maximum(metric_eigenvalues[..., 0], 1.0e-30)
    for region_name, mask in _compute_region_masks(case=case, grid_geometry=grid_geometry):
        if not np.any(mask):
            continue
        summaries.append(
            GeometryRegionSummary(
                region_name=region_name,
                point_fraction=float(np.sum(mask) / total_points),
                jacobian_summary=_field_summary(grid_geometry.jacobian, mask=mask),
                cell_volume_summary=_field_summary(grid_geometry.cell_volumes, mask=mask),
                spacing_summary=_field_summary(grid_geometry.spacing_measure, mask=mask),
                inverse_metric_trace_summary=_field_summary(inverse_metric_trace, mask=mask),
                metric_condition_summary=_field_summary(metric_condition, mask=mask),
                cell_volume_jacobian_relative_rms=float(
                    np.sqrt(np.mean(cell_volume_jacobian_relative[mask] ** 2))
                ),
                inverse_metric_relative_rms=float(
                    np.sqrt(np.mean(inverse_metric_relative[mask] ** 2))
                ),
            )
        )
    return tuple(summaries)


def _geometry_summary(
    grid_geometry: MonitorGridGeometry,
    case: BenchmarkCase,
) -> GeometryConsistencySummary:
    _, metric_reconstructed, inverse_metric_reconstructed, jacobian_reconstructed, cell_volumes_reconstructed = (
        _reconstruct_geometry_from_coordinates(grid_geometry)
    )
    dxi = _logical_spacing(grid_geometry.logical_x)
    deta = _logical_spacing(grid_geometry.logical_y)
    dzeta = _logical_spacing(grid_geometry.logical_z)
    logical_volume_element = dxi * deta * dzeta
    cell_volume_from_jacobian = grid_geometry.jacobian * logical_volume_element
    metric_det_sqrt = np.sqrt(np.clip(np.linalg.det(grid_geometry.metric_tensor), 0.0, None))
    identity = np.einsum("...ab,...bc->...ac", grid_geometry.metric_tensor, grid_geometry.inverse_metric_tensor)
    identity_mismatch = np.linalg.norm(identity - np.eye(3), axis=(-2, -1))
    inverse_metric_relative = _relative_mismatch(
        grid_geometry.inverse_metric_tensor,
        inverse_metric_reconstructed,
    )
    cell_volume_jacobian_relative = _relative_mismatch(
        grid_geometry.cell_volumes,
        cell_volume_from_jacobian,
    )
    return GeometryConsistencySummary(
        jacobian_summary=_field_summary(grid_geometry.jacobian),
        cell_volume_summary=_field_summary(grid_geometry.cell_volumes),
        spacing_summary=_field_summary(grid_geometry.spacing_measure),
        inverse_metric_diagonal_summaries=(
            _field_summary(grid_geometry.inverse_metric_tensor[..., 0, 0]),
            _field_summary(grid_geometry.inverse_metric_tensor[..., 1, 1]),
            _field_summary(grid_geometry.inverse_metric_tensor[..., 2, 2]),
        ),
        inverse_metric_offdiagonal_rms=(
            float(np.sqrt(np.mean(grid_geometry.inverse_metric_tensor[..., 0, 1] ** 2))),
            float(np.sqrt(np.mean(grid_geometry.inverse_metric_tensor[..., 0, 2] ** 2))),
            float(np.sqrt(np.mean(grid_geometry.inverse_metric_tensor[..., 1, 2] ** 2))),
        ),
        total_physical_volume=float(np.sum(grid_geometry.cell_volumes)),
        logical_cell_volume_element=logical_volume_element,
        cell_volume_vs_jacobian=GeometryMismatchSummary(
            label="cell_volume_vs_jacobian",
            absolute_summary=_field_summary(grid_geometry.cell_volumes - cell_volume_from_jacobian),
            relative_summary=_field_summary(cell_volume_jacobian_relative),
        ),
        reconstructed_jacobian=GeometryMismatchSummary(
            label="stored_vs_reconstructed_jacobian",
            absolute_summary=_field_summary(grid_geometry.jacobian - jacobian_reconstructed),
            relative_summary=_field_summary(
                _relative_mismatch(grid_geometry.jacobian, jacobian_reconstructed)
            ),
        ),
        reconstructed_inverse_metric=GeometryMismatchSummary(
            label="stored_vs_reconstructed_inverse_metric",
            absolute_summary=_field_summary(
                np.linalg.norm(grid_geometry.inverse_metric_tensor - inverse_metric_reconstructed, axis=(-2, -1))
            ),
            relative_summary=_field_summary(inverse_metric_relative),
        ),
        sqrt_det_metric_vs_jacobian=GeometryMismatchSummary(
            label="sqrt_det_metric_vs_jacobian",
            absolute_summary=_field_summary(metric_det_sqrt - grid_geometry.jacobian),
            relative_summary=_field_summary(
                _relative_mismatch(metric_det_sqrt, grid_geometry.jacobian)
            ),
        ),
        metric_inverse_identity_summary=_field_summary(identity_mismatch),
        region_summaries=_geometry_region_summaries(
            grid_geometry,
            cell_volume_jacobian_relative=cell_volume_jacobian_relative,
            inverse_metric_relative=inverse_metric_relative,
            case=case,
        ),
    )


def _gradient_reference_density(
    orbital: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> np.ndarray:
    field = validate_orbital_field(orbital, grid_geometry=grid_geometry, name="orbital")
    gradients = np.gradient(
        field,
        grid_geometry.logical_x,
        grid_geometry.logical_y,
        grid_geometry.logical_z,
        edge_order=2,
    )
    inverse_metric = np.asarray(grid_geometry.inverse_metric_tensor, dtype=np.float64)
    density = np.zeros(grid_geometry.spec.shape, dtype=np.float64)
    for axis_a in range(3):
        for axis_b in range(3):
            density += np.real(np.conjugate(gradients[axis_a]) * inverse_metric[..., axis_a, axis_b] * gradients[axis_b])
    return 0.5 * density


def _kinetic_identity_centerline_samples(
    orbital: np.ndarray,
    operator_indicator: np.ndarray,
    gradient_indicator: np.ndarray,
    grid_geometry: MonitorGridGeometry,
) -> tuple[KineticIdentityCenterlineSample, ...]:
    center_ix = grid_geometry.spec.nx // 2
    center_iy = grid_geometry.spec.ny // 2
    z_coordinates = grid_geometry.z_points[center_ix, center_iy, :]
    sample_indices = (
        0,
        len(z_coordinates) // 4,
        len(z_coordinates) // 2,
        3 * len(z_coordinates) // 4,
        len(z_coordinates) - 1,
    )
    return tuple(
        KineticIdentityCenterlineSample(
            sample_index=index,
            z_coordinate_bohr=float(z_coordinates[index]),
            orbital_value=float(orbital[center_ix, center_iy, index]),
            operator_indicator=float(operator_indicator[center_ix, center_iy, index]),
            gradient_indicator=float(gradient_indicator[center_ix, center_iy, index]),
            delta_indicator=float(
                operator_indicator[center_ix, center_iy, index]
                - gradient_indicator[center_ix, center_iy, index]
            ),
        )
        for index in sample_indices
    )


def _kinetic_identity_region_summaries(
    *,
    operator_indicator: np.ndarray,
    gradient_indicator: np.ndarray,
    grid_geometry: MonitorGridGeometry,
    denominator: float,
    case: BenchmarkCase,
) -> tuple[KineticIdentityRegionSummary, ...]:
    delta_indicator = operator_indicator - gradient_indicator
    summaries: list[KineticIdentityRegionSummary] = []
    total_points = float(np.prod(grid_geometry.spec.shape))
    for region_name, mask in _compute_region_masks(case=case, grid_geometry=grid_geometry):
        if not np.any(mask):
            continue
        weights = grid_geometry.cell_volumes[mask]
        op_values = operator_indicator[mask]
        grad_values = gradient_indicator[mask]
        delta_values = delta_indicator[mask]
        summaries.append(
            KineticIdentityRegionSummary(
                region_name=region_name,
                point_fraction=float(np.sum(mask) / total_points),
                operator_indicator_summary=_field_summary(op_values),
                gradient_indicator_summary=_field_summary(grad_values),
                delta_indicator_summary=_field_summary(delta_values),
                operator_weighted_contribution_ha=float(np.sum(op_values * weights) / denominator),
                gradient_weighted_contribution_ha=float(np.sum(grad_values * weights) / denominator),
                delta_weighted_contribution_mha=float(
                    1000.0 * np.sum(delta_values * weights) / denominator
                ),
            )
        )
    return tuple(summaries)


def _evaluate_field(
    *,
    orbital: np.ndarray,
    field_label: str,
    grid_geometry: MonitorGridGeometry,
    case: BenchmarkCase,
    source_eigenvalue_ha: float | None,
    source_residual_norm: float | None,
    source_converged: bool | None,
) -> KineticIdentityFieldResult:
    field = validate_orbital_field(orbital, grid_geometry=grid_geometry, name=field_label)
    kinetic_action = apply_monitor_grid_kinetic_operator(field, grid_geometry=grid_geometry)
    operator_indicator = np.real(np.conjugate(field) * kinetic_action)
    gradient_indicator = _gradient_reference_density(field, grid_geometry)
    denominator = float(np.real_if_close(_weighted_inner_product(field, field, grid_geometry)))
    operator_kinetic = float(
        np.real_if_close(_weighted_inner_product(field, kinetic_action, grid_geometry) / denominator)
    )
    gradient_kinetic = float(np.sum(gradient_indicator * grid_geometry.cell_volumes) / denominator)
    return KineticIdentityFieldResult(
        shape_label="baseline",
        field_label=field_label,
        weighted_norm=float(np.sqrt(max(denominator, 0.0))),
        operator_kinetic_ha=operator_kinetic,
        gradient_reference_ha=gradient_kinetic,
        delta_kinetic_mha=float((operator_kinetic - gradient_kinetic) * 1000.0),
        operator_action_summary=_field_summary(kinetic_action),
        operator_indicator_summary=_field_summary(operator_indicator),
        gradient_indicator_summary=_field_summary(gradient_indicator),
        delta_indicator_summary=_field_summary(operator_indicator - gradient_indicator),
        region_summaries=_kinetic_identity_region_summaries(
            operator_indicator=operator_indicator,
            gradient_indicator=gradient_indicator,
            grid_geometry=grid_geometry,
            denominator=denominator,
            case=case,
        ),
        centerline_samples=_kinetic_identity_centerline_samples(
            field,
            operator_indicator,
            gradient_indicator,
            grid_geometry,
        ),
        source_eigenvalue_ha=source_eigenvalue_ha,
        source_residual_norm=source_residual_norm,
        source_converged=source_converged,
    )


def _evaluate_bad_eigen_orbital(
    *,
    case: BenchmarkCase,
    grid_geometry: MonitorGridGeometry,
) -> tuple[np.ndarray, float, float, bool]:
    _, rho_up, rho_down, _ = _build_frozen_density(case=case, grid_geometry=grid_geometry)
    result = solve_fixed_potential_static_local_eigenproblem(
        grid_geometry=grid_geometry,
        rho_up=rho_up,
        rho_down=rho_down,
        spin_channel="up",
        k=1,
        case=case,
        tolerance=1.0e-3,
        ncv=20,
        use_monitor_patch=True,
        patch_parameters=_default_patch_parameters(),
    )
    return (
        result.orbitals[0],
        float(result.eigenvalues[0]),
        float(result.residual_norms[0]),
        bool(result.converged),
    )


def _diagnosis(
    *,
    geometry_summary: GeometryConsistencySummary,
    frozen_result: KineticIdentityFieldResult,
    bad_eigen_result: KineticIdentityFieldResult,
    smooth_field_results: tuple[KineticIdentityFieldResult, ...],
) -> str:
    raw_geometry_closed = (
        geometry_summary.reconstructed_jacobian.relative_summary.rms < 1.0e-10
        and geometry_summary.reconstructed_inverse_metric.relative_summary.rms < 1.0e-10
        and geometry_summary.cell_volume_vs_jacobian.relative_summary.rms < 1.0e-12
        and geometry_summary.sqrt_det_metric_vs_jacobian.relative_summary.rms < 1.0e-10
    )
    smooth_close = all(abs(item.delta_kinetic_mha) < 10.0 for item in smooth_field_results)
    frozen_close = abs(frozen_result.delta_kinetic_mha) < 20.0
    far_field_delta = next(
        region for region in bad_eigen_result.region_summaries if region.region_name == "far_field"
    )
    if raw_geometry_closed and smooth_close and frozen_close:
        if bad_eigen_result.gradient_reference_ha > 0.0 and bad_eigen_result.operator_kinetic_ha < 0.0:
            return (
                "The stored monitor-grid geometry looks internally self-consistent: jacobian, "
                "cell volumes, inverse metric, and det(metric)=J^2 all close numerically. The "
                "main failure is instead a kinetic-energy identity breakdown on the bad eigensolver "
                "orbital: <psi,Tpsi> is strongly negative while the gradient-based reference energy "
                "stays positive. The mismatch localizes primarily in the far-field region, which "
                "points more toward a geometry/operator/boundary coupling defect in the monitor-grid "
                "kinetic path than to a raw jacobian or inverse-metric construction bug."
            )
        if abs(far_field_delta.delta_weighted_contribution_mha) > 100.0:
            return (
                "The geometry storage itself looks mostly closed, but the operator/reference kinetic "
                "identity mismatch is already dominated by far-field contributions."
            )
    return (
        "The geometry-consistency audit found a real operator/reference kinetic mismatch, but the "
        "root cause still needs a narrower geometry-operator identity inspection."
    )


def run_h2_monitor_grid_geometry_consistency_audit(
    case: BenchmarkCase = H2_BENCHMARK_CASE,
) -> H2MonitorGridGeometryConsistencyAuditResult:
    """Run the H2 monitor-grid geometry-consistency audit."""

    grid_geometry = build_h2_local_patch_development_monitor_grid()
    geometry_summary = _geometry_summary(grid_geometry, case)
    frozen_trial_orbital, _, _, _ = _build_frozen_density(case=case, grid_geometry=grid_geometry)
    bad_eigen_orbital, bad_eigenvalue, bad_residual, bad_converged = _evaluate_bad_eigen_orbital(
        case=case,
        grid_geometry=grid_geometry,
    )

    frozen_result = _evaluate_field(
        orbital=frozen_trial_orbital,
        field_label="frozen_trial_orbital",
        grid_geometry=grid_geometry,
        case=case,
        source_eigenvalue_ha=None,
        source_residual_norm=None,
        source_converged=None,
    )
    bad_eigen_result = _evaluate_field(
        orbital=bad_eigen_orbital,
        field_label="bad_eigensolver_orbital_k1",
        grid_geometry=grid_geometry,
        case=case,
        source_eigenvalue_ha=bad_eigenvalue,
        source_residual_norm=bad_residual,
        source_converged=bad_converged,
    )
    smooth_field_results = (
        _evaluate_field(
            orbital=_build_smooth_field("gaussian", grid_geometry),
            field_label="smooth_gaussian",
            grid_geometry=grid_geometry,
            case=case,
            source_eigenvalue_ha=None,
            source_residual_norm=None,
            source_converged=None,
        ),
        _evaluate_field(
            orbital=_build_smooth_field("cosine", grid_geometry),
            field_label="smooth_cosine",
            grid_geometry=grid_geometry,
            case=case,
            source_eigenvalue_ha=None,
            source_residual_norm=None,
            source_converged=None,
        ),
    )
    result = H2MonitorGridGeometryConsistencyAuditResult(
        geometry_summary=geometry_summary,
        frozen_trial_result=frozen_result,
        bad_eigen_result=bad_eigen_result,
        smooth_field_results=smooth_field_results,
        legacy_frozen_kinetic_reference_ha=H2_KINETIC_OPERATOR_AUDIT_BASELINE.legacy_route.frozen_kinetic_ha,
        legacy_bad_eigen_kinetic_reference_ha=H2_KINETIC_OPERATOR_AUDIT_BASELINE.legacy_route.eigen_kinetic_ha,
        diagnosis="",
        note=(
            "This is a monitor-grid geometry-consistency audit only. It keeps the H2 singlet "
            "frozen density fixed, does not modify the geometry or kinetic implementation, and "
            "compares operator kinetic against a gradient-based reference kinetic identity on the "
            "same A-grid geometry."
        ),
    )
    return H2MonitorGridGeometryConsistencyAuditResult(
        geometry_summary=result.geometry_summary,
        frozen_trial_result=result.frozen_trial_result,
        bad_eigen_result=result.bad_eigen_result,
        smooth_field_results=result.smooth_field_results,
        legacy_frozen_kinetic_reference_ha=result.legacy_frozen_kinetic_reference_ha,
        legacy_bad_eigen_kinetic_reference_ha=result.legacy_bad_eigen_kinetic_reference_ha,
        diagnosis=_diagnosis(
            geometry_summary=result.geometry_summary,
            frozen_result=result.frozen_trial_result,
            bad_eigen_result=result.bad_eigen_result,
            smooth_field_results=result.smooth_field_results,
        ),
        note=result.note,
    )


def _print_field_result(result: KineticIdentityFieldResult) -> None:
    print(f"field: {result.field_label}")
    print(f"  weighted norm: {result.weighted_norm:.12f}")
    print(
        f"  operator <T> [Ha]: {result.operator_kinetic_ha:+.12f} | "
        f"gradient ref [Ha]: {result.gradient_reference_ha:+.12f} | "
        f"delta [mHa]: {result.delta_kinetic_mha:+.3f}"
    )
    if result.source_eigenvalue_ha is not None:
        print(
            f"  eig info: eps={result.source_eigenvalue_ha:+.12f} Ha, "
            f"res={result.source_residual_norm:.6e}, converged={result.source_converged}"
        )
    print(
        "  operator Tpsi summary: "
        f"min={result.operator_action_summary.minimum:+.6e}, "
        f"max={result.operator_action_summary.maximum:+.6e}, "
        f"rms={result.operator_action_summary.rms:.6e}"
    )
    print(
        "  density summaries: "
        f"operator_rms={result.operator_indicator_summary.rms:.6e}, "
        f"gradient_rms={result.gradient_indicator_summary.rms:.6e}, "
        f"delta_rms={result.delta_indicator_summary.rms:.6e}"
    )
    print("  region diagnostics:")
    for region in result.region_summaries:
        print(
            "    "
            f"{region.region_name}: op={region.operator_weighted_contribution_ha:+.12f} Ha, "
            f"grad={region.gradient_weighted_contribution_ha:+.12f} Ha, "
            f"delta={region.delta_weighted_contribution_mha:+.3f} mHa"
        )
    print("  center-line samples:")
    for sample in result.centerline_samples:
        print(
            "    "
            f"z[{sample.sample_index:02d}]={sample.z_coordinate_bohr:+.6f} -> "
            f"psi={sample.orbital_value:+.6e}, "
            f"op={sample.operator_indicator:+.6e}, "
            f"grad={sample.gradient_indicator:+.6e}, "
            f"delta={sample.delta_indicator:+.6e}"
        )


def print_h2_monitor_grid_geometry_consistency_audit_summary(
    result: H2MonitorGridGeometryConsistencyAuditResult,
) -> None:
    """Print the compact H2 monitor-grid geometry-consistency audit summary."""

    geometry = result.geometry_summary
    print("IsoGridDFT H2 monitor-grid geometry consistency audit")
    print(f"note: {result.note}")
    print(
        "legacy kinetic references: "
        f"frozen={result.legacy_frozen_kinetic_reference_ha:+.12f} Ha, "
        f"bad-eigen={result.legacy_bad_eigen_kinetic_reference_ha:+.12f} Ha"
    )
    print()
    print("geometry summary:")
    print(
        f"  monitor shape={H2_MONITOR_LOCAL_PATCH_BASELINE_SHAPE}, "
        f"box={H2_MONITOR_LOCAL_PATCH_BASELINE_BOX_HALF_EXTENTS_BOHR}"
    )
    print(
        f"  jacobian min/max={geometry.jacobian_summary.minimum:+.6e}/"
        f"{geometry.jacobian_summary.maximum:+.6e}"
    )
    print(
        f"  cell volume min/max={geometry.cell_volume_summary.minimum:+.6e}/"
        f"{geometry.cell_volume_summary.maximum:+.6e}"
    )
    print(
        f"  spacing mean/rms={geometry.spacing_summary.mean:+.6e}/"
        f"{geometry.spacing_summary.rms:.6e}"
    )
    print(
        "  inverse metric diag rms: "
        f"g11={geometry.inverse_metric_diagonal_summaries[0].rms:.6e}, "
        f"g22={geometry.inverse_metric_diagonal_summaries[1].rms:.6e}, "
        f"g33={geometry.inverse_metric_diagonal_summaries[2].rms:.6e}"
    )
    print(
        "  geometry mismatches: "
        f"cellvol/J rms={geometry.cell_volume_vs_jacobian.relative_summary.rms:.3e}, "
        f"J recon rms={geometry.reconstructed_jacobian.relative_summary.rms:.3e}, "
        f"ginv recon rms={geometry.reconstructed_inverse_metric.relative_summary.rms:.3e}, "
        f"sqrt(det g)-J rms={geometry.sqrt_det_metric_vs_jacobian.relative_summary.rms:.3e}, "
        f"metric*ginv-I rms={geometry.metric_inverse_identity_summary.rms:.3e}"
    )
    print("  region geometry diagnostics:")
    for region in geometry.region_summaries:
        print(
            "    "
            f"{region.region_name}: jac_rms={region.jacobian_summary.rms:.6e}, "
            f"spacing_mean={region.spacing_summary.mean:.6e}, "
            f"ginv_trace_mean={region.inverse_metric_trace_summary.mean:.6e}, "
            f"cond_mean={region.metric_condition_summary.mean:.6e}, "
            f"cellvol/J_rms={region.cell_volume_jacobian_relative_rms:.3e}, "
            f"ginv_recon_rms={region.inverse_metric_relative_rms:.3e}"
        )
    print()
    _print_field_result(result.frozen_trial_result)
    print()
    _print_field_result(result.bad_eigen_result)
    print()
    print("smooth probes:")
    for field in result.smooth_field_results:
        print(
            "  "
            f"{field.field_label}: op={field.operator_kinetic_ha:+.12f} Ha, "
            f"grad={field.gradient_reference_ha:+.12f} Ha, "
            f"delta={field.delta_kinetic_mha:+.3f} mHa"
        )
    print()
    print(f"reference kinetic-form baseline diagnosis: {H2_KINETIC_FORM_AUDIT_BASELINE.diagnosis}")
    print(f"diagnosis: {result.diagnosis}")


def main() -> int:
    result = run_h2_monitor_grid_geometry_consistency_audit()
    print_h2_monitor_grid_geometry_consistency_audit_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
