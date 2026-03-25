"""Minimal object tests for the H2 monitor-grid geometry consistency audit."""

from isogrid.audit.h2_monitor_grid_geometry_consistency_audit import GeometryConsistencySummary
from isogrid.audit.h2_monitor_grid_geometry_consistency_audit import GeometryMismatchSummary
from isogrid.audit.h2_monitor_grid_geometry_consistency_audit import GeometryRegionSummary
from isogrid.audit.h2_monitor_grid_geometry_consistency_audit import H2MonitorGridGeometryConsistencyAuditResult
from isogrid.audit.h2_monitor_grid_geometry_consistency_audit import KineticIdentityCenterlineSample
from isogrid.audit.h2_monitor_grid_geometry_consistency_audit import KineticIdentityFieldResult
from isogrid.audit.h2_monitor_grid_geometry_consistency_audit import KineticIdentityRegionSummary
from isogrid.audit.h2_monitor_grid_operator_audit import ScalarFieldSummary


def _summary() -> ScalarFieldSummary:
    return ScalarFieldSummary(minimum=-1.0, maximum=2.0, mean=0.1, rms=0.5)


def _geometry_summary() -> GeometryConsistencySummary:
    mismatch = GeometryMismatchSummary(
        label="cell_volume_vs_jacobian",
        absolute_summary=_summary(),
        relative_summary=_summary(),
    )
    return GeometryConsistencySummary(
        jacobian_summary=_summary(),
        cell_volume_summary=_summary(),
        spacing_summary=_summary(),
        inverse_metric_diagonal_summaries=(_summary(), _summary(), _summary()),
        inverse_metric_offdiagonal_rms=(0.1, 0.2, 0.3),
        total_physical_volume=12.0,
        logical_cell_volume_element=0.01,
        cell_volume_vs_jacobian=mismatch,
        reconstructed_jacobian=mismatch,
        reconstructed_inverse_metric=mismatch,
        sqrt_det_metric_vs_jacobian=mismatch,
        metric_inverse_identity_summary=_summary(),
        region_summaries=(
            GeometryRegionSummary(
                region_name="far_field",
                point_fraction=0.2,
                jacobian_summary=_summary(),
                cell_volume_summary=_summary(),
                spacing_summary=_summary(),
                inverse_metric_trace_summary=_summary(),
                metric_condition_summary=_summary(),
                cell_volume_jacobian_relative_rms=1.0e-12,
                inverse_metric_relative_rms=2.0e-12,
            ),
        ),
    )


def _field_result(label: str) -> KineticIdentityFieldResult:
    return KineticIdentityFieldResult(
        shape_label="baseline",
        field_label=label,
        weighted_norm=1.0,
        operator_kinetic_ha=-3.0,
        gradient_reference_ha=4.0,
        delta_kinetic_mha=-7000.0,
        operator_action_summary=_summary(),
        operator_indicator_summary=_summary(),
        gradient_indicator_summary=_summary(),
        delta_indicator_summary=_summary(),
        region_summaries=(
            KineticIdentityRegionSummary(
                region_name="far_field",
                point_fraction=0.3,
                operator_indicator_summary=_summary(),
                gradient_indicator_summary=_summary(),
                delta_indicator_summary=_summary(),
                operator_weighted_contribution_ha=-3.0,
                gradient_weighted_contribution_ha=4.0,
                delta_weighted_contribution_mha=-7000.0,
            ),
        ),
        centerline_samples=(
            KineticIdentityCenterlineSample(
                sample_index=0,
                z_coordinate_bohr=0.0,
                orbital_value=1.0,
                operator_indicator=-0.2,
                gradient_indicator=0.2,
                delta_indicator=-0.4,
            ),
        ),
        source_eigenvalue_ha=-6.0,
        source_residual_norm=3.0,
        source_converged=False,
    )


def test_geometry_consistency_result_fields_exist() -> None:
    result = H2MonitorGridGeometryConsistencyAuditResult(
        geometry_summary=_geometry_summary(),
        frozen_trial_result=_field_result("frozen_trial_orbital"),
        bad_eigen_result=_field_result("bad_eigensolver_orbital_k1"),
        smooth_field_results=(_field_result("smooth_gaussian"),),
        legacy_frozen_kinetic_reference_ha=1.0,
        legacy_bad_eigen_kinetic_reference_ha=0.4,
        diagnosis="geometry diagnosis",
        note="audit note",
    )

    assert result.geometry_summary.cell_volume_vs_jacobian.label == "cell_volume_vs_jacobian"
    assert result.bad_eigen_result.delta_kinetic_mha == -7000.0
    assert result.bad_eigen_result.region_summaries[0].region_name == "far_field"
    assert "geometry" in result.diagnosis
