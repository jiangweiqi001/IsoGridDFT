"""Minimal object tests for the H2 monitor-grid kinetic Green-identity audit."""

from isogrid.audit.h2_monitor_grid_kinetic_green_identity_audit import BoundaryFaceContribution
from isogrid.audit.h2_monitor_grid_kinetic_green_identity_audit import (
    GreenIdentityCenterlineSample,
)
from isogrid.audit.h2_monitor_grid_kinetic_green_identity_audit import (
    GreenIdentityFieldResult,
)
from isogrid.audit.h2_monitor_grid_kinetic_green_identity_audit import (
    GreenIdentityRegionSummary,
)
from isogrid.audit.h2_monitor_grid_kinetic_green_identity_audit import (
    H2MonitorGridKineticGreenIdentityAuditResult,
)
from isogrid.audit.h2_monitor_grid_operator_audit import ScalarFieldSummary


def _summary() -> ScalarFieldSummary:
    return ScalarFieldSummary(minimum=-1.0, maximum=2.0, mean=0.1, rms=0.5)


def _field_result(label: str) -> GreenIdentityFieldResult:
    return GreenIdentityFieldResult(
        shape_label="baseline",
        field_label=label,
        weighted_norm=1.0,
        operator_kinetic_ha=-3.0,
        gradient_kinetic_ha=4.0,
        delta_kinetic_mha=-7000.0,
        boundary_term_ha=-1.0,
        closure_mismatch_mha=-6000.0,
        operator_indicator_summary=_summary(),
        gradient_indicator_summary=_summary(),
        boundary_indicator_summary=_summary(),
        closure_indicator_summary=_summary(),
        face_contributions=(
            BoundaryFaceContribution(face_label="z_max", contribution_ha=-0.5),
        ),
        region_summaries=(
            GreenIdentityRegionSummary(
                region_name="far_field",
                point_fraction=0.2,
                operator_indicator_summary=_summary(),
                gradient_indicator_summary=_summary(),
                boundary_indicator_summary=_summary(),
                closure_indicator_summary=_summary(),
                operator_weighted_contribution_ha=-3.0,
                gradient_weighted_contribution_ha=4.0,
                boundary_weighted_contribution_ha=-1.0,
                delta_weighted_contribution_mha=-7000.0,
                closure_weighted_contribution_mha=-6000.0,
            ),
        ),
        centerline_samples=(
            GreenIdentityCenterlineSample(
                sample_index=0,
                z_coordinate_bohr=0.0,
                orbital_value=1.0,
                operator_indicator=-0.2,
                gradient_indicator=0.2,
                boundary_indicator=-0.1,
                closure_indicator=-0.3,
            ),
        ),
        source_eigenvalue_ha=-6.0,
        source_residual_norm=3.0,
        source_converged=False,
    )


def test_green_identity_field_fields_exist() -> None:
    field = _field_result("bad_eigensolver_orbital_k1")

    assert field.delta_kinetic_mha == -7000.0
    assert field.boundary_term_ha == -1.0
    assert field.region_summaries[0].region_name == "far_field"
    assert field.centerline_samples[0].boundary_indicator == -0.1


def test_green_identity_audit_result_fields_exist() -> None:
    result = H2MonitorGridKineticGreenIdentityAuditResult(
        frozen_trial_baseline=_field_result("frozen_trial_orbital"),
        bad_eigen_baseline=_field_result("bad_eigensolver_orbital_k1"),
        bad_eigen_finer_shape=_field_result("bad_eigensolver_orbital_k1"),
        smooth_field_results=(_field_result("smooth_gaussian"),),
        diagnosis="boundary diagnosis",
        note="audit note",
    )

    assert result.bad_eigen_baseline.operator_kinetic_ha == -3.0
    assert result.bad_eigen_baseline.boundary_term_ha == -1.0
    assert "boundary" in result.diagnosis
