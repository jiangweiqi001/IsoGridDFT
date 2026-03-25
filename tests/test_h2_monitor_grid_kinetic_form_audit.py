"""Minimal object tests for the H2 monitor-grid kinetic form audit."""

from isogrid.audit.h2_monitor_grid_kinetic_form_audit import H2MonitorGridKineticFormAuditResult
from isogrid.audit.h2_monitor_grid_kinetic_form_audit import KineticFormCenterlineSample
from isogrid.audit.h2_monitor_grid_kinetic_form_audit import KineticFormComparisonResult
from isogrid.audit.h2_monitor_grid_kinetic_form_audit import KineticFormRegionSummary
from isogrid.audit.h2_monitor_grid_kinetic_form_audit import KineticFormSelfAdjointnessComparison
from isogrid.audit.h2_monitor_grid_kinetic_operator_audit import ScalarFieldSummary
from isogrid.audit.h2_monitor_grid_operator_audit import SelfAdjointnessProbe


def _summary() -> ScalarFieldSummary:
    return ScalarFieldSummary(minimum=-1.0, maximum=2.0, mean=0.1, rms=0.5)


def _probe() -> SelfAdjointnessProbe:
    return SelfAdjointnessProbe(
        absolute_difference=1.0e-8,
        relative_difference=1.0e-4,
        left_inner_product_real=0.5,
        right_inner_product_real=0.5,
    )


def _comparison(label: str) -> KineticFormComparisonResult:
    return KineticFormComparisonResult(
        shape_label="baseline",
        orbital_label=label,
        production_label="monitor_production",
        reference_label="monitor_reference",
        weighted_norm=1.0,
        production_kinetic_quotient=-3.0,
        reference_kinetic_quotient=-2.9,
        delta_kinetic_quotient_mha=-100.0,
        production_tpsi_summary=_summary(),
        reference_tpsi_summary=_summary(),
        delta_tpsi_summary=_summary(),
        production_region_summaries=(
            KineticFormRegionSummary(
                region_name="far_field",
                point_fraction=0.2,
                field_summary=_summary(),
                weighted_mean=-0.2,
                weighted_rms=0.3,
                negative_fraction=0.6,
            ),
        ),
        reference_region_summaries=(
            KineticFormRegionSummary(
                region_name="far_field",
                point_fraction=0.2,
                field_summary=_summary(),
                weighted_mean=-0.1,
                weighted_rms=0.25,
                negative_fraction=0.5,
            ),
        ),
        delta_region_summaries=(
            KineticFormRegionSummary(
                region_name="far_field",
                point_fraction=0.2,
                field_summary=_summary(),
                weighted_mean=-0.1,
                weighted_rms=0.05,
                negative_fraction=0.4,
            ),
        ),
        centerline_samples=(
            KineticFormCenterlineSample(
                sample_index=0,
                z_coordinate_bohr=0.0,
                orbital_value=1.0,
                production_tpsi=-0.2,
                reference_tpsi=-0.1,
                delta_tpsi=-0.1,
            ),
        ),
        eigensolver_eigenvalue_ha=-6.0,
        eigensolver_residual_norm=3.0,
        eigensolver_converged=False,
    )


def test_kinetic_form_comparison_fields_exist() -> None:
    comparison = _comparison("bad_eigensolver_orbital_k1")

    assert comparison.delta_kinetic_quotient_mha == -100.0
    assert comparison.delta_region_summaries[0].region_name == "far_field"
    assert comparison.centerline_samples[0].delta_tpsi == -0.1


def test_kinetic_form_audit_result_fields_exist() -> None:
    comparison = _comparison("bad_eigensolver_orbital_k1")
    result = H2MonitorGridKineticFormAuditResult(
        frozen_trial_baseline=_comparison("frozen_trial_orbital"),
        bad_eigen_baseline=comparison,
        bad_eigen_finer_shape=comparison,
        smooth_field_results=(_comparison("smooth_gaussian"),),
        self_adjointness_baseline=KineticFormSelfAdjointnessComparison(
            shape_label="baseline",
            production_probe=_probe(),
            reference_probe=_probe(),
        ),
        self_adjointness_finer_shape=KineticFormSelfAdjointnessComparison(
            shape_label="finer-shape",
            production_probe=_probe(),
            reference_probe=_probe(),
        ),
        diagnosis="kinetic form diagnosis",
        note="audit note",
    )

    assert result.bad_eigen_baseline.production_kinetic_quotient == -3.0
    assert result.self_adjointness_baseline.reference_probe.relative_difference == 1.0e-4
    assert "diagnosis" in result.diagnosis
