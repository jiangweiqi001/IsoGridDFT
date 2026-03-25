"""Minimal object tests for the H2 monitor-grid kinetic operator audit."""

from isogrid.audit.h2_monitor_grid_kinetic_operator_audit import H2MonitorGridKineticOperatorAuditResult
from isogrid.audit.h2_monitor_grid_kinetic_operator_audit import KineticCenterlineSample
from isogrid.audit.h2_monitor_grid_kinetic_operator_audit import KineticOrbitalSummary
from isogrid.audit.h2_monitor_grid_kinetic_operator_audit import KineticRegionSummary
from isogrid.audit.h2_monitor_grid_kinetic_operator_audit import KineticRouteAuditResult
from isogrid.audit.h2_monitor_grid_kinetic_operator_audit import ScalarFieldSummary
from isogrid.audit.h2_monitor_grid_kinetic_operator_audit import SelfAdjointnessProbe
from isogrid.audit.h2_monitor_grid_kinetic_operator_audit import SmoothFieldKineticResult


def _summary() -> ScalarFieldSummary:
    return ScalarFieldSummary(minimum=-1.0, maximum=2.0, mean=0.1, rms=0.5)


def _probe() -> SelfAdjointnessProbe:
    return SelfAdjointnessProbe(
        absolute_difference=1.0e-12,
        relative_difference=1.0e-14,
        left_inner_product_real=0.5,
        right_inner_product_real=0.5,
    )


def _orbital_summary(label: str) -> KineticOrbitalSummary:
    return KineticOrbitalSummary(
        orbital_label=label,
        weighted_norm=1.0,
        kinetic_rayleigh_quotient=0.4,
        kinetic_action_summary=_summary(),
        local_indicator_summary=_summary(),
        negative_indicator_fraction=0.25,
        centerline_samples=(
            KineticCenterlineSample(
                sample_index=0,
                z_coordinate_bohr=0.0,
                orbital_value=1.0,
                kinetic_action_value=0.2,
                local_kinetic_indicator=0.2,
            ),
        ),
        region_summaries=(
            KineticRegionSummary(
                region_name="far_field",
                point_fraction=0.4,
                psi_abs_mean=0.01,
                kinetic_action_summary=_summary(),
                local_indicator_mean=-0.1,
                weighted_indicator_contribution_ha=-0.2,
                negative_indicator_fraction=0.5,
            ),
        ),
    )


def test_kinetic_route_result_fields_exist() -> None:
    route = KineticRouteAuditResult(
        path_type="monitor_a_grid_plus_patch",
        shape_label="baseline",
        grid_parameter_summary="shape=(67,67,81)",
        patch_parameter_summary=None,
        frozen_density_integral=2.0,
        frozen_orbital_summary=_orbital_summary("frozen"),
        eigen_orbital_summary=_orbital_summary("eigen"),
        eigenvalue_ha=-6.5,
        eigensolver_converged=False,
        eigensolver_weighted_residual_norm=3.3,
        self_adjoint_probe=_probe(),
    )

    assert route.frozen_orbital_summary.kinetic_rayleigh_quotient == 0.4
    assert route.eigen_orbital_summary.region_summaries[0].region_name == "far_field"
    assert route.self_adjoint_probe.relative_difference == 1.0e-14


def test_kinetic_audit_result_fields_exist() -> None:
    route = KineticRouteAuditResult(
        path_type="legacy",
        shape_label="baseline",
        grid_parameter_summary="legacy",
        patch_parameter_summary=None,
        frozen_density_integral=2.0,
        frozen_orbital_summary=_orbital_summary("frozen"),
        eigen_orbital_summary=_orbital_summary("eigen"),
        eigenvalue_ha=-0.2,
        eigensolver_converged=True,
        eigensolver_weighted_residual_norm=1.0e-3,
        self_adjoint_probe=_probe(),
    )
    smooth = SmoothFieldKineticResult(
        path_type="monitor_a_grid",
        field_label="gaussian",
        kinetic_rayleigh_quotient=0.6,
        weighted_norm=1.0,
        kinetic_action_summary=_summary(),
        negative_indicator_fraction=0.0,
    )
    result = H2MonitorGridKineticOperatorAuditResult(
        legacy_result=route,
        monitor_unpatched_baseline_result=route,
        monitor_patch_baseline_result=route,
        monitor_patch_finer_shape_result=route,
        smooth_field_results=(smooth,),
        diagnosis="kinetic diagnosis",
        note="audit note",
    )

    assert result.monitor_patch_baseline_result.eigenvalue_ha == -0.2
    assert result.smooth_field_results[0].field_label == "gaussian"
    assert "kinetic" in result.diagnosis
