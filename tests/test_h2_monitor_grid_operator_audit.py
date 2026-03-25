"""Minimal object tests for the H2 monitor-grid operator audit layer."""

from isogrid.audit.h2_monitor_grid_operator_audit import H2MonitorGridOperatorAuditResult
from isogrid.audit.h2_monitor_grid_operator_audit import H2StaticLocalOperatorRouteResult
from isogrid.audit.h2_monitor_grid_operator_audit import OperatorCenterlineSample
from isogrid.audit.h2_monitor_grid_operator_audit import ResidualRegionSummary
from isogrid.audit.h2_monitor_grid_operator_audit import ScalarFieldSummary
from isogrid.audit.h2_monitor_grid_operator_audit import SelfAdjointnessProbe
from isogrid.audit.h2_monitor_grid_operator_audit import WeightedExpectationSummary


def _expectation() -> WeightedExpectationSummary:
    return WeightedExpectationSummary(
        weighted_norm=1.0,
        denominator=1.0,
        rayleigh_quotient=-0.2,
        kinetic_expectation=0.4,
        local_ionic_expectation=-1.6,
        hartree_expectation=1.4,
        xc_expectation=-0.4,
    )


def _summary() -> ScalarFieldSummary:
    return ScalarFieldSummary(minimum=-1.0, maximum=2.0, mean=0.1, rms=0.5)


def _probe() -> SelfAdjointnessProbe:
    return SelfAdjointnessProbe(
        absolute_difference=1.0e-12,
        relative_difference=1.0e-14,
        left_inner_product_real=-0.5,
        right_inner_product_real=-0.5,
    )


def test_operator_route_result_fields_exist() -> None:
    route = H2StaticLocalOperatorRouteResult(
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        grid_parameter_summary="shape=(67,67,81)",
        patch_parameter_summary=None,
        frozen_density_integral=2.0,
        trial_expectation=_expectation(),
        eigen_expectation=_expectation(),
        eigenvalue=-6.5,
        weighted_residual_norm=3.3,
        residual_summary=_summary(),
        residual_centerline_samples=(
            OperatorCenterlineSample(
                sample_index=0,
                z_coordinate_bohr=0.0,
                orbital_value=1.0,
                residual_value=0.1,
            ),
        ),
        residual_regions=(
            ResidualRegionSummary(
                region_name="far_field",
                point_fraction=0.4,
                residual_summary=_summary(),
                weighted_rms=0.2,
                mean_signed_residual=0.01,
            ),
        ),
        self_adjoint_probe_total=_probe(),
        self_adjoint_probe_kinetic=_probe(),
        self_adjoint_probe_local_potential=_probe(),
        converged=False,
        patch_embedding_energy_mismatch=0.0,
        patch_embedded_correction_mha=77.815,
    )

    assert route.trial_expectation.rayleigh_quotient == -0.2
    assert route.eigen_expectation.kinetic_expectation == 0.4
    assert route.weighted_residual_norm == 3.3
    assert route.self_adjoint_probe_total.relative_difference == 1.0e-14
    assert route.residual_regions[0].region_name == "far_field"


def test_operator_audit_result_fields_exist() -> None:
    route = H2StaticLocalOperatorRouteResult(
        path_type="legacy",
        kinetic_version="production",
        grid_parameter_summary="legacy",
        patch_parameter_summary=None,
        frozen_density_integral=2.0,
        trial_expectation=_expectation(),
        eigen_expectation=_expectation(),
        eigenvalue=-0.2,
        weighted_residual_norm=1.0e-3,
        residual_summary=_summary(),
        residual_centerline_samples=(),
        residual_regions=(),
        self_adjoint_probe_total=_probe(),
        self_adjoint_probe_kinetic=_probe(),
        self_adjoint_probe_local_potential=_probe(),
        converged=True,
        patch_embedding_energy_mismatch=None,
        patch_embedded_correction_mha=None,
    )
    result = H2MonitorGridOperatorAuditResult(
        legacy_result=route,
        monitor_patch_production_result=route,
        monitor_patch_trial_fix_result=route,
        diagnosis="operator-level diagnosis",
        note="audit note",
    )

    assert result.monitor_patch_trial_fix_result.eigenvalue == -0.2
    assert "operator-level" in result.diagnosis
