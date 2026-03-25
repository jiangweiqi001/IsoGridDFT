"""Smoke tests for the H2 fixed-potential orbital-shape audit."""

from isogrid.audit.h2_monitor_grid_orbital_shape_audit import H2MonitorGridOrbitalShapeAuditResult
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import H2OrbitalShapeResult
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import H2OrbitalShapeRouteResult
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import OrbitalBoundarySummary
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import OrbitalCenterlineSample
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import OrbitalNodeSummary
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import OrbitalSymmetrySummary


def test_construct_h2_monitor_grid_orbital_shape_audit_result() -> None:
    sample = OrbitalCenterlineSample(
        sample_index=0,
        z_coordinate_bohr=0.0,
        orbital_value=1.0,
    )
    symmetry = OrbitalSymmetrySummary(
        inversion_overlap=1.0,
        inversion_best_parity="even",
        inversion_best_mismatch=0.0,
        z_mirror_overlap=1.0,
        z_mirror_best_parity="even",
        z_mirror_best_mismatch=0.0,
        z_center_of_mass_bohr=0.0,
    )
    node = OrbitalNodeSummary(
        centerline_sign_changes=0,
        node_positions_bohr=(),
        far_field_sign_changes=0,
        left_endpoint_value=0.1,
        center_value=1.0,
        right_endpoint_value=0.1,
    )
    boundary = OrbitalBoundarySummary(
        near_core_norm_fraction=0.3,
        center_norm_fraction=0.01,
        far_field_norm_fraction=0.001,
        boundary_layer_norm_fraction=0.0005,
        far_field_max_abs_value=0.01,
        boundary_layer_max_abs_value=0.02,
    )
    orbital = H2OrbitalShapeResult(
        solve_label="k1",
        orbital_index=0,
        eigenvalue_ha=-0.2,
        weighted_norm=1.0,
        residual_norm=1.0e-4,
        symmetry_summary=symmetry,
        node_summary=node,
        boundary_summary=boundary,
        centerline_samples=(sample,),
    )
    route = H2OrbitalShapeRouteResult(
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        grid_parameter_summary="shape=(67,67,81)",
        patch_parameter_summary=None,
        frozen_density_integral=2.0,
        converged_k1=True,
        converged_k2=True,
        k1_orbital=orbital,
        k2_orbitals=(orbital, orbital),
        k2_gap_ha=1.0e-4,
    )
    result = H2MonitorGridOrbitalShapeAuditResult(
        legacy_route=route,
        monitor_patch_trial_fix_route=route,
        diagnosis="shape smoke",
        note="audit smoke",
    )

    assert result.monitor_patch_trial_fix_route.kinetic_version == "trial_fix"
    assert result.monitor_patch_trial_fix_route.k1_orbital.eigenvalue_ha == -0.2
    assert result.monitor_patch_trial_fix_route.k2_orbitals[0].node_summary.centerline_sign_changes == 0
    assert (
        result.monitor_patch_trial_fix_route.k2_orbitals[0].boundary_summary.boundary_layer_norm_fraction
        == 0.0005
    )
