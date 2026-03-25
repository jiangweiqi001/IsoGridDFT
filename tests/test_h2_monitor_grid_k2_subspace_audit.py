"""Smoke tests for the H2 k=2 subspace audit."""

from isogrid.audit.h2_monitor_grid_k2_subspace_audit import H2K2SubspaceMatrixSummary
from isogrid.audit.h2_monitor_grid_k2_subspace_audit import H2K2SubspaceRotationSummary
from isogrid.audit.h2_monitor_grid_k2_subspace_audit import H2K2SubspaceRouteResult
from isogrid.audit.h2_monitor_grid_k2_subspace_audit import H2MonitorGridK2SubspaceAuditResult
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import H2OrbitalShapeResult
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import OrbitalBoundarySummary
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import OrbitalCenterlineSample
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import OrbitalNodeSummary
from isogrid.audit.h2_monitor_grid_orbital_shape_audit import OrbitalSymmetrySummary


def test_construct_h2_k2_subspace_audit_result() -> None:
    sample = OrbitalCenterlineSample(0, 0.0, 1.0)
    symmetry = OrbitalSymmetrySummary(1.0, "even", 0.0, 1.0, "even", 0.0, 0.0)
    node = OrbitalNodeSummary(0, (), 0, 0.1, 1.0, 0.1)
    boundary = OrbitalBoundarySummary(0.3, 0.01, 0.001, 0.0005, 0.01, 0.02)
    orbital = H2OrbitalShapeResult(
        solve_label="k2",
        orbital_index=0,
        eigenvalue_ha=-0.18,
        weighted_norm=1.0,
        residual_norm=1.0e-4,
        symmetry_summary=symmetry,
        node_summary=node,
        boundary_summary=boundary,
        centerline_samples=(sample,),
    )
    matrix = H2K2SubspaceMatrixSummary(
        label="z_mirror",
        matrix=((1.0, 0.0), (0.0, 1.0)),
        eigenvalues=(1.0, 1.0),
    )
    rotation = H2K2SubspaceRotationSummary(
        rotation_label="bonding_overlap_rotation",
        rotation_matrix=((1.0, 0.0), (0.0, 1.0)),
        raw_bonding_overlaps=(0.6, 0.6),
        rotated_bonding_overlaps=(0.85, 0.0),
        rotated_orbitals=(orbital, orbital),
        note="rotation smoke",
    )
    route = H2K2SubspaceRouteResult(
        path_type="monitor_a_grid_plus_patch",
        kinetic_version="trial_fix",
        grid_parameter_summary="shape=(67,67,81)",
        patch_parameter_summary=None,
        raw_k2_eigenvalues_ha=(-0.18, -0.18),
        raw_k2_gap_ha=1.0e-4,
        raw_k2_orbitals=(orbital, orbital),
        inversion_matrix=matrix,
        z_mirror_matrix=matrix,
        bonding_rotation=rotation,
    )
    result = H2MonitorGridK2SubspaceAuditResult(
        legacy_route=route,
        monitor_patch_trial_fix_route=route,
        diagnosis="subspace smoke",
        note="audit smoke",
    )

    assert result.monitor_patch_trial_fix_route.kinetic_version == "trial_fix"
    assert result.monitor_patch_trial_fix_route.bonding_rotation.rotation_label == "bonding_overlap_rotation"
    assert result.monitor_patch_trial_fix_route.raw_k2_orbitals[0].node_summary.centerline_sign_changes == 0
    assert (
        result.monitor_patch_trial_fix_route.bonding_rotation.rotated_orbitals[0]
        .boundary_summary.boundary_layer_norm_fraction
        == 0.0005
    )
