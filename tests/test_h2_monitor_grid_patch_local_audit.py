"""Minimal smoke tests for the H2 monitor-grid local-GTH patch audit."""

import numpy as np

from isogrid.audit.h2_monitor_grid_patch_local_audit import H2MonitorPatchParameterSummary
from isogrid.audit.h2_monitor_grid_patch_local_audit import H2TsElocPatchRouteResult
from isogrid.audit.h2_monitor_grid_ts_eloc_audit import H2GridEnergyGeometrySummary
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.pseudo import LocalPotentialPatchParameters
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential_with_patch


def test_monitor_grid_local_patch_path_runs() -> None:
    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(21, 21, 25),
        box_half_extents=(8.0, 8.0, 10.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    density_field = np.exp(
        -0.6
        * (
            grid_geometry.x_points**2
            + grid_geometry.y_points**2
            + grid_geometry.z_points**2
        )
    )
    base_evaluation = evaluate_monitor_grid_local_ionic_potential(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    patch_evaluation = evaluate_monitor_grid_local_ionic_potential_with_patch(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        density_field=density_field,
        patch_parameters=LocalPotentialPatchParameters(
            patch_radius_scale=0.75,
            patch_grid_shape=(11, 11, 11),
            correction_strength=1.0,
            interpolation_neighbors=4,
        ),
        base_evaluation=base_evaluation,
    )

    assert np.isfinite(patch_evaluation.uncorrected_local_energy)
    assert np.isfinite(patch_evaluation.corrected_local_energy)
    assert np.isfinite(patch_evaluation.total_patch_correction)
    assert len(patch_evaluation.atomic_patch_corrections) == len(H2_BENCHMARK_CASE.geometry.atoms)


def test_construct_patch_route_result_object() -> None:
    summary = H2GridEnergyGeometrySummary(
        grid_type="monitor_a_grid",
        grid_shape=(67, 67, 81),
        box_half_extents_bohr=(8.0, 8.0, 10.0),
        min_spacing_estimate_bohr=0.12,
        near_core_min_spacing_bohr=0.11,
        near_atom_spacing_bohr=0.21,
        far_field_spacing_bohr=0.43,
        min_jacobian=1.0,
        max_jacobian=2.0,
        center_line_local_potential_center=-1.0,
        center_line_local_potential_near_atom=-2.0,
    )
    result = H2TsElocPatchRouteResult(
        path_type="monitor_a_grid_plus_patch",
        grid_parameter_summary="shape=(67,67,81)",
        geometry_summary=summary,
        patch_parameter_summary=H2MonitorPatchParameterSummary(
            patch_radius_scale=0.75,
            patch_grid_shape=(25, 25, 25),
            correction_strength=1.3,
            interpolation_neighbors=8,
        ),
        kinetic_energy=1.9,
        local_ionic_energy=-4.3,
        ts_plus_eloc_energy=-2.4,
        reference_pyscf_total_energy=-1.13,
        reference_offset_ha=-1.27,
        reference_offset_mha=-1270.0,
        delta_vs_legacy_ha=-0.01,
        delta_vs_legacy_mha=-10.0,
        delta_vs_unpatched_monitor_ha=0.05,
        delta_vs_unpatched_monitor_mha=50.0,
        improvement_vs_unpatched_monitor_ha=0.05,
        improvement_vs_unpatched_monitor_mha=50.0,
        patch_correction_ha=0.05,
        patch_correction_mha=50.0,
    )

    assert result.path_type == "monitor_a_grid_plus_patch"
    assert result.patch_parameter_summary is not None
    assert result.patch_parameter_summary.correction_strength == 1.3
    assert result.improvement_vs_unpatched_monitor_mha == 50.0
