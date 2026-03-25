"""Minimal smoke tests for the H2 monitor-grid patch Hartree/XC audit."""

import numpy as np

from isogrid.audit.h2_monitor_grid_patch_hartree_xc_audit import H2StaticLocalRouteResult
from isogrid.audit.h2_monitor_grid_patch_local_audit import H2MonitorPatchParameterSummary
from isogrid.audit.h2_monitor_grid_ts_eloc_audit import H2GridEnergyGeometrySummary
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.poisson import evaluate_hartree_energy
from isogrid.poisson import solve_hartree_potential
from isogrid.pseudo import LocalPotentialPatchParameters
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential
from isogrid.pseudo import evaluate_monitor_grid_local_ionic_potential_with_patch
from isogrid.xc import evaluate_lsda_energy


def test_monitor_grid_patch_hartree_xc_components_are_finite() -> None:
    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(21, 21, 25),
        box_half_extents=(8.0, 8.0, 10.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    density_orbital = np.exp(
        -0.6
        * (
            grid_geometry.x_points**2
            + grid_geometry.y_points**2
            + grid_geometry.z_points**2
        )
    )
    norm = float(np.sqrt(np.sum(density_orbital * density_orbital * grid_geometry.cell_volumes)))
    orbital = density_orbital / norm
    rho_up = np.abs(orbital) ** 2
    rho_down = np.abs(orbital) ** 2
    rho_total = rho_up + rho_down

    base_local = evaluate_monitor_grid_local_ionic_potential(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
    )
    patch_eval = evaluate_monitor_grid_local_ionic_potential_with_patch(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        density_field=rho_total,
        patch_parameters=LocalPotentialPatchParameters(
            patch_radius_scale=0.75,
            patch_grid_shape=(11, 11, 11),
            correction_strength=1.2,
            interpolation_neighbors=4,
        ),
        base_evaluation=base_local,
    )
    hartree_result = solve_hartree_potential(
        grid_geometry=grid_geometry,
        rho=rho_total,
        tolerance=1.0e-6,
        max_iterations=200,
    )
    hartree_energy = evaluate_hartree_energy(
        rho=rho_total,
        grid_geometry=grid_geometry,
        hartree_potential=hartree_result,
    )
    xc_energy = evaluate_lsda_energy(
        rho_up=rho_up,
        rho_down=rho_down,
        grid_geometry=grid_geometry,
    )
    static_local_sum = patch_eval.corrected_local_energy + hartree_energy + xc_energy

    assert np.isfinite(patch_eval.corrected_local_energy)
    assert np.isfinite(hartree_energy)
    assert np.isfinite(xc_energy)
    assert np.isfinite(static_local_sum)


def test_construct_static_local_route_result_object() -> None:
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
    result = H2StaticLocalRouteResult(
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
        hartree_energy=0.7,
        xc_energy=-0.2,
        static_local_energy=-1.9,
        reference_pyscf_total_energy=-1.13,
        reference_offset_ha=-0.77,
        reference_offset_mha=-770.0,
        delta_vs_legacy_ha=-0.01,
        delta_vs_legacy_mha=-10.0,
        delta_vs_unpatched_monitor_ha=0.05,
        delta_vs_unpatched_monitor_mha=50.0,
        improvement_vs_unpatched_monitor_ha=0.05,
        improvement_vs_unpatched_monitor_mha=50.0,
        patch_correction_ha=0.07,
        patch_correction_mha=70.0,
        hartree_solver_method="scipy_bicgstab_monitor",
        hartree_solver_iterations=42,
        hartree_residual_max=1.0e-8,
    )

    assert result.path_type == "monitor_a_grid_plus_patch"
    assert result.patch_parameter_summary is not None
    assert result.patch_parameter_summary.correction_strength == 1.3
    assert result.hartree_energy == 0.7
    assert result.xc_energy == -0.2
