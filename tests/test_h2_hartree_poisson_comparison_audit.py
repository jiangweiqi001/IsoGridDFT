"""Minimal smoke tests for the H2 Hartree / Poisson comparison audit."""

from importlib import import_module

import numpy as np

from isogrid.audit.h2_hartree_poisson_comparison_audit import H2HartreeDifferenceSummary
from isogrid.audit.h2_hartree_poisson_comparison_audit import H2HartreeRouteResult
from isogrid.audit.h2_hartree_poisson_comparison_audit import HartreeBoundarySummary
from isogrid.audit.h2_hartree_poisson_comparison_audit import evaluate_h2_singlet_hartree_route
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_default_h2_grid_geometry
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case
from isogrid.poisson import evaluate_hartree_energy
from isogrid.poisson import solve_hartree_potential


def test_hartree_poisson_comparison_audit_module_imports() -> None:
    audit_module = import_module("isogrid.audit.h2_hartree_poisson_comparison_audit")

    assert hasattr(audit_module, "run_h2_hartree_poisson_comparison_audit")
    assert hasattr(audit_module, "evaluate_h2_singlet_hartree_route")


def test_small_monitor_grid_hartree_route_is_finite_and_symmetric() -> None:
    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(21, 21, 25),
        box_half_extents=(8.0, 8.0, 10.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    route = evaluate_h2_singlet_hartree_route(
        case=H2_BENCHMARK_CASE,
        grid_geometry=grid_geometry,
        grid_type="monitor_a_grid",
        multipole_order=2,
        tolerance=1.0e-6,
        max_iterations=200,
    )

    assert np.isfinite(route.hartree_energy)
    assert route.hartree_energy > 0.0
    assert np.isfinite(route.residual_max)
    assert route.mirror_symmetric


def test_small_monitor_grid_gaussian_density_hartree_is_finite() -> None:
    grid_geometry = build_monitor_grid_for_case(
        H2_BENCHMARK_CASE,
        shape=(19, 19, 23),
        box_half_extents=(8.0, 8.0, 10.0),
        element_parameters=build_h2_local_patch_development_element_parameters(),
    )
    rho = np.exp(
        -0.5
        * (
            grid_geometry.x_points**2
            + grid_geometry.y_points**2
            + grid_geometry.z_points**2
        )
    )
    result = solve_hartree_potential(
        grid_geometry=grid_geometry,
        rho=rho,
        tolerance=1.0e-6,
        max_iterations=200,
    )
    energy = evaluate_hartree_energy(
        rho=rho,
        grid_geometry=grid_geometry,
        hartree_potential=result,
    )

    assert np.all(np.isfinite(result.potential))
    assert np.isfinite(energy)
    assert energy > 0.0
    assert np.allclose(result.potential, result.potential[:, :, ::-1])


def test_construct_hartree_comparison_result_objects() -> None:
    legacy_route = H2HartreeRouteResult(
        grid_type="legacy",
        grid_parameter_summary="legacy structured sinh baseline",
        density_integral=2.0,
        density_integral_error=0.0,
        potential_min=0.1,
        potential_max=1.0,
        hartree_energy=1.7,
        solver_method="scipy_bicgstab",
        solver_iterations=10,
        residual_max=1.0e-8,
        boundary_summary=HartreeBoundarySummary(
            multipole_order=2,
            total_charge=2.0,
            dipole_norm=0.0,
            quadrupole_norm=1.0,
            boundary_min=0.1,
            boundary_max=0.2,
            boundary_mean=0.15,
            description="quadrupole boundary",
        ),
        center_potential=1.0,
        near_atom_potential=0.9,
        far_boundary_mean=0.15,
        centerline_samples=(),
        mirror_symmetric=True,
    )
    diff = H2HartreeDifferenceSummary(
        density_integral_difference=0.0,
        hartree_energy_difference_ha=-0.4,
        hartree_energy_difference_mha=-400.0,
        center_potential_difference=-0.2,
        near_atom_potential_difference=-0.1,
        far_boundary_mean_difference=-0.05,
        centerline_max_abs_difference=0.2,
        centerline_inner_mean_abs_difference=0.18,
        centerline_middle_mean_abs_difference=0.07,
        centerline_outer_mean_abs_difference=0.03,
        likely_difference_pattern="near_core_or_geometry_dominated",
    )

    assert legacy_route.boundary_summary.multipole_order == 2
    assert diff.hartree_energy_difference_mha == -400.0
    assert diff.likely_difference_pattern == "near_core_or_geometry_dominated"
