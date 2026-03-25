"""Minimal smoke tests for the A-grid Poisson-operator audit."""

from importlib import import_module

import numpy as np

from isogrid.audit.h2_monitor_grid_poisson_operator_audit import (
    BoundaryConditionSummary,
)
from isogrid.audit.h2_monitor_grid_poisson_operator_audit import (
    PoissonOperatorDifferenceSummary,
)
from isogrid.audit.h2_monitor_grid_poisson_operator_audit import (
    ScalarFieldSummary,
)
from isogrid.audit.h2_monitor_grid_poisson_operator_audit import (
    evaluate_poisson_operator_route,
)
from isogrid.config import H2_BENCHMARK_CASE
from isogrid.grid import build_h2_local_patch_development_element_parameters
from isogrid.grid import build_monitor_grid_for_case


def test_monitor_grid_poisson_operator_audit_module_imports() -> None:
    module = import_module("isogrid.audit.h2_monitor_grid_poisson_operator_audit")

    assert hasattr(module, "run_h2_monitor_grid_poisson_operator_audit")
    assert hasattr(module, "evaluate_poisson_operator_route")


def test_small_monitor_grid_poisson_operator_route_is_finite() -> None:
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
    rho *= 2.0 / float(np.sum(rho * grid_geometry.cell_volumes))

    result = evaluate_poisson_operator_route(
        case=H2_BENCHMARK_CASE,
        density_field=rho,
        density_label="gaussian_test_density",
        grid_geometry=grid_geometry,
        grid_type="monitor_a_grid",
        tolerance=1.0e-6,
        max_iterations=200,
    )

    assert np.isfinite(result.hartree_energy)
    assert np.isfinite(result.potential_summary.minimum)
    assert np.isfinite(result.residual_summary.rms)
    assert len(result.region_diagnostics) == 3


def test_construct_poisson_operator_difference_object() -> None:
    field = ScalarFieldSummary(minimum=-1.0, maximum=2.0, mean=0.1, rms=0.5)
    boundary = BoundaryConditionSummary(
        multipole_order=2,
        total_charge=2.0,
        dipole_norm=0.0,
        quadrupole_norm=1.0,
        boundary_min=0.1,
        boundary_max=0.2,
        boundary_mean=0.15,
        description="quadrupole boundary",
    )
    diff = PoissonOperatorDifferenceSummary(
        hartree_energy_difference_ha=-0.4,
        hartree_energy_difference_mha=-400.0,
        potential_min_difference=-0.3,
        potential_max_difference=-0.4,
        negative_interior_fraction_difference=0.9,
        residual_rms_ratio=1.0e6,
        centerline_max_abs_difference=0.4,
        centerline_inner_mean_abs_difference=0.39,
        centerline_middle_mean_abs_difference=0.38,
        centerline_outer_mean_abs_difference=0.40,
        likely_difference_pattern="broad_offset_like",
    )

    assert field.rms == 0.5
    assert boundary.multipole_order == 2
    assert diff.hartree_energy_difference_mha == -400.0
    assert diff.likely_difference_pattern == "broad_offset_like"
